from retrieval.vector_store import retrieve, retrieve_diverse
from llm.groq_client import generate_response, generate_response_stream
from llm.prompts import SYSTEM_PROMPT, OVERVIEW_SYSTEM_PROMPT, CASUAL_SYSTEM_PROMPT
from llm.query_pipeline import rewrite_and_route
from retrieval.schema import RetrievalMode
import json
import re

MAX_CONTEXT_CHARS = 12000

# ---------------------------------------------------------------------------
# Import pattern extraction for dependency-following retrieval
# ---------------------------------------------------------------------------

_IMPORT_PATTERNS = [
    # Python: from X import Y  /  import X
    re.compile(r'^\s*from\s+([\w.]+)\s+import', re.MULTILINE),
    re.compile(r'^\s*import\s+([\w.]+)', re.MULTILINE),
    # JS/TS: import ... from "X"  /  require("X")
    re.compile(r'''(?:from|require)\s*\(?\s*['"]([^'"]+)['"]''', re.MULTILINE),
    # Go: import "X"
    re.compile(r'^\s*"([^"]+)"', re.MULTILINE),
    # Java: import X.Y.Z;
    re.compile(r'^\s*import\s+([\w.]+);', re.MULTILINE),
    # Rust: use X::Y;
    re.compile(r'^\s*use\s+([\w:]+)', re.MULTILINE),
]


def _extract_imports(chunks):
    """Scan retrieved chunk contents for import statements and return
    a set of module/file names that could be resolved within the repo."""
    modules = set()
    for chunk in chunks:
        text = chunk.content
        for pattern in _IMPORT_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(1)
                # Skip stdlib / third-party looking names
                # Keep only relative-looking or single-word names
                # that are likely internal repo modules
                parts = re.split(r'[./:]+', raw)
                for part in parts:
                    # Filter out very short or obvious stdlib names
                    if len(part) > 2 and part not in (
                        'os', 'sys', 'json', 'math', 'typing', 're',
                        'abc', 'enum', 'time', 'datetime', 'logging',
                        'collections', 'functools', 'pathlib',
                        'react', 'vue', 'angular', 'express',
                        'http', 'fmt', 'log', 'strings', 'context',
                    ):
                        modules.add(part)
    return modules


def _retrieve_with_dependencies(query, repo_id, top_k=10):
    """Two-pass retrieval: initial search + dependency expansion.

    Pass 1: Standard hybrid retrieval for the query.
    Pass 2: Extract imports from Pass 1 results, then do a targeted
            keyword search to pull in the code those imports reference.
    Merge both passes, deduplicating by chunk ID.
    """
    # Pass 1 — normal hybrid retrieval
    pass1_chunks = retrieve(
        query=query, repo_id=repo_id, top_k=top_k
    )

    if not pass1_chunks:
        return pass1_chunks

    # Extract import targets from Pass 1 chunks
    import_targets = _extract_imports(pass1_chunks)
    if not import_targets:
        return pass1_chunks

    print(f"[TwoPass] Extracted imports: {import_targets}")

    # Pass 2 — keyword search for dependency chunks
    # Build a query string from the import targets
    dep_query = " ".join(import_targets)
    pass2_chunks = retrieve(
        query=dep_query,
        repo_id=repo_id,
        top_k=5,
        mode=RetrievalMode.KEYWORD,
    )

    # Merge: pass1 first (higher priority), then pass2 (fill gaps)
    seen_ids = {c.id for c in pass1_chunks}
    merged = list(pass1_chunks)

    for chunk in pass2_chunks:
        if chunk.id not in seen_ids:
            merged.append(chunk)
            seen_ids.add(chunk.id)

    print(f"[TwoPass] Pass1={len(pass1_chunks)}, "
          f"Pass2={len(pass2_chunks)}, Merged={len(merged)}")

    return merged

def trim_chunks(chunks):
    total = 0
    final = []

    for chunk in chunks:

        if total + len(chunk.content) > MAX_CONTEXT_CHARS:
            break

        final.append(chunk)
        total += len(chunk.content)

    return final


def build_context(chunks):
    if not chunks:
        return "No codebase context provided."

    context_parts = []

    for chunk in chunks:

        context_parts.append(
            f"""
FILE: {chunk.file_path}
LINES: {chunk.start_line}-{chunk.end_line}

{chunk.content}
"""
        )

    return "\n\n".join(context_parts)


def _build_messages(query, context, history=None, query_type="implementation"):
    """Build the messages list for the LLM call."""
    if query_type == "casual":
        sys_prompt = CASUAL_SYSTEM_PROMPT
    else:
        sys_prompt = SYSTEM_PROMPT
    
    messages = [
        {
            "role": "system",
            "content": sys_prompt
        }
    ]

    if history:
        messages.extend(history[-6:])

    messages.append({
        "role": "user",
        "content": f"""
Question:
{query}

Context:
{context}
"""
    })
    return messages


def chat_with_repo(repo_id, query, history=None):
    print("Starting retrieval...")

    # 1. Rewrite & Route
    route_result = rewrite_and_route(query, history)
    rewritten_query = route_result["rewritten_query"]
    query_type = route_result["query_type"]

    # 2. Retrieve
    if query_type == "casual":
        chunks = []
        print("Casual query, skipping retrieval.")
    elif query_type in ("overview", "architecture"):
        chunks = retrieve_diverse(
            query=rewritten_query,
            repo_id=repo_id,
            top_k=15,
            max_per_file=2
        )
    else:
        chunks = _retrieve_with_dependencies(
            query=rewritten_query,
            repo_id=repo_id,
            top_k=10
        )
        
    print(f"Retrieved {len(chunks)} chunks")
    chunks = trim_chunks(chunks)
    print("Building context...")
    context = build_context(chunks)
    print("Calling LLM...")
    
    # 3. Build messages with original query for the conversation, but context from rewritten
    messages = _build_messages(query, context, history, query_type)

    answer = generate_response(messages)
    print("LLM response received")
    return {
        "answer": answer,
        "sources": [
            {
                "file": c.file_path,
                "start": c.start_line,
                "end": c.end_line,
                "content": c.content
            }
            for c in chunks
        ]
    }


def chat_with_repo_stream(repo_id, query, history=None):
    """Generator that yields SSE-formatted events for streaming chat."""
    print("Starting retrieval (streaming)...")

    # 1. Rewrite & Route
    route_result = rewrite_and_route(query, history)
    rewritten_query = route_result["rewritten_query"]
    query_type = route_result["query_type"]

    # 2. Retrieve
    if query_type == "casual":
        chunks = []
        print("Casual query, skipping retrieval.")
    elif query_type in ("overview", "architecture"):
        chunks = retrieve_diverse(
            query=rewritten_query,
            repo_id=repo_id,
            top_k=15,
            max_per_file=2
        )
    else:
        chunks = _retrieve_with_dependencies(
            query=rewritten_query,
            repo_id=repo_id,
            top_k=10
        )
        
    print(f"Retrieved {len(chunks)} chunks")
    chunks = trim_chunks(chunks)
    context = build_context(chunks)

    # Send sources first
    sources = [
        {
            "file": c.file_path,
            "start": c.start_line,
            "end": c.end_line,
            "content": c.content
        }
        for c in chunks
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # Stream the LLM response token by token
    messages = _build_messages(query, context, history, query_type)

    for token in generate_response_stream(messages):
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    # Signal stream is done
    yield f"data: {json.dumps({'type': 'done'})}\n\n"