from retrieval.vector_store import retrieve, retrieve_diverse
from llm.groq_client import generate_response, generate_response_stream
from llm.prompts import SYSTEM_PROMPT, OVERVIEW_SYSTEM_PROMPT, CASUAL_SYSTEM_PROMPT
from llm.query_pipeline import rewrite_and_route
import json

MAX_CONTEXT_CHARS = 12000

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
        chunks = retrieve(
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
        chunks = retrieve(
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