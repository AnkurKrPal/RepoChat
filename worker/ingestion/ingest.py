from worker.ingestion.repo_loader import clone_repo
from worker.ingestion.file_walker import get_files
from backend.database import SessionLocal
from backend.models import Repo, Chunk
from worker.ingestion.chunker import (
    chunk_code_file,
    chunk_markdown_file
)

from worker.embeddings.embedder import generate_embeddings
from retrieval.vector_store import store_chunks
from retrieval.schema import ChunkSchema, ChunkType

import json
import os
import shutil
from collections import defaultdict


def _build_repo_overview(all_chunks, repo_path):
    """Build a signature-enhanced repository structure chunk.

    Groups every file's function/class signatures into a compact tree
    so the LLM can understand *which file does what* at a glance.
    """
    # Group signatures by file
    file_signatures = defaultdict(list)
    files_seen = set()

    for chunk in all_chunks:
        file_path = chunk.get("file", "")
        files_seen.add(file_path)

        sig = chunk.get("signature", "").strip()
        chunk_type = chunk.get("type", "")

        # Only collect meaningful signatures (functions, classes, methods)
        if sig and chunk_type not in ("plaintext", "markdown", ""):
            # Clean up the signature — first line only, no body
            short_sig = sig.split("\n")[0].strip()
            if short_sig and short_sig not in file_signatures[file_path]:
                file_signatures[file_path].append(short_sig)

    # Build the tree text
    lines = ["REPOSITORY STRUCTURE", "=" * 40, ""]

    # Sort files for consistent ordering
    for file_path in sorted(files_seen):
        # Make path relative to repo root for readability
        rel_path = os.path.relpath(file_path, repo_path)

        sigs = file_signatures.get(file_path, [])
        if sigs:
            lines.append(f"📄 {rel_path}")
            for sig in sigs:
                lines.append(f"   ├─ {sig}")
            lines.append("")
        else:
            lines.append(f"📄 {rel_path}")
            lines.append("")

    overview_text = "\n".join(lines)

    # Cap at ~3000 chars to avoid bloating embeddings
    if len(overview_text) > 3000:
        overview_text = overview_text[:2950] + "\n... (truncated)"

    return {
        "file": "REPO_OVERVIEW",
        "type": "repository_structure",
        "signature": "repository_structure",
        "start_line": 0,
        "end_line": 0,
        "content": overview_text,
    }


def ingest_repo(repo_id: int):
    db = SessionLocal()
    repo = None
    try:
        # Find repository row
        repo = db.query(Repo).filter(Repo.id == repo_id).first()

        if not repo:
            raise Exception("Repository not found")

        # Read GitHub URL
        repo_url = repo.github_url

        # Update status
        repo.status = "indexing"
        db.commit()

        # clone repo
        repo_path = clone_repo(repo_url)

        # get all files
        files = get_files(repo_path)

        all_chunks = []
        chunk_objects = []
        chunk_index = 0

        # process files
        for file in files:

            print(f"Processing: {file}")

            if file.endswith((".md", ".rst", ".txt")):

                chunks = chunk_markdown_file(file)

            else:

                chunks = chunk_code_file(file)

            for chunk in chunks:

                all_chunks.append(chunk)

                chunk_obj = Chunk(
                    repo_id=repo.id,
                    chunk_index=chunk_index,
                    file_path=chunk.get("file", ""),
                    chunk_type=chunk.get("type", ""),
                    signature=chunk.get("signature"),
                    start_line=chunk.get("start_line"),
                    end_line=chunk.get("end_line"),
                    content=chunk["content"]
                )

                chunk_objects.append(chunk_obj)

                chunk_index += 1

        print(f"\nTotal file chunks: {len(all_chunks)}")

        # Build and inject the repo overview chunk
        overview_chunk = _build_repo_overview(all_chunks, repo_path)
        all_chunks.append(overview_chunk)

        overview_obj = Chunk(
            repo_id=repo.id,
            chunk_index=chunk_index,
            file_path=overview_chunk["file"],
            chunk_type=overview_chunk["type"],
            signature=overview_chunk["signature"],
            start_line=overview_chunk["start_line"],
            end_line=overview_chunk["end_line"],
            content=overview_chunk["content"],
        )
        chunk_objects.append(overview_obj)
        chunk_index += 1

        print(f"Total chunks (with overview): {len(all_chunks)}")

        print("\nSaving chunks to Postgres...")

        db.add_all(chunk_objects)

        db.commit()

        print("Chunks saved")
        repo.total_files = len(files)
        repo.total_chunks = len(all_chunks)

        db.commit()
        # save chunks locally
        os.makedirs("chunks", exist_ok=True)

        with open("chunks/chunks.json", "w") as f:

            json.dump(all_chunks, f, indent=2)

        # generate embeddings
        print("\nGenerating embeddings...")

        embeddings = generate_embeddings(all_chunks)

        # store in chromadb
        print("\nStoring in ChromaDB...")

        schema_chunks = []
        for db_chunk in chunk_objects:
            # Safely parse the chunk type, fallback to PLAINTEXT if it doesn't strictly match the enum
            try:
                c_type = ChunkType(db_chunk.chunk_type)
            except ValueError:
                c_type = ChunkType.PLAINTEXT
                
            schema_chunks.append(ChunkSchema(
                id=str(db_chunk.id),  # This pulls the UUID generated by Postgres
                content=str(db_chunk.content),
                file_path=str(db_chunk.file_path),
                chunk_type=c_type,
                signature=str(db_chunk.signature),
                start_line=str(db_chunk.start_line),
                end_line=str(db_chunk.end_line)
            ))

        store_chunks(schema_chunks, embeddings, str(repo.id))

        # delete cloned repo
        print("\nDeleting cloned repository...")

        shutil.rmtree(repo_path)

        print("Repository deleted")

        print("\nDone")

        repo.status = "ready"
        db.commit()

    except Exception as e:
        if 'repo' in locals() and repo:
            repo.status = "error"
            repo.error_message = str(e)
            db.commit()
        raise

    finally:
        db.close()
