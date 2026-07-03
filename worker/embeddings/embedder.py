"""
Embedder — public API for generating embeddings.

Delegates to the configured EmbeddingProvider (see provider.py).
Preserves the original get_model() / generate_embeddings() signatures
so existing callers (vector_store.py, ingest.py) and test mocks keep working.
"""

from worker.embeddings.provider import get_provider


def get_model():
    """Return the active provider.

    The provider exposes .encode([texts]) via embed_query() / embed_texts(),
    but callers in vector_store.py use  model.encode([query]).
    We wrap the provider in a thin adapter so .encode() works seamlessly.
    """
    return _ProviderAdapter(get_provider())


def generate_embeddings(chunks, batch_size=64):
    """Generate embeddings for a list of chunk dicts (must have 'content' key)."""
    provider = get_provider()

    texts = [chunk["content"] for chunk in chunks]

    embeddings = provider.embed_texts(texts)

    return embeddings


class _ProviderAdapter:
    """Thin wrapper that gives every provider a .encode() method,
    matching the SentenceTransformer interface used by vector_store.py."""

    def __init__(self, provider):
        self._provider = provider

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self._provider.embed_query(texts)
        return self._provider.embed_query(texts[0]) if len(texts) == 1 else self._provider.embed_texts(texts)