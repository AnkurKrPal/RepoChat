"""
Embedding Provider — Strategy pattern for swappable embedding backends.

Switch providers by setting the EMBEDDING_PROVIDER env var:
    - "sentence_transformer"  → local SentenceTransformer (heavy, no API key needed)
    - "gemini"                → Google Gemini Embedding API (lightweight, needs EMBEDDING_API_KEY)

Adding a new provider:
    1. Subclass EmbeddingProvider
    2. Add it to _REGISTRY
"""

import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):
    """Common interface every embedding backend must implement."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape (n, dim) as numpy array."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns shape (1, dim) as numpy array."""
        ...


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------

class SentenceTransformerProvider(EmbeddingProvider):
    """Local SentenceTransformer model (all-MiniLM-L6-v2, 384-dim)."""

    def __init__(self):
        # Lazy import so PyTorch is never loaded when using a different provider
        from sentence_transformers import SentenceTransformer

        print("Loading SentenceTransformer model...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode([text])


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini gemini-embedding-2 via REST API (384-dim)."""

    _API_URL = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-embedding-2:embedContent"
    )
    _BATCH_URL = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-embedding-2:batchEmbedContents"
    )
    _DIM = 384
    _BATCH_SIZE = 100  # Gemini batch limit

    def __init__(self):
        import requests as _req  # ensure available at init time

        self._requests = _req
        self._api_key = os.environ.get("EMBEDDING_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "EMBEDDING_API_KEY env var is required when "
                "EMBEDDING_PROVIDER=gemini"
            )
        print("Using Gemini Embedding API")

    # -- public interface ---------------------------------------------------

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch, automatically chunking to respect API limits."""
        all_embeddings = []
        total = len(texts)

        for start in range(0, total, self._BATCH_SIZE):
            batch = texts[start : start + self._BATCH_SIZE]
            print(f"Embedding batch {start}–{start + len(batch)} / {total}")
            embeddings = self._batch_embed(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        embedding = self._single_embed(text)
        return np.array([embedding])

    # -- private helpers ----------------------------------------------------

    def _single_embed(self, text: str) -> List[float]:
        resp = self._requests.post(
            self._API_URL,
            params={"key": self._api_key},
            json={
                "model": "models/gemini-embedding-2",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": self._DIM,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        requests_payload = [
            {
                "model": "models/gemini-embedding-2",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self._DIM,
            }
            for t in texts
        ]

        resp = self._requests.post(
            self._BATCH_URL,
            params={"key": self._api_key},
            json={"requests": requests_payload},
            timeout=60,
        )
        resp.raise_for_status()
        return [
            item["values"]
            for item in resp.json()["embeddings"]
        ]


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    "sentence_transformer": SentenceTransformerProvider,
    "gemini": GeminiEmbeddingProvider,
}

_provider_instance: EmbeddingProvider | None = None


def get_provider() -> EmbeddingProvider:
    """Return the configured embedding provider (singleton)."""
    global _provider_instance

    if _provider_instance is None:
        name = os.environ.get("EMBEDDING_PROVIDER", "sentence_transformer")
        cls = _REGISTRY.get(name)

        if cls is None:
            available = ", ".join(_REGISTRY.keys())
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER={name!r}. "
                f"Available: {available}"
            )

        print(f"Initialising embedding provider: {name}")
        _provider_instance = cls()

    return _provider_instance


def reset_provider():
    """Drop the cached provider (useful for tests or config changes)."""
    global _provider_instance
    _provider_instance = None
