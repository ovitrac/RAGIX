"""
Embedding Backends for Memory Subsystem

Provides:
- MockEmbedder:  Deterministic embeddings for testing (no external deps)
- OllamaEmbedder: Real embeddings via Ollama API (nomic-embed-text etc.)

Both implement the same interface: embed_text() -> List[float],
embed_batch() -> List[List[float]], dimension property, model_name property.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------

class MemoryEmbedder(Protocol):
    """Protocol for memory embedding backends."""

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Model identifier string."""
        ...


# ---------------------------------------------------------------------------
# Mock Embedder (deterministic, for tests)
# ---------------------------------------------------------------------------

class MockEmbedder:
    """
    Deterministic embedding backend using hashing.

    Produces reproducible vectors for any input string.
    No external dependencies — suitable for unit tests.
    """

    def __init__(self, dimension: int = 768, seed: int = 42):
        """Initialize mock embedder with target dimension and deterministic seed."""
        self._dimension = dimension
        self._seed = seed

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return model identifier including dimension and seed."""
        return f"mock-{self._dimension}d-seed{self._seed}"

    def embed_text(self, text: str) -> List[float]:
        """Generate a deterministic pseudo-embedding from text hash."""
        # Hash text with seed for reproducibility
        h = hashlib.sha256(f"{self._seed}:{text}".encode()).digest()
        # Expand hash to fill dimension using repeated hashing
        raw = bytearray()
        block = h
        while len(raw) < self._dimension * 4:
            raw.extend(block)
            block = hashlib.sha256(block).digest()
        # Convert to floats in [-1, 1], then L2-normalize
        floats = []
        for i in range(self._dimension):
            val = struct.unpack_from("f", raw, i * 4)[0]
            # Clamp to reasonable range
            val = max(-1e6, min(1e6, val))
            floats.append(val)
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in floats)) or 1.0
        return [x / norm for x in floats]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts (sequential, deterministic)."""
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Cosine similarity (shared utility)
# ---------------------------------------------------------------------------

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Ollama Embedder (real backend)
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """
    Embedding backend using Ollama API.

    Requires a running Ollama instance with an embedding model
    (e.g. nomic-embed-text, mxbai-embed-large).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dimension: int = 768,
        base_url: str = "http://localhost:11434",
    ):
        """Initialize Ollama embedder with model name, dimension, and API URL."""
        self._model = model
        self._dimension = dimension
        self._base_url = base_url.rstrip("/")
        self._available: Optional[bool] = None

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension (may update after first call)."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return Ollama model identifier."""
        return self._model

    def _check_available(self) -> bool:
        """Check if Ollama is reachable."""
        if self._available is not None:
            return self._available
        try:
            import requests
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        if not self._available:
            logger.warning(f"Ollama not available at {self._base_url}")
        return self._available

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding via Ollama /api/embed endpoint."""
        if not self._check_available():
            raise RuntimeError(f"Ollama not available at {self._base_url}")
        import requests
        resp = requests.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"embeddings": [[...]]}
        embeddings = data.get("embeddings", [])
        if not embeddings or not embeddings[0]:
            raise RuntimeError(f"Empty embedding response from Ollama for model {self._model}")
        vec = embeddings[0]
        # Update dimension if different
        if len(vec) != self._dimension:
            logger.info(f"Updating dimension from {self._dimension} to {len(vec)}")
            self._dimension = len(vec)
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding (sequential calls to Ollama)."""
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Sentence Transformer Embedder (local, no Ollama needed)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """
    Embedding backend using sentence-transformers library.

    Loads a model locally — no external API needed.
    Requires: pip install sentence-transformers
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        """Initialize sentence-transformer embedder with model name and dimension."""
        self._model_name = model
        self._dimension = dimension
        self._model = None  # lazy load

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            # Auto-detect dimension from first embedding
            test_vec = self._model.encode(["test"], normalize_embeddings=True)
            self._dimension = test_vec.shape[1]
            logger.info(
                f"SentenceTransformerEmbedder loaded: {self._model_name} "
                f"(dim={self._dimension})"
            )
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension (auto-detected on first load)."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return sentence-transformers model identifier."""
        return self._model_name

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._load()
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedder(
    backend: str = "mock",
    model: str = "nomic-embed-text",
    dimension: int = 768,
    ollama_url: str = "http://localhost:11434",
    mock_seed: int = 42,
) -> MemoryEmbedder:
    """Create an embedder instance from config parameters."""
    if backend == "mock":
        return MockEmbedder(dimension=dimension, seed=mock_seed)
    elif backend == "ollama":
        return OllamaEmbedder(model=model, dimension=dimension, base_url=ollama_url)
    elif backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model=model, dimension=dimension)
    else:
        raise ValueError(f"Unknown embedder backend: {backend!r}")
