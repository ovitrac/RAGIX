"""
Embedding Backend for Semantic Code Search

Provides interfaces and implementations for generating vector embeddings from code chunks.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from typing import List, Protocol, Optional, Dict, Any
from pathlib import Path
import json
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Protocol for embedding backends."""

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector (list of floats)
        """
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        ...

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Get model identifier."""
        ...


@dataclass
class EmbeddingConfig:
    """Configuration for embedding backend."""

    model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None  # 'cpu', 'cuda', or None for auto
    batch_size: int = 32
    normalize: bool = True
    cache_dir: Optional[str] = None


class SentenceTransformerBackend:
    """
    Embedding backend using sentence-transformers.

    This is the default backend for RAGIX. It provides high-quality
    embeddings while being fast and local-first (no API calls).
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize sentence-transformers backend.

        Args:
            config: Optional configuration. Defaults to all-MiniLM-L6-v2.
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension = None

    def _ensure_model(self):
        """Lazy-load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model: {self.config.model_name}")
        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
            cache_folder=self.config.cache_dir,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self._dimension}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        self._ensure_model()
        embedding = self._model.encode(
            text, normalize_embeddings=self.config.normalize, convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self._ensure_model()

        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )

        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._ensure_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return self.config.model_name


class DummyEmbeddingBackend:
    """
    Dummy backend for testing without sentence-transformers.

    Generates random but deterministic embeddings based on text hash.
    DO NOT USE IN PRODUCTION.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize dummy backend.

        Args:
            dimension: Embedding dimension (default matches all-MiniLM-L6-v2)
        """
        self._dimension = dimension
        logger.warning("Using DummyEmbeddingBackend. Not suitable for production!")

    def embed_text(self, text: str) -> List[float]:
        """Generate deterministic random embedding based on text hash."""
        import hashlib
        import random

        # Use text hash as seed for reproducibility
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        # Generate random unit vector
        vec = [rng.gauss(0, 1) for _ in range(self._dimension)]

        # Normalize to unit length
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch."""
        return [self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return "dummy"


def create_embedding_backend(
    backend_type: str = "sentence-transformers", config: Optional[EmbeddingConfig] = None
) -> EmbeddingBackend:
    """
    Factory function to create embedding backend.

    Args:
        backend_type: Type of backend ('sentence-transformers' or 'dummy')
        config: Optional configuration

    Returns:
        EmbeddingBackend instance

    Raises:
        ValueError: If backend_type is unknown
    """
    if backend_type == "sentence-transformers":
        return SentenceTransformerBackend(config)
    elif backend_type == "dummy":
        dimension = 384
        if config and hasattr(config, "dimension"):
            dimension = config.dimension
        return DummyEmbeddingBackend(dimension)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Choose 'sentence-transformers' or 'dummy'."
        )


@dataclass
class ChunkEmbedding:
    """
    Embedding for a code chunk.

    Links a chunk identifier to its vector embedding and metadata.
    """

    chunk_id: str  # Unique identifier (e.g., "file.py:func_name")
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    embedding: List[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkEmbedding":
        """Deserialize from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            chunk_type=data["chunk_type"],
            name=data["name"],
            embedding=data["embedding"],
            metadata=data["metadata"],
        )


def embed_code_chunks(
    chunks: List[Any], backend: EmbeddingBackend, batch_size: int = 32
) -> List[ChunkEmbedding]:
    """
    Generate embeddings for a list of CodeChunk objects.

    Args:
        chunks: List of CodeChunk objects (from chunking.py)
        backend: Embedding backend to use
        batch_size: Batch size for embedding generation

    Returns:
        List of ChunkEmbedding objects with vectors
    """
    if not chunks:
        return []

    # Prepare texts for embedding
    texts = [chunk.content for chunk in chunks]

    # Generate embeddings in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = backend.embed_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)

    # Create ChunkEmbedding objects
    chunk_embeddings = []
    for chunk, embedding in zip(chunks, all_embeddings):
        # Generate unique ID: file_path:name
        chunk_id = f"{chunk.file_path}:{chunk.name}"

        chunk_emb = ChunkEmbedding(
            chunk_id=chunk_id,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
            name=chunk.name,
            embedding=embedding,
            metadata=chunk.metadata,
        )
        chunk_embeddings.append(chunk_emb)

    return chunk_embeddings


def save_embeddings(embeddings: List[ChunkEmbedding], output_path: Path):
    """
    Save embeddings to JSON file.

    Args:
        embeddings: List of ChunkEmbedding objects
        output_path: Path to output file
    """
    data = {
        "version": "0.6.0",
        "num_embeddings": len(embeddings),
        "dimension": len(embeddings[0].embedding) if embeddings else 0,
        "embeddings": [emb.to_dict() for emb in embeddings],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")


def load_embeddings(input_path: Path) -> List[ChunkEmbedding]:
    """
    Load embeddings from JSON file.

    Args:
        input_path: Path to input file

    Returns:
        List of ChunkEmbedding objects
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = [ChunkEmbedding.from_dict(emb_data) for emb_data in data["embeddings"]]

    logger.info(f"Loaded {len(embeddings)} embeddings from {input_path}")
    return embeddings
