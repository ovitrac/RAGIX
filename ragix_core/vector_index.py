"""
Vector Index for Semantic Code Search

Provides efficient similarity search over code chunk embeddings.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from typing import List, Tuple, Optional, Dict, Any, Protocol
from pathlib import Path
import json
import logging
from dataclasses import dataclass

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from similarity search."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    score: float  # Similarity score (higher = more similar)
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return f"<SearchResult {self.file_path}:{self.name} score={self.score:.3f}>"


class VectorIndex(Protocol):
    """Protocol for vector index implementations."""

    def add(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """Add embeddings to index with associated metadata."""
        ...

    def search(self, query_embedding: List[float], k: int = 10) -> List[SearchResult]:
        """Search for k nearest neighbors."""
        ...

    def save(self, path: Path):
        """Save index to disk."""
        ...

    def load(self, path: Path):
        """Load index from disk."""
        ...

    @property
    def size(self) -> int:
        """Number of vectors in index."""
        ...


class NumpyVectorIndex:
    """
    Simple vector index using NumPy for similarity search.

    Uses cosine similarity and exhaustive search. Suitable for
    small to medium codebases (< 100k chunks). For larger codebases,
    consider FAISSVectorIndex.
    """

    def __init__(self, dimension: int):
        """
        Initialize index.

        Args:
            dimension: Embedding dimension
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy not installed. Install with: pip install numpy")

        self.dimension = dimension
        self.vectors = None  # Will be (N, D) array
        self.metadata = []  # List of metadata dicts

    def add(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """
        Add embeddings to index.

        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dicts (one per embedding)
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")

        if not embeddings:
            return

        # Convert to numpy array
        new_vectors = np.array(embeddings, dtype=np.float32)

        # Validate dimension
        if new_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {new_vectors.shape[1]} "
                f"does not match index dimension {self.dimension}"
            )

        # Normalize for cosine similarity
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        new_vectors = new_vectors / norms

        # Add to index
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        self.metadata.extend(metadata)

        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.size}")

    def search(self, query_embedding: List[float], k: int = 10) -> List[SearchResult]:
        """
        Search for k nearest neighbors using cosine similarity.

        Args:
            query_embedding: Query vector
            k: Number of results to return

        Returns:
            List of SearchResult objects, sorted by score (descending)
        """
        if self.vectors is None or self.size == 0:
            return []

        # Convert query to numpy array and normalize
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(self.vectors, query.T).flatten()

        # Get top-k indices
        k = min(k, self.size)
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Build results
        results = []
        for idx in top_k_indices:
            meta = self.metadata[idx]
            result = SearchResult(
                chunk_id=meta["chunk_id"],
                file_path=meta["file_path"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                chunk_type=meta["chunk_type"],
                name=meta["name"],
                score=float(similarities[idx]),
                metadata=meta.get("metadata", {}),
            )
            results.append(result)

        return results

    def save(self, path: Path):
        """
        Save index to disk.

        Args:
            path: Directory path to save index
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save vectors
        if self.vectors is not None:
            np.save(path / "vectors.npy", self.vectors)

        # Save metadata
        index_data = {
            "dimension": self.dimension,
            "size": self.size,
            "metadata": self.metadata,
        }

        with open(path / "index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Saved index to {path}")

    def load(self, path: Path):
        """
        Load index from disk.

        Args:
            path: Directory path containing saved index
        """
        # Load metadata
        with open(path / "index.json", "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.dimension = index_data["dimension"]
        self.metadata = index_data["metadata"]

        # Load vectors
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            self.vectors = np.load(vectors_path)
        else:
            self.vectors = None

        logger.info(f"Loaded index from {path}. Size: {self.size}")

    @property
    def size(self) -> int:
        """Number of vectors in index."""
        if self.vectors is None:
            return 0
        return self.vectors.shape[0]


class FAISSVectorIndex:
    """
    High-performance vector index using FAISS.

    Suitable for large codebases (100k+ chunks). Uses approximate
    nearest neighbor search with IVF (Inverted File Index).
    """

    def __init__(self, dimension: int, use_gpu: bool = False):
        """
        Initialize FAISS index.

        Args:
            dimension: Embedding dimension
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: "
                "pip install faiss-cpu (or faiss-gpu for GPU support)"
            )

        self.dimension = dimension
        self.use_gpu = use_gpu
        self.metadata = []

        # Create index (flat L2 for simplicity, can be upgraded to IVF)
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)

        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        logger.info(f"Initialized FAISS index (dim={dimension}, gpu={use_gpu})")

    def add(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """Add embeddings to FAISS index."""
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")

        if not embeddings:
            return

        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        # Add to index
        self.index.add(vectors)
        self.metadata.extend(metadata)

        logger.info(f"Added {len(embeddings)} vectors to FAISS index. Total: {self.size}")

    def search(self, query_embedding: List[float], k: int = 10) -> List[SearchResult]:
        """Search using FAISS."""
        if self.size == 0:
            return []

        # Convert and normalize query
        query = np.array([query_embedding], dtype=np.float32)
        import faiss

        faiss.normalize_L2(query)

        # Search
        k = min(k, self.size)
        scores, indices = self.index.search(query, k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue

            meta = self.metadata[idx]
            result = SearchResult(
                chunk_id=meta["chunk_id"],
                file_path=meta["file_path"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                chunk_type=meta["chunk_type"],
                name=meta["name"],
                score=float(score),
                metadata=meta.get("metadata", {}),
            )
            results.append(result)

        return results

    def save(self, path: Path):
        """Save FAISS index to disk."""
        import faiss

        path.mkdir(parents=True, exist_ok=True)

        # Save index (convert from GPU if needed)
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(path / "faiss.index"))
        else:
            faiss.write_index(self.index, str(path / "faiss.index"))

        # Save metadata
        index_data = {
            "dimension": self.dimension,
            "size": self.size,
            "use_gpu": self.use_gpu,
            "metadata": self.metadata,
        }

        with open(path / "index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: Path):
        """Load FAISS index from disk."""
        import faiss

        # Load metadata
        with open(path / "index.json", "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.dimension = index_data["dimension"]
        self.metadata = index_data["metadata"]

        # Load index
        self.index = faiss.read_index(str(path / "faiss.index"))

        # Move to GPU if requested
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        logger.info(f"Loaded FAISS index from {path}. Size: {self.size}")

    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return self.index.ntotal


def create_vector_index(
    dimension: int, backend: str = "numpy", use_gpu: bool = False
) -> VectorIndex:
    """
    Factory function to create vector index.

    Args:
        dimension: Embedding dimension
        backend: 'numpy' or 'faiss'
        use_gpu: Whether to use GPU (FAISS only)

    Returns:
        VectorIndex instance

    Raises:
        ValueError: If backend is unknown
    """
    if backend == "numpy":
        return NumpyVectorIndex(dimension)
    elif backend == "faiss":
        return FAISSVectorIndex(dimension, use_gpu)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'numpy' or 'faiss'.")


def build_index_from_embeddings(
    embeddings: List[Any], backend: str = "numpy", use_gpu: bool = False
) -> VectorIndex:
    """
    Build vector index from ChunkEmbedding objects.

    Args:
        embeddings: List of ChunkEmbedding objects (from embeddings.py)
        backend: 'numpy' or 'faiss'
        use_gpu: Whether to use GPU (FAISS only)

    Returns:
        Populated VectorIndex
    """
    if not embeddings:
        raise ValueError("Cannot build index from empty embeddings list")

    # Get dimension from first embedding
    dimension = len(embeddings[0].embedding)

    # Create index
    index = create_vector_index(dimension, backend, use_gpu)

    # Extract vectors and metadata
    vectors = [emb.embedding for emb in embeddings]
    metadata = [
        {
            "chunk_id": emb.chunk_id,
            "file_path": emb.file_path,
            "start_line": emb.start_line,
            "end_line": emb.end_line,
            "chunk_type": emb.chunk_type,
            "name": emb.name,
            "metadata": emb.metadata,
        }
        for emb in embeddings
    ]

    # Add to index
    index.add(vectors, metadata)

    return index
