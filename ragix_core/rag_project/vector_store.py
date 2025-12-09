"""
RAGIX Project RAG - ChromaDB Vector Store

GPU-accelerated vector storage using ChromaDB.

Features:
    - CUDA acceleration when available (auto-detect)
    - Multiple collections (docs, code, mixed)
    - Efficient batch operations
    - Metadata filtering for targeted search

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import os

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    backend: str = "chroma"
    path: str = "chroma"               # Relative to .RAG/
    use_gpu: str = "auto"              # "auto", "always", "never"
    collection_prefix: str = "rag"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    distance_metric: str = "cosine"    # "cosine", "l2", "ip"


@dataclass
class SearchResult:
    """Search result from vector store."""
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience accessors
    @property
    def file_path(self) -> str:
        return self.metadata.get("file_path", "")

    @property
    def line_start(self) -> int:
        return self.metadata.get("line_start", 0)

    @property
    def line_end(self) -> int:
        return self.metadata.get("line_end", 0)

    @property
    def kind(self) -> str:
        return self.metadata.get("kind", "unknown")

    def get_citation(self) -> str:
        """Get citation string for this result."""
        if self.line_start and self.line_end:
            if self.line_start == self.line_end:
                return f"{self.file_path}:{self.line_start}"
            return f"{self.file_path}:{self.line_start}-{self.line_end}"
        return self.file_path


# =============================================================================
# Collection Names
# =============================================================================

class CollectionType:
    """Standard collection types."""
    DOCS = "docs"      # Documentation chunks
    CODE = "code"      # Source code chunks
    MIXED = "mixed"    # All chunks (union)
    CONFIG = "config"  # Configuration files


# =============================================================================
# Vector Store
# =============================================================================

class VectorStore:
    """
    ChromaDB-based vector store for project RAG.

    Provides:
        - GPU-accelerated embeddings (when CUDA available)
        - Multiple collections for different content types
        - Efficient batch add/query operations
        - Metadata filtering
    """

    def __init__(
        self,
        project_root: Path,
        config: Optional[VectorStoreConfig] = None,
    ):
        """
        Initialize vector store.

        Args:
            project_root: Project root directory
            config: Vector store configuration
        """
        self.project_root = project_root
        self.config = config or VectorStoreConfig()
        self.rag_dir = project_root / ".RAG"
        self.chroma_path = self.rag_dir / self.config.path

        # ChromaDB client and collections (lazy init)
        self._client = None
        self._embedding_fn = None
        self._collections: Dict[str, Any] = {}

        # Check GPU availability
        self._gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if GPU acceleration is available."""
        if self.config.use_gpu == "never":
            logger.info("GPU disabled by configuration (use_gpu='never')")
            return False

        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"CUDA available: {gpu_name} ({gpu_mem:.1f} GB)")
            else:
                logger.info("CUDA not available, using CPU for embeddings")
            return available
        except ImportError:
            logger.info("PyTorch not installed, using CPU for embeddings")
            return False

    def _ensure_client(self) -> None:
        """Lazily initialize ChromaDB client."""
        if self._client is not None:
            return

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

        # Create storage directory
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # Initialize client with persistent storage
        # ChromaDB 1.x API: PersistentClient takes just path
        try:
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
            logger.info(f"ChromaDB initialized at {self.chroma_path} (version 1.x)")
        except Exception as e:
            # Fall back to older API if needed
            try:
                from chromadb.config import Settings
                settings = Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.chroma_path),
                    anonymized_telemetry=False,
                )
                self._client = chromadb.Client(settings)
                logger.info(f"ChromaDB initialized at {self.chroma_path} (legacy API)")
            except Exception as e2:
                raise ImportError(f"Failed to initialize ChromaDB: {e}, {e2}")

    def _ensure_embedding_fn(self) -> None:
        """Lazily initialize embedding function."""
        if self._embedding_fn is not None:
            return

        try:
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError(
                "ChromaDB embedding functions not available. "
                "Install with: pip install chromadb sentence-transformers"
            )

        # Use sentence-transformers with GPU if available
        device = "cuda" if self._gpu_available and self.config.use_gpu != "never" else "cpu"

        # Try loading embedding model with multiple fallback strategies
        model_candidates = [
            self.config.embedding_model,
            "all-MiniLM-L6-v2",  # Short name (may work with some versions)
            "sentence-transformers/all-MiniLM-L6-v2",  # Full HF path
        ]

        for model_name in model_candidates:
            try:
                # Suppress HuggingFace warnings during model loading
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", message=".*additional_chat_templates.*")

                    self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=model_name,
                        device=device,
                    )
                logger.info(f"Embeddings: {model_name} on {device}")
                return
            except Exception as e:
                logger.debug(f"Failed to load {model_name}: {e}")
                continue

        # Final fallback: use ChromaDB's default embedding function
        logger.warning(f"Could not load sentence-transformers model, using ChromaDB default")
        try:
            self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            logger.info("Using ChromaDB default embedding function")
        except Exception as e:
            logger.error(f"Failed to initialize any embedding function: {e}")
            raise RuntimeError(
                f"No embedding function available. Install sentence-transformers: "
                f"pip install sentence-transformers"
            )

    def _get_collection(self, collection_type: str) -> Any:
        """Get or create a collection."""
        self._ensure_client()
        self._ensure_embedding_fn()

        collection_name = f"{self.config.collection_prefix}_{collection_type}"

        if collection_name not in self._collections:
            try:
                # Try to get existing collection
                self._collections[collection_name] = self._client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self._embedding_fn,
                    metadata={"hnsw:space": self.config.distance_metric},
                )
            except Exception as e:
                logger.error(f"Failed to get/create collection {collection_name}: {e}")
                raise

        return self._collections[collection_name]

    # -------------------------------------------------------------------------
    # Add Operations
    # -------------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_type: str = CollectionType.MIXED,
    ) -> int:
        """
        Add chunks to vector store.

        Args:
            chunks: List of chunk dictionaries with:
                - chunk_id: Unique identifier
                - content: Text content
                - metadata: Dict with file_path, line_start, line_end, kind, tags
            collection_type: Which collection to add to

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        collection = self._get_collection(collection_type)

        # Prepare batch
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")

            if not chunk_id or not content:
                continue

            ids.append(chunk_id)
            documents.append(content)

            # Prepare metadata (ChromaDB requires flat dict with primitive values)
            meta = {
                "file_path": chunk.get("file_path", ""),
                "file_id": chunk.get("file_id", ""),
                "line_start": chunk.get("line_start", 0),
                "line_end": chunk.get("line_end", 0),
                "kind": chunk.get("kind", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
            }

            # Add tags as comma-separated string (ChromaDB doesn't support lists)
            tags = chunk.get("tags", [])
            if tags:
                meta["tags"] = ",".join(str(t) for t in tags)

            metadatas.append(meta)

        if not ids:
            return 0

        # Add in batches
        batch_size = self.config.embedding_batch_size
        added = 0

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                )
                added += len(batch_ids)
            except Exception as e:
                logger.error(f"Failed to add batch {i // batch_size}: {e}")

        return added

    def add_to_multiple_collections(
        self,
        chunks: List[Dict[str, Any]],
        collection_types: List[str],
    ) -> Dict[str, int]:
        """
        Add chunks to multiple collections.

        Args:
            chunks: Chunks to add
            collection_types: List of collection types

        Returns:
            Dict mapping collection type to count added
        """
        results = {}
        for ctype in collection_types:
            results[ctype] = self.add_chunks(chunks, ctype)
        return results

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        collection_type: str = CollectionType.MIXED,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Query vector store.

        Args:
            query_text: Search query
            top_k: Number of results to return
            collection_type: Which collection to search
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        collection = self._get_collection(collection_type)

        # Build where clause from filters
        where = None
        if filters:
            where = self._build_where_clause(filters)

        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

        # Parse results
        search_results = []

        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, chunk_id in enumerate(ids):
                # Convert distance to similarity score (for cosine)
                distance = distances[i] if i < len(distances) else 0
                score = 1.0 - distance  # Cosine distance to similarity

                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    content=documents[i] if i < len(documents) else "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                ))

        return search_results

    def query_by_file(
        self,
        query_text: str,
        file_path: str,
        top_k: int = 10,
        collection_type: str = CollectionType.MIXED,
    ) -> List[SearchResult]:
        """Query within a specific file."""
        return self.query(
            query_text=query_text,
            top_k=top_k,
            collection_type=collection_type,
            filters={"file_path": file_path},
        )

    def query_by_kind(
        self,
        query_text: str,
        kind: str,
        top_k: int = 10,
        collection_type: str = CollectionType.MIXED,
    ) -> List[SearchResult]:
        """Query chunks of a specific kind (e.g., code_java, doc_markdown)."""
        return self.query(
            query_text=query_text,
            top_k=top_k,
            collection_type=collection_type,
            filters={"kind": kind},
        )

    def _build_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict]:
        """Build ChromaDB where clause from filters."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append({key: {"$eq": value}})
            elif isinstance(value, (int, float)):
                conditions.append({key: {"$eq": value}})
            elif isinstance(value, list):
                conditions.append({key: {"$in": value}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None

    # -------------------------------------------------------------------------
    # Delete Operations
    # -------------------------------------------------------------------------

    def delete_by_file(
        self,
        file_id: str,
        collection_type: str = CollectionType.MIXED,
    ) -> int:
        """
        Delete all chunks for a file.

        Args:
            file_id: File ID to delete chunks for
            collection_type: Collection to delete from

        Returns:
            Number of chunks deleted (approximate)
        """
        collection = self._get_collection(collection_type)

        try:
            # Get count before delete
            before = collection.count()

            collection.delete(
                where={"file_id": {"$eq": file_id}}
            )

            after = collection.count()
            return before - after

        except Exception as e:
            logger.error(f"Delete failed for file {file_id}: {e}")
            return 0

    def delete_by_ids(
        self,
        chunk_ids: List[str],
        collection_type: str = CollectionType.MIXED,
    ) -> int:
        """Delete specific chunks by ID."""
        if not chunk_ids:
            return 0

        collection = self._get_collection(collection_type)

        try:
            collection.delete(ids=chunk_ids)
            return len(chunk_ids)
        except Exception as e:
            logger.error(f"Delete by IDs failed: {e}")
            return 0

    def clear_collection(self, collection_type: str) -> bool:
        """Clear all data from a collection."""
        self._ensure_client()
        collection_name = f"{self.config.collection_prefix}_{collection_type}"

        try:
            self._client.delete_collection(name=collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
            logger.info(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {e}")
            return False

    def clear_all(self) -> bool:
        """Clear all collections."""
        success = True
        for ctype in [CollectionType.DOCS, CollectionType.CODE, CollectionType.MIXED, CollectionType.CONFIG]:
            if not self.clear_collection(ctype):
                success = False
        return success

    # -------------------------------------------------------------------------
    # Stats & Info
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        self._ensure_client()

        stats = {
            "path": str(self.chroma_path),
            "gpu_available": self._gpu_available,
            "embedding_model": self.config.embedding_model,
            "collections": {},
        }

        for ctype in [CollectionType.DOCS, CollectionType.CODE, CollectionType.MIXED, CollectionType.CONFIG]:
            try:
                collection = self._get_collection(ctype)
                stats["collections"][ctype] = {
                    "count": collection.count(),
                }
            except Exception:
                stats["collections"][ctype] = {"count": 0, "error": "not available"}

        # Disk size
        if self.chroma_path.exists():
            total_size = sum(
                f.stat().st_size for f in self.chroma_path.rglob("*") if f.is_file()
            )
            stats["disk_size_bytes"] = total_size
            stats["disk_size_mb"] = round(total_size / (1024 * 1024), 2)

        return stats

    def get_collection_count(self, collection_type: str = CollectionType.MIXED) -> int:
        """Get number of chunks in a collection."""
        try:
            collection = self._get_collection(collection_type)
            return collection.count()
        except Exception:
            return 0

    def is_initialized(self) -> bool:
        """Check if vector store has been initialized."""
        return self.chroma_path.exists() and any(self.chroma_path.iterdir())

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def persist(self) -> None:
        """Persist changes to disk (if using in-memory mode)."""
        if self._client is not None:
            try:
                # Newer ChromaDB versions auto-persist
                if hasattr(self._client, "persist"):
                    self._client.persist()
            except Exception as e:
                logger.warning(f"Persist call failed (may be auto-persisted): {e}")

    def close(self) -> None:
        """Close connections and cleanup."""
        self.persist()
        self._client = None
        self._collections = {}
        self._embedding_fn = None
