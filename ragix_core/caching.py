"""
Caching - LLM response caching and semantic deduplication

Provides caching strategies for LLM responses to reduce latency
and API costs for repeated or similar queries.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached response entry."""

    key: str
    query: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry by key."""
        pass

    @abstractmethod
    def set(self, entry: CacheEntry):
        """Store entry."""
        pass

    @abstractmethod
    def delete(self, key: str):
        """Delete entry by key."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all entries."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Number of entries."""
        pass


class InMemoryCache(CacheBackend):
    """
    In-memory LRU cache with size and TTL limits.

    Thread-safe implementation using OrderedDict.
    """

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = 3600):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds (None = no expiry)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry, moving to end (most recently used)."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.touch()
            return entry

    def set(self, entry: CacheEntry):
        """Store entry, evicting oldest if at capacity."""
        with self._lock:
            # Set default TTL if not specified
            if entry.ttl is None:
                entry.ttl = self.default_ttl

            # Remove existing entry if present
            if entry.key in self._cache:
                del self._cache[entry.key]

            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[entry.key] = entry

    def delete(self, key: str):
        """Delete entry by key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Number of entries."""
        return len(self._cache)

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        removed = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                removed += 1
        return removed


class DiskCache(CacheBackend):
    """
    Disk-based cache using JSON files.

    Each entry is stored as a separate file for simplicity.
    """

    def __init__(self, cache_dir: Path, max_size: int = 10000):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of entries
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index file for metadata
        self.index_path = cache_dir / "cache_index.json"
        self._index: Dict[str, float] = {}  # key -> created_at
        self._load_index()

    def _load_index(self):
        """Load index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}

    def _save_index(self):
        """Save index to disk."""
        try:
            with open(self.index_path, "w") as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from disk."""
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)

            entry = CacheEntry(
                key=data["key"],
                query=data["query"],
                response=data["response"],
                metadata=data.get("metadata", {}),
                created_at=data["created_at"],
                accessed_at=data.get("accessed_at", data["created_at"]),
                access_count=data.get("access_count", 0),
                ttl=data.get("ttl"),
            )

            # Check expiration
            if entry.is_expired():
                self.delete(key)
                return None

            # Update access time
            entry.touch()
            self.set(entry)  # Save updated access info

            return entry

        except Exception as e:
            logger.warning(f"Failed to load cache entry {key}: {e}")
            return None

    def set(self, entry: CacheEntry):
        """Store entry to disk."""
        # Evict if at capacity
        while len(self._index) >= self.max_size:
            oldest_key = min(self._index, key=self._index.get)
            self.delete(oldest_key)

        path = self._key_to_path(entry.key)
        try:
            data = {
                "key": entry.key,
                "query": entry.query,
                "response": entry.response,
                "metadata": entry.metadata,
                "created_at": entry.created_at,
                "accessed_at": entry.accessed_at,
                "access_count": entry.access_count,
                "ttl": entry.ttl,
            }
            with open(path, "w") as f:
                json.dump(data, f)

            self._index[entry.key] = entry.created_at
            self._save_index()

        except Exception as e:
            logger.warning(f"Failed to save cache entry {entry.key}: {e}")

    def delete(self, key: str):
        """Delete entry from disk."""
        path = self._key_to_path(key)
        try:
            if path.exists():
                path.unlink()
            if key in self._index:
                del self._index[key]
                self._save_index()
        except Exception as e:
            logger.warning(f"Failed to delete cache entry {key}: {e}")

    def clear(self):
        """Clear all entries."""
        for key in list(self._index.keys()):
            self.delete(key)

    def size(self) -> int:
        """Number of entries."""
        return len(self._index)


class LLMCache:
    """
    High-level LLM response cache with semantic similarity support.

    Features:
    - Exact match caching by query hash
    - Optional semantic similarity matching
    - TTL and size limits
    - Statistics tracking
    """

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        embedding_fn: Optional[callable] = None,
        similarity_threshold: float = 0.95,
    ):
        """
        Initialize LLM cache.

        Args:
            backend: Cache backend (default: InMemoryCache)
            embedding_fn: Function to compute embeddings for semantic matching
            similarity_threshold: Minimum similarity for semantic cache hit
        """
        self.backend = backend or InMemoryCache()
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold

        # Statistics
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0

        # Embedding cache for semantic matching
        self._embeddings: Dict[str, List[float]] = {}

    def _compute_key(self, query: str, context: Optional[str] = None) -> str:
        """Compute cache key from query and context."""
        content = query
        if context:
            content = f"{context}|||{query}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(
        self,
        query: str,
        context: Optional[str] = None,
        use_semantic: bool = True,
    ) -> Optional[str]:
        """
        Get cached response for query.

        Args:
            query: The query string
            context: Optional context (e.g., system prompt)
            use_semantic: Try semantic matching if exact match fails

        Returns:
            Cached response or None
        """
        # Try exact match first
        key = self._compute_key(query, context)
        entry = self.backend.get(key)

        if entry is not None:
            self.hits += 1
            logger.debug(f"Cache hit (exact): {key}")
            return entry.response

        # Try semantic matching if enabled
        if use_semantic and self.embedding_fn and self._embeddings:
            similar_key = self._find_similar(query)
            if similar_key:
                entry = self.backend.get(similar_key)
                if entry is not None:
                    self.hits += 1
                    self.semantic_hits += 1
                    logger.debug(f"Cache hit (semantic): {similar_key}")
                    return entry.response

        self.misses += 1
        return None

    def set(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
    ):
        """
        Cache a response.

        Args:
            query: The query string
            response: The response to cache
            context: Optional context
            metadata: Optional metadata to store
            ttl: Time to live (uses backend default if not specified)
        """
        key = self._compute_key(query, context)

        entry = CacheEntry(
            key=key,
            query=query,
            response=response,
            metadata=metadata or {},
            ttl=ttl,
        )

        self.backend.set(entry)

        # Store embedding for semantic matching
        if self.embedding_fn:
            try:
                embedding = self.embedding_fn(query)
                self._embeddings[key] = embedding
            except Exception as e:
                logger.warning(f"Failed to compute embedding: {e}")

        logger.debug(f"Cached response: {key}")

    def _find_similar(self, query: str) -> Optional[str]:
        """Find semantically similar cached query."""
        if not self.embedding_fn or not self._embeddings:
            return None

        try:
            query_embedding = self.embedding_fn(query)

            best_key = None
            best_similarity = 0.0

            for key, embedding in self._embeddings.items():
                similarity = self._cosine_similarity(query_embedding, embedding)
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_key = key

            return best_key

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def invalidate(self, query: str, context: Optional[str] = None):
        """Invalidate a specific cache entry."""
        key = self._compute_key(query, context)
        self.backend.delete(key)
        if key in self._embeddings:
            del self._embeddings[key]

    def clear(self):
        """Clear all cached entries."""
        self.backend.clear()
        self._embeddings.clear()
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        return {
            "size": self.backend.size(),
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "hit_rate": self.hit_rate,
        }


class ToolResultCache:
    """
    Cache for tool execution results.

    Caches results of deterministic tools (like file reads)
    to avoid repeated execution.
    """

    def __init__(self, max_size: int = 500, ttl: float = 300):
        """
        Initialize tool result cache.

        Args:
            max_size: Maximum cached results
            ttl: Time to live in seconds (default 5 minutes)
        """
        self.cache = InMemoryCache(max_size=max_size, default_ttl=ttl)

        # Tools that can be cached (deterministic, read-only)
        self.cacheable_tools = {
            "read_file",
            "grep_search",
            "find_files",
            "list_directory",
            "git_status",
            "git_log",
            "project_overview",
        }

    def _compute_key(self, tool: str, args: Dict[str, Any]) -> str:
        """Compute cache key from tool and arguments."""
        content = f"{tool}:{json.dumps(args, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_cacheable(self, tool: str) -> bool:
        """Check if tool results can be cached."""
        return tool in self.cacheable_tools

    def get(self, tool: str, args: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for tool call."""
        if not self.is_cacheable(tool):
            return None

        key = self._compute_key(tool, args)
        entry = self.cache.get(key)

        if entry is not None:
            return json.loads(entry.response)
        return None

    def set(self, tool: str, args: Dict[str, Any], result: Any):
        """Cache tool result."""
        if not self.is_cacheable(tool):
            return

        key = self._compute_key(tool, args)
        entry = CacheEntry(
            key=key,
            query=f"{tool}:{args}",
            response=json.dumps(result),
            metadata={"tool": tool},
        )
        self.cache.set(entry)

    def invalidate_file(self, file_path: str):
        """Invalidate cache entries related to a file."""
        # For now, just clear all - more sophisticated invalidation
        # would require tracking file dependencies
        self.cache.clear()

    def clear(self):
        """Clear all cached results."""
        self.cache.clear()


def create_llm_cache(
    cache_type: str = "memory",
    cache_dir: Optional[Path] = None,
    max_size: int = 1000,
    ttl: float = 3600,
    embedding_fn: Optional[callable] = None,
) -> LLMCache:
    """
    Create an LLM cache with specified backend.

    Args:
        cache_type: "memory" or "disk"
        cache_dir: Directory for disk cache
        max_size: Maximum entries
        ttl: Time to live in seconds
        embedding_fn: Function for semantic matching

    Returns:
        Configured LLMCache
    """
    if cache_type == "disk":
        if cache_dir is None:
            cache_dir = Path(".ragix/cache")
        backend = DiskCache(cache_dir, max_size)
    else:
        backend = InMemoryCache(max_size, ttl)

    return LLMCache(backend, embedding_fn)
