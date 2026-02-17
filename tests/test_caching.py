"""
Tests for Caching System

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import time
import pytest
from pathlib import Path

from ragix_core.caching import (
    CacheEntry,
    InMemoryCache,
    DiskCache,
    LLMCache,
    ToolResultCache,
    create_llm_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test_key",
            query="test query",
            response="test response",
        )
        assert entry.key == "test_key"
        assert entry.query == "test query"
        assert entry.response == "test response"
        assert entry.access_count == 0

    def test_entry_not_expired_without_ttl(self):
        """Test entry without TTL never expires."""
        entry = CacheEntry(
            key="test_key",
            query="test query",
            response="test response",
            ttl=None,
        )
        assert not entry.is_expired()

    def test_entry_expired_with_ttl(self):
        """Test entry expires after TTL."""
        entry = CacheEntry(
            key="test_key",
            query="test query",
            response="test response",
            ttl=0.01,  # 10ms
            created_at=time.time() - 1,  # Created 1 second ago
        )
        assert entry.is_expired()

    def test_entry_touch(self):
        """Test touch updates access time and count."""
        entry = CacheEntry(
            key="test_key",
            query="test query",
            response="test response",
        )
        initial_time = entry.accessed_at
        initial_count = entry.access_count

        time.sleep(0.01)
        entry.touch()

        assert entry.accessed_at > initial_time
        assert entry.access_count == initial_count + 1


class TestInMemoryCache:
    """Tests for InMemoryCache backend."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = InMemoryCache(max_size=100)
        entry = CacheEntry(
            key="key1",
            query="query1",
            response="response1",
        )
        cache.set(entry)

        result = cache.get("key1")
        assert result is not None
        assert result.response == "response1"

    def test_get_missing_key(self):
        """Test getting non-existent key returns None."""
        cache = InMemoryCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        cache = InMemoryCache(max_size=3)

        # Add 3 entries
        for i in range(3):
            cache.set(CacheEntry(
                key=f"key{i}",
                query=f"query{i}",
                response=f"response{i}",
            ))

        assert cache.size() == 3

        # Access key0 to make it recently used
        cache.get("key0")

        # Add a 4th entry - should evict key1 (least recently used)
        cache.set(CacheEntry(
            key="key3",
            query="query3",
            response="response3",
        ))

        assert cache.size() == 3
        assert cache.get("key0") is not None  # Still present (accessed)
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None  # Still present
        assert cache.get("key3") is not None  # New entry

    def test_delete(self):
        """Test deleting an entry."""
        cache = InMemoryCache()
        cache.set(CacheEntry(
            key="key1",
            query="query1",
            response="response1",
        ))

        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        """Test clearing all entries."""
        cache = InMemoryCache()
        for i in range(5):
            cache.set(CacheEntry(
                key=f"key{i}",
                query=f"query{i}",
                response=f"response{i}",
            ))

        assert cache.size() == 5
        cache.clear()
        assert cache.size() == 0

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = InMemoryCache(default_ttl=0.05)  # 50ms

        cache.set(CacheEntry(
            key="key1",
            query="query1",
            response="response1",
        ))

        # Should be available immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(0.1)

        # Should be expired
        assert cache.get("key1") is None

    def test_cleanup_expired(self):
        """Test cleanup_expired method."""
        cache = InMemoryCache(default_ttl=0.01)

        for i in range(5):
            cache.set(CacheEntry(
                key=f"key{i}",
                query=f"query{i}",
                response=f"response{i}",
            ))

        assert cache.size() == 5

        time.sleep(0.05)
        removed = cache.cleanup_expired()

        assert removed == 5
        assert cache.size() == 0


class TestDiskCache:
    """Tests for DiskCache backend."""

    def test_set_and_get(self, temp_dir: Path):
        """Test basic set and get with disk cache."""
        cache_dir = temp_dir / "cache"
        cache = DiskCache(cache_dir)

        entry = CacheEntry(
            key="key1",
            query="query1",
            response="response1",
        )
        cache.set(entry)

        result = cache.get("key1")
        assert result is not None
        assert result.response == "response1"

    def test_persistence(self, temp_dir: Path):
        """Test that cache persists across instances."""
        cache_dir = temp_dir / "cache"

        # Create cache and add entry
        cache1 = DiskCache(cache_dir)
        cache1.set(CacheEntry(
            key="key1",
            query="query1",
            response="response1",
        ))

        # Create new cache instance pointing to same directory
        cache2 = DiskCache(cache_dir)
        result = cache2.get("key1")

        assert result is not None
        assert result.response == "response1"

    def test_delete(self, temp_dir: Path):
        """Test deleting an entry from disk cache."""
        cache = DiskCache(temp_dir / "cache")
        cache.set(CacheEntry(
            key="key1",
            query="query1",
            response="response1",
        ))

        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self, temp_dir: Path):
        """Test clearing disk cache."""
        cache = DiskCache(temp_dir / "cache")
        for i in range(3):
            cache.set(CacheEntry(
                key=f"key{i}",
                query=f"query{i}",
                response=f"response{i}",
            ))

        assert cache.size() == 3
        cache.clear()
        assert cache.size() == 0


class TestLLMCache:
    """Tests for high-level LLM cache."""

    def test_get_set(self):
        """Test basic get and set."""
        cache = LLMCache()

        cache.set("What is Python?", "Python is a programming language.")
        result = cache.get("What is Python?")

        assert result == "Python is a programming language."

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LLMCache()
        result = cache.get("Unknown query")
        assert result is None

    def test_cache_with_context(self):
        """Test caching with context."""
        cache = LLMCache()

        cache.set("What is it?", "It's Python.", context="programming")
        cache.set("What is it?", "It's a snake.", context="animals")

        assert cache.get("What is it?", context="programming") == "It's Python."
        assert cache.get("What is it?", context="animals") == "It's a snake."

    def test_statistics(self):
        """Test cache statistics tracking."""
        cache = LLMCache()

        cache.set("query1", "response1")
        cache.set("query2", "response2")

        cache.get("query1")  # Hit
        cache.get("query1")  # Hit
        cache.get("query3")  # Miss

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 2

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = LLMCache()

        cache.set("query1", "response1")

        # 4 hits, 1 miss = 80% hit rate
        for _ in range(4):
            cache.get("query1")
        cache.get("nonexistent")

        assert cache.hit_rate == 0.8

    def test_invalidate(self):
        """Test invalidating specific entry."""
        cache = LLMCache()

        cache.set("query1", "response1")
        cache.invalidate("query1")

        assert cache.get("query1") is None

    def test_clear(self):
        """Test clearing cache."""
        cache = LLMCache()

        cache.set("query1", "response1")
        cache.set("query2", "response2")
        cache.get("query1")

        cache.clear()

        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
        assert cache.get("query1") is None


class TestToolResultCache:
    """Tests for tool result caching."""

    def test_is_cacheable(self):
        """Test identifying cacheable tools."""
        cache = ToolResultCache()

        assert cache.is_cacheable("read_file")
        assert cache.is_cacheable("grep_search")
        assert cache.is_cacheable("git_status")
        assert not cache.is_cacheable("bash")
        assert not cache.is_cacheable("write_file")

    def test_cache_tool_result(self):
        """Test caching tool results."""
        cache = ToolResultCache()

        cache.set("read_file", {"path": "src/main.py"}, "file contents here")
        result = cache.get("read_file", {"path": "src/main.py"})

        assert result == "file contents here"

    def test_cache_miss_for_different_args(self):
        """Test cache miss when args differ."""
        cache = ToolResultCache()

        cache.set("read_file", {"path": "src/main.py"}, "main.py contents")
        result = cache.get("read_file", {"path": "src/utils.py"})

        assert result is None

    def test_no_cache_for_uncacheable_tools(self):
        """Test that uncacheable tools are not cached."""
        cache = ToolResultCache()

        cache.set("bash", {"command": "ls"}, "output")
        result = cache.get("bash", {"command": "ls"})

        assert result is None


class TestCreateLLMCache:
    """Tests for create_llm_cache factory function."""

    def test_create_memory_cache(self):
        """Test creating memory cache."""
        cache = create_llm_cache(
            cache_type="memory",
            max_size=500,
            ttl=1800,
        )
        assert isinstance(cache, LLMCache)
        assert isinstance(cache.backend, InMemoryCache)

    def test_create_disk_cache(self, temp_dir: Path):
        """Test creating disk cache."""
        cache = create_llm_cache(
            cache_type="disk",
            cache_dir=temp_dir / "llm_cache",
            max_size=100,
        )
        assert isinstance(cache, LLMCache)
        assert isinstance(cache.backend, DiskCache)
