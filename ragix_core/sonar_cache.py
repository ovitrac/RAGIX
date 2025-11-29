"""
SonarQube Cache - TTL-based caching for SonarQube API responses

Provides time-based caching for SonarQube analysis results.
Caches are invalidated after a configurable TTL to ensure data freshness.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-29
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from .version import __version__

logger = logging.getLogger(__name__)


# Default TTL in seconds (5 minutes)
DEFAULT_TTL_SECONDS = 300


@dataclass
class SonarCacheMetadata:
    """Metadata about cached SonarQube analysis."""
    cache_key: str
    project_key: str
    server_url: str
    created_at: str
    expires_at: float  # Unix timestamp
    ragix_version: str


@dataclass
class CachedSonarAnalysis:
    """Complete cached SonarQube analysis results."""
    metadata: SonarCacheMetadata
    project_key: str = ""
    server: str = ""
    quality_gate: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: Dict[str, Any] = field(default_factory=dict)
    hotspots: int = 0
    top_issues: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': asdict(self.metadata),
            'project_key': self.project_key,
            'server': self.server,
            'quality_gate': self.quality_gate,
            'metrics': self.metrics,
            'issues': self.issues,
            'hotspots': self.hotspots,
            'top_issues': self.top_issues
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedSonarAnalysis':
        """Create from dictionary."""
        metadata = SonarCacheMetadata(**data['metadata'])
        return cls(
            metadata=metadata,
            project_key=data.get('project_key', ''),
            server=data.get('server', ''),
            quality_gate=data.get('quality_gate', ''),
            metrics=data.get('metrics', {}),
            issues=data.get('issues', {}),
            hotspots=data.get('hotspots', 0),
            top_issues=data.get('top_issues', [])
        )

    def is_expired(self) -> bool:
        """Check if this cached entry has expired."""
        return time.time() > self.metadata.expires_at


class SonarCache:
    """
    Manages cached SonarQube analysis results.

    Uses TTL-based caching since SonarQube data comes from external API
    and may change independently of local files.
    """

    DEFAULT_CACHE_DIR = Path.home() / '.ragix' / 'cache' / 'sonar'

    def __init__(self, cache_dir: Optional[Path] = None, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        """
        Initialize the SonarQube cache.

        Args:
            cache_dir: Custom cache directory. Defaults to ~/.ragix/cache/sonar/
            ttl_seconds: Time-to-live for cache entries in seconds (default 5 min)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.index_path = self.cache_dir / 'index.json'
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load Sonar cache index: {e}")
                return {}
        return {}

    def _save_index(self):
        """Save the cache index."""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self._index, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save Sonar cache index: {e}")

    def get_cache_key(self, project_key: str, server_url: str) -> str:
        """
        Generate a cache key for a SonarQube project.

        Args:
            project_key: The SonarQube project key
            server_url: The SonarQube server URL

        Returns:
            A hash-based cache key
        """
        key_string = f"{server_url}:{project_key}"
        hasher = hashlib.sha256()
        hasher.update(key_string.encode('utf-8'))
        return hasher.hexdigest()[:16]

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def is_cached(self, cache_key: str) -> bool:
        """Check if valid (non-expired) cache exists for the given key."""
        if cache_key not in self._index:
            return False

        # Check expiry from index
        expires_at = self._index[cache_key].get('expires_at', 0)
        if time.time() > expires_at:
            # Expired - clean up
            self.invalidate(cache_key)
            return False

        return self.get_cache_path(cache_key).exists()

    def load(self, cache_key: str) -> Optional[CachedSonarAnalysis]:
        """
        Load cached SonarQube analysis for the given cache key.

        Args:
            cache_key: The cache key

        Returns:
            CachedSonarAnalysis if found and not expired, None otherwise
        """
        if not self.is_cached(cache_key):
            return None

        cache_file = self.get_cache_path(cache_key)

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            cached = CachedSonarAnalysis.from_dict(data)

            # Double-check expiry
            if cached.is_expired():
                self.invalidate(cache_key)
                return None

            return cached
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load cached Sonar analysis: {e}")
            return None

    def save(
        self,
        project_key: str,
        server_url: str,
        quality_gate: str,
        metrics: Dict[str, Any],
        issues: Dict[str, Any],
        hotspots: int,
        top_issues: List[Dict[str, Any]]
    ) -> CachedSonarAnalysis:
        """
        Save SonarQube analysis results to cache.

        Args:
            project_key: SonarQube project key
            server_url: SonarQube server URL
            quality_gate: Quality gate status
            metrics: Project metrics dict
            issues: Issues summary dict
            hotspots: Number of security hotspots
            top_issues: List of top issues

        Returns:
            The created CachedSonarAnalysis object
        """
        cache_key = self.get_cache_key(project_key, server_url)
        now = time.time()
        expires_at = now + self.ttl_seconds

        metadata = SonarCacheMetadata(
            cache_key=cache_key,
            project_key=project_key,
            server_url=server_url,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at,
            ragix_version=__version__
        )

        analysis = CachedSonarAnalysis(
            metadata=metadata,
            project_key=project_key,
            server=server_url,
            quality_gate=quality_gate,
            metrics=metrics,
            issues=issues,
            hotspots=hotspots,
            top_issues=top_issues
        )

        # Save analysis
        cache_file = self.get_cache_path(cache_key)
        try:
            with open(cache_file, 'w') as f:
                json.dump(analysis.to_dict(), f)
        except IOError as e:
            logger.error(f"Failed to save Sonar analysis: {e}")
            raise

        # Update index
        self._index[cache_key] = {
            'project_key': project_key,
            'server_url': server_url,
            'created_at': metadata.created_at,
            'expires_at': expires_at,
            'ragix_version': __version__
        }
        self._save_index()

        logger.info(f"Cached Sonar analysis for {project_key} (TTL: {self.ttl_seconds}s)")
        return analysis

    def invalidate(self, cache_key: str) -> bool:
        """Remove a cached entry."""
        if cache_key not in self._index:
            return False

        cache_file = self.get_cache_path(cache_key)
        if cache_file.exists():
            try:
                cache_file.unlink()
            except IOError as e:
                logger.warning(f"Failed to remove cache file: {e}")

        del self._index[cache_key]
        self._save_index()
        return True

    def invalidate_project(self, project_key: str) -> int:
        """Remove all cached entries for a project key."""
        to_remove = [
            key for key, info in self._index.items()
            if info.get('project_key') == project_key
        ]
        for key in to_remove:
            self.invalidate(key)
        return len(to_remove)

    def cleanup_expired(self) -> int:
        """Remove all expired cache entries."""
        now = time.time()
        expired = [
            key for key, info in self._index.items()
            if info.get('expires_at', 0) < now
        ]
        for key in expired:
            self.invalidate(key)
        return len(expired)

    def clear(self) -> int:
        """Clear all cached entries."""
        count = len(self._index)
        for cache_key in list(self._index.keys()):
            cache_file = self.get_cache_path(cache_key)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except IOError:
                    pass

        self._index = {}
        self._save_index()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        valid_count = 0
        expired_count = 0
        now = time.time()

        for cache_key, info in self._index.items():
            cache_file = self.get_cache_path(cache_key)
            if cache_file.exists():
                total_size += cache_file.stat().st_size

            if info.get('expires_at', 0) > now:
                valid_count += 1
            else:
                expired_count += 1

        return {
            'entry_count': len(self._index),
            'valid_entries': valid_count,
            'expired_entries': expired_count,
            'total_size_bytes': total_size,
            'total_size_kb': round(total_size / 1024, 2),
            'ttl_seconds': self.ttl_seconds,
            'cache_dir': str(self.cache_dir)
        }


# Global cache instance
_sonar_cache: Optional[SonarCache] = None


def get_sonar_cache(cache_dir: Optional[Path] = None, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> SonarCache:
    """Get or create the global SonarQube cache instance."""
    global _sonar_cache
    if _sonar_cache is None or (cache_dir and _sonar_cache.cache_dir != cache_dir):
        _sonar_cache = SonarCache(cache_dir, ttl_seconds)
    return _sonar_cache
