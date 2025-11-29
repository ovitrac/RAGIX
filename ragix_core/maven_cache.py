"""
Maven Cache - Persistent caching for Maven POM analysis results

Provides fingerprint-based caching for Maven project analysis.
Tracks pom.xml files to detect when projects have changed.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-29
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from .version import __version__

logger = logging.getLogger(__name__)


# POM files to track for fingerprint
POM_PATTERNS = {'pom.xml'}

# Directories to exclude
EXCLUDED_DIRS = {
    '.git', '.svn', '.hg', 'node_modules', '__pycache__',
    '.idea', '.vscode', 'target', 'build', 'dist', 'out',
    '.ragix_cache', 'venv', 'env', '.env'
}


@dataclass
class MavenCacheMetadata:
    """Metadata about cached Maven analysis."""
    fingerprint: str
    project_path: str
    created_at: str
    ragix_version: str
    pom_count: int
    total_size: int


@dataclass
class CachedMavenAnalysis:
    """Complete cached Maven analysis results."""
    metadata: MavenCacheMetadata
    projects: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    conflict_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': asdict(self.metadata),
            'projects': self.projects,
            'conflicts': self.conflicts,
            'conflict_count': self.conflict_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedMavenAnalysis':
        """Create from dictionary."""
        metadata = MavenCacheMetadata(**data['metadata'])
        return cls(
            metadata=metadata,
            projects=data.get('projects', []),
            conflicts=data.get('conflicts', []),
            conflict_count=data.get('conflict_count', 0)
        )


class MavenCache:
    """
    Manages cached Maven analysis results.

    Uses POM file fingerprints (hash of pom.xml paths, sizes, mtimes) to detect
    when a project has changed and needs re-analysis.
    """

    DEFAULT_CACHE_DIR = Path.home() / '.ragix' / 'cache' / 'maven'

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the Maven cache.

        Args:
            cache_dir: Custom cache directory. Defaults to ~/.ragix/cache/maven/
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / 'index.json'
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load Maven cache index: {e}")
                return {}
        return {}

    def _save_index(self):
        """Save the cache index."""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self._index, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save Maven cache index: {e}")

    def get_fingerprint(self, project_path: Path) -> Tuple[str, int, int]:
        """
        Calculate a fingerprint based on POM files.

        Args:
            project_path: Path to the project root

        Returns:
            Tuple of (fingerprint_hash, pom_count, total_size)
        """
        project_path = Path(project_path).resolve()
        file_entries: List[str] = []
        total_size = 0

        for root, dirs, files in os.walk(project_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for filename in sorted(files):
                if filename in POM_PATTERNS:
                    filepath = Path(root) / filename
                    try:
                        stat = filepath.stat()
                        rel_path = filepath.relative_to(project_path)
                        entry = f"{rel_path}:{stat.st_size}:{int(stat.st_mtime)}"
                        file_entries.append(entry)
                        total_size += stat.st_size
                    except (OSError, ValueError):
                        continue

        # Sort for deterministic hash
        file_entries.sort()

        # Calculate hash
        hasher = hashlib.sha256()
        hasher.update('\n'.join(file_entries).encode('utf-8'))
        fingerprint = hasher.hexdigest()[:16]

        return fingerprint, len(file_entries), total_size

    def get_cache_path(self, fingerprint: str) -> Path:
        """Get the cache file path for a fingerprint."""
        return self.cache_dir / f"{fingerprint}.json"

    def is_cached(self, fingerprint: str) -> bool:
        """Check if analysis exists for the given fingerprint."""
        if fingerprint not in self._index:
            return False
        return self.get_cache_path(fingerprint).exists()

    def load(self, fingerprint: str) -> Optional[CachedMavenAnalysis]:
        """
        Load cached Maven analysis for the given fingerprint.

        Args:
            fingerprint: The project fingerprint

        Returns:
            CachedMavenAnalysis if found, None otherwise
        """
        if not self.is_cached(fingerprint):
            return None

        cache_file = self.get_cache_path(fingerprint)

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return CachedMavenAnalysis.from_dict(data)
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load cached Maven analysis: {e}")
            return None

    def save(
        self,
        fingerprint: str,
        project_path: Path,
        projects: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]],
        conflict_count: int,
        pom_count: int,
        total_size: int
    ) -> CachedMavenAnalysis:
        """
        Save Maven analysis results to cache.

        Args:
            fingerprint: Project fingerprint
            project_path: Path to project
            projects: List of parsed Maven projects
            conflicts: List of dependency conflicts
            conflict_count: Total number of conflicts
            pom_count: Number of POM files
            total_size: Total size of POM files

        Returns:
            The created CachedMavenAnalysis object
        """
        metadata = MavenCacheMetadata(
            fingerprint=fingerprint,
            project_path=str(project_path),
            created_at=datetime.now().isoformat(),
            ragix_version=__version__,
            pom_count=pom_count,
            total_size=total_size
        )

        analysis = CachedMavenAnalysis(
            metadata=metadata,
            projects=projects,
            conflicts=conflicts,
            conflict_count=conflict_count
        )

        # Save analysis
        cache_file = self.get_cache_path(fingerprint)
        try:
            with open(cache_file, 'w') as f:
                json.dump(analysis.to_dict(), f)
        except IOError as e:
            logger.error(f"Failed to save Maven analysis: {e}")
            raise

        # Update index
        self._index[fingerprint] = {
            'project_path': str(project_path),
            'created_at': metadata.created_at,
            'pom_count': pom_count,
            'ragix_version': __version__
        }
        self._save_index()

        logger.info(f"Cached Maven analysis for {project_path} (fingerprint: {fingerprint})")
        return analysis

    def invalidate(self, fingerprint: str) -> bool:
        """Remove a cached entry."""
        if fingerprint not in self._index:
            return False

        cache_file = self.get_cache_path(fingerprint)
        if cache_file.exists():
            try:
                cache_file.unlink()
            except IOError as e:
                logger.warning(f"Failed to remove cache file: {e}")

        del self._index[fingerprint]
        self._save_index()
        return True

    def clear(self) -> int:
        """Clear all cached entries."""
        count = len(self._index)
        for fingerprint in list(self._index.keys()):
            cache_file = self.get_cache_path(fingerprint)
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
        for fingerprint in self._index:
            cache_file = self.get_cache_path(fingerprint)
            if cache_file.exists():
                total_size += cache_file.stat().st_size

        return {
            'entry_count': len(self._index),
            'total_size_bytes': total_size,
            'total_size_kb': round(total_size / 1024, 2),
            'cache_dir': str(self.cache_dir)
        }


# Global cache instance
_maven_cache: Optional[MavenCache] = None


def get_maven_cache(cache_dir: Optional[Path] = None) -> MavenCache:
    """Get or create the global Maven cache instance."""
    global _maven_cache
    if _maven_cache is None or (cache_dir and _maven_cache.cache_dir != cache_dir):
        _maven_cache = MavenCache(cache_dir)
    return _maven_cache
