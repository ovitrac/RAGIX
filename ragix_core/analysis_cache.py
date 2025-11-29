"""
Analysis Cache - Persistent caching for AST analysis results

Provides fingerprint-based caching to avoid re-analyzing unchanged projects.
The cache stores symbols, dependencies, metrics, and cycle detection results.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-29
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

from .version import __version__

logger = logging.getLogger(__name__)


# File extensions to include in fingerprint calculation
TRACKED_EXTENSIONS = {
    '.java', '.py', '.js', '.ts', '.jsx', '.tsx',
    '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs',
    '.xml', '.json', '.yaml', '.yml', '.properties',
    '.gradle', '.pom'
}

# Directories to exclude from fingerprint
EXCLUDED_DIRS = {
    '.git', '.svn', '.hg', 'node_modules', '__pycache__',
    '.idea', '.vscode', 'target', 'build', 'dist', 'out',
    '.ragix_cache', 'venv', 'env', '.env'
}


@dataclass
class CacheMetadata:
    """Metadata about a cached analysis."""
    fingerprint: str
    project_path: str
    created_at: str
    ragix_version: str
    file_count: int
    total_size: int
    analysis_time_ms: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedAnalysis:
    """Complete cached analysis results."""
    metadata: CacheMetadata
    symbols: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    cycles: List[List[str]] = field(default_factory=list)
    packages: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': asdict(self.metadata),
            'symbols': self.symbols,
            'dependencies': self.dependencies,
            'metrics': self.metrics,
            'cycles': self.cycles,
            'packages': self.packages
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedAnalysis':
        """Create from dictionary."""
        metadata = CacheMetadata(**data['metadata'])
        return cls(
            metadata=metadata,
            symbols=data.get('symbols', []),
            dependencies=data.get('dependencies', []),
            metrics=data.get('metrics', {}),
            cycles=data.get('cycles', []),
            packages=data.get('packages', {})
        )


class AnalysisCache:
    """
    Manages cached analysis results for AST analysis.

    Uses project fingerprints (hash of file paths, sizes, mtimes) to detect
    when a project has changed and needs re-analysis.
    """

    DEFAULT_CACHE_DIR = Path.home() / '.ragix' / 'cache'

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the analysis cache.

        Args:
            cache_dir: Custom cache directory. Defaults to ~/.ragix/cache/
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
                logger.warning(f"Failed to load cache index: {e}")
                return {}
        return {}

    def _save_index(self):
        """Save the cache index."""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self._index, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save cache index: {e}")

    def get_fingerprint(self, project_path: Path) -> Tuple[str, int, int]:
        """
        Calculate a fingerprint for the project based on tracked files.

        Args:
            project_path: Path to the project root

        Returns:
            Tuple of (fingerprint_hash, file_count, total_size)
        """
        project_path = Path(project_path).resolve()
        file_entries: List[str] = []
        total_size = 0

        for root, dirs, files in os.walk(project_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for filename in sorted(files):
                ext = Path(filename).suffix.lower()
                if ext in TRACKED_EXTENSIONS:
                    filepath = Path(root) / filename
                    try:
                        stat = filepath.stat()
                        rel_path = filepath.relative_to(project_path)
                        # Include path, size, and mtime in fingerprint
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
        fingerprint = hasher.hexdigest()[:16]  # Use first 16 chars

        return fingerprint, len(file_entries), total_size

    def get_cache_path(self, fingerprint: str) -> Path:
        """Get the cache directory for a fingerprint."""
        return self.cache_dir / fingerprint

    def is_cached(self, fingerprint: str) -> bool:
        """Check if analysis exists for the given fingerprint."""
        if fingerprint not in self._index:
            return False

        cache_path = self.get_cache_path(fingerprint)
        return (cache_path / 'analysis.json').exists()

    def get_cache_info(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Get cache info without loading full analysis."""
        return self._index.get(fingerprint)

    def load(self, fingerprint: str) -> Optional[CachedAnalysis]:
        """
        Load cached analysis for the given fingerprint.

        Args:
            fingerprint: The project fingerprint

        Returns:
            CachedAnalysis if found, None otherwise
        """
        if not self.is_cached(fingerprint):
            return None

        cache_path = self.get_cache_path(fingerprint)
        analysis_file = cache_path / 'analysis.json'

        try:
            with open(analysis_file, 'r') as f:
                data = json.load(f)
            return CachedAnalysis.from_dict(data)
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load cached analysis: {e}")
            return None

    def save(
        self,
        fingerprint: str,
        project_path: Path,
        symbols: List[Dict[str, Any]],
        dependencies: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        cycles: List[List[str]],
        packages: Dict[str, int],
        file_count: int,
        total_size: int,
        analysis_time_ms: int = 0,
        settings: Optional[Dict[str, Any]] = None
    ) -> CachedAnalysis:
        """
        Save analysis results to cache.

        Args:
            fingerprint: Project fingerprint
            project_path: Path to project
            symbols: List of symbol dictionaries
            dependencies: List of dependency dictionaries
            metrics: Metrics dictionary
            cycles: List of cycles (each cycle is a list of node names)
            packages: Package name to count mapping
            file_count: Number of tracked files
            total_size: Total size of tracked files
            analysis_time_ms: Time taken for analysis in milliseconds
            settings: Analysis settings used

        Returns:
            The created CachedAnalysis object
        """
        cache_path = self.get_cache_path(fingerprint)
        cache_path.mkdir(parents=True, exist_ok=True)

        metadata = CacheMetadata(
            fingerprint=fingerprint,
            project_path=str(project_path),
            created_at=datetime.now().isoformat(),
            ragix_version=__version__,
            file_count=file_count,
            total_size=total_size,
            analysis_time_ms=analysis_time_ms,
            settings=settings or {}
        )

        analysis = CachedAnalysis(
            metadata=metadata,
            symbols=symbols,
            dependencies=dependencies,
            metrics=metrics,
            cycles=cycles,
            packages=packages
        )

        # Save analysis
        analysis_file = cache_path / 'analysis.json'
        try:
            with open(analysis_file, 'w') as f:
                json.dump(analysis.to_dict(), f)
        except IOError as e:
            logger.error(f"Failed to save analysis: {e}")
            raise

        # Update index
        self._index[fingerprint] = {
            'project_path': str(project_path),
            'created_at': metadata.created_at,
            'file_count': file_count,
            'total_size': total_size,
            'ragix_version': __version__
        }
        self._save_index()

        logger.info(f"Cached analysis for {project_path} (fingerprint: {fingerprint})")
        return analysis

    def invalidate(self, fingerprint: str) -> bool:
        """
        Remove a cached entry.

        Args:
            fingerprint: The fingerprint to invalidate

        Returns:
            True if entry was removed, False if not found
        """
        if fingerprint not in self._index:
            return False

        cache_path = self.get_cache_path(fingerprint)

        # Remove cache files
        if cache_path.exists():
            import shutil
            try:
                shutil.rmtree(cache_path)
            except IOError as e:
                logger.warning(f"Failed to remove cache directory: {e}")

        # Remove from index
        del self._index[fingerprint]
        self._save_index()

        return True

    def invalidate_project(self, project_path: Path) -> int:
        """
        Remove all cached entries for a project path.

        Args:
            project_path: The project path

        Returns:
            Number of entries removed
        """
        project_path = str(Path(project_path).resolve())
        to_remove = [
            fp for fp, info in self._index.items()
            if info.get('project_path') == project_path
        ]

        for fp in to_remove:
            self.invalidate(fp)

        return len(to_remove)

    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries removed
        """
        count = len(self._index)

        import shutil
        for fingerprint in list(self._index.keys()):
            cache_path = self.get_cache_path(fingerprint)
            if cache_path.exists():
                try:
                    shutil.rmtree(cache_path)
                except IOError:
                    pass

        self._index = {}
        self._save_index()

        return count

    def list_cached(self) -> List[Dict[str, Any]]:
        """
        List all cached analyses.

        Returns:
            List of cache info dictionaries
        """
        result = []
        for fingerprint, info in self._index.items():
            entry = {'fingerprint': fingerprint, **info}
            result.append(entry)
        return sorted(result, key=lambda x: x.get('created_at', ''), reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        for fingerprint in self._index:
            cache_path = self.get_cache_path(fingerprint)
            if cache_path.exists():
                for f in cache_path.iterdir():
                    if f.is_file():
                        total_size += f.stat().st_size

        return {
            'entry_count': len(self._index),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir)
        }


# Global cache instance
_global_cache: Optional[AnalysisCache] = None


def get_cache(cache_dir: Optional[Path] = None) -> AnalysisCache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None or (cache_dir and _global_cache.cache_dir != cache_dir):
        _global_cache = AnalysisCache(cache_dir)
    return _global_cache


def get_or_analyze(
    project_path: Path,
    analyze_func,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> Tuple[CachedAnalysis, bool]:
    """
    Get cached analysis or run analysis if not cached.

    This is a convenience function that handles the common pattern of:
    1. Calculate fingerprint
    2. Check cache
    3. Return cached if available, otherwise run analysis and cache

    Args:
        project_path: Path to the project
        analyze_func: Function to call for analysis. Should return a dict with:
                     symbols, dependencies, metrics, cycles, packages, analysis_time_ms
        use_cache: Whether to use caching (set False to force re-analysis)
        cache_dir: Custom cache directory

    Returns:
        Tuple of (CachedAnalysis, was_cached: bool)
    """
    cache = get_cache(cache_dir)
    project_path = Path(project_path).resolve()

    # Calculate fingerprint
    fingerprint, file_count, total_size = cache.get_fingerprint(project_path)

    # Check cache
    if use_cache:
        cached = cache.load(fingerprint)
        if cached:
            logger.info(f"Using cached analysis for {project_path}")
            return cached, True

    # Run analysis
    logger.info(f"Running fresh analysis for {project_path}")
    start_time = time.time()
    result = analyze_func(project_path)
    analysis_time_ms = int((time.time() - start_time) * 1000)

    # Cache results
    analysis = cache.save(
        fingerprint=fingerprint,
        project_path=project_path,
        symbols=result.get('symbols', []),
        dependencies=result.get('dependencies', []),
        metrics=result.get('metrics', {}),
        cycles=result.get('cycles', []),
        packages=result.get('packages', {}),
        file_count=file_count,
        total_size=total_size,
        analysis_time_ms=result.get('analysis_time_ms', analysis_time_ms),
        settings=result.get('settings')
    )

    return analysis, False
