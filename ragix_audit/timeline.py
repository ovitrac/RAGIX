"""
Timeline Module — Service Life Profiles without Git

Analyzes file timestamps, document versions, and Javadoc annotations
to build component lifecycle profiles.

Data sources (no git required):
- File mtime/ctime
- Doc versions from filenames (SFD V0.8, ISD V1.2)
- Javadoc @since, @version annotations
- Package-info.java dates
- POM artifact versions

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

from ragix_audit.component_mapper import ComponentMapper, ComponentType

logger = logging.getLogger(__name__)


class LifecycleCategory(Enum):
    """Component lifecycle stage based on age and activity."""
    NEW = "new"                     # age < 6 months
    ACTIVE = "active"               # age >= 6mo, last change < 3mo
    MATURE = "mature"               # age >= 1yr, last change >= 6mo
    LEGACY_HOT = "legacy_hot"       # age >= 3yr, still changing (MCO risk!)
    FROZEN = "frozen"               # age >= 2yr, no recent changes
    FROZEN_RISKY = "frozen_risky"   # frozen + high complexity (hidden risk)
    UNKNOWN = "unknown"


@dataclass
class FileTimeline:
    """Timeline information for a single file."""
    file_id: str                    # Unique identifier
    path: Path
    first_seen: datetime            # Estimated creation (ctime or earliest)
    last_modified: datetime         # mtime
    age_days: int
    size_bytes: int
    component_ids: List[str] = field(default_factory=list)  # SK02, SC04, etc.
    doc_versions: List[str] = field(default_factory=list)   # ["V0.1", "V0.8"]
    javadoc_since: Optional[str] = None
    javadoc_version: Optional[str] = None

    @property
    def age_months(self) -> float:
        return self.age_days / 30.44

    @property
    def age_years(self) -> float:
        return self.age_days / 365.25

    def to_dict(self) -> Dict:
        return {
            "file_id": self.file_id,
            "path": str(self.path),
            "first_seen": self.first_seen.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "age_days": self.age_days,
            "age_months": round(self.age_months, 1),
            "size_bytes": self.size_bytes,
            "component_ids": self.component_ids,
            "doc_versions": self.doc_versions,
            "javadoc_since": self.javadoc_since,
            "javadoc_version": self.javadoc_version,
        }


@dataclass
class ComponentTimeline:
    """Aggregated timeline for a business component (SK/SC/SG)."""
    component_id: str               # "SK02"
    type: ComponentType
    files: List[str] = field(default_factory=list)
    first_seen: Optional[datetime] = None
    last_change: Optional[datetime] = None
    age_days: int = 0
    estimated_changes: int = 0      # Estimated from file diversity and versions
    volatility: float = 0.0         # changes / age_months
    category: LifecycleCategory = LifecycleCategory.UNKNOWN
    doc_versions: List[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def age_months(self) -> float:
        return self.age_days / 30.44 if self.age_days > 0 else 0

    @property
    def age_years(self) -> float:
        return self.age_days / 365.25 if self.age_days > 0 else 0

    def to_dict(self) -> Dict:
        return {
            "component_id": self.component_id,
            "type": self.type.value,
            "file_count": self.file_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_change": self.last_change.isoformat() if self.last_change else None,
            "age_days": self.age_days,
            "age_months": round(self.age_months, 1),
            "age_years": round(self.age_years, 2),
            "estimated_changes": self.estimated_changes,
            "volatility": round(self.volatility, 4),
            "category": self.category.value,
            "doc_versions": self.doc_versions,
        }


class TimelineScanner:
    """
    Scans files and builds timeline profiles for components.

    Without git, uses:
    - File mtime (last modification)
    - File ctime (metadata change, often approximates creation on some systems)
    - Doc version patterns in filenames and content
    - Javadoc annotations (@since, @version)
    """

    # Patterns for extracting version info
    DOC_VERSION_PATTERN = re.compile(r'[Vv](\d+(?:\.\d+)*)', re.IGNORECASE)
    JAVADOC_SINCE_PATTERN = re.compile(r'@since\s+(\S+)')
    JAVADOC_VERSION_PATTERN = re.compile(r'@version\s+(\S+)')

    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize scanner.

        Args:
            reference_date: Date to calculate age from (default: now)
        """
        self.reference_date = reference_date or datetime.now()
        self.component_mapper = ComponentMapper()
        self.file_timelines: Dict[str, FileTimeline] = {}
        self.component_timelines: Dict[str, ComponentTimeline] = {}

    def scan_file(self, file_path: Path, content: Optional[str] = None) -> FileTimeline:
        """
        Extract timeline information from a single file.
        """
        stat = file_path.stat()

        # Get timestamps
        mtime = datetime.fromtimestamp(stat.st_mtime)
        ctime = datetime.fromtimestamp(stat.st_ctime)

        # On Linux, ctime is metadata change, not creation
        # Use the earlier of mtime and ctime as "first_seen" approximation
        first_seen = min(mtime, ctime)
        last_modified = mtime

        # Calculate age
        age_days = (self.reference_date - first_seen).days

        # Extract component IDs
        component_ids = self.component_mapper.map_file(file_path, content)

        # Extract version info from content
        doc_versions = []
        javadoc_since = None
        javadoc_version = None

        if content:
            # Doc versions from content
            versions = self.DOC_VERSION_PATTERN.findall(content)
            doc_versions = list(set(f"V{v}" for v in versions[:10]))  # Limit

            # Javadoc annotations
            since_match = self.JAVADOC_SINCE_PATTERN.search(content)
            if since_match:
                javadoc_since = since_match.group(1)

            version_match = self.JAVADOC_VERSION_PATTERN.search(content)
            if version_match:
                javadoc_version = version_match.group(1)

        # Also check filename for versions
        filename_versions = self.DOC_VERSION_PATTERN.findall(file_path.name)
        for v in filename_versions:
            version_str = f"V{v}"
            if version_str not in doc_versions:
                doc_versions.append(version_str)

        # Create file ID
        file_id = f"F{hash(str(file_path)) % 1000000:06d}"

        timeline = FileTimeline(
            file_id=file_id,
            path=file_path,
            first_seen=first_seen,
            last_modified=last_modified,
            age_days=max(0, age_days),
            size_bytes=stat.st_size,
            component_ids=component_ids,
            doc_versions=doc_versions,
            javadoc_since=javadoc_since,
            javadoc_version=javadoc_version,
        )

        self.file_timelines[file_id] = timeline
        return timeline

    def scan_directory(
        self,
        root_path: Path,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, FileTimeline]:
        """
        Scan all files in directory tree.

        Args:
            root_path: Root directory to scan
            extensions: File extensions to include (default: common code/doc types)

        Returns:
            Dictionary of file_id -> FileTimeline
        """
        if extensions is None:
            extensions = ['.java', '.xml', '.properties', '.md', '.txt', '.json', '.yaml', '.yml']

        root_path = Path(root_path)
        scanned = 0

        for ext in extensions:
            for file_path in root_path.rglob(f'*{ext}'):
                if file_path.is_file() and not self._should_skip(file_path):
                    try:
                        content = None
                        if ext in ['.java', '.xml', '.md', '.txt']:
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                        self.scan_file(file_path, content)
                        scanned += 1
                    except Exception as e:
                        logger.warning(f"Failed to scan {file_path}: {e}")

        logger.info(f"Scanned {scanned} files")
        return self.file_timelines

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_dirs = {'target', 'build', '.git', '.svn', 'node_modules', '__pycache__'}
        return any(part in skip_dirs for part in file_path.parts)

    def build_component_timelines(self) -> Dict[str, ComponentTimeline]:
        """
        Aggregate file timelines into component timelines.
        """
        # Group files by component
        component_files: Dict[str, List[FileTimeline]] = {}

        for file_timeline in self.file_timelines.values():
            for comp_id in file_timeline.component_ids:
                if comp_id not in component_files:
                    component_files[comp_id] = []
                component_files[comp_id].append(file_timeline)

        # Build component timelines
        for comp_id, files in component_files.items():
            comp = self.component_mapper.get_component(comp_id)
            comp_type = comp.type if comp else ComponentType.UNKNOWN

            # Aggregate timestamps
            first_seen = min(f.first_seen for f in files)
            last_change = max(f.last_modified for f in files)
            age_days = (self.reference_date - first_seen).days

            # Aggregate doc versions
            all_versions = set()
            for f in files:
                all_versions.update(f.doc_versions)

            # Estimate changes based on file diversity and modification spread
            # More files + wider modification time spread = more changes
            mod_times = [f.last_modified for f in files]
            if len(mod_times) > 1:
                time_spread = (max(mod_times) - min(mod_times)).days
                estimated_changes = len(files) + (time_spread // 30)  # Rough estimate
            else:
                estimated_changes = len(files)

            # Calculate volatility
            age_months = max(1, age_days / 30.44)
            volatility = estimated_changes / age_months

            # Classify lifecycle category
            category = self._classify_lifecycle(
                age_days=age_days,
                last_change=last_change,
                volatility=volatility
            )

            self.component_timelines[comp_id] = ComponentTimeline(
                component_id=comp_id,
                type=comp_type,
                files=[f.file_id for f in files],
                first_seen=first_seen,
                last_change=last_change,
                age_days=age_days,
                estimated_changes=estimated_changes,
                volatility=volatility,
                category=category,
                doc_versions=sorted(all_versions),
            )

        logger.info(f"Built timelines for {len(self.component_timelines)} components")
        return self.component_timelines

    def _classify_lifecycle(
        self,
        age_days: int,
        last_change: datetime,
        volatility: float
    ) -> LifecycleCategory:
        """
        Classify component into lifecycle category.

        Categories:
        - NEW: age < 6 months
        - ACTIVE: age >= 6mo, last change < 3mo
        - MATURE: age >= 1yr, last change >= 6mo
        - LEGACY_HOT: age >= 3yr, still changing (MCO risk!)
        - FROZEN: age >= 2yr, no recent changes
        """
        days_since_change = (self.reference_date - last_change).days

        # Thresholds (in days)
        SIX_MONTHS = 180
        ONE_YEAR = 365
        TWO_YEARS = 730
        THREE_YEARS = 1095
        THREE_MONTHS = 90

        if age_days < SIX_MONTHS:
            return LifecycleCategory.NEW

        if age_days >= THREE_YEARS and days_since_change < SIX_MONTHS:
            return LifecycleCategory.LEGACY_HOT

        if age_days >= TWO_YEARS and days_since_change >= SIX_MONTHS:
            return LifecycleCategory.FROZEN

        if age_days >= ONE_YEAR and days_since_change >= SIX_MONTHS:
            return LifecycleCategory.MATURE

        if days_since_change < THREE_MONTHS:
            return LifecycleCategory.ACTIVE

        return LifecycleCategory.MATURE

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.component_timelines:
            self.build_component_timelines()

        by_category = {}
        for cat in LifecycleCategory:
            components = [c for c in self.component_timelines.values() if c.category == cat]
            if components:
                by_category[cat.value] = {
                    "count": len(components),
                    "components": [c.component_id for c in components],
                    "avg_age_days": sum(c.age_days for c in components) / len(components),
                    "avg_volatility": sum(c.volatility for c in components) / len(components),
                }

        by_type = {}
        for comp_type in ComponentType:
            components = [c for c in self.component_timelines.values() if c.type == comp_type]
            if components:
                by_type[comp_type.value] = {
                    "count": len(components),
                    "components": [c.component_id for c in components],
                }

        return {
            "total_files": len(self.file_timelines),
            "total_components": len(self.component_timelines),
            "by_category": by_category,
            "by_type": by_type,
            "oldest_file": self._get_oldest_file(),
            "newest_file": self._get_newest_file(),
            "most_volatile": self._get_most_volatile(),
        }

    def _get_oldest_file(self) -> Optional[Dict]:
        if not self.file_timelines:
            return None
        oldest = min(self.file_timelines.values(), key=lambda f: f.first_seen)
        return {"path": str(oldest.path), "first_seen": oldest.first_seen.isoformat()}

    def _get_newest_file(self) -> Optional[Dict]:
        if not self.file_timelines:
            return None
        newest = max(self.file_timelines.values(), key=lambda f: f.last_modified)
        return {"path": str(newest.path), "last_modified": newest.last_modified.isoformat()}

    def _get_most_volatile(self) -> Optional[Dict]:
        if not self.component_timelines:
            return None
        most_volatile = max(self.component_timelines.values(), key=lambda c: c.volatility)
        return {
            "component_id": most_volatile.component_id,
            "volatility": most_volatile.volatility,
            "category": most_volatile.category.value,
        }


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "/home/olivi/Documents/Adservio/audit/IOWIZME/src"

    print(f"Scanning {path}...")
    scanner = TimelineScanner()
    scanner.scan_directory(Path(path))
    scanner.build_component_timelines()

    print("\n=== Timeline Summary ===\n")
    summary = scanner.get_summary()
    print(json.dumps(summary, indent=2, default=str))

    print("\n=== Component Details ===\n")
    for comp_id, timeline in sorted(scanner.component_timelines.items()):
        print(f"{comp_id} ({timeline.type.value})")
        print(f"  Category: {timeline.category.value}")
        print(f"  Age: {timeline.age_years:.1f} years ({timeline.age_days} days)")
        print(f"  Files: {timeline.file_count}")
        print(f"  Volatility: {timeline.volatility:.3f}")
        print(f"  Last change: {timeline.last_change}")
        print()
