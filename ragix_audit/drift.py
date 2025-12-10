"""
Drift Detection Module — Spec-Code Misalignment Analysis

Detects drift between code changes and documentation updates:
- Code evolved but docs frozen → UNDOCUMENTED_CHANGES
- Docs evolved but code frozen → SPEC_AHEAD_OF_CODE
- Both frozen → STABLE
- Both evolving → SYNCHRONIZED

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Type of spec-code drift."""
    SYNCHRONIZED = "synchronized"        # Both evolving together
    UNDOCUMENTED_CHANGES = "undocumented"  # Code changed, docs frozen
    SPEC_AHEAD_OF_CODE = "spec_ahead"    # Docs evolved, code frozen
    STABLE = "stable"                    # Both frozen, acceptable
    UNKNOWN = "unknown"


class Severity(Enum):
    """Drift severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DocInfo:
    """Information about a documentation file."""
    path: Path
    version: Optional[str] = None       # "V0.8", "V1.2"
    last_modified: Optional[datetime] = None
    component_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "path": str(self.path),
            "version": self.version,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "component_ids": self.component_ids,
        }


@dataclass
class DriftReport:
    """Report of spec-code drift for a component."""
    component_id: str
    drift_type: DriftType
    severity: Severity

    # Code timeline
    code_files_count: int = 0
    code_last_change: Optional[datetime] = None
    code_changes_estimated: int = 0

    # Doc timeline
    doc_files_count: int = 0
    doc_last_change: Optional[datetime] = None
    doc_version: Optional[str] = None

    # Drift metrics
    days_since_code_change: int = 0
    days_since_doc_change: int = 0
    drift_days: int = 0                 # Difference between code and doc changes

    message: str = ""
    affected_files: List[str] = field(default_factory=list)

    @property
    def gap_score(self) -> float:
        """Calculate documentation gap score (0-1)."""
        if self.drift_type == DriftType.SYNCHRONIZED:
            return 0.0
        elif self.drift_type == DriftType.STABLE:
            return 0.1
        elif self.drift_type == DriftType.UNDOCUMENTED_CHANGES:
            # Score based on drift magnitude
            return min(1.0, self.drift_days / 365)  # Max at 1 year drift
        elif self.drift_type == DriftType.SPEC_AHEAD_OF_CODE:
            return 0.3  # Less severe but still a gap
        return 0.2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "code_files_count": self.code_files_count,
            "code_last_change": self.code_last_change.isoformat() if self.code_last_change else None,
            "code_changes_estimated": self.code_changes_estimated,
            "doc_files_count": self.doc_files_count,
            "doc_last_change": self.doc_last_change.isoformat() if self.doc_last_change else None,
            "doc_version": self.doc_version,
            "days_since_code_change": self.days_since_code_change,
            "days_since_doc_change": self.days_since_doc_change,
            "drift_days": self.drift_days,
            "gap_score": round(self.gap_score, 3),
            "message": self.message,
            "affected_files": self.affected_files[:10],  # Limit
        }


class DriftAnalyzer:
    """
    Analyzes spec-code drift between documentation and code.

    Connects:
    - Code files (.java) with their modification times
    - Doc files (SFD, ISD, .md, .docx) with their versions and dates
    - RAG chunks if available for richer analysis
    """

    # Doc file patterns
    DOC_PATTERNS = [
        r'.*SFD.*\.(md|docx?|pdf|txt)$',
        r'.*ISD.*\.(md|docx?|pdf|txt)$',
        r'.*spec.*\.(md|docx?|pdf|txt)$',
        r'.*README.*\.md$',
        r'.*\.md$',
    ]

    # Version extraction
    VERSION_PATTERN = re.compile(r'[Vv](\d+(?:\.\d+)*)')

    def __init__(
        self,
        reference_date: Optional[datetime] = None,
        drift_threshold_days: int = 90
    ):
        """
        Initialize drift analyzer.

        Args:
            reference_date: Date to calculate drift from (default: now)
            drift_threshold_days: Days without doc update to consider "frozen"
        """
        self.reference_date = reference_date or datetime.now()
        self.drift_threshold_days = drift_threshold_days
        self.doc_files: Dict[str, List[DocInfo]] = {}  # component_id -> docs
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.DOC_PATTERNS]

    def scan_docs(
        self,
        root_path: Path,
        component_mapper: Optional[Any] = None
    ) -> Dict[str, List[DocInfo]]:
        """
        Scan directory for documentation files.

        Args:
            root_path: Root directory to scan
            component_mapper: Optional ComponentMapper for component detection
        """
        from ragix_audit.component_mapper import ComponentMapper, detect_component_id

        if component_mapper is None:
            component_mapper = ComponentMapper()

        root_path = Path(root_path)

        for file_path in root_path.rglob('*'):
            if not file_path.is_file():
                continue

            # Check if it's a doc file
            if not self._is_doc_file(file_path):
                continue

            # Extract info
            stat = file_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime)

            # Extract version from filename or content
            version = self._extract_version(file_path)

            # Detect component IDs
            comp_ids = []

            # From path
            path_comp = detect_component_id(str(file_path))
            if path_comp:
                comp_ids.append(path_comp)

            # From content (for text files)
            if file_path.suffix.lower() in ['.md', '.txt']:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    for match in re.finditer(r'\b(SK|SC|SG)[0-9]{2}\b', content, re.IGNORECASE):
                        comp_id = match.group(0).upper()
                        if comp_id not in comp_ids:
                            comp_ids.append(comp_id)
                except Exception:
                    pass

            # Create doc info
            doc_info = DocInfo(
                path=file_path,
                version=version,
                last_modified=last_modified,
                component_ids=comp_ids,
            )

            # Register with each component
            for comp_id in comp_ids:
                if comp_id not in self.doc_files:
                    self.doc_files[comp_id] = []
                self.doc_files[comp_id].append(doc_info)

        logger.info(f"Found docs for {len(self.doc_files)} components")
        return self.doc_files

    def _is_doc_file(self, file_path: Path) -> bool:
        """Check if file is a documentation file."""
        path_str = str(file_path)
        for pattern in self._compiled_patterns:
            if pattern.match(path_str):
                return True
        return False

    def _extract_version(self, file_path: Path) -> Optional[str]:
        """Extract version from filename."""
        match = self.VERSION_PATTERN.search(file_path.name)
        if match:
            return f"V{match.group(1)}"
        return None

    def analyze_component(
        self,
        component_id: str,
        code_last_change: datetime,
        code_files_count: int,
        code_changes_estimated: int = 0
    ) -> DriftReport:
        """
        Analyze drift for a single component.

        Args:
            component_id: Component ID (SK02, SC04, etc.)
            code_last_change: Last code modification time
            code_files_count: Number of code files
            code_changes_estimated: Estimated number of changes

        Returns:
            DriftReport with drift analysis
        """
        docs = self.doc_files.get(component_id, [])

        # Doc metrics
        doc_files_count = len(docs)
        doc_last_change = None
        doc_version = None

        if docs:
            # Get most recent doc
            most_recent = max(docs, key=lambda d: d.last_modified or datetime.min)
            doc_last_change = most_recent.last_modified
            doc_version = most_recent.version

        # Calculate days since changes
        days_since_code = (self.reference_date - code_last_change).days
        days_since_doc = (self.reference_date - doc_last_change).days if doc_last_change else 9999

        # Calculate drift
        if doc_last_change:
            drift_days = abs((code_last_change - doc_last_change).days)
        else:
            drift_days = days_since_code  # No docs = full drift

        # Determine drift type
        drift_type, severity, message = self._classify_drift(
            component_id=component_id,
            days_since_code=days_since_code,
            days_since_doc=days_since_doc,
            drift_days=drift_days,
            has_docs=doc_files_count > 0,
            doc_version=doc_version,
        )

        # Collect affected files
        affected = [str(d.path) for d in docs]

        return DriftReport(
            component_id=component_id,
            drift_type=drift_type,
            severity=severity,
            code_files_count=code_files_count,
            code_last_change=code_last_change,
            code_changes_estimated=code_changes_estimated,
            doc_files_count=doc_files_count,
            doc_last_change=doc_last_change,
            doc_version=doc_version,
            days_since_code_change=days_since_code,
            days_since_doc_change=days_since_doc,
            drift_days=drift_days,
            message=message,
            affected_files=affected,
        )

    def _classify_drift(
        self,
        component_id: str,
        days_since_code: int,
        days_since_doc: int,
        drift_days: int,
        has_docs: bool,
        doc_version: Optional[str],
    ) -> Tuple[DriftType, Severity, str]:
        """Classify drift type and generate message."""
        threshold = self.drift_threshold_days

        code_active = days_since_code < threshold
        doc_active = days_since_doc < threshold if has_docs else False

        if not has_docs:
            return (
                DriftType.UNDOCUMENTED_CHANGES,
                Severity.WARNING,
                f"{component_id}: No documentation found. Consider creating specs."
            )

        if code_active and doc_active:
            if drift_days < 30:
                return (
                    DriftType.SYNCHRONIZED,
                    Severity.INFO,
                    f"{component_id}: Code and documentation are synchronized."
                )
            else:
                return (
                    DriftType.UNDOCUMENTED_CHANGES,
                    Severity.WARNING,
                    f"{component_id}: Code changed {drift_days} days after last doc update ({doc_version}). Review for undocumented changes."
                )

        if code_active and not doc_active:
            severity = Severity.ERROR if drift_days > 180 else Severity.WARNING
            return (
                DriftType.UNDOCUMENTED_CHANGES,
                severity,
                f"{component_id}: Code actively changing but docs frozen at {doc_version} ({days_since_doc} days ago). Documentation gap of {drift_days} days."
            )

        if not code_active and doc_active:
            return (
                DriftType.SPEC_AHEAD_OF_CODE,
                Severity.INFO,
                f"{component_id}: Documentation updated recently but code unchanged. Verify spec implementation."
            )

        # Both frozen
        if drift_days > 365:
            return (
                DriftType.STABLE,
                Severity.INFO,
                f"{component_id}: Both code and docs stable for over a year. Legacy component."
            )
        else:
            return (
                DriftType.STABLE,
                Severity.INFO,
                f"{component_id}: Code and documentation both stable."
            )

    def analyze_all(
        self,
        timelines: Dict[str, Any]
    ) -> Dict[str, DriftReport]:
        """
        Analyze drift for all components with timelines.

        Args:
            timelines: ComponentTimeline dict from TimelineScanner

        Returns:
            Dictionary of component_id -> DriftReport
        """
        results = {}

        for comp_id, timeline in timelines.items():
            if timeline.last_change is None:
                continue

            results[comp_id] = self.analyze_component(
                component_id=comp_id,
                code_last_change=timeline.last_change,
                code_files_count=timeline.file_count,
                code_changes_estimated=timeline.estimated_changes,
            )

        return results

    def get_gap_scores(
        self,
        drift_reports: Dict[str, DriftReport]
    ) -> Dict[str, float]:
        """
        Extract gap scores for risk scoring integration.

        Returns:
            Dictionary of component_id -> gap_score (0-1)
        """
        return {comp_id: report.gap_score for comp_id, report in drift_reports.items()}

    def get_summary(
        self,
        drift_reports: Dict[str, DriftReport]
    ) -> Dict[str, Any]:
        """Generate summary of drift analysis."""
        by_type = {}
        for dtype in DriftType:
            reports = [r for r in drift_reports.values() if r.drift_type == dtype]
            if reports:
                by_type[dtype.value] = {
                    "count": len(reports),
                    "components": [r.component_id for r in reports],
                    "avg_drift_days": sum(r.drift_days for r in reports) / len(reports),
                }

        by_severity = {}
        for sev in Severity:
            reports = [r for r in drift_reports.values() if r.severity == sev]
            if reports:
                by_severity[sev.value] = {
                    "count": len(reports),
                    "components": [r.component_id for r in reports],
                }

        # Critical alerts (high drift)
        alerts = [
            r for r in drift_reports.values()
            if r.severity in (Severity.ERROR, Severity.CRITICAL)
        ]
        alerts.sort(key=lambda r: r.drift_days, reverse=True)

        return {
            "total_components": len(drift_reports),
            "by_type": by_type,
            "by_severity": by_severity,
            "alerts": [
                {"component": r.component_id, "message": r.message, "drift_days": r.drift_days}
                for r in alerts[:10]
            ],
            "avg_gap_score": sum(r.gap_score for r in drift_reports.values()) / len(drift_reports) if drift_reports else 0,
        }


if __name__ == "__main__":
    import sys
    import json
    from ragix_audit.timeline import TimelineScanner

    path = sys.argv[1] if len(sys.argv) > 1 else "/home/olivi/Documents/Adservio/audit/IOWIZME"

    print(f"Analyzing drift for {path}...")

    # Scan timelines
    scanner = TimelineScanner()
    src_path = Path(path) / "src" if (Path(path) / "src").exists() else Path(path)
    scanner.scan_directory(src_path)
    timelines = scanner.build_component_timelines()

    # Scan docs and analyze drift
    analyzer = DriftAnalyzer()
    analyzer.scan_docs(Path(path))
    drift_reports = analyzer.analyze_all(timelines)

    print("\n=== Drift Summary ===\n")
    summary = analyzer.get_summary(drift_reports)
    print(json.dumps(summary, indent=2))

    print("\n=== Drift Details ===\n")
    for comp_id, report in sorted(drift_reports.items()):
        print(f"{comp_id}: {report.drift_type.value} ({report.severity.value})")
        print(f"  {report.message}")
        print(f"  Code: {report.days_since_code_change}d ago, Docs: {report.days_since_doc_change}d ago")
        print()
