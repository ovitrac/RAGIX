"""
Kernel: Drift Detection
Stage: 2 (Analysis)
Wraps: ragix_audit.drift.DriftAnalyzer

Detects spec-code drift (documentation vs code misalignment):
- UNDOCUMENTED_CHANGES: Code evolved but docs frozen
- SPEC_AHEAD_OF_CODE: Docs evolved but code frozen
- SYNCHRONIZED: Both evolving together
- STABLE: Both frozen (acceptable)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DriftKernel(Kernel):
    """
    Detect spec-code drift between documentation and code.

    This kernel analyzes whether documentation is keeping pace with
    code changes. Useful for MCO (maintenance) risk assessment.

    Configuration options:
        doc_path: Path to documentation (default: project parent/doc)
        drift_threshold_days: Days without update = "frozen" (default: 90)

    Dependencies:
        timeline: Component lifecycle data

    Output:
        components: Per-component drift reports
        summary: Drift statistics
        alerts: High-severity drift warnings
        gap_scores: Integration data for risk kernel
    """

    name = "drift"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Detect spec-code drift"

    requires = ["timeline"]
    provides = ["drift", "gap_scores"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Analyze drift between code and documentation."""

        # Import here to avoid circular imports
        from ragix_audit.drift import DriftAnalyzer, DriftType, Severity
        from ragix_audit.timeline import ComponentTimeline

        # Get configuration
        drift_threshold = input.config.get("drift_threshold_days", 90)
        doc_path = input.config.get("doc_path")
        project_path = Path(input.config.get("project_path", ""))

        # Auto-detect doc path if not specified
        if doc_path:
            doc_path = Path(doc_path)
        else:
            # Try common doc locations
            for candidate in ["doc", "docs", "documentation", ".."]:
                candidate_path = project_path.parent / candidate
                if candidate_path.exists():
                    doc_path = candidate_path
                    break

        if not doc_path or not doc_path.exists():
            logger.warning(f"[drift] No documentation path found")
            return self._empty_result("No documentation path found")

        # Load timeline data
        timeline_path = input.dependencies.get("timeline")
        if not timeline_path or not timeline_path.exists():
            return self._empty_result("Missing timeline dependency")

        with open(timeline_path) as f:
            timeline_data = json.load(f)

        # Handle both list and dict formats for component timelines
        comp_data = timeline_data.get("data", {}).get("component_timelines", [])

        # Convert to ComponentTimeline objects
        timelines = {}

        if isinstance(comp_data, list):
            # List format (newer)
            for item in comp_data:
                comp_id = item.get("component_id")
                last_change_str = item.get("last_change")
                if comp_id and last_change_str:
                    try:
                        last_change = datetime.fromisoformat(last_change_str)
                        timeline = type('ComponentTimeline', (), {
                            'last_change': last_change,
                            'file_count': item.get('file_count', 0),
                            'estimated_changes': item.get('estimated_changes', 0),
                        })()
                        timelines[comp_id] = timeline
                    except (ValueError, TypeError):
                        pass
        elif isinstance(comp_data, dict):
            # Dict format (older)
            for comp_id, item in comp_data.items():
                last_change_str = item.get("last_change")
                if last_change_str:
                    try:
                        last_change = datetime.fromisoformat(last_change_str)
                        timeline = type('ComponentTimeline', (), {
                            'last_change': last_change,
                            'file_count': item.get('file_count', 0),
                            'estimated_changes': item.get('estimated_changes', 0),
                        })()
                        timelines[comp_id] = timeline
                    except (ValueError, TypeError):
                        pass

        logger.info(f"[drift] Analyzing {len(timelines)} components against docs in {doc_path}")

        # Initialize analyzer
        analyzer = DriftAnalyzer(drift_threshold_days=drift_threshold)

        # Scan documentation
        analyzer.scan_docs(doc_path)

        # Analyze all components
        drift_reports = analyzer.analyze_all(timelines)

        # Build result
        components = {}
        for comp_id, report in drift_reports.items():
            components[comp_id] = report.to_dict()

        # Get summary
        summary = analyzer.get_summary(drift_reports)

        # Get gap scores for risk integration
        gap_scores = analyzer.get_gap_scores(drift_reports)

        # Count by drift type
        by_type = {}
        for dtype in DriftType:
            count = sum(1 for r in drift_reports.values() if r.drift_type == dtype)
            if count > 0:
                by_type[dtype.value] = count

        # Count by severity
        alerts = []
        for report in drift_reports.values():
            if report.severity in (Severity.ERROR, Severity.CRITICAL):
                alerts.append({
                    "component": report.component_id,
                    "drift_type": report.drift_type.value,
                    "severity": report.severity.value,
                    "drift_days": report.drift_days,
                    "message": report.message,
                })

        # Sort alerts by severity and drift
        alerts.sort(key=lambda a: (
            0 if a["severity"] == "critical" else 1 if a["severity"] == "error" else 2,
            -a["drift_days"]
        ))

        return {
            "components": components,
            "summary": summary,
            "by_type": by_type,
            "alerts": alerts[:20],  # Top 20 alerts
            "gap_scores": gap_scores,
            "doc_path": str(doc_path),
            "total_components": len(components),
        }

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        """Return empty result with reason."""
        return {
            "components": {},
            "summary": {"total_components": 0, "message": reason},
            "by_type": {},
            "alerts": [],
            "gap_scores": {},
            "doc_path": None,
            "total_components": 0,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        total = data.get("total_components", 0)
        alerts = data.get("alerts", [])
        by_type = data.get("by_type", {})

        if total == 0:
            return "Drift analysis: No components to analyze or documentation not found."

        parts = [f"Drift analysis: {total} components analyzed."]

        # Type breakdown
        if by_type:
            type_parts = []
            if by_type.get("undocumented"):
                type_parts.append(f"{by_type['undocumented']} undocumented")
            if by_type.get("synchronized"):
                type_parts.append(f"{by_type['synchronized']} synchronized")
            if by_type.get("stable"):
                type_parts.append(f"{by_type['stable']} stable")
            if by_type.get("spec_ahead"):
                type_parts.append(f"{by_type['spec_ahead']} spec-ahead")
            if type_parts:
                parts.append(f"Types: {', '.join(type_parts)}.")

        # Alerts
        if alerts:
            parts.append(f"Alerts: {len(alerts)} high-severity drift issues.")
            top_alert = alerts[0]
            parts.append(f"Top: {top_alert['component']} ({top_alert['drift_days']} days drift).")

        return " ".join(parts)
