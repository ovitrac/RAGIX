"""
Kernel: Risk Assessment
Stage: 2 (Analysis)
Wraps: ragix_audit.risk.RiskScorer

Computes Service Life Risk scores for components:
- Volatility risk (change frequency)
- Impact risk (propagation, centrality)
- Complexity risk (CC, LOC)
- Maturity risk (lifecycle category)
- Documentation gap risk

Outputs risk levels: LOW, MEDIUM, HIGH, CRITICAL

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class RiskKernel(Kernel):
    """
    Compute risk scores using RiskScorer.

    This kernel wraps ragix_audit.risk.RiskScorer to compute
    Service Life Risk scores for components based on multiple factors.

    Configuration options:
        weights: Custom risk factor weights (optional)

    Dependencies:
        timeline: Component timelines (required)
        metrics: Code metrics for complexity (optional)
        coupling: Coupling data for impact (optional)

    Output:
        risks: Per-component risk assessments
        by_level: Components grouped by risk level
        critical_components: High/Critical risk components
        summary: Risk distribution summary
    """

    name = "risk"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Service Life Risk assessment"

    requires = ["timeline"]
    provides = ["risk_scores", "critical_components", "mco_risk"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Compute risk scores using RiskScorer."""

        # Import existing tools (REUSE!)
        from ragix_audit.risk import RiskScorer, RiskLevel
        from ragix_audit.timeline import ComponentTimeline, LifecycleCategory, ComponentType

        # Get configuration
        custom_weights = input.config.get("weights", None)

        # Load timeline data (required)
        timeline_path = input.dependencies.get("timeline")
        if not timeline_path or not timeline_path.exists():
            raise RuntimeError("Missing required dependency: timeline")

        with open(timeline_path) as f:
            timeline_data = json.load(f).get("data", {})

        # Load optional dependencies for enhanced risk scoring
        metrics_path = input.dependencies.get("metrics")
        coupling_path = input.dependencies.get("coupling")

        metrics_data = {}
        if metrics_path and metrics_path.exists():
            with open(metrics_path) as f:
                metrics_data = json.load(f).get("data", {})

        coupling_data = {}
        if coupling_path and coupling_path.exists():
            with open(coupling_path) as f:
                coupling_data = json.load(f).get("data", {})

        logger.info("[risk] Computing Service Life Risk scores")

        # Reconstruct ComponentTimeline objects from JSON
        component_timelines = {}
        for ct_dict in timeline_data.get("component_timelines", []):
            comp_id = ct_dict.get("component_id", "")
            if not comp_id:
                continue

            # Reconstruct the ComponentTimeline
            from datetime import datetime
            ct = ComponentTimeline(
                component_id=comp_id,
                type=ComponentType(ct_dict.get("type", "unknown")),
                files=ct_dict.get("files", []) if isinstance(ct_dict.get("files"), list) else [],
                age_days=ct_dict.get("age_days", 0),
                estimated_changes=ct_dict.get("estimated_changes", 0),
                volatility=ct_dict.get("volatility", 0.0),
                category=LifecycleCategory(ct_dict.get("category", "unknown")),
            )
            # Parse dates if present
            if ct_dict.get("first_seen"):
                try:
                    ct.first_seen = datetime.fromisoformat(ct_dict["first_seen"])
                except (ValueError, TypeError):
                    pass
            if ct_dict.get("last_change"):
                try:
                    ct.last_change = datetime.fromisoformat(ct_dict["last_change"])
                except (ValueError, TypeError):
                    pass

            component_timelines[comp_id] = ct

        if not component_timelines:
            logger.warning("[risk] No component timelines found")
            return {
                "risks": {},
                "by_level": {},
                "critical_components": [],
                "summary": {"total_components": 0},
            }

        # Build AST metrics for enhanced scoring
        ast_metrics = {}

        # Extract complexity from metrics kernel output
        if metrics_data:
            hotspots = metrics_data.get("hotspots", [])
            for h in hotspots:
                name = h.get("name", "")
                cc = h.get("complexity", 0)
                # Map to component (simplified - use prefix matching)
                for comp_id in component_timelines:
                    if comp_id in name or name.startswith(comp_id):
                        if comp_id not in ast_metrics:
                            ast_metrics[comp_id] = {"complexities": [], "avg_cc": 0}
                        ast_metrics[comp_id]["complexities"].append(cc)

            # Compute average CC per component
            for comp_id, data in ast_metrics.items():
                if data["complexities"]:
                    data["avg_cc"] = sum(data["complexities"]) / len(data["complexities"])

        # Extract propagation factors from coupling kernel output
        if coupling_data:
            propagation = coupling_data.get("propagation", {})
            critical_nodes = propagation.get("critical_nodes", [])
            for node in critical_nodes:
                node_id = node if isinstance(node, str) else node.get("node", "")
                pf = node.get("pf", 0.5) if isinstance(node, dict) else 0.5
                # Map to component
                for comp_id in component_timelines:
                    if comp_id in node_id:
                        if comp_id not in ast_metrics:
                            ast_metrics[comp_id] = {}
                        ast_metrics[comp_id]["propagation_factor"] = pf

        # Use existing RiskScorer (REUSE!)
        scorer = RiskScorer(
            weights=custom_weights,
            ast_metrics=ast_metrics,
        )
        risks = scorer.score_all(component_timelines)
        summary = scorer.get_risk_summary(risks)

        # Convert to JSON-serializable format
        risks_data = {}
        for comp_id, risk in risks.items():
            risks_data[comp_id] = risk.to_dict()

        # Group by risk level
        by_level = {}
        for level in RiskLevel:
            components = [
                {"component": r.component_id, "score": round(r.score, 3)}
                for r in risks.values() if r.level == level
            ]
            if components:
                by_level[level.value] = sorted(components, key=lambda x: -x["score"])

        # Critical components (HIGH + CRITICAL)
        critical_components = [
            {
                "component": r.component_id,
                "score": round(r.score, 3),
                "level": r.level.value,
                "factors": r.factors.to_dict(),
                "recommendation": r.recommendation,
            }
            for r in sorted(risks.values(), key=lambda x: -x.score)
            if r.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        ][:30]  # Limit output

        # Statistics
        statistics = {
            "total_components": len(risks),
            "by_level_count": {level.value: len([r for r in risks.values() if r.level == level]) for level in RiskLevel},
            "critical_count": sum(1 for r in risks.values() if r.level == RiskLevel.CRITICAL),
            "high_count": sum(1 for r in risks.values() if r.level == RiskLevel.HIGH),
            "avg_risk_score": summary.get("avg_risk_score", 0),
        }

        return {
            "risks": risks_data,
            "by_level": by_level,
            "critical_components": critical_components,
            "summary": summary,
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        by_level = stats.get("by_level_count", {})
        critical = data.get("critical_components", [])

        # Top 3 critical components
        top_critical = []
        for c in critical[:3]:
            top_critical.append(f"{c['component']}({c['score']:.2f})")
        top_str = ", ".join(top_critical) if top_critical else "none"

        return (
            f"Risk: {stats.get('total_components', 0)} components. "
            f"Critical: {stats.get('critical_count', 0)}, "
            f"High: {stats.get('high_count', 0)}, "
            f"Medium: {by_level.get('medium', 0)}, "
            f"Low: {by_level.get('low', 0)}. "
            f"Avg score: {stats.get('avg_risk_score', 0):.2f}. "
            f"Top risk: {top_str}."
        )
