"""
Risk Scoring Module — Service Life Risk Assessment

Computes risk scores for components based on:
- Volatility (change frequency)
- Impact (propagation factor, centrality)
- Complexity (CC, LOC)
- Maturity (lifecycle category)
- Documentation gaps

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

from ragix_audit.timeline import ComponentTimeline, LifecycleCategory, TimelineScanner
from ragix_audit.component_mapper import ComponentType

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"           # 0.0 - 0.25
    MEDIUM = "medium"     # 0.25 - 0.50
    HIGH = "high"         # 0.50 - 0.75
    CRITICAL = "critical" # 0.75 - 1.0


@dataclass
class RiskFactors:
    """Individual risk factor scores (0-1 scale)."""
    volatility: float = 0.0
    impact: float = 0.0
    complexity: float = 0.0
    maturity_penalty: float = 0.0
    doc_gap: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "volatility": round(self.volatility, 3),
            "impact": round(self.impact, 3),
            "complexity": round(self.complexity, 3),
            "maturity_penalty": round(self.maturity_penalty, 3),
            "doc_gap": round(self.doc_gap, 3),
        }


@dataclass
class ServiceLifeRisk:
    """Complete risk assessment for a component."""
    component_id: str
    score: float                    # 0-1 overall risk
    level: RiskLevel
    factors: RiskFactors
    recommendation: str
    timeline: Optional[ComponentTimeline] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "score": round(self.score, 3),
            "level": self.level.value,
            "factors": self.factors.to_dict(),
            "recommendation": self.recommendation,
            "timeline": self.timeline.to_dict() if self.timeline else None,
        }


class RiskScorer:
    """
    Computes Service Life Risk scores for components.

    Risk = weighted sum of:
    - Volatility score (normalized change frequency)
    - Impact score (propagation factor from AST, or file count proxy)
    - Complexity score (normalized CC or LOC)
    - Maturity penalty (based on lifecycle category)
    - Doc gap score (documentation drift)
    """

    # Default weights (configurable)
    DEFAULT_WEIGHTS = {
        "volatility": 0.20,
        "impact": 0.25,
        "complexity": 0.20,
        "maturity": 0.25,
        "doc_gap": 0.10,
    }

    # Maturity penalty by lifecycle category
    MATURITY_PENALTIES = {
        LifecycleCategory.NEW: 0.3,           # Some risk, learning curve
        LifecycleCategory.ACTIVE: 0.2,        # Managed risk
        LifecycleCategory.MATURE: 0.1,        # Low risk
        LifecycleCategory.LEGACY_HOT: 0.9,    # High MCO risk!
        LifecycleCategory.FROZEN: 0.2,        # Low but unknown
        LifecycleCategory.FROZEN_RISKY: 0.7,  # Hidden risk
        LifecycleCategory.UNKNOWN: 0.4,
    }

    # Normalization thresholds
    MAX_VOLATILITY = 0.5       # Changes per month
    MAX_FILE_COUNT = 100       # Files per component (proxy for impact)
    MAX_COMPLEXITY = 15        # Average CC threshold

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ast_metrics: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize risk scorer.

        Args:
            weights: Custom weights for risk factors
            ast_metrics: AST-derived metrics per component (CC, coupling, PF)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.ast_metrics = ast_metrics or {}

    def score_component(
        self,
        timeline: ComponentTimeline,
        doc_gap_score: float = 0.0
    ) -> ServiceLifeRisk:
        """
        Compute risk score for a component.

        Args:
            timeline: Component timeline data
            doc_gap_score: Pre-computed documentation gap score (0-1)

        Returns:
            ServiceLifeRisk with score, level, and recommendation
        """
        factors = RiskFactors()

        # 1. Volatility score
        factors.volatility = min(1.0, timeline.volatility / self.MAX_VOLATILITY)

        # 2. Impact score (use AST propagation factor if available, else file count proxy)
        if timeline.component_id in self.ast_metrics:
            metrics = self.ast_metrics[timeline.component_id]
            factors.impact = metrics.get("propagation_factor", 0.0)
        else:
            # Proxy: more files = higher impact
            factors.impact = min(1.0, timeline.file_count / self.MAX_FILE_COUNT)

        # 3. Complexity score (use AST CC if available, else estimate from file count)
        if timeline.component_id in self.ast_metrics:
            metrics = self.ast_metrics[timeline.component_id]
            avg_cc = metrics.get("avg_cc", 5.0)
            factors.complexity = min(1.0, avg_cc / self.MAX_COMPLEXITY)
        else:
            # Estimate: more files often means more complexity
            factors.complexity = min(1.0, (timeline.file_count / 50) * 0.5)

        # 4. Maturity penalty
        factors.maturity_penalty = self.MATURITY_PENALTIES.get(
            timeline.category,
            self.MATURITY_PENALTIES[LifecycleCategory.UNKNOWN]
        )

        # 5. Documentation gap
        factors.doc_gap = doc_gap_score

        # Calculate weighted score
        score = (
            self.weights["volatility"] * factors.volatility +
            self.weights["impact"] * factors.impact +
            self.weights["complexity"] * factors.complexity +
            self.weights["maturity"] * factors.maturity_penalty +
            self.weights["doc_gap"] * factors.doc_gap
        )
        score = min(1.0, max(0.0, score))

        # Determine risk level
        level = self._score_to_level(score)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            timeline, factors, score, level
        )

        return ServiceLifeRisk(
            component_id=timeline.component_id,
            score=score,
            level=level,
            factors=factors,
            recommendation=recommendation,
            timeline=timeline,
        )

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score < 0.25:
            return RiskLevel.LOW
        elif score < 0.50:
            return RiskLevel.MEDIUM
        elif score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _generate_recommendation(
        self,
        timeline: ComponentTimeline,
        factors: RiskFactors,
        score: float,
        level: RiskLevel
    ) -> str:
        """Generate human-readable recommendation."""
        comp_id = timeline.component_id
        age_years = round(timeline.age_years, 1)
        category = timeline.category.value.replace("_", " ").title()

        # Base message
        if level == RiskLevel.CRITICAL:
            base = f"{comp_id} requires immediate attention."
        elif level == RiskLevel.HIGH:
            base = f"{comp_id} has elevated MCO risk."
        elif level == RiskLevel.MEDIUM:
            base = f"{comp_id} needs monitoring."
        else:
            base = f"{comp_id} is stable."

        # Specific insights based on top factors
        insights = []

        if timeline.category == LifecycleCategory.LEGACY_HOT:
            insights.append(f"Legacy ({age_years}y) but still actively changing - high maintenance burden")

        if factors.volatility > 0.6:
            insights.append("High change frequency - consider stabilization")

        if factors.impact > 0.5:
            insights.append("High impact - changes affect many dependents")

        if factors.complexity > 0.6:
            insights.append("High complexity - refactoring candidate")

        if factors.doc_gap > 0.5:
            insights.append("Documentation gap - update specs")

        if timeline.category == LifecycleCategory.FROZEN and factors.complexity > 0.4:
            insights.append("Frozen but complex - hidden risk, needs review before changes")

        if timeline.category == LifecycleCategory.NEW:
            insights.append(f"New component ({timeline.age_days} days) - still stabilizing")

        if not insights:
            if level == RiskLevel.LOW:
                insights.append(f"Mature and stable ({category})")
            else:
                insights.append(f"Status: {category}")

        return f"{base} {'; '.join(insights)}."

    def score_all(
        self,
        timelines: Dict[str, ComponentTimeline],
        doc_gaps: Optional[Dict[str, float]] = None
    ) -> Dict[str, ServiceLifeRisk]:
        """
        Score all components.

        Args:
            timelines: Component timelines from TimelineScanner
            doc_gaps: Optional pre-computed doc gap scores per component

        Returns:
            Dictionary of component_id -> ServiceLifeRisk
        """
        doc_gaps = doc_gaps or {}
        results = {}

        for comp_id, timeline in timelines.items():
            doc_gap_score = doc_gaps.get(comp_id, 0.0)
            results[comp_id] = self.score_component(timeline, doc_gap_score)

        return results

    def get_risk_summary(
        self,
        risks: Dict[str, ServiceLifeRisk]
    ) -> Dict[str, Any]:
        """Generate summary statistics for risk assessment."""
        by_level = {}
        for level in RiskLevel:
            components = [r for r in risks.values() if r.level == level]
            if components:
                by_level[level.value] = {
                    "count": len(components),
                    "components": [r.component_id for r in components],
                    "avg_score": sum(r.score for r in components) / len(components),
                }

        critical_components = [
            r for r in risks.values()
            if r.level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        ]
        critical_components.sort(key=lambda r: r.score, reverse=True)

        return {
            "total_components": len(risks),
            "by_level": by_level,
            "critical_components": [
                {"id": r.component_id, "score": r.score, "recommendation": r.recommendation}
                for r in critical_components[:10]
            ],
            "avg_risk_score": sum(r.score for r in risks.values()) / len(risks) if risks else 0,
        }


def compute_risks_for_project(
    project_path: str,
    ast_metrics: Optional[Dict[str, Dict]] = None
) -> Dict[str, Any]:
    """
    Convenience function to compute risks for a project.

    Args:
        project_path: Path to project source
        ast_metrics: Optional AST-derived metrics

    Returns:
        Dictionary with risks and summary
    """
    from pathlib import Path

    scanner = TimelineScanner()
    scanner.scan_directory(Path(project_path))
    timelines = scanner.build_component_timelines()

    scorer = RiskScorer(ast_metrics=ast_metrics)
    risks = scorer.score_all(timelines)
    summary = scorer.get_risk_summary(risks)

    return {
        "risks": {k: v.to_dict() for k, v in risks.items()},
        "summary": summary,
        "timeline_summary": scanner.get_summary(),
    }


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "/path/to/enterprise-audit/src"

    print(f"Computing risks for {path}...")
    result = compute_risks_for_project(path)

    print("\n=== Risk Summary ===\n")
    print(json.dumps(result["summary"], indent=2))

    print("\n=== Component Risks (sorted by score) ===\n")
    sorted_risks = sorted(
        result["risks"].items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    for comp_id, risk in sorted_risks:
        print(f"{comp_id}: {risk['score']:.3f} ({risk['level'].upper()})")
        print(f"  Factors: V={risk['factors']['volatility']:.2f} I={risk['factors']['impact']:.2f} "
              f"C={risk['factors']['complexity']:.2f} M={risk['factors']['maturity_penalty']:.2f}")
        print(f"  {risk['recommendation']}")
        print()
