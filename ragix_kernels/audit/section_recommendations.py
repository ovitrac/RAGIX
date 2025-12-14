"""
Kernel: Recommendations Section
Stage: 3 (Report)

Generates the recommendations section with:
- Prioritized action items
- Effort estimates
- Risk mitigation strategies
- Action plan synthesis

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.audit.report.i18n import I18N, get_translator
from ragix_kernels.audit.report.templates import get_template_text
from ragix_kernels.audit.report.renderer import MarkdownRenderer, markdown_table

logger = logging.getLogger(__name__)


class SectionRecommendationsKernel(Kernel):
    """
    Generate recommendations section.

    Configuration options:
        language: Report language ("en" or "fr")
        max_recommendations: Maximum items per category (default: 10)
        include_effort: Include effort estimates (default: true)

    Dependencies:
        stats_summary: Recommendations from stats (required)
        risk: Risk assessment (optional)
        hotspots: Complexity hotspots (optional)
        coupling: Coupling analysis (optional)
        dead_code: Dead code analysis (optional)

    Output:
        markdown: Recommendations section Markdown content
        recommendations: Consolidated recommendations list
        action_plan: Prioritized action plan
    """

    name = "section_recommendations"
    version = "1.0.0"
    category = "audit"
    stage = 3
    description = "Generate recommendations section"

    requires = ["stats_summary"]
    provides = ["recommendations_section_md", "action_plan"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Generate recommendations section."""

        # Get configuration
        language = input.config.get("language", "en")
        max_per_category = input.config.get("max_recommendations", 10)
        include_effort = input.config.get("include_effort", True)

        # Initialize i18n and renderer
        i18n = get_translator(language)
        renderer = MarkdownRenderer(i18n=i18n)

        # Load dependencies
        stats_data = self._load_dependency(input, "stats_summary")
        risk_data = self._load_dependency(input, "risk")
        hotspots_data = self._load_dependency(input, "hotspots")
        coupling_data = self._load_dependency(input, "coupling")
        dead_code_data = self._load_dependency(input, "dead_code")

        logger.info(f"[section_recommendations] Generating in {language}")

        # Consolidate recommendations from all sources
        all_recommendations = self._consolidate_recommendations(
            stats_data, risk_data, hotspots_data, coupling_data, dead_code_data, i18n
        )

        # Prioritize and group
        grouped = self._group_by_priority(all_recommendations)

        # Generate Markdown
        md_lines = []

        # Section title
        md_lines.append(f"# {i18n.t('section.recommendations')}\n")

        # Introduction
        intro = get_template_text("recommendations_intro", language)
        md_lines.append(intro)

        # Summary counts
        total = len(all_recommendations)
        immediate = len(grouped.get("immediate", []))
        short_term = len(grouped.get("short_term", []))
        medium_term = len(grouped.get("medium_term", []))

        md_lines.append(f"\n**Total: {total} recommendations**\n")
        md_lines.append(f"- 🔴 Immediate: {immediate}")
        md_lines.append(f"- 🟠 Short-term: {short_term}")
        md_lines.append(f"- 🟡 Medium-term: {medium_term}")
        md_lines.append(f"- 🟢 Long-term: {len(grouped.get('long_term', []))}")
        md_lines.append("")

        # Immediate actions
        if grouped.get("immediate"):
            md_lines.append(f"## 🔴 {i18n.t('priority.immediate')}\n")
            md_lines.append(self._render_recommendations_list(
                grouped["immediate"][:max_per_category], i18n, include_effort
            ))

        # Short-term actions
        if grouped.get("short_term"):
            md_lines.append(f"## 🟠 {i18n.t('priority.short_term')}\n")
            md_lines.append(self._render_recommendations_list(
                grouped["short_term"][:max_per_category], i18n, include_effort
            ))

        # Medium-term actions
        if grouped.get("medium_term"):
            md_lines.append(f"## 🟡 {i18n.t('priority.medium_term')}\n")
            md_lines.append(self._render_recommendations_list(
                grouped["medium_term"][:max_per_category], i18n, include_effort
            ))

        # Long-term actions
        if grouped.get("long_term"):
            md_lines.append(f"## 🟢 {i18n.t('priority.long_term')}\n")
            md_lines.append(self._render_recommendations_list(
                grouped["long_term"][:max_per_category], i18n, include_effort
            ))

        # Action plan table
        md_lines.append(f"\n## {i18n.t('subsection.action_plan')}\n")
        md_lines.append(renderer.render_recommendations(
            all_recommendations[:20],
            caption=i18n.t("caption.recommendations"),
        ))

        # Effort summary
        if include_effort:
            total_effort = sum(
                r.get("effort_hours", 0) for r in all_recommendations
            )
            if total_effort > 0:
                md_lines.append(f"\n**Total estimated effort:** {total_effort:.0f} hours "
                              f"({total_effort/8:.1f} person-days)\n")

        markdown_content = "\n".join(md_lines)

        return {
            "markdown": markdown_content,
            "recommendations": all_recommendations,
            "action_plan": {
                "immediate": grouped.get("immediate", []),
                "short_term": grouped.get("short_term", []),
                "medium_term": grouped.get("medium_term", []),
                "long_term": grouped.get("long_term", []),
            },
            "counts": {
                "total": total,
                "immediate": immediate,
                "short_term": short_term,
                "medium_term": medium_term,
            },
            "language": language,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        counts = data.get("counts", {})
        return (
            f"Recommendations section generated. "
            f"Total: {counts.get('total', 0)}, "
            f"Immediate: {counts.get('immediate', 0)}, "
            f"Short-term: {counts.get('short_term', 0)}."
        )

    def _load_dependency(self, input: KernelInput, name: str) -> Dict[str, Any]:
        """Load dependency data."""
        path = input.dependencies.get(name)
        if path and path.exists():
            with open(path) as f:
                return json.load(f).get("data", {})
        return {}

    def _consolidate_recommendations(
        self,
        stats_data: Dict,
        risk_data: Dict,
        hotspots_data: Dict,
        coupling_data: Dict,
        dead_code_data: Dict,
        i18n: I18N,
    ) -> List[Dict[str, Any]]:
        """Consolidate recommendations from all analysis sources."""
        recommendations = []

        # From stats_summary
        for rec in stats_data.get("recommendations", []):
            rec["source"] = "quality"
            recommendations.append(rec)

        # From risk analysis
        critical_components = risk_data.get("critical_components", [])
        for comp in critical_components[:5]:
            recommendations.append({
                "title": i18n.t("reco.address_debt", component=comp.get("component", ""),
                              hours=8, days=1),
                "category": "risk",
                "priority": "immediate" if comp.get("level") == "critical" else "short_term",
                "severity": comp.get("level", "high"),
                "source": "risk",
                "component": comp.get("component"),
            })

        # From hotspots
        complexity_hotspots = hotspots_data.get("complexity_hotspots", [])
        for hs in complexity_hotspots[:5]:
            cc = hs.get("complexity", 0)
            if cc > 20:
                recommendations.append({
                    "title": i18n.t("reco.refactor_complexity",
                                  component=hs.get("name", ""), cc=cc),
                    "category": "complexity",
                    "priority": "immediate" if cc > 30 else "short_term",
                    "severity": "high" if cc > 30 else "medium",
                    "source": "hotspots",
                    "effort_hours": min(cc * 0.5, 16),
                })

        # From coupling
        sdp_violations = coupling_data.get("sdp_violations", [])
        for v in sdp_violations[:3]:
            recommendations.append({
                "title": i18n.t("reco.reduce_coupling",
                              package=v.get("source_package", ""),
                              ce=0, ca=0),
                "category": "architecture",
                "priority": "medium_term",
                "severity": v.get("severity", "medium"),
                "source": "coupling",
            })

        # From dead code
        dead_candidates = dead_code_data.get("dead_candidates", [])
        if dead_candidates:
            recommendations.append({
                "title": i18n.t("reco.remove_dead_code",
                              component="codebase",
                              count=len(dead_candidates)),
                "category": "maintenance",
                "priority": "medium_term",
                "severity": "medium",
                "source": "dead_code",
            })

        # Sort by priority
        priority_order = {"immediate": 0, "short_term": 1, "medium_term": 2, "long_term": 3}
        recommendations.sort(
            key=lambda x: priority_order.get(x.get("priority", "medium_term"), 2)
        )

        return recommendations

    def _group_by_priority(self, recommendations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group recommendations by priority."""
        grouped = {
            "immediate": [],
            "short_term": [],
            "medium_term": [],
            "long_term": [],
        }

        for rec in recommendations:
            priority = rec.get("priority", "medium_term").lower().replace("-", "_")
            if priority in grouped:
                grouped[priority].append(rec)
            else:
                grouped["medium_term"].append(rec)

        return grouped

    def _render_recommendations_list(
        self,
        recommendations: List[Dict],
        i18n: I18N,
        include_effort: bool,
    ) -> str:
        """Render recommendations as numbered list."""
        lines = []

        for i, rec in enumerate(recommendations, 1):
            title = rec.get("title", rec.get("description", ""))
            category = rec.get("category", "general").title()

            line = f"{i}. **[{category}]** {title}"

            if include_effort and rec.get("effort_hours"):
                line += f" *(~{rec['effort_hours']:.0f}h)*"

            lines.append(line)

        lines.append("")
        return "\n".join(lines)
