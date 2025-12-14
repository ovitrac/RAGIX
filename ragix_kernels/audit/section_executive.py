"""
Kernel: Executive Summary Section
Stage: 3 (Report)

Generates the executive summary section of the audit report:
- Key findings synthesis
- Critical metrics at a glance
- Risk overview
- Top recommendations

This kernel synthesizes data from Stage 1 & 2 kernels into
an executive-friendly format with language support.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.audit.report.i18n import I18N, get_translator
from ragix_kernels.audit.report.templates import FrontMatter, get_template_text
from ragix_kernels.audit.report.renderer import MarkdownRenderer, markdown_table

logger = logging.getLogger(__name__)


class SectionExecutiveKernel(Kernel):
    """
    Generate executive summary section.

    Configuration options:
        language: Report language ("en" or "fr", default: "en")
        max_findings: Maximum key findings (default: 5)
        max_recommendations: Maximum recommendations (default: 5)
        include_charts: Include ASCII charts (default: true)
        audit_context: Audit context with objectives and questions

    Dependencies:
        metrics: Code metrics (required)
        stats_summary: Statistical summary (required)
        risk: Risk assessment (optional)
        coupling: Coupling analysis (optional)

    Output:
        markdown: Executive summary Markdown content
        key_findings: List of key findings
        metrics_summary: Key metrics dictionary
        grade: Overall quality grade
    """

    name = "section_executive"
    version = "1.0.0"
    category = "audit"
    stage = 3
    description = "Generate executive summary section"

    requires = ["metrics", "stats_summary"]
    provides = ["executive_summary_md", "key_findings"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Generate executive summary section."""

        # Get configuration
        language = input.config.get("language", "en")
        max_findings = input.config.get("max_findings", 5)
        max_recommendations = input.config.get("max_recommendations", 5)
        include_charts = input.config.get("include_charts", True)
        audit_context = input.config.get("audit_context", {})
        project_config = input.config.get("project", {})

        # Initialize i18n and renderer
        i18n = get_translator(language)
        renderer = MarkdownRenderer(i18n=i18n)

        # Load Stage 1 & 2 outputs
        metrics_data = self._load_dependency(input, "metrics")
        stats_data = self._load_dependency(input, "stats_summary")
        risk_data = self._load_dependency(input, "risk") or {}
        coupling_data = self._load_dependency(input, "coupling") or {}
        hotspots_data = self._load_dependency(input, "hotspots") or {}

        logger.info(f"[section_executive] Generating in {language}")

        # Extract key metrics
        overview = stats_data.get("overview", {})
        quality = stats_data.get("quality", {})
        recommendations = stats_data.get("recommendations", [])

        # Build key findings
        key_findings = self._extract_key_findings(
            overview, quality, risk_data, coupling_data, hotspots_data, i18n
        )[:max_findings]

        # Build metrics summary
        metrics_summary = self._build_metrics_summary(overview, quality, risk_data)

        # Generate Markdown content
        md_lines = []

        # Section title
        md_lines.append(f"# {i18n.t('section.executive')}\n")

        # Context introduction
        intro = get_template_text("executive_intro", language)
        intro = intro.format(
            project=project_config.get("name", "Unknown"),
            file_count=overview.get("total_files", 0),
            class_count=overview.get("total_classes", 0),
            loc=overview.get("total_loc", 0),
        )
        md_lines.append(intro)

        # Audit objectives (if provided)
        if audit_context.get("objectives"):
            md_lines.append(f"\n## {i18n.t('context.audit_objectives')}\n")
            for obj in audit_context["objectives"]:
                md_lines.append(f"- {obj}")
            md_lines.append("")

        # Key findings
        md_lines.append(f"\n## {i18n.t('subsection.key_findings')}\n")
        for finding in key_findings:
            md_lines.append(f"- {finding}")
        md_lines.append("")

        # Metrics at a glance
        md_lines.append(f"\n## {i18n.t('subsection.metrics_summary')}\n")
        md_lines.append(renderer.render_metrics_table(
            metrics_summary,
            caption=i18n.t("caption.overview_metrics"),
        ))

        # Quality grades
        if quality:
            md_lines.append(f"\n### {i18n.t('label.quality')}\n")
            grade_data = {
                k: v for k, v in quality.items()
                if k not in ("overall_grade", "methods_per_class")
            }
            md_lines.append(renderer.render_quality_grades(
                grade_data,
                caption=i18n.t("caption.quality_grades"),
            ))

            # Overall grade with interpretation
            overall = quality.get("overall_grade", "C")
            md_lines.append(f"\n**{i18n.t('label.overall_grade')}:** {overall} ({i18n.grade(overall)})")

        # Risk overview (if available)
        if risk_data.get("statistics"):
            risk_stats = risk_data["statistics"]
            md_lines.append(f"\n## {i18n.t('section.risk')}\n")

            by_level = risk_stats.get("by_level_count", {})
            if by_level and include_charts:
                md_lines.append(renderer.render_risk_distribution(by_level))

            # Critical/high risk count
            critical = risk_stats.get("critical_count", 0)
            high = risk_stats.get("high_count", 0)
            if critical > 0 or high > 0:
                md_lines.append(f"\n> ⚠️ {i18n.t('interp.mco_risk')}")

        # Top recommendations
        if recommendations:
            md_lines.append(f"\n## {i18n.t('context.recommendations_summary')}\n")
            md_lines.append(renderer.render_recommendations(
                recommendations[:max_recommendations],
                caption=i18n.t("caption.recommendations"),
            ))

        # Conclusions (answer audit questions if provided)
        if audit_context.get("questions"):
            md_lines.append(f"\n## {i18n.t('context.key_findings')}\n")
            md_lines.append(self._generate_conclusions(
                audit_context["questions"],
                overview, quality, risk_data, i18n
            ))

        markdown_content = "\n".join(md_lines)

        return {
            "markdown": markdown_content,
            "key_findings": key_findings,
            "metrics_summary": metrics_summary,
            "grade": quality.get("overall_grade", "C"),
            "language": language,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        grade = data.get("grade", "C")
        findings_count = len(data.get("key_findings", []))
        lang = data.get("language", "en")

        return (
            f"Executive summary generated ({lang}). "
            f"Overall grade: {grade}. "
            f"Key findings: {findings_count}."
        )

    def _load_dependency(self, input: KernelInput, name: str) -> Optional[Dict[str, Any]]:
        """Load dependency data if available."""
        path = input.dependencies.get(name)
        if path and path.exists():
            with open(path) as f:
                return json.load(f).get("data", {})
        return None

    def _extract_key_findings(
        self,
        overview: Dict[str, Any],
        quality: Dict[str, Any],
        risk_data: Dict[str, Any],
        coupling_data: Dict[str, Any],
        hotspots_data: Dict[str, Any],
        i18n: I18N,
    ) -> List[str]:
        """Extract key findings from analysis data."""
        findings = []

        # Size finding
        total_loc = overview.get("total_loc", 0)
        total_files = overview.get("total_files", 0)
        total_classes = overview.get("total_classes", 0)
        findings.append(
            f"Codebase contains {total_files:,} files, {total_classes:,} classes, "
            f"and {total_loc:,} lines of code."
        )

        # Quality finding
        grade = quality.get("overall_grade", "C")
        mi = overview.get("maintainability_index", 0)
        if grade in ("A", "B"):
            findings.append(
                f"Overall quality is {i18n.grade(grade).lower()} (Grade {grade}). "
                f"Maintainability Index: {mi:.0f}/100."
            )
        else:
            findings.append(
                f"Quality requires attention (Grade {grade}). "
                f"Maintainability Index: {mi:.0f}/100."
            )

        # Complexity finding
        avg_cc = overview.get("avg_complexity", 0)
        if avg_cc > 10:
            findings.append(
                f"High average complexity (CC={avg_cc:.1f}). "
                f"Refactoring recommended for complex methods."
            )
        elif avg_cc > 5:
            findings.append(
                f"Moderate complexity (CC={avg_cc:.1f}). "
                f"Some methods may benefit from simplification."
            )

        # Technical debt finding
        debt_days = overview.get("technical_debt_days", 0)
        if debt_days > 0:
            findings.append(
                f"Technical debt estimated at {debt_days:.1f} person-days "
                f"({overview.get('technical_debt_hours', 0):.1f} hours)."
            )

        # Risk finding
        risk_stats = risk_data.get("statistics", {})
        critical = risk_stats.get("critical_count", 0)
        high = risk_stats.get("high_count", 0)
        if critical > 0:
            findings.append(
                f"⚠️ {critical} critical-risk components identified requiring immediate attention."
            )
        elif high > 0:
            findings.append(
                f"{high} high-risk components identified requiring monitoring."
            )

        # Coupling finding
        coupling_stats = coupling_data.get("statistics", {})
        sdp_violations = coupling_stats.get("sdp_violations", 0)
        if sdp_violations > 0:
            findings.append(
                f"{sdp_violations} Stable Dependencies Principle violations detected."
            )

        # Hotspots finding
        hotspots = hotspots_data.get("complexity_hotspots", [])
        if hotspots:
            findings.append(
                f"{len(hotspots)} complexity hotspots identified (CC > threshold)."
            )

        return findings

    def _build_metrics_summary(
        self,
        overview: Dict[str, Any],
        quality: Dict[str, Any],
        risk_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build key metrics summary dictionary."""
        risk_stats = risk_data.get("statistics", {})

        return {
            "files": overview.get("total_files", 0),
            "classes": overview.get("total_classes", 0),
            "methods": overview.get("total_methods", 0),
            "lines_of_code": overview.get("total_loc", 0),
            "complexity": f"{overview.get('avg_complexity', 0):.1f} (avg)",
            "maintainability": f"{overview.get('maintainability_index', 0):.0f}/100",
            "technical_debt": f"{overview.get('technical_debt_days', 0):.1f} days",
            "critical_risk": risk_stats.get("critical_count", 0),
            "high_risk": risk_stats.get("high_count", 0),
        }

    def _generate_conclusions(
        self,
        questions: List[str],
        overview: Dict[str, Any],
        quality: Dict[str, Any],
        risk_data: Dict[str, Any],
        i18n: I18N,
    ) -> str:
        """Generate conclusions answering audit questions."""
        lines = []

        for i, question in enumerate(questions, 1):
            lines.append(f"**Q{i}: {question}**\n")

            # Auto-generate answer based on question keywords
            q_lower = question.lower()

            if any(kw in q_lower for kw in ["quality", "qualite", "mainten"]):
                grade = quality.get("overall_grade", "C")
                mi = overview.get("maintainability_index", 0)
                lines.append(
                    f"> The codebase achieves a quality grade of **{grade}** "
                    f"with a Maintainability Index of **{mi:.0f}/100**.\n"
                )
            elif any(kw in q_lower for kw in ["risk", "risque", "mco"]):
                risk_stats = risk_data.get("statistics", {})
                critical = risk_stats.get("critical_count", 0)
                high = risk_stats.get("high_count", 0)
                lines.append(
                    f"> Risk assessment identifies **{critical} critical** and "
                    f"**{high} high-risk** components requiring attention.\n"
                )
            elif any(kw in q_lower for kw in ["debt", "dette"]):
                debt = overview.get("technical_debt_days", 0)
                lines.append(
                    f"> Technical debt is estimated at **{debt:.1f} person-days**. "
                    f"Prioritize refactoring in high-complexity areas.\n"
                )
            elif any(kw in q_lower for kw in ["complex", "hotspot"]):
                cc = overview.get("avg_complexity", 0)
                lines.append(
                    f"> Average complexity is **{cc:.1f}**. "
                    f"See hotspots section for detailed analysis.\n"
                )
            else:
                lines.append(
                    f"> See detailed analysis sections for comprehensive answer.\n"
                )

            lines.append("")

        return "\n".join(lines)
