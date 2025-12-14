"""
Kernel: Codebase Overview Section
Stage: 3 (Report)

Generates the codebase overview section with:
- Project structure summary
- Size and complexity metrics
- Quality grades breakdown
- Distribution analysis

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.audit.report.i18n import I18N, get_translator
from ragix_kernels.audit.report.templates import get_template_text
from ragix_kernels.audit.report.renderer import (
    MarkdownRenderer,
    markdown_table,
    ascii_distribution,
)

logger = logging.getLogger(__name__)


class SectionOverviewKernel(Kernel):
    """
    Generate codebase overview section.

    Configuration options:
        language: Report language ("en" or "fr")
        include_distributions: Include distribution charts (default: true)

    Dependencies:
        ast_scan: AST data (required)
        metrics: Code metrics (required)
        stats_summary: Statistical summary (required)
        partition: Partition data (optional)

    Output:
        markdown: Overview section Markdown content
        structure: Codebase structure summary
        metrics: Key metrics
    """

    name = "section_overview"
    version = "1.0.0"
    category = "audit"
    stage = 3
    description = "Generate codebase overview section"

    requires = ["ast_scan", "metrics", "stats_summary"]
    provides = ["overview_section_md", "codebase_structure"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Generate overview section."""

        # Get configuration
        language = input.config.get("language", "en")
        include_distributions = input.config.get("include_distributions", True)
        project_config = input.config.get("project", {})

        # Initialize i18n and renderer
        i18n = get_translator(language)
        renderer = MarkdownRenderer(i18n=i18n)

        # Load dependencies
        ast_data = self._load_dependency(input, "ast_scan")
        metrics_data = self._load_dependency(input, "metrics")
        stats_data = self._load_dependency(input, "stats_summary")
        partition_data = self._load_dependency(input, "partition")

        logger.info(f"[section_overview] Generating in {language}")

        # Extract data
        ast_stats = ast_data.get("statistics", {})
        overview = stats_data.get("overview", {})
        quality = stats_data.get("quality", {})
        distributions = stats_data.get("distributions", {})
        box_plot_data = stats_data.get("box_plot_data", {})

        # Generate Markdown
        md_lines = []

        # Section title
        md_lines.append(f"# {i18n.t('section.overview')}\n")

        # Introduction
        intro = get_template_text("overview_intro", language)
        md_lines.append(intro)

        # Project info
        md_lines.append(f"## {i18n.t('context.audit_scope')}\n")
        md_lines.append(f"- **Project:** {project_config.get('name', 'Unknown')}")
        md_lines.append(f"- **Path:** `{project_config.get('path', 'N/A')}`")
        md_lines.append(f"- **Language:** {project_config.get('language', 'Unknown')}")
        md_lines.append("")

        # Size metrics
        md_lines.append(f"## {i18n.t('subsection.metrics_summary')}\n")

        size_metrics = {
            i18n.t("label.files"): overview.get("total_files", 0),
            i18n.t("label.classes"): overview.get("total_classes", 0),
            i18n.t("label.methods"): overview.get("total_methods", 0),
            i18n.t("label.functions"): overview.get("total_functions", 0),
            i18n.t("label.lines_of_code"): f"{overview.get('total_loc', 0):,}",
            "Code LOC": f"{overview.get('code_loc', 0):,}",
            "Comment LOC": f"{overview.get('comment_loc', 0):,}",
        }
        md_lines.append(renderer.render_metrics_table(size_metrics))
        md_lines.append("")

        # Symbol type distribution
        by_type = ast_stats.get("by_type", {})
        if by_type and include_distributions:
            md_lines.append(f"### Symbol Distribution\n")
            md_lines.append("```")
            md_lines.append(ascii_distribution(by_type, width=30))
            md_lines.append("```")
            md_lines.append("")

        # Quality metrics
        md_lines.append(f"## {i18n.t('section.quality')}\n")

        quality_metrics = {
            i18n.t("label.complexity") + " (avg)": f"{overview.get('avg_complexity', 0):.2f}",
            i18n.t("label.complexity") + " (median)": f"{overview.get('median_complexity', 0):.2f}",
            i18n.t("label.maintainability"): f"{overview.get('maintainability_index', 0):.1f}/100",
            i18n.t("label.technical_debt"): f"{overview.get('technical_debt_hours', 0):.1f}h",
        }
        md_lines.append(renderer.render_metrics_table(quality_metrics))
        md_lines.append("")

        # Quality grades
        if quality:
            md_lines.append(f"### Quality Grades\n")
            grade_data = {
                k: v for k, v in quality.items()
                if k not in ("overall_grade", "methods_per_class") and isinstance(v, str)
            }
            if grade_data:
                md_lines.append(renderer.render_quality_grades(grade_data))
                md_lines.append("")

            overall = quality.get("overall_grade", "C")
            md_lines.append(f"**Overall Grade:** {overall} ({i18n.grade(overall)})\n")

        # Complexity distribution
        cc_dist = distributions.get("complexity", {})
        if cc_dist and include_distributions:
            md_lines.append(f"### Complexity Distribution\n")
            md_lines.append("```")
            md_lines.append(f"Mean:   {cc_dist.get('mean', 0):.2f}")
            md_lines.append(f"Median: {cc_dist.get('median', 0):.2f}")
            md_lines.append(f"Std:    {cc_dist.get('std', 0):.2f}")
            md_lines.append(f"Max:    {cc_dist.get('max', 0):.0f}")
            if cc_dist.get("skewness", 0) > 0.5:
                md_lines.append(f"Skewness: {cc_dist.get('skewness', 0):.2f} (right-skewed)")
            md_lines.append("```")
            md_lines.append("")

            # Interpretation
            if cc_dist.get("skewness", 0) > 1.0:
                md_lines.append(f"> {i18n.t('interp.high_complexity')}\n")

        # LOC distribution
        loc_dist = distributions.get("loc_per_file", {})
        if loc_dist and include_distributions:
            md_lines.append(f"### File Size Distribution\n")
            md_lines.append("```")
            md_lines.append(f"Mean:   {loc_dist.get('mean', 0):.0f} LOC/file")
            md_lines.append(f"Median: {loc_dist.get('median', 0):.0f} LOC/file")
            md_lines.append(f"Max:    {loc_dist.get('max', 0):.0f} LOC")
            outliers = loc_dist.get("outlier_count", 0)
            if outliers > 0:
                md_lines.append(f"Outliers: {outliers} large files")
            md_lines.append("```")
            md_lines.append("")

        # Partition summary (if available)
        if partition_data:
            partition_summary = partition_data.get("summary", {})
            if partition_summary:
                md_lines.append(f"### Partition Analysis\n")
                md_lines.append("```")
                md_lines.append(ascii_distribution(partition_summary, width=30))
                md_lines.append("```")
                md_lines.append("")

        markdown_content = "\n".join(md_lines)

        return {
            "markdown": markdown_content,
            "structure": {
                "files": overview.get("total_files", 0),
                "classes": overview.get("total_classes", 0),
                "methods": overview.get("total_methods", 0),
                "loc": overview.get("total_loc", 0),
            },
            "metrics": {
                "complexity_avg": overview.get("avg_complexity", 0),
                "maintainability_index": overview.get("maintainability_index", 0),
                "technical_debt_hours": overview.get("technical_debt_hours", 0),
            },
            "grade": quality.get("overall_grade", "C"),
            "language": language,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        structure = data.get("structure", {})
        grade = data.get("grade", "C")

        return (
            f"Overview section generated. "
            f"{structure.get('files', 0)} files, {structure.get('classes', 0)} classes, "
            f"{structure.get('loc', 0):,} LOC. Grade: {grade}."
        )

    def _load_dependency(self, input: KernelInput, name: str) -> Dict[str, Any]:
        """Load dependency data."""
        path = input.dependencies.get(name)
        if path and path.exists():
            with open(path) as f:
                return json.load(f).get("data", {})
        return {}
