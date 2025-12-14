"""
Markdown Renderer for KOAS Reports.

Generates Markdown content from kernel outputs with:
- Tables (Markdown and ASCII art)
- Charts (ASCII visualization)
- Sections with proper formatting
- Language-aware content

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ragix_kernels.audit.report.i18n import I18N, get_translator
from ragix_kernels.audit.report.templates import (
    FrontMatter,
    ReportTemplate,
    get_template_text,
    SECTION_TEMPLATES,
)


# =============================================================================
# ASCII Chart Generators
# =============================================================================

def ascii_bar(value: float, max_value: float, width: int = 30, filled: str = "█", empty: str = "░") -> str:
    """Generate ASCII progress bar."""
    if max_value <= 0:
        return empty * width
    ratio = min(value / max_value, 1.0)
    filled_count = int(ratio * width)
    return filled * filled_count + empty * (width - filled_count)


def ascii_distribution(data: Dict[str, int], width: int = 40) -> str:
    """Generate ASCII distribution chart."""
    if not data:
        return "No data available"

    max_value = max(data.values()) if data else 1
    lines = []

    for label, value in sorted(data.items(), key=lambda x: -x[1]):
        bar = ascii_bar(value, max_value, width=width)
        lines.append(f"{label:20s} {bar} {value:>5}")

    return "\n".join(lines)


def ascii_pie_chart(data: Dict[str, float], width: int = 40) -> str:
    """Generate ASCII representation of distribution."""
    if not data:
        return "No data available"

    total = sum(data.values())
    if total <= 0:
        return "No data available"

    lines = []
    for label, value in sorted(data.items(), key=lambda x: -x[1]):
        pct = (value / total) * 100
        bar_width = int((value / total) * width)
        bar = "█" * bar_width
        lines.append(f"{label:15s} {bar:40s} {pct:5.1f}%")

    return "\n".join(lines)


# =============================================================================
# Table Generators
# =============================================================================

def markdown_table(
    headers: List[str],
    rows: List[List[Any]],
    alignments: Optional[List[str]] = None,
) -> str:
    """
    Generate Markdown table.

    Args:
        headers: Column headers
        rows: List of row data
        alignments: Column alignments ("left", "center", "right")
    """
    if not headers or not rows:
        return ""

    # Default alignment
    if not alignments:
        alignments = ["left"] * len(headers)

    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Build table
    lines = []

    # Header row
    header_row = "|"
    for i, h in enumerate(headers):
        header_row += f" {str(h):<{widths[i]}} |"
    lines.append(header_row)

    # Separator row with alignment
    sep_row = "|"
    for i, align in enumerate(alignments):
        w = widths[i] if i < len(widths) else 3
        if align == "center":
            sep_row += f":{'-' * w}:|"
        elif align == "right":
            sep_row += f"{'-' * w}:|"
        else:
            sep_row += f":{'-' * w}|"
    lines.append(sep_row)

    # Data rows
    for row in rows:
        data_row = "|"
        for i, cell in enumerate(row):
            w = widths[i] if i < len(widths) else len(str(cell))
            data_row += f" {str(cell):<{w}} |"
        lines.append(data_row)

    return "\n".join(lines)


# =============================================================================
# Markdown Renderer
# =============================================================================

@dataclass
class MarkdownRenderer:
    """
    Markdown renderer for KOAS reports.

    Generates complete Markdown documents from kernel outputs.
    """

    i18n: I18N = field(default_factory=lambda: get_translator("en"))
    frontmatter: Optional[FrontMatter] = None

    def render_frontmatter(self, fm: FrontMatter) -> str:
        """Render YAML frontmatter."""
        self.frontmatter = fm
        return fm.to_markdown_header()

    def render_title_page(
        self,
        title: str,
        subtitle: Optional[str] = None,
        logo_html: str = "",
        date: Optional[str] = None,
        version: str = "1.0",
    ) -> str:
        """Render title page."""
        if subtitle is None:
            subtitle = get_template_text("title_page_subtitle", self.i18n.language)
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        return SECTION_TEMPLATES["title_page"].format(
            logo_html=logo_html,
            title=title,
            subtitle=subtitle,
            date=date,
            version=version,
        )

    def render_section(
        self,
        title: str,
        content: str,
        level: int = 1,
    ) -> str:
        """Render a section with title."""
        heading = "#" * level
        return f"\n{heading} {title}\n\n{content}\n"

    def render_metrics_table(
        self,
        metrics: Dict[str, Any],
        caption: Optional[str] = None,
    ) -> str:
        """Render metrics as a table."""
        headers = ["Metric", "Value"]
        rows = []

        for key, value in metrics.items():
            # Format value based on type
            if isinstance(value, float):
                formatted = f"{value:,.2f}"
            elif isinstance(value, int):
                formatted = f"{value:,}"
            else:
                formatted = str(value)

            # Translate metric name if possible
            label = self.i18n.t(f"label.{key}", default=key.replace("_", " ").title())
            rows.append([label, formatted])

        table = markdown_table(headers, rows, ["left", "right"])

        if caption:
            return f"*{caption}*\n\n{table}"
        return table

    def render_quality_grades(
        self,
        grades: Dict[str, str],
        caption: Optional[str] = None,
    ) -> str:
        """Render quality grades table."""
        headers = ["Aspect", "Grade", "Status"]
        rows = []

        for aspect, grade in grades.items():
            # Translate aspect and grade
            aspect_label = aspect.replace("_", " ").title()
            grade_label = self.i18n.grade(grade)

            # Status indicator
            if grade in ("A", "B"):
                status = "✓"
            elif grade == "C":
                status = "⚠"
            else:
                status = "✗"

            rows.append([aspect_label, f"{grade} ({grade_label})", status])

        table = markdown_table(headers, rows, ["left", "center", "center"])

        if caption:
            return f"*{caption}*\n\n{table}"
        return table

    def render_risk_distribution(
        self,
        distribution: Dict[str, int],
        caption: Optional[str] = None,
    ) -> str:
        """Render risk distribution with ASCII chart."""
        lines = []

        if caption:
            lines.append(f"*{caption}*\n")

        lines.append("```")
        lines.append(self.i18n.t("subsection.risk_distribution"))
        lines.append("")

        total = sum(distribution.values())
        for level, count in sorted(distribution.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            bar = ascii_bar(count, max(distribution.values()), width=30)
            level_label = self.i18n.risk_level(level)
            lines.append(f"{level_label:12s} {bar} {count:>4} ({pct:5.1f}%)")

        lines.append("```")

        return "\n".join(lines)

    def render_hotspots_table(
        self,
        hotspots: List[Dict[str, Any]],
        max_items: int = 20,
        caption: Optional[str] = None,
    ) -> str:
        """Render complexity hotspots table."""
        headers = ["Component", "CC", "LOC", "Risk"]
        rows = []

        for hs in hotspots[:max_items]:
            name = hs.get("name", "Unknown")
            cc = hs.get("complexity", hs.get("cc", 0))
            loc = hs.get("loc", 0)

            # Risk indicator
            if cc > 30:
                risk = "🔴"
            elif cc > 15:
                risk = "🟠"
            elif cc > 10:
                risk = "🟡"
            else:
                risk = "🟢"

            # Truncate name if too long
            if len(name) > 50:
                name = "..." + name[-47:]

            rows.append([name, cc, loc, risk])

        table = markdown_table(headers, rows, ["left", "right", "right", "center"])

        if caption:
            return f"*{caption}*\n\n{table}"
        return table

    def render_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        max_items: int = 10,
        caption: Optional[str] = None,
    ) -> str:
        """Render recommendations table."""
        headers = ["Priority", "Category", "Recommendation", "Effort"]
        rows = []

        for i, rec in enumerate(recommendations[:max_items], 1):
            priority = self.i18n.priority(rec.get("priority", "medium_term"))
            category = rec.get("category", "general").title()
            title = rec.get("title", "")
            effort = rec.get("effort", "")

            # Priority indicator
            prio_raw = rec.get("priority", "medium_term").lower()
            if "immediate" in prio_raw:
                prio_indicator = "🔴"
            elif "short" in prio_raw:
                prio_indicator = "🟠"
            elif "medium" in prio_raw:
                prio_indicator = "🟡"
            else:
                prio_indicator = "🟢"

            rows.append([f"{prio_indicator} {priority}", category, title, effort])

        table = markdown_table(headers, rows, ["left", "left", "left", "center"])

        if caption:
            return f"*{caption}*\n\n{table}"
        return table

    def render_coupling_zones(
        self,
        zones: Dict[str, int],
        caption: Optional[str] = None,
    ) -> str:
        """Render coupling zone distribution."""
        lines = []

        if caption:
            lines.append(f"*{caption}*\n")

        lines.append("```")
        lines.append(self.i18n.t("subsection.coupling_zones"))
        lines.append("")

        total = sum(zones.values())
        for zone_name, count in zones.items():
            pct = (count / total * 100) if total > 0 else 0
            bar = ascii_bar(count, max(zones.values()) if zones else 1, width=30)
            zone_label = self.i18n.zone(zone_name)
            lines.append(f"{zone_label:20s} {bar} {count:>4} ({pct:5.1f}%)")

        lines.append("```")

        return "\n".join(lines)

    def render_key_findings(
        self,
        findings: List[str],
    ) -> str:
        """Render key findings as bullet list."""
        lines = [f"- {finding}" for finding in findings]
        return "\n".join(lines)

    def render_interpretation(
        self,
        key: str,
        **kwargs,
    ) -> str:
        """Render translated interpretation text."""
        return f"> {self.i18n.t(key, **kwargs)}"

    def render_audit_context(
        self,
        context: Dict[str, Any],
    ) -> str:
        """Render audit context section."""
        lines = []

        # Objectives
        if "objectives" in context:
            lines.append(f"## {self.i18n.t('context.audit_objectives')}\n")
            for obj in context["objectives"]:
                lines.append(f"- {obj}")
            lines.append("")

        # Questions
        if "questions" in context:
            lines.append(f"## {self.i18n.t('context.audit_questions')}\n")
            for q in context["questions"]:
                lines.append(f"1. {q}")
            lines.append("")

        # Scope
        if "scope" in context:
            lines.append(f"## {self.i18n.t('context.audit_scope')}\n")
            scope = context["scope"]
            lines.append(f"- **Project:** {scope.get('project', 'N/A')}")
            lines.append(f"- **Path:** `{scope.get('path', 'N/A')}`")
            lines.append(f"- **Language:** {scope.get('language', 'N/A')}")
            lines.append(f"- **Files:** {scope.get('files', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    def render_methodology(self) -> str:
        """Render methodology section."""
        intro = get_template_text("methodology_intro", self.i18n.language)

        return f"""
{intro}

### {self.i18n.t("label.complexity")}
- **Cyclomatic Complexity (CC)**: McCabe's metric for control flow complexity
- **Maintainability Index (MI)**: Composite metric (Halstead, CC, LOC)

### Coupling Metrics (Martin)
- **Ca**: Afferent coupling (incoming dependencies)
- **Ce**: Efferent coupling (outgoing dependencies)
- **I**: Instability = Ce / (Ca + Ce)
- **A**: Abstractness = abstract classes / total classes
- **D**: Distance from Main Sequence = |A + I - 1|

### Risk Assessment
- Combines volatility, impact, complexity, and maturity factors
- Outputs: LOW, MEDIUM, HIGH, CRITICAL
"""
