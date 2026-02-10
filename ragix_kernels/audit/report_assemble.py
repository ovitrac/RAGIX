"""
Kernel: Report Assembly
Stage: 3 (Report)

Assembles all section kernels into a complete audit report:
- YAML frontmatter with custom header/footer
- All section content in order
- Table of contents generation
- Final Markdown output

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import json

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.audit.report.i18n import I18N, get_translator
from ragix_kernels.audit.report.templates import (
    FrontMatter,
    ReportTemplate,
    get_template_text,
    SECTION_TEMPLATES,
)
from ragix_kernels.audit.report.renderer import MarkdownRenderer

logger = logging.getLogger(__name__)


class ReportAssembleKernel(Kernel):
    """
    Assemble complete audit report from section kernels.

    Configuration options:
        language: Report language ("en" or "fr")
        template: Report template ("default", "executive", "full_audit", "mco_assessment")
        frontmatter: Custom frontmatter options (title, author, client, etc.)
        include_toc: Include table of contents (default: true)
        output_filename: Output filename (default: "audit_report.md")

    Dependencies:
        section_executive: Executive summary section (required)
        section_overview: Codebase overview section (required)
        section_recommendations: Recommendations section (required)
        section_risk: Risk assessment section (optional)
        section_coupling: Coupling analysis section (optional)
        section_hotspots: Hotspots section (optional)

    Output:
        report_path: Path to generated report file
        report_content: Complete Markdown content
        metadata: Report metadata (title, date, sections)
    """

    name = "report_assemble"
    version = "1.0.0"
    category = "audit"
    stage = 3
    description = "Assemble complete audit report"

    requires = ["section_executive", "section_overview", "section_recommendations"]
    provides = ["audit_report_md", "report_metadata"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Assemble complete audit report."""

        # Get configuration
        language = input.config.get("language", "en")
        template_name = input.config.get("template", "default")
        fm_config = input.config.get("frontmatter", {})
        include_toc = input.config.get("include_toc", True)
        output_filename = input.config.get("output_filename", "audit_report.md")
        project_config = input.config.get("project", {})
        audit_context = input.config.get("audit_context", {})

        # Initialize i18n and renderer
        i18n = get_translator(language)
        renderer = MarkdownRenderer(i18n=i18n)

        # Get report template
        template = self._get_template(template_name)

        logger.info(f"[report_assemble] Assembling report in {language} using '{template_name}' template")

        # Build frontmatter
        frontmatter = self._build_frontmatter(
            fm_config, project_config, language, i18n
        )

        # Load section content
        sections_content = self._load_sections(input, template.sections)

        # Assemble report
        report_lines = []

        # 1. Frontmatter
        report_lines.append(frontmatter.to_markdown_header())

        # 2. Title page (optional)
        if "title_page" in template.sections:
            logo_html = self._get_logo_html(fm_config.get("logo_path"))
            report_lines.append(renderer.render_title_page(
                title=frontmatter.title,
                subtitle=get_template_text("title_page_subtitle", language),
                logo_html=logo_html,
                date=frontmatter.date,
                version=frontmatter.version,
            ))

        # 3. Table of Contents (if enabled)
        if include_toc and "table_of_contents" in template.sections:
            report_lines.append(self._generate_toc(sections_content, i18n))

        # 4. Audit Context (if provided)
        if audit_context and "audit_context" in template.sections:
            report_lines.append(renderer.render_audit_context(audit_context))

        # 5. Methodology (if in template)
        if "methodology" in template.sections:
            report_lines.append(f"\n# {i18n.t('section.methodology')}\n")
            report_lines.append(renderer.render_methodology())

        # 6. Section content in order
        section_order = [
            "executive_summary",
            "overview",
            "quality",
            "architecture",
            "coupling",
            "risk",
            "drift",
            "hotspots",
            "dead_code",
            "maven",
            "spring",
            "recommendations",
        ]

        for section_key in section_order:
            if section_key in template.sections and section_key in sections_content:
                content = sections_content[section_key]
                if content:
                    report_lines.append(content)
                    report_lines.append("\n---\n")

        # 7. Appendix (if in template)
        if "appendix" in template.sections:
            appendix_content = self._generate_appendix(input, i18n)
            if appendix_content:
                report_lines.append(appendix_content)

        # 8. Footer / Generation info
        report_lines.append(self._generate_footer(i18n, frontmatter))

        # Combine all content
        report_content = "\n".join(report_lines)

        # Write to file (use workspace/stage3 as output directory)
        output_dir = input.workspace / f"stage{self.stage}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"[report_assemble] Report written to {output_path}")

        # Build metadata
        metadata = {
            "title": frontmatter.title,
            "author": frontmatter.author,
            "date": frontmatter.date,
            "language": language,
            "template": template_name,
            "sections": list(sections_content.keys()),
            "section_count": len(sections_content),
            "word_count": len(report_content.split()),
            "line_count": len(report_content.splitlines()),
            "output_path": str(output_path),
        }

        return {
            "report_path": str(output_path),
            "report_content": report_content,
            "metadata": metadata,
            "frontmatter": frontmatter.to_yaml(),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        metadata = data.get("metadata", {})
        return (
            f"Report assembled: {metadata.get('title', 'Audit Report')}. "
            f"Sections: {metadata.get('section_count', 0)}. "
            f"Words: {metadata.get('word_count', 0):,}. "
            f"Output: {metadata.get('output_path', 'N/A')}."
        )

    def _get_template(self, name: str) -> ReportTemplate:
        """Get report template by name."""
        templates = {
            "default": ReportTemplate(),
            "executive": ReportTemplate.executive(),
            "full_audit": ReportTemplate.full_audit(),
            "mco_assessment": ReportTemplate.mco_assessment(),
        }
        return templates.get(name, ReportTemplate())

    def _build_frontmatter(
        self,
        fm_config: Dict[str, Any],
        project_config: Dict[str, Any],
        language: str,
        i18n: I18N,
    ) -> FrontMatter:
        """Build frontmatter from configuration."""

        # Use Adservio defaults if not specified
        project_name = project_config.get("name", fm_config.get("project", "Unknown"))
        client = fm_config.get("client")

        if fm_config.get("adservio_branded", True):
            fm = FrontMatter.adservio_default(
                title=fm_config.get("title", f"{i18n.t('title.audit_report')} - {project_name}"),
                project=project_name,
                client=client,
                language=language,
            )
        else:
            fm = FrontMatter(
                title=fm_config.get("title", f"Technical Audit - {project_name}"),
                author=fm_config.get("author", "Adservio"),
                project=project_name,
                client=client,
                language=language,
            )

        # Override with explicit config
        if "header" in fm_config:
            fm.header = fm_config["header"]
        if "footer" in fm_config:
            fm.footer = fm_config["footer"]
        if "version" in fm_config:
            fm.version = fm_config["version"]
        if "status" in fm_config:
            fm.status = fm_config["status"]
        if "confidential" in fm_config:
            fm.confidential = fm_config["confidential"]
        if "logo_path" in fm_config:
            fm.logo_path = fm_config["logo_path"]
        if "audit_id" in fm_config:
            fm.audit_id = fm_config["audit_id"]

        return fm

    def _load_sections(
        self,
        input: KernelInput,
        template_sections: List[str],
    ) -> Dict[str, str]:
        """Load section content from dependencies."""
        sections_content = {}

        # Map template section names to kernel outputs
        section_mappings = {
            "executive_summary": "section_executive",
            "overview": "section_overview",
            "recommendations": "section_recommendations",
            "risk": "section_risk",
            "coupling": "section_coupling",
            "hotspots": "section_hotspots",
            "dead_code": "section_dead_code",
            "quality": "section_quality",
            "architecture": "section_architecture",
            "drift": "section_drift",
            "maven": "section_maven",
            "spring": "section_spring",
        }

        # Stage 3 output directory for fallback lookup
        stage3_dir = input.workspace / "stage3"

        for section_name, kernel_name in section_mappings.items():
            if section_name in template_sections:
                # First check dependencies
                path = input.dependencies.get(kernel_name)

                # Fallback: check stage3 folder directly for optional sections
                if (not path or not path.exists()) and stage3_dir.exists():
                    fallback_path = stage3_dir / f"{kernel_name}.json"
                    if fallback_path.exists():
                        path = fallback_path

                if path and path.exists():
                    try:
                        with open(path) as f:
                            data = json.load(f).get("data", {})
                            content = data.get("markdown", "")
                            if content:
                                sections_content[section_name] = content
                                logger.debug(f"Loaded section '{section_name}' from {kernel_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load section {section_name}: {e}")

        return sections_content

    def _generate_toc(
        self,
        sections_content: Dict[str, str],
        i18n: I18N,
    ) -> str:
        """Generate table of contents."""
        lines = []
        lines.append(f"\n# {i18n.t('section.toc')}\n")

        toc_items = [
            ("executive_summary", "section.executive"),
            ("overview", "section.overview"),
            ("quality", "section.quality"),
            ("architecture", "section.architecture"),
            ("coupling", "section.coupling"),
            ("risk", "section.risk"),
            ("drift", "section.drift"),
            ("hotspots", "section.hotspots"),
            ("dead_code", "section.dead_code"),
            ("maven", "section.maven"),
            ("spring", "section.spring"),
            ("recommendations", "section.recommendations"),
        ]

        item_num = 1
        for section_key, title_key in toc_items:
            if section_key in sections_content:
                title = i18n.t(title_key)
                # Create anchor link
                anchor = title.lower().replace(" ", "-").replace("'", "")
                lines.append(f"{item_num}. [{title}](#{anchor})")
                item_num += 1

        lines.append("")
        lines.append("---\n")

        return "\n".join(lines)

    def _generate_appendix(
        self,
        input: KernelInput,
        i18n: I18N,
    ) -> str:
        """Generate appendix with methodology and metrics details."""
        lines = []
        lines.append(f"\n# {i18n.t('section.appendix')}\n")

        # Kernel execution info
        lines.append(f"## {i18n.t('subsection.analysis_metadata')}\n")
        lines.append(f"- **Generated:** {datetime.now().isoformat()}")
        lines.append(f"- **KOAS Version:** 1.0.0")
        lines.append(f"- **Language:** {i18n.language.upper()}")
        lines.append("")

        # Metrics reference
        lines.append(f"## {i18n.t('subsection.metrics_reference')}\n")
        lines.append("""
### Complexity Metrics
- **CC (Cyclomatic Complexity):** McCabe's metric measuring control flow complexity
- **MI (Maintainability Index):** Composite metric from Halstead, CC, and LOC

### Coupling Metrics (Martin)
- **Ca (Afferent Coupling):** Number of packages that depend on this package
- **Ce (Efferent Coupling):** Number of packages this package depends on
- **I (Instability):** Ce / (Ca + Ce), ranges from 0 (stable) to 1 (unstable)
- **A (Abstractness):** Abstract classes / Total classes
- **D (Distance):** |A + I - 1|, distance from the main sequence

### Risk Levels
- **Critical:** Immediate action required (score ≥ 0.8)
- **High:** Near-term attention needed (score ≥ 0.6)
- **Medium:** Monitor and plan (score ≥ 0.4)
- **Low:** Acceptable risk (score < 0.4)
""")

        return "\n".join(lines)

    def _generate_footer(
        self,
        i18n: I18N,
        frontmatter: FrontMatter,
    ) -> str:
        """Generate report footer."""
        lines = []
        lines.append("\n---\n")
        lines.append(f"*{i18n.t('footer.generated_by')} KOAS (Kernel-Orchestrated Audit System)*")
        lines.append(f"*{i18n.t('footer.date')}: {frontmatter.date}*")

        if frontmatter.confidential:
            lines.append(f"\n**{i18n.t('footer.confidential')}**")

        lines.append("")
        return "\n".join(lines)

    def _get_logo_html(self, logo_path: Optional[str]) -> str:
        """Generate logo HTML if path provided."""
        if not logo_path:
            return ""

        # Check if it's an SVG or image
        if logo_path.endswith(".svg"):
            return f'<img src="{logo_path}" alt="Logo" width="150" />'
        else:
            return f'<img src="{logo_path}" alt="Logo" width="150" />'
