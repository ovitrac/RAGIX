"""
Report Templates for KOAS.

Provides:
- YAML frontmatter generation with variable substitution
- Custom header/footer templates
- Report structure templates
- Adservio-branded defaults

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import yaml


# =============================================================================
# Frontmatter Configuration
# =============================================================================

@dataclass
class FrontMatter:
    """
    YAML frontmatter for Markdown reports.

    Supports variable substitution in header/footer:
    - ${title} - Report title
    - ${author} - Author name
    - ${date} - Generation date
    - ${today} - Today's date (alias)
    - ${version} - Document version
    - ${pageNo} - Page number (for PDF)
    - ${pageCount} - Total pages (for PDF)
    """

    # Metadata
    title: str = "Technical Audit Report"
    author: str = "Adservio"
    creator: str = "Adservio Innovation Lab"
    subject: str = "Code Audit Report"
    keywords: List[str] = field(default_factory=lambda: ["audit", "code", "quality"])

    # Document info
    version: str = "1.0"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    language: str = "en"
    status: str = "DRAFT"

    # Header/footer templates
    header: str = "${title} | ${author} | ${date}"
    footer: str = "Page ${pageNo} / ${pageCount}"

    # Branding
    logo_path: Optional[str] = None
    confidential: bool = True

    # Custom fields
    client: Optional[str] = None
    project: Optional[str] = None
    audit_id: Optional[str] = None

    def to_yaml(self) -> str:
        """Generate YAML frontmatter string."""
        data = {
            "title": self.title,
            "author": self.author,
            "creator": self.creator,
            "subject": self.subject,
            "keywords": self.keywords,
            "version": self.version,
            "date": self.date,
            "language": self.language,
            "status": self.status,
            "header": self.header,
            "footer": self.footer,
        }

        if self.logo_path:
            data["logo"] = self.logo_path
        if self.confidential:
            data["confidential"] = True
        if self.client:
            data["client"] = self.client
        if self.project:
            data["project"] = self.project
        if self.audit_id:
            data["audit_id"] = self.audit_id

        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def to_markdown_header(self) -> str:
        """Generate complete Markdown frontmatter block."""
        return f"---\n{self.to_yaml()}---\n"

    def substitute(self, template: str) -> str:
        """Substitute variables in a template string."""
        replacements = {
            "${title}": self.title,
            "${author}": self.author,
            "${date}": self.date,
            "${today}": self.date,
            "${version}": self.version,
            "${status}": self.status,
            "${client}": self.client or "",
            "${project}": self.project or "",
            # Page variables are typically handled by PDF renderer
            "${pageNo}": "{pageNo}",
            "${pageCount}": "{pageCount}",
        }
        result = template
        for var, value in replacements.items():
            result = result.replace(var, str(value))
        return result

    @classmethod
    def adservio_default(
        cls,
        title: str,
        project: str,
        client: Optional[str] = None,
        language: str = "fr",
    ) -> "FrontMatter":
        """Create Adservio-branded frontmatter."""
        return cls(
            title=title,
            author="Dr Olivier Vitrac | Adservio Innovation Lab",
            creator="Adservio",
            subject=f"Audit Technique - {project}",
            keywords=["audit", "architecture", "qualite", "dette technique", project.lower()],
            version="1.0",
            language=language,
            header="Adservio | ${title} | ${date}",
            footer="${status} | Page ${pageNo} / ${pageCount}",
            logo_path="assets/adservio-logo.svg",
            confidential=True,
            client=client,
            project=project,
        )


# =============================================================================
# Report Templates
# =============================================================================

@dataclass
class ReportTemplate:
    """
    Report structure template with section definitions.

    Defines the order and content of report sections.
    """

    # Template metadata
    name: str = "default"
    description: str = "Standard technical audit report"

    # Section order
    sections: List[str] = field(default_factory=lambda: [
        "frontmatter",
        "title_page",
        "executive_summary",
        "table_of_contents",
        "methodology",
        "overview",
        "quality",
        "architecture",
        "risk",
        "drift",
        "coupling",
        "recommendations",
        "appendix",
    ])

    # Section configurations
    section_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_section_config(self, section: str) -> Dict[str, Any]:
        """Get configuration for a section."""
        return self.section_config.get(section, {})

    @classmethod
    def executive(cls) -> "ReportTemplate":
        """Executive summary template (shorter)."""
        return cls(
            name="executive",
            description="Executive summary for management",
            sections=[
                "frontmatter",
                "title_page",
                "executive_summary",
                "key_findings",
                "recommendations",
            ],
            section_config={
                "executive_summary": {"max_length": 500},
                "recommendations": {"max_items": 5},
            },
        )

    @classmethod
    def full_audit(cls) -> "ReportTemplate":
        """Full technical audit template."""
        return cls(
            name="full_audit",
            description="Comprehensive technical audit report",
            sections=[
                "frontmatter",
                "title_page",
                "executive_summary",
                "table_of_contents",
                "audit_context",
                "methodology",
                "overview",
                "quality",
                "architecture",
                "coupling",
                "risk",
                "drift",
                "dead_code",
                "recommendations",
                "action_plan",
                "appendix",
            ],
        )

    @classmethod
    def mco_assessment(cls) -> "ReportTemplate":
        """MCO (Maintenance) assessment template."""
        return cls(
            name="mco_assessment",
            description="Maintenance in Operational Conditions assessment",
            sections=[
                "frontmatter",
                "title_page",
                "executive_summary",
                "audit_context",
                "mco_overview",
                "risk",
                "timeline",
                "coupling",
                "recommendations",
                "cost_estimation",
            ],
        )


# =============================================================================
# Section Templates (Markdown Fragments)
# =============================================================================

SECTION_TEMPLATES = {
    "title_page": """
<table width="100%">
  <tr>
    <td style="text-align:center;">
      {logo_html}
    </td>
    <td style="text-align:center;">
      <h1>{title}</h1>
      <p><strong>{subtitle}</strong></p>
      <p>{date} | {version}</p>
    </td>
  </tr>
</table>

---
""",

    "executive_summary_header": """
# {section_title}

{context_intro}

## {key_findings_title}

{key_findings}

## {metrics_title}

{metrics_table}

## {recommendations_title}

{recommendations_summary}

---
""",

    "overview_header": """
# {section_title}

{intro_text}

## {metrics_title}

{metrics_table}

## {quality_title}

{quality_grades}

---
""",

    "risk_header": """
# {section_title}

{intro_text}

## {distribution_title}

{distribution_chart}

## {critical_title}

{critical_table}

---
""",

    "recommendations_header": """
# {section_title}

{intro_text}

## {action_plan_title}

{recommendations_table}

---
""",
}


# =============================================================================
# Adservio-specific Templates
# =============================================================================

ADSERVIO_TEMPLATES = {
    "fr": {
        "title_page_subtitle": "Rapport d'Audit Technique",
        "executive_intro": """
Ce document présente les résultats de l'audit technique réalisé sur le projet **{project}**.
L'analyse couvre {file_count} fichiers, {class_count} classes et {loc:,} lignes de code.
""",
        "methodology_intro": """
L'audit a été réalisé à l'aide de **KOAS** (Kernel-Orchestrated Audit System), un système
d'analyse de code souverain développé par Adservio. Les métriques sont calculées selon
les standards industriels (Martin, McCabe, Halstead).
""",
        "overview_intro": """
Cette section présente une vue d'ensemble du code source analysé, incluant les métriques
de taille, de complexité et de qualité.
""",
        "risk_intro": """
L'évaluation des risques identifie les composants présentant un risque élevé en termes
de maintenabilité, de couplage ou de complexité. Ces composants nécessitent une attention
particulière dans le cadre d'une stratégie de MCO.
""",
        "recommendations_intro": """
Les recommandations suivantes sont issues de l'analyse automatisée et classées par
priorité. Elles visent à améliorer la qualité du code et à réduire les risques MCO.
""",
    },
    "en": {
        "title_page_subtitle": "Technical Audit Report",
        "executive_intro": """
This document presents the results of the technical audit performed on the **{project}** project.
The analysis covers {file_count} files, {class_count} classes and {loc:,} lines of code.
""",
        "methodology_intro": """
The audit was conducted using **KOAS** (Kernel-Orchestrated Audit System), a sovereign
code analysis system developed by Adservio. Metrics are calculated according to
industry standards (Martin, McCabe, Halstead).
""",
        "overview_intro": """
This section presents an overview of the analyzed source code, including size,
complexity and quality metrics.
""",
        "risk_intro": """
Risk assessment identifies components with high risk in terms of maintainability,
coupling, or complexity. These components require particular attention in the context
of an MCO strategy.
""",
        "recommendations_intro": """
The following recommendations are derived from automated analysis and ranked by
priority. They aim to improve code quality and reduce MCO risks.
""",
    },
}


def get_template_text(key: str, language: str = "en") -> str:
    """Get language-specific template text."""
    lang_templates = ADSERVIO_TEMPLATES.get(language, ADSERVIO_TEMPLATES["en"])
    return lang_templates.get(key, ADSERVIO_TEMPLATES["en"].get(key, ""))
