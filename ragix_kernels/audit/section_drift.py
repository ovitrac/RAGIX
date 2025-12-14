"""
Kernel: Section - Drift Analysis
Stage: 3 (Report Generation)

Generates the drift analysis section for audit reports.
Shows spec-code misalignment and documentation gaps.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.audit.report.i18n import I18N, get_translator

logger = logging.getLogger(__name__)


class SectionDriftKernel(Kernel):
    """
    Generate Drift Analysis section for audit reports.

    Shows:
    - Drift type distribution (synchronized, undocumented, spec_ahead, stable)
    - High-severity alerts
    - Documentation gap scores
    - Per-component drift details

    Dependencies:
        drift: Drift analysis results from Stage 2

    Output:
        markdown: Rendered section markdown
        title: Section title
        order: Section ordering (e.g., 4)
    """

    name = "section_drift"
    version = "1.0.0"
    category = "audit"
    stage = 3
    description = "Generate Drift Analysis section"

    requires = ["drift"]
    provides = ["section_drift"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Generate drift section content."""

        # Get language config
        language = input.config.get("language", "en")
        i18n = get_translator(language)

        # Load drift data
        drift_path = input.dependencies.get("drift")
        if not drift_path or not drift_path.exists():
            return self._empty_section(i18n)

        with open(drift_path) as f:
            drift_data = json.load(f)

        data = drift_data.get("data", {})
        components = data.get("components", {})
        by_type = data.get("by_type", {})
        alerts = data.get("alerts", [])
        total = data.get("total_components", 0)

        if total == 0:
            return self._empty_section(i18n)

        # Build section using plain markdown
        lines = []
        title = "Analyse de la Dérive" if language == "fr" else "Drift Analysis"

        lines.append(f"# {title}")
        lines.append("")

        # Introduction
        if language == "fr":
            intro = f"Cette section analyse l'alignement entre le code et la documentation " \
                   f"pour {total} composants identifiés."
        else:
            intro = f"This section analyzes the alignment between code and documentation " \
                   f"for {total} identified components."
        lines.append(intro)
        lines.append("")

        # Summary table
        lines.append(f"## {'Résumé' if language == 'fr' else 'Summary'}")
        lines.append("")

        type_descriptions = {
            "synchronized": ("Synchronized" if language == "en" else "Synchronisé",
                           "Code and docs evolving together" if language == "en" else "Code et docs évoluent ensemble"),
            "undocumented": ("Undocumented" if language == "en" else "Non documenté",
                           "Code changed but docs frozen" if language == "en" else "Code modifié mais docs figés"),
            "spec_ahead": ("Spec Ahead" if language == "en" else "Spec en avance",
                          "Docs evolved but code unchanged" if language == "en" else "Docs évolués mais code inchangé"),
            "stable": ("Stable" if language == "en" else "Stable",
                      "Both frozen (acceptable)" if language == "en" else "Les deux figés (acceptable)"),
        }

        # Build table
        lines.append(f"| Type | {'Nombre' if language == 'fr' else 'Count'} | Description |")
        lines.append("|:-----|-------:|:------------|")
        for dtype, count in by_type.items():
            name, desc = type_descriptions.get(dtype, (dtype, ""))
            lines.append(f"| {name} | {count} | {desc} |")
        lines.append("")

        # Alerts
        if alerts:
            lines.append(f"## {'Alertes Critiques' if language == 'fr' else 'Critical Alerts'}")
            lines.append("")

            for alert in alerts[:10]:
                severity = alert.get("severity", "warning").upper()
                comp = alert.get("component", "Unknown")
                msg = alert.get("message", "")
                drift_days = alert.get("drift_days", 0)

                if language == "fr":
                    lines.append(f"- **[{severity}]** {comp}: {drift_days} jours de dérive")
                else:
                    lines.append(f"- **[{severity}]** {comp}: {drift_days} days of drift")

                if msg:
                    lines.append(f"  _{msg}_")

            lines.append("")

        # Components table (top 15)
        if components:
            lines.append(f"## {'Composants par Dérive' if language == 'fr' else 'Components by Drift'}")
            lines.append("")

            # Sort by gap score
            sorted_comps = sorted(
                components.items(),
                key=lambda x: x[1].get("gap_score", 0),
                reverse=True
            )[:15]

            lines.append(f"| {'Composant' if language == 'fr' else 'Component'} | Type | {'Dérive (j)' if language == 'fr' else 'Drift (d)'} | Score |")
            lines.append("|:----------|:-----|----------:|------:|")

            for comp_id, comp_data in sorted_comps:
                lines.append(f"| {comp_id} | {comp_data.get('drift_type', 'unknown')} | {comp_data.get('drift_days', 0)} | {comp_data.get('gap_score', 0):.2f} |")

            lines.append("")

        return {
            "markdown": "\n".join(lines),
            "title": title,
            "order": 4,  # After risk section
        }

    def _empty_section(self, i18n: I18N) -> Dict[str, Any]:
        """Return empty section when no drift data."""
        title = "Analyse de la Dérive" if i18n.language == "fr" else "Drift Analysis"
        lines = [f"# {title}", ""]

        if i18n.language == "fr":
            lines.append("Aucune donnée de dérive disponible.")
        else:
            lines.append("No drift data available.")

        return {
            "markdown": "\n".join(lines),
            "title": title,
            "order": 4,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        title = data.get("title", "Drift Analysis")
        content_len = len(data.get("markdown", ""))
        return f"Generated {title} section ({content_len} chars)"
