"""
Kernel: Section — Maven Dependencies & Supply Chain
Stage: 3 (Report Generation)

Generates the Maven dependencies section for audit reports:
- Dependency inventory table
- Module graph summary (with optional SVG reference)
- CVE findings table (severity, affected modules, remediation)
- Obsolescence assessment

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-09
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class SectionMavenKernel(Kernel):
    """
    Generate Maven Dependencies section for audit reports.

    Combines maven_deps, maven_graph, and maven_cve outputs into a
    structured Markdown section with dependency inventory, graph
    analysis summary, and vulnerability findings.
    """

    name = "section_maven"
    version = "1.0.0"
    category = "audit"
    stage = 3
    description = "Generate Maven Dependencies report section"
    requires = ["maven_deps", "maven_graph", "maven_cve"]
    provides = ["section_maven"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        language = input.config.get("language", "en")

        # Load dependencies
        deps_data = self._load_dep(input, "maven_deps")
        graph_data = self._load_dep(input, "maven_graph")
        cve_data = self._load_dep(input, "maven_cve")

        md_lines = []

        # Title
        title = "Dépendances Maven et Chaîne d'Approvisionnement" if language == "fr" else "Maven Dependencies & Supply Chain"
        md_lines.append(f"# {title}\n")

        # 1. Module inventory
        md_lines.append(self._render_module_inventory(deps_data, language))

        # 2. Graph analysis
        md_lines.append(self._render_graph_analysis(graph_data, language))

        # 3. CVE findings
        md_lines.append(self._render_cve_findings(cve_data, language))

        # 4. Dependency inventory
        md_lines.append(self._render_dependency_table(deps_data, language))

        markdown = "\n".join(md_lines)
        return {
            "markdown": markdown,
            "title": title,
            "order": 7,
            "language": language,
            "counts": {
                "modules": len(deps_data.get("modules", [])) if deps_data else 0,
                "dependencies": deps_data.get("statistics", {}).get("distinct_dependencies", 0) if deps_data else 0,
                "vulnerabilities": cve_data.get("statistics", {}).get("vulnerabilities_found", 0) if cve_data else 0,
            },
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        counts = data.get("counts", {})
        return (
            f"Maven section: {counts.get('modules', 0)} modules, "
            f"{counts.get('dependencies', 0)} dependencies, "
            f"{counts.get('vulnerabilities', 0)} CVE findings."
        )

    def _load_dep(self, input: KernelInput, name: str) -> Optional[Dict[str, Any]]:
        """Load a dependency's data from JSON."""
        path = input.dependencies.get(name)
        if not path:
            # Fallback: check stage directory
            for stage in (1, 2):
                fallback = input.workspace / f"stage{stage}" / f"{name}.json"
                if fallback.exists():
                    path = fallback
                    break
        if path and path.exists():
            with open(path) as f:
                return json.load(f).get("data", {})
        return None

    def _render_module_inventory(self, deps_data: Optional[Dict], language: str) -> str:
        """Render module inventory section."""
        if not deps_data:
            return ""
        modules = deps_data.get("modules", [])
        if not modules:
            return ""

        subtitle = "Inventaire des modules Maven" if language == "fr" else "Maven Module Inventory"
        lines = [f"\n## {subtitle}\n"]

        # Summary
        stats = deps_data.get("statistics", {})
        if language == "fr":
            lines.append(f"**{stats.get('modules_found', 0)} modules** analysés à partir de "
                         f"**{stats.get('pom_files_parsed', 0)} fichiers pom.xml**.\n")
        else:
            lines.append(f"**{stats.get('modules_found', 0)} modules** analyzed from "
                         f"**{stats.get('pom_files_parsed', 0)} pom.xml files**.\n")

        # Table
        hdr = "| Module | GroupId | Version | Packaging | Dependencies |" if language != "fr" \
            else "| Module | GroupId | Version | Packaging | Dépendances |"
        lines.append(hdr)
        lines.append("|--------|---------|---------|-----------|:------------:|")
        for m in sorted(modules, key=lambda x: x.get("artifactId", "")):
            dep_count = len(m.get("dependencies", []))
            lines.append(
                f"| {m.get('artifactId', '')} | {m.get('groupId', '')} | "
                f"{m.get('version', '')} | {m.get('packaging', 'jar')} | {dep_count} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _render_graph_analysis(self, graph_data: Optional[Dict], language: str) -> str:
        """Render graph analysis section."""
        if not graph_data:
            return ""

        subtitle = "Analyse du graphe de dépendances" if language == "fr" else "Dependency Graph Analysis"
        lines = [f"\n## {subtitle}\n"]

        stats = graph_data.get("statistics", {})
        hub = stats.get("hub_module", "N/A")
        hub_c = stats.get("hub_centrality", 0)

        if language == "fr":
            lines.append(f"- **Modules :** {stats.get('total_modules', 0)}")
            lines.append(f"- **Arêtes :** {stats.get('total_edges', 0)}")
            lines.append(f"- **Module hub :** `{hub}` (centralité = {hub_c})")
            lines.append(f"- **Cycles détectés :** {stats.get('cycle_count', 0)}")
            lines.append(f"- **Chemin critique :** {stats.get('critical_path_length', 0)} modules")
            lines.append(f"- **Modules racine :** {stats.get('root_modules', 0)}")
            lines.append(f"- **Modules feuille :** {stats.get('leaf_modules', 0)}")
        else:
            lines.append(f"- **Modules:** {stats.get('total_modules', 0)}")
            lines.append(f"- **Edges:** {stats.get('total_edges', 0)}")
            lines.append(f"- **Hub module:** `{hub}` (centrality={hub_c})")
            lines.append(f"- **Cycles detected:** {stats.get('cycle_count', 0)}")
            lines.append(f"- **Critical path:** {stats.get('critical_path_length', 0)} modules")
            lines.append(f"- **Root modules:** {stats.get('root_modules', 0)}")
            lines.append(f"- **Leaf modules:** {stats.get('leaf_modules', 0)}")

        # Reference visualization if it exists
        viz = stats.get("visualization")
        if viz:
            lines.append(f"\n![Maven Dependency Graph]({viz})\n")

        # Critical path
        cp = graph_data.get("critical_path", [])
        if cp:
            label = "Chemin critique" if language == "fr" else "Critical path"
            lines.append(f"\n**{label} :** `{'` → `'.join(cp)}`\n")

        # Top centrality table
        centrality = graph_data.get("centrality", {})
        if centrality:
            top_5 = sorted(centrality.items(), key=lambda x: -x[1].get("betweenness", 0))[:5]
            label = "Modules les plus centraux" if language == "fr" else "Top central modules"
            lines.append(f"\n### {label}\n")
            lines.append("| Module | Betweenness | In-degree | Out-degree | Transitive deps |")
            lines.append("|--------|:-----------:|:---------:|:----------:|:---------------:|")
            for name, detail in top_5:
                lines.append(
                    f"| {name} | {detail.get('betweenness', 0)} | "
                    f"{detail.get('in_degree', 0)} | {detail.get('out_degree', 0)} | "
                    f"{detail.get('transitive_deps', 0)} |"
                )
        lines.append("")
        return "\n".join(lines)

    def _render_cve_findings(self, cve_data: Optional[Dict], language: str) -> str:
        """Render CVE findings section."""
        if not cve_data:
            return ""

        subtitle = "Vulnérabilités identifiées (CVE)" if language == "fr" else "Identified Vulnerabilities (CVE)"
        lines = [f"\n## {subtitle}\n"]

        stats = cve_data.get("statistics", {})
        vulns = cve_data.get("vulnerabilities", [])

        if not vulns:
            msg = "Aucune vulnérabilité identifiée." if language == "fr" else "No vulnerabilities identified."
            lines.append(f"*{msg}*\n")
            return "\n".join(lines)

        by_sev = stats.get("by_severity", {})
        if language == "fr":
            lines.append(f"**{stats.get('vulnerabilities_found', 0)} vulnérabilités** identifiées "
                         f"sur {stats.get('dependencies_scanned', 0)} dépendances analysées.\n")
        else:
            lines.append(f"**{stats.get('vulnerabilities_found', 0)} vulnerabilities** found "
                         f"across {stats.get('dependencies_scanned', 0)} dependencies scanned.\n")

        # Severity summary
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            count = by_sev.get(sev, 0)
            if count > 0:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev, "⚪")
                lines.append(f"- {emoji} **{sev}**: {count}")
        lines.append("")

        # Detailed table
        hdr = "| CVE | Severity | CVSS | Dependency | Fixed in | Modules |" if language != "fr" \
            else "| CVE | Sévérité | CVSS | Dépendance | Corrigé dans | Modules |"
        lines.append(hdr)
        lines.append("|-----|:--------:|:----:|------------|----------|---------|")
        for v in vulns[:30]:  # Limit to 30 rows
            modules = ", ".join(v.get("modules_affected", [])[:5])
            lines.append(
                f"| {v.get('cve_id', '')} | {v.get('severity', '')} | "
                f"{v.get('cvss', '')} | {v.get('dependency', '')} | "
                f"{v.get('fixed_in', '')} | {modules} |"
            )
        if len(vulns) > 30:
            lines.append(f"| ... | | | | | ({len(vulns) - 30} more) |")

        lines.append("")
        return "\n".join(lines)

    def _render_dependency_table(self, deps_data: Optional[Dict], language: str) -> str:
        """Render full dependency table."""
        if not deps_data:
            return ""

        all_deps = deps_data.get("all_dependencies", [])
        if not all_deps:
            return ""

        subtitle = "Inventaire des dépendances" if language == "fr" else "Dependency Inventory"
        lines = [f"\n## {subtitle}\n"]

        lines.append("| GroupId | ArtifactId | Version | Scope | Used by |")
        lines.append("|---------|------------|---------|-------|---------|")
        for dep in sorted(all_deps, key=lambda x: f"{x.get('groupId', '')}:{x.get('artifactId', '')}"):
            used_by = ", ".join(dep.get("used_by", [])[:5])
            lines.append(
                f"| {dep.get('groupId', '')} | {dep.get('artifactId', '')} | "
                f"{dep.get('version', '')} | {dep.get('scope', '')} | {used_by} |"
            )
        lines.append("")
        return "\n".join(lines)
