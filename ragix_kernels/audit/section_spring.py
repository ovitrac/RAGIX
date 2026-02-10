"""
Kernel: Section — Spring Architecture
Stage: 3 (Report Generation)

Generates the Spring Architecture section for audit reports:
- Spring bean catalog (services, controllers, repositories, configurations)
- Dependency injection wiring graph
- Implicit entry point inventory (JMS, Scheduled, EventListener, REST)
- Corrected dead code analysis (with Spring-aware reachability)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-09
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class SectionSpringKernel(Kernel):
    """
    Generate Spring Architecture section for audit reports.

    Combines spring_wiring and dead_code outputs to produce a
    structured Markdown section showing bean inventory, wiring
    topology, entry points, and corrected dead code analysis.
    """

    name = "section_spring"
    version = "1.0.0"
    category = "audit"
    stage = 3
    description = "Generate Spring Architecture report section"
    requires = ["spring_wiring", "dead_code"]
    provides = ["section_spring"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        language = input.config.get("language", "en")

        # Load dependencies
        wiring_data = self._load_dep(input, "spring_wiring")
        dead_code_data = self._load_dep(input, "dead_code")

        md_lines = []

        # Title
        title = "Architecture Spring et Points d'Entrée" if language == "fr" else "Spring Architecture & Entry Points"
        md_lines.append(f"# {title}\n")

        # 1. Bean catalog
        md_lines.append(self._render_bean_catalog(wiring_data, language))

        # 2. Entry points
        md_lines.append(self._render_entry_points(wiring_data, language))

        # 3. Wiring graph
        md_lines.append(self._render_wiring(wiring_data, language))

        # 4. Corrected dead code
        md_lines.append(self._render_corrected_dead_code(wiring_data, dead_code_data, language))

        # 5. Reachability improvement
        md_lines.append(self._render_reachability_delta(wiring_data, language))

        markdown = "\n".join(md_lines)

        beans = wiring_data.get("beans", []) if wiring_data else []
        ep_summary = wiring_data.get("entry_point_summary", {}) if wiring_data else {}

        return {
            "markdown": markdown,
            "title": title,
            "order": 8,
            "language": language,
            "counts": {
                "beans": len(beans),
                "entry_points": ep_summary.get("total", 0),
                "wiring": wiring_data.get("statistics", {}).get("total_wiring", 0) if wiring_data else 0,
            },
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        counts = data.get("counts", {})
        return (
            f"Spring section: {counts.get('beans', 0)} beans, "
            f"{counts.get('entry_points', 0)} entry points, "
            f"{counts.get('wiring', 0)} injection points."
        )

    def _load_dep(self, input: KernelInput, name: str) -> Optional[Dict[str, Any]]:
        """Load a dependency's data from JSON."""
        path = input.dependencies.get(name)
        if not path:
            for stage in (1, 2):
                fallback = input.workspace / f"stage{stage}" / f"{name}.json"
                if fallback.exists():
                    path = fallback
                    break
        if path and path.exists():
            with open(path) as f:
                return json.load(f).get("data", {})
        return None

    def _render_bean_catalog(self, wiring_data: Optional[Dict], language: str) -> str:
        """Render Spring bean inventory."""
        if not wiring_data:
            return ""
        beans = wiring_data.get("beans", [])
        if not beans:
            return ""

        subtitle = "Catalogue des Beans Spring" if language == "fr" else "Spring Bean Catalog"
        lines = [f"\n## {subtitle}\n"]

        stats = wiring_data.get("statistics", {})
        by_type = stats.get("by_type", {})

        if language == "fr":
            lines.append(f"**{stats.get('total_beans', 0)} beans** identifiés "
                         f"dans {stats.get('classes_with_annotations', 0)} classes annotées.\n")
        else:
            lines.append(f"**{stats.get('total_beans', 0)} beans** identified "
                         f"across {stats.get('classes_with_annotations', 0)} annotated classes.\n")

        # Summary by type
        type_labels_fr = {
            "service": "Services", "controller": "Contrôleurs",
            "repository": "Repositories", "component": "Composants",
            "configuration": "Configurations", "factory_bean": "Factory beans",
        }
        type_labels_en = {
            "service": "Services", "controller": "Controllers",
            "repository": "Repositories", "component": "Components",
            "configuration": "Configurations", "factory_bean": "Factory beans",
        }
        labels = type_labels_fr if language == "fr" else type_labels_en
        for btype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            label = labels.get(btype, btype)
            lines.append(f"- **{label}:** {count}")
        lines.append("")

        # Detailed table
        hdr = "| Bean | Type | Annotations | Entry Point | File |" if language != "fr" \
            else "| Bean | Type | Annotations | Point d'entrée | Fichier |"
        lines.append(hdr)
        lines.append("|------|------|-------------|:-----------:|------|")
        for bean in sorted(beans, key=lambda b: b.get("class", "")):
            ann_str = ", ".join(f"@{a}" for a in bean.get("annotations", [])[:4])
            ep = "Yes" if bean.get("entry_point") else "" if language != "fr" \
                else ("Oui" if bean.get("entry_point") else "")
            file_short = Path(bean.get("file", "")).name
            lines.append(
                f"| `{bean.get('class', '').rsplit('.', 1)[-1]}` | {bean.get('type', '')} | "
                f"{ann_str} | {ep} | {file_short} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _render_entry_points(self, wiring_data: Optional[Dict], language: str) -> str:
        """Render implicit entry point inventory."""
        if not wiring_data:
            return ""

        entry_points = wiring_data.get("entry_points", {})
        ep_summary = wiring_data.get("entry_point_summary", {})
        total = ep_summary.get("total", 0)

        if total == 0:
            return ""

        subtitle = "Points d'Entrée Implicites" if language == "fr" else "Implicit Entry Points"
        lines = [f"\n## {subtitle}\n"]

        if language == "fr":
            lines.append(f"**{total} points d'entrée** identifiés, invisibles à l'analyse "
                         f"de code mort standard.\n")
        else:
            lines.append(f"**{total} entry points** identified, invisible to standard "
                         f"dead code analysis.\n")

        # Category breakdown
        categories = [
            ("rest_endpoints", "REST Endpoints", "Endpoints REST"),
            ("jms_listeners", "JMS Listeners", "Listeners JMS"),
            ("scheduled", "Scheduled Tasks", "Tâches planifiées"),
            ("event_listeners", "Event Listeners", "Listeners d'événements"),
            ("main_methods", "Main Methods", "Méthodes main()"),
        ]

        for key, label_en, label_fr in categories:
            items = entry_points.get(key, [])
            if not items:
                continue
            label = label_fr if language == "fr" else label_en
            lines.append(f"\n### {label} ({len(items)})\n")
            lines.append("| Method | File | Line |")
            lines.append("|--------|------|:----:|")
            for ep in items:
                method_short = ep.get("method", "").rsplit(".", 1)[-1]
                file_short = Path(ep.get("file", "")).name
                lines.append(f"| `{method_short}` | {file_short} | {ep.get('line', '')} |")

        lines.append("")
        return "\n".join(lines)

    def _render_wiring(self, wiring_data: Optional[Dict], language: str) -> str:
        """Render dependency injection wiring."""
        if not wiring_data:
            return ""
        wiring = wiring_data.get("wiring", [])
        if not wiring:
            return ""

        subtitle = "Injection de Dépendances" if language == "fr" else "Dependency Injection Wiring"
        lines = [f"\n## {subtitle}\n"]

        if language == "fr":
            lines.append(f"**{len(wiring)} points d'injection** détectés.\n")
        else:
            lines.append(f"**{len(wiring)} injection points** detected.\n")

        hdr = "| From (Bean) | Field | Type | Annotations |" if language != "fr" \
            else "| Bean source | Champ | Type | Annotations |"
        lines.append(hdr)
        lines.append("|-------------|-------|------|-------------|")
        for w in wiring[:50]:  # Limit to 50 rows
            from_short = w.get("from", "").rsplit(".", 1)[-1]
            field_short = w.get("field", "").rsplit(".", 1)[-1]
            ann_str = ", ".join(f"@{a}" for a in w.get("annotations", []))
            lines.append(f"| `{from_short}` | `{field_short}` | {w.get('type', '')} | {ann_str} |")
        if len(wiring) > 50:
            lines.append(f"| ... | | | ({len(wiring) - 50} more) |")

        lines.append("")
        return "\n".join(lines)

    def _render_corrected_dead_code(self, wiring_data: Optional[Dict],
                                     dead_code_data: Optional[Dict], language: str) -> str:
        """Render corrected dead code analysis with Spring-aware reachability."""
        if not dead_code_data:
            return ""

        subtitle = "Analyse Code Mort Corrigée (Spring-aware)" if language == "fr" \
            else "Corrected Dead Code Analysis (Spring-aware)"
        lines = [f"\n## {subtitle}\n"]

        candidates = dead_code_data.get("candidates", [])
        original_count = len(candidates)

        if not wiring_data or not candidates:
            if language == "fr":
                lines.append(f"*{original_count} candidats code mort identifiés "
                             f"(pas de données Spring pour correction).*\n")
            else:
                lines.append(f"*{original_count} dead code candidates identified "
                             f"(no Spring data for correction).*\n")
            return "\n".join(lines)

        # Build set of Spring-reachable classes/methods
        reachable = set()
        for bean in wiring_data.get("beans", []):
            reachable.add(bean.get("class", ""))
        for category in wiring_data.get("entry_points", {}).values():
            for ep in category:
                reachable.add(ep.get("method", ""))
                # Also mark the parent class
                parts = ep.get("method", "").rsplit(".", 1)
                if len(parts) > 1:
                    reachable.add(parts[0])

        # Classify candidates
        false_positives = []
        true_dead = []
        for c in candidates:
            qname = c.get("qualified_name", "")
            # Check if this candidate or its parent class is Spring-reachable
            parent = qname.rsplit(".", 1)[0] if "." in qname else qname
            if qname in reachable or parent in reachable:
                false_positives.append(c)
            else:
                true_dead.append(c)

        corrected_count = len(true_dead)
        fp_count = len(false_positives)

        if language == "fr":
            lines.append(f"L'analyse standard identifiait **{original_count} candidats** code mort. "
                         f"Après correction par la résolution Spring :\n")
            lines.append(f"- **{fp_count} faux positifs** éliminés (beans Spring, points d'entrée implicites)")
            lines.append(f"- **{corrected_count} véritables candidats** code mort restants")
            if original_count > 0:
                reduction = round(fp_count / original_count * 100, 1)
                lines.append(f"- **Réduction :** {reduction}%\n")
        else:
            lines.append(f"Standard analysis identified **{original_count} candidates** as dead code. "
                         f"After Spring wiring correction:\n")
            lines.append(f"- **{fp_count} false positives** eliminated (Spring beans, implicit entry points)")
            lines.append(f"- **{corrected_count} true dead code** candidates remaining")
            if original_count > 0:
                reduction = round(fp_count / original_count * 100, 1)
                lines.append(f"- **Reduction:** {reduction}%\n")

        lines.append("")
        return "\n".join(lines)

    def _render_reachability_delta(self, wiring_data: Optional[Dict], language: str) -> str:
        """Render reachability improvement summary."""
        if not wiring_data:
            return ""

        delta = wiring_data.get("reachability_delta", {})
        if not delta:
            return ""

        subtitle = "Amélioration de la Couverture de Reachability" if language == "fr" \
            else "Reachability Coverage Improvement"
        lines = [f"\n## {subtitle}\n"]

        before = delta.get("before", 1)
        after = delta.get("after", 1)
        ratio = delta.get("improvement_ratio", 1)
        new_beans = delta.get("new_bean_classes", 0)

        if language == "fr":
            lines.append(f"| Métrique | Avant | Après |")
            lines.append(f"|----------|:-----:|:-----:|")
            lines.append(f"| Points d'entrée connus | {before} | {after} |")
            lines.append(f"| Classes bean Spring | — | {new_beans} |")
            lines.append(f"| **Facteur d'amélioration** | | **×{ratio}** |")
        else:
            lines.append(f"| Metric | Before | After |")
            lines.append(f"|--------|:------:|:-----:|")
            lines.append(f"| Known entry points | {before} | {after} |")
            lines.append(f"| Spring bean classes | — | {new_beans} |")
            lines.append(f"| **Improvement factor** | | **x{ratio}** |")

        lines.append("")
        return "\n".join(lines)
