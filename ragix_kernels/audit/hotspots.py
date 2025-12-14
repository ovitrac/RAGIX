"""
Kernel: Complexity Hotspots
Stage: 2 (Analysis)
Dependencies: ast_scan, metrics, dependency

Identifies complexity hotspots:
- High cyclomatic complexity methods
- Large classes (God classes)
- Deep inheritance hierarchies
- High coupling components

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class HotspotsKernel(Kernel):
    """
    Identify complexity hotspots.

    This kernel analyzes code to find areas that need attention:
    - High complexity methods
    - Oversized classes
    - High coupling components
    - Potential God classes

    Configuration options:
        top_n: Number of top hotspots to report (default: 50)
        threshold_cc: Complexity threshold (default: 15)
        threshold_loc: Lines of code threshold (default: 500)
        threshold_methods: Methods per class threshold (default: 25)
        threshold_coupling: Coupling threshold (default: 20)

    Dependencies:
        ast_scan: Symbol data
        metrics: Code metrics
        dependency: Coupling data

    Output:
        complexity_hotspots: High-complexity methods
        size_hotspots: Large classes/files
        coupling_hotspots: High-coupling components
        ranked_hotspots: Combined ranked list
    """

    name = "hotspots"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Identify complexity hotspots"

    requires = ["ast_scan", "metrics", "dependency"]
    provides = ["hotspots", "risk_areas"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Identify complexity hotspots."""

        # Load dependencies
        ast_path = input.dependencies.get("ast_scan")
        metrics_path = input.dependencies.get("metrics")
        dep_path = input.dependencies.get("dependency")

        if not all([ast_path, metrics_path, dep_path]):
            raise RuntimeError("Missing required dependencies")

        with open(ast_path) as f:
            ast_data = json.load(f).get("data", {})
        with open(metrics_path) as f:
            metrics_data = json.load(f).get("data", {})
        with open(dep_path) as f:
            dep_data = json.load(f).get("data", {})

        # Get configuration
        top_n = input.config.get("top_n", 50)
        threshold_cc = input.config.get("threshold_cc", 15)
        threshold_loc = input.config.get("threshold_loc", 500)
        threshold_methods = input.config.get("threshold_methods", 25)
        threshold_coupling = input.config.get("threshold_coupling", 20)

        logger.info(f"[hotspots] Analyzing hotspots (CC>{threshold_cc}, LOC>{threshold_loc})")

        # Complexity hotspots from metrics
        complexity_hotspots = self._find_complexity_hotspots(
            metrics_data, threshold_cc, top_n
        )

        # Size hotspots (large classes/files)
        size_hotspots = self._find_size_hotspots(
            ast_data, metrics_data, threshold_loc, threshold_methods, top_n
        )

        # Coupling hotspots
        coupling_hotspots = self._find_coupling_hotspots(
            dep_data, threshold_coupling, top_n
        )

        # Combined ranked list
        ranked_hotspots = self._rank_all_hotspots(
            complexity_hotspots, size_hotspots, coupling_hotspots, top_n
        )

        # Statistics
        statistics = {
            "complexity_hotspots": len(complexity_hotspots),
            "size_hotspots": len(size_hotspots),
            "coupling_hotspots": len(coupling_hotspots),
            "total_unique": len(ranked_hotspots),
            "thresholds": {
                "complexity": threshold_cc,
                "loc": threshold_loc,
                "methods": threshold_methods,
                "coupling": threshold_coupling,
            }
        }

        return {
            "complexity_hotspots": complexity_hotspots,
            "size_hotspots": size_hotspots,
            "coupling_hotspots": coupling_hotspots,
            "ranked_hotspots": ranked_hotspots,
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        ranked = data.get("ranked_hotspots", [])

        # Top 3 hotspots
        top_3 = []
        for h in ranked[:3]:
            name = h.get("name", "unknown")
            if len(name) > 30:
                name = "..." + name[-27:]
            top_3.append(f"{name}({h.get('score', 0):.0f})")

        top_str = ", ".join(top_3) if top_3 else "none"

        return (
            f"Hotspots: {stats.get('complexity_hotspots', 0)} complexity, "
            f"{stats.get('size_hotspots', 0)} size, "
            f"{stats.get('coupling_hotspots', 0)} coupling. "
            f"Total unique: {stats.get('total_unique', 0)}. "
            f"Top: {top_str}."
        )

    def _find_complexity_hotspots(
        self,
        metrics_data: Dict[str, Any],
        threshold: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Find high-complexity methods."""
        hotspots = metrics_data.get("hotspots", [])

        result = []
        for h in hotspots:
            cc = h.get("complexity", 0)
            if cc >= threshold:
                result.append({
                    "name": h.get("name", "unknown"),
                    "type": "complexity",
                    "value": cc,
                    "severity": "critical" if cc > threshold * 2 else "high",
                    "reason": f"Cyclomatic complexity {cc} exceeds threshold {threshold}",
                })

        return sorted(result, key=lambda x: -x["value"])[:limit]

    def _find_size_hotspots(
        self,
        ast_data: Dict[str, Any],
        metrics_data: Dict[str, Any],
        threshold_loc: int,
        threshold_methods: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Find large classes and files."""
        result = []

        # Large files from metrics
        for f in metrics_data.get("files", []):
            loc = f.get("loc", 0)
            if loc > threshold_loc:
                result.append({
                    "name": f.get("path", "unknown"),
                    "type": "large_file",
                    "value": loc,
                    "severity": "high" if loc > threshold_loc * 2 else "medium",
                    "reason": f"File has {loc} LOC (threshold: {threshold_loc})",
                })

        # Large classes from AST
        symbols = ast_data.get("symbols", [])
        class_methods: Dict[str, int] = {}

        for sym in symbols:
            if sym.get("type") == "class":
                class_methods[sym.get("qualified_name", "")] = 0
            elif sym.get("type") == "method":
                # Find parent class from qualified name
                qname = sym.get("qualified_name", "")
                parts = qname.rsplit(".", 1)
                if len(parts) == 2:
                    class_name = parts[0]
                    if class_name in class_methods:
                        class_methods[class_name] += 1

        for class_name, method_count in class_methods.items():
            if method_count > threshold_methods:
                result.append({
                    "name": class_name,
                    "type": "god_class",
                    "value": method_count,
                    "severity": "high" if method_count > threshold_methods * 2 else "medium",
                    "reason": f"Class has {method_count} methods (threshold: {threshold_methods})",
                })

        return sorted(result, key=lambda x: -x["value"])[:limit]

    def _find_coupling_hotspots(
        self,
        dep_data: Dict[str, Any],
        threshold: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Find high-coupling components."""
        coupling = dep_data.get("coupling", {})
        high_coupling = coupling.get("high_coupling_nodes", [])

        result = []
        for node in high_coupling:
            ca = node.get("ca", 0)
            ce = node.get("ce", 0)
            total = ca + ce

            if total > threshold:
                result.append({
                    "name": node.get("id", "unknown"),
                    "type": "high_coupling",
                    "value": total,
                    "afferent": ca,
                    "efferent": ce,
                    "severity": "high" if total > threshold * 2 else "medium",
                    "reason": f"Total coupling {total} (Ca={ca}, Ce={ce}), threshold: {threshold}",
                })

        return sorted(result, key=lambda x: -x["value"])[:limit]

    def _rank_all_hotspots(
        self,
        complexity: List[Dict],
        size: List[Dict],
        coupling: List[Dict],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Combine and rank all hotspots."""
        # Normalize scores
        all_hotspots = []

        for h in complexity:
            score = h["value"] * 10  # Weight complexity highly
            all_hotspots.append({
                **h,
                "score": score,
                "categories": ["complexity"],
            })

        for h in size:
            score = h["value"] / 10  # Normalize LOC
            if h["type"] == "god_class":
                score = h["value"] * 5  # Weight god classes
            all_hotspots.append({
                **h,
                "score": score,
                "categories": ["size"],
            })

        for h in coupling:
            score = h["value"] * 3  # Weight coupling
            all_hotspots.append({
                **h,
                "score": score,
                "categories": ["coupling"],
            })

        # Sort by score
        all_hotspots.sort(key=lambda x: -x["score"])

        return all_hotspots[:limit]
