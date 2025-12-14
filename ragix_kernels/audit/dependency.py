"""
Kernel: Dependency Graph
Stage: 1 (Data Collection)
Wraps: ragix_core.dependencies.DependencyGraph

Builds and analyzes the dependency graph:
- Constructs symbol dependency relationships
- Computes coupling metrics (afferent/efferent)
- Detects circular dependencies
- Provides graph statistics

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
import json

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DependencyKernel(Kernel):
    """
    Build and analyze dependency graph.

    This kernel constructs the dependency graph from AST data
    and computes various dependency metrics.

    Configuration options:
        include_external: Include external dependencies (default: false)
        detect_cycles: Run cycle detection (default: true)
        max_cycles: Maximum cycles to report (default: 100)

    Dependencies:
        ast_scan: Provides symbol and dependency data

    Output:
        graph: Dependency graph data (nodes, edges)
        coupling: Afferent and efferent coupling per symbol
        cycles: Detected circular dependencies
        statistics: Graph statistics
    """

    name = "dependency"
    version = "1.0.0"
    category = "audit"
    stage = 1
    description = "Build and analyze dependency graph"

    requires = ["ast_scan"]
    provides = ["graph", "coupling", "cycles"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Build dependency graph and compute metrics."""

        # Import here to avoid circular imports
        from ragix_core.dependencies import DependencyGraph

        # Get configuration
        include_external = input.config.get("include_external", False)
        detect_cycles = input.config.get("detect_cycles", True)
        max_cycles = input.config.get("max_cycles", 100)

        # Load ast_scan output
        ast_scan_path = input.dependencies.get("ast_scan")
        if not ast_scan_path or not ast_scan_path.exists():
            raise RuntimeError("Missing required dependency: ast_scan")

        with open(ast_scan_path) as f:
            ast_data = json.load(f)

        # Reconstruct graph from cached data
        symbols = ast_data.get("data", {}).get("symbols", [])
        dependencies = ast_data.get("data", {}).get("dependencies", [])
        graph = DependencyGraph.from_cached_data(symbols, dependencies)

        logger.info(f"[dependency] Analyzing graph with {len(symbols)} symbols, {len(dependencies)} deps")

        # Get statistics
        stats = graph.get_stats()

        # Build nodes data (symbols with coupling info)
        nodes_data = []
        for sym_data in symbols:
            name = sym_data.get("qualified_name", sym_data.get("name", ""))
            nodes_data.append({
                "id": name,
                "name": sym_data.get("name", ""),
                "type": sym_data.get("type", "unknown"),
                "file": sym_data.get("file"),
                "line": sym_data.get("line", 0),
                "afferent_coupling": stats.afferent_coupling.get(name, 0),
                "efferent_coupling": stats.efferent_coupling.get(name, 0),
            })

        # Build edges data
        edges_data = []
        for dep in dependencies:
            edges_data.append({
                "source": dep.get("source", ""),
                "target": dep.get("target", ""),
                "type": dep.get("type", "unknown"),
            })

        # Filter external dependencies if requested
        if not include_external:
            internal_symbols = {n["id"] for n in nodes_data}
            edges_data = [
                e for e in edges_data
                if e["source"] in internal_symbols and e["target"] in internal_symbols
            ]

        # Detect cycles
        cycles_data = []
        if detect_cycles:
            cycles = graph.detect_cycles()
            cycles_data = cycles[:max_cycles]
            if len(cycles) > max_cycles:
                logger.warning(f"[dependency] Truncated cycles from {len(cycles)} to {max_cycles}")

        # Compute coupling summary
        coupling_summary = self._compute_coupling_summary(nodes_data)

        # Graph statistics
        graph_stats = {
            "total_nodes": len(nodes_data),
            "total_edges": len(edges_data),
            "total_cycles": len(cycles_data),
            "avg_afferent": coupling_summary["avg_afferent"],
            "avg_efferent": coupling_summary["avg_efferent"],
            "max_afferent": coupling_summary["max_afferent"],
            "max_efferent": coupling_summary["max_efferent"],
            "by_type": self._count_by_type(edges_data),
        }

        return {
            "graph": {
                "nodes": nodes_data,
                "edges": edges_data,
            },
            "coupling": coupling_summary,
            "cycles": cycles_data,
            "statistics": graph_stats,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        coupling = data.get("coupling", {})
        cycles = data.get("cycles", [])

        return (
            f"Dependency graph: {stats.get('total_nodes', 0)} nodes, "
            f"{stats.get('total_edges', 0)} edges. "
            f"Coupling: avg Ca={coupling.get('avg_afferent', 0):.1f}, "
            f"avg Ce={coupling.get('avg_efferent', 0):.1f}. "
            f"Circular dependencies: {len(cycles)}. "
            f"Most depended: {coupling.get('most_depended', 'N/A')} (Ca={coupling.get('max_afferent', 0)})."
        )

    def _compute_coupling_summary(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute coupling statistics summary."""
        if not nodes:
            return {
                "avg_afferent": 0,
                "avg_efferent": 0,
                "max_afferent": 0,
                "max_efferent": 0,
                "most_depended": None,
                "most_dependent": None,
                "high_coupling_nodes": [],
            }

        afferents = [n["afferent_coupling"] for n in nodes]
        efferents = [n["efferent_coupling"] for n in nodes]

        # Find nodes with highest coupling
        max_aff_node = max(nodes, key=lambda n: n["afferent_coupling"])
        max_eff_node = max(nodes, key=lambda n: n["efferent_coupling"])

        # Find high-coupling nodes (above threshold)
        threshold = 10
        high_coupling = [
            {"id": n["id"], "ca": n["afferent_coupling"], "ce": n["efferent_coupling"]}
            for n in nodes
            if n["afferent_coupling"] > threshold or n["efferent_coupling"] > threshold
        ]
        high_coupling.sort(key=lambda x: x["ca"] + x["ce"], reverse=True)

        return {
            "avg_afferent": sum(afferents) / len(afferents) if afferents else 0,
            "avg_efferent": sum(efferents) / len(efferents) if efferents else 0,
            "max_afferent": max(afferents) if afferents else 0,
            "max_efferent": max(efferents) if efferents else 0,
            "most_depended": max_aff_node["id"],
            "most_dependent": max_eff_node["id"],
            "high_coupling_nodes": high_coupling[:20],  # Top 20
        }

    def _count_by_type(self, edges: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count edges by type."""
        counts: Dict[str, int] = {}
        for edge in edges:
            t = edge.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts
