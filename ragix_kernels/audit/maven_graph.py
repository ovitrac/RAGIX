"""
Kernel: Maven Module Graph
Stage: 2 (Analysis)

Builds and analyzes the inter-module dependency graph from Maven
parent/child and dependency relationships extracted by maven_deps.

Computes:
- Adjacency list and transitive closure
- In-degree / out-degree per module
- Betweenness centrality
- Cycle detection
- Root modules, leaf modules, critical path
- Visualization (SVG via matplotlib, optional)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-09
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import logging

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)

# Optional matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MavenGraphKernel(Kernel):
    """
    Analyze Maven module dependency graph.

    Builds a directed graph of inter-module dependencies from maven_deps
    output, computes graph-theoretic metrics, and optionally generates
    SVG visualizations.

    Configuration:
        maven_graph.visualize: Generate SVG visualization (default: true)
        maven_graph.layout: Layout algorithm - "hierarchical" or "circular" (default: "hierarchical")
    """

    name = "maven_graph"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Analyze Maven module dependency graph"
    requires = ["maven_deps"]
    provides = ["maven_module_graph", "maven_centrality"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load maven_deps output
        deps_path = input.dependencies.get("maven_deps")
        if not deps_path or not deps_path.exists():
            raise RuntimeError("maven_deps output not found")

        with open(deps_path) as f:
            deps_data = json.load(f).get("data", {})

        modules = deps_data.get("modules", [])
        if not modules:
            return self._empty_result()

        # Build graph
        nodes, edges = self._build_graph(modules)
        adjacency = self._build_adjacency(nodes, edges)

        # Compute metrics
        in_degree = {n: 0 for n in nodes}
        out_degree = {n: 0 for n in nodes}
        for edge in edges:
            out_degree[edge["from"]] = out_degree.get(edge["from"], 0) + 1
            in_degree[edge["to"]] = in_degree.get(edge["to"], 0) + 1

        cycles = self._detect_cycles(nodes, adjacency)
        centrality = self._compute_betweenness(nodes, adjacency)
        roots = [n for n in nodes if in_degree.get(n, 0) == 0]
        leaves = [n for n in nodes if out_degree.get(n, 0) == 0]
        critical_path = self._find_longest_path(nodes, adjacency)
        transitive = self._transitive_closure(nodes, adjacency)

        # Module details with metrics
        module_details = {}
        for n in nodes:
            module_details[n] = {
                "in_degree": in_degree.get(n, 0),
                "out_degree": out_degree.get(n, 0),
                "betweenness": round(centrality.get(n, 0.0), 4),
                "transitive_deps": len(transitive.get(n, set())),
                "is_root": n in roots,
                "is_leaf": n in leaves,
            }

        # Visualization
        viz_path = None
        do_viz = input.config.get("maven_graph", {}).get("visualize", True)
        layout = input.config.get("maven_graph", {}).get("layout", "hierarchical")
        if do_viz and HAS_MATPLOTLIB:
            viz_path = self._generate_visualization(
                nodes, edges, module_details, input.workspace, layout
            )

        return {
            "graph": {
                "nodes": sorted(nodes),
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
            "centrality": module_details,
            "critical_path": critical_path,
            "cycles": cycles,
            "roots": sorted(roots),
            "leaves": sorted(leaves),
            "transitive_closure": {
                k: sorted(v) for k, v in transitive.items()
            },
            "statistics": {
                "total_modules": len(nodes),
                "total_edges": len(edges),
                "max_in_degree": max(in_degree.values()) if in_degree else 0,
                "max_out_degree": max(out_degree.values()) if out_degree else 0,
                "hub_module": max(centrality, key=centrality.get) if centrality else None,
                "hub_centrality": round(max(centrality.values()), 4) if centrality else 0,
                "cycle_count": len(cycles),
                "critical_path_length": len(critical_path),
                "root_modules": len(roots),
                "leaf_modules": len(leaves),
                "visualization": str(viz_path) if viz_path else None,
            },
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        stats = data.get("statistics", {})
        hub = stats.get("hub_module", "none")
        return (
            f"Maven graph: {stats.get('total_modules', 0)} modules, "
            f"{stats.get('total_edges', 0)} edges. "
            f"Hub: {hub} (centrality={stats.get('hub_centrality', 0)}). "
            f"Cycles: {stats.get('cycle_count', 0)}. "
            f"Critical path: {stats.get('critical_path_length', 0)} modules. "
            f"Roots: {stats.get('root_modules', 0)}, Leaves: {stats.get('leaf_modules', 0)}."
        )

    def _build_graph(
        self, modules: List[Dict[str, Any]]
    ) -> Tuple[Set[str], List[Dict[str, str]]]:
        """Build nodes and edges from maven_deps modules."""
        nodes: Set[str] = set()
        edges: List[Dict[str, str]] = []
        seen_edges: Set[Tuple[str, str]] = set()

        # Collect all module artifactIds
        module_ids = {m["artifactId"] for m in modules}
        nodes.update(module_ids)

        for module in modules:
            src = module["artifactId"]

            # Parent relationship
            parent = module.get("parent")
            if parent and parent.get("artifactId") in module_ids:
                edge = (src, parent["artifactId"])
                if edge not in seen_edges:
                    edges.append({"from": src, "to": parent["artifactId"], "type": "parent"})
                    seen_edges.add(edge)

            # Dependency relationships (only internal modules)
            for dep in module.get("dependencies", []):
                target = dep["artifactId"]
                if target in module_ids and target != src:
                    edge = (src, target)
                    if edge not in seen_edges:
                        edges.append({
                            "from": src,
                            "to": target,
                            "type": dep.get("scope", "compile"),
                        })
                        seen_edges.add(edge)

        return nodes, edges

    def _build_adjacency(
        self, nodes: Set[str], edges: List[Dict[str, str]]
    ) -> Dict[str, Set[str]]:
        """Build adjacency list (outgoing)."""
        adj: Dict[str, Set[str]] = {n: set() for n in nodes}
        for edge in edges:
            adj[edge["from"]].add(edge["to"])
        return adj

    def _detect_cycles(
        self, nodes: Set[str], adjacency: Dict[str, Set[str]]
    ) -> List[List[str]]:
        """Detect cycles using DFS (Tarjan-style back-edge detection)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}
        cycles = []
        path: List[str] = []

        def dfs(u):
            color[u] = GRAY
            path.append(u)
            for v in adjacency.get(u, set()):
                if color[v] == GRAY:
                    # Back edge → cycle
                    idx = path.index(v)
                    cycles.append(path[idx:] + [v])
                elif color[v] == WHITE:
                    dfs(v)
            path.pop()
            color[u] = BLACK

        for n in nodes:
            if color[n] == WHITE:
                dfs(n)

        return cycles

    def _compute_betweenness(
        self, nodes: Set[str], adjacency: Dict[str, Set[str]]
    ) -> Dict[str, float]:
        """Compute betweenness centrality (Brandes algorithm, unweighted)."""
        centrality = {n: 0.0 for n in nodes}
        node_list = sorted(nodes)

        for s in node_list:
            # BFS from s
            stack = []
            predecessors: Dict[str, List[str]] = {n: [] for n in nodes}
            sigma = {n: 0.0 for n in nodes}
            sigma[s] = 1.0
            dist = {n: -1 for n in nodes}
            dist[s] = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in adjacency.get(v, set()):
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

            # Accumulation
            delta = {n: 0.0 for n in nodes}
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    centrality[w] += delta[w]

        # Normalize
        n = len(nodes)
        if n > 2:
            norm = 1.0 / ((n - 1) * (n - 2))
            centrality = {k: v * norm for k, v in centrality.items()}

        return centrality

    def _find_longest_path(
        self, nodes: Set[str], adjacency: Dict[str, Set[str]]
    ) -> List[str]:
        """Find the longest path in DAG (critical path)."""
        # Topological sort (Kahn's algorithm)
        in_deg = {n: 0 for n in nodes}
        for n in nodes:
            for m in adjacency.get(n, set()):
                in_deg[m] = in_deg.get(m, 0) + 1

        queue = deque([n for n in nodes if in_deg[n] == 0])
        topo_order = []
        while queue:
            n = queue.popleft()
            topo_order.append(n)
            for m in adjacency.get(n, set()):
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)

        # If graph has cycles, topo_order is incomplete — use what we have
        if not topo_order:
            return []

        # Longest path via DP
        dist = {n: 0 for n in nodes}
        predecessor = {n: None for n in nodes}

        for u in topo_order:
            for v in adjacency.get(u, set()):
                if dist[u] + 1 > dist[v]:
                    dist[v] = dist[u] + 1
                    predecessor[v] = u

        # Reconstruct longest path
        if not dist:
            return []
        end_node = max(dist, key=dist.get)
        path = []
        node = end_node
        while node is not None:
            path.append(node)
            node = predecessor[node]
        path.reverse()
        return path

    def _transitive_closure(
        self, nodes: Set[str], adjacency: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Compute transitive closure (all reachable modules per node)."""
        closure: Dict[str, Set[str]] = {}
        for start in nodes:
            visited: Set[str] = set()
            queue = deque(adjacency.get(start, set()))
            while queue:
                n = queue.popleft()
                if n in visited:
                    continue
                visited.add(n)
                queue.extend(adjacency.get(n, set()) - visited)
            closure[start] = visited
        return closure

    def _generate_visualization(
        self,
        nodes: Set[str],
        edges: List[Dict[str, str]],
        module_details: Dict[str, Dict[str, Any]],
        workspace: Path,
        layout: str = "hierarchical",
    ) -> Optional[Path]:
        """Generate SVG visualization of the dependency graph."""
        if not HAS_MATPLOTLIB:
            return None

        try:
            positions = self._compute_layout(nodes, edges, module_details, layout)
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.set_aspect('equal')

            # Draw edges
            for edge in edges:
                src, tgt = edge["from"], edge["to"]
                if src in positions and tgt in positions:
                    x0, y0 = positions[src]
                    x1, y1 = positions[tgt]
                    edge_type = edge.get("type", "compile")
                    style = '--' if edge_type in ("test", "provided") else '-'
                    color = '#888888' if edge_type == "parent" else '#4a90d9'
                    ax.annotate(
                        "", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle='->', color=color,
                            linestyle=style, lw=1.2,
                            connectionstyle='arc3,rad=0.05',
                        ),
                    )

            # Draw nodes
            for name in nodes:
                if name not in positions:
                    continue
                x, y = positions[name]
                detail = module_details.get(name, {})
                betw = detail.get("betweenness", 0)
                in_deg = detail.get("in_degree", 0)

                # Size proportional to centrality
                size = 300 + betw * 8000
                # Color by role
                if detail.get("is_root"):
                    color = '#2ecc71'   # green
                elif betw > 0.1:
                    color = '#e74c3c'   # red (hub)
                elif detail.get("is_leaf"):
                    color = '#95a5a6'   # gray
                else:
                    color = '#3498db'   # blue

                ax.scatter(x, y, s=size, c=color, zorder=5, edgecolors='white', linewidth=1.5)

                # Label
                short_name = name.replace("iow-", "").replace("iog-", "").replace("iok-", "")
                fontsize = 7 if len(nodes) > 15 else 9
                ax.annotate(
                    short_name, (x, y),
                    fontsize=fontsize, ha='center', va='center',
                    fontweight='bold', color='white' if betw > 0.05 else 'black',
                    zorder=6,
                )

            # Legend
            legend_elements = [
                mpatches.Patch(facecolor='#2ecc71', label='Root module'),
                mpatches.Patch(facecolor='#e74c3c', label='Hub (high centrality)'),
                mpatches.Patch(facecolor='#3498db', label='Internal module'),
                mpatches.Patch(facecolor='#95a5a6', label='Leaf module'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

            ax.set_title("Maven Module Dependency Graph", fontsize=14, fontweight='bold')
            ax.axis('off')
            fig.tight_layout()

            # Save SVG
            assets_dir = workspace / "stage2" / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            output_path = assets_dir / "maven_graph.svg"
            fig.savefig(output_path, format="svg", bbox_inches='tight', dpi=150)
            plt.close(fig)

            # Also save PNG for embedding
            png_path = assets_dir / "maven_graph.png"
            fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
            ax2.set_aspect('equal')
            # Re-draw for PNG (same logic)
            for edge in edges:
                src, tgt = edge["from"], edge["to"]
                if src in positions and tgt in positions:
                    x0, y0 = positions[src]
                    x1, y1 = positions[tgt]
                    edge_type = edge.get("type", "compile")
                    style = '--' if edge_type in ("test", "provided") else '-'
                    color = '#888888' if edge_type == "parent" else '#4a90d9'
                    ax2.annotate(
                        "", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle='->', color=color,
                            linestyle=style, lw=1.2,
                            connectionstyle='arc3,rad=0.05',
                        ),
                    )
            for name in nodes:
                if name not in positions:
                    continue
                x, y = positions[name]
                detail = module_details.get(name, {})
                betw = detail.get("betweenness", 0)
                size = 300 + betw * 8000
                if detail.get("is_root"):
                    color = '#2ecc71'
                elif betw > 0.1:
                    color = '#e74c3c'
                elif detail.get("is_leaf"):
                    color = '#95a5a6'
                else:
                    color = '#3498db'
                ax2.scatter(x, y, s=size, c=color, zorder=5, edgecolors='white', linewidth=1.5)
                short_name = name.replace("iow-", "").replace("iog-", "").replace("iok-", "")
                fontsize = 7 if len(nodes) > 15 else 9
                ax2.annotate(
                    short_name, (x, y),
                    fontsize=fontsize, ha='center', va='center',
                    fontweight='bold', color='white' if betw > 0.05 else 'black',
                    zorder=6,
                )
            ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)
            ax2.set_title("Maven Module Dependency Graph", fontsize=14, fontweight='bold')
            ax2.axis('off')
            fig2.tight_layout()
            fig2.savefig(png_path, format="png", bbox_inches='tight', dpi=150)
            plt.close(fig2)

            logger.info(f"[maven_graph] Visualization saved to {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"[maven_graph] Visualization failed: {e}")
            return None

    def _compute_layout(
        self,
        nodes: Set[str],
        edges: List[Dict[str, str]],
        module_details: Dict[str, Dict[str, Any]],
        layout: str,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute node positions for visualization."""
        if layout == "circular":
            return self._circular_layout(nodes)
        return self._hierarchical_layout(nodes, edges, module_details)

    def _circular_layout(self, nodes: Set[str]) -> Dict[str, Tuple[float, float]]:
        """Simple circular layout."""
        import math
        sorted_nodes = sorted(nodes)
        n = len(sorted_nodes)
        positions = {}
        for i, name in enumerate(sorted_nodes):
            angle = 2 * math.pi * i / n
            positions[name] = (math.cos(angle), math.sin(angle))
        return positions

    def _hierarchical_layout(
        self,
        nodes: Set[str],
        edges: List[Dict[str, str]],
        module_details: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Hierarchical layout: roots at top, leaves at bottom.
        Layers assigned by longest path from any root.
        """
        # Build adjacency
        adj: Dict[str, Set[str]] = {n: set() for n in nodes}
        for e in edges:
            adj[e["from"]].add(e["to"])

        # Assign layers by BFS from roots
        roots = [n for n in nodes if module_details.get(n, {}).get("is_root", False)]
        if not roots:
            roots = sorted(nodes)[:1]

        layer: Dict[str, int] = {}
        queue = deque()
        for r in roots:
            layer[r] = 0
            queue.append(r)

        while queue:
            u = queue.popleft()
            for v in adj.get(u, set()):
                new_layer = layer[u] + 1
                if v not in layer or new_layer > layer[v]:
                    layer[v] = new_layer
                    queue.append(v)

        # Assign remaining unvisited nodes
        max_layer = max(layer.values()) if layer else 0
        for n in nodes:
            if n not in layer:
                layer[n] = max_layer + 1

        # Group by layer
        layers: Dict[int, List[str]] = defaultdict(list)
        for n, l in layer.items():
            layers[l].append(n)

        # Sort within each layer by centrality (hub in center)
        for l in layers:
            layers[l].sort(key=lambda n: -module_details.get(n, {}).get("betweenness", 0))

        # Compute positions
        positions = {}
        max_l = max(layers.keys()) if layers else 0
        for l, members in layers.items():
            count = len(members)
            for i, name in enumerate(members):
                x = (i - (count - 1) / 2) * 1.5
                y = -l * 2.0  # top to bottom
                positions[name] = (x, y)

        return positions

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "graph": {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0},
            "centrality": {},
            "critical_path": [],
            "cycles": [],
            "roots": [],
            "leaves": [],
            "transitive_closure": {},
            "statistics": {
                "total_modules": 0, "total_edges": 0,
                "max_in_degree": 0, "max_out_degree": 0,
                "hub_module": None, "hub_centrality": 0,
                "cycle_count": 0, "critical_path_length": 0,
                "root_modules": 0, "leaf_modules": 0,
                "visualization": None,
            },
        }
