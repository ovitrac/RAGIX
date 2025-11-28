"""
AST Visualization - Generate visual representations of AST and dependencies

Supports:
- DOT format (Graphviz)
- Mermaid format
- D3.js JSON format
- Interactive HTML

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

import html
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .ast_base import ASTNode, NodeType, Symbol
from .dependencies import Dependency, DependencyGraph, DependencyType


class LayoutDirection(str, Enum):
    """Graph layout direction."""
    LEFT_RIGHT = "LR"
    TOP_BOTTOM = "TB"
    RIGHT_LEFT = "RL"
    BOTTOM_TOP = "BT"


class ColorScheme(str, Enum):
    """Predefined color schemes."""
    DEFAULT = "default"
    PASTEL = "pastel"
    DARK = "dark"
    MONOCHROME = "monochrome"


@dataclass
class VizConfig:
    """Configuration for visualization."""
    direction: LayoutDirection = LayoutDirection.LEFT_RIGHT
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    show_types: bool = True
    show_visibility: bool = True
    cluster_by_file: bool = False
    cluster_by_package: bool = False
    max_label_length: int = 40
    font_name: str = "Helvetica"
    font_size: int = 10
    include_orphans: bool = False
    filter_types: Optional[List[NodeType]] = None
    filter_deps: Optional[List[DependencyType]] = None


# Color palettes
COLORS = {
    ColorScheme.DEFAULT: {
        NodeType.CLASS: "#4a90d9",
        NodeType.INTERFACE: "#50c878",
        NodeType.METHOD: "#f5a623",
        NodeType.FUNCTION: "#f5a623",
        NodeType.FIELD: "#9b59b6",
        NodeType.CONSTANT: "#e74c3c",
        NodeType.MODULE: "#34495e",
        NodeType.PACKAGE: "#2c3e50",
        NodeType.ENUM: "#1abc9c",
        "edge_import": "#999999",
        "edge_inheritance": "#3498db",
        "edge_call": "#2ecc71",
        "edge_type": "#9b59b6",
    },
    ColorScheme.PASTEL: {
        NodeType.CLASS: "#aed6f1",
        NodeType.INTERFACE: "#abebc6",
        NodeType.METHOD: "#fdebd0",
        NodeType.FUNCTION: "#fdebd0",
        NodeType.FIELD: "#d7bde2",
        NodeType.CONSTANT: "#f5b7b1",
        NodeType.MODULE: "#d5dbdb",
        NodeType.PACKAGE: "#bdc3c7",
        NodeType.ENUM: "#a3e4d7",
        "edge_import": "#bdc3c7",
        "edge_inheritance": "#85c1e9",
        "edge_call": "#82e0aa",
        "edge_type": "#c39bd3",
    },
    ColorScheme.DARK: {
        NodeType.CLASS: "#2980b9",
        NodeType.INTERFACE: "#27ae60",
        NodeType.METHOD: "#d35400",
        NodeType.FUNCTION: "#d35400",
        NodeType.FIELD: "#8e44ad",
        NodeType.CONSTANT: "#c0392b",
        NodeType.MODULE: "#2c3e50",
        NodeType.PACKAGE: "#1a252f",
        NodeType.ENUM: "#16a085",
        "edge_import": "#7f8c8d",
        "edge_inheritance": "#2980b9",
        "edge_call": "#27ae60",
        "edge_type": "#8e44ad",
    },
    ColorScheme.MONOCHROME: {
        NodeType.CLASS: "#666666",
        NodeType.INTERFACE: "#888888",
        NodeType.METHOD: "#aaaaaa",
        NodeType.FUNCTION: "#aaaaaa",
        NodeType.FIELD: "#cccccc",
        NodeType.CONSTANT: "#444444",
        NodeType.MODULE: "#333333",
        NodeType.PACKAGE: "#222222",
        NodeType.ENUM: "#999999",
        "edge_import": "#999999",
        "edge_inheritance": "#666666",
        "edge_call": "#333333",
        "edge_type": "#aaaaaa",
    },
}


def _get_color(node_type: NodeType, scheme: ColorScheme) -> str:
    """Get color for a node type."""
    colors = COLORS.get(scheme, COLORS[ColorScheme.DEFAULT])
    return colors.get(node_type, "#cccccc")


def _get_edge_color(dep_type: DependencyType, scheme: ColorScheme) -> str:
    """Get color for an edge type."""
    colors = COLORS.get(scheme, COLORS[ColorScheme.DEFAULT])
    key = f"edge_{dep_type.value}"
    return colors.get(key, "#999999")


def _sanitize_id(name: str) -> str:
    """Convert a name to a valid graph ID."""
    return name.replace(".", "_").replace("-", "_").replace(" ", "_").replace("<", "").replace(">", "")


def _truncate(text: str, max_length: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


class DotRenderer:
    """Render dependency graphs to DOT format."""

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def render(self, graph: DependencyGraph) -> str:
        """Render graph to DOT format."""
        lines = []
        lines.append("digraph DependencyGraph {")
        lines.append(f"  rankdir={self.config.direction.value};")
        lines.append(f"  fontname=\"{self.config.font_name}\";")
        lines.append(f"  fontsize={self.config.font_size};")
        lines.append("  node [shape=box, style=filled, fontname=\"" +
                     self.config.font_name + "\"];")
        lines.append("  edge [fontname=\"" + self.config.font_name + "\"];")
        lines.append("")

        # Get nodes and edges
        symbols = graph.get_symbols()
        deps = graph.get_all_dependencies()

        # Filter by type if specified
        if self.config.filter_types:
            symbols = [s for s in symbols if s.node_type in self.config.filter_types]

        if self.config.filter_deps:
            deps = [d for d in deps if d.dep_type in self.config.filter_deps]

        # Track which symbols have connections
        connected = set()
        for dep in deps:
            connected.add(dep.source)
            connected.add(dep.target)

        # Cluster by file if enabled
        if self.config.cluster_by_file:
            by_file: Dict[str, List[Symbol]] = {}
            for sym in symbols:
                file_key = str(sym.location.file.name) if sym.location.file else "unknown"
                by_file.setdefault(file_key, []).append(sym)

            for file_name, file_symbols in by_file.items():
                cluster_id = _sanitize_id(file_name)
                lines.append(f"  subgraph cluster_{cluster_id} {{")
                lines.append(f'    label="{file_name}";')
                lines.append("    style=rounded;")
                lines.append("    color=gray;")

                for sym in file_symbols:
                    if self.config.include_orphans or sym.qualified_name in connected:
                        lines.append(self._render_node(sym))

                lines.append("  }")
                lines.append("")
        else:
            # Render all nodes
            for sym in symbols:
                if self.config.include_orphans or sym.qualified_name in connected:
                    lines.append(self._render_node(sym))

        lines.append("")

        # Render edges
        for dep in deps:
            lines.append(self._render_edge(dep))

        lines.append("}")
        return "\n".join(lines)

    def _render_node(self, sym: Symbol) -> str:
        """Render a single node."""
        node_id = _sanitize_id(sym.qualified_name)
        label = _truncate(sym.name, self.config.max_label_length)

        if self.config.show_types:
            label = f"{sym.node_type.value}\\n{label}"

        if self.config.show_visibility and sym.visibility.value != "unknown":
            label = f"[{sym.visibility.value[0]}] {label}"

        color = _get_color(sym.node_type, self.config.color_scheme)

        return f'  "{node_id}" [label="{label}", fillcolor="{color}"];'

    def _render_edge(self, dep: Dependency) -> str:
        """Render a single edge."""
        src_id = _sanitize_id(dep.source)
        tgt_id = _sanitize_id(dep.target)
        color = _get_edge_color(dep.dep_type, self.config.color_scheme)

        style = ""
        if dep.dep_type == DependencyType.IMPORT:
            style = ", style=dashed"
        elif dep.dep_type == DependencyType.INHERITANCE:
            style = ", penwidth=2, arrowhead=empty"
        elif dep.dep_type == DependencyType.IMPLEMENTATION:
            style = ", style=dashed, arrowhead=empty"
        elif dep.dep_type == DependencyType.TYPE_REFERENCE:
            style = ", style=dotted"

        return f'  "{src_id}" -> "{tgt_id}" [color="{color}"{style}];'


class MermaidRenderer:
    """Render dependency graphs to Mermaid format."""

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def render(self, graph: DependencyGraph) -> str:
        """Render graph to Mermaid format."""
        lines = []
        lines.append(f"graph {self.config.direction.value}")

        deps = graph.get_all_dependencies()
        if self.config.filter_deps:
            deps = [d for d in deps if d.dep_type in self.config.filter_deps]

        # Track rendered nodes
        rendered_nodes: Set[str] = set()

        for dep in deps:
            src_id = _sanitize_id(dep.source)
            tgt_id = _sanitize_id(dep.target)

            # Define node shapes if first time
            if src_id not in rendered_nodes:
                lines.append(self._node_def(dep.source, src_id))
                rendered_nodes.add(src_id)

            if tgt_id not in rendered_nodes:
                lines.append(self._node_def(dep.target, tgt_id))
                rendered_nodes.add(tgt_id)

            # Add edge
            lines.append(self._edge_def(dep, src_id, tgt_id))

        return "\n".join(lines)

    def _node_def(self, name: str, node_id: str) -> str:
        """Define a node."""
        label = _truncate(name.split(".")[-1], self.config.max_label_length)
        return f"    {node_id}[{label}]"

    def _edge_def(self, dep: Dependency, src_id: str, tgt_id: str) -> str:
        """Define an edge."""
        if dep.dep_type == DependencyType.INHERITANCE:
            return f"    {src_id} -->|extends| {tgt_id}"
        elif dep.dep_type == DependencyType.IMPLEMENTATION:
            return f"    {src_id} -.->|implements| {tgt_id}"
        elif dep.dep_type == DependencyType.IMPORT:
            return f"    {src_id} -.->|import| {tgt_id}"
        elif dep.dep_type == DependencyType.CALL:
            return f"    {src_id} -->|calls| {tgt_id}"
        else:
            return f"    {src_id} --> {tgt_id}"


class D3Renderer:
    """Render dependency graphs to D3.js-compatible JSON."""

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def render(self, graph: DependencyGraph) -> str:
        """Render graph to D3.js JSON format."""
        data = self.to_dict(graph)
        return json.dumps(data, indent=2)

    def to_dict(self, graph: DependencyGraph) -> Dict[str, Any]:
        """Convert graph to D3.js-compatible dict."""
        symbols = graph.get_symbols()
        deps = graph.get_all_dependencies()

        if self.config.filter_types:
            symbols = [s for s in symbols if s.node_type in self.config.filter_types]

        if self.config.filter_deps:
            deps = [d for d in deps if d.dep_type in self.config.filter_deps]

        # Build node list
        nodes = []
        node_index: Dict[str, int] = {}

        for i, sym in enumerate(symbols):
            node_index[sym.qualified_name] = i
            nodes.append({
                "id": sym.qualified_name,
                "name": sym.name,
                "type": sym.node_type.value,
                "visibility": sym.visibility.value,
                "file": str(sym.location.file) if sym.location.file else None,
                "line": sym.location.line,
                "color": _get_color(sym.node_type, self.config.color_scheme),
            })

        # Build link list
        links = []
        for dep in deps:
            if dep.source in node_index and dep.target in node_index:
                links.append({
                    "source": node_index[dep.source],
                    "target": node_index[dep.target],
                    "type": dep.dep_type.value,
                    "color": _get_edge_color(dep.dep_type, self.config.color_scheme),
                })

        return {
            "nodes": nodes,
            "links": links,
            "stats": {
                "node_count": len(nodes),
                "link_count": len(links),
            }
        }


class HTMLRenderer:
    """Render interactive HTML visualization using D3.js with package clustering."""

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def render(self, graph: DependencyGraph, title: str = "Dependency Graph") -> str:
        """Render graph to interactive HTML with advanced features."""
        d3_renderer = D3Renderer(self.config)
        graph_data = d3_renderer.to_dict(graph)

        # Add package information to nodes
        for node in graph_data["nodes"]:
            parts = node["id"].split(".")
            node["package"] = ".".join(parts[:-1]) if len(parts) > 1 else ""

        # Compute package statistics
        packages: Dict[str, int] = {}
        for node in graph_data["nodes"]:
            pkg = node["package"]
            packages[pkg] = packages.get(pkg, 0) + 1

        return self._render_advanced_html(graph_data, title, packages)

    def _render_advanced_html(
        self,
        graph_data: Dict[str, Any],
        title: str,
        packages: Dict[str, int]
    ) -> str:
        """Render advanced HTML with clustering and edge bundling."""
        # Generate package colors
        package_colors = {}
        pkg_list = list(packages.keys())
        for i, pkg in enumerate(pkg_list):
            hue = (i * 137.5) % 360  # Golden angle for good distribution
            package_colors[pkg] = f"hsl({hue}, 60%, 85%)"

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: {self.config.font_name}, -apple-system, BlinkMacSystemFont, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #graph-container {{
            flex: 1;
            position: relative;
            overflow: hidden;
        }}
        #graph {{
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }}
        #controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            display: flex;
            gap: 5px;
            z-index: 100;
        }}
        .control-btn {{
            padding: 8px 12px;
            border: none;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        .control-btn:hover {{
            background: rgba(255,255,255,0.2);
        }}
        #minimap {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            width: 200px;
            height: 150px;
            background: rgba(0,0,0,0.5);
            border: 1px solid #444;
            border-radius: 4px;
        }}
        #sidebar {{
            width: 320px;
            background: #16213e;
            border-left: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        #sidebar-header {{
            padding: 20px;
            background: #0f3460;
        }}
        #sidebar-header h2 {{
            margin: 0 0 15px 0;
            font-size: 18px;
            color: #e94560;
        }}
        #search {{
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 14px;
        }}
        #search::placeholder {{ color: #888; }}
        #sidebar-content {{
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }}
        .section {{
            margin-bottom: 20px;
        }}
        .section-title {{
            font-size: 12px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}
        #stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #e94560;
        }}
        .stat-label {{
            font-size: 11px;
            color: #888;
            margin-top: 4px;
        }}
        #node-info {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 6px;
            font-size: 13px;
        }}
        #node-info.empty {{
            color: #666;
            text-align: center;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .info-label {{ color: #888; }}
        .info-value {{ color: #fff; word-break: break-all; text-align: right; max-width: 60%; }}
        #filters {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .filter-chip {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.2s;
        }}
        .filter-chip.active {{
            border-color: white;
        }}
        .filter-chip:hover {{
            opacity: 0.8;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
            font-size: 12px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            margin-right: 8px;
            border-radius: 3px;
        }}
        #packages-list {{
            max-height: 200px;
            overflow-y: auto;
        }}
        .package-item {{
            display: flex;
            align-items: center;
            padding: 6px 8px;
            margin: 2px 0;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .package-item:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .package-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
        }}
        .package-count {{
            margin-left: auto;
            color: #888;
            min-width: 30px;
            text-align: right;
        }}
        .package-checkbox {{
            margin-right: 8px;
            cursor: pointer;
            accent-color: #e94560;
        }}
        .package-controls {{
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }}
        .package-sort-btn {{
            padding: 4px 8px;
            font-size: 11px;
            border: 1px solid #555;
            background: transparent;
            color: #ccc;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .package-sort-btn:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .package-sort-btn.active {{
            background: #e94560;
            border-color: #e94560;
            color: white;
        }}
        .package-select-all {{
            font-size: 11px;
            color: #888;
            cursor: pointer;
            margin-left: auto;
        }}
        .package-select-all:hover {{
            color: #e94560;
        }}
        .package-item.hidden-pkg {{
            opacity: 0.4;
        }}
        .package-connections {{
            font-size: 10px;
            color: #666;
            margin-left: 4px;
        }}
        .large-graph-message {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(15, 52, 96, 0.95);
            border: 2px solid #e94560;
            border-radius: 12px;
            padding: 24px 32px;
            text-align: center;
            color: #fff;
            z-index: 100;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            max-width: 400px;
        }}
        .large-graph-message strong {{
            color: #e94560;
            font-size: 18px;
        }}
        .large-graph-message em {{
            color: #888;
            display: block;
            margin-top: 12px;
        }}
        /* Package search */
        .package-search-container {{
            margin-bottom: 8px;
        }}
        .package-search {{
            width: 100%;
            padding: 8px 12px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 6px;
            color: #fff;
            font-size: 13px;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .package-search:focus {{
            outline: none;
            border-color: #e94560;
            box-shadow: 0 0 0 2px rgba(233, 69, 96, 0.2);
        }}
        .package-search::placeholder {{
            color: #888;
        }}
        .package-search-hint {{
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }}
        .package-match-count {{
            font-size: 11px;
            color: #e94560;
            margin-top: 4px;
            display: none;
        }}
        .package-item.search-hidden {{
            display: none !important;
        }}
        .package-item.search-match {{
            border-left: 2px solid #e94560;
            padding-left: 6px;
            margin-left: -8px;
        }}
        /* Graph elements */
        .cluster-hull {{
            stroke-width: 2;
            stroke-opacity: 0.6;
            fill-opacity: 0.15;
        }}
        .node {{
            cursor: pointer;
            stroke: rgba(255,255,255,0.3);
            stroke-width: 1.5px;
            transition: all 0.2s;
        }}
        .node:hover {{
            stroke: white;
            stroke-width: 2px;
        }}
        .node.selected {{
            stroke: #e94560;
            stroke-width: 3px;
        }}
        .node.dimmed {{
            opacity: 0.15;
        }}
        .link {{
            stroke-opacity: 0.4;
        }}
        .link.highlighted {{
            stroke-opacity: 1;
            stroke-width: 2px;
        }}
        .link.dimmed {{
            stroke-opacity: 0.05;
        }}
        .node-label {{
            font-size: 10px;
            fill: #ccc;
            pointer-events: none;
        }}
        .node-label.dimmed {{
            opacity: 0.1;
        }}
        /* Tooltip */
        .tooltip {{
            position: absolute;
            background: rgba(15, 52, 96, 0.95);
            border: 1px solid #e94560;
            border-radius: 6px;
            padding: 12px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .tooltip-title {{
            font-weight: bold;
            color: #e94560;
            margin-bottom: 8px;
        }}
        .tooltip-row {{
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
        }}
        .tooltip-label {{ color: #888; }}
        .tooltip-value {{ color: #fff; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <svg id="graph"></svg>
            <div id="controls">
                <button class="control-btn" onclick="zoomIn()">+</button>
                <button class="control-btn" onclick="zoomOut()">-</button>
                <button class="control-btn" onclick="resetZoom()">Reset</button>
                <button class="control-btn" onclick="fitToView()">Fit</button>
                <button class="control-btn" onclick="toggleClusters()">Clusters</button>
                <button class="control-btn" onclick="toggleEdgeBundling()">Bundle</button>
                <button class="control-btn" onclick="exportSVG()">Export</button>
            </div>
            <svg id="minimap"></svg>
        </div>
        <div id="sidebar">
            <div id="sidebar-header">
                <h2>{html.escape(title)}</h2>
                <input type="text" id="search" placeholder="Search symbols...">
            </div>
            <div id="sidebar-content">
                <div class="section">
                    <div class="section-title">Statistics</div>
                    <div id="stats">
                        <div class="stat-card">
                            <div class="stat-value">{graph_data["stats"]["node_count"]}</div>
                            <div class="stat-label">Nodes</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{graph_data["stats"]["link_count"]}</div>
                            <div class="stat-label">Dependencies</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{len(packages)}</div>
                            <div class="stat-label">Packages</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="visible-count">{graph_data["stats"]["node_count"]}</div>
                            <div class="stat-label">Visible</div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Node Types</div>
                    <div id="filters">
                        <span class="filter-chip active" data-type="class" style="background: {_get_color(NodeType.CLASS, self.config.color_scheme)}">Class</span>
                        <span class="filter-chip active" data-type="interface" style="background: {_get_color(NodeType.INTERFACE, self.config.color_scheme)}">Interface</span>
                        <span class="filter-chip active" data-type="method" style="background: {_get_color(NodeType.METHOD, self.config.color_scheme)}">Method</span>
                        <span class="filter-chip active" data-type="enum" style="background: {_get_color(NodeType.ENUM, self.config.color_scheme)}">Enum</span>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Dependency Types</div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-color" style="background: #3498db"></div> Inheritance</div>
                        <div class="legend-item"><div class="legend-color" style="background: #27ae60"></div> Implementation</div>
                        <div class="legend-item"><div class="legend-color" style="background: #2ecc71"></div> Call</div>
                        <div class="legend-item"><div class="legend-color" style="background: #999; border: 1px dashed #666"></div> Import</div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Selected Node</div>
                    <div id="node-info" class="empty">Click a node to see details</div>
                </div>

                <div class="section">
                    <div class="section-title">Packages ({len(packages)})</div>
                    <div class="package-search-container">
                        <input type="text" id="package-search" class="package-search"
                               placeholder="Search packages... (e.g. print|alert)"
                               oninput="filterPackages(this.value)">
                        <div class="package-search-hint">Use | for OR (e.g. util|common|helper)</div>
                        <div id="package-match-count" class="package-match-count"></div>
                    </div>
                    <div class="package-controls">
                        <button class="package-sort-btn active" data-sort="size" onclick="sortPackages('size')">By Size</button>
                        <button class="package-sort-btn" data-sort="name" onclick="sortPackages('name')">By Name</button>
                        <button class="package-sort-btn" data-sort="connections" onclick="sortPackages('connections')">By Connections</button>
                        <span class="package-select-all" onclick="toggleAllPackages()">Select All</span>
                    </div>
                    <div id="packages-list">
                        {self._render_package_list(packages, package_colors)}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip" style="display: none;"></div>

    <script>
        // Graph data
        const data = {json.dumps(graph_data)};
        const packageColors = {json.dumps(package_colors)};

        // State
        let showClusters = true;
        let edgeBundling = true;
        let selectedNode = null;
        let highlightedNodes = new Set();
        let activeTypes = new Set(['class', 'interface', 'method', 'enum', 'function', 'field']);
        let visiblePackages = new Set(Object.keys(packageColors));
        visiblePackages.add('');  // Include default package
        let currentSort = 'size';

        // Compute package connection counts
        const packageConnections = {{}};
        data.links.forEach(link => {{
            const srcPkg = data.nodes[link.source]?.package || '';
            const tgtPkg = data.nodes[link.target]?.package || '';
            packageConnections[srcPkg] = (packageConnections[srcPkg] || 0) + 1;
            packageConnections[tgtPkg] = (packageConnections[tgtPkg] || 0) + 1;
        }});

        // Large graph mode: start with all packages hidden
        const LARGE_GRAPH_THRESHOLD = 5000;
        const isLargeGraph = data.nodes.length > LARGE_GRAPH_THRESHOLD;
        if (isLargeGraph) {{
            visiblePackages.clear();
        }}

        // Dimensions
        const container = document.getElementById('graph-container');
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Create SVG
        const svg = d3.select('#graph')
            .attr('width', width)
            .attr('height', height);

        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', handleZoom);

        svg.call(zoom);

        // Main group
        const g = svg.append('g');

        // Layers
        const clustersLayer = g.append('g').attr('class', 'clusters-layer');
        const linksLayer = g.append('g').attr('class', 'links-layer');
        const nodesLayer = g.append('g').attr('class', 'nodes-layer');
        const labelsLayer = g.append('g').attr('class', 'labels-layer');

        // Arrow markers
        const defs = svg.append('defs');
        const edgeTypes = ['inheritance', 'implementation', 'call', 'import', 'composition', 'type_reference'];
        const edgeColors = {{
            'inheritance': '#3498db',
            'implementation': '#27ae60',
            'call': '#2ecc71',
            'import': '#999',
            'composition': '#9b59b6',
            'type_reference': '#95a5a6'
        }};

        edgeTypes.forEach(type => {{
            defs.append('marker')
                .attr('id', `arrow-${{type}}`)
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 20)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-5L10,0L0,5')
                .attr('fill', edgeColors[type] || '#999');
        }});

        // Build clusters
        const clusters = new Map();
        data.nodes.forEach(node => {{
            const pkg = node.package || '';
            if (!clusters.has(pkg)) {{
                clusters.set(pkg, {{
                    name: pkg,
                    nodes: [],
                    x: Math.random() * width,
                    y: Math.random() * height,
                    color: packageColors[pkg] || '#666'
                }});
            }}
            const cluster = clusters.get(pkg);
            cluster.nodes.push(node);
            node.cluster = cluster;
        }});

        // Create simulation
        const simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links)
                .id((d, i) => i)
                .distance(80))
            .force('charge', d3.forceManyBody()
                .strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(15))
            .force('cluster', clusterForce(0.3))
            .on('tick', tick);

        // Cluster force
        function clusterForce(strength) {{
            return function(alpha) {{
                data.nodes.forEach(node => {{
                    if (node.cluster) {{
                        node.vx -= (node.x - node.cluster.x) * strength * alpha;
                        node.vy -= (node.y - node.cluster.y) * strength * alpha;
                    }}
                }});
            }};
        }}

        // Render elements
        renderClusters();
        renderLinks();
        renderNodes();
        renderLabels();

        function renderClusters() {{
            const clusterData = Array.from(clusters.values()).filter(c => c.nodes.length > 1);

            clustersLayer.selectAll('.cluster-hull')
                .data(clusterData, d => d.name)
                .join('path')
                .attr('class', 'cluster-hull')
                .attr('fill', d => d.color)
                .attr('stroke', d => d3.color(d.color).darker(0.5))
                .style('display', showClusters ? 'block' : 'none');
        }}

        function renderLinks() {{
            linksLayer.selectAll('.link')
                .data(data.links)
                .join('path')
                .attr('class', 'link')
                .attr('stroke', d => edgeColors[d.type] || '#999')
                .attr('stroke-width', d => d.type === 'inheritance' ? 2 : 1)
                .attr('stroke-dasharray', d => {{
                    if (d.type === 'import') return '4,4';
                    if (d.type === 'implementation') return '2,2';
                    return null;
                }})
                .attr('fill', 'none')
                .attr('marker-end', d => `url(#arrow-${{d.type || 'call'}})`);
        }}

        function renderNodes() {{
            nodesLayer.selectAll('.node')
                .data(data.nodes)
                .join('circle')
                .attr('class', 'node')
                .attr('r', 8)
                .attr('fill', d => d.color)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended))
                .on('click', selectNode)
                .on('mouseover', showTooltip)
                .on('mouseout', hideTooltip);
        }}

        function renderLabels() {{
            labelsLayer.selectAll('.node-label')
                .data(data.nodes)
                .join('text')
                .attr('class', 'node-label')
                .text(d => d.name.length > 18 ? d.name.slice(0, 18) + '...' : d.name)
                .attr('dx', 12)
                .attr('dy', 4);
        }}

        function tick() {{
            // Update cluster centers
            clusters.forEach(cluster => {{
                cluster.x = d3.mean(cluster.nodes, n => n.x) || cluster.x;
                cluster.y = d3.mean(cluster.nodes, n => n.y) || cluster.y;
            }});

            // Update cluster hulls
            if (showClusters) {{
                clustersLayer.selectAll('.cluster-hull')
                    .attr('d', d => {{
                        const points = d.nodes.map(n => [n.x, n.y]);
                        if (points.length < 3) {{
                            const cx = d3.mean(points, p => p[0]) || 0;
                            const cy = d3.mean(points, p => p[1]) || 0;
                            return `M${{cx-25}},${{cy}} a25,25 0 1,0 50,0 a25,25 0 1,0 -50,0`;
                        }}
                        const hull = d3.polygonHull(points);
                        if (!hull) return '';
                        // Expand hull slightly
                        const centroid = d3.polygonCentroid(hull);
                        const expanded = hull.map(p => [
                            p[0] + (p[0] - centroid[0]) * 0.2,
                            p[1] + (p[1] - centroid[1]) * 0.2
                        ]);
                        return 'M' + expanded.map(p => p.join(',')).join('L') + 'Z';
                    }});
            }}

            // Update links
            linksLayer.selectAll('.link')
                .attr('d', d => {{
                    if (edgeBundling && d.source.cluster !== d.target.cluster) {{
                        // Curve through center for bundling effect
                        const mx = (d.source.x + d.target.x) / 2;
                        const my = (d.source.y + d.target.y) / 2;
                        const cx = width / 2;
                        const cy = height / 2;
                        const controlX = mx + (cx - mx) * 0.2;
                        const controlY = my + (cy - my) * 0.2;
                        return `M${{d.source.x}},${{d.source.y}}Q${{controlX}},${{controlY}} ${{d.target.x}},${{d.target.y}}`;
                    }}
                    return `M${{d.source.x}},${{d.source.y}}L${{d.target.x}},${{d.target.y}}`;
                }});

            // Update nodes
            nodesLayer.selectAll('.node')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            // Update labels
            labelsLayer.selectAll('.node-label')
                .attr('x', d => d.x)
                .attr('y', d => d.y);

            // Update minimap
            updateMinimap();
        }}

        // Drag handlers
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        // Zoom handlers
        function handleZoom(event) {{
            g.attr('transform', event.transform);
            updateMinimapViewport(event.transform);
        }}

        function zoomIn() {{
            svg.transition().call(zoom.scaleBy, 1.3);
        }}

        function zoomOut() {{
            svg.transition().call(zoom.scaleBy, 0.7);
        }}

        function resetZoom() {{
            svg.transition().call(zoom.transform, d3.zoomIdentity);
        }}

        function fitToView() {{
            if (data.nodes.length === 0) return;
            const bounds = {{
                minX: d3.min(data.nodes, d => d.x) - 50,
                maxX: d3.max(data.nodes, d => d.x) + 50,
                minY: d3.min(data.nodes, d => d.y) - 50,
                maxY: d3.max(data.nodes, d => d.y) + 50
            }};
            const bWidth = bounds.maxX - bounds.minX;
            const bHeight = bounds.maxY - bounds.minY;
            const scale = 0.9 / Math.max(bWidth / width, bHeight / height);
            const tx = (width - scale * (bounds.minX + bounds.maxX)) / 2;
            const ty = (height - scale * (bounds.minY + bounds.maxY)) / 2;
            svg.transition().duration(750)
                .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
        }}

        // Toggle functions
        function toggleClusters() {{
            showClusters = !showClusters;
            clustersLayer.selectAll('.cluster-hull')
                .style('display', showClusters ? 'block' : 'none');
        }}

        function toggleEdgeBundling() {{
            edgeBundling = !edgeBundling;
            simulation.alpha(0.3).restart();
        }}

        // Package filtering functions
        function togglePackage(pkg) {{
            const checkbox = document.querySelector(`input[data-package="${{pkg}}"]`);
            const item = checkbox.closest('.package-item');
            if (visiblePackages.has(pkg)) {{
                visiblePackages.delete(pkg);
                item.classList.add('hidden-pkg');
                checkbox.checked = false;
            }} else {{
                visiblePackages.add(pkg);
                item.classList.remove('hidden-pkg');
                checkbox.checked = true;
            }}
            // Remove large graph message when user starts selecting
            const msg = document.querySelector('.large-graph-message');
            if (msg) msg.remove();
            updatePackageVisibility();
        }}

        function toggleAllPackages() {{
            const selectAllSpan = document.querySelector('.package-select-all');
            const shouldHide = selectAllSpan.textContent === 'Select None';
            // Remove large graph message
            const msg = document.querySelector('.large-graph-message');
            if (msg) msg.remove();

            if (shouldHide) {{
                // Hide all
                visiblePackages.clear();
                document.querySelectorAll('.package-checkbox').forEach(cb => {{
                    cb.checked = false;
                    cb.closest('.package-item').classList.add('hidden-pkg');
                }});
                selectAllSpan.textContent = 'Select All';
            }} else {{
                // Show all
                visiblePackages = new Set(Object.keys(packageColors));
                visiblePackages.add('');
                document.querySelectorAll('.package-checkbox').forEach(cb => {{
                    cb.checked = true;
                    cb.closest('.package-item').classList.remove('hidden-pkg');
                }});
                selectAllSpan.textContent = 'Select None';
            }}
            updatePackageVisibility();
        }}

        function updatePackageVisibility() {{
            // Update nodes visibility
            nodesLayer.selectAll('.node')
                .style('display', d => visiblePackages.has(d.package) ? 'block' : 'none');

            // Update labels visibility
            labelsLayer.selectAll('.node-label')
                .style('display', d => visiblePackages.has(d.package) ? 'block' : 'none');

            // Update links - hide if either end is hidden
            linksLayer.selectAll('.link')
                .style('display', d => {{
                    const srcNode = typeof d.source === 'object' ? d.source : data.nodes[d.source];
                    const tgtNode = typeof d.target === 'object' ? d.target : data.nodes[d.target];
                    return visiblePackages.has(srcNode.package) && visiblePackages.has(tgtNode.package) ? 'block' : 'none';
                }});

            // Update cluster hulls
            clustersLayer.selectAll('.cluster-hull')
                .style('display', d => visiblePackages.has(d.name) ? 'block' : 'none');

            // Update visible count
            const visibleCount = data.nodes.filter(n => visiblePackages.has(n.package)).length;
            document.getElementById('visible-count').textContent = visibleCount;
        }}

        function sortPackages(sortBy) {{
            currentSort = sortBy;

            // Update button styles
            document.querySelectorAll('.package-sort-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.sort === sortBy);
            }});

            // Get package data
            const packagesList = document.getElementById('packages-list');
            const items = Array.from(packagesList.querySelectorAll('.package-item'));

            // Sort items
            items.sort((a, b) => {{
                const pkgA = a.dataset.package;
                const pkgB = b.dataset.package;
                const countA = parseInt(a.querySelector('.package-count').textContent) || 0;
                const countB = parseInt(b.querySelector('.package-count').textContent) || 0;
                const connA = packageConnections[pkgA] || 0;
                const connB = packageConnections[pkgB] || 0;

                if (sortBy === 'name') {{
                    const nameA = pkgA || '(default)';
                    const nameB = pkgB || '(default)';
                    return nameA.localeCompare(nameB);
                }} else if (sortBy === 'connections') {{
                    return connB - connA;  // Descending
                }} else {{  // size
                    return countB - countA;  // Descending
                }}
            }});

            // Reorder DOM
            items.forEach(item => packagesList.appendChild(item));
        }}

        function filterPackages(query) {{
            const packagesList = document.getElementById('packages-list');
            const items = packagesList.querySelectorAll('.package-item');
            const matchCountEl = document.getElementById('package-match-count');

            // Trim and handle empty query
            query = query.trim().toLowerCase();

            if (!query) {{
                // Show all packages, remove highlighting
                items.forEach(item => {{
                    item.classList.remove('search-hidden', 'search-match');
                }});
                matchCountEl.style.display = 'none';
                return;
            }}

            // Split by | for OR search
            const terms = query.split('|').map(t => t.trim()).filter(t => t.length > 0);

            let matchCount = 0;
            items.forEach(item => {{
                const pkg = item.dataset.package || '(default)';
                const pkgLower = pkg.toLowerCase();

                // Check if any term matches
                const matches = terms.some(term => pkgLower.includes(term));

                if (matches) {{
                    item.classList.remove('search-hidden');
                    item.classList.add('search-match');
                    matchCount++;
                }} else {{
                    item.classList.add('search-hidden');
                    item.classList.remove('search-match');
                }}
            }});

            // Show match count
            matchCountEl.textContent = `${{matchCount}} package${{matchCount !== 1 ? 's' : ''}} found`;
            matchCountEl.style.display = 'block';
        }}

        // Selection
        function selectNode(event, d) {{
            event.stopPropagation();

            if (selectedNode === d) {{
                selectedNode = null;
                highlightedNodes.clear();
            }} else {{
                selectedNode = d;
                highlightConnectedNodes(d);
            }}

            updateHighlighting();
            updateNodeInfo(d);
        }}

        function highlightConnectedNodes(node) {{
            highlightedNodes.clear();
            highlightedNodes.add(node.id);

            data.links.forEach(link => {{
                const sourceId = typeof link.source === 'object' ? link.source.id : data.nodes[link.source].id;
                const targetId = typeof link.target === 'object' ? link.target.id : data.nodes[link.target].id;

                if (sourceId === node.id) highlightedNodes.add(targetId);
                if (targetId === node.id) highlightedNodes.add(sourceId);
            }});
        }}

        function updateHighlighting() {{
            const hasSelection = highlightedNodes.size > 0;

            nodesLayer.selectAll('.node')
                .classed('selected', d => d === selectedNode)
                .classed('dimmed', d => hasSelection && !highlightedNodes.has(d.id));

            linksLayer.selectAll('.link')
                .classed('highlighted', d => {{
                    if (!hasSelection) return false;
                    const sourceId = typeof d.source === 'object' ? d.source.id : data.nodes[d.source].id;
                    const targetId = typeof d.target === 'object' ? d.target.id : data.nodes[d.target].id;
                    return highlightedNodes.has(sourceId) && highlightedNodes.has(targetId);
                }})
                .classed('dimmed', d => {{
                    if (!hasSelection) return false;
                    const sourceId = typeof d.source === 'object' ? d.source.id : data.nodes[d.source].id;
                    const targetId = typeof d.target === 'object' ? d.target.id : data.nodes[d.target].id;
                    return !(highlightedNodes.has(sourceId) && highlightedNodes.has(targetId));
                }});

            labelsLayer.selectAll('.node-label')
                .classed('dimmed', d => hasSelection && !highlightedNodes.has(d.id));
        }}

        function updateNodeInfo(node) {{
            const info = document.getElementById('node-info');
            if (!node || selectedNode === null) {{
                info.innerHTML = 'Click a node to see details';
                info.className = 'empty';
                return;
            }}

            const inDeps = data.links.filter(l => {{
                const targetId = typeof l.target === 'object' ? l.target.id : data.nodes[l.target].id;
                return targetId === node.id;
            }}).length;

            const outDeps = data.links.filter(l => {{
                const sourceId = typeof l.source === 'object' ? l.source.id : data.nodes[l.source].id;
                return sourceId === node.id;
            }}).length;

            info.className = '';
            info.innerHTML = `
                <div class="info-row"><span class="info-label">Name</span><span class="info-value">${{node.name}}</span></div>
                <div class="info-row"><span class="info-label">Type</span><span class="info-value">${{node.type}}</span></div>
                <div class="info-row"><span class="info-label">Visibility</span><span class="info-value">${{node.visibility}}</span></div>
                <div class="info-row"><span class="info-label">Package</span><span class="info-value">${{node.package || 'N/A'}}</span></div>
                <div class="info-row"><span class="info-label">In deps</span><span class="info-value">${{inDeps}}</span></div>
                <div class="info-row"><span class="info-label">Out deps</span><span class="info-value">${{outDeps}}</span></div>
                <div class="info-row"><span class="info-label">File</span><span class="info-value">${{node.file ? node.file.split('/').pop() : 'N/A'}}</span></div>
                <div class="info-row"><span class="info-label">Line</span><span class="info-value">${{node.line || 'N/A'}}</span></div>
            `;
        }}

        // Tooltip
        function showTooltip(event, d) {{
            const tooltip = document.getElementById('tooltip');
            const inDeps = data.links.filter(l => {{
                const targetId = typeof l.target === 'object' ? l.target.id : data.nodes[l.target].id;
                return targetId === d.id;
            }}).length;
            const outDeps = data.links.filter(l => {{
                const sourceId = typeof l.source === 'object' ? l.source.id : data.nodes[l.source].id;
                return sourceId === d.id;
            }}).length;

            tooltip.innerHTML = `
                <div class="tooltip-title">${{d.name}}</div>
                <div class="tooltip-row"><span class="tooltip-label">Type:</span><span class="tooltip-value">${{d.type}}</span></div>
                <div class="tooltip-row"><span class="tooltip-label">Dependencies:</span><span class="tooltip-value">In: ${{inDeps}} | Out: ${{outDeps}}</span></div>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
        }}

        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        // Search
        document.getElementById('search').addEventListener('input', function(e) {{
            const term = e.target.value.toLowerCase();
            let visibleCount = 0;

            nodesLayer.selectAll('.node')
                .style('opacity', d => {{
                    const visible = d.name.toLowerCase().includes(term) || d.id.toLowerCase().includes(term);
                    if (visible) visibleCount++;
                    return visible || term === '' ? 1 : 0.1;
                }});

            labelsLayer.selectAll('.node-label')
                .style('opacity', d => {{
                    const visible = d.name.toLowerCase().includes(term) || d.id.toLowerCase().includes(term);
                    return visible || term === '' ? 1 : 0.1;
                }});

            document.getElementById('visible-count').textContent =
                term === '' ? data.nodes.length : visibleCount;
        }});

        // Filter chips
        document.querySelectorAll('.filter-chip').forEach(chip => {{
            chip.addEventListener('click', function() {{
                this.classList.toggle('active');
                const type = this.dataset.type;
                if (this.classList.contains('active')) {{
                    activeTypes.add(type);
                }} else {{
                    activeTypes.delete(type);
                }}
                applyFilters();
            }});
        }});

        function applyFilters() {{
            let visibleCount = 0;

            nodesLayer.selectAll('.node')
                .style('display', d => {{
                    const visible = activeTypes.has(d.type);
                    if (visible) visibleCount++;
                    return visible ? 'block' : 'none';
                }});

            labelsLayer.selectAll('.node-label')
                .style('display', d => activeTypes.has(d.type) ? 'block' : 'none');

            document.getElementById('visible-count').textContent = visibleCount;
        }}

        // Package click handling
        document.querySelectorAll('.package-item').forEach(item => {{
            item.addEventListener('click', function() {{
                const pkg = this.dataset.package;
                const cluster = clusters.get(pkg);
                if (cluster) {{
                    // Center view on this cluster
                    const cx = d3.mean(cluster.nodes, n => n.x);
                    const cy = d3.mean(cluster.nodes, n => n.y);
                    svg.transition().duration(750)
                        .call(zoom.transform, d3.zoomIdentity
                            .translate(width / 2 - cx, height / 2 - cy));
                }}
            }});
        }});

        // Minimap
        const minimapSvg = d3.select('#minimap');
        const minimapWidth = 200;
        const minimapHeight = 150;
        const minimapScale = Math.min(minimapWidth / width, minimapHeight / height) * 0.8;

        minimapSvg
            .attr('width', minimapWidth)
            .attr('height', minimapHeight);

        const minimapG = minimapSvg.append('g')
            .attr('transform', `translate(${{minimapWidth/2}}, ${{minimapHeight/2}}) scale(${{minimapScale}}) translate(${{-width/2}}, ${{-height/2}})`);

        const minimapNodes = minimapG.append('g');
        const minimapViewport = minimapSvg.append('rect')
            .attr('class', 'minimap-viewport')
            .attr('fill', 'rgba(233, 69, 96, 0.2)')
            .attr('stroke', '#e94560')
            .attr('stroke-width', 2);

        function updateMinimap() {{
            minimapNodes.selectAll('circle')
                .data(data.nodes)
                .join('circle')
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', 3)
                .attr('fill', d => d.color);
        }}

        function updateMinimapViewport(transform) {{
            const x = (-transform.x / transform.k) * minimapScale + minimapWidth / 2 - (width / 2) * minimapScale;
            const y = (-transform.y / transform.k) * minimapScale + minimapHeight / 2 - (height / 2) * minimapScale;
            const w = (width / transform.k) * minimapScale;
            const h = (height / transform.k) * minimapScale;

            minimapViewport
                .attr('x', x)
                .attr('y', y)
                .attr('width', w)
                .attr('height', h);
        }}

        // Export
        function exportSVG() {{
            const svgData = new XMLSerializer().serializeToString(svg.node());
            const blob = new Blob([svgData], {{ type: 'image/svg+xml;charset=utf-8' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'dependency-graph.svg';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // Clear selection on background click
        svg.on('click', function() {{
            selectedNode = null;
            highlightedNodes.clear();
            updateHighlighting();
            updateNodeInfo(null);
        }});

        // Initial minimap viewport
        updateMinimapViewport(d3.zoomIdentity);

        // Large graph mode initialization (after D3 is fully set up)
        if (isLargeGraph) {{
            // Hide all SVG elements
            updatePackageVisibility();

            // Update UI: uncheck all checkboxes and add hidden class
            document.querySelectorAll('.package-checkbox').forEach(cb => {{
                cb.checked = false;
                cb.closest('.package-item').classList.add('hidden-pkg');
            }});

            // Update Select All/None button text
            const selectAllSpan = document.querySelector('.package-select-all');
            if (selectAllSpan) {{
                selectAllSpan.textContent = 'Select All';
            }}

            // Update visible count to 0
            document.getElementById('visible-count').textContent = '0';

            // Show informational message
            const graphContainer = document.getElementById('graph-container');
            const message = document.createElement('div');
            message.className = 'large-graph-message';
            message.innerHTML = `
                <h3 style="margin: 0 0 12px 0; color: #e94560;">Large Graph Mode</h3>
                <p style="margin: 0 0 8px 0;">This graph has <strong>${{data.nodes.length.toLocaleString()}}</strong> nodes.</p>
                <p style="margin: 0; font-size: 13px; color: #aaa;">
                    All packages are hidden by default.<br>
                    Select specific packages from the sidebar to explore.
                </p>
            `;
            graphContainer.appendChild(message);
        }}
    </script>
</body>
</html>'''

    def _render_package_list(self, packages: Dict[str, int], colors: Dict[str, str]) -> str:
        """Render package list HTML with checkboxes for filtering."""
        items = []
        sorted_pkgs = sorted(packages.items(), key=lambda x: -x[1])
        for pkg, count in sorted_pkgs[:50]:  # Increased limit to 50
            display_name = pkg if pkg else "(default)"
            color = colors.get(pkg, "#666")
            escaped_pkg = html.escape(pkg)
            items.append(
                f'<div class="package-item" data-package="{escaped_pkg}">'
                f'<input type="checkbox" class="package-checkbox" '
                f'data-package="{escaped_pkg}" checked '
                f'onclick="togglePackage(\'{escaped_pkg.replace(chr(39), chr(92)+chr(39))}\')">'
                f'<div class="package-color" style="background: {color}"></div>'
                f'<span title="{html.escape(display_name)}">{html.escape(display_name[:35])}</span>'
                f'<span class="package-count">{count}</span>'
                f'</div>'
            )
        return "\n".join(items)


class DSMRenderer:
    """Render Dependency Structure Matrix (DSM) visualization."""

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def to_matrix(self, graph: DependencyGraph, level: str = "class") -> Dict[str, Any]:
        """
        Convert dependency graph to matrix format.

        Args:
            graph: Dependency graph
            level: Aggregation level ('class', 'package', 'file')

        Returns:
            Dictionary with matrix data
        """
        symbols = graph.get_symbols()
        deps = graph.get_all_dependencies()

        # Filter to classes/interfaces only for cleaner matrix
        if level == "class":
            symbols = [s for s in symbols if s.node_type in
                      (NodeType.CLASS, NodeType.INTERFACE, NodeType.ENUM)]
        elif level == "package":
            # Aggregate by package
            packages: Dict[str, Set[str]] = {}
            for sym in symbols:
                parts = sym.qualified_name.split(".")
                pkg = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
                packages.setdefault(pkg, set()).add(sym.qualified_name)

            # Create package-level symbols
            pkg_symbols = []
            for pkg in sorted(packages.keys()):
                pkg_symbols.append({
                    "id": pkg,
                    "name": pkg.split(".")[-1] if pkg else "(default)",
                    "full_name": pkg,
                    "count": len(packages[pkg])
                })
            symbols = pkg_symbols

        # Build adjacency matrix
        sym_ids = [s.qualified_name if hasattr(s, "qualified_name") else s["id"]
                   for s in symbols]
        sym_index = {sid: i for i, sid in enumerate(sym_ids)}
        n = len(sym_ids)
        matrix = [[0] * n for _ in range(n)]

        # Count dependencies
        for dep in deps:
            if level == "package":
                # Aggregate to package level
                src_parts = dep.source.split(".")
                tgt_parts = dep.target.split(".")
                src_pkg = ".".join(src_parts[:-1]) if len(src_parts) > 1 else src_parts[0]
                tgt_pkg = ".".join(tgt_parts[:-1]) if len(tgt_parts) > 1 else tgt_parts[0]

                if src_pkg in sym_index and tgt_pkg in sym_index:
                    i, j = sym_index[src_pkg], sym_index[tgt_pkg]
                    if i != j:
                        matrix[i][j] += 1
            else:
                if dep.source in sym_index and dep.target in sym_index:
                    i, j = sym_index[dep.source], sym_index[dep.target]
                    matrix[i][j] += 1

        # Find cycles (cells above diagonal with corresponding below-diagonal cells)
        cycles = []
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] > 0 and matrix[j][i] > 0:
                    cycles.append((i, j))

        return {
            "labels": [s.name if hasattr(s, "name") else s["name"] for s in symbols],
            "full_labels": sym_ids,
            "matrix": matrix,
            "cycles": cycles,
            "size": n,
            "total_deps": sum(sum(row) for row in matrix),
        }

    def render_html(self, graph: DependencyGraph, title: str = "Dependency Matrix",
                   level: str = "class") -> str:
        """Render DSM as interactive HTML."""
        data = self.to_matrix(graph, level)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: {self.config.font_name}, -apple-system, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #matrix-container {{
            flex: 1;
            overflow: auto;
            padding: 20px;
        }}
        #sidebar {{
            width: 300px;
            background: #16213e;
            border-left: 1px solid #0f3460;
            padding: 20px;
            overflow-y: auto;
        }}
        h2 {{
            margin: 0 0 20px 0;
            color: #e94560;
            font-size: 18px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .stat {{
            background: rgba(255,255,255,0.05);
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #e94560;
        }}
        .stat-label {{
            font-size: 11px;
            color: #888;
            margin-top: 4px;
        }}
        #matrix {{
            display: inline-block;
        }}
        .matrix-row {{
            display: flex;
        }}
        .matrix-cell {{
            width: 24px;
            height: 24px;
            border: 1px solid #2a2a4e;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .matrix-cell:hover {{
            transform: scale(1.2);
            z-index: 100;
            border-color: white;
        }}
        .matrix-cell.diagonal {{
            background: #333;
        }}
        .matrix-cell.has-value {{
            background: #3498db;
            color: white;
        }}
        .matrix-cell.cycle {{
            background: #e74c3c !important;
        }}
        .matrix-cell.high {{
            background: #1abc9c;
        }}
        .row-label, .col-label {{
            font-size: 10px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .row-label {{
            width: 120px;
            text-align: right;
            padding-right: 8px;
            line-height: 24px;
        }}
        .col-labels {{
            display: flex;
            margin-left: 120px;
        }}
        .col-label {{
            width: 24px;
            height: 100px;
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            padding-bottom: 8px;
        }}
        #cell-info {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 6px;
            font-size: 13px;
        }}
        #cell-info.empty {{
            color: #666;
            text-align: center;
        }}
        .info-row {{
            margin: 8px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .info-label {{
            color: #888;
            font-size: 11px;
        }}
        .info-value {{
            color: #fff;
            margin-top: 4px;
            word-break: break-all;
        }}
        .legend {{
            margin-top: 20px;
        }}
        .legend-title {{
            font-size: 12px;
            color: #888;
            margin-bottom: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
            font-size: 12px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 3px;
        }}
        .controls {{
            margin-bottom: 20px;
        }}
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            cursor: pointer;
            margin-right: 8px;
            font-size: 12px;
        }}
        .btn:hover {{
            background: rgba(255,255,255,0.2);
        }}
        .btn.active {{
            background: #e94560;
        }}
        .tooltip {{
            position: fixed;
            background: rgba(15, 52, 96, 0.95);
            border: 1px solid #e94560;
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="matrix-container">
            <div class="controls">
                <button class="btn" onclick="exportCSV()">Export CSV</button>
                <button class="btn" onclick="highlightCycles()">Show Cycles</button>
            </div>
            <div id="matrix-wrapper"></div>
        </div>
        <div id="sidebar">
            <h2>{html.escape(title)}</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{data["size"]}</div>
                    <div class="stat-label">Elements</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{data["total_deps"]}</div>
                    <div class="stat-label">Dependencies</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(data["cycles"])}</div>
                    <div class="stat-label">Cycles</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{round(data["total_deps"] / max(data["size"], 1), 1)}</div>
                    <div class="stat-label">Avg Deps</div>
                </div>
            </div>
            <h3 style="font-size: 14px; margin: 15px 0 10px;">Cell Info</h3>
            <div id="cell-info" class="empty">Hover over a cell</div>
            <div class="legend">
                <div class="legend-title">LEGEND</div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #333"></div> Diagonal (self)
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #3498db"></div> Has dependency
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #1abc9c"></div> High coupling (5+)
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #e74c3c"></div> Cyclic dependency
                </div>
            </div>
        </div>
    </div>
    <div class="tooltip" id="tooltip" style="display: none;"></div>
    <script>
        const data = {json.dumps(data)};
        let showCyclesOnly = false;

        function renderMatrix() {{
            const wrapper = document.getElementById('matrix-wrapper');
            let html = '';

            // Column labels
            html += '<div class="col-labels">';
            for (let j = 0; j < data.size; j++) {{
                html += `<div class="col-label" title="${{data.full_labels[j]}}">${{data.labels[j]}}</div>`;
            }}
            html += '</div>';

            // Matrix rows
            html += '<div id="matrix">';
            for (let i = 0; i < data.size; i++) {{
                html += '<div class="matrix-row">';
                html += `<div class="row-label" title="${{data.full_labels[i]}}">${{data.labels[i]}}</div>`;
                for (let j = 0; j < data.size; j++) {{
                    const value = data.matrix[i][j];
                    const isCycle = data.cycles.some(c => (c[0] === i && c[1] === j) || (c[0] === j && c[1] === i));
                    const isDiagonal = i === j;
                    const isHigh = value >= 5;

                    let classes = 'matrix-cell';
                    if (isDiagonal) classes += ' diagonal';
                    else if (isCycle) classes += ' cycle';
                    else if (isHigh) classes += ' high';
                    else if (value > 0) classes += ' has-value';

                    const display = isDiagonal ? '' : (value || '');
                    html += `<div class="${{classes}}" data-row="${{i}}" data-col="${{j}}" data-value="${{value}}"
                             onmouseover="showCellInfo(event, ${{i}}, ${{j}}, ${{value}})"
                             onmouseout="hideCellInfo()">${{display}}</div>`;
                }}
                html += '</div>';
            }}
            html += '</div>';

            wrapper.innerHTML = html;
        }}

        function showCellInfo(event, row, col, value) {{
            const info = document.getElementById('cell-info');
            const isCycle = data.cycles.some(c => (c[0] === row && c[1] === col) || (c[0] === col && c[1] === row));

            if (row === col) {{
                info.innerHTML = `
                    <div class="info-row">
                        <div class="info-label">Element</div>
                        <div class="info-value">${{data.labels[row]}}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Full Name</div>
                        <div class="info-value">${{data.full_labels[row]}}</div>
                    </div>
                `;
            }} else {{
                info.innerHTML = `
                    <div class="info-row">
                        <div class="info-label">From</div>
                        <div class="info-value">${{data.labels[row]}}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">To</div>
                        <div class="info-value">${{data.labels[col]}}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Dependencies</div>
                        <div class="info-value">${{value}}${{isCycle ? ' (CYCLIC!)' : ''}}</div>
                    </div>
                `;
            }}
            info.className = '';

            // Show tooltip
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = `${{data.labels[row]}} → ${{data.labels[col]}}: ${{value}}`;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.clientX + 15) + 'px';
            tooltip.style.top = (event.clientY - 10) + 'px';
        }}

        function hideCellInfo() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        function highlightCycles() {{
            showCyclesOnly = !showCyclesOnly;
            const cells = document.querySelectorAll('.matrix-cell');
            cells.forEach(cell => {{
                if (showCyclesOnly && !cell.classList.contains('cycle') && !cell.classList.contains('diagonal')) {{
                    cell.style.opacity = '0.2';
                }} else {{
                    cell.style.opacity = '1';
                }}
            }});
        }}

        function exportCSV() {{
            let csv = ',' + data.labels.join(',') + '\\n';
            for (let i = 0; i < data.size; i++) {{
                csv += data.labels[i] + ',' + data.matrix[i].join(',') + '\\n';
            }}
            const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'dependency-matrix.csv';
            a.click();
            URL.revokeObjectURL(url);
        }}

        renderMatrix();
    </script>
</body>
</html>'''

    def to_csv(self, graph: DependencyGraph, level: str = "class") -> str:
        """Export DSM as CSV."""
        data = self.to_matrix(graph, level)
        lines = []

        # Header
        lines.append("," + ",".join(data["labels"]))

        # Rows
        for i, label in enumerate(data["labels"]):
            row_values = [str(v) for v in data["matrix"][i]]
            lines.append(f"{label},{','.join(row_values)}")

        return "\n".join(lines)


class RadialExplorer:
    """
    Radial dependency explorer - ego-centric visualization.

    Shows a focal node at center with dependencies radiating outward
    in concentric circles by level (1st degree, 2nd degree, etc.).
    """

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def build_ego_graph(
        self,
        graph: DependencyGraph,
        focal_node: str,
        max_levels: int = 3
    ) -> Dict[str, Any]:
        """
        Build ego-centric graph data from a focal node.

        Args:
            graph: Full dependency graph
            focal_node: Qualified name of the center node
            max_levels: Maximum depth to explore

        Returns:
            Dict with nodes organized by level and links
        """
        from .dependencies import DependencyType

        symbols = {s.qualified_name: s for s in graph.get_symbols()}
        all_deps = graph.get_all_dependencies()

        # Build short name to qualified name mapping
        short_to_qualified: Dict[str, str] = {}
        for qname in symbols:
            short = qname.split('.')[-1]
            # Prefer shorter qualified names (avoid methods)
            if short not in short_to_qualified or len(qname) < len(short_to_qualified[short]):
                short_to_qualified[short] = qname

        def resolve_name(name: str) -> Optional[str]:
            """Resolve a name to its qualified form if possible."""
            if name in symbols:
                return name
            if name in short_to_qualified:
                return short_to_qualified[name]
            # Try extracting class part from method ref (Class.Class.method -> Class)
            parts = name.split('.')
            if len(parts) >= 2:
                class_name = parts[0]
                if class_name in short_to_qualified:
                    return short_to_qualified[class_name]
            return None

        # Skip import dependencies - focus on structural relationships
        meaningful_dep_types = {
            DependencyType.INHERITANCE,
            DependencyType.IMPLEMENTATION,
            DependencyType.COMPOSITION,
            DependencyType.TYPE_REFERENCE,
            DependencyType.CALL,
            DependencyType.INSTANTIATION,
        }

        # Build adjacency lists (both directions), filtering for meaningful deps
        # Resolve names to qualified forms
        outgoing: Dict[str, List[Tuple[str, Dependency]]] = {}
        incoming: Dict[str, List[Tuple[str, Dependency]]] = {}
        for dep in all_deps:
            if dep.dep_type in meaningful_dep_types:
                resolved_source = resolve_name(dep.source)
                resolved_target = resolve_name(dep.target)
                if resolved_source:
                    outgoing.setdefault(resolved_source, []).append((resolved_target or dep.target, dep))
                if resolved_target:
                    incoming.setdefault(resolved_target, []).append((resolved_source or dep.source, dep))

        # BFS to find nodes at each level
        # Only include nodes that exist in symbols (actual code entities)
        levels: Dict[int, Set[str]] = {0: {focal_node}}
        visited = {focal_node}
        links = []

        for level in range(1, max_levels + 1):
            levels[level] = set()
            for node_id in levels[level - 1]:
                # Outgoing dependencies (things this node depends on)
                for target, dep in outgoing.get(node_id, []):
                    resolved = resolve_name(target)
                    if resolved and resolved in symbols and resolved not in visited:
                        levels[level].add(resolved)
                        visited.add(resolved)
                    if resolved and resolved in visited:
                        links.append({
                            "source": node_id,
                            "target": resolved,
                            "type": dep.dep_type.value,
                            "direction": "out"
                        })

                # Incoming dependencies (things that depend on this node)
                for source, dep in incoming.get(node_id, []):
                    resolved = resolve_name(source)
                    if resolved and resolved in symbols and resolved not in visited:
                        levels[level].add(resolved)
                        visited.add(resolved)
                    if resolved and resolved in visited:
                        links.append({
                            "source": resolved,
                            "target": node_id,
                            "type": dep.dep_type.value,
                            "direction": "in"
                        })

        # Build node data with level info
        # Only include structural nodes (class, interface, enum) for cleaner viz
        from .ast_base import NodeType
        structural_types = {
            NodeType.CLASS, NodeType.INTERFACE, NodeType.ENUM,
            NodeType.ANNOTATION,  # Java annotations
        }

        nodes = []
        node_ids_included = set()
        for level, node_ids in levels.items():
            for node_id in node_ids:
                sym = symbols.get(node_id)
                if sym and sym.node_type in structural_types:
                    # Count connections
                    out_count = len(outgoing.get(node_id, []))
                    in_count = len(incoming.get(node_id, []))

                    nodes.append({
                        "id": node_id,
                        "name": sym.name,
                        "type": sym.node_type.value,
                        "visibility": sym.visibility.value,
                        "level": level,
                        "file": str(sym.location.file) if sym.location.file else None,
                        "line": sym.location.line,
                        "out_deps": out_count,
                        "in_deps": in_count,
                        "total_deps": out_count + in_count,
                        "color": _get_color(sym.node_type, self.config.color_scheme),
                    })
                    node_ids_included.add(node_id)

        # Aggregate links between same node pairs
        # Only include links where both endpoints are in the visualization
        link_counts: Dict[str, Dict[str, Any]] = {}
        for link in links:
            if link["source"] not in node_ids_included or link["target"] not in node_ids_included:
                continue
            key = f"{link['source']}|{link['target']}"
            if key not in link_counts:
                link_counts[key] = {
                    "source": link["source"],
                    "target": link["target"],
                    "types": [],
                    "count": 0
                }
            link_counts[key]["types"].append(link["type"])
            link_counts[key]["count"] += 1

        aggregated_links = []
        for lc in link_counts.values():
            # Determine primary type (most frequent)
            type_counts = {}
            for t in lc["types"]:
                type_counts[t] = type_counts.get(t, 0) + 1
            primary_type = max(type_counts, key=type_counts.get)

            aggregated_links.append({
                "source": lc["source"],
                "target": lc["target"],
                "type": primary_type,
                "count": lc["count"],
                "types": list(set(lc["types"]))
            })

        return {
            "focal": focal_node,
            "nodes": nodes,
            "links": aggregated_links,
            "levels": {str(k): list(v) for k, v in levels.items()},
            "stats": {
                "total_nodes": len(nodes),
                "total_links": len(aggregated_links),
                "max_level": max(levels.keys()) if levels else 0,
            }
        }

    def render_html(
        self,
        graph: DependencyGraph,
        focal_node: str,
        title: str = "Radial Dependency Explorer",
        max_levels: int = 3
    ) -> str:
        """Render radial explorer as interactive HTML."""
        ego_data = self.build_ego_graph(graph, focal_node, max_levels)

        # Get all symbols for the searchable list
        all_symbols = [
            {"id": s.qualified_name, "name": s.name, "type": s.node_type.value}
            for s in graph.get_symbols()
            if s.node_type in (NodeType.CLASS, NodeType.INTERFACE, NodeType.ENUM)
        ]

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: {self.config.font_name}, -apple-system, sans-serif;
            background: #0d1117;
            color: #e6edf3;
            overflow: hidden;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #graph-container {{
            flex: 1;
            position: relative;
        }}
        #graph {{
            width: 100%;
            height: 100%;
        }}
        #sidebar {{
            width: 350px;
            background: #161b22;
            border-left: 1px solid #30363d;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .sidebar-header {{
            padding: 20px;
            background: #21262d;
            border-bottom: 1px solid #30363d;
        }}
        .sidebar-header h2 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            color: #58a6ff;
        }}
        #search {{
            width: 100%;
            padding: 10px;
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #0d1117;
            color: #e6edf3;
            font-size: 14px;
        }}
        #search:focus {{
            outline: none;
            border-color: #58a6ff;
        }}
        #search-results {{
            max-height: 200px;
            overflow-y: auto;
            margin-top: 10px;
            display: none;
        }}
        .search-result {{
            padding: 8px 12px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 13px;
        }}
        .search-result:hover {{
            background: #30363d;
        }}
        .search-result .type {{
            color: #8b949e;
            font-size: 11px;
        }}
        .sidebar-content {{
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }}
        .section {{
            margin-bottom: 20px;
        }}
        .section-title {{
            font-size: 11px;
            text-transform: uppercase;
            color: #8b949e;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }}
        #focal-info {{
            background: #21262d;
            border-radius: 8px;
            padding: 15px;
        }}
        .focal-name {{
            font-size: 16px;
            font-weight: 600;
            color: #58a6ff;
            word-break: break-all;
            margin-bottom: 10px;
        }}
        .focal-type {{
            display: inline-block;
            padding: 2px 8px;
            background: #238636;
            border-radius: 12px;
            font-size: 11px;
            margin-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }}
        .metric {{
            background: #0d1117;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: bold;
            color: #58a6ff;
        }}
        .metric-label {{
            font-size: 10px;
            color: #8b949e;
            margin-top: 4px;
        }}
        #node-info {{
            background: #21262d;
            border-radius: 8px;
            padding: 15px;
        }}
        #node-info.empty {{
            color: #8b949e;
            text-align: center;
            font-size: 13px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #30363d;
            font-size: 13px;
        }}
        .info-row:last-child {{
            border-bottom: none;
        }}
        .info-label {{
            color: #8b949e;
        }}
        .info-value {{
            color: #e6edf3;
            text-align: right;
            max-width: 60%;
            word-break: break-all;
        }}
        .btn {{
            padding: 6px 12px;
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #21262d;
            color: #e6edf3;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        .btn:hover {{
            background: #30363d;
            border-color: #8b949e;
        }}
        .btn-primary {{
            background: #238636;
            border-color: #238636;
        }}
        .btn-primary:hover {{
            background: #2ea043;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 11px;
            padding: 4px 8px;
            background: #21262d;
            border-radius: 4px;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .level-ring {{
            fill: none;
            stroke: #30363d;
            stroke-dasharray: 4,4;
        }}
        .node {{
            cursor: pointer;
            transition: all 0.2s;
        }}
        .node:hover {{
            filter: brightness(1.3);
        }}
        .node.focal {{
            stroke: #58a6ff;
            stroke-width: 3px;
        }}
        .node.selected {{
            stroke: #f0883e;
            stroke-width: 3px;
        }}
        .node-label {{
            font-size: 11px;
            fill: #e6edf3;
            pointer-events: none;
            text-anchor: middle;
        }}
        .link {{
            fill: none;
            stroke-opacity: 0.6;
        }}
        .link:hover {{
            stroke-opacity: 1;
        }}
        .link.highlighted {{
            stroke-opacity: 1;
            stroke-width: 3px !important;
        }}
        .arc-label {{
            font-size: 9px;
            fill: #8b949e;
        }}
        .tooltip {{
            position: absolute;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            max-width: 250px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}
        .controls {{
            position: absolute;
            top: 15px;
            left: 15px;
            display: flex;
            gap: 8px;
            z-index: 100;
        }}
        .level-indicator {{
            position: absolute;
            bottom: 15px;
            left: 15px;
            background: #21262d;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 12px;
            border: 1px solid #30363d;
        }}
        .breadcrumb {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        .breadcrumb-item {{
            padding: 4px 10px;
            background: #21262d;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
        }}
        .breadcrumb-item:hover {{
            background: #30363d;
        }}
        .breadcrumb-item.active {{
            background: #238636;
        }}
        .breadcrumb-sep {{
            color: #8b949e;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <svg id="graph"></svg>
            <div class="controls">
                <button class="btn" onclick="zoomIn()">+</button>
                <button class="btn" onclick="zoomOut()">-</button>
                <button class="btn" onclick="resetView()">Reset</button>
                <button class="btn" onclick="exportSVG()">Export</button>
            </div>
            <div class="level-indicator">
                <span id="level-text">Showing 3 levels</span>
            </div>
        </div>
        <div id="sidebar">
            <div class="sidebar-header">
                <h2>Radial Dependency Explorer</h2>
                <input type="text" id="search" placeholder="Search classes to explore...">
                <div id="search-results"></div>
            </div>
            <div class="sidebar-content">
                <div class="section">
                    <div class="section-title">Navigation History</div>
                    <div class="breadcrumb" id="breadcrumb"></div>
                </div>

                <div class="section">
                    <div class="section-title">Focal Class</div>
                    <div id="focal-info"></div>
                </div>

                <div class="section">
                    <div class="section-title">Selected Node</div>
                    <div id="node-info" class="empty">Click a node to see details</div>
                </div>

                <div class="section">
                    <div class="section-title">Dependency Types</div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-dot" style="background: #3fb950"></div>Inheritance</div>
                        <div class="legend-item"><div class="legend-dot" style="background: #58a6ff"></div>Implementation</div>
                        <div class="legend-item"><div class="legend-dot" style="background: #f0883e"></div>Call</div>
                        <div class="legend-item"><div class="legend-dot" style="background: #8b949e"></div>Import</div>
                        <div class="legend-item"><div class="legend-dot" style="background: #a371f7"></div>Composition</div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Node Types</div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-dot" style="background: #4a90d9"></div>Class</div>
                        <div class="legend-item"><div class="legend-dot" style="background: #50c878"></div>Interface</div>
                        <div class="legend-item"><div class="legend-dot" style="background: #1abc9c"></div>Enum</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip" style="display: none;"></div>

    <script>
        // Initial data
        let egoData = {json.dumps(ego_data)};
        const allSymbols = {json.dumps(all_symbols)};
        const focalHistory = ['{html.escape(focal_node)}'];

        // Edge colors by type
        const edgeColors = {{
            'inheritance': '#3fb950',
            'implementation': '#58a6ff',
            'call': '#f0883e',
            'import': '#8b949e',
            'composition': '#a371f7',
            'annotation': '#f778ba',
            'type_reference': '#8b949e'
        }};

        // Dimensions
        const container = document.getElementById('graph-container');
        const width = container.clientWidth;
        const height = container.clientHeight;
        const centerX = width / 2;
        const centerY = height / 2;
        const levelRadius = Math.min(width, height) / 8;

        // SVG setup
        const svg = d3.select('#graph')
            .attr('width', width)
            .attr('height', height);

        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on('zoom', (e) => g.attr('transform', e.transform));

        svg.call(zoom);

        const g = svg.append('g');
        const ringsLayer = g.append('g').attr('class', 'rings');
        const linksLayer = g.append('g').attr('class', 'links');
        const nodesLayer = g.append('g').attr('class', 'nodes');
        const labelsLayer = g.append('g').attr('class', 'labels');

        let selectedNode = null;

        // Render the visualization
        function render() {{
            const nodes = egoData.nodes;
            const links = egoData.links;
            const maxLevel = egoData.stats.max_level;

            // Clear previous
            ringsLayer.selectAll('*').remove();
            linksLayer.selectAll('*').remove();
            nodesLayer.selectAll('*').remove();
            labelsLayer.selectAll('*').remove();

            // Draw level rings
            for (let l = 1; l <= maxLevel; l++) {{
                ringsLayer.append('circle')
                    .attr('class', 'level-ring')
                    .attr('cx', centerX)
                    .attr('cy', centerY)
                    .attr('r', l * levelRadius);

                // Level label
                ringsLayer.append('text')
                    .attr('class', 'arc-label')
                    .attr('x', centerX + l * levelRadius + 5)
                    .attr('y', centerY)
                    .text(`Level ${{l}}`);
            }}

            // Calculate node positions
            const nodePositions = {{}};
            const levelNodes = {{}};

            nodes.forEach(n => {{
                levelNodes[n.level] = levelNodes[n.level] || [];
                levelNodes[n.level].push(n);
            }});

            // Position nodes in concentric circles
            Object.keys(levelNodes).forEach(level => {{
                const lNodes = levelNodes[level];
                const radius = level * levelRadius;
                const angleStep = (2 * Math.PI) / Math.max(lNodes.length, 1);

                lNodes.forEach((n, i) => {{
                    if (parseInt(level) === 0) {{
                        // Focal node at center
                        n.x = centerX;
                        n.y = centerY;
                    }} else {{
                        const angle = i * angleStep - Math.PI / 2;
                        n.x = centerX + radius * Math.cos(angle);
                        n.y = centerY + radius * Math.sin(angle);
                    }}
                    nodePositions[n.id] = {{ x: n.x, y: n.y, level: n.level }};
                }});
            }});

            // Draw links as curved arcs
            linksLayer.selectAll('.link')
                .data(links)
                .join('path')
                .attr('class', 'link')
                .attr('d', d => {{
                    const src = nodePositions[d.source];
                    const tgt = nodePositions[d.target];
                    if (!src || !tgt) return '';

                    // Calculate arc
                    const dx = tgt.x - src.x;
                    const dy = tgt.y - src.y;
                    const dr = Math.sqrt(dx * dx + dy * dy) * 0.8;

                    // Determine curve direction based on levels
                    const sweep = src.level < tgt.level ? 1 : 0;

                    return `M${{src.x}},${{src.y}}A${{dr}},${{dr}} 0 0,${{sweep}} ${{tgt.x}},${{tgt.y}}`;
                }})
                .attr('stroke', d => edgeColors[d.type] || '#8b949e')
                .attr('stroke-width', d => Math.min(1 + d.count * 0.5, 4))
                .on('mouseover', (e, d) => showLinkTooltip(e, d))
                .on('mouseout', hideTooltip);

            // Draw nodes
            nodesLayer.selectAll('.node')
                .data(nodes)
                .join('circle')
                .attr('class', d => `node ${{d.level === 0 ? 'focal' : ''}}`)
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', d => d.level === 0 ? 20 : Math.min(8 + d.total_deps * 0.3, 15))
                .attr('fill', d => d.color)
                .on('click', (e, d) => selectNode(e, d))
                .on('dblclick', (e, d) => refocus(d.id))
                .on('mouseover', (e, d) => showNodeTooltip(e, d))
                .on('mouseout', hideTooltip);

            // Draw labels
            labelsLayer.selectAll('.node-label')
                .data(nodes.filter(n => n.level <= 1))
                .join('text')
                .attr('class', 'node-label')
                .attr('x', d => d.x)
                .attr('y', d => d.y + (d.level === 0 ? 35 : 20))
                .text(d => truncate(d.name, 15));

            // Update focal info
            updateFocalInfo();
            updateBreadcrumb();
        }}

        function truncate(text, len) {{
            return text.length > len ? text.slice(0, len) + '...' : text;
        }}

        function selectNode(event, d) {{
            event.stopPropagation();
            selectedNode = d;

            // Update node highlighting
            nodesLayer.selectAll('.node')
                .classed('selected', n => n === d);

            // Highlight connected links
            linksLayer.selectAll('.link')
                .classed('highlighted', l => l.source === d.id || l.target === d.id);

            updateNodeInfo(d);
        }}

        function refocus(nodeId) {{
            if (nodeId === egoData.focal) return;

            // Add to history
            focalHistory.push(nodeId);

            // In standalone mode, show the command to run
            const shortName = nodeId.split('.').pop();
            const cmd = `ragix-ast radial <path> --focal "${{shortName}}"`;

            // Copy to clipboard
            navigator.clipboard.writeText(cmd).then(() => {{
                alert(`Command copied to clipboard!\\n\\n${{cmd}}\\n\\nPaste and run in terminal to explore "${{shortName}}" as the center.`);
            }}).catch(() => {{
                alert(`To explore "${{shortName}}" as focal node, run:\\n\\n${{cmd}}`);
            }});
        }}

        function updateFocalInfo() {{
            const focal = egoData.nodes.find(n => n.level === 0);
            if (!focal) return;

            const levelCounts = {{}};
            egoData.nodes.forEach(n => {{
                levelCounts[n.level] = (levelCounts[n.level] || 0) + 1;
            }});

            document.getElementById('focal-info').innerHTML = `
                <div class="focal-name">${{focal.name}}</div>
                <span class="focal-type">${{focal.type}}</span>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value">${{focal.out_deps}}</div>
                        <div class="metric-label">Outgoing</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${{focal.in_deps}}</div>
                        <div class="metric-label">Incoming</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${{levelCounts[1] || 0}}</div>
                        <div class="metric-label">Level 1</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${{levelCounts[2] || 0}}</div>
                        <div class="metric-label">Level 2</div>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <div class="info-row">
                        <span class="info-label">Full Name</span>
                    </div>
                    <div style="font-size: 11px; color: #8b949e; word-break: break-all;">${{focal.id}}</div>
                </div>
            `;
        }}

        function updateNodeInfo(node) {{
            if (!node) {{
                document.getElementById('node-info').innerHTML = 'Click a node to see details';
                document.getElementById('node-info').className = 'empty';
                return;
            }}

            document.getElementById('node-info').className = '';
            document.getElementById('node-info').innerHTML = `
                <div style="margin-bottom: 10px;">
                    <strong style="color: #58a6ff;">${{node.name}}</strong>
                    <span class="focal-type" style="margin-left: 8px;">${{node.type}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Level</span>
                    <span class="info-value">${{node.level}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Outgoing Deps</span>
                    <span class="info-value">${{node.out_deps}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Incoming Deps</span>
                    <span class="info-value">${{node.in_deps}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">File</span>
                    <span class="info-value">${{node.file ? node.file.split('/').pop() : 'N/A'}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Line</span>
                    <span class="info-value">${{node.line || 'N/A'}}</span>
                </div>
                ${{node.level > 0 ? `
                <div style="margin-top: 10px;">
                    <button class="btn btn-primary" onclick="refocus('${{node.id}}')" style="width: 100%;">
                        Explore as Center
                    </button>
                    <div style="font-size: 11px; color: #8b949e; margin-top: 5px;">
                        Copies command to clipboard
                    </div>
                </div>
                ` : ''}}
            `;
        }}

        function updateBreadcrumb() {{
            const bc = document.getElementById('breadcrumb');
            bc.innerHTML = focalHistory.map((id, i) => {{
                const name = id.split('.').pop();
                const isActive = i === focalHistory.length - 1;
                return `
                    <span class="breadcrumb-item ${{isActive ? 'active' : ''}}"
                          onclick="goToHistory(${{i}})">${{truncate(name, 12)}}</span>
                    ${{i < focalHistory.length - 1 ? '<span class="breadcrumb-sep">→</span>' : ''}}
                `;
            }}).join('');
        }}

        function goToHistory(index) {{
            if (index >= focalHistory.length - 1) return;
            const targetId = focalHistory[index];
            focalHistory.splice(index + 1);
            refocus(targetId);
        }}

        function showNodeTooltip(event, d) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = `
                <strong>${{d.name}}</strong><br>
                <span style="color: #8b949e;">${{d.type}}</span><br>
                <hr style="border-color: #30363d; margin: 8px 0;">
                Level: ${{d.level}}<br>
                Out: ${{d.out_deps}} | In: ${{d.in_deps}}<br>
                <em style="color: #8b949e; font-size: 11px;">Double-click to explore</em>
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
        }}

        function showLinkTooltip(event, d) {{
            const tooltip = document.getElementById('tooltip');
            const srcName = d.source.split('.').pop();
            const tgtName = d.target.split('.').pop();
            tooltip.innerHTML = `
                <strong>${{srcName}}</strong> → <strong>${{tgtName}}</strong><br>
                <span style="color: ${{edgeColors[d.type] || '#8b949e'}};">${{d.type}}</span><br>
                Connections: ${{d.count}}
                ${{d.types.length > 1 ? `<br><span style="color: #8b949e;">Types: ${{d.types.join(', ')}}</span>` : ''}}
            `;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
        }}

        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        // Search functionality
        const searchInput = document.getElementById('search');
        const searchResults = document.getElementById('search-results');

        searchInput.addEventListener('input', function(e) {{
            const term = e.target.value.toLowerCase();
            if (term.length < 2) {{
                searchResults.style.display = 'none';
                return;
            }}

            const matches = allSymbols
                .filter(s => s.name.toLowerCase().includes(term) || s.id.toLowerCase().includes(term))
                .slice(0, 10);

            if (matches.length === 0) {{
                searchResults.style.display = 'none';
                return;
            }}

            searchResults.innerHTML = matches.map(s => `
                <div class="search-result" onclick="searchSelect('${{s.id}}')">
                    ${{s.name}}
                    <span class="type">${{s.type}}</span>
                </div>
            `).join('');
            searchResults.style.display = 'block';
        }});

        function searchSelect(nodeId) {{
            searchResults.style.display = 'none';
            searchInput.value = '';
            focalHistory.length = 0;
            focalHistory.push(nodeId);

            // Reload page with new focal (in real app, use API)
            window.location.href = window.location.pathname + '?focal=' + encodeURIComponent(nodeId);
        }}

        // Zoom controls
        function zoomIn() {{
            svg.transition().call(zoom.scaleBy, 1.3);
        }}

        function zoomOut() {{
            svg.transition().call(zoom.scaleBy, 0.7);
        }}

        function resetView() {{
            svg.transition().call(zoom.transform, d3.zoomIdentity);
        }}

        function exportSVG() {{
            const svgData = new XMLSerializer().serializeToString(svg.node());
            const blob = new Blob([svgData], {{ type: 'image/svg+xml;charset=utf-8' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'radial-deps.svg';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // Clear selection on background click
        svg.on('click', () => {{
            selectedNode = null;
            nodesLayer.selectAll('.node').classed('selected', false);
            linksLayer.selectAll('.link').classed('highlighted', false);
            updateNodeInfo(null);
        }});

        // Initial render
        render();
    </script>
</body>
</html>'''


def render_ast_tree(node: ASTNode, config: Optional[VizConfig] = None) -> str:
    """Render an AST tree to DOT format."""
    config = config or VizConfig()
    lines = []
    lines.append("digraph AST {")
    lines.append(f"  rankdir={config.direction.value};")
    lines.append(f"  fontname=\"{config.font_name}\";")
    lines.append("  node [shape=box, style=filled];")
    lines.append("")

    def visit(n: ASTNode, parent_id: Optional[str] = None):
        node_id = f"n{id(n)}"
        label = f"{n.node_type.value}"
        if n.name:
            label += f"\\n{_truncate(n.name, config.max_label_length)}"

        color = _get_color(n.node_type, config.color_scheme)
        lines.append(f'  "{node_id}" [label="{label}", fillcolor="{color}"];')

        if parent_id:
            lines.append(f'  "{parent_id}" -> "{node_id}";')

        for child in n.children:
            visit(child, node_id)

    visit(node)
    lines.append("}")
    return "\n".join(lines)


# Convenience functions

def graph_to_dot(
    graph: DependencyGraph,
    config: Optional[VizConfig] = None,
) -> str:
    """Render graph to DOT format."""
    renderer = DotRenderer(config)
    return renderer.render(graph)


def graph_to_mermaid(
    graph: DependencyGraph,
    config: Optional[VizConfig] = None,
) -> str:
    """Render graph to Mermaid format."""
    renderer = MermaidRenderer(config)
    return renderer.render(graph)


def graph_to_d3(
    graph: DependencyGraph,
    config: Optional[VizConfig] = None,
) -> str:
    """Render graph to D3.js JSON format."""
    renderer = D3Renderer(config)
    return renderer.render(graph)


def graph_to_html(
    graph: DependencyGraph,
    title: str = "Dependency Graph",
    config: Optional[VizConfig] = None,
) -> str:
    """Render graph to interactive HTML."""
    renderer = HTMLRenderer(config)
    return renderer.render(graph, title)
