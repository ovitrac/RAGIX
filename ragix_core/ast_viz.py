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
    # Solar system analogy: Classes=yellow stars, Constructors=red dwarfs, Methods=blue planets
    ColorScheme.DEFAULT: {
        NodeType.CLASS: "#FFD700",       # Gold (big stars)
        NodeType.INTERFACE: "#50c878",   # Green (nebulae)
        NodeType.METHOD: "#5B9BD5",      # Blue (planets)
        NodeType.FUNCTION: "#5B9BD5",    # Blue (planets)
        NodeType.FIELD: "#9b59b6",       # Purple (moons)
        NodeType.CONSTANT: "#CD5C5C",    # Indian red (asteroids)
        NodeType.MODULE: "#34495e",
        NodeType.PACKAGE: "#2c3e50",
        NodeType.ENUM: "#1abc9c",        # Teal (gas giants)
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

        # Always filter out imports and modules - they clutter visualization
        excluded_types = {NodeType.IMPORT, NodeType.IMPORT_FROM, NodeType.MODULE}
        symbols = [s for s in symbols if s.node_type not in excluded_types]

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

        # Build short name to qualified name mapping for name resolution
        short_to_qualified: Dict[str, str] = {}
        for qname in node_index:
            short = qname.split('.')[-1]
            # Prefer shorter qualified names (avoid method/field noise)
            if short not in short_to_qualified or len(qname) < len(short_to_qualified[short]):
                short_to_qualified[short] = qname

        def resolve_name(name: str) -> Optional[str]:
            """Resolve a dependency name to a node qualified name."""
            if name in node_index:
                return name
            if name in short_to_qualified:
                return short_to_qualified[name]
            # Try extracting last part (module.Class.method -> method, then Class)
            parts = name.split('.')
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i]
                if part in short_to_qualified:
                    return short_to_qualified[part]
            return None

        # Build link list with name resolution
        links = []
        seen_links = set()  # Avoid duplicate edges
        for dep in deps:
            source = resolve_name(dep.source)
            target = resolve_name(dep.target)
            if source and target and source in node_index and target in node_index:
                # Avoid self-loops and duplicates
                if source != target:
                    link_key = (node_index[source], node_index[target], dep.dep_type.value)
                    if link_key not in seen_links:
                        seen_links.add(link_key)
                        links.append({
                            "source": node_index[source],
                            "target": node_index[target],
                            "type": dep.dep_type.value,
                            "color": _get_edge_color(dep.dep_type, self.config.color_scheme),
                        })

        # Add structural edges: connect methods/fields/constructors to parent classes
        # These create the "solar system" effect - members orbit their class
        satellite_types = {"method", "field", "constructor", "constant"}
        class_types = {"class", "interface", "enum"}
        structural_color = "#333333"  # Dark color for structural links

        # Build class lookup for quick parent finding
        class_ids = {n["id"] for n in nodes if n["type"] in class_types}

        for node in nodes:
            if node["type"] in satellite_types:
                # Infer parent class from qualified name (e.g., MyClass.myMethod -> MyClass)
                parts = node["id"].split(".")
                if len(parts) >= 2:
                    parent_id = ".".join(parts[:-1])
                    if parent_id in node_index and parent_id in class_ids:
                        # Add structural edge (member -> class)
                        link_key = (node_index[node["id"]], node_index[parent_id], "structural")
                        if link_key not in seen_links:
                            seen_links.add(link_key)
                            links.append({
                                "source": node_index[node["id"]],
                                "target": node_index[parent_id],
                                "type": "structural",
                                "color": structural_color,
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


# Threshold for switching to vis.js renderer (GPU-accelerated)
VIS_RENDERER_THRESHOLD = 2000


class VisHTMLRenderer:
    """
    Render interactive HTML visualization using vis.js for large graphs.

    vis.js uses Canvas rendering (GPU-accelerated) which performs much better
    than D3.js SVG for graphs with >2000 nodes.

    Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-29
    """

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    @staticmethod
    def _hash_rand(seed: str, salt: str = "", mod: float = 1.0) -> float:
        """
        Deterministic hash-based random number generator.
        Returns a value in [0, mod) based on the hash of seed+salt.
        Produces stable, reproducible layouts across renders.
        """
        import hashlib
        h = hashlib.sha256((seed + salt).encode()).hexdigest()
        return (int(h[:8], 16) % 10_000_000) / 10_000_000.0 * mod

    def render(self, graph: DependencyGraph, title: str = "Dependency Graph") -> str:
        """Render graph to interactive HTML with vis.js."""
        d3_renderer = D3Renderer(self.config)
        graph_data = d3_renderer.to_dict(graph)

        # Container types: nodes that can have members
        container_types = ("class", "interface", "enum", "module", "namespace", "component", "object")

        # Build node lookup and identify containers
        node_by_id = {node["id"]: node for node in graph_data["nodes"]}
        container_ids = set()

        # First pass: assign packages to containers (real packages or (default))
        for node in graph_data["nodes"]:
            node_type = node.get("type", "")
            if node_type in container_types:
                container_ids.add(node["id"])
                parts = node["id"].split(".")
                node["package"] = ".".join(parts[:-1]) if len(parts) > 1 else "(default)"

        # Second pass: members inherit package from their container
        for node in graph_data["nodes"]:
            if node["id"] in container_ids:
                continue  # Already handled
            # Try to find parent container
            node_id = node["id"]
            if "." in node_id:
                parent_id = ".".join(node_id.split(".")[:-1])
                if parent_id in node_by_id:
                    # Inherit package from parent
                    parent_node = node_by_id[parent_id]
                    node["package"] = parent_node.get("package", "(default)")
                else:
                    # Parent not found, use derived package
                    node["package"] = parent_id if parent_id else "(default)"
            else:
                node["package"] = "(default)"

        # Compute package statistics (only count containers for cleaner list)
        packages: Dict[str, int] = {}
        for node in graph_data["nodes"]:
            pkg = node["package"]
            packages[pkg] = packages.get(pkg, 0) + 1

        return self._render_vis_html(graph_data, title, packages)

    def _render_vis_html(
        self,
        graph_data: Dict[str, Any],
        title: str,
        packages: Dict[str, int]
    ) -> str:
        """Render HTML with vis.js network visualization.

        Version 3: Implements physics-based hierarchical layout with:
        - Ghost edges for method-to-class clustering ("solar system" effect)
        - Node mass/sizing based on method count
        - Color-coded dependency edges
        """
        # --- Pre-process: Calculate class sizes and identify method-class relationships ---
        class_method_count: Dict[str, int] = {}
        method_to_class: Dict[str, str] = {}

        # Container types: nodes that can have members (classes, interfaces, enums, modules, etc.)
        container_types = ("class", "interface", "enum", "module", "namespace", "component", "object")

        # Satellite types: nodes that belong to a container (methods, fields, properties, etc.)
        satellite_types = ("method", "field", "constructor", "constant", "function", "property", "getter", "setter")

        # First pass: identify all container types
        for node in graph_data["nodes"]:
            node_type = node.get("type", "")
            if node_type in container_types:
                class_method_count[node["id"]] = 0

        # Build node ID set for fallback parent detection
        all_node_ids = {node["id"] for node in graph_data["nodes"]}

        # Second pass: link satellites to their parent containers
        # These are the "satellites" that should orbit around their class "sun"
        for node in graph_data["nodes"]:
            node_type = node.get("type", "")
            # Skip container types - they don't have parents in this sense
            if node_type in container_types:
                continue

            # For satellites and any unrecognized types with a dotted ID, try to find parent
            node_id = node["id"]
            if "." in node_id:
                parent_class_id = ".".join(node_id.split(".")[:-1])
                # Check if parent is a known container
                if parent_class_id in class_method_count:
                    class_method_count[parent_class_id] += 1
                    method_to_class[node_id] = parent_class_id
                # Fallback: if parent exists as any node, link to it anyway
                elif parent_class_id in all_node_ids:
                    # Add parent to class_method_count if not already there
                    if parent_class_id not in class_method_count:
                        class_method_count[parent_class_id] = 0
                    class_method_count[parent_class_id] += 1
                    method_to_class[node_id] = parent_class_id

        # --- Compute class positions using PCoA/MDS on topology-derived distances ---
        # This seeds 2D coordinates from actual graph structure (inheritance, calls)
        # rather than arbitrary geometric placement
        import math
        import numpy as np

        # Group classes by package (for later package centroid calculation)
        package_classes: Dict[str, List[str]] = {}
        for node in graph_data["nodes"]:
            if node.get("type") in ("class", "interface", "enum"):
                pkg = node.get("package", "(default)")
                if pkg not in package_classes:
                    package_classes[pkg] = []
                package_classes[pkg].append(node["id"])

        # Build list of class IDs for matrix indexing
        class_types = {"class", "interface", "enum"}
        class_ids = [n["id"] for n in graph_data["nodes"] if n.get("type") in class_types]
        class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        n_classes = len(class_ids)

        # Build weighted adjacency matrix (class-to-class only)
        # Weight: inheritance=3.0, implementation=3.0, call=1.0 (damped by out-degree)
        weight_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)

        # Count out-degree per class for call weight normalization
        out_degree: Dict[str, int] = {cid: 0 for cid in class_ids}
        for link in graph_data.get("links", []):
            source_id = graph_data["nodes"][link["source"]]["id"] if isinstance(link["source"], int) else link["source"]
            if source_id in out_degree:
                out_degree[source_id] += 1

        # Populate weight matrix from edges
        for link in graph_data.get("links", []):
            source_id = graph_data["nodes"][link["source"]]["id"] if isinstance(link["source"], int) else link["source"]
            target_id = graph_data["nodes"][link["target"]]["id"] if isinstance(link["target"], int) else link["target"]
            dep_type = link.get("type", "dependency")

            # Only process class-to-class edges
            if source_id not in class_to_idx or target_id not in class_to_idx:
                continue

            i, j = class_to_idx[source_id], class_to_idx[target_id]

            # Weight by dependency type
            if dep_type in ("inheritance", "implementation"):
                w = 3.0  # Strong structural relationship
            elif dep_type == "call":
                # Dampen by out-degree to avoid hubs dominating
                deg = max(out_degree.get(source_id, 1), 1)
                w = 1.0 / math.sqrt(deg)  # sqrt dampening
            else:
                w = 0.5  # Other dependencies (type_reference, import, etc.)

            # Symmetric: both directions
            weight_matrix[i, j] += w
            weight_matrix[j, i] += w

        # Convert weights to distances: dist = 1 / (epsilon + weight)
        epsilon = 1e-3
        dist_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    dist_matrix[i, j] = 0.0
                elif weight_matrix[i, j] > 0:
                    dist_matrix[i, j] = 1.0 / (epsilon + weight_matrix[i, j])
                else:
                    dist_matrix[i, j] = -1  # Mark disconnected, fill later

        # For disconnected pairs, use 95th percentile of finite distances
        finite_dists = dist_matrix[dist_matrix > 0]
        if len(finite_dists) > 0:
            fallback_dist = np.percentile(finite_dists, 95)
        else:
            fallback_dist = 100.0  # No edges at all

        dist_matrix[dist_matrix < 0] = fallback_dist

        # Run PCoA/MDS for 2D coordinates (deterministic)
        class_positions: Dict[str, Tuple[float, float]] = {}

        if n_classes >= 2:
            try:
                from sklearn.manifold import MDS
                mds = MDS(
                    n_components=2,
                    dissimilarity="precomputed",
                    random_state=42,  # Deterministic
                    max_iter=300,
                    n_init=4,  # Explicit to avoid FutureWarning (default changes in sklearn 1.9)
                    normalized_stress="auto"
                )
                coords_2d = mds.fit_transform(dist_matrix)

                # Scale coordinates by CONSTANT factor to preserve relative distances
                # Do NOT normalize to bounding box - that causes square confinement
                # MDS returns coords in distance-space units; scale for vis.js canvas
                coords_2d -= coords_2d.mean(axis=0)  # Center only
                coords_2d *= 2000.0  # Standard scale for class positions

                for idx, cid in enumerate(class_ids):
                    class_positions[cid] = (float(coords_2d[idx, 0]), float(coords_2d[idx, 1]))

            except ImportError:
                # Fallback: simple circular layout if sklearn not available
                for idx, cid in enumerate(class_ids):
                    angle = 2 * math.pi * idx / n_classes
                    class_positions[cid] = (2000 * math.cos(angle), 2000 * math.sin(angle))
        else:
            # Single class: center
            for cid in class_ids:
                class_positions[cid] = (0.0, 0.0)

        # Determine an outer ring radius for orphan members (no class or missing position)
        if class_positions:
            max_coord = max(
                max(abs(pos[0]), abs(pos[1]))
                for pos in class_positions.values()
            )
            # Shrink outer ring by factor 3 to bring orphans closer to galaxy
            outer_ring_radius = (max_coord + 500.0) / 3.0
        else:
            outer_ring_radius = 667.0  # 2000 / 3

        # Compute package positions as centroids of their classes
        package_positions: Dict[str, Tuple[float, float]] = {}
        for pkg, classes in package_classes.items():
            if classes:
                xs = [class_positions.get(c, (0, 0))[0] for c in classes]
                ys = [class_positions.get(c, (0, 0))[1] for c in classes]
                package_positions[pkg] = (sum(xs) / len(xs), sum(ys) / len(ys))
            else:
                package_positions[pkg] = (0.0, 0.0)

        # --- Prepare nodes with mass, size, and initial position ---
        vis_nodes = []
        for node in graph_data["nodes"]:
            node_id = node["id"]
            node_type = node.get("type", "class")
            node_name = node.get("name", node_id.split(".")[-1])
            pkg = node.get("package", "(default)")

            # Default mass and size
            mass = 1.0
            value = 10.0
            x, y = None, None
            node_fixed = None  # Will be set for anchored nodes

            # === MASS-BASED PHASE SEPARATION ===
            # Heavier nodes resist repulsion → stay closer to center/star
            # Lighter nodes pushed outward → settle in outer orbits
            # This creates natural "orbital bands" around class stars

            # Classes: Sun (heaviest, scaled by member count)
            if node_type == "class":
                member_count = class_method_count.get(node_id, 0)
                mass = 5.0 + member_count * 0.5  # Heavy sun
                value = 15 + min(member_count, 40)
                if node_id in class_positions:
                    x, y = class_positions[node_id]
            # Interfaces: Heavy planet (inner orbit)
            elif node_type == "interface":
                member_count = class_method_count.get(node_id, 0)
                mass = 4.0  # Heavy planet
                value = 12 + min(member_count * 0.3, 20)
                if node_id in class_positions:
                    x, y = class_positions[node_id]
            # Enums: Medium planet
            elif node_type == "enum":
                member_count = class_method_count.get(node_id, 0)
                mass = 3.0  # Medium planet
                value = 10 + min(member_count * 0.3, 15)
                if node_id in class_positions:
                    x, y = class_positions[node_id]
            # Methods: Light planet (outer orbit)
            elif node_type == "method":
                mass = 0.5  # Light planet
                value = 5
            # Constructors: Light planet (like methods)
            elif node_type == "constructor":
                mass = 0.5  # Light planet
                value = 6
            # Functions: Light planet (like methods)
            elif node_type == "function":
                mass = 0.5  # Light planet
                value = 5
            # Fields: Asteroid (lightest)
            elif node_type == "field":
                mass = 0.2  # Asteroid
                value = 4
            # Constants: Asteroid (lightest)
            elif node_type == "constant":
                mass = 0.2  # Asteroid
                value = 3
            # Imports: Comet (very light, outermost)
            elif node_type == "import":
                mass = 0.1  # Comet
                value = 2
            # Packages: Fixed anchors
            elif node_type == "package":
                mass = 10.0  # Packages are heavy anchors
                value = 25
                if pkg in package_positions:
                    x, y = package_positions[pkg]
                # Packages are pinned - they don't move during simulation
                node_fixed = {"x": True, "y": True}

            # Position satellites near their parent class using deterministic halos;
            # or place orphans on an outer ring if the class position is missing.
            if node_id in method_to_class:
                parent_id = method_to_class[node_id]
                parent_pos = class_positions.get(parent_id)
                if parent_pos:
                    cx, cy = parent_pos
                    m = class_method_count.get(parent_id, 1)  # Member count for this class

                    # Halo radius scaled for roughly uniform member density across classes
                    # R_mem ~ 20 + 10 * sqrt(m), capped to prevent over-expansion
                    halo_radius = min(20 + 10 * math.sqrt(m), 150)

                    # Get member index within this class for angular distribution
                    # Sort members by id for consistent ordering
                    class_members = [mid for mid, cid in method_to_class.items() if cid == parent_id]
                    class_members.sort()
                    member_idx = class_members.index(node_id) if node_id in class_members else 0

                    # Base angle distributed evenly, with deterministic jitter
                    base_angle = (2 * math.pi * member_idx) / max(m, 1)
                    angle_jitter = self._hash_rand(node_id, "mem_angle", math.pi / (2 * max(1, m)))
                    angle = base_angle + angle_jitter - math.pi / (4 * max(1, m))

                    # Radius jitter: r = R_mem * (0.7 + 0.3 * hash_rand) for organic feel
                    radius_factor = 0.7 + 0.3 * self._hash_rand(node_id, "mem_r")
                    offset = halo_radius * radius_factor

                    x = cx + offset * math.cos(angle)
                    y = cy + offset * math.sin(angle)
                else:
                    # Orphan member (missing class position) -> place on outer ring deterministically
                    angle = self._hash_rand(node_id, "orphan_angle", 2 * math.pi)
                    x = outer_ring_radius * math.cos(angle)
                    y = outer_ring_radius * math.sin(angle)

            node_data = {
                "id": node_id,
                "label": node_name,
                "group": node_type,
                "title": f"{node_name}\nType: {node_type}\nPackage: {pkg}",  # Plain text tooltip
                "package": pkg,
                "value": value,
                "mass": mass,
            }

            # Add initial position if computed
            if x is not None and y is not None:
                node_data["x"] = x
                node_data["y"] = y

            # Add fixed property for pinned nodes (packages)
            if node_fixed is not None:
                node_data["fixed"] = node_fixed

            # NOTE: Satellites now participate in physics for mass-based phase separation.
            # Their lighter mass causes them to be pushed outward by heavier nodes,
            # creating natural orbital bands around class stars.
            # Only packages remain fixed (anchored).

            vis_nodes.append(node_data)

        # --- Prepare edges with type information for color coding ---
        vis_edges = []
        edge_counter = 0
        real_edge_count = 0

        # Build node id -> name lookup for tooltips
        node_name_lookup = {n["id"]: n.get("name", n["id"].split(".")[-1]) for n in graph_data["nodes"]}

        # === EDGE LENGTH HIERARCHY (Force Priority) ===
        # Shorter length = stronger spring force = tighter coupling
        # This hierarchy ensures solar systems stay intact while
        # allowing weaker connections to stretch
        edge_length_by_type = {
            # Priority 2: Strong, medium-length (class hierarchies cluster)
            "inheritance": 100,
            "implementation": 100,
            # Priority 3: Medium (collaborating classes)
            "call": 180,
            "field_access": 180,
            # Priority 4: Weakest, longest (loose references)
            "type_reference": 250,
            "import": 250,
            "dependency": 200,  # Default for unknown types
        }
        # Note: structural edges (Priority 1) are handled separately below with length=50

        # Add real dependency edges
        for link in graph_data.get("links", []):
            source_id = graph_data["nodes"][link["source"]]["id"] if isinstance(link["source"], int) else link["source"]
            target_id = graph_data["nodes"][link["target"]]["id"] if isinstance(link["target"], int) else link["target"]
            dep_type = link.get("type", "dependency")

            # Get readable names for tooltip
            source_name = node_name_lookup.get(source_id, source_id.split(".")[-1])
            target_name = node_name_lookup.get(target_id, target_id.split(".")[-1])

            # Get edge length based on dependency type
            edge_length = edge_length_by_type.get(dep_type, 120)

            vis_edges.append({
                "id": edge_counter,
                "from": source_id,
                "to": target_id,
                "arrows": "to",
                "title": f"{source_name} → {target_name}\n({dep_type})",  # Plain text tooltip
                "edgeType": dep_type,  # For color coding in JS
                "length": edge_length,  # Spring length by type
            })
            edge_counter += 1
            real_edge_count += 1

        # Add "structural" edges (method/field -> parent class)
        # These create the "solar system" effect - members orbit their class
        # Hidden by default to reduce clutter, toggleable via UI
        # For large graphs (>5000 nodes), disable physics on structural edges
        structural_edge_count = 0

        total_nodes = len(graph_data["nodes"])
        # Keep structural physics ON even for large graphs to anchor members to classes.
        # Performance is mitigated by keeping these edges hidden and simple.
        structural_physics_enabled = True
        is_large_graph = total_nodes > 5000

        for method_id, class_id in method_to_class.items():
            method_name = node_name_lookup.get(method_id, method_id.split(".")[-1])
            class_name = node_name_lookup.get(class_id, class_id.split(".")[-1])

            # === STRUCTURAL EDGES: Priority 1 (Strongest, Shortest) ===
            # These are the "gravitational bonds" keeping solar systems intact.
            # Fixed short length ensures members stay close to their class star.
            vis_edges.append({
                "id": edge_counter,
                "from": method_id,
                "to": class_id,
                "edgeType": "structural",
                "title": f"{method_name} ∈ {class_name}\n(member of)",
                "hidden": False,  # Visible so springs participate in physics
                "physics": structural_physics_enabled,
                "length": 50,  # Priority 1: Shortest, strongest binding
                "smooth": False,  # Straight lines for performance
            })
            edge_counter += 1
            structural_edge_count += 1

        # Note: Class positions are seeded from PCoA/MDS (topology-based)
        # Physics only needs to do local refinement, not global rearrangement

        node_count = len(vis_nodes)
        edge_count = real_edge_count  # Only count visible edges for display

        # === REFINED PHYSICS CONSTANTS ===
        # Tuned for multi-body simulation with mass-based phase separation.
        # - Weaker global repulsion prevents infinite expansion
        # - Stronger central gravity holds the "galaxy" together
        # - Spring constant works with edge length hierarchy
        gravitational_constant = -35  # Weaker repulsion (balanced by mass)
        spring_length = 150  # Base spring length (overridden per edge type)
        spring_constant = 0.07  # Slightly stiffer springs
        central_gravity = 0.1  # Strong central pull (galactic core)
        avoid_overlap = 0.8  # Prevent node overlap

        # Physics control based on graph size
        # >10k nodes: physics OFF by default (user can toggle on)
        # <10k nodes: physics ON with stabilization
        if node_count > 10000:
            physics_enabled = False
            stabilization_iterations = 0
        elif node_count > 5000:
            physics_enabled = True
            stabilization_iterations = 100
        elif node_count > 2000:
            physics_enabled = True
            stabilization_iterations = 150
        else:
            physics_enabled = True
            stabilization_iterations = 250

        # Generate package list HTML
        package_items = []
        sorted_packages = sorted(packages.items(), key=lambda x: (-x[1], x[0]))
        for i, (pkg, count) in enumerate(sorted_packages):
            hue = (i * 137.5) % 360
            color = f"hsl({hue}, 60%, 50%)"
            display = pkg if len(pkg) <= 40 else "..." + pkg[-37:]
            # Use data attribute and read via dataset.pkg (auto-unescapes)
            # This ensures consistency between checkbox click and selectFiltered()
            package_items.append(
                f'<div class="pkg-item" data-pkg="{html.escape(pkg)}">'
                f'<input type="checkbox" checked onchange="togglePackage(this.parentElement.dataset.pkg, this.checked)">'
                f'<span class="pkg-color" style="background:{color}"></span>'
                f'<span class="pkg-name" title="{html.escape(pkg)}">{html.escape(display)}</span>'
                f'<span class="pkg-count">{count}</span>'
                f'</div>'
            )
        package_html = "\n".join(package_items)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
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
            gap: 8px;
            z-index: 100;
        }}
        .btn {{
            padding: 8px 14px;
            border: none;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.2s;
        }}
        .btn:hover {{ background: rgba(255,255,255,0.2); }}
        #info {{
            position: absolute;
            top: 10px;
            right: 340px;
            background: rgba(0,0,0,0.7);
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 13px;
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
            padding: 15px;
            background: #0f3460;
        }}
        #sidebar-header h2 {{
            margin: 0 0 12px 0;
            font-size: 16px;
            color: #e94560;
        }}
        #search {{
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 13px;
        }}
        #search::placeholder {{ color: #666; }}
        #stats {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }}
        .stat {{
            flex: 1;
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}
        .stat-value {{ font-size: 20px; font-weight: bold; color: #e94560; }}
        .stat-label {{ font-size: 11px; color: #888; }}
        #pkg-list {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }}
        .pkg-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 8px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .pkg-item:hover {{ background: rgba(255,255,255,0.05); }}
        .pkg-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            flex-shrink: 0;
        }}
        .pkg-name {{
            flex: 1;
            font-size: 12px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .pkg-count {{
            font-size: 11px;
            color: #888;
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 10px;
        }}
        .pkg-actions {{
            padding: 10px;
            display: flex;
            gap: 5px;
            border-top: 1px solid #0f3460;
        }}
        .pkg-actions .btn {{ flex: 1; font-size: 11px; }}
        #legend {{
            padding: 10px;
            border-top: 1px solid #0f3460;
            font-size: 11px;
        }}
        #legend h3 {{
            margin: 0 0 8px 0;
            font-size: 12px;
            color: #e94560;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin: 4px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 3px;
            border-radius: 1px;
        }}
        .type-toggle {{
            cursor: pointer;
            user-select: none;
        }}
        .type-toggle:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .type-toggle input {{
            margin-right: 4px;
            cursor: pointer;
        }}
        .type-toggle.disabled {{
            opacity: 0.4;
        }}
        /* Sidebar reorganization - collapsible sections */
        .sidebar-actions {{
            display: flex;
            gap: 4px;
            padding: 8px;
            background: #0f3460;
            align-items: center;
            flex-shrink: 0;
        }}
        .btn-sm {{
            padding: 6px 10px;
            font-size: 14px;
        }}
        .btn-mini {{
            padding: 2px 6px;
            font-size: 9px;
            background: rgba(255,255,255,0.1);
            border: none;
            color: #aaa;
            border-radius: 3px;
            cursor: pointer;
        }}
        .btn-mini:hover {{ background: rgba(255,255,255,0.2); color: #fff; }}
        .sidebar-section {{
            flex-shrink: 0;
            border-bottom: 1px solid #0f3460;
        }}
        .sidebar-section.packages-section {{
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
            border-bottom: none;
        }}
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 10px;
            background: rgba(15, 52, 96, 0.5);
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
            color: #e94560;
        }}
        .section-header:hover {{ background: rgba(15, 52, 96, 0.8); }}
        .collapse-icon {{
            font-size: 10px;
            color: #888;
            transition: transform 0.2s;
        }}
        .section-content {{
            padding: 6px 8px;
        }}
        .collapsible.collapsed .section-content {{ display: none; }}
        .collapsible.collapsed .collapse-icon {{ transform: rotate(0deg); }}
        .legend-group {{
            margin-bottom: 8px;
        }}
        .legend-group-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 10px;
            color: #888;
            margin-bottom: 4px;
            padding-bottom: 2px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .packages-section #pkg-list {{
            flex: 1;
            overflow-y: auto;
            min-height: 0;
        }}
        #loading {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(26, 26, 46, 0.95);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }}
        #loading.hidden {{ display: none; }}
        .spinner {{
            width: 60px;
            height: 60px;
            border: 4px solid rgba(233, 69, 96, 0.3);
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .loading-text {{
            margin-top: 20px;
            font-size: 14px;
            color: #888;
        }}
        .progress-bar {{
            width: 200px;
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            margin-top: 15px;
            overflow: hidden;
            position: relative;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e94560, #f5a623, #e94560);
            background-size: 200% 100%;
            width: 100%;
            animation: shimmer 1.5s ease-in-out infinite;
        }}
        .progress-fill.complete {{ background: #50c878; animation: none; }}
        @keyframes shimmer {{ 0% {{ background-position: 200% 0; }} 100% {{ background-position: -200% 0; }} }}
        /* Fullscreen mode */
        body.fullscreen #sidebar {{ display: none !important; }}
        body.fullscreen #container {{ width: 100vw; }}
        body.fullscreen #graph-container {{ width: 100%; flex: 1; }}
        body.fullscreen #graph {{ width: 100%; height: 100%; }}
        body.fullscreen #info {{ right: 20px; top: 10px; }}
        /* Help panel */
        #help-panel {{
            position: fixed;
            top: 0; right: -420px;
            width: 420px;
            height: 100vh;
            background: #16213e;
            border-left: 2px solid #e94560;
            padding: 20px;
            overflow-y: auto;
            transition: right 0.3s ease;
            z-index: 2000;
        }}
        #help-panel.visible {{ right: 0; }}
        #help-panel h2 {{
            color: #e94560;
            font-size: 18px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #help-panel .close-btn {{
            background: none; border: none; color: #888; font-size: 24px; cursor: pointer;
        }}
        #help-panel .close-btn:hover {{ color: #fff; }}
        .help-section {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .help-section h3 {{ color: #4fc3f7; font-size: 14px; margin-bottom: 10px; }}
        .help-section p {{ color: #ccc; font-size: 13px; line-height: 1.6; margin-bottom: 8px; }}
        .help-section ul {{ margin: 8px 0 0 20px; color: #aaa; font-size: 12px; }}
        .help-section li {{ margin-bottom: 6px; }}
        .help-section code {{ background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 3px; color: #f5a623; }}
        .help-overlay {{
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.5); opacity: 0; pointer-events: none;
            transition: opacity 0.3s; z-index: 1999;
        }}
        .help-overlay.visible {{ opacity: 1; pointer-events: auto; }}
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner"></div>
        <div class="loading-text">Building dependency graph...</div>
        <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
    </div>
    <div id="container">
        <div id="graph-container">
            <div id="graph"></div>
            <div id="controls">
                <button class="btn" onclick="network.fit()">Fit View</button>
                <button class="btn" onclick="togglePhysics()">Toggle Physics</button>
                <button class="btn" onclick="hideSelected()" title="Hide selected nodes (click to select)">Hide Selected</button>
                <button class="btn" onclick="showAll()" title="Show all hidden nodes">Show All</button>
                <button class="btn" onclick="exportPNG()">Export PNG</button>
                <button class="btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
                <button class="btn" onclick="toggleHelp()">❓ Help</button>
            </div>
    <div class="help-overlay" id="help-overlay" onclick="toggleHelp()"></div>
    <div id="help-panel">
        <h2>
            <span>🌐 Dependency Graph Guide</span>
            <button class="close-btn" onclick="toggleHelp()">×</button>
        </h2>
        <div class="help-section">
            <h3>What is this Graph?</h3>
            <p>This interactive visualization shows dependencies between code elements in your project. Nodes represent classes, methods, fields, etc. Edges show how they depend on each other (inheritance, method calls, imports).</p>
        </div>
        <div class="help-section">
            <h3>Navigation</h3>
            <ul>
                <li><strong>Scroll</strong> to zoom in/out</li>
                <li><strong>Drag background</strong> to pan the view</li>
                <li><strong>Click node</strong> to select it</li>
                <li><strong>Ctrl+Click</strong> to multi-select nodes</li>
                <li><strong>Fit View</strong> resets zoom to show all nodes</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Node Types (click legend to toggle)</h3>
            <ul>
                <li><code style="background:#FFD700;color:#1a1a2e">Gold circles</code> - Classes (stars ⭐)</li>
                <li><code style="background:#50c878">Green circles</code> - Interfaces (nebulae)</li>
                <li><code style="background:#5B9BD5">Blue dots</code> - Methods (planets 🔵)</li>
                <li><code style="background:#CD5C5C">Red dots</code> - Constructors (red dwarfs 🔴)</li>
                <li><code style="background:#9b59b6">Purple dots</code> - Fields (moons)</li>
                <li><code style="background:#34495e">Dark boxes</code> - Packages</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Edge Colors</h3>
            <ul>
                <li><code style="background:#d95f02">Orange</code> - Inheritance (extends)</li>
                <li><code style="background:#7570b3">Purple</code> - Implementation (implements)</li>
                <li><code style="background:#1b9e77">Green</code> - Method calls</li>
                <li><code style="background:#66a61e">Light green</code> - Imports</li>
                <li><code style="background:#e6ab02">Yellow</code> - Type references</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Package Filtering</h3>
            <ul>
                <li><strong>Checkboxes</strong> - Toggle package visibility</li>
                <li><strong>Search box</strong> - Filter packages by name</li>
                <li><strong>Select/Deselect Filtered</strong> - Bulk toggle filtered packages</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Exploring Dependencies</h3>
            <ul>
                <li><strong>↑ Parents</strong> - Show what selected nodes depend on</li>
                <li><strong>↓ Children</strong> - Show what depends on selected nodes</li>
                <li><strong>Hide Selected</strong> - Remove selected nodes from view</li>
                <li><strong>Show All</strong> - Restore all hidden nodes</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Physics Simulation</h3>
            <p>The graph uses physics-based layout where nodes repel each other and edges act as springs. Classes attract their methods (like solar systems).</p>
            <p><strong>Toggle Physics</strong> to enable/disable animation. For very large graphs, physics may be disabled for performance.</p>
        </div>
    </div>
            <div id="info">
                <span id="visible-count">{node_count}</span> / {node_count} nodes | {edge_count} edges
            </div>
            <div id="layout-info" style="display: block; position: absolute; bottom: 60px; left: 50%; transform: translateX(-50%); background: rgba(0,80,160,0.9); padding: 10px 18px; border-radius: 6px; font-size: 12px; max-width: 550px; text-align: center;">
                {'⚡ Large graph (' + f'{node_count:,}' + ' nodes): Physics disabled. PCoA layout based on inheritance/call topology. Filter packages to reduce, then use "🔄 Restart Sim".' if not physics_enabled else '🔬 PCoA-seeded layout: classes positioned by inheritance/call distances. Physics refines locally. Connected classes cluster together.'}
            </div>
        </div>
        <div id="sidebar">
            <!-- Actions bar - always visible at top -->
            <div class="sidebar-actions">
                <button class="btn btn-sm" onclick="restartSimulation(false)" title="Reset to PCoA topology">🔄</button>
                <button class="btn btn-sm" onclick="openSettings()" title="Physics settings">⚙️</button>
                <button class="btn btn-sm" onclick="expandParents()" title="Show parents">↑</button>
                <button class="btn btn-sm" onclick="expandChildren()" title="Show children">↓</button>
                <span style="flex:1"></span>
                <span id="visible-count" style="color:#4a90d9;font-weight:bold;">{node_count}</span>
                <span style="color:#666;font-size:11px;">/{node_count}</span>
            </div>

            <!-- Legend - collapsible (collapsed by default for large graphs) -->
            <div class="sidebar-section collapsible {'collapsed' if node_count > 500 else ''}">
                <div class="section-header" onclick="toggleSection(this)">
                    <span>🎨 Legend</span>
                    <span class="collapse-icon">{'▶' if node_count > 500 else '▼'}</span>
                </div>
                <div class="section-content" style="{'display:none' if node_count > 500 else ''}">
                    <div class="legend-group">
                        <div class="legend-group-header">
                            <span>Edges</span>
                            <div style="display:flex;gap:2px;">
                                <button class="btn-mini" onclick="event.stopPropagation();toggleAllEdges(true)">On</button>
                                <button class="btn-mini" onclick="event.stopPropagation();toggleAllEdges(false)">Off</button>
                            </div>
                        </div>
                        <div class="legend-item type-toggle" data-edge="inheritance" onclick="toggleEdgeType('inheritance')"><input type="checkbox" checked><span class="legend-color" style="background:#d95f02"></span> Inherit</div>
                        <div class="legend-item type-toggle" data-edge="implementation" onclick="toggleEdgeType('implementation')"><input type="checkbox" checked><span class="legend-color" style="background:#7570b3"></span> Impl</div>
                        <div class="legend-item type-toggle" data-edge="call" onclick="toggleEdgeType('call')"><input type="checkbox" checked><span class="legend-color" style="background:#1b9e77"></span> Call</div>
                        <div class="legend-item type-toggle" data-edge="import" onclick="toggleEdgeType('import')"><input type="checkbox" checked><span class="legend-color" style="background:#66a61e"></span> Import</div>
                        <div class="legend-item type-toggle" data-edge="type_reference" onclick="toggleEdgeType('type_reference')"><input type="checkbox"><span class="legend-color" style="background:#e6ab02"></span> TypeRef</div>
                        <div class="legend-item type-toggle disabled" data-edge="structural" onclick="toggleEdgeType('structural')"><input type="checkbox"><span class="legend-color" style="background:#2a2a3e"></span> Struct</div>
                    </div>
                    <div class="legend-group">
                        <div class="legend-group-header">
                            <span>Nodes</span>
                            <div style="display:flex;gap:2px;">
                                <button class="btn-mini" onclick="event.stopPropagation();toggleAllNodeTypes(true)">On</button>
                                <button class="btn-mini" onclick="event.stopPropagation();toggleAllNodeTypes(false)">Off</button>
                            </div>
                        </div>
                        <div class="legend-item type-toggle" data-type="class" onclick="toggleNodeType('class')"><input type="checkbox" checked><span class="legend-color" style="background:#FFD700; height:10px; width:10px; border-radius:50%"></span> Class</div>
                        <div class="legend-item type-toggle" data-type="interface" onclick="toggleNodeType('interface')"><input type="checkbox" checked><span class="legend-color" style="background:#50c878; height:10px; width:10px; border-radius:50%"></span> Interface</div>
                        <div class="legend-item type-toggle" data-type="enum" onclick="toggleNodeType('enum')"><input type="checkbox" checked><span class="legend-color" style="background:#1abc9c; height:8px; width:8px; border-radius:50%"></span> Enum</div>
                        <div class="legend-item type-toggle" data-type="method" onclick="toggleNodeType('method')"><input type="checkbox" checked><span class="legend-color" style="background:#5B9BD5; height:6px; width:6px; border-radius:50%"></span> Method</div>
                        <div class="legend-item type-toggle" data-type="constructor" onclick="toggleNodeType('constructor')"><input type="checkbox" checked><span class="legend-color" style="background:#CD5C5C; height:6px; width:6px; border-radius:50%"></span> Constr</div>
                        <div class="legend-item type-toggle" data-type="field" onclick="toggleNodeType('field')"><input type="checkbox" checked><span class="legend-color" style="background:#9b59b6; height:5px; width:5px; border-radius:50%"></span> Field</div>
                        <div class="legend-item type-toggle" data-type="constant" onclick="toggleNodeType('constant')"><input type="checkbox" checked><span class="legend-color" style="background:#CD5C5C; height:4px; width:4px; border-radius:50%"></span> Const</div>
                        <div class="legend-item type-toggle" data-type="package" onclick="toggleNodeType('package')"><input type="checkbox" checked><span class="legend-color" style="background:#34495e; height:12px; width:12px; border-radius:2px"></span> Package</div>
                    </div>
                </div>
            </div>

            <!-- Node Filter - filter by name within selected packages -->
            <div class="sidebar-section" style="padding:8px;">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                    <span style="font-size:11px;color:#e94560;font-weight:bold;">🔍 Filter Nodes</span>
                    <span id="filter-match-count" style="font-size:10px;color:#666;"></span>
                </div>
                <input type="text" id="node-filter" placeholder="Filter by class/method name..." oninput="filterNodes(this.value)" style="width:100%;padding:6px 8px;background:#2a2a3e;border:1px solid #444;color:#fff;border-radius:4px;font-size:11px;">
                <div style="display:flex;gap:4px;margin-top:4px;">
                    <button class="btn-mini" onclick="clearNodeFilter()" title="Clear filter">Clear</button>
                    <button class="btn-mini" onclick="filterNodes(document.getElementById('node-filter').value, true)" title="Show only exact matches">Exact</button>
                </div>
            </div>

            <!-- Packages - fills remaining space -->
            <div class="sidebar-section packages-section">
                <div class="section-header" style="cursor:default;">
                    <span>📦 Packages ({len(packages)})</span>
                    <div style="display:flex;gap:2px;">
                        <button class="btn-mini" onclick="selectFiltered()" title="Select filtered">✓</button>
                        <button class="btn-mini" onclick="deselectFiltered()" title="Deselect filtered">✗</button>
                    </div>
                </div>
                <input type="text" id="search" placeholder="Filter packages..." oninput="filterPackages(this.value)" style="width:calc(100% - 16px);margin:4px 8px;padding:6px 8px;background:#2a2a3e;border:1px solid #444;color:#fff;border-radius:4px;font-size:11px;">
                <div id="pkg-list">
                    {package_html}
                </div>
            </div>
        </div>
    </div>

    <!-- Settings Modal - Compact 4-column layout -->
    <div id="settings-modal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7); z-index:10000; justify-content:center; align-items:center;">
        <div style="background:#1a1a2e; border-radius:12px; padding:20px; width:720px; max-width:95vw; box-shadow:0 8px 32px rgba(0,0,0,0.5);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                <h2 style="margin:0; color:#fff; font-size:16px;">⚙️ Physics Settings</h2>
                <div style="display:flex; align-items:center; gap:12px;">
                    <select id="settings-template" onchange="applyTemplate(this.value)" style="padding:6px 10px; background:#2a2a3e; border:1px solid #444; color:#fff; border-radius:4px; font-size:12px;">
                        <option value="default">Default</option>
                        <option value="compact">Compact</option>
                        <option value="expanded">Expanded</option>
                        <option value="hierarchy">Hierarchy</option>
                        <option value="custom">Custom</option>
                    </select>
                    <button onclick="closeSettings()" style="background:none; border:none; color:#888; font-size:20px; cursor:pointer;">&times;</button>
                </div>
            </div>

            <!-- 4-column grid for all sliders -->
            <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px;">
                <!-- Force Parameters -->
                <div style="background:#2a2a3e; padding:10px; border-radius:6px;">
                    <label style="color:#888; font-size:10px; display:block; margin-bottom:4px;">Gravity</label>
                    <input type="range" id="settings-gravity" min="-100" max="0" value="-35" oninput="updateSettingValue('grav-val', this.value)" style="width:100%;">
                    <span id="grav-val" style="color:#4a90d9; font-size:11px;">-35</span>
                </div>
                <div style="background:#2a2a3e; padding:10px; border-radius:6px;">
                    <label style="color:#888; font-size:10px; display:block; margin-bottom:4px;">Central</label>
                    <input type="range" id="settings-central" min="0" max="0.3" step="0.01" value="0.1" oninput="updateSettingValue('central-val', this.value)" style="width:100%;">
                    <span id="central-val" style="color:#4a90d9; font-size:11px;">0.1</span>
                </div>
                <div style="background:#2a2a3e; padding:10px; border-radius:6px;">
                    <label style="color:#888; font-size:10px; display:block; margin-bottom:4px;">Spring</label>
                    <input type="range" id="settings-spring" min="0.01" max="0.2" step="0.01" value="0.07" oninput="updateSettingValue('spring-val', this.value)" style="width:100%;">
                    <span id="spring-val" style="color:#4a90d9; font-size:11px;">0.07</span>
                </div>
                <div style="background:#2a2a3e; padding:10px; border-radius:6px;">
                    <label style="color:#888; font-size:10px; display:block; margin-bottom:4px;">Damping</label>
                    <input type="range" id="settings-damping" min="0.1" max="1.0" step="0.05" value="0.5" oninput="updateSettingValue('damping-val', this.value)" style="width:100%;">
                    <span id="damping-val" style="color:#4a90d9; font-size:11px;">0.5</span>
                </div>
                <!-- Edge Lengths -->
                <div style="background:#1f1f35; padding:10px; border-radius:6px;">
                    <label style="color:#888; font-size:10px; display:block; margin-bottom:4px;">Structural</label>
                    <input type="range" id="settings-len-structural" min="20" max="150" value="50" oninput="updateSettingValue('len-structural-val', this.value)" style="width:100%;">
                    <span id="len-structural-val" style="color:#e94560; font-size:11px;">50</span>
                </div>
                <div style="background:#1f1f35; padding:10px; border-radius:6px;">
                    <label style="color:#888; font-size:10px; display:block; margin-bottom:4px;">Inheritance</label>
                    <input type="range" id="settings-len-inherit" min="50" max="250" value="100" oninput="updateSettingValue('len-inherit-val', this.value)" style="width:100%;">
                    <span id="len-inherit-val" style="color:#e94560; font-size:11px;">100</span>
                </div>
                <div style="background:#1f1f35; padding:10px; border-radius:6px;">
                    <label style="color:#888; font-size:10px; display:block; margin-bottom:4px;">Call</label>
                    <input type="range" id="settings-len-call" min="100" max="400" value="180" oninput="updateSettingValue('len-call-val', this.value)" style="width:100%;">
                    <span id="len-call-val" style="color:#e94560; font-size:11px;">180</span>
                </div>
                <div style="background:#1f1f35; padding:10px; border-radius:6px;">
                    <label style="color:#888; font-size:10px; display:block; margin-bottom:4px;">Other</label>
                    <input type="range" id="settings-len-other" min="150" max="500" value="250" oninput="updateSettingValue('len-other-val', this.value)" style="width:100%;">
                    <span id="len-other-val" style="color:#e94560; font-size:11px;">250</span>
                </div>
            </div>

            <div style="display:flex; gap:8px; margin-top:16px;">
                <button onclick="applySettings()" style="flex:1; padding:8px; background:#4a90d9; border:none; color:#fff; border-radius:6px; cursor:pointer; font-weight:bold; font-size:13px;">Apply & Restart</button>
                <button onclick="closeSettings()" style="flex:1; padding:8px; background:#444; border:none; color:#fff; border-radius:6px; cursor:pointer; font-size:13px;">Cancel</button>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        // Data
        const allNodes = {json.dumps(vis_nodes)};
        const allEdges = {json.dumps(vis_edges)};

        // Store original PCoA positions for reset functionality
        // These are the topology-based positions computed from inheritance/call graph
        const originalPositions = {{}};
        allNodes.forEach(n => {{
            if (n.x !== undefined && n.y !== undefined) {{
                originalPositions[n.id] = {{ x: n.x, y: n.y }};
            }}
        }});

        // Edge color scheme by dependency type
        const edgeColors = {{
            'inheritance': {{ color: '#d95f02', highlight: '#ff7f0e', opacity: 0.9 }},
            'implementation': {{ color: '#7570b3', highlight: '#9467bd', opacity: 0.9 }},
            'call': {{ color: '#1b9e77', highlight: '#2ca02c', opacity: 0.6 }},
            'import': {{ color: '#66a61e', highlight: '#8bc34a', opacity: 0.5 }},
            'type_reference': {{ color: '#e6ab02', highlight: '#ffeb3b', opacity: 0.5 }},
            'field_access': {{ color: '#a6761d', highlight: '#d4a574', opacity: 0.5 }},
            'structural': {{ color: '#2a2a3e', highlight: '#3a3a4e', opacity: 0.3 }},  // Dark visible link
            'default': {{ color: '#888888', highlight: '#aaaaaa', opacity: 0.4 }}
        }};

        // Apply colors and physics properties to edges
        allEdges.forEach(edge => {{
            const edgeType = edge.edgeType || 'default';
            const colorInfo = edgeColors[edgeType] || edgeColors['default'];

            if (edgeType === 'structural') {{
                // Structural edges: Priority 1 - strongest, shortest bonds
                // Low opacity/width for visual clarity, but physics enabled
                edge.color = {{ color: colorInfo.color, highlight: colorInfo.highlight, opacity: colorInfo.opacity }};
                edge.width = 0.5;  // Thin but visible
                edge.length = 50;  // Priority 1: Shortest spring (keeps solar systems intact)
                edge.smooth = false;
                edge.arrows = {{ to: {{ enabled: false }} }};  // No arrow for structural
                edge.dashes = false;
            }} else {{
                // Non-structural edges use type-based lengths from Python
                edge.color = {{ color: colorInfo.color, highlight: colorInfo.highlight, opacity: colorInfo.opacity }};
                edge.width = edgeType === 'inheritance' || edgeType === 'implementation' ? 1.5 : 0.8;
                // Length is set per edge in Python based on edge_length_by_type
            }}
        }});

        // vis.js datasets
        const nodes = new vis.DataSet(allNodes);
        const edges = new vis.DataSet(allEdges);

        // Package visibility state
        const packageVisible = {{}};
        allNodes.forEach(n => packageVisible[n.package] = true);

        // Node type visibility state (import hidden by default)
        const typeVisible = {{
            'class': true,
            'interface': true,
            'enum': true,
            'method': true,
            'constructor': true,
            'field': true,
            'constant': true,
            'function': true,
            'import': false,
            'package': true
        }};

        // Edge type visibility state (type_reference hidden by default - too cluttered)
        const edgeTypeVisible = {{
            'inheritance': true,
            'implementation': true,
            'call': true,
            'import': true,
            'type_reference': false,
            'structural': false,  // Hidden by default - toggle to see member halos
            'field_access': true,
            'default': true
        }};

        // Node name filter state
        let nodeNameFilter = '';
        let nodeNameFilterExact = false;

        // Filter nodes by name (within selected packages)
        function filterNodes(query, exact = false) {{
            nodeNameFilter = query.toLowerCase().trim();
            nodeNameFilterExact = exact;
            updateVisibility();

            // Update match count display
            const countEl = document.getElementById('filter-match-count');
            if (nodeNameFilter) {{
                const matches = allNodes.filter(n => {{
                    const name = (n.label || n.id).toLowerCase();
                    return nodeNameFilterExact ? name === nodeNameFilter : name.includes(nodeNameFilter);
                }}).length;
                countEl.textContent = `(${{matches}} matches)`;
            }} else {{
                countEl.textContent = '';
            }}
        }}

        function clearNodeFilter() {{
            document.getElementById('node-filter').value = '';
            nodeNameFilter = '';
            nodeNameFilterExact = false;
            document.getElementById('filter-match-count').textContent = '';
            updateVisibility();
        }}

        // Check if node name matches current filter
        function nodeMatchesFilter(node) {{
            if (!nodeNameFilter) return true;
            const name = (node.label || node.id).toLowerCase();
            return nodeNameFilterExact ? name === nodeNameFilter : name.includes(nodeNameFilter);
        }}

        // Toggle node type visibility from legend
        function toggleNodeType(nodeType) {{
            typeVisible[nodeType] = !typeVisible[nodeType];
            const item = document.querySelector(`.type-toggle[data-type="${{nodeType}}"]`);
            if (item) {{
                const cb = item.querySelector('input');
                cb.checked = typeVisible[nodeType];
                item.classList.toggle('disabled', !typeVisible[nodeType]);
            }}
            updateVisibility();
        }}

        // Toggle edge type visibility from legend
        function toggleEdgeType(edgeType) {{
            edgeTypeVisible[edgeType] = !edgeTypeVisible[edgeType];
            const item = document.querySelector(`.type-toggle[data-edge="${{edgeType}}"]`);
            if (item) {{
                const cb = item.querySelector('input');
                cb.checked = edgeTypeVisible[edgeType];
                item.classList.toggle('disabled', !edgeTypeVisible[edgeType]);
            }}
            updateEdgeVisibility();
        }}

        // Update edge visibility based on edge type toggles
        function updateEdgeVisibility() {{
            const updates = [];
            allEdges.forEach(edge => {{
                const edgeType = edge.edgeType || 'default';
                const visible = edgeTypeVisible[edgeType] !== false;
                updates.push({{ id: edge.id, hidden: !visible }});
            }});
            edges.update(updates);
        }}

        // Toggle collapsible sidebar sections
        function toggleSection(headerEl) {{
            const section = headerEl.parentElement;
            const content = section.querySelector('.section-content');
            const icon = headerEl.querySelector('.collapse-icon');
            const isCollapsed = section.classList.toggle('collapsed');
            content.style.display = isCollapsed ? 'none' : '';
            icon.textContent = isCollapsed ? '▶' : '▼';
        }}

        // Global toggle: all edge types on/off
        function toggleAllEdges(visible) {{
            Object.keys(edgeTypeVisible).forEach(edgeType => {{
                edgeTypeVisible[edgeType] = visible;
                const item = document.querySelector(`.type-toggle[data-edge="${{edgeType}}"]`);
                if (item) {{
                    const cb = item.querySelector('input');
                    cb.checked = visible;
                    item.classList.toggle('disabled', !visible);
                }}
            }});
            updateEdgeVisibility();
        }}

        // Global toggle: all node types on/off
        function toggleAllNodeTypes(visible) {{
            Object.keys(typeVisible).forEach(nodeType => {{
                typeVisible[nodeType] = visible;
                const item = document.querySelector(`.type-toggle[data-type="${{nodeType}}"]`);
                if (item) {{
                    const cb = item.querySelector('input');
                    cb.checked = visible;
                    item.classList.toggle('disabled', !visible);
                }}
            }});
            updateVisibility();
        }}

        // Create network
        const container = document.getElementById('graph');
        const data = {{ nodes, edges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                scaling: {{
                    min: 6,
                    max: 50,
                    label: {{
                        enabled: true,
                        min: 8,
                        max: 16,
                        drawThreshold: 12,  // Only show labels when zoomed in
                        maxVisible: 20      // Max font size when fully zoomed
                    }}
                }},
                font: {{
                    size: 11,
                    color: '#ffffff',
                    strokeWidth: 2,
                    strokeColor: '#1a1a2e',
                    face: 'arial',
                    vadjust: -2
                }},
                borderWidth: 2,
                borderWidthSelected: 4
            }},
            edges: {{
                smooth: {{
                    type: 'continuous',
                    roundness: 0.3
                }},
                arrows: {{
                    to: {{ enabled: true, scaleFactor: 0.5 }}
                }},
                selectionWidth: 2
            }},
            layout: {{
                improvedLayout: false  // Disable vis.js auto-compaction
            }},
            physics: {{
                enabled: {'true' if physics_enabled else 'false'},
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: {gravitational_constant},
                    centralGravity: {central_gravity},
                    springLength: {spring_length},
                    springConstant: {spring_constant},
                    damping: 0.5,
                    avoidOverlap: 0.8
                }},
                maxVelocity: 40,
                minVelocity: 0.1,  // Low threshold to preserve spread
                stabilization: {{
                    enabled: {stabilization_iterations} > 0,
                    iterations: {stabilization_iterations},
                    updateInterval: 15,
                    fit: false  // Don't auto-fit during stabilization
                }}
            }},
            interaction: {{
                tooltipDelay: 100,
                hideEdgesOnDrag: true,
                hideEdgesOnZoom: true,
                hover: true,
                multiselect: true,
                navigationButtons: false,
                keyboard: true
            }},
            // Solar system analogy: Classes=yellow stars, Constructors=red dwarfs, Methods=blue planets
            groups: {{
                "class": {{
                    color: {{ background: "#FFD700", border: "#DAA520" }},  // Gold (big stars)
                    font: {{ size: 12, color: '#1a1a2e', strokeWidth: 2, strokeColor: '#ffffff' }}
                }},
                "interface": {{
                    color: {{ background: "#50c878", border: "#2ca25f" }},  // Green (nebulae)
                    font: {{ size: 12, color: '#ffffff', strokeWidth: 2, strokeColor: '#1a1a2e' }}
                }},
                "enum": {{
                    color: {{ background: "#1abc9c", border: "#16a085" }},  // Teal (gas giants)
                    font: {{ size: 11, color: '#ffffff', strokeWidth: 2, strokeColor: '#1a1a2e' }}
                }},
                "method": {{
                    color: {{ background: "#5B9BD5", border: "#4080B0" }},  // Blue (planets)
                    font: {{ size: 0 }},  // Hidden - show on hover only
                    scaling: {{ min: 3, max: 10 }}
                }},
                "constructor": {{
                    color: {{ background: "#CD5C5C", border: "#A94442" }},  // Indian red (red dwarf stars)
                    font: {{ size: 0 }},  // Hidden
                    scaling: {{ min: 4, max: 12 }}
                }},
                "field": {{
                    color: {{ background: "#9b59b6", border: "#8e44ad" }},  // Purple (moons)
                    font: {{ size: 0 }},  // Hidden
                    scaling: {{ min: 2, max: 8 }}
                }},
                "constant": {{
                    color: {{ background: "#CD5C5C", border: "#A94442" }},  // Indian red (asteroids)
                    font: {{ size: 0 }},  // Hidden
                    scaling: {{ min: 2, max: 6 }}
                }},
                "function": {{
                    color: {{ background: "#5B9BD5", border: "#4080B0" }},  // Blue (planets)
                    font: {{ size: 0 }}  // Hidden
                }},
                "import": {{
                    color: {{ background: "#95a5a6", border: "#7f8c8d" }},
                    font: {{ size: 0 }},  // Hidden
                    scaling: {{ min: 1, max: 4 }}
                }},
                "package": {{
                    shape: "box",
                    color: {{ background: "#34495e", border: "#2c3e50" }},
                    font: {{ size: 14, color: '#ffffff', strokeWidth: 3, strokeColor: '#1a1a2e' }}
                }}
            }}
        }};

        // Simple loading - CSS animation handles the visual feedback
        function completeLoading() {{
            document.querySelector('.loading-text').textContent = 'Done!';
            document.getElementById('progress').classList.add('complete');
            setTimeout(() => {{
                document.getElementById('loading').classList.add('hidden');
            }}, 400);
        }}

        document.querySelector('.loading-text').textContent = 'Creating network...';

        const network = new vis.Network(container, data, options);
        let physicsEnabled = {'true' if physics_enabled else 'false'};

        // Apply initial edge type visibility (hides type_reference by default)
        updateEdgeVisibility();

        document.querySelector('.loading-text').textContent = physicsEnabled ? 'Stabilizing layout...' : 'Rendering graph...';

        // Auto-hide layout info after 5 seconds
        setTimeout(() => {{
            const info = document.getElementById('layout-info');
            if (info) info.style.display = 'none';
        }}, 5000);

        // Stabilization progress (text only, CSS handles animation)
        network.on("stabilizationProgress", function(params) {{
            const pct = Math.round((params.iterations / params.total) * 100);
            document.querySelector('.loading-text').textContent = 'Stabilizing layout... ' + pct + '%';
        }});

        network.on("stabilizationIterationsDone", function() {{
            completeLoading();
            network.setOptions({{ physics: false }});
            physicsEnabled = false;
        }});

        // For non-physics graphs, hide loading immediately after drawing
        if (!physicsEnabled) {{
            network.once("afterDrawing", function() {{
                completeLoading();
            }});
        }}

        // Toggle physics
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}

        // Restart simulation (optionally randomize positions)
        // When not randomizing, reset to original PCoA positions (topology-based)
        function restartSimulation(randomize) {{
            if (randomize) {{
                // Randomize node positions before restarting
                const positions = network.getPositions();
                const updates = [];
                const canvas = network.getViewPosition();
                const scale = network.getScale();
                const spreadX = 800 / scale;
                const spreadY = 600 / scale;

                Object.keys(positions).forEach(nodeId => {{
                    updates.push({{
                        id: nodeId,
                        x: canvas.x + (Math.random() - 0.5) * spreadX,
                        y: canvas.y + (Math.random() - 0.5) * spreadY
                    }});
                }});
                nodes.update(updates);
            }} else {{
                // Reset to original PCoA positions (topology-based layout)
                const updates = [];
                allNodes.forEach(n => {{
                    const orig = originalPositions[n.id];
                    if (orig) {{
                        updates.push({{ id: n.id, x: orig.x, y: orig.y }});
                    }}
                }});
                nodes.update(updates);
            }}

            // Enable physics and restart stabilization
            physicsEnabled = true;
            network.setOptions({{ physics: {{ enabled: true }} }});
            network.stabilize(500);  // Run stabilization for 500 iterations
        }}

        // Reset to original PCoA positions (topology-based layout)
        // This restores the initial positions computed from inheritance/call graph distances
        function resetToPCoA(runPhysics) {{
            const updates = [];
            allNodes.forEach(n => {{
                const orig = originalPositions[n.id];
                if (orig) {{
                    updates.push({{ id: n.id, x: orig.x, y: orig.y }});
                }}
            }});
            nodes.update(updates);

            if (runPhysics) {{
                // Optionally run physics to refine the layout
                physicsEnabled = true;
                network.setOptions({{ physics: {{ enabled: true }} }});
                network.stabilize(300);
            }} else {{
                // Just reset positions, no physics
                network.setOptions({{ physics: {{ enabled: false }} }});
                physicsEnabled = false;
                network.fit({{ animation: {{ duration: 500 }} }});
            }}
        }}

        // ============ Settings Modal Functions ============

        // Current physics settings (can be modified via Settings panel)
        // Defaults match the original working parameters from Python
        const physicsSettings = {{
            gravitationalConstant: -35,
            centralGravity: 0.1,
            springConstant: 0.07,
            damping: 0.5,
            edgeLengths: {{
                structural: 50,
                inheritance: 100,
                call: 180,
                other: 250
            }}
        }};

        // Template presets (all tested for stability)
        const settingsTemplates = {{
            default: {{ gravitationalConstant: -35, centralGravity: 0.1, springConstant: 0.07, damping: 0.5, edgeLengths: {{ structural: 50, inheritance: 100, call: 180, other: 250 }} }},
            compact: {{ gravitationalConstant: -25, centralGravity: 0.15, springConstant: 0.1, damping: 0.6, edgeLengths: {{ structural: 30, inheritance: 60, call: 100, other: 150 }} }},
            expanded: {{ gravitationalConstant: -50, centralGravity: 0.05, springConstant: 0.04, damping: 0.4, edgeLengths: {{ structural: 80, inheritance: 150, call: 280, other: 400 }} }},
            hierarchy: {{ gravitationalConstant: -35, centralGravity: 0.08, springConstant: 0.08, damping: 0.5, edgeLengths: {{ structural: 40, inheritance: 60, call: 200, other: 300 }} }}
        }};

        function openSettings() {{
            // Sync UI with current settings
            document.getElementById('settings-gravity').value = physicsSettings.gravitationalConstant;
            document.getElementById('grav-val').textContent = physicsSettings.gravitationalConstant;
            document.getElementById('settings-central').value = physicsSettings.centralGravity;
            document.getElementById('central-val').textContent = physicsSettings.centralGravity;
            document.getElementById('settings-spring').value = physicsSettings.springConstant;
            document.getElementById('spring-val').textContent = physicsSettings.springConstant;
            document.getElementById('settings-damping').value = physicsSettings.damping;
            document.getElementById('damping-val').textContent = physicsSettings.damping;
            document.getElementById('settings-len-structural').value = physicsSettings.edgeLengths.structural;
            document.getElementById('len-structural-val').textContent = physicsSettings.edgeLengths.structural;
            document.getElementById('settings-len-inherit').value = physicsSettings.edgeLengths.inheritance;
            document.getElementById('len-inherit-val').textContent = physicsSettings.edgeLengths.inheritance;
            document.getElementById('settings-len-call').value = physicsSettings.edgeLengths.call;
            document.getElementById('len-call-val').textContent = physicsSettings.edgeLengths.call;
            document.getElementById('settings-len-other').value = physicsSettings.edgeLengths.other;
            document.getElementById('len-other-val').textContent = physicsSettings.edgeLengths.other;

            document.getElementById('settings-modal').style.display = 'flex';
        }}

        function closeSettings() {{
            document.getElementById('settings-modal').style.display = 'none';
        }}

        function updateSettingValue(spanId, value) {{
            document.getElementById(spanId).textContent = value;
            document.getElementById('settings-template').value = 'custom';
        }}

        function applyTemplate(templateName) {{
            if (templateName === 'custom') return;
            const t = settingsTemplates[templateName];
            if (!t) return;

            document.getElementById('settings-gravity').value = t.gravitationalConstant;
            document.getElementById('grav-val').textContent = t.gravitationalConstant;
            document.getElementById('settings-central').value = t.centralGravity;
            document.getElementById('central-val').textContent = t.centralGravity;
            document.getElementById('settings-spring').value = t.springConstant;
            document.getElementById('spring-val').textContent = t.springConstant;
            document.getElementById('settings-damping').value = t.damping;
            document.getElementById('damping-val').textContent = t.damping;
            document.getElementById('settings-len-structural').value = t.edgeLengths.structural;
            document.getElementById('len-structural-val').textContent = t.edgeLengths.structural;
            document.getElementById('settings-len-inherit').value = t.edgeLengths.inheritance;
            document.getElementById('len-inherit-val').textContent = t.edgeLengths.inheritance;
            document.getElementById('settings-len-call').value = t.edgeLengths.call;
            document.getElementById('len-call-val').textContent = t.edgeLengths.call;
            document.getElementById('settings-len-other').value = t.edgeLengths.other;
            document.getElementById('len-other-val').textContent = t.edgeLengths.other;
        }}

        function applySettings() {{
            // Read values from UI
            physicsSettings.gravitationalConstant = parseFloat(document.getElementById('settings-gravity').value);
            physicsSettings.centralGravity = parseFloat(document.getElementById('settings-central').value);
            physicsSettings.springConstant = parseFloat(document.getElementById('settings-spring').value);
            physicsSettings.damping = parseFloat(document.getElementById('settings-damping').value);
            physicsSettings.edgeLengths.structural = parseInt(document.getElementById('settings-len-structural').value);
            physicsSettings.edgeLengths.inheritance = parseInt(document.getElementById('settings-len-inherit').value);
            physicsSettings.edgeLengths.call = parseInt(document.getElementById('settings-len-call').value);
            physicsSettings.edgeLengths.other = parseInt(document.getElementById('settings-len-other').value);

            // Update edge lengths
            const edgeUpdates = [];
            allEdges.forEach(edge => {{
                let newLength;
                switch (edge.edgeType) {{
                    case 'structural': newLength = physicsSettings.edgeLengths.structural; break;
                    case 'inheritance':
                    case 'implementation': newLength = physicsSettings.edgeLengths.inheritance; break;
                    case 'call': newLength = physicsSettings.edgeLengths.call; break;
                    default: newLength = physicsSettings.edgeLengths.other; break;
                }}
                edgeUpdates.push({{ id: edge.id, length: newLength }});
            }});
            edges.update(edgeUpdates);

            // Update physics options
            network.setOptions({{
                physics: {{
                    enabled: true,
                    forceAtlas2Based: {{
                        gravitationalConstant: physicsSettings.gravitationalConstant,
                        centralGravity: physicsSettings.centralGravity,
                        springConstant: physicsSettings.springConstant,
                        damping: physicsSettings.damping
                    }}
                }}
            }});

            // Reset to PCoA and restart
            restartSimulation(false);
            closeSettings();
        }}

        // Close modal on backdrop click
        document.getElementById('settings-modal').addEventListener('click', function(e) {{
            if (e.target === this) closeSettings();
        }});

        // ============ End Settings Modal Functions ============

        // Package filtering
        function togglePackage(pkg, visible) {{
            packageVisible[pkg] = visible;
            updateVisibility();
        }}

        // Build parent class map from structural edges (member -> class)
        // This is used to hide members when their parent class is hidden
        const parentClassMap = {{}};
        allEdges.forEach(e => {{
            if (e.edgeType === 'structural') {{
                // Structural edges: from=member, to=class
                parentClassMap[e.from] = e.to;
            }}
        }});

        // Helper: check if a node's parent class (if any) is visible
        function isParentClassVisible(nodeId) {{
            const parentId = parentClassMap[nodeId];
            if (!parentId) return true;  // No parent = always visible (it's a class/interface itself)
            const parentNode = allNodes.find(n => n.id === parentId);
            if (!parentNode) return true;  // Parent not found = assume visible
            // Check if parent class passes all visibility filters
            return packageVisible[parentNode.package] &&
                   typeVisible[parentNode.group] &&
                   !hiddenNodes.has(parentId);
        }}

        function updateVisibility() {{
            // Update node visibility using update() to preserve positions and not restart simulation
            // Key rules:
            // - Classes/interfaces: visible if their package is selected AND matches name filter
            // - Members (methods, fields): visible if their PARENT CLASS is visible AND matches name filter
            const nodeUpdates = [];
            let visibleCount = 0;
            allNodes.forEach(n => {{
                const hasParent = parentClassMap[n.id] !== undefined;
                const typeVis = typeVisible[n.group] !== false;
                const notHidden = !hiddenNodes.has(n.id);
                const matchesFilter = nodeMatchesFilter(n);
                let shouldBeVisible;

                if (hasParent) {{
                    // Members: visibility follows parent class + name filter
                    shouldBeVisible = isParentClassVisible(n.id) && typeVis && notHidden && matchesFilter;
                }} else {{
                    // Classes/interfaces/enums: visibility based on package selection + name filter
                    shouldBeVisible = packageVisible[n.package] && typeVis && notHidden && matchesFilter;
                }}

                nodeUpdates.push({{ id: n.id, hidden: !shouldBeVisible }});
                if (shouldBeVisible) visibleCount++;
            }});
            nodes.update(nodeUpdates);

            // Build set of visible node IDs for edge filtering
            const visibleIds = new Set(
                allNodes.filter(n => {{
                    const hasParent = parentClassMap[n.id] !== undefined;
                    const typeVis = typeVisible[n.group] !== false;
                    const notHidden = !hiddenNodes.has(n.id);
                    const matchesFilter = nodeMatchesFilter(n);
                    if (hasParent) {{
                        return isParentClassVisible(n.id) && typeVis && notHidden && matchesFilter;
                    }} else {{
                        return packageVisible[n.package] && typeVis && notHidden && matchesFilter;
                    }}
                }}).map(n => n.id)
            );

            // Update edge visibility - show only edges where both endpoints are visible
            const edgeUpdates = [];
            allEdges.forEach(e => {{
                const bothVisible = visibleIds.has(e.from) && visibleIds.has(e.to);
                const typeVis = edgeTypeVisible[e.edgeType || 'default'] !== false;
                edgeUpdates.push({{ id: e.id, hidden: !bothVisible || !typeVis }});
            }});
            edges.update(edgeUpdates);

            document.getElementById('visible-count').textContent = visibleCount;
        }}

        // Track hidden nodes (moved up for reference)
        const hiddenNodes = new Set();

        // Select/Deselect only filtered (visible in list) packages
        function selectFiltered() {{
            document.querySelectorAll('.pkg-item').forEach(item => {{
                if (item.style.display !== 'none') {{
                    const pkg = item.dataset.pkg;
                    const cb = item.querySelector('input');
                    cb.checked = true;
                    packageVisible[pkg] = true;
                }}
            }});
            updateVisibility();
        }}

        function deselectFiltered() {{
            document.querySelectorAll('.pkg-item').forEach(item => {{
                if (item.style.display !== 'none') {{
                    const pkg = item.dataset.pkg;
                    const cb = item.querySelector('input');
                    cb.checked = false;
                    packageVisible[pkg] = false;
                }}
            }});
            updateVisibility();
        }}

        function filterPackages(query) {{
            const q = query.toLowerCase();
            document.querySelectorAll('.pkg-item').forEach(item => {{
                const pkg = item.dataset.pkg.toLowerCase();
                item.style.display = pkg.includes(q) ? 'flex' : 'none';
            }});
        }}

        // Build dependency index for parent/child expansion
        // Edge direction: from -> to means "from" USES/DEPENDS ON "to"
        // Parent = what I depend on (targets of my outgoing edges)
        // Child = what depends on me (sources of edges pointing to me)
        const nodeIndex = {{}};  // Quick lookup for nodes
        allNodes.forEach(n => nodeIndex[n.id] = n);

        const dependsOn = {{}};  // nodeId -> [nodeIds that this node depends on] (parents)
        const dependedBy = {{}};  // nodeId -> [nodeIds that depend on this node] (children)

        allEdges.forEach(edge => {{
            if (edge.edgeType !== 'structural') {{
                // from depends on to (from -> to)
                if (!dependsOn[edge.from]) dependsOn[edge.from] = [];
                if (!dependedBy[edge.to]) dependedBy[edge.to] = [];
                dependsOn[edge.from].push(edge.to);  // to is a parent of from
                dependedBy[edge.to].push(edge.from);  // from is a child of to
            }}
        }});

        // Expand to show parent packages (nodes that visible nodes DEPEND ON)
        function expandParents() {{
            const currentVisibleIds = new Set(nodes.getIds());
            const newPackages = new Set();
            let newNodeCount = 0;

            currentVisibleIds.forEach(nodeId => {{
                const parents = dependsOn[nodeId] || [];
                parents.forEach(parentId => {{
                    const parentNode = nodeIndex[parentId];
                    if (parentNode) {{
                        const pkg = parentNode.package;
                        if (!packageVisible[pkg]) {{
                            newPackages.add(pkg);
                        }}
                        if (!currentVisibleIds.has(parentId)) {{
                            newNodeCount++;
                        }}
                    }}
                }});
            }});

            if (newPackages.size === 0) {{
                console.log('No new parent packages to add. All dependencies are in visible packages.');
                return;
            }}

            // Enable these packages
            newPackages.forEach(pkg => {{
                packageVisible[pkg] = true;
                const item = document.querySelector(`.pkg-item[data-pkg="${{CSS.escape(pkg)}}"]`);
                if (item) {{
                    const cb = item.querySelector('input');
                    if (cb) cb.checked = true;
                }}
            }});

            updateVisibility();
            console.log(`Added ${{newPackages.size}} parent packages (${{newNodeCount}} new nodes):`, Array.from(newPackages));
        }}

        // Expand to show child packages (nodes that DEPEND ON visible nodes)
        function expandChildren() {{
            const currentVisibleIds = new Set(nodes.getIds());
            const newPackages = new Set();
            let newNodeCount = 0;

            currentVisibleIds.forEach(nodeId => {{
                const children = dependedBy[nodeId] || [];
                children.forEach(childId => {{
                    const childNode = nodeIndex[childId];
                    if (childNode) {{
                        const pkg = childNode.package;
                        if (!packageVisible[pkg]) {{
                            newPackages.add(pkg);
                        }}
                        if (!currentVisibleIds.has(childId)) {{
                            newNodeCount++;
                        }}
                    }}
                }});
            }});

            if (newPackages.size === 0) {{
                console.log('No new child packages to add. Nothing else depends on visible nodes.');
                return;
            }}

            // Enable these packages
            newPackages.forEach(pkg => {{
                packageVisible[pkg] = true;
                const item = document.querySelector(`.pkg-item[data-pkg="${{CSS.escape(pkg)}}"]`);
                if (item) {{
                    const cb = item.querySelector('input');
                    if (cb) cb.checked = true;
                }}
            }});

            updateVisibility();
            console.log(`Added ${{newPackages.size}} child packages (${{newNodeCount}} new nodes):`, Array.from(newPackages));
        }}

        // Export to PNG
        function exportPNG() {{
            const canvas = document.querySelector('#graph canvas');
            if (canvas) {{
                const link = document.createElement('a');
                link.download = 'dependency-graph.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }}
        }}

        // Hide selected nodes
        function hideSelected() {{
            const selected = network.getSelectedNodes();
            if (selected.length === 0) {{
                console.log('No nodes selected. Click on nodes to select them first.');
                return;
            }}

            selected.forEach(nodeId => hiddenNodes.add(nodeId));
            updateVisibility();
            network.unselectAll();
            console.log(`Hidden ${{selected.length}} nodes. Total hidden: ${{hiddenNodes.size}}`);
        }}

        // Show all hidden nodes
        function showAll() {{
            hiddenNodes.clear();
            updateVisibility();
            console.log('All nodes restored');
        }}

        // Node click to select (for multi-select, hold Ctrl)
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                console.log('Selected:', node.label, '(' + node.group + ')');
            }}
        }});

        // Fullscreen toggle
        function toggleFullscreen() {{
            document.body.classList.toggle('fullscreen');
        }}

        // Help toggle
        function toggleHelp() {{
            document.getElementById('help-panel').classList.toggle('visible');
            document.getElementById('help-overlay').classList.toggle('visible');
        }}
    </script>
</body>
</html>'''


def get_optimal_renderer(node_count: int) -> str:
    """
    Determine the optimal renderer based on graph size.

    Returns:
        'vis' for large graphs (>2000 nodes), 'd3' for smaller graphs
    """
    if node_count > VIS_RENDERER_THRESHOLD:
        return 'vis'
    return 'd3'


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

        # Build short name to qualified name mapping for name resolution
        short_to_qualified: Dict[str, str] = {}
        for sid in sym_index:
            short = sid.split('.')[-1]
            if short not in short_to_qualified or len(sid) < len(short_to_qualified[short]):
                short_to_qualified[short] = sid

        def resolve_name(name: str) -> Optional[str]:
            """Resolve a dependency name to a symbol index key."""
            if name in sym_index:
                return name
            if name in short_to_qualified:
                return short_to_qualified[name]
            # Try extracting parts (module.Class.method -> Class)
            parts = name.split('.')
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i]
                if part in short_to_qualified:
                    return short_to_qualified[part]
            return None

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
                # Use name resolution for class-level matching
                source = resolve_name(dep.source)
                target = resolve_name(dep.target)
                if source and target and source in sym_index and target in sym_index:
                    i, j = sym_index[source], sym_index[target]
                    if i != j:
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
        /* Fullscreen mode */
        body.fullscreen #sidebar {{ display: none !important; }}
        body.fullscreen #row-selector {{ display: none !important; }}
        body.fullscreen #container {{ width: 100vw; height: 100vh; }}
        body.fullscreen #matrix-container {{ width: 100%; flex: 1; }}
        /* Loading overlay */
        #loading {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(26, 26, 46, 0.95);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }}
        #loading.hidden {{ display: none; }}
        .spinner {{
            width: 60px; height: 60px;
            border: 4px solid rgba(233, 69, 96, 0.3);
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .loading-text {{ margin-top: 20px; font-size: 14px; color: #888; }}
        .progress-bar {{
            width: 200px; height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px; margin-top: 15px; overflow: hidden;
            position: relative;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e94560, #f5a623, #e94560);
            background-size: 200% 100%;
            width: 100%;
            animation: shimmer 1.5s ease-in-out infinite;
        }}
        .progress-fill.complete {{ background: #50c878; animation: none; }}
        @keyframes shimmer {{ 0% {{ background-position: 200% 0; }} 100% {{ background-position: -200% 0; }} }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #matrix-container {{
            flex: 1;
            overflow: auto;
            padding: 20px;
        }}
        /* Row selector */
        #row-selector {{
            width: 200px;
            background: #16213e;
            border-right: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        #row-selector-header {{
            padding: 10px;
            background: #0f3460;
        }}
        #row-selector-header h3 {{
            margin: 0 0 8px 0;
            font-size: 12px;
            color: #e94560;
        }}
        #row-search {{
            width: 100%;
            padding: 8px;
            border: none;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 11px;
        }}
        #row-search::placeholder {{ color: #666; }}
        #row-list {{
            flex: 1;
            overflow-y: auto;
            padding: 5px;
        }}
        .row-item {{
            display: flex;
            align-items: center;
            padding: 4px 6px;
            font-size: 10px;
            cursor: pointer;
            border-radius: 3px;
        }}
        .row-item:hover {{ background: rgba(255,255,255,0.1); }}
        .row-item.selected {{ background: rgba(233, 69, 96, 0.3); }}
        .row-item.hidden {{ display: none; }}
        .row-item input {{ margin-right: 6px; }}
        .row-item-name {{
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .row-selector-actions {{
            padding: 8px;
            display: flex;
            gap: 5px;
            border-top: 1px solid #0f3460;
        }}
        .row-selector-actions .btn {{ flex: 1; font-size: 10px; padding: 6px; }}
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
            margin-bottom: 5px;
        }}
        .col-label {{
            width: 24px;
            min-width: 24px;
            height: 120px;
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            padding: 3px 2px;
            font-size: 10px;
            color: #ccc;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            box-sizing: border-box;
        }}
        .col-label:hover {{
            color: #e94560;
            background: rgba(255,255,255,0.1);
        }}
        .fullscreen-btn {{
            position: fixed;
            top: 10px;
            right: 320px;
            z-index: 1000;
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
        /* Help panel */
        #help-panel {{
            position: fixed;
            top: 0; right: -400px;
            width: 400px;
            height: 100vh;
            background: #16213e;
            border-left: 2px solid #e94560;
            padding: 20px;
            overflow-y: auto;
            transition: right 0.3s ease;
            z-index: 2000;
        }}
        #help-panel.visible {{ right: 0; }}
        #help-panel h2 {{
            color: #e94560;
            font-size: 18px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #help-panel .close-btn {{
            background: none; border: none; color: #888; font-size: 24px; cursor: pointer;
        }}
        #help-panel .close-btn:hover {{ color: #fff; }}
        .help-section {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .help-section h3 {{ color: #4fc3f7; font-size: 14px; margin-bottom: 10px; }}
        .help-section p {{ color: #ccc; font-size: 13px; line-height: 1.6; margin-bottom: 8px; }}
        .help-section ul {{ margin: 8px 0 0 20px; color: #aaa; font-size: 12px; }}
        .help-section li {{ margin-bottom: 6px; }}
        .help-section code {{ background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 3px; color: #f5a623; }}
        .help-overlay {{
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.5); opacity: 0; pointer-events: none;
            transition: opacity 0.3s; z-index: 1999;
        }}
        .help-overlay.visible {{ opacity: 1; pointer-events: auto; }}
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner"></div>
        <div class="loading-text">Building matrix...</div>
        <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
    </div>
    <div class="help-overlay" id="help-overlay" onclick="toggleHelp()"></div>
    <div id="container">
        <div id="row-selector">
            <div id="row-selector-header">
                <h3>📋 Row Selector</h3>
                <input type="text" id="row-search" placeholder="Search rows..." oninput="filterRows()">
            </div>
            <div id="row-list"></div>
            <div class="row-selector-actions">
                <button class="btn" onclick="selectAllRows()">All</button>
                <button class="btn" onclick="deselectAllRows()">None</button>
                <button class="btn" onclick="selectDependents()">↓ Deps</button>
            </div>
        </div>
        <div id="matrix-container">
            <div class="controls">
                <button class="btn" onclick="exportCSV()">📄 CSV</button>
                <button class="btn" onclick="exportPNG()">🖼️ PNG</button>
                <button class="btn" onclick="exportSVG()">📐 SVG</button>
                <button class="btn" onclick="highlightCycles()">🔄 Cycles</button>
                <button class="btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
                <button class="btn" onclick="toggleHelp()">❓ Help</button>
            </div>
    <div id="help-panel">
        <h2>
            <span>📊 DSM Matrix Guide</span>
            <button class="close-btn" onclick="toggleHelp()">×</button>
        </h2>
        <div class="help-section">
            <h3>What is a DSM?</h3>
            <p>A Dependency Structure Matrix (DSM) is a compact visual representation of dependencies between code elements. Each cell shows whether the row element depends on the column element.</p>
        </div>
        <div class="help-section">
            <h3>Reading the Matrix</h3>
            <ul>
                <li><strong>Rows</strong> represent source elements (who depends)</li>
                <li><strong>Columns</strong> represent target elements (what is depended on)</li>
                <li><strong>Cell value</strong> shows the number of dependencies</li>
                <li><strong>Diagonal cells</strong> represent self-references (always dark)</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Color Codes</h3>
            <ul>
                <li><code style="background:#333">Dark gray</code> - Diagonal (self)</li>
                <li><code style="background:#3498db">Blue</code> - Has dependency</li>
                <li><code style="background:#1abc9c">Teal</code> - High coupling (5+ deps)</li>
                <li><code style="background:#e74c3c">Red</code> - Cyclic dependency (bi-directional)</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Cyclic Dependencies</h3>
            <p>Red cells indicate cyclic dependencies - where A depends on B and B depends on A. These are often problematic and worth investigating.</p>
            <p>Use the <strong>Show Cycles</strong> button to highlight only cyclic dependencies.</p>
        </div>
        <div class="help-section">
            <h3>Interactivity</h3>
            <ul>
                <li><strong>Hover over cells</strong> to see detailed connection info</li>
                <li><strong>Click row/column labels</strong> to see element details</li>
                <li><strong>Export</strong> to CSV, PNG, or SVG for reports</li>
            </ul>
        </div>
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
        let selectedRows = new Set(); // Track selected rows
        let filterMode = 'all'; // 'all', 'selected', or 'deps'

        // Initialize selected rows (all selected by default)
        for (let i = 0; i < data.size; i++) {{
            selectedRows.add(i);
        }}

        // Simple loading - CSS animation handles the visual feedback
        function completeLoading() {{
            document.querySelector('.loading-text').textContent = 'Done!';
            document.getElementById('progress').classList.add('complete');
            setTimeout(() => {{
                document.getElementById('loading').classList.add('hidden');
            }}, 400);
        }}

        // Initialize row selector panel
        function initRowSelector() {{
            const rowList = document.getElementById('row-list');
            let html = '';
            for (let i = 0; i < data.size; i++) {{
                const checked = selectedRows.has(i) ? 'checked' : '';
                html += `<div class="row-item ${{selectedRows.has(i) ? 'selected' : ''}}" data-index="${{i}}" onclick="toggleRow(${{i}})">
                    <input type="checkbox" ${{checked}} onclick="event.stopPropagation(); toggleRow(${{i}})">
                    <span class="row-item-name" title="${{data.full_labels[i]}}">${{data.labels[i]}}</span>
                </div>`;
            }}
            rowList.innerHTML = html;
        }}

        // Filter rows based on search
        function filterRows() {{
            const search = document.getElementById('row-search').value.toLowerCase();
            const items = document.querySelectorAll('.row-item');
            items.forEach((item, i) => {{
                const label = data.labels[i].toLowerCase();
                const fullLabel = data.full_labels[i].toLowerCase();
                if (label.includes(search) || fullLabel.includes(search)) {{
                    item.classList.remove('hidden');
                }} else {{
                    item.classList.add('hidden');
                }}
            }});
        }}

        // Toggle single row selection
        function toggleRow(index) {{
            if (selectedRows.has(index)) {{
                selectedRows.delete(index);
            }} else {{
                selectedRows.add(index);
            }}
            updateRowItemUI(index);
            renderMatrix();
        }}

        function updateRowItemUI(index) {{
            const item = document.querySelector(`.row-item[data-index="${{index}}"]`);
            const checkbox = item.querySelector('input');
            if (selectedRows.has(index)) {{
                item.classList.add('selected');
                checkbox.checked = true;
            }} else {{
                item.classList.remove('selected');
                checkbox.checked = false;
            }}
        }}

        // Select all visible rows
        function selectAllRows() {{
            const items = document.querySelectorAll('.row-item:not(.hidden)');
            items.forEach(item => {{
                const i = parseInt(item.dataset.index);
                selectedRows.add(i);
                updateRowItemUI(i);
            }});
            renderMatrix();
        }}

        // Deselect all visible rows
        function deselectAllRows() {{
            const items = document.querySelectorAll('.row-item:not(.hidden)');
            items.forEach(item => {{
                const i = parseInt(item.dataset.index);
                selectedRows.delete(i);
                updateRowItemUI(i);
            }});
            renderMatrix();
        }}

        // Select rows that are dependencies of currently selected rows
        function selectDependents() {{
            const currentSelected = new Set(selectedRows);
            // Find all columns (targets) that have dependencies from selected rows
            currentSelected.forEach(rowIdx => {{
                for (let colIdx = 0; colIdx < data.size; colIdx++) {{
                    if (data.matrix[rowIdx][colIdx] > 0) {{
                        selectedRows.add(colIdx); // Add dependent
                    }}
                }}
            }});
            // Update UI
            for (let i = 0; i < data.size; i++) {{
                updateRowItemUI(i);
            }}
            renderMatrix();
        }}

        function renderMatrix() {{
            const wrapper = document.getElementById('matrix-wrapper');
            let html = '';

            // Get visible row/col indices based on selection
            const visibleIndices = Array.from(selectedRows).sort((a, b) => a - b);
            if (visibleIndices.length === 0) {{
                wrapper.innerHTML = '<div style="color:#666;padding:20px;">No rows selected. Select rows from the left panel.</div>';
                return;
            }}

            // Column labels (only for visible columns)
            html += '<div class="col-labels">';
            for (const j of visibleIndices) {{
                html += `<div class="col-label" title="${{data.full_labels[j]}}">${{data.labels[j]}}</div>`;
            }}
            html += '</div>';

            // Matrix rows (only visible rows/cols)
            html += '<div id="matrix">';
            for (const i of visibleIndices) {{
                html += '<div class="matrix-row">';
                html += `<div class="row-label" title="${{data.full_labels[i]}}">${{data.labels[i]}}</div>`;
                for (const j of visibleIndices) {{
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

        function exportPNG() {{
            const wrapper = document.getElementById('matrix-wrapper');
            html2canvas(wrapper, {{
                backgroundColor: '#1a1a2e',
                scale: 2
            }}).then(canvas => {{
                const link = document.createElement('a');
                link.download = 'dependency-matrix.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }}).catch(err => {{
                // Fallback: use simpler approach
                alert('PNG export requires html2canvas library. Try SVG export instead.');
            }});
        }}

        function exportSVG() {{
            const wrapper = document.getElementById('matrix-wrapper');
            const rect = wrapper.getBoundingClientRect();

            // Create SVG from current matrix
            let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${{rect.width}}" height="${{rect.height + 130}}" style="background:#1a1a2e">`;
            svg += `<style>.label {{ font-family: Arial, sans-serif; font-size: 10px; fill: #ccc; }} .cell {{ stroke: #2a2a4e; }}</style>`;

            const cellSize = 24;
            const labelWidth = 120;
            const colLabelHeight = 120;

            // Column labels (vertical)
            for (let j = 0; j < data.size; j++) {{
                const x = labelWidth + j * cellSize + cellSize/2;
                svg += `<text x="${{x}}" y="${{colLabelHeight - 5}}" class="label" transform="rotate(-90, ${{x}}, ${{colLabelHeight - 5}})" text-anchor="start">${{data.labels[j]}}</text>`;
            }}

            // Row labels and cells
            for (let i = 0; i < data.size; i++) {{
                const y = colLabelHeight + i * cellSize;
                // Row label
                svg += `<text x="${{labelWidth - 5}}" y="${{y + cellSize/2 + 3}}" class="label" text-anchor="end">${{data.labels[i]}}</text>`;

                for (let j = 0; j < data.size; j++) {{
                    const x = labelWidth + j * cellSize;
                    const val = data.matrix[i][j];
                    let fill = '#1a1a2e';
                    if (i === j) fill = '#333';
                    else if (val >= 5) fill = '#1abc9c';
                    else if (val > 0) fill = '#3498db';

                    // Check for cycle
                    const isCycle = data.cycles.some(c => (c[0] === i && c[1] === j) || (c[0] === j && c[1] === i));
                    if (isCycle && val > 0) fill = '#e74c3c';

                    svg += `<rect x="${{x}}" y="${{y}}" width="${{cellSize}}" height="${{cellSize}}" fill="${{fill}}" class="cell"/>`;
                    if (val > 0) {{
                        svg += `<text x="${{x + cellSize/2}}" y="${{y + cellSize/2 + 3}}" text-anchor="middle" fill="white" font-size="9">${{val}}</text>`;
                    }}
                }}
            }}

            svg += '</svg>';

            const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'dependency-matrix.svg';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function toggleFullscreen() {{
            document.body.classList.toggle('fullscreen');
            // Also hide row-selector in fullscreen
            const rowSelector = document.getElementById('row-selector');
            if (document.body.classList.contains('fullscreen')) {{
                rowSelector.style.display = 'none';
            }} else {{
                rowSelector.style.display = 'flex';
            }}
        }}

        function toggleHelp() {{
            document.getElementById('help-panel').classList.toggle('visible');
            document.getElementById('help-overlay').classList.toggle('visible');
        }}

        // Start render after showing loading screen
        setTimeout(() => {{
            document.querySelector('.loading-text').textContent = 'Building matrix...';
            setTimeout(() => {{
                try {{
                    initRowSelector();
                    renderMatrix();
                }} catch (e) {{
                    console.error('Render error:', e);
                    document.querySelector('.loading-text').textContent = 'Error: ' + e.message;
                }}
                setTimeout(completeLoading, 100);
            }}, 50);
        }}, 100);
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
