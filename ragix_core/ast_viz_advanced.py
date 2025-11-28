"""
AST Advanced Visualizations - Treemap, Sunburst, and Chord Diagram

Provides advanced D3.js-based visualizations for code analysis:
- Treemap: Package hierarchy by LOC/complexity
- Sunburst: Module structure drill-down
- Chord Diagram: Inter-module dependencies

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

import html
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ast_base import ASTNode, NodeType, Symbol
from .dependencies import Dependency, DependencyGraph, DependencyType
from .ast_viz import VizConfig, ColorScheme, _get_color


class TreemapMetric(str, Enum):
    """Metric for treemap sizing."""
    LOC = "loc"  # Lines of code
    COMPLEXITY = "complexity"  # Cyclomatic complexity
    COUNT = "count"  # Symbol count
    DEBT = "debt"  # Technical debt hours


@dataclass
class TreemapConfig:
    """Configuration for treemap visualization."""
    metric: TreemapMetric = TreemapMetric.LOC
    color_by: str = "type"  # type, complexity, package
    max_depth: int = 4
    min_size: int = 10  # Minimum size threshold
    show_labels: bool = True
    title: str = "Code Treemap"


@dataclass
class SunburstConfig:
    """Configuration for sunburst visualization."""
    max_depth: int = 5
    color_by: str = "type"  # type, complexity
    show_labels: bool = True
    title: str = "Code Sunburst"


@dataclass
class ChordConfig:
    """Configuration for chord diagram."""
    group_by: str = "package"  # package, file
    min_connections: int = 1  # Minimum connections to show
    show_labels: bool = True
    title: str = "Module Dependencies"


class TreemapRenderer:
    """Render code structure as an interactive treemap."""

    def __init__(self, config: Optional[TreemapConfig] = None):
        self.config = config or TreemapConfig()

    def render(
        self,
        graph: DependencyGraph,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render graph as interactive HTML treemap."""
        # Build hierarchical data
        hierarchy = self._build_hierarchy(graph, metrics)

        return self._render_html(hierarchy)

    def _build_hierarchy(
        self,
        graph: DependencyGraph,
        metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build hierarchical structure for treemap."""
        symbols = graph.get_symbols()

        # Build tree structure
        root: Dict[str, Any] = {
            "name": "root",
            "children": {}
        }

        for sym in symbols:
            parts = sym.qualified_name.split(".")
            current = root

            for i, part in enumerate(parts):
                if i >= self.config.max_depth:
                    break

                if part not in current["children"]:
                    current["children"][part] = {
                        "name": part,
                        "children": {},
                        "symbols": []
                    }
                current = current["children"][part]

            # Add symbol data
            size = self._get_size(sym, metrics)
            current["symbols"].append({
                "name": sym.name,
                "qualified_name": sym.qualified_name,
                "type": sym.node_type.value,
                "size": size,
                "file": str(sym.location.file) if sym.location.file else "",
                "line": sym.location.line
            })

        # Convert to D3 format
        return self._convert_to_d3(root)

    def _get_size(self, sym: Symbol, metrics: Optional[Dict[str, Any]]) -> int:
        """Get size value based on configured metric."""
        if self.config.metric == TreemapMetric.LOC:
            # Estimate LOC from location
            return max(10, sym.location.end_line - sym.location.line + 1 if sym.location.end_line else 10)
        elif self.config.metric == TreemapMetric.COMPLEXITY:
            if metrics and "complexity" in metrics:
                return metrics["complexity"].get(sym.qualified_name, 1)
            return 1
        elif self.config.metric == TreemapMetric.COUNT:
            return 1
        elif self.config.metric == TreemapMetric.DEBT:
            if metrics and "debt" in metrics:
                return int(metrics["debt"].get(sym.qualified_name, 0) * 60)  # Convert hours to minutes
            return 0
        return 1

    def _convert_to_d3(self, node: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """Convert tree to D3-compatible format."""
        result: Dict[str, Any] = {"name": node["name"]}

        children = []

        # Add child packages/modules
        for child_name, child_node in node.get("children", {}).items():
            converted = self._convert_to_d3(child_node, depth + 1)
            if converted.get("value", 0) > 0 or converted.get("children"):
                children.append(converted)

        # Add symbols as leaves
        for sym in node.get("symbols", []):
            if sym["size"] >= self.config.min_size or depth >= self.config.max_depth - 1:
                children.append({
                    "name": sym["name"],
                    "value": max(sym["size"], 1),
                    "type": sym["type"],
                    "qualified_name": sym["qualified_name"],
                    "file": sym["file"],
                    "line": sym["line"]
                })

        if children:
            result["children"] = children
        else:
            result["value"] = sum(s["size"] for s in node.get("symbols", [])) or 1

        return result

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render treemap as interactive HTML."""
        title = html.escape(self.config.title)
        metric = self.config.metric.value

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: #16213e;
            padding: 15px 20px;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        header h1 {{
            font-size: 20px;
            color: #e94560;
        }}
        .controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .control-btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            cursor: pointer;
            font-size: 13px;
        }}
        .control-btn:hover {{ background: rgba(255,255,255,0.2); }}
        #breadcrumb {{
            padding: 10px 20px;
            background: #0f3460;
            font-size: 13px;
            display: flex;
            gap: 8px;
        }}
        .crumb {{
            cursor: pointer;
            color: #4fc3f7;
        }}
        .crumb:hover {{ text-decoration: underline; }}
        .crumb-sep {{ color: #666; }}
        #container {{
            flex: 1;
            display: flex;
        }}
        #treemap {{
            flex: 1;
            overflow: hidden;
        }}
        #sidebar {{
            width: 300px;
            background: #16213e;
            border-left: 1px solid #0f3460;
            padding: 20px;
            overflow-y: auto;
        }}
        .sidebar-section {{
            margin-bottom: 20px;
        }}
        .section-title {{
            font-size: 11px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}
        #node-info {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 6px;
            font-size: 13px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .info-label {{ color: #888; }}
        .info-value {{ color: #fff; text-align: right; max-width: 60%; word-break: break-all; }}
        .node {{
            cursor: pointer;
            stroke: #1a1a2e;
            stroke-width: 1px;
        }}
        .node:hover {{ stroke-width: 2px; stroke: #fff; }}
        .node-label {{
            pointer-events: none;
            fill: #fff;
            font-size: 11px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
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
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            margin-right: 6px;
            border-radius: 2px;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
        }}
        /* Author Footer */
        .author-footer {{
            position: fixed;
            bottom: 8px;
            right: 12px;
            font-size: 11px;
            color: #666;
            opacity: 0.6;
        }}
        .author-footer:hover {{ opacity: 1; }}
        .author-footer a {{ color: #4fc3f7; text-decoration: none; }}
        .author-footer a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <div class="controls">
            <span style="color:#888;font-size:13px;">Metric: {metric.upper()}</span>
            <button class="control-btn" onclick="zoomOut()">↑ Zoom Out</button>
            <button class="control-btn" onclick="resetZoom()">⌂ Reset</button>
        </div>
    </header>
    <div id="breadcrumb"><span class="crumb" onclick="resetZoom()">root</span></div>
    <div id="container">
        <div id="treemap"></div>
        <div id="sidebar">
            <div class="sidebar-section">
                <div class="section-title">Selected Node</div>
                <div id="node-info">Click on a cell to see details</div>
            </div>
            <div class="sidebar-section">
                <div class="section-title">Legend</div>
                <div class="legend">
                    <div class="legend-item"><span class="legend-color" style="background:#4a90d9"></span>Class</div>
                    <div class="legend-item"><span class="legend-color" style="background:#50c878"></span>Interface</div>
                    <div class="legend-item"><span class="legend-color" style="background:#f5a623"></span>Method/Function</div>
                    <div class="legend-item"><span class="legend-color" style="background:#9b59b6"></span>Field</div>
                    <div class="legend-item"><span class="legend-color" style="background:#34495e"></span>Module</div>
                </div>
            </div>
        </div>
    </div>
    <div class="tooltip" style="display:none"></div>

    <footer class="author-footer">
        <span>Olivier Vitrac, PhD, HDR</span> |
        <span>Adservio</span> |
        <a href="mailto:olivier.vitrac@adservio.fr">olivier.vitrac@adservio.fr</a>
    </footer>

    <script>
        const data = {json.dumps(data)};

        const typeColors = {{
            'class': '#4a90d9',
            'interface': '#50c878',
            'method': '#f5a623',
            'function': '#f5a623',
            'field': '#9b59b6',
            'constant': '#e74c3c',
            'module': '#34495e',
            'package': '#2c3e50',
            'enum': '#1abc9c'
        }};

        const container = document.getElementById('treemap');
        const width = container.clientWidth;
        const height = container.clientHeight;

        const tooltip = d3.select('.tooltip');
        let currentRoot = data;
        let history = [];

        const treemap = d3.treemap()
            .size([width, height])
            .paddingOuter(3)
            .paddingTop(19)
            .paddingInner(1)
            .round(true);

        const svg = d3.select('#treemap')
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        function getColor(d) {{
            if (d.data.type) return typeColors[d.data.type] || '#666';
            // For groups, use depth-based coloring
            const hue = (d.depth * 60) % 360;
            return `hsl(${{hue}}, 40%, 35%)`;
        }}

        function render(root) {{
            currentRoot = root;

            const hierarchy = d3.hierarchy(root)
                .sum(d => d.value || 0)
                .sort((a, b) => b.value - a.value);

            treemap(hierarchy);

            svg.selectAll('*').remove();

            const nodes = svg.selectAll('g')
                .data(hierarchy.descendants())
                .join('g')
                .attr('transform', d => `translate(${{d.x0}},${{d.y0}})`);

            nodes.append('rect')
                .attr('class', 'node')
                .attr('width', d => Math.max(0, d.x1 - d.x0))
                .attr('height', d => Math.max(0, d.y1 - d.y0))
                .attr('fill', getColor)
                .on('click', (event, d) => {{
                    if (d.children) {{
                        history.push(currentRoot);
                        render(d.data);
                        updateBreadcrumb();
                    }}
                    showNodeInfo(d);
                }})
                .on('mouseover', (event, d) => {{
                    tooltip.style('display', 'block')
                        .html(`<strong>${{d.data.name}}</strong><br/>Size: ${{d.value}}`);
                }})
                .on('mousemove', (event) => {{
                    tooltip.style('left', (event.pageX + 10) + 'px')
                        .style('top', (event.pageY - 10) + 'px');
                }})
                .on('mouseout', () => tooltip.style('display', 'none'));

            // Add labels for larger nodes
            nodes.filter(d => (d.x1 - d.x0) > 40 && (d.y1 - d.y0) > 20)
                .append('text')
                .attr('class', 'node-label')
                .attr('x', 4)
                .attr('y', 14)
                .text(d => d.data.name)
                .each(function(d) {{
                    const textWidth = this.getComputedLength();
                    const boxWidth = d.x1 - d.x0 - 8;
                    if (textWidth > boxWidth) {{
                        const ratio = boxWidth / textWidth;
                        this.textContent = d.data.name.slice(0, Math.floor(d.data.name.length * ratio) - 2) + '…';
                    }}
                }});
        }}

        function showNodeInfo(d) {{
            const info = document.getElementById('node-info');
            const data = d.data;

            let html = `
                <div class="info-row">
                    <span class="info-label">Name</span>
                    <span class="info-value">${{data.name}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Size</span>
                    <span class="info-value">${{d.value}}</span>
                </div>
            `;

            if (data.type) {{
                html += `
                    <div class="info-row">
                        <span class="info-label">Type</span>
                        <span class="info-value">${{data.type}}</span>
                    </div>
                `;
            }}
            if (data.qualified_name) {{
                html += `
                    <div class="info-row">
                        <span class="info-label">Full Name</span>
                        <span class="info-value">${{data.qualified_name}}</span>
                    </div>
                `;
            }}
            if (data.file) {{
                html += `
                    <div class="info-row">
                        <span class="info-label">File</span>
                        <span class="info-value">${{data.file.split('/').pop()}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Line</span>
                        <span class="info-value">${{data.line || 'N/A'}}</span>
                    </div>
                `;
            }}
            if (d.children) {{
                html += `
                    <div class="info-row">
                        <span class="info-label">Children</span>
                        <span class="info-value">${{d.children.length}}</span>
                    </div>
                `;
            }}

            info.innerHTML = html;
        }}

        function updateBreadcrumb() {{
            const bc = document.getElementById('breadcrumb');
            let html = '<span class="crumb" onclick="resetZoom()">root</span>';

            history.forEach((h, i) => {{
                if (h.name !== 'root') {{
                    html += `<span class="crumb-sep">›</span><span class="crumb" onclick="goToHistory(${{i}})">${{h.name}}</span>`;
                }}
            }});

            if (currentRoot.name !== 'root') {{
                html += `<span class="crumb-sep">›</span><span>${{currentRoot.name}}</span>`;
            }}

            bc.innerHTML = html;
        }}

        function zoomOut() {{
            if (history.length > 0) {{
                const prev = history.pop();
                render(prev);
                updateBreadcrumb();
            }}
        }}

        function resetZoom() {{
            history = [];
            render(data);
            updateBreadcrumb();
        }}

        function goToHistory(index) {{
            const target = history[index];
            history = history.slice(0, index);
            render(target);
            updateBreadcrumb();
        }}

        // Initial render
        render(data);

        // Handle resize
        window.addEventListener('resize', () => {{
            const w = container.clientWidth;
            const h = container.clientHeight;
            svg.attr('width', w).attr('height', h);
            treemap.size([w, h]);
            render(currentRoot);
        }});
    </script>
</body>
</html>'''


class SunburstRenderer:
    """Render code structure as an interactive sunburst diagram."""

    def __init__(self, config: Optional[SunburstConfig] = None):
        self.config = config or SunburstConfig()

    def render(
        self,
        graph: DependencyGraph,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render graph as interactive HTML sunburst."""
        hierarchy = self._build_hierarchy(graph, metrics)
        return self._render_html(hierarchy)

    def _build_hierarchy(
        self,
        graph: DependencyGraph,
        metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build hierarchical structure for sunburst."""
        symbols = graph.get_symbols()

        root: Dict[str, Any] = {"name": "root", "children": {}}

        for sym in symbols:
            parts = sym.qualified_name.split(".")
            current = root

            for i, part in enumerate(parts):
                if i >= self.config.max_depth:
                    break
                if part not in current["children"]:
                    current["children"][part] = {"name": part, "children": {}}
                current = current["children"][part]

            # Store symbol info at leaf
            if "symbols" not in current:
                current["symbols"] = []

            loc = sym.location.end_line - sym.location.line + 1 if sym.location.end_line else 10
            current["symbols"].append({
                "name": sym.name,
                "type": sym.node_type.value,
                "size": max(loc, 1),
                "qualified_name": sym.qualified_name
            })

        return self._convert_to_d3(root)

    def _convert_to_d3(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to D3 format."""
        result: Dict[str, Any] = {"name": node["name"]}
        children = []

        for name, child in node.get("children", {}).items():
            converted = self._convert_to_d3(child)
            children.append(converted)

        for sym in node.get("symbols", []):
            children.append({
                "name": sym["name"],
                "value": sym["size"],
                "type": sym["type"],
                "qualified_name": sym["qualified_name"]
            })

        if children:
            result["children"] = children
        else:
            result["value"] = 1

        return result

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render sunburst as interactive HTML."""
        title = html.escape(self.config.title)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: #16213e;
            padding: 15px 20px;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        header h1 {{ font-size: 20px; color: #e94560; }}
        #container {{ flex: 1; display: flex; }}
        #sunburst {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        #sidebar {{
            width: 300px;
            background: #16213e;
            border-left: 1px solid #0f3460;
            padding: 20px;
            overflow-y: auto;
        }}
        .sidebar-section {{ margin-bottom: 20px; }}
        .section-title {{
            font-size: 11px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}
        #node-info {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 6px;
            font-size: 13px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .info-label {{ color: #888; }}
        .info-value {{ color: #fff; text-align: right; max-width: 60%; word-break: break-all; }}
        .arc {{ cursor: pointer; }}
        .arc:hover {{ opacity: 0.8; }}
        .center-label {{
            text-anchor: middle;
            fill: #fff;
            font-size: 14px;
            font-weight: bold;
        }}
        .center-value {{
            text-anchor: middle;
            fill: #888;
            font-size: 12px;
        }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .legend-item {{ display: flex; align-items: center; font-size: 11px; }}
        .legend-color {{ width: 12px; height: 12px; margin-right: 6px; border-radius: 2px; }}
        /* Author Footer */
        .author-footer {{
            position: fixed; bottom: 8px; right: 12px;
            font-size: 11px; color: #666; opacity: 0.6;
        }}
        .author-footer:hover {{ opacity: 1; }}
        .author-footer a {{ color: #4fc3f7; text-decoration: none; }}
    </style>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <button onclick="resetZoom()" style="padding:8px 16px;border:none;border-radius:4px;background:rgba(255,255,255,0.1);color:white;cursor:pointer;">⌂ Reset</button>
    </header>
    <div id="container">
        <div id="sunburst"></div>
        <div id="sidebar">
            <div class="sidebar-section">
                <div class="section-title">Selected</div>
                <div id="node-info">Click on a segment to see details</div>
            </div>
            <div class="sidebar-section">
                <div class="section-title">Legend</div>
                <div class="legend">
                    <div class="legend-item"><span class="legend-color" style="background:#4a90d9"></span>Class</div>
                    <div class="legend-item"><span class="legend-color" style="background:#50c878"></span>Interface</div>
                    <div class="legend-item"><span class="legend-color" style="background:#f5a623"></span>Method</div>
                    <div class="legend-item"><span class="legend-color" style="background:#9b59b6"></span>Field</div>
                    <div class="legend-item"><span class="legend-color" style="background:#34495e"></span>Module</div>
                </div>
            </div>
        </div>
    </div>

    <footer class="author-footer">
        <span>Olivier Vitrac, PhD, HDR</span> | <span>Adservio</span> |
        <a href="mailto:olivier.vitrac@adservio.fr">olivier.vitrac@adservio.fr</a>
    </footer>

    <script>
        const data = {json.dumps(data)};
        const typeColors = {{
            'class': '#4a90d9', 'interface': '#50c878', 'method': '#f5a623',
            'function': '#f5a623', 'field': '#9b59b6', 'module': '#34495e', 'enum': '#1abc9c'
        }};

        const container = document.getElementById('sunburst');
        const size = Math.min(container.clientWidth, container.clientHeight) - 40;
        const radius = size / 2;

        const svg = d3.select('#sunburst')
            .append('svg')
            .attr('width', size)
            .attr('height', size)
            .append('g')
            .attr('transform', `translate(${{size/2}},${{size/2}})`);

        const partition = d3.partition().size([2 * Math.PI, radius]);
        const arc = d3.arc()
            .startAngle(d => d.x0)
            .endAngle(d => d.x1)
            .innerRadius(d => d.y0)
            .outerRadius(d => d.y1 - 1);

        const root = d3.hierarchy(data)
            .sum(d => d.value || 0)
            .sort((a, b) => b.value - a.value);

        partition(root);

        let currentRoot = root;

        function getColor(d) {{
            if (d.data.type) return typeColors[d.data.type] || '#666';
            const hue = (d.depth * 45 + (d.data.name.charCodeAt(0) || 0) * 10) % 360;
            return `hsl(${{hue}}, 50%, ${{45 - d.depth * 5}}%)`;
        }}

        function render(p) {{
            currentRoot = p;

            svg.selectAll('path').remove();
            svg.selectAll('text').remove();

            const descendants = p.descendants().filter(d => d.depth > 0);

            svg.selectAll('path')
                .data(descendants)
                .join('path')
                .attr('class', 'arc')
                .attr('d', arc)
                .attr('fill', getColor)
                .on('click', (event, d) => {{
                    if (d.children) render(d);
                    showInfo(d);
                }});

            // Center text
            svg.append('text')
                .attr('class', 'center-label')
                .attr('dy', '-0.2em')
                .text(p.data.name === 'root' ? 'Project' : p.data.name);

            svg.append('text')
                .attr('class', 'center-value')
                .attr('dy', '1.2em')
                .text(`${{p.value}} items`);
        }}

        function showInfo(d) {{
            const info = document.getElementById('node-info');
            let html = `
                <div class="info-row"><span class="info-label">Name</span><span class="info-value">${{d.data.name}}</span></div>
                <div class="info-row"><span class="info-label">Size</span><span class="info-value">${{d.value}}</span></div>
            `;
            if (d.data.type) html += `<div class="info-row"><span class="info-label">Type</span><span class="info-value">${{d.data.type}}</span></div>`;
            if (d.data.qualified_name) html += `<div class="info-row"><span class="info-label">Full Name</span><span class="info-value">${{d.data.qualified_name}}</span></div>`;
            if (d.children) html += `<div class="info-row"><span class="info-label">Children</span><span class="info-value">${{d.children.length}}</span></div>`;
            info.innerHTML = html;
        }}

        function resetZoom() {{ render(root); }}

        render(root);
    </script>
</body>
</html>'''


class ChordRenderer:
    """Render inter-module dependencies as a chord diagram."""

    def __init__(self, config: Optional[ChordConfig] = None):
        self.config = config or ChordConfig()

    def render(self, graph: DependencyGraph) -> str:
        """Render graph as interactive HTML chord diagram."""
        matrix_data = self._build_matrix(graph)
        return self._render_html(matrix_data)

    def _build_matrix(self, graph: DependencyGraph) -> Dict[str, Any]:
        """Build adjacency matrix for chord diagram."""
        deps = graph.get_all_dependencies()

        # Group symbols by package or file
        groups: Dict[str, Set[str]] = {}
        for dep in deps:
            src_group = self._get_group(dep.source)
            tgt_group = self._get_group(dep.target)
            groups.setdefault(src_group, set()).add(dep.source)
            groups.setdefault(tgt_group, set()).add(dep.target)

        # Build matrix
        group_list = sorted(groups.keys())
        n = len(group_list)
        group_index = {g: i for i, g in enumerate(group_list)}

        matrix = [[0] * n for _ in range(n)]

        for dep in deps:
            src_group = self._get_group(dep.source)
            tgt_group = self._get_group(dep.target)
            if src_group in group_index and tgt_group in group_index:
                i = group_index[src_group]
                j = group_index[tgt_group]
                matrix[i][j] += 1

        # Filter by min connections
        if self.config.min_connections > 1:
            for i in range(n):
                for j in range(n):
                    if matrix[i][j] < self.config.min_connections:
                        matrix[i][j] = 0

        return {
            "groups": group_list,
            "matrix": matrix,
            "sizes": [len(groups[g]) for g in group_list]
        }

    def _get_group(self, qualified_name: str) -> str:
        """Get group name for a symbol."""
        parts = qualified_name.split(".")
        if self.config.group_by == "file":
            return ".".join(parts[:-1]) if len(parts) > 1 else qualified_name
        else:  # package
            # Get top 2-3 levels
            return ".".join(parts[:min(3, len(parts) - 1)]) if len(parts) > 1 else qualified_name

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render chord diagram as interactive HTML."""
        title = html.escape(self.config.title)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: #16213e;
            padding: 15px 20px;
            border-bottom: 1px solid #0f3460;
        }}
        header h1 {{ font-size: 20px; color: #e94560; }}
        #container {{ flex: 1; display: flex; }}
        #chord {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        #sidebar {{
            width: 320px;
            background: #16213e;
            border-left: 1px solid #0f3460;
            padding: 20px;
            overflow-y: auto;
        }}
        .sidebar-section {{ margin-bottom: 20px; }}
        .section-title {{
            font-size: 11px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}
        #info {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 6px;
            font-size: 13px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .info-label {{ color: #888; }}
        .info-value {{ color: #fff; }}
        .group {{ cursor: pointer; }}
        .group:hover {{ opacity: 0.8; }}
        .chord {{ fill-opacity: 0.7; }}
        .chord:hover {{ fill-opacity: 1; }}
        .group-label {{
            font-size: 10px;
            fill: #ccc;
        }}
        #groups-list {{
            max-height: 300px;
            overflow-y: auto;
        }}
        .group-item {{
            display: flex;
            align-items: center;
            padding: 6px 8px;
            margin: 2px 0;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
        }}
        .group-item:hover {{ background: rgba(255,255,255,0.1); }}
        .group-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
        }}
        /* Author Footer */
        .author-footer {{
            position: fixed; bottom: 8px; right: 12px;
            font-size: 11px; color: #666; opacity: 0.6;
        }}
        .author-footer:hover {{ opacity: 1; }}
        .author-footer a {{ color: #4fc3f7; text-decoration: none; }}
    </style>
</head>
<body>
    <header><h1>{title}</h1></header>
    <div id="container">
        <div id="chord"></div>
        <div id="sidebar">
            <div class="sidebar-section">
                <div class="section-title">Connection Info</div>
                <div id="info">Hover over a chord to see connection details</div>
            </div>
            <div class="sidebar-section">
                <div class="section-title">Modules ({len(data["groups"])})</div>
                <div id="groups-list"></div>
            </div>
        </div>
    </div>

    <footer class="author-footer">
        <span>Olivier Vitrac, PhD, HDR</span> | <span>Adservio</span> |
        <a href="mailto:olivier.vitrac@adservio.fr">olivier.vitrac@adservio.fr</a>
    </footer>

    <script>
        const data = {json.dumps(data)};
        const groups = data.groups;
        const matrix = data.matrix;

        const container = document.getElementById('chord');
        const size = Math.min(container.clientWidth, container.clientHeight) - 100;
        const outerRadius = size / 2;
        const innerRadius = outerRadius - 30;

        const color = d3.scaleOrdinal()
            .domain(d3.range(groups.length))
            .range(d3.schemeTableau10);

        const chord = d3.chord()
            .padAngle(0.05)
            .sortSubgroups(d3.descending);

        const arc = d3.arc()
            .innerRadius(innerRadius)
            .outerRadius(outerRadius);

        const ribbon = d3.ribbon()
            .radius(innerRadius);

        const svg = d3.select('#chord')
            .append('svg')
            .attr('width', size + 100)
            .attr('height', size + 100)
            .append('g')
            .attr('transform', `translate(${{(size+100)/2}},${{(size+100)/2}})`);

        const chords = chord(matrix);

        // Draw groups
        const groupG = svg.append('g')
            .selectAll('g')
            .data(chords.groups)
            .join('g')
            .attr('class', 'group');

        groupG.append('path')
            .attr('d', arc)
            .attr('fill', d => color(d.index))
            .attr('stroke', '#1a1a2e');

        // Add labels
        groupG.append('text')
            .attr('class', 'group-label')
            .each(d => {{ d.angle = (d.startAngle + d.endAngle) / 2; }})
            .attr('dy', '0.35em')
            .attr('transform', d => `
                rotate(${{d.angle * 180 / Math.PI - 90}})
                translate(${{outerRadius + 10}})
                ${{d.angle > Math.PI ? 'rotate(180)' : ''}}
            `)
            .attr('text-anchor', d => d.angle > Math.PI ? 'end' : 'start')
            .text(d => {{
                const name = groups[d.index];
                return name.length > 20 ? name.slice(0, 18) + '…' : name;
            }});

        // Draw chords
        svg.append('g')
            .attr('fill-opacity', 0.7)
            .selectAll('path')
            .data(chords)
            .join('path')
            .attr('class', 'chord')
            .attr('d', ribbon)
            .attr('fill', d => color(d.source.index))
            .attr('stroke', d => d3.rgb(color(d.source.index)).darker())
            .on('mouseover', (event, d) => {{
                showChordInfo(d);
            }})
            .on('mouseout', () => {{
                document.getElementById('info').innerHTML = 'Hover over a chord to see connection details';
            }});

        function showChordInfo(d) {{
            const src = groups[d.source.index];
            const tgt = groups[d.target.index];
            const value = d.source.value;

            document.getElementById('info').innerHTML = `
                <div class="info-row"><span class="info-label">From</span><span class="info-value">${{src}}</span></div>
                <div class="info-row"><span class="info-label">To</span><span class="info-value">${{tgt}}</span></div>
                <div class="info-row"><span class="info-label">Connections</span><span class="info-value">${{value}}</span></div>
            `;
        }}

        // Populate groups list
        const list = document.getElementById('groups-list');
        groups.forEach((g, i) => {{
            const item = document.createElement('div');
            item.className = 'group-item';
            item.innerHTML = `<span class="group-color" style="background:${{color(i)}}"></span>${{g}}`;
            list.appendChild(item);
        }});
    </script>
</body>
</html>'''
