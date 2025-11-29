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
        body.fullscreen #sidebar {{ display: none !important; }}
        body.fullscreen #container {{ width: 100vw; }}
        body.fullscreen #treemap {{ flex: 1; width: 100%; }}
        body.fullscreen #breadcrumb {{ display: none; }}
        body.fullscreen header {{ position: absolute; top: 0; left: 0; right: 0; z-index: 100; background: rgba(22, 33, 62, 0.9); }}
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
        .progress-fill.complete {{
            background: #50c878;
            animation: none;
        }}
        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
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
            z-index: 1000;
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
            background: none;
            border: none;
            color: #888;
            font-size: 24px;
            cursor: pointer;
        }}
        #help-panel .close-btn:hover {{ color: #fff; }}
        .help-section {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .help-section h3 {{
            color: #4fc3f7;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .help-section p {{
            color: #ccc;
            font-size: 13px;
            line-height: 1.6;
            margin-bottom: 8px;
        }}
        .help-section ul {{
            margin: 8px 0 0 20px;
            color: #aaa;
            font-size: 12px;
        }}
        .help-section li {{ margin-bottom: 6px; }}
        .help-section code {{
            background: rgba(0,0,0,0.3);
            padding: 2px 6px;
            border-radius: 3px;
            color: #f5a623;
        }}
        .help-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.5);
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
            z-index: 999;
        }}
        .help-overlay.visible {{ opacity: 1; pointer-events: auto; }}
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
    <div id="loading">
        <div class="spinner"></div>
        <div class="loading-text">Building treemap...</div>
        <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
    </div>
    <header>
        <h1>{title}</h1>
        <div class="controls">
            <span style="color:#888;font-size:13px;">Metric: {metric.upper()}</span>
            <button class="control-btn" onclick="zoomOut()">↑ Zoom Out</button>
            <button class="control-btn" onclick="resetZoom()">⌂ Reset</button>
            <button class="control-btn" onclick="exportPNG()">🖼️ PNG</button>
            <button class="control-btn" onclick="exportSVG()">📐 SVG</button>
            <button class="control-btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
            <button class="control-btn" onclick="toggleHelp()">❓ Help</button>
        </div>
    </header>
    <div class="help-overlay" id="help-overlay" onclick="toggleHelp()"></div>
    <div id="help-panel">
        <h2>
            <span>📊 Treemap Guide</span>
            <button class="close-btn" onclick="toggleHelp()">×</button>
        </h2>
        <div class="help-section">
            <h3>What is a Treemap?</h3>
            <p>A treemap visualizes hierarchical data using nested rectangles. The size of each rectangle represents a metric (lines of code, complexity, etc.), allowing you to quickly identify large or complex code elements.</p>
        </div>
        <div class="help-section">
            <h3>Navigation</h3>
            <ul>
                <li><strong>Click on a package/module</strong> to zoom into it and see its contents</li>
                <li><strong>Breadcrumb trail</strong> at the top shows your navigation path - click to jump back</li>
                <li><strong>Zoom Out (↑)</strong> button returns to the parent level</li>
                <li><strong>Reset (⌂)</strong> returns to the root view</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Understanding Colors</h3>
            <p>Colors indicate the type of code element:</p>
            <ul>
                <li><code style="background:#4a90d9">Blue</code> - Classes</li>
                <li><code style="background:#50c878">Green</code> - Interfaces</li>
                <li><code style="background:#f5a623">Orange</code> - Methods/Functions</li>
                <li><code style="background:#9b59b6">Purple</code> - Fields</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Sidebar Information</h3>
            <p>Click on any rectangle to see detailed information in the sidebar, including the element's name, type, file location, and size.</p>
        </div>
        <div class="help-section">
            <h3>Export Options</h3>
            <ul>
                <li><strong>PNG</strong> - High-resolution image for presentations</li>
                <li><strong>SVG</strong> - Vector format for editing or printing</li>
            </ul>
        </div>
    </div>
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

        // Simple loading - CSS animation handles the visual feedback
        function completeLoading() {{
            document.querySelector('.loading-text').textContent = 'Done!';
            document.getElementById('progress').classList.add('complete');
            setTimeout(() => {{
                document.getElementById('loading').classList.add('hidden');
            }}, 400);
        }}

        // Start render after showing loading screen
        setTimeout(() => {{
            document.querySelector('.loading-text').textContent = 'Building treemap...';
            setTimeout(() => {{
                try {{
                    render(data);
                }} catch (e) {{
                    console.error('Render error:', e);
                    document.querySelector('.loading-text').textContent = 'Error: ' + e.message;
                }}
                // Always complete after render attempt - use setTimeout to ensure it runs
                setTimeout(completeLoading, 100);
            }}, 50);
        }}, 100);

        // Export functions
        function exportSVG() {{
            const svgElement = document.querySelector('#treemap svg');
            if (!svgElement) {{ alert('No SVG to export'); return; }}
            const serializer = new XMLSerializer();
            let source = serializer.serializeToString(svgElement);
            // Add XML declaration and styles
            source = '<?xml version="1.0" encoding="UTF-8"?>' + source;
            const blob = new Blob([source], {{ type: 'image/svg+xml;charset=utf-8' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'treemap.svg';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function exportPNG() {{
            const svgElement = document.querySelector('#treemap svg');
            if (!svgElement) {{ alert('No SVG to export'); return; }}
            const canvas = document.createElement('canvas');
            const rect = svgElement.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = rect.height * 2;
            const ctx = canvas.getContext('2d');
            ctx.scale(2, 2);
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, rect.width, rect.height);
            const img = new Image();
            const serializer = new XMLSerializer();
            const svgStr = serializer.serializeToString(svgElement);
            img.onload = function() {{
                ctx.drawImage(img, 0, 0);
                const link = document.createElement('a');
                link.download = 'treemap.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }};
            img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgStr)));
        }}

        function toggleFullscreen() {{
            document.body.classList.toggle('fullscreen');
            setTimeout(() => {{
                const w = container.clientWidth;
                const h = container.clientHeight;
                svg.attr('width', w).attr('height', h);
                treemap.size([w, h]);
                render(currentRoot);
            }}, 100);
        }}

        function toggleHelp() {{
            document.getElementById('help-panel').classList.toggle('visible');
            document.getElementById('help-overlay').classList.toggle('visible');
        }}

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
        body.fullscreen #sidebar {{ display: none !important; }}
        body.fullscreen #container {{ width: 100vw; }}
        body.fullscreen #sunburst {{ flex: 1; width: 100%; }}
        body.fullscreen header {{ position: absolute; top: 0; left: 0; right: 0; z-index: 100; background: rgba(22, 33, 62, 0.9); }}
        header {{
            background: #16213e;
            padding: 15px 20px;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        header h1 {{ font-size: 20px; color: #e94560; }}
        .controls {{ display: flex; gap: 10px; align-items: center; }}
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
            width: 60px;
            height: 60px;
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
            z-index: 1000;
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
            transition: opacity 0.3s; z-index: 999;
        }}
        .help-overlay.visible {{ opacity: 1; pointer-events: auto; }}
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
    <div id="loading">
        <div class="spinner"></div>
        <div class="loading-text">Building sunburst...</div>
        <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
    </div>
    <header>
        <h1>{title}</h1>
        <div class="controls">
            <button class="control-btn" onclick="resetZoom()">⌂ Reset</button>
            <button class="control-btn" onclick="exportPNG()">🖼️ PNG</button>
            <button class="control-btn" onclick="exportSVG()">📐 SVG</button>
            <button class="control-btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
            <button class="control-btn" onclick="toggleHelp()">❓ Help</button>
        </div>
    </header>
    <div class="help-overlay" id="help-overlay" onclick="toggleHelp()"></div>
    <div id="help-panel">
        <h2>
            <span>🌞 Sunburst Guide</span>
            <button class="close-btn" onclick="toggleHelp()">×</button>
        </h2>
        <div class="help-section">
            <h3>What is a Sunburst Diagram?</h3>
            <p>A sunburst diagram displays hierarchical data in concentric rings. The center represents the root, and each ring outward represents deeper levels of the hierarchy. The arc size shows the relative size of each element.</p>
        </div>
        <div class="help-section">
            <h3>Navigation</h3>
            <ul>
                <li><strong>Click on any arc</strong> to zoom in and make it the new center</li>
                <li><strong>Click the center</strong> to see the current focus area statistics</li>
                <li><strong>Reset (⌂)</strong> returns to the root view</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Understanding the Rings</h3>
            <p>Each concentric ring represents a level of hierarchy:</p>
            <ul>
                <li><strong>Inner rings</strong> - Top-level packages</li>
                <li><strong>Middle rings</strong> - Sub-packages and classes</li>
                <li><strong>Outer rings</strong> - Methods and fields</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Colors</h3>
            <p>Colors indicate element types:</p>
            <ul>
                <li><code style="background:#4a90d9">Blue</code> - Classes</li>
                <li><code style="background:#50c878">Green</code> - Interfaces</li>
                <li><code style="background:#f5a623">Orange</code> - Methods</li>
                <li><code style="background:#9b59b6">Purple</code> - Fields</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Tips</h3>
            <p>Use sunburst to visualize the overall structure of your codebase and identify which packages contain the most code. Large arcs indicate areas that may need attention.</p>
        </div>
    </div>
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

        // Simple loading - CSS animation handles the visual feedback
        function completeLoading() {{
            document.querySelector('.loading-text').textContent = 'Done!';
            document.getElementById('progress').classList.add('complete');
            setTimeout(() => {{
                document.getElementById('loading').classList.add('hidden');
            }}, 400);
        }}

        // Start render after showing loading screen
        setTimeout(() => {{
            document.querySelector('.loading-text').textContent = 'Building sunburst...';
            setTimeout(() => {{
                try {{
                    render(root);
                }} catch (e) {{
                    console.error('Render error:', e);
                    document.querySelector('.loading-text').textContent = 'Error: ' + e.message;
                }}
                setTimeout(completeLoading, 100);
            }}, 50);
        }}, 100);

        // Export functions
        function exportSVG() {{
            const svgElement = document.querySelector('#sunburst svg');
            if (!svgElement) {{ alert('No SVG to export'); return; }}
            const serializer = new XMLSerializer();
            let source = serializer.serializeToString(svgElement);
            source = '<?xml version="1.0" encoding="UTF-8"?>' + source;
            const blob = new Blob([source], {{ type: 'image/svg+xml;charset=utf-8' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sunburst.svg';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function exportPNG() {{
            const svgElement = document.querySelector('#sunburst svg');
            if (!svgElement) {{ alert('No SVG to export'); return; }}
            const canvas = document.createElement('canvas');
            const rect = svgElement.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = rect.height * 2;
            const ctx = canvas.getContext('2d');
            ctx.scale(2, 2);
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, rect.width, rect.height);
            const img = new Image();
            const serializer = new XMLSerializer();
            const svgStr = serializer.serializeToString(svgElement);
            img.onload = function() {{
                ctx.drawImage(img, 0, 0);
                const link = document.createElement('a');
                link.download = 'sunburst.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }};
            img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgStr)));
        }}

        function toggleFullscreen() {{
            document.body.classList.toggle('fullscreen');
        }}

        function toggleHelp() {{
            document.getElementById('help-panel').classList.toggle('visible');
            document.getElementById('help-overlay').classList.toggle('visible');
        }}
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
        body.fullscreen #sidebar {{ display: none !important; }}
        body.fullscreen #container {{ width: 100vw; }}
        body.fullscreen #chord {{ flex: 1; width: 100%; }}
        body.fullscreen header {{ position: absolute; top: 0; left: 0; right: 0; z-index: 100; background: rgba(22, 33, 62, 0.9); }}
        header {{
            background: #16213e;
            padding: 15px 20px;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        header h1 {{ font-size: 20px; color: #e94560; }}
        .controls {{ display: flex; gap: 10px; align-items: center; }}
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
        .control-btn.active {{ background: #e94560; }}
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
            width: 60px;
            height: 60px;
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
            z-index: 1000;
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
            transition: opacity 0.3s; z-index: 999;
        }}
        .help-overlay.visible {{ opacity: 1; pointer-events: auto; }}
        #container {{ flex: 1; display: flex; }}
        #chord {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        #sidebar {{
            width: 360px;
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
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .section-controls {{
            display: flex;
            gap: 6px;
        }}
        .section-btn {{
            padding: 4px 8px;
            border: none;
            border-radius: 3px;
            background: rgba(255,255,255,0.1);
            color: #aaa;
            cursor: pointer;
            font-size: 10px;
        }}
        .section-btn:hover {{ background: rgba(255,255,255,0.2); color: #fff; }}
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
        .info-value {{ color: #fff; word-break: break-all; max-width: 200px; text-align: right; }}
        .group {{ cursor: pointer; }}
        .group:hover {{ opacity: 0.8; }}
        .chord {{ fill-opacity: 0.7; }}
        .chord:hover {{ fill-opacity: 1; }}
        .group-label {{
            font-size: 10px;
            fill: #ccc;
        }}
        .search-box {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #0f3460;
            border-radius: 4px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 12px;
            margin-bottom: 10px;
        }}
        .search-box::placeholder {{ color: #666; }}
        #groups-list {{
            max-height: 280px;
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
        .group-item.selected {{ background: rgba(233, 69, 96, 0.3); }}
        .group-item.hidden {{ display: none; }}
        .group-item input[type="checkbox"] {{
            margin-right: 8px;
            cursor: pointer;
        }}
        .group-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
            flex-shrink: 0;
        }}
        .group-name {{
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .group-count {{
            color: #666;
            font-size: 10px;
            margin-left: 8px;
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
    <div id="loading">
        <div class="spinner"></div>
        <div class="loading-text">Building chord diagram...</div>
        <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
    </div>
    <header>
        <h1>{title}</h1>
        <div class="controls">
            <button class="control-btn" onclick="exportPNG()">🖼️ PNG</button>
            <button class="control-btn" onclick="exportSVG()">📐 SVG</button>
            <button class="control-btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
            <button class="control-btn" onclick="toggleHelp()">❓ Help</button>
        </div>
    </header>
    <div class="help-overlay" id="help-overlay" onclick="toggleHelp()"></div>
    <div id="help-panel">
        <h2>
            <span>🔗 Chord Diagram Guide</span>
            <button class="close-btn" onclick="toggleHelp()">×</button>
        </h2>
        <div class="help-section">
            <h3>What is a Chord Diagram?</h3>
            <p>A chord diagram visualizes relationships between modules/packages. The arcs around the circle represent modules, and the ribbons connecting them show dependencies. The thickness of a ribbon indicates the strength of the connection.</p>
        </div>
        <div class="help-section">
            <h3>Interacting with Modules</h3>
            <ul>
                <li><strong>Checkboxes</strong> - Toggle module visibility in the diagram</li>
                <li><strong>Click on module row</strong> - Select/highlight the module (background turns pink)</li>
                <li><strong>Click on arc</strong> - Also selects the corresponding module</li>
                <li><strong>Hover over ribbon</strong> - See connection details in the sidebar</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Selection Controls</h3>
            <ul>
                <li><strong>All</strong> - Show all modules in the diagram</li>
                <li><strong>None</strong> - Hide all modules (clears the diagram)</li>
                <li><strong>↑ Parents</strong> - Add modules that your selected modules depend on</li>
                <li><strong>↓ Children</strong> - Add modules that depend on your selected modules</li>
            </ul>
            <p style="margin-top:10px;"><strong>Workflow:</strong> Start by selecting a few modules of interest, then use Parents/Children to explore their dependency neighborhood.</p>
        </div>
        <div class="help-section">
            <h3>Search</h3>
            <p>Type in the search box to filter the module list. This helps you quickly find specific packages in large codebases.</p>
        </div>
        <div class="help-section">
            <h3>Reading the Diagram</h3>
            <ul>
                <li><strong>Arc size</strong> - Proportional to total connections of that module</li>
                <li><strong>Ribbon thickness</strong> - Number of dependencies between two modules</li>
                <li><strong>Ribbon color</strong> - Matches the source module color</li>
            </ul>
        </div>
        <div class="help-section">
            <h3>Export Options</h3>
            <ul>
                <li><strong>PNG</strong> - High-resolution image for presentations</li>
                <li><strong>SVG</strong> - Vector format for editing or printing</li>
            </ul>
        </div>
    </div>
    <div id="container">
        <div id="chord"></div>
        <div id="sidebar">
            <div class="sidebar-section">
                <div class="section-title">Connection Info</div>
                <div id="info">Hover over a chord to see connection details</div>
            </div>
            <div class="sidebar-section">
                <div class="section-title">
                    <span>Modules ({len(data["groups"])})</span>
                    <div class="section-controls">
                        <button class="section-btn" onclick="selectAll()">All</button>
                        <button class="section-btn" onclick="selectNone()">None</button>
                        <button class="section-btn" onclick="addParents()">↑ Parents</button>
                        <button class="section-btn" onclick="addChildren()">↓ Children</button>
                    </div>
                </div>
                <input type="text" class="search-box" id="search" placeholder="Search modules..." oninput="filterModules()">
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
        const allGroups = data.groups;
        const fullMatrix = data.matrix;
        const sizes = data.sizes;

        let visibleGroups = new Set(allGroups.map((_, i) => i));
        let selectedGroups = new Set();

        const container = document.getElementById('chord');

        const color = d3.scaleOrdinal()
            .domain(d3.range(allGroups.length))
            .range(d3.schemeTableau10);

        // Build dependency indices (who depends on whom)
        const dependsOn = {{}};  // dependsOn[i] = set of indices i depends on
        const dependedBy = {{}};  // dependedBy[i] = set of indices that depend on i
        for (let i = 0; i < allGroups.length; i++) {{
            dependsOn[i] = new Set();
            dependedBy[i] = new Set();
        }}
        for (let i = 0; i < allGroups.length; i++) {{
            for (let j = 0; j < allGroups.length; j++) {{
                if (fullMatrix[i][j] > 0 && i !== j) {{
                    dependsOn[i].add(j);  // i depends on j
                    dependedBy[j].add(i);  // j is depended on by i
                }}
            }}
        }}

        function renderChord() {{
            const chordContainer = document.getElementById('chord');
            chordContainer.innerHTML = '';

            const visibleIndices = Array.from(visibleGroups).sort((a, b) => a - b);
            if (visibleIndices.length < 2) {{
                chordContainer.innerHTML = '<div style="color:#888;text-align:center;">Select at least 2 modules to display</div>';
                return;
            }}

            // Build filtered matrix
            const n = visibleIndices.length;
            const filteredMatrix = [];
            for (let i = 0; i < n; i++) {{
                const row = [];
                for (let j = 0; j < n; j++) {{
                    row.push(fullMatrix[visibleIndices[i]][visibleIndices[j]]);
                }}
                filteredMatrix.push(row);
            }}

            const size = Math.min(chordContainer.clientWidth, chordContainer.clientHeight) - 100;
            const outerRadius = size / 2;
            const innerRadius = outerRadius - 30;

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
                .attr('width', size + 150)
                .attr('height', size + 150)
                .append('g')
                .attr('transform', `translate(${{(size+150)/2}},${{(size+150)/2}})`);

            const chords = chord(filteredMatrix);

            // Draw groups
            const groupG = svg.append('g')
                .selectAll('g')
                .data(chords.groups)
                .join('g')
                .attr('class', 'group')
                .on('click', (event, d) => {{
                    const originalIndex = visibleIndices[d.index];
                    toggleGroupSelection(originalIndex);
                }});

            groupG.append('path')
                .attr('d', arc)
                .attr('fill', d => color(visibleIndices[d.index]))
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
                    const name = allGroups[visibleIndices[d.index]];
                    return name.length > 25 ? name.slice(0, 23) + '…' : name;
                }});

            // Draw chords
            svg.append('g')
                .attr('fill-opacity', 0.7)
                .selectAll('path')
                .data(chords)
                .join('path')
                .attr('class', 'chord')
                .attr('d', ribbon)
                .attr('fill', d => color(visibleIndices[d.source.index]))
                .attr('stroke', d => d3.rgb(color(visibleIndices[d.source.index])).darker())
                .on('mouseover', (event, d) => {{
                    showChordInfo(visibleIndices[d.source.index], visibleIndices[d.target.index], d.source.value);
                }})
                .on('mouseout', () => {{
                    document.getElementById('info').innerHTML = 'Hover over a chord to see connection details';
                }});
        }}

        function showChordInfo(srcIdx, tgtIdx, value) {{
            document.getElementById('info').innerHTML = `
                <div class="info-row"><span class="info-label">From</span><span class="info-value">${{allGroups[srcIdx]}}</span></div>
                <div class="info-row"><span class="info-label">To</span><span class="info-value">${{allGroups[tgtIdx]}}</span></div>
                <div class="info-row"><span class="info-label">Connections</span><span class="info-value">${{value}}</span></div>
            `;
        }}

        function toggleGroupSelection(idx) {{
            if (selectedGroups.has(idx)) {{
                selectedGroups.delete(idx);
            }} else {{
                selectedGroups.add(idx);
            }}
            updateGroupsList();
        }}

        function toggleGroupVisibility(idx, visible) {{
            if (visible) {{
                visibleGroups.add(idx);
            }} else {{
                visibleGroups.delete(idx);
            }}
            renderChord();
        }}

        function selectAll() {{
            allGroups.forEach((_, i) => visibleGroups.add(i));
            updateGroupsList();
            renderChord();
        }}

        function selectNone() {{
            visibleGroups.clear();
            updateGroupsList();
            renderChord();
        }}

        function addParents() {{
            // Add modules that selected modules depend on
            const toAdd = new Set();
            selectedGroups.forEach(idx => {{
                dependsOn[idx].forEach(dep => toAdd.add(dep));
            }});
            toAdd.forEach(idx => visibleGroups.add(idx));
            updateGroupsList();
            renderChord();
        }}

        function addChildren() {{
            // Add modules that depend on selected modules
            const toAdd = new Set();
            selectedGroups.forEach(idx => {{
                dependedBy[idx].forEach(dep => toAdd.add(dep));
            }});
            toAdd.forEach(idx => visibleGroups.add(idx));
            updateGroupsList();
            renderChord();
        }}

        function filterModules() {{
            const query = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.group-item').forEach(item => {{
                const name = item.dataset.name.toLowerCase();
                item.classList.toggle('hidden', !name.includes(query));
            }});
        }}

        function updateGroupsList() {{
            const list = document.getElementById('groups-list');
            list.innerHTML = '';
            allGroups.forEach((g, i) => {{
                const item = document.createElement('div');
                item.className = 'group-item' + (selectedGroups.has(i) ? ' selected' : '');
                item.dataset.name = g;
                item.innerHTML = `
                    <input type="checkbox" ${{visibleGroups.has(i) ? 'checked' : ''}} onchange="toggleGroupVisibility(${{i}}, this.checked)">
                    <span class="group-color" style="background:${{color(i)}}"></span>
                    <span class="group-name" title="${{g}}">${{g}}</span>
                    <span class="group-count">${{sizes[i]}}</span>
                `;
                item.addEventListener('click', (e) => {{
                    if (e.target.tagName !== 'INPUT') {{
                        toggleGroupSelection(i);
                    }}
                }});
                list.appendChild(item);
            }});
        }}

        // Export functions
        function exportSVG() {{
            const svgElement = document.querySelector('#chord svg');
            if (!svgElement) {{ alert('No SVG to export'); return; }}
            const serializer = new XMLSerializer();
            let source = serializer.serializeToString(svgElement);
            source = '<?xml version="1.0" encoding="UTF-8"?>' + source;
            const blob = new Blob([source], {{ type: 'image/svg+xml;charset=utf-8' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'chord-diagram.svg';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function exportPNG() {{
            const svgElement = document.querySelector('#chord svg');
            if (!svgElement) {{ alert('No SVG to export'); return; }}
            const canvas = document.createElement('canvas');
            const rect = svgElement.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = rect.height * 2;
            const ctx = canvas.getContext('2d');
            ctx.scale(2, 2);
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, rect.width, rect.height);
            const img = new Image();
            const serializer = new XMLSerializer();
            const svgStr = serializer.serializeToString(svgElement);
            img.onload = function() {{
                ctx.drawImage(img, 0, 0);
                const link = document.createElement('a');
                link.download = 'chord-diagram.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }};
            img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgStr)));
        }}

        function toggleFullscreen() {{
            document.body.classList.toggle('fullscreen');
            setTimeout(renderChord, 100);
        }}

        function toggleHelp() {{
            document.getElementById('help-panel').classList.toggle('visible');
            document.getElementById('help-overlay').classList.toggle('visible');
        }}

        // Simple loading - CSS animation handles the visual feedback
        function completeLoading() {{
            document.querySelector('.loading-text').textContent = 'Done!';
            document.getElementById('progress').classList.add('complete');
            setTimeout(() => {{
                document.getElementById('loading').classList.add('hidden');
            }}, 400);
        }}

        // Start render after showing loading screen
        setTimeout(() => {{
            document.querySelector('.loading-text').textContent = 'Building chord diagram...';
            setTimeout(() => {{
                try {{
                    updateGroupsList();
                    renderChord();
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
