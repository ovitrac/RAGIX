#!/usr/bin/env python3
"""
Standalone Radial Dependency Explorer Server

A minimal FastAPI server specifically for the live radial explorer.
Run with: python -m ragix_unix.radial_server --path /path/to/project

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("Error: FastAPI not installed.", file=sys.stderr)
    print("Install with: pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)

# Import AST components
try:
    from ragix_core.ast_java import get_java_backend  # Register Java backend
except ImportError:
    pass

from ragix_core import (
    build_dependency_graph,
    VizConfig,
    RadialExplorer,
)
from ragix_core.version import __version__ as RAGIX_VERSION

app = FastAPI(
    title="Radial Dependency Explorer",
    description="Interactive ego-centric dependency visualization",
    version=RAGIX_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
PROJECT_PATH: Optional[Path] = None
GRAPH_CACHE = None


def get_graph():
    """Get or build the dependency graph."""
    global GRAPH_CACHE
    if GRAPH_CACHE is None and PROJECT_PATH:
        print(f"Building dependency graph for {PROJECT_PATH}...")
        GRAPH_CACHE = build_dependency_graph([PROJECT_PATH])
        symbols = list(GRAPH_CACHE.get_symbols())
        stats = GRAPH_CACHE.get_stats()
        print(f"  Symbols: {len(symbols)}")
        print(f"  Dependencies: {stats.total_dependencies}")
    return GRAPH_CACHE


@app.get("/")
async def root():
    """Redirect to radial explorer."""
    if not PROJECT_PATH:
        return JSONResponse({"error": "No project path configured"}, status_code=500)
    return HTMLResponse(f'<script>window.location="/radial"</script>')


@app.get("/api/info")
async def get_info():
    """Get project info."""
    graph = get_graph()
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not available")

    symbols = list(graph.get_symbols())
    stats = graph.get_stats()
    return {
        "path": str(PROJECT_PATH),
        "symbols": len(symbols),
        "dependencies": stats.total_dependencies,
    }


@app.get("/api/radial")
async def get_radial_data(
    focal: Optional[str] = None,
    levels: int = 3
):
    """
    Get ego-centric radial graph data.

    Args:
        focal: Focal node (class name). Auto-selects if not provided.
        levels: Maximum depth levels (default 3)
    """
    from ragix_core.ast_base import NodeType

    graph = get_graph()
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not available")

    # Get all symbols
    all_symbols = {s.qualified_name: s for s in graph.get_symbols()}

    # Filter to structural types only (class, interface, enum)
    structural_types = {NodeType.CLASS, NodeType.INTERFACE, NodeType.ENUM}
    structural_symbols = {
        k: v for k, v in all_symbols.items()
        if v.node_type in structural_types
    }

    # Auto-select focal if not provided
    if not focal:
        deps = graph.get_all_dependencies()
        degree = {}
        for dep in deps:
            degree[dep.source] = degree.get(dep.source, 0) + 1
            degree[dep.target] = degree.get(dep.target, 0) + 1

        # Filter to structural symbols only
        class_degree = {
            k: v for k, v in degree.items()
            if k in structural_symbols
        }

        if class_degree:
            focal = max(class_degree.keys(), key=lambda k: class_degree[k])
        elif structural_symbols:
            # Fall back to first structural symbol
            focal = next(iter(structural_symbols.keys()))
        else:
            raise HTTPException(status_code=404, detail="No class symbols found")

    # Resolve focal node - check structural symbols first
    if focal not in structural_symbols:
        # Try to find a match in structural symbols
        matches = [s for s in structural_symbols.keys() if focal in s or s.endswith(f".{focal}") or s == focal]
        if len(matches) == 1:
            focal = matches[0]
        elif len(matches) > 1:
            # Prefer exact match or shortest
            exact = [m for m in matches if m == focal or m.endswith(f".{focal}")]
            focal = exact[0] if exact else min(matches, key=len)
        else:
            raise HTTPException(status_code=404, detail=f"Class not found: {focal}")

    # Build radial graph
    config = VizConfig()
    explorer = RadialExplorer(config)
    ego_data = explorer.build_ego_graph(graph, focal, max_levels=levels)

    return {
        "focal": focal,
        "levels": levels,
        **ego_data
    }


@app.get("/api/search")
async def search_nodes(q: str, limit: int = 20):
    """Search for nodes by name."""
    graph = get_graph()
    if not graph:
        raise HTTPException(status_code=500, detail="Graph not available")

    symbols = graph.get_symbols()
    query = q.lower()

    matches = []
    for sym in symbols:
        if query in sym.name.lower() or query in sym.qualified_name.lower():
            matches.append({
                "id": sym.qualified_name,
                "name": sym.name,
                "type": sym.node_type.value,
            })
            if len(matches) >= limit:
                break

    return {"results": matches}


@app.get("/radial", response_class=HTMLResponse)
async def radial_page():
    """Serve the radial explorer page."""
    if not PROJECT_PATH:
        return HTMLResponse("<h1>Error: No project configured</h1>", status_code=500)

    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Radial Dependency Explorer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background: #0d1117;
            color: #e6edf3;
            overflow: hidden;
        }
        #container { display: flex; height: 100vh; }
        #graph-container { flex: 1; position: relative; }
        #graph { width: 100%; height: 100%; }
        #sidebar {
            width: 350px;
            background: #161b22;
            border-left: 1px solid #30363d;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .sidebar-header {
            padding: 20px;
            background: #21262d;
            border-bottom: 1px solid #30363d;
        }
        .sidebar-header h2 {
            margin: 0 0 10px 0;
            font-size: 16px;
            color: #58a6ff;
        }
        #search {
            width: 100%;
            padding: 10px;
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #0d1117;
            color: #e6edf3;
            font-size: 14px;
        }
        #search:focus { outline: none; border-color: #58a6ff; }
        #search-results {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #30363d;
            border-top: none;
            border-radius: 0 0 6px 6px;
            display: none;
        }
        .search-result {
            padding: 8px 10px;
            cursor: pointer;
            border-bottom: 1px solid #21262d;
        }
        .search-result:hover { background: #21262d; }
        .search-result-name { font-weight: 500; }
        .search-result-type { font-size: 11px; color: #8b949e; }
        .controls {
            padding: 15px 20px;
            border-bottom: 1px solid #30363d;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        .btn {
            padding: 6px 12px;
            border: 1px solid #30363d;
            border-radius: 6px;
            background: #21262d;
            color: #e6edf3;
            cursor: pointer;
            font-size: 12px;
        }
        .btn:hover { background: #30363d; }
        .btn-primary { background: #238636; border-color: #238636; }
        .btn-primary:hover { background: #2ea043; }
        #node-info {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .info-section { margin-bottom: 20px; }
        .info-section h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #8b949e;
            text-transform: uppercase;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #21262d;
        }
        .info-label { color: #8b949e; }
        .info-value { color: #e6edf3; font-weight: 500; }
        #focal-info {
            padding: 15px 20px;
            background: #21262d;
            border-bottom: 1px solid #30363d;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        #breadcrumb {
            padding: 10px 20px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
            font-size: 12px;
            white-space: nowrap;
            overflow-x: auto;
        }
        .breadcrumb-item { color: #58a6ff; cursor: pointer; }
        .breadcrumb-item:hover { text-decoration: underline; }
        .breadcrumb-sep { color: #484f58; margin: 0 8px; }
        #levels-input {
            width: 50px;
            padding: 4px 8px;
            border: 1px solid #30363d;
            border-radius: 4px;
            background: #0d1117;
            color: #e6edf3;
            text-align: center;
        }
        .legend {
            padding: 10px 20px;
            border-top: 1px solid #30363d;
            font-size: 11px;
        }
        .legend-item { display: flex; align-items: center; margin: 4px 0; }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <svg id="graph"></svg>
            <div id="loading">
                <div class="spinner"></div>
                <p>Loading dependency graph...</p>
            </div>
        </div>
        <div id="sidebar">
            <div class="sidebar-header">
                <h2>Radial Explorer</h2>
                <input type="text" id="search" placeholder="Search classes..." autocomplete="off">
                <div id="search-results"></div>
            </div>
            <div id="breadcrumb"></div>
            <div class="controls">
                <button class="btn" onclick="resetView()">Reset View</button>
                <button class="btn" onclick="exportSVG()">Export SVG</button>
                <label style="font-size: 12px; color: #8b949e;">Levels:</label>
                <input type="number" id="levels-input" value="3" min="1" max="5">
                <button class="btn btn-primary" onclick="reloadWithLevels()">Apply</button>
            </div>
            <div id="focal-info"></div>
            <div id="node-info">
                <p style="color: #8b949e;">Click a node to see details<br>Double-click to explore</p>
            </div>
            <div class="legend">
                <div class="legend-item"><span class="legend-color" style="background: #f97583;"></span> Inheritance</div>
                <div class="legend-item"><span class="legend-color" style="background: #b392f0;"></span> Implementation</div>
                <div class="legend-item"><span class="legend-color" style="background: #79c0ff;"></span> Type Reference</div>
                <div class="legend-item"><span class="legend-color" style="background: #56d364;"></span> Method Call</div>
            </div>
        </div>
    </div>

    <script>
        let currentLevels = 3;
        let egoData = null;
        let selectedNode = null;
        let focalHistory = [];
        let searchTimeout = null;

        const svg = d3.select('#graph');
        const g = svg.append('g');
        const linksLayer = g.append('g').attr('class', 'links');
        const nodesLayer = g.append('g').attr('class', 'nodes');
        const labelsLayer = g.append('g').attr('class', 'labels');

        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (e) => g.attr('transform', e.transform));
        svg.call(zoom);

        // Load initial data
        loadData(null);

        async function loadData(focal) {
            document.getElementById('loading').style.display = 'block';

            const url = new URL('/api/radial', window.location.origin);
            url.searchParams.set('levels', currentLevels);
            if (focal) url.searchParams.set('focal', focal);

            try {
                const resp = await fetch(url);
                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || 'Unknown error');
                }
                egoData = await resp.json();

                if (!focalHistory.length || focalHistory[focalHistory.length - 1] !== egoData.focal) {
                    focalHistory.push(egoData.focal);
                }

                render();
                updateFocalInfo();
                updateBreadcrumb();
            } catch (e) {
                alert('Error: ' + e.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function render() {
            const width = svg.node().clientWidth;
            const height = svg.node().clientHeight;
            const centerX = width / 2;
            const centerY = height / 2;
            const maxLevel = Math.max(...egoData.nodes.map(n => n.level), 1);
            const levelRadius = Math.min(width, height) / (2 * (maxLevel + 1.5));

            // Position nodes
            const nodesByLevel = {};
            egoData.nodes.forEach(n => {
                nodesByLevel[n.level] = nodesByLevel[n.level] || [];
                nodesByLevel[n.level].push(n);
            });

            egoData.nodes.forEach(n => {
                if (n.level === 0) {
                    n.x = centerX;
                    n.y = centerY;
                } else {
                    const nodesAtLevel = nodesByLevel[n.level];
                    const idx = nodesAtLevel.indexOf(n);
                    const angle = (2 * Math.PI * idx) / nodesAtLevel.length - Math.PI / 2;
                    const radius = levelRadius * n.level;
                    n.x = centerX + radius * Math.cos(angle);
                    n.y = centerY + radius * Math.sin(angle);
                }
            });

            // Draw level circles
            g.selectAll('.level-circle').remove();
            for (let l = 1; l <= maxLevel; l++) {
                g.insert('circle', ':first-child')
                    .attr('class', 'level-circle')
                    .attr('cx', centerX)
                    .attr('cy', centerY)
                    .attr('r', levelRadius * l)
                    .attr('fill', 'none')
                    .attr('stroke', '#21262d')
                    .attr('stroke-dasharray', '4,4');
            }

            // Create node map
            const nodeMap = {};
            egoData.nodes.forEach(n => nodeMap[n.id] = n);

            // Draw links
            const links = linksLayer.selectAll('.link')
                .data(egoData.links, d => `${d.source}|${d.target}`);

            links.exit().remove();

            links.enter()
                .append('path')
                .attr('class', 'link')
                .merge(links)
                .attr('d', d => {
                    const s = nodeMap[d.source];
                    const t = nodeMap[d.target];
                    if (!s || !t) return '';
                    const dx = t.x - s.x;
                    const dy = t.y - s.y;
                    const dr = Math.sqrt(dx * dx + dy * dy) * 0.8;
                    return `M${s.x},${s.y}A${dr},${dr} 0 0,1 ${t.x},${t.y}`;
                })
                .attr('fill', 'none')
                .attr('stroke', d => getTypeColor(d.type))
                .attr('stroke-width', d => Math.min(d.count || 1, 5))
                .attr('stroke-opacity', 0.6);

            // Draw nodes
            const nodes = nodesLayer.selectAll('.node')
                .data(egoData.nodes, d => d.id);

            nodes.exit().remove();

            nodes.enter()
                .append('circle')
                .attr('class', 'node')
                .merge(nodes)
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', d => d.level === 0 ? 20 : Math.max(8, Math.min(15, 5 + (d.total_deps || 0) / 5)))
                .attr('fill', d => d.color || '#4a90d9')
                .attr('stroke', '#0d1117')
                .attr('stroke-width', 2)
                .style('cursor', 'pointer')
                .on('click', (e, d) => selectNode(d))
                .on('dblclick', (e, d) => { if (d.level > 0) loadData(d.id); });

            // Draw labels
            const labels = labelsLayer.selectAll('.node-label')
                .data(egoData.nodes.filter(n => n.level <= 1), d => d.id);

            labels.exit().remove();

            labels.enter()
                .append('text')
                .attr('class', 'node-label')
                .merge(labels)
                .attr('x', d => d.x)
                .attr('y', d => d.y + (d.level === 0 ? 35 : 25))
                .attr('text-anchor', 'middle')
                .attr('fill', '#e6edf3')
                .attr('font-size', d => d.level === 0 ? '14px' : '11px')
                .attr('font-weight', d => d.level === 0 ? 'bold' : 'normal')
                .text(d => d.name);

            // Reset zoom
            svg.call(zoom.transform, d3.zoomIdentity);
        }

        function getTypeColor(type) {
            const colors = {
                'inheritance': '#f97583',
                'implementation': '#b392f0',
                'type_reference': '#79c0ff',
                'call': '#56d364',
                'composition': '#ffa657',
                'instantiation': '#ff7b72'
            };
            return colors[type] || '#8b949e';
        }

        function selectNode(node) {
            selectedNode = node;
            nodesLayer.selectAll('.node')
                .attr('stroke', d => d.id === node.id ? '#58a6ff' : '#0d1117')
                .attr('stroke-width', d => d.id === node.id ? 3 : 2);
            updateNodeInfo(node);
        }

        function updateNodeInfo(node) {
            if (!node) {
                document.getElementById('node-info').innerHTML = '<p style="color: #8b949e;">Click a node to see details<br>Double-click to explore</p>';
                return;
            }
            document.getElementById('node-info').innerHTML = `
                <div class="info-section">
                    <h3>Selected Node</h3>
                    <div class="info-row">
                        <span class="info-label">Name</span>
                        <span class="info-value">${node.name}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Type</span>
                        <span class="info-value">${node.type}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Level</span>
                        <span class="info-value">${node.level}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Deps Out</span>
                        <span class="info-value">${node.out_deps || 0}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Deps In</span>
                        <span class="info-value">${node.in_deps || 0}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">File</span>
                        <span class="info-value" style="font-size: 10px; word-break: break-all;">${node.file || 'N/A'}</span>
                    </div>
                </div>
                ${node.level > 0 ? `
                <button class="btn btn-primary" onclick="loadData('${node.id}')" style="width: 100%;">
                    Explore as Center
                </button>
                ` : ''}
            `;
        }

        function updateFocalInfo() {
            const focal = egoData.nodes.find(n => n.level === 0);
            if (!focal) return;
            const levelCounts = {};
            egoData.nodes.forEach(n => levelCounts[n.level] = (levelCounts[n.level] || 0) + 1);

            document.getElementById('focal-info').innerHTML = `
                <div style="font-weight: bold; color: #58a6ff; margin-bottom: 5px;">${focal.name}</div>
                <div style="font-size: 12px; color: #8b949e;">
                    ${Object.entries(levelCounts).map(([l, c]) => `L${l}: ${c}`).join(' | ')}
                </div>
            `;
        }

        function updateBreadcrumb() {
            const container = document.getElementById('breadcrumb');
            if (focalHistory.length <= 1) {
                container.innerHTML = '';
                return;
            }
            container.innerHTML = focalHistory.map((f, i) => {
                const name = f.split('.').pop();
                const isLast = i === focalHistory.length - 1;
                return isLast
                    ? `<span style="color: #e6edf3;">${name}</span>`
                    : `<span class="breadcrumb-item" onclick="navigateTo(${i})">${name}</span><span class="breadcrumb-sep">›</span>`;
            }).join('');
        }

        function navigateTo(index) {
            const target = focalHistory[index];
            focalHistory = focalHistory.slice(0, index);
            loadData(target);
        }

        function resetView() {
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
        }

        function reloadWithLevels() {
            currentLevels = parseInt(document.getElementById('levels-input').value) || 3;
            const currentFocal = egoData?.focal;
            focalHistory = [];
            loadData(currentFocal);
        }

        function exportSVG() {
            const svgData = new XMLSerializer().serializeToString(svg.node());
            const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'radial-deps.svg';
            a.click();
            URL.revokeObjectURL(url);
        }

        // Search functionality
        const searchInput = document.getElementById('search');
        const searchResults = document.getElementById('search-results');

        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim();
            if (searchTimeout) clearTimeout(searchTimeout);

            if (!query) {
                searchResults.style.display = 'none';
                nodesLayer.selectAll('.node').attr('opacity', 1);
                return;
            }

            // Highlight matching nodes in view
            nodesLayer.selectAll('.node')
                .attr('opacity', d => d.name.toLowerCase().includes(query.toLowerCase()) ? 1 : 0.2);

            // Search API
            searchTimeout = setTimeout(async () => {
                try {
                    const resp = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=10`);
                    const data = await resp.json();

                    if (data.results.length) {
                        searchResults.innerHTML = data.results.map(r => `
                            <div class="search-result" onclick="loadData('${r.id}')">
                                <div class="search-result-name">${r.name}</div>
                                <div class="search-result-type">${r.type}</div>
                            </div>
                        `).join('');
                        searchResults.style.display = 'block';
                    } else {
                        searchResults.style.display = 'none';
                    }
                } catch (e) {
                    console.error('Search error:', e);
                }
            }, 300);
        });

        searchInput.addEventListener('blur', () => {
            setTimeout(() => searchResults.style.display = 'none', 200);
        });

        // Handle window resize
        window.addEventListener('resize', () => { if (egoData) render(); });
    </script>
</body>
</html>'''
    return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Radial Dependency Explorer Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ragix_unix.radial_server --path /path/to/java/project
  python -m ragix_unix.radial_server --path ./my-app --port 8888

Then open http://localhost:8080/radial in your browser.
"""
    )
    parser.add_argument(
        "--path", "-p",
        required=True,
        help="Path to project directory to analyze"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run server on (default: 8080)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    global PROJECT_PATH
    PROJECT_PATH = Path(args.path).expanduser().resolve()

    if not PROJECT_PATH.exists():
        print(f"Error: Path not found: {PROJECT_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Radial Dependency Explorer")
    print(f"=" * 40)
    print(f"Project: {PROJECT_PATH}")
    print(f"Server:  http://{args.host}:{args.port}/radial")
    print(f"=" * 40)

    # Pre-load graph
    get_graph()

    print(f"\nStarting server...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
