# RAGIX v0.10.1 - Advanced Visualization Complete

**Updated:** 2025-11-27
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## Current State

### Completed in v0.10.0
- ✅ Multi-language AST (Python + Java via javalang)
- ✅ Dependency graph with cycle detection
- ✅ AST query language (pattern-based search)
- ✅ Professional code metrics (cyclomatic complexity, technical debt)
- ✅ Maven POM parsing
- ✅ SonarQube/SonarCloud client
- ✅ Basic visualization (DOT, Mermaid, D3.js JSON, static HTML)
- ✅ CLI: `ragix-ast` with 10 subcommands

### Completed in v0.10.1 (This Session)
- ✅ **Enhanced HTML Renderer** - Package clustering with convex hulls
- ✅ **Edge bundling** - Cleaner inter-package dependencies visualization
- ✅ **Interactive controls** - Zoom, pan, search, type filters
- ✅ **Minimap** - Large graph navigation
- ✅ **Dependency Structure Matrix (DSM)** - Cycle detection, heatmap
- ✅ **AST API endpoints** - 6 new REST endpoints in web server
- ✅ **dependency_explorer.js** - Reusable D3.js component (600+ lines)
- ✅ **CLI: `ragix-ast matrix`** - New DSM command with CSV/JSON export
- ✅ **Radial Explorer** - Ego-centric visualization with focal node at center
- ✅ **CLI: `ragix-ast radial`** - New radial command with auto-selection

### Tested Successfully on Enterprise Codebase
- **Project:** 1,315 Java files, 18,210 symbols, 45,113 dependencies
- **Analysis time:** ~10 seconds
- **Technical debt:** 362.2 hours estimated
- **Hotspots:** VueToVueDtoConverter.convertTo (CC=71)
- **Visualization files generated:**
  - `/tmp/enterprise_deps_enhanced.html` (827KB) - Force-directed graph
  - `/tmp/enterprise_matrix_pkg.html` (254KB) - Package-level DSM
  - `/tmp/enterprise_matrix_class.html` (84KB) - Class-level DSM
  - `/tmp/enterprise_radial_final.html` (123KB) - Radial ego-centric explorer

---

## Quick Reference Commands

```bash
# Generate enhanced dependency visualization
ragix-ast graph /path/to/project --format html --output deps.html

# Generate dependency matrix (DSM)
ragix-ast matrix /path/to/project --level package --output matrix.html
ragix-ast matrix /path/to/project --level class --csv  # Export as CSV

# Generate radial ego-centric explorer
ragix-ast radial /path/to/project --output radial.html  # Auto-select focal
ragix-ast radial /path/to/project --focal ClassName --levels 3 --output radial.html

# Get metrics via API (requires web server)
curl "http://localhost:8080/api/ast/metrics?path=/path/to/project"
curl "http://localhost:8080/api/ast/hotspots?path=/path/to/project&limit=20"

# Open visualizations
xdg-open /tmp/project_deps_enhanced.html
xdg-open /tmp/project_matrix_pkg.html
xdg-open /tmp/project_radial_auto.html
```

---

## New API Endpoints (v0.10.1)

```
GET  /api/ast/status              # Check if AST analysis is available
GET  /api/ast/graph?path=...      # Get dependency graph as D3.js JSON
GET  /api/ast/metrics?path=...    # Get code metrics
GET  /api/ast/search?path=...&q=  # Search for symbols
GET  /api/ast/hotspots?path=...   # Get complexity hotspots
GET  /api/ast/visualize?path=...  # Generate HTML visualization
GET  /api/ast/radial?path=...     # Get ego-centric radial graph data
GET  /api/ast/radial/page?path=.. # Live interactive radial explorer page
```

### Live Radial Explorer

The `/api/ast/radial/page` endpoint serves a fully interactive radial explorer:

```bash
# Start the web server
ragix-web --port 8080

# Open the live radial explorer
xdg-open "http://localhost:8080/api/ast/radial/page?path=/path/to/project"

# With specific focal node and levels
xdg-open "http://localhost:8080/api/ast/radial/page?path=/path/to/project&focal=ProfilDto&levels=3"
```

Features:
- **Double-click to refocus**: Instantly loads new data without page reload
- **Breadcrumb navigation**: Go back to previous focal nodes
- **Adjustable levels**: Change depth on the fly
- **Real-time search**: Filter visible nodes
- **SVG export**: Download current view

### Standalone Radial Explorer Server

For production use without the full ragix-web stack, use the standalone radial server:

```bash
# Start the standalone radial server
python -m ragix_unix.radial_server --path /path/to/project --port 8090

# Open in browser
xdg-open "http://localhost:8090/radial"
```

**Endpoints:**
- `GET /` - Redirects to `/radial`
- `GET /api/info` - Project info (symbols, dependencies count)
- `GET /api/radial?focal=ClassName&levels=3` - Get radial graph data
- `GET /api/search?q=query` - Search for classes
- `GET /radial` - Interactive radial explorer page

**Key Features:**
- Lightweight FastAPI server (~800 lines)
- Graph caching (builds once, serves many requests)
- Auto-selects highest-connectivity class as initial focal
- Only shows structural types (class, interface, enum)
- Real-time search with autocomplete

---

## Key Files Updated

```
ragix_core/
├── ast_viz.py           # HTMLRenderer + DSMRenderer + RadialExplorer (2700+ lines)
├── dependencies.py      # Fixed import dependency source extraction
├── __init__.py          # Added DSMRenderer, RadialExplorer exports

ragix_unix/
├── ast_cli.py           # Added matrix, radial commands (1000+ lines, 12 commands)
├── radial_server.py     # Standalone radial explorer server (800+ lines)

ragix_web/
├── server.py            # Added 8 AST API endpoints including live radial (v0.10.1)
├── static/js/
│   └── dependency_explorer.js  # D3.js component (600+ lines)
```

---

## Visualization Features

### Force-Directed Graph (graph --format html)
- **Package clustering**: Nodes grouped by Java package with convex hulls
- **Edge bundling**: Curved edges between clusters for clarity
- **Node coloring**: By type (class=blue, interface=green, method=orange)
- **Interactive**: Click to select, search, filter by type
- **Minimap**: Overview of large graphs
- **Export**: SVG download button

### Dependency Matrix (matrix command)
- **Heatmap**: Cell color indicates dependency strength
- **Cycle detection**: Red cells for bidirectional dependencies
- **Aggregation**: Class-level or package-level views
- **Export**: CSV or JSON for further analysis
- **Interactive**: Hover for details, click to highlight

### Radial Explorer (radial command)
- **Ego-centric**: Selected class at center, dependencies radiating outward
- **Multi-level**: Concentric circles for Level 1, 2, 3 dependencies (configurable with `--levels`)
- **Class-focused**: Shows only classes, interfaces, and enums (filters out methods/imports)
- **Arc connections**: Colored by dependency type, thickness by connection count
- **Interactive**: Click to select, double-click to get command for refocusing
- **Auto-selection**: Automatically picks highest-connectivity project class
- **Search**: Find and explore any class in the project
- **Clipboard integration**: "Explore as Center" copies the ragix-ast command

---

## Next Priorities

### Phase 1: Remaining Visualization Types
1. **Treemap** - Package hierarchy by LOC/complexity
2. **Sunburst** - Module structure drill-down
3. **Chord diagram** - Inter-module dependencies

### Phase 2: Git Integration
1. Complexity evolution over commits
2. Hotspot emergence tracking
3. Technical debt accumulation

### Phase 3: Report Generation
1. PDF/HTML executive summary
2. Version comparison
3. Trend analysis

---

## Dependencies

```
javalang>=0.13.0    # Java AST parsing
jsonschema>=4.17.0  # Schema validation
requests>=2.31.0    # For Sonar API
d3.js v7            # Browser visualization (CDN)
```

---

## Enterprise Project Reference

```
/path/to/enterprise/audit/
├── app-bpm-main/    # BPM module
├── app-hab-main/    # HAB module (main, tested)
└── app-pre-main/    # PRE module
```

---

**Status:** Phase 1 Visualization Complete - Ready for Production Use
