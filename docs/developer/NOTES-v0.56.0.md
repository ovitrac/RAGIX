> **RAGIX 0.56.0** delivers **improved dead code detection**, **enhanced partitioner UI**, and all features from 0.55 including **enterprise-grade code audit**, **codebase partitioning**, and **MDS graph layout**.

------

# RAGIX v0.56.0 — Enterprise Code Audit & Codebase Partitioner

This release transforms RAGIX from a development assistant into a **professional code audit platform** capable of analyzing enterprise Java codebases with 15,000+ classes. Key capabilities:

- **Codebase Partitioner** — Automatic application boundary detection with D3.js force-directed visualization
- **Enterprise Service Detection** — Support for multiple service naming conventions (SK/SC/SG, spre##, custom patterns)
- **MDS Graph Layout** — Precomputed positions using eigendecomposition for large graphs
- **Multi-Module Maven Support** — Auto-detect complex project structures
- **Code Tracker** — Outliers, complexity hotspots, dead code, coupling issues
- **Project RAG** — ChromaDB vector store with concept exploration

---

## What's New in v0.56.0

| Category | Feature | Description |
|----------|---------|-------------|
| **Partitioner** | Force-Directed Graph | D3.js physics simulation with partition clustering |
| **Partitioner** | MDS Layout | Eigendecomposition-based positioning for <500 nodes |
| **Partitioner** | Partition-Based Layout | O(n) circular layout for >500 nodes (17K+ supported) |
| **Partitioner** | Export System | SVG, PNG, JSON, CSV formats |
| **Audit** | Service Pattern Presets | ACME-ERP (SK/SC/SG), Enterprise (spre##), Combined |
| **Audit** | Multi-Module Maven | Auto-detect `app-*/src/main/java/` structures |
| **Audit** | Risk Analysis | Timeline-based risk scoring with configurable weights |
| **Audit** | Service Lifecycle | NEW/ACTIVE/MATURE/LEGACY categorization |
| **Code Tracker** | Complexity Hotspots | Identify methods with CC > threshold |
| **Code Tracker** | Dead Code Detection | Isolated class detection (no callers AND no callees) |
| **Code Tracker** | Coupling Issues | Ca/Ce/Instability metrics per package |
| **Project RAG** | Concept Explorer | Dual-view (files + D3.js graph) |
| **Project RAG** | Knowledge Summary | LLM-powered concept summarization |
| **UI** | Audit Settings | Configurable thresholds, weights, service patterns |
| **UI** | Label Visibility | All node labels visible at max zoom |
| **UI** | Line Navigation | Smooth scrolling in code preview |

**Version History:**
- **v0.33** — Agentic reasoning, threads, RAG router
- **v0.35** — Project RAG with concept exploration
- **v0.40** — Code audit capabilities, service detection
- **v0.50** — Code tracker, RAG stats integration
- **v0.51** — Project discovery, type safety fixes
- **v0.55** — Codebase partitioner, MDS layout, enterprise patterns
- **v0.56** — Improved dead code detection (isolated classes only), partitioner UI

---

## Codebase Partitioner (New in v0.55)

The Partitioner analyzes dependency graphs to automatically identify application boundaries within monolithic codebases — essential for modernization and refactoring planning.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Codebase Partitioner                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Configuration          Visualization         Export            │
│  ┌─────────────┐       ┌─────────────┐      ┌─────────────┐    │
│  │ Application │       │ D3.js Force │      │ SVG / PNG   │    │
│  │ Patterns    │  ───► │ Directed    │ ───► │ JSON / CSV  │    │
│  │ (pkg/class/ │       │ Graph       │      │ XLSX        │    │
│  │  keyword)   │       │             │      │             │    │
│  └─────────────┘       └─────────────┘      └─────────────┘    │
│        │                     │                                  │
│        ▼                     ▼                                  │
│  ┌─────────────┐       ┌─────────────┐                         │
│  │ Fingerprint │       │ MDS / Part. │                         │
│  │ Matching    │       │ Layout      │                         │
│  └─────────────┘       └─────────────┘                         │
│        │                     │                                  │
│        ▼                     ▼                                  │
│  ┌─────────────┐       ┌─────────────┐                         │
│  │ Graph       │       │ Auto-Stop   │                         │
│  │ Propagation │       │ Simulation  │                         │
│  └─────────────┘       └─────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Fingerprint Matching

Classes are assigned to partitions based on configurable patterns:

```python
# Example configuration
{
    "applications": [
        {
            "name": "APP_A",
            "color": "#4fc3f7",
            "patterns": {
                "packages": ["com.company.app_a"],
                "classes": ["*ServiceA*", "*ControllerA*"],
                "keywords": ["module_a", "feature_a"]
            }
        },
        {
            "name": "APP_B",
            "color": "#81c784",
            "patterns": {
                "packages": ["com.company.app_b"],
                "classes": ["*ServiceB*"],
                "keywords": ["module_b"]
            }
        }
    ],
    "shared_patterns": {
        "packages": ["com.company.common", "com.company.util"],
        "classes": ["*Utils*", "*Helper*", "*Constants*"]
    }
}
```

### Graph Propagation

Unassigned classes are propagated based on dependency relationships:

1. **Fingerprint matching** — Direct pattern match → High confidence (0.9)
2. **Import propagation** — Majority of imports from partition → Medium confidence (0.7)
3. **Reverse propagation** — Used by partition classes → Low confidence (0.5)

### Dead Code Detection

Classes are classified as **DEAD_CODE** only when they are **completely isolated**:

```
DEAD_CODE criteria:
├── No incoming dependencies (nobody calls this class)
├── No outgoing dependencies (this class doesn't call anything)
└── Not an entry point pattern (*Controller, *Test, *Main, etc.)
```

**Why this approach?**

In Java/Spring applications, many classes appear to have no callers in static analysis but are actually used via:
- **Dependency Injection** — Spring wires `@Service`, `@Repository`, `@Component` via reflection
- **JPA/ORM** — Entities are loaded by Hibernate, not direct calls
- **Event listeners** — `@EventListener` methods are invoked by the framework
- **Scheduled tasks** — `@Scheduled` methods run via Spring scheduler

A class that **calls other classes but has no callers** is likely still active (loaded via reflection). Only classes with **no connections at all** are truly dead code.

| Condition | Classification |
|-----------|----------------|
| Has callers | **LIVE** |
| No callers, but calls others | **LIVE** (likely DI/reflection) |
| No callers AND no callees | **DEAD_CODE** |
| Entry point pattern (*Controller, *Test) | **LIVE** (always) |

### MDS Layout Algorithm

For graphs with <500 nodes, we use **Classical Multidimensional Scaling**:

```python
def _mds_layout(node_ids, width, height):
    """
    Classical MDS using eigendecomposition.

    1. Build shortest-path distance matrix (Floyd-Warshall for small graphs)
    2. Double-center the squared distance matrix
    3. Extract top 2 eigenvectors
    4. Scale to viewport dimensions
    """
    n = len(node_ids)

    # Build adjacency and compute shortest paths
    D = floyd_warshall(adjacency_matrix)

    # Double centering: B = -0.5 * J * D² * J where J = I - 1/n * 11'
    D_sq = D ** 2
    row_mean = D_sq.mean(axis=1, keepdims=True)
    col_mean = D_sq.mean(axis=0, keepdims=True)
    grand_mean = D_sq.mean()
    B = -0.5 * (D_sq - row_mean - col_mean + grand_mean)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]

    # Extract top 2 dimensions
    coords = eigenvectors[:, idx[:2]] * np.sqrt(eigenvalues[idx[:2]])

    # Scale to viewport
    return scale_to_viewport(coords, width, height)
```

### Partition-Based Layout (Large Graphs)

For graphs with >500 nodes (like enterprise codebases with 17K+ classes), we use a fast **partition-based circular layout**:

```python
def _partition_based_layout(node_ids, width, height):
    """
    O(n) layout using partition clustering.

    1. Group nodes by partition assignment
    2. Arrange partitions in a circle
    3. Place nodes in spiral pattern within partition sector
    """
    # Group by partition
    partition_groups = defaultdict(list)
    for node_id in node_ids:
        partition = assignments.get(node_id, "UNKNOWN")
        partition_groups[partition].append(node_id)

    # Arrange partitions in circle
    center_x, center_y = width / 2, height / 2
    radius = min(width, height) * 0.35

    positions = {}
    for i, (partition, nodes) in enumerate(partition_groups.items()):
        # Partition center on circle
        angle = 2 * pi * i / len(partition_groups)
        px = center_x + radius * cos(angle)
        py = center_y + radius * sin(angle)

        # Spiral arrangement within partition
        for j, node_id in enumerate(nodes):
            r = 20 + j * 3  # Expanding spiral
            a = angle + j * 0.3
            positions[node_id] = (px + r * cos(a), py + r * sin(a))

    return positions
```

### Force-Directed Visualization

The D3.js visualization includes:

| Control | Function |
|---------|----------|
| **Zoom/Pan** | Mouse wheel zoom, drag to pan |
| **Force Sliders** | Repulsion, Link Distance, Link Strength, Center Gravity, Collision |
| **Force Presets** | Separated, Clustered, Balanced configurations |
| **Labels Toggle** | Show/hide node labels |
| **Connections Toggle** | Show/hide cross-partition edges |
| **Auto-Stop** | Simulation stabilizes with precomputed positions |

### Export Formats

| Format | Content |
|--------|---------|
| **SVG** | Vector graphics for documentation |
| **PNG** | High-resolution raster (2x resolution) |
| **JSON** | Full metadata: FQN, partition, confidence, evidence, LOC |
| **CSV** | Spreadsheet-ready format |
| **XLSX** | Excel workbook with multiple sheets |

### API Endpoints

```
POST /api/ast/partition          # Run partition analysis
GET  /api/ast/partition/presets  # Get preset configurations
GET  /api/ast/partition/export   # Export results (format query param)
GET  /api/ast/partition/status   # Check partitioner availability
```

---

## Enterprise Service Detection

RAGIX v0.55 supports multiple service naming conventions for enterprise Java applications:

### Supported Patterns

| Pattern Type | Examples | Use Case |
|--------------|----------|----------|
| **SK/SC/SG** | `SK01`, `SC04`, `SG02` | Service Keys, Screen Codes, General Services |
| **spre##** | `spre28ws`, `spre13`, `sprebpm` | Enterprise web services, JMS handlers |
| **s[Action]** | `sAffecterTache`, `sCloturerIncident` | Task operations |
| **Custom** | User-defined regex patterns | Project-specific conventions |

### Detection Sources

```python
# Annotation patterns
@Service("SK04")
@Component("spre28ws")
@Component(value = "sAffecterTache")

# Package patterns
com.company.iok.sk04.*          → SK04 service
com.company.ws.spre28.*         → spre28 web service
com.company.jms.spre13.*        → spre13 JMS handler
```

### Configuration UI

The Audit Settings modal includes a **Service Detection Patterns** section:

```
┌─────────────────────────────────────────────────────────────────┐
│ Service Detection Patterns                                       │
├─────────────────────────────────────────────────────────────────┤
│ Service ID Patterns:          │ Package Patterns:               │
│ ┌─────────────────────────┐   │ ┌─────────────────────────────┐ │
│ │ SK\d{2}                 │   │ │ fr\.iowizmi\.iok\.sk        │ │
│ │ SC\d{2}                 │   │ │ fr\.iowizmi\.iok\.sc        │ │
│ │ SG\d{2}                 │   │ │ com\..*\.ws\.spre           │ │
│ │ spre\d{2}(?:ws)?        │   │ │ com\..*\.jms\.spre          │ │
│ │ sprebpm(?:ws)?          │   │ └─────────────────────────────┘ │
│ │ spremail                │   │                                 │
│ └─────────────────────────┘   │ Presets: [ACME-ERP ▼]           │
└─────────────────────────────────────────────────────────────────┘
```

### Presets

| Preset | Patterns |
|--------|----------|
| **ACME-ERP** | `SK\d{2}`, `SC\d{2}`, `SG\d{2}` |
| **Enterprise** | `spre\d{2}(?:ws)?`, `sprebpm(?:ws)?`, `spremail`, `s[A-Z][a-zA-Z]+` |
| **Combined** | All patterns from both presets |

---

## Multi-Module Maven Support

Enterprise Java projects often use complex multi-module structures. RAGIX v0.55 auto-detects these patterns:

### Supported Structures

```
project-root/
├── src/                    # Empty or config only
├── app-module-a/
│   └── submodule-1/
│       └── src/main/java/  # Actual source code
├── app-module-b/
│   └── src/main/java/
└── pom.xml                 # Parent POM
```

### Detection Algorithm

```python
def _get_source_path(project_path: Path) -> Path:
    """
    Auto-detect source path for multi-module Maven projects.

    1. Check if src/ exists
    2. If src/ has no Java files, scan from project root
    3. Otherwise use src/
    """
    src_path = project_path / "src"
    if src_path.exists():
        java_files = list(src_path.rglob("*.java"))[:1]
        if not java_files:
            # Multi-module project: scan from root
            return project_path
        return src_path
    return project_path
```

This enables analysis of codebases with:
- 17,000+ classes
- 500+ packages
- Multiple independent modules
- Shared libraries

---

## Code Tracker (v0.50 - v0.55)

The Code Tracker provides actionable insights into code quality:

### Tabs

| Tab | Content |
|-----|---------|
| **Outliers** | Files with metrics outside normal distribution |
| **High Complexity** | Methods with CC > threshold (default: 10) |
| **Dead Code** | Completely isolated classes (no callers AND no callees) |
| **Coupling Issues** | Packages in Zone of Pain or Uselessness |

### Statistics

| Metric | Description |
|--------|-------------|
| **Entropy** | Shannon entropy (bits), normalized (0-100%) |
| **Gini** | Concentration coefficient (0=equal, 1=concentrated) |
| **CR-4** | Concentration ratio of top 4 components |
| **Herfindahl** | Sum of squared market shares |

### Coupling Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Ca** | Afferent coupling | Incoming dependencies |
| **Ce** | Efferent coupling | Outgoing dependencies |
| **I** | Ce / (Ca + Ce) | Instability (0=stable, 1=unstable) |
| **A** | #abstract / #total | Abstractness |
| **D** | \|A + I - 1\| | Distance from main sequence |

### Zones

| Zone | Condition | Risk |
|------|-----------|------|
| **Pain** | A≈0, I≈0 | Concrete + stable = rigid |
| **Uselessness** | A≈1, I≈1 | Abstract + unstable = unused |
| **Main Sequence** | D≈0 | Balanced design |
| **Balanced** | Everything else | Normal |

---

## Project RAG (v0.35 - v0.55)

### Two-Level Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      Project RAG                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Project-Level (ChromaDB)          Chat-Level (BM25)          │
│  ┌─────────────────────┐          ┌─────────────────────┐     │
│  │ .RAG/ folder        │          │ .ragix/ folder      │     │
│  │ - Code chunks       │          │ - Uploaded docs     │     │
│  │ - Doc chunks        │          │ - Chat history      │     │
│  │ - Concept index     │          │ - Session context   │     │
│  └─────────────────────┘          └─────────────────────┘     │
│           │                                │                   │
│           ▼                                ▼                   │
│  ┌─────────────────────┐          ┌─────────────────────┐     │
│  │ Concept Explorer    │          │ RAG Context         │     │
│  │ - File view         │          │ - BM25 retrieval    │     │
│  │ - Graph view        │          │ - Chunk injection   │     │
│  │ - Knowledge summary │          │ - Auto-BYPASS mode  │     │
│  └─────────────────────┘          └─────────────────────┘     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Concept Explorer

| View | Features |
|------|----------|
| **File View** | Chunk previews, file navigation, chunk highlighting |
| **Graph View** | D3.js force-directed concept relationships |
| **Knowledge Summary** | LLM-generated summary with citations |

### Chunk Metadata

Each chunk includes:
- `file_path`: Source file
- `start_line`, `end_line`: Line range
- `cc_estimate`: Cyclomatic complexity estimate
- `is_complex`: Boolean flag for high complexity
- `is_code`: Boolean flag for code vs documentation

---

## Risk Analysis

### Timeline-Based Risk Scoring

```python
risk_score = (
    weights['age'] * age_score +
    weights['volatility'] * volatility_score +
    weights['complexity'] * complexity_score +
    weights['doc_coverage'] * doc_gap_score +
    weights['test_coverage'] * test_gap_score
)
```

### Configurable Weights

| Weight | Default | Description |
|--------|---------|-------------|
| **Age** | 0.20 | Older = higher risk |
| **Volatility** | 0.20 | More changes = higher risk |
| **Complexity** | 0.20 | Higher CC = higher risk |
| **Doc Coverage** | 0.15 | Less docs = higher risk |
| **Test Coverage** | 0.20 | Less tests = higher risk |

### Risk Levels

| Level | Score Range | Color |
|-------|-------------|-------|
| **Low** | 0.0 - 0.3 | Green |
| **Medium** | 0.3 - 0.6 | Yellow |
| **High** | 0.6 - 0.8 | Orange |
| **Critical** | 0.8 - 1.0 | Red |

### Service Lifecycle Categories

| Category | Criteria |
|----------|----------|
| **NEW** | Age < 180 days |
| **ACTIVE** | High volatility, regular updates |
| **MATURE** | Stable, well-documented |
| **LEGACY** | Age > 3 years, low activity |

---

## UI Improvements

### Audit Settings Modal

```
┌─────────────────────────────────────────────────────────────────┐
│ ⚙ Audit Settings                                           ✕   │
├─────────────────────────────────────────────────────────────────┤
│ Thresholds                      │ Risk Weights                  │
│ ┌─────────────────────────────┐ │ ┌────────────────────────────┐│
│ │ Drift Alert (days): [90  ]  │ │ │ Age:          [0.20]      ││
│ │ New Component (days): [180] │ │ │ Volatility:   [0.20]      ││
│ │ Legacy Threshold (yrs): [3] │ │ │ Complexity:   [0.20]      ││
│ └─────────────────────────────┘ │ │ Doc Coverage: [0.15]      ││
│                                 │ │ Test Coverage:[0.20]      ││
│ Service Detection Patterns      │ │ Total:        [0.95]      ││
│ ┌─────────────────────────────┐ │ └────────────────────────────┘│
│ │ Service ID Patterns:        │ │                               │
│ │ [textarea with regex]       │ │                               │
│ │                             │ │                               │
│ │ Package Patterns:           │ │                               │
│ │ [textarea with patterns]    │ │                               │
│ │                             │ │                               │
│ │ Presets: [Combined ▼]       │ │                               │
│ └─────────────────────────────┘ │                               │
├─────────────────────────────────────────────────────────────────┤
│                    [Reset Defaults]  [Save Settings]            │
└─────────────────────────────────────────────────────────────────┘
```

### Partition Graph Controls

| Control | Icon | Function |
|---------|------|----------|
| Zoom In | 🔍+ | Increase zoom level |
| Zoom Out | 🔍- | Decrease zoom level |
| Reset View | ↺ | Reset to initial view |
| Fit to View | ⛶ | Fit all nodes in viewport |
| Labels | Aa | Toggle node labels |
| Connections | ⟷ | Toggle cross-partition edges |
| Play/Pause | ▶/⏸ | Resume/pause simulation |

### Label Visibility

All node labels are now visible at maximum zoom with:
- Text-shadow outline for readability
- Primary text color (`--text-primary`)
- Font weight 500 for clarity

### Line Navigation

Smooth scrolling to target lines in code preview using:
```javascript
targetElement.scrollIntoView({
    behavior: 'smooth',
    block: 'center'
});
```

---

## Performance

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Partition analysis (17K classes) | 0.28s | With partition-based layout |
| MDS layout (500 nodes) | 1-2s | Eigendecomposition |
| Service detection (17K files) | 2-3s | Multi-module scan |
| Risk analysis (26 services) | <0.5s | Timeline + drift |
| D3.js rendering (17K nodes) | 3-5s | Initial render |
| Graph stabilization | 2-3s | With precomputed positions |

---

## API Reference

### Partitioner

```
POST /api/ast/partition
Body: {
    "path": "/path/to/project",
    "config": {
        "applications": [...],
        "shared_patterns": {...},
        "propagation_iterations": 3,
        "confidence_threshold": 0.5
    }
}

Response: {
    "project_path": "...",
    "total_classes": 17118,
    "partitioned_classes": 15234,
    "coverage": 89.0,
    "cross_partition_edges": 1247,
    "coupling_density": 0.082,
    "assignments": {...},
    "nodes": [...],
    "edges": [...]
}
```

### Audit

```
GET /api/audit/detect-services?project_path=...
GET /api/audit/risk?project_path=...
GET /api/audit/timeline?project_path=...
GET /api/audit/drift?project_path=...
POST /api/audit/config
```

### Code Tracker

```
GET /api/ast/tracker?project_path=...
GET /api/ast/file-view?file_path=...&start_line=...&end_line=...
```

---

## File Structure

```
ragix_audit/
├── __init__.py              # Public API
├── partitioner.py           # Codebase partitioner (NEW)
│   ├── CodebasePartitioner  # Main class
│   ├── compute_layout()     # MDS/partition-based layout
│   ├── _mds_layout()        # Classical MDS
│   └── _partition_based_layout()  # O(n) circular layout
│
├── component_mapper.py      # SK/SC/SG + spre## patterns
├── service_detector.py      # Multi-pattern service detection
├── timeline.py              # Service lifecycle analysis
├── risk.py                  # Risk scoring
├── drift.py                 # Spec-code drift detection
│
├── statistics.py            # Distribution statistics (planned)
├── entropy.py               # Entropy metrics (planned)
├── coupling.py              # Martin's metrics (planned)
├── dead_code.py             # Reachability analysis (planned)
└── mco.py                   # MCO estimation (planned)

ragix_web/
├── server.py                # FastAPI server
├── routers/
│   ├── audit.py             # /api/audit/* endpoints
│   │   └── _get_source_path() # Multi-module detection
│   ├── rag_project.py       # /api/rag-project/*
│   └── ...
└── static/
    ├── index.html           # Main UI
    │   ├── Partitioner section
    │   ├── Audit Settings modal
    │   └── Code Tracker section
    └── style.css            # 8000+ lines of CSS
```

---

## Migration Notes

### From v0.33 → v0.55

1. **New dependencies**: Ensure `numpy` is installed for MDS layout
2. **Service patterns**: Default patterns now include enterprise conventions
3. **Multi-module projects**: No configuration needed, auto-detected
4. **Risk analysis**: Services now use uppercase IDs for consistency

### Configuration

```yaml
# ragix.yaml additions for v0.55
audit:
  thresholds:
    drift_days: 90
    new_component_days: 180
    legacy_years: 3
  risk_weights:
    age: 0.20
    volatility: 0.20
    complexity: 0.20
    doc_coverage: 0.15
    test_coverage: 0.20
  service_patterns:
    component_patterns:
      - "SK\\d{2}"
      - "SC\\d{2}"
      - "SG\\d{2}"
      - "spre\\d{2}(?:ws)?"
      - "sprebpm(?:ws)?"
      - "spremail"
    package_patterns:
      - "fr\\.iowizmi\\.iok\\.sk"
      - "com\\..*\\.ws\\.spre"
```

---

## Known Limitations

1. **MDS layout**: Floyd-Warshall is O(n³), limited to <500 nodes
2. **Service patterns**: Backend API for custom patterns not yet implemented
3. **Risk scoring**: 20 detected services without file associations show no risk
4. **Export**: XLSX export requires `openpyxl` (optional dependency)

---

## Roadmap to v0.60

| Feature | Priority | Status |
|---------|----------|--------|
| Backend API for service patterns | High | Planned |
| Services table sorting/filtering | Medium | Planned |
| Risk badge colors | Medium | Planned |
| RAG partition metadata | High | Planned |
| Partition filter in search | High | Planned |

---

## Acknowledgments

RAGIX v0.55.0 represents extensive development at the **Adservio Innovation Lab**, focused on enterprise-grade code audit capabilities for legacy Java modernization.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr

**Target Applications:**
- Legacy Java codebases (10-20+ years)
- Multi-module Maven projects
- Enterprise service-oriented architectures
- Monolith-to-microservices modernization planning
