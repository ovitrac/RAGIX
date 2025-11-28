# RAGIX v0.10 Planning: AST Analysis & Dependency Intelligence

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
**Status:** PLANNING (Priority for next session)
**Previous:** v0.9.0 WASP Tools & Browser Runtime

---

## Executive Summary

v0.10 focuses on **AST-level code intelligence** with multi-language support:

| Feature | Priority | Languages |
|---------|----------|-----------|
| AST Parsing | HIGH | Python, Java, JavaScript |
| Dependency Graph | HIGH | Module/Class/Method level |
| Maven Integration | MEDIUM | Java projects |
| Sonar Integration | MEDIUM | Quality metrics |
| AST Visualization | HIGH | Interactive graphs |
| CLI AST Search | HIGH | Pattern matching |

---

## Task Breakdown

### Task 5.1: Multi-Language AST Parser

**Goal:** Unified AST parsing interface for Python, Java, JavaScript.

**Components:**

1. **AST Backend Protocol** (`ragix_core/ast_base.py`)
   ```python
   class ASTNode:
       type: NodeType  # class, method, function, import, etc.
       name: str
       file: Path
       line: int
       children: List[ASTNode]
       metadata: Dict[str, Any]
   
   class ASTBackend(Protocol):
       def parse_file(self, path: Path) -> ASTNode: ...
       def parse_string(self, content: str, lang: str) -> ASTNode: ...
       def get_symbols(self, node: ASTNode) -> List[Symbol]: ...
   ```

2. **Python AST Backend** (`ragix_core/ast_python.py`)
   - Uses stdlib `ast` module
   - Extract: classes, functions, imports, decorators
   - Docstring extraction

3. **Java AST Backend** (`ragix_core/ast_java.py`)
   - Uses `javalang` library
   - Extract: classes, interfaces, methods, fields, imports
   - Annotation extraction
   - Package structure

4. **JavaScript AST Backend** (`ragix_core/ast_javascript.py`)
   - Uses `esprima` or tree-sitter
   - Extract: classes, functions, imports/exports, modules
   - ES6+ support

**Dependencies:**
```toml
[project.optional-dependencies]
ast = ["javalang>=0.13.0", "esprima>=4.0.0"]
```

---

### Task 5.2: Dependency Graph Builder

**Goal:** Build dependency graphs at multiple granularity levels.

**Components:**

1. **Dependency Types** (`ragix_core/dependencies.py`)
   ```python
   class DependencyType(Enum):
       IMPORT = "import"           # Module imports
       INHERITANCE = "inheritance" # Class extends/implements
       COMPOSITION = "composition" # Field types
       CALL = "call"               # Method/function calls
       ANNOTATION = "annotation"   # Decorators/annotations
   
   class Dependency:
       source: Symbol
       target: Symbol
       dep_type: DependencyType
       file: Path
       line: int
   ```

2. **Graph Builder** (`ragix_core/dep_graph.py`)
   ```python
   class DependencyGraph:
       def add_file(self, path: Path): ...
       def add_directory(self, path: Path, patterns: List[str]): ...
       def get_dependencies(self, symbol: str) -> List[Dependency]: ...
       def get_dependents(self, symbol: str) -> List[Dependency]: ...
       def get_cycles(self) -> List[List[Symbol]]: ...
       def export_dot(self) -> str: ...
       def export_json(self) -> Dict: ...
   ```

3. **Analysis Functions**
   - Cycle detection
   - Unused imports
   - Orphan classes
   - Coupling metrics (afferent/efferent)
   - Abstractness/instability

---

### Task 5.3: Maven Integration

**Goal:** Extract dependencies and structure from Maven projects.

**Components:**

1. **POM Parser** (`ragix_core/maven.py`)
   ```python
   class MavenProject:
       group_id: str
       artifact_id: str
       version: str
       dependencies: List[MavenDependency]
       modules: List[str]  # Multi-module
       parent: Optional[MavenProject]
   
   def parse_pom(path: Path) -> MavenProject: ...
   def resolve_dependencies(project: MavenProject) -> DependencyTree: ...
   def find_effective_pom(path: Path) -> MavenProject: ...
   ```

2. **Dependency Resolution**
   - Parse `pom.xml` hierarchy
   - Resolve transitive dependencies
   - Detect version conflicts
   - Scope analysis (compile, test, provided)

3. **Integration with AST**
   - Map Maven modules to source directories
   - Link external dependencies to imports

---

### Task 5.4: Sonar Integration

**Goal:** Import and correlate Sonar quality metrics.

**Components:**

1. **Sonar Client** (`ragix_core/sonar.py`)
   ```python
   class SonarClient:
       def __init__(self, url: str, token: str): ...
       def get_project(self, key: str) -> SonarProject: ...
       def get_issues(self, key: str, types: List[str]) -> List[SonarIssue]: ...
       def get_metrics(self, key: str) -> Dict[str, float]: ...
       def get_hotspots(self, key: str) -> List[SecurityHotspot]: ...
   ```

2. **Metrics Mapping**
   - Code smells → AST nodes
   - Bugs → Method/class correlation
   - Vulnerabilities → Security analysis
   - Coverage → Test correlation

3. **Report Generation**
   - Quality gate status
   - Technical debt estimation
   - Priority remediation list

---

### Task 5.5: AST Visualization

**Goal:** Interactive visualization of code structure and dependencies.

**Components:**

1. **Graph Renderers** (`ragix_core/ast_viz.py`)
   ```python
   def render_dot(graph: DependencyGraph, options: RenderOptions) -> str: ...
   def render_mermaid(graph: DependencyGraph, options: RenderOptions) -> str: ...
   def render_d3_json(graph: DependencyGraph) -> Dict: ...
   ```

2. **Web Visualization** (`ragix_web/static/js/ast_viz.js`)
   - D3.js force-directed graph
   - Zoom/pan/filter
   - Click-to-navigate
   - Dependency highlighting

3. **CLI Visualization**
   - ASCII tree output
   - Terminal colors for types
   - Collapsible hierarchy

---

### Task 5.6: CLI AST Search

**Goal:** Pattern-based search at AST level.

**Components:**

1. **AST Query Language** (`ragix_core/ast_query.py`)
   ```python
   # Query examples:
   # "class:*Controller"           - All controller classes
   # "method:get*"                 - All getter methods
   # "import:java.util.*"          - All java.util imports
   # "calls:database.query"        - All calls to database.query
   # "extends:BaseService"         - All classes extending BaseService
   
   def parse_query(query: str) -> ASTQuery: ...
   def execute_query(query: ASTQuery, graph: DependencyGraph) -> List[Match]: ...
   ```

2. **CLI Commands** (`ragix_unix/ast_cli.py`)
   ```bash
   ragix-ast parse <file>              # Parse single file
   ragix-ast scan <dir> --lang py,java # Scan directory
   ragix-ast deps <symbol>             # Show dependencies
   ragix-ast search "class:*Service"   # AST pattern search
   ragix-ast graph <dir> --format dot  # Generate graph
   ragix-ast cycles <dir>              # Detect cycles
   ragix-ast metrics <dir>             # Coupling metrics
   ```

3. **WASP Integration**
   - `ast_parse` tool
   - `ast_search` tool
   - `ast_deps` tool

---

## File Structure (v0.10)

```
ragix_core/
├── ast_base.py           # Base AST types and protocols
├── ast_python.py         # Python AST backend
├── ast_java.py           # Java AST backend (javalang)
├── ast_javascript.py     # JavaScript AST backend
├── dependencies.py       # Dependency types and graph
├── dep_graph.py          # Dependency graph builder
├── maven.py              # Maven POM parsing
├── sonar.py              # Sonar API client
├── ast_viz.py            # Visualization renderers
└── ast_query.py          # AST query language

ragix_unix/
└── ast_cli.py            # AST CLI commands

ragix_web/static/js/
└── ast_viz.js            # D3.js visualization

wasp_tools/
├── ast_parse.py          # AST parsing tool
├── ast_search.py         # AST search tool
└── ast_deps.py           # Dependency tool

tests/
├── test_ast_python.py
├── test_ast_java.py
├── test_dep_graph.py
└── test_ast_query.py
```

---

## Dependencies

```toml
[project.optional-dependencies]
ast = [
    "javalang>=0.13.0",      # Java AST parsing
    "esprima>=4.0.0",        # JavaScript parsing (optional)
]
sonar = [
    "requests>=2.31.0",      # Already in base deps
]
viz = [
    "graphviz>=0.20",        # DOT rendering (optional)
]
```

---

## Implementation Order

### Phase 1: Core AST (Day 1 AM)
1. `ast_base.py` - Base types
2. `ast_python.py` - Python backend
3. `dependencies.py` - Dependency types
4. Basic tests

### Phase 2: Java Support (Day 1 PM)
5. `ast_java.py` - Java backend with javalang
6. `dep_graph.py` - Graph builder
7. Cycle detection

### Phase 3: CLI & Search (Day 2 AM)
8. `ast_query.py` - Query language
9. `ast_cli.py` - CLI commands
10. WASP tools integration

### Phase 4: Integration (Day 2 PM)
11. `maven.py` - Maven parsing
12. `sonar.py` - Sonar client
13. `ast_viz.py` - Visualization
14. Web integration

---

## Quick Reference: AST Node Types

### Python
- `module`, `class`, `function`, `async_function`
- `import`, `import_from`
- `assign`, `ann_assign` (type hints)
- `decorator`

### Java
- `class`, `interface`, `enum`, `annotation_type`
- `method`, `constructor`, `field`
- `import`, `package`
- `annotation`

### ragix-envJavaScript
- `module`, `class`, `function`, `arrow_function`
- `import`, `export`
- `method`, `property`

---

## Success Criteria

- [ ] Parse Python/Java/JS files to unified AST
- [ ] Build dependency graph with all dependency types
- [ ] Detect circular dependencies
- [ ] CLI search with pattern syntax
- [ ] Maven POM parsing (if available)
- [ ] Sonar integration (if available)
- [ ] DOT/Mermaid graph export
- [ ] Web visualization (D3.js)

---

## Notes for Tomorrow

1. Start with Python AST (stdlib, no deps)
2. Add Java with javalang (pip install javalang)
3. JavaScript optional (can defer if time-constrained)
4. Focus on practical CLI search first
5. Visualization can be incremental
6. Maven/Sonar are "nice to have" - implement if time permits
