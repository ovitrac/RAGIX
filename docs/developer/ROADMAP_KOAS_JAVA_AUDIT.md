# ROADMAP — KOAS Java Audit Extensions

**Version:** 1.0
**Date:** 2026-02-09
**Status:** PROPOSED
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## 1. Context & Motivation

The ACME-ERP audit (Dec 2025 – Jan 2026) exposed specific gaps in KOAS when applied to a production Java/Spring Boot codebase (24 Maven modules, 60K LOC, 4M MSG-HUB messages/day). Three categories of issues were identified:

**Parser failures (root cause of most anomalies):**
- CC=1.0 for all 2 704 methods → MI=100 → Grade A (all artificial)
- 0 services detected (Spring annotations invisible)
- 113 JAXB parsing errors (XSD-generated types)
- Dead code 98.2% (Spring DI, JMS listeners not resolved)

**Missing kernels:**
- No Maven `pom.xml` dependency extraction
- No OWASP/CVE scanning for Maven dependencies
- No inter-module Maven graph (parent/child/transitive)

**Uncalibrated thresholds:**
- 0 recommendations generated (thresholds tuned for "normal" CC, not CC=1.0)

This roadmap defines 6 new kernels + 2 kernel fixes, all following the standard `Kernel` base class contract and auto-discoverable by `KernelRegistry`.

---

## 2. Auto-Discovery Contract

Every new kernel MUST satisfy these conditions to be auto-discovered:

```python
# File: ragix_kernels/audit/<kernel_name>.py

from ragix_kernels.base import Kernel, KernelInput

class MyKernel(Kernel):
    name = "my_kernel"              # Unique identifier
    version = "1.0.0"               # Semantic versioning
    category = "audit"              # "audit" for audit pipeline
    stage = 1                       # 1=collection, 2=analysis, 3=reporting
    description = "Short description"
    requires = ["ast_scan"]         # Kernel dependencies (topologically sorted)
    provides = ["my_output"]        # Capabilities provided

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Pure computation — NO LLM, deterministic
        ...

    def summarize(self, data: Dict[str, Any]) -> str:
        # <500 chars for LLM consumption
        ...
```

**Discovery mechanism** (`ragix_kernels/registry.py`):
- `pkgutil.walk_packages` scans all modules under `ragix_kernels/`
- Finds all classes where `issubclass(cls, Kernel)` and `cls.name != "base"`
- Registers by `name`, indexes by `category` and `stage`
- Topological sort via `requires` for dependency resolution

**No changes needed to `registry.py` or `__init__.py`.** Just place the file in `ragix_kernels/audit/` and it's discovered.

---

## 3. Priority 1 — Fix Existing Kernels

### 3.1 FIX: `ast_scan` — Java CC Calculation

**File:** `ragix_kernels/audit/ast_scan.py`
**Problem:** CC=1.0 for all methods. The `javalang` parser extracts class/method structure but the CC counter doesn't walk statement-level nodes.
**Root cause to investigate:**

1. Does `javalang.parse.parse()` succeed on ACME-ERP files? (113 errors suggest partial failures)
2. Are `IfStatement`, `ForStatement`, `WhileStatement`, `SwitchStatement`, `CatchClause` nodes counted?
3. Are boolean operators (`&&`, `||`) in conditions counted?

**Fix approach:**

```python
# In ast_scan.py — add CC walker for Java methods
BRANCH_NODES = {
    'IfStatement', 'ForStatement', 'WhileStatement',
    'DoStatement', 'SwitchStatementCase', 'CatchClause',
    'ConditionalExpression',  # ternary
}

def _compute_cc_java(method_node) -> int:
    """Walk javalang AST to count branches."""
    cc = 1  # base path
    for _, node in method_node:
        if type(node).__name__ in BRANCH_NODES:
            cc += 1
        # Count && and || in conditions
        if type(node).__name__ == 'BinaryOperation':
            if node.operator in ('&&', '||'):
                cc += 1
    return cc
```

**Validation:** Re-run on `EchMsg-HubJmsQddcListener.java` (170 LOC, known to contain try/catch + conditional routing). Expected CC > 1.

**Deliverable:** Patch to `ast_scan.py`, regression test with 3 ACME-ERP Java files.

### 3.2 FIX: `ast_scan` — Spring Annotation Detection

**File:** `ragix_kernels/audit/ast_scan.py`
**Problem:** `services` kernel found 0 services because `ast_scan` doesn't extract annotation metadata.

**Fix approach:**

```python
# Extract annotations from class/method declarations
SPRING_ANNOTATIONS = {
    'Service', 'Component', 'Repository', 'Controller',
    'RestController', 'Configuration', 'Bean',
    'JmsListener', 'EventListener', 'Scheduled',
    'Autowired', 'Inject', 'Value',
}

def _extract_annotations(node) -> List[Dict]:
    """Extract Spring-relevant annotations from javalang node."""
    annotations = []
    for ann in getattr(node, 'annotations', []):
        ann_name = ann.name if hasattr(ann, 'name') else str(ann)
        if ann_name in SPRING_ANNOTATIONS:
            annotations.append({
                "name": ann_name,
                "element": getattr(ann, 'element', None),
            })
    return annotations
```

**Impact on downstream kernels:**
- `services`: Will detect `@Service`, `@Component`, `@RestController` → non-zero service count
- `dead_code`: `@JmsListener`, `@EventListener`, `@Scheduled` become entry points → realistic dead code %
- `hotspots`: CC values enable real hotspot detection

**Deliverable:** Patch to `ast_scan.py`, new `annotations` field in symbol output.

### 3.3 FIX: `ast_scan` — JAXB Error Tolerance

**File:** `ragix_kernels/audit/ast_scan.py`
**Problem:** 113 JAXB-generated files cause `javalang.parser.JavaSyntaxError`. These files use generated patterns (`@XmlType`, `@XmlAccessorType`) that sometimes confuse the parser.

**Fix approach:**
- Catch `JavaSyntaxError` per-file (already done) but report partial results
- Extract at minimum: class name, package, LOC, annotations from the portions that do parse
- Add a `parse_errors` counter in the summary for transparency

**Deliverable:** Improved error tolerance, `parse_errors` field in output.

---

## 4. Priority 2 — New Maven Kernels

### 4.1 NEW: `maven_deps` — Maven Dependency Extraction

**File:** `ragix_kernels/audit/maven_deps.py`
**Stage:** 1 (collection)
**Requires:** [] (no kernel dependencies — reads pom.xml directly from source)
**Provides:** `["maven_dependencies", "maven_versions"]`

**Purpose:** Parse all `pom.xml` files in the project, extract dependencies, versions, scopes, and parent relationships.

**Design:**

```python
class MavenDepsKernel(Kernel):
    name = "maven_deps"
    version = "1.0.0"
    category = "audit"
    stage = 1
    description = "Extract Maven dependencies from pom.xml files"
    requires = []
    provides = ["maven_dependencies", "maven_versions"]
```

**Implementation notes:**
- Use `xml.etree.ElementTree` (stdlib only — no external dependency)
- Handle Maven namespace: `{http://maven.apache.org/POM/4.0.0}`
- Resolve `${property}` references from `<properties>` block
- Walk parent chain: `<parent><groupId>...<artifactId>...<version>...`
- Collect: `groupId`, `artifactId`, `version`, `scope`, `type`, `exclusions`

**Output schema:**

```json
{
  "modules": [
    {
      "module_path": "iow-ech-msg_hub-master",
      "groupId": "com.iowizmi.ech",
      "artifactId": "iow-ech-msg_hub",
      "version": "1.2.0",
      "packaging": "jar",
      "parent": {"groupId": "...", "artifactId": "..."},
      "dependencies": [
        {"groupId": "org.springframework.boot", "artifactId": "spring-boot-starter-activemq", "version": "2.7.x", "scope": "compile"}
      ],
      "properties": {"java.version": "11", "spring-boot.version": "2.7.18"}
    }
  ],
  "all_dependencies": [
    {"groupId": "org.springframework.boot", "artifactId": "spring-boot-starter-activemq", "version": "2.7.18", "used_by": ["iow-ech-msg_hub", "iow-iok-sk01"], "scope": "compile"}
  ],
  "statistics": {
    "pom_files_parsed": 76,
    "modules_found": 24,
    "distinct_dependencies": 87,
    "parse_errors": 0
  }
}
```

### 4.2 NEW: `maven_graph` — Maven Module Graph

**File:** `ragix_kernels/audit/maven_graph.py`
**Stage:** 2 (analysis)
**Requires:** `["maven_deps"]`
**Provides:** `["maven_module_graph", "maven_centrality"]`

**Purpose:** Build and analyze the inter-module dependency graph from Maven parent/child and dependency relationships.

**Design:**

```python
class MavenGraphKernel(Kernel):
    name = "maven_graph"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Analyze Maven module dependency graph"
    requires = ["maven_deps"]
    provides = ["maven_module_graph", "maven_centrality"]
```

**Computed metrics (pure Python, no external deps):**
- Adjacency list (module → direct dependencies)
- Transitive closure (module → all reachable modules)
- In-degree / out-degree per module
- Betweenness centrality (which modules are hubs)
- Cycle detection (should be 0 for well-formed Maven, but verify)
- Root modules (no parent) and leaf modules (no dependents)
- Critical path: longest dependency chain

**Output schema:**

```json
{
  "graph": {
    "nodes": ["iow-ech-msg_hub", "iog-support-commons", ...],
    "edges": [
      {"from": "iow-ech-msg_hub", "to": "iog-support-commons", "type": "compile"},
      {"from": "iow-ech-msg_hub", "to": "iow-iog-models", "type": "compile"}
    ]
  },
  "centrality": {
    "iog-support-commons": {"betweenness": 0.85, "in_degree": 18, "out_degree": 3},
    ...
  },
  "critical_path": ["iow-ech-msg_hub", "iog-support-commons", "iog-support-platform"],
  "cycles": [],
  "roots": ["parent-acq"],
  "leaves": ["iow-iok-sk05", "iow-iok-sk06", ...]
}
```

### 4.3 NEW: `maven_cve` — Dependency Vulnerability Scan

**File:** `ragix_kernels/audit/maven_cve.py`
**Stage:** 2 (analysis)
**Requires:** `["maven_deps"]`
**Provides:** `["maven_vulnerabilities", "cve_report"]`

**Purpose:** Check extracted Maven dependencies against known CVE databases. Deterministic: compares version strings against a local vulnerability catalog.

**Design:**

```python
class MavenCveKernel(Kernel):
    name = "maven_cve"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Scan Maven dependencies for known vulnerabilities"
    requires = ["maven_deps"]
    provides = ["maven_vulnerabilities", "cve_report"]
```

**Implementation approach — two tiers:**

**Tier 1 (deterministic, no network):** Built-in catalog of known-vulnerable versions for common Java libraries. Covers the most critical advisories:
- Spring Boot / Spring Framework (CVE-2022-22965 Spring4Shell, etc.)
- Log4j (CVE-2021-44228 Log4Shell)
- Jackson, Apache Commons, JAXB, ActiveMQ
- Maintained as a JSON file: `ragix_kernels/audit/data/java_cve_catalog.json`

**Tier 2 (optional, network):** If `workspace/data/owasp_cache.json` exists (pre-downloaded from NVD or OWASP dependency-check), use it for comprehensive scanning. The kernel itself does NOT make network calls (sovereignty: local-only).

**Output schema:**

```json
{
  "vulnerabilities": [
    {
      "dependency": "org.springframework.boot:spring-boot-starter-activemq:2.7.18",
      "cve_id": "CVE-2023-XXXXX",
      "severity": "HIGH",
      "cvss": 8.1,
      "description": "...",
      "fixed_in": "2.7.19",
      "modules_affected": ["iow-ech-msg_hub", "iow-iok-sk01"]
    }
  ],
  "statistics": {
    "dependencies_scanned": 87,
    "vulnerabilities_found": 12,
    "by_severity": {"CRITICAL": 1, "HIGH": 4, "MEDIUM": 5, "LOW": 2},
    "catalog_version": "2026-02-01",
    "catalog_tier": "builtin"
  }
}
```

**CVE catalog maintenance:** The JSON catalog is versioned and updated manually (or via a helper script that downloads from NVD/OSV). The kernel itself is pure computation.

---

## 5. Priority 3 — New Analysis Kernel

### 5.1 NEW: `spring_wiring` — Spring DI Resolution

**File:** `ragix_kernels/audit/spring_wiring.py`
**Stage:** 2 (analysis)
**Requires:** `["ast_scan"]`
**Provides:** `["spring_beans", "spring_entry_points"]`

**Purpose:** Resolve Spring dependency injection to produce a realistic reachability graph. Addresses the dead code false positive problem (98.2% → ~5-15%).

**Design:**

```python
class SpringWiringKernel(Kernel):
    name = "spring_wiring"
    version = "1.0.0"
    category = "audit"
    stage = 2
    description = "Resolve Spring DI wiring and implicit entry points"
    requires = ["ast_scan"]
    provides = ["spring_beans", "spring_entry_points"]
```

**Logic:**

1. From `ast_scan` annotations, collect all classes annotated with:
   - `@Service`, `@Component`, `@Repository`, `@Controller`, `@RestController` → Spring beans
   - `@Configuration` + `@Bean` methods → factory-defined beans
2. From `@Autowired` / `@Inject` fields, build a wiring graph (bean → bean dependencies)
3. Identify implicit entry points (not in `main()` but reachable via framework):
   - `@JmsListener` methods → JMS-triggered entry points
   - `@EventListener` methods → event-triggered
   - `@Scheduled` methods → timer-triggered
   - `@RestController` / `@RequestMapping` methods → HTTP entry points
4. Output: augmented entry point list for `dead_code` kernel

**Impact:** The `dead_code` kernel already accepts entry points. This kernel provides additional entry points, reducing false positives from ~1 416 to a realistic count.

**Output schema:**

```json
{
  "beans": [
    {"class": "com.iowizmi.ech.msg_hub.infra.EchMsg-HubJmsQddcListener", "type": "component", "annotation": "JmsListener", "entry_point": true}
  ],
  "wiring": [
    {"from": "EchMsg-HubJmsQddcListener", "to": "ConcentrateurEchRepo", "type": "autowired"}
  ],
  "entry_points": {
    "jms_listeners": 7,
    "event_listeners": 0,
    "scheduled": 0,
    "rest_endpoints": 2,
    "main_methods": 3,
    "total": 12
  },
  "reachability_delta": {
    "before": 26,
    "after": 380,
    "improvement_ratio": 14.6
  }
}
```

### 5.2 NEW: `section_maven` — Maven Report Section

**File:** `ragix_kernels/audit/section_maven.py`
**Stage:** 3 (reporting)
**Requires:** `["maven_deps", "maven_graph", "maven_cve"]`
**Provides:** `["section_maven"]`

**Purpose:** Generate the "Dependencies & Supply Chain" section of the audit report.

**Output:** Markdown section with:
- Dependency inventory table (groupId, artifactId, version, scope, modules)
- Module graph visualization (Mermaid)
- CVE findings table (severity, affected modules, remediation)
- Obsolescence assessment (versions vs latest stable)

### 5.3 NEW: `section_spring` — Spring Architecture Section

**File:** `ragix_kernels/audit/section_spring.py`
**Stage:** 3 (reporting)
**Requires:** `["spring_wiring", "dead_code"]`
**Provides:** `["section_spring"]`

**Purpose:** Generate the "Spring Architecture" section: bean inventory, wiring graph, entry point catalog, corrected dead code analysis.

---

## 6. File Layout

```
ragix_kernels/audit/
├── __init__.py                    # (no changes — auto-discovery)
├── ast_scan.py                    # FIX: CC walker, annotations, JAXB tolerance
├── metrics.py                     # (no changes — consumes fixed ast_scan)
├── services.py                    # (no changes — consumes annotations from ast_scan)
├── dead_code.py                   # (no changes — consumes spring_wiring entry points)
├── maven_deps.py                  # NEW: pom.xml extraction (Stage 1)
├── maven_graph.py                 # NEW: module dependency graph (Stage 2)
├── maven_cve.py                   # NEW: CVE scanning (Stage 2)
├── spring_wiring.py               # NEW: Spring DI resolution (Stage 2)
├── section_maven.py               # NEW: Maven report section (Stage 3)
├── section_spring.py              # NEW: Spring report section (Stage 3)
├── data/
│   └── java_cve_catalog.json      # Built-in CVE catalog for common Java libs
├── ...                            # (existing kernels unchanged)
```

All 6 new files are auto-discovered. No changes to `__init__.py`, `registry.py`, or `base.py`.

---

## 7. Dependency Graph

```
                         Stage 1                    Stage 2                    Stage 3
                    ┌──────────────┐           ┌──────────────┐          ┌──────────────┐
                    │  ast_scan    │──────────▶│ spring_wiring│─────────▶│section_spring│
                    │  (FIX: CC,  │     ┌────▶│              │    ┌────▶│              │
                    │   annot.)   │─────┤     └──────────────┘    │     └──────────────┘
                    └──────────────┘     │                         │
                          │             │     ┌──────────────┐    │
                          │             └────▶│  dead_code   │────┘
                          │                   │  (existing)  │
                          ▼                   └──────────────┘
                    ┌──────────────┐
                    │   metrics    │           ┌──────────────┐
                    │  (existing)  │           │  hotspots    │
                    └──────────────┘           │  (existing)  │
                                              └──────────────┘
                    ┌──────────────┐           ┌──────────────┐
                    │  maven_deps  │──────────▶│ maven_graph  │─────────▶┌──────────────┐
                    │  (NEW)       │     ┌────▶│  (NEW)       │    ┌────▶│section_maven │
                    └──────────────┘     │     └──────────────┘    │     │  (NEW)       │
                          │             │                         │     └──────────────┘
                          │             │     ┌──────────────┐    │
                          └─────────────┴────▶│  maven_cve   │────┘
                                              │  (NEW)       │
                                              └──────────────┘
```

Topological sort resolves to:
1. `ast_scan` (fix), `maven_deps` (new) — Stage 1, independent
2. `metrics`, `services`, `spring_wiring`, `maven_graph`, `maven_cve` — Stage 2
3. `dead_code`, `hotspots`, `coupling`, `entropy`, `risk`, `drift` — Stage 2 (existing)
4. `section_maven`, `section_spring`, `section_recommendations` — Stage 3

---

## 8. Implementation Plan

### Phase A — Fixes (2 days)

| Task | File | Effort | Validation |
|------|------|--------|------------|
| A1. CC walker for Java AST | ast_scan.py | 4h | CC > 1 on ACME-ERP sample files |
| A2. Annotation extraction | ast_scan.py | 2h | `@Service` detected in SK04Controller |
| A3. JAXB error tolerance | ast_scan.py | 2h | 113 errors → partial results, not failures |
| A4. Re-run full pipeline | — | 1h | MI ≠ 100, Grade ≠ A, recommendations > 0 |

**Acceptance criteria:** On ACME-ERP snapshot, `metrics` kernel produces CC distribution with mean > 1.0, `services` kernel detects > 0 services, `section_recommendations` produces > 0 recommendations.

### Phase B — Maven Kernels (3 days)

| Task | File | Effort | Validation |
|------|------|--------|------------|
| B1. `maven_deps` kernel | maven_deps.py | 4h | 76 pom.xml parsed, 24 modules, deps extracted |
| B2. `maven_graph` kernel | maven_graph.py | 4h | Graph with centrality, iog-support-commons = top hub |
| B3. CVE catalog (initial) | data/java_cve_catalog.json | 4h | Cover Spring Boot 2.x, Log4j, Jackson, ActiveMQ |
| B4. `maven_cve` kernel | maven_cve.py | 4h | Scan against catalog, ≥0 findings |
| B5. `section_maven` kernel | section_maven.py | 4h | Markdown section generated |

**Acceptance criteria:** `maven_deps` parses all 76 ACME-ERP pom.xml files. `maven_graph` confirms `iog-support-commons` centrality. `maven_cve` runs without error on extracted dependencies.

### Phase C — Spring Wiring (2 days)

| Task | File | Effort | Validation |
|------|------|--------|------------|
| C1. `spring_wiring` kernel | spring_wiring.py | 6h | 7 JMS listeners detected as entry points |
| C2. Dead code integration | (config change) | 2h | Dead code drops from 98% to < 20% |
| C3. `section_spring` kernel | section_spring.py | 4h | Report section with bean catalog |

**Acceptance criteria:** `spring_wiring` identifies ≥ 7 JMS listener entry points in ECH-MSG-HUB. `dead_code` re-run with augmented entry points produces < 20% dead code ratio.

### Phase D — Integration & Report (1 day)

| Task | Effort | Validation |
|------|--------|------------|
| D1. Update `report_assemble` to include new sections | 2h | New sections appear in report |
| D2. Update manifest.yaml schema for new kernels | 1h | New kernels configurable |
| D3. MCP tool registration for new kernels | 2h | Callable from Claude Desktop |
| D4. Full pipeline re-run on ACME-ERP | 2h | Complete report with all new data |

---

## 9. Testing Strategy

### Unit Tests (per kernel)

Each new kernel gets a test file in `ragix_kernels/audit/tests/`:

```
ragix_kernels/audit/tests/
├── test_maven_deps.py          # Parse sample pom.xml, verify extraction
├── test_maven_graph.py         # Graph construction, centrality, cycle detection
├── test_maven_cve.py           # Catalog lookup, severity classification
├── test_spring_wiring.py       # Annotation-based entry point discovery
├── fixtures/
│   ├── sample_pom.xml          # Minimal valid pom.xml
│   ├── sample_pom_parent.xml   # Multi-module parent pom
│   ├── sample_pom_properties.xml  # Property substitution
│   └── sample_annotations.json    # ast_scan output with Spring annotations
```

### Integration Test

Re-run the full KOAS pipeline on ACME-ERP and compare:

| Metric | Before (v0.62) | After | Pass condition |
|--------|:--------------:|:-----:|----------------|
| CC mean | 1.0 | > 1.0 | Any increase |
| CC max | 1.0 | > 5 | Realistic Java CC |
| MI | 100 | < 90 | Non-artificial |
| Services detected | 0 | > 0 | Any detection |
| Dead code % | 98.2% | < 30% | Significant drop |
| Recommendations | 0 | > 0 | Any generation |
| Maven modules | N/A | 24 | Match known count |
| CVE findings | N/A | ≥ 0 | Runs without error |
| Spring entry points | 5 | > 10 | JMS listeners counted |

---

## 10. External Dependencies

| Dependency | Status | Required for | Fallback |
|------------|--------|-------------|----------|
| `javalang` | Already installed | ast_scan CC fix | None — required |
| `xml.etree.ElementTree` | stdlib | maven_deps | None — always available |
| `json` | stdlib | maven_cve catalog | None — always available |
| `httpx` | Optional | CVE catalog download (helper script only) | Manual download |
| Network access | NOT required by kernels | — | Sovereignty: local-only |

All kernels remain **local-only, deterministic, no LLM, no network**. The CVE catalog is a static JSON file updated out-of-band.

---

## 11. Risk Assessment

| Risk | Probability | Mitigation |
|------|:-----------:|------------|
| `javalang` cannot parse CC for ACME-ERP files | Medium | Fallback: regex-based CC counter on raw Java source |
| JAXB files still unparseable | Low | Already handled: partial results + `parse_errors` count |
| Maven property resolution incomplete (`${project.version}`) | Medium | Conservative: report unresolved as-is with warning |
| CVE catalog incomplete | Certain | Tier 1 covers critical CVEs only; Tier 2 optional |
| `dead_code` still high after Spring wiring | Low | Report with caveat; framework-managed beans noted |

---

## 12. Relation to ACME-ERP Baseline

This roadmap addresses items from `ANALYSE_ECARTS.md`:

| Écart | Kernel | Phase |
|-------|--------|:-----:|
| L1 — CC=1.0 | ast_scan fix | A |
| L2 — 113 JAXB errors | ast_scan fix | A |
| L3 — 0 services | ast_scan fix (annotations) | A |
| L4 — 0 recommendations | Cascading fix from CC | A |
| L5 — Dead code 98% | spring_wiring + dead_code | C |
| L6 — No OWASP | maven_cve | B |
| L8 — No inter-Maven | maven_graph | B |
| L9 — No pom.xml scan | maven_deps | B |

Items NOT addressed here (separate roadmaps):
- L7 — Document review → KOAS Reviewer pipeline (already exists)
- L10 — Semantic drift → Future R&D
- D1–D9 — Client data → External dependency

---

*Document de travail — RAGIX / KOAS Development Roadmap | Adservio Innovation Lab*
