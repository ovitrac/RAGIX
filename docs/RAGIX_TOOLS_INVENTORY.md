# RAGIX Tools Inventory

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Generated:** 2025-12-14
**Purpose:** Reference for KOAS kernel development

---

## Overview

This document inventories all reusable tools in `ragix_audit/` and `ragix_core/` that should be wrapped by KOAS kernels rather than reimplemented.

---

## ragix_audit/ — Audit Analysis Tools

### 1. StatisticsComputer (`statistics.py`)

**Purpose:** Computes comprehensive descriptive statistics from AST metrics.

| Aspect | Details |
|--------|---------|
| **Class** | `StatisticsComputer` |
| **Main Method** | `compute_from_metrics(file_metrics, component_mapping)` |
| **Input** | `List[FileMetrics]` + optional `Dict[str, str]` mapping |
| **Output** | `CodebaseStats` |
| **LOC** | 547 lines |

**Key Data Classes:**
- `DistributionStats` — Statistical profile (mean, std, quartiles, skewness, kurtosis)
- `ComponentStats` — Per-component statistics
- `CodebaseStats` — Global codebase statistics

**Reuse in Kernel:** `stats_summary.py` (Stage 2)

---

### 2. CouplingComputer (`coupling.py`)

**Purpose:** Computes Robert Martin's package coupling metrics (Ca, Ce, I, A, D).

| Aspect | Details |
|--------|---------|
| **Class** | `CouplingComputer` |
| **Main Method** | `compute_from_graph(dependencies, package_classes)` |
| **Input** | `Dict[str, Set[str]]` (package deps) + optional class info |
| **Output** | `CouplingAnalysis` |
| **LOC** | 505 lines |

**Key Data Classes:**
- `PackageCoupling` — Per-package metrics (Ca, Ce, I, A, D, zone)
- `SDPViolation` — Stable Dependencies Principle violation
- `CouplingAnalysis` — Complete analysis result
- `PropagationAnalysis` — Impact/propagation factors

**Key Methods:**
- `compute_from_graph()` — Main computation
- `compute_all_propagation_factors()` — Impact analysis

**Reuse in Kernel:** `coupling.py` (Stage 2)

---

### 3. DeadCodeDetector (`dead_code.py`)

**Purpose:** Detects unreachable code through entry point detection and BFS reachability.

| Aspect | Details |
|--------|---------|
| **Classes** | `DeadCodeDetector`, `EntryPointDetector` |
| **Main Method** | `analyze(file_contents, dependencies, class_to_file)` |
| **Input** | `Dict[str, str]` (files) + `Dict[str, Set[str]]` (deps) |
| **Output** | `DeadCodeAnalysis` |
| **LOC** | 427 lines |

**Key Data Classes:**
- `EntryPoint` — Detected entry point (main, controller, etc.)
- `DeadCodeCandidate` — Unreachable code candidate
- `DeadCodeAnalysis` — Complete analysis with reachability ratio

**Entry Point Types:**
- MAIN_METHOD, CONTROLLER, REST_ENDPOINT
- EVENT_HANDLER, SCHEDULED_TASK, MESSAGE_LISTENER
- TEST_CLASS, SPRING_BOOT

**Reuse in Kernel:** `dead_code.py` (Stage 2)

---

### 4. EntropyComputer (`entropy.py`)

**Purpose:** Information-theoretic distribution metrics (Shannon entropy, Gini, concentration).

| Aspect | Details |
|--------|---------|
| **Class** | `EntropyComputer` |
| **Main Method** | `compute_all(component_sizes, file_counts, complexities, degrees)` |
| **Input** | Multiple distribution dictionaries |
| **Output** | `EntropyMetrics` |
| **LOC** | 404 lines |

**Key Data Classes:**
- `EntropyMetrics` — Structural, complexity, coupling entropy
- `InequalityMetrics` — Gini, CR4, CR8, Herfindahl

**Helper Functions:**
- `shannon_entropy(distribution)` — Core entropy calculation
- `gini_coefficient(values)` — Inequality measure
- `concentration_ratio(values, top_k)` — CR-k metric

**Reuse in Kernel:** Could add `entropy.py` (Stage 2)

---

### 5. RiskScorer (`risk.py`)

**Purpose:** Multifactorial risk scoring combining volatility, impact, complexity, maturity.

| Aspect | Details |
|--------|---------|
| **Class** | `RiskScorer` |
| **Main Method** | `score_all(timelines, doc_gaps)` |
| **Input** | `Dict[str, ComponentTimeline]` + optional doc gaps |
| **Output** | `Dict[str, ServiceLifeRisk]` |
| **LOC** | 374 lines |

**Key Data Classes:**
- `RiskLevel` — LOW, MEDIUM, HIGH, CRITICAL
- `RiskFactors` — Volatility, impact, complexity, maturity, doc_gap
- `ServiceLifeRisk` — Complete risk assessment

**Default Weights:**
- Volatility: 20%
- Impact: 25%
- Complexity: 20%
- Maturity: 25%
- Documentation: 10%

**Reuse in Kernel:** Could add `risk.py` (Stage 2)

---

### 6. CodebasePartitioner (`partitioner.py`)

**Purpose:** Classifies classes into logical applications using fingerprint + graph propagation.

| Aspect | Details |
|--------|---------|
| **Class** | `CodebasePartitioner` |
| **Main Method** | `partition()` |
| **Input** | Classes + dependencies via `add_class()`, `add_dependency()` |
| **Output** | `PartitionResult` |
| **LOC** | 1,140 lines |

**Key Data Classes:**
- `ApplicationFingerprint` — Patterns for one app
- `ClassAssignment` — Label + confidence + evidence
- `PartitionConfig` — Algorithm configuration
- `PartitionResult` — Assignments + visualization data

**Algorithm Parameters:**
- `forward_weight`: 0.7 (classes I import)
- `reverse_weight`: 0.3 (classes importing me)
- `package_cohesion_bonus`: 0.2
- Phase thresholds: 0.8, 0.6, 0.4

**Reuse in Kernel:** `partition.py` (Stage 1) ✓ Already using

---

### 7. ServiceDetector (`service_detector.py`)

**Purpose:** Auto-detects services from filesystem, RAG, AST, and content analysis.

| Aspect | Details |
|--------|---------|
| **Class** | `ServiceDetector` |
| **Main Method** | `detect()` |
| **Input** | `project_path` + optional `rag_project`, `ast_graph` |
| **Output** | `AuditConfig` |
| **LOC** | 635 lines |

**Key Data Classes:**
- `DetectedService` — Service with confidence and evidence
- `AuditConfig` — Complete audit configuration

**Detection Sources:**
- FILESYSTEM, RAG_CONCEPT, RAG_CHUNK
- AST_CLASS, AST_PACKAGE
- CONTENT_ANNOTATION, CONTENT_JAVADOC, POM_ARTIFACT

**Reuse in Kernel:** `services.py` (Stage 1) ✓ Already using

---

### 8. DriftAnalyzer (`drift.py`)

**Purpose:** Detects spec-code drift between documentation and implementation.

| Aspect | Details |
|--------|---------|
| **Class** | `DriftAnalyzer` |
| **Main Method** | `analyze(spec_data, code_data)` |
| **Input** | Specification data + code analysis data |
| **Output** | `DriftReport` |
| **LOC** | ~300 lines |

**Reuse in Kernel:** Could add `drift.py` (Stage 2)

---

### 9. TimelineScanner (`timeline.py`)

**Purpose:** Builds component timelines from file modification history (without git).

| Aspect | Details |
|--------|---------|
| **Class** | `TimelineScanner` |
| **Main Method** | `scan_directory(path)`, `build_component_timelines()` |
| **Input** | Directory path |
| **Output** | `Dict[str, ComponentTimeline]` |
| **LOC** | ~400 lines |

**Key Data Classes:**
- `FileTimeline` — Per-file modification history
- `ComponentTimeline` — Aggregated component timeline
- `LifecycleCategory` — NEW, ACTIVE, MATURE, LEGACY_HOT, FROZEN, etc.

**Reuse in Kernel:** Could add `timeline.py` (Stage 1)

---

### 10. ComponentMapper (`component_mapper.py`)

**Purpose:** Maps files to logical components based on patterns.

| Aspect | Details |
|--------|---------|
| **Class** | `ComponentMapper` |
| **Main Method** | `map_directory(path)` |
| **Input** | Directory path + patterns |
| **Output** | Component mappings |
| **LOC** | ~300 lines |

**Component Types:**
- SERVICE (SK##), SCREEN (SC##), GENERAL (SG##)
- MODULE, LIBRARY, UNKNOWN

**Reuse in Kernel:** Used internally by other tools

---

## ragix_core/ — Core Infrastructure

### 11. DependencyGraph (`dependencies.py`)

**Purpose:** Build and query dependency graphs from AST.

| Aspect | Details |
|--------|---------|
| **Class** | `DependencyGraph` |
| **Main Methods** | `add_file()`, `add_directory()`, `get_symbols()`, `get_stats()` |
| **Input** | File paths or directories |
| **Output** | Graph with symbols and dependencies |
| **LOC** | ~600 lines |

**Key Methods:**
- `add_directory(path, patterns, recursive)` — Scan directory
- `get_symbols()` → `List[Symbol]`
- `get_all_dependencies()` → `List[Dependency]`
- `get_stats()` → `DependencyStats`
- `detect_cycles()` → `List[List[str]]`
- `from_cached_data(symbols, deps)` — Reconstruct from cache

**Reuse in Kernel:** `ast_scan.py`, `dependency.py` (Stage 1) ✓ Already using

---

### 12. calculate_metrics_from_graph (`code_metrics.py`)

**Purpose:** Calculate professional code metrics from dependency graph.

| Aspect | Details |
|--------|---------|
| **Function** | `calculate_metrics_from_graph(graph)` |
| **Input** | `DependencyGraph` |
| **Output** | `ProjectMetrics` |
| **LOC** | ~700 lines |

**Metrics Computed:**
- Lines of Code (LOC, SLOC, comments)
- Cyclomatic Complexity (CC)
- Maintainability Index (MI)
- Technical Debt estimation
- Hotspots (high-complexity methods)

**Reuse in Kernel:** `metrics.py` (Stage 1) ✓ Already using

---

## Summary: Kernel → Tool Mapping

| Kernel | Stage | Wraps Tool | Status |
|--------|-------|------------|--------|
| `ast_scan.py` | 1 | `DependencyGraph` | ✓ Done |
| `metrics.py` | 1 | `calculate_metrics_from_graph` | ✓ Done |
| `dependency.py` | 1 | `DependencyGraph.get_stats()` | ✓ Done |
| `partition.py` | 1 | `CodebasePartitioner` | ✓ Done |
| `services.py` | 1 | `ServiceDetector` | ✓ Done |
| `stats_summary.py` | 2 | `DistributionStats`, `complexity_histogram` | ✓ Done (rewritten) |
| `hotspots.py` | 2 | From `metrics` output | ✓ Done |
| `dead_code.py` | 2 | `DeadCodeDetector` | ✓ Done (rewritten) |
| `coupling.py` | 2 | `CouplingComputer`, `compute_all_propagation_factors` | ✓ Done (rewritten) |
| `entropy.py` | 2 | `EntropyComputer`, `compute_inequality_metrics` | ✓ Done |
| `risk.py` | 2 | `RiskScorer` | ✓ Done |
| `timeline.py` | 1 | `TimelineScanner` | ✓ Done |

---

## Input/Output Data Flow

```
                    PROJECT PATH
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: DATA COLLECTION                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ast_scan ──────► DependencyGraph.add_directory()           │
│      │                    │                                 │
│      │                    ▼                                 │
│      │           symbols, dependencies                      │
│      │                    │                                 │
│      ▼                    ▼                                 │
│  metrics ◄──── calculate_metrics_from_graph()               │
│      │                                                      │
│      ▼                                                      │
│  dependency ◄── DependencyGraph.get_stats()                 │
│      │              detect_cycles()                         │
│      │                                                      │
│      ▼                                                      │
│  partition ◄─── CodebasePartitioner.partition()             │
│      │                                                      │
│      ▼                                                      │
│  services ◄──── ServiceDetector.detect()                    │
│      │                                                      │
│      ▼                                                      │
│  timeline ◄──── TimelineScanner.build_component_timelines() │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 2: ANALYSIS                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  stats_summary ◄── StatisticsComputer.compute_from_metrics()│
│                                                             │
│  coupling ◄─────── CouplingComputer.compute_from_graph()    │
│                                                             │
│  dead_code ◄────── DeadCodeDetector.analyze()               │
│                                                             │
│  entropy ◄──────── EntropyComputer.compute_all()            │
│                                                             │
│  hotspots ◄─────── (from metrics.hotspots)                  │
│                                                             │
│  risk ◄─────────── RiskScorer.score_all()                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Notes for Kernel Implementation

### Principle: Wrap, Don't Reimplement

Each kernel should:
1. Import the existing tool class
2. Prepare inputs from previous kernel outputs
3. Call the tool's main method
4. Transform output to JSON-serializable format
5. Generate <500 char summary for LLM

### Example Pattern

```python
class MyKernel(Kernel):
    name = "my_kernel"
    requires = ["ast_scan"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Import existing tool
        from ragix_audit.some_tool import SomeTool

        # Load dependency outputs
        with open(input.dependencies["ast_scan"]) as f:
            ast_data = json.load(f).get("data", {})

        # Prepare inputs for tool
        prepared_input = self._prepare(ast_data)

        # Call existing tool (REUSE!)
        tool = SomeTool()
        result = tool.compute(prepared_input)

        # Return JSON-serializable output
        return result.to_dict()
```

---

**Document maintained for KOAS development reference.**
