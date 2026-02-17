# ROADMAP — Memory Reporting (v2.0 — Hybrid Architecture)

**Status**: PROPOSED
**Date**: 2026-02-16
**Supersedes**: v1.0 (full YAML DSL approach)

---

## 0. Decision Record

### Architecture: YAML = data, Python = control flow

The v1.0 roadmap proposed a full YAML DSL with `${...}` interpolation, `foreach` loops,
`capture`, `assert`, and `timed_block` directives. After review, **hybrid** was chosen:

| Aspect | Full YAML DSL | Hybrid (chosen) |
|--------|:-------------:|:----------------:|
| Target users | Non-developers | Developers (v1) |
| Scenario editing | YAML only | Python class + YAML config |
| Debugging | YAML path errors, opaque | Stack traces, IDE support |
| New step types | Registration + dispatch | Python method |
| Interpolation engine | ~300 lines, error-prone | Not needed |
| Estimated code | ~1500 lines | ~1000 lines |
| Delivery risk | High (DSL competes with core) | Low |

**What we keep from the DSL plan**:
- `timed_block` → Python context manager in `engine.py`
- `assert` → `engine.assert_*()` helper methods
- `capture` → local Python variables
- Tunable configs → YAML data files + CLI `--set` overrides

**What we lose**: Non-developers editing scenario logic directly. But a non-developer
editing a DSL with loops/capture/assert would fare worse than editing a plain YAML
config anyway.

---

## 1. Package Layout

```
ragix_core/memory/reporting/
  __init__.py              # Public: generate_report, list_scenarios
  api.py                   # generate_report() — public entry point
  engine.py                # ReportEngine (MockMCP adapter, metrics, assert helpers)
  renderer.py              # MarkdownRenderer (reusable section methods)
  schema.py                # Config dataclasses + validation
  io.py                    # YAML loader, path resolution, --set override merging
  scenarios/
    __init__.py            # ScenarioBase ABC + SCENARIO_REGISTRY
    summarize_content.py   # SummarizeContentScenario
    benchmarks.py          # BenchmarkScenario
    regression_min.py      # RegressionMinScenario
    domain_summary.py      # DomainSummaryScenario (M9)
  config/
    summarize_content.yml  # Data: queries, chains, budgets
    benchmarks.yml         # Data: query lists, thresholds
    regression_min.yml     # Data: smoke queries, latency limits
    domain_summary.yml     # Data: bucket defs, rendering opts (M9)
```

No new dependencies. Same stack as existing memory subsystem.

---

## 2. Design Contracts

### 2.1 Injection Format Contract

The `memory_recall` injection block is a parseable contract:

```
## Memory (Injected)
format_version: 1
injection_type: memory_recall
generated_at: <ISO8601>
budget_tokens: <int>          ← requested budget
matched: <int>                ← total items matching query
used: <int>                   ← actual tokens consumed (= len(inject_text) // 4)

[1] [TIER:validation] type — title
    content...
    provenance: ...
    tags: ...
    confidence: ...

--- End Memory (format_version=1, N items, T tokens) ---
```

**Rules**:
- `format_version` always present, always `1` (until explicitly bumped)
- `budget_tokens` = requested, `used` = actual consumed (single source of truth)
- Footer `T tokens` matches header `used` (fix current inconsistency)
- Scenarios can assert `format_version` via `engine.assert_format_version()`

### 2.2 Scenario Contract

```python
class ScenarioBase(ABC):
    """Base class for all reporting scenarios."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def default_config_name(self) -> str: ...

    @abstractmethod
    def run(self, engine: ReportEngine, config: dict) -> str:
        """Execute the scenario, return Markdown string."""
        ...
```

Every scenario:
- Receives a `ReportEngine` (tool access + metrics) and a validated `config` dict
- Returns a Markdown string
- May call `engine.tool(name, **kwargs)` for MCP tool access
- May call `engine.timed(label)` context manager for timing
- May call `engine.assert_*()` for runtime checks
- May use `MarkdownRenderer` for consistent formatting

### 2.3 Engine Contract

```python
class ReportEngine:
    """Orchestrates MCP tool calls, timing, and assertions."""

    def __init__(self, db_path, workspace, embedder="mock"): ...
    def tool(self, name: str, **kwargs) -> dict: ...
    def timed(self, label: str) -> ContextManager: ...
    def timings(self) -> dict[str, float]: ...
    def metrics_summary(self) -> dict: ...

    # Assert helpers (raise AssertionError with context)
    def assert_format_version(self, inject_text: str, expected: int = 1): ...
    def assert_min_count(self, label: str, actual: int, minimum: int): ...
    def assert_max_latency(self, label: str, max_ms: float): ...
    def assert_no_errors(self): ...
```

---

## 3. Public API

```python
# ragix_core/memory/reporting/api.py

def generate_report(
    *,
    db_path: str,
    workspace: str,
    scenario: str = "summarize_content",
    config_path: str | None = None,
    out_path: str | None = None,
    overrides: dict | None = None,
    embedder: str = "mock",
) -> str:
    """
    Run a reporting scenario against a RAGIX Memory DB.

    Args:
        db_path: Path to SQLite memory database.
        workspace: Named workspace (must exist or will be registered).
        scenario: Scenario name (builtin) or dotted module path.
        config_path: YAML config file. If None, uses builtin default.
        out_path: Write report here. If None, return string only.
        overrides: Dict of dotted-path overrides (e.g. {"recall.budgets": [500, 1500]}).
        embedder: Embedding backend ("mock" for now).

    Returns:
        Generated Markdown report as string.
    """
```

---

## 4. CLI Design

```
ragix-memory report \
  --db path/to/memory.db \
  --workspace rie-corp_energy \
  --scenario summarize_content \
  [--config path/to/config.yml] \
  [--set recall.budgets=500,1500,4000] \
  [--set search.k=8] \
  [--out report.md] \
  [--embedder mock] \
  [-v/--verbose]
```

**Behavior**:
- `--scenario` accepts builtin name or Python module path
- `--config` overrides the scenario's default YAML; if omitted, uses builtin
- `--set` applies dotted-path overrides to loaded config (repeatable)
- `--out` optional; if omitted, prints to stdout
- Wired into existing `ragix_core/memory/cli.py` as a new subcommand

---

## 5. YAML Config Schema (data only, no DSL)

### 5.1 summarize_content.yml

```yaml
meta:
  id: summarize_content
  title: "RAGIX Memory — Content Summary Report"
  language: fr

inventory:
  by_type: true
  by_tags: true
  top_tags_k: 15

queries:
  k: 5
  read_full_top_n: 3
  items:
    - question: "Quelles sont les non-conformites critiques ?"
      query: "non-conforme critique"
      type_filter: constraint
    - question: "Quel est le plan de remediation prioritaire ?"
      query: "plan remediation priorite"
    - question: "Quels RIE definissent les regles pour Oracle ?"
      query: "oracle 19c exadata support"
    - question: "ACME-ERP peut-il migrer vers Kubernetes ?"
      query: "kubernetes rancher containerisation"
    - question: "Quelles sont les derogations en cours ?"
      query: "derogation formelle echeance"
    - question: "Quels sont les risques de securite identifies ?"
      query: "vulnerabilite securite risque CVE"
    - question: "Quel est le cadre commun d'hebergement CORP-ENERGY ?"
      query: "cadre commun hebergement EDGAR"
      tags: cadre-commun
    - question: "Quels composants sont hors perimetre ?"
      query: "hors perimetre"
      k: 12

crossrefs:
  k: 2
  chains:
    - name: "Oracle: constraint vs. specification vs. remediation"
      facets:
        - label: "Oracle constraint"
          query: "oracle non-conforme"
          type_filter: constraint
        - label: "Oracle RIE spec"
          query: "oracle 19c exadata version"
          type_filter: note
        - label: "Oracle remediation"
          query: "oracle migration derogation"

    - name: "CentOS / RHEL: gap analysis chain"
      facets:
        - label: "CentOS finding"
          query: "centos 7 non-conforme"
          type_filter: constraint
        - label: "RHEL specification"
          query: "RHEL 8 9 regles"
          type_filter: note
        - label: "Remediation path"
          query: "centos RHEL migration"

    - name: "Hebergement: EDGAR vs. Cloud"
      facets:
        - label: "Cadre commun"
          query: "cadre commun hebergement EDGAR"
          type_filter: definition
        - label: "PostgreSQL violation"
          query: "postgresql azure cloud interdit"
          type_filter: constraint
        - label: "K8S option"
          query: "kubernetes EDGAR cloud"
          type_filter: note

recall:
  query: "non-conformite critique ACME-ERP oracle centos postgresql"
  budgets: [500, 1500, 4000]
  truncation_chars: 3000

filtered_views:
  - name: "All constraints (compliance violations)"
    query: "*"
    type_filter: constraint
    k: 10
  - name: "All decisions (aggregated verdicts)"
    query: "*"
    type_filter: decision
    k: 10
  - name: "All definitions (reference framework)"
    query: "*"
    type_filter: definition
    k: 10
  - name: "Tag filter: oracle"
    query: "oracle"
    tags: oracle
    k: 5
  - name: "Tag filter: securite"
    query: "securite vulnerabilite"
    tags: securite
    k: 5

links:
  create:
    - src_pattern: "Oracle 11gR2 non-conforme"
      dst_pattern: "RIE Oracle 19c"
      relation: supports
    - src_pattern: "CentOS 7 non-conforme"
      dst_pattern: "RIE RHEL"
      relation: supports
    - src_pattern: "PostgreSQL Azure non-conforme"
      dst_pattern: "RIE PostgreSQL 13"
      relation: supports
    - src_pattern: "Oracle 11gR2 non-conforme"
      dst_pattern: "Score conformite"
      relation: refines
    - src_pattern: "CentOS 7 non-conforme"
      dst_pattern: "Score conformite"
      relation: refines
    - src_pattern: "PostgreSQL Azure non-conforme"
      dst_pattern: "Score conformite"
      relation: refines
    - src_pattern: "Score conformite"
      dst_pattern: "Plan remediation"
      relation: supports
    - src_pattern: "Cadre commun"
      dst_pattern: "PostgreSQL Azure non-conforme"
      relation: supports
    - src_pattern: "Risques"
      dst_pattern: "Oracle 11gR2 non-conforme"
      relation: supports
  traverse:
    start_pattern: "Oracle 11gR2 non-conforme"
```

### 5.2 benchmarks.yml

```yaml
meta:
  id: benchmarks
  title: "RAGIX Memory — Benchmark Report"

search:
  k: 5
  queries:
    - "oracle conformite critique"
    - "migration java rhel kubernetes"
    - "derogation formelle acme_erp"
    - "sauvegarde commvault ransomware"
    - "spring4shell cve securite"
    - "postgresql azure cloud edgar"
    - "weblogic version cluster"
    - "centos rhel fin de vie"
    - "kubernetes rancher argocd"
    - "ports reseau jms msg_hub"

recall:
  runs:
    - query: "non-conformite critique acme_erp oracle centos postgresql"
      budget_tokens: 3000
    - query: "plan remediation derogation migration"
      budget_tokens: 1500
    - query: "kubernetes containerisation modernisation"
      budget_tokens: 500

thresholds:
  max_avg_search_ms: 50
  max_avg_recall_ms: 200
  format_version: 1
```

### 5.3 regression_min.yml

```yaml
meta:
  id: regression_min
  title: "RAGIX Memory — Regression Smoke Report"
  strict: true

search:
  k: 3
  queries:
    - "oracle conformite critique"
    - "centos rhel fin de vie"
    - "postgresql azure cloud"

recall:
  query: "non-conformite critique acme_erp oracle centos postgresql"
  budget_tokens: 800
  truncation_chars: 1200

thresholds:
  min_items_total: 1
  max_avg_search_ms: 25
  max_avg_recall_ms: 80
  format_version: 1
```

---

## 6. MarkdownRenderer — Reusable Section Methods

```python
class MarkdownRenderer:
    """Consistent Markdown formatting for report sections."""

    def __init__(self, title: str, author: str = "...", corpus: str = ""): ...

    # Reusable sections (called by multiple scenarios)
    def header(self, meta: dict) -> None: ...
    def section_inventory(self, engine, *, by_type, by_tags, top_tags_k) -> None: ...
    def section_queries(self, engine, queries_config: list) -> None: ...
    def section_crossrefs(self, engine, crossrefs_config: dict) -> None: ...
    def section_recall(self, engine, query, budgets, truncation_chars) -> None: ...
    def section_filtered(self, engine, views: list) -> None: ...
    def section_link_graph(self, engine, links_config: dict) -> None: ...
    def section_benchmark_table(self, engine, phase, rows, columns) -> None: ...
    def section_stats(self, engine) -> None: ...
    def footer(self, engine) -> None: ...

    def render(self) -> str: ...
```

Each method:
- Takes an `engine` (for tool calls) and config data
- Appends to internal line buffer
- Handles its own formatting (tables, blockquotes, code blocks)
- Is independently testable

---

## 7. Milestones

### M1: engine.py — Tool adapter + metrics + assert helpers
- `ReportEngine.__init__()`: MemoryStore + MockEmbedder + RecallEngine + Dispatcher + WorkspaceRouter + MetricsCollector + MockMCP
- `tool(name, **kwargs)`: dispatches to registered MCP tool
- `timed(label)`: context manager, stores (label → ms) in `self._timings`
- `assert_format_version()`, `assert_min_count()`, `assert_max_latency()`, `assert_no_errors()`
- Est: ~120 lines

### M2: renderer.py — MarkdownRenderer with 7+ section methods
- Port section logic from `run_rie_content_report.py`
- Each method: engine + config → appends Markdown lines
- `render()` joins and returns
- Est: ~300 lines

### M3: api.py — Public entry point
- `generate_report()` with the signature defined in §3
- Scenario resolution: builtin name → `SCENARIO_REGISTRY[name]`
- Config loading: builtin default or explicit path + overrides
- Write to file if `out_path` given
- Est: ~80 lines

### M4: schema.py + io.py — Config validation + YAML loader
- `schema.py`: dataclasses for query items, crossref chains, recall config, thresholds
- `io.py`: `load_config(path_or_builtin)`, `apply_overrides(config, overrides)`, `resolve_builtin_path(name)`
- Override syntax: `{"recall.budgets": [500, 1500]}` → nested dict merge
- Est: ~100 lines

### M5: summarize_content scenario — Port from run_rie_content_report.py
- `SummarizeContentScenario.run()`: calls renderer section methods in order
- `config/summarize_content.yml`: queries, chains, budgets, views, links (§5.1)
- Validates output matches current report structure
- Est: ~60 lines Python + 80 lines YAML

### M6: benchmarks scenario — Port from run_rie_benchmark.py
- `BenchmarkScenario.run()`: timed search batch, timed recall batch, consolidation, stats
- `config/benchmarks.yml`: query lists, recall runs, thresholds (§5.2)
- Renderer: `section_benchmark_table()` for timing tables
- Est: ~70 lines Python + 40 lines YAML

### M7: regression_min scenario — CI smoke test
- `RegressionMinScenario.run()`: preflight, 3 searches, 1 recall, consolidation, asserts
- `config/regression_min.yml`: minimal queries, strict thresholds (§5.3)
- All asserts must pass for CI green
- Est: ~50 lines Python + 25 lines YAML

### M8: CLI wiring — `ragix-memory report` subcommand
- Add `report` subparser to `ragix_core/memory/cli.py`
- Args: `--db`, `--workspace`, `--scenario`, `--config`, `--set` (repeatable), `--out`, `--embedder`, `-v`
- Calls `generate_report()` with parsed args
- Est: ~60 lines

### M9: domain_summary scenario — KOAS-level summarizer (advanced)
- `DomainSummaryScenario` with private helpers:
  - `_discover_domains()`: tags-first, lexical fallback
  - `_bucketize()`: constraint / decision / operational_rule classification
  - `_coverage_table()`: per-domain item counts
  - `_executive_bullets()`: one-liner per domain from top constraint
  - `_domain_section()`: 3 subsections with MID bullets
- `config/domain_summary.yml`: bucket defs, k_per_bucket, rendering opts, dedup threshold
- Renderer: `section_domain()` (new, domain-specific)
- Est: ~200 lines Python + 60 lines YAML

### M10: Tests + CI hooks
- `tests/test_reporting_engine.py`: engine init, tool dispatch, timed blocks, assert helpers
- `tests/test_reporting_renderer.py`: individual section methods with mock data
- `tests/test_reporting_regression_min.py`: full `generate_report(scenario="regression_min")` smoke
- `tests/test_reporting_format_contract.py`: injection block parsing, format_version stability
- Est: ~150 lines total

---

## 8. Injection Format Fix (from M5 onward)

Current inconsistency in report output:
```
Tokens used: 1524/1500    ← external counter (len(inject) // 4)
used: 1469                ← internal counter (from formatting loop)
```

**Fix**: Unify to a single counter. `used` in the injection header = `len(inject_text) // 4`
computed *after* the full block is rendered. The header `used:` line is written last
(or rewritten) to match the actual size.

This is a change in `ragix_core/memory/mcp/formatting.py` — not in the reporting
package itself. But the reporting package enforces it via `engine.assert_format_version()`.

---

## 9. Mapping from Current Scripts

| Source | Target |
|--------|--------|
| `run_rie_content_report.py` `MockMCP` | `engine.py` `ReportEngine` |
| `_section1_inventory()` | `renderer.section_inventory()` |
| `_section2_audit_queries()` | `renderer.section_queries()` |
| `_section3_cross_references()` | `renderer.section_crossrefs()` |
| `_section4_recall_injection()` | `renderer.section_recall()` |
| `_section5_filtered_discovery()` | `renderer.section_filtered()` |
| `_section6_link_graph()` | `renderer.section_link_graph()` |
| `_footer()` | `renderer.footer()` |
| `run_rie_benchmark.py` timing logic | `engine.timed()` + `BenchmarkScenario` |
| `run_rie_benchmark.py` search/recall tables | `renderer.section_benchmark_table()` |

---

## 10. "Done" Definition

### Library

```python
from ragix_core.memory.reporting import generate_report

md = generate_report(
    db_path="memory_bench/rie_bench.db",
    workspace="rie-corp_energy",
    scenario="summarize_content",
    out_path="report.md",
)
```

### CLI

```bash
# Content report
ragix-memory report --db rie_bench.db --workspace rie-corp_energy \
  --scenario summarize_content --out report.md

# Benchmark
ragix-memory report --db rie_bench.db --workspace rie-corp_energy \
  --scenario benchmarks --out benchmark.md

# Regression smoke (CI)
ragix-memory report --db rie_bench.db --workspace rie-corp_energy \
  --scenario regression_min --out regression.md

# Custom config + overrides
ragix-memory report --db rie_bench.db --workspace rie-corp_energy \
  --scenario summarize_content \
  --config ./custom_queries.yml \
  --set recall.budgets=500,2000 \
  --set search.k=8 \
  --out custom_report.md
```

### Tests

```bash
pytest ragix_core/memory/reporting/tests/ -q   # all pass, <3s
```
