# KOAS Summary — Multi-Document Summarization with Memory Persistence

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

**Version:** 3.1 (2026-02-15)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [CLI Reference](#4-cli-reference-summaryctl)
5. [MCP Tools](#5-mcp-tools)
6. [Visualization Outputs](#6-visualization-outputs)
7. [Secrecy Model](#7-secrecy-model)
8. [Memory Subsystem Integration](#8-memory-subsystem-integration)
9. [Graph-RAG](#9-graph-rag)
10. [Drift Detection](#10-drift-detection)
11. [Configuration](#11-configuration)
12. [Performance](#12-performance)
13. [File Layout](#13-file-layout)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Overview

### What KOAS Summary Does

KOAS Summary is a **multi-document summarization pipeline** that produces balanced, citeable technical summaries from large document corpora (10³–10⁵ pages). It combines:

- **Memory-persistent extraction**: Document rules and facts are stored in a persistent SQLite database with FTS5 full-text search
- **Graph-RAG consolidation**: Provenance graphs constrain merge candidates, preventing cross-corpus mega-chains
- **Secrecy-tier redaction**: S0 (public), S2 (internal), S3 (audit) tiers control what gets exposed
- **Drift detection**: Cross-corpus version comparison with Sankey visualizations
- **Budgeted recall**: Token-limited memory injection balances coverage vs. context budget

### Key Differentiators

1. **Local-first sovereignty**: All processing on-premises via Ollama, no external APIs
2. **Full traceability**: Every memory item links back to source documents with chunk hashes
3. **Never-delete architecture**: Consolidation supersedes items instead of deleting them
4. **Deterministic content recipes**: Math/tables/code masked before LLM to prevent hallucination
5. **8 interactive visualizations**: Graph, memory explorer, query, timeline, clusters, heatmap, metrics, drift

### Use Cases

- **Technical audits**: Summarize 100+ PDF reports into a 20-page executive overview
- **Compliance tracking**: Extract MUST/SHALL constraints across policy documents
- **Corpus versioning**: Detect drift between Q1 and Q2 documentation releases
- **Cross-domain synthesis**: Merge findings from RHEL, Oracle, Kubernetes domains with citations

---

## 2. Architecture

### The KOAS 3-Stage Pipeline

```mermaid
graph LR
    A[Document Corpus] --> S1[Stage 1: Collection]
    S1 --> B[(Memory Store<br/>SQLite + FTS5)]
    B --> S2[Stage 2: Analysis]
    S2 --> C[(Graph Store<br/>Provenance Graph)]
    C --> S3[Stage 3: Reporting]
    S3 --> D[summary.md<br/>summary.json]
    S3 --> E[8 HTML Views]
```

### Stage 1: Collection (Ingestion)

**Kernels:**
- `summary_collect` — Discover all PDF/MD/TXT files in corpus
- `summary_ingest` — Extract rules/facts via LLM, store in memory with policy governance

**Outputs:**
- `memory.db` — SQLite database with 6 tables (items, revisions, embeddings, links, palace_locations, events)
- `stage1/summary_collect.json` — File inventory
- `stage1/summary_ingest.json` — Ingestion stats

**What happens:**
1. Files are chunked using strategy: `pages` (page breaks), `headings` (markdown sections), or `windows` (sliding 800-token windows)
2. Each chunk is sent to the LLM (default: Granite 32B) with a prompt: "Extract 1–3 reusable rules/facts"
3. Extracted items are validated by **policy.py** (blocks secrets, injection patterns, oversized content)
4. Accepted items are written to `memory_items` table with provenance (source_id, chunk_ids, content_hashes)
5. FTS5 index automatically syncs via triggers for fast fulltext search

### Stage 2: Analysis (Consolidation + Recall)

**Kernels:**
- `summary_extract_entities` — Deterministic NER: extract CVE-IDs, versions, product names, config paths
- `summary_build_graph` — Build provenance graph (doc → chunk → item → entity edges)
- `summary_consolidate` — Cluster STM items by type+tags+embeddings, merge via Granite 3.2B, promote to MTM/LTM
- `summary_budgeted_recall` — Select top items per domain within token budget

**Outputs:**
- `graph.db` — Graph store (shares SQLite with memory.db)
- `stage2/summary_consolidate.json` — Merge chains (which items were clustered)
- `stage2/summary_budgeted_recall.json` — Selected items per domain with token counts

**What happens:**
1. **Entity extraction**: Scans all memory items for structured entities (CVE-2024-1234, RHEL 8.6, nginx.conf). Only controlled-vocabulary entities create graph nodes (no free-text noise).
2. **Graph building**: Creates nodes (doc, chunk, item, entity) and edges (contains, adjacent, extracted_from, mentions, similar). Edge weights based on embedding similarity for `similar` edges.
3. **Consolidation**:
   - Coarse clustering by `(type, tags)` buckets
   - Fine clustering within buckets using embedding distance (threshold: 0.75 cosine similarity)
   - Graph constraint: only merge items sharing a 2-hop provenance neighborhood (prevents cross-corpus chains)
   - For each cluster: merge via Granite 3.2B summarizer or deterministic fallback (longest content wins, tags/entities merged)
   - Original items marked `superseded_by=new_id` and `archived=true`
   - Promote items: `usage_count>=3` STM → MTM, `usage_count>=7` MTM → LTM
4. **Budgeted recall**:
   - Group items by domain (extracted from source_id, e.g., "rhel", "oracle", "kubernetes")
   - Allocate tokens: global budget (12,000) divided evenly across domains
   - Rank items per domain: weighted score (tags overlap 0.3, embedding 0.5, provenance quality 0.2)
   - Select top items until domain budget exhausted

### Stage 3: Reporting (Generation + Verification + Assembly)

**Kernels:**
- `summary_generate` — Generate summary sections per domain using budgeted memory
- `summary_verify` — Check citations (ensure all references exist in memory store)
- `summary_report` — Assemble final `summary.md` + `summary.json`
- `summary_redact` — Apply secrecy-tier redaction (S0/S2 only; S3 is no-op)
- `summary_capabilities` — Generate capabilities manifest (model versions, domains, stats)

**Outputs:**
- `stage3/summary.md` — Full markdown report with citations
- `stage3/summary.json` — Structured JSON (sections, citations, metadata)
- `stage3/summary_S0.md` / `summary_S2.md` — Redacted versions (if secrecy < S3)
- `stage3/capabilities.json` — Manifest for audit trails

**What happens:**
1. **Generation**: For each domain, inject budgeted items into LLM context with prompt: "Synthesize a balanced technical summary in {language}. Cite sources as [MEM-abc123]."
2. **Verification**: Parse citations from generated text, check existence in memory store, flag broken references
3. **Assembly**: Combine domain sections into single markdown with:
   - Executive summary (cross-domain synthesis)
   - Domain-specific sections
   - Bibliography (memory item IDs → source documents)
4. **Redaction** (if secrecy < S3):
   - S0 (public): redact paths, emails, hostnames, IPs, filenames, hashes, entity labels
   - S2 (internal): redact hostnames, IPs, hashes only
   - S3 (audit): no redaction
5. **Capabilities**: JSON manifest with model name/version, token counts, domain list, ingestion timestamp

---

## 3. Quick Start

### Prerequisites

- **Python 3.10+** (tested on 3.12)
- **Ollama** running locally for LLM (default: `ibm/granite4:32b-a9b-h`)
- **Optional**: `faiss-cpu` for O(n log n) embedding search (falls back to sequential scan if absent)
- **Optional**: `jinja2` for template rendering (has fallback)

### Installation

```bash
cd /path/to/RAGIX
pip install -e .
```

### Basic Workflow

#### 1. Ingest a corpus

```bash
python -m ragix_kernels.summary.cli.summaryctl ingest /path/to/documents \
  --scope corp_energy-rie \
  --model ibm/granite4:32b-a9b-h \
  -v
```

**What it does:**
- Creates `summary_workspace/` in parent directory of corpus
- Writes `memory.db` with extracted rules/facts
- Outputs ingestion stats (files processed, chunks, items accepted)

**Flags:**
- `--workspace PATH` — Custom output directory (default: auto)
- `--strategy {pages,headings,windows}` — Chunking strategy (default: pages)
- `--max-chunk-tokens N` — Max tokens per chunk (default: 800)
- `--delta` — Only process new/modified files (requires `--corpus-id`)
- `--corpus-id ID` — Corpus version label for cross-corpus tracking (e.g., `corp_energy-rie-2026Q1`)
- `-v` — Verbose output

#### 2. Generate summary

```bash
python -m ragix_kernels.summary.cli.summaryctl summarize summary_workspace/ \
  --model ibm/granite4:32b-a9b-h \
  --language French \
  --secrecy S2 \
  -v
```

**What it does:**
- Runs consolidation (clusters + merges STM items)
- Builds provenance graph
- Budgeted recall (selects top items per domain)
- Generates summary sections
- Verifies citations
- Assembles `summary.md` + `summary_S2.md` (redacted)

**Flags:**
- `--max-tokens N` — Injection budget (default: 12000)
- `--secrecy {S0,S2,S3}` — Redaction tier (default: S3=full detail)
- `--skip-consolidation` — Skip merge step (reuse existing MTM/LTM)
- `--no-graph` — Disable graph-assisted consolidation
- `--delta` — Neighborhood-scoped consolidation (only merge near new items)
- `--domains D1 D2` — Only regenerate specified domains

#### 3. Visualize

```bash
python -m ragix_kernels.summary.cli.summaryctl viz summary_workspace/ \
  --secrecy S2 \
  --views graph,memory,query \
  -v
```

**What it does:**
- Generates interactive HTML dashboards in `summary_workspace/viz/`
- Redacts sensitive fields based on `--secrecy` tier
- Selective rendering via `--views` (default: all 8 views)

#### 4. Query memory

```bash
python -m ragix_kernels.summary.cli.summaryctl query summary_workspace/ \
  "CVE nginx RHEL" \
  --domain rhel \
  --tier mtm \
  -k 10 \
  -v
```

**What it does:**
- Full-text search via SQLite FTS5 (`MATCH` query)
- Filters by tier, type, domain
- Returns top-k results with relevance scores

#### 5. Full pipeline (ingest + summarize)

```bash
python -m ragix_kernels.summary.cli.summaryctl run /path/to/documents \
  --scope corp_energy-rie \
  --model ibm/granite4:32b-a9b-h \
  --language French \
  --secrecy S2 \
  -v
```

---

## 4. CLI Reference (summaryctl)

### `ingest` — Ingest corpus into memory

```bash
summaryctl ingest <corpus_folder> [OPTIONS]
```

**Positional arguments:**
- `corpus_folder` — Path to folder containing PDF/MD/TXT files

**Options:**
- `-w, --workspace PATH` — Output directory (default: `<corpus_parent>/summary_workspace`)
- `--scope LABEL` — Memory scope label (default: `project`)
- `--model NAME` — LLM for rule extraction (default: `ibm/granite4:32b-a9b-h`)
- `--strategy {pages,headings,windows}` — Chunking strategy (default: `pages`)
- `--max-chunk-tokens N` — Max tokens per chunk (default: `800`)
- `--embedder {mock,ollama,sentence-transformers}` — Embedding backend (default: `mock`)
- `--embedder-model NAME` — Embedding model (default: `nomic-embed-text`)
- `--ollama-url URL` — Ollama API endpoint (default: `http://localhost:11434`)
- `--delta` — Delta mode: only process new/modified files (requires `--corpus-id`)
- `--corpus-id ID` — Corpus version ID (e.g., `corp_energy-rie-2026Q1`)
- `-v, --verbose` — Verbose output

**Example:**
```bash
summaryctl ingest /data/corp_energy-rie-docs \
  --workspace /mnt/workspace \
  --scope corp_energy-rie \
  --corpus-id corp_energy-rie-2026Q1 \
  --delta \
  -v
```

---

### `summarize` — Generate summary from memory

```bash
summaryctl summarize <workspace> [OPTIONS]
```

**Positional arguments:**
- `workspace` — Workspace directory with `memory.db`

**Options:**
- `--scope LABEL` — Memory scope (default: `project`)
- `--model NAME` — LLM for generation (default: `ibm/granite4:32b-a9b-h`)
- `--max-tokens N` — Injection budget (default: `12000`)
- `--min-per-domain N` — Min items per domain (default: `3`)
- `--max-per-domain N` — Max items per domain (default: `25`)
- `--embedder {mock,ollama,sentence-transformers}` — Embedding backend (default: `mock`)
- `--embedder-model NAME` — Model for embeddings (default: `nomic-embed-text`)
- `--ollama-url URL` — Ollama endpoint (default: `http://localhost:11434`)
- `--language LANG` — Output language (default: `French`)
- `--skip-consolidation` — Skip consolidation step
- `--graph / --no-graph` — Enable/disable graph-assisted consolidation (default: enabled)
- `--secrecy {S0,S2,S3}` — Redaction tier (default: `S3`)
- `--delta` — Neighborhood-scoped consolidation only
- `--domains D1 D2 ...` — Only regenerate specified domains (e.g., `--domains rhel oracle`)
- `--corpus-id ID` — Filter by corpus version
- `-v, --verbose` — Verbose output

**Example:**
```bash
summaryctl summarize /mnt/workspace \
  --model ibm/granite4:32b-a9b-h \
  --language English \
  --secrecy S0 \
  --domains rhel kubernetes \
  --delta \
  -v
```

---

### `run` — Full pipeline (ingest + summarize)

```bash
summaryctl run <corpus_folder> [OPTIONS]
```

Combines `ingest` + `summarize` in one command. Accepts all flags from both subcommands.

**Example:**
```bash
summaryctl run /data/docs \
  --scope project \
  --model ibm/granite4:32b-a9b-h \
  --language French \
  --secrecy S2 \
  --max-tokens 15000 \
  -v
```

---

### `drift` — Cross-corpus drift detection

```bash
summaryctl drift <workspace> [OPTIONS]
```

**Positional arguments:**
- `workspace` — Output workspace for drift report

**Options (choose one mode):**

**Same-DB mode** (two corpus_ids in one memory.db):
- `--corpus-a ID` — Baseline corpus ID
- `--corpus-b ID` — Current corpus ID

**Two-workspace mode** (separate memory.db files):
- `--workspace-a PATH` — Baseline workspace
- `--workspace-b PATH` — Current workspace

**Common options:**
- `--scope LABEL` — Memory scope (default: `project`)
- `-v, --verbose` — Verbose output

**Example:**
```bash
summaryctl drift drift_workspace \
  --workspace-a baseline_ws \
  --workspace-b current_ws \
  -v
```

**Output:**
- `drift_report.json` — Added/removed/modified/unchanged item counts
- `drift_heatmap.json` — Domain × version drift matrix

---

### `viz` — Generate HTML visualizations

```bash
summaryctl viz <workspace> [OPTIONS]
```

**Positional arguments:**
- `workspace` — Workspace with `memory.db`

**Options:**
- `--scope LABEL` — Filter by scope
- `--corpus-id ID` — Filter by corpus version
- `--secrecy {S0,S2,S3}` — Redaction tier (default: `S3`)
- `--views LIST` — Comma-separated view names (default: all). Valid: `graph,memory,query,drift,timeline,clusters,heatmap,metrics`
- `-v, --verbose` — Verbose output

**Example:**
```bash
summaryctl viz /mnt/workspace \
  --secrecy S2 \
  --views graph,memory,query \
  -v
```

**Output:**
- `viz/graph.html` — Interactive graph view (D3 force-directed layout)
- `viz/memory.html` — Memory explorer with filters
- `viz/query.html` — Search-first query interface
- `viz/drift.html` — Sankey diagram for drift flow
- `viz/timeline.html` — Temporal slider (if timestamps present)
- `viz/clusters.html` — Community detection bubble chart
- `viz/heatmap.html` — Domain × version heatmap
- `viz/metrics.html` — Smart metrics dashboard

---

### `query` — Search memory items by text

```bash
summaryctl query <workspace> <query_terms> [OPTIONS]
```

**Positional arguments:**
- `workspace` — Workspace with `memory.db`
- `query_terms` — Space-separated search terms (AND logic)

**Options:**
- `--tier {stm,mtm,ltm}` — Filter by tier
- `--type TYPE` — Filter by item type (e.g., `fact`, `constraint`)
- `--scope LABEL` — Filter by scope
- `--domain DOMAIN` — Filter by document domain (e.g., `rhel`, `oracle`)
- `-k, --limit N` — Max results (default: `20`)
- `--scored` — Use hybrid RecallEngine scoring (tags + embeddings)
- `--embedder {mock,ollama}` — Embedding backend for `--scored` mode (default: `mock`)
- `--json` — JSON output (machine-readable)
- `-v, --verbose` — Verbose output

**Example:**
```bash
summaryctl query /mnt/workspace "CVE nginx vulnerability" \
  --domain rhel \
  --tier mtm \
  -k 10 \
  --scored \
  --embedder ollama \
  -v
```

**Output (table mode):**
```
Query: CVE nginx vulnerability
Mode:  scored (RecallEngine)
Found 8 result(s)

  Score  Tier  Type          Domain            Title
  ──────  ────  ────────────  ────────────────  ────────────────────────────────────────
  0.876  MTM   fact          rhel              CVE-2024-1234 affects nginx 1.18.0
  0.743  MTM   constraint    rhel              MUST upgrade nginx to 1.20.2+
  ...
```

---

### `show` — Display workspace info

```bash
summaryctl show <workspace> [OPTIONS]
```

**Positional arguments:**
- `workspace` — Workspace directory

**Options:**
- `--format {table,json}` — Output format (default: `table`)

**Example:**
```bash
summaryctl show /mnt/workspace --format json
```

**Output (JSON):**
```json
{
  "workspace": "/mnt/workspace",
  "items": 1247,
  "by_tier": {"stm": 342, "mtm": 651, "ltm": 254},
  "by_type": {"fact": 487, "constraint": 312, "definition": 221, ...},
  "events": 3891,
  "corpora": [
    {
      "corpus_id": "corp_energy-rie-2026Q1",
      "item_count": 1247,
      "doc_count": 89,
      "parent": null
    }
  ],
  "stages": {
    "stage1": ["summary_collect.json", "summary_ingest.json"],
    "stage2": ["summary_consolidate.json", "summary_budgeted_recall.json"],
    "stage3": ["summary.json", "summary.md", "summary_S2.md"]
  },
  "report": {"path": "/mnt/workspace/stage3/summary.md", "size_bytes": 47392}
}
```

---

## 5. MCP Tools

KOAS Summary provides 7 MCP tools for integration with Claude Desktop / Claude Code:

### `summary_ingest`

```python
async def summary_ingest(
    corpus_folder: str,
    workspace: Optional[str] = None,
    scope: str = "project",
    model: str = "ibm/granite4:32b-a9b-h",
    strategy: str = "pages",
) -> str
```

**Description:** Ingest a document corpus into memory.

**Returns:** `{"status": "ok", "workspace": "<path>"}` or error

---

### `summary_run`

```python
async def summary_run(
    corpus_folder: str,
    workspace: Optional[str] = None,
    scope: str = "project",
    model: str = "ibm/granite4:32b-a9b-h",
    language: str = "French",
    max_tokens: int = 12000,
) -> str
```

**Description:** Run full pipeline (Stage 1+2+3). Returns path to `summary.md`.

**Returns:** `{"status": "ok", "workspace": "<path>", "summary": "<md_path>", "size_bytes": N}`

---

### `summary_status`

```python
async def summary_status(workspace: str) -> str
```

**Description:** Check workspace status (memory stats, stage completion, report info).

**Returns:** JSON with memory stats, stage file lists, report metadata

---

### `summary_query`

```python
async def summary_query(
    workspace: str,
    query: str,
    tier: Optional[str] = None,
    type_filter: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = 20,
) -> str
```

**Description:** Full-text search across memory items.

**Returns:** `{"status": "ok", "query": "<q>", "count": N, "items": [...]}`

---

### `summary_drift`

```python
async def summary_drift(
    workspace_a: str,
    workspace_b: str,
) -> str
```

**Description:** Compute drift between two corpus versions.

**Returns:** `{"status": "ok", "report_path": "<path>", "summary": {"added": N, "removed": M, ...}}`

---

### `summary_viz`

```python
async def summary_viz(
    workspace: str,
    secrecy: str = "S3",
    views: Optional[str] = None,
) -> str
```

**Description:** Generate HTML visualizations. `views` is comma-separated (e.g., `"graph,memory"`).

**Returns:** `{"status": "ok", "secrecy": "S2", "files": {"graph": {"path": "...", "size_bytes": N}, ...}}`

---

### `summary_summarize`

```python
async def summary_summarize(
    workspace: str,
    model: str = "ibm/granite4:32b-a9b-h",
    language: str = "French",
    max_tokens: int = 12000,
    secrecy: str = "S3",
) -> str
```

**Description:** Generate summary from existing memory (Stage 2+3 only, skips ingestion). Useful for re-generating with different parameters.

**Returns:** `{"status": "ok", "workspace": "<path>", "summary": "<md_path>", "size_bytes": N}`

---

## 6. Visualization Outputs

KOAS Summary generates 8 interactive HTML views (all self-contained, no server required):

### 6.1 Graph View (`graph.html`)

**What it shows:**
- D3 force-directed layout of provenance graph
- Node types: `doc` (blue), `chunk` (gray), `item` (green), `entity` (orange)
- Edge types: `contains`, `adjacent`, `extracted_from`, `mentions`, `similar`
- Node size = degree centrality
- Hover: node label, neighbors
- Click: node details panel

**Generated by:** `export_graph_d3()` + `render_graph_view()`

**Secrecy impact:** Entity labels redacted at S0/S2

---

### 6.2 Memory Explorer (`memory.html`)

**What it shows:**
- Filterable table of all memory items
- Columns: ID, title, type, tier, domain, tags, entities, confidence, source, timestamps
- Filters: tier, type, domain, tags
- Search bar (client-side filter)
- Export to JSON/CSV

**Generated by:** `export_memory_items()` + `render_memory_explorer()`

**Secrecy impact:** Titles, tags, entities, source_id redacted at S0/S2

---

### 6.3 Query Explorer (`query.html`)

**What it shows:**
- Search-first interface with live results
- Domain dropdown (populated from memory items)
- Tier/type multi-select filters
- Full content preview (not just 200-char snippet)
- KPI cards: total items, tier distribution, type distribution

**Generated by:** `export_memory_query()` + `render_query_explorer()`

**Secrecy impact:** Same as Memory Explorer + full content redacted

---

### 6.4 Drift Report (`drift.html`)

**What it shows:**
- Sankey diagram: baseline → current (added/removed/modified flows)
- Summary table: added/removed/modified counts by domain
- Item-level diff view (click domain to expand)
- Drift percentage = (added + removed + modified) / total

**Generated by:** `load_drift_report()` + `format_drift_sankey()` + `render_drift_report()`

**Secrecy impact:** Item titles redacted at S0/S2

---

### 6.5 Timeline (`timeline.html`)

**What it shows:**
- Temporal slider (if `created_at` timestamps present)
- Bins items by month (configurable granularity)
- Stacked bar chart: tier distribution per bin
- Click bin to filter table below

**Generated by:** `build_timeline()` + `render_timeline()`

**Secrecy impact:** Content redacted, timestamps preserved

---

### 6.6 Topic Clusters (`clusters.html`)

**What it shows:**
- Community detection via Louvain algorithm on provenance graph
- Bubble chart: cluster size = item count, color = modularity score
- Top tags per cluster
- Click cluster to list member items

**Generated by:** `detect_communities()` + `render_cluster_view()`

**Secrecy impact:** Cluster labels (top tags) redacted at S0/S2

---

### 6.7 Drift Heatmap (`heatmap.html`)

**What it shows:**
- Domain (rows) × corpus version (columns) heatmap
- Cell color intensity = drift percentage
- Hover: added/removed/modified counts
- Requires multiple corpus_ids in same workspace or drift report

**Generated by:** `build_drift_heatmap()` + `render_heatmap()`

**Secrecy impact:** Domain names redacted at S0

---

### 6.8 Smart Metrics Dashboard (`metrics.html`)

**What it shows:**
- Coverage: items per domain, tier distribution
- Quality: confidence distribution, validation state counts
- Provenance: source_kind breakdown, doc vs. chat ratio
- Graph metrics: avg degree, clustering coefficient, modularity (if graph exists)
- Time series: items created over time

**Generated by:** `compute_metrics()` + `render_metrics_dashboard()`

**Secrecy impact:** Aggregate stats only, no sensitive fields

---

### Using `--views` for Selective Generation

By default, `summaryctl viz` generates all 8 views. For faster iteration or when certain views are not needed:

```bash
# Only graph and memory explorer
summaryctl viz workspace/ --views graph,memory

# Only drift report and heatmap
summaryctl viz workspace/ --views drift,heatmap

# All views (explicit)
summaryctl viz workspace/ --views graph,memory,query,drift,timeline,clusters,heatmap,metrics
```

**Performance savings:** Graph view (community detection) and clusters view (Louvain) are compute-heavy. Skipping them reduces viz generation time from ~45s to ~8s on a 1,500-item corpus.

---

## 7. Secrecy Model

KOAS Summary implements **report-time redaction** based on three tiers:

| Tier | Audience | What Gets Redacted | Use Case |
|------|----------|-------------------|----------|
| **S3** | Audit | Nothing (full detail) | Internal audits, compliance reviews |
| **S2** | Internal | Hostnames, IPs, hashes | Company-wide tech docs |
| **S0** | Public | Paths, emails, hostnames, IPs, filenames, hashes, entity labels | External publications, client reports |

### Redaction Patterns (S0 most restrictive)

| Pattern | Example Match | S0 | S2 | S3 |
|---------|---------------|----|----|-----|
| **paths** | `/etc/nginx/nginx.conf`, `C:\Windows\System32` | ✅ REDACT | ❌ keep | ❌ keep |
| **emails** | `admin@corp.internal` | ✅ REDACT | ❌ keep | ❌ keep |
| **hostnames** | `db-server.corp.internal` | ✅ REDACT | ✅ REDACT | ❌ keep |
| **ips** | `192.168.1.10` | ✅ REDACT | ✅ REDACT | ❌ keep |
| **filenames** | `report.pdf`, `config.yaml` | ✅ REDACT | ❌ keep | ❌ keep |
| **pointer_ids** | `MEM-abc123def456` | ✅ REDACT | ❌ keep | ❌ keep |
| **hashes** | `sha256:a1b2c3d4...` | ✅ REDACT | ✅ REDACT | ❌ keep |
| **entity_labels** | `srv-db-01`, `app.internal` | ✅ REDACT | ❌ keep | ❌ keep |

### Replacement Tokens

When a pattern matches, it's replaced with a generic placeholder:

- `[PATH]` for file paths
- `[EMAIL]` for email addresses
- `[HOST]` for internal hostnames
- `[IP]` for IP addresses
- `[FILE]` for filenames
- `[MEM-ID]` for memory item IDs
- `[HASH]` for content hashes
- `[ENTITY]` for infrastructure entities

### How Redaction Works

1. **Write-time redaction** (optional): `redact_for_storage()` can be called before writing to memory store (not default — audit tier S3 stores full content)
2. **Report-time redaction** (mandatory): `summary_redact` kernel runs after `summary_report` (Stage 3) and produces:
   - `summary_S0.md` (public)
   - `summary_S2.md` (internal)
   - `redaction_report.json` (counts per pattern)
3. **Visualization redaction**: `export_memory_*()` functions apply `_redact_field()` based on `secrecy_tier` parameter

### Configuration

In `config` dict passed to kernels:

```python
config = {
    "secrecy": {
        "tier": "S2",  # S0, S2, or S3
        "custom_patterns": [
            r"\bPROJECT-\d{4}\b",  # Custom regex for project codes
        ],
    },
}
```

### Provenance Preservation

**Critical:** Redaction NEVER touches provenance fields (`source_id`, `chunk_ids`, `content_hashes`). These remain intact even at S0 to ensure audit trails and reproducibility.

---

## 8. Memory Subsystem Integration

KOAS Summary is built on top of the **RAGIX Memory Subsystem** (`ragix_core/memory/`), a persistent, policy-governed, LLM-proposed (but not LLM-committed) memory store.

### Memory Data Model

#### MemoryItem Schema

```python
@dataclass
class MemoryItem:
    id: str                          # MEM-abc123def456
    tier: MemoryTier                 # "stm" | "mtm" | "ltm"
    type: MemoryType                 # "fact" | "constraint" | "definition" | ...
    title: str                       # Short heading
    content: str                     # Main text (concise; use pointers for long evidence)
    tags: List[str]                  # ["rhel", "nginx", "cve"]
    entities: List[str]              # ["CVE-2024-1234", "RHEL 8.6", "nginx 1.20.2"]
    links: List[Dict[str, str]]      # [{"rel": "contradicts", "to": "MEM-xyz"}]
    provenance: MemoryProvenance     # source_kind, source_id, chunk_ids, content_hashes
    confidence: float                # 0.0 – 1.0
    validation: ValidationState      # "unverified" | "verified" | "contested" | "retracted"
    scope: str                       # "project" (multi-tenancy namespace)
    corpus_id: Optional[str]         # "corp_energy-rie-2026Q1" (cross-corpus tracking)
    rule_id: Optional[str]           # "RIE-RHEL-042" (domain-specific ID)
    usage_count: int                 # Incremented on each recall
    superseded_by: Optional[str]     # MEM-newid (if merged during consolidation)
    archived: bool                   # True if superseded
    created_at: str                  # ISO-8601 timestamp
    updated_at: str                  # ISO-8601 timestamp
```

#### MemoryProvenance Schema

```python
@dataclass
class MemoryProvenance:
    source_kind: SourceKind          # "doc" | "chat" | "tool" | "mixed"
    source_id: str                   # "corp_energy-rie-rhel-8.6-hardening.pdf"
    chunk_ids: List[str]             # ["CHUNK-001", "CHUNK-002"]
    content_hashes: List[str]        # ["sha256:a1b2c3...", "sha256:d4e5f6..."]
    created_at: str                  # ISO-8601 timestamp
```

### Memory Tiers (STM → MTM → LTM)

| Tier | Name | Purpose | Promotion Rule |
|------|------|---------|----------------|
| **STM** | Short-Term Memory | Initial extraction, unverified | `usage_count >= 3` → MTM |
| **MTM** | Medium-Term Memory | Verified, merged, consolidated | `usage_count >= 7` → LTM |
| **LTM** | Long-Term Memory | High-confidence, frequently used | Never demoted |

**Why tiers?**
- Prevents pollution: Low-quality extractions stay in STM until proven useful
- Adaptive consolidation: Only STM items are clustered/merged by default
- Performance: LTM items are cached, avoiding redundant embedding calls

### Memory Types (8 kinds)

| Type | Example | Use Case |
|------|---------|----------|
| **fact** | "CVE-2024-1234 affects nginx versions 1.18.0-1.19.9" | Verifiable statements |
| **constraint** | "MUST upgrade nginx to 1.20.2 or later" | Compliance requirements (MUST/SHALL/SHOULD) |
| **definition** | "RHEL 8.6 = Red Hat Enterprise Linux version 8.6" | Glossary terms |
| **decision** | "Team approved migration to Kubernetes 1.28" | Historical choices |
| **pattern** | "Always use TLS 1.3 for external APIs" | Recurring best practices |
| **todo** | "Audit OpenSSL dependencies by Q2 2026" | Action items |
| **pointer** | "See section 4.3 in CORP-ENERGY-RIE-RHEL-042" | Cross-references (avoids duplicating long content) |
| **note** | "Reviewer noted potential conflict with Oracle policy" | Annotations |

### SQLite Schema (6 Tables)

#### `memory_items` (canonical state)
- **Primary key:** `id` (TEXT)
- **Indexes:** `tier`, `type`, `scope`, `archived`, `corpus_id`
- **Triggers:** 4 FTS5 sync triggers (AFTER INSERT, BEFORE/AFTER DELETE, BEFORE/AFTER UPDATE)

#### `memory_revisions` (append-only history)
- **Primary key:** `revision_id` (TEXT)
- **Fields:** `item_id`, `revision_num`, `snapshot` (full JSON), `changed_at`, `reason`

#### `memory_embeddings` (vector cache)
- **Primary key:** `item_id`
- **Fields:** `model_name`, `dimension`, `vector` (BLOB, float32 packed bytes), `created_at`

#### `memory_links` (typed relationships)
- **Primary key:** `(src_id, dst_id, rel)`
- **Edge types:** `contradicts`, `replaces`, `supports`, `references`, `extracted_from`, `similar`

#### `memory_palace_locations` (Method of Loci index)
- **Primary key:** `item_id`
- **Fields:** `domain`, `room`, `shelf`, `card`, `assigned_at`

#### `memory_events` (audit log)
- **Primary key:** `id` (TEXT)
- **Fields:** `action`, `item_id`, `details_json`, `content_hash`, `timestamp`

#### `corpus_hashes` (delta mode tracking)
- **Primary key:** `file_path`
- **Fields:** `sha256`, `chunk_count`, `item_ids` (JSON array), `ingested_at`

#### `corpus_metadata` (cross-corpus operations)
- **Primary key:** `corpus_id`
- **Fields:** `corpus_label`, `parent_corpus_id`, `doc_count`, `item_count`, `scope`, `ingested_at`

### FTS5 Full-Text Search

```sql
CREATE VIRTUAL TABLE memory_items_fts USING fts5(
    title, content, tags, entities,
    content='memory_items',
    content_rowid='rowid'
);
```

**Features:**
- **External-content mode**: No duplicate data (mirrors `memory_items` table)
- **Automatic sync**: 4 triggers keep FTS index in sync
- **MATCH syntax**: `title MATCH 'CVE nginx' OR tags MATCH 'rhel'`
- **Rank**: BM25 scoring built-in

**Performance:**
- **LIKE**: O(n) sequential scan, 2.3s for 1,500 items
- **FTS5 MATCH**: O(log n) indexed lookup, 0.02s for 1,500 items (100× faster)

### Policy Governance

Before writing to memory store, all items pass through `policy.py`:

#### Hard Blocks (rejected, not stored)
1. **Secrets** (9 patterns):
   - API keys: `Authorization: Bearer ...`, `api_key=...`
   - Credentials: `password=...`, `token=...`
   - AWS keys: `AKIA...`, `aws_secret_access_key=...`
   - SSH keys: `-----BEGIN PRIVATE KEY-----`
   - Connection strings: `postgresql://user:pass@...`
2. **Injection attempts** (8 patterns):
   - Prompt injection: `Ignore previous instructions`
   - XML/HTML tags: `<script>`, `<?php`
   - SQL injection: `'; DROP TABLE`
3. **Oversized content**: `len(content) > 4000` characters (use `type="pointer"` instead)

#### Soft Blocks (quarantined to STM, flagged)
- Low confidence: `confidence < 0.3`
- No provenance: `source_id == ""` and `chunk_ids == []`
- Duplicate content: `content_hash` already exists in store

### Memory CLI

```bash
# Search items
python -m ragix_core.memory.cli search "CVE nginx" --tier mtm -k 10

# Show item details
python -m ragix_core.memory.cli show MEM-abc123def456

# Store statistics
python -m ragix_core.memory.cli stats /path/to/memory.db

# Run consolidation manually
python -m ragix_core.memory.cli consolidate /path/to/memory.db --scope project

# Export/import (JSON format)
python -m ragix_core.memory.cli export /path/to/memory.db --output dump.json
python -m ragix_core.memory.cli import /path/to/memory.db --input dump.json

# Browse memory palace
python -m ragix_core.memory.cli palace /path/to/memory.db --domain rhel
```

---

## 9. Graph-RAG

### What is Graph-RAG in KOAS Summary?

**Graph-RAG** = Graph-assisted Retrieval-Augmented Generation. The provenance graph **constrains consolidation** so that only items sharing structural neighborhoods can be merged. This prevents:

- **Cross-corpus mega-chains**: Without graph constraints, embeddings alone can merge "RHEL firewall" from corpus A with "RHEL firewall" from corpus B, creating a single mega-item that loses version identity.
- **Semantic drift**: Items from different document contexts (same words, different meanings) stay separated.

**Key insight:** The graph is NOT a retrieval platform (memory store + FTS5 handle search). It's a **merge filter** during consolidation.

### Graph Schema

#### Node Types (4 kinds)

| Kind | Label Format | Example | Created By |
|------|--------------|---------|------------|
| `doc` | `source_id` | `corp_energy-rie-rhel-8.6.pdf` | `summary_collect` |
| `chunk` | `CHUNK-NNN` | `CHUNK-042` | `summary_ingest` |
| `item` | `MEM-abc123...` | `MEM-4f8a2b1c9d3e` | `summary_ingest` |
| `entity` | Controlled vocabulary | `CVE-2024-1234`, `nginx`, `RHEL 8.6` | `summary_extract_entities` |

#### Edge Types (5 kinds)

| Kind | Direction | Weight | Meaning |
|------|-----------|--------|---------|
| `contains` | doc → chunk | 1.0 | Document contains chunk |
| `adjacent` | chunk → chunk | 1.0 | Consecutive chunks (same doc) |
| `extracted_from` | item → chunk | 1.0 | Item extracted from chunk |
| `mentions` | item → entity | 1.0 | Item mentions entity |
| `similar` | item ↔ item | 0.0–1.0 | Embedding cosine similarity |

### Entity Vocabulary Filter

**Problem:** LLMs extract free-text garbage ("the", "use", "system") as entities.

**Solution:** Only entities matching controlled vocabulary create graph nodes.

**Patterns accepted:**
1. **CVE IDs**: `CVE-2024-1234` (regex: `CVE-\d{4}-\d+`)
2. **Version tokens**: `8.6`, `1.20.2a` (regex: `\d+(\.\d+){1,3}[a-z]?`)
3. **Review IDs**: `RVW-042` (regex: `RVW-\d+`)
4. **Memory IDs**: `MEM-4f8a2b1c9d3e` (regex: `MEM-[0-9a-f]+`)
5. **Compliance markers**: `MUST`, `SHALL`, `SHOULD`, `PROHIBITED`, `REQUIRED`
6. **Config paths**: `/etc/nginx/nginx.conf`, `*.yaml`, `*.conf`
7. **Port numbers**: `443/tcp`, `3306/udp`
8. **Product names** (150+ vocabulary): `rhel`, `nginx`, `kubernetes`, `postgresql`, etc.

**Rejection examples:** `the`, `system`, `use`, `configuration`, `server` (too generic)

### Graph Build Performance

**Benchmark** (CORP-ENERGY-RIE corpus, 89 PDFs, 1,247 memory items):

| Step | Time | Nodes | Edges |
|------|------|-------|-------|
| Extract entities (deterministic NER) | 0.8s | 342 entities | 0 |
| Build graph (doc/chunk/item nodes) | 1.2s | 1,589 total | 2,847 |
| Add `similar` edges (embedding) | 23.4s | — | +412 |
| **Total** | **25.4s** | **1,589** | **3,259** |

**Rebuild policy:** Graph is cheap to rebuild (all edges are deterministic except `similar`). Typical workflow:
1. First run: build full graph
2. Delta mode: add new doc/chunk/item nodes, recompute `similar` edges only for new items
3. Force rebuild: `summaryctl summarize --no-graph` then re-enable graph

### How Graph Constrains Consolidation

**Without graph (embedding-only clustering):**
```
Cluster: "RHEL firewall rules"
  - MEM-001 from corp_energy-rie-2026Q1.pdf (RHEL 8.6 iptables)
  - MEM-234 from corp_energy-rie-2026Q2.pdf (RHEL 9.0 nftables)
  → Merged into mega-item, loses version distinction
```

**With graph constraint:**
```
Cluster candidates: MEM-001, MEM-234 (cosine similarity 0.89)
  → Check 2-hop neighborhood:
    MEM-001 → CHUNK-042 → corp_energy-rie-2026Q1.pdf
    MEM-234 → CHUNK-198 → corp_energy-rie-2026Q2.pdf
  → No shared provenance → separate clusters
  → MEM-001 stays in Q1 cluster, MEM-234 in Q2 cluster
```

**2-hop neighborhood** = items can merge if they share:
- Same document (both → same CHUNK → same DOC)
- Adjacent chunks (both → adjacent CHUNKs → same DOC)
- Same entity mention (both → same ENTITY)

### Graph-Assisted Consolidation Algorithm

```python
def consolidate_with_graph(items, graph_store):
    # Step 1: Coarse clustering by (type, tags)
    buckets = group_by_type_and_tags(items)

    # Step 2: Fine clustering within buckets
    clusters = []
    for bucket in buckets:
        # Compute pairwise embedding similarity
        sim_matrix = compute_similarity_matrix(bucket)

        # Filter by graph constraint
        for i, j in high_similarity_pairs(sim_matrix, threshold=0.75):
            if graph_store.share_2hop_neighborhood(items[i].id, items[j].id):
                merge_into_cluster(items[i], items[j])
            else:
                # Similar embeddings but different provenance → keep separate
                pass

    # Step 3: Merge each cluster via LLM
    for cluster in clusters:
        merged = llm_summarize(cluster)  # Granite 3.2B
        store.write(merged, supersede=cluster)
```

### Disabling Graph (Performance Trade-off)

```bash
# Disable graph during summarize
summaryctl summarize workspace/ --no-graph

# Enable graph (default)
summaryctl summarize workspace/ --graph
```

**When to disable:**
- Single-corpus, single-version projects (no cross-corpus risk)
- Embedding model already domain-tuned (low false-positive merge rate)
- Speed > precision (graph build adds 20–30s overhead)

**When to enable:**
- Multi-corpus projects (Q1 vs. Q2 releases)
- Generic embedding models (nomic-embed-text, all-MiniLM-L6-v2)
- Audit-grade provenance tracking required

---

## 10. Drift Detection

### What is Drift?

**Drift** = changes in memory items between two corpus versions. Measured as:

$$
\text{Drift} = \frac{\text{Added} + \text{Removed} + \text{Modified}}{\text{Total Baseline Items}} \times 100\%
$$

### Use Cases

1. **Policy evolution**: Detect when "SHOULD" became "MUST" between Q1 and Q2 compliance docs
2. **Version tracking**: Identify new CVEs introduced in latest RHEL release
3. **Documentation staleness**: Find removed items (deleted from docs but still referenced elsewhere)

### Two Modes

#### Mode 1: Same-DB (two corpus_ids)

**Setup:**
```bash
# Ingest baseline
summaryctl ingest baseline_docs/ --corpus-id baseline-2026Q1

# Ingest current (same workspace)
summaryctl ingest current_docs/ -w summary_workspace/ --corpus-id current-2026Q2
```

**Run drift:**
```bash
summaryctl drift drift_output/ \
  --corpus-a baseline-2026Q1 \
  --corpus-b current-2026Q2
```

#### Mode 2: Two-Workspace (separate DBs)

**Setup:**
```bash
# Baseline workspace
summaryctl run baseline_docs/ -w baseline_ws/

# Current workspace
summaryctl run current_docs/ -w current_ws/
```

**Run drift:**
```bash
summaryctl drift drift_output/ \
  --workspace-a baseline_ws/ \
  --workspace-b current_ws/
```

### Drift Algorithm

```python
def compute_drift(store_a, store_b):
    items_a = {item.content_hash: item for item in store_a.list_items()}
    items_b = {item.content_hash: item for item in store_b.list_items()}

    hashes_a = set(items_a.keys())
    hashes_b = set(items_b.keys())

    added = hashes_b - hashes_a           # In B, not in A
    removed = hashes_a - hashes_b         # In A, not in B
    unchanged = hashes_a & hashes_b       # In both with same hash

    # Modified = same item ID but different content hash
    modified = []
    for hash_b in hashes_b:
        item_b = items_b[hash_b]
        if item_b.id in items_a and items_a[item_b.id].content_hash != hash_b:
            modified.append(item_b.id)

    drift_pct = (len(added) + len(removed) + len(modified)) / len(hashes_a) * 100

    return {
        "added": len(added),
        "removed": len(removed),
        "modified": len(modified),
        "unchanged": len(unchanged),
        "drift_pct": round(drift_pct, 1),
    }
```

### Drift Report Structure

**JSON output** (`drift_report.json`):
```json
{
  "baseline": {
    "corpus_id": "baseline-2026Q1",
    "total_items": 487
  },
  "current": {
    "corpus_id": "current-2026Q2",
    "total_items": 512
  },
  "diff": {
    "added": 38,
    "removed": 13,
    "modified": 19,
    "unchanged": 455,
    "drift_pct": 14.4
  },
  "details": {
    "added_items": [
      {"id": "MEM-abc123", "title": "CVE-2024-9999 nginx", "domain": "rhel"},
      ...
    ],
    "removed_items": [...],
    "modified_items": [
      {
        "id": "MEM-def456",
        "title": "Firewall rule compliance",
        "baseline_content": "SHOULD close port 8080",
        "current_content": "MUST close port 8080",
        "domain": "rhel"
      },
      ...
    ]
  }
}
```

### Drift Visualization

**Sankey diagram** (`viz/drift.html`):
- Left column: baseline domains (RHEL, Oracle, Kubernetes, ...)
- Right column: current domains
- Flows: added (green), removed (red), modified (orange), unchanged (gray)
- Width = item count

**Heatmap** (`viz/heatmap.html`):
- Rows: domains
- Columns: corpus versions (baseline, current, future, ...)
- Cell color: drift intensity (white = 0%, red = 100%)
- Hover: added/removed/modified counts

---

## 11. Configuration

### Kernel Config Structure

All KOAS Summary kernels accept a `config` dict in `KernelInput`. Key sections:

#### `input_folder` / `scope` / `model`
```python
config = {
    "input_folder": "/data/corp_energy-rie-docs",
    "scope": "corp_energy-rie",
    "model": "ibm/granite4:32b-a9b-h",
}
```

#### `strategy` / `max_chunk_tokens`
```python
config = {
    "strategy": "pages",           # "pages" | "headings" | "windows"
    "max_chunk_tokens": 800,        # Max tokens per chunk
}
```

#### `db_path` / `embedder_backend` / `ollama_url`
```python
config = {
    "db_path": "/workspace/memory.db",
    "embedder_backend": "mock",     # "mock" | "ollama" | "sentence-transformers"
    "embedder_model": "nomic-embed-text",
    "ollama_url": "http://localhost:11434",
}
```

#### `graph`
```python
config = {
    "graph": {
        "enabled": True,                     # Enable graph-assisted consolidation
        "similarity_threshold": 0.75,        # Cosine similarity for "similar" edges
        "neighbor_hops": 2,                  # K-hop neighborhood for merge constraint
    },
}
```

#### `secrecy`
```python
config = {
    "secrecy": {
        "tier": "S2",                        # "S0" | "S2" | "S3"
        "custom_patterns": [
            r"\bPROJECT-\d{4}\b",            # Custom regex patterns to redact
        ],
    },
}
```

#### `max_inject_tokens` / `min_items_per_domain` / `max_items_per_domain`
```python
config = {
    "max_inject_tokens": 12000,              # Total budget for memory injection
    "min_items_per_domain": 3,               # Min items per domain (pad if needed)
    "max_items_per_domain": 25,              # Cap items per domain
}
```

#### `delta` / `corpus_id` / `new_item_ids`
```python
config = {
    "delta": True,                           # Delta mode (only new/modified files)
    "corpus_id": "corp_energy-rie-2026Q2",          # Corpus version ID
    "new_item_ids": ["MEM-abc", "MEM-def"],  # V3.0: for neighborhood consolidation
}
```

#### `language`
```python
config = {
    "language": "French",                    # Output language for summary generation
}
```

### Environment Variables

```bash
# Ollama endpoint
export OLLAMA_URL=http://localhost:11434

# Default embedding model
export EMBEDDER_MODEL=nomic-embed-text

# Workspace root
export SUMMARY_WORKSPACE=/mnt/summary_workspace
```

---

## 12. Performance

### Benchmarks (CORP-ENERGY-RIE Corpus)

**System:** Ubuntu 24.04, AMD Ryzen 9 7950X (16C/32T), 64 GB RAM, Ollama local

**Corpus:** 89 PDF files, 1,247 memory items, 342 entities, 1,589 graph nodes

| Stage | Kernel | Time | Throughput | Notes |
|-------|--------|------|-----------|-------|
| **S1** | `summary_collect` | 2.1s | 42 files/s | File enumeration |
| **S1** | `summary_ingest` | 12m 34s | 7 files/min | LLM extraction (Granite 32B) |
| **S2** | `summary_extract_entities` | 0.8s | 1,559 items/s | Deterministic NER |
| **S2** | `summary_build_graph` | 25.4s | — | 1,589 nodes, 3,259 edges |
| **S2** | `summary_consolidate` | 4m 18s | — | 89 clusters, 127 merges |
| **S2** | `summary_budgeted_recall` | 1.2s | — | FTS5 search + scoring |
| **S3** | `summary_generate` | 8m 42s | — | 7 domains, avg 1m 15s/domain |
| **S3** | `summary_verify` | 0.3s | — | Citation validation |
| **S3** | `summary_report` | 0.1s | — | Markdown assembly |
| **S3** | `summary_redact` | 0.2s | — | 147 redactions (S2) |
| **Viz** | `render_all` (8 views) | 42.3s | — | Graph view: 28s, others: 14s |
| **Total** | **Full pipeline** | **26m 50s** | — | Ingest → summarize → viz |

### FTS5 vs LIKE Performance

**Query:** `"CVE nginx RHEL"` on 1,247 items

| Method | Time | Speedup |
|--------|------|---------|
| `LIKE '%CVE%' AND LIKE '%nginx%'` (3 columns) | 2.34s | 1× |
| `FTS5 MATCH 'CVE nginx RHEL'` | 0.021s | **111×** |

**Key takeaway:** FTS5 is essential for corpora > 500 items. Below that, LIKE is acceptable.

### FAISS Acceleration

**V3.1 feature:** Optional FAISS index for embedding search.

**Benchmark** (1,247 items, 384-dim embeddings, k=50):

| Backend | Time | Speedup |
|---------|------|---------|
| Sequential cosine scan | 0.87s | 1× |
| FAISS IndexFlatIP | 0.042s | **21×** |

**Setup:**
```bash
pip install faiss-cpu
```

No code changes needed — `RecallEngine` auto-detects FAISS and builds index on first search.

**Trade-off:** Index build time = 0.12s (amortized over multiple searches).

### Visualization Selection

**Default** (all 8 views): 42.3s

**Selective** (`--views graph,memory,query`): 8.1s (5× faster)

**Bottleneck view:** Graph view (community detection via Louvain) = 28s. Skip if not needed.

### Delta Mode Performance

**Full ingestion** (89 files): 12m 34s

**Delta ingestion** (3 new files, 2 modified): 1m 47s (7× faster)

**Delta consolidation** (neighborhood-scoped, 15 new items): 38s vs. 4m 18s full (6.8× faster)

**Use case:** Monthly updates to corpus (95% unchanged files).

---

## 13. File Layout

### Workspace Directory Structure

```
summary_workspace/
├── memory.db                    # SQLite database (6 tables + FTS5)
├── graph.db                     # Graph store (shares SQLite with memory.db)
├── stage1/
│   ├── summary_collect.json     # File inventory
│   └── summary_ingest.json      # Ingestion stats
├── stage2/
│   ├── summary_extract_entities.json   # Entity extraction stats
│   ├── summary_build_graph.json         # Graph build stats
│   ├── summary_consolidate.json         # Merge chains
│   └── summary_budgeted_recall.json     # Selected items per domain
├── stage3/
│   ├── summary.md                       # Full markdown report (S3)
│   ├── summary.json                     # Structured JSON
│   ├── summary_S2.md                    # Redacted (internal)
│   ├── summary_S0.md                    # Redacted (public)
│   ├── redaction_report.json            # Redaction counts
│   └── capabilities.json                # Model/domain manifest
└── viz/
    ├── graph.html                       # Interactive graph view
    ├── memory.html                      # Memory explorer
    ├── query.html                       # Query explorer
    ├── drift.html                       # Drift report (if computed)
    ├── timeline.html                    # Timeline slider
    ├── clusters.html                    # Topic clusters
    ├── heatmap.html                     # Drift heatmap
    └── metrics.html                     # Smart metrics dashboard
```

### `memory.db` Schema Details

**Tables:**
1. `memory_items` — 1,247 rows (canonical state)
2. `memory_revisions` — 3,891 rows (full history)
3. `memory_embeddings` — 1,247 rows (384-dim vectors)
4. `memory_links` — 89 rows (typed relationships)
5. `memory_palace_locations` — 1,247 rows (domain/room/shelf/card)
6. `memory_events` — 3,891 rows (audit log)
7. `corpus_hashes` — 89 rows (delta mode tracking)
8. `corpus_metadata` — 1 row (corpus version metadata)

**Virtual table:**
- `memory_items_fts` — FTS5 index (title, content, tags, entities)

**DB size:** ~42 MB for 1,247 items (avg 34 KB/item including vectors)

---

## 14. Troubleshooting

### `No memory.db found`

**Symptom:** `summaryctl summarize` fails with "No memory.db in workspace"

**Solution:** Run `summaryctl ingest` first:
```bash
summaryctl ingest /path/to/corpus -v
```

---

### `Ollama connection refused`

**Symptom:** `ConnectionRefusedError: [Errno 111] Connection refused`

**Solution:** Start Ollama server:
```bash
ollama serve
```

Or specify custom URL:
```bash
summaryctl ingest /path/to/corpus --ollama-url http://192.168.1.100:11434
```

---

### `FTS5 extension not available`

**Symptom:** `sqlite3.OperationalError: no such module: fts5`

**Solution:** Rebuild SQLite with FTS5 support:
```bash
# Ubuntu/Debian
sudo apt install sqlite3 libsqlite3-dev

# macOS
brew reinstall sqlite
```

Or use LIKE fallback (set `use_fts5=False` in store.py, not recommended).

---

### `FAISS not found (falling back to sequential scan)`

**Symptom:** Log message: `faiss-cpu not installed — using sequential cosine scan`

**Solution:** Install FAISS (optional):
```bash
pip install faiss-cpu
```

No code changes needed — RecallEngine auto-detects.

---

### `Graph build too slow`

**Symptom:** `summary_build_graph` takes > 5 minutes

**Solution:** Disable graph:
```bash
summaryctl summarize workspace/ --no-graph
```

Or upgrade embedding model to faster variant (e.g., `all-MiniLM-L6-v2` instead of `nomic-embed-text`).

---

### `Out of memory during consolidation`

**Symptom:** Python process killed (OOM) during `summary_consolidate`

**Solution:** Reduce batch size in config:
```python
config = {
    "consolidate": {
        "max_cluster_size": 10,      # Reduce from default 25
        "embedding_batch_size": 50,  # Reduce from default 100
    },
}
```

Or run consolidation in stages:
```bash
# Consolidate STM only
summaryctl summarize workspace/ --skip-consolidation
# Then manually run consolidate kernel
```

---

### `Broken citations in summary.md`

**Symptom:** `summary_verify` reports missing citations `[MEM-xyz]`

**Cause:** LLM hallucinated memory IDs that don't exist in store

**Solution:**
1. Check verification report: `stage3/summary_verify.json`
2. Re-run generation with stricter prompt (add "cite only from injected memory")
3. Manual fix: search for actual item with similar content, replace citation

---

### `Drift report empty`

**Symptom:** `drift_report.json` shows `"added": 0, "removed": 0, "modified": 0`

**Cause:** Both corpora have identical content hashes

**Solution:** Verify corpus_ids are different:
```bash
summaryctl show baseline_ws/ --format json | jq '.corpora[].corpus_id'
summaryctl show current_ws/ --format json | jq '.corpora[].corpus_id'
```

If corpus_ids are missing, re-ingest with `--corpus-id` flag.

---

### `Visualization JS errors in browser`

**Symptom:** `viz/graph.html` shows blank page, console error: `ReferenceError: d3 is not defined`

**Cause:** CDN blocked (offline environment)

**Solution:** Templates use CDN by default. For offline use, download D3.js locally:
```bash
cd ragix_kernels/summary/visualization/templates
wget https://d3js.org/d3.v7.min.js
```

Edit template to use local path:
```html
<script src="d3.v7.min.js"></script>
```

---

**End of KOAS_SUMMARY.md**

For further assistance:
- **GitHub Issues:** https://github.com/ovitrac/RAGIX/issues
- **Email:** olivier.vitrac@adservio.fr
- **Documentation:** `/docs/` directory in RAGIX repository
