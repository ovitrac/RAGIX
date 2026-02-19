# CHANGELOG — RAGIX

All notable changes to the **RAGIX** project will be documented here.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## v0.69.0 — Bounded Recall-Answer Loop & Fixed-Point Convergence (2026-02-19)

### Highlights

**RAGIX v0.69.0 implements a bounded recall-answer loop with fixed-point convergence detection, two-tier text similarity (lexical + cosine), and a deterministic mock-LLM demo proving algorithm correctness — completing the full ROADMAP_LOOP.md (phases R1–R5).**

| Feature | Status |
|---------|--------|
| Text similarity module (Tier A lexical + Tier B cosine) | ✅ `similarity.py` — 160 LOC |
| Bounded recall-answer loop controller | ✅ `loop.py` — 310 LOC, 5 stopping conditions |
| `ragix-memory loop` CLI subcommand (16 flags) | ✅ 16 total subcommands |
| Act 9a — Deterministic mock LLM demo (JSON protocol) | ✅ Always runs, no LLM dependency |
| Act 9b — Live LLM loop demo (passive protocol) | ✅ Real model with safety bounds |
| 106 new tests (similarity + loop + interop) | ✅ 43 + 33 + 30 tests |
| Passive mode fix (no protocol injection) | ✅ `include_protocol` parameter |
| MockEmbedder removal from production CLI | ✅ Auto mode falls back to lexical |

### New Module: `ragix_core/memory/similarity.py`

Two-tier text similarity for fixed-point convergence detection:

```python
from ragix_core.memory.similarity import compute_similarity

# Tier A — lexical (no dependencies, always available)
score = compute_similarity("answer v1", "answer v2", method="lexical")
# → 0.5 * Jaccard(word_sets) + 0.5 * SequenceMatcher(texts)

# Tier B — cosine (requires embedder)
score = compute_similarity("answer v1", "answer v2", method="cosine", embedder=emb)

# Auto mode — uses cosine if embedder available, else lexical
score = compute_similarity("answer v1", "answer v2", method="auto")
```

**Key design decisions:**
- Tier A lexical uses `0.5 * Jaccard + 0.5 * SequenceMatcher` — no external dependencies
- Tier B cosine reuses existing `cosine_similarity()` from `embedder.py` — no duplication
- `auto` mode gracefully degrades: cosine → lexical if embedder is None or fails
- Edge cases handled: empty strings → 0.0, identical strings → 1.0

### New Module: `ragix_core/memory/loop.py`

Bounded recall-answer loop controller with 5 stopping conditions:

| Condition | Description |
|-----------|-------------|
| `llm_stop` | LLM signals `stop: true` in JSON protocol output |
| `fixed_point` | Consecutive answers exceed similarity threshold |
| `query_cycle` | LLM re-requests a previously tried query |
| `no_new_items` | Recall returns nothing new |
| `max_calls` | Iteration budget exhausted (safety bound) |

**Core components:**
- `LoopConfig` — dataclass with 16 configurable parameters (max_calls, threshold, protocol, similarity method, etc.)
- `LoopIteration` — per-iteration trace record (query, answer, similarity, stop reason)
- `LoopResult` — final result with answer, iterations list, stop reason, total timing
- `invoke_llm()` — subprocess-based LLM invocation with stdin piping and `include_protocol` control
- `parse_protocol_output()` — extracts JSON-first-line protocol signals from LLM output
- `run_loop()` — main controller orchestrating recall → LLM → parse → similarity → decide

**Two protocol modes:**
- `json` — LLM must emit JSON first line with `need_more`, `query`, `stop` fields; loop parses and acts on signals
- `passive` — LLM answers freely; loop monitors fixed-point convergence only (no protocol prompt injected)

### New CLI Subcommand: `ragix-memory loop`

```bash
# JSON protocol with mock LLM (deterministic)
ragix-memory --db $DB pipe "topic" --budget 3000 \
    | ragix-memory --db $DB loop --llm "bash mock_llm.sh" \
        --protocol json --max-calls 5 --threshold 0.92 \
        --trace-file trace.jsonl

# Passive protocol with real LLM
ragix-memory --db $DB pipe "topic" --budget 3000 \
    | ragix-memory --db $DB loop --llm "ollama run granite3.1-moe:3b" \
        --protocol passive --max-calls 3 --threshold 0.85 \
        --similarity lexical --no-recall --trace-file trace.jsonl
```

**Flags:** `--llm`, `--protocol`, `--max-calls`, `--threshold`, `--similarity`, `--no-recall`, `--trace-file`, `--system-prompt`, `--llm-mode`, `--budget`

### Demo: Act 9 Split (9a + 9b)

Act 9 restructured into two complementary sub-acts:

| Sub-act | LLM | Protocol | Runs when | Proves |
|---------|-----|----------|-----------|--------|
| 9a | `mock_llm.sh` (bash) | `json` | Always (even `--no-llm`) | Algorithm correctness, deterministic |
| 9b | Ollama/Claude (real) | `passive` | LLM available | Real-world convergence behavior |

**New file: `demos/ragix_memory_demo/mock_llm.sh`** — Stateful bash script (state counter in `/tmp/ragix_mock_llm_state`):
- Iteration 1: requests refinement on "memory consolidation tier promotion"
- Iteration 2: requests refinement on "injection block format token budget pipe"
- Iteration 3+: signals `stop: true`, emits comprehensive 5-section analysis

**Convergence behavior:**
- Mock LLM (9a): `llm_stop` after 3 iterations — deterministic, reproducible
- Small real LLMs (9b): `max_calls` safety bound — lexical similarity ~0.12–0.37 at default temperature is well below convergence thresholds; larger models or lower temperature achieve tighter convergence

### Bug Fixes

- **Passive mode protocol injection** — `invoke_llm()` always prepended `LOOP_PROTOCOL_PROMPT`, causing LLMs in passive mode to emit JSON protocol headers as their answer. Fixed with `include_protocol` parameter set to `False` when `config.protocol == "passive"`.
- **MockEmbedder in production CLI** — `cmd_loop()` used `MockEmbedder(dimension=768, seed=42)` which produces random vectors, making cosine similarity scores meaningless (~0.01–0.08). Removed; `embedder=None` lets auto mode fall back to lexical similarity.
- **Demo logging scope** — Logging functions (`recall_context`, `log_input`, `log_output`) were defined inside Act 8's `NO_LLM` guard, making them unavailable to Act 9a. Hoisted above the guard.

### Test Coverage

| Test file | Tests | Covers |
|-----------|-------|--------|
| `test_similarity.py` | 43 | Tier A/B similarity, edge cases, auto mode, error handling |
| `test_loop.py` | 33 | LoopConfig, parse_protocol_output, run_loop, all 5 stop conditions |
| `test_interop_protocol.py` | 30 | JSON protocol parsing, passive mode, CLI integration, trace files |
| **Total new** | **106** | Full coverage of R1–R5 ROADMAP_LOOP.md |

### Integration Points

| File | Changes |
|------|---------|
| `ragix_core/memory/similarity.py` | **NEW** — 160 lines, Tier A/B text similarity |
| `ragix_core/memory/loop.py` | **NEW** — 310 lines, bounded loop controller |
| `ragix_core/memory/cli.py` | **MODIFIED** — `loop` subcommand, MockEmbedder removal |
| `ragix_core/memory/loop.py:invoke_llm()` | **MODIFIED** — `include_protocol` parameter for passive mode |
| `demos/ragix_memory_demo/run_demo.sh` | **MODIFIED** — Act 9 split (9a/9b), logging hoist |
| `demos/ragix_memory_demo/mock_llm.sh` | **NEW** — 100 lines, deterministic 3-iteration mock |
| `demos/ragix_memory_demo/README.md` | **MODIFIED** — Act 9a/9b docs, convergence notes |
| `ragix_core/memory/tests/test_similarity.py` | **NEW** — 43 tests |
| `ragix_core/memory/tests/test_loop.py` | **NEW** — 33 tests |
| `ragix_core/memory/tests/test_interop_protocol.py` | **NEW** — 30 tests |
| `pyproject.toml` | **MODIFIED** — v0.69.0 |
| `ragix_core/version.py` | **MODIFIED** — v0.69.0, BUILD_DATE 2026-02-19 |

---

## v0.68.0 — Multi-Format Extract, Memory Demo & CLI Hardening (2026-02-18)

### Highlights

**RAGIX v0.68.0 extends memory ingestion to 46 file formats (including Office binaries and PDF), adds a flagship 8-act demo with dual LLM reasoning (Ollama + Claude), and hardens the CLI with `init`, `pull`, `serve` subcommands and environment variable support.**

| Feature | Status |
|---------|--------|
| Multi-format text extraction (46 extensions) | ✅ Markdown, source code, config, Office, PDF |
| Binary format support (docx/odt/pptx/odp/xlsx/ods/pdf) | ✅ Optional deps via `ragix[docs]` |
| Memory CLI hardening (15 subcommands) | ✅ `init`, `pull`, `serve` added |
| Environment variables (`RAGIX_MEMORY_DB`, `RAGIX_MEMORY_BUDGET`) | ✅ CLI override chain |
| 8-act flagship demo with LLM reasoning | ✅ Ollama (Granite) + Claude backends |
| Claude isolation pattern for clean piping | ✅ `--system-prompt` + `--tools ""` + `cd /tmp` |
| Per-question timing & logging infrastructure | ✅ Elapsed time, token rate, full I/O logs |
| Test infrastructure (contracts, CLI UX) | ✅ Fixtures + 2 test files |

### New Module: `ragix_core/memory/extract.py`

Unified text extraction layer supporting 46 file extensions:

```python
from ragix_core.memory.extract import read_file_text, ALL_INGESTABLE_EXTS, BINARY_EXTS

# Transparently handles text and binary formats
text = read_file_text("report.docx")   # python-docx
text = read_file_text("slides.pptx")   # python-pptx
text = read_file_text("data.xlsx")     # openpyxl
text = read_file_text("doc.odt")       # odfpy
text = read_file_text("paper.pdf")     # pdftotext (existing doc_tools)
text = read_file_text("code.py")       # direct read (utf-8)
```

**Key design decisions:**
- Single dispatch: `read_file_text(path)` handles all formats transparently
- Binary extractors are optional — missing library raises `ImportError` with install hint
- `ALL_INGESTABLE_EXTS` is the canonical set imported by `ingest.py` (prevents drift)
- PDF delegates to existing `doc_tools.extract_pdf_text()` (no duplication)

**Supported binary formats (7):**

| Format | Library | Install |
|--------|---------|---------|
| `.docx` | python-docx | `pip install ragix[docs]` |
| `.odt` | odfpy | `pip install ragix[docs]` |
| `.pptx` | python-pptx | `pip install ragix[docs]` |
| `.odp` | odfpy | `pip install ragix[docs]` |
| `.xlsx` | openpyxl | `pip install ragix[docs]` |
| `.ods` | odfpy | `pip install ragix[docs]` |
| `.pdf` | pdftotext (poppler) | system package |

### Memory CLI Hardening (`ragix_core/memory/cli.py`)

CLI expanded from 11 to **15 subcommands** (812 lines, +235 lines):

**New subcommands:**
- **`init [path]`** — Create memory workspace (SQLite DB, `config.yaml`, `.gitignore`)
- **`pull`** — Capture LLM output from stdin into memory (structured proposal parsing with fallback)
- **`serve`** — Start MCP server for memory tools (config-gated)
- **`push`** — Alias for `pipe` (ingest + recall)

**New environment variables:**
- `RAGIX_MEMORY_DB` — Override database path (precedence: CLI > env > last-db cache > `memory.db`)
- `RAGIX_MEMORY_BUDGET` — Default token budget for pipe/recall

**DB resolution chain:** `--db` flag → `RAGIX_MEMORY_DB` env → `~/.cache/ragix/last_memory_db` → `memory.db`

### Flagship Demo: `demos/ragix_memory_demo/`

8-act narrative demonstrating the complete ragix-memory lifecycle:

| Act | Title | What it demonstrates |
|-----|-------|---------------------|
| 1 | Init | Workspace creation |
| 2 | Ingest | Corpus ingestion (markdown docs, explicit glob patterns) |
| 3 | Search & Recall | FTS5/BM25 queries (4 diverse topics) |
| 4 | Idempotency | SHA-256 dedup proof (re-ingest → 0 new chunks) |
| 5 | Pull | Capture simulated LLM output → memory |
| 6 | Stats & Palace | Store statistics + spatial memory palace |
| 7 | Export | JSONL export with Unix pipe composability |
| 8 | LLM Reasoning | Dual-backend: Ollama (Q1-Q2) + Claude (Q3-Q4) |

**Act 8 — LLM Reasoning highlights:**
- **Ollama section:** 2 questions answered by `granite3.1-moe:3b` (local, sovereign)
- **Claude section:** 2 harder questions requiring deeper analysis
- **Per-question metrics:** elapsed time, estimated tokens, tok/s rate
- **Full I/O logging:** `$WORKSPACE/llm_logs/Q{n}_{context,input,output}.txt`
- **Graceful degradation:** skips if Ollama/Claude unavailable; `--no-llm` skips Act 8 entirely

**Claude isolation pattern** (prevents CLAUDE.md leaks into answers):
```bash
# cd to /tmp prevents project CLAUDE.md injection
# --system-prompt replaces default system prompt
# --tools "" disables all tools (pure LLM, no Bash/Read/Grep)
# --setting-sources "" skips project/local settings
cd /tmp && claude \
    --system-prompt "You are a doc analyst. Answer only from stdin." \
    --tools "" \
    --setting-sources "" \
    --no-session-persistence \
    -p < recalled_context.txt
```

**Demo options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--keep` | cleanup | Keep workspace after demo |
| `--workspace DIR` | `/tmp/ragix_memory_demo` | Custom workspace |
| `--budget N` | `2000` | Token budget for recall |
| `--skip-ingest` | false | Skip Act 2 (reuse existing DB) |
| `--corpus DIR` | `docs/` | Source corpus directory |
| `--model MODEL` | `granite3.1-moe:3b` | Ollama model for Act 8 |
| `--no-llm` | false | Skip Act 8 entirely |

### Integration Points

| File | Changes |
|------|---------|
| `ragix_core/memory/extract.py` | **NEW** — 280 lines, 46 extensions, 7 binary extractors |
| `ragix_core/memory/cli.py` | **MODIFIED** — +235 lines, `init`/`pull`/`serve`/`push`, env vars |
| `ragix_core/memory/ingest.py` | **MODIFIED** — Uses `extract.read_file_text()`, extended `_EXT_TAG_MAP` (+18 entries) |
| `demos/ragix_memory_demo/run_demo.sh` | **NEW** — 843 lines, 8-act demo with LLM logging |
| `demos/ragix_memory_demo/README.md` | **NEW** — 331 lines, architecture diagrams, format table |
| `ragix_core/memory/tests/test_cli_ux.py` | **NEW** — CLI UX test suite |
| `ragix_core/memory/tests/test_contracts.py` | **NEW** — Contract tests |
| `ragix_core/memory/tests/fixtures/contracts.json` | **NEW** — Test fixtures (7 entries) |
| `pyproject.toml` | **MODIFIED** — v0.68.0, added `docs` optional deps, `ragix-memory` entry point |
| `ragix_core/version.py` | **MODIFIED** — v0.68.0 |

### Ingestion Pipeline Change

Before v0.68.0, `ingest_file()` used `Path.read_text()` — binary files would fail silently or produce garbage. Now:

```
ingest_file(path)
    └─→ read_file_text(path)        # from extract.py
        ├─→ .docx → _extract_docx() # python-docx
        ├─→ .pptx → _extract_pptx() # python-pptx
        ├─→ .xlsx → _extract_xlsx() # openpyxl
        ├─→ .odt/.odp/.ods → _extract_odf() # odfpy
        ├─→ .pdf → doc_tools.extract_pdf_text()
        └─→ .* → Path.read_text(encoding="utf-8", errors="replace")
```

`_INGESTABLE_EXTS` in `ingest.py` now imports from `extract.py` (`ALL_INGESTABLE_EXTS`) — single source of truth.

### Demo Ingestion Fix

Act 2 uses **explicit glob patterns** (`docs/*.md`, `docs/developer/*.md`, `docs/archive/*.md`) instead of recursive directory walk. This avoids ingesting machine-generated logs (`.KOAS/runs/`) and binary noise that would pollute FTS5 search results.

---

## v0.67.0 — KOAS Memory, Summary & Memory Pipe (2026-02-16)

### Highlights

**RAGIX v0.67.0 introduces three major subsystems: episodic memory with policy-driven storage and MCP exposure, multi-document summarization with Graph-RAG and secrecy tiers, and a pipe-based demo for Claude integration.**

| Feature | Status |
|---------|--------|
| KOAS Memory (12 core + 9 MCP modules) | ✅ 17 MCP tools, FTS5+BM25 retrieval, Q\*-search |
| Memory CLI (`ragix-memory`) | ✅ 11 subcommands (search, recall, ingest, pipe, palace...) |
| Memory Palace (spatial metaphor) | ✅ Room/loci placement, guided tours |
| STM→MTM→LTM promotion | ✅ Policy-driven with SHA-256 corpus dedup |
| KOAS Summary (12 kernels) | ✅ 3-stage pipeline, Graph-RAG, secrecy tiers |
| Summary CLI (`summaryctl`) | ✅ `--graph/--no-graph`, `--secrecy` flags |
| Graph-RAG integration | ✅ Entity extraction → graph store → community detection |
| Secrecy tiers (S0/S2/S3) | ✅ Deterministic redaction before LLM exposure |
| Memory Pipe Demo | ✅ 6-act narrative with live Claude inference |
| Shared utilities | ✅ `gpu_detect`, `text_utils` in `ragix_core/shared/` |
| MCP memory integration | ✅ Config-gated `_register_memory_tools()` |
| Tests | ✅ 511 passing across 15 test files |

### New Module: KOAS Memory (`ragix_core/memory/`)

Episodic memory subsystem with policy-driven storage and multi-tier promotion:

```python
from ragix_core.memory import MemoryStore, MemoryPolicy

store = MemoryStore(db_path="memories.db")
policy = MemoryPolicy.default()

# Store with automatic tier assignment
store.add("The API uses JWT tokens for auth", tags=["api", "auth"])

# Retrieve with FTS5 + BM25 scoring
results = store.search("authentication method", top_k=5)
```

**Core modules (12):** types, store, embedder, policy, tools, proposer, middleware, recall, qsearch, palace, consolidate, cli

**Key features:**
- **FTS5 + BM25** full-text search with SQLite
- **Q\*-style search** — iterative deepening with relevance feedback
- **Memory Palace** — spatial metaphor with rooms, loci, guided tours
- **STM→MTM→LTM** tier promotion based on access frequency and recency
- **SHA-256 corpus dedup** — prevents duplicate memory storage
- **Consolidation** — merge related memories, prune stale entries

### New Module: KOAS Summary (`ragix_kernels/summary/`)

Multi-document summarization with Graph-RAG and secrecy:

```bash
# Standard summarization
summaryctl summarize ./workspace --format markdown

# With Graph-RAG (entity extraction + community detection)
summaryctl summarize ./workspace --graph

# With secrecy tier redaction
summaryctl summarize ./workspace --secrecy S2
```

**12 kernels** following the 3-stage pipeline:
- **S1 (Collection):** `summary_collect`, `summary_ingest`, `summary_extract_entities`, `summary_build_graph`
- **S2 (Analysis):** `summary_generate`, `summary_consolidate`, `summary_drift`, `summary_redact`, `summary_capabilities`, `summary_budgeted_recall`, `summary_verify`
- **S3 (Reporting):** `summary_report`

**Graph-RAG pipeline:** Entity extraction → Graph store (shared SQLite) → Community detection → Budgeted recall

### New Module: Memory MCP (`ragix_core/memory/mcp/`)

17 MCP tools for memory operations, exposed via the main RAGIX MCP server:

```python
# In MCP/ragix_mcp_server.py (line 3569)
if os.environ.get("RAGIX_MEMORY_ENABLED"):
    _register_memory_tools(mcp)
```

**MCP modules (9):** server, tools, session, workspace, metrics, rate_limiter, formatting, prompts, `__main__`

### Memory Pipe Demo (`demos/koas_pipe_demo/`)

6-act narrative demonstrating memory-augmented Claude inference:

```bash
# Ingest documents into memory, then query via pipe
ragix-memory pipe "What authentication methods are used?" \
    --source docs/ | claude

# Source accepts files, directories, and globs
ragix-memory pipe "Summarize security findings" \
    --source "reports/*.md"
```

**Features:**
- `resolve_sources()` — `--source` accepts files, directories, globs
- Last-DB reuse via `~/.cache/ragix/last_memory_db`
- 6 acts: corpus measurement → ingest → surgical recalls → numbers → synthesis → live inference

### Integration Points

| File | Changes |
|------|---------|
| `ragix_core/memory/` | **NEW** — 12 core modules, episodic memory subsystem |
| `ragix_core/memory/mcp/` | **NEW** — 9 MCP modules, 17 tools |
| `ragix_core/memory/tests/` | **NEW** — 15 test files, 511 tests |
| `ragix_core/memory/skills/` | **NEW** — 4 skill files (memory.md + 3 aliases) |
| `ragix_kernels/summary/` | **NEW** — 12 kernels, CLI, MCP tools, visualization APIs |
| `ragix_core/shared/` | **NEW** — `gpu_detect.py`, `text_utils.py` |
| `demos/koas_pipe_demo/` | **NEW** — Memory pipe demo (4 files) |
| `docs/KOAS_MEMORY_MCP.md` | **NEW** — Memory MCP documentation (v2.0.0, 636 lines) |
| `docs/KOAS_MEMORY_ARCHITECTURE.md` | **NEW** — Memory architecture documentation |
| `docs/KOAS_SUMMARY.md` | **NEW** — Summary subsystem documentation |
| `MCP/ragix_mcp_server.py` | **MODIFIED** — `_register_memory_tools()` (config-gated) |
| `pyproject.toml` | **MODIFIED** — Version 0.62.0 → 0.67.0, `ragix-memory` entry point |
| `unix-rag-agent.py` | **MODIFIED** — Memory middleware hooks |
| `demos/README.md` | **MODIFIED** — Added pipe demo documentation |

### Documentation

| Document | Version | Lines | Content |
|----------|---------|-------|---------|
| `docs/KOAS_MEMORY_MCP.md` | v2.0.0 | 636 | 17 MCP tools, workspace router, metrics, rate limiting |
| `docs/KOAS_MEMORY_ARCHITECTURE.md` | v1.0.0 | — | Core architecture, tier model, consolidation |
| `docs/KOAS_SUMMARY.md` | v1.0.0 | — | Summary pipeline, Graph-RAG, secrecy |
| `docs/developer/ROADMAP_MEMORY.md` | — | — | Memory subsystem blueprint |
| `docs/developer/PLAN_MEMORY_MCP_SKILL.md` | v1.2.0 | — | MCP skill implementation plan |

---

## v0.66.0 — Centralized Activity Logging & Broker Gateway (2026-01-30)

### Highlights

**KOAS v0.66.0 introduces centralized activity logging for complete observability of kernel and LLM operations, plus an optional broker gateway for critical applications requiring API authentication.**

| Feature | Status |
|---------|--------|
| Centralized Activity Logging | ✅ JSONL event stream at `.KOAS/activity/events.jsonl` |
| Event Schema (koas.event/1.0) | ✅ Structured events with sovereignty attestation |
| Kernel Start/End Events | ✅ All kernel executions logged with metrics |
| LLM Call Events | ✅ Model, cache_hit, duration logged (no content) |
| Broker Gateway (Optional) | ✅ FastAPI-based for restricted mode |
| ACL Support | ✅ API key + scope-based access control |
| Demo Setup | ✅ `demo/koas_docs_audit/` with 79 markdown files |
| Documentation Update | ✅ SOVEREIGN_LLM_OPERATIONS.md v1.4.0 |

### New Module: `ragix_kernels/activity.py`

Centralized activity logging with append-only JSONL events:

```python
from ragix_kernels.activity import init_activity_writer, get_activity_writer

# Initialize at workflow start
writer = init_activity_writer(workspace=workspace, run_id=run_id)

# Events emitted automatically by orchestrator and llm_wrapper
# - docs.kernel:start / docs.kernel:end
# - docs.llm:call / docs.llm:cache_hit
```

**Event Schema (koas.event/1.0):**
```json
{
  "v": "koas.event/1.0",
  "ts": "2026-01-30T20:50:11.142+00:00",
  "event_id": "uuid-v4",
  "run_id": "run_20260130_215011_348bc4",
  "actor": {"type": "system", "id": "koas", "auth": "none"},
  "scope": "docs.kernel",
  "phase": "end",
  "kernel": {"name": "doc_metadata", "version": "1.0.0", "stage": 1},
  "decision": {"success": true, "cache_hit": false},
  "metrics": {"duration_ms": 42, "item_count": 79},
  "sovereignty": {"local_only": true}
}
```

### Integration Points

| File | Changes |
|------|---------|
| `ragix_kernels/activity.py` | **NEW** - ActivityWriter, ActivityReader, event dataclasses |
| `ragix_kernels/orchestrator.py` | Emit kernel start/end events in `_run_kernel()` |
| `ragix_kernels/llm_wrapper.py` | Emit LLM call events with cache status |
| `ragix_kernels/run_doc_koas.py` | Initialize activity writer at workflow start |

### Broker Gateway (Optional)

For critical applications requiring API authentication:

```bash
# Start broker (separate terminal)
cd demo/koas_docs_audit && ./start_broker.sh

# Trigger via API
curl -X POST http://localhost:8080/koas/v1/jobs \
    -H "Authorization: Bearer $KOAS_API_KEY" \
    -d '{"mode": "pure_docs", "workspace": "./workspace"}'
```

**ACL Configuration (`acl.yaml`):**
```yaml
clients:
  claude-demo:
    key_hash: "sha256:..."
    type: external_orchestrator
    scopes: ["docs.trigger", "docs.status", "docs.export_external"]
    rate_limit: "30/min"
```

### Demo Setup

New demo at `demo/koas_docs_audit/`:
- **Corpus:** 79 markdown files (RAGIX documentation)
- **Relaxed Mode:** Direct CLI execution (default)
- **Restricted Mode:** Broker gateway with ACL

```bash
cd demo/koas_docs_audit
./setup.sh  # Create workspace with symlink to docs/
python -m ragix_kernels.run_doc_koas init --workspace ./workspace --project ./workspace/docs
python -m ragix_kernels.run_doc_koas run --workspace ./workspace --stage 1 --skip-preflight
```

### Validation

```bash
# Verify activity events
cat workspace/.KOAS/activity/events.jsonl | wc -l
# Expected: ~30 events for stage 1

# All events local
jq '.sovereignty.local_only' events.jsonl | sort -u
# Expected: true
```

### Documentation Updates

- `docs/SOVEREIGN_LLM_OPERATIONS.md` → v1.4.0
  - §6. Centralized Activity Logging (NEW)
  - §7. Broker Gateway for Critical Applications (NEW)
  - §12. Demo: KOAS Docs Audit (NEW)
  - Updated TOC and section numbering

### Sovereignty Guarantees

Every activity event includes:
- `sovereignty.local_only: true` — Confirms no external API calls
- `actor.type: "system"` — Internal KOAS processing
- **No content fields** — Prompts, responses, excerpts are NEVER logged

---

## v0.64.2 — Boilerplate Detection Enhancement & Output Path Fix (2026-01-29)

### Highlights

**KOAS-Docs v0.64.2 improves representative content quality by filtering release changelog patterns and infrastructure notation, and fixes the report output path to ensure audit artifacts don't contaminate future RAG indexing.**

| Feature | Status |
|---------|--------|
| Changelog boilerplate patterns | ✅ Filter `X.X.X.X MERGED/RELEASED` patterns |
| Infrastructure notation filter | ✅ Filter `VM1 VM2`, `Datacenter`, `Lien WAN` |
| Report output path fix | ✅ Write to `.KOAS/` instead of workspace root |
| Quality metric improvement | ✅ Sentences: 649 → 644 (-5 boilerplate) |

### Problem Solved

**Before v0.64.2:** Representative content excerpts contained non-informative text:
```
> 2 01 Oct 2025 0 issues MERGED ProjectX 3.4.3.18 02 Sept 2025 0 issues MERGED...
> 29 Oct 2025 5 issues RELEASED ProjectX 3.4.3.24...
> VM1 VM2 VMx...
```

**After v0.64.2:** Meaningful technical content preserved:
```
> **Sous-chapitre 3 : Appel à la fonction et scripts** Ce sous-chapitre décrit...
> ChangeSetState (Enumération) - ModificationType (Enumération)...
> Paris Risk Management Last risk assessment: 18 Dec 2025 Risk KPI...
```

### New Boilerplate Patterns (`config.py`)

Added `boilerplate_changelog` field to `QualityConfig`:

```python
boilerplate_changelog: List[str] = field(default_factory=lambda: [
    r"\d+\.\d+\.\d+\.\d+\s*.*?(?:MERGED|RELEASED|issues)",  # X.X.X.X MERGED
    r"(?:Jan|Feb|...|Dec)\s+\d{4}\s+\d+\s+issues",           # Month YYYY N issues
    r"\d+\s+issues\s+(?:MERGED|RELEASED|CLOSED)",            # N issues STATUS
    r"(?:NOT\s+)?DELIVERED",                                  # DELIVERED markers
    r"(?:VM\d+\s*)+",                                         # VM1 VM2 notation
    r"Sous-r[eé]seau\s+priv[eé]",                            # Network diagram text
    r"Fibre\s+noire",                                         # Infrastructure terms
    r"Datacenter\s+\w+",                                      # Datacenter refs
    r"Lien\s+WAN",                                            # Network link refs
])
```

### Output Path Fix (`doc_final_report.py`)

**Before:**
```python
run_dir = input.workspace  # → src2/ (pollutes RAG index)
```

**After:**
```python
run_dir = input.workspace / ".KOAS"  # → src2/.KOAS/ (excluded from indexing)
```

### Files Modified

| File | Changes |
|------|---------|
| `ragix_kernels/docs/config.py` | Added `boilerplate_changelog` field to `QualityConfig` |
| `ragix_kernels/docs/doc_extract.py` | Updated `_compile_boilerplate_pattern()` to include changelog patterns |
| `ragix_kernels/docs/doc_final_report.py` | Fixed `run_dir` to use `.KOAS/` subdirectory |

### Validation (Document Corpus Audit)

| Metric | v0.64.1 (Jan 29) | v0.64.2 (Jan 29) | Status |
|--------|------------------|------------------|--------|
| REP_QUALITY | Bad (changelog data) | Good (API docs) | ✅ Fixed |
| SENTENCE_SELECT | 649 | 644 (-5) | ✅ Improved |
| OUTPUT_PATH | `src2/final_report.md` | `src2/.KOAS/final_report.md` | ✅ Fixed |

### Regression Testing

Report history tracking established with KQI (Key Quality Indicators) metrics for measuring improvements across versions.

---

## v0.64.0 — Two-Tier Caching: LLM + Kernel Output Cache (2026-01-22)

### Highlights

**RAGIX now features dual caching layers for maximum iteration speed: LLM response caching and kernel output caching. When document list is stable, full audit pipeline runs in seconds instead of minutes.**

| Feature | Status |
|---------|--------|
| LLM response cache with modes | ✅ `write_through`, `read_only`, `read_prefer`, `off` |
| Kernel output cache | ✅ Cache entire kernel results by input hash |
| `--llm-cache` CLI flag | ✅ Control LLM caching behavior |
| `--kernel-cache` CLI flag | ✅ Control kernel output caching |
| Cache sovereignty tracking | ✅ Hostname, user, timestamp metadata |
| Input hash-based invalidation | ✅ Auto-invalidate on config/dependency changes |

### Performance Impact

**DOCSET src2 audit (137 documents):**

| Configuration | Stage 1 | Stage 2 | Stage 3 | Total |
|--------------|---------|---------|---------|-------|
| No cache | 54s | ~11 min | ~42s | ~12 min |
| LLM cache only | 54s | ~11 min | ~42s | ~12 min |
| Both caches | **2ms** | **~10s** | **34s** | **~45s** |

**Speedup: ~16x on cached runs**

### Two-Tier Caching System

```
┌─────────────────────────────────────┐
│  Kernel Input (config + deps)      │
│  → Hash: 5c9b158b                  │
└──────────┬──────────────────────────┘
           ↓
    ┌──────────────┐
    │ Kernel Cache │ ← New in v0.64
    │  Hit? Yes    │
    └──────────────┘
           ↓
    Return cached output (0ms)

    If cache miss:
           ↓
    ┌──────────────┐
    │ Run Kernel   │
    └──────────────┘
           ↓
    ┌──────────────┐
    │  LLM Cache   │ ← Existing
    │  Hit? Yes    │
    └──────────────┘
```

### LLM Cache Modes

```bash
# Default: normal operation, cache all responses
ragix-koas run --workspace ./ws --all --llm-cache=write_through

# Fast replay: fail on cache miss (deterministic)
ragix-koas run --workspace ./ws --all --llm-cache=read_only

# Prefer cache, fallback to LLM
ragix-koas run --workspace ./ws --all --llm-cache=read_prefer

# Disable LLM caching
ragix-koas run --workspace ./ws --all --llm-cache=off
```

### Kernel Cache Modes

```bash
# Default: cache kernel outputs
ragix-koas run --workspace ./ws --all --kernel-cache=write_through

# Ultra-fast: replay cached kernels (Stage 1: 2ms)
ragix-koas run --workspace ./ws --all --kernel-cache=read_only

# Combine both for maximum speed
ragix-koas run --workspace ./ws --all \
    --llm-cache=read_only \
    --kernel-cache=read_only
```

### Cache Invalidation

Kernel cache automatically invalidates when:
- Config changes (LLM model, parameters, options)
- Dependencies change (upstream kernel outputs modified)
- Kernel version changes (code updates)

**Input hash includes:**
```json
{
  "kernel": "doc_extract@1.0.0",
  "config": {"llm_model": "granite3.1-moe:3b", ...},
  "dependencies": {"doc_metadata": "/path/to/stage1/doc_metadata.json"}
}
```

### Cache Structure

```
.KOAS/cache/
├── llm_responses/              # LLM response cache
│   ├── granite3.1-moe_3b/
│   │   └── a3f2c8d1.json      # Cached LLM response
│   └── mistral_7b-instruct/
├── kernel_outputs/             # Kernel output cache (new)
│   ├── doc_extract_1f732e1c.json
│   ├── doc_quality_299be442.json
│   └── doc_summarize_8e4a1f3d.json
├── cache_stats.json            # LLM cache statistics
├── kernel_cache_stats.json     # Kernel cache statistics (new)
└── cache_index.json            # LLM cache index
```

### Sovereignty Tracking

Both caches track sovereignty metadata:

```json
{
  "hostname": "workstation-01",
  "user": "analyst",
  "endpoint": "http://127.0.0.1:11434",
  "local": true,
  "timestamp": "2026-01-22T17:51:18Z"
}
```

### Files Added/Modified

| File | Changes |
|------|---------|
| `ragix_kernels/cache.py` | Added `CacheMode` enum, `CacheMissError`, `KernelCache` class |
| `ragix_kernels/llm_wrapper.py` | **NEW** - Unified LLM call boundary with caching |
| `ragix_kernels/orchestrator.py` | Integrated kernel cache in `_run_kernel()` |
| `ragix_kernels/run_doc_koas.py` | Added `--llm-cache` and `--kernel-cache` CLI flags |
| `ragix_kernels/docs/doc_summarize.py` | Uses `llm_call_with_ollama_lib()` wrapper |
| `ragix_kernels/docs/doc_func_extract.py` | Uses `llm_call_with_ollama_lib()` wrapper |

### Usage Examples

**Scenario 1: Initial audit**
```bash
# Populate both caches (normal run)
ragix-koas run --workspace DOCSET/src2 --all
```

**Scenario 2: Iterate on report templates**
```bash
# Use cached kernel outputs + cached LLM responses
ragix-koas run --workspace DOCSET/src2 --all \
    --llm-cache=read_only \
    --kernel-cache=read_only
# Result: ~45 seconds instead of ~12 minutes
```

**Scenario 3: Update LLM prompts**
```bash
# Invalidate LLM cache, keep kernel cache
ragix-koas run --workspace DOCSET/src2 --all \
    --llm-cache=off \
    --kernel-cache=read_only
```

**Scenario 4: Fresh run after code changes**
```bash
# Bypass all caches
ragix-koas run --workspace DOCSET/src2 --all \
    --llm-cache=off \
    --kernel-cache=off
```

### Safety & Regression Risk

**LOW regression risk** - Comprehensive safeguards:

1. **Automatic invalidation**: Input hash includes all dependencies
2. **Default unchanged**: `write_through` mode preserves existing behavior
3. **Output consistency**: Cached outputs written to expected file locations
4. **Fail-fast**: `read_only` mode logs warnings on cache miss

**Tested on DOCSET audit**: All stages completed successfully with identical outputs.

---

## v0.63.0 — KOAS Document Analysis Enhancement & Deterministic Execution (2026-01-18)

### Highlights

**KOAS/docs kernels gain comprehensive final report generation, improved appendices with human-readable file names, dual LLM architecture (Worker + Tutor), and deterministic cache-based execution.**

| Feature | Status |
|---------|--------|
| `--use-cache` flag | ✅ Skip LLM kernels with cached outputs |
| Appendix D improvements | ✅ Type-specific discrepancy details |
| Appendix E improvements | ✅ File names instead of IDs |
| Appendix F (new) | ✅ Artifacts and visualizations catalog |
| `doc_final_report` kernel | ✅ Comprehensive report generation |
| `doc_visualize` kernel | ✅ Word clouds and concept graphs |
| Dual LLM (Worker + Tutor) | ✅ Granite 3b + Mistral 7b |
| Dual reconstruction | ✅ Pyramidal + Leiden clustering |

### Deterministic Execution (`--use-cache`)

New orchestrator flag to reuse cached kernel outputs:

```bash
# Skip kernels with existing cached outputs
ragix-koas run --workspace ./ws --all --use-cache

# Result: Stage 3 with 13 kernels in 0ms (all cached)
```

**Implementation:**
- Added `--use-cache, -C` argument to `run` command
- Added `_load_cached_output()` method to load existing kernel results
- Modified `_run_stage_sequential()` and `_run_stage_parallel()` to check cache before execution
- Enables rapid iteration: delete one cache file, regenerate only that kernel

### Appendix Improvements

#### Appendix D: Discrepancy Details

Now shows type-specific information:

| Discrepancy Type | Information Shown |
|------------------|-------------------|
| **Content Overlap** | File pairs with similarity percentage + text sample |
| **Terminology Variation** | Base term + all variants found |
| **Version Reference** | File name + file version + referenced version + context |

#### Appendix E: Clustering Analysis

- Shows **file names** (e.g., "2025.08.11 - CR CoStrat SURF4.pdf")
- Previously showed internal IDs (e.g., "F000012")
- Added `_build_file_id_mapping()` method for ID-to-name resolution

#### Appendix F: Artifacts Catalog (New)

Links to all generated visualizations:
- Word clouds (corpus, per-domain)
- Concept distribution charts
- Cluster visualizations
- All files in `visualizations/` directory

### Dual LLM Architecture

LLM-enabled kernels (`doc_summarize`, `doc_func_extract`) now use two models:

```
Worker (granite3.1-moe:3b) → Draft Output
        ↓
Tutor (mistral:7b-instruct) → Refined Output
```

**Benefits:**
- Speed: Small model handles bulk extraction
- Quality: Larger model validates and refines
- Cost: Minimizes expensive model usage

### Dual Reconstruction

Document clustering combines two algorithms:

| Algorithm | Purpose |
|-----------|---------|
| **Hierarchical (Pyramidal)** | Document → Group → Domain → Corpus |
| **Leiden Community Detection** | Graph-based optimized partitions |

### Files Modified

| File | Changes |
|------|---------|
| `ragix_kernels/orchestrator.py` | `--use-cache` flag, `_load_cached_output()` |
| `ragix_kernels/docs/doc_final_report.py` | Appendix D/E/F generation, file ID mapping |
| `ragix_kernels/README.md` | v1.2.0, --use-cache docs, dual LLM/reconstruction |

### Tested on DOCSET Corpus

- **159 documents** (DOCX, PDF, PPTX, XLSX)
- **5,515 chunks** indexed
- **33 content overlaps** detected with file pairs
- **7 terminology variations** identified
- **11 version references** flagged
- **Stage 3**: 45.2s full run → 0ms with cache

---

## v0.62.0 — KOAS MCP Consolidation, Demo UI & Academic Documentation (2025-12-20)

### Highlights

**KOAS is now fully consolidated with 38 MCP tools, an interactive demo UI, and comprehensive academic-level documentation explaining MCP as a protocol and reasoning engine architectures.**

| Feature | Status |
|---------|--------|
| Total MCP Tools | ✅ 38 tools (11 core + 5 system + 6 KOAS base + 8 security + 8 audit) |
| KOAS Demo UI | ✅ FastAPI + WebSocket with chat, history, markdown rendering |
| MCP Documentation | ✅ Protocol-first academic documentation (800+ lines) |
| Reasoning Documentation | ✅ ContractiveReasoner + v30 deep dive (600+ lines) |
| Documentation Index | ✅ Navigation hub linking all docs |

### KOAS Demo UI (`demos/koas_mcp_demo/`)

Interactive web interface for testing KOAS MCP tools:

```
demos/koas_mcp_demo/
├── server.py          # FastAPI server with WebSocket, memory management
├── static/
│   ├── index.html     # Demo UI with 4 tabs
│   ├── css/koas_demo.css
│   └── js/koas_client.js
└── README.md
```

**Features:**
- **Chat interface** with markdown rendering (marked.js)
- **History sidebar** — conversation tracking with save/restore sessions
- **Tool trace panel** — real-time tool execution visualization
- **Model selector** — Ollama model selection
- **Scenario browser** — Quick/Full Security/Audit presets
- **Dry-run mode** — safe testing without actual execution
- **Memory management** — conversation context with clear/save/load

### Academic Documentation

#### `docs/MCP.md` — Model Context Protocol (800+ lines)

Addresses common misconceptions about MCP:
- **Protocol vs System** — MCP is a communication protocol, not an orchestration engine
- **Local/Remote/Sovereign** — Works with any LLM deployment model
- **Stochastic vs Deterministic** — Hybrid backends explained
- **Complete tool reference** — All 38 tools documented with parameters
- **Deployment topologies** — Local, team server, hybrid, multi-server collective
- **Collective intelligence** — Multi-agent patterns, information exchange beyond text
- **FAQ section** — "Do we need MCP if tools are local?" and other questions

#### `docs/REASONING.md` — Reasoning Engines (600+ lines)

Comprehensive coverage of RAGIX reasoning systems:
- **ContractiveReasoner** — Tree-based decomposition with Banach fixed-point theorem
- **Reasoning v30** — Graph-based state machine with 7-node pipeline
- **Mathematical foundations** — Entropy metrics, decision logic, state machines
- **Comparison guide** — When to use which engine
- **MCP integration** — How reasoning engines leverage KOAS tools
- **Configuration reference** — Complete parameter documentation

#### `docs/INDEX.md` — Documentation Navigation Hub

- Visual documentation map
- Reading order by goal (audit, security, custom reasoning)
- Glossary of key terms
- Cross-references between documents

### MCP Tools Summary (38 Total)

```
RAGIX Core (11 tools)
├── ragix_chat, ragix_scan_repo, ragix_read_file, ragix_search
├── ragix_workflow, ragix_templates, ragix_config, ragix_health
└── ragix_logs, ragix_verify_logs, ragix_agent_step

System (5 tools)
├── ragix_ast_scan, ragix_ast_metrics
└── ragix_models_list, ragix_model_info, ragix_system_info

KOAS Base (6 tools)
├── koas_init, koas_run, koas_status
└── koas_summary, koas_list_kernels, koas_report

KOAS Security (8 tools)
├── koas_security_discover, koas_security_scan_ports
├── koas_security_ssl_check, koas_security_vuln_scan
├── koas_security_dns_check, koas_security_compliance
└── koas_security_risk, koas_security_report

KOAS Audit (8 tools)
├── koas_audit_scan, koas_audit_metrics
├── koas_audit_hotspots, koas_audit_dependencies
├── koas_audit_dead_code, koas_audit_risk
└── koas_audit_compliance, koas_audit_report
```

### KOAS Helpers (`MCP/koas_helpers.py`)

- Auto-workspace creation in `/tmp/koas_{category}_{timestamp}_{uuid}`
- Output simplification for LLM consumption (< 300 char summaries)
- Target resolution (`"discovered"` keyword for chaining)
- Port presets (common, web, database, admin, top100, full)
- Compliance framework presets (ANSSI, NIST, CIS)
- Dependency path loading for kernel chaining

### Design Principles

1. **Protocol-first** — MCP standardizes communication, not orchestration
2. **Deterministic kernels** — LLMs reason, kernels compute—no hallucinated metrics
3. **Single values** instead of arrays (use `"discovered"` for chaining)
4. **Preset strings** instead of complex configurations
5. **Mandatory summaries** (< 300 chars) in every response
6. **Auto-workspace** — temporary workspaces created automatically

### Target Hardware

Optimized for testing on DELL GB10 (NVIDIA OEM, 128GB VRAM):
- Mistral 7B (baseline, ~70% tool accuracy)
- Llama 3.1 70B (target, 90%+ accuracy)
- DeepSeek-V2 236B (95%+ accuracy)

---

## v0.61.0 — Security Kernels & Interactive Demos (2025-12-17)

### Highlights

**KOAS gains 10 security assessment kernels with compliance frameworks (ANSSI/NIST/CIS) and comprehensive interactive examples.**

| Feature | Status |
|---------|--------|
| Security Kernels | ✅ 10 kernels implemented |
| Compliance Framework | ✅ ANSSI, NIST CSF, CIS Controls v8 |
| Security Examples | ✅ 4 workspaces with demo script |
| Audit Examples | ✅ 4 ACME-ERP-based workspaces |
| Interactive Demos | ✅ Menu-driven bash scripts |

### New Security Kernels (`ragix_kernels/security/`)

| Kernel | Stage | Purpose |
|--------|-------|---------|
| `net_discover` | 1 | Network asset enumeration (nmap, arp-scan) |
| `port_scan` | 1 | Service and port detection |
| `dns_enum` | 1 | DNS analysis and subdomain enumeration |
| `config_parse` | 1 | Firewall config parsing (iptables, Cisco, pfSense) |
| `ssl_analysis` | 2 | TLS/certificate audit (testssl.sh) |
| `vuln_assess` | 2 | CVE mapping and vulnerability assessment |
| `web_scan` | 2 | Web application scanning (nikto, ZAP) |
| `compliance` | 2 | ANSSI/NIST/CIS compliance checking |
| `risk_network` | 2 | Network risk scoring |
| `section_security` | 3 | Security report generation |

### Compliance Framework

**ANSSI (Primary):**
- Guide d'hygiène informatique (42 rules)
- TLS/SSL best practices
- French regulatory compliance

**NIST CSF:**
- 5 functions: Identify, Protect, Detect, Respond, Recover
- Control mapping

**CIS Controls v8:**
- 18 controls
- Implementation groups (IG1, IG2, IG3)

### Security Examples (`examples/security/`)

```
examples/security/
├── run_security_demo.sh     # Interactive demo with menu
├── local_network/           # Network scanning demo
├── web_audit/              # Web application audit
├── compliance_check/       # ANSSI/NIST/CIS compliance
└── config_audit/           # Firewall config analysis
```

### Audit Examples (`examples/audit/`)

Based on ACME-ERP/MSG-HUB enterprise architecture (4M messages/day):

```
examples/audit/
├── run_audit_demo.sh        # Interactive demo with 5 options
├── volumetry_analysis/      # Risk weighted by traffic volume
├── microservices/           # Service catalog & dependencies
├── java_monolith/           # Complexity & refactoring
└── full_audit/              # Comprehensive system audit
```

### Interactive Demo Features

Both demos (`run_security_demo.sh`, `run_audit_demo.sh`) provide:
- ASCII art banners and color-coded output
- Prerequisites checking
- Command-line options (`--help`, `--all`, `-1` to `-5`)
- Interactive menus
- Dependency graphs and risk matrices in ASCII

### Previous v0.61.0 (2025-12-16)

**Volumetry Kernels (retained):**

### New KOAS Kernels

#### `volumetry` (Stage 1)
Ingest operational volumetry data:
- Flow definitions (volume/day, peak patterns)
- Module-to-flow mapping
- Incident tracking
- Normalized scores (0-10 scale)

```yaml
# Example volumetry.yaml
flows:
  - name: MSG-HUB
    volume_day: 4_000_000
    peak_hour: 5
    peak_window: "00:00-10:00"
```

#### `module_group` (Stage 1)
Group files into functional modules:
- Regex-based path extraction
- Maven/Gradle multi-module support
- Aggregated metrics (LOC, classes, methods)

#### `risk_matrix` (Stage 2)
Volumetry-weighted risk assessment:
- Formula: `Risk = (LOC × 0.25) + (Complexity × 0.25) + (Volumetry × 0.50)`
- Critical path identification
- Incident-based risk boosting

### Security Network Audit Design

New kernel collection designed for network security audits:

| Kernel | Stage | Purpose |
|--------|-------|---------|
| `net_discover` | 1 | Asset enumeration (nmap, arp-scan) |
| `port_scan` | 1 | Service detection |
| `dns_enum` | 1 | DNS analysis |
| `config_parse` | 1 | Firewall/router config parsing |
| `vuln_assess` | 2 | CVE mapping (nuclei) |
| `ssl_analysis` | 2 | TLS/certificate audit (testssl.sh) |
| `compliance` | 2 | CIS/NIST/PCI-DSS checks |
| `risk_network` | 2 | Network risk scoring |
| `section_security` | 3 | Security report generation |

**Design document:** `docs/developer/SECURITY_NETWORK_KERNELS_DESIGN.md`

### Documentation

- `docs/developer/VOLUMETRY_KERNELS_DESIGN.md` — Volumetry kernel architecture
- `docs/KOAS.md` — Updated with volumetry integration

---

## v0.60.0 — MCP Enhancement, KOAS Parallel Execution & System Tools (2025-12-14)

### Highlights

**RAGIX MCP server gains 5 new tools, parallel KOAS execution, and comprehensive system introspection for industrial-scale code auditing.**

| Feature | Status |
|---------|--------|
| MCP Server v0.8.0 | ✅ 22 tools total |
| Parallel KOAS | ✅ koas_run(parallel=True) |
| AST Tools | ✅ ragix_ast_scan, ragix_ast_metrics |
| Model Management | ✅ ragix_models_list, ragix_model_info |
| System Info | ✅ ragix_system_info (GPU, CPU, memory) |
| French i18n | ✅ Proper UTF-8 diacritics |
| Test Suite | ✅ 18 new MCP tests |

### New MCP Tools (v0.8.0)

#### `ragix_ast_scan(path, language, include_private)`
Extract AST symbols from source code with fallback to Python AST:
- Classes, methods, functions, fields
- Symbol visibility tracking
- Multi-language support (auto-detection)

#### `ragix_ast_metrics(path, language)`
Compute code quality metrics:
- Total files, LOC, avg LOC/file
- Complexity hotspots identification
- Basic metrics fallback when ragix-ast unavailable

#### `ragix_models_list()`
List available Ollama models:
- Model name, size, family
- Recommended model selection
- Current model indicator

#### `ragix_model_info(model)`
Detailed model information:
- Parameter count, quantization
- Context length
- Capability inference (text, code, vision)

#### `ragix_system_info()`
Comprehensive system introspection:
- Platform (OS, Python version)
- CPU (cores, architecture)
- Memory (total, available)
- GPU (CUDA availability, devices, memory)
- Disk usage
- Ollama status

### KOAS Enhancement

#### Parallel Execution
`koas_run` now supports parallel kernel execution:
```python
koas_run(workspace, parallel=True, workers=4)
```
- Dependency-aware batching
- Stage-by-stage parallelization
- Duration tracking in response

#### French Report Fixes
Fixed UTF-8 diacritics in French audit reports:
- Méthodologie, Synthèse Exécutive, Complexité
- 50+ translation strings corrected
- Templates and drift analysis updated

### Claude Code Slash Commands

New and updated commands in `.claude/commands/`:
- `/koas-audit` — Updated with `--parallel` option
- `/ragix-system` — System info and deployment check
- `/ragix-models` — Model management and selection

### Test Suite

New `tests/test_mcp_server.py` with 18 tests:
- AST scan/metrics tests
- Model list/info tests
- System info tests
- KOAS parallel parameter tests
- Tool availability validation

### Files Modified

| File | Changes |
|------|---------|
| `MCP/ragix_mcp_server.py` | +400 lines, 5 new tools |
| `ragix_kernels/audit/report/i18n.py` | UTF-8 fixes |
| `ragix_kernels/audit/report/templates.py` | UTF-8 fixes |
| `ragix_kernels/audit/section_drift.py` | UTF-8 fixes |
| `tests/test_mcp_server.py` | New, 18 tests |
| `.claude/commands/*.md` | New/updated |
| `pyproject.toml` | Version 0.60.0 |

### Performance

Industrial-scale audit capability:
- **60K LOC Java project**: 3.4s full audit (parallel)
- **Stage 1**: ~2.1s (data collection)
- **Stage 2**: ~0.5s (analysis)
- **Stage 3**: ~0.02s (reporting)
- **Throughput**: 3-20 codebases/hour depending on size

### Migration from v0.59.0

No breaking changes. New MCP tools available immediately.

---

## v0.10.1 — Advanced Visualization & Live Explorer (2025-11-27)

### Highlights

**Interactive visualization suite for dependency analysis with live exploration capabilities.**

| Feature | Status |
|---------|--------|
| Enhanced HTML Renderer | ✅ Package clustering |
| DSM (Dependency Structure Matrix) | ✅ Heatmap + cycle detection |
| Radial Explorer | ✅ Ego-centric visualization |
| Standalone Radial Server | ✅ FastAPI live app |
| AST API Endpoints | ✅ 8 new REST endpoints |

### Tested on Production Codebase (Enterprise Client)

- **1,315 Java files** analyzed
- **18,210 symbols** extracted
- **45,113 dependencies** mapped
- **Technical debt:** 362.2 hours
- **Visualization outputs:**
  - Force-directed graph (827KB HTML)
  - Package-level DSM (254KB HTML)
  - Class-level DSM (84KB HTML)
  - Radial explorer (123KB HTML)

### New Features

#### Enhanced HTML Renderer (`ragix_core/ast_viz.py`)

Interactive D3.js force-directed graph with:
- **Package clustering** — Nodes grouped by Java package with convex hulls
- **Edge bundling** — Curved edges between clusters for clarity
- **Node coloring** — By type (class=blue, interface=green, method=orange)
- **Interactive controls** — Click to select, search, filter by type
- **Minimap** — Overview navigation for large graphs
- **SVG export** — Download current view

```bash
ragix-ast graph /path/to/project --format html --output deps.html
```

#### Dependency Structure Matrix (DSM)

Heatmap visualization for dependency analysis:
- **Cell color** — Indicates dependency strength
- **Cycle detection** — Red cells for bidirectional dependencies
- **Aggregation levels** — Package-level or class-level views
- **Export formats** — HTML, CSV, JSON

```bash
ragix-ast matrix /path/to/project --level package --output matrix.html
ragix-ast matrix /path/to/project --level class --csv  # Export as CSV
```

#### Radial Explorer (Ego-Centric Visualization)

Focus on a single class with dependencies radiating outward:
- **Ego-centric layout** — Selected class at center
- **Multi-level rings** — Concentric circles for Level 1, 2, 3 dependencies
- **Arc connections** — Colored by dependency type
- **Auto-selection** — Picks highest-connectivity class automatically
- **Interactive** — Click to select, double-click to refocus

```bash
ragix-ast radial /path/to/project --output radial.html  # Auto-select focal
ragix-ast radial /path/to/project --focal ClassName --levels 3 --output radial.html
```

#### Standalone Radial Server (`ragix_unix/radial_server.py`)

Lightweight FastAPI server for live exploration:

```bash
# Start the server
python -m ragix_unix.radial_server --path /path/to/project --port 8090

# Open in browser
xdg-open "http://localhost:8090/radial"
```

**Features:**
- Graph caching (builds once, serves many requests)
- Auto-selects highest-connectivity class as initial focal
- Real-time search with autocomplete
- Breadcrumb navigation for exploration history
- Adjustable depth levels (1-5)
- SVG export

**Endpoints:**
- `GET /` — Redirects to `/radial`
- `GET /api/info` — Project info (symbols, dependencies count)
- `GET /api/radial?focal=ClassName&levels=3` — Get radial graph data
- `GET /api/search?q=query` — Search for classes
- `GET /radial` — Interactive radial explorer page

#### AST API Endpoints (`ragix_web/server.py`)

8 new REST endpoints for programmatic access:

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

### CLI Commands

New and updated `ragix-ast` commands (now 12 total):

```bash
ragix-ast parse file.py --symbols      # Parse and show symbols
ragix-ast scan ./src --lang java       # Scan directory
ragix-ast deps ./src "ClassName"       # Show dependencies
ragix-ast search ./src "query"         # Pattern search
ragix-ast graph ./src --format html    # Force-directed graph
ragix-ast cycles ./src                 # Detect circular deps
ragix-ast metrics ./src                # Professional metrics
ragix-ast maven ./project              # Maven analysis
ragix-ast sonar project-key            # Sonar metrics
ragix-ast info                         # Show supported languages
ragix-ast matrix ./src --level package # DSM visualization (NEW)
ragix-ast radial ./src --focal Class   # Radial explorer (NEW)
```

### Files Added/Modified

| File | Description |
|------|-------------|
| `ragix_core/ast_viz.py` | HTMLRenderer + DSMRenderer + RadialExplorer (2700+ lines) |
| `ragix_core/dependencies.py` | Fixed import dependency source extraction |
| `ragix_core/__init__.py` | Added DSMRenderer, RadialExplorer exports |
| `ragix_unix/ast_cli.py` | Added matrix, radial commands (1000+ lines, 12 commands) |
| `ragix_unix/radial_server.py` | Standalone radial explorer server (800+ lines) |
| `ragix_web/server.py` | Added 8 AST API endpoints |
| `ragix_web/static/js/dependency_explorer.js` | D3.js component (600+ lines) |

### Bug Fixes

- **Import dependency source extraction** — Fixed incorrect source names in dependency graph
- **Name resolution in BFS** — Fixed short names vs qualified names mismatch
- **Structural type filtering** — Radial explorer now shows only classes, interfaces, enums

---

## v0.10.0 — AST Analysis, Code Metrics & Multi-Language Dependencies (2025-11-27)

### Highlights

**RAGIX gains professional-grade AST analysis for Python and Java, with dependency tracking, coupling metrics, and technical debt estimation.**

| Feature | Status |
|---------|--------|
| Multi-Language AST | ✅ Python + Java |
| Dependency Graph | ✅ Full tracking |
| AST Query Language | ✅ Pattern-based search |
| Code Metrics | ✅ Cyclomatic + Technical Debt |
| Maven Integration | ✅ POM parsing |
| Sonar Integration | ✅ API client |
| Interactive Visualization | ✅ HTML/D3.js |

### Tested on Production Codebase

Successfully analyzed **1,315 Java files** from a real enterprise project:
- **18,210 symbols** extracted
- **45,113 dependencies** mapped
- **362 hours** of technical debt estimated
- Analysis completed in **~10 seconds**

### New Features

#### AST Base & Multi-Language Support (`ragix_core/ast_base.py`)

Unified AST representation supporting Python, Java, and extensible to other languages:

```python
from ragix_core import ASTNode, NodeType, Language, get_ast_registry

registry = get_ast_registry()
backend = registry.get_backend(Language.PYTHON)
ast = backend.parse_file(Path("mycode.py"))
symbols = backend.get_symbols(ast)
```

- **Language enum** — Python, Java, JavaScript, TypeScript, Go, Rust, C, C++
- **NodeType enum** — MODULE, CLASS, INTERFACE, METHOD, FIELD, etc. (20+ types)
- **Visibility tracking** — PUBLIC, PRIVATE, PROTECTED, PACKAGE
- **Type information** — generics, arrays, optionals
- **Symbol extraction** — qualified names, locations, metadata

#### Python AST Backend (`ragix_core/ast_python.py`)

Uses Python's stdlib `ast` module:
- Classes with inheritance
- Functions and methods with parameters
- Import tracking (import and from...import)
- Decorators and type annotations
- Docstrings and visibility by naming convention

#### Java AST Backend (`ragix_core/ast_java.py`)

Uses `javalang` library for comprehensive Java parsing:
- Classes, interfaces, enums, annotations
- Methods, constructors, fields
- Generics and type parameters
- Annotations/decorators
- Modifiers (static, final, abstract)
- Method call extraction

#### Dependency Graph (`ragix_core/dependencies.py`)

Full dependency tracking across symbols:

```python
from ragix_core import DependencyGraph, build_dependency_graph

graph = build_dependency_graph([Path("./src")])
deps = graph.get_dependencies("MyClass.method")
dependents = graph.get_dependents("MyInterface")
cycles = graph.detect_cycles()
```

- **Dependency types** — import, inheritance, implementation, call, composition, annotation
- **Cycle detection** — find circular dependencies
- **Coupling metrics** — afferent/efferent coupling, instability index
- **Export formats** — DOT, Mermaid, JSON

#### AST Query Language (`ragix_core/ast_query.py`)

Pattern-based search for code symbols:

```bash
ragix-ast search ./src "type:class name:*Service"
ragix-ast search ./src "@Transactional"
ragix-ast search ./src "extends:Base*"
ragix-ast search ./src "!visibility:private"
```

Query predicates:
- `type:pattern` — Match node type (class, method, function)
- `name:pattern` — Match by name (wildcards supported)
- `extends:pattern` — Match classes extending
- `implements:pattern` — Match classes implementing
- `@annotation` — Match by decorator/annotation
- `visibility:public` — Match by visibility
- `!predicate` — Negate any predicate

#### Professional Code Metrics (`ragix_core/code_metrics.py`)

Industry-standard metrics for code quality assessment:

```bash
ragix-ast metrics ./project
```

Metrics calculated:
- **Cyclomatic complexity** — decision point counting
- **Cognitive complexity** — readability impact
- **Lines of code** — total, code, comments, blank
- **Technical debt** — estimated remediation effort in hours
- **Maintainability index** — 0-100 scale
- **Complexity hotspots** — top complex methods

#### Maven Integration (`ragix_core/maven.py`)

Parse Maven POM files for Java projects:

```bash
ragix-ast maven ./java-project --conflicts
```

- Project coordinates (groupId, artifactId, version)
- Dependency extraction with scopes
- Property resolution (${version} placeholders)
- Multi-module project support
- Dependency conflict detection

#### Sonar Integration (`ragix_core/sonar.py`)

Query SonarQube/SonarCloud for quality metrics:

```bash
ragix-ast sonar my-project --verbose
```

- Quality gate status
- Bugs, vulnerabilities, code smells
- Test coverage and duplication
- Security hotspots
- Issue filtering by severity

#### Interactive Visualization (`ragix_core/ast_viz.py`)

Generate visual dependency graphs:

```bash
ragix-ast graph ./src --format html --output deps.html
ragix-ast graph ./src --format dot --colors pastel
ragix-ast graph ./src --format mermaid
```

- **DOT format** — for Graphviz
- **Mermaid format** — for Markdown embedding
- **D3.js JSON** — for custom visualization
- **Interactive HTML** — zoomable, searchable, draggable nodes
- **Color schemes** — default, pastel, dark, monochrome

### CLI Commands

New `ragix-ast` command with subcommands:

```bash
ragix-ast parse file.py --symbols      # Parse and show symbols
ragix-ast scan ./src --lang java       # Scan directory
ragix-ast deps ./src "ClassName"       # Show dependencies
ragix-ast search ./src "query"         # Pattern search
ragix-ast graph ./src --format html    # Generate visualization
ragix-ast cycles ./src                 # Detect circular deps
ragix-ast metrics ./src                # Professional metrics
ragix-ast maven ./project              # Maven analysis
ragix-ast sonar project-key            # Sonar metrics
ragix-ast info                         # Show supported languages
```

### Dependencies

New optional dependencies:
```bash
pip install ragix[ast]  # javalang, jsonschema
```

### Files Added

- `ragix_core/ast_base.py` — Base AST types and registry
- `ragix_core/ast_python.py` — Python AST backend
- `ragix_core/ast_java.py` — Java AST backend
- `ragix_core/ast_query.py` — Query language
- `ragix_core/ast_viz.py` — Visualization renderers
- `ragix_core/dependencies.py` — Dependency graph
- `ragix_core/code_metrics.py` — Professional metrics
- `ragix_core/maven.py` — Maven POM parsing
- `ragix_core/sonar.py` — SonarQube client
- `ragix_unix/ast_cli.py` — AST CLI
- `tests/test_ast.py` — Comprehensive tests

---

## v0.8.0 — Plugin System, SWE Workflows & WASP Foundation (2025-11-26)

### Highlights

**RAGIX becomes a true platform with extensible plugins, chunked workflows for large codebases, and WASP sandbox abstraction.**

| Feature | Status |
|---------|--------|
| Plugin System | ✅ Implemented |
| Unified Tool Registry | ✅ Enhanced |
| SWE Chunked Workflows | ✅ Implemented |
| WASP Sandbox Abstraction | ✅ Foundation |
| Built-in Plugins | ✅ 2 examples |
| CLI Plugin Commands | ✅ Implemented |

### New Features

#### Plugin System (`ragix_core/plugin_system.py`)

RAGIX now supports extensible plugins for tools and workflows:

- **Plugin types** — `tool`, `workflow` (future: `agent`, `backend`, `search`)
- **Trust levels** — `builtin`, `trusted`, `untrusted` with capability restrictions
- **Safe loading** — explicit allowlist, capability-based permissions
- **Plugin manifest** — YAML-based definition with tools, workflows, dependencies

```yaml
# plugin.yaml example
name: json-validator
version: 1.0.0
type: tool
trust: builtin
capabilities:
  - file:read
tools:
  - name: validate_json
    entry: json_tools:validate_json
    parameters:
      - name: content
        type: string
        required: true
```

#### Unified Tool Registry Enhancement (`ragix_core/tool_registry.py`)

- **Provider tracking** — tools tagged with source (`builtin`, `plugin`, `mcp`, `wasm`)
- **Unified API** — same tools available via CLI, Web UI, MCP server
- **Plugin sync** — automatic registration of plugin tools
- **Export formats** — CLI-friendly and MCP-compatible exports

#### SWE Chunked Workflows (`ragix_core/swe_workflows.py`)

For large codebase operations:

- **Chunked processing** — split large file sets into manageable chunks
- **Checkpoint resumption** — save/restore workflow state across interruptions
- **Circuit breaker** — automatic pause on repeated failures
- **Progress tracking** — real-time progress and ETA estimation

```python
from ragix_core import FileProcessingWorkflow, ChunkConfig

workflow = FileProcessingWorkflow(
    workflow_id="review-2024",
    root_path=Path("./src"),
    file_patterns=["*.py"],
    config=ChunkConfig(chunk_size=50),
)
results = workflow.run_on_files()
```

#### WASP Sandbox Abstraction (`ragix_core/sandbox_base.py`, `wasm_sandbox.py`)

Foundation for WebAssembly tool execution:

- **BaseSandbox protocol** — unified interface for all sandbox types
- **SandboxConfig** — capability-based security model
- **ExecutionResult** — unified result format across backends
- **HybridSandbox** — routes to WASM or shell based on availability
- **WasmSandbox** — WASM execution (requires `wasmtime>=14.0.0`)

```python
from ragix_core import create_sandbox, SandboxType

# Create hybrid sandbox (WASM when available, shell fallback)
sandbox = create_sandbox("hybrid", root_path=Path.cwd())
result = sandbox.run("validate_json {...}")
```

#### Plugin CLI Commands

New `ragix plugin` subcommands:

```bash
ragix plugin list              # List available plugins
ragix plugin info <name>       # Show plugin details
ragix plugin load <name>       # Load a plugin
ragix plugin unload <name>     # Unload a plugin
ragix plugin create <name>     # Create new plugin scaffold
ragix tools                    # List all available tools
```

#### Built-in Example Plugins

Two example plugins in `plugins/`:

1. **json-validator** — JSON/YAML validation and diff tools
   - `validate_json` — validate and format JSON
   - `validate_yaml` — validate YAML, convert to JSON
   - `json_diff` — compare two JSON objects

2. **file-stats** — File and codebase statistics
   - `file_stats` — size, lines, encoding
   - `directory_stats` — file counts, sizes, types
   - `code_stats` — lines of code, comments, blanks

### Files Added/Modified

| File | Description |
|------|-------------|
| `ragix_core/plugin_system.py` | Plugin system (~600 lines) |
| `ragix_core/swe_workflows.py` | Chunked workflows (~650 lines) |
| `ragix_core/sandbox_base.py` | Sandbox abstraction (~400 lines) |
| `ragix_core/wasm_sandbox.py` | WASM sandbox (~450 lines) |
| `ragix_core/tool_registry.py` | Enhanced with providers (~200 lines added) |
| `ragix_core/cli.py` | Plugin commands (~350 lines added) |
| `plugins/json-validator/` | Example tool plugin |
| `plugins/file-stats/` | Example tool plugin |
| `pyproject.toml` | Version 0.8.0, added `wasm` optional dep |

### New Dependencies

```toml
[project.optional-dependencies]
wasm = ["wasmtime>=14.0.0"]  # Optional, for WASM sandbox
```

### Architecture

```
v0.8 Architecture:

                    ┌─────────────────────────────────────┐
                    │         ragix_core/cli.py           │
                    │    ragix plugin list/load/...       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       PluginManager                 │
                    │   - discover()                      │
                    │   - load_plugin()                   │
                    │   - get_tool()                      │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │ Tool Plugin │         │  Workflow   │         │  Built-in   │
   │  (trusted)  │         │   Plugin    │         │   Tools     │
   └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       Unified Tool Registry         │
                    │   - ToolProvider: builtin/plugin/mcp│
                    │   - export_for_cli()                │
                    │   - export_for_mcp()                │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │   CLI       │         │   Web UI    │         │ MCP Server  │
   │ ragix tools │         │ Streamlit   │         │   Claude    │
   └─────────────┘         └─────────────┘         └─────────────┘
```

### Migration from v0.7.1

- **No breaking changes** — all v0.7.1 features preserved
- **New imports** — plugin and workflow classes in `ragix_core`
- **Optional WASM** — `pip install ragix[wasm]` for WASP features
- **Plugin directory** — create `plugins/` in project or `~/.ragix/plugins/` global

---

## v0.9.0 — WASP Tools & Browser Runtime (2025-11-26)

### Highlights

**WASP (WebAssembly-ready Agentic System Protocol) delivers deterministic, sandboxed tools for RAGIX agents with browser-side execution capability.**

| Feature | Status |
|---------|--------|
| WASP Tools (Python) | ✅ 18 tools |
| WASP CLI | ✅ Implemented |
| wasp_task Protocol | ✅ Implemented |
| Browser Runtime (JS) | ✅ Implemented |
| Virtual Filesystem | ✅ Implemented |
| Test Suite | ✅ 73 tests |

### New Features

#### WASP Tools (`wasp_tools/`)

18 deterministic tools across three categories:

**Validation:**
- `validate_json` — Validate JSON with optional schema
- `validate_yaml` — Validate YAML with optional schema
- `format_json` — Format/prettify JSON
- `format_yaml` — Format/prettify YAML
- `json_to_yaml` — Convert JSON to YAML
- `yaml_to_json` — Convert YAML to JSON

**Markdown:**
- `parse_markdown` — Parse to structured AST
- `extract_headers` — Extract headers
- `extract_code_blocks` — Extract code blocks
- `extract_links` — Extract links
- `extract_frontmatter` — Extract YAML frontmatter
- `renumber_sections` — Renumber section headers
- `generate_toc` — Generate table of contents

**Search:**
- `search_pattern` — Regex pattern search
- `search_lines` — Search with line context
- `count_matches` — Count pattern matches
- `extract_matches` — Extract with groups
- `replace_pattern` — Replace matches

#### WASP CLI (`ragix-wasp`)

```bash
ragix-wasp list              # List available tools
ragix-wasp info <tool>       # Show tool details
ragix-wasp run <tool> <args> # Run tool directly
ragix-wasp validate <file>   # Validate manifest
ragix-wasp categories        # List categories
```

#### wasp_task Protocol (`ragix_core/orchestrator.py`)

New action type for agent protocol:

```json
{
  "action": "wasp_task",
  "tool": "validate_json",
  "inputs": {"content": "..."}
}
```

#### WASP Executor (`ragix_core/wasp_executor.py`)

- Tool registry and execution
- Input validation
- Timing and metrics
- Custom tool registration
- Prompt generation for agents

#### Browser Runtime (`ragix_web/static/js/`)

- `wasp_runtime.js` — Client-side tool execution
- `virtual_fs.js` — In-memory filesystem
- `browser_tools.js` — UI integration

### Files Added/Modified

| File | Description |
|------|-------------|
| `wasp_tools/__init__.py` | Tool registry (~150 lines) |
| `wasp_tools/validate.py` | Validation tools (~350 lines) |
| `wasp_tools/mdparse.py` | Markdown tools (~400 lines) |
| `wasp_tools/search.py` | Search tools (~300 lines) |
| `wasp_tools/manifest.yaml` | Tool definitions |
| `ragix_unix/wasp_cli.py` | WASP CLI (~300 lines) |
| `ragix_core/wasp_executor.py` | Executor (~280 lines) |
| `ragix_core/orchestrator.py` | wasp_task action |
| `ragix_web/static/js/wasp_runtime.js` | Browser runtime |
| `ragix_web/static/js/virtual_fs.js` | Virtual filesystem |
| `ragix_web/static/js/browser_tools.js` | UI integration |
| `tests/test_wasp_tools.py` | Tool tests |
| `tests/test_wasp_integration.py` | Integration tests |
| `docs/WASP_GUIDE.md` | Documentation |

### Architecture

```
Agent Action
    │
    ▼
┌─────────────────┐
│  WaspExecutor   │
│  - Registry     │
│  - Validation   │
│  - Timing       │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Python │ │Browser│
│wasp_  │ │Wasp   │
│tools/ │ │Runtime│
└───────┘ └───────┘
```

### Migration from v0.8.0

- **No breaking changes** — all v0.8.0 features preserved
- **New imports** — wasp_tools module, WaspExecutor class
- **New CLI** — `ragix-wasp` command
- **New action** — `wasp_task` in agent protocol

---

## [Unreleased] — v1.0

### Planned Features

- **WASM Tools** — Compile tools to WebAssembly
- **AST-aware Search** — tree-sitter integration
- **Agent Improvements** — Multi-step reasoning, memory
- **VS Code Extension** — IDE integration

---

## [Future] — v1.0+ Ideas

### Agent Improvements
- [ ] Autonomous multi-step reasoning with self-correction
- [ ] Memory and context persistence across sessions
- [ ] Agent specialization profiles (security, performance, refactoring)
- [ ] Inter-agent communication protocol

### Search & Retrieval
- [ ] Incremental index updates (watch mode)
- [ ] Cross-repository search federation
- [ ] AST-aware code search (tree-sitter.wasm)
- [ ] Natural language to code search

### Integration
- [ ] VS Code extension
- [ ] GitHub Actions integration
- [ ] GitLab CI/CD integration
- [ ] Jupyter notebook support

### Performance
- [ ] GPU acceleration for embeddings (CUDA/MPS)
- [ ] Distributed index sharding
- [ ] Response streaming for large outputs
- [ ] Persistent connection pooling for Ollama

### Security & Compliance
- [ ] Audit log export (JSON, CSV)
- [ ] Role-based access control for tools
- [ ] Secrets scanning integration
- [ ] SBOM generation for analyzed repos

---

## v0.7.1 — Unified Configuration & Compliance (2025-11-26)

### Highlights

**Response to external code review — consolidation release addressing gaps identified in v0.7.0.**

| Feature | Status |
|---------|--------|
| Unified config (`ragix.yaml`) | ✅ Implemented |
| Log hashing (SHA256) | ✅ Implemented |
| Log viewer in GUI | ✅ Implemented |
| `ragix` CLI commands | ✅ Implemented |
| Full MCP instantiation | ✅ Implemented |

### New Features

#### Unified Configuration (`ragix.yaml`)
- **Single config file** — all settings in one place
- **Environment variable overrides** — `RAGIX_*` variables take precedence
- **Backward compatibility** — legacy `UNIX_RAG_*` variables still work
- **Auto-discovery** — searches cwd, `.ragix/`, `~/.config/ragix/`
- **Data classes** — `RAGIXConfig`, `LLMConfig`, `MCPConfig`, `SafetyConfig`, etc.

```yaml
# ragix.yaml example
llm:
  backend: ollama
  model: mistral
safety:
  profile: dev
  air_gapped: false
  log_hashing: true
mcp:
  enabled: true
  port: 5173
```

#### Log Integrity (`ragix_core/log_integrity.py`)
- **ChainedLogHasher** — blockchain-style hash chain for logs
- **SHA256 signatures** — each entry includes hash of previous entry
- **Tamper detection** — verify chain integrity on demand
- **AuditLogManager** — unified audit logging with optional hashing
- **Log export** — download logs and hash files from GUI

#### Web UI Log Viewer (new tab in `ragix_app.py`)
- **Recent Entries** — color-coded by type (CMD, EDIT, EVENT, ERROR)
- **Search Logs** — filter by type and search pattern
- **Integrity Verification** — verify hash chain with one click
- **Export** — download log files and hash signatures

#### RAGIX CLI (`ragix` command)
- `ragix install` — setup environment, create directories, default config
- `ragix doctor` — comprehensive system diagnostics
- `ragix config` — show current configuration
- `ragix status` — quick status check
- `ragix logs [-n 50]` — view recent log entries
- `ragix verify` — verify log integrity
- `ragix mcp` — start MCP server
- `ragix web` — start web interface
- `ragix run` — start interactive agent
- `ragix upgrade` — upgrade instructions

#### Enhanced MCP Server (4 new tools)
- `ragix_config()` — get current configuration
- `ragix_verify_logs()` — verify log integrity
- `ragix_logs(n)` — get recent log entries
- `ragix_agent_step(prompt)` — config-aware agent execution

### Files Added/Modified

| File | Description |
|------|-------------|
| `ragix.yaml` | Sample unified configuration |
| `ragix_core/config.py` | Configuration loader (~350 lines) |
| `ragix_core/log_integrity.py` | Log hashing (~450 lines) |
| `ragix_core/cli.py` | CLI commands (~550 lines) |
| `ragix_app.py` | Added Logs page (~220 lines) |
| `MCP/ragix_mcp_server.py` | Added 4 new MCP tools |
| `pyproject.toml` | Updated version, added `ragix` entry point |

### Gap Analysis Summary (from external review)

| Review Point | v0.7.0 Status | v0.7.1 Status |
|--------------|---------------|---------------|
| Modular package | ✅ Exceeded | ✅ Maintained |
| MCP integration | ⚠️ Partial | ✅ Full |
| Multi-agent | ✅ Exceeded | ✅ Maintained |
| Hybrid retrieval | ✅ Full | ✅ Maintained |
| Web UI | ⚠️ Partial | ✅ Full (+ logs) |
| Reproducibility | ⚠️ Partial | ✅ CLI added |
| Security | ⚠️ Partial | ✅ Log hashing |
| WASP (WASM) | Planned | Deferred to v0.8 |

---

## v0.7.0 — Launcher, Web GUI & Multi-Agent Platform (2025-11-25)

### Highlights

**RAGIX evolves from a CLI tool to a complete multi-agent orchestration platform.**

| Metric | Value |
|--------|-------|
| New code | ~10,000+ lines |
| New modules | 12 |
| Workflow templates | 8 |
| LLM backends | 3 |

### New Features

#### Launcher & Environment (`launch_ragix.sh`)
- **Portable conda initialization** — searches `~/anaconda3`, `~/miniconda3`, `~/miniforge3`
- **Auto-environment creation** — creates `ragix-env` if missing
- **Dependency management** — installs from `environment.yaml` and `requirements.txt`
- **Ollama health check** — verifies status and lists available models with sizes
- **Interactive menu** — 6 options: GUI, Demo, MCP, Test, Shell, Status
- **Direct launch modes** — `./launch_ragix.sh gui|demo|mcp|test`

#### Web Interface (`ragix_app.py`)
- **Dashboard** — sovereignty status, model inventory, quick actions
- **Hybrid Search** — BM25 + Vector search with fusion strategy selector
- **LLM Chat** — direct conversation with local Ollama models
- **Workflow Browser** — view and launch 8 pre-built templates
- **System Monitor** — health checks, environment info, refresh controls
- **About Page** — architecture diagram, documentation links

#### LLM Backends (`ragix_core/llm_backends.py`)
- **SovereigntyStatus enum** — `SOVEREIGN`, `CLOUD`, `HYBRID`
- **OllamaLLM** — 🟢 100% local, no data leaves machine
- **ClaudeLLM** — 🔴 Anthropic API (optional, with sovereignty warnings)
- **OpenAILLM** — 🔴 OpenAI API (optional, with sovereignty warnings)
- **Factory functions** — `create_llm_backend()`, `get_backend_from_env()`
- **Automatic warnings** — logs sovereignty status on initialization

#### Real Integration Testing (`examples/test_llm_backends.sh`)
- **Actual Ollama calls** — not mocked, real API requests
- **Model comparison** — mistral vs granite3.1-moe speed benchmark
- **Response timing** — average response time per model
- **Speed ranking** — automated fastest-to-slowest ranking

### Configuration Files

| File | Purpose |
|------|---------|
| `environment.yaml` | Conda environment (Python 3.10-3.12, numpy, scipy) |
| `requirements.txt` | Full v0.7 dependencies (15+ packages) |
| `launch_ragix.sh` | One-command setup and launch |
| `ragix_app.py` | Streamlit web interface |

### Documentation Updates
- **README.md** — Added "Option A: Using the Launcher" installation
- **README.md** — Updated Quick Start with Web UI instructions
- **examples/README.md** — Added launcher quick start and web interface docs

---

## v0.6.0 — Production Monitoring & Resilience (2025-11-24)

### New Features

#### Monitoring (`ragix_core/monitoring.py`)
- **MetricsCollector** — counters, gauges, histograms, timers
- **HealthChecker** — pluggable health checks with status aggregation
- **AgentMonitor** — execution tracking, tool call statistics
- **RateLimiter** — token bucket algorithm for API protection
- **CircuitBreaker** — failure protection with recovery timeout
- **Built-in checks** — `check_ollama_health()`, `check_disk_space()`, `check_memory_usage()`

#### Resilience Patterns (`ragix_core/resilience.py`)
- **RetryConfig** — configurable retry with 4 backoff strategies
  - `CONSTANT`, `LINEAR`, `EXPONENTIAL`, `EXPONENTIAL_JITTER`
- **@retry / @retry_async** — decorators for automatic retry
- **FallbackChain** — ordered fallback execution
- **Timeout** — async timeout wrapper with cancellation
- **Bulkhead** — concurrency limiting (semaphore-based)
- **GracefulDegradation** — automatic fallback on failure

#### Caching (`ragix_core/caching.py`)
- **InMemoryCache** — LRU eviction with TTL support
- **DiskCache** — persistent JSON-based caching
- **LLMCache** — specialized for LLM responses with semantic keys
- **ToolResultCache** — caches deterministic tool outputs
- **Statistics** — hit rate, miss rate, eviction counts

### Integration
- All monitoring integrated into `GraphExecutor`
- Health checks available via MCP (`ragix_health` tool)
- Metrics exposed for external monitoring systems

---

## v0.5.0 — Core Orchestrator & Modular Tooling (2025-11-23)

### Highlights

**Major architectural refactoring: monolithic agent → modular ragix_core package.**

### New Package: `ragix_core/`

#### Agent System (`ragix_core/agents/`)
- **BaseAgent** — abstract base with capabilities enum
- **CodeAgent** — code analysis, editing, search
- **DocAgent** — documentation generation
- **GitAgent** — version control operations
- **TestAgent** — test execution and coverage
- **AgentCapability** — 12 capability types

#### Graph Execution (`ragix_core/agent_graph.py`, `graph_executor.py`)
- **AgentNode** — node with config, capabilities, status
- **AgentEdge** — transitions with conditions
- **AgentGraph** — DAG with validation
- **GraphExecutor** — async execution with dependency resolution
- **SyncGraphExecutor** — synchronous wrapper
- **StreamEvent** — real-time execution events

#### Workflow Templates (`ragix_core/workflow_templates.py`)
- **TemplateManager** — template registry and instantiation
- **8 built-in templates:**
  - `bug_fix` — locate, diagnose, fix, test
  - `feature_addition` — design, implement, test, document
  - `code_review` — quality and security review
  - `refactoring` — analyze, plan, refactor, verify
  - `documentation` — code analysis, doc generation
  - `security_audit` — SAST, dependency checks
  - `test_coverage` — coverage analysis, test generation
  - `exploration` — codebase mapping and analysis

#### Hybrid Search (`ragix_core/hybrid_search.py`, `bm25_index.py`)
- **BM25Index** — sparse keyword search with code tokenization
- **HybridSearchEngine** — BM25 + vector fusion
- **FusionStrategy** — 5 strategies:
  - `RRF` (Reciprocal Rank Fusion)
  - `WEIGHTED`
  - `INTERLEAVE`
  - `BM25_ONLY`
  - `VECTOR_ONLY`
- **Code-aware tokenization** — handles camelCase, snake_case, PascalCase

#### Embeddings & Vector Search
- **EmbeddingBackend** — abstract interface
- **SentenceTransformerBackend** — all-MiniLM-L6-v2 default
- **DummyEmbeddingBackend** — testing without ML deps
- **VectorIndex** — NumPy and FAISS implementations
- **Chunking** — Python, Markdown, Generic chunkers

#### Tool Infrastructure
- **ToolRegistry** — centralized tool management
- **ToolDefinition** — schema with permissions
- **ToolExecutor** — safe execution with logging
- **LLMAgentExecutor** — full agent loop with tool calling

#### Prompt Engineering (`ragix_core/prompt_templates.py`)
- **TaskType enum** — 10 task types
- **PromptTemplate** — structured templates with few-shot examples
- **detect_task_type()** — automatic task classification
- **build_prompt()** — context-aware prompt construction

### Existing Improvements
- **ShellSandbox** — enhanced command filtering
- **AgentLogger** — structured logging with levels
- **Profiles** — `safe-read-only`, `dev`, `unsafe` modes
- **Secrets vault** — encrypted storage for sensitive data

---

## v0.4.0 — MCP Integration & Unix Toolbox (2025-11-20)

### New Features
- Full **MCP server** (`MCP/ragix_mcp_server.py`)
  - Tools: `ragix_chat`, `ragix_scan_repo`, `ragix_read_file`
  - Compatible with Claude Desktop, Claude Code, Codex
- **ragix_tools.py** — sovereign Unix toolbox
  - `rt-find`, `rt-grep`, `rt-stats`, `rt-lines`, `rt-top`, `rt-replace`, `rt-doc2md`
- **Bash surrogates** — `rt.sh`, `rt-find.sh`, `rt-grep.sh`
- **Tool spec** — `MCP/ragix_tools_spec.json`

### Architecture
- Unified naming (RAGIX everywhere)
- Environment variables: `UNIX_RAG_MODEL`, `UNIX_RAG_SANDBOX`, `UNIX_RAG_PROFILE`
- Project overview pre-scan at startup
- Enhanced denylist enforcement

### Documentation
- Rewritten README.md
- Added README_RAGIX_TOOLS.md
- Added MCP/README_MCP.md
- Updated demo.md

---

## v0.3.0 — Original Release (2025-11)

### Features
- `unix-rag-agent.py` — main agent script
- JSON action protocol: `bash`, `bash_and_respond`, `edit_file`, `respond`
- Git awareness (status, diff, log)
- Sandboxed shell with denylist
- Structured logging (`.agent_logs/commands.log`)
- Basic Unix-RAG retrieval

---

## v0.2.0 — Experimental (2025-10)

- Shell sandbox drafts
- Local LLM integration (Ollama)
- Unix-RAG prompt engineering experiments

---

## v0.1.0 — Prototype (2025-09)

- First prototype: bash via LLM
- Pure sandbox experiment
- Hardcoded reasoning

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| **v0.68.0** | 2026-02-18 | Multi-format extract (46 exts), 8-act Memory Demo, CLI hardening (init/pull/serve) |
| **v0.67.0** | 2026-02-16 | KOAS Memory (17 MCP tools), Summary (12 kernels, Graph-RAG), Memory Pipe demo |
| **v0.66.0** | 2026-01-30 | Centralized Activity Logging, Broker Gateway, Demo setup |
| **v0.64.2** | 2026-01-29 | Boilerplate detection (changelog patterns), output path fix |
| **v0.64** | 2026-01-22 | Two-tier caching (LLM + kernel), 16x speedup on cached runs |
| **v0.63** | 2026-01-18 | KOAS/docs enhancement, --use-cache, dual LLM (Worker+Tutor), improved appendices |
| **v0.62** | 2025-12-20 | KOAS MCP Consolidation, Demo UI, Academic docs (MCP, REASONING) |
| **v0.61** | 2025-12-17 | Security Kernels (10), ANSSI/NIST/CIS compliance |
| **v0.60** | 2025-12-14 | MCP Enhancement, Parallel KOAS, System tools |
| **v0.10.1** | 2025-11-27 | Advanced Visualization, DSM, Radial Explorer |
| **v0.10** | 2025-11-27 | AST Analysis, Code Metrics, Multi-Language |
| **v0.9** | 2025-11-26 | WASP Tools, Browser Runtime |
| **v0.8** | 2025-11-26 | Plugin System, SWE Workflows, WASP Foundation |
| **v0.7.1** | 2025-11-26 | Unified config, log hashing, CLI |
| **v0.7** | 2025-11-25 | Launcher, Web GUI, LLM backends |
| **v0.6** | 2025-11-24 | Monitoring, resilience, caching |
| **v0.5** | 2025-11-23 | ragix_core package, workflows, hybrid search |
| **v0.4** | 2025-11-20 | MCP integration, Unix toolbox |

## Related Documents

| Document | Purpose |
|----------|---------|
| `V08_WASP_PLANNING.md` | Detailed v0.8 WASP specifications |
| `WASM.md` | WASM architecture rationale |
| `README.md` | Usage documentation |
| `MCP/README_MCP.md` | MCP integration guide |

---

*For detailed usage instructions, see [README.md](README.md).*
