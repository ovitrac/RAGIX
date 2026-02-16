# KOAS Memory Pipe Demo

**Purpose:** Demonstrate RAGIX Memory's `pipe` command — a single CLI invocation that ingests source files and retrieves relevant content for LLM context injection.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## What Is This Demo?

This demo shows how RAGIX can turn a large codebase into a **searchable memory store** and retrieve precisely the right context for an AI assistant — all without needing a running LLM, a vector database, or an internet connection.

### The Problem It Solves

When you ask an AI assistant about a large codebase:

- Feeding entire files overwhelms the context window (truncation, summarization loss)
- Feeding random snippets misses critical sections
- There is no persistence — every session starts from scratch

### The Solution: `ragix-memory pipe`

```
Source files ──→ chunk ──→ store ──→ search ──→ injection block ──→ AI assistant
                  │          │          │              │
            (paragraphs) (SQLite)   (FTS5/BM25)  (token-budgeted)
```

One command does it all:

```bash
ragix-memory --db project.db pipe "your question" --source file1.py file2.md --budget 3000
```

**Key properties:**
- **Deterministic** — no LLM needed, pure text processing + BM25 search
- **Idempotent** — re-running with same files creates zero new chunks (SHA-256 dedup)
- **Composable** — ingest status goes to stderr, injection block to stdout
- **Sovereign** — everything runs locally, data never leaves your machine

---

## Quick Start

```bash
cd demos/koas_pipe_demo
./run_demo.sh
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--query-only` | false | Skip ingest, query existing database |
| `--budget N` | 3000 | Max tokens per query result |
| `--db PATH` | `/tmp/koas_pipe_demo.db` | Database file path |

---

## What Happens During the Demo

### Step 1: Ingest + Query

The demo ingests 8 RAGIX source files into a SQLite memory store:

| File | What It Contains |
|------|-----------------|
| `docs/KOAS.md` | KOAS philosophy, 5 kernel families, 75 kernels |
| `docs/ARCHITECTURE.md` | System architecture, 3-stage pipeline |
| `ragix_kernels/base.py` | Kernel base class (`compute()` / `summarize()` interface) |
| `ragix_kernels/registry.py` | Auto-discovery registry |
| `ragix_kernels/reviewer/kernels/md_edit_plan.py` | Reviewer kernel (LLM-driven document editing) |
| `ragix_kernels/presenter/kernels/pres_slide_plan.py` | Presenter kernel (slide generation) |
| `ragix_kernels/summary/cli/summaryctl.py` | Summary CLI (Graph-RAG pipeline) |
| `ragix_core/memory/cli.py` | Memory CLI (this pipe feature itself) |

Each file is:
1. Read and hashed (SHA-256) for deduplication
2. Split at paragraph boundaries (`\n\n`)
3. Merged into chunks respecting a token budget (default: 1800 tokens/chunk)
4. Stored with auto-inferred tags (e.g., `markdown`, `python`, `docs`)
5. Indexed by SQLite FTS5 for full-text search

Then, a query runs against the store and returns the most relevant chunks.

### Steps 2-3: Recall-Only Queries

Once ingested, subsequent queries skip the ingest phase entirely. Different queries return different chunks — FTS5 BM25 ranking ensures the most relevant content surfaces for each question.

### Step 4: Idempotency Check

Re-running `pipe` with `--source` on already-ingested files shows "0 new chunks" — the SHA-256 dedup mechanism prevents duplicate storage.

---

## Expected Output

### Step 1 output (first run)

```
================================================================
  KOAS Memory Pipe Demo
  RAGIX — Retrieval-Augmented Generative Interactive eXecution
================================================================

Database : /tmp/koas_pipe_demo.db
Budget   : 3000 tokens per query
Mode     : full (ingest + query)

[check] Python and RAGIX...
  OK

[check] Source files...
  KOAS.md
  ARCHITECTURE.md
  base.py
  registry.py
  md_edit_plan.py
  pres_slide_plan.py
  summaryctl.py
  cli.py
  All 8 files found

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: Ingest + Query — "What is KOAS?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[pipe] Ingested 41 chunks from 8 files (0 skipped)
---
format_version: 1
total_matched: 2
budget_tokens: 3000
---

## Memory (Injected)

### [1/2] KOAS.md — chunk 0 (stm, score: unranked)
[path:/home/.../docs/KOAS.md chunk:0 lines:0-98]
# KOAS — Kernel-Oriented Audit System
...
(Content about KOAS philosophy, kernel families, pipeline stages)

### [2/2] ARCHITECTURE.md — chunk 3 (stm, score: unranked)
[path:/home/.../docs/ARCHITECTURE.md chunk:3 lines:156-220]
## Kernel Contract
...
(Content about kernel interfaces, compute/summarize, 3-stage pipeline)
```

The `## Memory (Injected)` block is the injection block — structured markdown ready to be prepended to an LLM conversation. Each section includes:
- **Provenance**: source file, chunk index, line range
- **Content**: the actual text from the chunk
- **Metadata header**: format version, match count, token budget

### Step 2 output (different query, different results)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: Recall-only — "How do kernels work?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---
format_version: 1
total_matched: 3
budget_tokens: 3000
---

## Memory (Injected)

### [1/3] base.py — chunk 0 (stm, score: unranked)
[path:/home/.../ragix_kernels/base.py chunk:0 lines:0-85]
...
(Kernel base class, compute() method, KernelInput dataclass)
```

Note: no `[pipe] Ingested...` line — ingest phase was skipped (no `--source`).

---

## How It Works (Technical Details)

### Pipeline Architecture

```
                         ragix-memory pipe
                    ┌──────────┴──────────┐
                    │                     │
              [Phase 1: Ingest]    [Phase 2: Recall]
                    │                     │
            ┌───────┴───────┐     ┌───────┴────────┐
            │               │     │                │
        read files     chunk & store   FTS5 search   format
            │               │     │                │
        SHA-256 hash   paragraph-aware  BM25 rank   token budget
        dedup check    splitting        injectable   injection block
            │               │     filter           │
            └───────┬───────┘     └────────┬───────┘
                    │                      │
                 stderr                  stdout
              (status msgs)          (injection block)
```

### What Is KOAS?

KOAS (**Kernel-Oriented Audit System**) is the processing architecture behind RAGIX. Key concepts:

- **Kernel**: a self-contained processing unit with a `compute()` method and a `summarize()` method
- **3-stage pipeline**: Collection (S1) gathers raw data, Analysis (S2) processes it, Reporting (S3) produces human-readable output
- **5 kernel families**: Docs, Audit, Reviewer, Presenter, Summary — each with 8-17 specialized kernels
- **75 total kernels**: all following the same interface, discoverable via auto-registry

This demo uses `pipe` to search through KOAS source code and documentation, retrieving exactly the chunks needed to explain any aspect of the architecture.

### FTS5/BM25 Search

The recall engine uses SQLite FTS5 (Full-Text Search version 5) with BM25 ranking:

- **Indexing**: title, content, tags, and entities are indexed at ingest time
- **Querying**: all query terms must appear in the chunk (AND logic)
- **Ranking**: BM25 (Best Match 25) scores documents by term frequency and inverse document frequency
- **No embeddings needed**: this is pure keyword-based search, making it fast and deterministic

### Anti-Context-Poisoning

Raw ingested chunks are marked `injectable=false` by default. Only two paths make items injectable:

1. **`pipe` command**: sets `injectable=true` (interactive use, user chose the files)
2. **`consolidate` command**: promotes STM items to MTM/LTM with `injectable=true` (production pipeline)

The `recall` command and injection block formatter both filter on `injectable=true`, preventing unvetted content from entering an LLM context.

---

## Composability with LLMs

The pipe output is designed to be composed with LLM tools:

```bash
# Save injection block to a file, then reference in Claude Code
ragix-memory --db project.db pipe "authentication flow" \
    --source src/**/*.py --budget 2000 > /tmp/context.md

# Or pipe directly (when LLM tool supports stdin)
ragix-memory --db project.db pipe "database schema" --budget 1500 | \
    some-llm-tool --context-stdin "Explain the database design"
```

The `format_version: 1` header and `## Memory (Injected)` structure are designed for LLM consumption — the model knows this is retrieved context, not user instructions.

---

## Cleanup

```bash
# Remove the demo database
rm /tmp/koas_pipe_demo.db
```

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [docs/KOAS.md](../../docs/KOAS.md) | KOAS architecture and philosophy |
| [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) | System architecture reference |
| [docs/KOAS_MEMORY_MCP.md](../../docs/KOAS_MEMORY_MCP.md) | Memory MCP server (17 tools) |
| [docs/developer/ROADMAP_MEMORY_PIPE.md](../../docs/developer/ROADMAP_MEMORY_PIPE.md) | Pipe implementation roadmap (v3.0) |
