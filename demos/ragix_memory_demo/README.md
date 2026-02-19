# RAGIX Memory — Full Lifecycle Demo

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## Purpose

This demo walks through the **complete ragix-memory lifecycle** in 9 acts,
using the entire `docs/` corpus as a real-world document collection.
It demonstrates that the memory CLI is a self-contained, Unix-composable
system for knowledge ingestion, retrieval, management, LLM-augmented
reasoning, and **iterative loop convergence** — with both local (Ollama)
and cloud (Claude) backends.

**Audience:** developers evaluating RAGIX, demo sessions, CI smoke tests.

---

## Quick Start

```bash
cd demos/ragix_memory_demo
./run_demo.sh
```

To keep the workspace for manual exploration:

```bash
./run_demo.sh --keep
```

To run without any LLM (Acts 1–7 only, skips Acts 8–9):

```bash
./run_demo.sh --no-llm
```

---

## Options

| Flag              | Default                     | Description                           |
|-------------------|-----------------------------|---------------------------------------|
| `--keep`          | cleanup                     | Keep workspace after demo             |
| `--workspace DIR` | `/tmp/ragix_memory_demo`    | Custom workspace path                 |
| `--budget N`      | `2000`                      | Token budget for recall queries       |
| `--skip-ingest`   | false                       | Skip Act 2 (reuse existing DB)        |
| `--corpus DIR`    | `docs/`                     | Source corpus directory                |
| `--model MODEL`   | `granite3.1-moe:3b`        | Ollama model for Acts 8–9             |
| `--no-llm`        | false                       | Skip Acts 8–9 (no LLM required)      |
| `--help`          |                             | Show usage                            |

---

## The 9 Acts

### Act 1 — Init

Creates a fresh memory workspace: SQLite database, config, gitignore.

```
ragix-memory init /tmp/ragix_memory_demo
```

### Act 2 — Ingest Corpus

Ingests all documents from `docs/` into memory. Files are chunked at
paragraph boundaries, tagged automatically from paths and extensions,
and stored with SHA-256 content hashes for deduplication.

```
ragix-memory --db $DB ingest --source docs/ --injectable --format auto --tags ragix,documentation
```

### Act 3 — Search & Recall

Runs 4 diverse queries against the ingested corpus, demonstrating
FTS5/BM25 full-text search:

- "What is KOAS and how do kernels work?"
- "memory architecture and consolidation"
- "MCP tools and server integration"
- "sovereign AI and local-first design"

### Act 4 — Idempotency Proof

Re-ingests the same corpus. The SHA-256 dedup ensures **zero new chunks**
are created — proving idempotent ingestion.

### Act 5 — Pull (Capture)

Pipes a simulated LLM response into `ragix-memory pull`, storing it as
a tagged note. Then searches for it to confirm retrieval alongside
the original corpus.

```bash
echo "Summary text..." | ragix-memory --db $DB pull --tags demo,summary --title "Demo Summary"
```

### Act 6 — Stats & Palace

Displays store statistics (items by tier, type, corpora) and the memory
palace spatial view.

### Act 7 — Export & Compose

Exports all memory items as JSONL, demonstrating Unix pipe composability:

```bash
ragix-memory --db $DB export | wc -l        # count items
ragix-memory --db $DB export | jq '.title'   # extract titles
```

### Act 8 — LLM Reasoning over Memory

The flagship act: pipes token-budgeted recall into LLM backends for
complex reasoning tasks. Two sections demonstrate the same Unix pattern
with different backends:

**Section A — Ollama (local, sovereign)**

Two questions answered by a local Granite model with no cloud dependency:

- Q1: "What is RAGIX?" — 5 bullet points
- Q2: "KOAS kernels inventory & ranking" — audit usefulness

```bash
ragix-memory --db $DB pipe "RAGIX architecture" --budget 3000 \
    | { cat; echo '---'; echo 'Explain what RAGIX is in 5 bullets.'; } \
    | ollama run granite3.1-moe:3b
```

**Section B — Claude (cloud, powerful reasoning)**

Two harder questions requiring deeper analysis:

- Q3: "MCP tools & skills taxonomy" — classify all tools into categories
- Q4: "Is RAGIX an agent?" — evidence-based argument from docs

```bash
ragix-memory --db $DB pipe "MCP tools skills" --budget 4000 \
    | claude -p "Enumerate all MCP tools RAGIX provides. Classify into categories."
```

Each question shows: elapsed time, estimated tokens, tok/s rate.
Graceful degradation: if Ollama or Claude is unavailable, that section
is skipped with a hint on how to install.

**Important:** Prompts explicitly instruct the LLM to use ONLY the piped
`docs/` content — not system instructions, CLAUDE.md, or prior knowledge.
This ensures answers are grounded exclusively in the ingested corpus.

### Act 9 — Iterative Loop (Bounded Recall-Answer Loop)

Demonstrates the bounded recall-answer loop in two complementary parts.
The loop controller calls an LLM repeatedly, parses its protocol output,
and monitors answer similarity. Five stopping conditions are enforced:

1. **LLM stop** — the LLM signals `stop: true` (JSON protocol)
2. **Fixed point** — consecutive answers exceed similarity threshold
3. **Query cycle** — the LLM re-requests a previously tried query
4. **No new items** — recall returns nothing new
5. **Max calls** — iteration budget exhausted (safety bound)

#### Act 9a — Deterministic Loop (mock LLM, JSON protocol)

A deterministic bash mock LLM follows a scripted 3-iteration sequence.
No external LLM needed — always runs, fully reproducible.

```bash
ragix-memory --db $DB pipe "RAGIX data flow" --budget 3000 \
    | ragix-memory --db $DB loop --llm "bash mock_llm.sh" \
        --protocol json --max-calls 5 --threshold 0.92 \
        --trace-file loop_mock_trace.jsonl
```

```
iter=1  query=memory consolidation tier promotion       sim=  —    -> continue
iter=2  query=injection block format token budget pipe  sim=0.172  -> continue
iter=3  query=—                                         sim=0.167  -> llm_stop
```

The mock LLM requests refinements on iterations 1–2, then signals stop.
The controller traces every iteration, enforces bounds, and terminates
cleanly. This proves the loop mechanism works correctly, deterministically,
and reproducibly — without any LLM dependency.

#### Act 9b — Live LLM Loop (real model, passive protocol)

Same loop mechanism, real LLM. Uses `--protocol passive` (no JSON
parsing — the LLM just answers, and the controller monitors stability).

```bash
ragix-memory --db $DB pipe "data flow ingestion" --budget 3000 \
    | ragix-memory --db $DB loop --llm "ollama run granite3.1-moe:3b" \
        --protocol passive --max-calls 3 --threshold 0.85 \
        --similarity lexical --no-recall --trace-file loop_live_trace.jsonl
```

```
iter=1  sim=  —    method=      —  -> continue
iter=2  sim=0.146  method=lexical  -> continue
iter=3  sim=0.151  method=lexical  -> max_calls
```

With small models (3B) at default temperature, each generation produces
different phrasings (lexical similarity ~0.15), so the safety bound
(`max_calls`) is the expected termination. The key demonstration: the
loop controller monitors stability and enforces bounds regardless of
the LLM backend. Larger models or lower temperature achieve tighter
convergence.

---

## Architecture

```
                          ragix-memory CLI
                    ┌──────────┴──────────┐
              [Ingest Pipeline]     [Recall Pipeline]
                    │                     │
          ┌─────────┴─────────┐     ┌─────┴──────┐
          │                   │     │            │
     read_file_text()    chunk_paragraphs()   FTS5/BM25
     (text + binary)     SHA-256 dedup     search + rank
          │                   │     │            │
          │              write_item()     format_injection_block()
          │                   │                  │
          └─────────┬─────────┘          ┌───────┘
                    │                    │
               SQLite + FTS5 Index    stdout ──┐
                    │                          │
         ┌──────────┼──────────┐         ┌─────┴─────┐
         │          │          │         │           │
       stats     palace     export    Ollama      Claude
                                    (local)     (cloud)
```

### LLM Integration Pattern

The key insight: **one Unix pipe connects memory to any LLM**.

```
ragix-memory pipe "topic" --budget N  →  stdout (injection block)
                                            │
                              ┌─────────────┼─────────────┐
                              │             │             │
                      ollama run model  claude -p     ragix-memory loop
                      (single-shot)     (single-shot)  (iterative, bounded)
                              │             │             │
                              └─────────────┼─────────────┘
                                            │
                                      ragix-memory pull
                                      (capture to memory)
```

### Supported File Formats

| Category     | Extensions                                        | Method            |
|--------------|---------------------------------------------------|-------------------|
| Markdown     | `.md`                                             | Direct read       |
| Text         | `.txt`, `.rst`, `.csv`, `.tsv`                    | Direct read       |
| Source code  | `.py`, `.java`, `.js`, `.ts`, `.go`, `.rs`, etc.  | Direct read       |
| Config       | `.yaml`, `.yml`, `.json`, `.toml`, `.xml`         | Direct read       |
| Web          | `.html`, `.htm`, `.css`                           | Direct read       |
| Shell        | `.sh`, `.bash`, `.sql`                            | Direct read       |
| Office docs  | `.docx`, `.odt`                                   | python-docx/odfpy |
| Slides       | `.pptx`, `.odp`                                   | python-pptx/odfpy |
| Spreadsheets | `.xlsx`, `.ods`                                   | openpyxl/odfpy    |
| PDF          | `.pdf`                                            | pdftotext/poppler |

Install Office format support: `pip install ragix[docs]`

---

## Technical Details

- **Chunking:** paragraph-bounded, ~1800 tokens per chunk (configurable via `--chunk-tokens`)
- **Dedup:** SHA-256 file hash stored in `corpus_hashes` table; re-ingest is a no-op for unchanged files
- **Search:** SQLite FTS5 with BM25 ranking; supports tier/type filtering
- **Recall:** token-budgeted injection blocks formatted for LLM context windows
- **Storage:** SQLite with WAL mode; all items have provenance (source path, chunk ID, content hash)
- **LLM grounding:** prompts explicitly restrict answers to the piped corpus content only

---

## Expected Output (Abbreviated)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RAGIX Memory — Full Lifecycle Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [info]  Checking prerequisites...
  [ok]    ragix-memory importable
  [ok]    Corpus: 21 files (638K) in .../docs
  [ok]    Ollama: model 'granite3.1-moe:3b' available
  [ok]    Claude CLI available

  ┌─────────────────────────────────────────────────────────────┐
  │  Act 1 — Initialize Memory Workspace
  └─────────────────────────────────────────────────────────────┘

  [ok]    Workspace initialized

  ┌─────────────────────────────────────────────────────────────┐
  │  Act 2 — Ingest Corpus into Memory
  └─────────────────────────────────────────────────────────────┘

  [demo]  Ingested 157 chunks from 21 file(s) (0 skipped)
  [ok]    Ingestion complete

  ...

  ┌─────────────────────────────────────────────────────────────┐
  │  Act 8 — LLM Reasoning over Memory
  └─────────────────────────────────────────────────────────────┘

  ── Ollama / granite3.1-moe:3b  (local, sovereign) ──

  Q1: What is RAGIX? (5 bullets)
  $ ragix-memory --db $DB pipe "RAGIX architecture" --budget 3000 \
      | { cat; echo '---'; echo 'prompt'; } | ollama run granite3.1-moe:3b

    • RAGIX is a local-first development assistant...
    • Uses SQLite FTS5 for retrieval...
    [ok]  Answer received (12 lines, ~340 tok, 8s, ~42 tok/s)

  ...

  ── Claude  (cloud, powerful reasoning) ──

  Q3: MCP tools & skills taxonomy
  $ ragix-memory --db $DB pipe "MCP tools skills" --budget 4000 \
      | claude -p "Enumerate all MCP tools..."

    ## Memory Management
    - memory_store, memory_recall, memory_search...
    [ok]  Answer received (45 lines, ~1200 tok, 22s, ~54 tok/s)

  ...

  ┌─────────────────────────────────────────────────────────────┐
  │  Act 9 — Iterative Loop — Bounded Recall-Answer Loop
  └─────────────────────────────────────────────────────────────┘

  ── 9a: Deterministic Loop (mock LLM, JSON protocol) ──

  Q5a: RAGIX data flow analysis (mock, deterministic)
  $ ragix-memory --db $DB pipe "topic" --budget 3000 \
      | ragix-memory --db $DB loop --llm "bash mock_llm.sh" \
          --protocol json --max-calls 5 --threshold 0.92 ...

    # RAGIX Memory Data Flow — Complete Analysis
    ## 1. Document Ingestion ...
    ## 2. Recall & Retrieval ...
    [ok]  Answer: 31 lines, ~384 tok, 1s, ~384 tok/s

    Convergence trace (deterministic):
      iter=1  query=memory consolidation tier promotion       sim=  —    -> continue
      iter=2  query=injection block format token budget pipe  sim=0.172  -> continue
      iter=3  query=—                                         sim=0.167  -> llm_stop
    [ok]  Mock loop completed: 3 iteration(s), deterministic, reproducible

  ── 9b: Live LLM Loop (ollama/granite3.1-moe:3b, passive) ──

  Q5b: Data flow analysis (live LLM, passive)
    Data Flow from Document Ingestion to LLM Context Injection...
    [ok]  Answer: 21 lines, ~638 tok, 103s, ~6 tok/s

    Convergence trace (live):
      iter=1  sim=  —    method=      —  -> continue
      iter=2  sim=0.146  method=lexical  -> continue
      iter=3  sim=0.151  method=lexical  -> max_calls
    [ok]  Live loop completed: 3 iteration(s)

  ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Demo Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Summary Dashboard
  ─────────────────────────────────────────
  Workspace:      /tmp/ragix_memory_demo
  Corpus:         21 top-level docs (5.4M)
  JSONL records:  336
  Search queries: 4
  LLM questions:  6 (Ollama + Claude + mock + loop)
  Loop iters:     6 (3 mock + 3 live)
  Total time:     ~120s
  Dedup:          SHA-256 (idempotent)
  Search engine:  FTS5/BM25
  Local LLM:      granite3.1-moe:3b (Ollama)
  Cloud LLM:      Claude
  ─────────────────────────────────────────
```

---

## Cleanup

The workspace is automatically removed unless `--keep` is used.
To manually clean:

```bash
rm -rf /tmp/ragix_memory_demo
```

---

## Known Limitations

- **Claude system instructions:** When using `claude -p`, the Claude CLI
  loads its own system instructions (CLAUDE.md). The demo prompts explicitly
  tell the LLM to ignore these and use only the piped content. If you observe
  references to CLAUDE.md in answers, this is a system instruction leak —
  the grounding prompt can be strengthened further.
- **Ollama cold start:** The first query to Ollama may be slower if the model
  needs to be loaded into memory. Subsequent queries are faster.
- **Token estimates:** Token counts are approximated as `characters / 4`,
  not actual tokenizer counts.
- **Loop convergence:** Act 9 uses `--protocol passive` (fixed-point only)
  with `--similarity lexical`. Small models (3B) at default temperature
  produce lexical similarity of ~0.15–0.20 between runs, so the `max_calls`
  safety bound is the expected termination. Larger models or lower temperature
  achieve tighter convergence. The full `--protocol json` mode (where the
  LLM proposes refined queries) requires models that can follow the
  JSON-first-line protocol consistently.

---

## Related Documentation

| Document                    | Description                          |
|-----------------------------|--------------------------------------|
| `docs/KOAS_MEMORY_MCP.md`  | Memory MCP tools reference           |
| `docs/ARCHITECTURE.md`     | RAGIX system architecture            |
| `docs/CLI_GUIDE.md`        | Full CLI reference                   |
| `demos/koas_pipe_demo/`    | Memory pipe demo                     |
