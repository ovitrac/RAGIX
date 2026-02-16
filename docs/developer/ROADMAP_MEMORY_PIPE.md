# ROADMAP: Memory Pipe — Ingest + Recall + Pipe CLI

**Version**: 3.0 (revised 2026-02-16)
**Author**: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## Motivation

Claude Code + direct piping breaks on large inputs because:

- stdin gets truncated / summarized unpredictably
- crucial sections are missed (the "200 lines cap" effect)
- no persistence — all context is lost next session

**Solution**: pipe large files into **RAGIX Memory** (chunked, deduplicated, safe), then use **recall** to inject token-budgeted, relevance-ranked context on demand.

```
                                   ┌─── ragix-memory pipe ───┐
                                   │                         │
large files ──→ ingest ──→ memory.db ──→ FTS5 recall ──→ injection block ──→ Claude
                  │              │                              │
             (idempotent)   (SQLite)                      (token-budgeted)
```

---

## Decision Record

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chunking | Paragraph-aware (`\n\s*\n`) | Preserves semantic boundaries, reuses pattern from `md_chunk.py` |
| Overlap | None | Recall engine handles relevance; overlap doubles storage for marginal gain |
| Storage type | `note` + `provenance.source_id` | Consistent with existing items in the store |
| Tier | `STM` (ingest) / injectable (pipe) | Raw chunks are unverified; `pipe` marks injectable for immediate use |
| Injectable | `false` by default (ingest) | **Anti-context-poisoning**: raw chunks never injected unless explicitly requested |
| Dedup | SHA-256 via `corpus_hashes` table | Already in `MemoryStore`; re-ingest is idempotent |
| stdin | Slurp (not streaming) | Piped files are finite |
| Candidate generation | FTS5 `search_fulltext()` | BM25-ranked shortlist, then re-ranked with tag/embedding/provenance scoring |
| Hooks | Deferred to M3 | Claude Code hooks don't expose per-turn user message or context injection |
| `ragix-claude ask` | Deferred to M3 | Depends on stable `claude --inject-file` API or equivalent |

---

## M1: Two Composable CLI Commands — DONE

### M1a: `ragix-memory recall`

Thin CLI wrapper: dispatcher search (FTS5) → read enrichment → injectable filter → `format_injection_block()`.

```bash
ragix-memory --db memory.db recall "compliance risks" --budget 2500
ragix-memory --db memory.db recall "remediation plan" --no-header
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `query` | str | required | Natural language search query (positional) |
| `--budget` | int | 1500 | Max tokens for injection block |
| `-w` | str | None | Named workspace |
| `--tier` | str | None | Filter by tier (stm/mtm/ltm) |
| `--no-header` | flag | false | Strip the `format_version` / metadata header |

### M1b: `ragix-memory ingest`

Chunks large files into memory items. Raw chunks are `injectable=false` by default.

```bash
ragix-memory --db memory.db ingest --source docs/*.md --tags rie --format auto
ragix-memory --db memory.db ingest --source docs/*.md --injectable   # expert only
cat report.md | ragix-memory --db memory.db ingest --source - --tags stdin
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--source` | str(s) | required | File paths or `-` for stdin |
| `--chunk-tokens` | int | 1800 | Max tokens per chunk (`len(text) // 4`) |
| `--tags` | str | None | Comma-separated tags applied to all chunks |
| `--format` | str | `text` | `text` (plain) / `auto` (infer type, tags, title from path/content) |
| `--injectable` | flag | false | Mark chunks as injectable (expert only) |
| `--scope` | str | `audit` | Scope for items |
| `--corpus` | str | None | Corpus ID |

### M1.5: Consolidation injectable fix — DONE

Verified and fixed: `consolidate.py` now explicitly sets `injectable=True` on promoted items (MTM/LTM). Three paths covered:

1. **Merge path** (`consolidation_merge`): `merged.injectable = promote`
2. **Delta merge path** (`delta_consolidation_merge`): `merged.injectable = promote`
3. **Single-item promotion** (`_try_promote`): `update_item(..., injectable=True)`

The anti-context-poisoning pipeline works:

```bash
ragix-memory ingest --source docs/*.md --tags rie    # injectable=false
ragix-memory consolidate --scope audit               # promote → injectable=true
ragix-memory recall "compliance" --budget 2500       # returns only injectable items
```

### Implementation files (M1)

| File | Action | Lines |
|------|--------|-------|
| `ragix_core/memory/ingest.py` | NEW — `chunk_paragraphs()`, `ingest_file()`, `ingest_stdin()` | 254 |
| `ragix_core/memory/cli.py` | EDIT — `cmd_recall()`, `cmd_ingest()`, `cmd_pipe()` + parsers | +170 |
| `ragix_core/memory/tests/test_ingest.py` | NEW — 18 tests | 226 |
| **Total** | | **~650** |

### Tests: 18/18 new + 511 total (zero regression)

| Test | Verified |
|------|----------|
| `test_chunk_paragraphs_basic` (4 tests) | Splits, token limit, empty, line ranges |
| `test_chunk_paragraphs_merge` (2 tests) | Small merged, large emitted alone |
| `test_ingest_idempotent` (2 tests) | Skip same, re-ingest modified |
| `test_ingest_injectable_default` (2 tests) | Default false, flag respected |
| `test_ingest_provenance` (2 tests) | Fields present, content header |
| `test_ingest_format_auto` (2 tests) | Tags inferred, title inferred |
| `test_recall_format_version` (2 tests) | Header present, empty returns empty |
| `test_recall_excludes_non_injectable` (2 tests) | Filter works, all-false yields empty |

---

## M2: `ragix-memory pipe` — Unified Command — DONE

### The problem M2 solves

M1 required 3 commands for a full cycle: `ingest` → `consolidate` → `recall`. For interactive Claude Code usage, a single command is needed.

### Design

`pipe` = `ingest` (if `--source` given) + `recall` (always), in one invocation:

- Ingest uses `--format auto` and `injectable=True` (no consolidation needed)
- Recall uses FTS5 via the fixed RecallEngine (Step 1 bug fix)
- Ingest status goes to stderr, injection block goes to stdout

```bash
# Full pipeline: ingest + recall
ragix-memory --db project.db pipe "KOAS architecture" --source ragix_core/memory/*.py --budget 2000

# Recall-only (no --source): query existing DB
ragix-memory --db project.db pipe "write governance" --budget 1500

# Re-run is idempotent (skips already-ingested files)
ragix-memory --db project.db pipe "consolidation promote" --source ragix_core/memory/*.py --budget 2000
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `query` | str | required | Natural language query (positional) |
| `--source` | str(s) | None | File paths to ingest (optional, idempotent) |
| `--budget` | int | 2000 | Token budget for injection block |
| `-w` | str | None | Named workspace |
| `--tier` | str | None | Filter by tier |
| `--chunk-tokens` | int | 1800 | Max tokens per chunk |
| `--tags` | str | None | Comma-separated tags |
| `--scope` | str | `audit` | Item scope |

### Key difference from `ingest` + `recall`

| Feature | `ingest` + `recall` | `pipe` |
|---------|-------------------|--------|
| Commands | 2 (or 3 with consolidate) | 1 |
| Injectable default | `false` | `true` |
| Format mode | configurable | always `auto` |
| Consolidation needed | yes (for injectable) | no |
| Use case | Production pipeline | Interactive Claude Code |

---

## Bug Fix: RecallEngine FTS5 Candidate Path — DONE

### Problem

`RecallEngine.search()` called `store.list_items()` (plain SQL scan) then scored with `w_tag * tag_overlap + w_emb * emb_score + w_prov * prov_score`. With mock embedder (no stored embeddings), all items scored identically → every query returned the same results.

### Fix

When a non-empty `query` is provided, use `store.search_fulltext()` (FTS5 BM25) as the candidate generator. The FTS5 shortlist is then re-ranked by the existing tag/embedding/provenance scoring pipeline.

Applied to both `search()` and `search_with_scores()` in `recall.py`.

### Verified

```
Q: PolicyVerdict quarantine rejected     → ['policy [1/2]', 'tools [1/2]']
Q: consolidate promote stm mtm          → ['consolidate [5/5]', 'consolidate [1/5]', 'tools [3/5]']
Q: MemoryItem tier provenance            → ['consolidate [4/5]', 'ingest [2/2]', ...]
```

FTS5 returns differentiated, query-relevant results.

---

## M3: Hooks + Claude Integration (deferred)

### Why deferred

Claude Code hooks (as of 2026-02) provide:

| Hook | Trigger | Problem |
|------|---------|---------|
| `PreToolUse` | Before each tool call | Fires per tool, not per turn |
| `PostToolUse` | After each tool call | No user message in env |
| `Notification` | Status changes | Not useful for injection |
| `Stop` | Agent finishes | Too late |

Missing from Claude Code hook API:
- Per-turn trigger (before assistant starts reasoning)
- `$CLAUDE_USER_MESSAGE` env var
- Context injection file mechanism (`$CLAUDE_INJECT_FILE`)

### When to revisit

Implement M3 when Claude Code provides any of:
- A **turn-start hook** with user message access
- A **context prepend** mechanism (file or env-based)
- An officially supported **tool-as-context-supplier** pattern

### M3 scope (when unblocked)

1. **Hook config**: `ragix-memory hooks --install` writes `.claude/settings.json`
2. **`ragix-claude ask` wrapper**: single command for pipe + claude invocation
3. **Auto-recall hook**: pre-turn recall injection (requires Claude Code API changes)

---

## Why This Works

| Problem | Solution |
|---------|----------|
| Large stdin truncated by Claude | Stored once in memory (chunked + dedup) |
| Context window overflow | Token-budgeted recall (inject only what fits) |
| Crucial sections missed | FTS5 BM25 ranking returns query-relevant chunks |
| No persistence across sessions | SQLite store with full provenance |
| Context poisoning from raw dumps | `injectable=false` (ingest) → consolidate → `injectable=true` |
| Re-ingest overhead | SHA-256 dedup via `corpus_hashes` — idempotent |
| Too many commands for simple use | `pipe` = ingest + recall in one shot |

---

*v3.0 — revised 2026-02-16 after M1+M2 implementation, FTS5 recall fix, consolidation injectable fix*
