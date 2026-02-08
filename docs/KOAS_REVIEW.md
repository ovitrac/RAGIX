# KOAS-Review — Traceable, Reversible Markdown Review

**Kernel-Enforced Document Review for Documents Larger Than Context Windows**

**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Version:** 0.5.0
**Date:** 2026-02-08
**RAGIX Version:** 0.66+
**KOAS Version:** 1.0

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Design Philosophy](#2-design-philosophy)
3. [Architecture](#3-architecture)
4. [Three-Stage Pipeline](#4-three-stage-pipeline)
5. [Pyramidal Knowledge Base](#5-pyramidal-knowledge-base)
6. [Edit Operations and Traceability](#6-edit-operations-and-traceability)
7. [Worker + Tutor Pattern](#7-worker--tutor-pattern)
   - [v7 Preflight Pipeline](#v7-preflight-pipeline)
   - [Adaptive Tier Escalation (v7.1)](#adaptive-tier-escalation-v71)
   - [v7.2 Content Recipes](#v72-content-recipes)
   - [v7.3 Expanded Triggers and Masking](#v73-expanded-triggers-and-masking)
   - [v7.3.1 Policy D — Markdown Stripping](#v731-policy-d--markdown-stripping)
   - [v7.3.2 Acceptance Fixes](#v732-acceptance-fixes)
   - [Blank-Line Targeting Guard (v2.4.0)](#blank-line-targeting-guard-v240)
   - [Benchmark Results](#benchmark-results)
   - [Production Baseline Contract (v7.3.2)](#production-baseline-contract-v732)
8. [Selective Revert](#8-selective-revert)
9. [CLI Reference](#9-cli-reference)
10. [MCP Integration](#10-mcp-integration)
11. [Configuration Reference](#11-configuration-reference)
12. [Data Model Reference](#12-data-model-reference)
13. [Workspace Layout](#13-workspace-layout)
14. [Design Rationale](#14-design-rationale)

---

## 1. Introduction

**KOAS-Review** extends the Kernel-Orchestrated Audit System (KOAS) to **Markdown document review and editing**. While KOAS-Docs addresses summarization of large document corpora, KOAS-Review addresses a complementary challenge: **reviewing, correcting, and annotating a single Markdown document** that may exceed any LLM's context window.

### Core Capabilities

1. **Traceable edits** — Every change is identified (`RVW-0001`), timestamped, and stored in an append-only ledger
2. **Selective revert** — Any change can be individually undone via its inverse patch, at any time
3. **Protected regions** — Code fences, math blocks, tables, YAML front matter, and HTML are never touched
4. **Pyramidal context** — A 4-level hierarchical summary gives the LLM document-wide awareness when reviewing each chunk
5. **Worker + Tutor validation** — A fast model proposes edits; an optional larger model validates them
6. **Sovereignty** — All processing runs locally via Ollama; non-local backends require explicit opt-in
7. **Deterministic core** — The edit engine, patch generator, and ledger are purely deterministic; LLMs operate only at the analysis edge

### What KOAS-Review Is Not

KOAS-Review is not a general-purpose text editor, a spell-checker, or a grammar tool. It is a **kernel-enforced review system** designed for:

- Technical specifications and standards documents
- Multi-author Markdown documents with accumulated AI-generated artifacts
- Regulatory and compliance documents where every change must be auditable
- Large documents (50+ pages) that cannot be reviewed in a single LLM pass

---

## 2. Design Philosophy

### "Kernels compute, LLMs interpret — changes are always reversible."

This extends the foundational KOAS principle with a traceability guarantee. The system is built on three invariants:

1. **Every change has an ID.** Changes follow the format `RVW-NNNN` (zero-padded, sequential). No change is anonymous.

2. **Every change has an inverse.** For each forward patch (`RVW-0034.patch`), an inverse patch (`RVW-0034.inverse.patch`) is stored alongside it. Reverting a change is as reliable as applying it.

3. **Protected content is immutable.** Code fences, math blocks, tables, YAML front matter, and HTML blocks are marked at Stage 1 and excluded from all edits. An edit that would cross a protected boundary is rejected unless explicitly flagged for human attention.

### Visibility Tiers

Not all changes require human review. KOAS-Review classifies edits into two visibility tiers:

| Tier | Examples | Behavior |
|------|----------|----------|
| **Silent** | Typos, punctuation, capitalization, spacing, grammar | Applied automatically, logged in ledger, no inline note |
| **Attention** | Terminology changes, logic rewrites, deletions, structural changes | Applied with a GitHub Alert block (`> [!WARNING]`) injected after the modified text |

The boundary between tiers is configurable. By default, changes whose `kind` is in `["typo", "punctuation", "capitalization", "spacing", "grammar"]` are silent. Everything else requires attention.

### Deletion Policy

Deletions are treated with maximum caution:

- A deletion **always** requires `needs_attention: true`
- A deletion **always** requires a `review_note` with `alert: CAUTION`
- The review note **must** start with `REVIEWER: RVW-NNNN` followed by a human-readable explanation
- Deleted text is never discarded: it remains in the inverse patch and can be restored

---

## 3. Architecture

### File Tree

```
ragix_kernels/reviewer/
    __init__.py               # Package registration, kernel list
    config.py                 # ReviewerConfig, ChunkConfig, LLMConfig, StyleRules
    models.py                 # Core data models (ChangeID, EditOp, LedgerEntry, ...)
    md_parser.py              # Heading extraction, protected region detection
    patch_engine.py           # Forward/inverse unified diffs
    ledger.py                 # Append-only JSONL ledger (thread-safe)
    llm_backend.py            # Dual backend: Ollama (sovereign) + Claude API
    context.py                # Adaptive context assembly for edit-time injection
    kernels/
        md_inventory.py       # [S1] File stats, SHA-256, snapshot
        md_structure.py       # [S1] Heading tree, anchors, numbering
        md_protected_regions.py # [S1] Immutable span detection
        md_chunk.py           # [S1] Structure-aligned chunking
        md_consistency_scan.py  # [S2] AI leftovers, duplicates, refs, numbers, tables, terms (v2.0)
        md_numbering_control.py # [S2] Heading/figure/table numbering
        md_pyramid.py         # [S2] Bottom-up hierarchical summaries
        md_fingerprint_chunk.py # [S2] Structural fingerprint per chunk (deterministic)
        md_edit_plan.py       # [S2] LLM-driven edit ops (Worker+Tutor, v2.4.0)
        md_apply_ops.py       # [S3] Validate + apply + patch + ledger (v1.1.0)
        md_inline_notes_inject.py # [S3] GitHub Alert blocks
        md_review_report_assemble.py # [S3] REVIEW_doc.md generation
        md_revert.py          # [S3] Selective inverse-patch application
    prompts/
        review.j2             # Main review prompt (Jinja2)
        tutor_validate.j2     # Tutor validation prompt
        regenerate.j2         # Conservative regeneration after rejection
    cli/
        reviewctl.py          # CLI: review, report, revert, show, grep
    mcp/
        tools.py              # 4 MCP tools for Claude Code/Desktop
```

### Kernel Map

All 13 kernels subclass `Kernel` from `ragix_kernels/base.py` and implement `compute()` + `summarize()`. They are auto-discovered by the KOAS `KernelRegistry`.

| # | Kernel | Stage | Deterministic | Depends On | Produces |
|---|--------|-------|:---:|------------|----------|
| 1 | `md_inventory` | S1 | Yes | *(none)* | `file_stats`, `doc_hash` |
| 2 | `md_structure` | S1 | Yes | `md_inventory` | `heading_tree`, `section_index`, `anchor_map` |
| 3 | `md_protected_regions` | S1 | Yes | `md_inventory` | `protected_spans` |
| 4 | `md_chunk` | S1 | Yes | `md_structure`, `md_protected_regions` | `chunks` |
| 5 | `md_consistency_scan` v2.0 | S2 | Yes | `md_chunk`, `md_protected_regions`, `md_structure` | `issues` |
| 6 | `md_numbering_control` | S2 | Yes | `md_structure` | `numbering_findings`, `renumber_ops` |
| 7 | `md_pyramid` | S2 | LLM edge | `md_chunk` | `pyramid`, `glossary` |
| 8 | `md_fingerprint_chunk` | S2 | Yes | `md_chunk` | `chunk_fingerprints` |
| 9 | `md_edit_plan` v2.4.0 | S2 | LLM edge | `md_chunk`, `md_consistency_scan`, `md_pyramid`, `md_fingerprint_chunk` | `edit_plan` |
| 10 | `md_apply_ops` v1.1.0 | S3 | Yes | `md_protected_regions` | `patches`, `ledger`, `edited_doc` |
| 11 | `md_inline_notes_inject` | S3 | Yes | `md_apply_ops` | `annotated_doc` |
| 12 | `md_review_report_assemble` | S3 | Yes | `md_apply_ops` | `review_report` |
| 13 | `md_revert` | S3 | Yes | `md_apply_ops` | `reverted_doc` |

### Dependency DAG

```
md_inventory ──┬── md_structure ──┬── md_chunk ──┬── md_consistency_scan ──┐
               │                  │              │                        │
               └── md_protected_regions ─────────┤                        │
                                  │              │                        │
                                  └──────────────┤                        │
                                                 ├── md_pyramid ──────────┤
                                                 │                        │
                                                 ├── md_fingerprint_chunk │
                                                 │                        │
                                 md_structure ── md_numbering_control     │
                                                                          │
                                                 md_edit_plan ────────────┤
                                                                          │
                    md_protected_regions ──────── md_apply_ops ──┬── md_inline_notes_inject
                                                                │
                                                                ├── md_review_report_assemble
                                                                │
                                                                └── md_revert
```

---

## 4. Three-Stage Pipeline

### Stage 1 — Collection (Deterministic)

Stage 1 analyzes the document structure without any LLM involvement. All outputs are deterministic and reproducible.

**`md_inventory`** — Creates an immutable snapshot (`doc.raw.md`) and computes:
- SHA-256 hash of the entire document
- Line count, byte size
- YAML front matter detection
- Code fence and table counts

**`md_structure`** — Parses all ATX headings (`# ... ######`) into a recursive tree:
- Each heading receives a stable section ID (e.g., `S2.3.1`)
- GitHub-compatible anchor slugs are generated
- Explicit numbering patterns (e.g., `2.3 Methods`) are detected
- A flat section index and anchor map are saved alongside the tree

**`md_protected_regions`** — Detects and marks immutable spans. Policy is **conservative**: when in doubt, mark as protected.

| Kind | Detection |
|------|-----------|
| `code_fence` | Triple backticks or tildes (``` ``` ```, `~~~`) |
| `yaml_front_matter` | `---` at line 1, closed by `---` |
| `math_block` | `$$` delimiters |
| `table` | Pipe-separated rows with `|---|` separator |
| `html_block` | Block-level `<tag>...</tag>` |
| `link_ref_def` | `[label]: URL` definitions |

Each span stores its content hash so that any accidental modification is detectable.

**`md_chunk`** — Produces a structure-aligned chunk plan:
- Chunks follow section boundaries (never cross headings)
- Large sections are sub-split at paragraph boundaries
- Chunk IDs are hash-stable: `{anchor}_{sha256[:8]}`
- Protected spans are never split
- Default budget: 1,800 tokens per chunk

### Stage 2 — Analysis (Deterministic + LLM Edge)

Stage 2 combines deterministic detection with optional LLM-driven analysis.

**`md_consistency_scan`** v2.0 (deterministic) — 7 detectors across two generations:

*v1 detectors:*
- AI-generated leftovers (10 configurable regex patterns: "As an AI", "Certainly!", "I hope this helps", etc.)
- Duplicated paragraphs (MD5 hash comparison, normalized whitespace)
- Broken cross-references — English patterns ("see Section 4.2" where Section 4.2 does not exist)

*v2 detectors (added 2026-02-08):*
- French cross-references — Validates `§X.Y`, `voir X.Y`, `cf. X.Y` against the section index from `md_structure`. Suppresses false positives for explicitly deleted sections (`SUPPRIMÉ` in line) and deduplicates same-line references.
- Numerical contradictions — Cross-checks quantity claims ("deux approches", "5 classes") against nearby markdown table row counts. Forward-only 5-line window, claim range 2–10.
- Table structure validation — Column count consistency (header vs separator vs data rows) and percentage column sum validation (flags totals outside 95–105%).
- Terminology drift — Tracks language pair mixing per section (e.g., "code mort" vs "dead code", "refactoring" vs "refonte"). Flags sections using both French and English forms.

Issues inside protected regions are automatically filtered out.

**`md_numbering_control`** (deterministic) — Validates:
- Heading level continuity (no H2 -> H4 skip)
- Figure, Table, and Equation numbering sequences (gaps, duplicates)
- Cross-reference existence ("see Figure 3" — does Figure 3 exist?)

**`md_pyramid`** (LLM edge) — Builds a 4-level hierarchical summary. See [Section 5](#5-pyramidal-knowledge-base).

**`md_fingerprint_chunk`** (deterministic, v1.0.0) — Computes a structural fingerprint per chunk: table row/count, math expressions, emoji code points, digit density, bullet count, blockquote lines, safety keyword hits, and content tokens. Used by `md_edit_plan` to decide whether content-level masking recipes should be applied before the LLM call. See [Section 7.11](#711-v72-content-recipes).

**`md_edit_plan`** (LLM edge, v2.4.0) — Generates edit operations per chunk using a **tolerant multi-format extraction ladder**. Accepts JSON, YAML, tagged payloads, plain-text directives, or freeform prose; the kernel normalizes all formats into canonical `EditOp` structures. Writes per-chunk artifacts (raw LLM output + canonical ops + status) immediately, enabling mid-run inspection. v2.4.0 adds a blank-line targeting guard to prevent off-by-one edits. See [Section 7](#7-worker--tutor-pattern).

### Stage 3 — Reporting (Deterministic)

Stage 3 applies edits, generates output documents, and maintains the audit trail.

**`md_apply_ops`** v1.1.0 — The critical deterministic engine. For each edit op:
1. **Pre-validation normalization** (`_normalize_op`, v1.1.0): three deterministic fixes applied before validation:
   - *HASH_RECOMPUTE*: recomputes `before_hash` from original document lines (fixes hash mismatches caused by masked-text hashing)
   - *NOTE_INJECT*: injects fallback `ReviewNote(alert=NOTE)` when `needs_attention=true` but note is missing
   - *NOTE_ID_FIX*: prepends `REVIEWER: {op.id} —` to review note text if the change ID is absent
2. Validates the op against 6 invariants (hash match, protected spans, deletion policy, ID format, ID uniqueness, review note format)
3. Applies the edit (bottom-up by line number to avoid offset shift)
4. Generates forward and inverse unified diff patches
5. Writes the ledger entry with full metadata
6. Saves `doc.edited.md` and `apply_log.jsonl`

Rejected ops are classified via `_classify_rejection()` into a funnel taxonomy: `REJECT_HASH`, `REJECT_SCHEMA_NOTE`, `REJECT_SCHEMA_ID`, `REJECT_LEDGER`, `REJECT_PROTECTED`, `REJECT_RANGE`, `REJECT_OTHER`.

**`md_inline_notes_inject`** — For each `needs_attention` edit, injects a GitHub Alert block immediately after the modified text:

```markdown
> [!WARNING]
> REVIEWER: RVW-0034 — Terminology change: "bandwidth" replaced with "throughput" for consistency with Section 2.
```

Alert type is mapped from severity:

| Severity | Alert |
|----------|-------|
| `minor` | `[!NOTE]` |
| `attention` | `[!WARNING]` |
| `deletion` | `[!CAUTION]` |
| `critical` | `[!IMPORTANT]` |

**`md_review_report_assemble`** — Generates `REVIEW_doc.md` with:
- Run metadata (date, tool version, document hash)
- Summary counts (attention changes, deletions, silent edits, reverts)
- Per-section change table
- Full chronological ledger table

**`md_revert`** — Applies inverse patches to undo specific changes. See [Section 8](#8-selective-revert).

---

## 5. Pyramidal Knowledge Base

The central technical innovation. When an LLM reviews a single chunk (~1,800 tokens), it has no awareness of the rest of the document. The pyramid solves this by providing compressed document-wide context.

### 4-Level Hierarchy

```
Level 0: Document abstract         (~500 tokens, 1 node)
  Level 1: Top-section summaries   (~200 tokens each, ~8 nodes)
    Level 2: Subsection summaries  (~150 tokens each, ~20 nodes)
      Level 3: Paragraph-group     (~100 tokens each, ~80 nodes)
               micro-summaries
```

### Bottom-Up Construction

The pyramid is built leaf-first:

1. **Level 3**: Each leaf section (heading with no children) is summarized by the LLM
2. **Level 2**: Parent sections are summarized from their children's summaries
3. **Level 1**: Top-level sections are summarized from their children
4. **Level 0**: The document abstract is synthesized from all Level 1 summaries

### Hash-Based Incremental Updates

Every pyramid node stores a `content_hash` of its source text. On re-run:
- Unchanged sections skip LLM calls entirely (hash match → return cached node)
- Only modified branches are recomputed
- Level 3 nodes are parallelizable (independent leaf nodes, up to 4 threads)

For a 200-page document with ~109 pyramid nodes, incremental mode typically requires 0 LLM calls if the document has not changed.

### Context Injection at Edit Time

When `md_edit_plan` reviews a chunk, the `context.py` module assembles a priority-ordered context window:

| Priority | Component | Budget |
|----------|-----------|--------|
| 1 (always) | Target chunk text | ~40% |
| 2 (always) | Edit instructions | ~8% |
| 3 (high) | Document abstract (L0) | ~14% |
| 4 (high) | Current section summary (L1) | ~10% |
| 5 (high) | Issue findings for this chunk | ~6% |
| 6 (medium) | Current subsection summary (L2) | ~5% |
| 7 (medium) | 2 nearest sibling sections (L1) | ~10% |
| 8 (low) | Global glossary | ~5% |

When the budget is tight (small local model with 4K context), lower-priority items are dropped automatically. With a 128K model, full sibling context is included.

### Skipping the Pyramid

For quick deterministic-only runs, pass `--skip-pyramid` (CLI) or `skip_pyramid: true` (config). This disables `md_pyramid` and `md_edit_plan`, running only the deterministic consistency and numbering checks.

---

## 6. Edit Operations and Traceability

### EditOp Schema

Every edit operation follows this exact JSON schema:

```json
{
  "id": "RVW-0034",
  "action": "replace",
  "target": {
    "chunk_id": "methods_a3f8bc21",
    "anchor": "methods",
    "node_path": "S2.3.p5",
    "line_start": 142,
    "line_end": 145
  },
  "before_hash": "sha256:a3f8bc21...",
  "before_text": "The bandwith of the system...",
  "after_text": "The throughput of the system...",
  "needs_attention": true,
  "kind": "terminology",
  "severity": "attention",
  "silent": false,
  "review_note": {
    "alert": "WARNING",
    "text": "REVIEWER: RVW-0034 — 'bandwidth' replaced with 'throughput' for consistency with Section 2."
  },
  "rationale": "Section 2 uses 'throughput' consistently; 'bandwidth' here is a register shift."
}
```

### Allowed Actions

| Action | Effect |
|--------|--------|
| `replace` | Replace `before_text` at target lines with `after_text` |
| `insert` | Insert `after_text` after the target line range |
| `delete` | Remove the text at target lines (requires review note) |
| `flag_only` | No modification; records the finding in the ledger |

### Validation Invariants

`md_apply_ops` enforces 6 invariants before applying any edit:

1. **ID format**: Must match `RVW-NNNN` (canonical `ChangeID`)
2. **ID uniqueness**: No collision with existing ledger entries
3. **Content hash**: `before_hash` must match the actual SHA-256 of the target text
4. **Protected spans**: Edit must not cross protected regions (unless `needs_attention` + `review_note`)
5. **Deletion policy**: `action: delete` requires `review_note`
6. **Attention policy**: `needs_attention: true` requires `review_note` starting with `REVIEWER:`

Any validation failure causes the op to be **rejected** (not applied) with the error recorded.

### The Ledger

All applied changes are recorded in an append-only JSONL file (`review/ledger.jsonl`). Each line is a `LedgerEntry`:

```json
{
  "id": "RVW-0034",
  "doc": "spec.md",
  "doc_hash_before": "sha256:...",
  "doc_hash_after": "sha256:...",
  "timestamp": "2026-02-06T14:23:01.234567+00:00",
  "actor": {"tool": "reviewctl", "operator": "koas_reviewer"},
  "scope": {"anchor": "methods", "node_path": "S2.3.p5"},
  "kind": "terminology",
  "severity": "attention",
  "silent": false,
  "summary": "'bandwidth' replaced with 'throughput' for consistency",
  "rationale": "Section 2 uses 'throughput' consistently...",
  "patch_forward": "review/patches/RVW-0034.patch",
  "patch_inverse": "review/patches/RVW-0034.inverse.patch",
  "review_note": {"alert": "WARNING", "text": "REVIEWER: RVW-0034 — ..."},
  "is_revert": false,
  "reverted_id": ""
}
```

The ledger is **thread-safe** (uses `threading.Lock`) and **never mutated** — reverts create new entries with `is_revert: true`.

---

## 7. Worker + Tutor Pattern

`md_edit_plan` v2.4.0 processes each chunk through a **tolerant extraction pipeline** that accepts any LLM output format. The kernel — not the LLM — is responsible for producing canonical `EditOp` structures.

### Design Principle

> **The kernel must not block the whole pipeline on "perfect JSON".**

Local sovereign models (7B–120B) produce output in wildly varying formats: strict JSON, JSON with trailing commas, YAML, fenced code blocks, plain-text directives, or free prose. Requiring strict JSON compliance causes 50–80% parse failure rates with large local models. The extraction ladder solves this by accepting all formats.

### Extraction Ladder (6 Levels)

When the LLM returns a response, the kernel runs an ordered sequence of parsers. The first parser that succeeds determines the extraction method.

| Level | Method | Description |
|-------|--------|-------------|
| 1 | `json_strict` | Standard JSON array parse (strips markdown fences first) |
| 2 | `json_relaxed` | Tolerates trailing commas, single quotes, `//` comments, single objects wrapped in array |
| 3 | `yaml` | YAML parse (handles models that emit YAML instead of JSON) |
| 4 | `tagged_payload` | Extracts content between `BEGIN_EDIT_OPS`/`END_EDIT_OPS` or `<EDIT_OPS>` tags, then applies levels 1–3 |
| 5 | `plaintext_regex` | Matches `[RVW] action=replace` blocks and structured `replace:`, `delete:`, `flag:` patterns |
| 6 | `freeform_salvage` | Detects "no changes needed" (EN/FR variants) → empty ops; otherwise emits a single `flag_only` op with the LLM's prose as rationale |

If all 6 levels fail (should be rare), the chunk is skipped with an empty op set.

### Kernel-Owned ID Allocation

The LLM never generates `RVW-NNNN` change IDs. Instead:

1. The LLM outputs raw edit proposals (action, target lines, text, rationale)
2. The kernel's `_normalize_op()` assigns sequential IDs (`RVW-0001`, `RVW-0002`, ...) after extraction
3. Missing fields are filled with sensible defaults (`severity: "minor"`, `kind: "style"`, etc.)
4. `before_hash` is computed from the actual document lines, not from LLM output
5. Deletion policy is enforced: any `delete` action gets `needs_attention: true` + `review_note` with `alert: CAUTION`
6. Ops touching protected lines are silently filtered out

### Circuit Breaker

A rolling-window circuit breaker monitors parse success rates:

- **Window**: 20 chunks (configurable)
- **Trip threshold**: 60% failure rate over the window
- **Action on trip**: Switches from the structured `review.j2` prompt to the simpler `review_plain.j2` prompt

The plain-text prompt uses a minimal format (`[RVW] action=replace / lines=N-M / before="..." / after="..."`) that models with weak JSON capabilities can produce reliably. The circuit breaker logs a warning when it trips.

### Worker + Tutor Orchestration

When `tutor_model` is configured, the kernel uses a two-pass validation loop:

```
For each chunk:
  1. WORKER (sovereign model, e.g., gpt-oss-safeguard:120b)
     └── Proposes edit ops in any format

  2. Extraction ladder → canonical ops

  3. Filter: remove ops touching protected lines

  4. TUTOR (validation model, e.g., mistral:instruct)
     └── Validates proposed ops
     └── Returns VERDICT: ACCEPTED | CORRECTED | REJECTED

  5. If ACCEPTED:  use worker's ops as-is
     If CORRECTED: use tutor's corrected ops (re-extracted via ladder)
     If REJECTED:  one regeneration attempt with conservative prompt
                   └── If regeneration also fails → skip chunk
```

### Verdict Parsing

The tutor response is parsed via regex:

```
VERDICT: ACCEPTED
REASON: All proposed edits are accurate and well-scoped.
```

```
VERDICT: CORRECTED
REASON: Line numbers were off by one; severity should be 'minor' not 'attention'.
CORRECTION:
[{ ... corrected ops ... }]
```

```
VERDICT: REJECTED
REASON: Proposed edit would introduce a factual error in the definition of X.
```

### Per-Chunk Artifact Writing

Every chunk writes artifacts **immediately** after processing (not batched at kernel end):

| File | Content | Always written? |
|------|---------|-----------------|
| `ops/{chunk_id}.raw.txt` | Raw LLM response verbatim | Yes |
| `ops/{chunk_id}.json` | Canonical normalized ops | Only if ops > 0 |
| `status.jsonl` | One-line status (chunk_id, parse result, extraction method, op count, latency) | Yes (append) |

This enables:
- **Mid-run inspection**: `tail -f stage2/ops/status.jsonl` to monitor progress
- **Partial exploitation**: Canonical ops are usable even if the pipeline is interrupted
- **Debugging**: Raw LLM output is always preserved for post-mortem analysis

### Prompt Templates

Two Jinja2 templates, selected by the circuit breaker:

| Template | Usage | LLM output format |
|----------|-------|-------------------|
| `review.j2` | Default (structured) | `BEGIN_EDIT_OPS [...] END_EDIT_OPS` JSON array |
| `review_plain.j2` | Fallback (circuit breaker tripped) | `[RVW] action=replace / lines=N-M / ...` plain text |
| `regenerate.j2` | After tutor rejection | Conservative `BEGIN_EDIT_OPS` with extra caution instructions |

### Quality Metrics

`md_edit_plan` tracks per-chunk statistics:

| Metric | Description |
|--------|-------------|
| `worker_calls` | Number of LLM calls to the worker model |
| `tutor_calls` | Number of LLM calls to the tutor model |
| `verdict` | `accepted`, `corrected`, `rejected`, `regenerated`, `single_model`, `empty`, `filtered` |
| `ops_proposed` | Edit ops proposed by the worker |
| `ops_accepted` | Edit ops that passed normalization |
| `parse_success` | Whether extraction ladder found ops |
| `extraction_method` | Which ladder level succeeded (e.g., `json_strict`, `plaintext_regex`, `freeform_salvage`) |
| `latency_ms` | Wall-clock time for the chunk |

Aggregate statistics reported in the kernel summary: `acceptance_rate`, `parse_success_rate`, `extraction_method_distribution`, `circuit_breaker_tripped`.

### Single-Model Mode

When `tutor_model` is not set (the default), the system runs in single-model mode: the worker's output is extracted and normalized directly without tutor validation. This is faster but less safe. For production use on important documents, enabling the tutor is recommended.

### Retry Policy

When the worker LLM returns an empty response (0 tokens generated), the kernel retries with increasing temperature:

| Attempt | Temperature | Wait |
|---------|-------------|------|
| 1 (initial) | 0.10 | — |
| 2 (retry) | 0.15 | 3–8s jitter |
| 3 (retry) | 0.20 | 3–8s jitter |

Configurable via `llm.empty_retries` (default: 2). The temperature nudge helps when the model's sampling distribution collapses to zero output at low temperatures.

### Timeout Configuration

Large models (65B+) can take 30–60s per chunk. The per-request timeout is configurable:

| Config key | Default | Description |
|------------|---------|-------------|
| `llm.chunk_timeout` | 300 | HTTP timeout in seconds per LLM call |
| `llm.empty_retries` | 2 | Max retries on empty response |

The `status.jsonl` log includes `t_request_ms` (time spent in LLM calls only) and `retries` (number of retry attempts) for post-mortem latency analysis.

### Graceful Degradation

- If the worker LLM returns empty text → retried up to `empty_retries` times with temperature jitter
- If still empty after retries → chunk is skipped with `extraction_method: "empty_response"`
- If the extraction ladder finds zero ops but detects "no changes needed" → chunk recorded as clean (`freeform_salvage`)
- If all 6 ladder levels fail → chunk is skipped, raw output preserved in `ops/{chunk_id}.raw.txt`
- If the tutor LLM fails → all ops for that chunk are rejected (fail-safe)
- If `--no-llm` is set → `md_edit_plan` returns an empty result; only deterministic passes run
- If the circuit breaker trips → prompt switches to plain-text mode for remaining chunks

### v7 Preflight Pipeline

`md_edit_plan` v2.4.0 includes a **preflight pipeline** (introduced in v2.2.0) — three deterministic guards applied to each chunk *before* the LLM call:

1. **Math masking** — `$...$` / `$$...$$` expressions are replaced with inert `[MATH_N]` placeholders. LaTeX content triggers degenerate decoding loops in many local models; masking eliminates this failure mode entirely.

2. **Sub-chunk splitting** — Chunks exceeding `max_edit_chunk_tokens` (default 900) are split at paragraph boundaries into `{chunk_id}_a` / `{chunk_id}_b` sub-chunks. Never splits inside protected spans. Single-level only (no recursion).

3. **Context tiering** — Four tiers control how much pyramid context is injected into the prompt:

| Tier | Content | Use case |
|------|---------|----------|
| 0 | Skeleton only (chunk text + instructions) | Large chunks, fallback for degenerate models |
| 1 | + section summary | Default for sovereign models |
| 2 | + abstract + issue findings | Escalation when ops=0 but issues exist |
| 3 | Full context (all pyramid levels + siblings) | Large-context models (128K+) |

Tier is computed from chunk token estimate and budget: `compute_context_tier(chunk_tokens, budget, default)`.

### Adaptive Tier Escalation (v7.1)

After the first LLM call, three deterministic retry policies are applied in order:

**Policy A** — Degenerate fallback: If the model emits tokens but produces 0 visible characters (`decoded_empty_saturated`) at tier > 0, retry at tier=0 (skeleton only). This removes context complexity as the source of failure.

**Policy B** — Ops escalation: If parsing succeeded but ops=0 *and* the chunk has known issues from `md_consistency_scan`, escalate to `min(tier + 1, tier_max)` if budget allows. More context helps the model find issues it missed with less context.

**Policy C** — Content recipes (v7.2+): If still degenerate at tier=0 *and* the chunk fingerprint triggers content recipes, apply deterministic masking transforms and rerun once. See next section.

**Policy D** — Markdown stripping (v7.3.1): If still degenerate after Policy C *and* post-mask density thresholds exceeded, apply a 9-step plain-text projection and rerun once. See [v7.3.1 Policy D](#v731-policy-d--markdown-stripping).

### v7.2 Content Recipes

Content recipes are deterministic, reversible text transforms applied to chunk content before the LLM call. They reduce structural complexity that causes model degenerate decoding — a failure mode where the model emits the full token budget (e.g., 2048 tokens) but produces 0 visible characters.

**Design principle:** Tiering is a model-control lever (how much *context* the model sees). Content recipes are a content-control lever (how complex the *chunk itself* appears). The two are orthogonal.

#### Fingerprinting

The `md_fingerprint_chunk` kernel (pure deterministic, no LLM) extracts per-chunk structural features:

| Feature | Description |
|---------|-------------|
| `table_rows` / `table_count` | Markdown pipe-table statistics |
| `math_count` | Display + inline LaTeX expressions |
| `emoji_count` | Unicode emoji code points |
| `digit_density` | Numeric content ratio (digits / non-whitespace) |
| `bullet_count` | List items (-, *, N.) |
| `blockquote_lines` | Lines starting with `>` |
| `safety_keyword_hits` | Security terms that trigger model self-censoring |
| `content_tokens` | Rough token estimate |

A chunk `triggers_recipe()` when structural complexity exceeds safe thresholds:
- Table rows >= 3 combined with emoji or blockquote presence
- Math combined with tables
- Emoji count >= 8
- Blockquote lines >= 8

#### Recipe Pipeline

When Policy C fires, `_apply_content_recipe` applies transforms in order:

| Order | Transform | Placeholder | Trigger |
|-------|-----------|-------------|---------|
| 1 | `blockquote_flatten` | `[[QUOTE_BLOCK_N]]` | `blockquote_lines >= blockquote_min_lines` (default 5) |
| 2 | `table_mask` | `[[TABLE_BLOCK_N]]` | `table_rows >= table_mask_min_rows` (default 3) |
| 3 | `emoji_mask` | `[[EMOJI_N]]` | `emoji_count > 0` |
| 4 | `digit_truncation` | `[[NUMERIC_TRUNCATED_N]]` | `digit_density >= threshold` (default 0.25) |
| 5 | `synonym_map` | *(inline substitution)* | `safety_keyword_hits > 0` |

All placeholder formats use double brackets `[[...]]` to distinguish from the single-bracket `[MATH_N]` math placeholders.

Each transform returns `(modified_text, mapping)`. Mappings are merged into a single dict passed to `_unmask_ops()` after extraction, which restores original content in `before_text` / `after_text` fields so that `md_apply_ops` matches real document content.

#### Traceability

Content recipe application produces three categories of artifacts:

**Per-chunk masks** (`stage2/masks/`):
- `{chunk_id}_QUOTE_N.md` — Original blockquote block text
- `{chunk_id}_TABLE_N.md` — Original table block text
- `{chunk_id}_EMOJI.json` — Emoji placeholder mapping
- `{chunk_id}_recipe.json` — Full recipe log: transforms applied, mapping keys, token delta

**Degenerate exemplars** (`stage2/exemplars/`):
- `{chunk_id}.json` — Snapshot of chunks that remain degenerate after all policies: chunk text, fingerprint, tier history, extraction method, recipes applied

**Status log** (`stage2/ops/status.jsonl`):
- Per-chunk entries include `content_recipe` field when a recipe fires (e.g., `["blockquote_flatten", "emoji_mask"]`)
- Adaptive escalation history recorded in `adaptive_escalation` field

#### Configuration

```yaml
reviewer:
  content_recipes:
    enabled: true                    # Master switch
    blockquote_flatten: true         # Replace blockquote blocks with placeholders
    blockquote_min_lines: 5          # Minimum contiguous blockquote lines to flatten
    table_mask_min_rows: 3           # Mask tables with >= N data rows
    emoji_mask: true                 # Mask emoji code points
    digit_truncation_threshold: 0.25 # Digit density trigger
    digit_truncation_min_lines: 5    # Min block size for truncation
    synonym_map_enabled: true        # Safety vocabulary substitution
```

#### Quality Metrics

The `_compute_quality` aggregator tracks:
- `recipe_applied` — Number of chunks where content recipes were applied
- `recipe_recovered` — Number of those that recovered (no longer degenerate after recipe)

### v7.3 Expanded Triggers and Masking

v7.3 expands the fingerprint trigger conditions and adds new masking functions, driven by analysis of a 249-chunk production run where 33/38 remaining degenerates had untriggered structural features.

#### Expanded Trigger Conditions

| Trigger | v7.2 | v7.3 | Rationale |
|---------|:----:|:----:|-----------|
| `table_rows >= 3` + emoji/blockquote | Yes | Yes | Unchanged |
| `math_count > 0` + `table_count > 0` | Yes | Yes | Unchanged |
| `emoji_count >= 8` | Yes | `>= 5` | 8 missed emoji-heavy chunks in production |
| `blockquote_lines >= 5` | Yes | Yes | Unchanged |
| `code_fence_count >= 1` | — | **New** | Mermaid diagrams / code blocks cause degenerates |
| `table_rows >= 3` (standalone) | — | **New** | Tables alone trigger without needing emoji/blockquote |
| `math_count >= 3` (standalone) | — | **New** | Math-heavy chunks without tables also degenerate |

#### New Masking Functions

8 masking functions applied in order (v7.3):

1. `_flatten_blockquotes` — Collapses contiguous `>` lines into `[[QUOTE_BLOCK_N]]`
2. `_mask_code_fences` — Replaces fenced code blocks (including Mermaid) with `[[CODE_BLOCK_N]]`
3. `_mask_table_blocks` — Replaces pipe-table blocks with `[[TABLE_BLOCK_N]]`
4. `_mask_emoji` — Replaces emoji code points with `[[EMOJI_N]]`
5. `_consolidate_math_placeholders` — Groups math-heavy paragraphs into `[[FORMULA_BLOCK_N]]`
6. `_truncate_digit_blocks` — Replaces high-density numeric blocks with `[[NUMERIC_TRUNCATED_N]]`
7. `_apply_synonym_map` — Safety vocabulary substitution (reversible)

Policy C now applies to both the main path and the split path (sub-chunks get recipes too).

### v7.3.1 Policy D — Markdown Stripping

Policy D is a late-stage retry-only transform that fires after Policy C when masking alone is insufficient because residual markdown formatting (bold markers, backticks, empty headings, horizontal rules) causes a degenerate generation loop.

#### Trigger Conditions

Policy D fires when all three conditions are met:
1. Policy C content recipe already fired for this chunk
2. Model still degenerate after recipe application
3. Post-mask density thresholds exceeded (any single metric or combined total)

**Density thresholds** (computed by `_compute_post_mask_density()`):

| Metric | Threshold | Description |
|--------|:---------:|-------------|
| `bold_markers` | >= 8 | `**...**` pairs |
| `backtick_count` | >= 12 | Inline backticks |
| `code_span_count` | >= 8 | `` `...` `` spans |
| `empty_heading_count` | >= 1 | `###` lines with no text |
| `hr_count` | >= 1 | `---` / `***` horizontal rules |
| `combined` | >= 20 | Sum of all metrics |

#### Markdown Strip Transform

`_markdown_strip()` is a 9-step deterministic plain-text projection:

1. Protect `[[...]]` and `[MATH_N]` placeholders
2. Remove horizontal rules (`---`, `***`)
3. Normalize headings (`### Title` → `Title:`)
4. Flatten blockquotes (remove `>` prefix)
5. Strip bold/italic markers (`**`, `*`, `__`, `_`)
6. Strip backticks
7. Normalize list markers (`-`, `*`, `N.` → `•`)
8. Collapse blank lines
9. Restore placeholders

**Post-strip fixup** (`_fixup_ops_for_stripped`): Because the LLM received stripped text but the document contains markdown, extracted `before_text` is replaced with original content using line numbers, and `before_hash` is recomputed against the real document.

**Artifacts**: `masks/{chunk}_MDSTRIP_0.txt`, `masks/{chunk}_MDSTRIP_0.recipe.json`

**Configuration** (`reviewer.md_strip.*`): `enabled`, `backtick_threshold`, `bold_threshold`, `empty_heading_threshold`, `hr_threshold`, `code_span_threshold`, `combined_threshold`.

### v7.3.2 Acceptance Fixes

`md_apply_ops` v1.1.0 introduces three pre-validation deterministic fixes in `_normalize_op()`, applied before the 6-invariant validation. These fixes address systematic mismatches between LLM-generated ops (which see masked/stripped text) and the real document.

| Fix | Trigger | Action |
|-----|---------|--------|
| **HASH_RECOMPUTE** | Hash mismatch from masked text | Recompute `before_hash` from original document lines at `line_start:line_end` |
| **NOTE_INJECT** | `needs_attention=true` but no `review_note` | Inject `ReviewNote(alert=NOTE, text="REVIEWER: {id} — flagged for attention")` |
| **NOTE_ID_FIX** | `review_note.text` missing change ID | Prepend `REVIEWER: {op.id} — ` to the note text |

**Rejection classification** (`_classify_rejection()`): funnel taxonomy for rejected ops — `REJECT_HASH`, `REJECT_SCHEMA_NOTE`, `REJECT_SCHEMA_ID`, `REJECT_LEDGER`, `REJECT_PROTECTED`, `REJECT_RANGE`, `REJECT_OTHER`.

**Result on production run**: 0% → 100% acceptance (169/169 ops applied). Fixups: 90 hash recomputes, 76 note injections, 27 ID fixes. 103 inline review notes injected, 169 report changes (52 attention, 67 silent).

### Blank-Line Targeting Guard (v2.4.0)

Quality analysis of the 169 applied ops revealed that 34/46 harmful edits shared the same root cause: the LLM targets line N (a blank separator between paragraphs) and replaces it with hallucinated content, leaving the real target at line N+1 untouched. This off-by-one pattern accounts for 73.9% of all harmful edits.

v2.4.0 adds a blank-line targeting guard in `_build_edit_op`, after line clamping but before the protected region check:

1. **Detection**: If `action == "replace"` and the target span is entirely blank but the LLM's `before_text` contains real content (>10 chars)
2. **Relocation attempt**: Search ±3 lines from the target for a fuzzy match (first 40 chars, case-insensitive)
3. **If match found**: Relocate the target to the matching line (`_blank_line_guard: relocated_delta=N`)
4. **If no match**: Downgrade the action to `flag_only` (`_blank_line_guard: downgraded`)

This eliminates the off-by-one class of harmful edits deterministically. The guard is logged in the op's audit trail for traceability.

### Benchmark Results

#### Integration Test (13 Chunks)

Tested on a 5,600-line French technical audit document (SIAS v7), 13 chunks, using `gpt-oss-safeguard:120b` (sovereign model via Ollama).

| Metric | v6 (baseline) | v7.0 (preflight) | v7.1 (adaptive tier) | v7.2 (content recipes) |
|--------|:---:|:---:|:---:|:---:|
| Parse success | 62% | 100% | 100% | 100% |
| Degenerate chunks | 5 | 5 | 2 | **0** |
| Edit ops produced | 7 | 3 | 5 | 8 |
| `json_strict` extractions | — | 7 | 10 | 12 |
| Policy A recoveries | — | — | 3 | 3 |
| Policy C fires | — | — | — | 2 |
| Policy C recoveries | — | — | — | **2** |
| Short-circuited (END_EDIT_OPS) | — | — | — | 12 |
| Total retries | — | — | — | 0 |
| Avg chunk latency (OK) | — | — | — | 49.3s |

**Key findings (13-chunk):**

- v7.0 (preflight guards) fixed parse failures entirely: 62% → 100% success. Math masking and context tiering eliminated the parsing issue.
- v7.1 (adaptive tier) recovered 3 degenerate chunks via Policy A (tier=0 fallback). Degenerates 5 → 2.
- v7.2 (content recipes) recovered the 2 remaining degenerates via Policy C:
  - `15-vulnérabilités-et-risques-critiques`: recovered by `table_mask` + `synonym_map` (79% token reduction, 373→80t)
  - `122-résultats-consolidés`: recovered by `blockquote_flatten` + `emoji_mask` (57% token reduction, 423→180t). A 13-line blockquote with embedded table, math, and 9 emojis was flattened to a single placeholder.

#### Production Run (249 Chunks, v7.3.1)

Full-document production run on the same SIAS v7 document (5,609 lines, 249 chunks), `gpt-oss-safeguard:120b`.

| Metric | v7.3 (expanded triggers) | v7.3.1 (+ Policy D) |
|--------|:---:|:---:|
| Parse success | 91.2% (227/249) | **100%** |
| Degenerate chunks | 13/249 (5.2%) | **13/249 (5.2%)** |
| Edit ops produced | 147 | 169 |
| `json_strict` extractions | 70.3% (175/249) | 70.3% |
| Short-circuited (END_EDIT_OPS) | 70.3% | 70.3% |
| Policy C recipe fires | 55 | 55 |
| Policy C recoveries | 44 (80%) | 44 (80%) |
| Policy D fires | — | 12 |
| Policy D recoveries | — | 8 (67%) |
| Avg chunk latency | 44.8s | 44.8s |
| Total wall-clock time | 3.1h | 3.1h |

#### v7.3.2 Acceptance Results

| Metric | Before (v7.3.1) | After (v7.3.2) |
|--------|:---:|:---:|
| Ops submitted to `md_apply_ops` | 169 | 169 |
| Ops accepted | **0** (0%) | **169** (100%) |
| Hash recomputes (HASH_RECOMPUTE) | — | 90 |
| Note injections (NOTE_INJECT) | — | 76 |
| ID fixes (NOTE_ID_FIX) | — | 27 |
| Inline review notes | 0 | 103 |
| Attention changes | 0 | 52 |
| Silent minor changes | 0 | 67 |
| Document diff | 0 lines | +340 lines, +2,912 bytes |

#### Quality Assessment (v7.3.2 Applied Ops)

Manual review of the 169 applied ops on the SIAS document:

| Category | Count | Percentage | Description |
|----------|:-----:|:----------:|-------------|
| **VALUABLE** | 21 | 12.4% | Genuine fixes: typos, grammar, factual errors |
| **COSMETIC** | 61 | 36.1% | Correct but low-value: formatting, redundant punctuation |
| **GARBLED** | 41 | 24.3% | Malformed or confused outputs from LLM |
| **HARMFUL** | 46 | 27.2% | Edits that would damage the document |

**Root cause of HARMFUL edits:** 34/46 (73.9%) share the off-by-one blank-line targeting bug — the LLM targets a blank separator line and replaces it with hallucinated content. The v2.4.0 blank-line guard eliminates this class entirely, projecting HARMFUL from 27.2% to ~7.1%.

**Single highest-value finding:** `deux approches complémentaires` → `trois approches complémentaires` (line 367) — a factual error where the document claimed 2 approaches but the following table lists 3.

#### Deterministic Scan Results (v2.0)

`md_consistency_scan` v2.0 on the same document (no LLM cost):

| Detector | Issues Found | Examples |
|----------|:---:|---------|
| French cross-refs | 2 | `§10.2`, `§6.4` on navigation bar — sections not in heading tree |
| Numerical contradictions | 3 | `deux approches` vs 3 table rows (L367), `5 classes` vs 2 rows (L1210), `2 workflows` vs 3 rows (L2845) |
| Table percentage mismatch | 3 | Sums of 119.8%, 137.0%, 165.0% (likely multi-category overlap) |
| Terminology drift | 1 | "refactoring" (2×) vs "refonte" (1×) in section S17.13 |
| AI leftovers | 0 | — |
| Duplicated paragraphs | 0 | — |
| Broken refs (English) | 0 | — |

The deterministic scan found the same `deux approches` bug that was the single highest-value LLM finding — at zero LLM cost. This validates the design principle: run deterministic checks first, then target LLM review where anomalies are flagged.

#### Multi-Model Comparison: Sovereign 120b vs Granite4 32b

The same v7.3.1 pipeline was run on the identical 249-chunk document using a second sovereign model — IBM Granite4 (32b-a9b parameter-count, instruction-tuned). This validates that the kernel contract is model-agnostic and provides a cross-model quality comparison.

**Pipeline Metrics:**

| Metric | gpt-oss-safeguard:120b | ibm/granite4:32b | Notes |
|--------|:---:|:---:|-------|
| Chunks processed | 249 | 249 | Same document, same chunking |
| Parse success | 100% | 100% | Both achieve full parse |
| Degenerate chunks | 13 (5.2%) | **0 (0%)** | Granite4 never degenerates |
| Edit ops produced | 169 | 474 | 2.8× more ops from Granite4 |
| `json_strict` extractions | 70.3% (175) | **97.2% (242)** | Near-perfect JSON compliance |
| Short-circuited (END_EDIT_OPS) | 70.3% | **97.6% (243)** | Almost all emit end marker |
| Avg chunk latency | 44.8s | **20.6s** | 2.2× faster per chunk |
| Total wall-clock time | 3.1h | **1.4h** | 2.2× faster end-to-end |
| TFV (time-to-first-visible) | 342–2,007 tokens | **1 token** | Instruct model responds instantly |
| Adaptive fallbacks | 6 | 0 | No tier fallback needed |
| Content recipes fired | 55 | 0 | No content masking needed |
| Policy D fires | 12 | 0 | No markdown stripping needed |

**Key pipeline finding:** Granite4's instruction-following compliance eliminates the need for all adaptive policies (A, B, C, D). Every chunk succeeds at the default context tier (tier=1) on the first attempt. The 6-level extraction ladder still operates (5 split-merged, 2 freeform salvage), but the adaptive retry and content recipe layers are never triggered.

**Stage 3 Acceptance (Granite4):**

| Metric | Value |
|--------|:---:|
| Ops submitted | 474 |
| Ops accepted | **474 (100%)** |
| Hash recomputes | 369 |
| Note injections | 348 |
| ID fixes | 0 |
| Attention changes | 75 |
| Silent minor changes | 122 |

All 474 ops pass the same v7.3.2 acceptance normalization pipeline. Hash recompute rate is higher (78% vs 53%) because Granite4 operates at tier=1 (section context included), producing more masked-text hash mismatches.

**Quality Assessment (Granite4 474 ops):**

| Category | Count | Percentage | Description |
|----------|:-----:|:----------:|-------------|
| **VALUABLE** | 8 | 1.7% | Genuine fixes: typos, grammar, missing prepositions |
| **COSMETIC** | 382 | 80.6% | Correct but low-value: formatting, punctuation, style |
| **GARBLED** | 39 | 8.2% | Malformed or confused outputs from LLM |
| **HARMFUL** | 45 | 9.5% | Edits that would damage the document |

**Comparison (120b vs Granite4):**

| Quality Metric | 120b (169 ops) | Granite4 (474 ops) |
|----------------|:---:|:---:|
| VALUABLE rate | **12.4%** | 1.7% |
| VALUABLE count | **21** | 8 |
| COSMETIC rate | 36.1% | 80.6% |
| GARBLED rate | 24.3% | 8.2% |
| HARMFUL rate | **27.2%** | 9.5% |
| Signal-to-noise | **1:7** | 1:58 |

**Key quality findings:**

1. **Granite4 is mechanically superior** — 0 degenerates, 2.2× faster, near-perfect JSON compliance. Ideal for pipeline development and regression testing.
2. **120b produces higher-value findings** — 12.4% vs 1.7% valuable rate. The larger model catches factual errors, missing logical connections, and subtle grammar issues that escape the smaller model.
3. **Granite4 failure mode: cosmetic flood** — 80.6% of ops are low-value formatting changes (bold/italic adjustments, synonym substitutions, punctuation micro-edits). This inflates the op count without adding value.
4. **Language constraint is essential** — without an explicit language directive in the prompt, Granite4 translates French terms to English (e.g., warning headers, section titles). Adding `style.language=fr` to the configuration eliminates this failure mode.
5. **Harmful rate is lower for Granite4** (9.5% vs 27.2%) — fewer garbled outputs means fewer off-target edits. The blank-line targeting guard applies to both models.

**Conclusion:** The multi-model comparison validates that the kernel contract (extraction ladder, streaming artifacts, acceptance normalization) works model-agnostically. Model selection is a quality/speed tradeoff: Granite4 for fast iteration and pipeline validation, larger sovereign models for production audit quality.

### Production Baseline Contract (v7.3.2)

v7.3.2 delivers a fully traceable, reversible Markdown reviewer pipeline for `gpt-oss-safeguard:120b`, validated on a 249-chunk production run (5,609-line SIAS v7 document). The pipeline achieves 100% parse success, 100% op acceptance, and produces 169 edit operations with full traceability. Reliability comes from six layers of deterministic kernel controls:

1. **Extraction ladder** (v2) — 6-level tolerant parsing
2. **Streaming artifacts** (v6) — per-chunk raw output + status log
3. **Adaptive tier escalation** (v7.1) — Policies A/B for context control
4. **Content recipes** (v7.2/v7.3) — fingerprint-triggered masking with 8 transforms
5. **Markdown stripping** (v7.3.1) — Policy D for residual formatting cleanup
6. **Acceptance normalization** (v7.3.2) — 3 pre-validation fixes in `_normalize_op()`

#### Non-Negotiable Kernel Invariants

The following seven properties are declared as the production contract for `md_edit_plan` v2.4.0+ and `md_apply_ops` v1.1.0+. Any future change that weakens these invariants is a breaking change requiring a new kernel version and regression validation.

| # | Invariant | Guarantee |
|---|-----------|-----------|
| 1 | **Marker-first skeleton output** | The prompt forces `BEGIN_EDIT_OPS` as the first generated token. Combined with `END_EDIT_OPS` short-circuit, this bounds LLM output and enables streaming artifact writes. |
| 2 | **Extraction ladder** | Six parser levels (json_strict → json_relaxed → yaml → tagged → plaintext → freeform) are tried in order. The kernel never rejects output that contains recoverable structure. |
| 3 | **Per-chunk artifacts** | Every chunk writes `ops/{chunk_id}.raw.txt` (raw LLM response) and appends to `status.jsonl` immediately. Partial exploitation is always possible. |
| 4 | **Kernel-owned IDs** | `RVW-NNNN` change IDs are allocated sequentially by the kernel after extraction. The LLM never generates IDs — only raw edit proposals. |
| 5 | **Adaptive tiering** | Four deterministic retry policies (A: degenerate→tier=0, B: ops=0→escalate, C: fingerprint→content recipe, D: markdown strip) fire in order. Each is logged in `adaptive_escalation` with `from`/`to`/`reason` fields. |
| 6 | **Deterministic content recipes** | Content-level masking (8 transforms: blockquote flatten, code fence mask, table mask, emoji mask, math consolidation, digit truncation, synonym map, markdown strip) is applied *before* the LLM call, reversed *after* extraction. All mappings are persisted in `stage2/masks/`. |
| 7 | **Acceptance normalization** | `md_apply_ops` applies 3 deterministic fixes (hash recompute, note injection, ID fix) *before* the 6-invariant validation, bridging the gap between masked-text LLM output and the real document. Blank-line targeting guard prevents off-by-one edits. |

#### Causal Chain

The progression from v2 to v7.2 follows a single design principle: **you don't make the model smarter; you change the interface contract so the model can be used reliably.**

```
v2 (parsing solved)
│   Problem: LLMs produce wildly varying output formats.
│   Solution: 6-level extraction ladder — accept any format, normalize to canonical EditOps.
│   Result: Parse failure eliminated as a failure mode.
│
├── v6 (skeleton + END short-circuit)
│   Problem: LLMs emit unbounded output; no streaming artifact visibility.
│   Solution: Marker-first skeleton forces BEGIN_EDIT_OPS first; END_EDIT_OPS short-circuits.
│   Result: 30-60% token savings; per-chunk artifacts written immediately.
│
├── v7.1 (adaptive tier escalation)
│   Problem: Context complexity causes degenerate decoding (2048 tokens, 0 visible chars).
│   Solution: Policy A reduces context to skeleton (tier=0); Policy B escalates for missed issues.
│   Result: Degenerates 5 → 2. Three chunks recovered by removing context overhead.
│
├── v7.2 (content recipes)
│   Problem: Content complexity (nested blockquotes, tables, emoji, math) causes degenerate
│   decoding even at tier=0.
│   Solution: Fingerprint-triggered deterministic masking removes structural triggers.
│   Result: Degenerates 2 → 0 (13-chunk). Both recovered by content simplification.
│
├── v7.3 (expanded triggers + masking)
│   Problem: v7.2 triggers miss 33/38 degenerates in 249-chunk production run (5.2% residual).
│   Solution: Data-driven trigger expansion (emoji≥5, code_fence≥1, standalone table/math).
│   8 masking functions incl. code fences and math consolidation.
│   Result: Recipe coverage 55/249, recovery 44 (80%). Degenerates 38 → 13 (5.2%).
│
├── v7.3.1 (Policy D — markdown stripping)
│   Problem: 11 chunks degenerate despite masking — residual markdown formatting
│   (bold, backticks, empty headings, HR) causes generation loop.
│   Solution: 9-step deterministic plain-text projection as late-stage retry.
│   Result: 8/12 Policy D fires recovered (67%). Parse 91.2% → 100%.
│
└── v7.3.2 (acceptance normalization)
    Problem: 169 ops produced but 0% accepted — hash mismatches (masked text ≠ real text),
    missing review notes, absent change IDs in note text.
    Solution: 3 pre-validation deterministic fixes in _normalize_op() + rejection taxonomy.
    Blank-line targeting guard in _build_edit_op (prevents 73.9% of harmful edits).
    Result: 0% → 100% acceptance. 169 ops applied, 52 attention, 67 silent, 103 notes.
```

Each step is a kernel-side intervention — the model, prompt template, and temperature remain unchanged across versions. The improvement comes entirely from the contract between kernel and model.

#### Regression Suite

The SIAS 13-chunk baseline is frozen as `tests/fixtures/sias_v72_regression.json`. The regression suite (`test_v72_regression.py`) validates:

- **Aggregate invariants**: total_chunks=13, parse_success_all=true, degenerate_max=0, json_strict_min=10, ops_min=5
- **Per-chunk invariants**: each of 13 chunks checked for parse success and non-degenerate status
- **Recipe triggers**: `122-résultats-consolidés` (blockquote_flatten + emoji_mask) and `15-vulnérabilités` (table_mask + synonym_map) must fire with expected fingerprint thresholds
- **Recipe effectiveness**: artifacts must include tokens_before/after, reduction_pct, recovered=true, ops_count, extraction_method

Run with: `python -m pytest ragix_kernels/reviewer/tests/test_v72_regression.py -v`

---

## 8. Selective Revert

### Single-Change Revert

```bash
reviewctl revert spec.md RVW-0034
```

This:
1. Loads the current `doc.edited.md`
2. Finds `RVW-0034.inverse.patch` in the patches directory
3. Applies the inverse patch (content-hash verified)
4. Generates a new pair of patches for the revert operation itself
5. Appends a revert entry to the ledger: `REVERT-RVW-0034`
6. Saves the updated document

### Bulk Revert

```bash
reviewctl revert spec.md RVW-0034 RVW-0035 RVW-0036
```

Multiple changes are reverted in **reverse chronological order** (highest sequence number first) to minimize patch conflicts. The operation stops at the first failure.

### Revert Safety

- A change can only be reverted once (double-revert is rejected)
- A revert entry cannot itself be reverted (revert-of-revert is rejected)
- If the document has been modified since the change was applied, the content-hash check in the inverse patch will detect the conflict and reject the revert cleanly

---

## 9. CLI Reference

The `reviewctl` command provides five subcommands:

### `reviewctl review`

Run the full review pipeline on a Markdown document.

```bash
reviewctl review spec.md [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-w`, `--workspace` | auto | Workspace directory |
| `--in-place` | off | Overwrite original document |
| `--strict` | off | Refuse edits touching protected regions |
| `--skip-pyramid` | off | Skip pyramidal KB construction |
| `--no-llm` | off | Deterministic passes only |
| `--backend` | `ollama` | LLM backend (`ollama` or `claude`) |
| `--endpoint` | `http://127.0.0.1:11434` | Ollama endpoint URL |
| `--model` | `mistral:instruct` | Worker model |
| `--tutor-model` | *(none)* | Tutor model for validation |
| `--strict-sovereign` | on | Reject non-local backends |
| `-v`, `--verbose` | off | Verbose output with tracebacks |

**Example — deterministic only (no LLM, no Ollama required):**

```bash
reviewctl review spec.md --no-llm
```

**Example — full pipeline with tutor validation:**

```bash
reviewctl review spec.md --model mistral:instruct --tutor-model granite3.1-moe:3b
```

**Example — non-sovereign mode (Claude API):**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
reviewctl review spec.md --backend claude --no-strict-sovereign
```

### `reviewctl report`

Generate or display the review report.

```bash
reviewctl report spec.md [-w WORKSPACE]
```

If the report already exists, it is printed to stdout. Otherwise, the `md_review_report_assemble` kernel is run to generate it.

### `reviewctl revert`

Revert one or more changes.

```bash
reviewctl revert spec.md RVW-0001 [RVW-0002 ...]
```

### `reviewctl show`

Show details of a specific change from the ledger.

```bash
reviewctl show spec.md RVW-0034 [--patch]
```

Displays: ID, timestamp, kind, severity, silent flag, rationale, scope, hashes, and optionally the full forward patch.

### `reviewctl grep`

Search ledger entries by kind, severity, or text pattern.

```bash
reviewctl grep spec.md [PATTERN] [-k KIND] [-s SEVERITY] [--silent true|false] [--reverts true|false]
```

| Option | Description |
|--------|-------------|
| `PATTERN` | Regex to match against summary, rationale, or ID |
| `-k`, `--kind` | Filter by kind (e.g., `typo`, `logic_flow`) |
| `-s`, `--severity` | Filter by severity (`minor`, `attention`, `deletion`, `critical`) |
| `--silent` | Show only silent (`true`) or non-silent (`false`) entries |
| `--reverts` | Show only revert entries (`true`) or non-reverts (`false`) |

**Examples:**

```bash
# Find all attention-level changes
reviewctl grep spec.md -s attention

# Find all terminology changes
reviewctl grep spec.md -k terminology

# Search for a specific word in summaries
reviewctl grep spec.md "bandwidth"

# Show only reverts
reviewctl grep spec.md --reverts true
```

---

## 10. MCP Integration

KOAS-Review provides 4 MCP tools compatible with Claude Code and Claude Desktop. They are registered via a single call:

```python
from ragix_kernels.reviewer.mcp import register_reviewer_tools
register_reviewer_tools(mcp_server)
```

### `review_md_run`

Run the full review pipeline. Accepts the same parameters as the CLI `review` command.

```python
result = review_md_run(
    doc_path="docs/spec.md",
    backend="ollama",
    model="mistral:instruct",
    skip_pyramid=False,
    no_llm=False,
)
# result["status"] == "completed"
# result["total_changes"] == 12
# result["workspace"] == "/path/to/.review/spec_a3f8bc21f0e1"
```

### `review_md_status`

Query review status and statistics without running the pipeline.

```python
result = review_md_status(doc_path="docs/spec.md")
# result["stages_completed"] == ["stage1", "stage2", "stage3"]
# result["counts"]["total_changes"] == 12
```

### `review_md_revert`

Revert one or more changes by their IDs.

```python
result = review_md_revert(
    doc_path="docs/spec.md",
    change_ids="RVW-0034,RVW-0035",
)
# result["total_reverted"] == 2
```

### `review_md_show_change`

Show details of a specific ledger entry.

```python
result = review_md_show_change(
    doc_path="docs/spec.md",
    change_id="RVW-0034",
    include_patch=True,
)
# result["change"]["severity"] == "attention"
# result["patch"] == "--- a/spec.md\n+++ b/spec.md\n..."
```

### Return Value Convention

All MCP tools return `Dict[str, Any]` with:

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | `"completed"`, `"error"`, `"not_found"`, `"partial"` |
| `summary` | `str` | Human-readable summary (< 300 chars) |
| `workspace` | `str` | Path to the workspace directory |
| `error` | `str` | Error message (only on failure) |

Tool-specific fields are documented in each tool's docstring.

---

## 11. Configuration Reference

### ReviewerConfig

The top-level configuration object, typically passed as `input.config["reviewer"]`:

```python
ReviewerConfig(
    # Sub-configs
    chunk=ChunkConfig(...),
    llm=LLMConfig(...),
    style=StyleRules(...),

    # Output
    in_place=False,              # True → overwrite original
    output_suffix=".REVIEWED.md",

    # Pipeline
    skip_pyramid=False,          # Skip md_pyramid
    no_llm=False,                # Deterministic only
    strict=False,                # Refuse edits on protected regions

    # Pyramid
    pyramid_levels=4,            # 1-4
    summary_max_tokens=200,

    # Cache
    cache_mode="write_through",  # write_through|read_only|read_prefer|off

    # Alert mapping
    severity_alert_map={
        "minor": "NOTE",
        "attention": "WARNING",
        "deletion": "CAUTION",
        "critical": "IMPORTANT",
    },
)
```

### ChunkConfig

```python
ChunkConfig(
    max_chunk_tokens=1800,   # Maximum tokens per chunk
    overlap_tokens=100,      # Overlap between adjacent chunks
    min_chunk_tokens=50,     # Minimum viable chunk size
)
```

### LLMConfig

```python
LLMConfig(
    backend="ollama",                    # ollama | claude
    endpoint="http://127.0.0.1:11434",  # Ollama API endpoint
    pyramid_model="granite3.1-moe:3b",  # Model for pyramid summaries
    edit_model="mistral:instruct",       # Worker model for edit plan
    tutor_model=None,                    # Tutor model (None = single-model)
    temperature=0.1,
    timeout=120,
    num_predict=2048,
    strict_sovereign=True,               # Reject non-local backends
)
```

### StyleRules

```python
StyleRules(
    silent_allowlist=[
        "typo", "punctuation", "capitalization", "spacing", "grammar",
    ],
    ai_leftover_patterns=[
        r"(?i)\bAs an AI\b",
        r"(?i)\bAs a language model\b",
        r"(?i)\bIn conclusion\b",
        r"(?i)\bSure,?\s+here['']?s\b",
        r"(?i)\bI hope this helps\b",
        r"(?i)\bLet me know if\b",
        r"(?i)\bHere is (?:a|an|the)\b",
        r"(?i)\bCertainly!",
        r"(?i)\bAbsolutely!",
        r"(?i)\bGreat question",
    ],
    protect_tables=True,
    protect_math=True,
)
```

### LLM Backend Selection

| Backend | Command | Sovereignty | Requirements |
|---------|---------|:-----------:|--------------|
| Ollama (default) | `--backend ollama` | Local | Ollama running on `--endpoint` |
| Claude API | `--backend claude --no-strict-sovereign` | Non-local | `ANTHROPIC_API_KEY` env var |

When `--strict-sovereign` is active (default), requesting the Claude backend raises `SovereigntyError`.

---

## 12. Data Model Reference

### ChangeID

Format: `RVW-NNNN` (zero-padded, sequential). Allocated by the `Ledger` via `allocate_id()`.

```python
cid = ChangeID(seq=34)
cid.canonical   # "RVW-0034"
cid.display     # "RVW-0034" (or "RVW-NS-0034" with namespace)

ChangeID.parse("RVW-0034")  # → ChangeID(seq=34)
```

### ProtectedSpan

```python
ProtectedSpan(
    kind=ProtectedKind.CODE_FENCE,
    line_start=42,       # 1-based inclusive
    line_end=58,         # 1-based inclusive
    content_hash="sha256:...",
    info="python",       # Language tag for code fences
)
```

Methods: `contains_line(line)`, `overlaps(start, end)`, `to_dict()`, `from_dict()`.

### HeadingNode

Recursive tree node representing a section:

```python
HeadingNode(
    id="S2.3",
    level=3,
    title="Methods",
    anchor="methods",
    line_start=42,
    line_end=128,
    numbering="2.3",
    children=[HeadingNode(...), ...],
)
```

### ReviewChunk

```python
ReviewChunk(
    chunk_id="methods_a3f8bc21",
    section_id="S2.3",
    line_start=43,       # 1-based
    line_end=87,         # 1-based
    token_estimate=1200,
    content_hash="sha256:...",
)
```

### PyramidNode

```python
PyramidNode(
    node_id="S2.3",
    level=1,             # 0=abstract, 1=section, 2=subsection, 3=paragraph
    heading="Methods",
    anchor="methods",
    content_hash="sha256:...",
    summary="This section describes the experimental methods...",
    children=["S2.3.1", "S2.3.2"],
    token_count=350,
)
```

### Utility Functions

```python
content_hash("text")     # → "sha256:a3f8bc21..."
estimate_tokens("text")  # → len(text) // 4  (rough approximation)
```

---

## 13. Workspace Layout

Each review creates an isolated workspace, either auto-generated or user-specified:

```
.review/spec_a3f8bc21f0e1/       # Auto-generated: {stem}_{hash[:12]}
    stage1/
        doc.raw.md                # Immutable snapshot of the input
        md_inventory.json         # Kernel output (file stats, hash)
        md_structure.json         # Kernel output (heading tree)
        outline.json              # Flat section index
        anchors.json              # Anchor → section ID map
        md_protected_regions.json # Kernel output (protected spans)
        protected_spans.json      # Detailed protected regions
        md_chunk.json             # Kernel output (chunk plan)
        chunks.json               # Detailed chunk list
    stage2/
        md_consistency_scan.json  # Kernel output (issues)
        coherence_issues.json     # Detailed issues
        md_numbering_control.json # Kernel output (findings)
        numbering_findings.json   # Detailed findings
        pyramid.json              # Pyramid nodes (if not skipped)
        pyramid.md                # Human-readable pyramid
        md_fingerprint_chunk.json # Chunk fingerprints (v7.2)
        edit_plan.json            # Merged edit plan (all ops)
        md_edit_plan.json         # Kernel summary output
        ops/                      # Per-chunk artifacts (written immediately)
            status.jsonl          # Append-only: one line per chunk (parse, method, ops, latency)
            ops_{chunk_id}.raw.txt    # Raw LLM response (always)
            ops_{chunk_id}.json       # Canonical normalized ops (if ops > 0)
        masks/                    # Content recipe artifacts (v7.2)
            {chunk_id}_QUOTE_N.md     # Original blockquote block text
            {chunk_id}_TABLE_N.md     # Original table block text
            {chunk_id}_EMOJI.json     # Emoji placeholder mapping
            {chunk_id}_recipe.json    # Full recipe log (transforms, token delta)
        exemplars/                # Degenerate chunk snapshots (v7.2)
            {chunk_id}.json           # Fingerprint + tier history + extraction method
    stage3/
        doc.edited.md             # Edited document
        apply_log.jsonl           # Applied ops log
        doc.REVIEWED.md           # Annotated document (with alerts)
        REVIEW_spec.md            # Human-readable report (copy)
    review/
        ledger.jsonl              # Append-only change ledger
        REVIEW_spec.md            # Full review report
        patches/
            RVW-0001.patch        # Forward unified diff
            RVW-0001.inverse.patch # Inverse unified diff
            RVW-0002.patch
            RVW-0002.inverse.patch
            ...
```

### Key Invariants

- `doc.raw.md` is **never modified** after creation (immutable snapshot)
- `ledger.jsonl` is **append-only** (entries are never mutated or deleted)
- Each `RVW-NNNN.patch` + `RVW-NNNN.inverse.patch` pair is immutable once written
- Revert operations create **new** patch files (`REVERT-RVW-NNNN.patch`)
- Per-chunk raw output (`ops_*.raw.txt`) is written immediately and never overwritten — preserves LLM forensics
- `status.jsonl` is append-only with one entry per chunk, enabling `tail -f` monitoring during long runs

---

## 14. Design Rationale

### Why Kernels Instead of a Monolithic LLM Pipeline?

A monolithic approach (feed entire document to an LLM, get corrections back) fails at scale:

1. **Context window limits.** A 200-page specification does not fit in any local model's context. Even 128K-token models cannot reliably attend to the full document.

2. **Non-reproducibility.** LLM outputs vary across runs. By isolating LLM involvement to two specific kernels (`md_pyramid` and `md_edit_plan`), the rest of the pipeline is fully deterministic.

3. **No audit trail.** An LLM that silently rewrites text provides no traceability. KOAS-Review records every change with its rationale, produces diffable patches, and supports selective undo.

4. **Protected content.** Technical documents contain code, formulas, and tables that must not be altered. A monolithic LLM cannot reliably respect these boundaries. KOAS-Review marks them at Stage 1 and enforces them at Stage 3.

### Why Bottom-Up Pyramid Instead of RAG?

Traditional RAG retrieves relevant chunks via similarity search. This works for question-answering but fails for document review because:

- **Spatial awareness matters.** A reviewer needs to know that Section 3 uses "throughput" while Section 5 uses "bandwidth" — this requires structured hierarchy, not cosine similarity.
- **Completeness.** RAG retrieves *some* relevant chunks. A pyramid guarantees every section is represented at every level.
- **Incremental updates.** Hash-based caching means the pyramid is free to maintain on re-runs. RAG indices must be fully recomputed.

### Why Forward + Inverse Patches?

Storing only the edited document makes revert impossible. Storing the full text before and after each change is wasteful. Unified diffs are compact, standard, and support partial application.

The inverse patch is generated at the same time as the forward patch (before the document state advances to the next edit), ensuring it always corresponds exactly to the forward change.

### Why Append-Only Ledger?

The ledger design follows the principle of **immutable audit trails**:

- No entry is ever deleted or modified
- Reverts are new entries that reference the original
- The ledger can be replayed to reconstruct any historical state of the document
- Thread-safe concurrent access is trivial (append is the only write operation)

### Why a 6-Level Extraction Ladder Instead of Strict JSON?

Requiring strict JSON from LLMs is fragile in practice:

- **Local models (7B–120B)** frequently produce trailing commas, single quotes, unescaped characters, or wrap JSON in markdown fences — all of which break `json.loads()`.
- **Large sovereign models** (e.g., `gpt-oss-safeguard:120b`) were observed to produce valid edit operations in prose, YAML, or non-standard JSON 50–80% of the time when tested on real 5,600-line audit documents.
- **Model-agnosticism** is a design goal: the kernel should yield useful results regardless of which LLM is behind it.

The extraction ladder resolves this by trying progressively more tolerant parsers. The kernel logs which level succeeded per chunk (`extraction_method` field), providing a quantitative measure of LLM format compliance. Combined with the circuit breaker (which switches to a simpler prompt when structured output fails too often), this makes the pipeline resilient to model-specific formatting quirks while preserving full traceability.

### Why Content Recipes Instead of Prompt Engineering?

The v7.2 content recipe approach embodies three design principles validated by benchmarking:

1. **Adaptive tiering is model-agnostic.** Reducing context (tier=0) recovers chunks where context complexity, not content complexity, causes degenerate decoding. This works across model families because it reduces prompt size, which is a universal cost.

2. **Content recipes convert intrinsic failure to recoverable preprocessing.** When the *content itself* triggers degenerate decoding (nested blockquotes + embedded tables + math + emoji), no amount of prompt rewording helps — the model cannot even begin generating visible output. Deterministic masking removes the structural triggers before the LLM sees the text.

3. **Sovereign models are operationally usable with kernel normalization.** A 120B local model that produces valid JSON 85% of the time and degenerates on 15% of structurally complex chunks is *usable* when the kernel provides: (a) a 6-level extraction ladder, (b) adaptive tier escalation, and (c) content recipes. The combination brings a 62% baseline parse rate to 100% with 0-1 degenerate chunks.

The alternative — prompt engineering to "convince" the model to handle complex content — was rejected because:
- It is model-specific (a prompt that works for Mistral fails for GPT-OSS)
- It increases token budget (more instructions = less room for content)
- It provides no guarantee (the model may still degenerate on structurally adversarial content)
- It is not auditable (the "fix" is invisible in the prompt template)

Content recipes are deterministic, auditable (every transform is logged with its mapping), reversible (unmasking restores original content in extracted ops), and model-independent.

### Why `REVIEWER:` Prefix in Alert Blocks?

The prefix serves two purposes:

1. **Machine-parseable.** Downstream tools can `grep -n "REVIEWER: RVW-"` to find all review notes and their IDs.
2. **Human-visible.** The reader immediately knows this annotation was placed by the KOAS Reviewer, not by a human editor, and can look up the change ID in the ledger for full context.

---

*Generated by KOAS Reviewer v0.4.0 — kernel-enforced, fully traceable, production validated on 249 chunks.*
