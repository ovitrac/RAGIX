This document describe the design for a new KOAS (with a similar architecture of ragix_kernel/docs/ and to be developed in ragix_kernerls/reviewer/) for a **traceable + reversible Markdown reviewer/editor tool** that works on documents larger than context windows, produces **human-navigable inline review notes**, and supports **selective undo by change ID**.

The pyramidal construction to overcome context limitation may require discussion as it can use different approaches including KOAS docs capabilities. 

---

# GENERAL SPECIFICATIONS
## 1) Core objectives

### Non-negotiables

- **Everything traceable**: every edit has an ID, provenance, scope, and rationale.
    
- **Everything reversible**: selective revert (by edit ID) without rolling back unrelated edits.
    
- **Markdown-only** (for now): preserve structure, links, code fences, tables, footnotes, front-matter.
    
- **Inline visibility when human attention is needed**: use GitHub Alerts; **must start with `REVIEWER:`** for fast navigation.
    
- **No silent deletions**: any deletion that changes meaning or removes content must be explicitly commented inline.
    
- **Undecidable cases remain as text**: tool can flag and propose; user decides.
    

### What can be silent

- Typos, punctuation, capitalization, spacing, basic grammar _if_:
    
    - each is still logged (with IDs),
        
    - reversible,
        
    - summarized in a generated review report file `REVIEW_<original>.md`.
        

---

## 2) Output artifacts and file layout

Given `doc.md`, the tool produces:

1. **Edited document** (in place, or as `doc.REVIEWED.md` depending on your workflow)
    
2. **Review report**: `REVIEW_doc.md`
    
3. **Edit ledger** (machine-readable): `.review/ledger.jsonl`
    
4. **Patch store** (one patch per change ID): `.review/patches/<CHANGE_ID>.patch`
    
5. **Pyramid KB** (summaries + anchors): `.review/kb/<doc_hash>/...`
    

Suggested structure:

```
doc.md
REVIEW_doc.md
.review/
  ledger.jsonl
  patches/
    RVW-0001.patch
    RVW-0002.patch
    ...
  kb/
    <doc_hash>/
      outline.json
      pyramid.jsonl
      section_summaries/
        S1.md
        S1.1.md
        ...
  runs/
    2026-02-06T11-00-00Z.json
```

---

## 3) Change identification and selective revert

### Change IDs

Use a stable prefix + zero-padded integer:

- `RVW-0001`, `RVW-0002`, …
    

Optionally allow namespaces:

- `RVW-GRAM-0007` (grammar)
    
- `RVW-STRUCT-0012` (structure/numbering)
    
- `RVW-FACT-0003` (factual/logic)  
    …but keep **one canonical ID** for revert.
    

### Revert command (selective)

- `reviewctl revert doc.md RVW-0012`
    
- `reviewctl revert doc.md RVW-0012 RVW-0031`
    
- `reviewctl revert doc.md --since 2026-02-06T10:00:00Z`
    
- `reviewctl revert doc.md --all` (revert entire run)
    

Mechanism: each change stores an **inverse patch** (or stores both forward+inverse), so revert is deterministic and local.

---

## 4) Inline review notes format (GitHub Alerts)

When human attention is needed (meaning, style/flow ambiguity, questionable deletion, inconsistencies), append an alert block **immediately after** the modified paragraph/section.

**Required header**: `REVIEWER:` at the start of the alert content.

Example:

```md
Rewritten sentence here for coherence and tense alignment.

> [!WARNING]
> REVIEWER: RVW-0034 — Flow inconsistency detected (likely leftover AI sentence).
> I removed the sentence “In conclusion, the future is bright.” because it breaks the argumentative chain.
> If you intended a transition, consider adding a bridge sentence connecting §2.3 to §2.4.
```

Deletion must never be silent:

```md
> [!CAUTION]
> REVIEWER: RVW-0041 — Content deletion.
> I deleted a duplicated paragraph (same meaning as §1.2, different wording). Please confirm the deletion is acceptable.
```

Undecidable case:

```md
> [!NOTE]
> REVIEWER: RVW-0050 — Undecidable revision.
> The claim “X always implies Y” may be too strong. I did not change it.
> Options: (a) add conditions, (b) cite [XXX], or (c) weaken wording (“often”, “typically”).
```

---

## 5) Ledger format (JSONL) and patch store

### `ledger.jsonl` entry (one line per change)

Each entry should contain enough to audit, filter, and revert:

```json
{
  "id": "RVW-0034",
  "doc": "doc.md",
  "doc_hash_before": "sha256:...",
  "timestamp": "2026-02-06T10:58:21Z",
  "actor": {"tool": "reviewctl", "model": "local_llm_X", "operator": "Claude Code"},
  "scope": {"section": "2.3", "heading": "Reasoning, Acting, and Their Limitations", "anchor": "doc.md#reasoning-acting"},
  "kind": "logic_flow",
  "severity": "attention",
  "silent": false,
  "summary": "Removed illogical transition sentence; added reviewer note",
  "rationale": "Sentence was off-topic and broke argument chain; likely leftover from prior agent.",
  "patch_forward": ".review/patches/RVW-0034.patch",
  "patch_inverse": ".review/patches/RVW-0034.inverse.patch",
  "review_note": {"type": "WARNING", "inserted": true, "text_hash": "sha256:..."},
  "constraints": {"no_silent_deletion": true, "md_preserve_code_fences": true}
}
```

Silent edits still log:

```json
{
  "id": "RVW-0102",
  "kind": "typo",
  "severity": "minor",
  "silent": true,
  "summary": "Fixed comma splice in §1.1",
  ...
}
```

### Patch format

Use standard unified diff (`.patch`) targeting exact ranges. For robustness across shifting line numbers, you can:

- anchor patches on **section content hashes**, or
    
- store **AST-aware patch operations** (see next section).
    

---

## 6) Parsing and safety: Markdown AST + “protected regions”

To avoid corrupting Markdown:

- Parse Markdown to an AST (e.g., `markdown-it` / `mistune` / `remark` stack).
    
- Mark **protected blocks** that must never be edited unless explicitly allowed:
    
    - fenced code blocks
        
    - inline code
        
    - YAML front matter
        
    - tables (optional: edit carefully with a table-aware routine)
        
    - math blocks (even if Markdown supports them)
        
    - reference link definitions
        

A practical rule set:

- **Default editable nodes**: paragraphs, headings, list items, blockquotes.
    
- **Default protected nodes**: code fences, HTML blocks, tables unless table-mode.
    

If an edit crosses protected boundaries, create a `REVIEWER:` alert and require manual approval (or refuse).

---

## 7) Numbering control (headings, figures, tables, equations)

Implement a deterministic “structure auditor” pass:

### Checks

- Heading order (no skipping levels unless allowed)
    
- Unique anchors for headings
    
- Table/Figure numbering sequence if you use explicit labels (`Table 1`, `Figure 2`, etc.)
    
- Cross-references:
    
    - references to missing `Table X`, `Figure Y`
        
    - duplicates (`Figure 3` used twice)
        
- Optional: enforce consistent caption style
    

### Behavior

- If fix is **purely mechanical** and unambiguous: can be silent (but logged).
    
- If fix changes meaning (renumbering may affect external references): add alert.
    

---

## 8) Detecting “multi-human / multi-agent drift” and incoherence

Add a dedicated pass to detect typical failure modes:

- abrupt register shifts (formal → casual)
    
- contradictory statements across sections
    
- duplicated paragraphs with paraphrase-level similarity
    
- leftover agent phrases (“As an AI…”, “In conclusion…”, “Sure, here’s…”)
    
- broken referents (“this”, “it”, “they” without antecedent)
    
- inconsistent terminology (same concept with 3 different names)
    

### Policy

- **Deletions are never silent**.
    
- For “AI leftovers”, default action can be:
    
    - remove + comment (`REVIEWER:` WARNING), or
        
    - keep + flag (NOTE) if unsure.
        

---

## 9) Handling documents larger than context: Pyramid KB

### Goal

Maintain global coherence while editing locally.

### Representation

Create a hierarchical memory structure aligned to the document outline:

- Level 0: document abstract (≤ N tokens)
    
- Level 1: per top-level section abstracts
    
- Level 2: per subsection abstracts
    
- Level 3: optional micro-summaries per paragraph group (only when needed)
    

Store:

- `outline.json` (headings tree with anchors, line ranges, hashes)
    
- `pyramid.jsonl` entries:
    
    - `{ "node": "2.3", "hash": "...", "summary": "...", "keywords": [...], "claims": [...], "dependencies": [...] }`
        

### Editing workflow with pyramid

1. Build/refresh outline + hashes.
    
2. Build summaries bottom-up:
    
    - summarize leaf sections first,
        
    - then summarize parents using children summaries.
        
3. When editing a local chunk:
    
    - inject into context:
        
        - doc abstract,
            
        - sibling summaries (small),
            
        - current section summary (detailed),
            
        - the target text chunk.
            

### Granularity adaptation

If the context budget is tight:

- keep parent summaries,
    
- drop distant siblings,
    
- keep a “global glossary/terminology map” extracted once.
    

---

## 10) Two-phase engine: deterministic core + LLM reviewer

To keep it robust and audit-friendly:

### Phase A — Deterministic passes (no LLM)

- Markdown parse + protected region marking
    
- numbering/structure auditor
    
- “agent leftovers” regex detector (cheap wins)
    
- duplicate detection via hashes/similarity
    
- glossary/term consistency check (optional)
    
- pyramid KB refresh triggers (only for changed sections)
    

### Phase B — LLM-assisted revisions

- Provide constrained editing instructions:
    
    - “Edit only editable nodes”
        
    - “No silent deletion”
        
    - “If uncertain: add `REVIEWER:` NOTE”
        
    - “Every attention-worthy change must have alert block after it”
        
- LLM outputs **structured edit ops** rather than raw text:
    
    - `replace_range`, `insert_after`, `delete_range` (but deletion always coupled with a note insertion)
        

Then deterministic core:

- validates ops,
    
- applies ops,
    
- generates patches + inverse patches,
    
- updates ledger,
    
- regenerates affected pyramid nodes.
    

---

## 11) CLI interface (minimal but complete)

### Review command

- `reviewctl review doc.md`  
    Options:
    
- `--in-place` / `--output doc.REVIEWED.md`
    
- `--max-chunk-tokens 1800`
    
- `--silent-minor true`
    
- `--strict` (refuse edits touching protected areas)
    
- `--rules rules.yaml` (your house style)
    

### Report

- `reviewctl report doc.md`  
    Regenerates `REVIEW_doc.md` from ledger (useful if you want clean reports).
    

### Revert

- `reviewctl revert doc.md RVW-0034`
    
- `reviewctl revert doc.md --range RVW-0034:RVW-0040`
    
- `reviewctl revert doc.md --kind typo` (bulk revert category)
    

### Inspect

- `reviewctl show doc.md RVW-0034` (shows before/after snippet + rationale)
    
- `reviewctl grep doc.md --severity attention`
    

---

## 12) REVIEW_doc.md content template

`REVIEW_doc.md` should be **human-scannable** and aligned with your traceability constraints:

- Run metadata (date, model/tool versions, doc hash)
    
- Summary counts:
    
    - attention changes
        
    - deletions
        
    - silent minor edits
        
- Per-section table:
    
    - section id, change IDs, severities, short rationales
        
- Full change log (chronological)
    

Example skeleton:

```md
# Review Report — doc.md
**Run ID:** [XXX]
**Date:** 2026-02-06
**Tool:** reviewctl [version]
**Model:** [local LLM name]
**Doc hash (before):** sha256:[...]

## Overview
- Attention changes: 7
- Deletions (commented): 2
- Silent minor edits (logged): 31

## By section
### 2.3 Reasoning, Acting, and Their Limitations
- RVW-0034 (WARNING): Removed illogical leftover sentence; added inline note.
- RVW-0039 (NOTE): Undecidable strength of claim “X always implies Y” — left as-is, suggested options.

## Full ledger (chronological)
- RVW-0001 ...  
- RVW-0002 ...
```

---

## 13) Integration with Claude Code

The cleanest integration pattern is:

- Claude Code orchestrates tasks (read → plan → call `reviewctl` → inspect outputs).
    
- `reviewctl` is the deterministic editor/ledger/patch engine.
    
- The LLM is used only for:
    
    - proposing edits as structured ops,
        
    - generating reviewer notes text,
        
    - generating pyramid summaries.
        

This matches your “tool + trace + deterministic contracts” philosophy: Claude Code becomes the _agentic shell_, while `reviewctl` is the _auditable kernel_.

---

## 14) A minimal “edit op” schema (for tool-calling models)

If you want to drive edits via structured outputs (highly recommended), define:

```json
{
  "ops": [
    {
      "id": "RVW-0034",
      "action": "replace",
      "target": {"anchor": "doc.md#reasoning-acting", "node_path": "2.3.p5"},
      "before_hash": "sha256:...",
      "after_text": "....",
      "needs_attention": true,
      "review_note": {
        "alert": "WARNING",
        "text": "REVIEWER: RVW-0034 — ... explanation ..."
      }
    }
  ]
}
```

The deterministic engine rejects ops if:

- `before_hash` mismatches (document changed),
    
- target crosses protected nodes,
    
- deletion occurs without `review_note`,
    
- ID collisions occur.
    

---

If you want, I can turn this into a **ready-to-implement spec pack** (still Markdown-only):

- `PHILOSOPHY.md` (traceability rules, attention policy, undecidability policy)
    
- `SPEC.md` (schemas, CLI, file layout, invariants)
    
- `rules.example.yaml` (style rules, protected regions, alert thresholds)
    
- `CHANGE_ID_POLICY.md` (ID formats, revert semantics)
    

…and keep it aligned with your existing “kernel-orchestrated” approach so Claude Code can run it as a deterministic tool in the loop.

---
# KOAS integration

## 1) Where it plugs into KOAS / RAGIX

### Reuse KOAS principles verbatim

- **Kernels are pure computation units** with explicit `requires/provides`, deterministic outputs, and a chain-hashed audit trail.
    
- **Three-stage pipeline** (collection → analysis → reporting) and DAG orchestration remain identical.
    
- **LLM at the edge** (only where natural language synthesis is required) is the KOAS-Docs pattern you already formalized.
    
- **MCP exposure** should follow KOAS MCP design: simplified interfaces, mandatory summaries, auto-workspace, details in files.
    


This family extends KOAS-Docs from “summarize/compare corpus” to “edit one Markdown document with reversible trace”.

It uses the KOAS-Docs pyramid idea _inside a single document_ (section summaries + hierarchy) to handle context limits OR create something similar.

---

## 2) Kernel map (KOAS stages) for reversible Markdown editing

### Stage 1 — Collection (deterministic)

1. `md_inventory`
    

- Input: `doc_path`
    
- Output: file stats, hashes, detected front-matter, code fences, tables, heading map
    

2. `md_structure`
    

- Extract heading tree, section anchors, numbering patterns (explicit “1.2”, “Figure 3”, etc.)
    
- Output: `section_index`, `heading_tree`, `anchor_map`
    

3. `md_protected_regions`
    

- Locate protected spans (fenced code blocks, inline code, YAML front matter, link refs, tables if “protect_tables=true”)
    
- Output: `protected_spans`
    

4. `md_chunk`
    

- Produce chunk plan aligned to structure + protected spans
    
- Output: `chunks.json` (stable chunk IDs based on content hash + structural anchor)
    

_(This is the “doc_structure” spirit applied to one Markdown file.)_

---

### Stage 2 — Analysis (mostly deterministic, with optional LLM at edge)

5. `md_consistency_scan`  
    Detect:
    

- “multi-agent drift” leftovers, broken discourse, register shifts, duplicated paragraphs, contradictory local claims, orphan references.
    
- Output: `issues.json` with severity + locations + evidence (hashes/snippets)
    

6. `md_numbering_control`
    

- Validate heading numbering, table/figure numbering, cross-references (“see Figure 4” exists?), duplicates.
    
- Output: `numbering_findings.json`, and (optionally) deterministic renumber proposal ops.
    

7. `md_pyramid`
    

- Build **hierarchical summaries**: section → subsection → doc abstract.
    
- Output: `pyramid.json` + `pyramid.md` (stable per section hash)
    
- This is the “doc_pyramid” idea but single-document scoped.
    

8. `md_edit_plan` (**LLM edge, constrained**)
    

- Input: current chunk + relevant pyramid slices + issue list + style rules
    
- Output: **structured edit ops** (no raw freeform rewriting), each with:
    
    - change ID (`RVW-0001`, …),
        
    - action (`replace/insert/delete`),
        
    - target (chunk id + local offsets + before_hash),
        
    - `needs_attention` boolean,
        
    - optional `review_note` (GitHub alert block text, **starting with `REVIEWER:`**),
        
    - rationale.
        

This is where you enforce: **“no silent deletion”** (deletion requires a note). Everything else can be silent if categorized minor and still logged.

_(Analogous to KOAS-Docs where the LLM consumes structured extracts rather than raw corpora.)_

---

### Stage 3 — Reporting / Application (deterministic)

9. `md_apply_ops`
    

- Validate ops against protected spans + hashes
    
- Apply edits
    
- Emit:
    
    - forward patch
        
    - inverse patch
        
    - updated doc hash
        
- Output: `.review/patches/*.patch`, `.inverse.patch`, `apply_log.jsonl`
    

10. `md_inline_notes_inject`
    

- For every `needs_attention=true`, inject GitHub alert block right after the change.
    
- Output: updated doc text (same file or `*.REVIEWED.md`)
    

11. `md_review_report_assemble`
    

- Generate `REVIEW_<doc>.md`:
    
    - run metadata, counts, per-section lists, full ledger summary
        
- Output: `REVIEW_doc.md`
    

12. `md_revert`
    

- Deterministic revert by change IDs using stored inverse patches:
    
    - `revert(RVW-0034)` applies only the inverse patch for that change.
        
- Output: reverted document + updated ledger entry (“revert event”).
    

This mirrors KOAS “stage3 report assembly” and auditability chain style.

---

## 3) Workspace, audit trail, and sovereignty guarantees

Follow KOAS-Docs workspace conventions (stage1/stage2/stage3 + audit_trail).

Recommended workspace per run:

```
.KOAS/runs/run_<timestamp>_<uid>/
  stage1/
  stage2/
  stage3/
  review/
    ledger.jsonl
    patches/
    pyramid/
    protected_spans.json
    ops/
```

Audit trail should include:

- doc hash before/after
    
- chunk hashes
    
- model name + digest (if local via Ollama, record endpoint + model digest)
    
- “external_calls: 0” assertion (same sovereignty framing you use).
    

---

## 4) MCP tool surface (RAGIX-friendly, KOAS-style simplification)

Following KOAS MCP rules: **single values, simplified presets, mandatory summaries, details_file, auto-workspace**.

Minimal MCP tools (first usable set):

1. `koas_docs_review_md_run`
    

- params: `doc_path`, `language`, `style_preset`, `model` (optional), `workspace` (optional)
    
- returns: `summary`, `workspace`, `report_file`, `edited_file`, `details_file`
    

2. `koas_docs_review_md_status`
    

- params: `workspace`
    
- returns: progress, last kernel summary
    

3. `koas_docs_review_md_revert`
    

- params: `workspace`, `doc_path`, `change_ids` (string: `"RVW-0004,RVW-0012"`)
    
- returns: summary + updated hashes
    

4. `koas_docs_review_md_show_change`
    

- params: `workspace`, `change_id`
    
- returns: short before/after + rationale + patch file references
    

This matches the KOAS MCP “simplified tool” direction and makes Claude Code orchestration trivial.

---

## 5) Behavior rules encoded as deterministic checks

These should be **kernel-enforced invariants** (not “prompt hopes”):

- **Deletion requires a note**: `md_apply_ops` rejects any `delete` without `review_note`.
    
- **Protected spans are immutable** unless `allow_protected_edits=true` _and_ every such edit is `needs_attention=true`.
    
- **Alert formatting contract**:
    
    - always a GitHub alert block,
        
    - starts with `REVIEWER:` and includes the change ID.
        
- **Minor silent edits**: allowed only for categories in `silent_allowlist` (typos, punctuation, capitalization) but still logged and revertible.
    
- **Undecidable**: the LLM must output a `review_note` without applying the change (op kind `flag_only`), so it stays as text.
    

This is the same “interaction contract” mindset you use for KOAS generally: kernels enforce truth; LLM proposes.

---

## 6) What you need to give me next (to lock it to _your_ KOAS_DOCS examples)

I already have the KOAS + KOAS-Docs + MCP reference you uploaded.  
To align perfectly with your existing KOAS_DOCS examples, the next step is:

- the **exact alert block variants** you want to allow (`NOTE`, `WARNING`, `CAUTION`, `IMPORTANT`, etc.)
    
- your **naming conventions** for run folders and stage filenames (so the outputs look identical to KOAS-Docs)
    
- whether the edited doc is **in-place** or emitted as `*.REVIEWED.md`
    

---
# DETAILS


1. **KOAS-Workspace Init (stage0)**
    

- Create run workspace (`.KOAS/runs/<run_id>/`), capture inputs, tool/model identifiers, and immutable “run metadata”.
    
- Emit `audit_trail` header (hashes, timestamps, operator, versions).
    

2. **Document Intake & Fingerprint (stage1 / collect)**
    

- Read `doc.md` (Markdown-only), compute `sha256` (full file) + line-count + size.
    
- Store immutable snapshot `stage1/doc.raw.md` and `stage1/doc.meta.json`.
    

3. **Markdown Structure Indexing (stage1 / collect)**
    

- Parse Markdown → build heading tree, anchors, section boundaries, and stable section IDs.
    
- Output `stage1/outline.json` + `stage1/anchors.json`.
    

4. **Protected Regions Marking (stage1 / collect)**
    

- Detect and mark immutable spans: YAML front matter, fenced code blocks, inline code, link refs, (optionally) tables.
    
- Output `stage1/protected_spans.json`.
    

5. **Deterministic Preflight Checks (stage2 / analyze)**
    

- Run KOAS-style deterministic validators:
    
    - malformed Markdown patterns,
        
    - broken fences,
        
    - duplicate anchors,
        
    - illegal edits inside protected spans (policy).
        
- Output `stage2/preflight_findings.json`; fail-fast if hard violations.
    

6. **Numbering & Cross-Reference Control (stage2 / analyze)**
    

- Validate heading numbering and sequences; detect inconsistencies in “Figure/Table/Eq” numbering if present; check referenced items exist.
    
- Output `stage2/numbering_findings.json` + optional deterministic “safe fixes” plan.
    

7. **Coherence / Drift / Artifact Scan (stage2 / analyze)**
    

- Detect multi-human / multi-agent artifacts (leftover AI sentences, register shifts, duplicated paragraphs, broken referents, contradictions).
    
- Output `stage2/coherence_issues.json` with evidence (snippets + hashes + locations).
    

8. **Pyramidal KB Construction (stage2 / analyze)**
    

- Build section-level summaries bottom-up (granularity adapted to context budget): subsection → section → doc abstract.
    
- Output `stage2/pyramid.json` + `stage2/pyramid.md` (stable by section hash).
    

9. **Chunk Plan for Context-Limited Review (stage2 / analyze)**
    

- Create a deterministic chunk schedule aligned to structure and protected spans (chunk IDs tied to content hashes).
    
- Output `stage2/chunks.json`.
    

10. **LLM Edge: Produce Structured Edit Operations (stage2 / analyze, non-deterministic but constrained)**
    

- For each chunk, provide LLM with: local chunk + relevant pyramid nodes + issue lists + style constraints.
    
- LLM must output **edit ops** (not raw rewrite): `replace/insert/delete/flag_only`, each with:
    
    - `RVW-####` ID,
        
    - `before_hash`,
        
    - `target` (chunk id + offsets),
        
    - `needs_attention` boolean,
        
    - optional GitHub alert block text starting with `REVIEWER:` (mandatory for attention items; mandatory for deletions).
        
- Output `stage2/ops/ops_<chunk_id>.json`.
    

11. **Kernel Apply: Validate + Apply Ops + Generate Forward/Inverse Patches (stage3 / report/apply)**
    

- Deterministic kernel validates:
    
    - hashes match,
        
    - protected spans untouched,
        
    - **no silent deletion** (delete requires reviewer note),
        
    - alert formatting contract (`REVIEWER:` + change ID).
        
- Apply edits, emit:
    
    - `.review/patches/RVW-####.patch` + `.inverse.patch`,
        
    - `stage3/apply_log.jsonl`,
        
    - updated doc snapshot `stage3/doc.edited.md`.
        
- Update audit trail hashes.
    

12. **Inline Reviewer Notes Injection (stage3 / report/apply)**
    

- Insert GitHub alert blocks (NOTE/WARNING/CAUTION/IMPORTANT, per your policy) **right after** the edited text for any `needs_attention=true`.
    
- Ensure each alert starts with `REVIEWER:` to enable human navigation.
    
- Output final edited doc (in-place or `*.REVIEWED.md` depending on policy).
    

13. **KOAS Report Assembly + Selective Revert Interface (stage3 / report)**
    

- Generate `REVIEW_<doc>.md` from the ledger:
    
    - run metadata, totals, per-section list of change IDs, rationales,
        
    - silent minor edits summary (still reversible).
        
- Persist machine ledger `ledger.jsonl` and expose deterministic commands:
    
    - `revert(doc, RVW-0034)` applies stored inverse patch only.
        
- Output: `REVIEW_doc.md`, `.review/ledger.jsonl`, and revert-ready patches.