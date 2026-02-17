Below is a **full, implementation-ready roadmap** for **KOAS Summary**, designed to:

* **consume the existing `memory.*` tool API as the core persistent store** (no parallel memory store),
* implement a **flexible 3-stage KOAS pipeline** (collect → analyze → report), consistent with KOAS v2.0 conventions ,
* expose **autodiscoverable capabilities** so even a “public very fat LLM” can ask **RAGIX KOAS**: “summarize this folder” **without direct access** to the folder (the kernels read locally; the external model only sees structured summaries) ,
* support **secrecy levels** via deterministic redaction + policy gating (your memory governance already has hard blocks + provenance, we extend it to Summary outputs) ,
* improve the exact pain points you measured in the CORP-ENERGY RIE benchmark (domain truncation by injection cap, consultative style, weak proof of memory usage) .

---

# A. What “Summary” should do vs what “KOAS Summary” will do

## A1) What “Summary” should improve (system-level behavior)

These are outcomes you want regardless of whether you use a graph:

1. **Coverage & balance**: avoid “one document dominates” summaries (your RHEL domination failure mode). 
2. **Rule-first outputs**: produce structured, non-consultative summaries (template-driven; bullet rules, constraints, decisions). 
3. **Forced memory usage**: summary must cite memory IDs deterministically (measurable).
4. **Multi-file digestion**: consume huge corpora (folder, multiple files, or stdin pipe) with stable chunking and traceability.
5. **Secrecy control**: generate “clean” outputs at multiple secrecy levels (redaction + pointer control + safe quoting).
6. **Autodiscoverable execution**: expose a capabilities manifest so an external LLM can call “summary” without seeing the corpus.

## A2) What “KOAS Summary” will do (kernel-level responsibilities)

KOAS Summary is the deterministic + auditable execution system implementing those outcomes:

* **Stage 1**: deterministically *collect and chunk* a corpus (folder / file list / stdin), generate pointers and metadata.
* **Stage 2**: build *memory + (optional) graph index*:

  * propose/store memory items (via `memory.propose` → governance → `memory.write`) 
  * consolidate STM→MTM/LTM (granite 3.2b) and assign palace locations where helpful 
  * build a **RAG graph** that links docs/sections/rules/pointers and enables coverage budgeting.
* **Stage 3**: produce the final report:

  * strict templates
  * per-domain/cluster budgeted retrieval (prevents truncation)
  * memory ID citations + deterministic citation verification
  * secrecy-aware redaction.

---

# B. KOAS Summary: the 3-stage pipeline (full design)

This follows KOAS’s three-stage pipeline model and kernel output contract  .

## Stage 1 — Data Collection (deterministic, corpus ingestion)

### Stage 1 goals

* Turn *any corpus* into:

  * a stable **file registry**
  * a stable **chunk registry**
  * stable **pointer objects** (hashes + offsets)
* No LLM needed.

### Stage 1 inputs (supported)

1. `--folder <path>` (recursive)
2. `--files <list>`
3. `--stdin` (pipe) → spooled into a workspace file for audit

### Stage 1 outputs

* `stage1/files.json` (file list, mime, bytes, hashes)
* `stage1/chunks.jsonl` (chunk plan + offsets)
* `stage1/pointers.jsonl` (pointer records for later memory provenance)
* `stage1/manifest.json` (corpus_id, secrecy, run metadata, hashes)

### Stage 1 kernels (proposed family: `summary/*`)

1. `summary.collect`

   * enumerate files, determine types, compute hashes
2. `summary.spool_stdin`

   * read stdin → write `stdin.txt` (or `.md`) with hash
3. `summary.extract_text`

   * deterministic text extraction per type (pdf → text, docx → text, md/txt passthrough)
4. `summary.chunk_plan`

   * hybrid chunking: headings if possible, else windowing with overlap
5. `summary.pointer_index`

   * create pointer entries `(doc_id, chunk_id, offsets, hashes)` for provenance

**Why this matters for your benchmark:** it creates the “doc skill/agent to read large docs” as deterministic kernels, not “LLM pretending it read the file”. This is aligned with Unix-RAG / tool-first philosophy .

---

## Stage 2 — Analysis (hybrid: memory extraction + graph + consolidation)

### Stage 2 goals

* Extract **rules/definitions/decisions/patterns** into Memory (STM first).
* Normalize them (version-aware, domain-aware).
* Consolidate, deduplicate, promote.
* Build **graph index** for coverage + bridging.

### Stage 2 inputs

* `stage1/*`
* configuration:

  * `scope` (e.g., `corp_energy-rie`)
  * `secrecy_level` (see section E)
  * `embedder_backend` (ollama or sentence-transformers fallback) 
  * consolidation thresholds

### Stage 2 outputs

* `stage2/memory_export.jsonl` (optional export for audit)
* `stage2/graph.jsonl` (edges) or `stage2/graph.sqlite`
* `stage2/coverage.json` (per-domain coverage metrics)
* `stage2/consolidation.json` (merge map, superseded_by links)

### Stage 2 kernels (summary family)

1. `summary.propose_memory` *(LLM-assisted, but tool-governed)*

   * For each chunk: ask an LLM worker (granite 3.2b recommended) to produce MemoryProposals.
   * Write via `memory.propose()` then store accepted via `memory.write()` under STM. 
   * Enforce “version-aware title prefix” at code level if needed (fixes the “Oracle 19c exists only in doc title” failure) 

2. `summary.memory_tagging` *(deterministic)*

   * auto-tag `doc_source`, `technology`, `version`, `section`
   * add canonical `rie-{docname}` tags (your own tuning reco) 

3. `summary.build_graph` *(deterministic)*
   Build nodes + edges:

   * Nodes: Doc, Chunk, MemoryItem(rule), MemoryItem(pointer), Cluster
   * Edges: contains, extracted-from, supported-by, superseded_by, belongs-to-domain
     This leverages fields you already added (`superseded_by`, `archived`) .

4. `summary.coverage_metrics` *(deterministic)*

   * compute coverage by domain/technology/version/section:

     * #chunks
     * #rules
     * entropy/diversity
     * “dominance ratio” (detect RHEL-style domination) 

5. `summary.consolidate_memory` *(hybrid)*

   * call `memory.consolidate(scope=..., tiers=["stm"], promote=True)` with:

     * **context-fraction trigger** (your “15%” rule moved here as a deterministic trigger) 
   * use Granite 3.2b as merge proposer (already in your blueprint) 
   * ensure consolidation never deletes; only supersedes/archives 

6. `summary.graph_bridge` *(deterministic-first; optional LLM later)*

   * implement the missing “bridge” operator without full KOAS verify yet:

     * shortest-path between clusters via shared tags/entities/sections
     * return “bridge candidates” for Stage 3 retrieval

**Why Stage 2 matters:** it makes the graph useful for **budgeted retrieval** and it makes consolidation measurable (unlike the mock-embedder situation that produced 0 merges) .

---

## Stage 3 — Reporting (structured summary, forced memory citations, secrecy)

### Stage 3 goals

* Produce a **clean summary** with:

  * per-domain balanced coverage
  * strict template (rules-first, no consultative style)
  * memory ID citations (mandatory)
  * optionally: annexes (tables, checklists, compliance matrix)
* Provide an output that a public LLM can consume **without corpus exposure**.

### Stage 3 outputs

* `stage3/summary.md` (main deliverable)
* `stage3/summary.json` (structured)
* `stage3/citation_map.json` (every bullet → memory IDs → pointer IDs)
* `stage3/redaction_report.json` (what was removed/blurred due to secrecy)

### Stage 3 kernels

1. `summary.plan_outline` *(deterministic + optional LLM)*

   * outline derived from coverage metrics:

     * “must include these technologies / domains”
     * “must include minimum N rules per domain”
   * this is what fixes the “RHEL dominates injection” issue before tokens are spent 

2. `summary.retrieve_budgeted` *(deterministic — the key kernel)*

   * retrieve memory items using:

     * graph partitions (domain nodes) + coverage metrics
     * per-domain quota
     * cluster-level dedup
   * output is a **support set**: `(MemoryItem IDs + short catalog + pointers optionally)`
     This directly implements the benchmark fix you already identified (per-domain injection) .

3. `summary.generate_sections` *(LLM: gpt-oss-safeguard 120b)*

   * for each outline section:

     * provide budgeted support set
     * require each bullet ends with `[MID: ...]`
     * forbid “I can do X” consultative framing (hard prompt rule) 

4. `summary.verify_citations` *(deterministic)*

   * check:

     * every bullet has ≥1 MID
     * every MID exists
     * optional: MID has pointer provenance for secrecy level ≥S2
   * if missing: re-run retrieval for that section or flag as incomplete (but deterministic)

5. `summary.assemble_report` *(deterministic)*

   * stitch sections
   * add executive summary + coverage table + appendix list
   * embed “capabilities manifest” pointer for autodiscovery (see D)

---

# C. Autodiscoverable capabilities (so an external fat LLM can ask without corpus access)

KOAS already emphasizes “LLM-ready summaries” and modular kernels with `provides` metadata . KOAS Summary should publish a **capabilities manifest** generated deterministically.

## C1) Capability contract

Create a kernel: `summary.capabilities` (stage 3 or “meta”) that emits:

* `capability_id`: `koas.summary.v1`
* supported inputs: folder/files/stdin
* supported secrecy levels
* supported output formats: md/json
* supported “profiles”: `rules`, `digest`, `checklist`, `compliance_matrix`
* constraints: max corpus size, max run time (optional), supported mime types

This allows a public LLM to do:

> “RAGIX KOAS summarize me this folder at secrecy S2, profile=rules, output=md.”

The public LLM never sees the folder; KOAS reads it locally and returns only Stage-3 artifacts.

---

# D. Secrecy levels: “clean summary with given level of secrecy”

You already have memory write governance (secrets/injection patterns, provenance requirements, quarantine) . KOAS Summary extends this to **outputs**, with deterministic redaction and pointer control.

## D1) Define secrecy levels (example spec)

* **S0 (public)**: no file paths, no usernames, no IPs, no hostnames, no internal identifiers; only generalized statements.
* **S1 (internal-light)**: allowed: technology names + versions + generic constraints; still redact identifiers.
* **S2 (internal-standard)**: allowed: named systems/roles if they pass policy; include MIDs but not raw pointers.
* **S3 (audit)**: include MIDs + pointer IDs + chunk hashes + exact citations (still no raw secrets).
* **S4 (forensic)**: include full pointer offsets and excerpt snippets (bounded) where policy allows.

## D2) Deterministic redaction kernel

Stage 3 kernel: `summary.redact`:

* apply regex-based scrubbing:

  * secrets patterns (reuse policy patterns)
  * file system paths
  * emails, hostnames, internal IDs
* output redaction report (what replaced and why)

This is crucial if a “public fat LLM” is consuming the result: it must not receive sensitive artifacts.

---

# E. Roadmap: implementation phases, deliverables, acceptance tests

Below is a structured plan you can execute incrementally, each phase leaving the system in a working state.

---

## Phase 0 — Spec & folder skeleton (1–2 days)

**Deliverables**

* `ragix_core/koas/kernels/summary/` family skeleton
* `KOAS_SUMMARY.md` design doc: inputs/outputs/secrecy/profiles
* kernel registry entries with `provides` metadata 

**Acceptance**

* `koas list` shows `summary.*` kernels with stages and provides.

---

## Phase 1 — Stage 1: deterministic corpus ingestion (2–4 days)

**Implement kernels**

* `summary.collect`
* `summary.spool_stdin`
* `summary.extract_text`
* `summary.chunk_plan`
* `summary.pointer_index`

**Acceptance tests**

* Given a folder with mixed files:

  * stable file hashes across runs
  * stable chunk IDs
  * pointer registry matches chunk registry

**Why it’s critical**

* This is the “agent skill to read huge corpora” made deterministic.

---

## Phase 2 — Stage 2: memory ingestion via `memory.*` (3–6 days)

**Implement**

* `summary.propose_memory` (granite 3.2b)
* `summary.memory_tagging`
* `summary.consolidate_memory` (triggered by thresholds)

**Use memory tools**

* `memory.propose`, `memory.write`, `memory.consolidate` 

**Acceptance**

* For a sample corpus:

  * memory items written with provenance pointers
  * version-aware titles enforced (code-level)
  * consolidation produces MTM items and superseded_by links when embeddings are real 

---

## Phase 3 — Stage 2 graph index (Graph-RAG minimal) (3–5 days)

**Implement**

* `summary.build_graph`
* `summary.coverage_metrics`
* `summary.graph_bridge` (basic)

**Acceptance**

* graph contains edges:

  * doc→chunk, chunk→rule, rule→pointer, rule→cluster
* coverage metrics detect dominance and list missing domains.

**What it improves**

* makes it possible to do budgeted retrieval and ensure balanced coverage (direct fix for your benchmark) 

---

## Phase 4 — Stage 3 reporting with forced memory citations (4–8 days)

**Implement**

* `summary.plan_outline`
* `summary.retrieve_budgeted` (**must-have**)
* `summary.generate_sections` (gpt-oss 120b)
* `summary.verify_citations`
* `summary.assemble_report`

**Acceptance**

* every bullet has ≥1 MID
* per-domain quotas respected
* summary no longer “consultative” (template compliance)
* citation map exists

**Direct benchmark KPI**

* Stage 3 must include Oracle/WebLogic/PostgreSQL/Java/Tomcat sections even if RHEL dominates corpus. 

---

## Phase 5 — Secrecy & external consumption (2–5 days)

**Implement**

* `summary.redact`
* `summary.capabilities` (autodiscovery manifest)

**Acceptance**

* Run with `--secrecy S0`:

  * no raw paths/emails/hostnames
  * no pointers
* Run with `--secrecy S3`:

  * includes pointer IDs + hashes, still redacted secrets
* Manifest allows an external LLM to request summaries safely.

---

## Phase 6 — “Pipe-first” mode + incremental updates (optional, but very valuable) (3–7 days)

**Implement**

* `summary.incremental`:

  * detect changed files by hash
  * only re-chunk and re-propose memory on changed chunks
  * maintain corpus_id continuity
* `summary.stdin_watch` (batch)
* `summary.merge_runs` (combine multiple corpora)

**Acceptance**

* re-run on same corpus → no-op except report regeneration
* change 1 file → only affected chunks updated, memory supersedes old, graph updated

---

# F. CLI and orchestration UX (what an LLM or human calls)

Provide one high-level entrypoint:

```bash
koas summary run \
  --input folder:/path/to/corpus \
  --scope corp_energy-rie \
  --profile rules \
  --secrecy S2 \
  --memory-scope corp_energy-rie \
  --consolidator granite3.2b \
  --writer gpt-oss-120b \
  --ctx-trigger 0.15 \
  --out /path/to/workspace
```

And a shorthand for stdin:

```bash
cat huge.txt | koas summary run --input stdin --profile digest --secrecy S0
```

This explicitly matches your “digest any corpus including standard output from a pipe”.

---

# G. Non-negotiable engineering constraints (to keep KOAS philosophy intact)

These are the constraints that make KOAS “KOAS”:

1. **Determinism in kernels**: no hidden state, no network, stable hashing 
2. **LLM use only at well-defined boundaries**: propose_memory, generate_sections.
3. **Memory is tool-governed**: proposals never directly committed; policy gates apply 
4. **Audit trail everywhere**: Stage1/2/3 artifacts + logs + hashes 
5. **Budgeted retrieval**: per-domain quotas is mandatory to avoid your observed truncation failure .

---

## The bottom line

* **Summary (as a feature)** is about *balanced, rule-first, secrecy-aware large summaries forced to cite memory*.
* **KOAS Summary (as a system)** is the 3-stage deterministic pipeline that:

  * reads any corpus locally (folder/files/stdin),
  * fills/uses your `memory.*` subsystem as the only persistent store,
  * optionally builds a graph index to enforce coverage and bridging,
  * outputs a clean, redacted, citeable report consumable by an external fat LLM without corpus access.

