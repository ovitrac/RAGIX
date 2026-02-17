For ragix_kernels/docs/, a forced caching mechanism must be proposed to force to store all requests to LLM, this cache aims at being reused to work without LLM for debugging, traceability, etc. This caching is independent of the existing mechanism and can be combined with the current mechanism targeting file summary.

Here the ideas:

You can solve this cleanly if you treat the cache as a first-class, content-addressed “LLM call log” with explicit provenance, and you separate two problems:
	1.	replaying an LLM call deterministically without an LLM
	2.	reusing an LLM result in a similar context (approximate match)

Your “forced caching” should guarantee (1) always, and support (2) optionally.

Below are practical solutions you can implement in ragix_kernels/docs/ without breaking the existing “file summary” cache.

⸻

0) Design rule: cache the call, not the summary

You want a cache that can be used for:
	•	debugging
	•	traceability
	•	offline re-run / “LLM disabled”
	•	regression tests

So you must store, for every LLM interaction:
	•	request (model + params + messages + tools if any)
	•	response (raw text + structured parse if you do one)
	•	provenance (which kernel, which stage, which doc nodes)
	•	inputs fingerprints (stable hashes)

This becomes a call ledger.

⸻

1) Two complementary cache keys (exact + semantic)

Key A — Exact replay key (must exist)

A content-addressed hash that guarantees that “same call → same cache entry”.

Compute a hash over a canonical JSON including:
	•	model id (e.g., mistral:7b-instruct)
	•	sampling params (temperature, top_p, seed, etc.)
	•	system prompt / role prompts
	•	messages, after canonicalization
	•	tool schema if used
	•	strict context pointers (doc/chunk IDs and their hashes)

Call it:
	•	call_hash = SHA256(canonical_request_json)

✅ If call_hash matches, you can replay exactly.

⸻

Key B — Reuse key (optional, approximate)

A second key that aims to say: “this is essentially the same request”.

Examples:
	•	prompt template id + version
	•	kernel name + stage
	•	document node identity (domain/community/cluster id)
	•	and a vector hash / MinHash of the content

Call it:
	•	reuse_key = (template_id, template_version, kernel_id, stage_id, node_path, content_signature)

Where content_signature can be:
	•	simhash(text) or minhash(shingles)
	•	and/or embedding centroid id if you already compute embeddings

✅ This supports “similar context” reuse, but you should treat it as best-effort with confidence.

⸻

2) The hard part I named: pyramidal decomposition provenance

In a pyramid (chunk → doc → group → domain), it’s easy to lose traceability because summaries are “derived” objects.

The solution is to make provenance explicit and machine-checkable:

2.1 Introduce a NodeRef for every synthesis target

Every LLM call must declare what it is summarizing:

node_ref:
  level: chunk | doc | group | domain | corpus
  node_id: "doc:DOCSET:doc_013"
  parents: ["group:DOCSET:g_02", "domain:DOCSET:d_05"]
  children: ["chunk:doc_013:c_001", "chunk:doc_013:c_002", ...]

2.2 Store input digests for all children used

For each child input text, store:
	•	child_id
	•	child_hash (sha256 of canonical text)
	•	optional child_excerpt (first N chars for sanity)

Then compute:
	•	inputs_merkle_root (Merkle tree root over the ordered list)

Now the “context identity” becomes stable:
	•	a domain summary is uniquely defined by the set of children digests + prompt template + params.

This is robust even if chunk IDs change, as long as content hashes remain.

⸻

3) Canonicalization: make requests hashable and stable

To avoid “cache misses due to formatting noise”, canonicalize:
	•	normalize whitespace (\r\n→\n, trim)
	•	remove volatile metadata (timestamps, run IDs) from prompts
	•	enforce stable ordering of message fields
	•	ensure deterministic JSON serialization (sorted keys)

For pyramid stages, ensure the child ordering is deterministic:
	•	by document path
	•	then chunk index
	•	then stable id

If ordering is not stable, your hashes won’t be stable.

⸻

4) Cache storage format: append-only ledger + materialized views

4.1 Append-only call log (source of truth)

Store each call as a JSONL entry:
	•	calls.jsonl (or sharded per day/run)
	•	each entry includes call_hash, reuse_key, node_ref, inputs_merkle_root, request, response, timestamps

Append-only makes it audit-friendly.

4.2 Materialized indices (fast lookup)

Build small indices for:
	•	call_hash -> file offset
	•	reuse_key -> list(call_hash)
	•	node_id -> list(call_hash)
	•	inputs_merkle_root -> call_hash

These can be SQLite or simple key-value.

⸻

5) Forced caching mechanism: wrapper at the LLM boundary

Do not sprinkle caching everywhere. Put one wrapper:
	•	llm_call() is the only function that touches Ollama.
	•	Kernel code calls it with a CallContext.

Example context object:

call_context:
  kernel: "doc_pyramid"
  stage: "domain_summary"
  template_id: "docs/domain_summary"
  template_version: "1.3"
  node_ref: ...
  inputs_merkle_root: ...
  run: {run_id, mode, profile}

Behavior modes (UI + CLI switch)
	•	--llm-cache=write_through (default for forced caching)
	•	--llm-cache=read_only (no LLM; fail if missing exact call)
	•	--llm-cache=read_prefer (use cache if exists; else call LLM and store)
	•	--llm-cache=off (not recommended; but keep for dev)

This gives us “work without LLM” immediately.

⸻

6) Solving “reusable in similar context” safely

Reusing in a similar context is dangerous unless we track similarity and expose it.

Option 1 (safe): exact-only replay

Use only call_hash.
Pros: deterministic and traceable.
Cons: less reuse.

This is the best default for KOAS.

Option 2 (controlled reuse): similarity with threshold + provenance check

If exact missing, look up reuse_key candidates and choose best by:
	•	SimHash distance or MinHash Jaccard
	•	plus constraint: same template_id/version, kernel/stage, and same node_level

Return:
	•	cached response
	•	plus reuse_confidence
	•	plus the “closest match” details

And mark the output as reused, not generated.

Option 3 (hybrid): replay + patch

Use cached response as draft but ask the LLM only for a minimal patch.
This defeats “no LLM” mode, so it’s for production acceleration, not debugging.

⸻

7) Traceability in the report: embed cache pointers

Whenever you generate Markdown artefacts, include in metadata:

llm_trace:
  call_hash: "sha256:..."
  inputs_merkle_root: "sha256:..."
  template: "docs/domain_summary@1.3"
  model: "mistral:7b-instruct"
  cache_status: hit | miss | reused

Now any statement in the report can be traced back to the cached call.

Metadata should be removable 

⸻

8) Minimal set of solutions I recommend you implement now

If you want the highest leverage with minimal risk:
	1.	Exact replay cache (call_hash) + append-only log
	2.	NodeRef + Merkle inputs root for pyramid stages
	3.	LLM boundary wrapper with read_only mode
	4.	Metadata embedding in all generated Markdown outputs
	5.	Optional later: similarity reuse with SimHash/MinHash

That solves your “forced caching” requirement and makes pyramid traceability tractable.

⸻

9) One more subtle but important trick: “template contracts”

Version your prompt templates explicitly.

If you don’t, then “similar context” reuse becomes impossible to justify.

So every call must include:
	•	template_id
	•	template_version
	•	schema_version (expected output structure)

This is what turns the cache into a regression-test asset.



⸻

Core rule (what Claude must obey)

Markdown metadata (front-matter or any metadata blocks) must be stripped before any content is:
	•	shown in the UI,
	•	included in a user-facing report,
	•	summarized / analyzed by an LLM,
	•	used for embeddings / retrieval (unless explicitly requested).

Exception: only if a config flag or prompt explicitly says to keep metadata for a specific purpose.

⸻

What counts as “metadata” (define it precisely)

Treat as metadata and remove by default:
	1.	YAML front matter:

---
title: ...
tags: ...
llm_trace: ...
---

	2.	TOML / JSON front matter:

+++
...
+++

```json
{ ... }

3) **Any fenced “metadata” blocks you use** (if you have a convention):
```md
```metadata
...

4) **Inline provenance blocks** you embed (e.g., `llm_trace:` YAML snippets) — keep them for internal debugging, but not in user-facing outputs unless requested.

---

## Required behavior (pipeline contract)

### A) Two-channel representation
Claude should implement (or respect) this conceptual split:

- `content_clean`: Markdown content with metadata removed (default for users, LLM input, embedding)
- `content_meta`: extracted metadata object kept internally (for traceability, caching, diagnostics)

### B) Non-leak guarantee
Claude must ensure:
- no metadata keys/values appear in summaries,
- no metadata is quoted,
- no metadata is embedded in “report narratives”.

If metadata is needed for traceability, it must be referenced **indirectly**, e.g.:
- “Trace available internally”  
or shown only in a dedicated “Internal Appendix (not for client)” if explicitly requested.

---

## How to phrase it to Claude Code (copy/paste)

Use something like this in your prompt / CLAUDE.md:

> **Markdown metadata handling (STRICT):**  
> All Markdown may contain metadata blocks (YAML/TOML/JSON front matter or fenced `metadata` blocks). These blocks are **internal control-plane** and must be **removed before any user-facing output** (UI display, PDF/MD reports, summaries, analysis, embeddings, retrieval snippets).  
>  
> Implement a preprocessing step that produces:  
> 1) `content_clean` (metadata stripped) — **the only text allowed** for summarization, analysis, retrieval, and report generation.  
> 2) `metadata` (parsed object) — retained **only** for internal traceability/logging.  
>  
> **Never leak metadata** into any report narrative, summary, or user-visible view, unless explicitly instructed for a specific deliverable. If unsure, default to stripping.

---

## Implementation hint (optional but helpful)
Tell Claude it can do this deterministically:

- Detect front matter only when it starts at the very beginning of the file.
- Parse the first block delimited by `--- ... ---` (YAML) or `+++ ... +++` (TOML).
- Remove it and keep the parsed dict.
- Also strip any fenced block tagged `metadata`.

And add a unit test rule:
- “A document summary must not contain any of the metadata keys.”

---

## UI and report-level enforcement (defense in depth)

Even if preprocessing exists, enforce again at output time:

- The report builder should only accept `content_clean`.
- The UI preview should render `content_clean`.
- The embedding pipeline should ingest `content_clean`.

This makes leakage nearly impossible.

---

## A simple “red flag” audit check (cheap and effective)
Maintain a list of metadata keys that must never appear in outputs, e.g.:
- `llm_trace`
- `call_hash`
- `inputs_merkle_root`
- `run_id`
- `endpoint`
- `model`

Then automatically scan generated reports for these tokens and fail the build if found.

---

All cached data should be located in .RAG/ 
