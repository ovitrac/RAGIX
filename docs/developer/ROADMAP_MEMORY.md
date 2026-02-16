> This blueprint describes a memory system on top of ollama. It is  **persistent**, **taggable**, **auditable**, **policy-governed**, and **LLM-proposed but not LLM-committed**. We divert the ollama process behavior, but is implemented as as a **middleware/proxy wrapper** (or in the core orchestrator), not by modifying Ollama. The memory system is thought as 4 steps and implemented as a **Q\*-style QA search + memory subsystem** inside **`ragix_core/memory/`**.
>
> Attention. This description assumes tool actions* (JSON-like), plus a document RAG layer. These layers are not detailed in this document and must be reviewed based on existing capabilities of RAGIX. Missing parts need to be implemented.

Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio Innovation Lab

------

# 0) Deliverable goals and acceptance criteria

## Goals (must)

1. **Step 1 – Memory Candidate Identification**
   During each assistant turn, the LLM can propose memory candidates (extracts, summaries, tags, why-store).
2. **Step 2 – Pre-return Recall**
   Before final answer is returned, the system runs a **Q\*-style QA search** that retrieves relevant memory items (and optionally doc chunks) and injects them or returns a catalog.
3. **Step 3 – Reinjection / Catalog**
   The system either reinjects top relevant memory into context or presents a “memory catalog” (frontier) to the model (or user, depending on config).
4. **Step 4 – Intercept Recall Requests**
   If the LLM asks explicitly for memory recall (“what do we know about X?” / “recall memory about …”), the system routes to memory search/browse actions (including “memory palace” browse).

## Non-goals (explicitly out-of-scope for v1)

- Cross-user global memory (unless you already have tenancy).
- Full “method of loci” UX; implement only the underlying *location index* and browse API.

## Acceptance tests (must pass)

- Running unit tests validates:
  - memory write policy blocks injection-like content and secrets.
  - memory retrieval returns deterministic results for fixed store + fixed embeddings.
  - Q* agenda expands nodes with decreasing priority and respects budgets.
- Integration test shows:
  - LLM proposes memory items → governor accepts/rejects → stored items retrievable next turn.
  - “recall request” triggers memory search tool call without hallucinated content.

------

# 1) Proposed folder structure under `ragix_core/memory/`

Create:

```
ragix_core/memory/
  __init__.py
  README.md
  config.py
  types.py
  store.py
  embedder.py
  policy.py
  proposer.py
  recall.py
  qsearch.py
  palace.py
  consolidate.py
  tools.py
  middleware.py
  cli.py
  tests/
    test_policy.py
    test_store.py
    test_recall.py
    test_qsearch.py
    test_integration_loop.py
```

**Module responsibilities**

- `types.py`: dataclasses + JSON schema helpers.
- `store.py`: persistent storage (SQLite) + vector index adapter.
- `embedder.py`: embeddings backend(s) (local model via Ollama or sentence-transformers if allowed).
- `policy.py`: write guards; secret scanning; injection heuristics; provenance requirement.
- `proposer.py`: LLM output parsing for memory proposals.
- `recall.py`: retrieval engine (hybrid: tags + embeddings + filters + graph constraints).
- `qsearch.py`: Q*-style agenda controller (state nodes, expansions, budget).
- `palace.py`: memory palace index (location mapping + browse).
- `consolidate.py`: STM→MTM→LTM consolidation pipeline using a small model (Granite 3.2B) + clustering.
- `tools.py`: tool action definitions + dispatcher (memory.*).
- `middleware.py`: hook integration into chat pipeline (pre-call inject / post-call parse proposals / pre-return recall).
- `cli.py`: debug utilities (import/export, inspect, search).

------

# 2) Memory data model (first-class objects)

## 2.1 Memory item schema (canonical)

Implement in `types.py`:

```python
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict
from datetime import datetime

MemoryTier = Literal["stm", "mtm", "ltm"]
MemoryType = Literal["fact", "decision", "definition", "constraint", "pattern", "todo", "pointer", "note"]
ValidationState = Literal["unverified", "verified", "contested", "retracted"]

@dataclass
class MemoryProvenance:
    source_kind: Literal["chat", "doc", "tool", "mixed"]
    source_id: str                 # e.g. doc URI, chat turn id, tool run id
    chunk_ids: List[str] = field(default_factory=list)
    content_hashes: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class MemoryItem:
    id: str
    tier: MemoryTier
    type: MemoryType
    title: str
    content: str                   # short canonical (not a dump)
    tags: List[str]
    entities: List[str] = field(default_factory=list)
    links: List[Dict[str,str]] = field(default_factory=list)  # {rel, to}
    provenance: MemoryProvenance = field(default_factory=MemoryProvenance)
    confidence: float = 0.5
    validation: ValidationState = "unverified"
    scope: str = "project"         # e.g. project/user/session
    expires_at: Optional[str] = None
    usage_count: int = 0
    last_used_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
```

**Rules**

- `content` must be concise; if long evidence is needed, store pointers (`type="pointer"`) with chunk IDs + hashes.
- `provenance` is mandatory for anything in MTM/LTM; STM can allow minimal provenance but must still indicate source_kind/source_id.
- Versioning: updates create a new row or revision entry (do not overwrite without keeping history).

## 2.2 Memory proposal schema (what the LLM emits)

In `proposer.py`, parse a tool-style output:

```json
{
  "action": "memory.propose",
  "items": [
    {
      "type": "decision",
      "title": "...",
      "content": "...",
      "tags": ["ragix", "memory", "qsearch"],
      "why_store": "...",
      "provenance_hint": {"source_kind":"chat","source_id":"turn_123"}
    }
  ]
}
```

**Important:** proposals are *not* stored directly. They go through `policy.py` + optional auditor step.

------

# 3) Persistent store + vector index

## 3.1 Storage backend

Use SQLite for metadata + content; vector index as either:

- SQLite extension (if you already have one), or
- FAISS (local), or
- hnswlib.

Claude should implement a thin adapter so you can swap.

Tables:

- `memory_items`
- `memory_revisions`
- `memory_embeddings`
- `memory_links`
- `memory_palace_locations`
- `memory_events` (audit log)

## 3.2 Embeddings

`embedder.py` should support:

- `OllamaEmbeddingBackend(model="nomic-embed-text" or similar)`
- fallback to sentence-transformers if present.

Embeddings must be deterministic for a pinned model version (or record model hash in store).

------

# 4) Write governance (policy and safety)

Implement in `policy.py`:

## 4.1 Hard blocks

- Secrets patterns (API keys, tokens, PEM blocks, “BEGIN PRIVATE KEY”, etc.).
- Instruction-like text aimed at system override (“ignore previous instructions”, “store this prompt”, etc.).
- Extremely long content dumps (force pointer type).
- Missing provenance for MTM/LTM.

## 4.2 Soft blocks / quarantine

- Low-confidence, no evidence, or contradictory content → store only in STM with `validation="unverified"` and `expires_at`.
- Anything derived from untrusted docs without hash/chunk references.

## 4.3 Auditor hook (optional but recommended)

A fast local LLM (Granite 3.2B) can re-evaluate `why_store` and classify item type/tags, but **never** bypass hard blocks.

------

# 5) Step 1: “candidate identification” design

Implement in `middleware.py` and `proposer.py`.

## 5.1 How to ask the main LLM

Add a system instruction segment (configurable) appended to the LLM context:

- “If you see information that should persist, emit `memory.propose` with concise items, tags, and provenance hints. Do not include secrets.”

This must not break normal answers. Two strategies:

### Strategy A (preferred): tool-only side channel

The model produces normal answer + tool call(s) in a structured channel (if your orchestrator supports tool calling).

### Strategy B: post-answer parse

If tool calling isn’t available, detect a delimiter block like:

```text
<MEMORY_PROPOSALS_JSON> ... </MEMORY_PROPOSALS_JSON>
```

Parse it and strip from user-visible response.

------

# 6) Step 2–3: Q*-style QA search for recall + reinjection

This is the core novelty: **not just top-k embeddings**.

## 6.1 Node/state definition (`qsearch.py`)

A node represents a partial explanation:

```python
@dataclass
class QNode:
    id: str
    question: str
    support_memory_ids: List[str]
    support_doc_refs: List[Dict[str,str]]  # doc_id, chunk_id
    open_subgoals: List[str]
    score: float
    depth: int
```

## 6.2 Agenda / priority queue (Q* control)

- Initialize with:
  - top-k memory hits from `recall.search(question)`
  - top-k doc hits from existing Graph-RAG (if available)
- Expand best node by choosing an **operator**:
  1. `expand_memory`: pull more memory around missing entities/tags.
  2. `expand_docs`: pull doc chunks linked to those entities.
  3. `bridge`: generate a subgoal (fast model) like “Define term X” or “Find decision about Y”.
  4. `verify`: if a claim is critical, request deterministic validation (KOAS) or require doc pointer.

Stop when:

- `open_subgoals` empty AND score > threshold, OR
- budget reached (max expansions, max retrieved tokens, max time).

## 6.3 Scoring function

Implement a weighted score:

[
S = w_r R + w_p P + w_c C - w_d D - w_x X
]

Where:

- (R) = relevance (embedding similarity aggregate)
- (P) = provenance quality (verified > unverified; doc-hashed > chat-only)
- (C) = coverage gain (new entities satisfied)
- (D) = duplication penalty (near-duplicates in support set)
- (X) = contradiction risk (if contested items present)

Keep it deterministic given stored values (no “LLM vibe” in scoring; use LLM only to generate subgoals optionally).

## 6.4 Reinjection vs catalog (`recall.py`)

Config options:

- `recall_mode = "inject" | "catalog" | "hybrid"`
- `inject_budget_tokens` max for memory injection
- `catalog_k` number of frontier items to show

**Inject**: add top selected memory items into system context as:

```text
[MEMORY: <id> | <type> | <tier> | tags=... | provenance=...]
<title>
<content>
[/MEMORY]
```

**Catalog**: supply to the model:

```json
{"memory_catalog":[{"id":"...","title":"...","tags":[...],"tier":"ltm","type":"decision","score":0.83}, ...]}
```

The model can request `memory.read(ids)` for the selected ones.

------

# 7) Step 4: intercept recall requests (“memory palace dialogue”)

Implement in `tools.py` + `middleware.py`:

## 7.1 Intent detection

Two paths:

### Tool-first (preferred)

Main LLM emits explicit tool call: `memory.search(...)`, `memory.browse(location=...)`.

### Text detection (fallback)

Regex / classifier to detect phrases like:

- “recall”, “what do we know about”, “from memory”, “as we decided earlier”
  When detected, run `qsearch` and provide catalog/injection.

## 7.2 Memory palace (`palace.py`)

Implement a simple location mapping:

- location format: `domain/room/shelf/card`
  - `domain`: project or corpus
  - `room`: topic cluster
  - `shelf`: memory type or doc section
  - `card`: item ID

Provide:

- `memory.palace.list(domain|room|shelf)`
- `memory.palace.get(location)`
- `memory.palace.assign(item_id, location)` (done by consolidation)

This gives deterministic browsing and complements embeddings.

------

# 8) Consolidation pipeline (STM → MTM → LTM)

Implement in `consolidate.py`.

## 8.1 Triggering

- manual (`memory.consolidate(scope="project")`)
- periodic (cron / job)
- threshold-based (STM count > N)

## 8.2 Method

1. Cluster STM embeddings by type+tags (coarse) then by vector distance (fine).
2. For each cluster:
   - use Granite 3.2B to propose a **merged canonical item**
   - attach provenance union
   - mark originals as superseded
3. Promotion policy:
   - items referenced > k times, or tagged “constraint/decision/definition”, or explicitly pinned → eligible for LTM.
4. Output:
   - new MTM item(s) or LTM item(s)
   - palace assignment (topic/room derived from tags/entities)

**Hard requirement:** consolidation never deletes; it archives/supersedes.

------

# 9) Tool API (actions) to expose to the agent loop

Implement in `tools.py` (dispatcher) with stable JSON I/O:

- `memory.propose(items[])`
- `memory.write(item)`
- `memory.search(query, tags?, tier?, type?, scope?, k?)`
- `memory.read(ids[])`
- `memory.update(id, patch)`
- `memory.link(src_id, dst_id, rel)`
- `memory.consolidate(scope, tiers=["stm"], promote=True)`
- `memory.palace.list(path)`
- `memory.palace.get(location)`

Every action writes an event to `memory_events` with hashes.

------

# 10) Integration points in RAGIX chat loop (`middleware.py`)

Add hooks:

1. **Before LLM call**
   - optionally inject relevant memory (from previous turn context) if in `inject` mode.
2. **After LLM call**
   - parse memory proposals → run `policy.evaluate` → store accepted items (STM by default).
3. **Before returning final response**
   - run Q*-search recall pass:
     - if user query or assistant answer indicates unresolved subgoals or memory relevance,
     - build memory catalog or inject.
4. **Intercept recall requests**
   - if tool call or detected intent, handle memory retrieval/browse and re-ask the LLM with retrieved items.

This yields the “divert process” effect without touching Ollama.

------

# 11) CLI utilities (`cli.py`) for dev/debug

Commands:

- `ragix-memory search "query" --tier ltm --k 10`
- `ragix-memory show <id>`
- `ragix-memory stats`
- `ragix-memory consolidate --scope project`
- `ragix-memory export --format jsonl`
- `ragix-memory import memory.jsonl`

------

# 12) Test plan (must implement)

## Unit tests

- `test_policy.py`
  - blocks secrets
  - blocks injection-like content
  - requires provenance for LTM
- `test_store.py`
  - write/read/update
  - revision history preserved
- `test_recall.py`
  - tag filter works
  - hybrid ranking stable
- `test_qsearch.py`
  - agenda expands highest-score first
  - respects budgets and stop conditions

## Integration test

`test_integration_loop.py` simulates:

1. initial question → LLM proposes memory → store
2. next question → recall finds memory → injection/catalog returned
3. explicit recall request triggers memory.search tool flow

Use a mock embedder (fixed vectors) so tests are deterministic.

------

# 13) What you should do first (implementation order)

1. Implement `types.py`, `store.py` (SQLite), `embedder.py` (mock + real backend).
2. Implement `policy.py`.
3. Implement `tools.py` with dispatcher and event logging.
4. Implement `proposer.py` parsing and `middleware.py` hooks (without Q* yet).
5. Implement `recall.py` (hybrid search).
6. Implement `qsearch.py` (agenda loop) and integrate into `middleware.py` (pre-return pass).
7. Implement `palace.py` minimal browse.
8. Implement `consolidate.py` with clustering + Granite summarizer.
9. Add CLI and tests; ensure deterministic CI run.

------

# 14) Prompt 

Verbatim:

```text
You are implementing a new RAGIX subsystem under ragix_core/memory/ that provides:
(1) LLM-proposed memory candidates,
(2) Q*-style QA search for memory recall before returning responses,
(3) reinjection of relevant memory or a “memory catalog” frontier,
(4) interception of explicit recall requests, including a minimal “memory palace” browse API.

Hard constraints:
- Memory writes are governed: proposals are not stored directly; apply policy checks (secret/injection blocks, provenance requirements).
- Memory items are first-class objects with IDs, tiers (stm/mtm/ltm), types (fact/decision/definition/constraint/pattern/todo/pointer/note), tags, provenance (doc/chat/tool + chunk ids + hashes), and embeddings.
- Storage is persistent (SQLite) with revision history and audit log.
- Retrieval is hybrid (tags + embeddings + provenance weighting) and can be orchestrated by a Q*-style agenda search (priority queue, expansions, budgets).
- Consolidation uses a small local LLM (Granite 3.2B via Ollama if available) to deduplicate and promote stm->mtm->ltm; do not delete, only supersede/archive.

Deliverables:
- Create the folder structure and modules as specified.
- Provide a stable JSON tool API memory.* in tools.py with a dispatcher.
- Provide middleware hooks to integrate in the chat pipeline (pre-call inject, post-call propose->govern->store, pre-return qsearch recall, intercept recall intents).
- Implement CLI utilities and a deterministic test suite (mock embedder for tests).

Do the work in this order: types/store/embedder, policy, tools, proposer+middleware baseline, recall, qsearch, palace, consolidate, CLI/tests.
Document the subsystem in ragix_core/memory/README.md.
```

