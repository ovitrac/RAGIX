## Baseline proposal — Centralized, auditable activity logging for KOAS (Docs)

### Context

For **KOAS Docs**, we have already implemented systematic logging, primarily motivated by **sovereignty constraints**:

- traceability of LLM usage,
- local execution,
- replayability without external dependencies.

This is the **right direction**, but the current logging remains **function-scoped and sovereignty-oriented**.
It does **not yet provide a unified view of activities across kernels, orchestration steps, and actors**.

As the number of kernels grows and orchestration becomes more complex (including possible external LLM orchestrators), this becomes a limitation.

------

## Problem statement

Today, we lack a **central, auditable activity layer** that can answer questions such as:

- *Which kernels ran, in which order, and why?*
- *Which data was processed, at which abstraction level (chunk, doc, cluster)?*
- *Which LLM calls were triggered, reused from cache, or replayed offline?*
- *Who (or what) initiated the workflow: internal process, operator, external orchestrator?*

Current logs are:

- technically useful,
- sovereignty-compliant,
- but **not sufficient for activity tracking, governance, or security review**.

------

## Core idea

Introduce a **centralized, auditable activity logging system** for KOAS, spanning:

- all document-related kernels,
- the LLM call boundary,
- and the orchestration layer (including LLM-driven orchestration).

This system is **orthogonal** to:

- existing LLM cache mechanisms,
- file summary caches,
- kernel-specific debug logs.

It is a **governance and security layer**, not a performance feature.

------

## Key design principles

### 1. Activity logging ≠ content logging

The system must log **what happened**, not **what was processed**.

Tracked elements include:

- kernel name, version, stage,
- execution start/end,
- input/output *identifiers* (hashes, node IDs),
- decisions taken (e.g. cache hit / miss, branch selected),
- orchestration commands.

It must **never log raw document content**.

------

### 2. Kernel-agnostic, orchestration-aware

All kernels should emit events into the same activity stream, with a minimal common schema.

The **LLM orchestrator** (human, Claude, internal agent) must also be a first-class actor:

- initiating workflows,
- triggering kernel execution,
- requesting summaries or comparisons.

This allows end-to-end reconstruction of *who did what*.

------

### 3. Sovereignty-first, but activity-oriented

The goal is no longer only “prove we stayed local”, but also:

- ensure **accountability**,
- support **audits**,
- enable **post-mortem analysis**,
- and detect misuse or abnormal behavior.

Sovereignty remains a constraint, but activity traceability becomes the primary objective.

------

## Access control and key mechanism (critical)

If the system is sovereign, **access to activity logs must be strictly controlled**, especially when orchestration comes from outside.

### Proposed baseline rules

- Activity logs are stored locally and append-only.
- Logs are **never directly accessible** without explicit authorization.
- Access is gated by a **key-based mechanism**, with:
  - scoped permissions (read / replay / audit),
  - actor identity (internal service, operator, external orchestrator),
  - time or session constraints.

### External orchestrator access

An external LLM orchestrator:

- may trigger workflows,
- may receive *high-level status and metrics*,
- but **must not have unrestricted access** to activity logs.

Any access must be:

- explicit,
- logged,
- revocable.

This preserves sovereignty **and** operational safety.

------

## Expected benefits

- Clear, auditable reconstruction of KOAS activity chains
- Separation between *execution*, *content*, and *governance*
- Stronger security posture when exposing orchestration interfaces
- Foundation for compliance, certification, or third-party audits
- Easier debugging without leaking sensitive data

------

## Non-goals (at this stage)

- No real-time monitoring UI required
- No behavioral analytics or anomaly detection yet
- No correlation with user identities beyond system actors
- No replacement of existing kernel logs or LLM caches

This is a **baseline governance layer**, not a full observability stack.

------

## Summary (one paragraph)

> KOAS already logs for sovereignty; the next step is to log for **activity governance**.
> A centralized, auditable activity logging system—spanning kernels and orchestration—provides traceability, accountability, and controlled external interaction, while preserving strict sovereignty guarantees through key-based access control.



----

my recommendation for **sovereignty, auditability, low operational complexity, and “KOAS-style determinism”**.

------

## 1) Activity Event Schema

### Recommendation: **Both**

- **Write path:** flat **JSON Lines** (`.jsonl`) for every event (append-only, grep-able, resilient)
- **Export path:** an **enveloped structured format** generated on demand (for audit bundles / compliance export)

#### Why “both” is the sweet spot

- Flat JSONL is unbeatable for:
  - performance
  - robustness under crashes
  - easy local debugging (`grep`, `jq`)
- Envelopes are necessary for:
  - consistent cross-kernel correlation
  - signatures
  - downstream ingestion into an auditor tool
  - explicit schema versioning

#### Concrete model

**Store** one event per line, but **each line is an envelope** (so you still get both):

```json
{
  "v": "koas.event/1.0",
  "ts": "2026-01-30T20:14:12.231Z",
  "event_id": "01J...ULID",
  "run_id": "20260130_201200",
  "actor": {"type":"external_orchestrator","id":"claude-prod","auth":"hmac"},
  "scope": "docs.audit",
  "kernel": {"name":"doc_extract","ver":"0.6.2"},
  "stage": "representative_sentences",
  "node_ref": {"level":"doc","id":"doc:VDP:013"},
  "io": {"inputs_root":"sha256:...","outputs_root":"sha256:..."},
  "decision": {"cache":"hit","llm":"disabled"},
  "metrics": {"sentences":42,"kept":5,"score_mean":0.71},
  "refs": {"call_hash":"sha256:..."},
  "sig": "ed25519:..."
}
```

This is still JSONL, still grep-able, but now it’s **structured and versioned**.

**Key point:**
Do **not** create “two parallel logs”. Create **one canonical event stream** and one “exporter” that produces audit bundles.

------

## 2) Key Mechanism (authn/authz)

### Recommendation: **API keys + ACL file** as the baseline

This is the best fit for “sovereign / local / revocable / no infra”.

- Each client gets a long random API key.
- You map keys → identity → scopes in a local ACL file.
- You can rotate/revoke instantly (delete entry).
- You can keep everything offline.

#### Minimal ACL example

```yaml
clients:
  claude-prod:
    key_hash: "sha256:..."      # store hashed key, never plaintext
    scopes: ["docs.trigger", "docs.status"]
    rate_limit: "30/min"
    allowed_sources: ["127.0.0.1", "10.0.0.0/8"]
    expires: "2026-06-30"
```

### Why not JWT as your baseline

JWT is great when you already have:

- an issuer,
- key rotation infrastructure,
- distributed services.

But it adds moving parts and typically becomes an “IAM project”. For KOAS-docs, start simpler.

### Where HMAC fits

HMAC is excellent for **request signing** (integrity + replay protection) but weak for rich authorization unless you reinvent claims.

**Best combo (very strong, still simple):**

- API key identifies the client (authn)
- HMAC signs each request (integrity, anti-replay)
- ACL controls scopes (authz)

That gives you:

- easy revocation
- time-bound anti-replay (`timestamp + nonce`)
- no JWT complexity

------

## 3) External Orchestrator Protocol (Claude/external LLM)

### Recommendation: **Never let the external LLM authenticate directly**

This is the most important security decision.

**Claude (or any external LLM) must not hold secrets.**
Instead, use a **broker / gateway** controlled by you.

#### Pattern: External LLM → Broker → KOAS

- Claude sends an instruction to the broker (no secrets in prompt).
- The broker:
  - authenticates to KOAS using API key/HMAC
  - enforces policy (scopes, rate limits, allowed endpoints)
  - sanitizes outputs (no metadata leakage)
  - logs everything (activity event stream)

So Claude never sees:

- API keys
- internal endpoints
- file paths
- raw logs

This matches your “orchestrator sees only metrics” isolation goal.

------

## Concrete protocol options (pick one)

### Option A — Local-only orchestrator (best sovereignty)

- Claude is only used as a “reasoner” inside your controlled runtime.
- It never reaches KOAS endpoints directly.
- No external auth problem.

### Option B — External orchestrator via broker (recommended if Claude is outside)

**Broker provides**:

- `POST /jobs` (trigger workflow)
- `GET /jobs/{id}` (status + metrics only)
- `GET /jobs/{id}/report` (external-safe deliverable only)

KOAS endpoints never exposed externally.

------

## Final choices summary

1. **Event schema:** JSONL with envelope + schema version, plus an exporter for audit bundles
2. **Key mechanism:** API key + ACL baseline, optionally add HMAC signing for request integrity and anti-replay
3. **External orchestrator:** Claude must authenticate **through a broker**, not directly to KOAS