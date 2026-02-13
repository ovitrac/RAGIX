# KOAS Activity Logging — Centralized Audit Trail

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 1.0.0
**Date:** 2026-02-13
**RAGIX Version:** 0.66+
**KOAS Version:** 1.0

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Design Principles](#2-design-principles)
3. [Event Schema](#3-event-schema-koasevent10)
4. [Actor Model](#4-actor-model)
5. [Event Types](#5-event-types)
6. [Sovereignty Attestation](#6-sovereignty-attestation)
7. [Hash Chain Integrity](#7-hash-chain-integrity)
8. [Querying Events](#8-querying-events)
9. [Broker Integration](#9-broker-integration)
10. [Configuration](#10-configuration)
11. [API Reference](#11-api-reference)
12. [Related Documentation](#12-related-documentation)

---

## 1. Introduction

KOAS Activity Logging provides a **centralized, auditable governance layer** for all kernel and LLM operations. It answers:

- *Which kernels ran, in which order, and why?*
- *Which LLM calls were triggered, reused from cache, or replayed offline?*
- *Who (or what) initiated the workflow: internal process, operator, external orchestrator?*
- *Can we prove that all processing happened locally?*

### Activity Logging ≠ Content Logging

The system logs **what happened**, not **what was processed**.

| Logged | NOT Logged |
|--------|------------|
| Kernel name, version, stage | Document content |
| Execution timestamps | Raw text excerpts |
| Input/output hashes | File contents |
| Decisions (cache hit/miss) | Prompt/response text |
| Actor identity | Personal data |
| Metrics (counts, scores) | Internal paths (in external view) |

### Single Canonical Event Stream

All activity flows into one append-only file:

```
.KOAS/activity/events.jsonl    <- canonical source
.KOAS/activity/exports/        <- audit bundles (generated on demand)
```

**Source:** `ragix_kernels/activity.py` (731 lines)

---

## 2. Design Principles

| Principle | Mechanism |
|-----------|-----------|
| **Prove local execution** | `sovereignty.local_only: true` in every event |
| **Track who did what** | Actor field with type + auth method |
| **Enable audits** | Structured events + append-only stream |
| **Detect misuse** | Centralized stream + rate limiting (broker) |
| **No content leakage** | Only hashes, metrics, and decisions recorded |
| **Forward compatibility** | Schema versioned as `koas.event/1.0` |

---

## 3. Event Schema (`koas.event/1.0`)

Every event is a self-contained JSON envelope:

```json
{
  "v": "koas.event/1.0",
  "ts": "2026-01-30T20:50:11.142+00:00",
  "event_id": "a7b3c4d5-e6f7-8901-2345-67890abcdef0",
  "run_id": "run_20260130_215011_348bc4",
  "actor": {
    "type": "system",
    "id": "koas",
    "auth": "none"
  },
  "scope": "docs.kernel",
  "phase": "end",
  "kernel": {
    "name": "doc_metadata",
    "version": "1.0.0",
    "stage": 1
  },
  "decision": {
    "success": true,
    "cache_hit": false
  },
  "metrics": {
    "duration_ms": 42,
    "item_count": 79
  },
  "sovereignty": {
    "local_only": true
  }
}
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `v` | string | Yes | Schema version (`koas.event/1.0`) |
| `ts` | ISO 8601 | Yes | UTC timestamp with milliseconds |
| `event_id` | UUID v4 | Yes | Unique event identifier |
| `run_id` | string | Yes | Groups events within a single pipeline run |
| `actor` | object | Yes | Who/what initiated the action |
| `scope` | string | Yes | Event category (e.g., `docs.kernel`, `docs.llm`) |
| `phase` | string | No | Phase within scope (`start`, `end`, `cache_hit`) |
| `kernel` | object | No | Kernel name, version, stage (for kernel events) |
| `node_ref` | object | No | Document hierarchy reference (level, id) |
| `io` | object | No | Input/output hashes for provenance |
| `decision` | object | No | Decision metadata (cache hit/miss, branch) |
| `metrics` | object | No | Quantitative: duration_ms, item_count, scores |
| `refs` | object | No | References to related entities |
| `sovereignty` | object | No | Sovereignty attestation |

---

## 4. Actor Model

Every event carries an `actor` identifying who initiated the action.

### Actor Types (`ActorType`)

| Type | Description | Auth |
|------|-------------|------|
| `system` | Internal KOAS process (default) | `none` |
| `operator` | Human operator via CLI/web | `api_key` or `none` |
| `external_orchestrator` | External LLM (Claude, GPT-4) via broker | `api_key` or `hmac` |
| `auditor` | Read-only audit access | `api_key` |

### Authentication Methods (`AuthMethod`)

| Method | Description |
|--------|-------------|
| `none` | Direct CLI/MCP access (relaxed mode) |
| `api_key` | SHA256-hashed API key with ACL scopes |
| `hmac` | HMAC-signed requests (production) |

---

## 5. Event Types

### Kernel Events (`scope: docs.kernel` / `audit.kernel` / etc.)

| Phase | When | Key Metrics |
|-------|------|-------------|
| `start` | Kernel begins execution | kernel name, version, stage |
| `end` | Kernel completes | `duration_ms`, `item_count`, `success` |

### LLM Events (`scope: docs.llm`)

| Phase | When | Key Metrics |
|-------|------|-------------|
| `call` | LLM invocation | model, prompt_hash, temperature |
| `cache_hit` | Response served from cache | model, call_hash |
| `end` | LLM response received | `duration_ms`, `eval_count` |

### Orchestration Events (`scope: orchestration`)

| Phase | When | Key Metrics |
|-------|------|-------------|
| `started` | Pipeline run begins | run_id, config summary |
| `completed` | Pipeline run finishes | total_duration_ms, kernel_count |

### Authentication Events (`scope: system.auth`)

| Phase | When | Key Metrics |
|-------|------|-------------|
| `success` | Valid key + allowed scope | client_id, scope |
| `denied` | Invalid key or scope violation | client_id, reason |

---

## 6. Sovereignty Attestation

Every event includes a `sovereignty` field:

```json
{
  "sovereignty": {
    "local_only": true,
    "endpoint": "http://localhost:11434"
  }
}
```

| Field | Description |
|-------|-------------|
| `local_only` | `true` if all processing was local (Ollama) |
| `endpoint` | LLM endpoint used (for verification) |

For LLM events, the sovereignty field is populated by `SovereigntyInfo` from the cache layer, which captures hostname, user, endpoint, and timestamp.

**Data Containment:** Under correct configuration (`enforce_local: true`), no document content traverses the sovereign perimeter. The activity stream itself contains only hashes, metrics, and decisions — never raw content.

---

## 7. Hash Chain Integrity

The orchestrator maintains a SHA256 chain across kernel executions (`orchestrator.py:366`):

```python
entry["chain_hash"] = hashlib.sha256(chain_content.encode()).hexdigest()[:16]
```

Each audit log entry's hash incorporates the previous entry, creating a tamper-evident chain. If any entry is modified or removed, the chain breaks.

Additionally, the Merkle tree module (`merkle.py`) computes `inputs_merkle_root` for pyramidal provenance — ensuring that document-level summaries can be traced back to their source chunks via cryptographic hashes.

---

## 8. Querying Events

### Using `jq` (Command Line)

```bash
# Count events by scope
cat .KOAS/activity/events.jsonl | \
  jq -s 'group_by(.scope) | map({scope: .[0].scope, count: length})'

# List all kernel executions with duration
cat .KOAS/activity/events.jsonl | \
  jq 'select(.scope | endswith(".kernel")) | select(.phase == "end") |
      {kernel: .kernel.name, duration_ms: .metrics.duration_ms}'

# Verify sovereignty — find any non-local events
cat .KOAS/activity/events.jsonl | \
  jq 'select(.sovereignty.local_only != true)'
# Expected: empty (all local)

# Find LLM cache hits
cat .KOAS/activity/events.jsonl | \
  jq 'select(.scope | endswith(".llm")) | select(.decision.cache_hit == true)'

# Count events per run
cat .KOAS/activity/events.jsonl | \
  jq -s 'group_by(.run_id) | map({run_id: .[0].run_id, events: length})'
```

### Using `ActivityReader` (Python API)

```python
from ragix_kernels.activity import ActivityReader

reader = ActivityReader(Path(".KOAS/activity/events.jsonl"))

# Get all events for a run
events = reader.get_events(run_id="run_20260130_215011_348bc4")

# Filter by scope
kernel_events = reader.get_events(scope="docs.kernel")

# Get summary statistics
summary = reader.get_summary()
print(f"Total events: {summary['total']}")
print(f"Kernel executions: {summary['kernel_count']}")
print(f"LLM calls: {summary['llm_count']}")
print(f"Cache hit rate: {summary['cache_hit_rate']:.1%}")
```

---

## 9. Broker Integration

When a broker gateway mediates access, activity logging captures additional information:

| Direct Mode | Brokered Mode |
|-------------|---------------|
| Actor: `system` / `operator` | Actor: `external_orchestrator` |
| Auth: `none` | Auth: `api_key` / `hmac` |
| No scope filtering | Scopes enforced per client |
| Full event stream visible | Only `docs.status` + `docs.export_external` |

The broker logs authentication events (`system.auth`) before forwarding requests to KOAS. Failed authentication attempts are logged with `phase: denied`.

See the demo at `demos/koas_docs_audit/` for a working example with relaxed and restricted modes.

---

## 10. Configuration

Activity logging is enabled in the workspace manifest:

```yaml
activity:
  enabled: true
  stream: ".KOAS/activity/events.jsonl"

# Optional: broker authentication
auth:
  enabled: false          # true for restricted mode
  acl_file: ".KOAS/auth/acl.yaml"
  require_hmac: false     # true for production
```

The activity writer is initialized at workflow start:

```python
from ragix_kernels.activity import init_activity_writer, get_activity_writer

# At workflow start
writer = init_activity_writer(workspace=workspace, run_id=run_id)

# Events are then emitted automatically by:
# - orchestrator.py (kernel start/end)
# - llm_wrapper.py (LLM call/cache_hit)
```

---

## 11. API Reference

### Core Classes

| Class | Purpose |
|-------|---------|
| `ActivityEvent` | Event data structure with `to_dict()` / `to_json()` |
| `ActivityWriter` | Append-only JSONL writer (thread-safe) |
| `ActivityReader` | Event querying and summary generation |
| `Actor` | Actor identity (type, id, auth, session) |
| `KernelInfo` | Kernel metadata (name, version, stage) |
| `NodeRef` | Document hierarchy reference (level, id) |

### Enums

| Enum | Values |
|------|--------|
| `ActorType` | `SYSTEM`, `OPERATOR`, `EXTERNAL_ORCHESTRATOR`, `AUDITOR` |
| `AuthMethod` | `NONE`, `API_KEY`, `HMAC` |

### Functions

| Function | Purpose |
|----------|---------|
| `init_activity_writer(workspace, run_id)` | Initialize global writer |
| `get_activity_writer()` | Get current writer instance |

---

## 12. Related Documentation

| Document | Description |
|----------|-------------|
| [SOVEREIGN_LLM_OPERATIONS.md](SOVEREIGN_LLM_OPERATIONS.md) | Sovereignty architecture and policy enforcement |
| [KOAS.md](KOAS.md) | KOAS philosophy and kernel families |
| [developer/ROADMAP_ACTIVITY_LOGGING.md](developer/ROADMAP_ACTIVITY_LOGGING.md) | Design specification and implementation roadmap |
| [ragix_kernels/README.md](../ragix_kernels/README.md) | Kernel developer reference |
