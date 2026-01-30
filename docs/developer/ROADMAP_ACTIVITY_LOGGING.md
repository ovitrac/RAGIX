# Roadmap: Centralized Auditable Activity Logging for KOAS

**Project:** RAGIX — KOAS Document Pipeline
**Reference:** `docs/developer/Baseline proposal - Centralized, auditable activity logging for KOAS (Docs).md`
**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Date:** 2026-01-30
**Status:** Design Specification

---

## Executive Summary

KOAS already logs for **sovereignty**; the next step is to log for **activity governance**. This roadmap specifies a centralized, auditable activity logging system spanning kernels and orchestration, with key-based access control for external interaction.

### Two Modes of Operation

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Direct** | Claude manages KOAS via CLI/MCP | Default. Most deployments. |
| **Brokered** | Authenticated gateway for critical apps | Strict isolation, rate limits, multi-tenant. |

**The broker is OPTIONAL** — only for very critical applications requiring strict access control.

### Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Event Schema** | JSONL with envelope | Grep-able + structured + versioned |
| **Key Mechanism** | API key + ACL + optional HMAC | Simple, revocable, no infra |
| **External Auth** | Direct (default) or Broker (critical) | Flexibility vs. security |
| **Storage** | Append-only `.jsonl` + exporter | Resilient + audit bundles |
| **Broker Role** | "Front desk + guard + bookkeeper" | Never reimplements KOAS |

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Activity Event Schema](#2-activity-event-schema)
3. [Key Mechanism (AuthN/AuthZ)](#3-key-mechanism-authnauthz)
4. [External Orchestrator Protocol](#4-external-orchestrator-protocol)
5. [Implementation Phases](#5-implementation-phases)
6. [File Structure](#6-file-structure)
7. [Integration Points](#7-integration-points)
8. [Security Considerations](#8-security-considerations)

---

## 1. Design Principles

### 1.1 Activity Logging ≠ Content Logging

The system logs **what happened**, not **what was processed**.

| Logged | NOT Logged |
|--------|------------|
| Kernel name, version, stage | Document content |
| Execution timestamps | Raw text excerpts |
| Input/output hashes | File contents |
| Decisions (cache hit/miss) | Prompt/response text |
| Actor identity | Personal data |
| Metrics (counts, scores) | Internal paths (in external view) |

### 1.2 Single Canonical Event Stream

**Do NOT create parallel logs.** All activity flows into one append-only stream:

```
.KOAS/activity/events.jsonl    ← canonical source
.KOAS/activity/exports/        ← audit bundles (generated on demand)
```

### 1.3 Sovereignty + Governance

| Goal | Mechanism |
|------|-----------|
| Prove local execution | Sovereignty attestation in events |
| Track who did what | Actor field with auth method |
| Enable audits | Structured events + signatures |
| Detect misuse | Centralized stream + access logging |

---

## 2. Activity Event Schema

### 2.1 Event Envelope (v1.0)

Each line in `events.jsonl` is a self-contained envelope:

```json
{
  "v": "koas.event/1.0",
  "ts": "2026-01-30T20:14:12.231Z",
  "event_id": "01JFXYZ...",
  "run_id": "run_20260130_201200_abc123",
  "actor": {
    "type": "external_orchestrator",
    "id": "claude-prod",
    "auth": "hmac",
    "session": "sess_01JFX..."
  },
  "scope": "docs.audit",
  "kernel": {
    "name": "doc_extract",
    "version": "1.2.0",
    "stage": 2
  },
  "phase": "representative_sentences",
  "node_ref": {
    "level": "doc",
    "id": "doc:PROJECT:013"
  },
  "io": {
    "inputs_merkle": "sha256:a1b2c3...",
    "outputs_merkle": "sha256:d4e5f6..."
  },
  "decision": {
    "cache": "hit",
    "llm": "disabled",
    "reason": "kernel_cache_valid"
  },
  "metrics": {
    "sentences_total": 142,
    "sentences_kept": 8,
    "quality_mean": 0.71,
    "duration_ms": 234
  },
  "refs": {
    "call_hash": "sha256:...",
    "parent_event": "01JFXYW..."
  },
  "sovereignty": {
    "local_only": true,
    "endpoint": "localhost:11434"
  },
  "sig": "ed25519:..."
}
```

### 2.2 Event Types

| Event Type | Scope | Emitted By |
|------------|-------|------------|
| `workflow.start` | `docs.audit` | Orchestrator |
| `workflow.end` | `docs.audit` | Orchestrator |
| `kernel.start` | `docs.kernel` | Kernel wrapper |
| `kernel.end` | `docs.kernel` | Kernel wrapper |
| `llm.call` | `docs.llm` | LLM wrapper |
| `llm.cache_hit` | `docs.llm` | LLM wrapper |
| `auth.granted` | `system.auth` | Auth layer |
| `auth.denied` | `system.auth` | Auth layer |
| `export.requested` | `docs.export` | Export handler |

### 2.3 Minimal Event (Required Fields)

```json
{
  "v": "koas.event/1.0",
  "ts": "2026-01-30T20:14:12.231Z",
  "event_id": "01JFXYZ...",
  "run_id": "run_20260130_201200_abc123",
  "actor": {"type": "system", "id": "koas"},
  "scope": "docs.kernel",
  "kernel": {"name": "doc_metadata", "version": "1.0.0"}
}
```

### 2.4 Schema Versioning

The `v` field enables forward compatibility:

```python
SCHEMA_VERSION = "koas.event/1.0"

def parse_event(line: str) -> dict:
    event = json.loads(line)
    if event["v"] != SCHEMA_VERSION:
        return migrate_event(event)  # Handle older schemas
    return event
```

---

## 3. Key Mechanism (AuthN/AuthZ)

### 3.1 Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  External       │     │    Broker       │     │     KOAS        │
│  Orchestrator   │────▶│   (Gateway)     │────▶│   Activity      │
│  (Claude)       │     │                 │     │   Stream        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │ No secrets            │ API key + HMAC        │ Append-only
        │ No direct access      │ ACL enforcement       │ Signed events
```

### 3.2 ACL File Format

**File:** `.KOAS/auth/acl.yaml`

```yaml
# KOAS Activity ACL — v1.0
# This file controls access to activity logs and workflow triggers

schema_version: "koas.acl/1.0"

clients:
  # Internal system (full access)
  koas-system:
    key_hash: null  # No key required for internal
    type: system
    scopes:
      - "*"  # All scopes

  # Human operator (read + trigger)
  operator-alice:
    key_hash: "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    type: operator
    scopes:
      - "docs.trigger"
      - "docs.status"
      - "docs.export"
      - "activity.read"
    rate_limit: "60/min"
    expires: "2026-12-31"

  # External orchestrator (limited)
  claude-prod:
    key_hash: "sha256:..."
    type: external_orchestrator
    scopes:
      - "docs.trigger"
      - "docs.status"
      # NOT: activity.read, docs.export.full
    rate_limit: "30/min"
    allowed_sources:
      - "127.0.0.1"
      - "10.0.0.0/8"
    expires: "2026-06-30"
    restrictions:
      - "no_content_access"
      - "metrics_only"

  # Audit service (read-only)
  auditor-external:
    key_hash: "sha256:..."
    type: auditor
    scopes:
      - "activity.read"
      - "activity.export"
    allowed_sources:
      - "192.168.1.100"
    session_max_duration: "1h"

# Scope definitions
scopes:
  docs.trigger:
    description: "Trigger KOAS workflow execution"
    endpoints: ["/jobs"]
    methods: ["POST"]

  docs.status:
    description: "View workflow status and metrics"
    endpoints: ["/jobs/*"]
    methods: ["GET"]
    filters: ["metrics_only"]

  docs.export:
    description: "Export external-safe reports"
    endpoints: ["/jobs/*/report"]
    methods: ["GET"]
    output_level: "external"

  activity.read:
    description: "Read activity event stream"
    endpoints: ["/activity/events"]
    methods: ["GET"]

  activity.export:
    description: "Export audit bundles"
    endpoints: ["/activity/export"]
    methods: ["POST"]
```

### 3.3 Key Generation and Storage

```bash
# Generate API key for a client
python -m ragix_kernels.auth generate-key --client claude-prod

# Output:
# API Key (store securely, shown once): koas_key_a1b2c3d4e5f6...
# Key Hash (for ACL): sha256:9f86d081884c7d659a2feaa0c55ad015...

# Add to ACL
# Edit .KOAS/auth/acl.yaml with the key_hash
```

**Key format:** `koas_key_<32 random hex bytes>` (256-bit entropy)

**Storage:** Only the hash is stored in ACL. The key itself is given to the client once.

### 3.4 HMAC Request Signing (Optional, Recommended)

For request integrity and anti-replay:

```python
import hmac
import hashlib
import time

def sign_request(api_key: str, method: str, path: str, body: str, timestamp: int, nonce: str) -> str:
    """Sign a request with HMAC-SHA256."""
    message = f"{timestamp}:{nonce}:{method}:{path}:{hashlib.sha256(body.encode()).hexdigest()}"
    return hmac.new(api_key.encode(), message.encode(), hashlib.sha256).hexdigest()

# Request headers:
# X-KOAS-Key-ID: claude-prod
# X-KOAS-Timestamp: 1706648052
# X-KOAS-Nonce: abc123xyz
# X-KOAS-Signature: <hmac signature>
```

**Anti-replay:** Reject requests with timestamp > 5 minutes old or reused nonce.

---

## 4. External Orchestrator Protocol

### 4.1 Two Modes of Operation

KOAS supports two modes for external orchestration:

| Mode | Use Case | Security Level | Complexity |
|------|----------|----------------|------------|
| **Direct** | Claude manages KOAS directly via CLI/MCP | Standard | Low |
| **Brokered** | Critical applications requiring strict isolation | High | Medium |

**Default:** Direct mode. Claude calls KOAS CLI commands directly.

**Brokered mode:** Optional, for very critical applications where:
- External orchestrator must never see internal paths/IDs
- Strict audit trail is mandatory
- Rate limiting and access control are required

### 4.2 Direct Mode (Default)

Claude interacts with KOAS via CLI or MCP tools:

```bash
# Claude executes directly
python -m ragix_kernels.run_doc_koas run --workspace ./audit --all --output-level=external
```

Activity logging still applies (events written to `events.jsonl`), but no broker intermediary.

### 4.3 Brokered Mode (Critical Applications)

> A minimal broker is a **local authenticated job gateway** that triggers KOAS workflows, exposes only status/metrics and external-safe artifacts, and writes an append-only auditable activity log for every action.

Think: **"front desk + guard + bookkeeper"** — not a reimplementation of KOAS.

#### 4.3.1 What the Broker Does (and Does NOT Do)

| Responsibility | Broker Does | Broker Does NOT |
|----------------|-------------|-----------------|
| **AuthN/AuthZ** | ✅ Keys + scopes + rate limits | ❌ User management |
| **Sanitization** | ✅ Strip metadata, redact paths | ❌ Content processing |
| **Job Queue** | ✅ Queue, status, artifacts | ❌ Kernel logic |
| **Logging** | ✅ Activity events | ❌ Debug logs |
| **Execution** | ✅ Subprocess call to KOAS CLI | ❌ Reimplementing KOAS |

### 4.4 Broker API Surface (3 Endpoints)

**Base URL:** `http://localhost:8080/koas/v1` (local broker)

#### Endpoint 1: `POST /jobs` — Trigger Workflow

**Request:**
```json
{
  "mode": "pure_docs",
  "profile": "docs_techspec_v1",
  "input_ref": "workspace:/path/to/audit",
  "actions": ["index", "koas_audit", "export_report_external"]
}
```

**Response:**
```json
{
  "job_id": "01JFXYZ...",
  "status": "queued"
}
```

**Scope required:** `docs.trigger`

---

#### Endpoint 2: `GET /jobs/{job_id}` — Status & Metrics

**Response:**
```json
{
  "job_id": "01JFXYZ...",
  "status": "running",
  "progress": {
    "stage": "doc_cluster",
    "pct": 62
  },
  "metrics": {
    "docs": 137,
    "chunks": 5481,
    "llm_cache_hit": 0.83
  }
}
```

**Scope required:** `docs.status`

**Note:** No file paths, no content, no internal IDs exposed.

---

#### Endpoint 3: `GET /jobs/{job_id}/artifact?view=external` — Download Artifacts

**Response:** File stream (PDF + clean Markdown zip)

**Scope required:** `docs.export_external`

**Never returns:** Raw logs, internal IDs, call hashes, paths.

---

#### Optional Endpoint 4 (Auditors Only): `GET /jobs/{job_id}/artifact?view=internal`

**Response:** Full trace bundle (call_hash, Merkle roots, provenance)

**Scope required:** `docs.export_internal`

### 4.5 Broker Auth Mechanism

#### API Key in Header

```http
Authorization: Bearer <api_key>
```

Store only `sha256(api_key)` in ACL.

#### Optional HMAC Signing (Recommended if Not Localhost)

```http
X-KOAS-Timestamp: 1706648052
X-KOAS-Nonce: abc123xyz
X-KOAS-Signature: <hmac_sha256(secret, method + path + timestamp + nonce + body_hash)>
```

### 4.6 Broker Implementation Architecture

**Single-process, robust design:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     Minimal Broker                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   FastAPI   │  │   SQLite    │  │   Event     │              │
│  │   (HTTP)    │  │   (Jobs)    │  │   Stream    │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Subprocess Runner                           │   │
│  │  python -m ragix_kernels.run_doc_koas run ...            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   KOAS Pipeline     │
                    │   (existing CLI)    │
                    └─────────────────────┘
```

**Components:**
- **FastAPI** (or Flask) for HTTP
- **SQLite** for job state (recommended) or JSON files
- **Subprocess runner** calls KOAS CLI (preserves existing architecture)
- **Event stream**: append-only `events.jsonl`
- **Artifacts folder**: `output/jobs/{job_id}/...`

### 4.7 Job Runner Behavior

```python
# Simplified job state machine

class JobState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

def run_job(job: Job, activity: ActivityWriter):
    # 1. Mark started
    job.status = JobState.RUNNING
    job.started_at = datetime.now()
    activity.emit(ActivityEvent(scope="broker.job", phase="started", refs={"job_id": job.id}))

    # 2. Call KOAS CLI (subprocess)
    result = subprocess.run([
        "python", "-m", "ragix_kernels.run_doc_koas", "run",
        "--workspace", job.workspace,
        "--all",
        "--output-level", "external",
        "--quiet"
    ], capture_output=True)

    # 3. Handle result
    if result.returncode == 0:
        job.status = JobState.COMPLETED
        activity.emit(ActivityEvent(scope="broker.job", phase="completed", refs={"job_id": job.id}))
    else:
        job.status = JobState.FAILED
        # Log error category only (no secrets)
        activity.emit(ActivityEvent(
            scope="broker.job",
            phase="failed",
            refs={"job_id": job.id},
            decision={"error_code": result.returncode}
        ))

    job.ended_at = datetime.now()
```

### 4.8 What Claude Sees vs. What KOAS Knows

| Information | Direct Mode | Brokered Mode (Claude) | Brokered Mode (KOAS) |
|-------------|-------------|------------------------|----------------------|
| Job ID | N/A (run_id) | ✅ `01JFXYZ...` | ✅ Full mapping |
| Status | ✅ stdout | ✅ `running` | ✅ Detailed state |
| Progress | ✅ stdout | ✅ `62%` | ✅ Per-kernel timing |
| Metrics | ✅ Full | ✅ Aggregates only | ✅ Full breakdown |
| File paths | ✅ Visible | ❌ Redacted | ✅ Full paths |
| Document content | ❌ Never | ❌ Never | ❌ Never |
| API key | N/A | ❌ Never holds it | ✅ Key ID logged |
| Activity stream | ✅ File access | ❌ No access | ✅ Full stream |

---

## 5. Implementation Phases

### Phase 1: Event Schema & Writer (v0.66.0)

**Goal:** Establish canonical event stream.

**Deliverables:**
- `ragix_kernels/activity.py` — Event schema, writer, reader
- Event emission from kernel wrapper
- Append-only `.KOAS/activity/events.jsonl`

**Files:**
```python
# ragix_kernels/activity.py

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import uuid

SCHEMA_VERSION = "koas.event/1.0"

@dataclass
class Actor:
    type: str  # system | operator | external_orchestrator | auditor
    id: str
    auth: str = "none"  # none | api_key | hmac
    session: Optional[str] = None

@dataclass
class KernelInfo:
    name: str
    version: str
    stage: int = 0

@dataclass
class ActivityEvent:
    scope: str
    actor: Actor
    kernel: Optional[KernelInfo] = None
    phase: Optional[str] = None
    node_ref: Optional[Dict[str, str]] = None
    io: Optional[Dict[str, str]] = None
    decision: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, Any]] = None
    refs: Optional[Dict[str, str]] = None
    sovereignty: Dict[str, Any] = field(default_factory=lambda: {"local_only": True})

    # Auto-generated
    v: str = field(default=SCHEMA_VERSION)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(',', ':'))

class ActivityWriter:
    def __init__(self, workspace: Path):
        self.log_path = workspace / ".KOAS" / "activity" / "events.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def emit(self, event: ActivityEvent) -> None:
        event.run_id = self.run_id
        with open(self.log_path, "a") as f:
            f.write(event.to_json() + "\n")
```

**Effort:** 3 days

---

### Phase 2: ACL & Auth Layer (v0.66.0)

**Goal:** Key-based access control.

**Deliverables:**
- `ragix_kernels/auth.py` — ACL parser, key validation, scope checking
- `.KOAS/auth/acl.yaml` template
- CLI: `ragix-koas auth generate-key`, `ragix-koas auth check`

**Files:**
```python
# ragix_kernels/auth.py

import hashlib
import hmac
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Set

@dataclass
class Client:
    id: str
    type: str
    scopes: Set[str]
    key_hash: Optional[str]
    rate_limit: Optional[str] = None
    expires: Optional[str] = None

class ACLManager:
    def __init__(self, acl_path: Path):
        with open(acl_path) as f:
            self.config = yaml.safe_load(f)
        self.clients = self._parse_clients()

    def _parse_clients(self) -> dict[str, Client]:
        clients = {}
        for cid, cfg in self.config.get("clients", {}).items():
            clients[cid] = Client(
                id=cid,
                type=cfg.get("type", "unknown"),
                scopes=set(cfg.get("scopes", [])),
                key_hash=cfg.get("key_hash"),
                rate_limit=cfg.get("rate_limit"),
                expires=cfg.get("expires"),
            )
        return clients

    def authenticate(self, key_id: str, api_key: str) -> Optional[Client]:
        client = self.clients.get(key_id)
        if not client or not client.key_hash:
            return None

        provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if f"sha256:{provided_hash}" == client.key_hash:
            return client
        return None

    def authorize(self, client: Client, scope: str) -> bool:
        if "*" in client.scopes:
            return True
        return scope in client.scopes
```

**Effort:** 2 days

---

### Phase 3: Broker Gateway (v0.67.0) — OPTIONAL

**Goal:** Authenticated job gateway for critical applications.

> **Note:** This phase is OPTIONAL. Most deployments use Direct Mode where Claude manages KOAS via CLI.

**When to implement:**
- Strict isolation required (external orchestrator must never see paths/IDs)
- Rate limiting and access control mandatory
- Multi-tenant or shared infrastructure

**Deliverables:**
- `ragix_kernels/broker/` — FastAPI gateway module
  - `main.py` — HTTP endpoints
  - `jobs.py` — Job state machine + SQLite storage
  - `runner.py` — Subprocess KOAS CLI invocation
  - `sanitize.py` — Output sanitization (uses `output_sanitizer.py`)
- Endpoints: `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/artifact`
- Activity logging for all broker requests

**KOAS CLI Integration:**

The broker calls KOAS via subprocess (preserves existing architecture):

```python
# runner.py
import subprocess

def execute_koas_workflow(workspace: str, profile: str, output_level: str = "external") -> int:
    """Execute KOAS pipeline via CLI."""
    cmd = [
        "python", "-m", "ragix_kernels.run_doc_koas", "run",
        "--workspace", workspace,
        "--all",
        "--output-level", output_level,
        "--quiet"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode
```

**Effort:** 4 days

---

### Phase 4: Audit Export (v0.67.0)

**Goal:** Generate audit bundles on demand.

**Deliverables:**
- `ragix_kernels/audit_export.py` — Bundle generator
- Signed audit packages (`.tar.gz` with manifest + signature)
- CLI: `ragix-koas activity export --from 2026-01-01 --to 2026-01-31`

**Effort:** 2 days

---

### Phase 5: Event Signing (v0.68.0)

**Goal:** Tamper-evident event stream.

**Deliverables:**
- Ed25519 key pair generation
- Per-event signatures
- Chain verification (optional hash chain)

**Effort:** 3 days

---

## 6. File Structure

```
.KOAS/
├── activity/
│   ├── events.jsonl           # Canonical event stream (append-only)
│   ├── events.jsonl.sig       # Stream signature (optional)
│   └── exports/
│       └── audit_202601.tar.gz  # Exported audit bundle
├── auth/
│   ├── acl.yaml               # Access control list
│   ├── keys/
│   │   └── signing.key        # Ed25519 private key (protected)
│   └── sessions/
│       └── sess_01JFX....json # Active session tokens
├── cache/
│   └── ...                    # Existing caches
└── logs/
    └── ...                    # Existing logs
```

---

## 7. Integration Points

### 7.1 Kernel Wrapper Integration

```python
# In ragix_kernels/base.py or orchestrator.py

def run_kernel(kernel: Kernel, input: KernelInput, activity: ActivityWriter) -> dict:
    # Emit start event
    activity.emit(ActivityEvent(
        scope="docs.kernel",
        actor=Actor(type="system", id="koas"),
        kernel=KernelInfo(name=kernel.name, version=kernel.version, stage=kernel.stage),
        phase="start",
    ))

    t0 = time.time()
    result = kernel.compute(input)
    duration = time.time() - t0

    # Emit end event with metrics
    activity.emit(ActivityEvent(
        scope="docs.kernel",
        actor=Actor(type="system", id="koas"),
        kernel=KernelInfo(name=kernel.name, version=kernel.version, stage=kernel.stage),
        phase="end",
        metrics={"duration_ms": int(duration * 1000), **result.get("_audit", {})},
        io={
            "inputs_merkle": compute_inputs_merkle_root(input.dependencies),
            "outputs_merkle": compute_response_hash(json.dumps(result)),
        }
    ))

    return result
```

### 7.2 LLM Wrapper Integration

```python
# In ragix_kernels/llm_wrapper.py

def call_llm(request: dict, activity: ActivityWriter) -> str:
    call_hash = compute_call_hash(request)

    # Check cache
    cached = cache.get(call_hash)
    if cached:
        activity.emit(ActivityEvent(
            scope="docs.llm",
            actor=Actor(type="system", id="koas"),
            phase="cache_hit",
            refs={"call_hash": call_hash},
            decision={"cache": "hit", "llm": "disabled"},
        ))
        return cached

    # Call LLM
    activity.emit(ActivityEvent(
        scope="docs.llm",
        actor=Actor(type="system", id="koas"),
        phase="call_start",
        refs={"call_hash": call_hash},
    ))

    response = ollama.generate(...)

    activity.emit(ActivityEvent(
        scope="docs.llm",
        actor=Actor(type="system", id="koas"),
        phase="call_end",
        refs={"call_hash": call_hash, "response_hash": compute_response_hash(response)},
        metrics={"tokens": response.usage},
        sovereignty={"local_only": True, "endpoint": "localhost:11434"},
    ))

    return response
```

---

## 8. Security Considerations

### 8.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| API key theft | Hash-only storage, rotation support |
| Replay attacks | HMAC with timestamp + nonce |
| Log tampering | Ed25519 signatures, append-only |
| Scope escalation | ACL enforcement at broker |
| Content leakage | Output sanitization, metrics-only for external |

### 8.2 Key Rotation

```bash
# Generate new key for client
ragix-koas auth rotate-key --client claude-prod

# Old key remains valid for grace period (configurable)
# After grace period, only new key works
```

### 8.3 Audit Bundle Integrity

Each exported audit bundle includes:
- `manifest.json` — List of events, date range, hash
- `events.jsonl` — Event data
- `signature.sig` — Ed25519 signature of manifest

```bash
# Verify audit bundle
ragix-koas activity verify audit_202601.tar.gz
# Output: ✓ Valid signature, 1,234 events, 2026-01-01 to 2026-01-31
```

---

## Summary

| Phase | Version | Scope | Required | Effort |
|-------|---------|-------|----------|--------|
| 1 | v0.66.0 | Event schema + writer | ✅ Yes | 3 days |
| 2 | v0.66.0 | ACL + auth layer | ✅ Yes | 2 days |
| 3 | v0.67.0 | Broker gateway | ⚠️ Optional | 4 days |
| 4 | v0.67.0 | Audit export | ✅ Yes | 2 days |
| 5 | v0.68.0 | Event signing | ⚠️ Optional | 3 days |

**Core implementation:** Phases 1, 2, 4 = **7 days**
**Full implementation:** All phases = **14 days**

### Mode Selection Guide

| Deployment | Phases Needed | Effort |
|------------|---------------|--------|
| **Standard** (Claude direct) | 1, 2, 4 | 7 days |
| **Critical** (with broker) | 1, 2, 3, 4 | 11 days |
| **Certified** (full audit) | All | 14 days |

---

## References

| Document | Description |
|----------|-------------|
| `docs/developer/Baseline proposal - Centralized, auditable activity logging for KOAS (Docs).md` | Original problem statement |
| `docs/SOVEREIGN_LLM_OPERATIONS.md` | Sovereignty requirements |
| `ragix_kernels/output_sanitizer.py` | Output isolation (existing) |
| `ragix_kernels/merkle.py` | Provenance hashing (existing) |

---

*RAGIX KOAS-Docs | Adservio Innovation Lab | 2026-01-30*
