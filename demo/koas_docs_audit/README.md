# KOAS Docs Audit Demo

**Corpus:** `docs/**/*.md` (79 markdown files)
**Purpose:** Validate activity logging implementation in two modes

---

## Requirements

All requirements are met with a full RAGIX install.

### Core (Relaxed Mode)

```bash
# Already in RAGIX requirements
pip install ragix  # or install from source
```

| Package | Version | Purpose |
|---------|---------|---------|
| `ragix_kernels` | ≥0.65.0 | KOAS pipeline |
| `pyyaml` | ≥6.0 | Configuration |
| `ollama` | running | Local LLM |

### Broker (Restricted Mode)

```bash
# Additional for broker gateway
pip install fastapi uvicorn
```

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥0.100 | HTTP API |
| `uvicorn` | ≥0.20 | ASGI server |
| `pyyaml` | ≥6.0 | ACL parsing |

### LLM Models

```bash
# Required models (via Ollama)
ollama pull granite3.1-dense:8b   # Worker
ollama pull mistral:7b-instruct   # Tutor
```

---

## Demo Modes

| Mode | Description | Access Control | Broker |
|------|-------------|----------------|--------|
| **Relaxed** | Claude manages KOAS directly via CLI | None | No |
| **Restricted** | Full access control with broker gateway | API key + ACL | Yes |

---

## Quick Start

### 1. Setup Demo Workspace

```bash
cd /home/olivi/Documents/Adservio/Projects/RAGIX/demo/koas_docs_audit

# Create workspace with symlink to docs corpus
./setup.sh
```

### 2. Run in Relaxed Mode (Direct)

```bash
# Claude/operator runs KOAS directly
./run_relaxed.sh

# Or manually:
python -m ragix_kernels.run_doc_koas run \
    --workspace ./workspace \
    --all \
    --output-level=internal
```

### 3. Run in Restricted Mode (Brokered)

```bash
# Start broker (separate terminal)
./start_broker.sh

# Trigger via API (simulates external orchestrator)
./run_restricted.sh

# Or manually with curl:
curl -X POST http://localhost:8080/koas/v1/jobs \
    -H "Authorization: Bearer $KOAS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"mode": "pure_docs", "workspace": "./workspace"}'
```

---

## Directory Structure

```
demo/koas_docs_audit/
├── README.md              # This file
├── setup.sh               # Initialize workspace
├── run_relaxed.sh         # Run in relaxed mode
├── run_restricted.sh      # Run in restricted mode
├── start_broker.sh        # Start broker gateway
├── config/
│   ├── relaxed.yaml       # Config for relaxed mode
│   ├── restricted.yaml    # Config for restricted mode
│   └── acl.yaml           # ACL for restricted mode
├── workspace/             # Created by setup.sh
│   ├── docs/ -> symlink   # Symlink to ../../docs/
│   ├── .RAG/              # RAG index (auto-generated)
│   └── .KOAS/             # KOAS outputs
│       ├── activity/
│       │   └── events.jsonl   # Activity stream
│       ├── auth/
│       │   └── acl.yaml       # ACL (restricted mode)
│       └── ...
└── broker/                # Broker implementation (restricted mode)
    └── main.py
```

---

## Expected Outputs

### Relaxed Mode

After running `./run_relaxed.sh`:

```bash
# Check activity log
cat workspace/.KOAS/activity/events.jsonl | jq -s 'length'
# Expected: ~20-30 events (kernel starts/ends, LLM calls)

# Check report
head -50 workspace/.KOAS/final_report.md
```

### Restricted Mode

After running `./run_restricted.sh`:

```bash
# Check job status
curl http://localhost:8080/koas/v1/jobs/JOB_ID

# Check activity log (includes auth events)
cat workspace/.KOAS/activity/events.jsonl | jq 'select(.scope == "system.auth")'

# Get external-safe artifact
curl http://localhost:8080/koas/v1/jobs/JOB_ID/artifact?view=external -o report.zip
```

---

## Validation Checklist

### Phase 1: Event Schema (v0.66.0)

- [ ] Events written to `events.jsonl` for every kernel
- [ ] Events have envelope format with `v`, `ts`, `event_id`, `run_id`
- [ ] Actor field populated (`system` for direct, `external_orchestrator` for brokered)
- [ ] Kernel info captured (name, version, stage)
- [ ] Metrics captured (duration, counts)

### Phase 2: ACL (v0.66.0)

- [ ] ACL file parsed correctly
- [ ] Invalid key rejected
- [ ] Scope enforcement works (trigger OK, export denied)
- [ ] Rate limiting applied

### Phase 3: Broker (v0.67.0) — OPTIONAL

- [ ] POST /jobs creates job
- [ ] GET /jobs/{id} returns status + metrics only
- [ ] GET /jobs/{id}/artifact returns sanitized output
- [ ] All broker requests logged to activity stream

---

## Test Commands

```bash
# Verify event schema
cat workspace/.KOAS/activity/events.jsonl | head -1 | jq '.v'
# Expected: "koas.event/1.0"

# Count events by scope
cat workspace/.KOAS/activity/events.jsonl | jq -s 'group_by(.scope) | map({scope: .[0].scope, count: length})'

# Verify no content in activity log
grep -c "content" workspace/.KOAS/activity/events.jsonl
# Expected: 0 (no raw content logged)

# Verify sovereignty attestation
cat workspace/.KOAS/activity/events.jsonl | jq 'select(.sovereignty.local_only == true)' | head -1
```

---

## API Keys

### Demo Keys (Auto-Generated)

`setup.sh` creates `.demo_keys` with pre-configured demo credentials:

```bash
# Created by setup.sh (chmod 600)
DEMO_OPERATOR_KEY=koas_key_demo_operator_12345
CLAUDE_DEMO_KEY=koas_key_claude_demo_67890
```

These match the `key_hash` values in `config/acl.yaml`.

### Production Keys

For production deployments, generate secure keys and compute their hashes:

```bash
# Generate a secure random key
NEW_KEY="koas_key_$(openssl rand -hex 16)"
echo "Key: $NEW_KEY"

# Compute SHA256 hash for ACL
KEY_HASH=$(echo -n "$NEW_KEY" | sha256sum | cut -d' ' -f1)
echo "Hash: sha256:$KEY_HASH"
```

Add to `config/acl.yaml`:

```yaml
clients:
  my-client:
    key_hash: "sha256:<computed_hash>"
    type: operator
    scopes: ["docs.trigger", "docs.status"]
```

Store keys securely (vault, secrets manager) — never commit to git.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Ollama not responding" | Ollama not running | `ollama serve` |
| "workspace not initialized" | Setup not run | `./setup.sh` |
| "401 Unauthorized" | Invalid API key | Check `.demo_keys` and `config/acl.yaml` |
| "403 Forbidden" | Scope not allowed | Check client scopes in ACL |
| Empty events.jsonl | Activity writer not integrated | Check Phase 1 implementation |

---

*Demo for KOAS Activity Logging — v0.66.0*
