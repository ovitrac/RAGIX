# KOAS Memory MCP Server

> Persistent Structured Memory for Large Document Operations

**Version:** 2.0.0 (Phase 5 — Production Wiring)
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Last updated:** 2026-02-15

---

## 1. Overview

The RAGIX Memory MCP Server exposes persistent structured memory as [Model Context Protocol](https://modelcontextprotocol.io/) tools, enabling LLM assistants (Claude Desktop, Claude Code, VS Code, Cursor) to accumulate, search, and recall domain knowledge across sessions.

**Key capabilities:**
- **17 MCP tools** (11 core + 2 session bridge + 3 workspace + 1 metrics) wrapping the RAGIX Memory subsystem
- **FTS5 + FAISS-accelerated search** with BM25 ranking and optional embedding similarity
- **Three-tier memory** (STM → MTM → LTM) with automatic consolidation and promotion
- **Write governance** — 9 secret patterns, 8 injection patterns, 8 instructional-content blockers
- **Named workspace routing** — `workspace` parameter on all scope-aware tools resolves via WorkspaceRouter
- **Per-tool metrics** — call counts, latency tracking, error rates via `memory_metrics` tool
- **Token-bucket rate limiting** — per-session rate limits wired into every tool call + per-turn proposal caps
- **Secrecy tiers** — S0/S2/S3 field-level redaction for safe sharing

### 1.1 Architecture

```
MCP Client (Claude Desktop / Code / VS Code)
    │
    ▼
┌───────────────────────────────────────────────┐
│  MCP Transport (stdio / SSE)                  │
├───────────────────────────────────────────────┤
│  FastMCP Server (mcp/server.py)               │
│  ├─ Rate Limiter (mcp/rate_limiter.py)        │
│  ├─ Metrics Collector (mcp/metrics.py)        │
│  └─ Workspace Router (mcp/workspace.py)       │
├───────────────────────────────────────────────┤
│  MCP Tools Layer (mcp/tools.py)               │
│  ├─ 11 core tools (recall, search, propose…)  │
│  ├─ 2 session bridge tools (inject, store)    │
│  ├─ 3 workspace tools (list, register, remove)│
│  ├─ 1 metrics tool (memory_metrics)           │
│  └─ Formatting (mcp/formatting.py)            │
├───────────────────────────────────────────────┤
│  Memory Tool Dispatcher (tools.py)            │
│  ├─ Store (SQLite + FTS5 + WAL)               │
│  ├─ Policy Guard (policy.py)                  │
│  ├─ Embedder (FAISS / Mock / Ollama)          │
│  ├─ Recall Engine (recall.py)                 │
│  ├─ Consolidator (consolidate.py)             │
│  ├─ Graph Store (graph_store.py)              │
│  └─ Memory Palace (palace.py)                 │
└───────────────────────────────────────────────┘
```

**Design principle:** Zero business logic in the MCP layer. Every tool is a thin wrapper that delegates 1:1 to `MemoryToolDispatcher`. All domain logic resides in `ragix_core/memory/`.

---

## 2. Getting Started

### 2.1 Standalone Server

```bash
# Start with defaults (mock embedder, French FTS, S3 secrecy)
python -m ragix_core.memory.mcp --db ./project_memory.db

# With Ollama embeddings and English stemming
python -m ragix_core.memory.mcp \
    --db ./memory.db \
    --embedder ollama \
    --model nomic-embed-text \
    --fts-tokenizer en \
    --inject-budget 2000 \
    -v
```

### 2.2 Claude Desktop Integration

Add to `~/.config/claude-desktop/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ragix-memory": {
      "command": "python",
      "args": [
        "-m", "ragix_core.memory.mcp",
        "--db", "/path/to/project/memory.db",
        "--embedder", "mock",
        "--fts-tokenizer", "fr"
      ],
      "env": {
        "PYTHONPATH": "/path/to/RAGIX"
      }
    }
  }
}
```

### 2.3 Embedded in RAGIX MCP Server

When running the main RAGIX MCP server (`MCP/ragix_mcp_server.py`), memory tools are optionally registered alongside the existing 31+ tools:

```bash
export RAGIX_MEMORY_ENABLED=1
export RAGIX_MEMORY_DB=/path/to/memory.db
export RAGIX_MEMORY_EMBEDDER=mock
export RAGIX_MEMORY_FTS=fr
python MCP/ragix_mcp_server.py
```

### 2.4 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGIX_MEMORY_ENABLED` | `0` | Enable memory tools in main MCP server (`1`/`true`/`yes`) |
| `RAGIX_MEMORY_DB` | `memory.db` | SQLite database path |
| `RAGIX_MEMORY_EMBEDDER` | `mock` | Embedding backend (`mock`/`ollama`) |
| `RAGIX_MEMORY_FTS` | `fr` | FTS5 tokenizer preset (`fr`/`en`/`raw`) |
| `RAGIX_MEMORY_SECRECY` | `S3` | Default secrecy tier (`S0`/`S2`/`S3`) |
| `RAGIX_MEMORY_BUDGET` | `1500` | Default injection budget (tokens) |

---

## 3. Tool Reference

### 3.1 Tool Hierarchy

| Category | Tool | Purpose | Rate limited | Workspace-aware |
|----------|------|---------|:---:|:---:|
| **PRIMARY** | `memory_recall` | Token-budgeted context injection (default retrieval) | yes | yes |
| **SEARCH** | `memory_search` | Interactive discovery with filters | yes | yes |
| **WRITE** | `memory_propose` | Default write path (governed by policy) | yes + proposal cap | yes |
| **WRITE** | `memory_write` | Privileged direct write (dev/admin) | yes | yes |
| **CRUD** | `memory_read` | Read items by ID | yes | — |
| **CRUD** | `memory_update` | Update item fields | yes | — |
| **CRUD** | `memory_link` | Create typed relationships | yes | — |
| **LIFECYCLE** | `memory_consolidate` | Trigger dedup + merge + promotion | yes | yes |
| **LIFECYCLE** | `memory_stats` | Store statistics and health | yes | yes |
| **PALACE** | `memory_palace_list` | Browse memory palace hierarchy | yes | — |
| **PALACE** | `memory_palace_get` | Get item with palace location | yes | — |
| **SESSION** | `memory_session_inject` | Pre-call context augmentation | yes (per session_id) | — |
| **SESSION** | `memory_session_store` | Post-call knowledge extraction | yes (per session_id) | — |
| **MANAGEMENT** | `memory_workspace_list` | List registered workspaces | yes | — |
| **MANAGEMENT** | `memory_workspace_register` | Register or update a workspace | yes | — |
| **MANAGEMENT** | `memory_workspace_remove` | Remove a workspace | yes | — |
| **OBSERVABILITY** | `memory_metrics` | Per-tool call metrics and rate limiter status | **no** | — |

### 3.2 PRIMARY: memory_recall

Token-budgeted memory retrieval for context injection. This is the canonical entry point for LLM context augmentation.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | str | required | Natural language search query |
| `budget_tokens` | int | 1500 | Maximum tokens for injection block |
| `mode` | str | `hybrid` | Retrieval mode: `inject`/`catalog`/`hybrid` |
| `tier` | str | None | Filter by tier (`stm`/`mtm`/`ltm`) |
| `scope` | str | None | Filter by scope |
| `workspace` | str | None | Named workspace (resolves to scope + corpus_id) |

**Returns:**
```json
{
  "status": "ok",
  "inject_text": "[RAGIX_MEMORY format_version=1 ...]\n...",
  "items": [...],
  "tokens_used": 342,
  "matched": 5,
  "format_version": 1
}
```

**Injection format contract (format_version=1):**
```
[RAGIX_MEMORY format_version=1 type=memory_recall matched=5 injected=3 budget=1500]
--- ITEM 1/3 [MEM-abc123 | fact | ltm | tags=oracle,cve] ---
Oracle CVE-2024-1234 Patch Required
Critical vulnerability in Oracle Database 19c requires immediate patching...
--- ITEM 2/3 [...] ---
...
[/RAGIX_MEMORY tokens_used=342]
```

### 3.3 SEARCH: memory_search

Interactive discovery with composable filters. Unlike `memory_recall`, returns structured data (not formatted for injection).

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | str | required | Search text (FTS5 BM25 ranked) |
| `tags` | str | None | Comma-separated tags |
| `tier` | str | None | Filter by tier |
| `type_filter` | str | None | Filter by type |
| `domain` | str | None | Filter by document domain |
| `scope` | str | None | Filter by scope |
| `workspace` | str | None | Named workspace (resolves to scope + corpus_id) |
| `k` | int | 10 | Max results |

### 3.4 WRITE: memory_propose

Default write path. Items pass through the full governance pipeline (secret detection, injection patterns, instructional-content blockers). Approved items are stored as STM.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `items` | str (JSON) | required | JSON array of proposals |
| `scope` | str | `project` | Memory scope |
| `workspace` | str | None | Named workspace (resolves to scope + corpus_id) |
| `source_doc` | str | None | Source document for provenance |

**Proposal format:**
```json
[{
  "title": "Oracle CVE-2024-1234",
  "content": "Critical vulnerability requiring immediate patching.",
  "type": "fact",
  "tags": ["oracle", "cve", "security"],
  "provenance_hint": {
    "source_id": "audit_report_v3.md",
    "source_kind": "doc"
  }
}]
```

**Item types:** `fact`, `decision`, `definition`, `constraint`, `pattern`, `todo`, `pointer`, `note`

### 3.5 SESSION BRIDGE: memory_session_inject / memory_session_store

For hook-based integration (e.g., Claude Code hooks). Call `memory_session_inject` at the START of each turn and `memory_session_store` at the END.

**Typical flow:**
```
Turn N:
  1. Hook: memory_session_inject(query, session_id, system_context)
     → Returns augmented_context with injected memory
  2. LLM processes augmented context
  3. Hook: memory_session_store(response_text, session_id)
     → Extracts proposals, applies governance, stores accepted items
```

---

## 4. Multi-Workspace Support

Each workspace maps a human-friendly name to a `(scope, corpus_id)` pair, enabling parallel document audits with isolated memory. A protected `"default"` workspace always exists (scope=`"project"`, corpus_id=`""`).

### 4.1 Workspace Management Tools

**`memory_workspace_register`** — create or update a named workspace:

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Unique workspace name |
| `scope` | str | name | Memory scope (defaults to name) |
| `corpus_id` | str | `""` | Corpus identifier for cross-corpus operations |
| `description` | str | `""` | Free-text description |

**`memory_workspace_list`** — list all registered workspaces (no params).

**`memory_workspace_remove`** — remove a workspace by name. The `"default"` workspace is protected and cannot be removed.

### 4.2 Using the `workspace` Parameter

Six tools accept an optional `workspace` parameter: `memory_recall`, `memory_search`, `memory_propose`, `memory_write`, `memory_consolidate`, `memory_stats`. When provided, `workspace` resolves to `(scope, corpus_id)` via the WorkspaceRouter, overriding the explicit `scope` parameter.

```
# Register a workspace for the SIAS audit
memory_workspace_register(name="sias", scope="sias-audit", corpus_id="sias-v7",
                          description="SIAS technical audit 2026")

# All subsequent calls use workspace= instead of scope=
memory_search(query="oracle CVE", workspace="sias")
memory_propose(items='[...]', workspace="sias")
memory_recall(query="migration risks", workspace="sias")
```

### 4.3 Scope Isolation

- `workspace="default"` or omitted → scope=`"project"` (all items)
- `workspace="sias"` → scope=`"sias-audit"`, corpus_id=`"sias-v7"`
- `workspace=None` with `scope=None` → unfiltered cross-workspace query
- Unknown workspace → immediate error response (`"Unknown workspace: ..."`)

### 4.4 Backward Compatibility

When no `WorkspaceRouter` is provided to the server (e.g., old Phase 1 setup), the `workspace` parameter is silently ignored and `scope` is used directly.

---

## 5. Security and Governance

### 5.1 Write Governance Pipeline

Every item passes through the policy guard before storage:

```
Proposal → Secret Check (9 patterns) → Injection Check (8 patterns)
         → Instructional Content (8 BLOCK + 4 QUARANTINE patterns)
         → Size Check → Provenance Check → Accept / Quarantine / Reject
```

### 5.2 Pattern Groups

| Group | Count | Action | Examples |
|-------|-------|--------|----------|
| Secrets | 9 | REJECT | API keys, SSH keys, JWT tokens, AWS credentials |
| Injection | 8 | REJECT | "Ignore previous instructions", system prompt overrides |
| Instructional BLOCK | 8 | REJECT | Tool invocation syntax, system/role fragments, JSON payloads |
| Instructional QUARANTINE | 4 | QUARANTINE | "Always remember to...", "In future sessions..." |

### 5.3 Quarantine Semantics

Quarantined items are stored but **excluded from automatic injection**:

| Tool | Injectable=True | Injectable=False (quarantined) |
|------|----------------|-------------------------------|
| `memory_recall` | Included | **Excluded** |
| `memory_session_inject` | Included | **Excluded** |
| `memory_search` | Shown | Shown + `[QUARANTINED]` flag |
| `memory_stats` | Counted | Counted separately |

This prevents accidental amplification of instructional content across sessions while preserving full auditability.

### 5.4 Secrecy Tiers

| Tier | Paths | Emails | IPs | Hostnames | Hashes |
|------|-------|--------|-----|-----------|--------|
| S0 (public) | REDACT | REDACT | REDACT | REDACT | REDACT |
| S2 (internal) | keep | keep | REDACT | REDACT | REDACT |
| S3 (full) | keep | keep | keep | keep | keep |

### 5.5 Rate Limiting

Per-session token-bucket rate limiting is wired into every tool call via the `_rate_guard()` closure. When a tool call exceeds the rate limit, it returns an error response with `retry_after_ms` instead of executing.

| Limit | Default | Description |
|-------|---------|-------------|
| `calls_per_minute` | 60 | Max tool calls per minute per session |
| `proposals_per_turn` | 10 | Max memory proposals per turn (checked in `memory_propose`) |
| `max_content_length` | 5000 | Max content length per proposal |
| `burst_multiplier` | 1.5 | Allows burst up to 1.5x base rate |

**Session isolation:** Core tools share a default bucket (`"mcp"`). Session bridge tools (`memory_session_inject`, `memory_session_store`) use their explicit `session_id`, so a busy session cannot starve core tools.

**Exemptions:** `memory_metrics` is not rate-limited (read-only observability tool).

**CLI flags:**
```bash
python -m ragix_core.memory.mcp --db memory.db --no-rate-limit   # disable rate limiting
python -m ragix_core.memory.mcp --db memory.db --rate-limit      # enable (default)
```

**Rate-limited error response:**
```json
{
  "status": "error",
  "message": "Rate limit exceeded for memory_search",
  "reason": "rate_limit_exceeded",
  "retry_after_ms": 1042
}
```

---

## 6. Metrics and Observability

### 6.1 Per-Tool Metrics via `memory_metrics`

Every tool call is automatically timed via the `_timed()` closure, which records call count, latency, and errors in the `MetricsCollector`. The `memory_metrics` tool exposes this data.

**Aggregate summary** (no params):

```json
{
  "status": "ok",
  "total_calls": 342,
  "total_errors": 2,
  "avg_latency_ms": 12.5,
  "tools_used": 8,
  "uptime_seconds": 3600.0,
  "started_at": "2026-02-15T10:00:00+00:00",
  "by_tool": [
    {"tool_name": "memory_recall", "call_count": 150, "avg_latency_ms": 8.3, "error_count": 0, ...},
    {"tool_name": "memory_search", "call_count": 89, "avg_latency_ms": 15.1, "error_count": 1, ...}
  ],
  "rate_limiter": {
    "tokens": 58.5,
    "max_tokens": 90.0,
    "proposals_this_turn": 3,
    "proposals_limit": 10,
    "current_turn": 7,
    "enabled": true
  }
}
```

**Per-tool detail** (`tool_name="memory_search"`):

```json
{
  "status": "ok",
  "metrics": {
    "tool_name": "memory_search",
    "call_count": 89,
    "total_latency_ms": 1344.2,
    "error_count": 1,
    "last_call_at": "2026-02-15T11:15:22+00:00",
    "avg_latency_ms": 15.1
  }
}
```

### 6.2 Store Statistics via `memory_stats`

The `memory_stats` tool returns store-level statistics (item counts, tier distribution, FTS5 status). It is separate from `memory_metrics` (which tracks MCP-layer call performance).

### 6.3 JSON Logging

With `--verbose`, all tool calls are logged with structured JSON:

```json
{"timestamp": "2026-02-15T10:30:00Z", "tool": "memory_recall", "latency_ms": 8.3, "status": "ok", "items_returned": 3}
```

---

## 7. Consolidation

### 7.1 Automatic Triggers

Consolidation fires automatically when:
1. **STM count >= threshold** (default: 20 items)
2. **Context fraction >= 15%** of model context limit

### 7.2 Consolidation Pipeline

```
STM items → Cluster by type+tags+embeddings
          → Merge duplicates (deterministic or via Granite 3.2B)
          → Promote high-value items: STM → MTM → LTM
          → Graph edges updated (similarity + semantic links)
```

### 7.3 Promotion Rules

| Condition | Promotion |
|-----------|-----------|
| Usage count >= 5 | MTM → LTM |
| Type = constraint/decision/definition | Auto-promote on consolidation |
| Validation = verified | Eligible for LTM |
| Confidence >= 0.8 + provenance | Eligible for LTM |

---

## 8. Memory Data Model

### 8.1 MemoryItem Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique ID (MEM-xxxx format) |
| `title` | str | Concise title |
| `content` | str | Full content (max 2000 chars, or pointer type) |
| `type` | str | One of 8 types (fact, decision, definition, ...) |
| `tier` | str | Memory tier: stm / mtm / ltm |
| `tags` | list | 3-7 lowercase hyphenated tags |
| `entities` | list | Named entities extracted from content |
| `confidence` | float | 0.0-1.0 confidence score |
| `validation` | str | unverified / verified / contradicted |
| `injectable` | bool | Whether included in automatic recall/inject |
| `provenance` | obj | Source tracking (source_id, source_kind, content_hash) |
| `scope` | str | Workspace scope |
| `corpus_id` | str | Corpus identifier |

### 8.2 Three-Tier Model

```
STM (Short-Term Memory)
  │ — New items land here
  │ — Quarantined items stay here (never promoted)
  │ — Expires after 72h if unverified
  │
  ▼ (consolidation promotes)
MTM (Medium-Term Memory)
  │ — Verified findings, recurring patterns
  │ — Usage tracking enables LTM promotion
  │
  ▼ (usage + verification promotes)
LTM (Long-Term Memory)
  │ — Core domain knowledge
  │ — Decisions, constraints, definitions
  │ — Persists indefinitely
```

---

## 9. CLI Quick Reference

### 9.1 Standalone MCP Server

```bash
# Basic
python -m ragix_core.memory.mcp --db memory.db

# Full options
python -m ragix_core.memory.mcp \
    --db /path/to/memory.db \
    --embedder ollama \
    --model nomic-embed-text \
    --fts-tokenizer fr \
    --secrecy S3 \
    --inject-budget 2000 \
    --rate-limit \
    -v

# Without rate limiting (development/debugging)
python -m ragix_core.memory.mcp --db memory.db --no-rate-limit
```

### 9.2 summaryctl Query (CLI Search)

```bash
# Full-text search
summaryctl query /path/to/workspace "oracle CVE" -v

# With filters
summaryctl query /path/to/workspace "migration" --tier ltm --domain oracle -k 5

# JSON output
summaryctl query /path/to/workspace "sécurité" --json

# Hybrid scored search (with embeddings)
summaryctl query /path/to/workspace "architecture" --scored --embedder ollama
```

### 9.3 Memory CLI

```bash
# Search
python -m ragix_core.memory.cli search "oracle" --db memory.db

# Show item
python -m ragix_core.memory.cli show MEM-abc123 --db memory.db

# Statistics
python -m ragix_core.memory.cli stats --db memory.db

# Consolidate
python -m ragix_core.memory.cli consolidate --db memory.db

# Export / Import
python -m ragix_core.memory.cli export --db memory.db > backup.json
python -m ragix_core.memory.cli import backup.json --db memory.db
```

---

## 10. Testing

### 10.1 Test Suite

```bash
# Run all memory tests (493 tests, ~1.8s)
pytest ragix_core/memory/tests/ -v

# Run specific phase tests
pytest ragix_core/memory/tests/test_memory_mcp.py -v       # Phase 1: MCP wrapper (38 tests)
pytest ragix_core/memory/tests/test_memory_mcp_phase2.py -v # Phase 2: Session bridge (55 tests)
pytest ragix_core/memory/tests/test_memory_mcp_phase3.py -v # Phase 3: Middleware (47 tests)
pytest ragix_core/memory/tests/test_memory_mcp_phase4.py -v # Phase 4: Production modules (49 tests)
pytest ragix_core/memory/tests/test_memory_mcp_phase5.py -v # Phase 5: Wiring integration (38 tests)
```

### 10.2 Coverage Summary

| Phase | Focus | Tests |
|-------|-------|-------|
| Core (pre-MCP) | Store, policy, recall, consolidation, palace, qsearch, graph, GPU | 266 |
| Phase 1 | MCP tool wrappers, formatting, injection contract | 38 |
| Phase 2 | Session bridge, instructional governance, injectable field | 55 |
| Phase 3 | Middleware hooks, auto-consolidation, agent wiring | 47 |
| Phase 4 | WorkspaceRouter, MetricsCollector, RateLimiter modules | 49 |
| Phase 5 | Wiring: workspace resolution, rate guard, metrics timing, management tools | 38 |
| **Total** | | **493** |

---

## 11. Comparison with Claude Code Auto-Memory

| Feature | Claude Code Auto-Memory | RAGIX Memory MCP |
|---------|------------------------|-------------------|
| Storage | Markdown files (MEMORY.md) | SQLite + FTS5 |
| Capacity | ~200 lines | Unlimited (tested at 1,199 items) |
| Search | Linear scan | BM25 + embeddings + tags |
| Tiers | None (flat) | STM → MTM → LTM |
| Governance | None | 25+ patterns (secrets, injection, instructional) |
| Graph | None | Entity links + similarity edges |
| Consolidation | Manual | Automatic clustering + merging |
| Secrecy | None | S0/S2/S3 redaction |
| Embeddings | None | FAISS-accelerated |

**Recommendation:** Use Claude Code auto-memory for behavioral preferences and session notes. Use RAGIX Memory for accumulated domain knowledge across document corpora.

---

## 12. File Map

| File | Purpose | Lines |
|------|---------|-------|
| `ragix_core/memory/mcp/server.py` | FastMCP server setup, CLI args, module wiring | ~250 |
| `ragix_core/memory/mcp/tools.py` | 17 MCP tool wrappers + rate guard/metrics/workspace closures | ~990 |
| `ragix_core/memory/mcp/formatting.py` | Injection block formatter | ~150 |
| `ragix_core/memory/mcp/session.py` | Session state management | ~215 |
| `ragix_core/memory/mcp/workspace.py` | SQLite-backed named workspace router | ~325 |
| `ragix_core/memory/mcp/metrics.py` | Thread-safe per-tool metrics collector | ~250 |
| `ragix_core/memory/mcp/rate_limiter.py` | Token-bucket per-session rate limiter | ~235 |
| `ragix_core/memory/mcp/prompts/memory_guide.md` | MCP prompt resource | ~30 |
| `ragix_core/memory/tools.py` | Core dispatcher (9 actions) | ~370 |
| `ragix_core/memory/store.py` | SQLite + FTS5 store | ~1200 |
| `ragix_core/memory/policy.py` | Write governance engine | ~290 |
| `ragix_core/memory/middleware.py` | Chat pipeline hooks | ~350 |

---

## 13. References

- [Model Context Protocol specification](https://modelcontextprotocol.io/)
- [RAGIX Memory architecture](../ragix_core/memory/README.md)
- [Memory subsystem roadmap](developer/ROADMAP_MEMORY.md)
- [MCP server plan](developer/PLAN_MEMORY_MCP_SKILL.md)
- [KOAS Summary documentation](KOAS_SUMMARY.md)
