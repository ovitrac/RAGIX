# RAGIX Demos

**Hands-on demonstrations of RAGIX capabilities**

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**RAGIX Version:** 0.66+
**Updated:** 2026-02-13

---

## Overview

This directory contains executable demonstrations covering four areas of RAGIX:

| Demo | Directory/File | What It Shows |
|------|---------------|---------------|
| **KOAS Docs Audit** | `koas_docs_audit/` | Activity logging, broker gateway, ACL enforcement |
| **KOAS MCP Interactive** | `koas_mcp_demo/` | Web UI for KOAS tools via MCP (16 tools, scenarios) |
| **KOAS Memory Pipe** | `koas_pipe_demo/` | Ingest + FTS5 recall via single `pipe` command (no LLM needed) |
| **Reasoning v30 Benchmark** | `reasoning_v30_demo.py` + scripts | Reasoning Graph with real LLM calls and multi-model comparison |

---

## Prerequisites

All demos require a working RAGIX installation and a running Ollama instance.

### Shared Requirements

| Requirement | Purpose | Check |
|-------------|---------|-------|
| Python 3.10+ | Runtime | `python3 --version` |
| RAGIX (source install) | Core + kernels | `python -c "import ragix_core"` |
| Ollama | Local LLM serving | `curl http://localhost:11434/api/tags` |

### LLM Models (via Ollama)

```bash
# Minimum set for all demos
ollama pull mistral:7b-instruct     # Tutor / general reasoning
ollama pull granite3.1-dense:8b     # Worker (KOAS docs)

# Optional — for reasoning benchmarks
ollama pull qwen2.5:7b
ollama pull deepseek-r1:14b
ollama pull llama3.2:3b
```

### Additional Python Packages

```bash
# For broker (koas_docs_audit restricted mode) and MCP demo
pip install fastapi uvicorn httpx pyyaml
```

---

## Directory Structure

```
demos/
├── README.md                       # This file
│
├── koas_docs_audit/                # Demo 1 — Activity Logging & Broker
│   ├── README.md                   # Full documentation
│   ├── setup.sh                    # Initialize workspace + configs + keys
│   ├── run_relaxed.sh              # Direct CLI mode (no auth)
│   ├── run_restricted.sh           # Brokered mode (API key + ACL)
│   ├── start_broker.sh             # Launch broker gateway
│   ├── config/
│   │   ├── relaxed.yaml            # Pipeline config (internal output)
│   │   ├── restricted.yaml         # Pipeline config (external-safe output)
│   │   └── acl.yaml                # Access control list
│   ├── broker/
│   │   └── main.py                 # FastAPI broker implementation
│   └── workspace/                  # Created by setup.sh
│       ├── docs/ -> ../../docs/    # Symlink to RAGIX docs corpus
│       ├── .KOAS/activity/         # Activity event stream
│       ├── stage1/, stage2/, stage3/
│       └── ...
│
├── koas_mcp_demo/                  # Demo 2 — Interactive MCP Web UI
│   ├── README.md                   # Full documentation
│   ├── run_demo.sh                 # Launcher with dependency checks
│   ├── server.py                   # FastAPI + WebSocket server (42K)
│   ├── test_koas_tools.py          # Tool integration tests
│   ├── static/
│   │   ├── index.html              # Single-page web UI
│   │   ├── css/koas_demo.css       # Styles
│   │   └── js/koas_client.js       # Client-side logic
│   └── scenarios/                  # Pre-built workflow definitions
│
├── koas_pipe_demo/                 # Demo 3 — Memory Pipe (ingest + recall)
│   ├── README.md                   # Full documentation + expected output
│   └── run_demo.sh                 # Self-contained demo (no LLM needed)
│
├── reasoning_v30_demo.py           # Demo 4a — Reasoning Graph demo
├── run_reasoning_benchmark.sh      # Demo 4b — Multi-model benchmark runner
├── test_classification.py          # Test — task classification
├── test_simple_execution.py        # Test — SIMPLE task flow
└── test_moderate_execution.py      # Test — MODERATE task flow
```

---

## Demo 1: KOAS Docs Audit

**Purpose:** Validate KOAS activity logging and broker gateway using RAGIX's own documentation corpus (79 Markdown files).

**Two operating modes:**

| Mode | Auth | Broker | Output Level | Use Case |
|------|------|--------|-------------|----------|
| **Relaxed** | None | No | `internal` (full traces) | Development, direct CLI |
| **Restricted** | API key + ACL | Yes | `external` (sanitized) | Production-like, external orchestrators |

### Quick Start

```bash
cd demos/koas_docs_audit

# 1. Initialize workspace, configs, and demo API keys
./setup.sh

# 2a. Run in relaxed mode (direct)
./run_relaxed.sh

# 2b. Or run in restricted mode (brokered)
./start_broker.sh          # Terminal 1 — starts broker on :8080
./run_restricted.sh        # Terminal 2 — triggers via API
```

### Key Concepts Demonstrated

- **Centralized activity logging** — every kernel execution writes to `events.jsonl`
- **Sovereignty attestation** — `sovereignty.local_only: true` per event
- **Broker gateway** — Core-Shell architecture mediating external access
- **ACL enforcement** — scope-based access control (`docs.trigger`, `docs.export_external`)
- **Output sanitization** — internal vs. external output levels

See [`koas_docs_audit/README.md`](koas_docs_audit/README.md) for validation checklist, API key management, and troubleshooting.

---

## Demo 2: KOAS MCP Interactive

**Purpose:** Interactive web interface for exploring KOAS tools (8 security + 8 audit) with real-time tool-call visualization and LLM chat.

### Quick Start

```bash
cd demos/koas_mcp_demo

# Start the demo server (opens browser automatically)
./run_demo.sh

# Or manually:
python server.py            # http://127.0.0.1:8080
```

### Features

- **16 simplified KOAS tools** — 8 security (discover, port scan, DNS, SSL, vuln scan, compliance, risk, report) + 8 audit (AST scan, metrics, dependencies, hotspots, dead code, risk, compliance, report)
- **Model selection** — choose from available Ollama models
- **Tool trace visualization** — real-time display of tool calls as they execute
- **Scenario browser** — pre-built workflows (quick scan, full assessment, quick audit, full audit)
- **Dry-run mode** — test without actual network scans
- **WebSocket updates** — real-time status via `/ws`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/tools` | GET | List available tools |
| `/api/scenarios` | GET | List pre-built scenarios |
| `/api/models` | GET | List Ollama models |
| `/api/tool` | POST | Execute a single tool |
| `/api/scenario` | POST | Run a multi-tool scenario |
| `/api/chat` | POST | Chat with LLM |
| `/api/health` | GET | Health check |
| `/ws` | WS | Real-time updates |

See [`koas_mcp_demo/README.md`](koas_mcp_demo/README.md) for programmatic usage examples and LLM accuracy benchmarks.

---

## Demo 3: KOAS Memory Pipe

**Purpose:** Demonstrate the `ragix-memory pipe` command — a single CLI invocation that ingests RAGIX source files into a SQLite memory store and retrieves relevant chunks via FTS5 full-text search. No LLM or Ollama required.

### Quick Start

```bash
cd demos/koas_pipe_demo
./run_demo.sh
```

### What It Exercises

1. **Paragraph-aware chunking** — files split at `\n\n` boundaries, merged up to token budget
2. **SHA-256 deduplication** — re-running is idempotent (zero new chunks on re-ingest)
3. **FTS5 BM25 search** — full-text search with differentiated, query-relevant results
4. **Token-budgeted injection** — output is capped to fit within an LLM context window
5. **Anti-context-poisoning** — raw ingest defaults to `injectable=false`; `pipe` uses `injectable=true` for interactive use

### Key Difference from Other Demos

| Feature | Docs Audit / MCP | Memory Pipe |
|---------|-----------------|-------------|
| Requires LLM | Yes (Ollama) | No (pure deterministic) |
| Focus | Kernel execution, UI | Data ingestion + retrieval |
| Output | Reports, web UI | Injection block (text) |
| Composability | Standalone | Pipes into LLM tools |

See [`koas_pipe_demo/README.md`](koas_pipe_demo/README.md) for expected output and technical details.

---

## Demo 4: Reasoning v30 Benchmark

**Purpose:** Exercise the full Reasoning Graph v30 pipeline (classify → plan → execute → reflect → verify) with real Ollama LLM calls and compare accuracy across models.

### Files

| File | Description |
|------|-------------|
| `reasoning_v30_demo.py` | Comprehensive demo: task classification, plan generation, step execution with Unix tools, reflection on failures, experience corpus learning |
| `run_reasoning_benchmark.sh` | Benchmark runner: tests multiple Ollama models, saves structured logs, produces comparison tables |

### Quick Start

```bash
cd demos

# Run the demo with the default model
python reasoning_v30_demo.py

# Run benchmarks across all available models
./run_reasoning_benchmark.sh --all

# Compare results
./run_reasoning_benchmark.sh --compare

# Benchmark a specific model
./run_reasoning_benchmark.sh -m mistral:7b-instruct
```

### What It Exercises

1. **Task classification** — BYPASS / SIMPLE / MODERATE / COMPLEX
2. **Plan generation** — structured step plans with confidence scoring
3. **Step execution** — Unix tool calls in sandboxed shell
4. **Reflection** — failure analysis and plan adjustment
5. **Experience corpus** — hybrid learning from session traces

### Test Scripts

Focused test scripts for individual reasoning stages:

| Script | Tests |
|--------|-------|
| `test_classification.py` | Task complexity classification accuracy |
| `test_simple_execution.py` | End-to-end SIMPLE task flow (classify → execute) |
| `test_moderate_execution.py` | End-to-end MODERATE task flow (classify → plan → execute → verify) |

```bash
# Run individual tests
python test_classification.py
python test_simple_execution.py
python test_moderate_execution.py
```

---

## Comparison of Demos

```
                         ┌───────────────────────────────────────────┐
                         │             RAGIX Demos                   │
                         └──────────────────┬────────────────────────┘
                                            │
     ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
     │                  │                   │                   │                  │
     ▼                  ▼                   ▼                   ▼                  │
┌──────────┐   ┌────────────────┐   ┌──────────────┐   ┌──────────────┐          │
│Docs Audit│   │MCP Interactive │   │ Memory Pipe  │   │Reasoning v30 │          │
│          │   │                │   │              │   │              │          │
│Activity  │   │ Web UI         │   │ Ingest+Recall│   │Graph pipeline│          │
│Broker+ACL│   │ 16 KOAS tools  │   │ FTS5/BM25   │   │Multi-model   │          │
│2 modes   │   │ Scenarios      │   │ No LLM      │   │Benchmarks    │          │
│          │   │ WebSocket      │   │ Composable   │   │              │          │
│ragix_    │   │ ragix_kernels  │   │ ragix_core   │   │ragix_core    │          │
│ kernels  │   │  + MCP server  │   │  memory      │   │ reasoning_v30│          │
└────┬─────┘   └───────┬────────┘   └──────┬───────┘   └──────┬───────┘          │
     │                 │                    │                  │                  │
     ▼                 ▼                    ▼                  ▼                  │
 Sovereignty     Tool orchestration   Data retrieval     LLM reasoning           │
 & compliance      & integration      & injection       & benchmarking           │
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: ragix_core` | RAGIX not installed | `pip install -e .` from RAGIX root |
| "Ollama not responding" | Ollama not running | `ollama serve` |
| "No models available" | Models not pulled | `ollama pull mistral:7b-instruct` |
| Port 8080 in use | Another service on same port | Use `--port 9000` or stop conflicting service |
| "workspace not initialized" | setup.sh not run | `cd koas_docs_audit && ./setup.sh` |
| "401 Unauthorized" | Invalid API key | Check `.demo_keys` and `config/acl.yaml` |
| Import errors in test scripts | Wrong Python environment | Activate ragix-env: `conda activate ragix-env` |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [docs/KOAS.md](../docs/KOAS.md) | KOAS architecture and philosophy (5 families, 75 kernels) |
| [docs/KOAS_ACTIVITY.md](../docs/KOAS_ACTIVITY.md) | Activity logging reference (event schema, actor model) |
| [docs/KOAS_DOCS.md](../docs/KOAS_DOCS.md) | Document summarization kernels (17 kernels) |
| [docs/KOAS_MEMORY_MCP.md](../docs/KOAS_MEMORY_MCP.md) | Memory MCP server (17 tools) |
| [docs/developer/ROADMAP_MEMORY_PIPE.md](../docs/developer/ROADMAP_MEMORY_PIPE.md) | Memory pipe implementation roadmap (v3.0) |
| [docs/REASONING.md](../docs/REASONING.md) | Reasoning engines deep dive |
| [docs/MCP.md](../docs/MCP.md) | MCP protocol and tool reference |
| [ragix_kernels/README.md](../ragix_kernels/README.md) | Kernel developer reference (all 75 kernels) |
