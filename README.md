<p align="center">
  <img src="assets/ragix-logo.png" alt="RAGIX Logo" height="128"><br>
</p>

# RAGIX v0.69.0

*(Retrieval-Augmented Generative Interactive eXecution Agent)*

**A Sovereign Development Platform for AI-Assisted Software Engineering**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![DeepWiki](https://img.shields.io/badge/Docs-DeepWiki-purple.svg)](https://deepwiki.com/ovitrac/RAGIX)

---

**Version:** 0.69.0 | **Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Updated:** 2026-02-19 | **Codebase:** 500K+ LOC analyzed in production

---

## Why RAGIX?

**RAGIX** is a **sovereign development platform** for building AI-assisted tools that transforms local LLMs into disciplined software engineering assistants. Unlike cloud-based AI coding tools, RAGIX provides:

### Sovereignty by Design

- **100% local execution** — All LLM inference runs via Ollama on `localhost:11434`; zero data exfiltration, air-gapped mode available
- **Model-agnostic** — Works with Ollama (Mistral, Granite, Qwen, DeepSeek-R1) or commercial APIs with explicit opt-in
- **Sovereignty attestation** — Every operation records `sovereignty.local_only: true`, verifiable per-event
- **Auditable operations** — SHA256 hash chain on command logs, Merkle roots for pyramidal provenance

### Production-Ready Architecture: KOAS

**KOAS** (Kernel-Orchestrated Audit System) is the computational core — **87 deterministic kernels** across 6 families:

| Family | Kernels | Purpose |
|--------|---------|---------|
| **audit** | 27 | Code quality: AST, complexity, coupling, risk matrices, CVE scanning |
| **docs** | 17 | Document analysis: hierarchical summarization, clustering, discrepancy detection |
| **presenter** | 8 | Slide generation: MARP decks from document corpora (full/compressed/executive) |
| **reviewer** | 13 | Traceable Markdown review: chunk-level edits with selective revert |
| **security** | 10 | Infrastructure: network discovery, vulnerability assessment, compliance |
| **summary** | 12 | Multi-document summarization: Graph-RAG, secrecy tiers, drift detection |

**Architectural guarantee:** Kernels compute deterministically — LLMs are used only for planning and reasoning, never for metrics. No hallucinated numbers.

### Multi-Model Reasoning

- **Planner-Worker-Verifier** — Tiered model selection: reasoning models (7B+) for planning, fast models (3B) for execution, validation models for verification
- **Worker + Tutor** — Two-stage LLM pattern in KOAS: small model drafts, larger model refines
- **4 reasoning engines** — ReasoningLoop, ReasoningGraph (v30), ContractiveReasoner, Interpreter-Tutor (PCG)

### Centralized Activity Logging

- **Append-only JSONL** event stream at `.KOAS/activity/events.jsonl`
- **SHA256 hash chain** for tamper-evidence
- **Sovereignty flag** per event (`sovereignty.local_only: true`)
- **Actor tracking** — system, operator, external orchestrator, auditor

### Broker Gateway (Optional)

For regulated environments requiring strict access control:

- **Core-Shell architecture** — External orchestrators (Claude, GPT-4) can trigger and monitor audits **without ever seeing document content**
- **ACL with scopes** — `docs.trigger`, `docs.status`, `docs.export_external` (no `docs.export_internal`)
- **Output sanitization** — 4 isolation levels: INTERNAL, EXTERNAL, ORCHESTRATOR, COMPLIANCE

### Enterprise Capabilities

- **Code audit & metrics** — AST analysis, complexity hotspots, tech debt estimation, risk matrices
- **Security scanning** — Network discovery, vulnerability assessment, compliance checking (ANSSI/NIST/CIS)
- **Multiple interfaces** — Web UI, CLI, REST API, MCP server, WebSocket streaming

### Episodic Memory System

Policy-driven memory with multi-tier promotion, bounded loops, and MCP exposure:

- **FTS5 + BM25 retrieval** — SQLite-backed full-text search with relevance scoring
- **Bounded recall-answer loop** — Iterative LLM refinement with 5 stopping conditions (llm_stop, fixed_point, query_cycle, no_new_items, max_calls)
- **Fixed-point convergence** — Lexical similarity detection (Jaccard + SequenceMatcher) stops when answers stabilize
- **Q\*-style search** — Iterative deepening with relevance feedback
- **Memory Palace** — Spatial metaphor for memory organization (rooms, loci, guided tours)
- **STM→MTM→LTM promotion** — Access frequency and recency-based tier advancement
- **SHA-256 corpus dedup** — Prevents duplicate storage
- **17 MCP tools** — Full memory operations exposed to Claude Desktop / Claude Code
- **Pipe + Loop** — `ragix-memory pipe "question" | ragix-memory loop --llm "ollama run model" --trace`

### Development Platform

RAGIX is not just a tool—it's a platform for building sovereign AI systems:
- **Extensible tooling** — Add custom MCP tools, reasoning engines, or domain-specific kernels
- **Collective intelligence** — Multiple RAGIX instances can collaborate on complex tasks
- **Local-first research** — Experiment with 200B+ parameter models on local hardware (GB10-ready)

> **Make local LLMs behave like disciplined software engineers.**
> Reproducible, auditable, sovereign—without cloud dependencies.


<video src="https://github.com/user-attachments/assets/601670ed-2fa9-4dc6-a6e2-c23b9cde090b"
       autoplay
       loop
       muted
       playsinline
       controls
       width="960">
</video>

▶️ <strong>Watch the full audit demo on YouTube:</strong>
<a href="https://youtu.be/vDHI70ZPnDE">https://youtu.be/vDHI70ZPnDE</a>


---

## Quick Start

### Option A: One-Click Launcher (Conda + Streamlit GUI)

```bash
git clone https://github.com/ovitrac/RAGIX.git
cd RAGIX
./launch_ragix.sh
```

This script automatically:
- Initializes Conda and creates `ragix-env` environment
- Installs all dependencies + RAGIX package
- Checks Ollama status and available models
- Presents an interactive menu (select **option 7** for `ragix-web`)

### Option B: Manual Setup (venv)

```bash
git clone https://github.com/ovitrac/RAGIX.git
cd RAGIX
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies then RAGIX package (BOTH required!)
pip install -r requirements.txt
pip install -e .   # MUST run after requirements.txt (required to install ragix-web)

# Start the Web UI
ragix-web
```

> **Common Error:** `ModuleNotFoundError: No module named 'ragix_core'`
> **Solution:** Run `pip install -e .` — this installs RAGIX itself, not just dependencies.

### Option C: Conda Environment

```bash
git clone https://github.com/ovitrac/RAGIX.git
cd RAGIX
conda env create -f environment.yaml
conda activate ragix-env
pip install -e .

ragix-web
```
### Troubleshooting

On some OS or machine, they may be some issues or errors.

#### Error 1 :
- OS : MacOS
- Message : `OSError: cannot load library 'libgobject-2.0-0'`
- Solution : 
  - Option 1 : Try to uninstall the lib from local python and use the OS available one `pip uninstall weasyprint -y`
  - Alternative : Try to install the required lib using this command `brew install pango gdk-pixbuf libffi cairo`

---

### Web UI Access

**`ragix-web`** is the most advanced interface with full AI capabilities. It opens **http://localhost:8080** and provides:

- **Dashboard** — Project overview and metrics
- **Chat** — AI-assisted coding with local LLMs
- **Reasoning** — Multi-step workflow visualization
- **AST Analysis** — Code structure and dependencies
- **Project RAG** — Semantic search across your codebase
- **Audit** — Risk matrix, drift tracking, compliance
- **Partitioner** — Visual dependency graphs with propagation algorithms

> **Note:** `ragix-web` creates configuration and cache files in `$HOME/.ragix/` for session persistence.

> **Tip:** For full AI capabilities, start Ollama: `ollama serve && ollama pull mistral`

### Available Launchers

| Launcher | Description |
|----------|-------------|
| `./launch_ragix.sh` | Interactive menu with 7 options (Conda) |
| `./launch_ragix.sh gui` | Direct: Streamlit GUI (port 8501) |
| `./launch_ragix.sh web` | Direct: **RAGIX-WEB** FastAPI (port 8080) |
| `./launch_ragix.sh mcp` | Direct: MCP server |
| `source ragixinit.sh` | Set environment variables (model, sandbox, profile) |

### CLI Entry Points (after `pip install -e .`)

| Command | Description |
|---------|-------------|
| `ragix-web` | **Most advanced UI** — FastAPI (port 8080), creates `~/.ragix/` |
| `ragix` | Core CLI with subcommands |
| `ragix-unix-agent` | Interactive Unix-RAG agent |
| `ragix-ast` | AST analysis (scan, metrics, hotspots) |
| `ragix-koas` | KOAS kernel orchestrator |
| `ragix-index` | Project indexing for RAG |
| `ragix-batch` | Batch processing pipelines |
| `ragix-vault` | Encrypted credential vault |
| `ragix-wasp` | WASP security scanner |
| `ragix-memory` | Episodic memory CLI (search, recall, ingest, pipe, palace) |

### Claude Code Integration

Register MCP servers, safety hooks, and audit logging in one command:

```bash
bash scripts/install_claude.sh            # Memory MCP + 4 hooks
bash scripts/install_claude.sh --profile full  # + main MCP (38 tools)
```

Installs: memory auto-recall on every prompt, dangerous-command safety guard, tool-action audit log, and a test reminder on stop. See `RAGIX_COMPONENTS.md` for the full component reference.

> **Without install:** `PYTHONPATH=. python ragix_web/server.py`

### Returning Users: Reactivate Your Environment

If you installed RAGIX before and want to use it again, you need to **reactivate** the environment first:

| Install Method | Environment Name | Location | Reactivation Command |
|----------------|------------------|----------|----------------------|
| Option A (Conda launcher) | `ragix-env` | `~/anaconda3/envs/ragix-env/` | `conda activate ragix-env` |
| Option B (venv) | `.venv` | `./RAGIX/.venv/` | `cd RAGIX && source .venv/bin/activate` |
| Option C (Conda manual) | `ragix-env` | `~/anaconda3/envs/ragix-env/` | `conda activate ragix-env` |

**Quick reference:**

```bash
# If you used venv (Option B) — must be in RAGIX folder
cd /path/to/RAGIX
source .venv/bin/activate
ragix-web

# If you used Conda (Option A or C) — works from anywhere
conda activate ragix-env
ragix-web
```

> **How to know if environment is active?**
> - Your terminal prompt shows `(.venv)` or `(ragix-env)` at the beginning
> - Run `which python` — should point to the environment, not system Python

---

## Platform Capabilities

### 🖥️ **Four Access Modes**

| Interface | Use Case | Features |
|-----------|----------|----------|
| **Web UI** | Interactive development | 7 specialized tabs, real-time streaming, visualizations |
| **CLI** | Automation & scripting | 15+ commands, batch processing, CI/CD integration |
| **MCP Server** | Claude Desktop | Native integration, tool schemas, bidirectional |
| **REST API** | Custom integrations | FastAPI, WebSocket, OpenAPI docs |

### 🔍 **Dual RAG Architecture**

| RAG Type | Scope | Storage | Use Case |
|----------|-------|---------|----------|
| **Unix-RAG** | Shell-native | In-memory | Real-time code exploration with `grep`, `find`, `awk` |
| **Project RAG** | Per-project | ChromaDB | Semantic search, concept discovery, knowledge graphs |

Both support **hybrid search** (BM25 + vector) with multiple fusion strategies: RRF, Weighted, Interleave, Rerank.

### 🌳 **AST Analysis & Code Metrics**

Professional-grade static analysis for **Java** and **Python**:

```bash
ragix-ast scan ./src --lang java          # Extract 18K+ symbols
ragix-ast metrics ./src                   # Complexity, debt, maintainability
ragix-ast hotspots ./src --top 20         # Find complexity hotspots
ragix-ast graph ./src --output deps.html  # Interactive dependency graph
ragix-ast matrix ./src --level package    # DSM with cycle detection
```

**Production-tested:** 1,315 Java files, 18,210 symbols, 45,113 dependencies, 362h tech debt estimated in ~10 seconds.

### 📊 **Code Audit & Governance**

| Feature | Description |
|---------|-------------|
| **Risk Matrix** | Visual risk assessment with configurable weights |
| **Drift Tracking** | Monitor code changes over time |
| **Dead Code Detection** | Find isolated classes (no callers AND no callees) |
| **Service Detection** | Enterprise patterns (MSG-HUB, ACME-ERP) |
| **Compliance Reports** | PDF/HTML audit reports |

### 🧩 **Codebase Partitioner**

Visual tool for understanding and decomposing large codebases:

- **Graph Propagation Algorithm** — Multi-phase directional classification
- **Dependency Visualization** — Force-directed graphs with D3.js
- **MDS Layout** — Eigendecomposition for optimal positioning
- **Export** — JSON, CSV, SVG, PNG for documentation

### 🤖 **Multi-Agent Workflows**

Pre-built templates for common tasks:

| Template | Steps | Use Case |
|----------|-------|----------|
| `bug_fix` | Locate → Diagnose → Fix → Test | Bug resolution |
| `feature_addition` | Design → Implement → Test → Document | New features |
| `code_review` | Quality + Security review | PR review |
| `refactoring` | Analyze → Plan → Refactor → Verify | Tech debt |
| `security_audit` | Static analysis + Dependency checks | Security |

### 🔒 **Safety & Sovereignty**

| Feature | Description |
|---------|-------------|
| **Three Profiles** | `safe-read-only`, `dev`, `unsafe` |
| **Dangerous Pattern Filtering** | Blocks `rm -rf`, `dd`, `mkfs`, etc. |
| **Git Protection** | Prevents destructive operations |
| **Audit Trail** | SHA256 hash chain for all operations |
| **Air-Gapped Mode** | Enforces local-only processing |

---

## The RAGIX Ecosystem

RAGIX is part of a suite of sovereign AI tools:

### 🔐 **CloakMCP** — Secret Sanitization

Deterministic sanitizer for secrets before LLM exposure.

- **Reversible redaction** with encrypted vaults
- **Policy-driven detection** (regex, entropy, JWT, SSH keys)
- **VS Code integration** and CLI tools
- **Repository:** [github.com/ovitrac/CloakMCP](https://github.com/ovitrac/CloakMCP)

### 📚 **RAGGAE** — Enterprise Document RAG

Production-ready RAG for document processing (tenders, CVs, reports).

- **Hybrid retrieval** (FAISS + BM25)
- **NLI compliance checking** via Ollama
- **Multi-format** (PDF, DOCX, ODT, Markdown)
- **Full traceability** (document → page → block → bounding box)
- **Repository:** [github.com/ovitrac/RAGGAE](https://github.com/ovitrac/RAGGAE)
- **Website:** [raggae.pages.dev](https://raggae.pages.dev/)

---

## Documentation

### 📖 **Documentation Index**

**[docs/INDEX.md](docs/INDEX.md)** — Complete documentation navigation hub with reading guides by topic and goal.

### 📖 **DeepWiki** (AI-Generated)

**[deepwiki.com/ovitrac/RAGIX](https://deepwiki.com/ovitrac/RAGIX)** — AI-generated documentation covering:

- Six-tier architecture (Presentation → Storage)
- Agent orchestration and reasoning graphs
- LLM integration patterns
- RAG system internals
- Security model

> **Note:** DeepWiki is generated by Devin and refreshes every ~5 days. This repository is typically ahead with latest features.

### 📁 **Core Documentation**

| Document | Description |
|----------|-------------|
| [docs/INDEX.md](docs/INDEX.md) | **Documentation navigation hub** |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |
| [docs/SOVEREIGN_LLM_OPERATIONS.md](docs/SOVEREIGN_LLM_OPERATIONS.md) | Sovereignty and confidential AI operations |
| [docs/MCP.md](docs/MCP.md) | Model Context Protocol in RAGIX |
| [docs/REASONING.md](docs/REASONING.md) | Reasoning engines (ContractiveReasoner, v30, Interpreter-Tutor) |

### 📁 **KOAS Kernel Documentation**

| Document | Description |
|----------|-------------|
| [docs/KOAS.md](docs/KOAS.md) | KOAS philosophy and architecture (6 families, 87 kernels) |
| [docs/KOAS_DOCS.md](docs/KOAS_DOCS.md) | Document summarization system (17 kernels) |
| [docs/KOAS_SUMMARY.md](docs/KOAS_SUMMARY.md) | Multi-document summarization with Graph-RAG (12 kernels) |
| [docs/KOAS_PRESENTER.md](docs/KOAS_PRESENTER.md) | Slide deck generation from documents (8 kernels) |
| [docs/KOAS_REVIEW.md](docs/KOAS_REVIEW.md) | Traceable Markdown review (13 kernels) |
| [docs/KOAS_ACTIVITY.md](docs/KOAS_ACTIVITY.md) | Centralized activity logging and governance |
| [docs/KOAS_MEMORY_MCP.md](docs/KOAS_MEMORY_MCP.md) | Episodic memory MCP tools (17 tools) |
| [docs/KOAS_MEMORY_ARCHITECTURE.md](docs/KOAS_MEMORY_ARCHITECTURE.md) | Memory architecture and design |
| [docs/KOAS_MCP_REFERENCE.md](docs/KOAS_MCP_REFERENCE.md) | KOAS MCP tool reference |
| [ragix_kernels/README.md](ragix_kernels/README.md) | Kernel developer reference (v1.4.0) |

### 📁 **Guides & References**

| Document | Description |
|----------|-------------|
| [QUICKSTART_CLAUDE_CODE.md](QUICKSTART_CLAUDE_CODE.md) | Getting started guide |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | REST API documentation |
| [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) | Command-line interface |
| [docs/AST_GUIDE.md](docs/AST_GUIDE.md) | AST analysis guide |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Architecture

```mermaid
flowchart TB
    subgraph "User Interfaces"
        WEB["🌐 Web UI (ragix-web)"]
        CLI["🖥️ CLI Tools"]
        MCP["🔌 MCP Server"]
        API["📡 REST API"]
    end

    subgraph "Orchestration"
        ORCH["🧠 Graph Executor"]
        REASON["🔄 Reasoning Loop"]
        TEMPLATES["📋 Workflow Templates"]
    end

    subgraph "Analysis & Audit"
        AST["🌳 AST Parser"]
        METRICS["📈 Code Metrics"]
        AUDIT["📊 Risk Matrix"]
        PARTITION["🧩 Partitioner"]
    end

    subgraph "Retrieval"
        UNIX_RAG["🔧 Unix-RAG"]
        PROJ_RAG["📚 Project RAG"]
        HYBRID["🔍 Hybrid Search"]
    end

    subgraph "Execution"
        SHELL["🐚 Sandboxed Shell"]
        LLM["🤖 LLM Backends"]
        SAFETY["🔒 Safety Policies"]
    end

    WEB --> ORCH
    CLI --> ORCH
    MCP --> ORCH
    API --> ORCH

    ORCH --> REASON
    ORCH --> TEMPLATES

    REASON --> AST
    REASON --> UNIX_RAG
    REASON --> PROJ_RAG

    AST --> METRICS
    AST --> AUDIT
    AST --> PARTITION

    UNIX_RAG --> SHELL
    PROJ_RAG --> HYBRID

    SHELL --> SAFETY
    REASON --> LLM
    LLM --> SAFETY
```

---

## Installation

### Option A: Quick Install

```bash
git clone https://github.com/ovitrac/RAGIX.git
cd RAGIX
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: With Ollama (Recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull mistral        # Fast, general purpose
ollama pull qwen2.5:14b    # Better reasoning
ollama pull deepseek-coder # Code-specialized

# Start RAGIX
python ragix_web/server.py
```

### Option C: Using Launcher

```bash
./start_ragix.sh          # Interactive launcher
./start_ragix.sh --web    # Direct web server start
./start_ragix.sh --cli    # CLI mode
```

---

## LLM Backend Configuration

RAGIX supports multiple LLM backends with a sovereignty-first model:

| Backend | Type | Configuration |
|---------|------|---------------|
| **Ollama** | Local | Default, no API key needed |
| **Claude** | Cloud | `ANTHROPIC_API_KEY` env var |
| **OpenAI** | Cloud | `OPENAI_API_KEY` env var |

Configure in `ragix.yaml` or via Web UI Settings:

```yaml
llm:
  backend: ollama
  model: mistral
  temperature: 0.7

sovereignty:
  enforce_local: true  # Block cloud backends
```

---

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/sessions` | POST | Create session |
| `/api/threads/{id}/messages` | POST | Send message |
| `/api/ast/scan` | POST | Scan codebase |
| `/api/ast/metrics` | GET | Get metrics |
| `/api/ast/partition` | POST | Run partitioner |
| `/api/rag/search` | POST | Search documents |
| `/api/rag/index` | POST | Index files |

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8421/ws/chat');
ws.send(JSON.stringify({ message: "Analyze this code", session_id: "..." }));
```

Full API documentation: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

---

## Tools & Applications

### 🚀 **Main Applications**

| Application | Entry Point | Description |
|-------------|-------------|-------------|
| **Web UI** | `python ragix_web/server.py` | Full-featured dashboard (8 tabs) |
| **Unix-RAG Agent** | `python unix-rag-agent.py` | Interactive shell-based assistant |
| **MCP Server** | `python MCP/ragix_mcp_server.py` | Claude Desktop integration |
| **Radial Server** | `python ragix_unix/radial_server.py` | Live dependency explorer |

### 🛠️ **CLI Tools**

| Tool | Command | Description |
|------|---------|-------------|
| **AST Analysis** | `ragix-ast scan/metrics/graph/matrix/radial` | Code analysis & visualization |
| **Project Index** | `ragix-unix index_cli.py` | Index codebase for RAG |
| **Batch Processing** | `ragix-unix batch_cli.py` | Bulk file operations |
| **Vault Manager** | `ragix-unix vault_cli.py` | Secure credential storage |
| **WASP Tools** | `ragix-unix wasp_cli.py` | Workflow automation |
| **Core CLI** | `ragix_core/cli.py` | Main orchestrator CLI |

### 📡 **API Routers** (REST + WebSocket)

| Router | Endpoints | Purpose |
|--------|-----------|---------|
| `sessions.py` | `/api/sessions/*` | Session management |
| `threads.py` | `/api/threads/*` | Conversation threads |
| `reasoning.py` | `/api/reasoning/*` | Multi-step workflows |
| `rag.py` | `/api/rag/*` | Document RAG |
| `rag_project.py` | `/api/rag/project/*` | Project-level RAG |
| `audit.py` | `/api/audit/*` | Code audit & partitioner |
| `agents.py` | `/api/agents/*` | Agent configuration |
| `memory.py` | `/api/memory/*` | Episodic memory |
| `context.py` | `/api/context/*` | Global context |
| `logs.py` | `/api/logs/*` | Operation logs |

### 🔧 **Shell Scripts**

| Script | Purpose |
|--------|---------|
| `launch_ragix.sh` | Interactive launcher with menu |
| `ragixinit.sh` | Environment initialization |
| `release.sh` | Release automation |
| `rt.sh`, `rt-find.sh`, `rt-grep.sh` | Testing utilities |

### 🧪 **Test Suites**

| Test File | Coverage |
|-----------|----------|
| `test_ast.py` | AST parsing & metrics |
| `test_reasoning_graph.py` | Multi-step reasoning |
| `test_graph_executor.py` | Workflow execution |
| `test_hybrid_search.py` | BM25 + Vector search |
| `test_bm25_index.py` | Keyword indexing |
| `test_workflow_templates.py` | Pre-built workflows |
| `test_wasp_*.py` | WASP integration |
| `tests/audit/*` | Audit & partitioner |

### 📚 **Examples**

| Example | Description |
|---------|-------------|
| `claude_demo.py` | Claude API integration |
| `hybrid_search_example.py` | Hybrid RAG demo |
| `resilience_example.py` | Retry & circuit breaker |
| `workflow_example.py` | Multi-agent workflow |
| `test_llm_backends.sh` | Backend comparison |

### 🔒 **Security Examples** (`examples/security/`)

| Example | Description |
|---------|-------------|
| `local_network/` | Network discovery & port scanning |
| `web_audit/` | Web application security audit |
| `compliance_check/` | ANSSI/NIST/CIS compliance |
| `config_audit/` | Firewall configuration analysis |
| `run_security_demo.sh` | Interactive security demo |

### 📊 **Audit Examples** (`examples/audit/`)

| Example | Description |
|---------|-------------|
| `volumetry_analysis/` | Risk weighted by traffic (ACME-ERP 4M msg/day) |
| `microservices/` | Service catalog & dependency analysis |
| `java_monolith/` | Complexity hotspots & refactoring |
| `full_audit/` | Comprehensive system audit |
| `run_audit_demo.sh` | Interactive audit demo |

Full CLI reference: [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md)

---

## Project Structure

```
RAGIX/
├── ragix_core/          # Core platform (150+ modules, ~60K LOC)
│   ├── agents/          # Agent implementations (code, doc, git, test)
│   ├── memory/          # Episodic memory (12 core + 9 MCP + 15 tests)
│   │   ├── mcp/         # MCP tools (17 tools, workspace router, metrics)
│   │   ├── tests/       # 511 tests across 15 files
│   │   └── skills/      # Memory skills (4 files)
│   ├── shared/          # Shared utilities (gpu_detect, text_utils)
│   ├── reasoning_slim/  # ContractiveReasoner (tree-based, entropy)
│   ├── reasoning_v30/   # ReasoningGraph (graph state machine)
│   ├── reasoning_tutor/ # Interpreter-Tutor (game-theoretic PCG)
│   ├── rag_project/     # Project RAG (ChromaDB, knowledge graph)
│   ├── ast_*.py         # AST analysis (Java, Python)
│   └── llm_backends.py  # Sovereign/Cloud LLM abstraction
├── ragix_kernels/       # KOAS kernels (160+ files, 87 kernels)
│   ├── audit/           # Code audit (27 kernels)
│   ├── docs/            # Document analysis (17 kernels)
│   ├── presenter/       # Slide generation (8 kernels)
│   ├── reviewer/        # Markdown review (13 kernels)
│   ├── security/        # Security scanning (10 kernels)
│   ├── summary/         # Multi-document summarization (12 kernels)
│   ├── shared/          # Shared kernel utilities (md_renumber, md_toc)
│   ├── activity.py      # Centralized activity logging
│   ├── merkle.py        # SHA256 Merkle provenance
│   └── orchestrator.py  # Kernel execution with hash chain
├── ragix_web/           # Web application
│   ├── server.py        # FastAPI server (port 8080)
│   ├── routers/         # API routes (10 routers)
│   └── static/          # Frontend (D3.js visualizations)
├── ragix_unix/          # Unix-RAG agent
├── MCP/                 # MCP integration (55 tools)
├── demos/               # Demos (koas_pipe_demo, koas_docs_audit)
├── docs/                # Documentation (see Index)
├── tests/               # Test suite
└── examples/            # Usage examples (audit, security)
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/audit/ -v
pytest tests/ast/ -v

# With coverage
pytest tests/ --cov=ragix_core --cov-report=html
```

---

## What's New

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

**Latest: v0.71.0** (2026-03-03)
- **Presenter v2.1**: 24-transform MARP post-processing pipeline (3,149 LOC), layout directives, hand-crafted presentation workflow, typography refinement
- **Maven graph fix**: staircase layout prevents label overlap in dense dependency layers
- **KOAS_PRESENTER docs**: updated to v2.1.0 (1,198 lines) with full post-processing reference

**Recent highlights:**
- v0.70.0: Claude Code integration — installer, safety hooks, audit logging, component manifest
- v0.69.0: Bounded recall-answer loop, fixed-point convergence, text similarity
- v0.67.0: Episodic memory (17 MCP tools), multi-document summarization (12 kernels), Graph-RAG
- v0.66.0: Centralized activity logging, broker gateway with ACL scopes

---

## Roadmap

| Priority | Feature | Status |
|----------|---------|--------|
| ✅ | Multi-agent workflows | Complete |
| ✅ | Hybrid RAG (BM25 + Vector) | Complete |
| ✅ | AST analysis (Java/Python) | Complete |
| ✅ | Code audit (27 kernels) & partitioning | Complete |
| ✅ | Document analysis (17 kernels) | Complete |
| ✅ | Security scanning (10 kernels) | Complete |
| ✅ | Traceable Markdown review (13 kernels) | Complete |
| ✅ | Slide deck generation (8 kernels) | Complete |
| ✅ | MCP integration (55 tools) | Complete |
| ✅ | Centralized activity logging | Complete |
| ✅ | Broker Gateway with ACL | Complete |
| ✅ | Episodic memory (17 MCP tools, Memory Palace) | Complete |
| ✅ | Multi-document summarization (12 kernels) | Complete |
| ✅ | Graph-RAG (entity extraction, community detection) | Complete |
| ✅ | Memory Pipe demo for Claude integration | Complete |
| ✅ | KOAS Presenter v2.1 (24 transforms, layout directives, HTML export) | Complete |
| 🔄 | KOAS Presenter LLM normalization | Phase 1 done (deterministic) |
| 🔄 | Interpreter-Tutor reasoning engine | Research (v0.5.0) |
| 📋 | Multi-language AST (Go, Rust) | Planned |
| 📋 | Editable PPTX export (python-pptx) | Planned |

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Author

**Olivier Vitrac, PhD, HDR**
Head of Innovation Lab, Adservio
olivier.vitrac@adservio.fr

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow existing code style and documentation standards
4. Add tests for new functionality
5. Submit a pull request

---

## Acknowledgments

- **Ollama** — Local LLM serving
- **FastAPI** — Web framework
- **D3.js** — Visualizations
- **ChromaDB** — Vector storage
- **javalang** — Java AST parsing
