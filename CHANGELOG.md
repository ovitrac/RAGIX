# CHANGELOG — RAGIX

All notable changes to the **RAGIX** project will be documented here.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## [Unreleased] — v0.8: WASP Integration

**WASP** = WebAssembly Agentic System Protocol / Pipelines

v0.8 introduces WASM as a **secure, portable execution layer** for RAGIX:
- Deterministic tool execution (no shell unpredictability)
- Stronger sandboxing (WASI capabilities model)
- Cross-platform portability (Linux/macOS/Windows/browser)
- Full auditability (reproducible, logged execution)

**Reference:** `V08_WASP_PLANNING.md` for detailed specifications

### Architecture Evolution

```
v0.7 (Current):
  LLM → RAGIX Orchestrator → ShellSandbox → bash/grep/git

v0.8 (Target):
  LLM → RAGIX Orchestrator → ┬→ ShellSandbox (legacy)
                              └→ WasmSandbox (new) → .wasm modules
```

### Task Breakdown (~7,680 lines estimated)

| Task | Description | Lines | Priority |
|------|-------------|-------|----------|
| 4.1 | Sandbox Abstraction Layer | ~230 | CRITICAL |
| 4.2 | WasmSandbox Implementation | ~700 | CRITICAL |
| 4.3 | WASP Tool Registry | ~450 | HIGH |
| 4.4 | JSON Protocol Extension (`wasp_task`) | ~200 | HIGH |
| 4.5 | Priority WASM Tools | ~1,400 | HIGH |
| 4.6 | Browser WASM Runtime | ~1,100 | MEDIUM |
| 4.7 | Testing & Validation | ~2,000 | HIGH |
| 4.8 | Documentation | ~1,600 | MEDIUM |

### Phase 1: Foundation
- [ ] `ragix_core/sandbox_base.py` — Base sandbox protocol
- [ ] `ragix_core/sandbox_factory.py` — Pluggable sandbox factory
- [ ] `ragix_core/wasm_runtime.py` — wasmtime Python wrapper
- [ ] `ragix_core/wasm_sandbox.py` — WASM-based sandbox
- [ ] `ragix_core/hybrid_sandbox.py` — Shell/WASM routing

### Phase 2: Tools & Protocol
- [ ] `ragix_core/wasp_manifest.py` — Tool manifest schema
- [ ] `ragix_core/wasp_registry.py` — Tool registry
- [ ] `ragix_unix/wasp_cli.py` — CLI (`ragix-wasp list|info|install|run`)
- [ ] JSON protocol: `wasp_task` action support
- [ ] **validate.wasm** — JSON/YAML schema validator
- [ ] **mdparse.wasm** — Markdown parser/AST
- [ ] **rg.wasm** — ripgrep wrapper

### Phase 3: Browser Runtime
- [ ] `ragix_web/static/wasm_runtime.js` — Browser WASI runtime
- [ ] `ragix_web/static/virtual_fs.js` — In-memory filesystem
- [ ] `ragix_web/static/browser_tools.js` — Client-side tool execution
- [ ] File System Access API integration
- [ ] Offline-capable RAGIX

### Phase 4: Quality & Documentation
- [ ] Unit tests for WASM components (~800 lines)
- [ ] Integration tests (~500 lines)
- [ ] Performance benchmarks (WASM vs shell)
- [ ] `docs/WASP_GUIDE.md` — Architecture guide
- [ ] `docs/WASP_TOOL_DEV.md` — Tool development (Rust)
- [ ] `docs/BROWSER_RAGIX.md` — Browser setup
- [ ] `docs/MIGRATION_V07_V08.md` — Upgrade guide

### New Dependencies

```toml
# Python (optional)
[project.optional-dependencies]
wasm = ["wasmtime>=14.0.0"]

# Rust (wasp_tools/Cargo.toml)
[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1.0", features = ["derive"] }
pulldown-cmark = "0.9"
```

### Success Criteria
- [ ] WasmSandbox executes all priority tools
- [ ] Browser RAGIX runs without Python server
- [ ] Hybrid sandbox seamlessly falls back
- [ ] All WASM tools have >90% test coverage
- [ ] Performance within 2x of native shell

---

## [Future] — v0.9+ Ideas

### Agent Improvements
- [ ] Autonomous multi-step reasoning with self-correction
- [ ] Memory and context persistence across sessions
- [ ] Agent specialization profiles (security, performance, refactoring)
- [ ] Inter-agent communication protocol

### Search & Retrieval
- [ ] Incremental index updates (watch mode)
- [ ] Cross-repository search federation
- [ ] AST-aware code search (tree-sitter.wasm)
- [ ] Natural language to code search

### Integration
- [ ] VS Code extension
- [ ] GitHub Actions integration
- [ ] GitLab CI/CD integration
- [ ] Jupyter notebook support

### Performance
- [ ] GPU acceleration for embeddings (CUDA/MPS)
- [ ] Distributed index sharding
- [ ] Response streaming for large outputs
- [ ] Persistent connection pooling for Ollama

### Security & Compliance
- [ ] Audit log export (JSON, CSV)
- [ ] Role-based access control for tools
- [ ] Secrets scanning integration
- [ ] SBOM generation for analyzed repos

---

## v0.7.0 — Launcher, Web GUI & Multi-Agent Platform (2025-11-25)

### Highlights

**RAGIX evolves from a CLI tool to a complete multi-agent orchestration platform.**

| Metric | Value |
|--------|-------|
| New code | ~10,000+ lines |
| New modules | 12 |
| Workflow templates | 8 |
| LLM backends | 3 |

### New Features

#### Launcher & Environment (`launch_ragix.sh`)
- **Portable conda initialization** — searches `~/anaconda3`, `~/miniconda3`, `~/miniforge3`
- **Auto-environment creation** — creates `ragix-env` if missing
- **Dependency management** — installs from `environment.yaml` and `requirements.txt`
- **Ollama health check** — verifies status and lists available models with sizes
- **Interactive menu** — 6 options: GUI, Demo, MCP, Test, Shell, Status
- **Direct launch modes** — `./launch_ragix.sh gui|demo|mcp|test`

#### Web Interface (`ragix_app.py`)
- **Dashboard** — sovereignty status, model inventory, quick actions
- **Hybrid Search** — BM25 + Vector search with fusion strategy selector
- **LLM Chat** — direct conversation with local Ollama models
- **Workflow Browser** — view and launch 8 pre-built templates
- **System Monitor** — health checks, environment info, refresh controls
- **About Page** — architecture diagram, documentation links

#### LLM Backends (`ragix_core/llm_backends.py`)
- **SovereigntyStatus enum** — `SOVEREIGN`, `CLOUD`, `HYBRID`
- **OllamaLLM** — 🟢 100% local, no data leaves machine
- **ClaudeLLM** — 🔴 Anthropic API (optional, with sovereignty warnings)
- **OpenAILLM** — 🔴 OpenAI API (optional, with sovereignty warnings)
- **Factory functions** — `create_llm_backend()`, `get_backend_from_env()`
- **Automatic warnings** — logs sovereignty status on initialization

#### Real Integration Testing (`examples/test_llm_backends.sh`)
- **Actual Ollama calls** — not mocked, real API requests
- **Model comparison** — mistral vs granite3.1-moe speed benchmark
- **Response timing** — average response time per model
- **Speed ranking** — automated fastest-to-slowest ranking

### Configuration Files

| File | Purpose |
|------|---------|
| `environment.yaml` | Conda environment (Python 3.10-3.12, numpy, scipy) |
| `requirements.txt` | Full v0.7 dependencies (15+ packages) |
| `launch_ragix.sh` | One-command setup and launch |
| `ragix_app.py` | Streamlit web interface |

### Documentation Updates
- **README.md** — Added "Option A: Using the Launcher" installation
- **README.md** — Updated Quick Start with Web UI instructions
- **examples/README.md** — Added launcher quick start and web interface docs

---

## v0.6.0 — Production Monitoring & Resilience (2025-11-24)

### New Features

#### Monitoring (`ragix_core/monitoring.py`)
- **MetricsCollector** — counters, gauges, histograms, timers
- **HealthChecker** — pluggable health checks with status aggregation
- **AgentMonitor** — execution tracking, tool call statistics
- **RateLimiter** — token bucket algorithm for API protection
- **CircuitBreaker** — failure protection with recovery timeout
- **Built-in checks** — `check_ollama_health()`, `check_disk_space()`, `check_memory_usage()`

#### Resilience Patterns (`ragix_core/resilience.py`)
- **RetryConfig** — configurable retry with 4 backoff strategies
  - `CONSTANT`, `LINEAR`, `EXPONENTIAL`, `EXPONENTIAL_JITTER`
- **@retry / @retry_async** — decorators for automatic retry
- **FallbackChain** — ordered fallback execution
- **Timeout** — async timeout wrapper with cancellation
- **Bulkhead** — concurrency limiting (semaphore-based)
- **GracefulDegradation** — automatic fallback on failure

#### Caching (`ragix_core/caching.py`)
- **InMemoryCache** — LRU eviction with TTL support
- **DiskCache** — persistent JSON-based caching
- **LLMCache** — specialized for LLM responses with semantic keys
- **ToolResultCache** — caches deterministic tool outputs
- **Statistics** — hit rate, miss rate, eviction counts

### Integration
- All monitoring integrated into `GraphExecutor`
- Health checks available via MCP (`ragix_health` tool)
- Metrics exposed for external monitoring systems

---

## v0.5.0 — Core Orchestrator & Modular Tooling (2025-11-23)

### Highlights

**Major architectural refactoring: monolithic agent → modular ragix_core package.**

### New Package: `ragix_core/`

#### Agent System (`ragix_core/agents/`)
- **BaseAgent** — abstract base with capabilities enum
- **CodeAgent** — code analysis, editing, search
- **DocAgent** — documentation generation
- **GitAgent** — version control operations
- **TestAgent** — test execution and coverage
- **AgentCapability** — 12 capability types

#### Graph Execution (`ragix_core/agent_graph.py`, `graph_executor.py`)
- **AgentNode** — node with config, capabilities, status
- **AgentEdge** — transitions with conditions
- **AgentGraph** — DAG with validation
- **GraphExecutor** — async execution with dependency resolution
- **SyncGraphExecutor** — synchronous wrapper
- **StreamEvent** — real-time execution events

#### Workflow Templates (`ragix_core/workflow_templates.py`)
- **TemplateManager** — template registry and instantiation
- **8 built-in templates:**
  - `bug_fix` — locate, diagnose, fix, test
  - `feature_addition` — design, implement, test, document
  - `code_review` — quality and security review
  - `refactoring` — analyze, plan, refactor, verify
  - `documentation` — code analysis, doc generation
  - `security_audit` — SAST, dependency checks
  - `test_coverage` — coverage analysis, test generation
  - `exploration` — codebase mapping and analysis

#### Hybrid Search (`ragix_core/hybrid_search.py`, `bm25_index.py`)
- **BM25Index** — sparse keyword search with code tokenization
- **HybridSearchEngine** — BM25 + vector fusion
- **FusionStrategy** — 5 strategies:
  - `RRF` (Reciprocal Rank Fusion)
  - `WEIGHTED`
  - `INTERLEAVE`
  - `BM25_ONLY`
  - `VECTOR_ONLY`
- **Code-aware tokenization** — handles camelCase, snake_case, PascalCase

#### Embeddings & Vector Search
- **EmbeddingBackend** — abstract interface
- **SentenceTransformerBackend** — all-MiniLM-L6-v2 default
- **DummyEmbeddingBackend** — testing without ML deps
- **VectorIndex** — NumPy and FAISS implementations
- **Chunking** — Python, Markdown, Generic chunkers

#### Tool Infrastructure
- **ToolRegistry** — centralized tool management
- **ToolDefinition** — schema with permissions
- **ToolExecutor** — safe execution with logging
- **LLMAgentExecutor** — full agent loop with tool calling

#### Prompt Engineering (`ragix_core/prompt_templates.py`)
- **TaskType enum** — 10 task types
- **PromptTemplate** — structured templates with few-shot examples
- **detect_task_type()** — automatic task classification
- **build_prompt()** — context-aware prompt construction

### Existing Improvements
- **ShellSandbox** — enhanced command filtering
- **AgentLogger** — structured logging with levels
- **Profiles** — `safe-read-only`, `dev`, `unsafe` modes
- **Secrets vault** — encrypted storage for sensitive data

---

## v0.4.0 — MCP Integration & Unix Toolbox (2025-11-20)

### New Features
- Full **MCP server** (`MCP/ragix_mcp_server.py`)
  - Tools: `ragix_chat`, `ragix_scan_repo`, `ragix_read_file`
  - Compatible with Claude Desktop, Claude Code, Codex
- **ragix_tools.py** — sovereign Unix toolbox
  - `rt-find`, `rt-grep`, `rt-stats`, `rt-lines`, `rt-top`, `rt-replace`, `rt-doc2md`
- **Bash surrogates** — `rt.sh`, `rt-find.sh`, `rt-grep.sh`
- **Tool spec** — `MCP/ragix_tools_spec.json`

### Architecture
- Unified naming (RAGIX everywhere)
- Environment variables: `UNIX_RAG_MODEL`, `UNIX_RAG_SANDBOX`, `UNIX_RAG_PROFILE`
- Project overview pre-scan at startup
- Enhanced denylist enforcement

### Documentation
- Rewritten README.md
- Added README_RAGIX_TOOLS.md
- Added MCP/README_MCP.md
- Updated demo.md

---

## v0.3.0 — Original Release (2025-11)

### Features
- `unix-rag-agent.py` — main agent script
- JSON action protocol: `bash`, `bash_and_respond`, `edit_file`, `respond`
- Git awareness (status, diff, log)
- Sandboxed shell with denylist
- Structured logging (`.agent_logs/commands.log`)
- Basic Unix-RAG retrieval

---

## v0.2.0 — Experimental (2025-10)

- Shell sandbox drafts
- Local LLM integration (Ollama)
- Unix-RAG prompt engineering experiments

---

## v0.1.0 — Prototype (2025-09)

- First prototype: bash via LLM
- Pure sandbox experiment
- Hardcoded reasoning

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| **v0.8** | *Planned* | WASP: WebAssembly sandbox, browser runtime |
| **v0.7** | 2025-11-25 | Launcher, Web GUI, LLM backends |
| **v0.6** | 2025-11-24 | Monitoring, resilience, caching |
| **v0.5** | 2025-11-23 | ragix_core package, workflows, hybrid search |
| **v0.4** | 2025-11-20 | MCP integration, Unix toolbox |
| **v0.3** | 2025-11 | Original release |
| **v0.2** | 2025-10 | Experimental |
| **v0.1** | 2025-09 | Prototype |

## Related Documents

| Document | Purpose |
|----------|---------|
| `V08_WASP_PLANNING.md` | Detailed v0.8 WASP specifications |
| `WASM.md` | WASM architecture rationale |
| `README.md` | Usage documentation |
| `MCP/README_MCP.md` | MCP integration guide |

---

*For detailed usage instructions, see [README.md](README.md).*
