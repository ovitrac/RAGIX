# CHANGELOG — RAGIX

All notable changes to the **RAGIX** project will be documented here.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## v0.8.0 — Plugin System, SWE Workflows & WASP Foundation (2025-11-26)

### Highlights

**RAGIX becomes a true platform with extensible plugins, chunked workflows for large codebases, and WASP sandbox abstraction.**

| Feature | Status |
|---------|--------|
| Plugin System | ✅ Implemented |
| Unified Tool Registry | ✅ Enhanced |
| SWE Chunked Workflows | ✅ Implemented |
| WASP Sandbox Abstraction | ✅ Foundation |
| Built-in Plugins | ✅ 2 examples |
| CLI Plugin Commands | ✅ Implemented |

### New Features

#### Plugin System (`ragix_core/plugin_system.py`)

RAGIX now supports extensible plugins for tools and workflows:

- **Plugin types** — `tool`, `workflow` (future: `agent`, `backend`, `search`)
- **Trust levels** — `builtin`, `trusted`, `untrusted` with capability restrictions
- **Safe loading** — explicit allowlist, capability-based permissions
- **Plugin manifest** — YAML-based definition with tools, workflows, dependencies

```yaml
# plugin.yaml example
name: json-validator
version: 1.0.0
type: tool
trust: builtin
capabilities:
  - file:read
tools:
  - name: validate_json
    entry: json_tools:validate_json
    parameters:
      - name: content
        type: string
        required: true
```

#### Unified Tool Registry Enhancement (`ragix_core/tool_registry.py`)

- **Provider tracking** — tools tagged with source (`builtin`, `plugin`, `mcp`, `wasm`)
- **Unified API** — same tools available via CLI, Web UI, MCP server
- **Plugin sync** — automatic registration of plugin tools
- **Export formats** — CLI-friendly and MCP-compatible exports

#### SWE Chunked Workflows (`ragix_core/swe_workflows.py`)

For large codebase operations:

- **Chunked processing** — split large file sets into manageable chunks
- **Checkpoint resumption** — save/restore workflow state across interruptions
- **Circuit breaker** — automatic pause on repeated failures
- **Progress tracking** — real-time progress and ETA estimation

```python
from ragix_core import FileProcessingWorkflow, ChunkConfig

workflow = FileProcessingWorkflow(
    workflow_id="review-2024",
    root_path=Path("./src"),
    file_patterns=["*.py"],
    config=ChunkConfig(chunk_size=50),
)
results = workflow.run_on_files()
```

#### WASP Sandbox Abstraction (`ragix_core/sandbox_base.py`, `wasm_sandbox.py`)

Foundation for WebAssembly tool execution:

- **BaseSandbox protocol** — unified interface for all sandbox types
- **SandboxConfig** — capability-based security model
- **ExecutionResult** — unified result format across backends
- **HybridSandbox** — routes to WASM or shell based on availability
- **WasmSandbox** — WASM execution (requires `wasmtime>=14.0.0`)

```python
from ragix_core import create_sandbox, SandboxType

# Create hybrid sandbox (WASM when available, shell fallback)
sandbox = create_sandbox("hybrid", root_path=Path.cwd())
result = sandbox.run("validate_json {...}")
```

#### Plugin CLI Commands

New `ragix plugin` subcommands:

```bash
ragix plugin list              # List available plugins
ragix plugin info <name>       # Show plugin details
ragix plugin load <name>       # Load a plugin
ragix plugin unload <name>     # Unload a plugin
ragix plugin create <name>     # Create new plugin scaffold
ragix tools                    # List all available tools
```

#### Built-in Example Plugins

Two example plugins in `plugins/`:

1. **json-validator** — JSON/YAML validation and diff tools
   - `validate_json` — validate and format JSON
   - `validate_yaml` — validate YAML, convert to JSON
   - `json_diff` — compare two JSON objects

2. **file-stats** — File and codebase statistics
   - `file_stats` — size, lines, encoding
   - `directory_stats` — file counts, sizes, types
   - `code_stats` — lines of code, comments, blanks

### Files Added/Modified

| File | Description |
|------|-------------|
| `ragix_core/plugin_system.py` | Plugin system (~600 lines) |
| `ragix_core/swe_workflows.py` | Chunked workflows (~650 lines) |
| `ragix_core/sandbox_base.py` | Sandbox abstraction (~400 lines) |
| `ragix_core/wasm_sandbox.py` | WASM sandbox (~450 lines) |
| `ragix_core/tool_registry.py` | Enhanced with providers (~200 lines added) |
| `ragix_core/cli.py` | Plugin commands (~350 lines added) |
| `plugins/json-validator/` | Example tool plugin |
| `plugins/file-stats/` | Example tool plugin |
| `pyproject.toml` | Version 0.8.0, added `wasm` optional dep |

### New Dependencies

```toml
[project.optional-dependencies]
wasm = ["wasmtime>=14.0.0"]  # Optional, for WASM sandbox
```

### Architecture

```
v0.8 Architecture:

                    ┌─────────────────────────────────────┐
                    │         ragix_core/cli.py           │
                    │    ragix plugin list/load/...       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       PluginManager                 │
                    │   - discover()                      │
                    │   - load_plugin()                   │
                    │   - get_tool()                      │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │ Tool Plugin │         │  Workflow   │         │  Built-in   │
   │  (trusted)  │         │   Plugin    │         │   Tools     │
   └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       Unified Tool Registry         │
                    │   - ToolProvider: builtin/plugin/mcp│
                    │   - export_for_cli()                │
                    │   - export_for_mcp()                │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │   CLI       │         │   Web UI    │         │ MCP Server  │
   │ ragix tools │         │ Streamlit   │         │   Claude    │
   └─────────────┘         └─────────────┘         └─────────────┘
```

### Migration from v0.7.1

- **No breaking changes** — all v0.7.1 features preserved
- **New imports** — plugin and workflow classes in `ragix_core`
- **Optional WASM** — `pip install ragix[wasm]` for WASP features
- **Plugin directory** — create `plugins/` in project or `~/.ragix/plugins/` global

---

## v0.9.0 — WASP Tools & Browser Runtime (2025-11-26)

### Highlights

**WASP (WebAssembly-ready Agentic System Protocol) delivers deterministic, sandboxed tools for RAGIX agents with browser-side execution capability.**

| Feature | Status |
|---------|--------|
| WASP Tools (Python) | ✅ 18 tools |
| WASP CLI | ✅ Implemented |
| wasp_task Protocol | ✅ Implemented |
| Browser Runtime (JS) | ✅ Implemented |
| Virtual Filesystem | ✅ Implemented |
| Test Suite | ✅ 73 tests |

### New Features

#### WASP Tools (`wasp_tools/`)

18 deterministic tools across three categories:

**Validation:**
- `validate_json` — Validate JSON with optional schema
- `validate_yaml` — Validate YAML with optional schema
- `format_json` — Format/prettify JSON
- `format_yaml` — Format/prettify YAML
- `json_to_yaml` — Convert JSON to YAML
- `yaml_to_json` — Convert YAML to JSON

**Markdown:**
- `parse_markdown` — Parse to structured AST
- `extract_headers` — Extract headers
- `extract_code_blocks` — Extract code blocks
- `extract_links` — Extract links
- `extract_frontmatter` — Extract YAML frontmatter
- `renumber_sections` — Renumber section headers
- `generate_toc` — Generate table of contents

**Search:**
- `search_pattern` — Regex pattern search
- `search_lines` — Search with line context
- `count_matches` — Count pattern matches
- `extract_matches` — Extract with groups
- `replace_pattern` — Replace matches

#### WASP CLI (`ragix-wasp`)

```bash
ragix-wasp list              # List available tools
ragix-wasp info <tool>       # Show tool details
ragix-wasp run <tool> <args> # Run tool directly
ragix-wasp validate <file>   # Validate manifest
ragix-wasp categories        # List categories
```

#### wasp_task Protocol (`ragix_core/orchestrator.py`)

New action type for agent protocol:

```json
{
  "action": "wasp_task",
  "tool": "validate_json",
  "inputs": {"content": "..."}
}
```

#### WASP Executor (`ragix_core/wasp_executor.py`)

- Tool registry and execution
- Input validation
- Timing and metrics
- Custom tool registration
- Prompt generation for agents

#### Browser Runtime (`ragix_web/static/js/`)

- `wasp_runtime.js` — Client-side tool execution
- `virtual_fs.js` — In-memory filesystem
- `browser_tools.js` — UI integration

### Files Added/Modified

| File | Description |
|------|-------------|
| `wasp_tools/__init__.py` | Tool registry (~150 lines) |
| `wasp_tools/validate.py` | Validation tools (~350 lines) |
| `wasp_tools/mdparse.py` | Markdown tools (~400 lines) |
| `wasp_tools/search.py` | Search tools (~300 lines) |
| `wasp_tools/manifest.yaml` | Tool definitions |
| `ragix_unix/wasp_cli.py` | WASP CLI (~300 lines) |
| `ragix_core/wasp_executor.py` | Executor (~280 lines) |
| `ragix_core/orchestrator.py` | wasp_task action |
| `ragix_web/static/js/wasp_runtime.js` | Browser runtime |
| `ragix_web/static/js/virtual_fs.js` | Virtual filesystem |
| `ragix_web/static/js/browser_tools.js` | UI integration |
| `tests/test_wasp_tools.py` | Tool tests |
| `tests/test_wasp_integration.py` | Integration tests |
| `docs/WASP_GUIDE.md` | Documentation |

### Architecture

```
Agent Action
    │
    ▼
┌─────────────────┐
│  WaspExecutor   │
│  - Registry     │
│  - Validation   │
│  - Timing       │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Python │ │Browser│
│wasp_  │ │Wasp   │
│tools/ │ │Runtime│
└───────┘ └───────┘
```

### Migration from v0.8.0

- **No breaking changes** — all v0.8.0 features preserved
- **New imports** — wasp_tools module, WaspExecutor class
- **New CLI** — `ragix-wasp` command
- **New action** — `wasp_task` in agent protocol

---

## [Unreleased] — v1.0

### Planned Features

- **WASM Tools** — Compile tools to WebAssembly
- **AST-aware Search** — tree-sitter integration
- **Agent Improvements** — Multi-step reasoning, memory
- **VS Code Extension** — IDE integration

---

## [Future] — v1.0+ Ideas

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

## v0.7.1 — Unified Configuration & Compliance (2025-11-26)

### Highlights

**Response to external code review — consolidation release addressing gaps identified in v0.7.0.**

| Feature | Status |
|---------|--------|
| Unified config (`ragix.yaml`) | ✅ Implemented |
| Log hashing (SHA256) | ✅ Implemented |
| Log viewer in GUI | ✅ Implemented |
| `ragix` CLI commands | ✅ Implemented |
| Full MCP instantiation | ✅ Implemented |

### New Features

#### Unified Configuration (`ragix.yaml`)
- **Single config file** — all settings in one place
- **Environment variable overrides** — `RAGIX_*` variables take precedence
- **Backward compatibility** — legacy `UNIX_RAG_*` variables still work
- **Auto-discovery** — searches cwd, `.ragix/`, `~/.config/ragix/`
- **Data classes** — `RAGIXConfig`, `LLMConfig`, `MCPConfig`, `SafetyConfig`, etc.

```yaml
# ragix.yaml example
llm:
  backend: ollama
  model: mistral
safety:
  profile: dev
  air_gapped: false
  log_hashing: true
mcp:
  enabled: true
  port: 5173
```

#### Log Integrity (`ragix_core/log_integrity.py`)
- **ChainedLogHasher** — blockchain-style hash chain for logs
- **SHA256 signatures** — each entry includes hash of previous entry
- **Tamper detection** — verify chain integrity on demand
- **AuditLogManager** — unified audit logging with optional hashing
- **Log export** — download logs and hash files from GUI

#### Web UI Log Viewer (new tab in `ragix_app.py`)
- **Recent Entries** — color-coded by type (CMD, EDIT, EVENT, ERROR)
- **Search Logs** — filter by type and search pattern
- **Integrity Verification** — verify hash chain with one click
- **Export** — download log files and hash signatures

#### RAGIX CLI (`ragix` command)
- `ragix install` — setup environment, create directories, default config
- `ragix doctor` — comprehensive system diagnostics
- `ragix config` — show current configuration
- `ragix status` — quick status check
- `ragix logs [-n 50]` — view recent log entries
- `ragix verify` — verify log integrity
- `ragix mcp` — start MCP server
- `ragix web` — start web interface
- `ragix run` — start interactive agent
- `ragix upgrade` — upgrade instructions

#### Enhanced MCP Server (4 new tools)
- `ragix_config()` — get current configuration
- `ragix_verify_logs()` — verify log integrity
- `ragix_logs(n)` — get recent log entries
- `ragix_agent_step(prompt)` — config-aware agent execution

### Files Added/Modified

| File | Description |
|------|-------------|
| `ragix.yaml` | Sample unified configuration |
| `ragix_core/config.py` | Configuration loader (~350 lines) |
| `ragix_core/log_integrity.py` | Log hashing (~450 lines) |
| `ragix_core/cli.py` | CLI commands (~550 lines) |
| `ragix_app.py` | Added Logs page (~220 lines) |
| `MCP/ragix_mcp_server.py` | Added 4 new MCP tools |
| `pyproject.toml` | Updated version, added `ragix` entry point |

### Gap Analysis Summary (from external review)

| Review Point | v0.7.0 Status | v0.7.1 Status |
|--------------|---------------|---------------|
| Modular package | ✅ Exceeded | ✅ Maintained |
| MCP integration | ⚠️ Partial | ✅ Full |
| Multi-agent | ✅ Exceeded | ✅ Maintained |
| Hybrid retrieval | ✅ Full | ✅ Maintained |
| Web UI | ⚠️ Partial | ✅ Full (+ logs) |
| Reproducibility | ⚠️ Partial | ✅ CLI added |
| Security | ⚠️ Partial | ✅ Log hashing |
| WASP (WASM) | Planned | Deferred to v0.8 |

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
| **v0.7.1** | 2025-11-26 | Unified config, log hashing, CLI, MCP consolidation |
| **v0.7.0** | 2025-11-25 | Launcher, Web GUI, LLM backends |
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
