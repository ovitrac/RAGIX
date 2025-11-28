# RAGIX v0.7 Planning Document

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
**Status:** PLANNING - Ready for implementation

---

## Context: v0.6 → v0.7

### v0.6 Achievements (COMPLETE ✅)

**Core Implementation: ~6,500 lines of production code**

1. **Graph-Based Multi-Agent Orchestration** (~1,285 lines)
   - DAG workflow representation (AgentGraph, AgentNode, AgentEdge)
   - Async executor with parallelization (GraphExecutor)
   - Specialized agent types (CodeAgent, DocAgent, GitAgent, TestAgent)

2. **Local Web UI** (~1,000 lines)
   - FastAPI backend with WebSocket support
   - Terminal-inspired frontend (vanilla JS, no build)
   - Real-time chat and tool traces

3. **Hybrid Retrieval System** (~1,443 lines)
   - AST-based code chunking (Python, Markdown, generic)
   - Sentence-transformers embeddings (local)
   - NumPy and FAISS vector indexes
   - CLI: `ragix-index`

4. **Batch Mode for CI/CD** (~1,260 lines)
   - YAML playbook configuration
   - Sequential/parallel workflow execution
   - GitHub Actions and GitLab CI templates
   - CLI: `ragix-batch`

5. **Local Secrets Vault** (~730 lines)
   - Fernet encryption (AES-128-CBC)
   - PBKDF2 key derivation (100k iterations)
   - Access control policies
   - CLI: `ragix-vault`

**Optional Dependencies Added:**
- `web`: FastAPI, uvicorn, websockets
- `retrieval`: sentence-transformers, numpy
- `vault`: cryptography
- `ci`: pyyaml

**CLI Tools:**
- `ragix-unix-agent` (interactive agent)
- `ragix-web` (Web UI server)
- `ragix-index` (semantic indexing)
- `ragix-batch` (CI workflows)
- `ragix-vault` (secrets management)

---

## v0.7 Goals: Integration & Intelligence

### Theme: **"From Platform to Orchestrator"**

v0.6 built the **infrastructure**. v0.7 focuses on **integration** and **intelligence**:

1. **LLM Integration** - Connect graph executor to actual LLM reasoning
2. **Tool Execution** - Implement real tool calls from agents
3. **Enhanced Retrieval** - BM25 + vector hybrid, query rewriting
4. **Workflow Templates** - Pre-built workflows for common tasks
5. **Advanced UI** - Workflow visualization, diff viewer, logs
6. **Performance** - Caching, streaming, incremental indexing
7. **Testing** - Unit and integration test suites
8. **Production Readiness** - Error recovery, monitoring, observability

---

## v0.7 Task Breakdown

### Task 3.1: LLM Integration with Agents ⏸️

**Goal:** Connect agent framework to actual LLM reasoning loop.

**Components:**

1. **Agent-LLM Bridge** (`ragix_core/agent_llm_bridge.py` ~300 lines)
   - `LLMAgentExecutor` class
   - Tool call parsing and execution
   - Streaming response handling
   - Context window management
   - Error recovery and retries

2. **Tool Registry** (`ragix_core/tool_registry.py` ~200 lines)
   - `ToolRegistry` class
   - Tool definitions with schemas
   - Dynamic tool loading
   - Permission checks

3. **Prompt Templates** (`ragix_core/prompt_templates.py` ~150 lines)
   - Few-shot examples for tool usage
   - Chain-of-thought templates
   - Error recovery prompts

**Integration Points:**
- Update `BaseAgent.run()` to call LLM
- Implement tool execution loop
- Connect to existing `ShellSandbox`, `GrepRetriever`, etc.

**Estimated:** ~650 lines

---

### Task 3.2: Enhanced Hybrid Retrieval ⏸️

**Goal:** Improve search with BM25 + vector fusion and query optimization.

**Components:**

1. **BM25 Index** (`ragix_core/bm25_index.py` ~250 lines)
   - Sparse keyword index
   - TF-IDF scoring
   - Integration with vector search

2. **Hybrid Search** (`ragix_core/hybrid_search.py` ~300 lines)
   - Fusion strategies (reciprocal rank, weighted)
   - Query rewriting with LLM
   - Multi-stage retrieval
   - Result deduplication and reranking

3. **Incremental Indexing** (update `index_cli.py` ~150 lines)
   - Track file modifications
   - Update only changed chunks
   - Index versioning

**Estimated:** ~700 lines

---

### Task 3.3: Workflow Template Library ⏸️

**Goal:** Pre-built workflows for common development tasks.

**Components:**

1. **Template Definitions** (`templates/workflows/` ~600 lines total)
   - `bug_fix.yaml` - Locate, fix, test bugs
   - `feature_addition.yaml` - Design, implement, document features
   - `code_review.yaml` - Automated code review
   - `refactoring.yaml` - Identify and refactor code smells
   - `documentation.yaml` - Generate/update docs from code
   - `security_audit.yaml` - Scan for vulnerabilities

2. **Template Manager** (`ragix_core/workflow_templates.py` ~200 lines)
   - Load and instantiate templates
   - Parameter substitution
   - Template validation

3. **CLI Integration** (update `batch_cli.py` ~100 lines)
   - `ragix-batch --template bug_fix --params "..."`

**Estimated:** ~900 lines

---

### Task 3.4: Advanced Web UI Features ⏸️

**Goal:** Enhance UI with visualization and interactive tools.

**Components:**

1. **Workflow Visualizer** (`ragix_web/static/workflow.js` ~400 lines)
   - D3.js or Cytoscape.js graph rendering
   - Real-time node status updates
   - Interactive node inspection

2. **Diff Viewer** (`ragix_web/static/diff.js` ~300 lines)
   - Side-by-side or inline diff display
   - Syntax highlighting
   - Accept/reject changes

3. **Log Viewer** (`ragix_web/static/logs.js` ~250 lines)
   - Real-time log streaming
   - Filtering by level, agent, tool
   - Search and export

4. **File Browser** (`ragix_web/static/files.js` ~200 lines)
   - Tree view of sandbox
   - File preview
   - Quick navigation

**Backend Updates** (update `server.py` ~200 lines)
- WebSocket endpoints for real-time updates
- Streaming log APIs
- File system APIs

**Estimated:** ~1,350 lines

---

### Task 3.5: Performance Optimization ⏸️

**Goal:** Improve speed and resource efficiency.

**Components:**

1. **LLM Response Caching** (`ragix_core/caching.py` ~200 lines)
   - Semantic cache for similar queries
   - Cache invalidation strategies
   - TTL and size limits

2. **Streaming Execution** (update `graph_executor.py` ~150 lines)
   - Stream agent outputs as they execute
   - Progressive result display in UI
   - Early termination on failure

3. **Parallel Tool Calls** (update `agent_llm_bridge.py` ~100 lines)
   - Execute independent tool calls concurrently
   - Dependency detection
   - Result aggregation

4. **Index Optimization** (update `vector_index.py` ~150 lines)
   - Lazy loading of embeddings
   - Memory-mapped arrays
   - Quantization for large indexes

**Estimated:** ~600 lines

---

### Task 3.6: Testing and Quality Assurance ⏸️

**Goal:** Comprehensive test suite for production readiness.

**Components:**

1. **Unit Tests** (`tests/test_v06/` ~1,500 lines)
   - Agent graph and executor tests
   - Embedding and indexing tests
   - Batch mode tests
   - Vault encryption tests

2. **Integration Tests** (`tests/integration/` ~1,000 lines)
   - End-to-end workflow execution
   - Multi-agent coordination
   - Tool execution and safety
   - Web UI interaction

3. **Performance Tests** (`tests/performance/` ~500 lines)
   - Indexing speed benchmarks
   - Search latency tests
   - Concurrent execution tests

4. **CI/CD Pipeline** (`.github/workflows/tests.yml`)
   - Automated test runs
   - Coverage reporting
   - Performance regression detection

**Estimated:** ~3,000 lines

---

### Task 3.7: Production Readiness ⏸️

**Goal:** Monitoring, error recovery, and operational tools.

**Components:**

1. **Observability** (`ragix_core/observability.py` ~300 lines)
   - Structured logging (JSON)
   - Metrics collection (execution time, token usage)
   - Tracing with OpenTelemetry (optional)

2. **Error Recovery** (update agents ~200 lines)
   - Automatic retry with exponential backoff
   - Fallback strategies
   - Partial result preservation

3. **Health Checks** (`ragix_web/health.py` ~150 lines)
   - `/health` endpoint with detailed status
   - Ollama connectivity check
   - Index availability check
   - Disk space monitoring

4. **Admin Tools** (`ragix_unix/admin_cli.py` ~250 lines)
   - `ragix-admin stats` - Usage statistics
   - `ragix-admin cleanup` - Clean logs and caches
   - `ragix-admin diagnose` - System diagnostics

**Estimated:** ~900 lines

---

### Task 3.8: Documentation and Examples ⏸️

**Goal:** Complete documentation for v0.7 launch.

**Components:**

1. **Architecture Guide** (`docs/ARCHITECTURE.md` ~600 lines)
   - System overview
   - Component interaction diagrams
   - Data flow documentation

2. **API Reference** (`docs/API_REFERENCE.md` ~800 lines)
   - All public classes and functions
   - Code examples
   - Type signatures

3. **Workflow Guide** (`docs/WORKFLOW_GUIDE.md` ~500 lines)
   - Creating custom workflows
   - Agent coordination patterns
   - Best practices

4. **Examples** (`examples/v07/` ~1,000 lines)
   - Complete working examples
   - Jupyter notebooks for tutorials
   - Video demos (scripts)

5. **Migration Guide** (`docs/MIGRATION_V06_V07.md` ~300 lines)
   - Breaking changes
   - Upgrade steps
   - Compatibility notes

**Estimated:** ~3,200 lines

---

## Total v0.7 Estimates

| Task | Description | Lines | Priority |
|------|-------------|-------|----------|
| 3.1 | LLM Integration | ~650 | CRITICAL |
| 3.2 | Enhanced Retrieval | ~700 | HIGH |
| 3.3 | Workflow Templates | ~900 | HIGH |
| 3.4 | Advanced UI | ~1,350 | MEDIUM |
| 3.5 | Performance | ~600 | MEDIUM |
| 3.6 | Testing | ~3,000 | CRITICAL |
| 3.7 | Production Readiness | ~900 | HIGH |
| 3.8 | Documentation | ~3,200 | HIGH |
| **TOTAL** | | **~11,300** | |

---

## Implementation Strategy

### Phase 1: Core Integration (Weeks 1-2)
1. Task 3.1: LLM Integration
2. Task 3.2: Enhanced Retrieval (basic hybrid)
3. Basic testing for 3.1 + 3.2

### Phase 2: Workflows & UI (Weeks 3-4)
4. Task 3.3: Workflow Templates
5. Task 3.4: Advanced UI (workflow viz + diff viewer)

### Phase 3: Optimization & Quality (Weeks 5-6)
6. Task 3.5: Performance Optimization
7. Task 3.6: Testing Suite
8. Task 3.7: Production Readiness

### Phase 4: Documentation & Launch (Week 7)
9. Task 3.8: Documentation
10. Final integration testing
11. v0.7 release

---

## Dependencies & Prerequisites

### External Libraries (to add)
```toml
[project.optional-dependencies]
llm = ["tiktoken>=0.5.0"]  # Token counting
ui = ["d3js-integration>=1.0"]  # Graph visualization (if needed)
testing = ["pytest>=7.0", "pytest-asyncio>=0.21", "pytest-cov>=4.0"]
observability = ["opentelemetry-api>=1.20", "opentelemetry-sdk>=1.20"]  # Optional
```

### Infrastructure
- Ollama installed and running (for LLM integration tests)
- FAISS (optional, for large-scale testing)
- Redis (optional, for distributed caching in future)

---

## Success Criteria

### Functional
- ✅ LLM-driven agent execution with real tool calls
- ✅ Hybrid search (BM25 + vector) improves recall
- ✅ Workflow templates run end-to-end
- ✅ Web UI visualizes workflows in real-time
- ✅ All tests pass (unit + integration)
- ✅ Production-ready error handling

### Quality
- ✅ >80% code coverage
- ✅ All public APIs documented
- ✅ Performance benchmarks established
- ✅ Security audit passed (no secrets leaked, sandbox secure)

### Performance
- ✅ <500ms search latency (medium codebase)
- ✅ <2s workflow startup time
- ✅ Streaming responses <100ms first token
- ✅ <5% overhead from caching layer

---

## Risk Assessment

### High Risk
1. **LLM reliability** - Models may produce invalid tool calls
   - Mitigation: Robust parsing, retry logic, fallbacks

2. **Performance degradation** - Added features may slow system
   - Mitigation: Benchmarks, profiling, lazy loading

### Medium Risk
3. **UI complexity** - Advanced features may be hard to maintain
   - Mitigation: Modular design, clear separation of concerns

4. **Test coverage** - Large surface area to test
   - Mitigation: Prioritize critical paths, incremental coverage

### Low Risk
5. **Documentation drift** - Docs may fall behind code
   - Mitigation: Docstring-driven docs, CI checks

---

## Post-v0.7: v0.8 WASP Integration (Preview)

### Theme: **"WebAssembly-Powered Agentic Pipelines"**

v0.8 introduces **WASP** (WebAssembly Agentic System Protocol) as a secure, portable execution layer for RAGIX agents. See `V08_WASP_PLANNING.md` and `WASM.md` for full specifications.

### Core v0.8 Features

1. **WasmSandbox** - WASM-based tool execution alongside ShellSandbox
2. **WASP Tool Registry** - `.wasm` modules for deterministic operations
3. **JSON Action: `wasp_task`** - New protocol action for WASM execution
4. **Browser-Native RAGIX** - Web UI with client-side WASM tools

### Priority WASM Tools (in order)

1. **JSON/YAML Validator** - Deterministic schema validation
2. **Markdown Parser** - Structural parsing for documentation tasks
3. **ripgrep.wasm** - Portable search for agents (server + browser)

### Architecture

```
LLM → RAGIX Orchestrator → ┬→ Unix-RAG (ShellSandbox)
                           └→ WASP (WasmSandbox)
                                 ↓
                           .wasm modules
                           (deterministic, auditable)
```

### Implementation Strategy

- **Server-side**: Python + wasmtime bindings (parallel)
- **Browser-side**: JS + WASI (parallel)
- **Unified protocol**: Same JSON actions work in both environments

### Why WASP?

- **Stronger sandboxing**: WASI capabilities model
- **Portability**: Same tools run on Linux/macOS/Windows/browser
- **Auditability**: Deterministic execution, full logging
- **Sovereignty**: No cloud dependencies, local-first

---

## Beyond v0.8 (v0.9+)

### Potential Features
- **Multi-repo support** - Work across multiple repositories
- **Remote execution** - Distributed agent execution
- **Plugin system** - Third-party agent and tool plugins
- **Advanced debugging** - Step-through debugging for workflows
- **Cloud integrations** - Optional AWS/GCP/Azure tool providers
- **Custom LLMs** - Support for OpenAI, Anthropic, etc. (in addition to Ollama)
- **Collaboration** - Multi-user sessions (optional)
- **WASP scientific kernels** - SFPPy-lite, signal processing in WASM

---

## v0.6 → v0.7 Migration Notes

### Breaking Changes (anticipated)
- Agent `run()` signature may change to include LLM executor
- Tool registry replaces ad-hoc tool definitions
- Workflow YAML schema may add new fields

### Backward Compatibility
- v0.6 workflows should run in v0.7 with deprecation warnings
- Existing indexes can be upgraded incrementally
- Web UI v0.6 → v0.7 graceful degradation

---

## Quick Start Context for v0.7 Implementation

When resuming work on v0.7:

1. **Read this document** (`V07_PLANNING.md`)
2. **Check v0.6 status** (`V06_PROGRESS.md` - all tasks complete)
3. **Review v0.6 code structure** (8 new modules, 5 CLI tools)
4. **Start with Task 3.1** (LLM Integration) - highest priority
5. **Follow implementation strategy** (Phase 1 → Phase 4)

**Key Files to Start:**
- `ragix_core/agent_llm_bridge.py` (new)
- `ragix_core/tool_registry.py` (new)
- `ragix_core/agents/base_agent.py` (update `run()` method)

**First Milestone:** Working LLM-driven agent that can execute bash and search_project tools.

---

**v0.7 Status:** READY TO START - All v0.6 dependencies in place
