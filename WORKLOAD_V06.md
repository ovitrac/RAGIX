# RAGIX v0.6 Implementation Workload

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Date:** 2025-11-24
**Goal:** Web UI + Real Multi-Agent + Hybrid Retrieval

---

## Overview

v0.6 builds on the solid modular foundation from v0.5 to introduce:
1. **Local Web UI** with chat interface and tool traces
2. **Graph-based multi-agent workflows** (agent orchestration)
3. **Hybrid retrieval** with embeddings + semantic search
4. **Early CI integration** (batch mode)
5. **Local secrets vault** abstraction

**Philosophy:** Maintain local-first, sovereign AI principles. No cloud dependencies.

---

## Task 2.1: Graph-Based Agent Orchestrator

**Goal:** Extend orchestrator to support multi-agent workflows with explicit graph representation.

### Subtasks

**2.1.1: Agent Graph Data Structure**
- Create `ragix_core/agent_graph.py`:
  - `AgentNode` class (agent instance + tools + entry/exit conditions)
  - `AgentEdge` class (transitions based on outcomes)
  - `AgentGraph` class (DAG representation)
- Support linear chains, parallel branches, conditional routing
- Serialization to/from JSON/YAML

**2.1.2: Graph Executor**
- Create `ragix_core/graph_executor.py`:
  - `GraphExecutor` class
  - Execute nodes in topological order
  - Handle parallel execution (when branches independent)
  - State management between nodes
  - Error handling and rollback

**2.1.3: Pre-Built Agent Roles**
- Create `ragix_core/agents/`:
  - `CodeAgent` - codebase exploration and editing
  - `DocAgent` - documentation and README analysis
  - `GitAgent` - git operations (status, diff, log)
  - `TestAgent` - running tests and interpreting results
- Each agent has specific tool subset and system prompt

**2.1.4: Task Delegation API**
- Primary agent can spawn sub-agents for sub-tasks
- Result aggregation from multiple agents
- Context sharing between agents (project state)

**Deliverables:**
- `ragix_core/agent_graph.py` (200 lines)
- `ragix_core/graph_executor.py` (300 lines)
- `ragix_core/agents/__init__.py`, `code_agent.py`, `doc_agent.py`, `git_agent.py`, `test_agent.py` (400 lines total)
- Unit tests for graph execution
- Example workflow: "Refactor module X" graph

**Time estimate:** Medium complexity

---

## Task 2.2: Local Web UI

**Goal:** Create local-only web interface for RAGIX with chat, traces, and workspace management.

### Subtasks

**2.2.1: Backend API Server**
- Create `ragix_web/` package:
  - FastAPI server (`server.py`)
  - WebSocket endpoint for chat streaming
  - REST endpoints for:
    - `/api/sessions` (list, create, delete)
    - `/api/sandbox` (set root, get info)
    - `/api/logs` (get command logs, events)
    - `/api/config` (get/set profile, model, etc.)
- Serve static files from `ragix_web/static/`

**2.2.2: Frontend (Single-Page App)**
- Create `ragix_web/static/`:
  - `index.html` - Main layout
  - `app.js` - Chat UI, WebSocket handling
  - `traces.js` - Command trace viewer
  - `style.css` - Clean, terminal-inspired design
- Use vanilla JS or lightweight framework (Alpine.js / htmx)
- No Node.js build required (keep it simple)

**2.2.3: Chat Interface**
- Real-time message streaming via WebSocket
- Display:
  - User messages
  - Agent responses
  - Command outputs (collapsible)
  - File edits (diff view)
- Input: Textarea with syntax highlighting (markdown)

**2.2.4: Tool Trace Viewer**
- Timeline view of all tool calls:
  - Bash commands (with stdout/stderr)
  - File edits (with diffs)
  - Retrieval queries (with results)
  - Agent transitions (in multi-agent mode)
- Filterable by type, status, timestamp

**2.2.5: Workspace Management**
- Multiple workspaces (tabs/projects)
- Each workspace has:
  - Sandbox root
  - Session history
  - Command logs
- Persist workspace state to disk

**2.2.6: CLI Integration**
- New command: `ragix-web --port 8080 --sandbox-root ~/project`
- Opens browser automatically
- Graceful shutdown (Ctrl+C)

**Deliverables:**
- `ragix_web/server.py` (400 lines)
- `ragix_web/static/index.html` (200 lines)
- `ragix_web/static/app.js` (500 lines)
- `ragix_web/static/traces.js` (300 lines)
- `ragix_web/static/style.css` (200 lines)
- Entry point in `pyproject.toml`: `ragix-web = "ragix_web.server:main"`
- Screenshot for documentation

**Time estimate:** High complexity (frontend + backend integration)

---

## Task 2.3: Hybrid Retrieval System

**Goal:** Implement embeddings-backed hybrid retrieval (keyword + semantic) for codebases.

### Subtasks

**2.3.1: Retrieval Backend Interface**
- Extend `ragix_core/retrieval.py`:
  - `HybridRetriever` class
  - Combines keyword (grep) + semantic (embeddings)
  - Ranking/reranking logic

**2.3.2: Code Chunking**
- Create `ragix_core/chunking.py`:
  - Parse files into semantic chunks:
    - Functions/classes (via AST for Python, JS, etc.)
    - Docstrings/comments
    - Markdown sections
  - Chunk metadata: file, start_line, end_line, type, name

**2.3.3: Embedding Backend**
- Create `ragix_core/embeddings.py`:
  - `EmbeddingBackend` protocol
  - `SentenceTransformerBackend` implementation:
    - Uses local models (e.g., `all-MiniLM-L6-v2`)
    - No external API calls
  - Batch encoding for efficiency

**2.3.4: Vector Index**
- Create `ragix_core/vector_index.py`:
  - Simple in-memory vector store (NumPy-based)
  - Save/load to disk (`.ragix_index/vectors.npy`, `metadata.json`)
  - Cosine similarity search
  - Optional: use FAISS for large indices

**2.3.5: Indexing CLI**
- New command: `ragix index <path> --profile code`
  - Scans files (`.py`, `.js`, `.md`, etc.)
  - Chunks code
  - Generates embeddings
  - Saves index to `.ragix_index/`
  - Progress bar
- Incremental indexing (detect changed files)

**2.3.6: Search Action**
- Add new JSON action: `{"action": "search_project", "query": "...", "max_results": 20}`
- Hybrid search:
  - Keyword match (grep) - fast, precise
  - Semantic match (embeddings) - handles synonyms, concepts
  - Combine and rank results
- Return: file:line:content + relevance score

**Deliverables:**
- `ragix_core/chunking.py` (200 lines)
- `ragix_core/embeddings.py` (150 lines)
- `ragix_core/vector_index.py` (250 lines)
- Enhanced `ragix_core/retrieval.py` (add HybridRetriever, 200 lines)
- CLI command in `ragix_unix/cli.py`: `ragix index` subcommand
- Unit tests for chunking, embedding, indexing
- Example: Index RAGIX itself, query "how to add a new tool"

**Dependencies:**
- `sentence-transformers` (local models)
- `numpy`
- Optional: `faiss-cpu` for large-scale

**Time estimate:** High complexity (chunking + embeddings + indexing pipeline)

---

## Task 2.4: Batch Mode for CI Integration

**Goal:** Enable RAGIX to run in non-interactive batch mode for CI pipelines.

### Subtasks

**2.4.1: Batch Mode Executor**
- Create `ragix_core/batch_mode.py`:
  - `BatchExecutor` class
  - Reads workflow from YAML/JSON
  - Executes steps sequentially
  - Collects all outputs
  - Exits with status code (0 = success, non-zero = failure)

**2.4.2: Workflow Definition Format**
- Define YAML schema for workflows:
```yaml
name: "CI Lint and Review"
steps:
  - name: "Check code style"
    agent: "code_agent"
    action: "bash"
    command: "ruff check src/"

  - name: "Generate docstring suggestions"
    agent: "doc_agent"
    action: "search_and_suggest"
    query: "functions without docstrings"

  - name: "Run tests"
    agent: "test_agent"
    action: "bash"
    command: "pytest tests/"
```

**2.4.3: CI CLI Command**
- New command: `ragix run-ci-check --playbook ci_checks.yaml --output report.json`
- Options:
  - `--dry-run` - Preview without execution
  - `--fail-on-warnings` - Exit non-zero if warnings found
  - `--output` - Save results as JSON/Markdown

**2.4.4: CI Templates**
- Create `templates/ci/`:
  - `github_actions.yaml` - GitHub Actions workflow
  - `gitlab_ci.yaml` - GitLab CI config
  - `README_CI.md` - Setup instructions
- Example: Run RAGIX on PR, post results as comment

**Deliverables:**
- `ragix_core/batch_mode.py` (300 lines)
- CLI command in `ragix_unix/cli.py`: `ragix run-ci-check`
- `templates/ci/github_actions.yaml`
- `templates/ci/gitlab_ci.yaml`
- `templates/ci/README_CI.md`
- Example workflow: `examples/ci_workflow.yaml`

**Time estimate:** Medium complexity

---

## Task 2.5: Local Secrets Vault Abstraction

**Goal:** Implement secure local secrets storage with access control policies.

### Subtasks

**2.5.1: Vault Interface**
- Create `ragix_core/secrets_vault.py`:
  - `SecretProvider` protocol
  - Methods: `store(key, value)`, `retrieve(key)`, `delete(key)`, `list_keys()`
  - Policy enforcement (which agent can access which secrets)

**2.5.2: File-Based Encrypted Vault**
- Create `ragix_core/vault_backends.py`:
  - `EncryptedFileVault` implementation:
    - Store secrets in `~/.ragix_vault/secrets.enc`
    - Encrypt with user password (using `cryptography` library)
    - Fernet symmetric encryption
  - `EnvVault` (read-only, for backward compatibility)

**2.5.3: Vault CLI Commands**
- New commands:
  - `ragix vault init` - Initialize encrypted vault
  - `ragix vault set <key> <value>` - Store secret (prompts for vault password)
  - `ragix vault get <key>` - Retrieve secret
  - `ragix vault list` - List all keys (not values)
  - `ragix vault delete <key>` - Remove secret

**2.5.4: Agent Access Control**
- Extend profiles to include secret access policies:
```toml
[unix_agent.secrets]
allowed_patterns = ["API_KEY_*", "GITHUB_TOKEN"]
denied_patterns = ["PROD_*"]
require_confirmation = true  # Prompt before allowing access
```

**2.5.5: Integration with Logging**
- Ensure vault operations are logged (but not secret values)
- Log: key accessed, by which agent, at what time
- Never log plaintext secrets

**Deliverables:**
- `ragix_core/secrets_vault.py` (200 lines)
- `ragix_core/vault_backends.py` (300 lines)
- CLI commands in `ragix_unix/cli.py`: `ragix vault` subcommand
- Policy enforcement in `ragix_core/profiles.py`
- Unit tests for vault (encryption, access control)
- Documentation: `SECRETS_VAULT.md`

**Dependencies:**
- `cryptography` library (for Fernet encryption)

**Time estimate:** Medium complexity

---

## Task 2.6: Enhanced Agent System Prompt for Multi-Agent

**Goal:** Update system prompts to support multi-agent workflows and new capabilities.

### Subtasks

**2.6.1: Update AGENT_SYSTEM_PROMPT**
- Extend `ragix_unix/agent.py`:
  - Document new action: `search_project`
  - Document agent delegation: `{"action": "delegate", "target_agent": "doc_agent", "task": "..."}`
  - Document multi-step workflows

**2.6.2: Agent-Specific Prompts**
- Create prompts for each agent role:
  - `CodeAgent`: Emphasize code analysis, refactoring, bug finding
  - `DocAgent`: Emphasize documentation quality, README clarity
  - `GitAgent`: Emphasize commit history analysis, branch management
  - `TestAgent`: Emphasize test coverage, failure analysis

**2.6.3: Workflow Examples**
- Add example workflows to documentation:
  - "Bug localization and fix" (CodeAgent → TestAgent)
  - "Feature implementation" (CodeAgent → DocAgent → TestAgent)
  - "Code review" (CodeAgent → GitAgent)

**Deliverables:**
- Updated `ragix_unix/agent.py` (system prompt extensions)
- Agent-specific prompts in `ragix_core/agents/*.py`
- Example workflows in `EXAMPLES_MULTI_AGENT.md`

**Time estimate:** Low complexity (mostly documentation)

---

## Task 2.7: Testing and Documentation

**Goal:** Comprehensive testing and documentation for all v0.6 features.

### Subtasks

**2.7.1: Unit Tests**
- Agent graph execution tests
- Hybrid retrieval tests (chunking, embedding, search)
- Vault encryption/decryption tests
- Batch mode workflow tests
- Web API endpoint tests

**2.7.2: Integration Tests**
- End-to-end multi-agent workflow
- Web UI → Backend → Agent → Tool chain
- Indexing → Search → Retrieval flow
- CI batch mode on real repository

**2.7.3: Documentation Updates**
- Update `README.md`:
  - Add v0.6 features section
  - Add Web UI screenshots
  - Add multi-agent examples
- Create new docs:
  - `WEB_UI_GUIDE.md` - How to use Web UI
  - `MULTI_AGENT_GUIDE.md` - How to create agent graphs
  - `HYBRID_RETRIEVAL.md` - How to use semantic search
  - `CI_INTEGRATION.md` - How to use in CI pipelines
  - `SECRETS_VAULT.md` - How to manage secrets
- Update `CHANGELOG.md` for v0.6

**2.7.4: Example Gallery**
- Create `examples/v06/`:
  - `bug_fix_workflow.yaml` - Multi-agent bug fix
  - `feature_addition_workflow.yaml` - Multi-agent feature implementation
  - `code_review_workflow.yaml` - Automated code review
  - `ci_lint_workflow.yaml` - CI integration example

**Deliverables:**
- Test suite in `tests/test_v06/`
- All documentation files listed above
- Example workflows in `examples/v06/`
- Updated `V06_PROGRESS.md` tracking document

**Time estimate:** Medium complexity

---

## Task 2.8: Packaging and Dependencies

**Goal:** Update package configuration for new dependencies and entry points.

### Subtasks

**2.8.1: Update pyproject.toml**
- Add new dependencies:
  - `fastapi` - Web API
  - `uvicorn` - ASGI server
  - `websockets` - WebSocket support
  - `sentence-transformers` - Embeddings
  - `numpy` - Vector operations
  - `cryptography` - Vault encryption
  - Optional: `faiss-cpu` - Large-scale vector search

**2.8.2: New Entry Points**
- `ragix-web` → `ragix_web.server:main`
- `ragix index` → `ragix_unix.cli:index_command` (subcommand)
- `ragix vault` → `ragix_unix.cli:vault_command` (subcommand)
- `ragix run-ci-check` → `ragix_unix.cli:ci_command` (subcommand)

**2.8.3: Optional Dependencies**
- Create optional dependency groups:
```toml
[project.optional-dependencies]
web = ["fastapi>=0.104.0", "uvicorn>=0.24.0", "websockets>=12.0"]
retrieval = ["sentence-transformers>=2.2.0", "numpy>=1.24.0"]
vault = ["cryptography>=41.0.0"]
ci = ["pyyaml>=6.0"]
all = ["ragix[web,retrieval,vault,ci]"]
```

**Deliverables:**
- Updated `pyproject.toml`
- Installation guide in `README.md`

**Time estimate:** Low complexity

---

## Success Criteria for v0.6

### Functional Requirements
- ✅ Multi-agent workflows can be defined as graphs and executed
- ✅ Web UI accessible at `http://localhost:8080` with chat and traces
- ✅ Hybrid retrieval (keyword + semantic) working for codebases
- ✅ Batch mode can execute workflows non-interactively for CI
- ✅ Secrets vault stores encrypted secrets with access control

### Quality Requirements
- ✅ All unit tests pass
- ✅ Integration tests demonstrate end-to-end workflows
- ✅ Documentation complete and up-to-date
- ✅ Backward compatibility with v0.5 (all v0.5 features still work)

### Performance Requirements
- ✅ Web UI responsive (<100ms latency for user actions)
- ✅ Indexing: ~1000 files/minute on typical codebase
- ✅ Search: <200ms for hybrid retrieval on medium codebase (<10k files)
- ✅ Vault operations: <50ms for encrypt/decrypt

---

## Implementation Order (Recommended)

**Phase 1: Core Infrastructure (2.1, 2.5)**
1. Task 2.1: Graph-based orchestrator
2. Task 2.5: Secrets vault abstraction

**Phase 2: Retrieval (2.3)**
3. Task 2.3: Hybrid retrieval system

**Phase 3: Interfaces (2.2, 2.4, 2.6)**
4. Task 2.6: Enhanced prompts
5. Task 2.4: Batch mode for CI
6. Task 2.2: Web UI (most complex, do last)

**Phase 4: Polish (2.7, 2.8)**
7. Task 2.8: Packaging updates
8. Task 2.7: Testing and documentation

---

## Estimated Complexity

**Total new code:** ~4000-5000 lines
**Total new tests:** ~1500 lines
**Total documentation:** ~3000 lines (guides + examples)

**Breakdown by complexity:**
- **High:** Task 2.2 (Web UI), Task 2.3 (Hybrid Retrieval)
- **Medium:** Task 2.1 (Agent Graph), Task 2.4 (Batch Mode), Task 2.5 (Vault), Task 2.7 (Testing)
- **Low:** Task 2.6 (Prompts), Task 2.8 (Packaging)

---

## Notes

**Local-First Philosophy:**
- All components run locally (no cloud dependencies)
- Embeddings via local sentence-transformers models
- Web UI served locally (not a cloud service)
- Secrets stored in encrypted local vault

**Backward Compatibility:**
- All v0.5 features remain functional
- `ragix-unix-agent` CLI unchanged
- Existing config files still work
- Gradual adoption of new features

**Optional Features:**
- Web UI is optional (CLI still fully functional)
- Hybrid retrieval is optional (grep-only mode still available)
- Vault is optional (can use environment variables)

---

**v0.6 transforms RAGIX from a CLI tool into a platform with Web UI, multi-agent orchestration, and semantic search capabilities.**
