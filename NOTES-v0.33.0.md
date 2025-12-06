> **RAGIX 0.33.0** is the largest leap forward since the project began.
> It delivers a **production-ready agentic reasoning framework**, **Claude Code-style shell execution**, **hybrid RAG retrieval**, and a **full graphical web interface**.

------

# 🚀 RAGIX v0.33.0 — The Agentic Reasoning Platform

This release marks RAGIX's transition from a Unix-only automation agent (0.11) into a **complete AI-powered development assistant** with:

- **Reflective Reasoning Graph** (v0.30) — Production-ready multi-node reasoning
- **Claude Code-style Shell Sandbox** — Safe command execution with audit logging
- **Hybrid RAG** — BM25 + Vector search with multiple fusion strategies
- **AST Analysis** — Python/Java code understanding
- **RAGIX-web** — Full graphical interface

---

## 📋 What's New in v0.33.0

| Category | Feature | Description |
|----------|---------|-------------|
| **Web UI** | RAG Router | `/api/rag` endpoints for index management |
| **Web UI** | Threads Router | `/api/threads` for multi-conversation support |
| **Web UI** | Cancellation | Interrupt long-running LLM operations |
| **Web UI** | RAG Context Display | Show retrieved documents in chat |
| **Core** | Session Management | Full session lifecycle with memory |
| **Core** | Global Context | Inject persistent context across messages |
| **Research** | Contractive Reasoning | Entropy-based tree decomposition |
| **Research** | Peer Review | Multi-model answer validation |

**Version History:**
- **v0.30** — Reasoning Graph architecture
- **v0.31** — Agent specialization (code/doc/git/test)
- **v0.32** — Hybrid RAG, AST analysis
- **v0.33** — Sessions, threads, RAG router, experimental reasoning

---

## ✨ Core Architecture (Production)

### 1. **Reflective Reasoning Graph (reasoning_v30/)**

The production reasoning engine implements a **state machine graph** with specialized nodes:

```
CLASSIFY → DIRECT_EXEC (bypass/simple)
    ↓
   PLAN → EXECUTE → REFLECT → VERIFY → RESPOND
              ↑         ↓
              └─────────┘ (retry loop)
```

| Node | Function |
|------|----------|
| **CLASSIFY** | Route by task complexity: BYPASS / SIMPLE / MODERATE / COMPLEX |
| **DIRECT_EXEC** | Handle simple tasks without formal planning |
| **PLAN** | Generate structured execution plans |
| **EXECUTE** | Execute plan steps with tool calls |
| **REFLECT** | Diagnose failures, generate revised plans |
| **VERIFY** | Validate complex task results |
| **RESPOND** | Format final response, emit events |

**Key features:**
- Full execution trace for debugging
- Event emission for experience corpus
- Configurable iteration limits
- Tool result integration

### 2. **Shell Sandbox (Claude Code-style)**

```python
from ragix_core.tools_shell import ShellSandbox

sandbox = ShellSandbox(
    root="/workspace",
    dry_run=False,
    profile="dev",
    allow_git_destructive=False
)

result = sandbox.run("grep -r 'def main' .")
```

**Safety features:**
- Confines execution to sandbox root
- Hard denylist (DANGEROUS_PATTERNS)
- Blocks destructive git commands by default
- Full command logging with timestamps
- Dry-run mode for testing

### 3. **Hybrid RAG Retrieval**

```python
from ragix_core.hybrid_search import HybridSearchEngine, FusionStrategy

engine = HybridSearchEngine(
    bm25_index=bm25_idx,
    vector_index=vec_idx,
    fusion_strategy=FusionStrategy.RRF,  # Reciprocal Rank Fusion
)

results = engine.search("authentication middleware", top_k=10)
```

**Fusion strategies:**
| Strategy | Description |
|----------|-------------|
| `RRF` | Reciprocal Rank Fusion (default) |
| `WEIGHTED` | Weighted score combination |
| `INTERLEAVE` | Round-robin interleaving |
| `BM25_RERANK` | Vector search, BM25 rerank |
| `VECTOR_RERANK` | BM25 search, vector rerank |

### 4. **AST Analysis (Python + Java)**

```python
from ragix_core.ast_python import PythonASTBackend

backend = PythonASTBackend()
ast_tree = backend.parse_file(Path("src/main.py"))

# Extract classes, functions, imports, decorators, type annotations
for node in ast_tree.children:
    print(f"{node.node_type}: {node.name}")
```

**Extracts:**
- Classes with inheritance hierarchy
- Functions and methods
- Imports (import and from...import)
- Decorators
- Type annotations
- Docstrings
- Module-level constants

---

## ✨ RAGIX-web: Full Graphical Interface

RAGIX-web is the **flagship interface** — a production-ready FastAPI application with WebSocket real-time communication.

### Architecture

```
ragix_web/
├── server.py              # FastAPI server (50k+ lines)
├── routers/               # Modular API endpoints
│   ├── agents.py          # Agent configuration & personas
│   ├── reasoning.py       # Reasoning graph state & control
│   ├── rag.py             # RAG index management (v0.33)
│   ├── sessions.py        # Session lifecycle management
│   ├── memory.py          # Message history & context
│   ├── context.py         # User context injection
│   ├── threads.py         # Multi-thread conversations (v0.33)
│   └── logs.py            # Audit log access
│
└── static/                # Frontend application
    ├── app.js             # Main WebSocket client
    ├── workflow.js        # D3.js reasoning graph visualization
    ├── diff.js            # Side-by-side diff viewer
    ├── files.js           # File browser
    ├── logs.js            # Log viewer
    └── js/
        ├── dependency_explorer.js  # Force-directed dependency graph
        ├── virtual_fs.js           # In-browser filesystem
        ├── browser_tools.js        # Browser-based tools
        └── wasp_runtime.js         # WASP workflow runtime
```

### REST API Endpoints

| Router | Prefix | Features |
|--------|--------|----------|
| **Sessions** | `/api/sessions` | Create/list/delete sessions, model selection |
| **Memory** | `/api/sessions/{id}/memory` | View/edit/delete message history |
| **Context** | `/api/sessions/{id}/context` | Global context injection |
| **Agents** | `/api/agents` | Agent config, model assignment, personas |
| **Reasoning** | `/api/reasoning` | Graph state, step control, experience corpus |
| **RAG** | `/api/rag` | Index status, document upload, enable/disable |
| **Threads** | `/api/threads` | Multi-thread management, thread switching |
| **Logs** | `/api/logs` | Audit log browsing, command history |

### WebSocket Real-Time Features

```javascript
// Connect to session
ws = new WebSocket(`ws://localhost:8000/ws/chat/${sessionId}`);

// Message types
- 'user_message'   // User input
- 'agent_message'  // Agent response
- 'thinking'       // Progress indicator (cancellable)
- 'rag_context'    // RAG retrieval notification
- 'cancel_ack'     // Cancellation confirmed
- 'status'         // Connection status
```

### Interactive Visualizations

**Workflow Visualizer (D3.js):**
- Force-directed graph of reasoning nodes
- Real-time status colors (pending/running/completed/failed)
- Zoom/pan navigation
- Node inspection on click

**Diff Viewer:**
- Split and unified view modes
- Accept/reject changes
- Syntax highlighting
- Multi-file navigation

**Dependency Explorer:**
- Package clustering
- Edge bundling for clarity
- Search and filter
- Export to SVG/PNG

### Optional Integrations

| Integration | Availability | Purpose |
|-------------|--------------|---------|
| **AST Analysis** | Optional | Python/Java code graphs, metrics |
| **Maven Parser** | Optional | POM.xml analysis, dependency conflicts |
| **SonarQube** | Optional | Code quality reports |
| **Prompt Database** | Optional | Reusable prompt templates |
| **Workflow Templates** | Optional | Predefined task workflows |

### Security Model

- **Launch Directory Confinement:** Sandbox restricted to server start directory
- **CORS:** Enabled for local development (localhost only)
- **Session Isolation:** Each session has isolated state
- **Cancellation Support:** Background tasks can be interrupted

### Quick Start

```bash
# Start the web server
ragix-web

# With custom port and sandbox
ragix-web --port 8080 --sandbox ./my-project

# Open browser
http://localhost:8000
```

---

## ✨ Unix-RAG Agent

The original RAGIX pattern — lightweight Unix-based retrieval with LLM orchestration:

```python
from ragix_unix import UnixRAGAgent

agent = UnixRAGAgent(
    llm=ollama_llm,
    sandbox=ShellSandbox(root="./project"),
)

# Uses grep, find, sed, awk for retrieval — no vector DB required
response = agent.run("Find all functions that handle authentication")
```

**Unix-RAG Tools:**
| Tool | Command | Purpose |
|------|---------|---------|
| `grep` | `grep -r -n` | Content search with line numbers |
| `find` | `find -name` | File discovery by pattern |
| `head/tail` | `head -n`, `tail -n` | View file portions |
| `sed` | `sed -n 'X,Yp'` | Extract line ranges |
| `wc` | `wc -l` | Count lines |
| `awk` | `awk '{...}'` | Field extraction |

**Key advantages:**
- Zero-setup (no embedding models, no vector DB)
- Works on any Unix system
- Predictable, auditable retrieval
- Low memory footprint

---

## ✨ Agent Framework

### Specialized Agents

| Agent | Purpose |
|-------|---------|
| `code_agent` | Code generation, refactoring, bug fixing |
| `doc_agent` | Documentation generation and analysis |
| `git_agent` | Git operations, commit analysis |
| `test_agent` | Test generation and execution |

### Agent Configuration

```python
from ragix_core.agent_config import AgentConfig

config = AgentConfig(
    model="mistral:7b-instruct",
    max_iterations=10,
    reflection_enabled=True,
    experience_corpus_path="./experience",
)
```

### Experience Corpus

The system learns from past executions:

```python
from ragix_core.experience_corpus import ExperienceCorpus

corpus = ExperienceCorpus("./experience")
similar = corpus.find_similar(goal="implement authentication", top_k=3)
```

---

## ✨ CLI Improvements

```bash
# Run reasoning on a goal
ragix reason "Implement user authentication with JWT"

# Interactive shell mode
ragix shell --sandbox /workspace

# Profile a model
ragix profile model mistral:7b-instruct

# Export reasoning trace
ragix reason -o trace.json "Fix the bug in login.py"
```

---

## 🧪 Experimental: Contractive Reasoning (reasoning_slim/)

For **research purposes**, we include an experimental entropy-based reasoning engine:

```bash
cd ragix_core/reasoning_slim
python ContractiveReasoner.py "Complex question" --entropy-decompose-threshold 0.3
```

**Features:**
- Entropy-based decomposition
- BM25 relevance pruning
- Optional peer review (multi-model validation)
- Mermaid diagram export

**Note:** This is experimental research code, not production-ready.

---

# 📦 How to Use

## Prerequisites

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull models
ollama serve &
ollama pull mistral:7b-instruct
ollama pull granite3.1-moe:3b

# 3. Install RAGIX
pip install -e .
```

## Quick Start: Web Interface

```bash
ragix-web
# Open http://localhost:8000
```

## Quick Start: CLI Agent

```bash
# Interactive shell with sandbox
ragix shell --sandbox ./workspace

# Execute a goal
ragix reason "Add input validation to the user registration form"
```

## Quick Start: Python API

```python
import asyncio
from ragix_core.reasoning_v30 import ReasoningGraph, create_default_graph
from ragix_core.tools_shell import ShellSandbox

async def main():
    # Create sandbox
    sandbox = ShellSandbox(root="./workspace", dry_run=False, profile="dev")

    # Create reasoning graph
    graph = create_default_graph(
        llm_fn=my_llm_function,
        tool_fn=sandbox.run,
    )

    # Execute
    result = await graph.run("Implement a REST API endpoint for user login")
    print(result.final_answer)

asyncio.run(main())
```

## Configuration File

```yaml
# ragix.yaml
model: mistral:7b-instruct
sandbox:
  root: ./workspace
  profile: dev
  allow_git_destructive: false

reasoning:
  max_iterations: 10
  reflection_enabled: true

rag:
  fusion_strategy: rrf
  bm25_weight: 0.5
  vector_weight: 0.5
```

---

# 🧭 Roadmap

## Completed in 0.33.0

| Feature | Status |
|---------|--------|
| Reflective Reasoning Graph (v0.30) | ✅ Production |
| Shell Sandbox (Claude Code-style) | ✅ Production |
| Hybrid RAG (BM25 + Vector) | ✅ Production |
| AST Analysis (Python + Java) | ✅ Production |
| RAGIX-web graphical interface | ✅ Production |
| Agent specialization (code/doc/git/test) | ✅ Production |
| Experience corpus & learning | ✅ Production |
| Contractive reasoning (experimental) | ✅ Research |

## v0.34.0 — Interactive Reasoning

| Feature | Priority |
|---------|----------|
| Clarification nodes (LLM asks user questions) | High |
| Human-in-the-loop approval before tool execution | High |
| Branch injection (user adds sub-goals mid-reasoning) | Medium |

## v0.35.0 — Multi-Agent Orchestration

| Feature | Priority |
|---------|----------|
| Agent-to-agent debate | High |
| Specialist routing (domain-specific models) | Medium |
| Parallel agent execution | Medium |

## v0.36.0 — Memory & Persistence

| Feature | Priority |
|---------|----------|
| Long-term memory integration | High |
| Session persistence (resume interrupted reasoning) | Medium |
| Timeline view of reasoning evolution | Medium |

## Future (v0.40+)

- Collaborative multi-user sessions (web)
- Fine-tuning integration for domain adaptation
- Formal verification of reasoning chains
- Distributed reasoning across GPU clusters

---

# 🏗️ Architecture Overview

```
RAGIX v0.33.0
├── ragix_core/
│   ├── reasoning_v30/          # Production reasoning graph
│   │   ├── graph.py            # Graph orchestration
│   │   ├── nodes.py            # CLASSIFY, PLAN, EXECUTE, REFLECT, VERIFY
│   │   ├── types.py            # ReasoningState, Plan, ToolCall
│   │   └── experience.py       # Experience corpus
│   │
│   ├── tools_shell.py          # Shell sandbox (Claude Code-style)
│   ├── orchestrator.py         # Action protocol & JSON extraction
│   ├── profiles.py             # Safety profiles & patterns
│   │
│   ├── hybrid_search.py        # BM25 + Vector fusion
│   ├── bm25_index.py           # Keyword search
│   ├── vector_index.py         # Semantic search
│   ├── embeddings.py           # Embedding backends
│   │
│   ├── ast_base.py             # Unified AST interface
│   ├── ast_python.py           # Python parser
│   ├── ast_java.py             # Java parser
│   ├── ast_query.py            # AST queries
│   │
│   ├── agents/                 # Specialized agents
│   │   ├── code_agent.py
│   │   ├── doc_agent.py
│   │   ├── git_agent.py
│   │   └── test_agent.py
│   │
│   ├── maven/                  # Maven integration (optional)
│   ├── sonar/                  # SonarQube integration (optional)
│   │
│   └── reasoning_slim/         # Experimental contractive reasoning
│       ├── ContractiveReasoner.py
│       ├── peer_review.py
│       └── run_production.py   # Benchmark runner
│
├── ragix_web/                  # Full graphical interface
│   ├── server.py               # FastAPI + WebSocket server
│   ├── routers/                # Modular REST endpoints
│   │   ├── agents.py           # /api/agents
│   │   ├── reasoning.py        # /api/reasoning
│   │   ├── rag.py              # /api/rag (v0.33)
│   │   ├── sessions.py         # /api/sessions
│   │   ├── memory.py           # /api/sessions/{id}/memory
│   │   ├── context.py          # /api/sessions/{id}/context
│   │   ├── threads.py          # /api/threads (v0.33)
│   │   └── logs.py             # /api/logs
│   └── static/                 # Frontend (D3.js, WebSocket client)
│
├── ragix_unix/                 # Unix-RAG agent
│   └── unix_rag_agent.py       # grep/find/sed-based retrieval
│
└── demos/                      # Example applications
```

---

# 📊 Performance Characteristics

| Component | Typical Latency | Notes |
|-----------|----------------|-------|
| CLASSIFY node | <100ms | LLM classification |
| DIRECT_EXEC | 1-3s | Simple tool + response |
| PLAN node | 2-5s | Structured plan generation |
| EXECUTE step | 0.5-10s | Depends on tool |
| Hybrid search | <50ms | BM25 + Vector fusion |
| AST parse | <100ms/file | Python/Java |

---

# 🙏 Acknowledgments

RAGIX 0.33.0 represents months of development at the **Adservio Innovation Lab**.

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr

Special thanks to:
- The Ollama team for enabling local LLM deployment
- The open-source community for BM25 and embedding models
