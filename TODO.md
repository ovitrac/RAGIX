# TODO — RAGIX Roadmap

**Updated:** 2025-11-28 (Session 5)
**Reference:** See `ACTION_PLAN.md` for detailed implementation plan

---

## TOP PRIORITY — Multi-Agent LLM Configuration (v0.11.0)

### Augmented Reasoning with Light Models (3B/7B)

| Task | Effort | Status |
|------|--------|--------|
| **AgentConfig class** - mode/model per agent | 4h | ✅ Done |
| **Auto-detect installed Ollama models** | 2h | ✅ Done |
| **UI toggle** - Minimal (3B) vs Strict (7B+) mode | 3h | ✅ Done |
| **Validate model size** against requirements | 1h | ✅ Done |
| **Granite 3B persona** - Worker/Verifier prompts | 2h | ✅ Done |
| **ragix.yaml agents section** - schema + loader | 2h | ✅ Done |
| **KnowledgeBase** - Pattern storage for 7B models | 4h | ✅ Done |

**Implemented in:**
- `ragix_core/agent_config.py` - Full AgentConfig, model detection, personas
- `ragix.yaml` - Agent configuration section (lines 139-159)

**Reference:** See `RAGIX_REASONING.md` §8 for full specification

**Agent Architecture:**
```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Planner  │───▶│  Worker  │───▶│ Verifier │
│ (≥7B/3B) │    │   (3B)   │    │   (3B)   │
└──────────┘    └──────────┘    └──────────┘
```

**Modes:**
- **Minimal** (default): All 3B — for 8GB VRAM / CPU
- **Strict**: Planner ≥7B, Worker/Verifier 3B — for 12GB+ VRAM

**Installed Models:**
| Model | Size | Recommended Role |
|-------|------|------------------|
| `granite3.1-moe:3b` | 2.0 GB | Worker, Verifier, Minimal Planner |
| `mistral:latest` | 4.4 GB | Strict Planner |
| `deepseek-r1:14b` | 9.0 GB | Advanced Planner |

---

## Session Completed (2025-11-28 Session 5)

### Knowledge Base & Web UI Consolidation

| Task | Status |
|------|--------|
| **KnowledgeBase system** - Pattern/rule storage for 7B models | ✅ Done |
| **Session Memory viewer** - View/delete/clear message history | ✅ Done |
| **User Context management** - System instructions like Claude/ChatGPT | ✅ Done |
| **Modular routers** - sessions, memory, context, agents, logs | ✅ Done |

**Implemented in:**
- `ragix_core/knowledge_base.py` - CommandPattern, ReasoningRule, KnowledgeBase
- `ragix_core/knowledge_rules.yaml` - Extensible YAML patterns
- `ragix_web/routers/` - Modular API routers
- `ragix_web/static/index.html` - Memory & Context tabs in Settings
- `ragix_web/static/style.css` - Memory/Context panel styles
- `ragix_web/server.py` - Router registration

**New API Endpoints:**
```
GET/DELETE /api/sessions/{session_id}/memory      # Session memory
GET/DELETE /api/sessions/{session_id}/memory/{i}  # Specific message
GET/POST/DELETE /api/sessions/{session_id}/context # User context
```

---

## Session Completed (2025-11-28 Session 4)

### Reasoning Loop & Web UI Improvements

| Task | Status |
|------|--------|
| **Direct conversational responses** (greeting, identity, help) | ✅ Done |
| **Improved Planner prompts** - line count filtering examples | ✅ Done |
| **Improved Worker prompts** - JSON templates for commands | ✅ Done |
| **CommandResult formatting** - clean output with newlines | ✅ Done |
| **JSON copy button** - optional JSON copy in details section | ✅ Done |
| **Collapsible details** - show/hide execution details | ✅ Done |

**Implemented in:**
- `ragix_core/reasoning.py` - Direct responses, improved prompts, formatting
- `ragix_web/static/app.js` - JSON copy button, details toggle
- `ragix_web/static/style.css` - JSON button styling

**Command Examples Added:**
```bash
# Files > N lines
find . -name "*.md" -type f -exec wc -l {} + | grep -v " total$" | awk '$1 > 1000'
# Largest file
find . -name "*.md" -type f -exec wc -l {} + | grep -v " total$" | sort -n | tail -1
```

---

## Session Completed (2025-11-27)

### HTMLRenderer Improvements (ast_viz.py)

| Task | Status |
|------|--------|
| Large Graph Mode (>5000 nodes) - auto-hide packages on load | ✅ Done |
| Package search with OR support (`pkg1\|pkg2\|pkg3`) | ✅ Done |
| Large Graph Mode informational message overlay | ✅ Done |
| Search match highlighting (red border on matches) | ✅ Done |

**Technical Details:**
- Large Graph Mode threshold: 5000 nodes
- Package search: case-insensitive, `|` for OR
- Tested on GRDF codebase (18,210 nodes)

---

## Immediate (v0.11.0) — ragix-web Consolidation

### Critical Priority

| Task | Effort | Status |
|------|--------|--------|
| Fix ragix-web server startup | 2h | Pending |
| Connect trace viewer to log_integrity.py | 4h | Pending |
| Integrate radial explorer into ragix-web | 2h | Pending |
| Add project selector to dashboard | 3h | Pending |
| Dashboard with quick stats from AST | 4h | Pending |

### High Priority

| Task | Effort | Status |
|------|--------|--------|
| Modular router structure (sessions, memory, context, agents, logs) | 6h | ✅ Done |
| WebSocket for live updates | 6h | Pending |
| Full MCP wrapper for rt-* tools | 4h | Pending |
| Add `rt-checksum` and `rt-metadata` tools | 3h | Pending |

---

## Phase 2 (v0.11.1) — Visualization Completion

### New Visualization Types

| Visualization | Purpose | Effort | Status |
|---------------|---------|--------|--------|
| **Treemap** | Package hierarchy by LOC/complexity | 8h | Pending |
| **Sunburst** | Module structure drill-down | 8h | Pending |
| **Chord Diagram** | Inter-module dependencies | 8h | Pending |

### Existing Visualizations (Complete)

- [x] Force-directed graph with package clustering
- [x] Dependency Structure Matrix (DSM)
- [x] Radial ego-centric explorer
- [x] D3.js interactive HTML

---

## Phase 3 (v0.11.2) — Report Generation

| Report Type | Format | Effort | Status |
|-------------|--------|--------|--------|
| Executive Summary | PDF/HTML | 8h | Pending |
| Technical Audit | PDF/HTML | 8h | Pending |
| Compliance Report | PDF | 6h | Pending |
| Report Engine + Templates | — | 10h | Pending |

**Dependencies:** `weasyprint` or `pdfkit`, `jinja2`

---

## Phase 4 (v0.12.0) — Git Integration

| Feature | Description | Effort | Status |
|---------|-------------|--------|--------|
| Complexity Evolution | Track CC over commits | 12h | Pending |
| Hotspot Emergence | Files becoming complex | 8h | Pending |
| Debt Accumulation | Technical debt timeline | 8h | Pending |
| Churn Analysis | Most-changed files | 6h | Pending |

---

## Future (v1.0+)

### Agent Improvements
- [x] Autonomous multi-step reasoning with self-correction (Planner/Worker/Verifier loop)
- [x] Memory and context persistence across sessions (EpisodicMemory in reasoning.py)
- [ ] Agent specialization profiles (security, performance, refactoring)
- [ ] Inter-agent communication protocol

### Tool Enhancements
- [ ] WASM-compiled tools for browser execution
- [ ] AST-aware search (tree-sitter integration)
- [ ] Pyodide sandbox for browser-in-agent (not urgent)

### Integrations
- [ ] VS Code Extension
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

## Completed (v0.10.1 → v0.11.0)

### v0.11.0 (In Progress)
- [x] AgentConfig class with mode/model per agent
- [x] Auto-detect installed Ollama models
- [x] Model size validation against requirements
- [x] Agent persona prompts (Planner/Worker/Verifier)
- [x] ragix.yaml agents section with schema
- [x] EpisodicMemory for session context persistence
- [x] ReasoningLoop (Planner/Worker/Verifier orchestration)
- [x] Direct conversational responses (greetings, identity, help)
- [x] Improved planner/worker prompts with Unix command templates
- [x] CommandResult formatting with proper newlines
- [x] Web UI: JSON copy button, collapsible details
- [x] Web UI: Reasoning traces display
- [x] Web UI: Agent mode selector (Minimal/Strict/Custom)
- [x] KnowledgeBase for 7B model reasoning improvement
- [x] Session memory viewer with selective delete/clear
- [x] User context management (system instructions, preferences)
- [x] Modular router structure (ragix_web/routers/)

### v0.10.1
- [x] Multi-language AST (Python + Java via javalang)
- [x] Dependency graph with cycle detection
- [x] AST query language (pattern-based search)
- [x] Professional code metrics (cyclomatic complexity, technical debt)
- [x] Maven POM parsing
- [x] SonarQube/SonarCloud client
- [x] Enhanced HTML visualization (package clustering, edge bundling)
- [x] Dependency Structure Matrix (DSM) with heatmap
- [x] Radial ego-centric explorer
- [x] Standalone radial server
- [x] 8 AST API endpoints in ragix-web
- [x] CLI: `ragix-ast` with 12 subcommands
- [x] Plugin system with tool/workflow types
- [x] WASP tools (18 deterministic tools)
- [x] SWE chunked workflows
- [x] Unified configuration (ragix.yaml)
- [x] Log integrity (SHA256 hash chain)
- [x] Large Graph Mode for HTMLRenderer (>5000 nodes threshold)
- [x] Package search with OR support in HTMLRenderer

---

## Quick Reference

```bash
# Current CLI capabilities
ragix-ast scan ./project          # Full AST analysis
ragix-ast metrics ./project       # Code quality metrics
ragix-ast graph ./project -f html # Force-directed graph
ragix-ast matrix ./project        # DSM visualization
ragix-ast radial ./project        # Radial explorer

# Live radial server
python -m ragix_unix.radial_server --path ./project --port 8090

# ragix-web (needs consolidation)
ragix-web --port 8080
```

---

*See `ACTION_PLAN.md` for implementation timeline and architecture.*
