# TODO — RAGIX Roadmap

**Updated:** 2025-12-02 (v0.23.0 Release)
**Reference:** See `ACTION_PLAN.md` for detailed implementation plan

---

## Session Completed (2025-12-02 - v0.23.0)

### Unified Model Inheritance & Web UI Fixes

| Task | Status |
|------|--------|
| **Model Inheritance Hierarchy** - Session → Agent Config → Reasoning | ✅ Done |
| **Fixed Agent Config Router** - `/api/agents/config` reads from `active_sessions` | ✅ Done |
| **MINIMAL Mode Inheritance** - Planner/Worker/Verifier inherit session model | ✅ Done |
| **UI Consistency** - All panels show correct model | ✅ Done |
| **Session Auto-Creation** - Handle server restart gracefully | ✅ Done |
| **Settings Modal Sync** - Session ID consistent between Chat and Settings | ✅ Done |
| **Removed Redundant Settings** - Reasoning model selector now inherits | ✅ Done |
| **Version Bump** - 0.23.0 centralized in `ragix_core/version.py` | ✅ Done |

**Key Files Modified:**
- `ragix_web/routers/agents.py` - Major fix for model inheritance
- `ragix_web/server.py` - Session management and auto-creation
- `ragix_web/static/app.js` - Settings modal fixes
- `ragix_web/static/index.html` - UI consistency updates

**Model Inheritance Architecture:**
```
Session (Settings → Session tab)
    └── Agent Config (in MINIMAL mode, inherits session model)
            └── Reasoning (Planner/Worker/Verifier inherit from Agent Config)
```

---

## Next Steps (v0.24.0) — Reflective Reasoning Graph

**Reference:** See `PLAN_v0.23_REASONING.md` for full specification

### Phase 1: Foundation (Schema + Hardening)

| Task | Effort | Status |
|------|--------|--------|
| **Create `reasoning_types.py`** - Pydantic schemas for Plan, Step, State | 4h | Pending |
| **Harden `execute_step`** - returncode/stderr capture, structured results | 3h | Pending |
| **ReasoningEvent schema** - Unified event for experience corpus | 2h | Pending |

### Phase 2: Reasoning Graph

| Task | Effort | Status |
|------|--------|--------|
| **BaseNode class** - Abstract node with `run(state) -> (state, next)` | 2h | Pending |
| **ReasoningGraph orchestrator** - Graph execution with trace | 3h | Pending |
| **ClassifyNode** - SIMPLE vs MODERATE vs COMPLEX routing | 2h | Pending |
| **PlanNode** - Generate plan with reflection context | 3h | Pending |
| **ExecuteNode** - Step execution with reflection routing | 4h | Pending |

### Phase 3: REFLECT Node with Tool Access

| Task | Effort | Status |
|------|--------|--------|
| **ReflectNode** - Read-only tools (`ls`, `find`, `grep`, etc.) | 4h | Pending |
| **Experience corpus query** - Search past failures | 2h | Pending |
| **Reflection prompt** - Diagnosis + new plan generation | 2h | Pending |
| **ReflectionAttempt tracking** - Record each attempt | 1h | Pending |

### Phase 4: Hybrid Experience Corpus

| Task | Effort | Status |
|------|--------|--------|
| **ExperienceCorpus class** - JSONL storage with TTL pruning | 3h | Pending |
| **HybridExperienceCorpus** - Global (`~/.ragix/`) + Project (`.ragix/`) | 3h | Pending |
| **Keyword + recency search** - Simple retrieval for LLM context | 2h | Pending |
| **Success bonus scoring** - Prefer successful recoveries | 1h | Pending |

### Phase 5: Graceful Degradation

| Task | Effort | Status |
|------|--------|--------|
| **RespondNode** - Build attempt summary on max_reflections | 2h | Pending |
| **Recommendation generator** - Pattern-based suggestions | 2h | Pending |
| **Partial results collector** - Gather successful steps | 1h | Pending |

### Phase 6: Evaluation Harness

| Task | Effort | Status |
|------|--------|--------|
| **Scenario YAML format** - Test case definition | 2h | Pending |
| **Harness runner** - Execute scenarios with metrics | 4h | Pending |
| **Mock repo fixtures** - Test file structures | 2h | Pending |
| **Success metrics** - Plan success rate, recovery rate | 2h | Pending |

**Total Estimated Effort:** ~52 hours

**File Structure for v0.24:**
```
ragix_core/
├── reasoning.py              # Existing - hardened
├── reasoning_types.py        # NEW - Pydantic schemas
├── reasoning_graph.py        # NEW - Graph + nodes
├── reasoning_nodes.py        # NEW - Node implementations
├── experience_corpus.py      # NEW - Hybrid corpus

tests/reasoning/
├── harness.py                # Evaluation runner
├── scenarios/                # YAML test cases
│   ├── file_search.yaml
│   └── code_analysis.yaml
└── test_reasoning_graph.py

~/.ragix/experience/          # Global experience corpus
.ragix/experience/            # Project experience corpus
```

---

## Session Completed (2025-11-28 - v0.20.0)

### Report Generation & Documentation Coverage

| Task | Status |
|------|--------|
| **Report Engine** - Jinja2 templates for PDF/HTML reports | ✅ Done |
| **Executive Summary** - High-level metrics, risks, recommendations | ✅ Done |
| **Technical Audit** - Detailed code metrics and hotspots | ✅ Done |
| **Compliance Report** - Security findings and coverage | ✅ Done |
| **Treemap Visualization** - Package hierarchy by LOC/complexity | ✅ Done |
| **Sunburst Visualization** - Module structure drill-down | ✅ Done |
| **Chord Diagram** - Inter-module dependency visualization | ✅ Done |
| **Maven Integration** - POM parsing in reports | ✅ Done |
| **SonarQube Integration** - Quality gate data in reports | ✅ Done |
| **Documentation Coverage Fix** - Filter placeholder Javadocs | ✅ Done |
| **Separate Doc Metrics** - Class vs Method coverage (50/50 weighted) | ✅ Done |
| **Methods Count Fix** - Include class methods in total | ✅ Done |
| **Web UI Defensive JS** - Handle undefined API responses | ✅ Done |

**Implemented in:**
- `ragix_core/report_engine.py` - ReportEngine, generators, templates
- `ragix_core/ast_viz_advanced.py` - TreemapRenderer, SunburstRenderer, ChordRenderer
- `ragix_core/code_metrics.py` - total_methods, doc_coverage, class_doc_coverage
- `ragix_core/ast_java.py` - _get_javadoc() filters placeholders
- `ragix_unix/ast_cli.py` - New CLI commands (treemap, sunburst, chord, report)
- `ragix_web/server.py` - New API endpoints
- `ragix_web/static/index.html` - New cards for visualizations and reports

**New API Endpoints:**
```
GET /api/ast/treemap?path=...      # Treemap visualization
GET /api/ast/sunburst?path=...     # Sunburst visualization
GET /api/ast/chord?path=...        # Chord diagram
GET /api/ast/maven?path=...        # Maven analysis
GET /api/ast/sonar?url=...         # SonarQube integration
GET /api/ast/cycles?path=...       # Cycle detection
GET /api/reports/generate          # Generate reports
```

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

## Phase 2 (v0.20.0) — Visualization Completion ✅ COMPLETE

### New Visualization Types

| Visualization | Purpose | Effort | Status |
|---------------|---------|--------|--------|
| **Treemap** | Package hierarchy by LOC/complexity | 8h | ✅ Done |
| **Sunburst** | Module structure drill-down | 8h | ✅ Done |
| **Chord Diagram** | Inter-module dependencies | 8h | ✅ Done |

### All Visualizations (Complete)

- [x] Force-directed graph with package clustering
- [x] Dependency Structure Matrix (DSM)
- [x] Radial ego-centric explorer
- [x] Treemap (package hierarchy)
- [x] Sunburst (module drill-down)
- [x] Chord diagram (inter-module deps)
- [x] D3.js interactive HTML

---

## Phase 3 (v0.20.0) — Report Generation ✅ COMPLETE

| Report Type | Format | Effort | Status |
|-------------|--------|--------|--------|
| Executive Summary | PDF/HTML | 8h | ✅ Done |
| Technical Audit | PDF/HTML | 8h | ✅ Done |
| Compliance Report | PDF | 6h | ✅ Done |
| Report Engine + Templates | — | 10h | ✅ Done |
| Maven Integration | — | 4h | ✅ Done |
| SonarQube Integration | — | 4h | ✅ Done |
| Documentation Coverage Fix | — | 4h | ✅ Done |

**Dependencies:** `weasyprint`, `jinja2` (added to pyproject.toml)

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

### Agent Reasoning Improvements (from Gemini 2.5 Review)
- [ ] **Automated Self-Correction Loop** — Verifier failure → Planner re-plan → Re-execute (High priority)
- [ ] **Dynamic Tool Selection** — Agent decides which ragix-* tool to use (High priority)
- [ ] **Structured Episodic Memory** — Store successful workflows for reference (Medium priority)
- [ ] **Confidence Scoring** — Agent outputs confidence (1-10), pauses on low (Medium priority)
- [x] Autonomous multi-step reasoning with self-correction (Planner/Worker/Verifier loop)
- [x] Memory and context persistence across sessions (EpisodicMemory in reasoning.py)
- [ ] Agent specialization profiles (security, performance, refactoring)
- [ ] Inter-agent communication protocol

### Tool Usability Improvements (from Gemini 2.5 Review)
- [ ] **Unified CLI Entry Point** — `ragix ast search` instead of `ragix-ast search` (High priority)
- [ ] **Interactive ragix-ast Mode** — `--interactive` REPL to avoid re-parsing (Medium priority)
- [ ] **Cross-Panel Context** — Click node in graph → show file in code panel (Medium priority)
- [ ] **`ragix config view`** — Show resolved config with sources (High priority)
- [ ] **Visual Workflow Builder** — Drag-and-drop workflow creation (Low priority)

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

## Completed (v0.10.1 → v0.20.0)

### v0.20.0 (Current Release)
- [x] Report Generation (Executive Summary, Technical Audit, Compliance)
- [x] Advanced Visualizations (Treemap, Sunburst, Chord Diagram)
- [x] Maven integration in reports
- [x] SonarQube integration in reports
- [x] Documentation coverage fix (filters placeholder Javadocs)
- [x] Separate class/method doc coverage metrics
- [x] Methods count fix (includes class methods)
- [x] Web UI defensive JavaScript (handles undefined responses)
- [x] New API endpoints (treemap, sunburst, chord, maven, sonar, cycles, reports)

### v0.11.0
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
