# RAGIX v0.20.0 Action Plan

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Date:** 2025-11-28 (v0.20.0 Release)
**Status:** Phase 2 & 3 Complete

---

## Recent Progress (2025-11-28 - v0.20.0 Release)

### Report Generation & Visualization ✅ COMPLETE

**Phase 2 - Advanced Visualizations:**
| Component | Status | Location |
|-----------|--------|----------|
| TreemapRenderer | ✅ Done | `ragix_core/ast_viz_advanced.py` |
| SunburstRenderer | ✅ Done | `ragix_core/ast_viz_advanced.py` |
| ChordRenderer | ✅ Done | `ragix_core/ast_viz_advanced.py` |
| Web UI Cards | ✅ Done | `ragix_web/static/index.html` |
| API Endpoints | ✅ Done | `ragix_web/server.py` |

**Phase 3 - Report Generation:**
| Component | Status | Location |
|-----------|--------|----------|
| ReportEngine | ✅ Done | `ragix_core/report_engine.py` |
| ExecutiveSummaryGenerator | ✅ Done | `ragix_core/report_engine.py` |
| TechnicalAuditGenerator | ✅ Done | `ragix_core/report_engine.py` |
| ComplianceReportGenerator | ✅ Done | `ragix_core/report_engine.py` |
| Maven Integration | ✅ Done | Reports include Maven data |
| SonarQube Integration | ✅ Done | Reports include SonarQube data |

**Documentation Coverage Fix:**
| Issue | Solution | Status |
|-------|----------|--------|
| Methods count = 0 | Added `total_methods` property (includes class methods) | ✅ Done |
| Doc coverage = 100% | Filter placeholder Javadocs ("The Class X") | ✅ Done |
| Misleading scores | Separate class/method doc coverage (50/50 weighted) | ✅ Done |

**Web UI Stability:**
| Fix | Status |
|-----|--------|
| Defensive JS for undefined responses | ✅ Done |
| Logs page flatMap error | ✅ Done |
| formatLogLine includes error | ✅ Done |
| All data arrays with fallbacks | ✅ Done |

---

## Recent Progress (2025-11-28 Session 4)

### Reasoning Loop & Web UI Improvements

**Completed:**
1. **Direct Conversational Responses** - Greetings, identity questions, help now bypass agent
2. **Improved Planner Prompts** - Added templates for line count filtering, file listing
3. **Improved Worker Prompts** - Copy-paste JSON templates for common commands
4. **CommandResult Formatting** - Proper newline handling, truncation for long output
5. **Web UI Enhancements** - JSON copy button, collapsible details section

**Files Modified:**
- `ragix_core/reasoning.py` - Direct responses, command templates, formatting
- `ragix_web/static/app.js` - JSON copy button, details toggle
- `ragix_web/static/style.css` - JSON button styling (yellow accent)

---

## Recent Progress (2025-11-28 Session 3)

### Multi-Agent LLM Configuration — ✅ MOSTLY COMPLETE

**Implementation Status:**

| Task | Status | Location |
|------|--------|----------|
| AgentConfig class | ✅ Done | `ragix_core/agent_config.py` |
| Auto-detect Ollama models | ✅ Done | `detect_ollama_models()` |
| Model size validation | ✅ Done | `AgentConfig.validate()` |
| Granite 3B persona prompts | ✅ Done | `AGENT_PERSONAS` dict |
| ragix.yaml agents section | ✅ Done | `ragix.yaml:139-159` |
| UI toggle Minimal/Strict | ⏳ Pending | ragix-web settings |

**Configuration Format** (`ragix.yaml`):
```yaml
agents:
  mode: minimal  # or "strict" or "custom"
  models:
    planner: granite3.1-moe:3b
    worker: granite3.1-moe:3b
    verifier: granite3.1-moe:3b
```

**Installed Models Detected:**
| Model | Size | Role |
|-------|------|------|
| `granite3.1-moe:3b` | 2.0 GB | All (Minimal mode) |
| `mistral:7b-instruct` | 4.4 GB | Planner (Strict mode) |
| `qwen2.5:7b` | 4.7 GB | Alternative Planner |
| `deepseek-r1:14b` | 9.0 GB | Advanced Planner |

**Remaining Tasks:**
- UI toggle Minimal/Strict (3h)

---

## Recent Progress (2025-11-27 Session 2)

### HTMLRenderer Enhancements (ast_viz.py)

**Completed:**
1. **Large Graph Mode** - Graphs with >5000 nodes now:
   - Start with all packages unchecked/hidden
   - Display informational overlay explaining the mode
   - Visible count shows "0" initially
   - User must select packages to explore

2. **Package Search Feature** - New search box with:
   - Case-insensitive filtering
   - OR support using `|` (e.g., `util|common|helper`)
   - Live match count display
   - Visual highlighting of matching packages (red border)
   - Hidden non-matching packages during search

**Files Modified:**
- `ragix_core/ast_viz.py` (CSS, HTML, JavaScript additions)

**Tested On:**
- GRDF codebase: 18,210 nodes, Large Graph Mode works correctly
- Package search with `print|alert` pattern validated

---

## Executive Summary

This document defines the roadmap to consolidate RAGIX tools through a unified **ragix-web** interface, complete the visualization suite, and establish report generation capabilities.

---

## Current State Assessment

### What We Have (v0.10.1)

| Component | Status | Maturity |
|-----------|--------|----------|
| **ragix-core** | Complete | Production |
| **ragix-unix CLI** | Complete | Production |
| **ragix-ast CLI** | Complete (12 commands) | Production |
| **WASP Tools** | Complete (18 tools) | Production |
| **Plugin System** | Complete | Production |
| **Standalone Radial Server** | Complete | Production |
| **ragix-web** | Partial (v0.6) | Alpha |

### Gap Analysis

```
ragix-web Current State:
├── Viewer          → Not connected
├── Traces          → Not connected
├── AST Endpoints   → Added but untested
├── Search          → Hybrid search ready
├── Agent Chat      → Basic only
├── Workflow Runner → Template-based
└── Visualizations  → Static files only
```

---

## Strategy: Modular Tool Anchoring

### Architecture Vision

```
                    ┌─────────────────────────────────────┐
                    │         ragix-web (v0.11)           │
                    │   Unified Web Interface + API       │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
    ┌─────▼─────┐           ┌─────▼─────┐           ┌─────▼─────┐
    │  Explorer │           │  Analysis │           │  Reports  │
    │   Module  │           │   Module  │           │   Module  │
    └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
          │                        │                        │
    ┌─────┴─────────────────┐     │     ┌─────────────────┴─────┐
    │ - File Browser        │     │     │ - PDF Generation       │
    │ - Code Viewer         │     │     │ - HTML Reports         │
    │ - Search Interface    │     │     │ - Executive Summary    │
    │ - Trace Viewer        │     │     │ - Trend Analysis       │
    └───────────────────────┘     │     └───────────────────────┘
                                  │
    ┌─────────────────────────────┴─────────────────────────────┐
    │                    Visualization Module                    │
    ├───────────────────────────────────────────────────────────┤
    │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ │
    │ │  Force  │ │  DSM    │ │ Radial  │ │Treemap  │ │Sunbur │ │
    │ │ Graph   │ │ Matrix  │ │Explorer │ │         │ │  st   │ │
    │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────┘ │
    └───────────────────────────────────────────────────────────┘
                                  │
    ┌─────────────────────────────┴─────────────────────────────┐
    │                    Backend Services                        │
    ├───────────────────────────────────────────────────────────┤
    │ ragix-core: AST, Dependencies, Metrics, WASP, Plugins     │
    └───────────────────────────────────────────────────────────┘
```

### Modular Design Principles

1. **On-Demand Loading**: Tools loaded only when requested
2. **Lazy Analysis**: Graph/metrics computed on first access, cached
3. **Unified API**: All tools accessible via REST + WebSocket
4. **Progressive Enhancement**: Basic HTML → Interactive D3.js
5. **State Persistence**: Analysis state saved per project

---

## Phase 1: ragix-web Consolidation (v0.11.0)

### 1.1 Unified Dashboard

**Goal:** Single entry point for all RAGIX capabilities

```
/                        → Dashboard (project selector, quick stats)
/explorer                → File browser + code viewer
/search                  → Hybrid search interface
/traces                  → Agent trace viewer
/ast                     → AST analysis hub
  /ast/graph             → Force-directed dependency graph
  /ast/matrix            → DSM visualization
  /ast/radial            → Radial explorer (live)
  /ast/treemap           → Package treemap (NEW)
  /ast/sunburst          → Module sunburst (NEW)
  /ast/metrics           → Code metrics dashboard
  /ast/hotspots          → Complexity hotspots
/reports                 → Report generation
/workflows               → Workflow templates
/settings                → Configuration
```

### 1.2 API Standardization

All endpoints follow pattern:
```
GET  /api/{module}/status     → Module health
GET  /api/{module}/data       → Primary data
POST /api/{module}/action     → Trigger action
WS   /ws/{module}/stream      → Real-time updates
```

**AST Module API (complete):**
```
GET  /api/ast/status              ✅ Exists
GET  /api/ast/graph?path=...      ✅ Exists
GET  /api/ast/metrics?path=...    ✅ Exists
GET  /api/ast/hotspots?path=...   ✅ Exists
GET  /api/ast/search?path=...     ✅ Exists
GET  /api/ast/radial?path=...     ✅ Exists
GET  /api/ast/radial/page?path=...✅ Exists
GET  /api/ast/treemap?path=...    ❌ TODO
GET  /api/ast/sunburst?path=...   ❌ TODO
GET  /api/ast/chord?path=...      ❌ TODO
POST /api/ast/report?path=...     ❌ TODO
```

### 1.3 Implementation Tasks

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Fix ragix-web server startup | High | 2h | None |
| Connect trace viewer to logs | High | 4h | log_integrity.py |
| Integrate radial explorer | High | 2h | radial_server.py |
| Add project selector | High | 3h | None |
| Dashboard with quick stats | Medium | 4h | AST endpoints |
| WebSocket for live updates | Medium | 6h | None |

---

## Phase 2: Visualization Completion (v0.11.1)

### 2.1 Remaining Visualizations

| Visualization | Purpose | Input | Implementation |
|---------------|---------|-------|----------------|
| **Treemap** | Package hierarchy by LOC/complexity | DependencyGraph + Metrics | D3.js treemap |
| **Sunburst** | Module drill-down | DependencyGraph hierarchy | D3.js sunburst |
| **Chord Diagram** | Inter-module dependencies | Package-level adjacency | D3.js chord |

### 2.2 Treemap Implementation

**Use Case:** Visualize which packages have most code, highest complexity, or most debt.

```python
class TreemapRenderer:
    """Generate treemap visualization by package."""

    def build_hierarchy(self, graph: DependencyGraph, metric: str = "loc") -> Dict:
        """Build hierarchical data for treemap.

        Args:
            graph: Dependency graph with symbols
            metric: "loc", "complexity", "debt", or "files"
        """
        # Group symbols by package
        # Calculate metric for each package
        # Return nested hierarchy
```

**API:**
```
GET /api/ast/treemap?path=/project&metric=complexity
```

**Output:** Interactive treemap with:
- Color by metric intensity
- Size by LOC
- Click to zoom into package
- Breadcrumb navigation

### 2.3 Sunburst Implementation

**Use Case:** Explore module structure from outside-in (project → package → class → method).

```python
class SunburstRenderer:
    """Generate sunburst visualization for module hierarchy."""

    def build_hierarchy(self, graph: DependencyGraph) -> Dict:
        """Build radial hierarchy.

        Structure:
        - Level 0: Project root
        - Level 1: Top-level packages
        - Level 2: Sub-packages
        - Level 3: Classes
        - Level 4: Methods (optional)
        """
```

**API:**
```
GET /api/ast/sunburst?path=/project&depth=4
```

### 2.4 Chord Diagram Implementation

**Use Case:** Show which modules depend on which (circular layout, arcs for deps).

```python
class ChordRenderer:
    """Generate chord diagram for inter-module dependencies."""

    def build_matrix(self, graph: DependencyGraph, level: str = "package") -> Dict:
        """Build adjacency matrix for chord diagram.

        Args:
            level: "package" or "module" (top-level directories)
        """
```

**API:**
```
GET /api/ast/chord?path=/project&level=package
```

---

## Phase 3: Report Generation (v0.11.2)

### 3.1 Report Types

| Report | Format | Content | Audience |
|--------|--------|---------|----------|
| **Executive Summary** | PDF/HTML | High-level metrics, risks, recommendations | Management |
| **Technical Audit** | PDF/HTML | Detailed metrics, hotspots, dependencies | Tech leads |
| **Compliance Report** | PDF | Security findings, OWASP, coverage | Auditors |
| **Trend Analysis** | HTML | Metrics over time (requires git integration) | All |

### 3.2 Executive Summary Template

```markdown
# Code Quality Report: {project_name}
Generated: {date}

## Overview
- Files analyzed: {file_count}
- Total symbols: {symbol_count}
- Technical debt: {debt_hours} hours

## Risk Assessment
| Risk Level | Count | Examples |
|------------|-------|----------|
| Critical   | {n}   | {top_3}  |
| High       | {n}   | {top_3}  |
| Medium     | {n}   | ...      |

## Key Findings
1. **Complexity Hotspots**: {top_5_methods}
2. **Circular Dependencies**: {cycle_count} cycles detected
3. **Coupling Issues**: {high_coupling_packages}

## Recommendations
1. {rec_1}
2. {rec_2}
3. {rec_3}

## Visualizations
[Embedded: Dependency Graph, DSM, Treemap]
```

### 3.3 Report Engine

```python
class ReportEngine:
    """Generate PDF/HTML reports from analysis data."""

    def __init__(self, template_dir: Path):
        self.templates = self._load_templates(template_dir)

    def generate(
        self,
        report_type: str,
        graph: DependencyGraph,
        metrics: Dict,
        output_format: str = "html"
    ) -> Path:
        """Generate report.

        Args:
            report_type: "executive", "technical", "compliance"
            output_format: "html", "pdf"
        """
```

**Dependencies:**
- `weasyprint` or `pdfkit` for PDF generation
- `jinja2` for templating
- `matplotlib` for static charts in PDF

---

## Phase 4: Git Integration (v0.12.0)

### 4.1 Features

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **Complexity Evolution** | Track CC over commits | git log + AST analysis per commit |
| **Hotspot Emergence** | Identify files that became complex | Diff analysis |
| **Debt Accumulation** | Technical debt timeline | Metrics per commit |
| **Churn Analysis** | Files changed most frequently | git log --numstat |

### 4.2 API

```
GET /api/git/evolution?path=/project&metric=complexity&commits=100
GET /api/git/hotspots?path=/project&since=2024-01-01
GET /api/git/churn?path=/project&top=20
```

---

## Implementation Timeline

```
Week 1-2: Phase 1 (ragix-web consolidation)
├── Day 1-2: Fix server startup, project selector
├── Day 3-4: Connect trace viewer, dashboard
├── Day 5-7: Integrate existing visualizations
└── Day 8-10: Testing, documentation

Week 3-4: Phase 2 (visualizations)
├── Day 1-3: Treemap implementation
├── Day 4-6: Sunburst implementation
├── Day 7-9: Chord diagram implementation
└── Day 10: Integration testing

Week 5-6: Phase 3 (reports)
├── Day 1-3: Report engine + templates
├── Day 4-6: PDF generation
├── Day 7-10: Executive summary + technical audit

Week 7+: Phase 4 (git integration)
└── Complexity evolution, trend analysis
```

---

## File Structure

```
ragix_web/
├── server.py              # Main FastAPI server
├── routers/               # NEW: Modular routers
│   ├── __init__.py
│   ├── ast.py             # AST endpoints
│   ├── search.py          # Search endpoints
│   ├── traces.py          # Trace viewer endpoints
│   ├── reports.py         # Report generation
│   └── workflows.py       # Workflow endpoints
├── templates/             # Jinja2 templates
│   ├── base.html
│   ├── dashboard.html
│   ├── explorer.html
│   └── reports/
│       ├── executive.html
│       └── technical.html
├── static/
│   ├── js/
│   │   ├── dependency_explorer.js  # Existing
│   │   ├── treemap.js              # NEW
│   │   ├── sunburst.js             # NEW
│   │   └── chord.js                # NEW
│   └── css/
└── report_engine.py       # NEW: Report generator

ragix_core/
├── ast_viz.py             # Add TreemapRenderer, SunburstRenderer, ChordRenderer
└── report_templates/      # NEW: Report templates
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| All tools accessible via ragix-web | 100% | Feature checklist |
| Visualization coverage | 6 types | Graph, DSM, Radial, Treemap, Sunburst, Chord |
| Report generation | 3 types | Executive, Technical, Compliance |
| Documentation coverage | 100% | README + API docs |
| Test coverage | >80% | pytest --cov |

---

## Next Steps (Immediate)

### Multi-Agent LLM Configuration (1 task remaining)
1. ✅ ~~Implement AgentConfig class~~ — Done (`ragix_core/agent_config.py`)
2. ✅ ~~Add Ollama model detection~~ — Done (`detect_ollama_models()`)
3. ⏳ **Add UI toggle** in ragix-web settings page (3h)
4. ✅ ~~Update ragix.yaml schema~~ — Done (lines 139-159)
5. ✅ ~~Create Granite 3B personas~~ — Done (`AGENT_PERSONAS`)

### Reasoning Loop (Done)
6. ✅ Direct conversational responses (greetings, identity, help)
7. ✅ Improved planner/worker prompts with command templates
8. ✅ CommandResult formatting with proper newlines
9. ✅ Web UI JSON copy button and collapsible details

### ragix-web Consolidation (Pending)
10. Update CHANGELOG.md with v0.10.1 → v0.11.0 features
11. Update README.md to v0.11 level
12. Fix ragix-web server.py to start properly
13. Create router structure for modular endpoints
14. Integrate standalone radial server into ragix-web

### Testing & Validation
15. Test reasoning loop with various models (granite, mistral, qwen)
16. Test "list files with > N lines" queries
17. Test conversational queries (hello, who are you, help)
18. Validate JSON copy button functionality

---

## Future Improvements (from Review)

Based on Gemini 2.5 Pro review (`FORREVIEW_IMPROVEMENT_SUGGESTIONS.md`):

### Agent Reasoning Improvements

| Feature | Priority | Effort | Description |
|---------|----------|--------|-------------|
| **Automated Self-Correction Loop** | High | 16h | Verifier failure → Planner re-plan → Re-execute automatically |
| **Dynamic Tool Selection** | High | 12h | Agent decides which ragix-* tool to use for the task |
| **Structured Episodic Memory** | Medium | 16h | Store successful workflows for future reference |
| **Confidence Scoring** | Medium | 8h | Agent outputs confidence (1-10), pauses on low confidence |

### Tool Usability Improvements

| Feature | Priority | Effort | Description |
|---------|----------|--------|-------------|
| **Unified CLI Entry Point** | High | 8h | `ragix ast search` instead of `ragix-ast search` |
| **Interactive ragix-ast Mode** | Medium | 12h | `--interactive` REPL to avoid re-parsing |
| **Cross-Panel Context** | Medium | 12h | Click node in graph → show file in code panel |
| **Visual Workflow Builder** | Low | 24h | Drag-and-drop workflow creation in Web UI |
| **`ragix config view`** | High | 4h | Show resolved config with sources |

### Implementation Notes

**Self-Correction Loop:**
```python
# In graph_executor.py
async def execute_with_retry(self, max_retries: int = 3):
    for attempt in range(max_retries):
        result = await self.execute()
        if result.success:
            return result
        # Feed failure to planner
        self.graph = self.planner.replan(
            original_goal=self.goal,
            failure_report=result.error,
            attempt=attempt + 1
        )
    raise MaxRetriesExceeded()
```

**Confidence Scoring:**
```yaml
# Agent response format
response:
  plan: "..."
  confidence: 7  # 1-10 scale
  reasoning: "Medium confidence due to ambiguous requirements"
```

**Unified CLI:**
```bash
# Current:
ragix-ast scan ./project
ragix-web --port 8080
ragix-batch --template bug_fix

# Proposed:
ragix ast scan ./project
ragix web --port 8080
ragix batch --template bug_fix
ragix config view
```

---

*"Augment reasoning with light models, consolidate infrastructure."*
