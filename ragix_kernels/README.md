# KOAS — Kernel-Orchestrated Audit System

**A Sovereign, LLM-Agnostic Platform for AI-Assisted Analysis**

**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Version:** 1.4.0
**Date:** 2026-02-09

KOAS brings the Virtual Hybrid Lab paradigm to automated analysis:
- **Kernels compute** — Fast, deterministic, scalable
- **LLMs orchestrate** — Strategic decisions, synthesis, interpretation
- **RAG provides context** — Domain knowledge, patterns, standards

## Quick Start

```bash
# Install RAGIX with KOAS support
pip install -e ".[koas]"

# Code Audit
ragix-koas init --project /path/to/your/project --name "My Project" --language python
ragix-koas run --workspace /tmp/koas_my_project --all

# Document Analysis (see docs/KOAS_DOCS.md)
python run_doc_koas.py --project /path/to/project/src --language fr
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        KOAS APPROACH                                │
│  Kernel computes → LLM reads summary → Kernel computes → LLM writes│
│  ✅ Fast, efficient, unlimited scale, fully reproducible            │
└─────────────────────────────────────────────────────────────────────┘
```

### Three-Stage Pipeline

| Stage | Purpose | Description |
|-------|---------|-------------|
| **1** | Collection | Extract raw data (AST, metadata, structure) |
| **2** | Analysis | Derive insights (metrics, clusters, risks) |
| **3** | Reporting | Generate reports and summaries |

---

## Kernel Categories

KOAS provides four kernel families for different analysis domains:

| Category | Module | Purpose | Kernels |
|----------|--------|---------|---------|
| **audit** | `ragix_kernels.audit` | Code quality analysis | 27 kernels |
| **docs** | `ragix_kernels.docs` | Document summarization | 17 kernels |
| **reviewer** | `ragix_kernels.reviewer` | Traceable Markdown review | 13 kernels |
| **security** | `ragix_kernels.security` | Network/infrastructure security | 10 kernels |

---

## Code Audit Kernels (`ragix_kernels.audit`)

For source code analysis (Java, Python, etc.)

### Stage 1: Data Collection

| Kernel | Description | Output |
|--------|-------------|--------|
| `ast_scan` (v1.1) | AST extraction, symbols, annotations, CC | Classes, methods, fields |
| `metrics` | Code metrics (CC, LOC, MI) | Per-file metrics |
| `dependency` | Dependency graph construction | Import relationships |
| `partition` | Codebase partitioning | Component clusters |
| `services` | Service detection | Controllers, services, repos |
| `timeline` | Component lifecycle profiles | Modification history |
| `volumetry` | Size and distribution analysis | LOC distributions |
| `module_group` | Module grouping | Logical boundaries |
| `maven_deps` | Maven pom.xml extraction | Modules, dependencies, versions |

### Stage 2: Analysis

| Kernel | Description | Output |
|--------|-------------|--------|
| `stats_summary` | Statistical aggregation | Mean, median, outliers |
| `hotspots` | Complexity hotspot identification | High-risk areas |
| `dead_code` | Dead code detection | Unreachable elements |
| `coupling` | Martin coupling metrics | Ca, Ce, I, A, D |
| `entropy` | Information-theoretic analysis | Token entropy |
| `risk` | Service Life Risk assessment | Risk levels |
| `drift` | Spec-code alignment | Synchronization status |
| `risk_matrix` | Risk matrix generation | Risk visualization |
| `maven_graph` | Maven module graph + visualization | Centrality, cycles, SVG/PNG |
| `maven_cve` | Dependency vulnerability scan | CVE findings, severity |
| `spring_wiring` | Spring DI resolution | Beans, entry points, wiring |

### Stage 3: Reporting

| Kernel | Description | Output |
|--------|-------------|--------|
| `section_executive` | Executive summary | Key findings |
| `section_overview` | Codebase overview | Structure, distributions |
| `section_drift` | Drift analysis | Alignment tables |
| `section_maven` | Maven dependencies & supply chain | Modules, graph, CVEs |
| `section_spring` | Spring architecture & entry points | Beans, wiring, dead code |
| `section_recommendations` | Prioritized recommendations | Action plan |
| `report_assemble` | Final assembly | audit_report.md |

---

## Document Analysis Kernels (`ragix_kernels.docs`)

For specification and document corpus analysis. See [KOAS_DOCS.md](../docs/KOAS_DOCS.md) for full documentation.

### Stage 1: Collection

| Kernel | Description | LLM | Output |
|--------|-------------|-----|--------|
| `doc_metadata` | Document inventory from RAG | No | File list, statistics |
| `doc_concepts` | Concept extraction from knowledge graph | No | Concept hierarchy |
| `doc_structure` | Section/heading detection | No | Document outlines |

### Stage 2: Analysis

| Kernel | Description | LLM | Output |
|--------|-------------|-----|--------|
| `doc_cluster` | Hierarchical document clustering | No | Document groups |
| `doc_cluster_leiden` | Leiden community detection clustering | No | Optimized partitions |
| `doc_cluster_reconcile` | Reconcile hierarchical + Leiden clusters | No | Unified clusters |
| `doc_extract` (v1.2) | Quality-filtered sentence extraction | No | Key sentences |
| `doc_coverage` | Concept coverage analysis | No | Gaps, overlaps |
| `doc_quality` | Document quality scoring | No | Quality metrics |
| `doc_func_extract` | SPD functionality extraction | **Yes** | Structured functionalities |

### Stage 3: Synthesis

| Kernel | Description | LLM | Output |
|--------|-------------|-----|--------|
| `doc_compare` | Inter-document discrepancy detection | No | Discrepancies |
| `doc_pyramid` | Hierarchical summary structure | No | 4-level pyramid |
| `doc_report_assemble` | Report generation | No | doc_report.md |
| `doc_summarize` | Per-document LLM summaries | **Yes** | Natural language summaries |
| `doc_summarize_tutored` | Worker+Tutor LLM summaries | **Yes** | Refined summaries |
| `doc_visualize` | Generate word clouds and charts | No | PNG visualizations |
| `doc_final_report` | Comprehensive report with appendices | No | final_report.md + A-F appendices |

### Report Appendices (doc_final_report)

The `doc_final_report` kernel generates six appendices:

| Appendix | Content |
|----------|---------|
| **A** | Corpus Summary — file inventory, statistics by type |
| **B** | Domain Summaries — per-domain concept analysis |
| **C** | Functionality Catalog — extracted SPD functionalities |
| **D** | Discrepancy Details — content overlap, terminology variations, version references |
| **E** | Clustering Analysis — document clusters with file names |
| **F** | Artifacts Catalog — links to visualizations (word clouds, graphs) |

Appendices D and E display **human-readable file names** instead of internal IDs.

---

## Security Kernels (`ragix_kernels.security`)

For network and infrastructure security analysis.

### Stage 1: Discovery

| Kernel | Description | Output |
|--------|-------------|--------|
| `net_discover` | Network topology discovery | Host inventory |
| `port_scan` | Port and service detection | Open ports, services |
| `dns_enum` | DNS enumeration | DNS records |
| `config_parse` | Configuration file parsing | Security settings |

### Stage 2: Analysis

| Kernel | Description | Output |
|--------|-------------|--------|
| `ssl_analysis` | TLS/SSL certificate analysis | Certificate status |
| `vuln_assess` | Vulnerability assessment | CVE matches |
| `web_scan` | Web application scanning | Web vulnerabilities |
| `compliance` | Compliance checking | Policy violations |
| `risk_network` | Network risk scoring | Risk levels |

### Stage 3: Reporting

| Kernel | Description | Output |
|--------|-------------|--------|
| `section_security` | Security report section | Findings summary |

---

## Document Reviewer Kernels (`ragix_kernels.reviewer`)

For traceable, reversible review of large Markdown documents (beyond LLM context windows). See [KOAS_REVIEW.md](../docs/KOAS_REVIEW.md) for full documentation.

### Stage 1: Collection (deterministic)

| Kernel | Description | Output |
|--------|-------------|--------|
| `md_inventory` | File stats, SHA-256, front-matter detection | Document metadata |
| `md_structure` | Heading tree, anchors, numbering patterns | Section hierarchy |
| `md_protected_regions` | Code fences, inline code, YAML, tables, math | Protected spans |
| `md_chunk` | Structure-aligned chunk plan with hash-stable IDs | Chunk inventory |

### Stage 2: Analysis (deterministic + LLM edge)

| Kernel | Description | LLM | Output |
|--------|-------------|-----|--------|
| `md_consistency_scan` (v2.1) | AI leftovers, duplicates, broken refs, tables, term drift | No | Issue list |
| `md_numbering_control` | Heading/figure/table numbering validation | No | Numbering issues |
| `md_fingerprint_chunk` | Structural fingerprint for content-recipe routing | No | ChunkFingerprint |
| `md_pyramid` | Bottom-up hierarchical summaries | No | Multi-level abstracts |
| `md_edit_plan` (v2.4) | Constrained edit ops per chunk with preflight pipeline | **Yes** | Structured edit ops |

### Stage 3: Reporting (deterministic)

| Kernel | Description | Output |
|--------|-------------|--------|
| `md_apply_ops` (v1.1) | Validate + apply edit ops + forward/inverse patches | Patched document |
| `md_inline_notes_inject` | GitHub alert blocks (REVIEWER: prefix) | Annotated document |
| `md_review_report_assemble` | Generate REVIEW_doc.md from ledger | Review report |
| `md_revert` | Selective inverse-patch application by change ID | Reverted document |

### Key Features

- **Preflight pipeline** (v7+): Math masking, sub-chunk splitting, context tiering, content recipes, markdown stripping — all deterministic transforms before LLM call
- **Adaptive tier escalation**: Automatic retry with skeleton/full context based on degenerate detection
- **Content recipes**: Fingerprint-driven masking (tables, blockquotes, emojis, code fences, math) to maximize LLM compliance
- **Full traceability**: Append-only JSONL ledger, RVW-NNNN change IDs, SHA-256 content hashing, forward + inverse patches
- **Selective revert**: Undo individual changes by ID without affecting others
- **CLI**: `reviewctl` with review/report/revert/show/grep subcommands
- **MCP**: 4 tools via `register_reviewer_tools(mcp_server)`

---

## CLI Usage

```bash
# Initialize workspace
ragix-koas init --project /path/to/project --name "Project Name" --language python

# Run specific stage
ragix-koas run --workspace /path/to/workspace --stage 1

# Run all stages
ragix-koas run --workspace /path/to/workspace --all

# Run with cache (skip LLM kernels if outputs exist)
ragix-koas run --workspace /path/to/workspace --all --use-cache

# Get summary
ragix-koas summary --workspace /path/to/workspace

# List available kernels
ragix-koas list
ragix-koas list --stage 1
ragix-koas list --category docs
```

### Deterministic Execution (--use-cache)

The `--use-cache` flag enables deterministic re-execution by reusing cached kernel outputs:

```bash
# First run: all kernels execute (includes LLM calls)
ragix-koas run --workspace ./ws --all
# → Stage 3: 13 kernels in 45.2s

# Subsequent runs: skip kernels with cached outputs
ragix-koas run --workspace ./ws --all --use-cache
# → Stage 3: 13 kernels in 0ms (all cached)

# Regenerate single kernel: delete its cache, then run
rm ./ws/stage3/doc_final_report.json
ragix-koas run --workspace ./ws --stage 3 --use-cache
# → 12 cached + 1 regenerated
```

Benefits:
- **Reproducibility**: Same inputs always produce same outputs
- **Speed**: Skip expensive LLM calls on unchanged data
- **Iteration**: Modify one kernel without re-running the entire pipeline

## Configuration

### Manifest (manifest.yaml)

```yaml
audit:
  name: "My Project Technical Audit"
  version: "1.0"

project:
  name: "My Project"
  path: "/path/to/project"
  language: "python"

output:
  format: "markdown"
  template: "default"
  language: "fr"  # or "en"

stage1:
  ast_scan:
    enabled: true
  metrics:
    enabled: true
    options:
      complexity_threshold: 10

stage2:
  hotspots:
    enabled: true
    options:
      top_n: 50
      threshold_cc: 15

sovereignty:
  mode: local
  audit_trail: required
```

## Language Support (i18n)

KOAS supports English and French report generation:

```yaml
output:
  language: "fr"  # Generates French report
```

French output includes:
- Synthèse Exécutive
- Vue d'Ensemble du Code Source
- Recommandations priorisées
- Évaluation des Risques

## MCP Integration

KOAS tools are exposed via the RAGIX MCP server:

```python
# Available MCP tools
koas_init(project_path, project_name, language, output_language)
koas_run(workspace, stage, kernels)
koas_status(workspace)
koas_summary(workspace, stage)
koas_list_kernels(stage, category)
koas_report(workspace, max_chars)
```

## Design Principles

### 1. Local-First, Sovereignty by Default
All computation happens locally. Data never leaves the infrastructure.

### 2. LLM-Agnostic Architecture
Works with ANY LLM: Ollama (local), Claude, GPT-4, etc.

### 3. Kernel-Centric Computation
Kernels are pure computation units — no LLM inside (except explicitly marked):
- Accept structured input (JSON/YAML)
- Produce structured output + human-readable summary
- Fully deterministic and reproducible

### 4. Reuse Over Reimplementation
KOAS wraps existing RAGIX tools rather than reimplementing.

### 5. Dual LLM Architecture (Worker + Tutor)

LLM-enabled kernels use a **two-model approach**:

```
┌────────────────────────────────────────────────────────┐
│                 Dual LLM Pattern                        │
│                                                         │
│  Input → Worker LLM (fast, small) → Draft Output       │
│              ↓                                          │
│         Tutor LLM (larger, smarter) → Refined Output   │
└────────────────────────────────────────────────────────┘
```

| Role | Model | Purpose |
|------|-------|---------|
| **Worker** | `granite3.1-moe:3b` | Fast extraction, initial processing |
| **Tutor** | `mistral:7b-instruct` | Quality refinement, validation |

Benefits:
- **Speed**: Small model handles bulk work
- **Quality**: Larger model refines results
- **Cost**: Minimizes expensive model usage

### 6. Dual Reconstruction (Pyramidal + Leiden)

Document clustering uses two complementary algorithms:

| Algorithm | Purpose | Output |
|-----------|---------|--------|
| **Hierarchical (Pyramidal)** | Bottom-up aggregation | Document → Group → Domain → Corpus |
| **Leiden Community Detection** | Graph-based clustering | Optimized partitions from similarity graph |

The combination provides robust document organization regardless of corpus structure.

## Kernel Development

Create a new kernel by extending the base class:

```python
from ragix_kernels.base import Kernel, KernelInput
from typing import Dict, Any

class MyKernel(Kernel):
    name = "my_kernel"
    version = "1.0.0"
    category = "audit"  # or "docs", "security"
    stage = 2

    requires = ["ast_scan", "metrics"]  # Dependencies
    provides = ["my_analysis"]          # Capabilities

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load dependencies
        ast_data = self._load_dependency(input, "ast_scan")

        # Pure computation here
        result = analyze(ast_data)

        return {"analysis": result}

    def summarize(self, data: Dict[str, Any]) -> str:
        return f"Analyzed {len(data['analysis'])} items."
```

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Codebase support | 500K+ LOC without context issues |
| Document support | 1000+ documents |
| Report length | 50+ pages, modular sections |
| Execution time | Stage 1+2 < 5 minutes for 1000 classes |
| LLM efficiency | < 10% time in LLM for code audit |
| Reproducibility | Same input = same output (deterministic) |
| Auditability | Full SHA256 chain of all operations |

## Documentation

| Document | Description |
|----------|-------------|
| [KOAS.md](../docs/KOAS.md) | Philosophy and code audit details |
| [KOAS_DOCS.md](../docs/KOAS_DOCS.md) | Document summarization system |
| [KOAS_REVIEW.md](../docs/KOAS_REVIEW.md) | Document reviewer pipeline |
| [KOAS_MCP_REFERENCE.md](../docs/KOAS_MCP_REFERENCE.md) | MCP tool reference |
| [ROADMAP_KOAS_JAVA_AUDIT.md](../docs/developer/ROADMAP_KOAS_JAVA_AUDIT.md) | Java audit extensions roadmap |

## Author

**Olivier Vitrac, PhD, HDR**
Head of Innovation Lab, Adservio
olivier.vitrac@adservio.fr

---

*"Science is no longer what the model can explain, but what a group of agents can coordinate."*
— Generative Simulation Initiative
