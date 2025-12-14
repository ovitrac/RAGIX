# KOAS — Kernel-Orchestrated Audit System

**A Sovereign, LLM-Agnostic Platform for AI-Assisted Code Analysis**

KOAS brings the Virtual Hybrid Lab paradigm to code auditing:
- **Kernels compute** — Fast, deterministic, scalable
- **LLMs orchestrate** — Strategic decisions, synthesis, interpretation
- **RAG provides context** — Domain knowledge, patterns, standards

## Quick Start

```bash
# Install RAGIX with KOAS support
pip install -e ".[koas]"

# Initialize an audit workspace
ragix-koas init --project /path/to/your/project --name "My Project" --language python

# Run all stages
ragix-koas run --workspace /tmp/koas_my_project --all

# View the report
cat /tmp/koas_my_project/stage3/audit_report.md
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

| Stage | Purpose | Kernels |
|-------|---------|---------|
| **1** | Data Collection | ast_scan, metrics, dependency, partition, services, timeline |
| **2** | Analysis | stats_summary, hotspots, dead_code, coupling, entropy, risk |
| **3** | Reporting | section_executive, section_overview, section_recommendations, report_assemble |

## Kernels

### Stage 1: Data Collection

| Kernel | Description | Wraps |
|--------|-------------|-------|
| `ast_scan` | AST extraction and symbol enumeration | ragix-ast scan |
| `metrics` | Code metrics (CC, LOC, MI) | ragix-ast metrics |
| `dependency` | Dependency graph construction | DependencyGraphBuilder |
| `partition` | Codebase partitioning | Partitioner |
| `services` | Service detection | ServiceDetector |
| `timeline` | Component lifecycle profiles | TimelineScanner |

### Stage 2: Analysis

| Kernel | Description | Wraps |
|--------|-------------|-------|
| `stats_summary` | Statistical aggregation | StatisticsComputer |
| `hotspots` | Complexity hotspot identification | metrics data |
| `dead_code` | Dead code detection | DeadCodeDetector |
| `coupling` | Martin coupling metrics | CouplingComputer |
| `entropy` | Information-theoretic analysis | EntropyComputer |
| `risk` | Service Life Risk assessment | RiskScorer |

### Stage 3: Reporting

| Kernel | Description | Output |
|--------|-------------|--------|
| `section_executive` | Executive summary | key findings, metrics |
| `section_overview` | Codebase overview | structure, distributions |
| `section_recommendations` | Prioritized recommendations | action plan |
| `report_assemble` | Final assembly | audit_report.md |

## CLI Usage

```bash
# Initialize workspace
ragix-koas init --project /path/to/project --name "Project Name" --language python

# Run specific stage
ragix-koas run --workspace /path/to/workspace --stage 1

# Run all stages
ragix-koas run --workspace /path/to/workspace --all

# Get summary
ragix-koas summary --workspace /path/to/workspace

# List available kernels
ragix-koas list
ragix-koas list --stage 1
```

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
- Synthese Executive
- Vue d'Ensemble du Code Source
- Recommandations priorisees
- Evaluation des Risques

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

## Shell Scripts

```bash
# Quick single-project audit
./scripts/koas-quick-audit.sh /path/to/project "My Project" --fr

# Batch audit multiple projects
./scripts/koas-batch-audit.sh projects.txt --lang fr --output-dir ./reports
```

## Design Principles

### 1. Local-First, Sovereignty by Default
All computation happens locally. Data never leaves the infrastructure.

### 2. LLM-Agnostic Architecture
Works with ANY LLM: Ollama (local), Claude, GPT-4, etc.

### 3. Kernel-Centric Computation
Kernels are pure computation units — no LLM inside:
- Accept structured input (JSON/YAML)
- Produce structured output + human-readable summary
- Fully deterministic and reproducible

### 4. Reuse Over Reimplementation
KOAS wraps existing RAGIX tools rather than reimplementing.

## Kernel Development

Create a new kernel by extending the base class:

```python
from ragix_kernels.base import Kernel, KernelInput

class MyKernel(Kernel):
    name = "my_kernel"
    version = "1.0.0"
    category = "audit"
    stage = 2

    requires = ["ast_scan", "metrics"]
    provides = ["my_analysis"]

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
| Report length | 50+ pages, modular sections |
| Execution time | Stage 1+2 < 5 minutes for 1000 classes |
| LLM efficiency | < 10% time in LLM, > 90% in kernels |
| Reproducibility | Same input = same output (deterministic) |
| Auditability | Full SHA256 chain of all operations |

## Author

**Olivier Vitrac, PhD, HDR**
Head of Innovation Lab, Adservio
olivier.vitrac@adservio.fr

---

*"Science is no longer what the model can explain, but what a group of agents can coordinate."*
— Generative Simulation Initiative
