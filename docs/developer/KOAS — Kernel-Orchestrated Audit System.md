# KOAS — Kernel-Orchestrated Audit System

**A Sovereign, LLM-Agnostic Platform for AI-Assisted Code Analysis**
    
**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Version:** 1.0 DRAFT
**Date:** 2025-12-12
    
---

## Sommaire

[TOC]

---

# 1 | Philosophy & Intent

## 1.1 | The Problem with Current AI Code Analysis

Today's AI-assisted code analysis tools suffer from fundamental limitations:
    
| Problem | Consequence |
|---------|-------------|
| **Context window limits** | Cannot process large codebases (500K+ LOC) |
| **LLM computational waste** | Using LLMs for tasks better suited to deterministic algorithms |
| **Vendor lock-in** | Dependent on specific cloud providers |
| **Black-box reasoning** | No auditability or reproducibility |
| **Monolithic architecture** | Cannot scale or specialize |

## 1.2 | The KOAS Solution

KOAS (Kernel-Orchestrated Audit System) follows the **Virtual Hybrid Lab** paradigm proven in scientific computing at CARGILL,
FDA, and NWO/DIFFER:
    
> *"Each interaction costs a few seconds; most time is spent inside the **scientific kernels**, not the LLMs."*
> — Generative Simulation Initiative

**Core Principle:**
- **Kernels** do the heavy computation (AST parsing, metrics, graph analysis)
- **LLMs** orchestrate and interpret (what to run, what does it mean)
- **RAG** provides context (patterns, standards, domain knowledge)
  
```
┌─────────────────────────────────────────────────────────────────────┐
│ TRADITIONAL APPROACH                                                │
│  LLM reads entire codebase → LLM computes metrics → LLM writes      │
│  ❌ Slow, expensive, context-limited, non-reproducible              │
└─────────────────────────────────────────────────────────────────────┘
    
vs.
    
┌─────────────────────────────────────────────────────────────────────┐
│    KOAS APPROACH  │
│  Kernel computes → LLM reads summary → Kernel computes → LLM writes│
│  ✅ Fast, efficient, unlimited scale, fully reproducible  │
└─────────────────────────────────────────────────────────────────────┘
```

## 1.3 | Design Principles

### 1.3.1 | Local-First, Sovereignty by Default

All computation happens locally. Data never leaves the infrastructure unless explicitly configured.
    
```yaml
sovereignty:
  mode: local # local | hybrid | cloud
  allow_cloud_llm: false
  audit_trail: required  # every operation logged with SHA256 chain
```

### 1.3.2 | LLM-Agnostic Architecture

KOAS works with ANY LLM. The orchestrator can be:
- **Local**: Ollama (Mistral, Qwen, DeepSeek)
- **Cloud**: Claude, GPT-4, Gemini
- **Hybrid**: Local for agents, Cloud for discovery
  
```yaml
llm:
  discovery:    claude-sonnet# Large context, synthesis
  orchestrator: mistral # Coordination, scheduling
  agents:  mistral # Python-savvy, narrow tasks
```

### 1.3.3 | Kernel-Centric Computation

Kernels are **pure computation units** — no LLM inside. They:
- Accept structured input (JSON/YAML)
- Produce structured output + human-readable summary
- Are fully deterministic and reproducible
- Wrap existing tools (RAGIX, shell scripts, Python libraries)
  
### 1.3.4 | Reuse Over Reimplementation

KOAS maximizes reuse of existing RAGIX tools:
    
| RAGIX Tool | KOAS Kernel |
|------------|-------------|
| `ragix-ast scan` | `kernels/audit/ast_scan.py` |
| `ragix-ast metrics` | `kernels/audit/metrics.py` |
| `ragix_audit/partitioner.py` | `kernels/audit/partition.py` |
| `ragix_audit/service_detector.py` | `kernels/audit/services.py` |

---

# 2 | Architecture Overview

## 2.1 | System Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DISCOVERY LAYER  │
│  "What are the risks?" "How should we prioritize?" │
│  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  🔍 DISCOVERY AGENT│   │
│  │  LLM: Claude/GPT-4 (large context, synthesis)   │   │
│  │  Role: Strategic questions, report synthesis, recommendations  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│    ORCHESTRATION LAYER│
│  Workflow coordination, dependency resolution, scheduling│
│  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  🧠 ORCHESTRATOR AGENT  │   │
│  │  LLM: Mistral (coordination)│   │
│  │  Role: Execute manifest, manage stages, handle errors│   │
│  │  Kernel: workflow_engine (dependency graph, scheduling)   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
   │
  ┌─────────────────────────┼─────────────────────────┐
  ▼▼▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│  📊 METRICS AGENT│ │  🧩 STRUCTURE AGENT   │ │  📈 QUALITY AGENT│
│   │ │   │ │   │
│  LLM: Mistral    │ │  LLM: Mistral    │ │  LLM: Mistral    │
│  Kernels:   │ │  Kernels:   │ │  Kernels:   │
│  • ast_scan │ │  • partition│ │  • hotspots │
│  • metrics  │ │  • dependency    │ │  • dead_code│
│  • tech_debt│ │  • services │ │  • coupling │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘
  │││
  └─────────────────────────┼─────────────────────────┘
   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ KERNEL LAYER│
│  Pure computation — no LLM inside — deterministic & reproducible  │
│  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ast_scan │ │metrics  │ │partition│ │services │ │hotspots │ │coupling │  │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │
│  │ │ │ │ │ │   │
│  ▼ ▼ ▼ ▼ ▼ ▼   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RAGIX TOOLS (Reused)  │   │
│  │  ragix-ast | partitioner.py | service_detector.py | ...  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STORAGE LAYER    │
│  Audit workspace with full traceability  │
│  │
│  /audit/{PROJECT}/   │
│  ├── manifest.yaml # Configuration  │
│  ├── stage1/  # Raw kernel outputs (JSON)│
│  ├── stage2/  # Analysis results (JSON + summaries)│
│  ├── stage3/  # Report sections (Markdown)    │
│  ├── report/  # Final assembled report   │
│  ├── kb/ # Knowledge base (RAG index)    │
│  └── logs/    # Execution audit trail    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.2 | Data Flow

```
┌──────────────┐
│   manifest   │
│    .yaml│
└──────┬───────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Data Collection   │
│  Kernels: ast_scan → metrics → dependency → partition → services    │
│  Output: stage1/*.json (raw structured data)    │
└──────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Analysis│
│  Kernels: stats_summary → hotspot_analysis → coupling → dead_code   │
│  Output: stage2/*.json + stage2/*_summary.txt (for LLM)   │
└──────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Report Generation │
│  LLM + Kernels: section_executive → section_risk → section_debt → ...   │
│  Output: stage3/*.md (modular sections)    │
└──────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ASSEMBLY    │
│  Kernel: report_assemble    │
│  Output: report/Audit_{PROJECT}_{DATE}.md → PDF │
└──────────────────────────────────────────────────────────────────────────┘
```

---

# 3 | Kernel Architecture

## 3.1 | What is a Kernel?

A **kernel** in KOAS is a specialized computation unit that:
    
1. **Encapsulates domain logic** — AST parsing, metrics calculation, graph analysis
2. **Wraps existing tools** — RAGIX CLI, Python libraries, shell scripts
3. **Produces structured output** — JSON data + human-readable summary
4. **Is fully deterministic** — Same input always produces same output
5. **Contains no LLM** — Pure computation, no AI reasoning inside
  
### 3.1.1 | Kernel vs. Agent

| Aspect | Kernel | Agent |
|--------|--------|-------|
| **Contains LLM** | No | Yes |
| **Deterministic** | Yes | No (LLM variability) |
| **Purpose** | Compute | Orchestrate + Interpret |
| **Reusability** | High (pure function) | Medium (context-dependent) |
| **Testability** | Unit testable | Integration testable |

### 3.1.2 | Why Kernels?

Following the VirtualHybridLab pattern:
    
> *"Training a domain-specific LLM is inefficient and costly. Instead, GS uses **scientific agents**, which embed one or more 
LLMs and operate **scientific kernels** — specialized software libraries designed for machine-to-machine interaction."*

Applied to code audit:
- **Kernel** = Specialized audit computation (metrics, AST, graphs)
- **Agent** = LLM that operates the kernel and interprets results
  
## 3.2 | Kernel Categories

Kernels are organized by category in `ragix/kernels/`:
    
```
ragix/kernels/
├── __init__.py
├── base.py# Base classes and interfaces
├── registry.py # Kernel discovery and registration
│
├── audit/ # Code audit kernels
│   ├── __init__.py
│   ├── ast_scan.py # AST extraction
│   ├── metrics.py  # Code metrics (CC, LOC, MI)
│   ├── dependency.py    # Dependency graph
│   ├── partition.py# Codebase partitioning
│   ├── services.py # Service detection
│   ├── hotspots.py # Complexity hotspots
│   ├── dead_code.py# Dead code detection
│   ├── coupling.py # Coupling metrics (Ca, Ce, I, A, D)
│   ├── stats_summary.py # Statistical aggregation
│   └── report_*.py # Report section generators
│
├── transform/  # Code transformation kernels (future)
│   ├── refactor.py
│   └── migrate.py
│
├── test/  # Test generation kernels (future)
│   ├── unit_gen.py
│   └── coverage.py
│
└── docs/  # Documentation kernels (future)
    ├── docstring.py
    └── readme_gen.py
```

## 3.3 | Kernel Interface

### 3.3.1 | Base Class

```python
# ragix/kernels/base.py
"""
KOAS Kernel Base Classes
    
Following the VirtualHybridLab pattern:
- Kernels do computation (no LLM inside)
- Produce structured output + summary for LLM consumption
- Fully traceable and reproducible
"""
    
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json
    
@dataclass
class KernelInput:
    """Standard input for any kernel."""
    workspace: Path# Audit workspace root
    config: Dict[str, Any]  # Kernel-specific configuration
    dependencies: Dict[str, Path]# Required input files from previous stages
    
@dataclass
class KernelOutput:
    """Standard output from any kernel."""
    success: bool
    data: Dict[str, Any]    # Full structured data
    summary: str   # < 500 chars for LLM consumption
    output_file: Path  # Persisted JSON path
    
    # Traceability
    kernel_name: str
    kernel_version: str
    execution_time_ms: int
    input_hash: str    # SHA256 of inputs for reproducibility
    dependencies_used: List[str]
    
    # Optional diagnostics
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
class Kernel(ABC):
    """
    Abstract base class for all KOAS kernels.
    
    Design principles:
    1. No LLM inside — pure computation
    2. Deterministic — same input = same output
    3. Traceable — full audit trail
    4. Composable — clear dependencies
    """
    
    # Kernel metadata (override in subclasses)
    name: str = "base"
    version: str = "1.0.0"
    category: str = "base"
    stage: int = 0
    
    # Dependency declaration
    requires: List[str] = []# Kernel names this depends on
    provides: List[str] = []# What this kernel provides
    
    @abstractmethod
    def compute(self, input: KernelInput) -> Dict[str, Any]:
   """
   Core computation logic. Must be deterministic.
   Returns raw data dictionary.
   """
   pass
    
    @abstractmethod
    def summarize(self, data: Dict[str, Any]) -> str:
   """
   Generate human-readable summary (< 500 chars).
   This is what the LLM will see.
   """
   pass
    
    def run(self, input: KernelInput) -> KernelOutput:
   """
   Execute kernel with full traceability.
   Do not override — override compute() and summarize() instead.
   """
   start_time = datetime.now()
    
   # Compute input hash for reproducibility
   input_hash = self._hash_input(input)
    
   try:
  # Run computation
  data = self.compute(input)
  summary = self.summarize(data)
  success = True
  errors = []
   except Exception as e:
  data = {"error": str(e)}
  summary = f"Kernel {self.name} failed: {str(e)[:100]}"
  success = False
  errors = [str(e)]
    
   # Calculate execution time
   execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
    
   # Persist output
   output_file = input.workspace / f"stage{self.stage}" / f"{self.name}.json"
   output_file.parent.mkdir(parents=True, exist_ok=True)
   output_file.write_text(json.dumps(data, indent=2, default=str))
    
   # Also write summary for LLM
   summary_file = output_file.with_suffix('.summary.txt')
   summary_file.write_text(summary)
    
   return KernelOutput(
  success=success,
  data=data,
  summary=summary,
  output_file=output_file,
  kernel_name=self.name,
  kernel_version=self.version,
  execution_time_ms=execution_time_ms,
  input_hash=input_hash,
  dependencies_used=list(input.dependencies.keys()),
  errors=errors
   )
    
    def _hash_input(self, input: KernelInput) -> str:
   """Generate SHA256 hash of inputs for reproducibility."""
   content = json.dumps({
  "config": input.config,
  "dependencies": {k: str(v) for k, v in input.dependencies.items()}
   }, sort_keys=True)
   return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### 3.3.2 | Example: AST Scan Kernel

```python
# ragix/kernels/audit/ast_scan.py
"""
Kernel: AST Scan
Stage: 1 (Data Collection)
Wraps: ragix-ast scan
    
Extracts Abstract Syntax Tree data from source code.
"""
    
import subprocess
import json
from pathlib import Path
from typing import Dict, Any
    
from ..base import Kernel, KernelInput
    
class ASTScanKernel(Kernel):
    """Extract AST data from source code using RAGIX tools."""
    
    name = "ast_scan"
    version = "1.0.0"
    category = "audit"
    stage = 1
    
    requires = []  # No dependencies — first kernel
    provides = ["symbols", "files", "classes", "methods"]
    
    def compute(self, input: KernelInput) -> Dict[str, Any]:
   """Run ragix-ast scan and parse results."""
    
   project_path = input.config["project"]["path"]
   language = input.config["project"].get("language", "java")
    
   # Call existing RAGIX tool (REUSE, don't reimplement!)
   result = subprocess.run(
  [
 "ragix-ast", "scan",
 project_path,
 "--lang", language,
 "--json"
  ],
  capture_output=True,
  text=True,
  cwd=str(input.workspace)
   )
    
   if result.returncode != 0:
  raise RuntimeError(f"ragix-ast scan failed: {result.stderr}")
    
   data = json.loads(result.stdout)
    
   # Enrich with computed statistics
   data["statistics"] = {
  "total_files": len(data.get("files", [])),
  "total_classes": sum(f.get("class_count", 0) for f in data.get("files", [])),
  "total_methods": sum(f.get("method_count", 0) for f in data.get("files", [])),
  "total_loc": sum(f.get("loc", 0) for f in data.get("files", [])),
  "parse_errors": len(data.get("errors", [])),
   }
    
   return data
    
    def summarize(self, data: Dict[str, Any]) -> str:
   """Generate LLM-consumable summary."""
   stats = data.get("statistics", {})
   return (
  f"AST scan complete. "
  f"Analyzed {stats.get('total_files', 0)} files: "
  f"{stats.get('total_classes', 0)} classes, "
  f"{stats.get('total_methods', 0)} methods, "
  f"{stats.get('total_loc', 0):,} LOC. "
  f"Parse errors: {stats.get('parse_errors', 0)}."
   )
```

## 3.4 | Kernel Discovery

The orchestrator discovers kernels automatically via the registry:
    
```python
# ragix/kernels/registry.py
"""
Kernel Registry — Automatic discovery and dependency resolution.
"""
    
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Type
from .base import Kernel
    
class KernelRegistry:
    """Discovers and manages available kernels."""
    
    _kernels: Dict[str, Type[Kernel]] = {}
    _categories: Dict[str, List[str]] = {}
    
    @classmethod
    def discover(cls, package_path: str = "ragix.kernels"):
   """Auto-discover all kernels in the package."""
   package = importlib.import_module(package_path)
    
   for _, module_name, is_pkg in pkgutil.walk_packages(
  package.__path__, prefix=f"{package_path}."
   ):
  if is_pkg:
 continue
    
  module = importlib.import_module(module_name)
    
  for attr_name in dir(module):
 attr = getattr(module, attr_name)
 if (
isinstance(attr, type)
and issubclass(attr, Kernel)
and attr is not Kernel
 ):
cls.register(attr)
    
    @classmethod
    def register(cls, kernel_class: Type[Kernel]):
   """Register a kernel class."""
   name = kernel_class.name
   category = kernel_class.category
    
   cls._kernels[name] = kernel_class
    
   if category not in cls._categories:
  cls._categories[category] = []
   cls._categories[category].append(name)
    
    @classmethod
    def get(cls, name: str) -> Type[Kernel]:
   """Get kernel class by name."""
   if name not in cls._kernels:
  raise KeyError(f"Kernel '{name}' not found. Available: {list(cls._kernels.keys())}")
   return cls._kernels[name]
    
    @classmethod
    def list_category(cls, category: str) -> List[str]:
   """List all kernels in a category."""
   return cls._categories.get(category, [])
    
    @classmethod
    def resolve_dependencies(cls, kernel_names: List[str]) -> List[str]:
   """Topologically sort kernels by dependencies."""
   # Build dependency graph
   graph = {}
   for name in kernel_names:
  kernel = cls.get(name)
  graph[name] = kernel.requires
    
   # Topological sort
   sorted_kernels = []
   visited = set()
    
   def visit(name):
  if name in visited:
 return
  visited.add(name)
  for dep in graph.get(name, []):
 visit(dep)
  sorted_kernels.append(name)
    
   for name in kernel_names:
  visit(name)
    
   return sorted_kernels
```

---

# 4 | Integration with RAGIX

## 4.1 | Existing Tools Reused

KOAS kernels wrap existing RAGIX tools without reimplementation:
    
| RAGIX Tool | Location | KOAS Kernel | Stage |
|------------|----------|-------------|-------|
| `ragix-ast scan` | `ragix_unix/ast_cli.py` | `ast_scan` | 1 |
| `ragix-ast metrics` | `ragix_unix/ast_cli.py` | `metrics` | 1 |
| `ragix-ast graph` | `ragix_unix/ast_cli.py` | `dependency` | 1 |
| `ragix-ast hotspots` | `ragix_unix/ast_cli.py` | `hotspots` | 2 |
| `partitioner.py` | `ragix_audit/partitioner.py` | `partition` | 1 |
| `service_detector.py` | `ragix_audit/service_detector.py` | `services` | 1 |
| `risk.py` | `ragix_audit/risk.py` | `risk_matrix` | 2 |
| `drift.py` | `ragix_audit/drift.py` | `drift` | 2 |

## 4.2 | Directory Structure in RAGIX

```
ragix/
├── kernels/# NEW: Kernel implementations
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── audit/
│   │   ├── __init__.py
│   │   ├── ast_scan.py
│   │   ├── metrics.py
│   │   └── ...
│   └── [future categories]/
│
├── ragix_audit/ # EXISTING: Audit tools (wrapped by kernels)
│   ├── partitioner.py
│   ├── service_detector.py
│   └── ...
│
├── ragix_unix/  # EXISTING: CLI tools (wrapped by kernels)
│   ├── ast_cli.py
│   └── ...
│
└── ragix_web/   # EXISTING: Web UI (can call kernels via API)
    └── ...
```

---

# 5 | MCP, Skills & Scripts Integration

## 5.1 | Interface Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│CLAUDE CODE / CLAUDE DESKTOP│
│  │
│  User: "Audit the IOWIZME codebase and identify risks" │
└─────────────────────────────────────────────────────────────────────────────┘
   │
┌─────────────────┼─────────────────┐
▼  ▼  ▼
  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
  │    MCP   │ │    SKILLS│ │   SCRIPTS│
  │   Server │ │  (Prompts)    │ │   (Shell)│
  ├───────────────┤ ├───────────────┤ ├───────────────┤
  │ Tool calls    │ │ Methodology   │ │ Batch ops│
  │ from Claude   │ │ templates│ │ automation    │
  │ Desktop  │ │ for Claude    │ ││
  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
│  │  │
└─────────────────┼─────────────────┘
   ▼
┌─────────────────────────────────────┐
│    KOAS ORCHESTRATOR │
│  │
│  - Parses manifest.yaml   │
│  - Resolves kernel dependencies│
│  - Executes kernels in order   │
│  - Collects summaries for LLM  │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│  KERNELS   │
└─────────────────────────────────────┘
```

## 5.2 | MCP Tools

The MCP server exposes KOAS operations as tools:
    
```json
// MCP/koas_tools_spec.json
{
  "tools": [
    {
 "name": "koas_init",
 "description": "Initialize a new audit workspace",
 "parameters": {
   "project_name": "string",
   "project_path": "string",
   "language": "string",
   "preset": "string (optional)"
 }
    },
    {
 "name": "koas_run_stage",
 "description": "Execute all kernels in a stage",
 "parameters": {
   "workspace": "string",
   "stage": "integer (1, 2, or 3)"
 }
    },
    {
 "name": "koas_run_kernel",
 "description": "Execute a specific kernel",
 "parameters": {
   "workspace": "string",
   "kernel": "string"
 }
    },
    {
 "name": "koas_get_summary",
 "description": "Get summaries from completed kernels",
 "parameters": {
   "workspace": "string",
   "stage": "integer (optional)"
 }
    },
    {
 "name": "koas_list_kernels",
 "description": "List available kernels by category",
 "parameters": {
   "category": "string (optional)"
 }
    }
  ]
}
```

## 5.3 | Skills (Claude Code)

Skills define methodology for specific audit types:
    
```markdown
<!-- .claude/commands/audit-full.md -->
# Full Codebase Audit
    
Execute a comprehensive code audit using KOAS:
    
1. Initialize workspace:
```
   Call koas_init with project details
   ```
    
2. Stage 1 - Data Collection:
   ```
   Call koas_run_stage(stage=1)
   ```
   Review summaries and confirm to proceed.
    
3. Stage 2 - Analysis:
   ```
   Call koas_run_stage(stage=2)
   ```
   Review findings and prioritize concerns.
    
4. Stage 3 - Report Generation:
   For each section, review and refine the generated content.
   ```
   Call koas_run_stage(stage=3)
   ```
    
5. Final assembly and review.
   ```

## 5.4 | Shell Scripts

For batch operations and CI/CD:
    
```bash
#!/bin/bash
# scripts/run_audit.sh — Execute full audit pipeline
    
WORKSPACE=$1
MANIFEST=$2
    
echo "=== KOAS Audit Pipeline ==="
    
# Stage 1: Data Collection
echo "[Stage 1] Data Collection..."
python -m ragix.kernels.orchestrator run \
    --workspace "$WORKSPACE" \
    --manifest "$MANIFEST" \
    --stage 1
    
# Stage 2: Analysis
echo "[Stage 2] Analysis..."
python -m ragix.kernels.orchestrator run \
    --workspace "$WORKSPACE" \
    --manifest "$MANIFEST" \
    --stage 2
    
# Stage 3: Report
echo "[Stage 3] Report Generation..."
python -m ragix.kernels.orchestrator run \
    --workspace "$WORKSPACE" \
    --manifest "$MANIFEST" \
    --stage 3
    
echo "=== Audit Complete ==="
echo "Report: $WORKSPACE/report/"
```

---

# 6 | Configuration

## 6.1 | Manifest Structure

Each audit is configured via `manifest.yaml`:
    
```yaml
# manifest.yaml — Audit Configuration
# Location: /audit/{PROJECT}/manifest.yaml
    
# ============================================================================
# METADATA
# ============================================================================
audit:
  name: "IOWIZME Technical Audit"
  version: "1.0"
  date: "2025-12-12"
  author: "Olivier Vitrac, PhD, HDR | Adservio"
    
# ============================================================================
# PROJECT DEFINITION
# ============================================================================
project:
  name: "IOWIZME"
  path: "/path/to/IOWIZME/src"
  language: "java"
    
  # Multi-module support
  modules:
    - name: "app-pre-main"
 path: "app-pre-main/src/main/java"
    - name: "sat-pre-echange"
 path: "sat-pre-echange/src/main/java"
    
  # Exclusions
  exclude:
    - "**/test/**"
    - "**/generated/**"
    
# ============================================================================
# LLM CONFIGURATION
# ============================================================================
llm:
  # Discovery agent — large context, synthesis capabilities
  discovery:
    provider: "claude" # claude | openai | ollama
    model: "claude-sonnet-4"
    temperature: 0.3
    role: "Strategic analysis and report synthesis"
    
  # Orchestrator — coordination and scheduling
  orchestrator:
    provider: "ollama"
    model: "mistral"
    temperature: 0.2
    role: "Workflow coordination and error handling"
    
  # Agents — narrow tasks, Python-savvy
  agents:
    provider: "ollama"
    model: "mistral"
    temperature: 0.1
    role: "Kernel operation and result interpretation"
    
# ============================================================================
# SOVEREIGNTY SETTINGS
# ============================================================================
sovereignty:
  mode: "hybrid"  # local | hybrid | cloud
  allow_cloud_discovery: true    # Allow Claude for discovery
  allow_cloud_agents: false # Keep agents local
  audit_trail: "required"   # required | optional | none
    
# ============================================================================
# STAGE 1: DATA COLLECTION KERNELS
# ============================================================================
stage1:
  ast_scan:
    enabled: true
    options:
 include_tests: false
 max_depth: 10
 parse_javadoc: true
    
  metrics:
    enabled: true
    options:
 complexity_threshold: 10
 loc_threshold: 300
 debt_rate_hours_per_issue: 0.5
    
  dependency:
    enabled: true
    options:
 include_external: false
 detect_cycles: true
    
  partition:
    enabled: true
    options:
 preset: "sias_ticc"
 propagation_iterations: 5
 confidence_threshold: 0.7
 # Graph propagation algorithm settings
 forward_weight: 0.7
 reverse_weight: 0.3
 package_cohesion_bonus: 0.2
    
  services:
    enabled: true
    options:
 patterns:
   - "spre##"
   - "SK*"
   - "SC*"
   - "SG*"
    
# ============================================================================
# STAGE 2: ANALYSIS KERNELS
# ============================================================================
stage2:
  stats_summary:
    enabled: true
    depends_on: [ast_scan, metrics]
    
  hotspots:
    enabled: true
    depends_on: [metrics]
    options:
 top_n: 50
 threshold_cc: 15
 threshold_loc: 500
    
  dead_code:
    enabled: true
    depends_on: [dependency, partition]
    options:
 strict: true   # Require no callers AND no callees
    
  coupling:
    enabled: true
    depends_on: [dependency, partition]
    options:
 # Martin metrics
 compute_instability: true
 compute_abstractness: true
 compute_distance: true
    
  pattern_search:
    enabled: true
    depends_on: [ast_scan, metrics]
    options:
 patterns:
   - name: "God Classes"
query: "methods > 30 OR loc > 1000"
   - name: "Deep Inheritance"
query: "inheritance_depth > 4"
   - name: "High Coupling"
query: "efferent_coupling > 20"
    
# ============================================================================
# STAGE 3: REPORT GENERATION
# ============================================================================
stage3:
  sections:
    - id: "executive"
 title: "Executive Summary"
 kernel: "section_executive"
 depends_on: [stats_summary]
 max_pages: 2
 llm_role: "synthesis"
    
    - id: "methodology"
 title: "Methodology"
 kernel: "section_methodology"
 depends_on: []
 max_pages: 3
 template: "methodology_standard.md"
    
    - id: "overview"
 title: "Codebase Overview"
 kernel: "section_overview"
 depends_on: [stats_summary, partition]
 max_pages: 5
    
    - id: "risk"
 title: "Risk Assessment"
 kernel: "section_risk"
 depends_on: [hotspots, coupling, dead_code]
 max_pages: 8
    
    - id: "debt"
 title: "Technical Debt Analysis"
 kernel: "section_debt"
 depends_on: [metrics, hotspots]
 max_pages: 10
    
    - id: "architecture"
 title: "Architecture Analysis"
 kernel: "section_architecture"
 depends_on: [partition, dependency, coupling]
 max_pages: 15
    
    - id: "dead_code"
 title: "Dead Code & Cleanup"
 kernel: "section_dead_code"
 depends_on: [dead_code]
 max_pages: 5
    
    - id: "recommendations"
 title: "Recommendations"
 kernel: "section_recommendations"
 depends_on: [all]
 max_pages: 10
 llm_role: "discovery"
    
# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
output:
  format: "markdown"    # markdown | html | pdf
  template: "adservio_audit" # Report template
  assets:
    logo: "assets/adservio-logo.svg"
    header_image: "assets/adservio-experience.jpg"
    
  # PDF generation (if format includes pdf)
  pdf:
    engine: "pandoc"    # pandoc | weasyprint
    toc: true
    toc_depth: 3
```

## 6.2 | Workspace Structure

```
/home/olivi/Documents/Adservio/audit/IOWIZME/
├── manifest.yaml  # Configuration
│
├── stage1/   # Raw kernel outputs
│   ├── ast_scan.json
│   ├── ast_scan.summary.txt # For LLM consumption
│   ├── metrics.json
│   ├── metrics.summary.txt
│   ├── dependency.json
│   ├── partition.json
│   └── services.json
│
├── stage2/   # Analysis results
│   ├── stats_summary.json
│   ├── stats_summary.summary.txt
│   ├── hotspots.json
│   ├── dead_code.json
│   ├── coupling.json
│   └── pattern_search.json
│
├── stage3/   # Report sections (Markdown)
│   ├── 01_executive.md
│   ├── 02_methodology.md
│   ├── 03_overview.md
│   ├── 04_risk.md
│   ├── 05_debt.md
│   ├── 06_architecture.md
│   ├── 07_dead_code.md
│   └── 08_recommendations.md
│
├── report/   # Final assembled report
│   ├── Audit_IOWIZME_2025-12-12.md
│   └── Audit_IOWIZME_2025-12-12.pdf
│
├── assets/   # Report assets
│   ├── adservio-logo.svg
│   └── adservio-experience.jpg
│
├── kb/  # Knowledge base (RAG index)
│   └── findings.idx
│
└── logs/# Execution audit trail
    ├── 2025-12-12_stage1.log
    ├── 2025-12-12_stage2.log
    └── audit_trail.json# SHA256 chain
```

---

# 7 | Workplan

## 7.1 | Phase 1: Foundation (Kernels Infrastructure)

| Task | Description | Deliverable |
|------|-------------|-------------|
| **1.1** | Create `ragix/kernels/` directory structure | Directory layout |
| **1.2** | Implement `base.py` with `Kernel` and `KernelOutput` | Base classes |
| **1.3** | Implement `registry.py` for kernel discovery | Auto-discovery |
| **1.4** | Create first Stage 1 kernel: `ast_scan.py` | Working kernel |
| **1.5** | Validate kernel wraps `ragix-ast scan` correctly | Integration test |

## 7.2 | Phase 2: Stage 1 Kernels (Data Collection)

| Task | Description | Wraps |
|------|-------------|-------|
| **2.1** | `metrics.py` | `ragix-ast metrics` |
| **2.2** | `dependency.py` | `ragix-ast graph` + `build_dependency_graph` |
| **2.3** | `partition.py` | `ragix_audit/partitioner.py` |
| **2.4** | `services.py` | `ragix_audit/service_detector.py` |

## 7.3 | Phase 3: Stage 2 Kernels (Analysis)

| Task | Description | Dependencies |
|------|-------------|--------------|
| **3.1** | `stats_summary.py` | ast_scan, metrics |
| **3.2** | `hotspots.py` | metrics |
| **3.3** | `dead_code.py` | dependency, partition |
| **3.4** | `coupling.py` | dependency, partition |
| **3.5** | `pattern_search.py` | ast_scan, metrics |

## 7.4 | Phase 4: Orchestrator

| Task | Description |
|------|-------------|
| **4.1** | Implement `orchestrator.py` CLI |
| **4.2** | Manifest parsing and validation |
| **4.3** | Dependency resolution (topological sort) |
| **4.4** | Stage execution with progress reporting |
| **4.5** | Summary collection for LLM |

## 7.5 | Phase 5: Stage 3 Kernels (Reporting)

| Task | Description |
|------|-------------|
| **5.1** | Report section template system |
| **5.2** | `section_executive.py` |
| **5.3** | `section_overview.py` |
| **5.4** | `section_risk.py` |
| **5.5** | `section_debt.py` |
| **5.6** | `section_architecture.py` |
| **5.7** | `report_assemble.py` |

## 7.6 | Phase 6: Integration

| Task | Description |
|------|-------------|
| **6.1** | MCP tools for KOAS operations |
| **6.2** | Claude Code skills/commands |
| **6.3** | Shell scripts for batch operations |
| **6.4** | Documentation |

## 7.7 | Phase 7: Validation

| Task | Description |
|------|-------------|
| **7.1** | Test on IOWIZME codebase |
| **7.2** | Generate 50+ page report |
| **7.3** | Validate with SIAS audit format |
| **7.4** | Performance optimization |

---

# 8 | Success Criteria

| Criterion | Target |
|-----------|--------|
| **Codebase support** | 500K+ LOC without context issues |
| **Report length** | 50+ pages, modular sections |
| **Execution time** | Stage 1+2 < 5 minutes for 1000 classes |
| **LLM efficiency** | < 10% time in LLM, > 90% in kernels |
| **Reproducibility** | Same input = same output (deterministic) |
| **Auditability** | Full SHA256 chain of all operations |
| **Reusability** | Template works for any Java/Python project |

---

# 9 | Conclusion

KOAS brings the **Virtual Hybrid Lab** paradigm to code auditing:
    
- **Kernels compute** — Fast, deterministic, scalable
- **LLMs orchestrate** — Strategic decisions, synthesis, interpretation
- **RAG provides context** — Domain knowledge, patterns, standards
  

This architecture enables sophisticated 50+ page audit reports on 500K+ LOC codebases while maintaining:
    
- Full sovereignty (local-first)
- Complete auditability (SHA256 chain)
- LLM agnosticism (any provider)
- Maximum reuse (RAGIX tools)
  
> *"Science is no longer what the model can explain, but what a group of agents can coordinate."*
> — Generative Simulation Initiative

---

**Document Version:** 1.0 DRAFT
**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Date:** 2025-12-12