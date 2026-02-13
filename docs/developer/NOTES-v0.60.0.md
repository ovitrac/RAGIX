# RAGIX v0.60.0 Release Notes

**Release Date:** 2025-12-14
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

---

## Overview

Version 0.60.0 enhances RAGIX's position as a **sovereign swiss knife** for industrial-scale code auditing. This release focuses on:

1. **MCP Server Enhancement** — 5 new tools for comprehensive system introspection
2. **Parallel KOAS Execution** — Faster audits through dependency-aware parallelization
3. **French i18n Fixes** — Proper UTF-8 diacritics in audit reports
4. **System Introspection** — GPU, CPU, memory, and model management

---

## Key Features

### 1. MCP Server v0.8.0 (22 Tools Total)

Five new tools expand RAGIX's MCP capabilities:

| Tool | Purpose |
|------|---------|
| `ragix_ast_scan` | Extract AST symbols from source code |
| `ragix_ast_metrics` | Compute code quality metrics |
| `ragix_models_list` | List available Ollama models |
| `ragix_model_info` | Get detailed model information |
| `ragix_system_info` | Comprehensive system introspection |

**Example: System Info**
```python
result = ragix_system_info()
# Returns: platform, cpu, memory, gpu (CUDA), disk, ollama status
```

**Example: AST Scan**
```python
result = ragix_ast_scan("/path/to/project", language="auto")
# Returns: symbols, summary (classes, methods, functions)
```

### 2. Parallel KOAS Execution

The `koas_run` tool now supports parallel kernel execution:

```python
# Sequential (default)
koas_run(workspace, stage=0)  # ~5-10s for 60K LOC

# Parallel (new)
koas_run(workspace, stage=0, parallel=True, workers=4)  # ~3.4s for 60K LOC
```

**Performance gains:**
- Stage 1 (Data Collection): 2-3x faster
- Stage 2 (Analysis): ~2x faster with batching
- Stage 3 (Reporting): Unchanged (sequential for consistency)

### 3. French i18n Corrections

Fixed 50+ translation strings for proper UTF-8 diacritics in French reports:

| Before | After |
|--------|-------|
| Methodologie | Méthodologie |
| Synthese Executive | Synthèse Exécutive |
| Complexite | Complexité |
| Indice de Maintenabilite | Indice de Maintenabilité |
| Évaluation des Risques | Évaluation des Risques |

**Files updated:**
- `ragix_kernels/audit/report/i18n.py`
- `ragix_kernels/audit/report/templates.py`
- `ragix_kernels/audit/section_drift.py`

### 4. Claude Code Slash Commands

New and updated commands:

| Command | Description |
|---------|-------------|
| `/koas-audit` | Full audit with `--parallel` option |
| `/ragix-system` | System introspection for deployment |
| `/ragix-models` | Model management and selection |

---

## Technical Details

### New MCP Tool Signatures

```python
# AST Analysis
def ragix_ast_scan(
    path: str,
    language: str = "auto",
    include_private: bool = False,
) -> Dict[str, Any]: ...

def ragix_ast_metrics(
    path: str,
    language: str = "auto",
) -> Dict[str, Any]: ...

# Model Management
def ragix_models_list() -> Dict[str, Any]: ...
def ragix_model_info(model: str) -> Dict[str, Any]: ...

# System Info
def ragix_system_info() -> Dict[str, Any]: ...
```

### KOAS Run Enhanced Signature

```python
def koas_run(
    workspace: str,
    stage: int = 0,
    kernels: str = "",
    parallel: bool = False,  # NEW
    workers: int = 4,        # NEW
) -> Dict[str, Any]:
    # Returns: status, stages_run, results, report_path,
    #          execution_mode, workers, duration_seconds
```

---

## Test Coverage

New test file: `tests/test_mcp_server.py`

**18 tests covering:**
- AST scan single file and directory
- Private symbol filtering
- Non-existent path handling
- Metrics computation
- Model listing (mocked Ollama)
- Model info retrieval
- System info structure
- KOAS parallel parameter validation
- Tool availability checks (core, KOAS, v0.8.0)

```bash
pytest tests/test_mcp_server.py -v
# Result: 18 passed in 4.34s
```

---

## Performance Benchmarks

### IOWIZME Audit (Java, 60K LOC)

| Metric | Value |
|--------|-------|
| Total Files | 806 |
| Total Classes | 582 |
| Total Methods | 2,704 |
| Lines of Code | 60,157 |
| Audit Time (parallel) | **3.4 seconds** |
| Maintainability Index | 100/100 (Grade A) |
| Technical Debt | 0.0 days |

### Throughput Estimates

| Project Size | Time (parallel) | Hourly Rate |
|--------------|-----------------|-------------|
| 10K LOC | ~1s | 3,600/hour |
| 60K LOC | ~3-4s | 900-1,200/hour |
| 200K LOC | ~10-15s | 240-360/hour |
| 1M LOC | ~60s | 60/hour |

---

## Installation

No new dependencies required. Existing installations update automatically:

```bash
pip install --upgrade ragix
# Or from source:
pip install -e .
```

---

## Migration Notes

### From v0.59.0

- No breaking changes
- All existing MCP tools continue to work
- KOAS workspaces are fully compatible
- New tools available immediately after upgrade

### For MCP Clients

Update tool invocations to use new features:

```json
// Old
{"tool": "koas_run", "workspace": "/path"}

// New (with parallel)
{"tool": "koas_run", "workspace": "/path", "parallel": true, "workers": 4}
```

---

## Known Issues

None reported.

---

## What's Next (v0.61.0)

Potential areas for future improvement:
- GPU acceleration for AST parsing (CUDA)
- Streaming audit progress via MCP
- Multi-repository batch auditing
- Real-time audit dashboard

---

## Video Demonstration

Watch the RAGIX demo showcasing key features:
**https://www.youtube.com/watch?v=vDHI70ZPnDE**

---

## Industrial Applications

RAGIX v0.60.0 is ready for:

- **Enterprise Code Audits**: Thousands of files, millions of classes
- **MCO (Maintenance) Assessments**: Technical debt quantification
- **Continuous Quality Monitoring**: Regular codebase reviews
- **Documentation Gap Analysis**: Code-spec drift detection

With proper GPU and CPU resources, organizations can audit:
- **3-20 codebases per hour** (depending on size)
- **Hundreds of projects per day**
- **Full deterministic, statistical, and AI-powered analysis**

---

## Contributors

- **Olivier Vitrac** — Architecture, implementation, testing

---

*Generated by KOAS (Kernel-Orchestrated Audit System)*
*RAGIX — Retrieval-Augmented Generative Interactive eXecution Agent*
