# KOAS — Kernel-Orchestrated Audit System

**Philosophy, Origins, and Application**

**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Version:** 1.0
**Date:** 2025-12-14

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Origins: Scientific Kernels](#2-origins-scientific-kernels)
3. [From Virtual Labs to Code Audits](#3-from-virtual-labs-to-code-audits)
4. [KOAS Architecture](#4-koas-architecture)
5. [Kernel Philosophy](#5-kernel-philosophy)
6. [The Three-Stage Pipeline](#6-the-three-stage-pipeline)
7. [Orchestration and Execution](#7-orchestration-and-execution)
8. [RAG Integration](#8-rag-integration)
9. [Quality Standards](#9-quality-standards)
10. [References](#10-references)

---

## 1. Introduction

**KOAS (Kernel-Orchestrated Audit System)** is a sovereign, local-first code analysis framework that applies scientific computation principles to software quality assessment. It combines Abstract Syntax Tree (AST) analysis, graph-based metrics, and structured reporting into a reproducible, auditable pipeline.

### Core Principles

1. **Sovereignty**: All processing happens locally — no cloud dependencies
2. **Reproducibility**: Deterministic kernels produce identical outputs for identical inputs
3. **Auditability**: Complete execution trail with cryptographic verification
4. **Composability**: Modular kernels with explicit dependencies
5. **LLM-Ready**: Structured summaries designed for AI-assisted interpretation

---

## 2. Origins: Scientific Kernels

KOAS inherits its architecture from the **Generative Simulation (GS)** initiative, which developed a pattern for augmenting AI agents with specialized scientific computation.

### The Scientific Agent Pattern

In the GS framework, **scientific agents** are not raw LLMs. Instead, they embed one or more LLMs that operate **scientific kernels** — specialized software libraries designed for machine-to-machine interaction rather than human use.

```
┌─────────────────────────────────────────────────────────────┐
│                    SCIENTIFIC AGENT                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│   │   Encoder   │───▶│   Kernel    │───▶│   Decoder   │     │
│   │    (LLM)    │    │(Computation)│    │    (LLM)    │     │
│   └─────────────┘    └─────────────┘    └─────────────┘     │
│                            │                                │
│                            ▼                                │
│                     ┌─────────────┐                         │
│                     │ Structured  │                         │
│                     │   Output    │                         │
│                     └─────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key insight from GS:**

> "Each interaction costs a few seconds; most time is spent inside the scientific kernels, not the LLMs."

Scientific kernels include:
- **FEniCS, OpenFOAM** for fluid dynamics
- **LAMMPS, Gromacs** for molecular simulation
- **SFPPy** for mass transfer and migration modeling
- **Pizza3** for soft-matter mechanics

These kernels are narrow but powerful: they respond to one supervisor and produce strongly contextualized outputs.

### The RAG Dual Role

In the GS pattern, RAG (Retrieval-Augmented Generation) serves two key roles:

1. **Upstream (Encoder)**: Encode and contextualize user requests into domain-aware instructions aligned with kernel semantics
2. **Downstream (Decoder)**: Decode and frame kernel outputs into interpretable, structured insights for the LLM

This bidirectional RAG loop ensures precision, traceability, and scientific consistency across reasoning steps.

### The Orchestrator Pattern

Above individual agents sits an **orchestrator agent** that coordinates multiple scientific agents asynchronously:

- Scheduling and dependency management
- Progress monitoring and diagnostics
- Convergence by majority voting or weighted consensus

The orchestrator's kernel typically includes logic or numerical solvers, while an LLM with large context handles summarization and dashboards.

---

## 3. From Virtual Labs to Code Audits

The GS scientific agent pattern maps directly to code audit systems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VIRTUAL HYBRID LAB                    →    CODE AUDIT LAB                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Scientific Kernels                    →    Audit Kernels                   │
│  (FEniCS, LAMMPS, SFPPy, Pizza3)            (AST scan, metrics, coupling)   │
│                                                                             │
│  Scientific Agent                      →    Audit Agent                     │
│  (narrow, operates one kernel)              (operates one audit kernel)     │
│                                                                             │
│  Orchestrator Agent                    →    Audit Orchestrator              │
│  (coordinates, schedules, diagnoses)        (stages, dependencies, flow)    │
│                                                                             │
│  Discovery Agent                       →    Audit Discovery                 │
│  ("Interpret these experiments")            ("What are the risks here?")    │
│                                                                             │
│  Knowledge Base (RAG)                  →    Audit KB (RAG)                  │
│  (regulatory data, domain knowledge)        (patterns, standards, history)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Applied to Code Audit

- **Kernels** do the heavy computation (AST parsing, metrics, graph analysis)
- **LLM** orchestrates and interprets (what to run next, what does this mean)
- **RAG** provides context (patterns, standards, previous audits)

This separation ensures:
- Reproducible, deterministic analysis
- Explainable results with full audit trail
- Efficient use of LLM resources

---

## 4. KOAS Architecture

### Code Audit Lab Architecture

```
                    ┌──────────────────────────────────────┐
                    │      DISCOVERY AGENT                 │
                    │  "What technical debt should we      │
                    │   prioritize? What are the risks?"   │
                    │                                      │
                    │  LLM: Claude/GPT-4 (large context)   │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │      ORCHESTRATOR AGENT              │
                    │  Coordinates audit workflow          │
                    │  Handles dependencies & scheduling   │
                    │                                      │
                    │  Engine: Kernel Orchestrator         │
                    └──────────────────┬───────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  METRICS KERNEL   │    │  STRUCTURE        │    │  QUALITY KERNEL   │
│                   │    │  KERNEL           │    │                   │
│  • ast_scan       │    │  • partition      │    │  • hotspots       │
│  • metrics        │    │  • dependency     │    │  • dead_code      │
│  • coupling       │    │  • services       │    │  • risk           │
└─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                 ┌─────────────────▼──────────────────┐
                 │        AUDIT KNOWLEDGE BASE        │
                 │                                    │
                 │  stage1/  → Raw kernel outputs     │
                 │  stage2/  → Analysis summaries     │
                 │  stage3/  → Report sections        │
                 │  logs/    → Full audit trail       │
                 └────────────────────────────────────┘
```

---

## 5. Kernel Philosophy

### What is a Kernel?

A kernel is a **pure computation unit** that:

1. Takes structured input (workspace path, configuration, dependencies)
2. Produces JSON output + human-readable summary
3. Is fully deterministic (no randomness, no external network calls)
4. Contains no LLM logic (pure computation)

### Kernel Interface

```python
class Kernel:
    name: str           # Unique identifier (e.g., "ast_scan")
    version: str        # Semantic version
    stage: int          # Pipeline stage (1, 2, or 3)
    category: str       # Functional category
    requires: List[str] # Dependencies (other kernel outputs)
    provides: List[str] # Capabilities provided

    def compute(input: KernelInput) -> Dict[str, Any]:
        """Execute kernel computation. Pure function."""
        ...

    def summarize(data: Dict) -> str:
        """Generate LLM-consumable summary (<500 chars)."""
        ...
```

### Kernel Output Contract

Every kernel produces:

```python
@dataclass
class KernelOutput:
    success: bool              # Execution status
    data: Dict[str, Any]       # Full structured data
    summary: str               # LLM-ready summary (<500 chars)
    output_file: Path          # Persisted JSON location
    dependencies_used: List    # Traceability
```

### Why This Design?

1. **Separation of Concerns**: Computation is isolated from orchestration
2. **Testability**: Kernels can be tested in isolation
3. **Parallelism**: Independent kernels run concurrently
4. **Reproducibility**: No hidden state or external dependencies
5. **Auditability**: Clear inputs, outputs, and execution trace

---

## 6. The Three-Stage Pipeline

KOAS organizes computation into three stages, each building on the previous:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        KOAS Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │   STAGE 1       │  │   STAGE 2       │  │   STAGE 3       │      │
│  │ Data Collection │─▶│    Analysis     │─▶│   Reporting     │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
│                                                                     │
│  • AST Scanning       • Statistics         • Executive Summary      │
│  • Metrics            • Hotspots           • Overview               │
│  • Dependencies       • Dead Code          • Risk Assessment        │
│  • Partitioning       • Coupling           • Drift Analysis         │
│  • Services           • Entropy            • Recommendations        │
│  • Timeline           • Risk               • Report Assembly        │
│                       • Drift                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Data Collection

Extracts raw information from source code:

| Kernel | Purpose | Output |
|--------|---------|--------|
| `ast_scan` | Parse source files, extract symbols | Classes, methods, fields |
| `metrics` | Calculate code metrics | CC, LOC, MI per file |
| `dependency` | Build dependency graph | Import relationships |
| `partition` | Identify logical boundaries | Component clusters |
| `services` | Detect architectural patterns | Controllers, services, repos |
| `timeline` | Build lifecycle profiles | Modification history |

### Stage 2: Analysis

Processes Stage 1 data to derive insights:

| Kernel | Purpose | Output |
|--------|---------|--------|
| `stats_summary` | Distributional statistics | Mean, median, outliers |
| `hotspots` | Identify high-risk areas | Complexity/size hotspots |
| `dead_code` | Detect unused code | Unreachable elements |
| `coupling` | Martin coupling metrics | Ca, Ce, I, A, D |
| `entropy` | Information-theoretic analysis | Token/symbol entropy |
| `risk` | MCO risk assessment | Risk levels per component |
| `drift` | Spec-code alignment | Synchronization status |

### Stage 3: Reporting

Generates human-readable documentation:

| Kernel | Purpose | Output |
|--------|---------|--------|
| `section_executive` | Executive summary | Key findings |
| `section_overview` | Codebase metrics | Quality grades |
| `section_drift` | Drift analysis | Alignment tables |
| `section_recommendations` | Action items | Prioritized list |
| `report_assemble` | Final assembly | Complete markdown |

---

## 7. Orchestration and Execution

### Dependency Resolution

Kernels declare explicit dependencies. The orchestrator:

1. Builds a dependency graph from `requires` declarations
2. Performs topological sort
3. Identifies independent kernels (same topological level)
4. Executes in batches respecting dependencies

```
Stage 1 Execution:
  Batch 1: ast_scan (no dependencies)
  Batch 2: metrics, dependency, timeline, services (depend on ast_scan)
  Batch 3: partition (depends on dependency)

Stage 2 Execution:
  All 7 kernels run in parallel (independent)

Stage 3 Execution:
  Batch 1: section_executive, section_overview, section_drift, section_recommendations
  Batch 2: report_assemble (depends on all sections)
```

### Parallel Execution

Independent kernels execute concurrently using thread pools:

```python
# Parallel execution with dependency awareness
koas_run(workspace, parallel=True, workers=4)
```

Performance characteristics:
- **60K LOC Java project**: ~3.4 seconds total
- **Stage 1**: ~2.1s (I/O bound - file parsing)
- **Stage 2**: ~0.5s (CPU bound - analysis)
- **Stage 3**: ~0.02s (sequential for consistency)

### Audit Trail

Every execution is logged with:
- Timestamp and kernel name
- Input hash (SHA256 of configuration)
- Execution duration
- Success/failure status
- Output hash (SHA256 of results)
- Chain hash linking to previous entry

This creates a blockchain-style integrity chain that can be verified at any time.

---

## 8. RAG Integration

### Semantic Code Search

KOAS integrates with vector-based retrieval for semantic queries:

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Source Code ──▶ Chunker ──▶ Embeddings ──▶ Vector Index    │
│                                                             │
│  Query ──▶ Embedding ──▶ Similarity Search ──▶ Context      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Components:
- **ChromaDB**: Vector store for embeddings
- **Sentence Transformers**: Embedding models (all-MiniLM-L6-v2)
- **BM25**: Sparse keyword search
- **Hybrid Fusion**: RRF, weighted, or interleave strategies

### Upstream RAG (Encoder Role)

When a user asks: *"What are the risks in this codebase?"*

RAG retrieves relevant context:
- Previous audit findings for similar projects
- Industry standards and thresholds
- Component patterns and anti-patterns

This context enriches the orchestrator's decision-making.

### Downstream RAG (Decoder Role)

After kernel execution, RAG helps interpret results:
- Maps metrics to qualitative assessments
- Retrieves remediation patterns
- Generates contextualized recommendations

---

## 9. Quality Standards

### Complexity Metrics (McCabe)

Cyclomatic Complexity thresholds:

| Level | CC Range | Action |
|-------|----------|--------|
| Low | 1-10 | Normal maintenance |
| Moderate | 11-20 | Review recommended |
| High | 21-50 | Refactoring advised |
| Very High | >50 | Critical attention |

### Maintainability Index (Halstead)

Based on the SEI formula:

```
MI = 171 - 5.2 × ln(HV) - 0.23 × CC - 16.2 × ln(LOC)
```

Normalized to 0-100 scale:

| Grade | MI Range | Assessment |
|-------|----------|------------|
| A | 80-100 | Excellent |
| B | 60-79 | Good |
| C | 40-59 | Moderate |
| D | 20-39 | Poor |
| F | 0-19 | Critical |

### Coupling Metrics (Martin)

Robert C. Martin's package coupling metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Ca | Afferent coupling | Incoming dependencies |
| Ce | Efferent coupling | Outgoing dependencies |
| I | Ce / (Ca + Ce) | Instability (0=stable, 1=unstable) |
| A | Abstract / Total | Abstractness ratio |
| D | \|A + I - 1\| | Distance from main sequence |

### Technical Debt Estimation

Following the SQALE method:

```
Debt (hours) = Σ (violation_count × remediation_time)
```

Where remediation_time is based on industry benchmarks for each violation type.

---

## 10. References

### Academic Foundations

1. **McCabe, T.J.** (1976). "A Complexity Measure". *IEEE Transactions on Software Engineering*, SE-2(4), 308-320.

2. **Martin, R.C.** (2003). "Agile Software Development: Principles, Patterns, and Practices". *Pearson Education*.

3. **Halstead, M.H.** (1977). "Elements of Software Science". *Elsevier North-Holland*.

4. **SQALE Method** — Software Quality Assessment based on Lifecycle Expectations. SQALE Consortium.

### GS Framework

5. **Vitrac, O.** (2025). "Virtual/Hybrid R&D Laboratories built with Augmented-AI Agents". *LinkedIn Publication / Generative Simulation Initiative*.

### Software Standards

6. **ISO/IEC 25010:2011** — Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE).

7. **IEEE 1061-1998** — Standard for a Software Quality Metrics Methodology.

---

## Appendix: Design Philosophy Summary

> **"Science is no longer what the model can explain, but what a group of agents can coordinate."**
> — *Generative Simulation Initiative*

KOAS embodies this philosophy:

1. **Kernels are narrow but powerful** — Each does one thing well
2. **Orchestration enables emergence** — Complex insights from simple components
3. **RAG bridges language and computation** — Context-aware encoding/decoding
4. **Sovereignty ensures trust** — All data stays local
5. **Reproducibility enables auditability** — Same input, same output, always

This architecture enables industrial-scale code analysis while maintaining scientific rigor, explainability, and full control over sensitive codebases.

---

*KOAS is part of the RAGIX project — Retrieval-Augmented Generative Interactive eXecution Agent*

*Adservio Innovation Lab | 2025*
