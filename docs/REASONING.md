# Reasoning Engines in RAGIX

**From Local LLMs to Deep Reasoners: Architectural Patterns for Sovereign AI**

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 0.66.0
**Updated:** 2026-02-13

---

## Table of Contents

1. [The Reasoning Challenge](#1-the-reasoning-challenge)
2. [RAGIX Reasoning Architecture](#2-ragix-reasoning-architecture)
3. [The Four Reasoning Engines](#3-the-four-reasoning-engines)
4. [ContractiveReasoner: Branching Contractive Reasoning](#4-contractivereasoner)
5. [Reasoning v30: Graph-Based Reflective Reasoning](#5-reasoning-v30)
6. [Multi-Model Architecture](#6-multi-model-architecture)
7. [Comparison and Selection Guide](#7-comparison-and-selection-guide)
8. [Integration with MCP Tools](#8-integration-with-mcp-tools)
9. [Collective Intelligence: Multi-Agent Reasoning](#9-collective-intelligence)
10. [Configuration Reference](#10-configuration-reference)
11. [Benchmarking and Evaluation](#11-benchmarking-and-evaluation)
12. [Related Documentation](#12-related-documentation)

---

## 1. The Reasoning Challenge

### 1.1 Why Local LLMs Need External Reasoning

Local LLMs (7B-70B parameters) have limitations:

| Limitation | Manifestation | Impact |
|------------|---------------|--------|
| **Shallow reasoning** | Single-pass answers | Misses complex dependencies |
| **Hallucination** | Confident but wrong | Unreliable for critical tasks |
| **Context limits** | 4K-128K tokens | Cannot hold full problem state |
| **Inconsistency** | Different answers per run | Hard to verify |

**Solution:** Externalize reasoning into structured, controllable processes.

### 1.2 The Externalization Principle

Instead of asking an LLM to "think deeply," we:

```
Traditional:  LLM("Solve complex problem X") → (hopes for good answer)

Externalized:
  1. Decompose X into subproblems          [External logic]
  2. Solve each subproblem                 [LLM calls]
  3. Verify solutions                      [External checks]
  4. Integrate into final answer           [External logic]
  5. Review and iterate if needed          [External control]
```

**Benefits:**

- **Controllable:** We define the reasoning structure
- **Auditable:** Each step is logged and traceable
- **Iterative:** Can retry failed steps
- **Composable:** Different strategies for different problems

### 1.3 RAGIX's Four-Engine Approach

RAGIX provides four complementary reasoning engines, each suited to different task profiles:

| Engine | Paradigm | Best For | Location |
|--------|----------|----------|----------|
| **ReasoningLoop** (v1) | Iterative plan-execute | Production agent sessions | `ragix_core/reasoning.py` |
| **ReasoningGraph** (v30) | Graph state machine | Structured workflows, tool integration | `ragix_core/reasoning_v30/` |
| **ContractiveReasoner** | Tree-based decomposition | Deep exploration, uncertainty handling | `ragix_core/reasoning_slim/` |
| **Interpreter-Tutor** | Game-theoretic proof game | Hallucination suppression in slim LLMs | `ragix_core/reasoning_tutor/` |

All four engines:
- Work with local Ollama models (3B-14B)
- Are fully auditable with event traces
- Support sovereignty constraints (local-only inference)
- Can be combined with MCP tools and KOAS kernels

---

## 2. RAGIX Reasoning Architecture

### 2.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                             │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   Reasoning Engines                         │
│  ReasoningLoop │ ReasoningGraph │ Contractive │ Interp-Tutor│
│  (Iterative)   │ (Graph v30)    │ (Tree)      │ (Proof Game)│
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                       Tool Layer                            │
│    MCP Tools  │  Shell Commands  │  KOAS Kernels  │  APIs   │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLM Layer (Ollama)                      │
│  Planner: DeepSeek-R1/Mistral  │  Worker: Granite 3B-8B     │
│  Tutor: Mistral 7B             │  Verifier: Granite 3B      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Abstractions

| Abstraction | Purpose | Implementation |
|-------------|---------|----------------|
| **State** | Track reasoning progress | Dataclasses with history |
| **Node** | Single reasoning step | Classify, Plan, Execute, Reflect |
| **Transition** | State change logic | Conditional edges in graph |
| **Entropy** | Uncertainty measure | Model disagreement across samples |
| **Relevance** | Topic drift detection | BM25-style similarity |

### 2.3 Information Flow

```
Query → [Classify Complexity] → [Plan/Decompose] → [Execute Steps]
                                                          │
                                                          ▼
                                                   [Verify Results]
                                                          │
                                    ┌─────────────────────┴─────────────────────┐
                                    │                                           │
                                    ▼                                           ▼
                             [Pass: Respond]                          [Fail: Reflect]
                                                                                │
                                                                                ▼
                                                                    [Retry with feedback]
```

---

## 3. The Four Reasoning Engines

RAGIX provides four reasoning engines, progressively more sophisticated. Each addresses a different class of problems and failure modes.

### 3.1 ReasoningLoop (v1) — Production Agent Baseline

The original reasoning engine, used in `ragix_unix/agent.py` and `ragix_web/server.py` for production agent sessions.

- **Architecture:** Iterative plan-execute loop with configurable retry
- **States:** Classify → Plan → Execute → Verify → Respond
- **Strengths:** Simple, predictable, battle-tested in production
- **Integration:** Default engine for CLI and web UI sessions
- **Source:** `ragix_core/reasoning.py:470` (`class ReasoningLoop`)

### 3.2 ReasoningGraph (v30) — Reflective Graph

The advanced graph-based engine with reflective capabilities. Covered in detail in §5.

- **Architecture:** Graph state machine with 7 node types
- **States:** START → CLASSIFY → DIRECT_EXEC | (PLAN → EXECUTE → REFLECT → VERIFY) → RESPOND
- **Strengths:** Reflection on failures, experience corpus learning, task complexity classification
- **Integration:** Available in web UI and CLI via `--reasoning=graph`
- **Source:** `ragix_core/reasoning_v30/`

### 3.3 ContractiveReasoner — Deep Exploration

Tree-based reasoning with entropy-driven decomposition. Covered in detail in §4.

- **Architecture:** Tree of reasoning nodes with automatic branching
- **States:** NEW → NORMALIZE → DECOMPOSE → SOLVE → PRE-REVIEW → PEER-REVIEW → COLLAPSE → DONE
- **Strengths:** Handles uncertainty, built-in peer review, mathematical foundations (Banach contraction)
- **Integration:** Available via API, partial web integration
- **Source:** `ragix_core/reasoning_slim/`

### 3.4 Interpreter-Tutor — Hallucination Suppression

A game-theoretic architecture that transforms unreliable slim LLMs into disciplined agents. Instead of asking LLMs to reason, interaction is framed as a **turn-based proof game**: the LLM proposes moves, a deterministic Tutor validates them against evidence.

- **Architecture:** Turn-based proof game (PCG — Proof-Carrying Graph)
- **Key innovation:** Hallucination suppression by structure — illegal moves are rejected, truths require evidence proofs
- **Benchmarked:** 300+ game sessions across 10 benchmarks, 4 capability bands (search, formal reasoning, governance, engineering diagnosis)
- **Models tested:** DeepSeek-R1 14B, Granite 3B-8B, Mistral 7B, and others
- **Integration:** Standalone research engine (not yet wired into ragix-web)
- **Source:** `ragix_core/reasoning_tutor/`
- **Publication:** Research manuscript in preparation (see `ragix_core/reasoning_tutor/PUBLICATION_SUMMARY_v5a.md` for a summary of findings)

> **Note:** The Interpreter-Tutor is the subject of an ongoing research publication. Only a summary of the architecture and benchmark results is available in the repository. Full details will be published separately.

---

## 4. ContractiveReasoner

### 3.1 Overview

**ContractiveReasoner** implements branching contractive reasoning—a mathematically principled approach to deep reasoning with local LLMs.

**Location:** `ragix_core/reasoning_slim/`

**Key Idea:** Transform reasoning into a tree of contractive transformations, converging to a stable answer through fixed-point iteration.

### 3.2 Mathematical Foundation

#### Fixed-Point Iteration

Let $X$ be the problem. Define reasoning operators:
- $f_1$: Normalize (standardize input)
- $f_2$: Decompose (split into subproblems)
- $f_3$: Solve (answer leaf questions)
- $f_4$: Pre-review (internal quality check)
- $f_5$: Peer-review (external validation)
- $f_6$: Collapse (merge answers)

Composite operator:
$$F(X) = f_6(f_5(f_4(f_3(f_2(f_1(X))))))$$

Each operator is **contractive**:
$$|f_k(a) - f_k(b)| \le \alpha |a - b|, \quad \alpha < 1$$

By Banach's fixed-point theorem, iteration converges:
$$X_{n+1} = F(X_n) \rightarrow X^* \text{ (stable answer)}$$

#### Entropy Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Model entropy** | $H = -\sum p_i \log p_i$ | Disagreement across $k$ LLM samples |
| **Structural entropy** | $H_s = \log(\text{len}(Q))$ | Question complexity |
| **BM25 relevance** | $\text{BM25}(Q_{node}, Q_{root})$ | Topic similarity to original question |

### 3.3 Decision State Machine

```
                    ┌─────────────────────────────────────────────┐
                    │                  NEW NODE                   │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │  Compute Entropy  │
                              └───────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │  H >= threshold_high  │           │  H < threshold_high   │
        │     (uncertain)       │           │    (confident)        │
        └───────────────────────┘           └───────────────────────┘
                    │                                       │
                    ▼                                       ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │      DECOMPOSE        │           │     SOLVE LEAF        │
        │   (create children)   │           │   (direct answer)     │
        └───────────────────────┘           └───────────────────────┘
                    │                                       │
                    ▼                                       │
        ┌───────────────────────┐                           │
        │  Children all solved? │◄──────────────────────────┘
        └───────────────────────┘
                    │ YES
                    ▼
        ┌───────────────────────┐
        │     PRE-REVIEW        │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │    PEER REVIEW        │
        │   (if configured)     │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │      COLLAPSE         │
        │ (merge best answers)  │
        └───────────────────────┘
```

### 3.4 Quick Start

```python
import asyncio
from ragix_core.reasoning_slim import ContractiveReasoner

async def main():
    engine = ContractiveReasoner(
        model="granite3.1-moe:3b",
        max_depth=3,
        max_loops=6,
        entropy_decompose_threshold=0.5,
    )

    result = await engine.solve("What are the key factors in food packaging safety?")

    print(result.final_answer)
    print(result.to_mermaid())  # Visualization

asyncio.run(main())
```

### 3.5 CLI Usage

```bash
# Basic
python ContractiveReasoner.py "What causes ocean tides?"

# With telemetry
python ContractiveReasoner.py \
  --model mistral:7b-instruct \
  --max-depth 3 \
  --entropy-decompose-threshold 0.4 \
  --print-events \
  --export-trace trace.json \
  "Explain polymer migration in food packaging."

# Interactive chat
python ContractiveReasoner.py --chat
```

### 3.6 Peer Review System

The peer reviewer acts as a **constructive academic reviewer**:

```
Phase 1: PRE-REVIEW (Internal)
├── Semantic alignment check
├── Topic coverage analysis
├── Answer completeness scoring
└── Confidence assessment

Phase 2: PEER REVIEW (External LLM)
├── Call peer model with context
├── Parse verdict: approved/minor_revision/major_revision/reject
└── Extract improvement suggestions

Phase 3: RANKING & COLLAPSE
├── Combine pre-review and peer scores
├── Filter rejected branches
└── Merge best answers
```

**Peer Verdicts:**

| Verdict | Score Range | Action |
|---------|-------------|--------|
| `approved` | ≥ 0.6 | Keep as-is |
| `minor_revision` | 0.4 - 0.6 | Keep, flag for improvement |
| `major_revision` | 0.3 - 0.4 | Keep with noted issues |
| `reject` | < 0.3 | Remove from collapse |

### 3.7 Presets

```yaml
# Available presets
factual:    # Simple Q&A - high threshold, few iterations
complex:    # Multi-step - lower threshold, more samples
math:       # Mathematical - low threshold, deep trees
exploration: # Research - very low threshold, max depth
fast:       # Quick responses - shallow, few iterations
deep:       # Thorough - deep trees, many iterations
```

---

## 5. Reasoning v30

### 4.1 Overview

**Reasoning v30** implements graph-based reflective reasoning—a structured workflow approach with explicit state management and tool integration.

**Location:** `ragix_core/reasoning_v30/`
**Version:** 0.30.0

**Key Idea:** Model reasoning as a state machine where nodes perform specific functions and edges encode transition logic.

### 4.2 Architecture

```
Input
  ↓
[Classify] ─── Complexity classification (BYPASS/SIMPLE/MODERATE/COMPLEX)
  ↓
[DirectExec] ── Simple answers or routing
  ↓
[Plan] ─────── Multi-step planning with confidence tracking
  ↓
[Execute] ──── Tool-augmented step execution
  ↓
[Reflect] ──── Self-critique with experience corpus
  ↓
[Verify] ───── Answer verification
  ↓
[Respond] ──── Final response formatting
```

### 4.3 Core Components

#### Task Complexity Levels

```python
class TaskComplexity(Enum):
    BYPASS = "bypass"      # Greetings, simple queries → direct response
    SIMPLE = "simple"      # Single-step tasks → direct execution
    MODERATE = "moderate"  # 2-3 step tasks → light planning
    COMPLEX = "complex"    # Multi-step tasks → full planning + reflection
```

#### State Management

```python
@dataclass
class ReasoningState:
    goal: str                              # User's original query
    session_id: str                        # Unique session identifier
    complexity: TaskComplexity = None      # Classified complexity
    plan: Plan = None                      # Structured execution plan
    execution_results: List[ToolResult]    # Step execution results
    reflections: List[ReflectionAttempt]   # Self-critique history
    final_answer: str = None               # Final response
    events: List[ReasoningEvent]           # Audit trail
```

#### Node Types

| Node | Purpose | I/O |
|------|---------|-----|
| **ClassifyNode** | Determine task complexity | Query → Complexity |
| **DirectExecNode** | Handle simple tasks | Query → Answer |
| **PlanNode** | Generate execution plan | Query + Context → Plan |
| **ExecuteNode** | Run plan steps with tools | Plan → Results |
| **ReflectNode** | Self-critique with shell access | Results → Feedback |
| **VerifyNode** | Validate final answer | Answer → Verdict |
| **RespondNode** | Format final response | State → Response |

### 4.4 Quick Start

```python
from ragix_core.reasoning_v30 import (
    ReasoningGraph, GraphBuilder, ReasoningState,
    ClassifyNode, DirectExecNode, PlanNode,
    ExecuteNode, ReflectNode, VerifyNode, RespondNode,
    HybridExperienceCorpus,
)

# Define LLM functions
async def classify_fn(query: str) -> TaskComplexity: ...
async def llm_answer_fn(query: str) -> str: ...
async def generate_plan_fn(query: str, context: str) -> str: ...
async def execute_step_fn(step: PlanStep) -> ToolResult: ...
async def llm_reflect_fn(state: ReasoningState) -> str: ...
async def llm_verify_fn(answer: str, goal: str) -> bool: ...

# Build graph
corpus = HybridExperienceCorpus("experiences/")
graph = (GraphBuilder()
    .add_node(ClassifyNode(classify_fn))
    .add_node(DirectExecNode(llm_answer_fn))
    .add_node(PlanNode(generate_plan_fn, parse_plan_fn))
    .add_node(ExecuteNode(execute_step_fn))
    .add_node(ReflectNode(llm_reflect_fn, corpus, shell_fn))
    .add_node(VerifyNode(llm_verify_fn))
    .add_node(RespondNode())
    .build())

# Run
state = ReasoningState(goal="Find security vulnerabilities", session_id="001")
final_state = graph.run(state)
print(final_state.final_answer)
```

### 4.5 Experience Corpus

The experience corpus enables learning from past sessions:

```python
class HybridExperienceCorpus:
    """
    Persistent experience storage with:
    - Pattern matching for similar problems
    - Success/failure tracking
    - Retrieval for reflection context
    """

    def add_experience(self, state: ReasoningState, success: bool): ...
    def find_similar(self, query: str, k: int = 5) -> List[Experience]: ...
    def get_patterns(self, category: str) -> List[Pattern]: ...
```

### 4.6 Configuration

```yaml
# reasoning_config.yaml
graph:
  max_reflections: 3
  max_plan_retries: 2
  timeout_per_step: 60

reflect:
  enabled: true
  use_shell: true
  shell_timeout: 30
  max_commands: 5

experience:
  enabled: true
  corpus_path: "experiences/"
  similarity_threshold: 0.7

traces:
  enabled: true
  output_dir: "traces/"
  format: "ndjson"
```

---

## 6. Multi-Model Architecture

RAGIX supports **tiered model assignment**, where different reasoning roles use models optimized for their purpose. This is a key differentiator from single-model architectures.

### 6.1 Planner-Worker-Verifier (Agent Sessions)

Used in `ragix_web` and `ragix_unix` agent sessions. Configurable per session.

| Role | Default Model | Purpose | Config Parameter |
|------|---------------|---------|------------------|
| **Planner** | Mistral 7B / DeepSeek-R1 14B | Task classification, plan generation | `planner_model` |
| **Worker** | Granite 3B-8B | Step execution, fast inference | `worker_model` |
| **Verifier** | Granite 3B | Output validation, deterministic checks | `verifier_model` |

### 6.2 Worker + Tutor (KOAS Kernel Families)

Used in the `docs` and `reviewer` kernel families for LLM-assisted steps.

| Role | Model | Purpose |
|------|-------|---------|
| **Worker** | Granite 3B-8B | Generate initial summary / edit plan (fast, cheap) |
| **Tutor** | Mistral 7B | Refine, critique, and improve Worker output (domain knowledge) |

### 6.3 Tiered LLM Budget (Presenter)

The `presenter` family uses a graduated compute budget:

| Tier | LLM Usage | When |
|------|-----------|------|
| T0 | No LLM | Default: pure deterministic layout |
| T1 | Top-K clusters only | When semantic normalization requested |
| T2 | All clusters | Full normalization mode |
| T3 | Full polish | Executive mode with extensive rewriting |

### 6.4 Which Engine Uses Which Model

| Engine | Where Used | Model Assignment |
|--------|-----------|------------------|
| **ReasoningLoop** | ragix_web, ragix_unix | Planner-Worker-Verifier |
| **ReasoningGraph** (v30) | ragix_web, demos | Planner-Worker-Verifier |
| **ContractiveReasoner** | API, research | Single model (any 7B+) |
| **Interpreter-Tutor** | Research, benchmarks | Player (any LLM) + deterministic Tutor |
| **KOAS docs/reviewer** | KOAS pipelines | Worker + Tutor |
| **KOAS presenter** | KOAS pipelines | Tiered (T0-T3) |

---

## 7. Comparison and Selection Guide

### 7.1 Feature Comparison

| Feature | ReasoningLoop | ReasoningGraph v30 | ContractiveReasoner | Interpreter-Tutor |
|---------|---------------|---------------------|---------------------|-------------------|
| **Paradigm** | Iterative loop | Graph state machine | Tree decomposition | Turn-based proof game |
| **Branching** | None | Explicit (complexity) | Automatic (entropy) | Deterministic validation |
| **Reflection** | Retry on failure | Built-in reflect node | Built-in peer review | Tutor rejects illegal moves |
| **Tool Integration** | Shell + MCP | Built-in nodes | External | Shell (deterministic Tutor) |
| **Learning** | None | Persistent corpus | Per-session | Session traces |
| **Hallucination Control** | Verification step | Reflect + verify | Peer review | Structural suppression |
| **Production-ready** | Yes | Yes | Partial | Research |
| **Min Model Size** | Any | Any | 7B+ | 3B+ |

### 7.2 When to Use What

| Scenario | Recommended Engine | Rationale |
|----------|-------------------|-----------|
| **Production agent sessions** | ReasoningLoop | Simple, predictable, battle-tested |
| **Multi-step workflows** | ReasoningGraph v30 | Structured execution with reflection |
| **Research questions** | ContractiveReasoner | Deep exploration, entropy-based branching |
| **Uncertainty handling** | ContractiveReasoner | Automatic decomposition |
| **Tool-heavy tasks** | ReasoningGraph v30 | Native tool support |
| **Slim LLM reliability** | Interpreter-Tutor | Structural hallucination suppression |
| **Academic analysis** | ContractiveReasoner | Peer review built-in |
| **Benchmark evaluation** | Interpreter-Tutor | Game-theoretic scoring |

### 5.3 Combining Engines

The engines can be combined:

```python
# Use ContractiveReasoner for complex analysis
# Use Reasoning v30 for structured execution

async def hybrid_reasoning(query: str):
    # Stage 1: Analyze with ContractiveReasoner
    cr = ContractiveReasoner(model="mistral:7b-instruct")
    analysis = await cr.solve(f"Analyze: {query}")

    # Stage 2: Execute with Reasoning v30
    v30_state = ReasoningState(
        goal=f"Execute based on analysis: {analysis.final_answer}",
        session_id="hybrid"
    )
    result = v30_graph.run(v30_state)

    return result.final_answer
```

---

## 8. Integration with MCP Tools

### 6.1 Tool Access Patterns

Both engines can leverage MCP tools:

**ContractiveReasoner + MCP:**
```python
# During solve, inject tool results as context
engine = ContractiveReasoner(model="mistral:7b-instruct")

# Pre-gather context with MCP tools
scan_result = await mcp.call("ragix_scan_repo")
metrics = await mcp.call("ragix_ast_metrics", path="src/")

# Include in query
result = await engine.solve(
    f"Given this codebase structure: {scan_result}\n"
    f"And these metrics: {metrics}\n"
    f"What are the main architectural concerns?"
)
```

**Reasoning v30 + MCP:**
```python
# ExecuteNode natively calls tools
class MCPExecuteNode(ExecuteNode):
    def __init__(self, mcp_client):
        self.mcp = mcp_client

    async def execute(self, step: PlanStep) -> ToolResult:
        if step.tool_name.startswith("koas_"):
            return await self.mcp.call(step.tool_name, **step.params)
        return await super().execute(step)
```

### 6.2 KOAS Integration

KOAS kernels provide deterministic results for reasoning:

```
┌─────────────────────────────────────────────────────────────┐
│                    Reasoning Engine                         │
│         (ContractiveReasoner or Reasoning v30)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ "Analyze code quality"
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      KOAS Kernels                           │
│   ast_scan → metrics → hotspots → risk → recommendations    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Deterministic results
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Reasoning Engine                         │
│         (Interprets results, generates insights)            │
└─────────────────────────────────────────────────────────────┘
```

**Key benefit:** LLM reasons about **verified data**, not hallucinations.

---

## 9. Collective Intelligence

### 7.1 Multi-Agent Patterns

RAGIX enables collective intelligence through multiple reasoning agents:

#### Pattern 1: Ensemble Reasoning

```
Query → ┌─ ContractiveReasoner (exploration) ─┐
        │                                     │
        ├─ Reasoning v30 (structured) ────────┼─→ Merge → Final Answer
        │                                     │
        └─ Direct LLM (baseline) ─────────────┘
```

#### Pattern 2: Specialist Collaboration

```
Query → [Router] → ┌─ Security Specialist (KOAS Security) ─┐
                   │                                       │
                   ├─ Code Analyst (KOAS Audit) ───────────┼─→ Synthesize
                   │                                       │
                   └─ Documentation Writer ────────────────┘
```

#### Pattern 3: Hierarchical Reasoning

```
                    ┌─────────────────┐
                    │ Master Reasoner │
                    │ (ContractiveR)  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │Worker 1 │        │Worker 2 │        │Worker 3 │
    │(v30)    │        │(v30)    │        │(v30)    │
    └─────────┘        └─────────┘        └─────────┘
```

### 7.2 Information Exchange Beyond Text

Reasoning agents exchange structured data:

| Data Type | Purpose | Example |
|-----------|---------|---------|
| **Numbers** | Quantitative analysis | Risk scores, metrics |
| **Graphs** | Structural relationships | Dependency trees |
| **Vectors** | Semantic similarity | Embeddings for retrieval |
| **Schemas** | Data contracts | JSON schemas for tools |
| **Events** | Audit trails | Reasoning step logs |

### 7.3 Synchronous vs Asynchronous

| Mode | Use Case | Implementation |
|------|----------|----------------|
| **Sync** | Sequential dependencies | `await engine.solve()` |
| **Async** | Parallel exploration | `asyncio.gather(*)` |
| **Streaming** | Real-time feedback | WebSocket events |
| **Background** | Long-running analysis | Task queues |

---

## 10. Configuration Reference

### 8.1 ContractiveReasoner Parameters

```python
engine = ContractiveReasoner(
    # Ollama Connection
    base_url="http://localhost:11434",
    model="granite3.1-moe:3b",

    # Tree Structure
    max_depth=4,                         # Maximum tree depth
    max_loops=10,                        # Global iterations
    max_global_tokens=64000,             # Token budget
    max_branch_tokens=16000,             # Per-branch limit
    max_concurrent_branches=4,           # Parallelism

    # Entropy Thresholds
    entropy_decompose_threshold=0.5,     # High → decompose
    entropy_collapse_threshold=0.4,      # Low → collapse
    entropy_gamma_min_reduction=0.05,    # Min reduction required
    k_entropy_samples=4,                 # Samples for estimation

    # Semantic Filtering
    min_relevance_threshold=0.15,        # Prune below this
    max_rebranch_attempts=2,             # Auto-rebranch limit

    # Timeouts
    timeout_sec=120,                     # Per-call timeout
)
```

### 8.2 Reasoning v30 Configuration

```yaml
# reasoning_config.yaml
version: "0.30.0"

graph:
  max_reflections: 3
  max_plan_retries: 2
  timeout_per_step: 60
  complexity_bypass_threshold: 0.9

reflect:
  enabled: true
  use_shell: true
  shell_timeout: 30
  max_commands: 5
  read_only: true

experience:
  enabled: true
  corpus_path: "experiences/"
  similarity_threshold: 0.7
  max_similar: 5

traces:
  enabled: true
  output_dir: "traces/"
  format: "ndjson"
  include_prompts: false

profiles:
  developer:
    max_reflections: 2
    verbose: true
  production:
    max_reflections: 1
    verbose: false
```

### 8.3 Model-Specific Tuning

| Model | Entropy Threshold | Max Depth | Notes |
|-------|-------------------|-----------|-------|
| `granite3.1-moe:3b` | 0.3 - 0.5 | 3-4 | High entropy baseline |
| `mistral:7b-instruct` | 0.4 - 0.6 | 3 | Balanced |
| `qwen2.5:7b` | 0.4 - 0.6 | 3 | Good instruction following |
| `llama3.2:3b` | 0.3 - 0.5 | 3 | Fast, shallow |
| `llama3.1:70b` | 0.5 - 0.7 | 4 | Deep reasoning capable |

---

## 11. Benchmarking and Evaluation

### 9.1 ContractiveReasoner Benchmarks

**Framework:** `ragix_core/reasoning_slim/benchmarks/`

**Paper Scenarios (S0-S6):**

| Scenario | Description | Key Settings |
|----------|-------------|--------------|
| S0 | Brute force (no reasoning) | Direct LLM call |
| S1 | Baseline fast | No peer, shallow |
| S2 | Self-consistency | Multiple samples |
| S3 | Light peer | Fast peer checker |
| S4 | Diverse peer | Different model |
| S5 | Strong peer | Same family peer |
| S6 | Safety focus | Clarification requests |

**Running Benchmarks:**
```bash
python benchmarks/run_paper_experiments.py \
  --scenarios benchmarks/configs/paper_scenarios.yaml \
  --tasks benchmarks/configs/task_set.yaml \
  --output-dir benchmarks/results/
```

### 9.2 Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Keyword coverage** | `found / expected` | Answer completeness |
| **Token efficiency** | `quality / tokens` | Cost-effectiveness |
| **Decomposition rate** | `branched / total` | Reasoning depth |
| **Peer agreement** | `approved / reviewed` | External validation |
| **Time to solution** | Wall clock | Practical efficiency |

### 9.3 Quality Assessment

```python
# Example evaluation
def evaluate_result(result, expected_keywords):
    return {
        "keywords_found": sum(1 for k in expected_keywords if k in result.final_answer),
        "keywords_total": len(expected_keywords),
        "nodes_created": len(result.tree),
        "max_depth": result.max_depth_reached,
        "total_tokens": result.metrics.total_tokens,
        "duration_sec": result.metrics.duration,
    }
```

---

## 12. Related Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| [MCP.md](MCP.md) | Model Context Protocol in RAGIX |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Overall RAGIX architecture |
| [KOAS.md](KOAS.md) | Kernel-Orchestrated Audit System |

### Detailed References

| Resource | Location |
|----------|----------|
| ReasoningLoop (v1) | `ragix_core/reasoning.py` |
| ReasoningGraph (v30) API | `ragix_core/reasoning_v30/__init__.py` |
| ContractiveReasoner README | `ragix_core/reasoning_slim/README.md` |
| Interpreter-Tutor README | `ragix_core/reasoning_tutor/README.md` |
| Interpreter-Tutor publication summary | `ragix_core/reasoning_tutor/PUBLICATION_SUMMARY_v5a.md` |
| Reasoning v30 demo | `demos/reasoning_v30_demo.py` |
| Multi-model benchmark runner | `demos/run_reasoning_benchmark.sh` |
| Benchmark configs | `ragix_core/reasoning_slim/benchmarks/configs/` |

### External Resources

| Resource | URL |
|----------|-----|
| Ollama | https://ollama.ai/ |
| RAGIX Repository | https://github.com/ovitrac/RAGIX |
| Fixed-Point Theory | https://en.wikipedia.org/wiki/Banach_fixed-point_theorem |

---

## Appendix A: Reasoning Engine Quick Reference

### ReasoningLoop (v1) States

```
CLASSIFY → PLAN → EXECUTE → VERIFY → RESPOND
                     │          │
                     └──────────┘ (retry on failure)
```

### ContractiveReasoner States

```
NEW → NORMALIZE → DECOMPOSE → SOLVE → PRE-REVIEW → PEER-REVIEW → COLLAPSE → DONE
                      │                                              │
                      └──────────── (children) ──────────────────────┘
```

### Reasoning v30 States

```
START → CLASSIFY → DIRECT_EXEC ──────────────────────────────→ RESPOND
                       │
                       └→ PLAN → EXECUTE → REFLECT → VERIFY → RESPOND
                                    │          │
                                    └──────────┘ (retry loop)
```

### Interpreter-Tutor States

```
PLAYER (LLM)              TUTOR (Deterministic)
────────────              ─────────────────────
Propose move       ──▶    Validate legality
                   ◀──    Accept / Reject + reason
Execute (if legal) ──▶    Verify outcome
                   ◀──    Score + next prompt
Repeat until goal or budget exhausted
```

### Decision Matrix

| Query Type | Complexity | Engine | Strategy |
|------------|------------|--------|----------|
| Greeting | BYPASS | v30 | Direct response |
| Factual Q&A | SIMPLE | ReasoningLoop | Direct exec |
| Multi-step task | MODERATE | ReasoningLoop / v30 | Light planning |
| Research question | COMPLEX | Contractive | Deep exploration |
| Uncertain problem | HIGH ENTROPY | Contractive | Decomposition |
| Tool workflow | STRUCTURED | v30 | Plan + Execute |
| Slim LLM task | RELIABILITY | Interpreter-Tutor | Proof game |
| Benchmark eval | SCORING | Interpreter-Tutor | Game sessions |

---

**Document Version:** 2.0.0
**Last Updated:** 2026-02-13
**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
