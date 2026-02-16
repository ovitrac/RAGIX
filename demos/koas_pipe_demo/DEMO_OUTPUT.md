# KOAS Memory Pipe — Demo Output

**What you're looking at:** A single CLI tool that reads 68,511 tokens of source code,
and for each question, instantly retrieves the exact ~2,000 tokens that answer it —
from a different file each time. No AI model involved. Pure search.

---

## The Problem

You have 8 source files describing an architecture. Together they are **7,283 lines**
and **68,511 tokens**. An AI assistant's injection budget is 3,000 tokens — that's
**4.4% of the corpus**.

Which 4.4% do you pick? And what if the next question needs a completely different 4.4%?

```
File                                      Lines  ~Tokens
──────────────────────────────────────── ────── ────────
KOAS.md                                     614     7333     <- philosophy
ARCHITECTURE.md                             708     6603     <- system design
base.py                                     349     2790     <- kernel interface
registry.py                                 342     2491     <- auto-discovery
md_edit_plan.py                            2964    27250     <- complex real kernel
pres_slide_plan.py                          825     8051     <- slide generation
summaryctl.py                               911     8764     <- summary pipeline
cli.py                                      570     5229     <- memory CLI itself
────────────────────────────────────────────────────────
TOTAL                                      7283    68511
```

**One command** chunks all 8 files into 41 searchable memory items (1.1 MB SQLite database).
Re-running is instant — SHA-256 dedup skips unchanged files.

---

## Four Questions, Four Different Answers

Each query hits a **different file** and retrieves a **different cross-section** of
the codebase. This is the core value: Memory doesn't return the same boilerplate
every time — it understands what each question needs.

---

### Q1: "What problem does KOAS solve?"

**Memory returns:** `KOAS.md`, chunk 0 (1,870 tokens)

> KOAS (Kernel-Orchestrated Audit System) is a sovereign, local-first
> computation framework that applies scientific kernel principles to software
> audits, document analysis, slide generation, Markdown review, and security
> scanning. As of v0.66, KOAS comprises **75 deterministic kernels across 5
> families**, each following the three-stage pipeline pattern (data collection
> -> analysis -> reporting).
>
> Core Principles:
> 1. **Sovereignty**: All processing happens locally -- no cloud dependencies
> 2. **Reproducibility**: Deterministic kernels produce identical outputs
> 3. **Auditability**: Complete execution trail with cryptographic verification
> 4. **Composability**: Modular kernels with explicit dependencies
> 5. **LLM-Ready**: Structured summaries designed for AI-assisted interpretation
>
> KOAS inherits its architecture from the **Generative Simulation (GS)**
> initiative. In the GS framework, **scientific agents** embed one or more
> LLMs that operate **scientific kernels** -- specialized software libraries
> designed for machine-to-machine interaction rather than human use.
>
> ```
>          SCIENTIFIC AGENT
>    ┌─────────┐   ┌──────────┐   ┌─────────┐
>    │ Encoder │──>│  Kernel  │──>│ Decoder │
>    │  (LLM)  │   │(Compute) │   │  (LLM)  │
>    └─────────┘   └──────────┘   └─────────┘
> ```
>
> Key insight from GS: "Most time is spent inside the scientific kernels,
> not the LLMs."

**Why this is the right answer:** The question asks about the *problem* KOAS solves.
Memory retrieved the philosophical introduction — origins, principles, the scientific
agent pattern — not implementation details. From 68,511 tokens, it found the 1,870
that explain *why KOAS exists*.

---

### Q2: "What is the kernel interface contract?"

**Memory returns:** `base.py`, chunk 0 (2,084 tokens)

> ```python
> class Kernel(ABC):
>     """
>     A kernel is a specialized computation unit that:
>     1. Encapsulates domain logic (AST parsing, metrics, graph analysis)
>     2. Wraps existing tools (RAGIX CLI, Python libraries, shell scripts)
>     3. Produces structured output (JSON data + human-readable summary)
>     4. Is fully deterministic (same input = same output)
>     5. Contains no LLM (pure computation, no AI reasoning inside)
>
>     Subclasses must implement:
>         - compute(): Core computation logic
>         - summarize(): Generate human-readable summary
>     """
>
>     @abstractmethod
>     def compute(self, input: KernelInput) -> Dict[str, Any]: ...
>
>     @abstractmethod
>     def summarize(self, data: Dict[str, Any]) -> str: ...
>
>     def run(self, input: KernelInput) -> KernelOutput:
>         """Execute kernel with full traceability.
>         Do NOT override -- override compute() and summarize() instead."""
> ```

**Why this is the right answer:** The question asks about the *contract*. Memory
skipped all 8 files and went straight to `base.py` — the abstract base class that
defines `compute()` and `summarize()`. This is the actual Python code, not a
description of it. A developer can read this and immediately understand the interface.

---

### Q3: "What are the five kernel families?"

**Memory returns:** `ARCHITECTURE.md`, chunk 1 (2,086 tokens)

> As of v0.66, RAGIX includes **75 deterministic computation kernels**
> organized into 5 families via the KOAS architecture.
>
> | Family        | Kernels | Scope                                          | LLM Usage                  |
> |---------------|---------|------------------------------------------------|----------------------------|
> | **audit**     | 27      | Java codebase analysis (AST, metrics, risk)    | None (pure deterministic)  |
> | **docs**      | 17      | Document summarization (Leiden clustering)      | Worker + Tutor pattern     |
> | **presenter** | 8       | MARP slide deck generation (3 compression)     | Optional normalization     |
> | **reviewer**  | 13      | Traceable Markdown review (chunk edits)        | Worker + Tutor pattern     |
> | **security**  | 10      | Vulnerability scanning, secrets detection      | None (pure deterministic)  |
>
> Three-Stage Pipeline:
> ```
> Stage 1: Collection     Stage 2: Analysis       Stage 3: Reporting
> ─────────────────────   ──────────────────────   ──────────────────
> - Parse, extract        - Cross-reference        - Generate sections
> - Inventory             - Score, classify        - Assemble report
> - Build indexes         - Detect patterns        - Apply output level
> ```

**Why this is the right answer:** The question asks about the *five families*. Memory
found the ARCHITECTURE.md section with the exact table — family names, kernel counts,
scope, and LLM usage — plus the 3-stage pipeline pattern. Not KOAS.md (philosophy),
not base.py (code). The architectural reference.

---

### Q4: "How does a real kernel handle complexity?"

**Memory returns:** `md_edit_plan.py`, chunk 10 (1,993 tokens)

> ```python
> # --- Preflight 2: Sub-chunk splitting ---
> sub_chunks = _maybe_split_chunk(
>     chunk, lines, protected_spans or [], max_tokens=max_edit_chunk_tokens,
> )
> was_split = len(sub_chunks) > 1
>
> if was_split:
>     # Process each sub-chunk independently, merge results
>     all_ops: List[Dict[str, Any]] = []
>     merged_stats: Dict[str, Any] = {
>         "chunk_id": chunk.chunk_id,
>         "worker_calls": 0,
>         "tutor_calls": 0,
>         "preflight_tier": context_tier_default,
>         "preflight_math_masked": math_masked,
>         "preflight_split": True,
>         "sub_chunks": [],
>         ...
>     }
>     for sci, sub in enumerate(sub_chunks):
>         sub_result = _run_single_edit_plan(...)
>         # ...adaptive tier escalation, content recipes, Policy C/D...
> ```

**Why this is the right answer:** The question asks about real-world *complexity*.
Memory bypassed all documentation files and went deep into `md_edit_plan.py` — the
2,964-line reviewer kernel — finding the preflight pipeline: sub-chunk splitting,
adaptive tier escalation, content recipes. This is not documentation *about*
complexity — it's the actual code that *handles* it.

---

## The Numbers

```
Corpus:            68,511 tokens across 8 files, stored as 41 chunks
Budget per query:   3,000 tokens (4.4% of corpus)
Avg. tokens used:   2,008 (2.9% of corpus)

Query differentiation:
  Q1 (philosophy)      → KOAS.md:0
  Q2 (interface)       → base.py:0
  Q3 (families)        → ARCHITECTURE.md:1
  Q4 (implementation)  → md_edit_plan.py:10

  4/4 queries returned different chunks from different files.
```

---

## The Synthesis

No single file contains all of this. But from 4 targeted recalls, an AI assistant
could produce:

> KOAS is a sovereign computation framework (from KOAS.md) built on a formal
> kernel contract (from base.py): every kernel implements `compute()` for
> deterministic processing and `summarize()` for LLM consumption. Five families
> -- audit, docs, presenter, reviewer, summary -- organize 75 kernels into a
> 3-stage pipeline: collection, analysis, reporting (from ARCHITECTURE.md).
> Real implementations like md_edit_plan.py show how this contract scales to
> complex tasks: 2,964 lines of preflight masking, adaptive tier escalation,
> and content recipes -- all deterministic guards before the LLM call.

This brief was assembled from **4 different source files**, each surfaced by a
different question. Without Memory, a human would need to manually find the right
2.9% of 68,511 tokens for each question. With Memory, each query takes
**less than 1 second**.

---

## What Makes This Different

| Without Memory | With Memory |
|----------------|-------------|
| Feed all 68K tokens to the LLM | Select 2K tokens per question |
| Same context for every question | Different context for each question |
| Truncation loses critical sections | BM25 ranking surfaces the relevant section |
| No persistence — repeat every session | Ingest once, query forever |
| Manual file selection | Automatic cross-file retrieval |
| Requires an LLM to process | Pure search — no AI, no cloud, no cost |

---

*Generated by `demos/koas_pipe_demo/run_demo.sh --verbose` on 2026-02-16.
No LLM was used to produce any of the recalled content.*
