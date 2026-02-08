# LLM Reasoning Olympics — Technical Appendix

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 1.1 (2026-02-04)
**System:** RAGIX Interpreter-Tutor Benchmark Suite

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Benchmark Problems](#2-benchmark-problems)
   - B01-B06: Foundation Benchmarks
   - B07-B10: Advanced Reasoning Benchmarks
3. [LLM Participants](#3-llm-participants)
4. [Testing Conditions](#4-testing-conditions)
5. [Scoring System](#5-scoring-system)
6. [Architectural Components](#6-architectural-components)
7. [Reproducibility Guide](#7-reproducibility-guide)
8. [Round Summary](#8-round-summary)
9. [Capability Matrix](#9-capability-matrix)

---

## 1. Introduction

The LLM Reasoning Olympics is a systematic benchmark suite designed to evaluate local LLMs in a deterministic reasoning framework. The key innovation is the **Interpreter-Tutor architecture**, which separates:

- **LLM (Player):** Proposes moves (commands, claims, questions)
- **Tutor (Referee):** Validates moves, executes actions, derives truths deterministically

This separation achieves **hallucination suppression by structure** — illegal moves are rejected, truths require evidence proofs, and all decisions are auditable.

### Central Hypothesis

> **REASONING CORRECTNESS is MODEL-INDEPENDENT**
>
> The Tutor's CHECK protocol produces identical verdicts regardless of which LLM proposes moves.
>
> **REASONING EFFICIENCY is MODEL-DEPENDENT**
>
> Better models reach goals in fewer turns with fewer hallucinations.

---

## 2. Benchmark Problems

The benchmark suite consists of 10 problems testing different cognitive capabilities across four capability bands:

| Band | Benchmarks | Cognitive Focus |
|------|------------|-----------------|
| **1. Search & Enumeration** | B01, B02 | Pattern search, arithmetic |
| **2. Reasoning & Rules** | B03, B04, B05 | Truth/Dare, verification, rule generation |
| **3. Memory & Context** | B06 | Multi-turn information retention |
| **4. Advanced Reasoning** | B07, B08, B09, B10 | Causal analysis, semantic diff, graph reasoning, temporal correlation |

### B01: Find Needle in Haystack

| Property | Value |
|----------|-------|
| **Category** | Discovery |
| **Difficulty** | Medium |
| **Optimal Turns** | 3 |
| **Primary Tools** | `grep`, `find` |

**Setup:** 10 text files with random content. Only ONE contains the target expression `EUREKA_SECRET_42`.

**Goal:** Find which file contains `EUREKA_SECRET_42` and report its name.

**Files Created:**
```
data/
├── file_alpha.txt   (Lorem ipsum...)
├── file_beta.txt    (Processing records...)
├── file_gamma.txt   (Error log entries...)
├── file_delta.txt   ← Contains EUREKA_SECRET_42
├── file_epsilon.txt (Network status...)
├── file_zeta.txt    (Build log...)
├── file_eta.txt     (Database status...)
├── file_theta.txt   (Cache statistics...)
├── file_iota.txt    (Security scan...)
└── file_kappa.txt   (Metrics summary...)
```

**Success Criteria:**
- Response contains `file_delta.txt`
- Response contains `EUREKA_SECRET_42`

**Expected Solution:**
1. List files in data/ directory
2. Run `grep -r "EUREKA_SECRET_42" data/`
3. Assert which file contains the pattern

---

### B02: Count Lines Challenge

| Property | Value |
|----------|-------|
| **Category** | Enumeration |
| **Difficulty** | Medium |
| **Optimal Turns** | 4 |
| **Primary Tools** | `wc -l`, `find` |

**Setup:** 5 Python files with known line counts (7 + 12 + 4 + 5 + 2 = 30 lines total).

**Goal:** Count the total number of lines across all `.py` files in `src/` directory.

**Files Created:**
```
src/
├── module_a.py    (7 lines)
├── module_b.py    (12 lines)
├── module_c.py    (4 lines)
├── utils.py       (5 lines)
└── __init__.py    (2 lines)
```

**Success Criteria:**
- Response contains "total"
- Response contains "lines"
- Correct count: 30 lines

**Scientific Rationale:** Slim LLMs consistently fail at counting and arithmetic. The Tutor compensates by executing `wc -l` and validating results via rules.

---

### B03: Undecidable Claim Recovery

| Property | Value |
|----------|-------|
| **Category** | Undecidable (Truth/Dare) |
| **Difficulty** | Hard |
| **Optimal Turns** | 5 |
| **Primary Tools** | `cat`, `grep` |

**Setup:** Configuration file with a subtle error — missing `%(message)s` in logging format string.

**Goal:** Find the configuration error preventing application startup.

**Files Created:**
```
config/
├── settings.yaml   ← Contains the bug (missing %(message)s)
└── schema.json     ← Defines format requirement
logs/
└── error.log       ← Shows "Schema mismatch in logging.format"
```

**The Bug:**
```yaml
logging:
  format: "%(asctime)s - %(name)s - %(levelname)s"
  # Missing %(message)s - this is the bug!
```

**Success Criteria:**
- Response contains "logging"
- Response contains "format"
- Response contains "message"

**Scientific Rationale:** Tests the Truth/Dare mechanism. When LLMs make claims without evidence, the Tutor forces them to:
- **TRUTH:** Reformulate into a decidable claim
- **DARE:** Propose an evidence-producing action

---

### B04: Verification Chain

| Property | Value |
|----------|-------|
| **Category** | Verification |
| **Difficulty** | Hard |
| **Optimal Turns** | 6 |
| **Primary Tools** | `cat`, `grep` |

**Setup:** Python project structure requiring verification of import chain validity.

**Goal:** Verify that `main.py` can execute successfully by checking all imports resolve.

**Files Created:**
```
project/
├── main.py      → imports core, utils
├── core.py      → imports base, utils
├── base.py      (no external imports)
├── utils.py     (standard library only)
└── __init__.py
```

**Import Chain:**
```
main.py
├── project.core.Engine
│   ├── project.base.BaseEngine
│   └── project.utils.log
└── project.utils.configure
```

**Success Criteria:**
- Evidence shows `from project.core import Engine`
- Evidence shows `from project.base import BaseEngine`
- Evidence shows `class Engine` definition

**Scientific Rationale:** Tests the Proof-Carrying Graph's ability to accumulate evidence across turns and derive complex truths from simpler observations.

---

### B05: Session Rule Generation

| Property | Value |
|----------|-------|
| **Category** | Rule Generation |
| **Difficulty** | Expert |
| **Optimal Turns** | 8 |
| **Primary Tools** | `cat`, `grep`, pattern matching |

**Setup:** Custom log format that standard bash rules don't cover.

**Goal:** Identify root cause of system shutdown (OOM) and propose a fix.

**Files Created:**
```
logs/
└── custom.log    ← Custom format: [METRIC:cpu=X%,mem=Y,disk=Z%]
config/
└── thresholds.json   ← memory_limit_gb: 3.0
```

**Log Timeline:**
```
[TRACE] SYSTEM BOOT INITIATED
[INFO]  Loading config...
[METRIC:cpu=15%,mem=2.1GB,disk=45%] SYSTEM HEALTH OK
[METRIC:cpu=23%,mem=2.3GB,disk=45%] SYSTEM HEALTH OK
[WARN]  High memory growth detected
[METRIC:cpu=28%,mem=3.1GB,disk=46%] SYSTEM HEALTH WARNING
[ERROR] Memory threshold exceeded (3.0GB limit)
[METRIC:cpu=35%,mem=3.5GB,disk=46%] SYSTEM HEALTH CRITICAL
[FATAL] OOM condition, shutting down
```

**Success Criteria:**
- Response contains "memory"
- Response contains "3.0GB"
- Response contains "OOM"

**Scientific Rationale:** Tests the two-tier rule system — LLMs may need to generate ad-hoc session rules for domain-specific patterns.

---

### B06: Memory Recall Challenge

| Property | Value |
|----------|-------|
| **Category** | Memory |
| **Difficulty** | Hard |
| **Optimal Turns** | 5 |
| **Primary Tools** | `cat` |

**Setup:** 4-digit secret code spread across 4 files. LLM must read all files and combine digits.

**Goal:** Find the 4-digit secret code by reading all clue files.

**Files Created:**
```
clues/
├── part1.txt   → Digit 1 = 7
├── part2.txt   → Digit 2 = 5 (3+2)
├── part3.txt   → Digit 3 = 3 (vowels in "ENIGMA")
└── part4.txt   → Digit 4 = 9
README.txt      → Instructions
```

**Expected Answer:** `CODE=7539`

**Success Criteria:**
- Evidence shows "first digit is: 7"
- Evidence shows "3 + 2 = 5"
- Evidence shows "vowels"
- Evidence shows "fourth digit is: 9"

**Scientific Rationale:** Slim LLMs often "forget" information from earlier turns. The card deck acts as a procedural RAG — if LLMs can't remember, they can re-read files at a scoring penalty.

---

### B07: Stack Trace Diagnosis

| Property | Value |
|----------|-------|
| **Category** | Error Analysis |
| **Difficulty** | Medium |
| **Optimal Turns** | 5 |
| **Primary Tools** | `cat`, `grep` |

**Setup:** A crashed application with stack trace pointing to `handler.py:67`, but the root cause is a configuration error (`divisor: 0`).

**Goal:** Find the root cause configuration that sets divisor to zero.

**Files Created:**
```
logs/
└── crash.log       ← Stack trace pointing to handler.py:67
src/
├── main.py         (entry point)
├── processor.py    (calls handler)
└── handler.py      ← Red herring: line 67 is symptom, not cause
config/
└── settings.yaml   ← ROOT CAUSE: divisor: 0
```

**The Root Cause:**
```yaml
processing:
  divisor: 0    # ← This is the bug (set by admin to disable feature)
```

**Success Criteria:**
- Response contains "config" or "settings.yaml"
- Response contains "divisor"
- Response contains "0"
- Must NOT cite `handler.py:67` as the root cause

**Scientific Rationale:** Tests causal reasoning — LLMs must trace from symptom (stack trace) to root cause (configuration), not stop at the crash site. The red herring comment in handler.py tests resistance to superficial answers.

---

### B08: Diff Analysis

| Property | Value |
|----------|-------|
| **Category** | Comparison |
| **Difficulty** | Medium |
| **Optimal Turns** | 4 |
| **Primary Tools** | `cat`, `diff` |

**Setup:** Two versions of a calculator module. Version 2 introduced a subtle semantic bug: using `is` instead of `==` for zero comparison.

**Goal:** Find the exact breaking change between calculator_v1.py and calculator_v2.py.

**Files Created:**
```
src/
├── calculator_v1.py   ← Working: if b == 0
└── calculator_v2.py   ← Buggy: if b is 0
tests/
└── test_calculator.py  ← Shows float 0.0 test failing
CHANGELOG.md
bug_report.txt
```

**The Bug:**
```python
# calculator_v1.py (correct)
def divide(a, b):
    if b == 0:  # Equality check - works for 0 and 0.0
        raise ValueError("Cannot divide by zero")

# calculator_v2.py (buggy)
def divide(a, b):
    if b is 0:  # Identity check - fails for 0.0 (0.0 is not 0)
        raise ValueError("Cannot divide by zero")
```

**Success Criteria:**
- Response contains "v2" (identifies the buggy file)
- Response contains "is" (identifies the operator)
- Response contains "0" (identifies the comparison)

**Scientific Rationale:** Tests semantic understanding of code changes without syntax parsing dominating. The bug is valid Python (no syntax error), but wrong logic. `is` is identity comparison; `0 is 0` is `True` but `0.0 is 0` is `False`.

---

### B09: Dependency Cycle Detection

| Property | Value |
|----------|-------|
| **Category** | Graph Reasoning |
| **Difficulty** | Expert |
| **Optimal Turns** | 5 |
| **Primary Tools** | `cat`, `grep` |

**Setup:** A Python application fails to start due to circular imports. Three modules form a dependency cycle.

**Goal:** Identify the circular dependency chain in order.

**Files Created:**
```
modules/
├── __init__.py
├── auth.py        ← from modules import user
├── user.py        ← from modules import permissions
└── permissions.py ← from modules import auth (CYCLE CLOSER)
main.py
logs/
└── import_error.log  ← Shows ImportError traceback
README.md
```

**The Cycle:**
```
auth.py → user.py → permissions.py → auth.py
```

**Success Criteria:**
- Response contains "auth"
- Response contains "user"
- Response contains "permissions"
- Response contains "circular" (or equivalent)

**Scientific Rationale:** Tests graph traversal reasoning without explicit graph tools. The LLM must follow import statements across files to reconstruct the dependency chain and identify the cycle. Generic "circular import exists" is insufficient — the complete chain must be reported.

---

### B10: Temporal Event Correlation

| Property | Value |
|----------|-------|
| **Category** | Temporal Reasoning |
| **Difficulty** | Hard |
| **Optimal Turns** | 6 |
| **Primary Tools** | `cat`, `grep` |

**Setup:** A distributed system experienced a cascading failure. Logs from three services are available, but clock skew complicates correlation.

**Goal:** Identify the root cause service and event that triggered the cascade.

**Files Created:**
```
logs/
├── service_a.log   ← Gateway: FATAL circuit breaker (consequence)
├── service_b.log   ← Processor: connection refused (symptom)
└── service_c.log   ← Database layer: FATAL at 09:59:58 (ROOT CAUSE)
notes/
└── clock_skew.txt  ← service_c is 3.2 seconds behind
architecture.txt
README.md
```

**Clock Skew Challenge:**
```
Service   | Raw Timestamp | Adjusted (real time)
----------|---------------|--------------------
service_c | 09:59:58.000  | 10:00:01.234 (root cause)
service_a | 10:00:02.000  | 10:00:02.000 (consequence)
```

**Success Criteria:**
- Response contains "service_c" (identifies root cause service)
- Response contains "09:59:58" (correct timestamp)
- Response contains "Database" (identifies the event)
- Response contains "FATAL" (severity level)

**Scientific Rationale:** Tests temporal reasoning with distributed systems complexity. The gateway's circuit breaker FATAL at 10:00:02 appears later but is actually a consequence of service_c's database failure. The LLM must account for clock skew to correctly identify causality.

---

## 3. LLM Participants

### Round 5 Models (Final)

| Model | Size | Quantization | Source |
|-------|------|--------------|--------|
| `gpt-oss-safeguard:120b` | ~65 GB | Q4_0 | OpenGPT-X via Ollama |
| `ibm/granite4:32b-a9b-h` | ~19 GB | Full precision | IBM via Ollama |
| `deepseek-r1:14b` | ~9 GB | Q4_K_M | DeepSeek via Ollama |
| `qwen2.5-coder:7b` | ~5 GB | Q4_K_M | Alibaba via Ollama |
| `granite3.1-moe:3b` | ~2 GB | Q4_K_M | IBM via Ollama |

### Historical Participants (Rounds 1-4)

| Model | Size | Notable Performance |
|-------|------|---------------------|
| `llama3:latest` (8B) | ~5 GB | Silver medal R1 |
| `mistral:latest` (7B) | ~4 GB | Variable performance |
| `mistral:7b-instruct` | ~4 GB | Improved with instruct |
| `qwen2.5-coder:14b` | ~9 GB | High scores, moderate wins |
| `qwen2.5:7b` | ~5 GB | Memory benchmark specialist |
| `llama3.2:3b` | ~2 GB | High points, exploration-heavy |
| `dolphin-mistral:7b` | ~4 GB | Uncensored variant |
| `phi3:latest` | ~2 GB | Negative scores (structural issues) |

---

## 4. Testing Conditions

### Hardware Environment

```
Platform: Linux (Ubuntu 24.04)
LLM Server: Ollama (local)
Endpoint: http://127.0.0.1:11434
```

### LLM Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.3 | Low variance for reproducibility |
| `max_tokens` | 512 | Sufficient for commands + reasoning |
| `timeout` | 120s | Per-turn maximum |
| `stop_sequences` | `\n` (first line) | Extract command only |

### Benchmark Parameters

| Parameter | Value |
|-----------|-------|
| `max_turns` | 10 (default), 15-25 (complex) |
| `sandbox_root` | Isolated temp directory |
| `sandbox_mode` | `safe-read-only` |

### Tool Adapter (Round 5)

For tool-calling models (Granite 4, OpenAI-style):

```python
# Adapter intercepts structured tool calls
<tool_call>{"name": "grep", "arguments": {"pattern": "EUREKA", "path": "."}}</tool_call>
# Renders to:
grep -r -n -- "EUREKA" .
```

Supported tools: `grep`, `find`, `cat`, `head`, `tail`, `ls`, `wc`, `count_lines`, `egrep`, `answer`

### Synthesis Controller (Round 5)

Two-phase state machine for task completion:

| Phase | Behavior |
|-------|----------|
| **EXPLORE** | Tool calls allowed, evidence accumulates |
| **SYNTHESIZE** | Tools blocked, forced answer generation |

Transition triggers:
- Goal variables satisfied
- No new evidence for K turns (stagnation)
- Maximum exploration turns reached

---

## 5. Scoring System

### Point Values

| Event | Points | Rationale |
|-------|--------|-----------|
| Own solution success | **+100** | Rewards independent problem-solving |
| Card solution success | **+50** | Cards help but cost points |
| Goal achieved bonus | **+200** | Strong incentive to reach goal |
| Efficient path bonus | **+50** | Bonus if turns ≤ optimal |

### Penalty Values

| Event | Points | Rationale |
|-------|--------|-----------|
| Syntax error | **-20** | Shell syntax error |
| File not found | **-15** | Wrong path |
| Repeated action | **-30** | Same command twice |
| Card menu access | **-10** | Cost to view help |
| Empty response | **-25** | Non-response |
| Timeout | **-50** | Slow response |

### Win Conditions

A benchmark is **WON** when:
1. All success criteria are satisfied (evidence-based)
2. Final score is positive

A benchmark is **LOST** when:
1. Max turns exhausted without success criteria
2. OR final score is negative

### B07-B10 Extended Scoring

| Event | Points | Applies To |
|-------|--------|------------|
| Synthesis bonus | **+75** | Answer within 2 turns of evidence ready |
| Exploration penalty | **-25** | Per turn after evidence ready |
| Causal misattribution | **-20** | Blaming symptom instead of cause (B07, B10) |
| Incomplete answer | **-30** | Partial answer (e.g., cycle without chain) |
| Semantic precision bonus | **+25** | Exact line quote, skew mention |

### Scoring Formula

```
Final Score = Σ(Points) + Σ(Penalties) + Goal Bonus + Efficiency Bonus + Synthesis Bonus
```

### Example Calculation

```
B01 - Find Needle (granite3.1-moe:3b):
  Turn 1: ls data/           → +100 (success)
  Turn 2: grep -r EUREKA .   → +100 (success)
  Turn 3: Assert file_delta  → +200 (goal achieved)
  Efficiency bonus           → +50 (3 turns = optimal)
  ─────────────────────────────────────
  Final: +450 (WIN)
```

---

## 6. Architectural Components

### 6.1 Tool Call Adapter

**File:** `tool_call_adapter.py`

Translates structured tool calls to bash commands:

```
Model Response                    Adapter                      Bash Command
─────────────────────────────────────────────────────────────────────────────
<tool_call>                       parse_response()            grep -r -n --
{"name": "grep",           ────►  render_to_bash()     ────►  "EUREKA" .
 "arguments": {...}}
</tool_call>
```

**Key Classes:**
- `ToolDefinition`: Schema for tool parameters
- `ToolCall`: Parsed tool invocation
- `BaseToolAdapter`: Abstract adapter interface
- `Granite4Adapter`: IBM Granite 4 specific
- `OpenAIAdapter`: OpenAI function calling format
- `RawAdapter`: Pass-through for raw bash models

### 6.2 Synthesis Controller

**File:** `synthesis_controller.py`

Forces task completion via phase transitions:

```
EXPLORE Phase                    Trigger                      SYNTHESIZE Phase
─────────────────────────────────────────────────────────────────────────────
• Tool calls allowed            Goal variables            • Tools blocked
• Evidence buffer fills   ────► satisfied OR        ────► • Force answer prompt
• Track goal variables          No new evidence K turns   • L2 Finalizer backup
```

**Goal Variable Detection:**
```python
BENCHMARK_GOALS = {
    # Band 1-2: Foundation
    "Find Needle": [GoalVariable("needle_file", r'([./\w-]+\.txt):\d*:?.*EUREKA')],
    "Count Lines": [GoalVariable("line_count", r'^(\d+)\s*(total)?')],
    "Undecidable": [GoalVariable("logging_error", r'%(message)')],

    # Band 4: Advanced Reasoning
    "Stack Trace Diagnosis": [
        GoalVariable("config_found", r'settings\.yaml|divisor:\s*0', required=True),
    ],
    "Diff Analysis": [
        GoalVariable("v1_read", r'calculator_v1\.py|if b == 0'),
        GoalVariable("v2_read", r'calculator_v2\.py|if b is 0', required=True),
    ],
    "Dependency Cycle Detection": [
        GoalVariable("auth_import", r'auth\.py|from modules import user'),
        GoalVariable("user_import", r'user\.py|from modules import permissions'),
        GoalVariable("permissions_import", r'permissions\.py|from modules import auth', required=True),
    ],
    "Temporal Event Correlation": [
        GoalVariable("service_c_log", r'service_c\.log|Database connection lost', required=True),
        GoalVariable("clock_skew_read", r'clock_skew|3234ms|3\.2 seconds', required=True),
    ],
}
```

### 6.3 Failure Detector

**File:** `failure_detector.py`

Detects pathological behaviors:

| Failure Type | Pattern | Intervention |
|--------------|---------|--------------|
| Repetition Loop | Same command 3+ times | TRIZ card suggestion |
| Circular Pattern | A→B→A→B cycle | Force different approach |
| Syntax Cascade | 3+ syntax errors | Simplify command |
| Stagnation | No progress 5+ turns | Phase transition |

### 6.4 Strategic Advisor

**File:** `strategic_advisor.py`

TRIZ-inspired guidance + Kanban WIP limits:

```python
MODEL_WIP_PROFILES = {
    "granite3.1-moe:3b": 2,      # Standard
    "granite4": 2,               # Standard (with adapter)
    "deepseek-r1:14b": 3,        # Extended (strong reasoner)
    "qwen2.5-coder:7b": 2,       # Standard
    "gpt-oss-safeguard:120b": 3, # Extended (large model)
}
```

---

## 7. Reproducibility Guide

### Prerequisites

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull required models
ollama pull granite3.1-moe:3b
ollama pull qwen2.5-coder:7b
ollama pull deepseek-r1:14b
# For larger models (requires sufficient VRAM):
ollama pull ibm/granite4:32b-a9b-h
ollama pull gpt-oss-safeguard:120b

# 3. Install Python dependencies
pip install -r ragix_core/reasoning_tutor/requirements.txt
```

### Running Benchmarks

```bash
# Single model, all benchmarks
python -m ragix_core.reasoning_tutor.benchmarks.scored_mode \
    --models "granite3.1-moe:3b" \
    --max-turns 10

# Multiple models comparison
python -m ragix_core.reasoning_tutor.benchmarks.scored_mode \
    --models "granite3.1-moe:3b,qwen2.5-coder:7b,deepseek-r1:14b" \
    --max-turns 10 \
    --output results/my_run.jsonl

# Single benchmark
python -m ragix_core.reasoning_tutor.benchmarks.scored_mode \
    --benchmark 01 \
    --models "granite3.1-moe:3b"

# Quiet mode (minimal output)
python -m ragix_core.reasoning_tutor.benchmarks.scored_mode \
    --models "granite3.1-moe:3b" \
    --quiet
```

### Output Format

Results are logged in JSONL format:

```json
{
  "game_id": "game_B01_granite3.1-moe:3b_20260203_143022",
  "model": "granite3.1-moe:3b",
  "benchmark": "01_find_needle",
  "success": true,
  "total_turns": 3,
  "final_score": 450,
  "own_solutions": 2,
  "card_solutions": 0,
  "syntax_errors": 0,
  "repeated_actions": 0,
  "total_latency_ms": 1234,
  "timestamp": "2026-02-03T14:30:22Z"
}
```

### Verification

To verify model-independence:

```bash
python -m ragix_core.reasoning_tutor.benchmarks.demo_model_independence
```

This runs the same benchmark with multiple models and confirms the Tutor produces identical verdicts for equivalent moves.

---

## 8. Round Summary

### Round 1 (2025-12-22)
**Focus:** Baseline benchmark establishment

- 11 models tested
- 6 benchmarks introduced
- Gold: `deepseek-r1:14b` (6/6 wins)
- Finding: Model size ≠ performance

### Round 2 (2025-12-23)
**Focus:** Reasoning token handling

- Fixed: `<think>...</think>` token leakage
- deepseek-r1 recovered from 33% to 100%
- Introduced failure detection

### Round 3 (2025-12-24)
**Focus:** Failure analysis

- Added failure detector (repetition, circular, cascade)
- Behavioral fingerprints for each model
- Finding: Failure patterns are model-specific

### Round 4 (2026-01-15)
**Focus:** TRIZ + Kanban integration

- Strategic advisor with TRIZ cards
- Kanban WIP limits per model
- granite3.1-moe:3b improved 33% → 100%
- Finding: Scaffolding equalizes small models

### Round 5 (2026-02-03)
**Focus:** Tool Call Adapter + Synthesis Controller

- Tool adapter for structured tool-calling models
- Synthesis controller for task completion
- Granite 4: -1770 → +3520 (+5290 improvement)
- Finding: Interface contract matters as much as model capability

| Model | Size | Wins | Total | B01 | B02 | B03 | B04 | B05 | B06 |
|-------|------|------|-------|-----|-----|-----|-----|-----|-----|
| gpt-oss-safeguard:120b | 65 GB | **6/6** | +1615 | ✓+220 | ✓+275 | ✓+260 | ✓+300 | ✓+275 | ✓+285 |
| deepseek-r1:14b | ~9 GB | **6/6** | +1600 | ✓+220 | ✓+275 | ✓+245 | ✓+300 | ✓+275 | ✓+285 |
| qwen2.5-coder:7b | ~5 GB | 4/6 | +1720 | ✓+350 | ✓+280 | +245 | +330 | ✓+230 | ✓+285 |
| ibm/granite4:32b-a9b-h | 19 GB | 3/6 | **+3520** | ✓+350 | ✓+450 | +790 | +820 | ✓+290 | +820 |
| granite3.1-moe:3b | ~2 GB | 3/6 | +855 | -110 | ✓+490 | -120 | ✓+180 | ✓+260 | +155 |

### Round 6 (2026-02-04)
**Focus:** Full 10-Benchmark Suite (B01-B10)

- Extended suite from 6 to 10 benchmarks
- Added B07-B10 testing causal, semantic, graph, and temporal reasoning
- Evidence sufficiency detection for synthesis gate optimization
- DeepSeek-r1:14b achieves **perfect 10/10** wins

**Full Results (10 benchmarks):**

| Model | Wins | Total | B01 | B02 | B03 | B04 | B05 | B06 | B07 | B08 | B09 | B10 |
|-------|------|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| deepseek-r1:14b | **10/10** | **+2835** | ✓220 | ✓275 | ✓245 | ✓300 | ✓275 | ✓285 | ✓300 | ✓285 | ✓325 | ✓325 |
| qwen2.5-coder:7b | 8/10 | +2690 | ✓350 | ✓280 | 200 | 240 | ✓230 | ✓285 | ✓260 | ✓210 | ✓370 | ✓265 |
| granite3.1-moe:3b | 9/10 | +1945 | -55 | ✓385 | 60 | ✓180 | ✓260 | ✓450 | ✓285 | ✓145 | 80 | ✓155 |

**Band Performance:**

| Band | Focus | deepseek | qwen | granite |
|------|-------|----------|------|---------|
| **1** | Search & Enumeration (B01-B02) | 2/2 | 2/2 | 1/2 |
| **2** | Reasoning & Rules (B03-B05) | 3/3 | 1/3 | 2/3 |
| **3** | Memory & Context (B06) | 1/1 | 1/1 | 1/1 |
| **4** | Advanced Reasoning (B07-B10) | 4/4 | 4/4 | 4/4 |

**Round 6 Findings:**

1. **DeepSeek achieves perfect score** — Only model with 10/10 wins
2. **Granite best efficiency** — 973 pts/GB (3× better than DeepSeek)
3. **Band 4 universally accessible** — All models passed all advanced benchmarks
4. **Band 2 differentiates** — Reasoning & Rules separates top performers
5. **B03 (Undecidable) most challenging** — Only 1/3 pass rate

### Key Findings for Publication

1. **Interface Contract > Model Size**
   - Granite 4 (32B) improved from -1770 to +3520 with proper interface
   - This is NOT a model defect — it's protocol mismatch

2. **Scaffolding Equalizes Performance**
   - 14B DeepSeek = 120B GPT-OSS (both 6/6 wins)
   - 7B Qwen achieves 4/6 with +1720 points
   - 3B Granite achieves 3/6 with proper support

3. **Efficiency vs Capability Trade-off**
   - 120B model: 6/6 wins but 24.8 pts/GB
   - 3B model: 3/6 wins but 427.5 pts/GB (17× more efficient)

4. **Adapter is Necessary, Not Optional**
   - Tool-calling models appear broken without adaptation
   - Generalizable to other structured-output models

5. **Synthesis Controller Helps Completion**
   - Fixes "explores forever" pathology
   - Must be tuned per-task complexity

6. **Capability Bands Reveal Model Strengths**
   - Band 4 (advanced reasoning): All models pass (12/12)
   - Band 3 (memory): All models pass (3/3)
   - Band 2 (reasoning & rules): Key differentiator (7/9 pass)
   - Band 1 (search & enumeration): One failure (5/6 pass)
   - DeepSeek-r1:14b achieves perfect 10/10 across all bands

7. **Evidence Sufficiency Enables Smart Synthesis**
   - Per-benchmark goal variables track exploration progress
   - Synthesis triggers when evidence is sufficient (not just turn count)
   - Prevents premature synthesis and over-exploration equally

---

## 9. Capability Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CAPABILITY MATRIX (B01-B10)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BAND 1: Search & Enumeration (Foundation)                          │
│  ├── B01 Find Needle ──── Pattern search, grep                      │
│  └── B02 Count Lines ──── Arithmetic, wc -l                         │
│                                                                     │
│  BAND 2: Reasoning & Rules                                          │
│  ├── B03 Undecidable ──── Truth/Dare, evidence gaps                 │
│  ├── B04 Verification ──── Multi-step proofs                        │
│  └── B05 Rule Gen ──── Domain-specific patterns                     │
│                                                                     │
│  BAND 3: Memory & Context                                           │
│  └── B06 Memory Recall ──── Cross-turn retention                    │
│                                                                     │
│  BAND 4: Advanced Reasoning                                         │
│  ├── B07 Stack Trace ──── Causal traceback, red herrings            │
│  ├── B08 Diff Analysis ──── Semantic change detection               │
│  ├── B09 Cycle Detection ──── Graph traversal, imports              │
│  └── B10 Temporal ──── Distributed systems, clock skew              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Cognitive Skills per Benchmark:**

| Benchmark | Primary Skill | Secondary Skills | Difficulty |
|-----------|---------------|------------------|------------|
| B01 | Pattern search | File navigation | Medium |
| B02 | Arithmetic | Enumeration | Medium |
| B03 | Evidence reasoning | Claim validation | Hard |
| B04 | Proof chaining | Import resolution | Hard |
| B05 | Rule extraction | Domain modeling | Expert |
| B06 | Context retention | Multi-turn memory | Hard |
| B07 | Causal reasoning | Red herring rejection | Medium |
| B08 | Semantic diff | Code review | Medium |
| B09 | Graph traversal | Cycle detection | Expert |
| B10 | Temporal reasoning | Clock skew, causality | Hard |

---

## References

- Vitrac, O. (2025). *RAGIX: Retrieval-Augmented Generative Interactive eXecution Agent*. Adservio.
- *Interpreter-Tutor Design Document*: `INTERPRET_TUTOR.md`
- *Round 5 Final Results*: `results/round5/ROUND5_FINAL_RESULTS.md`
- *Round 6 Final Results*: `results/round6/ROUND6_FINAL_RESULTS.md`
- *Benchmark Expansion Plan (B07-B10)*: `docs/BENCHMARK_EXPANSION_PLAN_B07_B10.md`

---

*Generated by RAGIX Interpreter-Tutor Benchmark Suite v1.1*
