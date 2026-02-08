# Benchmark Expansion Plan: B07–B10

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 1.0 (2026-02-04)
**Status:** Specification Draft — Pending Implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Philosophy](#2-design-philosophy)
3. [Benchmark Specifications](#3-benchmark-specifications)
   - [B07: Stack Trace Diagnosis](#b07-stack-trace-diagnosis)
   - [B08: Diff Analysis](#b08-diff-analysis)
   - [B09: Dependency Cycle Detection](#b09-dependency-cycle-detection)
   - [B10: Temporal Event Correlation](#b10-temporal-event-correlation)
4. [Synthesis Gate Mechanism](#4-synthesis-gate-mechanism)
5. [Scoring System Extensions](#5-scoring-system-extensions)
6. [Difficulty Calibration](#6-difficulty-calibration)
7. [Updated Capability Matrix](#7-updated-capability-matrix)
8. [Implementation Checklist](#8-implementation-checklist)

---

## 1. Executive Summary

This document specifies four additional benchmarks (B07–B10) that extend the LLM Reasoning Olympics suite along **orthogonal cognitive axes**:

| Benchmark | Cognitive Axis | Real-World Task |
|-----------|----------------|-----------------|
| B07 | Causal traceback | Debugging from stack traces |
| B08 | Semantic diff | Code review |
| B09 | Graph cycles | Architecture analysis |
| B10 | Temporal causality | Distributed systems ops |

**Design Goals:**
- Unambiguous: single correct target per benchmark
- Robust: resistant to superficial strategies
- Scoreable: compatible with existing scored_mode.py
- Synthesis-aware: explicitly test termination (Granite 4 deficit)

**Key Innovation:** A **Synthesis Gate** mechanism that detects evidence sufficiency and enforces answer generation, directly targeting the "exploration without synthesis" failure mode observed in Round 5.

---

## 2. Design Philosophy

### 2.1 Benchmark Template Structure

Each benchmark follows a standardized structure:

```
benchmarks/b07_stack_trace/
├── assets/                    # File tree for sandbox setup
│   ├── logs/crash.log
│   ├── main.py
│   ├── processor.py
│   ├── handler.py
│   └── config.yaml
├── 07_stack_trace.yaml        # Benchmark definition
└── README_TASK.txt            # Human-readable description (optional)
```

### 2.2 Specification Components

| Component | Description |
|-----------|-------------|
| **Assets** | File tree created in sandbox |
| **Prompt** | Minimal story + explicit goal + output contract |
| **Allowed Tools** | bash + read_file; optionally "answer" tool |
| **Success Criteria** | Machine-checkable regex or exact match |
| **Evidence Sufficiency** | Deterministic detector for synthesis gate |
| **Scoring Hooks** | Points, penalties, synthesis bonus |

### 2.3 Filesystem-Centric Design

All benchmarks remain compatible with the Unix-RAG pattern:
- Primary tools: `grep`, `find`, `cat`, `head`, `tail`, `wc`, `diff`
- No external APIs or databases
- Self-contained sandbox environments
- Deterministic verification

### 2.4 Synthesis Bonus (Global Scoring Knob)

> **Synthesis Bonus:** If the model outputs `answer` within ≤2 turns after evidence sufficiency is reached, award **+75 points**. If it continues exploring beyond the "ready" state, apply **-25 per extra explore-turn**.

This directly targets "exploration without synthesis" while remaining model-agnostic.

---

## 3. Benchmark Specifications

---

### B07: Stack Trace Diagnosis

| Property | Value |
|----------|-------|
| **Category** | Error Analysis (Causal Traceback) |
| **Difficulty** | Hard |
| **Optimal Turns** | 5 |
| **Primary Tools** | `cat`, `grep` |
| **Cognitive Axis** | Backward causal reasoning |

#### 3.1 Assets

```
b07_stack_trace/
├── logs/
│   └── crash.log
├── main.py
├── processor.py
├── handler.py
├── config.yaml
└── README_TASK.txt
```

#### 3.2 File Contents

**logs/crash.log:**
```
Application crash report - 2026-02-04T10:15:32

Traceback (most recent call last):
  File "main.py", line 45, in run
    result = processor.execute(data)
  File "processor.py", line 23, in execute
    return self.handler.transform(item)
  File "handler.py", line 67, in transform
    return item["value"] / self.config["divisor"]
ZeroDivisionError: division by zero

Process terminated with exit code 1
```

**main.py:**
```python
# main.py - Application entry point
from processor import Processor

def run():
    data = {"value": 100, "name": "test_item"}
    proc = Processor()
    result = proc.execute(data)  # line 45
    return result

if __name__ == "__main__":
    run()
```

**processor.py:**
```python
# processor.py - Data processing module
from handler import Handler

class Processor:
    def __init__(self):
        self.handler = Handler()

    def execute(self, item):
        # Process the item through handler
        return self.handler.transform(item)  # line 23
```

**handler.py:**
```python
# handler.py - Item transformation handler
import yaml

class Handler:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def transform(self, item):
        # Apply transformation based on config
        # NOTE: divisor comes from config, not hardcoded
        return item["value"] / self.config["divisor"]  # line 67
```

**config.yaml:**
```yaml
# Application configuration
mode: "production"
divisor: 0
max_retries: 3
timeout_seconds: 30
```

**Red Herring (in handler.py comments):**
```python
# handler.py - Item transformation handler
# Default divisor was 1, changed to use config
# Old code: return item["value"] / 1  # divisor=1 hardcoded
```

#### 3.3 Prompt

```
You have access to a crashed application's repository.
The crash log is in logs/crash.log.

GOAL: Find the root cause configuration that sets divisor to zero.

OUTPUT FORMAT: Report the exact file and line, e.g.:
  config.yaml: divisor: 0
```

#### 3.4 Success Criteria

**Regex match:**
```regex
^config\.yaml:\s*divisor:\s*0\s*$
```

**Required elements:**
- Must mention `config.yaml`
- Must identify `divisor: 0` as the root cause
- Must NOT cite `handler.py:67` as the answer (that's the symptom)

#### 3.5 Evidence Sufficiency Detector

```python
def is_evidence_sufficient_b07(evidence_buffer: List[str]) -> bool:
    """B07: Ready when config.yaml content with divisor: 0 is observed."""
    for evidence in evidence_buffer:
        if "config.yaml" in evidence and re.search(r'divisor:\s*0', evidence):
            return True
    return False
```

#### 3.6 Failure Modes Caught

| Failure Mode | Symptom | Penalty |
|--------------|---------|---------|
| Restates stack trace | No config reference | -30 |
| Blames handler.py:67 | Symptom not cause | -20 |
| Finds config but no answer | Exploration loops | Synthesis penalty |
| Cites red herring comment | Wrong divisor source | -15 |

#### 3.7 Scoring

| Event | Points |
|-------|--------|
| Correct answer (config.yaml: divisor: 0) | +200 (goal) |
| Each successful file read | +50 |
| Synthesis within 2 turns of evidence | +75 (bonus) |
| Answering handler.py as root cause | -20 |
| Extra exploration after evidence ready | -25/turn |

---

### B08: Diff Analysis

| Property | Value |
|----------|-------|
| **Category** | Comparison (Semantic Change Detection) |
| **Difficulty** | Medium |
| **Optimal Turns** | 4 |
| **Primary Tools** | `diff`, `cat` |
| **Cognitive Axis** | Semantic diff reasoning |

#### 3.8 Design Decision: Semantic Bug vs Syntax Error

**Option A (Recommended): Semantic Bug — Valid Python, Wrong Logic**

Tests semantic understanding without syntax parsing dominating.

**Option B: Syntax Error — Invalid Token**

Tests code review + syntax awareness but may be "trivially" detectable.

**Selected: Option A** — More aligned with real code review tasks.

#### 3.9 Assets

```
b08_diff_analysis/
├── src/
│   ├── calculator_v1.py
│   └── calculator_v2.py
├── tests/
│   └── test_calculator.py
└── CHANGELOG.txt
```

#### 3.10 File Contents

**src/calculator_v1.py (working version):**
```python
# calculator_v1.py - Mathematical operations (v1.0)

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

**src/calculator_v2.py (buggy version):**
```python
# calculator_v2.py - Mathematical operations (v2.0)
# Updated: performance improvements

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b is 0:  # BUG: identity vs equality
        raise ValueError("Cannot divide by zero")
    return a / b
```

**tests/test_calculator.py:**
```python
# test_calculator.py
import unittest
from src.calculator_v2 import divide

class TestCalculator(unittest.TestCase):
    def test_divide_normal(self):
        self.assertEqual(divide(10, 2), 5)

    def test_divide_by_zero(self):
        # This test FAILS with v2 for certain inputs
        with self.assertRaises(ValueError):
            divide(10, 0)  # Works
        with self.assertRaises(ValueError):
            divide(10, 0.0)  # FAILS! 0.0 is not 0

if __name__ == "__main__":
    unittest.main()
```

**CHANGELOG.txt:**
```
Version 2.0 (2026-02-01)
- Performance improvements in divide function
- Minor code cleanup

Version 1.0 (2026-01-15)
- Initial release
```

#### 3.11 Prompt

```
A bug was introduced between calculator_v1.py and calculator_v2.py.
The divide function now fails for some zero inputs.

GOAL: Identify the exact breaking change (one line).

OUTPUT FORMAT: Report the file and describe the change:
  calculator_v2.py: uses "is" instead of "==" in zero check
```

#### 3.12 Success Criteria

**Regex match (any of):**
```regex
calculator_v2\.py.*\bis\b.*\b(0|zero)\b
calculator_v2\.py.*identity.*equality
calculator_v2\.py.*==.*is
```

**Required elements:**
- Must mention `calculator_v2.py`
- Must identify `is 0` vs `== 0` issue
- Must reference the `if` condition

#### 3.13 Evidence Sufficiency Detector

```python
def is_evidence_sufficient_b08(evidence_buffer: List[str]) -> bool:
    """B08: Ready when both versions are read and 'is 0' pattern detected."""
    has_v1 = any("== 0" in e and "v1" in e for e in evidence_buffer)
    has_v2 = any("is 0" in e and "v2" in e for e in evidence_buffer)
    return has_v1 and has_v2
```

#### 3.14 Scoring

| Event | Points |
|-------|--------|
| Correct identification of `is` vs `==` | +200 (goal) |
| Using `diff` command effectively | +50 |
| Quoting the exact line | +25 (bonus) |
| Generic "there is a bug" without specifying | -30 |
| Synthesis within 2 turns of evidence | +75 (bonus) |

---

### B09: Dependency Cycle Detection

| Property | Value |
|----------|-------|
| **Category** | Graph Reasoning |
| **Difficulty** | Expert |
| **Optimal Turns** | 6 |
| **Primary Tools** | `cat`, `grep` |
| **Cognitive Axis** | Graph traversal, cycle detection |

#### 3.15 Assets

```
b09_cycle_detection/
├── modules/
│   ├── __init__.py
│   ├── auth.py
│   ├── user.py
│   └── permissions.py
├── main.py
└── logs/
    └── import_error.log
```

#### 3.16 File Contents

**modules/auth.py:**
```python
# modules/auth.py - Authentication module
from modules import user

def authenticate(username, password):
    """Authenticate user credentials."""
    usr = user.get_user(username)
    if usr and usr["password"] == password:
        return {"status": "authenticated", "user": usr}
    return {"status": "failed"}

def get_session():
    return {"active": True}
```

**modules/user.py:**
```python
# modules/user.py - User management
from modules import permissions

def get_user(username):
    """Retrieve user with permissions."""
    perms = permissions.get_permissions(username)
    return {
        "username": username,
        "permissions": perms,
        "password": "hashed_value"
    }

def list_users():
    return ["admin", "guest"]
```

**modules/permissions.py:**
```python
# modules/permissions.py - Permission system
from modules import auth  # CREATES CYCLE: auth -> user -> permissions -> auth

def get_permissions(username):
    """Get user permissions, requires active session."""
    session = auth.get_session()
    if session["active"]:
        return ["read", "write"]
    return ["read"]

def check_permission(user, action):
    return action in get_permissions(user)
```

**modules/__init__.py:**
```python
# modules/__init__.py - Package initialization
```

**main.py:**
```python
# main.py - Application entry point
from modules import auth

def main():
    result = auth.authenticate("admin", "secret")
    print(result)

if __name__ == "__main__":
    main()
```

**logs/import_error.log:**
```
$ python main.py
Traceback (most recent call last):
  File "main.py", line 2, in <module>
    from modules import auth
  File "/app/modules/auth.py", line 2, in <module>
    from modules import user
  File "/app/modules/user.py", line 2, in <module>
    from modules import permissions
  File "/app/modules/permissions.py", line 2, in <module>
    from modules import auth
ImportError: cannot import name 'auth' from partially initialized module 'modules'
(most likely due to a circular import)
```

#### 3.17 Prompt

```
The application fails to start due to a circular import error.
See logs/import_error.log for the traceback.

GOAL: Identify the circular dependency chain in order.

OUTPUT FORMAT: Report the cycle as:
  auth -> user -> permissions -> auth
```

#### 3.18 Success Criteria

**Exact sequence match (normalized):**
```
auth -> user -> permissions -> auth
```

**Normalization rules:**
- Whitespace flexible: `auth->user` = `auth -> user`
- Arrow flexible: `->` = `→` = `-->`
- Must be complete cycle (4 nodes, ending at start)

#### 3.19 Evidence Sufficiency Detector

```python
def is_evidence_sufficient_b09(evidence_buffer: List[str]) -> bool:
    """B09: Ready when all 3 import statements are observed."""
    imports_found = {
        "auth_imports_user": False,
        "user_imports_permissions": False,
        "permissions_imports_auth": False
    }
    for evidence in evidence_buffer:
        if "auth.py" in evidence and "from modules import user" in evidence:
            imports_found["auth_imports_user"] = True
        if "user.py" in evidence and "from modules import permissions" in evidence:
            imports_found["user_imports_permissions"] = True
        if "permissions.py" in evidence and "from modules import auth" in evidence:
            imports_found["permissions_imports_auth"] = True
    return all(imports_found.values())
```

#### 3.20 Scoring

| Event | Points |
|-------|--------|
| Correct cycle chain | +200 (goal) |
| Each module file read | +50 |
| Citing import line numbers | +25 (bonus) |
| "Circular import exists" without chain | -30 |
| Incomplete chain (missing node) | -20 |
| Synthesis within 2 turns of evidence | +75 (bonus) |

---

### B10: Temporal Event Correlation

| Property | Value |
|----------|-------|
| **Category** | Temporal Reasoning |
| **Difficulty** | Hard |
| **Optimal Turns** | 6 |
| **Primary Tools** | `cat`, `grep` |
| **Cognitive Axis** | Timestamp parsing, causality, clock skew |

#### 3.21 Assets

```
b10_temporal_correlation/
├── logs/
│   ├── service_a.log
│   ├── service_b.log
│   └── service_c.log
├── notes/
│   └── clock_skew.txt
└── architecture.txt
```

#### 3.22 File Contents

**logs/service_a.log (gateway):**
```
2026-02-04T10:00:01.234 INFO  [gateway] Request received id=req-abc123 from client
2026-02-04T10:00:01.345 INFO  [gateway] Forwarding to service_b
2026-02-04T10:00:01.567 ERROR [gateway] Timeout waiting for service_b (5000ms)
2026-02-04T10:00:01.568 ERROR [gateway] Request req-abc123 failed: upstream timeout
2026-02-04T10:00:02.000 FATAL [gateway] Circuit breaker OPEN for service_b
```

**logs/service_b.log (processor):**
```
2026-02-04T10:00:00.890 INFO  [processor] Received request req-abc123 from gateway
2026-02-04T10:00:00.891 INFO  [processor] Querying service_c for data enrichment
2026-02-04T10:00:05.123 ERROR [processor] service_c connection refused after 4s retry
2026-02-04T10:00:05.124 ERROR [processor] Cannot process req-abc123: dependency failed
```

**logs/service_c.log (database layer):**
```
2026-02-04T09:59:55.000 WARN  [db-layer] Connection pool exhausted, waiting...
2026-02-04T09:59:57.500 ERROR [db-layer] Database primary node unreachable
2026-02-04T09:59:58.000 FATAL [db-layer] Database connection lost - cannot recover
2026-02-04T09:59:58.001 INFO  [db-layer] Initiating graceful shutdown
2026-02-04T09:59:58.500 INFO  [db-layer] Service stopped
```

**notes/clock_skew.txt:**
```
NTP Synchronization Report - 2026-02-04

Observed clock offsets (relative to service_a):
- service_a: reference (0ms)
- service_b: -344ms (behind)
- service_c: -3234ms (behind, ~3.2 seconds)

Note: service_c runs on legacy hardware with drift issues.
Timestamps in service_c.log should be adjusted +3.2s for correlation.
```

**architecture.txt:**
```
System Architecture
===================

Request flow:
  Client -> service_a (gateway) -> service_b (processor) -> service_c (db-layer)

Dependencies:
- service_a depends on service_b
- service_b depends on service_c
- service_c depends on PostgreSQL primary

SLA: 5000ms end-to-end timeout at gateway
```

**Red Herring (in service_a.log):**
```
2026-02-04T10:00:02.000 FATAL [gateway] Circuit breaker OPEN for service_b
```
This FATAL is a **consequence**, not the root cause.

#### 3.23 Prompt

```
A distributed system experienced a cascading failure.
Logs from three services are available in logs/.
Clock skew information is in notes/clock_skew.txt.

GOAL: Identify the root cause service and event that triggered the cascade.

OUTPUT FORMAT: Report the root cause as:
  service_c.log: 2026-02-04T09:59:58.000 FATAL Database connection lost
```

#### 3.24 Success Criteria

**Required elements (all must match):**
```regex
service_c
09:59:58
(Database connection lost|FATAL.*db)
```

**Must NOT cite as root cause:**
- service_a circuit breaker (consequence)
- service_b connection refused (symptom)

#### 3.25 Evidence Sufficiency Detector

```python
def is_evidence_sufficient_b10(evidence_buffer: List[str]) -> bool:
    """B10: Ready when service_c FATAL is seen AND clock skew is known."""
    has_root_cause = any(
        "service_c" in e and "FATAL" in e and "Database" in e
        for e in evidence_buffer
    )
    has_skew_info = any(
        "clock_skew" in e or "service_c" in e and "-3" in e
        for e in evidence_buffer
    )
    return has_root_cause and has_skew_info
```

#### 3.26 Scoring

| Event | Points |
|-------|--------|
| Correct root cause (service_c FATAL) | +200 (goal) |
| Each log file read | +50 |
| Reading clock_skew.txt | +30 |
| Mentioning skew reconciliation | +25 (bonus) |
| Citing service_a/b errors as root cause | -30 |
| Synthesis within 2 turns of evidence | +75 (bonus) |

---

## 4. Synthesis Gate Mechanism

### 4.1 Rationale

Round 5 revealed that Granite 4 (and other models) suffer from **"exploration without synthesis"** — they gather sufficient evidence but fail to terminate with a final answer. The Synthesis Gate addresses this directly.

### 4.2 Evidence Sufficiency Framework

Each benchmark defines a deterministic `is_evidence_sufficient()` function:

```python
# synthesis_controller.py extension

EVIDENCE_DETECTORS: Dict[str, Callable[[List[str]], bool]] = {
    "Stack Trace Diagnosis": is_evidence_sufficient_b07,
    "Diff Analysis": is_evidence_sufficient_b08,
    "Dependency Cycle Detection": is_evidence_sufficient_b09,
    "Temporal Event Correlation": is_evidence_sufficient_b10,
}
```

### 4.3 Intervention Protocol

When `is_evidence_sufficient()` returns `True`:

1. **Turn N (evidence ready):** No intervention, allow one more exploration turn
2. **Turn N+1:** Inject system message:
   ```
   SYSTEM: You have gathered sufficient evidence. Output your final answer now.
   ```
3. **Turn N+2:** Block further tool calls, force answer generation
4. **Turn N+3+:** Apply `-25` penalty per additional turn

### 4.4 Implementation Hook

```python
# In scored_mode.py game loop

if synthesis_controller.phase == Phase.EXPLORE:
    if is_evidence_sufficient(benchmark_name, evidence_buffer):
        synthesis_controller.mark_evidence_ready()

if synthesis_controller.turns_since_ready > 1:
    # Inject synthesis prompt
    prompt += "\n\nSYSTEM: You have enough evidence. Output the final answer now."

if synthesis_controller.turns_since_ready > 2:
    # Block tools, force answer
    allowed_tools = ["answer"]
```

### 4.5 Scoring Impact

| Condition | Points |
|-----------|--------|
| Answer within 2 turns of evidence ready | **+75** (synthesis bonus) |
| Each extra exploration turn after ready | **-25** (per turn) |
| Never reaching answer (timeout) | **-100** (existing penalty) |

---

## 5. Scoring System Extensions

### 5.1 New Scoring Constants

```python
# Add to ScoringConfig in scored_mode.py

@dataclass
class ScoringConfig:
    # ... existing fields ...

    # B07-B10 specific
    synthesis_bonus: int = 75           # Answer within 2 turns of evidence ready
    exploration_penalty: int = -25      # Per turn after evidence ready
    causal_misattribution: int = -20    # Blaming symptom instead of cause
    incomplete_answer: int = -30        # Partial answer (e.g., cycle without chain)
    semantic_precision_bonus: int = 25  # Exact line quote, skew mention, etc.
```

### 5.2 Benchmark-Specific Scoring Hooks

```python
# B07 specific
def score_b07(answer: str, evidence: List[str]) -> int:
    score = 0
    if re.match(r'config\.yaml:\s*divisor:\s*0', answer):
        score += 200  # Goal achieved
    elif "handler.py" in answer and "67" in answer:
        score -= 20   # Symptom not cause
    return score

# B08 specific
def score_b08(answer: str, evidence: List[str]) -> int:
    score = 0
    if re.search(r'calculator_v2.*\bis\b.*0', answer, re.IGNORECASE):
        score += 200  # Goal achieved
        if "if b is 0" in answer:
            score += 25  # Exact quote bonus
    elif "bug" in answer.lower() and "calculator_v2" not in answer:
        score -= 30   # Generic without specifics
    return score

# B09 specific
def score_b09(answer: str, evidence: List[str]) -> int:
    score = 0
    chain = normalize_cycle_chain(answer)
    if chain == ["auth", "user", "permissions", "auth"]:
        score += 200  # Goal achieved
    elif "circular" in answer.lower() and "->" not in answer:
        score -= 30   # Exists without chain
    return score

# B10 specific
def score_b10(answer: str, evidence: List[str]) -> int:
    score = 0
    if "service_c" in answer and "09:59:58" in answer:
        score += 200  # Goal achieved
        if "skew" in answer.lower() or "clock" in answer.lower():
            score += 25  # Skew awareness bonus
    elif "service_a" in answer or "circuit breaker" in answer.lower():
        score -= 30   # Wrong service (consequence)
    return score
```

---

## 6. Difficulty Calibration

### 6.1 Difficulty Knobs

| Benchmark | Difficulty | Calibration Mechanism |
|-----------|------------|----------------------|
| B07 | Hard | Red herring divisor in code comments |
| B08 | Medium | Single-file, single-line change |
| B09 | Expert | 3-module cycle (can extend to 4+) |
| B10 | Hard | Multiple FATAL events, requires skew reasoning |

### 6.2 B07 Hardening (Red Herrings)

Add misleading content:

```python
# handler.py comments (red herring)
# Old implementation: divisor = 1  (hardcoded)
# TODO: Consider divisor = 2 for double precision
```

```yaml
# config.yaml alternate section (not used)
legacy:
  old_divisor: 1
```

### 6.3 B09 Hardening (Extended Cycle)

For increased difficulty, add a 4th module:

```
auth -> user -> permissions -> roles -> auth
```

But keep the **shortest cycle** as the primary target.

### 6.4 B10 Hardening (Multiple FATALs)

Add unrelated FATAL to service_a:

```
2026-02-04T09:58:00.000 FATAL [gateway] Config validation failed: missing TLS cert
2026-02-04T09:58:00.001 INFO  [gateway] Using fallback HTTP mode
```

This FATAL is **before** the incident and unrelated — tests causal discrimination.

---

## 7. Updated Capability Matrix

### 7.1 Four-Band Classification

The complete 10-benchmark suite now covers four cognitive bands:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CAPABILITY MATRIX (B01-B10)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BAND 1: Search & Enumeration (Foundation)                          │
│  ├── B01 Find Needle ──── Pattern search, grep                      │
│  └── B02 Count Lines ──── Arithmetic, wc -l                         │
│                                                                     │
│  BAND 2: Formal Reasoning Under Constraints                         │
│  ├── B03 Undecidable ──── Truth/Dare mechanism                      │
│  └── B04 Verification ──── Proof chain building                     │
│                                                                     │
│  BAND 3: Rule & Memory Governance                                   │
│  ├── B05 Session Rules ──── Domain-specific pattern extraction      │
│  └── B06 Memory Recall ──── Multi-turn context retention            │
│                                                                     │
│  BAND 4: Real-World Engineering Diagnosis (NEW)                     │
│  ├── B07 Stack Trace ──── Causal traceback, debugging               │
│  ├── B08 Diff Analysis ──── Semantic change detection, code review  │
│  ├── B09 Cycle Detection ──── Graph reasoning, architecture         │
│  └── B10 Temporal ──── Distributed systems, clock skew              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Cognitive Skills per Benchmark

| Benchmark | Primary Skill | Secondary Skills | Synthesis Requirement |
|-----------|---------------|------------------|----------------------|
| B01 | Pattern search | grep, discovery | Low |
| B02 | Enumeration | arithmetic, aggregation | Low |
| B03 | Evidence gathering | Truth/Dare protocol | Medium |
| B04 | Proof building | multi-file reasoning | Medium |
| B05 | Rule extraction | custom parsing | Medium |
| B06 | Context retention | multi-turn memory | High |
| B07 | Causal reasoning | error correlation | **High** |
| B08 | Semantic comparison | change detection | **Medium** |
| B09 | Graph traversal | cycle detection | **High** |
| B10 | Temporal reasoning | clock skew, causality | **Very High** |

### 7.3 Empirical Story for Publication

The benchmark suite now supports a clean empirical narrative:

1. **Adapter fixes GROUNDING** (tool/action expression)
   - Measured: B01-B02 performance pre/post adapter
   - Finding: Granite 4 improved +5290 points

2. **Synthesis Gate fixes TERMINATION** (declare final answer)
   - Measured: B07-B10 performance with/without gate
   - Finding: (To be measured in Round 6)

3. **Remaining deltas = genuine REASONING CAPACITY**
   - Band 1-2: Basic capabilities (most models pass)
   - Band 3: Memory/rules (differentiates mid-tier)
   - Band 4: Engineering diagnosis (differentiates top-tier)

---

## 8. Implementation Checklist

### 8.1 File Creation

- [ ] `benchmarks/b07_stack_trace/` directory + assets
- [ ] `benchmarks/b08_diff_analysis/` directory + assets
- [ ] `benchmarks/b09_cycle_detection/` directory + assets
- [ ] `benchmarks/b10_temporal_correlation/` directory + assets
- [ ] `benchmarks/07_stack_trace.yaml` benchmark definition
- [ ] `benchmarks/08_diff_analysis.yaml` benchmark definition
- [ ] `benchmarks/09_cycle_detection.yaml` benchmark definition
- [ ] `benchmarks/10_temporal_correlation.yaml` benchmark definition

### 8.2 Code Modifications

- [ ] `synthesis_controller.py`: Add evidence sufficiency detectors for B07-B10
- [ ] `synthesis_controller.py`: Add `BENCHMARK_GOALS` entries for B07-B10
- [ ] `scored_mode.py`: Add benchmark-specific scoring hooks
- [ ] `scored_mode.py`: Add synthesis bonus/penalty constants
- [ ] `scored_mode.py`: Integrate evidence-ready detection in game loop

### 8.3 Documentation Updates

- [ ] `README.md`: Update benchmark list (6 → 10)
- [ ] `README.md`: Update roadmap (v0.5.0 → v0.6.0)
- [ ] `docs/LLM_OLYMPICS_TECHNICAL_APPENDIX.md`: Add B07-B10 specifications

### 8.4 Validation

- [ ] Run B07-B10 with `granite3.1-moe:3b` (baseline)
- [ ] Run B07-B10 with `deepseek-r1:14b` (expected best)
- [ ] Run B07-B10 with `granite4` (test synthesis gate)
- [ ] Verify scoring produces expected penalties/bonuses
- [ ] Verify evidence sufficiency detectors trigger correctly

### 8.5 Round 6 Planning

- [ ] Full 10-benchmark run with all Round 5 models
- [ ] Compare synthesis gate effectiveness
- [ ] Document findings for publication

---

## References

- Vitrac, O. (2026). *LLM Reasoning Olympics — Technical Appendix*. `docs/LLM_OLYMPICS_TECHNICAL_APPENDIX.md`
- Round 5 Results: `results/round5/ROUND5_FINAL_RESULTS.md`
- Synthesis Controller: `synthesis_controller.py`
- Tool Call Adapter: `tool_call_adapter.py`

---

*Specification authored for RAGIX Interpreter-Tutor Benchmark Suite*
