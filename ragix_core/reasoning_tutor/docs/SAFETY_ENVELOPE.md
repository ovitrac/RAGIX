# Safety Envelope Architecture

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 0.1.0 (2025-12-27)
**Status:** Specification (implementation pending)

---

## Overview

The Safety Envelope is an optional wrapper architecture that extends the existing `Tutor` with additional safety layers while preserving **strict backward compatibility**. It follows the **"Law vs. Lawyer"** model:

- **Law (Tutor):** Deterministic verification, CHECK protocol, scoring — immutable baseline
- **Lawyer (TutorEnvelope):** Optional layers for schema enforcement, repair, and policy optimization

**Critical invariant:** The existing `Tutor` behavior is the **Law**. All previous results must remain reproducible.

---

## The Law vs. Lawyer Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RUNTIME (Law)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      TutorEnvelope (Optional)                        │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│   │  │   Layer 1   │  │   Layer 2   │  │        Layer 3              │  │   │
│   │  │   Schema    │→ │   Repair    │→ │   Policy Bundle Execution   │  │   │
│   │  │ Enforcement │  │  (BestOfN)  │  │   (Frozen prompts/params)   │  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│   └───────────────────────────────────┬─────────────────────────────────┘   │
│                                       │                                     │
│                                       ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         Tutor (Baseline)                            │   │
│   │  - CHECK protocol (CheckVerdict: PROVABLE/REFUTABLE/UNDECIDABLE/    │   │
│   │                                  ILL_TYPED)                         │   │
│   │  - Move validation (MoveVerdict: LEGAL/ILLEGAL/BLOCKED)             │   │
│   │  - Deterministic scoring (SCORE_TRUTH_DIRECT, SCORE_TRUTH_RULE...)  │   │
│   │  - PCG state management                                             │   │
│   │  - ShellExecutor (sandboxed)                                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE (Lawyer) - Optional                         │
├─────────────────────────────────────────────────────────────────────────────┤
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      DSPy Policy Compiler                           │   │
│   │  - Optimizes prompts and exemplars against Tutor-grounded metric    │   │
│   │  - Outputs frozen Policy Bundle (JSON/YAML)                         │   │
│   │  - NEVER runs at runtime                                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Definitions

### Layer 0: Hard Gate (Always Active)

The existing `Tutor` implements the hard safety gate:

| Component | Class | Method | Behavior |
|-----------|-------|--------|----------|
| CHECK Protocol | `Tutor` | `check()` | Returns `CheckVerdict` |
| Move Validation | `Tutor` | `execute_move()` | Returns `MoveVerdict` |
| Command Denylist | `ShellExecutor` | `execute()` | Blocks dangerous commands |
| Scoring | `Tutor` | internal | Deterministic score updates |

**Verdicts (from `tutor.py`):**

```python
class CheckVerdict(Enum):
    PROVABLE = "provable"       # Has valid proof
    REFUTABLE = "refutable"     # Contradicted by evidence
    UNDECIDABLE = "undecidable" # Missing information
    ILL_TYPED = "ill-typed"     # Invalid formulation

class MoveVerdict(Enum):
    LEGAL = "legal"
    ILLEGAL = "illegal"
    BLOCKED = "blocked"         # Blocked by constraint
```

### Layer 1: Interface Safety (Optional)

Schema enforcement to prevent format interference (the "Chatty Parser" problem).

| Feature | Purpose | Default |
|---------|---------|---------|
| Move Schema | Validate JSON structure before processing | OFF |
| Format Normalization | Strip markdown fences, backticks | OFF |
| Type Coercion | Ensure correct field types | OFF |

**When enabled:**
- Parses raw LLM output into structured `Move` dataclass
- Applies format normalization (removes ```` ```bash ```` wrappers)
- Rejects malformed moves before they reach Layer 0

**When disabled:**
- Uses existing `parse_move()` from `moves.py` unchanged

### Layer 2: Inference-Time Repair (Optional)

Model-agnostic retry loop for recovering from CHECK failures.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repair_enabled` | bool | False | Enable retry loop |
| `repair_max_attempts` | int | 0 | Max retries per turn |
| `repair_trigger_verdicts` | tuple | `("ILL_TYPED", "UNDECIDABLE")` | Which verdicts trigger retry |
| `repair_temperature` | float | 0.0 | Temperature for retries |

**Algorithm:**
```
1. Generate candidate move from LLM
2. Run NON-COMMITTING CHECK (no shell, no PCG mutation)
3. If verdict in trigger set AND attempts < max:
   - Generate new candidate
   - Repeat from step 2
4. Select best candidate by deterministic scoring proxy
5. Execute selected move through Layer 0
```

**Critical:** Non-committing CHECK requires `Tutor.check()` to support a `commit=False` mode.

### Layer 3: Policy Bundle Execution (Optional)

Runtime execution of frozen, pre-compiled policy bundles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_enabled` | bool | False | Use policy bundle |
| `policy_bundle_path` | str | None | Path to bundle file |

**Policy Bundle Contents:**
- Frozen prompts (system, action menu, etc.)
- Exemplars (few-shot examples)
- Model parameters (temperature, top_p, etc.)
- Metadata (compiler version, hash, timestamp)

**DSPy Integration:**
- Bundles are compiled OFFLINE by DSPy
- Runtime NEVER imports or calls DSPy
- Bundles are plain JSON/YAML files

---

## Non-Negotiable Invariants

### I0: Baseline Equivalence

When `TutorEnvelope` is instantiated with all layers disabled:

```python
envelope = TutorEnvelope(
    base_tutor=tutor,
    layer1_schema="off",
    layer2_repair="off",
    layer3_policy="off"
)
```

The following must hold:
- Prompts issued to model are **identical** to baseline
- Number of model calls per turn is **identical**
- Parsing logic uses legacy `parse_move()` **unchanged**
- `CheckVerdict`, score deltas, state updates are **identical**
- Output artifacts differ **only** by manifest file addition

### I1: Determinism Boundary

The `Tutor` class remains the single source of truth:

- `CheckVerdict` semantics unchanged
- Scoring rules (`SCORE_TRUTH_DIRECT`, etc.) unchanged
- `FailureDetector` and routing rules unchanged
- `StrategicAdvisor` behavior unchanged

### I2: No Hidden Compute

Layer 2 repair introduces retries **only when explicitly enabled**:

- Budgets are deterministic and recorded in manifest
- Retry count is logged per turn
- Total model calls are auditable

### I3: Reproducibility

Every run writes a manifest (see `REPRODUCIBILITY.md`).

### I4: DSPy Isolation

DSPy is **never** imported at runtime:

```python
# CORRECT: Lazy import only in offline compiler
def compile_policy():
    import dspy  # Only here, never at runtime
    ...

# WRONG: Import at module level
import dspy  # Never do this in runtime code
```

---

## Integration with Existing Components

### StrategicAdvisor (R4)

The `StrategicAdvisor` from Round 4 operates **within** Layer 0:

```
TutorEnvelope
    └── Layer 1 (Schema)
    └── Layer 2 (Repair)
    └── Layer 3 (Policy)
    └── Tutor (Layer 0)
            └── StrategicAdvisor
                    └── FailureDetector
                    └── MetaCardSelector
                    └── ContradictionDetector
```

The envelope wraps the Tutor; it does not replace the Strategic Advisor.

### FailureDetector

The `FailureDetector` continues to operate unchanged:

| Failure Type | Detection | Treatment |
|--------------|-----------|-----------|
| `REPETITION` | Same command ≥3 times | Meta-card injection |
| `CIRCULAR` | Loop in action sequence | WIP enforcement |
| `EXPLICIT_ERROR` | Shell error code ≠ 0 | Error handling |
| `STALL` | No progress for N turns | Goal refocus |

### SemanticIntentTracker

Intent tracking operates at Layer 0, unaffected by envelope layers.

---

## Configuration Schema

```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class EnvelopeConfig:
    """Configuration for TutorEnvelope."""

    # Layer 1: Interface Safety
    layer1_schema: Literal["off", "strict", "lenient"] = "off"

    # Layer 2: Repair
    layer2_repair: Literal["off", "bestofn"] = "off"
    repair_max_attempts: int = 0
    repair_trigger_verdicts: tuple = ("ILL_TYPED", "UNDECIDABLE")
    repair_temperature: float = 0.0  # Must be 0.0 for determinism
    repair_seed: Optional[int] = None

    # Layer 3: Policy
    layer3_policy: Literal["off", "bundle"] = "off"
    policy_bundle_path: Optional[str] = None

    # Manifest
    manifest_enabled: bool = True
    manifest_path: Optional[str] = None

    # Safety
    deterministic_mode: bool = True  # Enforce temp=0.0
```

---

## Implementation Checklist

### Phase 1: Preparation (No Behavior Change)

- [ ] Add `TUTOR_API_VERSION` constant to `tutor.py`
- [ ] Create `TutorV1 = Tutor` alias for explicit baseline reference
- [ ] Add `commit` parameter to `Tutor.check()` for non-committing mode
- [ ] Generate golden traces for compatibility tests

### Phase 2: Envelope Structure

- [ ] Create `tutor_envelope.py` with `TutorEnvelope` class
- [ ] Implement `EnvelopeConfig` dataclass
- [ ] Ensure `TutorEnvelope(all_off)` passes golden trace test

### Phase 3: Layer Implementation

- [ ] Layer 1: Schema parser with format normalization
- [ ] Layer 2: BestOfN repair loop with non-committing CHECK
- [ ] Layer 3: Policy bundle loader and executor

### Phase 4: Manifest & Testing

- [ ] Run manifest writer (see `REPRODUCIBILITY.md`)
- [ ] Golden trace compatibility tests
- [ ] Anti-regression CI integration

### Phase 5: DSPy Adapter (Optional)

- [ ] Create `dspy_adapter/` module with lazy imports
- [ ] Implement policy compiler
- [ ] Document bundle format (see `DSPY_INTEGRATION.md`)

---

## References

- `tutor.py` — Baseline Tutor implementation
- `strategic_advisor.py` — R4 Strategic Advisor
- `failure_detector.py` — Failure detection patterns
- `pcg.py` — Proof-Carrying Graph
- `REPRODUCIBILITY.md` — Run manifest specification
- `DSPY_INTEGRATION.md` — DSPy integration guide

---

*RAGIX Interpreter-Tutor Safety Envelope Specification v0.1.0*
