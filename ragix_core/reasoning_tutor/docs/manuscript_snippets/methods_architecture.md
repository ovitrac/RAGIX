# Manuscript Snippet: Methods — Safety Envelope Architecture

**Target location:** Methods / Architecture
**Version:** 1.0.0 (2025-12-27)
**Status:** Paper-ready

---

## Safety Envelope Architecture

The Interpreter-Tutor system employs a layered safety envelope that separates *verification* (the "Law") from *optimization* (the "Lawyer"). This design ensures that baseline behavior remains deterministic and reproducible while allowing optional enhancement layers to be enabled incrementally.

### Layer 0: Hard Gate (Always Active)

The core Tutor implements deterministic verification through the CHECK protocol:

- **CHECK verdicts:** PROVABLE, REFUTABLE, UNDECIDABLE, ILL_TYPED
- **Move verdicts:** LEGAL, ILLEGAL, BLOCKED
- **Scoring:** Deterministic rules (e.g., +3 for validated truth, −3 for illegal move)
- **Execution:** Sandboxed shell with command denylist

This layer is immutable and forms the basis for all reproducibility guarantees.

### Layer 1: Interface Safety (Optional)

Schema enforcement prevents format interference:

- **Schema validation:** Reject unknown action types before processing
- **Format normalization:** Strip markdown fences, backticks (when enabled)
- **Command validation:** Distinguish shell commands from natural language

When disabled, parsing follows the legacy path unchanged.

### Layer 2: Inference-Time Repair (Optional)

Model-agnostic retry loop for recovering from CHECK failures:

- **Non-committing CHECK:** Evaluate candidates without shell execution or state mutation
- **Trigger verdicts:** ILL_TYPED, UNDECIDABLE (configurable)
- **Budget:** Deterministic maximum retries per turn
- **Selection:** Best candidate by evidence-gain proxy

When disabled, no additional model calls occur.

### Layer 3: Policy Bundle Execution (Optional)

Runtime execution of frozen, pre-compiled policy bundles:

- **Contents:** Prompts, exemplars, model parameters
- **Format:** Plain YAML/JSON (no runtime optimizer dependency)
- **Compilation:** Offline only (DSPy or manual)

When disabled, default prompts are used unchanged.

### Invariant: Baseline Equivalence

When all optional layers are disabled:

```
TutorEnvelope(layer1="off", layer2="off", layer3="off") ≡ TutorV1
```

This equivalence is verified by golden trace compatibility tests.

---

*Methods snippet v1.0.0*
