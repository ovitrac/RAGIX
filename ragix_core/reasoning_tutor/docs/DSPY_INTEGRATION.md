# DSPy Integration Guide

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 0.1.0 (2025-12-27)
**Status:** Specification (implementation pending)

---

## Overview

This document specifies how DSPy integrates with the Interpreter-Tutor system under the **Law vs. Lawyer** model. DSPy is an **optional offline optimizer** that compiles policy bundles; runtime execution **never** imports or depends on DSPy.

**Critical principle:** DSPy is the "Lawyer" — it advises and optimizes. The Tutor is the "Law" — it enforces and verifies. The Law never changes based on the Lawyer's advice.

---

## The Law vs. Lawyer Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RUNTIME (Law)                                     │
│                                                                             │
│   Policy Bundle ─────────────►  TutorEnvelope.Layer3  ────────►  Tutor      │
│   (frozen JSON/YAML)            (executes frozen params)         (verifies) │
│                                                                             │
│   • No DSPy imports                                                         │
│   • Deterministic execution                                                 │
│   • Auditable via manifest                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE (Lawyer)                                    │
│                                                                             │
│   DSPy Optimizer ─────────────►  Policy Bundle                              │
│   (imports dspy-ai)              (JSON/YAML output)                         │
│                                                                             │
│   • Optimizes prompts and exemplars                                         │
│   • Uses Tutor-grounded metric                                              │
│   • Runs only during development/research                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Separation?

1. **Reproducibility:** Paper results don't depend on DSPy version/behavior
2. **Deployment simplicity:** Production runs require only the Tutor
3. **Auditability:** Policy bundles are inspectable artifacts
4. **Scientific rigor:** Optimization is a controlled experiment, not magic

---

## Module Structure

### Directory Layout

```
ragix_core/reasoning_tutor/
├── tutor.py                    # Law: Baseline Tutor (unchanged)
├── tutor_envelope.py           # Optional: TutorEnvelope wrapper
├── manifest.py                 # Run manifest writer
├── metrics.py                  # Tutor-grounded metric
├── schema.py                   # Move schema + parser
│
├── dspy_adapter/               # Lawyer: DSPy integration (optional)
│   ├── __init__.py             # Lazy imports only
│   ├── compiler.py             # Policy bundle compiler
│   ├── signatures.py           # DSPy signatures for Tutor
│   ├── modules.py              # DSPy modules (MoveGenerator, etc.)
│   └── optimizers.py           # Optimizer wrappers (MIPRO, etc.)
│
├── policies/                   # Compiled policy bundles
│   ├── baseline.yaml           # Default (no optimization)
│   └── round4_optimized.yaml   # Example optimized bundle
│
└── tests/
    ├── golden/                 # Golden traces for compatibility
    └── test_compatibility_golden_trace.py
```

### Lazy Import Pattern

DSPy is **never** imported at module level. All imports are deferred to function scope:

```python
# dspy_adapter/compiler.py

def compile_policy(
    train_traces: list,
    metric_config: dict,
    output_path: str
) -> dict:
    """Compile optimized policy bundle using DSPy.

    DSPy import happens HERE, not at module level.
    This function is ONLY called during offline compilation.
    """
    # Lazy import — DSPy only loaded when this function is called
    import dspy
    from dspy.teleprompt import MIPRO

    # ... optimization logic ...

    return bundle
```

**Verification:** The following must always succeed without DSPy installed:

```python
from ragix_core.reasoning_tutor import Tutor, TutorEnvelope
# No ImportError — DSPy not required for runtime
```

---

## Policy Bundle Format

### Schema Definition

Policy bundles are frozen YAML/JSON files containing all parameters needed for Layer 3 execution:

```yaml
# policies/round4_optimized.yaml

bundle_version: "1.0.0"
bundle_id: "round4_optimized_v1"
bundle_hash: "sha256:abc123..."

metadata:
  compiled_at: "2025-12-27T10:00:00Z"
  compiled_by: "dspy_adapter.compiler"
  dspy_version: "2.5.0"
  optimizer: "MIPRO"
  optimizer_config:
    num_candidates: 10
    num_threads: 4
  training_traces: 50
  validation_score: 185.3

prompts:
  system: |
    You are a reasoning agent operating under the Truth-Dare protocol.
    Available actions: PROPOSE, ASSERT, DARE, CHECK, ABORT.

    Rules:
    - PROPOSE introduces a hypothesis (requires evidence)
    - ASSERT claims a proven fact (must be PROVABLE)
    - DARE challenges a claim (shifts burden of proof)
    - CHECK verifies current state
    - ABORT terminates with failure acknowledgment

  action_menu: |
    Based on the current state, choose your next action.
    Consider: What evidence do you have? What can you prove?

  error_recovery: |
    Your previous action was rejected. Analyze the error and try again.
    Common issues: insufficient evidence, invalid syntax, blocked move.

exemplars:
  - input: "Goal: Prove that file.py contains a function named 'process'"
    output: |
      {"action": "bash", "command": "grep -n 'def process' file.py"}
    rationale: "Direct evidence gathering via grep"

  - input: "Evidence shows: def process(data): found at line 42"
    output: |
      {"action": "respond", "message": "ASSERT: file.py contains function 'process' at line 42. Evidence: grep output."}
    rationale: "Valid assertion with direct evidence"

model_params:
  temperature: 0.0          # REQUIRED: determinism
  top_p: 1.0
  max_tokens: 2048
  stop_sequences: ["```\n\n"]

metric_config:
  version: "1.0.0"
  weights:
    goal: 200
    provable: 5
    refutable: 3
    undecidable: -10
    ill_typed: -20
    illegal: -30
    evidence_node: 1
    evidence_edge: 1
    undecidable_resolved: 5
    repetition_penalty: -10
    noop_penalty: -5
  repetition_window: 3
  low_info_actions: ["CHECK"]
```

### Bundle Loader

```python
# In tutor_envelope.py

from pathlib import Path
import yaml
import hashlib

def load_policy_bundle(path: str) -> dict:
    """Load and validate a policy bundle.

    Args:
        path: Path to YAML/JSON bundle file

    Returns:
        Validated bundle dictionary

    Raises:
        ValueError: If bundle is invalid or corrupted
    """
    path = Path(path)

    with open(path) as f:
        if path.suffix in ('.yaml', '.yml'):
            bundle = yaml.safe_load(f)
        else:
            bundle = json.load(f)

    # Validate required fields
    required = ['bundle_version', 'prompts', 'model_params']
    for field in required:
        if field not in bundle:
            raise ValueError(f"Missing required field: {field}")

    # Verify hash if present
    if 'bundle_hash' in bundle:
        computed = compute_bundle_hash(bundle)
        if computed != bundle['bundle_hash']:
            raise ValueError("Bundle hash mismatch — file may be corrupted")

    return bundle
```

---

## Tutor-Grounded Metric

### Design Principles

The DSPy metric must:

1. **Be deterministic:** Same trace → same score
2. **Use Tutor artifacts:** Only data the Tutor already computes
3. **Resist gaming:** Penalize cheap/repetitive moves
4. **Align with Law:** Reward what the Tutor considers "good"

### Formal Definition

For a trace τ = (s_t, m_t, v_t, Δ_t) where t = 1...T:

- s_t = state at turn t
- m_t = proposed move
- v_t = CHECK verdict (PROVABLE, REFUTABLE, UNDECIDABLE, ILL_TYPED)
- Δ_t = state delta (PCG changes)

```
M(τ) = S_goal + S_verdict + S_evidence + S_repeat + S_noop + S_time
```

### Component Definitions

#### Verdict Scoring (Epistemic Safety)

Using the current Tutor nomenclature:

```python
def score_verdict(verdict: CheckVerdict, weights: dict) -> float:
    """Score a CHECK verdict.

    Maps Tutor verdicts to reward signals.
    """
    mapping = {
        CheckVerdict.PROVABLE: weights.get('provable', 5),
        CheckVerdict.REFUTABLE: weights.get('refutable', 3),
        CheckVerdict.UNDECIDABLE: weights.get('undecidable', -10),
        CheckVerdict.ILL_TYPED: weights.get('ill_typed', -20),
    }
    return mapping.get(verdict, 0)
```

#### Evidence Gain (Anti-Stagnation)

```python
@dataclass
class PCGDelta:
    """Tracks changes to Proof-Carrying Graph."""
    nodes_added: int = 0
    edges_added: int = 0
    undecidables_resolved: int = 0

def score_evidence(delta: PCGDelta, weights: dict) -> float:
    """Score evidence progress.

    Rewards meaningful state changes, penalizes stagnation.
    """
    return (
        weights.get('evidence_node', 1) * delta.nodes_added +
        weights.get('evidence_edge', 1) * delta.edges_added +
        weights.get('undecidable_resolved', 5) * delta.undecidables_resolved
    )
```

#### Anti-Gaming Penalties

```python
def score_repetition(
    move_signatures: list[str],
    window_k: int,
    penalty: float
) -> float:
    """Penalize repetitive moves.

    A move signature captures action type + normalized arguments.
    """
    penalties = 0
    for t in range(window_k, len(move_signatures)):
        window = set(move_signatures[t-window_k:t])
        if move_signatures[t] in window:
            penalties += penalty
    return penalties  # Already negative

def score_noop(
    moves: list,
    deltas: list[PCGDelta],
    low_info_actions: set,
    penalty: float
) -> float:
    """Penalize low-information moves with zero evidence gain."""
    penalties = 0
    for move, delta in zip(moves, deltas):
        is_low_info = move.action in low_info_actions
        zero_gain = (delta.nodes_added + delta.edges_added +
                     delta.undecidables_resolved) == 0
        if is_low_info and zero_gain:
            penalties += penalty
    return penalties  # Already negative
```

### Complete Metric Implementation

```python
# metrics.py

from dataclasses import dataclass
from typing import Optional
from .tutor import CheckVerdict, MoveVerdict

@dataclass
class MetricConfig:
    """Configuration for Tutor-grounded metric."""
    version: str = "1.0.0"

    # Goal attainment
    weight_goal: float = 200.0
    weight_time: float = -1.0  # Per-turn cost

    # Verdict weights
    weight_provable: float = 5.0
    weight_refutable: float = 3.0
    weight_undecidable: float = -10.0
    weight_ill_typed: float = -20.0
    weight_illegal: float = -30.0

    # Evidence weights
    weight_node: float = 1.0
    weight_edge: float = 1.0
    weight_resolved: float = 5.0

    # Anti-gaming
    repetition_window: int = 3
    repetition_penalty: float = -10.0
    noop_penalty: float = -5.0
    low_info_actions: tuple = ("CHECK",)


def score_trace(
    trace: list[dict],
    goal_reached: bool,
    config: Optional[MetricConfig] = None
) -> float:
    """Compute Tutor-grounded metric for a trace.

    Args:
        trace: List of turn records with verdicts, deltas, signatures
        goal_reached: Whether the episode succeeded
        config: Metric configuration (uses defaults if None)

    Returns:
        Total score for the trace
    """
    if config is None:
        config = MetricConfig()

    score = 0.0

    # Goal attainment
    if goal_reached:
        score += config.weight_goal

    # Time cost
    score += config.weight_time * len(trace)

    # Per-turn scoring
    signatures = []
    for turn in trace:
        # Verdict scoring
        verdict = turn.get('check_verdict')
        if verdict == 'PROVABLE':
            score += config.weight_provable
        elif verdict == 'REFUTABLE':
            score += config.weight_refutable
        elif verdict == 'UNDECIDABLE':
            score += config.weight_undecidable
        elif verdict == 'ILL_TYPED':
            score += config.weight_ill_typed

        # Move verdict
        move_verdict = turn.get('move_verdict')
        if move_verdict == 'ILLEGAL':
            score += config.weight_illegal

        # Evidence gain
        delta = turn.get('pcg_delta', {})
        score += config.weight_node * delta.get('nodes_added', 0)
        score += config.weight_edge * delta.get('edges_added', 0)
        score += config.weight_resolved * delta.get('undecidables_resolved', 0)

        # Collect signature for repetition check
        signatures.append(turn.get('move_signature', ''))

    # Repetition penalty
    k = config.repetition_window
    for t in range(k, len(signatures)):
        window = set(signatures[t-k:t])
        if signatures[t] and signatures[t] in window:
            score += config.repetition_penalty

    # No-op penalty
    for turn in trace:
        action = turn.get('action_type', '')
        delta = turn.get('pcg_delta', {})
        is_low_info = action in config.low_info_actions
        zero_gain = (
            delta.get('nodes_added', 0) +
            delta.get('edges_added', 0) +
            delta.get('undecidables_resolved', 0)
        ) == 0
        if is_low_info and zero_gain:
            score += config.noop_penalty

    return score
```

---

## Offline Compilation Workflow

### Prerequisites

```bash
# 1. Install DSPy (only needed for compilation)
pip install dspy-ai>=2.5.0

# 2. Generate training traces
cd ragix_core/reasoning_tutor
python -m benchmarks.scored_mode \
    --round 4 \
    --models "granite3.1-moe:3b" \
    --output traces/training_traces.jsonl
```

### Compilation Script

```python
# dspy_adapter/compiler.py

def compile_policy(
    training_traces_path: str,
    output_path: str,
    metric_config: dict,
    optimizer: str = "MIPRO",
    num_candidates: int = 10
) -> dict:
    """Compile optimized policy bundle.

    This function imports DSPy — it is NEVER called at runtime.

    Args:
        training_traces_path: Path to JSONL training traces
        output_path: Output path for compiled bundle
        metric_config: Metric weights and parameters
        optimizer: DSPy optimizer to use
        num_candidates: Number of candidates to evaluate

    Returns:
        Compiled bundle dictionary
    """
    # Lazy import — DSPy only here
    import dspy
    from dspy.teleprompt import MIPRO, BootstrapFewShot

    # Load training traces
    traces = load_traces(training_traces_path)

    # Create DSPy metric from config
    def tutor_metric(example, prediction, trace=None):
        # Evaluate prediction against Tutor
        # Return score using our metric
        return score_trace(trace, example.goal_reached, metric_config)

    # Configure optimizer
    if optimizer == "MIPRO":
        teleprompter = MIPRO(
            metric=tutor_metric,
            num_candidates=num_candidates
        )
    elif optimizer == "BootstrapFewShot":
        teleprompter = BootstrapFewShot(
            metric=tutor_metric,
            max_bootstrapped_demos=3
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Create and optimize module
    module = TutorMoveGenerator()
    optimized = teleprompter.compile(module, trainset=traces)

    # Extract frozen parameters
    bundle = {
        'bundle_version': '1.0.0',
        'bundle_id': f'{optimizer.lower()}_optimized',
        'metadata': {
            'compiled_at': datetime.utcnow().isoformat() + 'Z',
            'compiled_by': 'dspy_adapter.compiler',
            'dspy_version': dspy.__version__,
            'optimizer': optimizer,
            'training_traces': len(traces)
        },
        'prompts': extract_prompts(optimized),
        'exemplars': extract_exemplars(optimized),
        'model_params': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': 2048
        },
        'metric_config': metric_config
    }

    # Compute hash
    bundle['bundle_hash'] = compute_bundle_hash(bundle)

    # Save
    with open(output_path, 'w') as f:
        yaml.dump(bundle, f, default_flow_style=False)

    return bundle
```

### Running Compilation

```bash
# Compile policy bundle (requires DSPy)
cd ragix_core/reasoning_tutor
python -m dspy_adapter.compiler \
    --traces traces/training_traces.jsonl \
    --output policies/round4_optimized.yaml \
    --optimizer MIPRO \
    --candidates 10

# Verify bundle
python -c "
from tutor_envelope import load_policy_bundle
bundle = load_policy_bundle('policies/round4_optimized.yaml')
print(f'Bundle: {bundle[\"bundle_id\"]}')
print(f'Hash: {bundle[\"bundle_hash\"]}')
"
```

---

## Runtime Execution (No DSPy)

### Using a Policy Bundle

```python
from tutor import Tutor
from tutor_envelope import TutorEnvelope, EnvelopeConfig

# Create envelope with policy bundle
config = EnvelopeConfig(
    layer1_schema="off",
    layer2_repair="off",
    layer3_policy="bundle",
    policy_bundle_path="policies/round4_optimized.yaml"
)

envelope = TutorEnvelope(
    base_tutor=Tutor(),
    config=config
)

# Run benchmark — NO DSPy imports occur
result = envelope.run_episode(benchmark)
```

### Verification: No DSPy Dependency

```python
# This test must pass with DSPy NOT installed
import sys

# Ensure DSPy is not available
if 'dspy' in sys.modules:
    del sys.modules['dspy']

# Import and use envelope
from tutor import Tutor
from tutor_envelope import TutorEnvelope, EnvelopeConfig

config = EnvelopeConfig(
    layer3_policy="bundle",
    policy_bundle_path="policies/round4_optimized.yaml"
)

envelope = TutorEnvelope(base_tutor=Tutor(), config=config)
# Should work without ImportError
```

---

## Mapping to Current Implementation

### Verdict Mapping

| DSPy Metric Term | Current Tutor | Location |
|------------------|---------------|----------|
| PROVABLE | `CheckVerdict.PROVABLE` | `tutor.py` |
| REFUTABLE | `CheckVerdict.REFUTABLE` | `tutor.py` |
| UNDECIDABLE | `CheckVerdict.UNDECIDABLE` | `tutor.py` |
| ILL_TYPED | `CheckVerdict.ILL_TYPED` | `tutor.py` |

### Score Mapping

| DSPy Metric Term | Current Tutor | Value |
|------------------|---------------|-------|
| Goal success | `SCORE_WIN` | 200 |
| Truth direct | `SCORE_TRUTH_DIRECT` | 3 |
| Truth by rule | `SCORE_TRUTH_RULE` | 2 |
| Illegal move | `SCORE_ILLEGAL_MOVE` | -3 |

### PCG Delta (To Be Implemented)

The `PCGDelta` dataclass needs to be added to track:

```python
@dataclass
class PCGDelta:
    """Track PCG changes for metric computation."""
    nodes_added: int = 0
    edges_added: int = 0
    undecidables_resolved: int = 0

    @classmethod
    def compute(cls, before: GameState, after: GameState) -> "PCGDelta":
        """Compute delta between two states."""
        # Implementation depends on PCG structure
        ...
```

---

## Anti-Gaming Guarantees

### Why the Metric Resists Gaming

| Gaming Strategy | Metric Response |
|-----------------|-----------------|
| Spam valid no-ops | `S_noop` penalty + zero `S_evidence` |
| Repeat same move | `S_repeat` penalty (window-based) |
| Generate plausible but unverifiable claims | `S_verdict` penalizes UNDECIDABLE |
| Syntax errors for retry farming | `S_verdict` penalizes ILL_TYPED |
| Delayed goal completion | `S_time` penalty per turn |

### Metric-Hacking Regression Tests

```python
# tests/test_metric_gaming.py

def test_noop_spam_penalized():
    """Spamming CHECK with no evidence gain is penalized."""
    trace = [
        {'action_type': 'CHECK', 'pcg_delta': {'nodes_added': 0}},
        {'action_type': 'CHECK', 'pcg_delta': {'nodes_added': 0}},
        {'action_type': 'CHECK', 'pcg_delta': {'nodes_added': 0}},
    ]
    score = score_trace(trace, goal_reached=False)
    assert score < 0, "No-op spam should be penalized"

def test_decisive_dare_beats_noop():
    """One decisive DARE beats multiple no-op moves."""
    noop_trace = [
        {'action_type': 'CHECK', 'check_verdict': 'PROVABLE',
         'pcg_delta': {'nodes_added': 0}},
    ] * 5

    dare_trace = [
        {'action_type': 'DARE', 'check_verdict': 'PROVABLE',
         'pcg_delta': {'undecidables_resolved': 1}},
    ]

    noop_score = score_trace(noop_trace, goal_reached=False)
    dare_score = score_trace(dare_trace, goal_reached=False)

    assert dare_score > noop_score, "Decisive DARE should beat no-ops"

def test_repetition_penalized():
    """Repeating the same move is penalized."""
    trace = [
        {'move_signature': 'bash:ls', 'pcg_delta': {'nodes_added': 1}},
        {'move_signature': 'bash:ls', 'pcg_delta': {'nodes_added': 0}},
        {'move_signature': 'bash:ls', 'pcg_delta': {'nodes_added': 0}},
        {'move_signature': 'bash:ls', 'pcg_delta': {'nodes_added': 0}},
    ]
    score = score_trace(trace, goal_reached=False)

    # Compare to varied trace
    varied = [
        {'move_signature': 'bash:ls', 'pcg_delta': {'nodes_added': 1}},
        {'move_signature': 'bash:grep', 'pcg_delta': {'nodes_added': 1}},
        {'move_signature': 'bash:cat', 'pcg_delta': {'nodes_added': 1}},
        {'move_signature': 'respond', 'pcg_delta': {'nodes_added': 1}},
    ]
    varied_score = score_trace(varied, goal_reached=False)

    assert varied_score > score, "Varied moves should beat repetition"
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure (No DSPy)

- [ ] Add `PCGDelta` dataclass to `tutor.py`
- [ ] Implement `metrics.py` with `score_trace()`
- [ ] Create `policies/baseline.yaml` (default, no optimization)
- [ ] Add metric-hacking regression tests

### Phase 2: DSPy Adapter (Optional Module)

- [ ] Create `dspy_adapter/__init__.py` with lazy imports
- [ ] Implement `dspy_adapter/compiler.py`
- [ ] Implement `dspy_adapter/signatures.py`
- [ ] Add compilation CLI

### Phase 3: Integration Testing

- [ ] Test runtime without DSPy installed
- [ ] Verify compiled bundles produce valid output
- [ ] Compare optimized vs. baseline performance

---

## References

- `SAFETY_ENVELOPE.md` — Envelope architecture
- `REPRODUCIBILITY.md` — Run manifest specification
- `tutor.py` — Baseline Tutor implementation
- [DSPy Documentation](https://dspy-docs.vercel.app/) — DSPy framework

---

*RAGIX Interpreter-Tutor DSPy Integration Guide v0.1.0*
