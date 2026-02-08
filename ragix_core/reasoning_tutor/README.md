# Interpreter-Tutor: Rule-Based Reasoning for Slim LLMs

**A Game-Theoretic Architecture for Reliable AI-Assisted Shell Operations**

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 0.5.0 (2026-02-03)

---

## Abstract

This module implements a novel **Interpreter-Tutor** architecture that transforms unreliable slim LLMs into disciplined software engineering assistants. Instead of asking LLMs to reason, we frame interaction as a **turn-based proof game** where:

- **LLM (Player):** Proposes moves (commands, claims, questions)
- **Tutor (Referee):** Validates moves, executes actions, derives truths deterministically

This separation achieves **hallucination suppression by structure** — illegal moves are rejected, truths require evidence proofs, and all decisions are auditable.

---

## Installation

The reasoning_tutor requires additional dependencies beyond the core RAGIX requirements.

```bash
# 1. Install core RAGIX dependencies (from project root)
pip install -r requirements.txt

# 2. Install reasoning_tutor analysis dependencies
pip install -r ragix_core/reasoning_tutor/requirements.txt

# 3. (Optional) Install DSPy for policy compilation
pip install dspy-ai>=2.5.0
```

**Core packages (required):**

| Package | Purpose |
|---------|---------|
| `numpy` | Semantic similarity calculations |
| `pandas` | Feature matrix analysis |
| `matplotlib` | Visualization (PCA plots, charts) |
| `scipy` | Dendrograms, hierarchical clustering |
| `scikit-learn` | PCA, StandardScaler |

**Optional packages:**

| Package | Purpose |
|---------|---------|
| `dspy-ai` | Offline policy compilation (TutorEnvelope Layer 3) |

**Note:** If you only use the core Tutor (no visualization/analysis), `numpy` is the only required addition.

### Backward Compatibility

The Tutor is designed with strict backward compatibility:

- **TutorV1** (existing) remains unchanged and is the default
- **TutorEnvelope** (optional wrapper) adds layers without modifying TutorV1
- **DSPy is never required at runtime** — only for offline policy compilation
- **Previous production scripts continue to work unchanged**

This follows the **"Law vs. Lawyer"** model:
- **Law (TutorV1):** Deterministic verification, CHECK protocol, scoring — immutable
- **Lawyer (DSPy):** Optional optimizer that compiles policy bundles offline

See `REVIEW_extensions.md` for the full integration specification.

---

## Motivation

### The Problem with Slim LLMs

Basic and mid-scale LLMs (3B-7B parameters) exhibit structural limitations:

| Limitation | Consequence |
|------------|-------------|
| Short working memory | Cannot track long reasoning chains |
| Weak enumeration | Fails at counting, listing |
| Hallucination under uncertainty | Invents plausible but false facts |
| Non-deterministic | Same input → different outputs |

**Key insight:** These are not flaws to fix — they are intrinsic properties of probabilistic language models.

### The Solution: Game Framing

Instead of fighting LLM limitations, we **leverage LLM strengths**:

| LLM Weakness | Tutor Compensates | LLM Strength | We Exploit |
|--------------|-------------------|--------------|------------|
| Memory | Proof-Carrying Graph | Creativity | Move proposals |
| Enumeration | Deterministic rules | Pattern matching | JSON generation |
| Truth validation | Evidence proofs | Language fluency | Intent expression |
| Determinism | Append-only log | Flexibility | Adaptation |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         GAME LOOP                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐         ┌──────────┐         ┌──────────┐     │
│   │   LLM    │ PROPOSE │  TUTOR   │ EXECUTE │  SHELL   │     │
│   │ (Player) │────────▶│(Referee) │────────▶│(Sandbox)│       │
│   └──────────┘         └────┬─────┘         └────┬─────┘        │
│        ▲                    │                    │             │
│        │                    ▼                    ▼              │
│        │              ┌──────────┐         ┌──────────┐         │
│        │              │  RULES   │         │   OBS    │         │
│        │              │ (YAML)   │────────▶│(Evidence)│         │
│        │              └──────────┘  DERIVE └────┬─────┘         │
│        │                                        │               │
│        │              ┌──────────┐              │               │
│        └──────────────│   PCG    │◀─────────────┘               │
│           CONTEXT     │ (Graph)  │    APPEND                    │
│                       └──────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Role | Implementation |
|-----------|------|----------------|
| **PCG** | Append-only proof graph | `pcg.py` — Nodes, edges, events |
| **Rules** | Declarative inference | `rules.py` — YAML + typed operators |
| **Tutor** | Deterministic referee | `tutor.py` — CHECK protocol |
| **Moves** | LLM output parser | `moves.py` — JSON structured output |

---

## The CHECK Protocol

Every claim must pass through the **decidability ladder**:

```
CHECK(T) → { provable | refutable | undecidable | ill-typed }
```

| Verdict | Meaning | Action |
|---------|---------|--------|
| **provable** | Has evidence proof | Promote to validated |
| **refutable** | Contradicted by evidence | Mark refuted |
| **undecidable** | Missing information | Truth/Dare mechanism |
| **ill-typed** | Invalid formulation | Reject move |

### Truth/Dare Mechanism

When a claim is **undecidable**, exactly two options:

- **TRUTH:** Reformulate into a decidable claim
- **DARE:** Propose an evidence-producing action

No third option exists. This forces progress without hallucination.

---

## Rule System

### Two-Tier Rules

| Tier | Lifetime | Source | Trust |
|------|----------|--------|-------|
| **Permanent** | Persisted | Human-curated | Audited |
| **Session** | Game only | Fat LLM generated | Disposable |

### YAML Format with Typed Operators

```yaml
- id: R_file_exists_test
  soundness: sound
  description: "test -f succeeds means file exists"
  match:
    - obs.tool: {eq: "bash"}
    - obs.command: {matches: "test -[fe] "}
    - obs.rc: {eq: 0}
  extract:
    path:
      regex: "test -[fe] ['\"]?([^'\"\\s]+)"
      from: obs.command
  conclude:
    truth:
      text: "File '{path}' exists"
      kind: existence
      scope: "{path}"
    entity:
      kind: file
      value: "{path}"
```

### Typed Operators (Safe, No Eval)

| Operator | Meaning |
|----------|---------|
| `eq`, `neq` | Equality |
| `in` | Value in list |
| `contains` | Substring |
| `matches` | Regex |
| `gt`, `lt`, `gte`, `lte` | Numeric |
| `exists` | Field present |

---

## Benchmarks

### Central Hypothesis: Model Independence

The benchmark suite tests a fundamental claim:

> **REASONING CORRECTNESS is MODEL-INDEPENDENT**
>
> The Tutor's CHECK protocol produces identical verdicts regardless of which LLM proposes moves.
>
> **REASONING EFFICIENCY is MODEL-DEPENDENT**
>
> Better models reach goals in fewer turns with fewer hallucinations.

This enables **fair comparison** across LLM architectures and sizes.

### Scenario Categories

| Category | Description | Tests | Metrics |
|----------|-------------|-------|---------|
| **Discovery** | Find files matching criteria | grep, find | Turns to solution |
| **Enumeration** | Count, list, aggregate | wc, arithmetic | Hallucination rate |
| **Verification** | Confirm multi-step properties | Import chains | Proof chain length |
| **Undecidable** | Force Truth/Dare mechanism | Missing evidence | Resolution quality |
| **Rule Generation** | Create session rules | Custom formats | Rule reusability |
| **Memory** | Context retention across turns | Multi-file synthesis | Memory efficiency |

### Benchmark Suite

```
benchmarks/
├── __init__.py                    # Package exports
├── 01_find_needle.yaml            # Find expression in 10 files
├── 02_count_lines.yaml            # Count and sum (tests enumeration)
├── 03_undecidable_claim.yaml      # Force undecidable → Truth/Dare
├── 04_verification_chain.yaml     # Multi-step proof building
├── 05_session_rules.yaml          # Domain-specific rule generation
├── 06_memory_recall.yaml          # Multi-file context retention test
├── metrics.py                     # GameMetrics, BenchmarkResult, ModelComparison
├── runner.py                      # Multi-model benchmark runner
├── assisted_mode.py               # Card-assisted exploration mode
├── scored_mode.py                 # Full scoring with points/penalties
├── diagnose.py                    # Diagnostic tracing for failures
├── diagnose_with_menu.py          # Action menu diagnostic mode
└── demo_model_independence.py     # Determinism verification
```

### Benchmark Modes

The framework provides multiple execution modes for different purposes:

| Mode | Purpose | Key Feature |
|------|---------|-------------|
| **Assisted Mode** | Card-based exploration | LLM chooses from action menu (TRIZ-like RAG) |
| **Scored Mode** | Quantitative evaluation | Points/penalties for systematic comparison |
| **Diagnose Mode** | Failure analysis | Full conversation tracing |
| **Diagnose with Menu** | Semantic action testing | Tests if LLMs understand intent vs syntax |

#### Assisted Mode (`assisted_mode.py`)

LLMs explore benchmarks using a "card deck" of generic solutions:

```bash
python -m ragix_core.reasoning_tutor.benchmarks.assisted_mode \
    --benchmark 01 --models "granite3.1-moe:3b,mistral:latest"
```

The card deck acts as a **procedural RAG** — generic retrieval methods that slim LLMs can invoke when they lack specific bash syntax knowledge. Cards are "decimated" (removed) after use to ensure exploration.

#### Scored Mode (`scored_mode.py`)

Full quantitative evaluation with points and penalties:

```bash
python -m ragix_core.reasoning_tutor.benchmarks.scored_mode \
    --models "granite3.1-moe:3b,mistral:latest" --max-turns 6
```

**Scoring System:**

| Event | Points | Rationale |
|-------|--------|-----------|
| Own solution success | +100 | Rewards independent problem-solving |
| Card solution success | +50 | Cards help but cost points |
| Goal achieved bonus | +200 | Strong incentive to reach goal |
| Efficient path bonus | +50 | Bonus if turns ≤ optimal |
| Syntax error | -20 | Penalize invalid commands |
| File not found | -15 | Penalize wrong paths |
| Repeated action | -30 | Discourage loops |
| Card menu cost | -10 | Cost to access help |
| Empty response | -25 | Penalize non-responses |
| Timeout | -50 | Penalize slow responses |

This scoring encourages LLMs to **try their own solutions first** before falling back to the card menu.

### Running Benchmarks

```bash
# Run all benchmarks with granite3.1-moe:3b
python -m ragix_core.reasoning_tutor.benchmarks.runner

# Compare two models
python -m ragix_core.reasoning_tutor.benchmarks.runner --models granite3.1-moe:3b,qwen2.5-coder:7b

# Demonstrate model-independence
python -m ragix_core.reasoning_tutor.benchmarks.demo_model_independence
```

### Key Metrics

| Metric | Type | Definition |
|--------|------|------------|
| **Turns to Goal** | Efficiency | Number of turns to satisfy goal |
| **Rule Coverage** | Correctness | % of observations matched by rules |
| **Truth Accuracy** | Correctness | % of validated truths that are correct |
| **Hallucination Rate** | Robustness | % of ASSERT moves rejected (undecidable) |
| **Truth/Dare Ratio** | Reasoning | How undecidable claims are resolved |
| **Session Rules Generated** | Reasoning | Ad-hoc rules created during game |

### Metric Collection

```python
from ragix_core.reasoning_tutor.benchmarks import GameMetrics, ModelComparison

# Metrics for a single game
metrics = GameMetrics(game_id="test_001", model_name="granite3.1-moe:3b")
metrics.add_move(move_record)
metrics.finalize(goal_achieved=True)

# Cross-model comparison
comparison = ModelComparison(benchmark_id="01", benchmark_name="Find Needle")
comparison.add_result("granite3.1-moe:3b", result_granite)
comparison.add_result("qwen2.5-coder:7b", result_qwen)
print(comparison.generate_report())
```

### Benchmark Results (2025-12-22)

Results from testing with `granite3.1-moe:3b` and `mistral:latest` on Ollama.

#### Assisted Mode Results

Card-assisted exploration with max 6 turns:

| Benchmark | granite3.1-moe:3b | mistral:latest |
|-----------|-------------------|----------------|
| B01 - Find Needle | ✓ Turn 1 | ✓ Turn 2 |
| B02 - Count Lines | ✓ Turn 2 | ✓ Turn 2 |
| B03 - Undecidable | ✓ Turn 2 | ✓ Turn 1 |
| B04 - Verification | ✗ (partial) | ✗ (partial) |
| B05 - Session Rules | ✓ Turn 3 | ✓ Turn 3 |
| **Success Rate** | **4/5 (80%)** | **4/5 (80%)** |

**Observations:**
- Both models achieve similar success rates with card assistance
- B04 (Verification) fails because models read files but don't explicitly state "valid" in their findings
- Card decimation (removing used cards) prevents infinite loops

#### Scored Mode Results

Full scoring with points/penalties, all 6 benchmarks:

| Model | Success | Total Score | Avg Score/Benchmark |
|-------|---------|-------------|---------------------|
| granite3.1-moe:3b | 2/6 (33%) | 1250 | 208 |
| mistral:latest | 1/6 (17%) | 815 | 136 |

**Key Insight:** Granite achieves higher total score despite similar/fewer successes because:
- Fewer syntax errors (-20 each)
- Fewer repeated actions (-30 each)
- More successful intermediate commands (+50/+100 each)

**Benchmark Details (Scored Mode):**

| Benchmark | granite3.1-moe:3b | mistral:latest |
|-----------|-------------------|----------------|
| B01 - Find Needle | ✓ 350 pts | ✗ 125 pts |
| B02 - Count Lines | ✓ 350 pts | ✗ 125 pts |
| B03 - Undecidable | ✗ 200 pts | ✓ 300 pts |
| B04 - Verification | ✗ 100 pts | ✗ 75 pts |
| B05 - Session Rules | ✗ 150 pts | ✗ 100 pts |
| B06 - Memory Recall | ✗ 100 pts | ✗ 90 pts |

#### Memory Benchmark Insights

The B06 Memory Recall benchmark reveals a key limitation of slim LLMs:

- **Task:** Read 4 clue files, remember digits (7, 5, 3, 9), report CODE=7539
- **Observation:** Both models successfully read all files but fail to synthesize the final answer
- **Root cause:** Working memory limitations — models "forget" earlier digits by the time they reach the final file
- **Implication:** The card deck as procedural RAG provides a recovery mechanism (re-read files at a point penalty)

---

## LLM Reasoning Olympics — Round Summary

The benchmark suite has been systematically improved across 5 rounds of testing:

### Round 1 (2025-12-22) — Baseline
- 11 models tested across 6 benchmarks
- Gold: `deepseek-r1:14b` (6/6 wins)
- Finding: Model size ≠ performance

### Round 2 (2025-12-23) — Token Handling
- Fixed `<think>...</think>` token leakage
- deepseek-r1 recovered from 33% to 100%
- Introduced failure detection

### Round 3 (2025-12-24) — Failure Analysis
- Added failure detector (repetition, circular, cascade)
- Behavioral fingerprints for each model
- Finding: Failure patterns are model-specific

### Round 4 (2026-01-15) — TRIZ + Kanban
- Strategic advisor with TRIZ cards
- Kanban WIP limits per model
- granite3.1-moe:3b improved 33% → 100%
- Finding: Scaffolding equalizes small models

### Round 5 (2026-02-03) — Tool Adapter + Synthesis Controller
- Tool adapter for structured tool-calling models
- Synthesis controller for task completion
- Granite 4: -1770 → +3520 (+5290 improvement)
- Finding: Interface contract matters as much as model capability

#### Round 5 Final Results

| Model | Size | Wins | Total Score |
|-------|------|------|-------------|
| gpt-oss-safeguard:120b | 65 GB | **6/6** | +1615 |
| deepseek-r1:14b | ~9 GB | **6/6** | +1600 |
| qwen2.5-coder:7b | ~5 GB | 4/6 | +1720 |
| ibm/granite4:32b-a9b-h | 19 GB | 3/6 | **+3520** |
| granite3.1-moe:3b | ~2 GB | 3/6 | +855 |

**Key Findings:**
1. **Interface Contract > Model Size** — Granite 4 (32B) improved +5290 points with proper adapter
2. **Scaffolding Equalizes** — 14B DeepSeek = 120B GPT-OSS (both 6/6)
3. **Efficiency Trade-off** — 3B model: 427.5 pts/GB vs 120B model: 24.8 pts/GB

For complete technical details, see: [`docs/LLM_OLYMPICS_TECHNICAL_APPENDIX.md`](docs/LLM_OLYMPICS_TECHNICAL_APPENDIX.md)

---

## Scientific Contributions

### Novelty

1. **Game-theoretic framing** — LLM as player, not reasoner
2. **Structural hallucination suppression** — No prompt engineering
3. **Decidability as first-class concept** — Uncertainty is modeled
4. **Two-tier rule system** — Human + LLM collaboration on rules
5. **Append-only auditability** — Complete provenance chain

### Research Questions

1. **Minimal LLM capability:** What is the smallest model that can play the game effectively?
2. **Rule library design:** How to balance coverage vs. complexity?
3. **Session rule quality:** Can LLM-generated rules be automatically promoted?
4. **Convergence:** Does the game always terminate? Under what conditions?
5. **Optimality:** Can we minimize turns to goal with action selection?

### Comparison with Existing Approaches

| Approach | LLM Role | Truth Source | Hallucination Control |
|----------|----------|--------------|----------------------|
| Chain-of-Thought | Reasoner | LLM claims | Prompt engineering |
| ReAct | Actor | LLM claims | Observation feedback |
| Tool Learning | Caller | Tool output | Tool selection |
| **Interpreter-Tutor** | **Player** | **Evidence proofs** | **Structural** |

---

## Usage

### Quick Start

```python
from ragix_core.reasoning_tutor import Tutor, parse_move

# Initialize
tutor = Tutor(game_id="my_game", sandbox_root="/safe/path")
tutor.setup_game("Find all Python files with 'TODO' comments")

# Game loop
while not tutor.check_goal_satisfaction():
    context = tutor.get_context_for_llm()
    llm_response = call_your_llm(context)

    for move in parse_move(llm_response):
        result = tutor.execute_move(move)
        print(f"{move.move_type}: {result.verdict}")
```

### Demo

```bash
# Basic demo with granite3.1-moe:3b
python -m ragix_core.reasoning_tutor.demo

# Interactive mode
python -m ragix_core.reasoning_tutor.demo -i

# Different model
python -m ragix_core.reasoning_tutor.demo --model qwen2.5-coder:7b
```

---

## Testing & CI

### Regression Testing Framework

The benchmark suite includes a comprehensive regression testing framework:

```bash
# List available baselines
python -m ragix_core.reasoning_tutor.tests.regression_runner list

# Run regression tests with fast model (granite3.1-moe:3b)
python -m ragix_core.reasoning_tutor.tests.regression_runner run --fast

# Run with gold standard model (deepseek-r1:14b)
python -m ragix_core.reasoning_tutor.tests.regression_runner run --gold

# Compare historical results against baseline
python -m ragix_core.reasoning_tutor.tests.regression_runner compare results/round5/final/granite3_3b.jsonl
```

### Pytest Integration

```bash
# Quick smoke tests (no Ollama required, ~5 seconds)
pytest ragix_core/reasoning_tutor/tests/test_benchmarks.py -v -m "smoke"

# Full regression tests (requires Ollama, ~2 minutes)
pytest ragix_core/reasoning_tutor/tests/test_benchmarks.py -v -m "regression"
```

### Creating Baselines

```bash
# Create baseline from fresh run
python -m ragix_core.reasoning_tutor.tests.regression_runner baseline --model granite3.1-moe:3b

# Create baseline from existing results file
python -m ragix_core.reasoning_tutor.tests.regression_runner baseline \
    --from-file results/round5/final/deepseek_14b.jsonl --force
```

### CI Helper Script

```bash
# Quick smoke tests
./ragix_core/reasoning_tutor/tests/ci_test.sh smoke

# Full regression
./ragix_core/reasoning_tutor/tests/ci_test.sh regression

# Both
./ragix_core/reasoning_tutor/tests/ci_test.sh all
```

---

## File Structure

```
ragix_core/reasoning_tutor/
├── __init__.py                 # Module exports
├── pcg.py                      # Proof-Carrying Graph (~445 lines)
├── rules.py                    # Rule engine with typed operators (~340 lines)
├── tutor.py                    # Deterministic referee + CHECK protocol (~430 lines)
├── moves.py                    # LLM move parser (~300 lines)
├── action_menu.py              # TRIZ-like action menu for slim LLMs
├── failure_detector.py         # Meta-cognitive failure detection (Round 3)
├── strategic_advisor.py        # TRIZ + Kanban integration (Round 4)
├── tool_call_adapter.py        # Structured tool-call adapter (Round 5)
├── synthesis_controller.py     # EXPLORE → SYNTHESIZE phases (Round 5)
├── demo.py                     # Basic interactive demo
├── README.md                   # This documentation
├── INTERPRET_TUTOR.md          # Full design document
├── docs/
│   └── LLM_OLYMPICS_TECHNICAL_APPENDIX.md  # Publication appendix
├── schema/
│   └── rule.schema.json        # JSON Schema for rule validation
├── rules/
│   ├── bash.rules.yaml         # Shell operation rules (25 rules)
│   └── adhoc/                  # Session rules (auto-generated)
├── results/
│   ├── round2/                 # Round 2 results
│   ├── round5/                 # Round 5 results (latest)
│   │   └── final/              # Final comparison results
│   └── comparison_*/           # Cross-round comparisons
├── tests/
│   ├── __init__.py             # Test package
│   ├── regression_runner.py    # CLI for regression testing
│   ├── test_benchmarks.py      # Pytest test suite
│   ├── smoke_test.py           # Quick validation script
│   ├── ci_test.sh              # CI helper script
│   ├── conftest.py             # Pytest configuration
│   └── baselines/              # Stored baseline results
│       ├── baseline_granite3.1-moe_3b.json
│       ├── baseline_deepseek-r1_14b.json
│       └── baseline_qwen2.5-coder_7b.json
└── benchmarks/
    ├── __init__.py             # Benchmark package exports
    ├── metrics.py              # GameMetrics, ModelComparison classes
    ├── runner.py               # Multi-model benchmark runner
    ├── assisted_mode.py        # Card-assisted exploration mode
    ├── scored_mode.py          # Full scoring with points/penalties (~500 lines)
    ├── diagnose.py             # Diagnostic tracing for failure analysis
    ├── diagnose_with_menu.py   # Semantic action menu diagnostics
    ├── demo_model_independence.py  # Determinism verification demo
    ├── 01_find_needle.yaml     # Scenario: Find expression in files
    ├── 02_count_lines.yaml     # Scenario: Count/sum (tests enumeration)
    ├── 03_undecidable_claim.yaml   # Scenario: Force Truth/Dare
    ├── 04_verification_chain.yaml  # Scenario: Multi-step proofs
    ├── 05_session_rules.yaml   # Scenario: Ad-hoc rule generation
    ├── 06_memory_recall.yaml   # Scenario: Context retention test
    ├── 07_stack_trace.yaml     # Scenario: Causal traceback (B07)
    ├── 08_diff_analysis.yaml   # Scenario: Semantic diff (B08)
    ├── 09_cycle_detection.yaml # Scenario: Graph traversal (B09)
    └── 10_temporal_correlation.yaml  # Scenario: Distributed systems (B10)
```

---

## Roadmap

### v0.1.0 — Core Implementation ✓
- [x] Proof-Carrying Graph (PCG) with append-only semantics
- [x] Rule engine with typed operators (no eval)
- [x] CHECK protocol with decidability ladder
- [x] Move parser for structured LLM output
- [x] 25 bash rules library
- [x] Basic demo with Ollama integration
- [x] 5 benchmark scenarios
- [x] Metrics collection (GameMetrics, ModelComparison)
- [x] Model-independence demonstration

### v0.2.0 — Benchmark Suite ✓
- [x] 6 benchmark scenarios (+1 memory recall)
- [x] Assisted mode with card-based exploration (TRIZ-like RAG)
- [x] Scored mode with points/penalties system
- [x] Diagnostic modes for failure analysis
- [x] Cross-model comparison (granite vs mistral)
- [x] Action menu as procedural RAG for slim LLMs

### v0.3.0 — Failure Analysis ✓
- [x] Failure detector (repetition, circular, cascade patterns)
- [x] Behavioral fingerprints per model
- [x] Reasoning token stripping (`<think>...</think>`)

### v0.4.0 — Strategic Scaffolding ✓
- [x] TRIZ-inspired strategic advisor
- [x] Kanban WIP limits per model profile
- [x] Model-specific intervention thresholds

### v0.5.0 — Tool Adapter & Synthesis (Current)
- [x] Tool Call Adapter for structured tool-calling models
- [x] Synthesis Controller (EXPLORE → SYNTHESIZE phases)
- [x] Goal variable detection per benchmark
- [x] 5-round Olympics with 15+ model configurations
- [x] Technical documentation for publication
- [x] CI integration for regression testing
- [x] B07: Stack Trace Diagnosis (causal traceback)
- [x] B08: Diff Analysis (semantic change detection)
- [x] B09: Dependency Cycle Detection (graph reasoning)
- [x] B10: Temporal Event Correlation (distributed systems)
- [ ] LaTeX export for scientific papers

### v0.6.0 — Benchmark Expansion (Planned)
- [ ] B11+ additional benchmark scenarios
- [ ] Synthesis Gate for evidence sufficiency

### v0.7.0 — Fat LLM Integration (Planned)
- [ ] Rule generation from undecidable claims
- [ ] Automatic rule validation against schema
- [ ] Rule promotion workflow (session → permanent)
- [ ] Quality scoring for generated rules

### v0.8.0 — Optimization (Planned)
- [ ] Information-gain action selection
- [ ] Shortest-path to goal estimation
- [ ] Multi-goal games with priority ordering
- [ ] Caching for repeated patterns

---

## References

- Vitrac, O. (2025). *RAGIX: Retrieval-Augmented Generative Interactive eXecution Agent*. Adservio.
- *Reasoning Tutor Design Document*: `INTERPRET_TUTOR.md`

---

## License

MIT License — See repository root.
