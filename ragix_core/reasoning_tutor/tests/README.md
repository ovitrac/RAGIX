# Test Suite for RAGIX Reasoning Tutor v0.3

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Date:** 2025-12-23
**Validated:** LLM Olympics 2025-12-23 (11 models × 6 benchmarks = 66 games)

---

## Overview

This test suite validates the **Meta-Cognitive Architecture** introduced in v0.3:

| Component | Module | Tests | Purpose |
|-----------|--------|-------|---------|
| **R1** | `failure_detector.py` | 30 | Detect stuck states (perseveration, disorientation, agnosia) |
| **R2** | `meta_cards.py` | 40 | Strategic interventions for wall-breaking |
| **R3** | `justification_protocol.py` | 25 | Fix metric bias (activity ≠ achievement) |
| **Semantic P1** | `semantic_intent.py` | 44 | Semantic intent tracking (converging vs wandering) |
| **Semantic P2** | `semantic_intent.py` | 15 | Error comprehension analysis |
| **Semantic P3** | `semantic_intent.py` | 15 | Semantic card relevance ranking |
| **Integration** | — | 15 | Full pipeline validation |

**Total: 184 tests** — All passing

---

## Running Tests

```bash
# From reasoning_tutor directory
cd ragix_core/reasoning_tutor

# Run all tests
python3 -m pytest tests/ -v

# Run specific component
python3 -m pytest tests/test_failure_detector.py -v
python3 -m pytest tests/test_meta_cards.py -v
python3 -m pytest tests/test_justification_protocol.py -v
python3 -m pytest tests/test_integration.py -v
python3 -m pytest tests/test_semantic_intent.py -v

# Run with coverage
python3 -m pytest tests/ --cov=. --cov-report=term-missing

# Use the test runner script
./tests/run_tests.sh
```

---

## Test Files

### `test_failure_detector.py` (R1)

Tests the FailureDetector class that implements meta-cognitive detection:

| Test Class | Coverage |
|------------|----------|
| `TestFailureDetectorInit` | Initialization, thresholds, defaults |
| `TestRepetitionLoopDetection` | Perseveration (syntactic aphasia) |
| `TestCircularPatternDetection` | Disorientation (strategic confusion) |
| `TestExplicitErrorDetection` | Agnosia (error recognition failure) |
| `TestProgressStallDetection` | Confabulation (verbose but unproductive) |
| `TestExhaustionDetection` | Card exhaustion |
| `TestTurnTracking` | Turn counting, history |
| `TestFailureContext` | Context data structure |
| `TestFatLLMInstruction` | Intervention prompt generation |
| `TestReset` | State reset |
| `TestOlympicsValidation` | Olympics pattern reproduction |

**Key Olympics Validation Tests:**
- `test_olympics_granite_b01_pattern` — Reproduces granite's B01 repetition failure
- `test_olympics_dolphin_pattern` — Reproduces dolphin's ls→cat→ls→cat cycle
- `test_olympics_phi3_cascade` — Reproduces phi3's 26-failure error cascade
- `test_olympics_llama32_confabulation` — Reproduces llama3.2's stall pattern

### `test_justification_protocol.py` (R3)

Tests the Justification Protocol that fixes the Metric Bias problem:

| Test Class | Coverage |
|------------|----------|
| `TestJustifiedAction` | Action structure, score multiplier |
| `TestJustificationQuality` | Quality enum (EXCELLENT → NONE) |
| `TestGoalProximity` | Proximity enum (DIRECT → IRRELEVANT) |
| `TestJustificationEvaluator` | Hypothesis evaluation, scoring |
| `TestMetricBiasFix` | llama3.2:3b anomaly fix |
| `TestResponseParsing` | LLM response parsing |
| `TestEvaluatorSummary` | Statistics aggregation |
| `TestPromptTemplate` | Prompt structure validation |
| `TestKeywordExtraction` | Keyword extraction helper |

**Key Metric Bias Tests:**
- `test_confabulation_penalty` — Verifies >50% score reduction for unjustified actions
- `test_justified_actions_preserved` — Ensures justified actions keep their score

### `test_meta_cards.py` (R2)

Tests the MetaCards system for strategic interventions:

| Test Class | Coverage |
|------------|----------|
| `TestMetaCardType` | Card type enum |
| `TestCardTemplates` | Template structure, content |
| `TestMetaCard` | Card creation, formatting |
| `TestMetaCardSelector` | Card selection logic |
| `TestCardEffectivenessTracking` | Success rate tracking |
| `TestSelectorStatistics` | Usage statistics |
| `TestContextEnrichment` | Context-specific hints |
| `TestFailureTypeMapping` | Failure → Card mapping |
| `TestOlympicsValidation` | Olympics pattern cards |

**Key Card Tests:**
- `test_granite_b01_card` — Card for granite's repetition pattern
- `test_dolphin_circular_card` — Card for dolphin's cycling
- `test_phi3_error_cascade_card` — Card for phi3's error cascade
- `test_llama32_stall_card` — Card for llama3.2's confabulation

### `test_semantic_intent.py` (Phase 1 Semantic Tool)

Tests the Semantic Intent Tracker that addresses the llama3.2:3b "Metric Bias" anomaly:

| Test Class | Coverage |
|------------|----------|
| `TestIntentCategory` | Intent enum (CONVERGING, STABLE, WANDERING, DIVERGING) |
| `TestIntentAnalysis` | Analysis dataclass, serialization |
| `TestEmbeddingProvider` | Embedding generation, caching, distance computation |
| `TestSemanticIntentTracker` | Tracker initialization, analysis, history |
| `TestIntentClassification` | Converging vs wandering pattern detection |
| `TestConfigIntegration` | Enable/disable via config, multiplier values |
| `TestComputeSemanticScore` | Score computation helper |
| `TestOlympicsPatternValidation` | deepseek vs llama3.2 pattern discrimination |
| `TestEdgeCases` | Empty actions, unicode, special characters |
| `TestTrajectoryAnalysis` | Trajectory trend detection |

**Key Olympics Pattern Tests:**
- `test_deepseek_converging_pattern` — Validates focused, converging actions score well
- `test_llama32_wandering_pattern` — Validates wandering at high distance is detected
- `test_deepseek_vs_llama_score_differential` — Confirms deepseek earns more than llama3.2
- `test_granite_mixed_pattern` — Validates recovery after card intervention

### Phase 2: Error Comprehension Tests

Tests for semantic error comprehension that addresses the phi3:latest error cascade pattern:

| Test Class | Coverage |
|------------|----------|
| `TestErrorComprehensionLevel` | Enum (FULL, PARTIAL, MINIMAL, NONE) |
| `TestErrorComprehensionAnalysis` | Analysis dataclass, serialization |
| `TestSemanticErrorComprehension` | Analyzer class, history, triggers |

**Key Tests:**
- `test_analyze_response` — Analyzes error-response pairs semantically
- `test_should_trigger_card` — Detects when to inject ERROR_ANALYSIS card
- `test_phi3_error_cascade_pattern` — Validates phi3 repeated-error detection

### Phase 3: Card Relevance Tests

Tests for semantic card selection that improves card targeting:

| Test Class | Coverage |
|------------|----------|
| `TestCardRelevanceScore` | Relevance score dataclass |
| `TestSemanticCardRelevance` | Card ranking, registration, selection |

**Key Tests:**
- `test_rank_cards` — Ranks cards by semantic relevance to failure context
- `test_repetition_failure_selects_escape_loop` — Validates correct card for repetition
- `test_error_failure_selects_error_analysis` — Validates correct card for errors

### Unified Semantic Analyzer Tests

Tests for the combined semantic analyzer (all 3 phases):

| Test Class | Coverage |
|------------|----------|
| `TestSemanticAnalyzer` | Unified interface, combined multipliers |
| `TestOlympicsPhase2Phase3` | Olympics pattern validation |

### `test_integration.py`

Tests the full pipeline: FailureDetector → MetaCards → JustificationProtocol

| Test Class | Coverage |
|------------|----------|
| `TestFailureToCardFlow` | Failure detection triggers card selection |
| `TestCardImpactOnJustification` | Cards improve justification quality |
| `TestFullPipelineScenarios` | Olympics scenarios end-to-end |
| `TestDeepseekPerfectRun` | Clean run (0 failures) |
| `TestCardEffectivenessTracking` | Cross-session tracking |
| `TestJustificationSummary` | Session-level statistics |
| `TestEndToEndScenarios` | Complete Failure→Card→Recovery→Success |

**Key Integration Tests:**
- `test_granite_recovery_scenario` — Full card rescue flow
- `test_phi3_failure_cascade` — Multiple failures, cards don't save
- `test_llama32_confabulation_detection` — Stall + score reduction
- `test_recovery_after_card_intervention` — End-to-end success

---

## Olympics Validation

Tests reproduce actual patterns from LLM Olympics 2025-12-23:

### Model Patterns Tested

| Model | Pattern | Test |
|-------|---------|------|
| deepseek-r1:14b | Clean run (0 failures) | `test_clean_run_pattern` |
| granite3.1-moe:3b | Repetition → Card rescue | `test_granite_recovery_scenario` |
| llama3.2:3b | Confabulation (high pts, low wins) | `test_llama32_confabulation_detection` |
| phi3:latest | Error cascade (26 failures) | `test_phi3_failure_cascade` |
| dolphin-mistral | Circular pattern | `test_dolphin_circular_card` |

### Key Validations

1. **Failure Rate ↔ Success Rate Correlation**
   - Zero failures → 100% wins (deepseek)
   - High failures → Low wins (phi3)

2. **Metric Bias Fix**
   - llama3.2 pattern: +4490 pts, 1/6 wins
   - With justification: ~75% score reduction

3. **Card Effectiveness**
   - Cards help break failure patterns
   - Some models (phi3) don't recover even with cards

---

## Test Dependencies

```
pytest>=7.0.0
pytest-cov>=4.0.0  # Optional, for coverage
```

---

## Coverage Targets

| Component | Target | Current |
|-----------|--------|---------|
| failure_detector.py | 90% | ~85% |
| meta_cards.py | 90% | ~90% |
| justification_protocol.py | 85% | ~80% |
| semantic_intent.py | 85% | ~85% |
| config.py | 80% | ~75% |
| Integration | 80% | ~75% |

---

## Adding New Tests

When adding tests for new features:

1. **Follow the pattern**: Each test class focuses on one aspect
2. **Use fixtures**: `@pytest.fixture` for common setup
3. **Validate against Olympics**: Add tests that reproduce real model behavior
4. **Document expectations**: Clear docstrings explaining what's tested

Example:
```python
def test_new_failure_pattern(self):
    """
    Detect [pattern name] based on Olympics observation.

    Model: [model name]
    Benchmark: [B0X]
    Expected: [FAILURE_TYPE] at turn [N]
    """
    fd = FailureDetector()
    # ... test implementation
```

---

*Generated by RAGIX Interpreter-Tutor Test Suite*
