# Round 5 Extension Plan: Large Model Evaluation

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Date:** 2026-02-03
**Status:** Approved — Ready for Execution

---

## 1. Executive Summary

This document defines the extension of the LLM Olympics study (Rounds 1-4) to evaluate **large-scale models** on the new NVIDIA DIGITS hardware. The goal is to determine whether meta-cognitive scaffolding benefits diminish, persist, or transform at the 32B-120B parameter scale.

**Key Questions:**
1. Does the Minimum Viable Threshold (MVT) have a **ceiling** where scaffolding becomes unnecessary?
2. What is the **efficiency crossover point** where raw capability beats scaffolded slim models?
3. Do large models exhibit the **same pathologies** or new failure modes?
4. Can large models serve as **rule generators** for slim LLM execution (v0.3.0 preview)?

---

## 2. Context

### 2.1 Previous Study (Rounds 1-4)

| Round | Intervention | Key Finding |
|-------|--------------|-------------|
| R1 | Baseline (Safety Net) | High variance, interface failures |
| R2 | Token stripping, model reduction | Symbol grounding fixed |
| R3 | Intent Tracker, 2-Strike Rule | Reward hacking eliminated |
| R4 | TRIZ + Kanban Strategic Advisor | **granite3.1-moe:3b = 100%** |

**Principal Result:** granite3.1-moe:3b (3B params) matched deepseek-r1:14b (14B params) at **8.5× lower inference latency** (1.5s vs. 12.8s per turn).

### 2.2 New Hardware Environment

| Property | Value |
|----------|-------|
| **Platform** | Dell Pro Max with NVIDIA GB10 (DIGITS) |
| **Architecture** | ARM64 (Cortex-X925 + Cortex-A725) |
| **GPU** | NVIDIA GB10 (Blackwell), CUDA 13.0 |
| **Memory** | 120 GB unified RAM |
| **Storage** | 3.6 TB NVMe |

This configuration enables local inference of models up to ~120B parameters.

### 2.3 New Models Available

| Model | Parameters | Size | Rationale |
|-------|------------|------|-----------|
| `ibm/granite4:32b-a9b-h` | 32B | 19 GB | Next-gen Granite, tests scaling hypothesis |
| `gpt-oss-safeguard:120b` | 120B | 65 GB | "Fat LLM" reference, tests ceiling hypothesis |

**Comparison with R4 models:**

| Model | Parameters | R4 Win Rate | Expected R5 Role |
|-------|------------|-------------|------------------|
| granite3.1-moe:3b | 3B | 100% | Baseline champion |
| deepseek-r1:14b | 14B | 100% | Previous ceiling |
| ibm/granite4:32b-a9b-h | 32B | ? | Scaling test |
| gpt-oss-safeguard:120b | 120B | ? | Ceiling test |

---

## 3. Scientific Hypotheses

### H1: MVT Ceiling Hypothesis

> **Large models (≥32B) satisfy MIC without scaffolding, making the Strategic Advisor unnecessary.**

**Prediction:** If true, granite4:32b and gpt-oss:120b achieve 100% win rate **without** TRIZ/Kanban interventions.

**Test:** Run both with `--no-scaffold` flag and compare to scaffolded runs.

### H2: Efficiency Crossover Hypothesis

> **There exists a parameter count N where raw model capability exceeds scaffolded slim model efficiency.**

**Prediction:** Define efficiency as `E = WinRate / AvgLatency`. At some scale, `E(large, raw) > E(slim, scaffolded)`.

**Test:** Compute efficiency ratio for all models; identify crossover point if it exists.

### H3: Pathology Persistence Hypothesis

> **Large models exhibit the same pathology classes but at lower incidence rates.**

**Prediction:** Reward Hacking, Policy Overfitting, Format Interference should decrease with scale; Feedback Insensitivity (terminal) should be absent.

**Test:** Apply diagnostic probes; classify failures using R4 taxonomy.

### H4: Fat-Tutor-Slim-Player Hypothesis (v0.3.0 Preview)

> **A 120B model can generate high-quality session rules that improve 3B model performance.**

**Prediction:** gpt-oss-safeguard:120b generates rules → granite3.1-moe:3b uses them → performance ≥ R4 with less scaffolding.

**Test:** Hybrid experiment (Phase 4).

---

## 4. Experimental Design

### 4.1 Benchmark Suite (Frozen)

The same 6 benchmarks from Rounds 1-4 are used without modification:

| ID | Name | Capability | Difficulty |
|----|------|------------|------------|
| B01 | Find Needle | File search | Low |
| B02 | Count Lines | Enumeration | Low |
| B03 | Undecidable Claim | Truth/Dare | High |
| B04 | Verification Chain | Multi-step proof | High |
| B05 | Session Rules | Rule generation | High |
| B06 | Memory Recall | Context retention | High |

**Invariance Statement:** Benchmark YAML files are version-controlled and will not be modified for Round 5.

### 4.2 Execution Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.0 | Deterministic sampling (reproducibility) |
| `max-turns` | 20 | Same as R4 |
| `scorer` | scored_mode.py | Same scoring system |
| `output_format` | JSONL + CSV | Compatible with analysis pipeline |

### 4.3 Experimental Phases

```
Phase 1: Baseline Extension
    │ Run new models on same benchmarks
    │ Output: olympics_round5_large.jsonl
    ↓
Phase 2: Comparison Analysis
    │ Compare R4 champions vs. R5 large models
    │ Output: comparison_r4_r5/
    ↓
Phase 3: Ablation — Scaffolding Benefit
    │ Run large models with/without Strategic Advisor
    │ Output: round5/ablation_scaffold/
    ↓
Phase 4: Hybrid Experiment (Optional)
    │ gpt-oss:120b generates rules → granite:3b executes
    │ Output: hybrid_experiment/
```

---

## 5. Implementation Plan

### 5.1 Directory Structure

```
ragix_core/reasoning_tutor/
├── results/                      # Existing (FROZEN)
│   ├── olympics_round[1-4]*.jsonl
│   ├── olympics_round[1-4]*.csv
│   └── comparison_r*/
│
├── results/round5/               # NEW
│   ├── olympics_round5_large.jsonl
│   ├── olympics_round5_features.csv
│   ├── ablation_scaffold/
│   │   ├── with_scaffold.jsonl
│   │   └── without_scaffold.jsonl
│   └── comparison_r4_r5/
│
├── results/hybrid_experiment/    # NEW (Phase 4)
│   ├── generated_rules/
│   └── slim_execution/
│
├── docs/                         # NEW
│   ├── ROUND5_EXTENSION_PLAN.md  # This file
│   └── ROUND5_RESULTS.md         # Post-analysis
│
└── scripts/                      # Execution scripts
    ├── run_round5.sh             # Phase 1
    ├── run_round5_ablation.sh    # Phase 3
    └── run_round5_hybrid.sh      # Phase 4
```

### 5.2 Script: run_round5.sh

```bash
#!/bin/bash
# ============================================================================
# LLM Olympics Round 5 — Large Model Extension
# ============================================================================
# Tests scaling hypothesis with 32B and 120B models on NVIDIA DIGITS hardware.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Date: 2026-02-03
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Output directory
OUTPUT_DIR="results/round5"
mkdir -p "$OUTPUT_DIR"

OUTPUT="$OUTPUT_DIR/olympics_round5_large.jsonl"
CSV_OUTPUT="$OUTPUT_DIR/olympics_round5_features.csv"

# New large models
MODELS=(
    "ibm/granite4:32b-a9b-h"    # 32B — Scaling test
    "gpt-oss-safeguard:120b"    # 120B — Ceiling test
)

# Include R4 champions for direct comparison
REFERENCE_MODELS=(
    "granite3.1-moe:3b"         # R4 champion (slim)
    "deepseek-r1:14b"           # R4 reference (mid)
)

# Benchmarks (frozen from R1-R4)
BENCHMARKS=(
    "benchmarks/01_find_needle.yaml"
    "benchmarks/02_count_lines.yaml"
    "benchmarks/03_undecidable.yaml"
    "benchmarks/04_verification_chain.yaml"
    "benchmarks/05_session_rules.yaml"
    "benchmarks/06_memory_recall.yaml"
)

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           LLM OLYMPICS ROUND 5 — LARGE MODEL EXTENSION           ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Hardware: NVIDIA DIGITS (GB10 Blackwell, 120GB RAM)             ║"
echo "║  New Models: granite4:32b, gpt-oss-safeguard:120b                ║"
echo "║  Hypothesis: MVT ceiling, efficiency crossover                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Large models: ${MODELS[*]}"
echo "Reference models: ${REFERENCE_MODELS[*]}"
echo "Benchmarks: ${#BENCHMARKS[@]}"
echo "Total games: $(( (${#MODELS[@]} + ${#REFERENCE_MODELS[@]}) * ${#BENCHMARKS[@]} ))"
echo "Output: $OUTPUT"
echo ""

# Verify models are available
echo "Checking model availability..."
for model in "${MODELS[@]}" "${REFERENCE_MODELS[@]}"; do
    if ! ollama list | grep -q "$(echo $model | cut -d: -f1)"; then
        echo "WARNING: Model $model may not be available"
    fi
done
echo ""

# Clear previous results
> "$OUTPUT"

# Run large models first (most interesting)
for model in "${MODELS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  MODEL: $model (LARGE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for benchmark in "${BENCHMARKS[@]}"; do
        echo ""
        echo "  Benchmark: $(basename "$benchmark" .yaml)"
        echo "  ───────────────────────────────────────────"

        # Record start time for latency measurement
        START_TIME=$(date +%s.%N)

        python3 benchmarks/scored_mode.py \
            --benchmark "$benchmark" \
            --model "$model" \
            --output "$OUTPUT" \
            --max-turns 20

        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
        echo "  Elapsed: ${ELAPSED}s"

        # Longer pause for large models (GPU cooldown)
        sleep 5
    done
done

# Run reference models for comparison
for model in "${REFERENCE_MODELS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  MODEL: $model (REFERENCE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for benchmark in "${BENCHMARKS[@]}"; do
        echo ""
        echo "  Benchmark: $(basename "$benchmark" .yaml)"

        python3 benchmarks/scored_mode.py \
            --benchmark "$benchmark" \
            --model "$model" \
            --output "$OUTPUT" \
            --max-turns 20

        sleep 2
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ROUND 5 COMPLETE                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Generate feature CSV
echo "Generating feature CSV..."
python3 analysis/extract_features.py \
    --input "$OUTPUT" \
    --output "$CSV_OUTPUT"

# Compare with Round 4
echo "Generating Round 4 vs Round 5 comparison..."
python3 analysis/compare_rounds.py \
    --r1 "results/olympics_round4_features.csv" \
    --r2 "$CSV_OUTPUT" \
    --output "$OUTPUT_DIR/comparison_r4_r5/" \
    --r1-name "Round 4 (slim champions)" \
    --r2-name "Round 5 (large models)"

echo ""
echo "Results saved to:"
echo "  • $OUTPUT (raw JSONL)"
echo "  • $CSV_OUTPUT (feature CSV)"
echo "  • $OUTPUT_DIR/comparison_r4_r5/ (comparison report)"
echo ""
echo "Done!"
```

### 5.3 Script: run_round5_ablation.sh

```bash
#!/bin/bash
# ============================================================================
# Round 5 Ablation: Scaffolding Benefit for Large Models
# ============================================================================
# Tests whether large models benefit from Strategic Advisor or perform
# better without scaffolding intervention.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Date: 2026-02-03
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

OUTPUT_DIR="results/round5/ablation_scaffold"
mkdir -p "$OUTPUT_DIR"

MODELS=(
    "ibm/granite4:32b-a9b-h"
    "gpt-oss-safeguard:120b"
)

BENCHMARKS=(
    "benchmarks/01_find_needle.yaml"
    "benchmarks/02_count_lines.yaml"
    "benchmarks/03_undecidable.yaml"
    "benchmarks/04_verification_chain.yaml"
    "benchmarks/05_session_rules.yaml"
    "benchmarks/06_memory_recall.yaml"
)

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       ROUND 5 ABLATION: SCAFFOLDING BENEFIT FOR LARGE MODELS     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Condition 1: WITH scaffolding (Strategic Advisor enabled)
echo "=== Condition 1: WITH Strategic Advisor ==="
OUTPUT_WITH="$OUTPUT_DIR/with_scaffold.jsonl"
> "$OUTPUT_WITH"

for model in "${MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        echo "  $model — $(basename "$benchmark" .yaml) — SCAFFOLDED"
        python3 benchmarks/scored_mode.py \
            --benchmark "$benchmark" \
            --model "$model" \
            --output "$OUTPUT_WITH" \
            --max-turns 20 \
            --enable-triz \
            --enable-kanban
        sleep 3
    done
done

# Condition 2: WITHOUT scaffolding (raw model capability)
echo ""
echo "=== Condition 2: WITHOUT Strategic Advisor ==="
OUTPUT_WITHOUT="$OUTPUT_DIR/without_scaffold.jsonl"
> "$OUTPUT_WITHOUT"

for model in "${MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        echo "  $model — $(basename "$benchmark" .yaml) — RAW"
        python3 benchmarks/scored_mode.py \
            --benchmark "$benchmark" \
            --model "$model" \
            --output "$OUTPUT_WITHOUT" \
            --max-turns 20 \
            --disable-triz \
            --disable-kanban
        sleep 3
    done
done

echo ""
echo "Ablation complete. Compare results:"
echo "  • $OUTPUT_WITH"
echo "  • $OUTPUT_WITHOUT"
```

---

## 6. Analysis Plan

### 6.1 Primary Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Win Rate** | Wins / 6 benchmarks | Task completion ability |
| **Avg Latency** | Mean time per turn (s) | Inference cost |
| **Efficiency** | WinRate / AvgLatency | Cost-adjusted performance |
| **Failure Rate** | Detected failures / total actions | Scaffolding load |
| **Recovery Rate** | Recovered / Detected | Resilience |

### 6.2 Comparison Matrix

```
                    ┌─────────────┬─────────────┬─────────────┬─────────────┐
                    │ granite:3b  │ deepseek:14b│ granite4:32b│ gpt-oss:120b│
┌───────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Parameters        │     3B      │     14B     │     32B     │    120B     │
│ Win Rate (R4)     │   100%      │    100%     │      ?      │      ?      │
│ Win Rate (R5)     │     ?       │      ?      │      ?      │      ?      │
│ Latency/turn      │    1.5s     │    12.8s    │   ~15-30s?  │   ~60-120s? │
│ Efficiency        │    0.67     │    0.08     │      ?      │      ?      │
│ Scaffold benefit? │    YES      │    MINIMAL  │      ?      │      ?      │
└───────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

### 6.3 Pathology Analysis

For each large model, classify failures using the R4 taxonomy:

| Pathology | Detection Method | Expected for Large Models |
|-----------|------------------|---------------------------|
| Symbol Grounding Failure | Parse error rate | LOW (better syntax) |
| Reward Hacking | High action, low goal | LOW (better planning) |
| Policy Overfitting | Repeated strategies | UNKNOWN |
| Format Interference | Markdown in output | MEDIUM (chat training) |
| Feedback Insensitivity | <10% correction rate | ABSENT (above MVT) |

### 6.4 Visualization Outputs

1. **Scaling curve**: Win rate vs. parameter count (log scale)
2. **Efficiency frontier**: Win rate vs. latency (Pareto plot)
3. **Pathology heatmap**: Model × Pathology incidence
4. **Ablation delta**: Scaffolded vs. raw performance per model

---

## 7. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model OOM | Low | High | Monitor GPU memory; reduce batch size |
| Slow inference | High | Medium | Budget extra time; run overnight |
| Results overwrite | Low | Critical | Separate `round5/` directory |
| Benchmark drift | Low | High | YAML files frozen, version-controlled |
| Hardware thermal | Medium | Medium | Add sleep between runs; monitor temp |

### 7.1 Pre-Flight Checklist

```bash
# Before running Round 5:

# 1. Verify models are pulled
ollama list | grep -E "granite4|gpt-oss"

# 2. Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# 3. Check available memory
free -h

# 4. Verify benchmark files unchanged
git diff --name-only benchmarks/*.yaml  # Should be empty

# 5. Create output directory
mkdir -p results/round5

# 6. Backup existing results (optional)
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/olympics_round*.jsonl
```

---

## 8. Timeline

| Phase | Duration (est.) | Dependency |
|-------|-----------------|------------|
| Phase 1: Baseline | 4-6 hours | Models available |
| Phase 2: Comparison | 30 min | Phase 1 complete |
| Phase 3: Ablation | 4-6 hours | Phase 1 insights |
| Phase 4: Hybrid | 2-3 hours | Optional |
| Documentation | 2 hours | All phases |

**Total estimated time:** 12-18 hours (can run overnight)

---

## 9. Success Criteria

### 9.1 Minimum Deliverables

- [ ] `olympics_round5_large.jsonl` with complete runs for both large models
- [ ] `comparison_r4_r5/` with comparison report
- [ ] Updated PUBLICATION_SUMMARY with Round 5 results

### 9.2 Extended Deliverables (if time permits)

- [ ] Ablation study: scaffolding benefit for large models
- [ ] Diagnostic probe results for any failures
- [ ] Hybrid experiment: fat-tutor-slim-player

### 9.3 Publication Integration

If results are significant, integrate into PUBLICATION_SUMMARY_v4.md:

- **Section 2.3**: Add granite4:32b and gpt-oss:120b to model selection
- **Section 3**: Add Round 5 results table
- **Section 4**: Update pathology taxonomy if new patterns emerge
- **Section 6**: Revise conclusions on scaling hypothesis

---

## 10. Decision Points — CONFIRMED

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Model priority** | Start with `gpt-oss-safeguard:120b`, then `granite4:32b` | Test ceiling hypothesis first |
| **Scaffolding strategy** | Test **both** with and without | Full ablation required |
| **Hybrid experiment** | **Include** in Round 5 | Fat-tutor-slim-player is key v0.3.0 preview |
| **Publication update** | Create **PUBLICATION_SUMMARY_v4.md** | Extend with Round 5 results |

### Execution Order

```
1. gpt-oss-safeguard:120b — WITH scaffolding (all 6 benchmarks)
2. gpt-oss-safeguard:120b — WITHOUT scaffolding (all 6 benchmarks)
3. ibm/granite4:32b-a9b-h — WITH scaffolding (all 6 benchmarks)
4. ibm/granite4:32b-a9b-h — WITHOUT scaffolding (all 6 benchmarks)
5. Reference: granite3.1-moe:3b, deepseek-r1:14b (verification run)
6. Hybrid experiment: gpt-oss:120b generates rules → granite:3b executes
```

**Total runs:** 4 models × 6 benchmarks × 2 conditions = **48 benchmark sessions** (+ hybrid)

---

## Appendix A: Model Specifications

### ibm/granite4:32b-a9b-h

| Property | Value |
|----------|-------|
| Parameters | 32B |
| Architecture | Dense (?) |
| Context window | 8192 tokens (est.) |
| Quantization | Unknown |
| Warmup time | ~30s |
| Inference timeout | 60s |
| Output format | **Clean** — direct command output |

### gpt-oss-safeguard:120b

| Property | Value |
|----------|-------|
| Parameters | 120B |
| Architecture | Dense |
| Context window | 8192+ tokens (est.) |
| Quantization | Mixed precision (65GB) |
| Warmup time | ~90s |
| Inference timeout | 120s |
| Output format | **Thinking pattern** — requires stripping |

### Output Format Analysis (2026-02-03)

Testing revealed different output patterns requiring format normalization:

| Model | Pattern | Example | Handling |
|-------|---------|---------|----------|
| `gpt-oss-safeguard:120b` | `Thinking...` reasoning block | `Thinking...\n<reasoning>\n...done thinking.\n<command>` | Strip via regex |
| `ibm/granite4:32b-a9b-h` | Clean output | `grep -l "KEY" {a,b,c}` | No stripping needed |
| `deepseek-r1:14b` | `<think>` XML tags | `<think>...</think><command>` | Already handled (R2) |
| `mistral:7b-instruct` | Markdown fences | `` ```bash\n<cmd>\n``` `` | Strip fences |

**Implemented fix:** Extended `strip_reasoning_tokens()` in `scored_mode.py` to handle:
- `Thinking...` / `...done thinking.` preambles
- Markdown code fences (`` ```bash `` / `` ``` ``)
- Inline backticks

---

## Appendix B: References

1. PUBLICATION_SUMMARY_v3.md — Round 1-4 methodology and results
2. run_round4.sh — Execution script template
3. README.md — Interpreter-Tutor architecture
4. ~/Desktop/host_description.md — NVIDIA DIGITS hardware specs

---

*RAGIX Interpreter-Tutor Extension Study*
*2026-02-03*
