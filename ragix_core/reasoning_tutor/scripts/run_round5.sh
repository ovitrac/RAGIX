#!/bin/bash
# ============================================================================
# LLM Olympics Round 5 — Large Model Extension
# ============================================================================
# Tests scaling hypothesis with 32B and 120B models on NVIDIA DIGITS hardware.
# Includes full ablation: WITH and WITHOUT scaffolding for each model.
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
mkdir -p "$OUTPUT_DIR/ablation_scaffold"

# Timestamps for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/round5_${TIMESTAMP}.log"

# Main output files
OUTPUT_WITH="$OUTPUT_DIR/olympics_round5_with_scaffold.jsonl"
OUTPUT_WITHOUT="$OUTPUT_DIR/olympics_round5_without_scaffold.jsonl"
CSV_OUTPUT="$OUTPUT_DIR/olympics_round5_features.csv"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Large models (Round 5 focus) — Order: gpt-oss first, then granite4
LARGE_MODELS=(
    "gpt-oss-safeguard:120b"    # 120B — Ceiling test (run first)
    "ibm/granite4:32b-a9b-h"    # 32B — Scaling test
)

# Reference models (R4 champions for comparison)
REFERENCE_MODELS=(
    "granite3.1-moe:3b"         # R4 champion (slim)
    "deepseek-r1:14b"           # R4 reference (mid) — NOW AVAILABLE
    "qwen2.5-coder:7b"          # Extension model (code specialist)
)

# Slim models (may still be downloading)
SLIM_MODELS=(
    "mistral:7b-instruct"
    "llama3.2:3b"
    "phi3:latest"
)

# Benchmark IDs (frozen from R1-R4)
# scored_mode.py expects IDs like "01", not full paths
BENCHMARKS=(
    "01"  # find_needle
    "02"  # count_lines
    "03"  # undecidable_claim
    "04"  # verification_chain
    "05"  # session_rules
    "06"  # memory_recall
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_model_available() {
    local model="$1"
    local model_base=$(echo "$model" | cut -d: -f1 | sed 's|/|.|g')
    if ollama list 2>/dev/null | grep -qi "$model_base"; then
        return 0
    else
        return 1
    fi
}

check_gpu_temp() {
    local temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo "0")
    echo "$temp"
}

wait_for_cooldown() {
    local threshold=75
    local temp=$(check_gpu_temp)
    while [ "$temp" -gt "$threshold" ] 2>/dev/null; do
        log "GPU temperature ${temp}°C > ${threshold}°C, waiting for cooldown..."
        sleep 30
        temp=$(check_gpu_temp)
    done
}

run_benchmark() {
    local model="$1"
    local benchmark_id="$2"
    local output_file="$3"
    local condition="$4"  # "scaffold" or "raw"

    local start_time=$(date +%s.%N)

    log "  Running: $model — B$benchmark_id ($condition)"

    # Run scored_mode with benchmark ID
    python3 benchmarks/scored_mode.py \
        --benchmark "$benchmark_id" \
        --models "$model" \
        --output "$output_file" \
        --max-turns 20 2>&1 | tee -a "$LOG_FILE"

    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)
    log "  Completed in ${elapsed}s"

    # Cooldown between runs
    wait_for_cooldown
    sleep 3
}

# ============================================================================
# HEADER
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           LLM OLYMPICS ROUND 5 — LARGE MODEL EXTENSION           ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Hardware: NVIDIA DIGITS (GB10 Blackwell, 120GB RAM)             ║"
echo "║  Focus: gpt-oss-safeguard:120b, ibm/granite4:32b-a9b-h           ║"
echo "║  Ablation: WITH and WITHOUT scaffolding                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

log "Round 5 started"
log "Output directory: $OUTPUT_DIR"
log "Log file: $LOG_FILE"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PRE-FLIGHT CHECKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check large models
echo ""
echo "Large models (required):"
AVAILABLE_LARGE=()
for model in "${LARGE_MODELS[@]}"; do
    if check_model_available "$model"; then
        echo "  ✓ $model — AVAILABLE"
        AVAILABLE_LARGE+=("$model")
    else
        echo "  ✗ $model — NOT FOUND (will skip)"
    fi
done

# Check reference models
echo ""
echo "Reference models (optional):"
AVAILABLE_REF=()
for model in "${REFERENCE_MODELS[@]}"; do
    if check_model_available "$model"; then
        echo "  ✓ $model — AVAILABLE"
        AVAILABLE_REF+=("$model")
    else
        echo "  ○ $model — downloading or not found"
    fi
done

# Check slim models (informational)
echo ""
echo "Slim models (may still be downloading):"
for model in "${SLIM_MODELS[@]}"; do
    if check_model_available "$model"; then
        echo "  ✓ $model — AVAILABLE"
    else
        echo "  ○ $model — downloading..."
    fi
done

# GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo "  GPU info unavailable"

# Abort if no large models available
if [ ${#AVAILABLE_LARGE[@]} -eq 0 ]; then
    echo ""
    echo "ERROR: No large models available. Please ensure models are downloaded:"
    echo "  ollama pull gpt-oss-safeguard:120b"
    echo "  ollama pull ibm/granite4:32b-a9b-h"
    exit 1
fi

echo ""
echo "Ready to run ${#AVAILABLE_LARGE[@]} large model(s) × 6 benchmarks × 2 conditions"
echo "Estimated time: $((${#AVAILABLE_LARGE[@]} * 6 * 2 * 5)) minutes (very rough estimate)"
echo ""

# Clear previous output files
> "$OUTPUT_WITH"
> "$OUTPUT_WITHOUT"

# ============================================================================
# PHASE 1: LARGE MODELS — WITH SCAFFOLDING
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: LARGE MODELS — WITH SCAFFOLDING                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

for model in "${AVAILABLE_LARGE[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "MODEL: $model — WITH SCAFFOLDING"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for benchmark_id in "${BENCHMARKS[@]}"; do
        run_benchmark "$model" "$benchmark_id" "$OUTPUT_WITH" "scaffold"
    done
done

# ============================================================================
# PHASE 2: LARGE MODELS — WITHOUT SCAFFOLDING
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: LARGE MODELS — WITHOUT SCAFFOLDING                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

for model in "${AVAILABLE_LARGE[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "MODEL: $model — WITHOUT SCAFFOLDING"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for benchmark_id in "${BENCHMARKS[@]}"; do
        run_benchmark "$model" "$benchmark_id" "$OUTPUT_WITHOUT" "raw"
    done
done

# ============================================================================
# PHASE 3: REFERENCE MODELS (if available)
# ============================================================================

if [ ${#AVAILABLE_REF[@]} -gt 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: REFERENCE MODELS (R4 Champions)                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    for model in "${AVAILABLE_REF[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "REFERENCE: $model"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        for benchmark_id in "${BENCHMARKS[@]}"; do
            run_benchmark "$model" "$benchmark_id" "$OUTPUT_WITH" "scaffold"
        done
    done
fi

# ============================================================================
# POST-PROCESSING
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ROUND 5 MAIN RUNS COMPLETE                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

log "Main runs complete"

# Merge results for analysis
COMBINED_OUTPUT="$OUTPUT_DIR/olympics_round5_combined.jsonl"
cat "$OUTPUT_WITH" "$OUTPUT_WITHOUT" > "$COMBINED_OUTPUT"
log "Combined results: $COMBINED_OUTPUT"

# Generate feature CSV (if analysis script exists)
if [ -f "analysis/extract_features.py" ]; then
    echo ""
    echo "Generating feature CSV..."
    python3 analysis/extract_features.py \
        --input "$COMBINED_OUTPUT" \
        --output "$CSV_OUTPUT" 2>&1 | tee -a "$LOG_FILE" || log "Feature extraction failed (non-critical)"
fi

# Compare with Round 4 (if comparison script and R4 data exist)
if [ -f "analysis/compare_rounds.py" ] && [ -f "results/olympics_round4_features.csv" ]; then
    echo ""
    echo "Generating Round 4 vs Round 5 comparison..."
    mkdir -p "$OUTPUT_DIR/comparison_r4_r5"
    python3 analysis/compare_rounds.py \
        --r1 "results/olympics_round4_features.csv" \
        --r2 "$CSV_OUTPUT" \
        --output "$OUTPUT_DIR/comparison_r4_r5/" \
        --r1-name "Round 4 (slim champions)" \
        --r2-name "Round 5 (large models)" 2>&1 | tee -a "$LOG_FILE" || log "Comparison failed (non-critical)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RESULTS SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Output files:"
echo "  • $OUTPUT_WITH (with scaffolding)"
echo "  • $OUTPUT_WITHOUT (without scaffolding)"
echo "  • $COMBINED_OUTPUT (merged)"
echo "  • $CSV_OUTPUT (features)"
echo "  • $LOG_FILE (full log)"
echo ""
echo "Next steps:"
echo "  1. Run hybrid experiment: ./scripts/run_round5_hybrid.sh"
echo "  2. Review comparison: cat $OUTPUT_DIR/comparison_r4_r5/comparison_report.md"
echo "  3. Update publication: docs/PUBLICATION_SUMMARY_v4.md"
echo ""

log "Round 5 finished"
echo "Done!"
