#!/bin/bash
# ============================================================================
# LLM Olympics Round 4 — TRIZ + Kanban Strategic Advisor
# ============================================================================
#
# Round 4 introduces:
# - TRIZ Meta-Cards (Segmentation, Prior Action, Inversion)
# - Kanban WIP Management (model-specific limits)
# - Focus View (compressed DONE column for 3B models)
# - Mistral "Straightjacket" (WIP=1)
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Date: 2025-12-23
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Output file
OUTPUT="results/olympics_round4.jsonl"
CSV_OUTPUT="results/olympics_round4_features.csv"

# Models to benchmark (Phi-3 RETIRED due to terminal agnosia)
MODELS=(
    "deepseek-r1:14b"      # Champion (WIP=3, TRIZ off unless Turn>15)
    "llama3.2:3b"          # Rehabilitated (WIP=2, TRIZ on)
    "granite3.1-moe:3b"    # Stable specialist (WIP=3, TRIZ on)
    "mistral:latest"       # Stubborn (WIP=1 STRAIGHTJACKET, TRIZ on)
)

# Benchmarks (6 total)
BENCHMARKS=(
    "benchmarks/01_find_needle.yaml"
    "benchmarks/02_count_lines.yaml"
    "benchmarks/03_undecidable.yaml"
    "benchmarks/04_verification_chain.yaml"
    "benchmarks/05_session_rules.yaml"
    "benchmarks/06_memory_recall.yaml"
)

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           LLM OLYMPICS ROUND 4 — TRIZ + KANBAN                   ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Strategic Advisor Features:                                     ║"
echo "║  • TRIZ #1:  SEGMENT_TASK (Divide & Conquer)                    ║"
echo "║  • TRIZ #10: DEFINE_CRITERIA (Prior Action)                     ║"
echo "║  • TRIZ #13: LIST_INSTEAD (Inversion)                           ║"
echo "║  • Kanban WIP Limits (Mistral=1, Default=2)                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Models: ${MODELS[*]}"
echo "Benchmarks: ${#BENCHMARKS[@]}"
echo "Total games: $((${#MODELS[@]} * ${#BENCHMARKS[@]}))"
echo "Output: $OUTPUT"
echo ""

# Clear previous results
> "$OUTPUT"

# Run benchmarks
for model in "${MODELS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  MODEL: $model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for benchmark in "${BENCHMARKS[@]}"; do
        echo ""
        echo "  📋 Benchmark: $(basename "$benchmark" .yaml)"
        echo "  ───────────────────────────────────────────"

        python3 benchmarks/scored_mode.py \
            --benchmark "$benchmark" \
            --model "$model" \
            --output "$OUTPUT" \
            --max-turns 20

        # Brief pause between games
        sleep 2
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ROUND 4 COMPLETE                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Generate feature CSV
echo "Generating feature CSV..."
python3 analysis/extract_features.py \
    --input "$OUTPUT" \
    --output "$CSV_OUTPUT"

echo ""
echo "Results saved to:"
echo "  • $OUTPUT (raw JSONL)"
echo "  • $CSV_OUTPUT (feature CSV)"
echo ""

# Compare with Round 3
echo "Generating Round 3 vs Round 4 comparison..."
python3 analysis/compare_rounds.py \
    --r1 "results/olympics_round3_features.csv" \
    --r2 "$OUTPUT" \
    --output "results/comparison_r3_r4/" \
    --r1-name "Round 3" \
    --r2-name "Round 4"

echo ""
echo "Comparison report: results/comparison_r3_r4/comparison_report.md"
echo "Done!"
