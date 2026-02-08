#!/bin/bash
# ============================================================================
# LLM Olympics Round 5 — Quick Test (Single Model, Single Benchmark)
# ============================================================================
# Use this script to verify setup before running full Round 5.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Date: 2026-02-03
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Defaults (can be overridden via arguments)
MODEL="${1:-gpt-oss-safeguard:120b}"
BENCHMARK="${2:-01}"  # Benchmark ID (01-06), not path

OUTPUT_DIR="results/round5/quick_test"
mkdir -p "$OUTPUT_DIR"

OUTPUT="$OUTPUT_DIR/quick_test_$(date +%Y%m%d_%H%M%S).jsonl"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           ROUND 5 QUICK TEST                                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL"
echo "Benchmark: $BENCHMARK"
echo "Output: $OUTPUT"
echo ""

# Check model availability
echo "Checking model availability..."
if ollama list | grep -qi "$(echo $MODEL | cut -d: -f1 | sed 's|/|.|g')"; then
    echo "  ✓ Model available"
else
    echo "  ✗ Model not found: $MODEL"
    echo ""
    echo "Available models:"
    ollama list
    exit 1
fi

# Check benchmark ID is valid
if [[ ! "$BENCHMARK" =~ ^0[1-6]$ ]]; then
    echo "  ⚠ Benchmark ID should be 01-06, got: $BENCHMARK"
fi
echo "  ✓ Benchmark ID: $BENCHMARK"

echo ""
echo "Starting benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

START_TIME=$(date +%s)

python3 benchmarks/scored_mode.py \
    --benchmark "$BENCHMARK" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --max-turns 10

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Quick test complete in ${ELAPSED}s"
echo "  Output: $OUTPUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show result summary
if [ -f "$OUTPUT" ]; then
    echo "Result summary:"
    tail -1 "$OUTPUT" | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    print(f\"  Model: {data.get('model', 'N/A')}\")
    print(f\"  Benchmark: {data.get('benchmark', 'N/A')}\")
    print(f\"  Success: {data.get('goal_achieved', 'N/A')}\")
    print(f\"  Score: {data.get('total_score', 'N/A')}\")
    print(f\"  Turns: {data.get('turns', 'N/A')}\")
except:
    print('  (Could not parse result)')
" 2>/dev/null || echo "  (Result parsing unavailable)"
fi

echo ""
echo "If this looks good, run the full suite:"
echo "  ./scripts/run_round5.sh"
