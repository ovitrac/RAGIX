#!/bin/bash
# =============================================================================
# LLM Olympics Round 3 — Campaign Launch Script
# =============================================================================
#
# Round 3 implements the recommendations from Reviewer N°2:
#
# P0 FIXES (Mandatory):
#   1. deepseek <think> token stripping (strip_reasoning_tokens)
#   2. 2-Strike Rule for phi3/llama3.2 (model-specific thresholds)
#
# SEMANTIC LAYER (Enabled):
#   - Phase 1: Intent Tracker (wandering detection for llama3.2)
#   - Phase 2: Error Comprehension (phi3 "amnesia" fix)
#   - Phase 3: Card Relevance (optimal card selection)
#
# EXPECTED OUTCOMES:
#   - deepseek-r1:14b: 33% → 100% (after token fix)
#   - granite3.1-moe:3b: Hold at 100%
#   - mistral:latest: Hold at 100%
#   - phi3:latest: 17% → 60-70%
#   - llama3.2:3b: 33% → 50-60%
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Date: 2025-12-23
# =============================================================================

set -e

# Change to the reasoning_tutor directory
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║           LLM OLYMPICS ROUND 3 — META-COGNITION VALIDATION                   ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Date: $(date '+%Y-%m-%d %H:%M')                                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Show current configuration
echo "📊 Configuration Status:"
python3 -c "from config import print_config_summary; print_config_summary()"
echo ""

# Models to test
MODELS="deepseek-r1:14b,granite3.1-moe:3b,llama3.2:3b,mistral:latest,phi3:latest"
MAX_TURNS=8
OUTPUT="results/olympics_round3.jsonl"

echo "🎯 Round 3 Parameters:"
echo "   Models: $MODELS"
echo "   Max turns: $MAX_TURNS"
echo "   Output: $OUTPUT"
echo ""

# Show Round 3 fixes active
echo "🔧 Round 3 Fixes Active:"
echo "   ✓ deepseek <think> token stripping"
echo "   ✓ 2-Strike Rule for phi3:latest (threshold=2)"
echo "   ✓ 2-Strike Rule for llama3.2:3b (threshold=2)"
echo "   ✓ Semantic Intent Tracker (Phase 1)"
echo "   ✓ Error Comprehension (Phase 2)"
echo "   ✓ Card Relevance (Phase 3)"
echo ""

# Confirm before launching
read -p "🚀 Launch Round 3? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                          ROUND 3 IN PROGRESS..."
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Run the benchmark campaign
START_TIME=$(date +%s)

python3 -m benchmarks.scored_mode \
    --all \
    --models "$MODELS" \
    --max-turns $MAX_TURNS \
    --output "$OUTPUT"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                          ROUND 3 COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "⏱️  Duration: ${DURATION}s"
echo "📁 Output: $OUTPUT"
echo ""

# Extract features
echo "📊 Extracting features..."
python3 analysis/extract_features.py "$OUTPUT" --output results/olympics_round3_features.csv

# Generate visualizations
echo "📈 Generating visualizations..."
python3 analysis/visualize.py results/olympics_round3_features.csv --output results/round3/

# Compare with Round 2
echo "🔄 Comparing with Round 2..."
python3 analysis/compare_rounds.py \
    --r1 results/olympics_round2_features.csv \
    --r2 "$OUTPUT" \
    --output results/comparison_r2_r3/

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                      ROUND 3 ANALYSIS COMPLETE                               ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Results:      results/olympics_round3.jsonl                                 ║"
echo "║  Features:     results/olympics_round3_features.csv                          ║"
echo "║  Visualizations: results/round3/                                             ║"
echo "║  Comparison:   results/comparison_r2_r3/                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Show quick summary
echo "📋 Quick Summary:"
cat results/round3/report.md | head -30
