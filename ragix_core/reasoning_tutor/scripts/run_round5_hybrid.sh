#!/bin/bash
# ============================================================================
# LLM Olympics Round 5 — Hybrid Experiment: Fat-Tutor-Slim-Player
# ============================================================================
# Tests whether a large model (120B) can generate session rules that
# improve slim model (3B) performance — previewing v0.3.0 architecture.
#
# Workflow:
#   1. gpt-oss-safeguard:120b analyzes benchmark and generates rules
#   2. Rules are saved as session rules (YAML)
#   3. granite3.1-moe:3b executes benchmark with generated rules
#   4. Compare: slim+generated-rules vs slim-alone vs fat-alone
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Date: 2026-02-03
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Output directory
OUTPUT_DIR="results/round5/hybrid_experiment"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/generated_rules"
mkdir -p "$OUTPUT_DIR/slim_execution"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/hybrid_${TIMESTAMP}.log"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

FAT_MODEL="gpt-oss-safeguard:120b"      # Rule generator (Tutor)
SLIM_MODEL="granite3.1-moe:3b"          # Executor (Player)
BACKUP_FAT="ibm/granite4:32b-a9b-h"     # Fallback if 120B unavailable

# Benchmarks for hybrid testing (start with subset)
HYBRID_BENCHMARKS=(
    "benchmarks/03_undecidable.yaml"        # High difficulty — benefits from rules
    "benchmarks/04_verification_chain.yaml" # Multi-step — needs strategy
    "benchmarks/05_session_rules.yaml"      # Meta: rule generation benchmark
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

# Generate session rules using fat model
generate_rules() {
    local benchmark="$1"
    local output_rules="$2"
    local bench_name=$(basename "$benchmark" .yaml)

    log "Generating rules for $bench_name using $FAT_MODEL..."

    # Read benchmark to understand the task
    local task_prompt=$(python3 -c "
import yaml
with open('$benchmark') as f:
    b = yaml.safe_load(f)
print(b.get('task', {}).get('prompt', 'Unknown task'))
" 2>/dev/null || echo "Analyze and solve this benchmark")

    # Prompt for rule generation
    local rule_gen_prompt="You are an expert at creating inference rules for shell-based reasoning tasks.

TASK: $task_prompt

Generate 3-5 YAML rules that would help a smaller LLM (3B parameters) solve this task.
Each rule should follow this format:

\`\`\`yaml
- id: R_descriptive_name
  soundness: sound
  description: \"What this rule detects\"
  match:
    - obs.tool: {eq: \"bash\"}
    - obs.command: {matches: \"pattern\"}
    - obs.rc: {eq: 0}
  extract:
    variable_name:
      regex: \"capture_pattern\"
      from: obs.stdout
  conclude:
    truth:
      text: \"What we can conclude\"
      kind: existence|count|property
      scope: \"{variable_name}\"
\`\`\`

Focus on rules that:
1. Detect successful command patterns
2. Extract key information from output
3. Help build evidence chains

Output ONLY valid YAML, no explanation."

    # Call fat model to generate rules
    local response=$(ollama run "$FAT_MODEL" "$rule_gen_prompt" 2>/dev/null)

    # Extract YAML from response (handle markdown fences)
    echo "$response" | sed -n '/^```yaml/,/^```/p' | sed '1d;$d' > "$output_rules"

    # If no fenced block, try raw output
    if [ ! -s "$output_rules" ]; then
        echo "$response" | grep -A1000 "^- id:" > "$output_rules" 2>/dev/null || true
    fi

    # Validate generated rules
    if [ -s "$output_rules" ]; then
        local rule_count=$(grep -c "^- id:" "$output_rules" 2>/dev/null || echo "0")
        log "  Generated $rule_count rules → $output_rules"
        return 0
    else
        log "  WARNING: No valid rules generated"
        return 1
    fi
}

# Run benchmark with slim model using generated rules
run_with_rules() {
    local benchmark="$1"
    local rules_file="$2"
    local output_file="$3"
    local bench_name=$(basename "$benchmark" .yaml)

    log "Executing $bench_name with $SLIM_MODEL + generated rules..."

    # Check if scored_mode supports session rules injection
    if python3 benchmarks/scored_mode.py --help 2>&1 | grep -q "session-rules"; then
        python3 benchmarks/scored_mode.py \
            --benchmark "$benchmark" \
            --model "$SLIM_MODEL" \
            --output "$output_file" \
            --max-turns 20 \
            --session-rules "$rules_file" 2>&1 | tee -a "$LOG_FILE"
    else
        # Fallback: copy rules to adhoc directory and run normally
        mkdir -p "rules/adhoc"
        cp "$rules_file" "rules/adhoc/$(basename "$rules_file")"
        python3 benchmarks/scored_mode.py \
            --benchmark "$benchmark" \
            --model "$SLIM_MODEL" \
            --output "$output_file" \
            --max-turns 20 2>&1 | tee -a "$LOG_FILE"
    fi
}

# Run benchmark with slim model alone (baseline)
run_slim_alone() {
    local benchmark="$1"
    local output_file="$2"
    local bench_name=$(basename "$benchmark" .yaml)

    log "Executing $bench_name with $SLIM_MODEL alone (baseline)..."

    python3 benchmarks/scored_mode.py \
        --benchmark "$benchmark" \
        --model "$SLIM_MODEL" \
        --output "$output_file" \
        --max-turns 20 2>&1 | tee -a "$LOG_FILE"
}

# ============================================================================
# HEADER
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     ROUND 5 HYBRID EXPERIMENT: FAT-TUTOR-SLIM-PLAYER             ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Fat Model (Tutor):  $FAT_MODEL                       ║"
echo "║  Slim Model (Player): $SLIM_MODEL                       ║"
echo "║  Workflow: Fat generates rules → Slim executes with rules        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

log "Hybrid experiment started"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo "Checking model availability..."

# Check fat model
if ! check_model_available "$FAT_MODEL"; then
    log "WARNING: $FAT_MODEL not available, trying fallback..."
    if check_model_available "$BACKUP_FAT"; then
        FAT_MODEL="$BACKUP_FAT"
        log "Using fallback: $FAT_MODEL"
    else
        log "ERROR: No fat model available"
        exit 1
    fi
fi
echo "  ✓ Fat model: $FAT_MODEL"

# Check slim model
if ! check_model_available "$SLIM_MODEL"; then
    log "ERROR: Slim model $SLIM_MODEL not available"
    exit 1
fi
echo "  ✓ Slim model: $SLIM_MODEL"

echo ""

# ============================================================================
# PHASE 1: GENERATE RULES WITH FAT MODEL
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: RULE GENERATION (Fat Model)                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

for benchmark in "${HYBRID_BENCHMARKS[@]}"; do
    bench_name=$(basename "$benchmark" .yaml)
    rules_file="$OUTPUT_DIR/generated_rules/${bench_name}_rules.yaml"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Benchmark: $bench_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    generate_rules "$benchmark" "$rules_file" || true

    sleep 5  # Cooldown between fat model calls
done

# ============================================================================
# PHASE 2: EXECUTE WITH SLIM MODEL + GENERATED RULES
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: EXECUTION (Slim Model + Generated Rules)               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

OUTPUT_HYBRID="$OUTPUT_DIR/slim_execution/hybrid_results.jsonl"
> "$OUTPUT_HYBRID"

for benchmark in "${HYBRID_BENCHMARKS[@]}"; do
    bench_name=$(basename "$benchmark" .yaml)
    rules_file="$OUTPUT_DIR/generated_rules/${bench_name}_rules.yaml"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Benchmark: $bench_name (with generated rules)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -s "$rules_file" ]; then
        run_with_rules "$benchmark" "$rules_file" "$OUTPUT_HYBRID"
    else
        log "  Skipping (no rules generated)"
    fi

    sleep 2
done

# ============================================================================
# PHASE 3: BASELINE (Slim Model Alone)
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: BASELINE (Slim Model Alone)                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

OUTPUT_BASELINE="$OUTPUT_DIR/slim_execution/baseline_results.jsonl"
> "$OUTPUT_BASELINE"

for benchmark in "${HYBRID_BENCHMARKS[@]}"; do
    bench_name=$(basename "$benchmark" .yaml)

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Benchmark: $bench_name (slim alone)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    run_slim_alone "$benchmark" "$OUTPUT_BASELINE"

    sleep 2
done

# ============================================================================
# ANALYSIS
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    HYBRID EXPERIMENT COMPLETE                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

log "Hybrid experiment complete"

# Generate comparison report
REPORT_FILE="$OUTPUT_DIR/HYBRID_EXPERIMENT_REPORT.md"
cat > "$REPORT_FILE" << 'REPORT_HEADER'
# Round 5 Hybrid Experiment Report

**Fat Model (Tutor):** gpt-oss-safeguard:120b
**Slim Model (Player):** granite3.1-moe:3b
**Date:** $(date '+%Y-%m-%d %H:%M')

## Hypothesis

> A large model (120B) can generate session rules that improve slim model (3B) performance,
> enabling "fat-tutor-slim-player" collaboration.

## Results

| Benchmark | Slim Alone | Slim + Rules | Δ |
|-----------|------------|--------------|---|
REPORT_HEADER

# TODO: Add automated result extraction here
echo "| (results pending analysis) | | | |" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'REPORT_FOOTER'

## Generated Rules

See `generated_rules/` directory for YAML rule files.

## Observations

(To be filled after analysis)

## Conclusion

(To be filled after analysis)

---
*RAGIX Round 5 Hybrid Experiment*
REPORT_FOOTER

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Output files:"
echo "  • $OUTPUT_HYBRID (slim + rules)"
echo "  • $OUTPUT_BASELINE (slim alone)"
echo "  • $OUTPUT_DIR/generated_rules/ (YAML rules)"
echo "  • $REPORT_FILE (analysis report)"
echo "  • $LOG_FILE (full log)"
echo ""
echo "Next steps:"
echo "  1. Compare hybrid vs baseline results"
echo "  2. Evaluate rule quality"
echo "  3. Document findings in PUBLICATION_SUMMARY_v4.md"
echo ""

log "Done!"
echo "Done!"
