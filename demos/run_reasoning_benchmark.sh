#!/bin/bash
#
# RAGIX v0.30 Reasoning Graph Benchmark Runner
#
# This script runs the reasoning demo with different LLM models and saves
# structured logs for comparison and ranking.
#
# Usage:
#   ./run_reasoning_benchmark.sh                    # Interactive mode
#   ./run_reasoning_benchmark.sh -m mistral:7b-instruct
#   ./run_reasoning_benchmark.sh --all              # Run all available models
#   ./run_reasoning_benchmark.sh --compare          # Compare existing results
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-/home/olivi/anaconda3/envs/ragix-env/bin/python}"
DEMO_SCRIPT="$SCRIPT_DIR/reasoning_v30_demo.py"

# Output directories
LOG_DIR="$PROJECT_ROOT/benchmark_logs"
RESULTS_DIR="$LOG_DIR/results"
CONSOLE_DIR="$LOG_DIR/console"

# Default models to benchmark (in order of preference)
# These are checked against actual installed models
DEFAULT_MODELS=(
    "mistral:7b-instruct"
    "qwen2.5:7b"
    "granite3.1-moe:3b"
    "dolphin-mistral:7b-v2.6-dpo-laser"
    "llama3:latest"
    "deepseek-r1:14b"
    "mistral:latest"
    "llama3.2:3b"
    "deepseek-r1:7b"
    "phi3:mini"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}======================================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}======================================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Ollama is running
check_ollama() {
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama is not installed. Please install it first."
        exit 1
    fi

    if ! ollama list &> /dev/null; then
        print_error "Ollama is not running. Please start it with: ollama serve"
        exit 1
    fi
}

# Get list of available models
get_available_models() {
    ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | sort
}

# Check if a specific model is available
is_model_available() {
    local model="$1"
    ollama list 2>/dev/null | grep -q "^${model}"
}

# Generate timestamp for filenames
get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# Sanitize model name for filename
sanitize_model_name() {
    echo "$1" | sed 's/:/_/g' | sed 's/\./_/g'
}

# =============================================================================
# Main Functions
# =============================================================================

# List available models
list_models() {
    print_header "Available Ollama Models"

    echo "Currently installed models:"
    echo ""
    ollama list
    echo ""

    echo "Default benchmark models:"
    for model in "${DEFAULT_MODELS[@]}"; do
        if is_model_available "$model"; then
            echo -e "  ${GREEN}✓${NC} $model (available)"
        else
            echo -e "  ${YELLOW}○${NC} $model (not installed)"
        fi
    done
    echo ""

    echo "To install a model: ollama pull <model-name>"
}

# Run benchmark for a single model
run_single_benchmark() {
    local model="$1"
    local timestamp=$(get_timestamp)
    local model_safe=$(sanitize_model_name "$model")

    # Create output directories
    mkdir -p "$RESULTS_DIR" "$CONSOLE_DIR"

    # Output files
    local json_file="$RESULTS_DIR/${model_safe}_${timestamp}.json"
    local console_file="$CONSOLE_DIR/${model_safe}_${timestamp}.log"

    print_header "Benchmarking: $model"
    print_info "JSON output: $json_file"
    print_info "Console log: $console_file"
    echo ""

    # Check if model is available
    if ! is_model_available "$model"; then
        print_warning "Model '$model' is not available. Attempting to pull..."
        if ! ollama pull "$model"; then
            print_error "Failed to pull model '$model'. Skipping."
            return 1
        fi
    fi

    # Run the demo
    print_info "Running benchmark..."

    if "$PYTHON" "$DEMO_SCRIPT" \
        --model "$model" \
        --output "$json_file" \
        2>&1 | tee "$console_file"; then

        print_success "Benchmark completed for $model"

        # Extract summary from JSON
        if [ -f "$json_file" ]; then
            echo ""
            print_info "Quick Summary:"
            "$PYTHON" -c "
import json
with open('$json_file') as f:
    data = json.load(f)
    s = data['summary']
    print(f\"  Pass Rate: {s['pass_rate']*100:.0f}%\")
    print(f\"  Complexity Accuracy: {s['complexity_accuracy']*100:.0f}%\")
    print(f\"  Avg Confidence: {s['avg_confidence']:.2f}\")
    print(f\"  Total Time: {s['total_elapsed_seconds']:.1f}s\")
    print(f\"  Reflections: {s['total_reflections']}\")
"
        fi
        return 0
    else
        print_error "Benchmark failed for $model"
        return 1
    fi
}

# Run benchmarks for all available default models
run_all_benchmarks() {
    local use_all_installed="${1:-false}"

    print_header "Running All Available Model Benchmarks"

    local available_models=()
    local skipped_models=()

    if [ "$use_all_installed" = "true" ]; then
        # Use ALL installed models (from ollama list)
        print_info "Detecting all installed Ollama models..."
        while IFS= read -r model; do
            if [ -n "$model" ]; then
                available_models+=("$model")
            fi
        done < <(get_available_models)
    else
        # Use only DEFAULT_MODELS that are installed
        for model in "${DEFAULT_MODELS[@]}"; do
            if is_model_available "$model"; then
                available_models+=("$model")
            else
                skipped_models+=("$model")
            fi
        done
    fi

    if [ ${#available_models[@]} -eq 0 ]; then
        print_error "No benchmark models are available."
        echo "Install models with: ollama pull <model-name>"
        echo "Available default models: ${DEFAULT_MODELS[*]}"
        exit 1
    fi

    print_info "Will benchmark: ${available_models[*]}"
    if [ ${#skipped_models[@]} -gt 0 ]; then
        print_warning "Skipping (not installed): ${skipped_models[*]}"
    fi
    echo ""

    local successful=0
    local failed=0

    for model in "${available_models[@]}"; do
        if run_single_benchmark "$model"; then
            ((successful++))
        else
            ((failed++))
        fi
        echo ""
    done

    print_header "Benchmark Summary"
    print_info "Successful: $successful"
    if [ $failed -gt 0 ]; then
        print_warning "Failed: $failed"
    fi

    # Generate comparison if multiple models were run
    if [ $successful -gt 1 ]; then
        echo ""
        compare_results
    fi
}

# Compare benchmark results
compare_results() {
    print_header "Model Comparison (Ranking)"

    if [ ! -d "$RESULTS_DIR" ] || [ -z "$(ls -A "$RESULTS_DIR" 2>/dev/null)" ]; then
        print_warning "No benchmark results found in $RESULTS_DIR"
        return 1
    fi

    # Use Python to analyze and compare results
    "$PYTHON" << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path
from datetime import datetime

results_dir = Path(os.environ.get('RESULTS_DIR', 'benchmark_logs/results'))

# Load all results
results = []
for f in results_dir.glob('*.json'):
    try:
        with open(f) as fp:
            data = json.load(fp)
            data['_file'] = f.name
            results.append(data)
    except Exception as e:
        print(f"Warning: Could not load {f}: {e}")

if not results:
    print("No valid results found.")
    exit(1)

# Group by model, keep latest run for each
by_model = {}
for r in results:
    model = r['meta']['model']
    ts = r['meta']['timestamp']
    if model not in by_model or ts > by_model[model]['meta']['timestamp']:
        by_model[model] = r

# Calculate composite score for ranking
# Score = pass_rate * 0.3 + complexity_accuracy * 0.3 + confidence * 0.2 + speed_score * 0.2
def calculate_score(r):
    s = r['summary']
    # Speed score: normalize to 0-1 (faster is better, assume 30s baseline, 180s max)
    elapsed = s['total_elapsed_seconds']
    speed_score = max(0, min(1, (180 - elapsed) / 150))

    score = (
        s['pass_rate'] * 0.30 +
        s['complexity_accuracy'] * 0.30 +
        s['avg_confidence'] * 0.20 +
        speed_score * 0.20
    )
    return score

# Rank models
ranked = []
for model, r in by_model.items():
    score = calculate_score(r)
    ranked.append({
        'model': model,
        'score': score,
        'data': r
    })

ranked.sort(key=lambda x: x['score'], reverse=True)

# Print ranking table
print("\n" + "="*80)
print(f"{'Rank':<6}{'Model':<30}{'Score':<8}{'Pass%':<8}{'Cmplx%':<8}{'Conf':<8}{'Time(s)':<10}")
print("="*80)

for i, entry in enumerate(ranked, 1):
    r = entry['data']
    s = r['summary']
    medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(i, '  ')
    print(f"{medal}{i:<4}{entry['model']:<30}{entry['score']:.3f}   "
          f"{s['pass_rate']*100:>5.0f}%  {s['complexity_accuracy']*100:>5.0f}%  "
          f"{s['avg_confidence']:>5.2f}   {s['total_elapsed_seconds']:>7.1f}")

print("="*80)
print("\nScoring: Pass Rate (30%) + Complexity Accuracy (30%) + Confidence (20%) + Speed (20%)")
print(f"\nTotal models compared: {len(ranked)}")

# Show best model details
if ranked:
    best = ranked[0]
    print(f"\n🏆 Best Model: {best['model']} (Score: {best['score']:.3f})")
    print(f"   Timestamp: {best['data']['meta']['timestamp']}")

PYTHON_SCRIPT
}

# Interactive model selection
interactive_mode() {
    print_header "RAGIX v0.30 Reasoning Benchmark"

    echo "Select an option:"
    echo ""
    echo "  1) Run benchmark with specific model"
    echo "  2) Run benchmarks for all available models"
    echo "  3) List available models"
    echo "  4) Compare existing results"
    echo "  5) Exit"
    echo ""

    read -p "Enter choice [1-5]: " choice

    case $choice in
        1)
            echo ""
            echo "Available models:"
            get_available_models | head -20
            echo ""
            read -p "Enter model name: " model
            if [ -n "$model" ]; then
                run_single_benchmark "$model"
            else
                print_error "No model specified."
            fi
            ;;
        2)
            run_all_benchmarks
            ;;
        3)
            list_models
            ;;
        4)
            compare_results
            ;;
        5)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid choice."
            exit 1
            ;;
    esac
}

# Show usage
show_usage() {
    cat << EOF
RAGIX v0.30 Reasoning Graph Benchmark Runner

Usage:
    $(basename "$0") [OPTIONS]

Options:
    -m, --model MODEL     Run benchmark for specific model
    -a, --all             Run benchmarks for all available default models
    -A, --all-installed   Run benchmarks for ALL installed Ollama models
    -l, --list            List available Ollama models
    -c, --compare         Compare existing benchmark results
    -h, --help            Show this help message

Examples:
    $(basename "$0")                              # Interactive mode
    $(basename "$0") -m mistral:7b-instruct       # Single model benchmark
    $(basename "$0") -m granite3.1-moe:3b         # Alternative model
    $(basename "$0") --all                        # Benchmark default models (if installed)
    $(basename "$0") --all-installed              # Benchmark ALL installed models
    $(basename "$0") --compare                    # Show ranking of tested models

Output:
    Results are saved to:
    - $LOG_DIR/results/*.json     (structured benchmark data)
    - $LOG_DIR/console/*.log      (console output with colors)

Default Models (in preference order):
    ${DEFAULT_MODELS[*]}

Environment Variables:
    PYTHON          Path to Python interpreter (default: ragix-env)
    RESULTS_DIR     Override results directory

EOF
}

# =============================================================================
# Main Entry Point
# =============================================================================

# Export for Python subprocesses
export RESULTS_DIR

# Check prerequisites
check_ollama

# Parse arguments
if [ $# -eq 0 ]; then
    interactive_mode
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            if [ -z "$2" ]; then
                print_error "Model name required after $1"
                exit 1
            fi
            run_single_benchmark "$2"
            shift 2
            ;;
        -a|--all)
            run_all_benchmarks "false"
            shift
            ;;
        -A|--all-installed)
            run_all_benchmarks "true"
            shift
            ;;
        -l|--list)
            list_models
            shift
            ;;
        -c|--compare)
            compare_results
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done
