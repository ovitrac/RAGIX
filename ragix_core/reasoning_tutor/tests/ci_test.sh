#!/bin/bash
# CI Test Script for Interpreter-Tutor Benchmarks
#
# Usage:
#   ./ci_test.sh smoke      # Quick tests (no LLM, ~5 seconds)
#   ./ci_test.sh regression # Full regression (requires Ollama, ~2 minutes)
#   ./ci_test.sh all        # Both smoke and regression
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Interpreter-Tutor CI Tests"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(python --version)"
echo ""

case "${1:-smoke}" in
    smoke)
        echo "Running SMOKE tests (no LLM required)..."
        python -m pytest ragix_core/reasoning_tutor/tests/test_benchmarks.py \
            -v -m "smoke" \
            --tb=short
        ;;

    regression)
        echo "Running REGRESSION tests (requires Ollama)..."

        # Check Ollama
        if ! command -v ollama &> /dev/null; then
            echo "ERROR: Ollama not found"
            exit 1
        fi

        if ! ollama list &> /dev/null; then
            echo "ERROR: Ollama not running"
            exit 1
        fi

        python -m pytest ragix_core/reasoning_tutor/tests/test_benchmarks.py \
            -v -m "regression" \
            --tb=short \
            --timeout=300
        ;;

    full)
        echo "Running FULL regression with regression_runner..."
        python -m ragix_core.reasoning_tutor.tests.regression_runner run --fast
        ;;

    all)
        echo "Running ALL tests..."
        $0 smoke
        echo ""
        $0 regression
        ;;

    *)
        echo "Usage: $0 {smoke|regression|full|all}"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "CI Tests PASSED"
echo "========================================"
