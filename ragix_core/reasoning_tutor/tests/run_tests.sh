#!/bin/bash
# ============================================================================
# Test Runner for RAGIX Reasoning Tutor v0.3
# ============================================================================
#
# Runs all meta-cognitive component tests:
# - R1: FailureDetector
# - R2: MetaCards
# - R3: JustificationProtocol
# - Integration tests
#
# Usage:
#   ./run_tests.sh          # Run all tests
#   ./run_tests.sh -v       # Verbose output
#   ./run_tests.sh -k "pattern"  # Run tests matching pattern
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Date: 2025-12-23
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUTOR_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}  RAGIX Reasoning Tutor v0.3 — Test Suite${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Check pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found. Install with: pip install pytest${NC}"
    exit 1
fi

# Change to tutor directory for imports
cd "$TUTOR_DIR"

echo -e "${YELLOW}Running tests from: $TUTOR_DIR${NC}"
echo ""

# Run tests with coverage if available
if python3 -c "import pytest_cov" 2>/dev/null; then
    echo -e "${GREEN}Running with coverage...${NC}"
    pytest tests/ \
        --cov=. \
        --cov-report=term-missing \
        --cov-report=html:tests/coverage_html \
        -v \
        "$@"
else
    echo -e "${YELLOW}Running without coverage (install pytest-cov for coverage)${NC}"
    pytest tests/ -v "$@"
fi

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}  Test suite completed${NC}"
echo -e "${GREEN}============================================================================${NC}"
