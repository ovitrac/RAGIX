#!/bin/bash
# =============================================================================
# KOAS Quick Audit Script
# =============================================================================
# Run a quick KOAS audit on a single project.
#
# Usage:
#   ./koas-quick-audit.sh <project_path> [project_name] [--fr]
#
# Examples:
#   ./koas-quick-audit.sh /path/to/my-project
#   ./koas-quick-audit.sh /path/to/my-project "My Project" --fr
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# =============================================================================

set -e

# Parse arguments
PROJECT_PATH=""
PROJECT_NAME=""
LANG="en"

for arg in "$@"; do
    case $arg in
        --fr)
            LANG="fr"
            ;;
        --en)
            LANG="en"
            ;;
        -h|--help)
            echo "Usage: $0 <project_path> [project_name] [--fr|--en]"
            echo ""
            echo "Options:"
            echo "  --fr    Generate French report"
            echo "  --en    Generate English report (default)"
            exit 0
            ;;
        *)
            if [[ -z "$PROJECT_PATH" ]]; then
                PROJECT_PATH="$arg"
            elif [[ -z "$PROJECT_NAME" ]]; then
                PROJECT_NAME="$arg"
            fi
            ;;
    esac
done

if [[ -z "$PROJECT_PATH" ]]; then
    echo "Error: Project path required"
    echo "Usage: $0 <project_path> [project_name] [--fr|--en]"
    exit 1
fi

# Resolve absolute path
PROJECT_PATH=$(realpath "$PROJECT_PATH")

if [[ ! -d "$PROJECT_PATH" ]]; then
    echo "Error: Project directory not found: $PROJECT_PATH"
    exit 1
fi

# Default project name from directory
PROJECT_NAME="${PROJECT_NAME:-$(basename "$PROJECT_PATH")}"

# Detect language from files
if ls "$PROJECT_PATH"/*.py &>/dev/null || ls "$PROJECT_PATH"/**/*.py &>/dev/null 2>/dev/null; then
    PROJECT_LANG="python"
elif ls "$PROJECT_PATH"/*.java &>/dev/null || ls "$PROJECT_PATH"/**/*.java &>/dev/null 2>/dev/null; then
    PROJECT_LANG="java"
elif ls "$PROJECT_PATH"/*.ts &>/dev/null || ls "$PROJECT_PATH"/**/*.ts &>/dev/null 2>/dev/null; then
    PROJECT_LANG="typescript"
else
    PROJECT_LANG="python"  # Default
fi

# Create workspace
WORKSPACE_NAME=$(echo "$PROJECT_NAME" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
WORKSPACE="/tmp/koas_${WORKSPACE_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "KOAS Quick Audit"
echo "============================================================"
echo "Project: $PROJECT_NAME"
echo "Path: $PROJECT_PATH"
echo "Language: $PROJECT_LANG"
echo "Output: $LANG"
echo "Workspace: $WORKSPACE"
echo ""

# Initialize
echo "[1/4] Initializing workspace..."
ragix-koas init \
    --project "$PROJECT_PATH" \
    --name "$PROJECT_NAME" \
    --language "$PROJECT_LANG" \
    --workspace "$WORKSPACE"

# Update output language
if [[ -f "$WORKSPACE/manifest.yaml" ]]; then
    sed -i "s/language: \"en\"/language: \"$LANG\"/" "$WORKSPACE/manifest.yaml" 2>/dev/null || true
    # Add output section if not present
    if ! grep -q "^output:" "$WORKSPACE/manifest.yaml"; then
        echo -e "\noutput:\n  language: \"$LANG\"" >> "$WORKSPACE/manifest.yaml"
    fi
fi

# Run stages
echo ""
echo "[2/4] Running Stage 1 (Data Collection)..."
ragix-koas run --workspace "$WORKSPACE" --stage 1 --quiet

echo ""
echo "[3/4] Running Stage 2 (Analysis)..."
ragix-koas run --workspace "$WORKSPACE" --stage 2 --quiet

echo ""
echo "[4/4] Running Stage 3 (Report Generation)..."
ragix-koas run --workspace "$WORKSPACE" --stage 3 --quiet

echo ""
echo "============================================================"
echo "Audit Complete!"
echo "============================================================"

# Show report location
REPORT="$WORKSPACE/stage3/audit_report.md"
if [[ -f "$REPORT" ]]; then
    echo "Report: $REPORT"
    echo ""
    echo "Quick summary:"
    head -50 "$REPORT" | grep -E "^#|^\*\*|^-" | head -20
else
    echo "Warning: Report not generated"
fi
