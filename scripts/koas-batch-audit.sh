#!/bin/bash
# =============================================================================
# KOAS Batch Audit Script
# =============================================================================
# Run KOAS audits on multiple projects or modules.
#
# Usage:
#   ./koas-batch-audit.sh <projects_file> [--lang fr|en] [--output-dir <dir>]
#
# projects_file format (one project per line):
#   /path/to/project1|Project Name 1|python
#   /path/to/project2|Project Name 2|java
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# =============================================================================

set -e

# Defaults
LANG="en"
OUTPUT_DIR="./koas_reports"
PARALLEL=1

# Parse arguments
PROJECTS_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --lang)
            LANG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <projects_file> [--lang fr|en] [--output-dir <dir>] [--parallel <n>]"
            echo ""
            echo "projects_file format (pipe-separated):"
            echo "  /path/to/project|Project Name|language"
            exit 0
            ;;
        *)
            PROJECTS_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "$PROJECTS_FILE" ]]; then
    echo "Error: Projects file required"
    echo "Usage: $0 <projects_file> [--lang fr|en] [--output-dir <dir>]"
    exit 1
fi

if [[ ! -f "$PROJECTS_FILE" ]]; then
    echo "Error: Projects file not found: $PROJECTS_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "KOAS Batch Audit"
echo "============================================================"
echo "Projects file: $PROJECTS_FILE"
echo "Output language: $LANG"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Process each project
TOTAL=0
SUCCESS=0
FAILED=0

while IFS='|' read -r PROJECT_PATH PROJECT_NAME PROJECT_LANG || [[ -n "$PROJECT_PATH" ]]; do
    # Skip empty lines and comments
    [[ -z "$PROJECT_PATH" || "$PROJECT_PATH" =~ ^# ]] && continue

    TOTAL=$((TOTAL + 1))

    # Default values
    PROJECT_NAME="${PROJECT_NAME:-$(basename "$PROJECT_PATH")}"
    PROJECT_LANG="${PROJECT_LANG:-python}"

    echo "------------------------------------------------------------"
    echo "[$TOTAL] Auditing: $PROJECT_NAME"
    echo "    Path: $PROJECT_PATH"
    echo "    Language: $PROJECT_LANG"
    echo ""

    # Create workspace name
    WORKSPACE_NAME=$(echo "$PROJECT_NAME" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    WORKSPACE="$OUTPUT_DIR/${WORKSPACE_NAME}_audit"

    # Initialize workspace
    if ragix-koas init \
        --project "$PROJECT_PATH" \
        --name "$PROJECT_NAME" \
        --language "$PROJECT_LANG" \
        --workspace "$WORKSPACE" 2>/dev/null; then

        # Update manifest with output language
        if [[ -f "$WORKSPACE/manifest.yaml" ]]; then
            # Add/update output.language
            if grep -q "^output:" "$WORKSPACE/manifest.yaml"; then
                sed -i "s/language: \"en\"/language: \"$LANG\"/" "$WORKSPACE/manifest.yaml"
            else
                echo -e "\noutput:\n  language: \"$LANG\"" >> "$WORKSPACE/manifest.yaml"
            fi
        fi

        # Run all stages
        if ragix-koas run --workspace "$WORKSPACE" --all --quiet 2>&1 | tail -5; then
            SUCCESS=$((SUCCESS + 1))

            # Copy report to output directory
            if [[ -f "$WORKSPACE/stage3/audit_report.md" ]]; then
                cp "$WORKSPACE/stage3/audit_report.md" "$OUTPUT_DIR/${WORKSPACE_NAME}_report.md"
                echo "    Report: $OUTPUT_DIR/${WORKSPACE_NAME}_report.md"
            fi
        else
            FAILED=$((FAILED + 1))
            echo "    ERROR: Stage execution failed"
        fi
    else
        FAILED=$((FAILED + 1))
        echo "    ERROR: Workspace initialization failed"
    fi

    echo ""
done < "$PROJECTS_FILE"

echo "============================================================"
echo "Batch Audit Complete"
echo "============================================================"
echo "Total projects: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "Reports in: $OUTPUT_DIR"
echo ""

# Exit with error if any failed
[[ $FAILED -gt 0 ]] && exit 1
exit 0
