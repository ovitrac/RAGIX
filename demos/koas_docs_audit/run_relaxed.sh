#!/bin/bash
# Run KOAS in relaxed mode (direct CLI access)
# No broker, no auth, full output

set -e

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$DEMO_DIR/workspace"

echo "=== KOAS Docs Audit — Relaxed Mode ==="
echo "Workspace: $WORKSPACE"
echo "Output level: internal (full)"
echo ""

# Check workspace exists
if [ ! -d "$WORKSPACE/docs" ]; then
    echo "ERROR: Workspace not initialized. Run ./setup.sh first."
    exit 1
fi

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "WARNING: Ollama not responding. LLM calls will fail."
    echo "Run 'ollama serve' in another terminal."
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Initialize activity log
ACTIVITY_LOG="$WORKSPACE/.KOAS/activity/events.jsonl"
mkdir -p "$(dirname "$ACTIVITY_LOG")"

# Write start event (manual, until Phase 1 is integrated)
START_EVENT=$(cat << EOF
{"v":"koas.event/1.0","ts":"$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)","event_id":"$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid)","run_id":"demo_relaxed_$(date +%Y%m%d_%H%M%S)","actor":{"type":"operator","id":"demo-cli","auth":"none"},"scope":"demo.workflow","phase":"start","decision":{"mode":"relaxed"}}
EOF
)
echo "$START_EVENT" >> "$ACTIVITY_LOG"

echo "[1/3] Running KOAS pipeline..."
echo ""

# Run KOAS
python -m ragix_kernels.run_doc_koas run \
    --workspace "$WORKSPACE" \
    --all \
    --output-level=internal \
    --llm-cache=write_through \
    --kernel-cache=write_through \
    2>&1 | tee "$WORKSPACE/.KOAS/logs/run_relaxed_$(date +%Y%m%d_%H%M%S).log"

KOAS_EXIT=$?

# Write end event
END_EVENT=$(cat << EOF
{"v":"koas.event/1.0","ts":"$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)","event_id":"$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid)","run_id":"demo_relaxed_$(date +%Y%m%d_%H%M%S)","actor":{"type":"operator","id":"demo-cli","auth":"none"},"scope":"demo.workflow","phase":"end","decision":{"exit_code":$KOAS_EXIT}}
EOF
)
echo "$END_EVENT" >> "$ACTIVITY_LOG"

echo ""
echo "[2/3] Checking outputs..."

# Check outputs
if [ -f "$WORKSPACE/.KOAS/final_report.md" ]; then
    REPORT_SIZE=$(wc -c < "$WORKSPACE/.KOAS/final_report.md")
    echo "    Final report: $REPORT_SIZE bytes"
else
    echo "    Final report: NOT FOUND"
fi

# Check activity log
if [ -f "$ACTIVITY_LOG" ]; then
    EVENT_COUNT=$(wc -l < "$ACTIVITY_LOG")
    echo "    Activity events: $EVENT_COUNT"
else
    echo "    Activity events: NOT FOUND"
fi

echo ""
echo "[3/3] Summary"
echo ""

if [ $KOAS_EXIT -eq 0 ]; then
    echo "✓ KOAS completed successfully"
    echo ""
    echo "Outputs:"
    echo "  Report:   $WORKSPACE/.KOAS/final_report.md"
    echo "  Activity: $ACTIVITY_LOG"
    echo ""
    echo "Inspect activity log:"
    echo "  cat $ACTIVITY_LOG | jq ."
    echo "  cat $ACTIVITY_LOG | jq -s 'group_by(.scope) | map({scope: .[0].scope, count: length})'"
else
    echo "✗ KOAS failed with exit code $KOAS_EXIT"
    echo "  Check logs: $WORKSPACE/.KOAS/logs/"
fi
