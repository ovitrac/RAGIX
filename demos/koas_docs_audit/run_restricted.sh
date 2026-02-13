#!/bin/bash
# Run KOAS in restricted mode via broker API
# Requires broker to be running (./start_broker.sh)

set -e

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
BROKER_URL="http://localhost:8080/koas/v1"

# Load demo keys
if [ -f "$DEMO_DIR/.demo_keys" ]; then
    source "$DEMO_DIR/.demo_keys"
else
    echo "ERROR: Demo keys not found. Run ./setup.sh first."
    exit 1
fi

# Use external orchestrator key (limited scopes)
API_KEY="${CLAUDE_DEMO_KEY}"

echo "=== KOAS Docs Audit — Restricted Mode ==="
echo "Broker: $BROKER_URL"
echo "Client: claude-demo (external_orchestrator)"
echo ""

# Check broker is running
if ! curl -s "$BROKER_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Broker not responding at $BROKER_URL"
    echo "Run './start_broker.sh' in another terminal first."
    exit 1
fi

echo "[1/4] Triggering job via broker..."

# Trigger job
RESPONSE=$(curl -s -X POST "$BROKER_URL/jobs" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "mode": "pure_docs",
        "profile": "docs_audit",
        "workspace": "./workspace",
        "actions": ["index", "koas_audit", "export_report_external"]
    }')

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
STATUS=$(echo "$RESPONSE" | jq -r '.status')

if [ "$JOB_ID" == "null" ] || [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to create job"
    echo "Response: $RESPONSE"
    exit 1
fi

echo "    Job ID: $JOB_ID"
echo "    Status: $STATUS"
echo ""

echo "[2/4] Polling job status..."

# Poll until complete
MAX_POLLS=60
POLL_INTERVAL=5

for i in $(seq 1 $MAX_POLLS); do
    STATUS_RESPONSE=$(curl -s "$BROKER_URL/jobs/$JOB_ID" \
        -H "Authorization: Bearer $API_KEY")

    STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
    PROGRESS=$(echo "$STATUS_RESPONSE" | jq -r '.progress.pct // 0')

    echo "    [$i/$MAX_POLLS] Status: $STATUS, Progress: ${PROGRESS}%"

    if [ "$STATUS" == "completed" ] || [ "$STATUS" == "failed" ]; then
        break
    fi

    sleep $POLL_INTERVAL
done

echo ""

if [ "$STATUS" != "completed" ]; then
    echo "ERROR: Job did not complete (status: $STATUS)"
    exit 1
fi

echo "[3/4] Retrieving metrics..."

# Get final metrics
METRICS=$(curl -s "$BROKER_URL/jobs/$JOB_ID" \
    -H "Authorization: Bearer $API_KEY" | jq '.metrics')

echo "$METRICS" | jq .
echo ""

echo "[4/4] Downloading external-safe artifact..."

# Download artifact
ARTIFACT_PATH="$DEMO_DIR/output_restricted_$(date +%Y%m%d_%H%M%S).zip"

curl -s "$BROKER_URL/jobs/$JOB_ID/artifact?view=external" \
    -H "Authorization: Bearer $API_KEY" \
    -o "$ARTIFACT_PATH"

if [ -f "$ARTIFACT_PATH" ]; then
    ARTIFACT_SIZE=$(wc -c < "$ARTIFACT_PATH")
    echo "    Artifact: $ARTIFACT_PATH ($ARTIFACT_SIZE bytes)"
else
    echo "    Artifact: DOWNLOAD FAILED"
fi

echo ""
echo "=== Restricted Mode Complete ==="
echo ""
echo "Summary:"
echo "  Job ID:    $JOB_ID"
echo "  Status:    $STATUS"
echo "  Artifact:  $ARTIFACT_PATH"
echo ""
echo "Verify external-safe output:"
echo "  unzip -l $ARTIFACT_PATH"
echo "  # Should NOT contain: internal paths, call_hash, merkle roots"
echo ""
echo "Check activity log (requires operator key):"
echo "  curl $BROKER_URL/activity/events -H 'Authorization: Bearer \$DEMO_OPERATOR_KEY'"
