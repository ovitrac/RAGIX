#!/bin/bash
# Start KOAS Broker for restricted mode
# Requires: pip install fastapi uvicorn pyyaml

set -e

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Starting KOAS Broker ==="
echo ""

# Check dependencies
python -c "import fastapi, uvicorn, yaml" 2>/dev/null || {
    echo "Missing dependencies. Installing..."
    pip install fastapi uvicorn pyyaml
}

# Check workspace
if [ ! -d "$DEMO_DIR/workspace" ]; then
    echo "ERROR: Workspace not initialized. Run ./setup.sh first."
    exit 1
fi

echo "Broker URL: http://localhost:8080/koas/v1"
echo "Health:     http://localhost:8080/koas/v1/health"
echo ""
echo "Demo API keys (from .demo_keys):"
echo "  Operator: koas_key_demo_operator_12345"
echo "  Claude:   koas_key_claude_demo_67890"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$DEMO_DIR"
python broker/main.py
