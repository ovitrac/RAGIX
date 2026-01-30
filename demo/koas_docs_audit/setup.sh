#!/bin/bash
# Setup demo workspace for KOAS docs audit
# Uses docs/**/*.md as corpus

set -e

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
RAGIX_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
WORKSPACE="$DEMO_DIR/workspace"

echo "=== KOAS Docs Audit Demo Setup ==="
echo "Demo dir: $DEMO_DIR"
echo "RAGIX root: $RAGIX_ROOT"
echo "Workspace: $WORKSPACE"
echo ""

# Create workspace structure
echo "[1/5] Creating workspace structure..."
mkdir -p "$WORKSPACE"
mkdir -p "$WORKSPACE/.KOAS/activity"
mkdir -p "$WORKSPACE/.KOAS/auth"
mkdir -p "$WORKSPACE/.KOAS/cache"
mkdir -p "$WORKSPACE/.KOAS/logs"

# Create symlink to docs corpus
echo "[2/5] Linking docs corpus..."
if [ -L "$WORKSPACE/docs" ]; then
    rm "$WORKSPACE/docs"
fi
ln -s "$RAGIX_ROOT/docs" "$WORKSPACE/docs"

# Count corpus files (use -L to follow symlinks)
CORPUS_COUNT=$(find -L "$WORKSPACE/docs" -name "*.md" -type f | wc -l)
echo "    Corpus: $CORPUS_COUNT markdown files"

# Copy configurations
echo "[3/5] Installing configurations..."
mkdir -p "$DEMO_DIR/config"

# Create relaxed config
cat > "$DEMO_DIR/config/relaxed.yaml" << 'EOF'
# KOAS Relaxed Mode Configuration
# Direct CLI access, no broker, minimal auth

project:
  name: "RAGIX Documentation Audit"
  language: "en"
  author: "Demo"

pipeline:
  stages: [1, 2, 3]
  parallel: true
  workers: 4

llm:
  worker_model: "granite3.1-dense:8b"
  tutor_model: "mistral:7b-instruct"
  endpoint: "http://localhost:11434"
  temperature: 0.3

cache:
  llm_cache: "write_through"
  kernel_cache: "write_through"

output:
  level: "internal"  # Full output for development
  format: "markdown"

activity:
  enabled: true
  stream: ".KOAS/activity/events.jsonl"

# No auth required in relaxed mode
auth:
  enabled: false
EOF

# Create restricted config
cat > "$DEMO_DIR/config/restricted.yaml" << 'EOF'
# KOAS Restricted Mode Configuration
# Broker-mediated access, full auth, external-safe output

project:
  name: "RAGIX Documentation Audit"
  language: "en"
  author: "Demo"

pipeline:
  stages: [1, 2, 3]
  parallel: true
  workers: 4

llm:
  worker_model: "granite3.1-dense:8b"
  tutor_model: "mistral:7b-instruct"
  endpoint: "http://localhost:11434"
  temperature: 0.3

cache:
  llm_cache: "write_through"
  kernel_cache: "write_through"

output:
  level: "external"  # Sanitized output
  format: "markdown"
  redact_paths: true
  anonymize_ids: true

activity:
  enabled: true
  stream: ".KOAS/activity/events.jsonl"

auth:
  enabled: true
  acl_file: ".KOAS/auth/acl.yaml"
  require_hmac: false  # Set true for production
EOF

# Create ACL for restricted mode
cat > "$DEMO_DIR/config/acl.yaml" << 'EOF'
# KOAS Activity ACL — Demo
# Access control for restricted mode

schema_version: "koas.acl/1.0"

clients:
  # System (internal, no key required)
  koas-system:
    key_hash: null
    type: system
    scopes: ["*"]

  # Demo operator (full access)
  demo-operator:
    # Key: koas_key_demo_operator_12345
    key_hash: "sha256:a]5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"
    type: operator
    scopes:
      - "docs.trigger"
      - "docs.status"
      - "docs.export_external"
      - "docs.export_internal"
      - "activity.read"
    rate_limit: "60/min"

  # External orchestrator (limited)
  claude-demo:
    # Key: koas_key_claude_demo_67890
    key_hash: "sha256:d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
    type: external_orchestrator
    scopes:
      - "docs.trigger"
      - "docs.status"
      - "docs.export_external"
      # NOT: activity.read, docs.export_internal
    rate_limit: "30/min"
    restrictions:
      - "no_content_access"
      - "metrics_only"

# Scope definitions
scopes:
  docs.trigger:
    description: "Trigger KOAS workflow"
  docs.status:
    description: "View job status and metrics"
  docs.export_external:
    description: "Download external-safe artifacts"
  docs.export_internal:
    description: "Download full artifacts with traces"
  activity.read:
    description: "Read activity event stream"
EOF

# Copy ACL to workspace
cp "$DEMO_DIR/config/acl.yaml" "$WORKSPACE/.KOAS/auth/acl.yaml"

# Create API keys file (for demo only - in production, keys are generated and shown once)
echo "[4/5] Creating demo API keys..."
cat > "$DEMO_DIR/.demo_keys" << 'EOF'
# Demo API Keys (DO NOT USE IN PRODUCTION)
# These are pre-generated for demo purposes only

DEMO_OPERATOR_KEY=koas_key_demo_operator_12345
CLAUDE_DEMO_KEY=koas_key_claude_demo_67890
EOF

chmod 600 "$DEMO_DIR/.demo_keys"

# Verify Ollama
echo "[5/5] Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "    Ollama: OK"
    MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null | head -5 | tr '\n' ', ')
    echo "    Models: $MODELS"
else
    echo "    Ollama: NOT RUNNING"
    echo "    Run 'ollama serve' before starting the demo"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Relaxed mode:    ./run_relaxed.sh"
echo "  2. Restricted mode: ./start_broker.sh  (then ./run_restricted.sh)"
echo ""
echo "Corpus: $CORPUS_COUNT markdown files in workspace/docs/"
