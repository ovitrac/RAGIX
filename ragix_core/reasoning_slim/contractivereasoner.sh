#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# contractive_reasoner.sh
#
# Thin CLI wrapper around ContractiveReasoner.py
# Makes it easy to call the contractive reasoning engine from the shell.
#
# Location: same folder as ContractiveReasoner.py
# Usage:
#   ./contractive_reasoner.sh "Your question here"
#   ./contractive_reasoner.sh -m granite3.1-moe:3b -d 3 -l 6 "Complex question ..."
#   ./contractive_reasoner.sh --config config.yaml --chat
#
# Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio Innovation Lab
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
SCRIPT_PY="$SCRIPT_DIR/ContractiveReasoner.py"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] "your question ..."

Options:
  -m MODEL              Ollama model name (default: granite3.1-moe:3b)
  -d DEPTH              Max reasoning depth (default: 3)
  -l LOOPS              Max reasoning loops (default: 6)
  --config PATH         YAML/JSON config for parameters (CLI flags override)
  --chat                Interactive loop (repeat questions until exit)
  --export-trace PATH   Export full trace JSON
  --export-mermaid PATH Export Mermaid graph
  --log-events PATH     NDJSON per-step events
  -h                    Show this help message and exit

Examples:
  $(basename "$0") "Design a safe-by-design migration study for a new polymer."
  $(basename "$0") -m mistral -d 4 -l 8 "Explain the physics of deep frying."
  $(basename "$0") --config config.yaml --chat

Environment:
  PYTHON         Python interpreter to use (default: python3)

Note:
  ContractiveReasoner.py already has an internal CLI using argparse.
  This wrapper simply forwards the model, depth, loops and question.
EOF
}

MODEL="granite3.1-moe:3b"
MAX_DEPTH=3
MAX_LOOPS=6
CONFIG=""
CHAT=""
EXPORT_TRACE=""
EXPORT_MERMAID=""
LOG_EVENTS=""

LONG_OPTS="config:,chat,export-trace:,export-mermaid:,log-events:"

# Parse short and long options
PARSED=$(getopt -o "m:d:l:h" --long "$LONG_OPTS" -- "$@") || {
    usage
    exit 1
}
eval set -- "$PARSED"

while true; do
    case "$1" in
        -m) MODEL="$2"; shift 2 ;;
        -d) MAX_DEPTH="$2"; shift 2 ;;
        -l) MAX_LOOPS="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --chat) CHAT="1"; shift 1 ;;
        --export-trace) EXPORT_TRACE="$2"; shift 2 ;;
        --export-mermaid) EXPORT_MERMAID="$2"; shift 2 ;;
        --log-events) LOG_EVENTS="$2"; shift 2 ;;
        -h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Error: invalid option $1" >&2
            usage
            exit 1
            ;;
    esac
done

shift $((OPTIND - 1))

if [ "$#" -lt 1 ] && [ -z "$CHAT" ]; then
    echo "Error: you must provide a question unless using --chat." >&2
    usage
    exit 1
fi

QUESTION="$*"

if [ ! -f "$SCRIPT_PY" ]; then
    echo "Error: ContractiveReasoner.py not found at: $SCRIPT_PY" >&2
    exit 1
fi

# Run the Python script with the forwarded parameters
exec "$PYTHON" "$SCRIPT_PY" \
    --model "$MODEL" \
    --max-depth "$MAX_DEPTH" \
    --max-loops "$MAX_LOOPS" \
    ${CONFIG:+--config "$CONFIG"} \
    ${CHAT:+--chat} \
    ${EXPORT_TRACE:+--export-trace "$EXPORT_TRACE"} \
    ${EXPORT_MERMAID:+--export-mermaid "$EXPORT_MERMAID"} \
    ${LOG_EVENTS:+--log-events "$LOG_EVENTS"} \
    "$QUESTION"
