#!/usr/bin/env bash
# ragix_memory_inject.sh — UserPromptSubmit hook for Claude Code
# Auto-recalls relevant memory context before each user turn.
#
# Protocol: reads JSON from stdin (user_prompt), calls ragix-memory recall,
#   outputs JSON {"systemMessage": "..."} on stdout.
#   Graceful degradation: if no DB, CLI missing, or recall fails -> exits 0 silently.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

set -uo pipefail

# Project root = where .claude/ lives (parent of scripts/hooks/)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DB="${RAGIX_MEMORY_DB:-$PROJECT_ROOT/memory.db}"

# Bail silently if no database
[ -f "$DB" ] || exit 0

# Bail silently if ragix-memory not available
command -v ragix-memory >/dev/null 2>&1 || exit 0

# Read the hook payload from stdin
INPUT=$(cat)

# Extract user prompt
PROMPT=$(printf '%s' "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('user_prompt', ''))
except Exception:
    print('')
" 2>/dev/null)

# Skip empty prompts or very short ones (< 10 chars)
[ ${#PROMPT} -lt 10 ] && exit 0

# Call ragix-memory recall with budget
CONTEXT=$(ragix-memory recall "$PROMPT" --budget 1500 --db "$DB" --format plain 2>/dev/null) || exit 0

# Skip if recall returned nothing useful
[ -z "$CONTEXT" ] && exit 0

# Output the system message as JSON
python3 -c "
import json, sys
ctx = sys.stdin.read()
if ctx.strip():
    print(json.dumps({'systemMessage': '[RAGIX Memory Context]\n' + ctx}))
" <<< "$CONTEXT"

exit 0
