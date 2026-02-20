#!/usr/bin/env bash
# ragix_safety_guard.sh — PreToolUse hook for Claude Code
# Blocks dangerous shell commands (mirrors unix-rag-agent.py:71-113 denylist)
#
# Protocol: reads JSON from stdin, checks tool_input.command against patterns.
#   Exit 0 = allow, Exit 2 = block (stderr = feedback to Claude)
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

set -euo pipefail

# Read the hook payload from stdin
INPUT=$(cat)

# Extract the command field from tool_input
COMMAND=$(printf '%s' "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    cmd = data.get('tool_input', {}).get('command', '')
    print(cmd)
except Exception:
    print('')
" 2>/dev/null)

# Nothing to check if command is empty
[ -z "$COMMAND" ] && exit 0

# --- DANGEROUS_PATTERNS (hard denylist, never allowed) ---
DANGEROUS_PATTERNS=(
    'rm\s+-rf\s+/'
    'rm\s+-rf\s+\.\s*$'
    'mkfs\.'
    'dd\s+if='
    'shutdown\b'
    'reboot\b'
    ':\s*\(\)\s*\{'
    '\bsudo\b'
    '\bsu\s+-'
    '\bsu\s+root\b'
    '\bcurl\b.*\|\s*sh'
    '\bwget\b.*\|\s*sh'
    '\bcurl\b.*\|\s*bash'
    '\bwget\b.*\|\s*bash'
)

# --- GIT_DESTRUCTIVE_PATTERNS (blocked by default) ---
GIT_PATTERNS=(
    'git\s+reset\s+--hard'
    'git\s+clean\s+-fd'
    'git\s+push\s+--force'
    'git\s+push\s+-f'
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
    if echo "$COMMAND" | grep -qP "$pattern"; then
        echo "RAGIX safety guard: blocked dangerous command matching /$pattern/" >&2
        exit 2
    fi
done

for pattern in "${GIT_PATTERNS[@]}"; do
    if echo "$COMMAND" | grep -qP "$pattern"; then
        echo "RAGIX safety guard: blocked git-destructive command matching /$pattern/. Use the git CLI directly if you really need this." >&2
        exit 2
    fi
done

exit 0
