#!/usr/bin/env bash
# ragix_audit_logger.sh — PostToolUse hook for Claude Code
# Logs tool actions to .agent_logs/commands.log
#
# Protocol: reads JSON from stdin (tool_name, tool_input), appends entry.
#   Always exits 0 (logging never blocks).
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

set -uo pipefail

# Project root = where .claude/ lives
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/.agent_logs"
LOG_FILE="$LOG_DIR/commands.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR" 2>/dev/null || exit 0

# Read the hook payload from stdin
INPUT=$(cat)

# Extract tool name and a summary of the input
ENTRY=$(printf '%s' "$INPUT" | python3 -c "
import sys, json, datetime
try:
    data = json.load(sys.stdin)
    tool = data.get('tool_name', 'unknown')
    inp = data.get('tool_input', {})
    # Summarize input (first 200 chars of command or description)
    if isinstance(inp, dict):
        summary = inp.get('command', inp.get('file_path', inp.get('pattern', str(inp)[:200])))
    else:
        summary = str(inp)[:200]
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    print(f'{ts} | {tool} | {summary}')
except Exception as e:
    print(f'{datetime.datetime.now().isoformat(timespec=\"seconds\")} | error | {e}')
" 2>/dev/null)

# Append to log
[ -n "$ENTRY" ] && echo "$ENTRY" >> "$LOG_FILE" 2>/dev/null

exit 0
