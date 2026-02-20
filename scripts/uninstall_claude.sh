#!/usr/bin/env bash
# uninstall_claude.sh — Remove RAGIX hooks and MCP registration from Claude Code
#
# Convenience wrapper around install_claude.sh --uninstall.
# Removes MCP server entries and hooks. Does NOT delete .claude/commands/ (in repo).
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SETTINGS_LOCAL="$PROJECT_ROOT/.claude/settings.local.json"
SETTINGS_PROJECT="$PROJECT_ROOT/.claude/settings.json"

# Colors
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
    BOLD='\033[1m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; BOLD=''; NC=''
fi

info()  { printf "${GREEN}[ok]${NC}   %s\n" "$*"; }
warn()  { printf "${YELLOW}[warn]${NC} %s\n" "$*"; }

echo ""
printf "${BOLD}RAGIX Claude Code Uninstaller${NC}\n"
echo "============================================"
echo ""

# Backup helper
backup_file() {
    local f="$1"
    if [ -f "$f" ]; then
        local bak="${f}.bak.$(date +%Y%m%d_%H%M%S)"
        cp "$f" "$bak"
        info "Backup: $bak"
    fi
}

REMOVED=0

# Remove MCP entries from settings.local.json
if [ -f "$SETTINGS_LOCAL" ]; then
    if python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
sys.exit(0 if 'mcpServers' in data and ('ragix-memory' in data['mcpServers'] or 'ragix-mcp' in data['mcpServers']) else 1)
" "$SETTINGS_LOCAL" 2>/dev/null; then
        backup_file "$SETTINGS_LOCAL"
        python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
mcp = data.get('mcpServers', {})
removed = []
for k in ['ragix-memory', 'ragix-mcp']:
    if k in mcp:
        del mcp[k]
        removed.append(k)
if not mcp:
    data.pop('mcpServers', None)
with open(sys.argv[1], 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
for r in removed:
    print(f'Removed MCP server: {r}')
" "$SETTINGS_LOCAL"
        REMOVED=$((REMOVED + 1))
    else
        info "No RAGIX MCP servers found in settings.local.json"
    fi
else
    info "No settings.local.json found"
fi

# Remove hooks from settings.json
if [ -f "$SETTINGS_PROJECT" ]; then
    if python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
sys.exit(0 if 'hooks' in data else 1)
" "$SETTINGS_PROJECT" 2>/dev/null; then
        backup_file "$SETTINGS_PROJECT"
        python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
hooks = data.get('hooks', {})
count = 0
for event in list(hooks.keys()):
    if isinstance(hooks[event], list):
        before = len(hooks[event])
        hooks[event] = [h for h in hooks[event]
                        if not (isinstance(h, dict) and 'ragix' in json.dumps(h).lower())]
        count += before - len(hooks[event])
        if not hooks[event]:
            del hooks[event]
if not hooks:
    data.pop('hooks', None)
with open(sys.argv[1], 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
print(f'Removed {count} RAGIX hook entries')
" "$SETTINGS_PROJECT"
        REMOVED=$((REMOVED + 1))
    else
        info "No hooks found in settings.json"
    fi
else
    info "No settings.json found"
fi

echo ""
if [ "$REMOVED" -gt 0 ]; then
    info "RAGIX integration removed. Restart your Claude Code session."
else
    info "Nothing to remove — RAGIX integration was not installed."
fi
info "Slash commands in .claude/commands/ are preserved (they're part of the repo)."
info "Hook scripts in scripts/hooks/ are preserved (re-run install to re-enable)."
echo ""
