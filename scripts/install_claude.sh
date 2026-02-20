#!/usr/bin/env bash
# install_claude.sh — Register RAGIX skills, hooks & MCP servers with Claude Code
#
# Idempotent installer: safe to run multiple times. Backs up before modifying.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SETTINGS_LOCAL="$PROJECT_ROOT/.claude/settings.local.json"
SETTINGS_PROJECT="$PROJECT_ROOT/.claude/settings.json"
HOOKS_DIR="$PROJECT_ROOT/scripts/hooks"
COMMANDS_DIR="$PROJECT_ROOT/.claude/commands"
MANIFEST="$PROJECT_ROOT/RAGIX_COMPONENTS.md"

# Defaults
PROFILE="memory"
DRY_RUN=0
HOOKS_ONLY=0
MCP_ONLY=0
NO_HOOKS=0
NO_MCP=0
UNINSTALL=0
VERBOSE=0

# Colors (disable if not a terminal)
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; NC=''
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { printf "${GREEN}[ok]${NC}   %s\n" "$*"; }
warn()  { printf "${YELLOW}[warn]${NC} %s\n" "$*"; }
err()   { printf "${RED}[err]${NC}  %s\n" "$*" >&2; }
step()  { printf "${CYAN}[step]${NC} ${BOLD}%s${NC}\n" "$*"; }
dry()   { printf "${YELLOW}[dry]${NC}  %s\n" "$*"; }

usage() {
    cat <<'EOF'
Usage: scripts/install_claude.sh [OPTIONS]

Register RAGIX skills, hooks & MCP servers with Claude Code.

Options:
  --profile memory  Default: register Memory MCP only
  --profile full    Also register main RAGIX MCP server (38 tools)
  --dry-run         Show what would be done without modifying anything
  --hooks-only      Only install/update hooks
  --mcp-only        Only register MCP servers
  --no-hooks        Skip hook installation
  --no-mcp          Skip MCP registration
  --uninstall       Remove RAGIX hooks and MCP registration
  --verbose         Show detailed output
  --help            Show this message

Examples:
  bash scripts/install_claude.sh                  # Default: memory MCP + hooks
  bash scripts/install_claude.sh --profile full   # All MCP servers + hooks
  bash scripts/install_claude.sh --dry-run        # Preview changes
  bash scripts/install_claude.sh --uninstall      # Remove RAGIX integration
EOF
    exit 0
}

backup_file() {
    local f="$1"
    if [ -f "$f" ]; then
        local bak="${f}.bak.$(date +%Y%m%d_%H%M%S)"
        cp "$f" "$bak"
        [ "$VERBOSE" -eq 1 ] && info "Backup: $bak"
    fi
}

# Safe JSON merge using Python (no jq dependency)
# Uses atomic write (tmp + rename) to minimize file watcher interference.
json_merge() {
    # $1 = target file, $2 = JSON string to deep-merge
    local target="$1" patch="$2"
    python3 -c "
import json, sys, os, tempfile

target_path = sys.argv[1]
patch = json.loads(sys.argv[2])

try:
    with open(target_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}

def deep_merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base

result = deep_merge(data, patch)
target_dir = os.path.dirname(target_path) or '.'
fd, tmp = tempfile.mkstemp(dir=target_dir, suffix='.tmp')
try:
    with os.fdopen(fd, 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    os.replace(tmp, target_path)
except Exception:
    os.unlink(tmp)
    raise
" "$target" "$patch"
}

# Remove specific keys from JSON (atomic write)
json_remove_keys() {
    # $1 = target file, $2 = Python expression for keys to remove
    local target="$1" remove_script="$2"
    python3 -c "
import json, sys, os, tempfile

target_path = sys.argv[1]
try:
    with open(target_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(0)

$remove_script

target_dir = os.path.dirname(target_path) or '.'
fd, tmp = tempfile.mkstemp(dir=target_dir, suffix='.tmp')
try:
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    os.replace(tmp, target_path)
except Exception:
    os.unlink(tmp)
    raise
" "$target"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --profile)   PROFILE="${2:-memory}"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --hooks-only) HOOKS_ONLY=1; shift ;;
        --mcp-only)  MCP_ONLY=1; shift ;;
        --no-hooks)  NO_HOOKS=1; shift ;;
        --no-mcp)    NO_MCP=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        --verbose)   VERBOSE=1; shift ;;
        --help|-h)   usage ;;
        *) err "Unknown option: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
printf "${BOLD}RAGIX Claude Code Installer${NC}\n"
echo "============================================"
[ "$DRY_RUN" -eq 1 ] && printf "${YELLOW}DRY-RUN MODE — no files will be modified${NC}\n"
[ "$UNINSTALL" -eq 1 ] && printf "${RED}UNINSTALL MODE — removing RAGIX integration${NC}\n"
echo ""

# ---------------------------------------------------------------------------
# Phase 0: Uninstall (if requested)
# ---------------------------------------------------------------------------
if [ "$UNINSTALL" -eq 1 ]; then
    step "Removing RAGIX integration..."

    if [ "$DRY_RUN" -eq 1 ]; then
        dry "Would remove mcpServers.ragix-memory from $SETTINGS_LOCAL"
        dry "Would remove mcpServers.ragix-mcp from $SETTINGS_LOCAL"
        dry "Would remove hooks entries from $SETTINGS_PROJECT"
    else
        # Remove MCP entries
        if [ -f "$SETTINGS_LOCAL" ]; then
            backup_file "$SETTINGS_LOCAL"
            json_remove_keys "$SETTINGS_LOCAL" "
mcp = data.get('mcpServers', {})
mcp.pop('ragix-memory', None)
mcp.pop('ragix-mcp', None)
if not mcp:
    data.pop('mcpServers', None)
"
            info "Removed MCP server entries from settings.local.json"
        fi

        # Remove hooks from settings.json
        if [ -f "$SETTINGS_PROJECT" ]; then
            backup_file "$SETTINGS_PROJECT"
            json_remove_keys "$SETTINGS_PROJECT" "
hooks = data.get('hooks', {})
for event in list(hooks.keys()):
    if isinstance(hooks[event], list):
        hooks[event] = [h for h in hooks[event]
                        if not (isinstance(h, dict) and 'ragix' in json.dumps(h).lower())]
        if not hooks[event]:
            del hooks[event]
if not hooks:
    data.pop('hooks', None)
"
            info "Removed RAGIX hooks from settings.json"
        fi
    fi

    info "Uninstall complete. Slash commands in .claude/commands/ are preserved (they're in the repo)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 1: Prerequisites check
# ---------------------------------------------------------------------------
step "Phase 1: Prerequisites check"
PREREQ_OK=1

# Python 3.10+
if command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
        info "Python $PY_VER"
    else
        warn "Python $PY_VER found (3.10+ recommended)"
    fi
else
    err "Python 3 not found"; PREREQ_OK=0
fi

# ragix-memory CLI
if command -v ragix-memory >/dev/null 2>&1; then
    info "ragix-memory CLI available"
else
    warn "ragix-memory not in PATH (install with: pip install -e .)"
fi

# Ollama (optional)
if command -v ollama >/dev/null 2>&1; then
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        info "Ollama running"
    else
        warn "Ollama installed but not running (start with: ollama serve)"
    fi
else
    warn "Ollama not found (optional — needed for main MCP server)"
fi

# .claude/ directory
if [ -d "$PROJECT_ROOT/.claude" ]; then
    info ".claude/ directory exists"
else
    err ".claude/ directory not found — is this a Claude Code project?"
    PREREQ_OK=0
fi

[ "$PREREQ_OK" -eq 0 ] && { err "Prerequisites check failed. Aborting."; exit 1; }
echo ""

# ---------------------------------------------------------------------------
# Phase 2: MCP Server registration
# ---------------------------------------------------------------------------
if [ "$HOOKS_ONLY" -eq 0 ] && [ "$NO_MCP" -eq 0 ]; then
    step "Phase 2: MCP Server registration (profile=$PROFILE)"

    # Memory MCP server (always in both profiles)
    MEMORY_MCP='{"mcpServers":{"ragix-memory":{"command":"ragix-memory","args":["serve","--db","memory.db","--fts-tokenizer","fr"]}}}'

    if [ "$DRY_RUN" -eq 1 ]; then
        dry "Would add ragix-memory to mcpServers in $SETTINGS_LOCAL"
        if [ "$PROFILE" = "full" ]; then
            dry "Would add ragix-mcp to mcpServers in $SETTINGS_LOCAL"
        fi
    else
        backup_file "$SETTINGS_LOCAL"
        json_merge "$SETTINGS_LOCAL" "$MEMORY_MCP"
        info "Registered ragix-memory MCP server (17 tools)"

        # Full profile: also register main MCP
        if [ "$PROFILE" = "full" ]; then
            MAIN_MCP='{"mcpServers":{"ragix-mcp":{"command":"python3","args":["-m","MCP.ragix_mcp_server"]}}}'
            json_merge "$SETTINGS_LOCAL" "$MAIN_MCP"
            info "Registered ragix-mcp MCP server (38 tools)"
        fi
    fi

    if [ "$PROFILE" != "full" ]; then
        echo ""
        info "Main RAGIX MCP server (38 tools) not registered by default."
        info "To enable: bash scripts/install_claude.sh --profile full"
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Phase 3: Hooks installation
# ---------------------------------------------------------------------------
if [ "$MCP_ONLY" -eq 0 ] && [ "$NO_HOOKS" -eq 0 ]; then
    step "Phase 3: Hooks installation"

    # Verify hook scripts exist
    for hook in ragix_safety_guard.sh ragix_memory_inject.sh ragix_audit_logger.sh; do
        if [ ! -x "$HOOKS_DIR/$hook" ]; then
            err "Hook script not found or not executable: $HOOKS_DIR/$hook"
            exit 1
        fi
    done

    HOOKS_JSON=$(cat <<'HOOKS_EOF'
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$PROJECT_ROOT/scripts/hooks/ragix_safety_guard.sh"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "$PROJECT_ROOT/scripts/hooks/ragix_memory_inject.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "$PROJECT_ROOT/scripts/hooks/ragix_audit_logger.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "[RAGIX] If any code files were modified in this session, remind the user to run tests before committing (pytest or the relevant test suite). List the files that changed."
          }
        ]
      }
    ]
  }
}
HOOKS_EOF
    )

    # Substitute $PROJECT_ROOT in the JSON
    HOOKS_JSON=$(echo "$HOOKS_JSON" | sed "s|\\\$PROJECT_ROOT|$PROJECT_ROOT|g")

    if [ "$DRY_RUN" -eq 1 ]; then
        dry "Would install 4 hooks into $SETTINGS_PROJECT:"
        dry "  PreToolUse:Bash        -> ragix_safety_guard.sh"
        dry "  UserPromptSubmit:*     -> ragix_memory_inject.sh"
        dry "  PostToolUse:Bash|Write|Edit -> ragix_audit_logger.sh"
        dry "  Stop:*                 -> test reminder (prompt)"
    else
        # Create or update settings.json (project-level, not local)
        if [ ! -f "$SETTINGS_PROJECT" ]; then
            echo '{}' > "$SETTINGS_PROJECT"
        fi
        backup_file "$SETTINGS_PROJECT"
        json_merge "$SETTINGS_PROJECT" "$HOOKS_JSON"
        info "Installed 4 hooks:"
        info "  PreToolUse:Bash        -> safety guard"
        info "  UserPromptSubmit:*     -> memory inject"
        info "  PostToolUse:Bash|W|E   -> audit logger"
        info "  Stop:*                 -> test reminder"
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Phase 4: Slash commands verification
# ---------------------------------------------------------------------------
step "Phase 4: Slash commands verification"

EXPECTED_COMMANDS=(
    memory memory-add memory-search memory-recall
    koas-audit koas-kernels koas-report-fr koas-status
    ragix-system ragix-models
)
MISSING=0
for cmd in "${EXPECTED_COMMANDS[@]}"; do
    if [ -f "$COMMANDS_DIR/$cmd.md" ]; then
        [ "$VERBOSE" -eq 1 ] && info "/$cmd"
    else
        warn "Missing: .claude/commands/$cmd.md"
        MISSING=$((MISSING + 1))
    fi
done
FOUND=$(( ${#EXPECTED_COMMANDS[@]} - MISSING ))
info "$FOUND/${#EXPECTED_COMMANDS[@]} slash commands present"
echo ""

# ---------------------------------------------------------------------------
# Phase 5: Manifest generation
# ---------------------------------------------------------------------------
step "Phase 5: Generating RAGIX_COMPONENTS.md"

VERSION=$(python3 -c "
try:
    import tomllib
    with open('$PROJECT_ROOT/pyproject.toml', 'rb') as f:
        print(tomllib.load(f)['project']['version'])
except Exception:
    print('0.69.0')
" 2>/dev/null || echo "0.69.0")

if [ "$DRY_RUN" -eq 1 ]; then
    dry "Would generate $MANIFEST (v$VERSION)"
else
    cat > "$MANIFEST" <<MANIFEST_EOF
# RAGIX Components Reference -- v$VERSION

> Auto-generated by \`scripts/install_claude.sh\` on $(date -u +%Y-%m-%dT%H:%M:%SZ)
> Re-run the installer to refresh this file.

---

## Slash Commands (${#EXPECTED_COMMANDS[@]})

| Command | Description | Requires |
|---------|-------------|----------|
| /memory | Master memory command (7 subcommands) | Memory MCP |
| /memory-add | Store findings with tags (alias) | Memory MCP |
| /memory-search | FTS5 + hybrid search (alias) | Memory MCP |
| /memory-recall | Token-budgeted context injection (alias) | Memory MCP |
| /koas-audit | Run complete KOAS audit on a project | Main MCP |
| /koas-kernels | List available KOAS kernels | Main MCP |
| /koas-report-fr | KOAS audit with French report | Main MCP |
| /koas-status | Check KOAS workspace status | Main MCP |
| /ragix-system | System info for deployment assessment | Main MCP |
| /ragix-models | List/manage Ollama models | Main MCP |

## MCP Servers (2)

| Server | ID | Tools | Command | Profile |
|--------|----|-------|---------|---------|
| Memory | ragix-memory | 17 | \`ragix-memory serve --db memory.db\` | memory (default) |
| Main | ragix-mcp | 38 | \`python3 -m MCP.ragix_mcp_server\` | full (opt-in) |

### Memory MCP Tools (17)

| Tool | Purpose |
|------|---------|
| memory_recall | Token-budgeted retrieval for context injection |
| memory_search | FTS5 + hybrid scoring search |
| memory_propose | Governed write path (policy validation) |
| memory_write | Direct write (bypass governance) |
| memory_read | Read a memory item by ID |
| memory_update | Update an existing item |
| memory_link | Create semantic links between items |
| memory_consolidate | Dedup, merge, tier promotion cycle |
| memory_stats | Store statistics |
| memory_palace_list | List memory palaces |
| memory_palace_get | Get palace contents |
| memory_session_inject | Inject context into session |
| memory_session_store | Store session context |
| memory_workspace_list | List registered workspaces |
| memory_workspace_register | Register a new workspace |
| memory_workspace_remove | Remove a workspace |
| memory_metrics | Tool call metrics and performance |

### Main MCP Tools (38)

| Tool | Category | Purpose |
|------|----------|---------|
| ragix_chat | Core | Chat with local LLM |
| ragix_scan_repo | Core | Scan and index a repository |
| ragix_read_file | Core | Read file contents |
| ragix_search | Core | Hybrid search (RRF) |
| ragix_workflow | Core | Execute workflow templates |
| ragix_health | Core | System health check |
| ragix_templates | Core | List workflow templates |
| ragix_config | Core | Show configuration |
| ragix_verify_logs | Core | Verify audit logs integrity |
| ragix_logs | Core | View recent command logs |
| ragix_agent_step | Core | Single agent reasoning step |
| ragix_models_list | Models | List available Ollama models |
| ragix_model_info | Models | Model details and parameters |
| ragix_system_info | System | System info (CPU, GPU, RAM) |
| ragix_ast_scan | AST | Scan codebase AST structure |
| ragix_ast_metrics | AST | Compute code metrics |
| koas_init | KOAS | Initialize audit workspace |
| koas_run | KOAS | Execute audit pipeline |
| koas_status | KOAS | Audit workspace status |
| koas_summary | KOAS | Stage summary report |
| koas_list_kernels | KOAS | List available kernels |
| koas_report | KOAS | Generate audit report |
| koas_security_discover | Security | Network discovery scan |
| koas_security_scan_ports | Security | Port scanning |
| koas_security_ssl_check | Security | SSL/TLS certificate check |
| koas_security_vuln_scan | Security | Vulnerability scanning |
| koas_security_dns_check | Security | DNS configuration check |
| koas_security_compliance | Security | Compliance assessment |
| koas_security_risk | Security | Risk scoring |
| koas_security_report | Security | Security report generation |
| koas_audit_scan | Audit | Code audit scan |
| koas_audit_metrics | Audit | Code quality metrics |
| koas_audit_hotspots | Audit | Complexity hotspots |
| koas_audit_dependencies | Audit | Dependency analysis |
| koas_audit_dead_code | Audit | Dead code detection |
| koas_audit_risk | Audit | Risk assessment |
| koas_audit_compliance | Audit | Standards compliance |
| koas_audit_report | Audit | Audit report generation |

## Hooks (4)

| Hook | Event | Matcher | Type | Script | Purpose |
|------|-------|---------|------|--------|---------|
| Safety guard | PreToolUse | Bash | command | ragix_safety_guard.sh | Block dangerous shell commands |
| Memory inject | UserPromptSubmit | * | command | ragix_memory_inject.sh | Auto-recall relevant memory |
| Audit logger | PostToolUse | Bash\|Write\|Edit | command | ragix_audit_logger.sh | Log tool actions |
| Stop validator | Stop | * | prompt | (inline) | Remind to test if code changed |

### Safety Guard Patterns

Mirrors \`unix-rag-agent.py:71-113\` denylist:
- System destruction: \`rm -rf /\`, \`mkfs.\`, \`dd if=\`, \`shutdown\`, \`reboot\`
- Privilege escalation: \`sudo\`, \`su -\`, \`su root\`
- Pipe-to-shell: \`curl|sh\`, \`wget|bash\`
- Git destructive: \`git reset --hard\`, \`git clean -fd\`, \`git push --force\`

## CLI Entry Points (10)

| Command | Module | Purpose |
|---------|--------|---------|
| ragix | ragix_core.cli | Main RAGIX CLI |
| ragix-unix-agent | ragix_unix.cli | Unix-RAG interactive agent |
| ragix-web | ragix_web.server | Web UI server |
| ragix-index | ragix_unix.index_cli | Repository indexing |
| ragix-batch | ragix_unix.batch_cli | Batch processing |
| ragix-vault | ragix_unix.vault_cli | Secret vault management |
| ragix-wasp | ragix_unix.wasp_cli | WASP protocol handler |
| ragix-ast | ragix_unix.ast_cli | AST analysis CLI |
| ragix-koas | ragix_kernels.orchestrator | KOAS audit orchestrator |
| ragix-memory | ragix_core.memory.cli | Memory CLI (19 subcommands) |

## Agent Modes (3)

| Mode | Planner | Worker | Verifier | Use Case |
|------|---------|--------|----------|----------|
| MINIMAL | session model | session model | session model | Low VRAM / CPU-only |
| STRICT | mistral:7b+ | granite3.1-moe:3b | granite3.1-moe:3b | Balanced quality/speed |
| CUSTOM | user-defined | user-defined | user-defined | Full control |

## Quick Start

\`\`\`bash
# Install with default profile (memory MCP + hooks)
bash scripts/install_claude.sh

# Install with all MCP servers
bash scripts/install_claude.sh --profile full

# Preview without modifying
bash scripts/install_claude.sh --dry-run

# Remove RAGIX integration
bash scripts/install_claude.sh --uninstall
\`\`\`

## Verification

\`\`\`bash
# Check settings validity
python3 -m json.tool .claude/settings.local.json
python3 -m json.tool .claude/settings.json

# Test safety guard
echo '{"tool_input":{"command":"rm -rf /"}}' | bash scripts/hooks/ragix_safety_guard.sh
# Expected: exit 2 (blocked)

echo '{"tool_input":{"command":"ls -la"}}' | bash scripts/hooks/ragix_safety_guard.sh
# Expected: exit 0 (allowed)

# Test audit logger
echo '{"tool_name":"Bash","tool_input":{"command":"ls"}}' | bash scripts/hooks/ragix_audit_logger.sh
cat .agent_logs/commands.log
\`\`\`
MANIFEST_EOF
    info "Generated $MANIFEST (v$VERSION)"
fi
echo ""

# ---------------------------------------------------------------------------
# Phase 6: Verification
# ---------------------------------------------------------------------------
step "Phase 6: Verification"

if [ "$DRY_RUN" -eq 0 ]; then
    # Verify JSON validity
    if [ -f "$SETTINGS_LOCAL" ]; then
        if python3 -m json.tool "$SETTINGS_LOCAL" >/dev/null 2>&1; then
            info "settings.local.json: valid JSON"
        else
            err "settings.local.json: INVALID JSON — check backup and restore"
        fi
    fi
    if [ -f "$SETTINGS_PROJECT" ]; then
        if python3 -m json.tool "$SETTINGS_PROJECT" >/dev/null 2>&1; then
            info "settings.json: valid JSON"
        else
            err "settings.json: INVALID JSON — check backup and restore"
        fi
    fi
fi

# Summary
echo ""
echo "============================================"
printf "${BOLD}Installation Summary${NC}\n"
echo "============================================"
if [ "$NO_MCP" -eq 0 ] && [ "$HOOKS_ONLY" -eq 0 ]; then
    info "MCP: ragix-memory registered (profile=$PROFILE)"
    [ "$PROFILE" = "full" ] && info "MCP: ragix-mcp registered"
fi
if [ "$NO_HOOKS" -eq 0 ] && [ "$MCP_ONLY" -eq 0 ]; then
    info "Hooks: 4 installed (safety, memory, audit, stop)"
fi
info "Commands: $FOUND/${#EXPECTED_COMMANDS[@]} slash commands present"
info "Manifest: RAGIX_COMPONENTS.md generated"
echo ""
info "Restart your Claude Code session to activate changes."
echo ""
