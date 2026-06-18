#!/usr/bin/env bash
# ragix_test_reminder.sh — Stop hook for Claude Code
# Reminds the user to run tests before committing IF code files changed.
#
# Replaces the previous `type: "prompt"` Stop hook, which had no loop guard:
# a prompt-type Stop hook re-fires on every turn end forever. A command-type
# Stop hook receives `stop_hook_active` on stdin and uses it to break the loop.
#
# Protocol:
#   - reads hook JSON from stdin (stop_hook_active, cwd, ...)
#   - if stop_hook_active is true  -> exit 0 (already in a continuation; do not
#                                      re-trigger -> breaks the loop)
#   - else detect changed code files via `git status --porcelain`
#       - none changed -> exit 0 (stop normally, silent)
#       - some changed -> emit {"decision":"block","reason":"..."} so the
#                         reminder surfaces exactly once, then stops next time
#
# Gitignored paths (e.g. base/) never appear in `git status`, so draft docs do
# not trigger the reminder.
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOOK_INPUT="$(cat)"

export PROJECT_ROOT HOOK_INPUT

python3 <<'PYEOF'
import sys, os, json, subprocess

# 1. Loop guard: if already continuing because of a Stop hook, stop now.
try:
    data = json.loads(os.environ.get("HOOK_INPUT") or "{}")
except Exception:
    data = {}
if data.get("stop_hook_active"):
    sys.exit(0)

root = os.environ.get("PROJECT_ROOT", ".")

# 2. Find changed (tracked + staged + untracked) files via git.
try:
    out = subprocess.run(
        ["git", "-C", root, "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, timeout=10,
    ).stdout
except Exception:
    sys.exit(0)  # no git / error -> never block

CODE_EXT = (
    ".py", ".sh", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
)

changed = []
for line in out.splitlines():
    if not line.strip():
        continue
    path = line[3:]                       # strip the XY status + space
    if " -> " in path:                    # rename: keep the destination
        path = path.split(" -> ", 1)[1]
    path = path.strip().strip('"')
    if path.endswith(CODE_EXT):
        changed.append(path)

if not changed:
    sys.exit(0)  # no code changed -> stop normally, no reminder

# 3. Surface the reminder once.
files = "\n".join("  - " + p for p in sorted(set(changed)))
reason = (
    "[RAGIX] Code files changed this session. Run the test suite before committing "
    "(e.g. `python -m pytest` or the relevant tests). Changed code files:\n" + files
)
print(json.dumps({"decision": "block", "reason": reason}))
sys.exit(0)
PYEOF
exit 0
