# RAGIX Profile Behavior Matrix

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24

---

## Overview

RAGIX supports three execution profiles with different safety/productivity trade-offs:

- **`strict`** (safe-read-only): Maximum safety, read-only operations
- **`dev`** (default): Balanced development mode with strong safety checks
- **`unsafe`**: Expert workbench mode with minimal restrictions

This document defines the exact behavior for each profile.

---

## Profile Comparison Matrix

| Feature | `strict` | `dev` | `unsafe` |
|---------|----------|-------|----------|
| **File Writes** | ❌ Blocked | ✅ Allowed | ✅ Allowed |
| **Shell Mutations** | ❌ Blocked | ✅ Allowed | ✅ Allowed |
| **Git Destructive Commands** | ❌ Blocked | ❌ Blocked | ⚠️ Requires `--allow-git-destructive` |
| **Dangerous Patterns** | ✅ Enforced | ✅ Enforced | ✅ Enforced |
| **Dry-Run Mode** | 🔒 Always ON | ⚙️ Configurable | ⚙️ Configurable |
| **Use Case** | Exploration, audits | Daily development | Debugging, recovery |

---

## Profile Details

### Profile: `strict` (safe-read-only)

**Philosophy:** Absolute safety. No modifications to the filesystem or repository state.

**Restrictions:**
- ✅ **Read operations:** `ls`, `find`, `grep`, `cat`, `head`, `tail`, `git log`, `git diff`
- ❌ **Write operations:** Blocked (all writes are dry-run)
- ❌ **Shell mutations:** No command that modifies state
- ❌ **Git mutations:** `git add`, `git commit`, `git push` all blocked
- 🔒 **Dry-run:** Always enabled (cannot be overridden)

**Typical Commands Allowed:**
```bash
ls -la
find . -name "*.py" | head -n 50
grep -R -n "function_name" .
git log --oneline -n 20
git diff HEAD~1
wc -l src/*.py
```

**Typical Commands Blocked:**
```bash
rm file.txt                # Blocked (write)
echo "test" > file.txt     # Blocked (write)
git add .                  # Blocked (git mutation)
mkdir new_dir              # Blocked (write)
```

**When to Use:**
- Initial project exploration
- Security audits
- Read-only code reviews
- Working in production environments
- Untrusted codebases

---

### Profile: `dev` (default)

**Philosophy:** Balanced mode for daily development. Allows normal workflows with strong safety guardrails.

**Restrictions:**
- ✅ **Read operations:** All allowed
- ✅ **Write operations:** Allowed (create, edit, delete files)
- ✅ **Shell mutations:** Normal commands (mkdir, cp, mv, etc.)
- ✅ **Git safe commands:** `git add`, `git commit`, `git push`
- ❌ **Git destructive:** `git reset --hard`, `git clean -fd`, `git push --force` blocked
- ❌ **Dangerous patterns:** Hard denylist enforced (rm -rf /, mkfs, dd, shutdown, etc.)
- ⚙️ **Dry-run:** Configurable (default OFF)

**Typical Commands Allowed:**
```bash
echo "new content" > file.txt
mkdir -p src/new_module
git add .
git commit -m "feature: add new module"
git push origin feature-branch
python -m pytest tests/
```

**Typical Commands Blocked:**
```bash
rm -rf /                   # Blocked (dangerous pattern)
git reset --hard HEAD~1    # Blocked (git destructive)
git clean -fd              # Blocked (git destructive)
git push --force           # Blocked (git destructive)
dd if=/dev/zero of=/dev/sda # Blocked (dangerous pattern)
```

**When to Use:**
- Daily development work
- Feature implementation
- Refactoring existing code
- Running tests
- Normal git workflows

---

### Profile: `unsafe`

**Philosophy:** Expert mode for debugging and recovery operations. Use with caution.

**Restrictions:**
- ✅ **Read operations:** All allowed
- ✅ **Write operations:** All allowed
- ✅ **Shell mutations:** All allowed
- ✅ **Git safe commands:** All allowed
- ⚠️ **Git destructive:** Requires explicit `--allow-git-destructive` flag
- ❌ **Dangerous patterns:** Still enforced (last line of defense)
- ⚙️ **Dry-run:** Configurable (default OFF)

**Typical Commands Allowed:**
```bash
# All dev commands plus:
# (with --allow-git-destructive flag)
git reset --hard HEAD~1
git clean -fd
git push --force origin feature-branch
```

**Typical Commands Still Blocked:**
```bash
rm -rf /                   # Blocked (dangerous pattern - always enforced)
mkfs.ext4 /dev/sda         # Blocked (dangerous pattern - always enforced)
shutdown -h now            # Blocked (dangerous pattern - always enforced)
```

**When to Use:**
- Emergency debugging
- Repository recovery operations (rebase, reset, etc.)
- Advanced git workflows
- Experimental/throwaway work
- Expert users only

**⚠️ Warning:** This profile allows potentially destructive operations. Use only when necessary and understand the risks.

---

## Configuration

### Setting Profile via CLI

```bash
# Use strict profile (read-only)
ragix-unix-agent --profile strict

# Use dev profile (default)
ragix-unix-agent --profile dev

# Use unsafe profile (expert mode)
ragix-unix-agent --profile unsafe

# Unsafe + allow git destructive commands
ragix-unix-agent --profile unsafe --allow-git-destructive
```

### Setting Profile via Environment Variable

```bash
export UNIX_RAG_PROFILE=strict
ragix-unix-agent

export UNIX_RAG_PROFILE=dev
ragix-unix-agent

export UNIX_RAG_PROFILE=unsafe
export UNIX_RAG_ALLOW_GIT_DESTRUCTIVE=1
ragix-unix-agent
```

### Setting Profile via Config File

**`~/.ragix.toml`:**
```toml
[unix_agent]
profile = "dev"                 # strict / dev / unsafe
allow_git_destructive = false   # true to allow (only in unsafe profile)
```

### Config Precedence

Configuration is resolved in this order (highest to lowest priority):

1. **CLI arguments** (`--profile`, `--allow-git-destructive`)
2. **Environment variables** (`UNIX_RAG_PROFILE`, `UNIX_RAG_ALLOW_GIT_DESTRUCTIVE`)
3. **Config file** (`~/.ragix.toml`)
4. **Defaults** (profile: `dev`, git-destructive: `false`)

---

## Dangerous Patterns Denylist

The following patterns are **always blocked** in all profiles:

### Hard Safety Denylist

```regex
rm\s+-rf\s+/              # Delete root filesystem
rm\s+-rf\s+\.\s*$         # Delete current directory recursively
mkfs\.                    # Format filesystem
dd\s+if=                  # Raw disk write
shutdown\b                # System shutdown
reboot\b                  # System reboot
:\s*(){                   # Fork bomb
```

### Git Destructive Patterns (blocked in `strict`/`dev`, requires flag in `unsafe`)

```regex
git\s+reset\s+--hard      # Hard reset (loses uncommitted work)
git\s+clean\s+-fd         # Force clean (deletes untracked files)
git\s+push\s+--force      # Force push (rewrites remote history)
git\s+push\s+-f           # Force push (short form)
```

---

## Custom Denylist Extension

You can extend the denylist via config file:

**`~/.ragix.toml`:**
```toml
[unix_agent.safety]
additional_denylist = [
    "curl.*sudo.*bash",        # Block piped-to-shell downloads
    "wget.*\\|.*bash",          # Same for wget
    "eval.*\\$\\(",             # Block dynamic eval
]
```

These patterns are **merged** with the base denylist and enforced in all profiles.

---

## Profile Behavior Implementation

Profiles are implemented in `ragix_core/profiles.py`:

```python
from ragix_core import get_profile_restrictions

# Get restrictions for a profile
restrictions = get_profile_restrictions("dev")

print(restrictions["allow_writes"])           # True
print(restrictions["allow_git_destructive"])  # False
print(restrictions["enforce_denylist"])       # True
print(restrictions["description"])            # Human-readable description
```

Denylist merging:

```python
from ragix_core import merge_denylist_from_config

# Load config and merge denylists
config = load_config_from_toml("~/.ragix.toml")
full_denylist = merge_denylist_from_config(config)

# full_denylist = DANGEROUS_PATTERNS + config['unix_agent.safety.additional_denylist']
```

---

## Choosing the Right Profile

### Use `strict` when:
- ✅ Exploring unfamiliar codebases
- ✅ Performing security audits
- ✅ Working in production environments
- ✅ Read-only analysis required
- ✅ Maximum safety is priority

### Use `dev` when:
- ✅ Daily feature development
- ✅ Writing tests
- ✅ Normal refactoring
- ✅ Standard git workflows
- ✅ Balanced safety + productivity

### Use `unsafe` when:
- ⚠️ Debugging complex git issues
- ⚠️ Repository recovery needed
- ⚠️ Advanced git operations (rebase --interactive, etc.)
- ⚠️ You understand the risks
- ⚠️ Working in throwaway environments

---

## Example Workflows

### Workflow 1: Initial Exploration (strict)

```bash
# Start in strict profile
ragix-unix-agent --profile strict --sandbox-root ~/audit/project-x

# Agent can:
# - List files, search code
# - Read git history
# - Generate reports
# - Answer questions

# Agent cannot:
# - Modify any files
# - Run tests that write artifacts
# - Commit changes
```

### Workflow 2: Feature Development (dev)

```bash
# Start in dev profile (default)
ragix-unix-agent --sandbox-root ~/projects/my-app

# Agent can:
# - Create/edit files
# - Run tests
# - Git add/commit/push
# - Install dependencies

# Agent cannot:
# - Force-push branches
# - Hard reset commits
# - Run dangerous system commands
```

### Workflow 3: Git Recovery (unsafe + flag)

```bash
# Start in unsafe profile with git-destructive flag
ragix-unix-agent --profile unsafe \
                 --allow-git-destructive \
                 --sandbox-root ~/projects/broken-repo

# Agent can:
# - All dev operations
# - git reset --hard
# - git clean -fd
# - git push --force

# Agent still cannot:
# - rm -rf /
# - Format disks
# - Shutdown system
```

---

## Safety Philosophy

RAGIX follows a **defense-in-depth** approach:

1. **Profile-based restrictions** — Coarse-grained policy enforcement
2. **Denylist matching** — Pattern-based command blocking
3. **Sandbox root** — Filesystem boundary enforcement
4. **Logging & audit** — Complete command trace in `.agent_logs/`
5. **Dry-run mode** — Optional preview before execution

The goal is to enable productive agent-assisted development while maintaining control and auditability.

---

## References

- **Profile implementation:** `ragix_core/profiles.py`
- **CLI configuration:** `ragix_unix/cli.py`
- **Shell sandbox:** `ragix_core/tools_shell.py`
- **RAGIX architecture:** `README.md`
- **Quick start guide:** `QUICKSTART_CLAUDE_CODE.md`

---

**Last updated:** 2025-11-24
**Version:** RAGIX v0.5.0-dev
