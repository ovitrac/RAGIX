"""
Profile Definitions and Safety Policies for RAGIX

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from enum import Enum
from typing import List, Dict, Any, Optional


class Profile(str, Enum):
    """
    Execution profiles for RAGIX agents.

    - STRICT: read-only, no writes, no shell mutations
    - DEV: default development mode with strong safety policies
    - UNSAFE: expert/workbench mode, minimal restrictions (use with caution)
    """
    STRICT = "strict"  # Also known as "safe-read-only"
    DEV = "dev"
    UNSAFE = "unsafe"


# Hard safety denylist (never allowed in any profile)
DANGEROUS_PATTERNS: List[str] = [
    # System destruction
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+\.\s*$",
    r"mkfs\.",
    r"dd\s+if=",
    r"shutdown\b",
    r"reboot\b",
    r":\s*(){",  # fork-bomb style

    # Package managers (v0.32.1: prevent unauthorized installs)
    r"\bpip\s+install\b",
    r"\bpip3\s+install\b",
    r"\bconda\s+install\b",
    r"\bnpm\s+install\b",
    r"\byarn\s+add\b",
    r"\bapt\s+install\b",
    r"\bapt-get\s+install\b",
    r"\byum\s+install\b",
    r"\bdnf\s+install\b",
    r"\bbrew\s+install\b",
    r"\bcargo\s+install\b",
    r"\bgem\s+install\b",

    # Sudo/privilege escalation
    r"\bsudo\b",
    r"\bsu\s+-\b",
    r"\bsu\s+root\b",

    # Network exfiltration
    r"\bcurl\b.*\|\s*sh",
    r"\bwget\b.*\|\s*sh",
    r"\bcurl\b.*\|\s*bash",
    r"\bwget\b.*\|\s*bash",
]

# Git-destructive commands: blocked unless explicitly allowed
GIT_DESTRUCTIVE_PATTERNS: List[str] = [
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-fd",
    r"git\s+push\s+--force",
    r"git\s+push\s+-f",
]


def compute_dry_run_from_profile(profile: str, default_dry_run: bool = False) -> bool:
    """
    Decide dry-run behavior based on profile.

    - strict (safe-read-only): always dry-run True
    - dev: use default_dry_run parameter
    - unsafe: use default_dry_run parameter

    Args:
        profile: Profile name (strict/dev/unsafe)
        default_dry_run: Default dry-run behavior for non-strict profiles

    Returns:
        bool: Whether to enable dry-run mode
    """
    if profile in ("strict", "safe-read-only"):
        return True
    elif profile in ("dev", "unsafe"):
        return default_dry_run
    else:
        # Unknown profile: default to "dev" semantics
        return default_dry_run


def get_profile_restrictions(profile: str) -> Dict[str, Any]:
    """
    Get profile-specific restrictions and permissions.

    Returns a dictionary describing what operations are allowed/blocked
    for the given profile.

    Args:
        profile: Profile name (strict/dev/unsafe)

    Returns:
        Dict with keys:
        - allow_writes: Can write to files
        - allow_shell_mutations: Can run commands that modify system state
        - allow_git_destructive: Can run destructive git commands
        - enforce_denylist: Whether to enforce DANGEROUS_PATTERNS
        - description: Human-readable description
    """
    restrictions = {
        "strict": {
            "allow_writes": False,
            "allow_shell_mutations": False,
            "allow_git_destructive": False,
            "enforce_denylist": True,
            "description": "Read-only mode: no writes, no shell mutations, all operations are dry-run"
        },
        "dev": {
            "allow_writes": True,
            "allow_shell_mutations": True,
            "allow_git_destructive": False,
            "enforce_denylist": True,
            "description": "Development mode: writes allowed, git destructive blocked, strong safety checks"
        },
        "unsafe": {
            "allow_writes": True,
            "allow_shell_mutations": True,
            "allow_git_destructive": False,  # Still requires explicit flag
            "enforce_denylist": True,
            "description": "Expert mode: minimal restrictions, use with caution"
        }
    }

    normalized = profile.lower()
    if normalized in ("strict", "safe-read-only"):
        return restrictions["strict"]
    elif normalized == "dev":
        return restrictions["dev"]
    elif normalized == "unsafe":
        return restrictions["unsafe"]
    else:
        # Unknown profile: default to dev semantics
        return restrictions["dev"]


def merge_denylist_from_config(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Merge base denylist with user-configured patterns from config file.

    The config file can contain:
    ```toml
    [unix_agent.safety]
    additional_denylist = ["pattern1", "pattern2"]
    ```

    Args:
        config: Configuration dictionary (from TOML/YAML)

    Returns:
        Combined list of denylist patterns
    """
    base_patterns = DANGEROUS_PATTERNS.copy()

    if config is None:
        return base_patterns

    # Extract additional patterns from config
    safety_config = config.get("unix_agent", {}).get("safety", {})
    additional = safety_config.get("additional_denylist", [])

    if not isinstance(additional, list):
        return base_patterns

    # Filter for valid regex patterns (strings only)
    valid_additional = [p for p in additional if isinstance(p, str) and p.strip()]

    return base_patterns + valid_additional
