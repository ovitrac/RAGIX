"""
Shell Sandbox and Command Execution for RAGIX

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

from .profiles import DANGEROUS_PATTERNS, GIT_DESTRUCTIVE_PATTERNS
from .logging_utils import AgentLogger


@dataclass
class CommandResult:
    """
    Container for the result of a shell command.
    """
    command: str
    cwd: str
    stdout: str
    stderr: str
    returncode: int

    def as_text_block(self) -> str:
        """
        Format the command result as a text block for feeding back into the LLM.
        """
        lines = [f"$ {self.command}", f"(cwd: {self.cwd})", ""]
        # Always show output section
        lines.append("Output:")
        if self.stdout.strip():
            lines.append(self.stdout.strip())
        else:
            lines.append("(no output)")
        if self.stderr:
            lines.append("")
            lines.append("STDERR:")
            lines.append(self.stderr.strip())
        lines.append("")
        lines.append(f"Return code: {self.returncode}")
        return "\n".join(lines)


class ShellSandbox:
    """
    ShellSandbox
    ------------

    A controlled shell environment that:

    - Confines execution to a sandbox root directory
    - Applies hard safety denylist (DANGEROUS_PATTERNS)
    - Optionally blocks destructive git commands (GIT_DESTRUCTIVE_PATTERNS)
    - Supports dry-run mode (no actual execution)
    - Logs all commands and results
    """

    def __init__(
        self,
        root: str,
        dry_run: bool,
        profile: str,
        allow_git_destructive: bool = False
    ):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

        self.dry_run = dry_run
        self.profile = profile
        self.allow_git_destructive = allow_git_destructive

        # Setup logging
        log_dir = Path(self.root) / ".agent_logs"
        self.logger = AgentLogger(log_dir)

    # ------------------------ Safety & Logging ------------------------

    def _matches_any(self, cmd: str, patterns: list[str]) -> bool:
        """Check if command matches any pattern in the list."""
        return any(re.search(p, cmd) for p in patterns)

    def _is_blocked_by_policy(self, cmd: str) -> Tuple[bool, str]:
        """
        Check if a command is blocked by either the hard denylist or
        the git-destructive policy.

        Returns:
            (is_blocked, reason)
        """
        if self._matches_any(cmd, DANGEROUS_PATTERNS):
            # Show what was blocked (truncate long commands)
            cmd_preview = cmd[:80] + "..." if len(cmd) > 80 else cmd
            return True, f"Blocked by hard safety policy: `{cmd_preview}`"

        # Git destructive commands: only allowed if explicitly confirmed at startup
        if self._matches_any(cmd, GIT_DESTRUCTIVE_PATTERNS):
            if not self.allow_git_destructive:
                cmd_preview = cmd[:80] + "..." if len(cmd) > 80 else cmd
                return True, f"Blocked by git-destructive policy: `{cmd_preview}`"

        return False, ""

    def log_command_result(self, result: CommandResult):
        """
        Log a shell command result to the agent log.

        Args:
            result: CommandResult instance
        """
        self.logger.log_command(result.command, result.cwd, result.returncode)

    def log_event(self, event: str, details: str = ""):
        """
        Log arbitrary events (e.g. file edits) as pseudo-commands.

        Args:
            event: Event type
            details: Optional event details
        """
        self.logger.log_event(event, details)

    # ------------------------ Command Execution ------------------------

    def run(self, cmd: str, timeout: int = 60) -> CommandResult:
        """
        Execute a shell command inside the sandbox.

        Args:
            cmd: Shell command to execute
            timeout: Maximum execution time in seconds

        Returns:
            CommandResult with stdout, stderr, and return code
        """
        cmd = cmd.strip()
        if not cmd:
            result = CommandResult(cmd, self.root, "", "Empty command", 1)
            self.log_command_result(result)
            return result

        # Check safety policies
        blocked, reason = self._is_blocked_by_policy(cmd)
        if blocked:
            result = CommandResult(cmd, self.root, "", reason, 1)
            self.log_command_result(result)
            return result

        # Dry-run mode
        if self.dry_run:
            result = CommandResult(cmd, self.root, "[DRY RUN: command not executed]", "", 0)
            self.log_command_result(result)
            return result

        # Execute command
        try:
            cp = subprocess.run(
                cmd,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=os.setsid,
            )
            result = CommandResult(cmd, self.root, cp.stdout, cp.stderr, cp.returncode)
            self.log_command_result(result)
            return result
        except subprocess.TimeoutExpired:
            result = CommandResult(cmd, self.root, "", "Timeout", 124)
            self.log_command_result(result)
            return result
