#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unix-RAG Agent (Extended Ollama Edition)
========================================

Overview
--------
This script implements a "Claude Code–style" local development assistant using:

- A local LLM served by Ollama (e.g. Mistral).
- A sandboxed Unix shell (bash) for project exploration/analysis.
- A Unix-RAG pattern: grep/head/tail/find/sed/awk/python for retrieval.
- Git-aware commands (status, diff, log, show, grep).
- A file-editing tool exposed via a JSON action: {"action": "edit_file", ...}.
- Structured logging of all executed commands and edit operations.
- Configurable profiles / modes ("safe-read-only", "dev", "unsafe").

The agent communicates with the LLM using a JSON action protocol:

- {"action": "bash", "command": "..."}
- {"action": "bash_and_respond", "command": "...", "message": "..."}
- {"action": "respond", "message": "..."}
- {"action": "edit_file", "path": "...", "old": "...", "new": "..."}

The model is instructed to use Unix tools for retrieval, and to remain within
a sandbox directory for safety.


Author: Olivier Vitrac | Adservio Innovation Lab | olivier.vitrac@adservio.fr
Contact: olivier.vitrac@adservio.fr

"""

import os
import re
import json
import signal
import textwrap
import subprocess
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# ---------------------------------------------------------------------------
# 0. GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------

# LLM model served by Ollama (e.g. "mistral", "mistral:instruct")
OLLAMA_MODEL = os.environ.get("UNIX_RAG_MODEL", "mistral")

# Sandbox root directory – all shell operations are confined here
SANDBOX_ROOT = os.path.abspath(os.environ.get("UNIX_RAG_SANDBOX", "~/agent-sandbox"))
SANDBOX_ROOT = os.path.expanduser(SANDBOX_ROOT)

# Default dry-run behavior in "dev" profile (can be overridden by profile)
DEFAULT_DRY_RUN = False

# Profiles / modes:
# - "safe-read-only": dry-run ON, very conservative behavior.
# - "dev": default; dry-run as per DEFAULT_DRY_RUN, strong denylist.
# - "unsafe": same as dev but allows git destructive commands if explicitly enabled.
AGENT_PROFILE = os.environ.get("UNIX_RAG_PROFILE", "dev").lower()

# Allow destructive git operations if explicitly confirmed via environment variable.
# This is interpreted as a coarse-grained "confirmation" at startup.
ALLOW_GIT_DESTRUCTIVE = os.environ.get("UNIX_RAG_ALLOW_GIT_DESTRUCTIVE", "0") == "1"

# Hard safety denylist (never allowed).
DANGEROUS_PATTERNS = [
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

# Git-destructive commands: blocked unless explicitly allowed.
GIT_DESTRUCTIVE_PATTERNS = [
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-fd",
    r"git\s+push\s+--force",
    r"git\s+push\s+-f",
]


# ---------------------------------------------------------------------------
# 1. LLM BACKEND (OLLAMA)
# ---------------------------------------------------------------------------

class OllamaLLM:
    """
    Simple wrapper around Ollama's /api/chat endpoint.

    Assumes Ollama is running locally:
        ollama serve

    The model is specified by OLLAMA_MODEL.
    """

    def __init__(self, model: str):
        self.model = model

    def generate(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Call Ollama's chat API with:
            - a system prompt
            - a history of messages (user/assistant)

        Returns the assistant's message as plain text.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        r = requests.post("http://localhost:11434/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"].strip()


# ---------------------------------------------------------------------------
# 2. SHELL SANDBOX WITH LOGGING
# ---------------------------------------------------------------------------

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
        if self.stdout:
            lines.append("STDOUT:")
            lines.append(self.stdout.strip())
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

    - Confines execution to a sandbox root directory.
    - Applies hard safety denylist (DANGEROUS_PATTERNS).
    - Optionally blocks destructive git commands (GIT_DESTRUCTIVE_PATTERNS).
    - Supports dry-run mode (no actual execution).
    - Logs all commands and results under SANDBOX_ROOT/.agent_logs/commands.log.
    """

    def __init__(self,
                 root: str,
                 dry_run: bool,
                 profile: str,
                 allow_git_destructive: bool):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

        self.dry_run = dry_run
        self.profile = profile
        self.allow_git_destructive = allow_git_destructive

        # Setup logging directory/file
        self.log_dir = os.path.join(self.root, ".agent_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "commands.log")

    # ------------------------ Safety & Logging ------------------------

    def _matches_any(self, cmd: str, patterns: List[str]) -> bool:
        return any(re.search(p, cmd) for p in patterns)

    def _is_blocked_by_policy(self, cmd: str) -> Tuple[bool, str]:
        """
        Check if a command is blocked by either the hard denylist or
        the git-destructive policy.
        """
        if self._matches_any(cmd, DANGEROUS_PATTERNS):
            return True, "Blocked by hard safety policy."

        # Git destructive commands: only allowed if explicitly confirmed at startup.
        if self._matches_any(cmd, GIT_DESTRUCTIVE_PATTERNS):
            if not self.allow_git_destructive:
                return True, "Blocked by git-destructive policy."
            # If allowed, we still let it through (caller is assumed to know).

        return False, ""

    def _log(self, line: str):
        """
        Append a single log line to the commands log, with timestamp.
        """
        timestamp = datetime.utcnow().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {line}\n")

    def log_command_result(self, result: CommandResult):
        """
        Structured logging of a shell command result:
        - timestamp
        - cwd
        - command
        - return code
        """
        summary = (
            f'CMD="{result.command}" '
            f"CWD='{result.cwd}' "
            f"RC={result.returncode}"
        )
        self._log(summary)

    def log_event(self, event: str, details: str = ""):
        """
        Log arbitrary events (e.g. file edits) as pseudo-commands.
        """
        line = f"EVENT={event}"
        if details:
            line += f" DETAILS={details}"
        self._log(line)

    # ------------------------ Command Execution ------------------------

    def run(self, cmd: str, timeout: int = 60) -> CommandResult:
        """
        Execute a shell command inside the sandbox.

        - Applies denylist policies.
        - Honors dry_run (unless you call subprocess directly yourself).
        - Logs every attempt.
        """
        cmd = cmd.strip()
        if not cmd:
            result = CommandResult(cmd, self.root, "", "Empty command", 1)
            self.log_command_result(result)
            return result

        blocked, reason = self._is_blocked_by_policy(cmd)
        if blocked:
            result = CommandResult(cmd, self.root, "", reason, 1)
            self.log_command_result(result)
            return result

        # In safe-read-only mode, dry_run is typically True.
        if self.dry_run:
            result = CommandResult(cmd, self.root, "[DRY RUN: command not executed]", "", 0)
            self.log_command_result(result)
            return result

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


# ---------------------------------------------------------------------------
# 3. JSON ACTION PROTOCOL & SYSTEM PROMPT
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from a text response.

    The model is instructed to respond with pure JSON, but this function
    is defensive in case extra text appears.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


AGENT_SYSTEM_PROMPT = textwrap.dedent("""
You are a local Unix-RAG development assistant working inside a sandboxed
project directory.

You can:
- Run bash commands: ls, find, tree, grep -R -n, head, tail, wc -l,
  sed -n, awk, cut, sort, uniq, python one-liners, etc.
- Explore git repositories: `git status -sb`, `git diff`, `git log --oneline --graph -n 20`,
  `git show <commit>`, `git grep <pattern>`.
- Edit files using a high-level "edit_file" action (described below).

IMPORTANT BEHAVIOR (Unix-RAG style):
- When exploring, start with listing and discovery:
  * `ls`, `find . -maxdepth ...`
  * `grep -R -n "keyword" .`
- Never dump entire large files:
  * Use `wc -l file` to gauge size.
  * Use `head`, `tail`, or `sed -n 'start,endp'` to show relevant parts.
- For logs and diagnostics:
  * Use `grep` + `head`/`tail` to extract a few examples.
  * Use `awk`, `python` scripts to summarize or aggregate.
- For git:
  * Prefer read-only commands (`status`, `diff`, `log`, `show`, `grep`).
  * Some destructive commands such as `git reset --hard`, `git clean -fd`,
    `git push --force` may be blocked by policy. If they fail with a message
    about policy, explain that the user must run them manually.

FILE EDITING TOOL:
- You have access to a high-level action:
    {"action": "edit_file", "path": "...", "old": "...", "new": "..."}
  This means:
    * Open the file at `path` (relative to the sandbox root).
    * Find the first occurrence of the exact text `old`.
    * Replace it with `new`.
    * Save the file.
    * Then, you should typically run `git diff -- <path>` (via a bash action)
      to show the changes, or at least explain what changed.

  Strategy for edits:
    - First inspect the file content using bash (grep/sed/head).
    - Only then propose an `edit_file` action with a precise `old` snippet.
    - After edit, use git diff or show the relevant lines again.

SWE-STYLE NAVIGATION & EDITING (RECOMMENDED FOR SYSTEMATIC CODEBASE WORK):
- For systematic codebase exploration and editing, use these commands via bash:

  NAVIGATION (100-line windows with 2-line overlap):
    rt open <path>                → view lines 1-100
    rt open <path>:<line>         → view 100-line window centered on <line>
    rt open <path>:<start>-<end>  → view explicit range (max 200 lines)
    rt scroll <path> +            → scroll down (with 2-line overlap)
    rt scroll <path> -            → scroll up

  SEARCH:
    rt grep-file "<pattern>" <path>  → search within single file (with line numbers)
    rt grep "<pattern>" [root]       → recursive search (existing tool)
    rt find <pattern> [root]         → find files by name (existing tool)

  EDITING (line-based, reads from stdin):
    rt edit <path> <start> <end> << 'EOF'
    <new text>
    EOF

    rt insert <path> <line> << 'EOF'
    <new text>
    EOF

- BEST PRACTICES:
  * Use `rt open path:line` to jump directly to relevant code (from grep results).
  * AVOID repeated `rt scroll` — prefer direct jumps when line number is known.
  * View windows are 100 lines with 2-line overlap between scrolls.
  * After edits, optionally run `git diff -- <path>` to verify changes.
  * Line numbers are 1-based (consistent with grep -n, sed -n).
  * All SWE commands respect the sandbox and create .bak backups.

- RECOMMENDED WORKFLOW:
  1. Discover: `rt find`, `rt grep` to locate relevant files
  2. Navigate: `rt open path:line` to jump to specific locations
  3. Edit: Use `rt edit`/`rt insert` for line-based changes (or fallback to JSON `edit_file`)
  4. Verify: `git diff` or `rt open` to review changes

SAFETY & SANDBOX:
- You must assume you are in a sandbox; stay within the repository.
- Disallowed commands (rm -rf /, mkfs, dd, shutdown, reboot, etc.) are blocked.
- Destructive git commands may be blocked; if so, do not try to circumvent this.
- Be cautious and incremental when editing files.

RESPONSE PROTOCOL (VERY IMPORTANT):
- You must ALWAYS respond with a single JSON object, and NOTHING else.
- Do NOT use markdown fences, backticks, or explanations outside JSON.
- Valid actions are:

  1) Run a bash command:
     {"action": "bash", "command": "<shell command>"}

  2) Respond with a natural language answer:
     {"action": "respond", "message": "<your explanation or summary>"}

  3) Run a command AND respond:
     {"action": "bash_and_respond",
      "command": "<shell command>",
      "message": "<what you plan / summary>"}

  4) Edit a file using the high-level API:
     {"action": "edit_file",
      "path": "<relative path>",
      "old": "<exact text to replace>",
      "new": "<replacement text>"}

- The JSON MUST be syntactically valid.
""").strip()


# ---------------------------------------------------------------------------
# 4. AGENT CLASS (PROJECT OVERVIEW + EDIT FILE TOOL)
# ---------------------------------------------------------------------------

@dataclass
class UnixRAGAgent:
    """
    UnixRAGAgent
    ------------

    Encapsulates:
    - the LLM (OllamaLLM),
    - the sandboxed shell (ShellSandbox),
    - the chat history,
    - the JSON action protocol.

    Additional behavior:
    - On initialization, it generates an automatic project overview using
      `find` (limited depth and file types) and feeds this as context to
      the conversation.
    """
    llm: OllamaLLM
    shell: ShellSandbox
    system_prompt: str = AGENT_SYSTEM_PROMPT
    history: List[Dict[str, str]] = field(default_factory=list)
    # V3.3: Optional persistent memory middleware (config-gated)
    memory_enabled: bool = False
    memory_config: Optional[Dict[str, Any]] = None
    _memory: Any = field(default=None, init=False, repr=False)
    _turn_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """
        After initialization, run a project overview command and store its
        result as part of the initial context.
        """
        # Wire memory middleware if enabled
        if self.memory_enabled:
            try:
                from ragix_core.memory.middleware import MemoryMiddleware
                from ragix_core.memory.tools import create_dispatcher
                from ragix_core.memory.config import MemoryConfig
                mem_cfg = MemoryConfig.from_dict(self.memory_config or {})
                dispatcher = create_dispatcher(mem_cfg)
                self._memory = MemoryMiddleware(dispatcher, mem_cfg)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Memory middleware init failed (continuing without memory): {e}"
                )
                self._memory = None
        overview_cmd = (
            "find . -maxdepth 4 -type f "
            "\\( -name '*.py' -o -name '*.ipynb' -o -name 'Makefile' "
            "-o -name 'CMakeLists.txt' \\) 2>/dev/null | head -n 200"
        )
        result = self.shell.run(overview_cmd)
        overview_text = result.stdout.strip()
        if overview_text:
            # Feed project structure as initial context from the "user".
            self.history.append({
                "role": "user",
                "content": (
                    "Initial project overview (file list sample):\n" +
                    overview_text
                )
            })

    # -------------------------- Core methods --------------------------

    def _add_message(self, role: str, content: str):
        """
        Append a message to the chat history.
        """
        self.history.append({"role": role, "content": content})

    def _edit_file(self, path: str, old: str, new: str) -> str:
        """
        Implement the high-level file editing tool:
        - Ensure the path is inside the sandbox.
        - Replace the first occurrence of `old` with `new`.
        - Log the edit event.
        - Return a textual summary of what happened (for the LLM/user).
        """
        # Resolve absolute path and enforce sandbox boundary
        abs_path = os.path.abspath(os.path.join(self.shell.root, path))
        if not abs_path.startswith(self.shell.root):
            msg = f"Refused to edit '{path}': outside sandbox boundary."
            self.shell.log_event("EDIT_FILE_BLOCKED", msg)
            return msg

        if not os.path.exists(abs_path):
            msg = f"File not found: {path}"
            self.shell.log_event("EDIT_FILE_ERROR", msg)
            return msg

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            msg = f"Could not read file '{path}': {e}"
            self.shell.log_event("EDIT_FILE_ERROR", msg)
            return msg

        idx = content.find(old)
        if idx == -1:
            msg = f"Pattern 'old' not found in file '{path}'. No edit applied."
            self.shell.log_event("EDIT_FILE_NO_MATCH", msg)
            return msg

        # Perform a single replacement
        new_content = content.replace(old, new, 1)

        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            msg = f"Could not write file '{path}': {e}"
            self.shell.log_event("EDIT_FILE_ERROR", msg)
            return msg

        # Log the edit action
        summary = f"Edited file '{path}': replaced first occurrence of given 'old' text."
        self.shell.log_event("EDIT_FILE_OK", summary)
        return summary

    def step(self, user_text: str) -> Tuple[Optional[CommandResult], Optional[str]]:
        """
        Perform a single reasoning/acting step for a given user input.

        Returns:
        - CommandResult or None (if no bash command executed)
        - Natural-language response or None
        """
        self._turn_id += 1

        # Hook 1: inject relevant memory into system context
        effective_prompt = self.system_prompt
        if self._memory:
            effective_prompt = self._memory.pre_call(
                user_text, self.system_prompt, turn_id=str(self._turn_id)
            )

        # Feed user message into history
        self._add_message("user", user_text)

        # Query the LLM for the next action
        raw = self.llm.generate(effective_prompt, self.history)
        self._add_message("assistant", raw)

        # Hook 2: parse proposals, govern, store accepted items
        if self._memory:
            raw, _summary = self._memory.post_call(
                raw, tool_calls=[], turn_id=str(self._turn_id)
            )

        # Parse JSON action
        action = extract_json_object(raw)
        if action is None:
            # Fallback: treat entire raw output as a message.
            cmd_result, response = None, raw
        else:
            kind = action.get("action")

            # -------- case: execute bash command only --------
            if kind == "bash":
                cmd = action.get("command", "")
                result = self.shell.run(cmd)
                self._add_message("user", "Command result:\n" + result.as_text_block())
                cmd_result, response = result, None

            # -------- case: bash + respond --------
            elif kind == "bash_and_respond":
                cmd = action.get("command", "")
                msg = action.get("message", "")
                result = self.shell.run(cmd)
                self._add_message("user", "Command result:\n" + result.as_text_block())
                cmd_result, response = result, msg or None

            # -------- case: respond only --------
            elif kind == "respond":
                msg = action.get("message", "")
                cmd_result, response = None, msg or None

            # -------- case: edit_file tool --------
            elif kind == "edit_file":
                path = action.get("path", "").strip()
                old = action.get("old", "")
                new = action.get("new", "")
                summary = self._edit_file(path, old, new)
                self._add_message("user", f"Edit_file result for '{path}': {summary}")
                cmd_result, response = None, summary

            # -------- unknown action fallback --------
            else:
                cmd_result, response = None, f"Unknown action kind: {kind!r}"

        # Hook 3: Q*-search recall pass before final response
        if self._memory and response:
            response, _catalog = self._memory.pre_return(
                user_text, response, turn_id=str(self._turn_id)
            )

        return cmd_result, response

    def interactive_loop(self):
        """
        Simple interactive REPL on the terminal.

        - Ctrl+D or empty line to exit.
        - Each user message triggers a step.
        """
        print("=============================================")
        print("Unix-RAG Agent (Ollama Edition)")
        print("---------------------------------------------")
        print(f"Sandbox root : {self.shell.root}")
        print(f"Profile      : {self.shell.profile}")
        print(f"Dry-run      : {self.shell.dry_run}")
        print(f"Model        : {self.llm.model}")
        print("=============================================\n")
        print("Type your request. Empty line to quit.\n")

        # (Optional) Inform the user that an initial overview has been captured.
        print("Initial project overview has been collected (if available).")
        print("You can start by asking, for example:")
        print("  - \"Summarize the structure of this project.\"")
        print("  - \"Search where a certain function or variable is defined.\"")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\n[EOF received, exiting]")
                break

            if not user_input:
                print("[Empty input, exiting]")
                break

            cmd_result, msg = self.step(user_input)

            if cmd_result is not None:
                print("\n[Command executed]")
                print(cmd_result.as_text_block())
                print("-" * 60)

            if msg is not None:
                print("\nAssistant:")
                print(msg)
                print("-" * 60)


# ---------------------------------------------------------------------------
# 5. MAIN ENTRY POINT (PROFILE HANDLING)
# ---------------------------------------------------------------------------

def compute_dry_run_from_profile(profile: str) -> bool:
    """
    Decide dry-run behavior based on profile.

    - safe-read-only: always dry-run True.
    - dev: use DEFAULT_DRY_RUN.
    - unsafe: follow DEFAULT_DRY_RUN (intended for expert/workbench usage).
    """
    if profile == "safe-read-only":
        return True
    elif profile == "dev":
        return DEFAULT_DRY_RUN
    elif profile == "unsafe":
        return DEFAULT_DRY_RUN
    else:
        # Unknown profile: default to "dev" semantics.
        return DEFAULT_DRY_RUN


def main():
    # Ensure sandbox directory exists
    os.makedirs(SANDBOX_ROOT, exist_ok=True)

    # Configure dry-run based on profile
    dry_run = compute_dry_run_from_profile(AGENT_PROFILE)

    # Instantiate components
    llm = OllamaLLM(OLLAMA_MODEL)
    shell = ShellSandbox(
        root=SANDBOX_ROOT,
        dry_run=dry_run,
        profile=AGENT_PROFILE,
        allow_git_destructive=ALLOW_GIT_DESTRUCTIVE,
    )
    agent = UnixRAGAgent(llm=llm, shell=shell)

    # Start interactive loop
    agent.interactive_loop()


if __name__ == "__main__":
    main()
