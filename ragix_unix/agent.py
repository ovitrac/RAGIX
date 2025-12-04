"""
Unix-RAG Agent Implementation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from ragix_core import OllamaLLM, ShellSandbox, CommandResult, extract_json_object
from ragix_core.agent_config import AgentConfig, AgentMode, AgentRole
from ragix_core.reasoning import ReasoningLoop, EpisodicMemory, TaskComplexity, create_reasoning_loop


# System prompt for Unix-RAG agent
AGENT_SYSTEM_PROMPT = textwrap.dedent("""
You are RAGIX, a local Unix-RAG development assistant working inside a sandboxed
project directory.

IDENTITY & CONVERSATIONAL QUERIES:
When users ask "who are you", "hello", "help", or similar:
- Respond as RAGIX: "I am RAGIX, a Unix-RAG development assistant. I can explore
  your codebase, run bash commands, and edit files. How can I help you today?"
- Do NOT analyze the project files for identity questions.
- Just respond conversationally with a brief introduction.

You can:
- Run bash commands: ls, find, tree, grep -R -n, head, tail, wc -l,
  sed -n, awk, cut, sort, uniq, python one-liners, etc.
- Explore git repositories: `git status -sb`, `git diff`, `git log --oneline --graph -n 20`,
  `git show <commit>`, `git grep <pattern>`.
- Edit files using a high-level "edit_file" action (described below).

IMPORTANT BEHAVIOR (Unix-RAG style):
- When exploring, start with listing and discovery:
  * `ls`, `find . -maxdepth ...`
  * `grep -rn "keyword" --include="*.py" --exclude-dir=.git --exclude-dir=__pycache__ .`
- For grep searches, ALWAYS use these options:
  * `--include="*.py"` (or appropriate extension) to limit file types
  * `--exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=node_modules --exclude-dir=.ragix`
  * Example: `grep -rn "TODO" --include="*.py" --include="*.md" --exclude-dir=.git --exclude-dir=__pycache__ .`
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
    - the reasoning loop (Planner/Worker/Verifier)
    - episodic memory for session context

    Additional behavior:
    - On initialization, it generates an automatic project overview using
      `find` (limited depth and file types) and feeds this as context to
      the conversation.
    - For complex tasks, uses a [PLAN] template and executes step-by-step
    - Maintains episodic memory across turns for continuity
    """
    llm: OllamaLLM
    shell: ShellSandbox
    system_prompt: str = AGENT_SYSTEM_PROMPT
    history: List[Dict[str, str]] = field(default_factory=list)
    agent_config: Optional[AgentConfig] = None
    use_reasoning_loop: bool = True  # Enable Planner/Worker/Verifier

    # Internal: initialized in __post_init__
    _episodic_memory: Optional[EpisodicMemory] = field(default=None, init=False)
    _reasoning_loop: Optional[ReasoningLoop] = field(default=None, init=False)
    # Token statistics tracking
    _token_stats: Dict = field(default_factory=lambda: {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
        "last_request": {}
    })

    def __post_init__(self):
        """
        After initialization, run a project overview command and store its
        result as part of the initial context. Also initialize episodic memory
        and reasoning loop.
        """
        # Initialize agent config if not provided
        if self.agent_config is None:
            self.agent_config = AgentConfig()

        # Initialize episodic memory
        ragix_dir = Path(self.shell.root) / ".ragix"
        self._episodic_memory = EpisodicMemory(storage_path=ragix_dir)

        # Initialize reasoning loop using factory (honors RAGIX_REASONING_STRATEGY)
        if self.use_reasoning_loop:
            self._reasoning_loop = create_reasoning_loop(
                llm_generate=self._llm_generate_for_reasoning,
                agent_config=self.agent_config,
                episodic_memory=self._episodic_memory,
                shell_executor=self.shell,
                project_path=Path(self.shell.root),
            )

        # Run project overview
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

    def _llm_generate_for_reasoning(self, system_prompt: str, messages: List[Dict]) -> str:
        """Wrapper for LLM generation used by reasoning loop."""
        # Use generate_with_stats if available to track token usage
        if hasattr(self.llm, 'generate_with_stats'):
            response, stats = self.llm.generate_with_stats(system_prompt, messages)
            self._update_token_stats(stats)
            return response
        return self.llm.generate(system_prompt, messages)

    def _update_token_stats(self, stats: Dict):
        """Update cumulative token statistics."""
        self._token_stats["total_prompt_tokens"] += stats.get("prompt_tokens", 0)
        self._token_stats["total_completion_tokens"] += stats.get("completion_tokens", 0)
        self._token_stats["total_tokens"] += stats.get("total_tokens", 0)
        self._token_stats["request_count"] += 1
        self._token_stats["last_request"] = stats

    def get_token_stats(self) -> Dict:
        """Get current token usage statistics."""
        return self._token_stats.copy()

    def reset_token_stats(self):
        """Reset token statistics (e.g., for new session)."""
        self._token_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
            "last_request": {}
        }

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
        # Feed user message into history
        self._add_message("user", user_text)

        # Query the LLM for the next action
        raw = self.llm.generate(self.system_prompt, self.history)
        self._add_message("assistant", raw)

        # Parse JSON action
        action = extract_json_object(raw)
        if action is None:
            # Fallback: treat entire raw output as a message.
            return None, raw

        kind = action.get("action")

        # -------- case: execute bash command only --------
        if kind == "bash":
            cmd = action.get("command", "")
            result = self.shell.run(cmd)
            # Feed command result back as context
            self._add_message("user", "Command result:\n" + result.as_text_block())
            return result, None

        # -------- case: bash + respond --------
        elif kind == "bash_and_respond":
            cmd = action.get("command", "")
            msg = action.get("message", "")
            result = self.shell.run(cmd)
            self._add_message("user", "Command result:\n" + result.as_text_block())
            return result, msg or None

        # -------- case: respond only --------
        elif kind == "respond":
            msg = action.get("message", "")
            return None, msg or None

        # -------- case: edit_file tool --------
        elif kind == "edit_file":
            path = action.get("path", "").strip()
            old = action.get("old", "")
            new = action.get("new", "")
            summary = self._edit_file(path, old, new)
            # Feed summary into the conversation so the model can follow up with git diff, etc.
            self._add_message("user", f"Edit_file result for '{path}': {summary}")
            # We return a "response" to the human as well.
            return None, summary

        # -------- fallback: treat common shell commands as bash action --------
        # Some LLMs return {"action": "ls"} or {"action": "pwd"} instead of bash
        common_shell_cmds = {"ls", "pwd", "cat", "head", "tail", "find", "grep", "wc", "tree", "echo", "cd", "mkdir", "touch", "rm", "cp", "mv", "which", "env", "date", "whoami", "file", "stat", "du", "df"}

        # Command aliases for LLMs that use descriptive action names
        cmd_aliases = {
            "current_path": "pwd",
            "current_directory": "pwd",
            "list_files": "ls",
            "list_directory": "ls -la",
            "show_files": "ls",
            "read_file": "cat",
            "view_file": "cat",
            "search": "grep -r",
            "find_files": "find . -name",
            "directory_contents": "ls -la",
            "working_directory": "pwd",
            "get_path": "pwd",
            "show_path": "pwd",
        }

        # Check for aliases first
        if kind in cmd_aliases:
            base_cmd = cmd_aliases[kind]
            cmd_args = action.get("args", action.get("arguments", action.get("path", action.get("query", ""))))
            if cmd_args:
                cmd = f"{base_cmd} {cmd_args}"
            else:
                cmd = base_cmd
            result = self.shell.run(cmd)
            self._add_message("user", "Command result:\n" + result.as_text_block())
            return result, None

        if kind in common_shell_cmds:
            # The action type itself is the command, maybe with args
            cmd_args = action.get("args", action.get("arguments", action.get("path", "")))
            if cmd_args:
                cmd = f"{kind} {cmd_args}"
            else:
                cmd = kind
            result = self.shell.run(cmd)
            self._add_message("user", "Command result:\n" + result.as_text_block())
            return result, None

        # -------- truly unknown action fallback --------
        else:
            return None, f"Unknown action kind: {kind!r}. Use action='bash' with command='...'"

    def step_with_reasoning(self, user_text: str) -> Tuple[Optional[CommandResult], Optional[str], List[Dict]]:
        """
        Perform a reasoning step using the Planner/Worker/Verifier loop.

        For complex tasks, this will:
        1. Generate a [PLAN] using the Planner agent
        2. Execute each step using the Worker agent
        3. Verify results using the Verifier agent
        4. Maintain episodic memory for context

        Returns:
        - CommandResult or None (if no bash command executed)
        - Natural-language response or None
        - List of reasoning traces for visualization
        """
        if not self._reasoning_loop:
            # Fallback to regular step if reasoning loop not initialized
            result, msg = self.step(user_text)
            return result, msg, []

        # Record the goal in episodic memory
        self._episodic_memory.record_goal(user_text)

        # Classify task complexity
        complexity = self._reasoning_loop.classify_task(user_text)

        if complexity == TaskComplexity.BYPASS:
            # For BYPASS tasks (conceptual questions), generate direct LLM response
            # without any tool execution
            result, msg = self.step(user_text)

            traces = [{
                "type": "bypass",
                "timestamp": "",
                "content": f"Bypass (conceptual): {user_text[:100]}",
            }]
            return None, msg, traces

        if complexity == TaskComplexity.SIMPLE:
            # For simple tasks, use direct step but still record in memory
            result, msg = self.step(user_text)

            # Record in episodic memory
            if result and hasattr(result, 'command'):
                self._episodic_memory.record_command(result.command)

            traces = [{
                "type": "direct_execution",
                "timestamp": "",
                "content": f"Simple task executed: {user_text[:100]}",
            }]
            return result, msg, traces

        # For moderate/complex tasks, use the full reasoning loop
        response, traces = self._reasoning_loop.run(user_text, self.step)

        return None, response, traces

    def get_reasoning_traces(self) -> List[Dict]:
        """Get reasoning traces from the current session."""
        if self._reasoning_loop:
            return self._reasoning_loop.get_traces()
        return []

    def get_session_context(self) -> str:
        """Get the current session context from episodic memory."""
        if self._episodic_memory:
            return self._episodic_memory.get_current_session_context()
        return ""

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
