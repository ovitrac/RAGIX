"""
Agent-LLM Bridge - Connects RAGIX agents to LLM reasoning with tool execution

Provides the execution loop that drives agents using LLM reasoning and tool calls.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, AsyncIterator
from pathlib import Path

from .tool_registry import (
    ToolRegistry,
    ToolDefinition,
    ToolPermission,
    get_tool_registry,
)
from .tools_shell import ShellSandbox, CommandResult
from .llm_backends import OllamaLLM
from .logging_utils import AgentLogger

logger = logging.getLogger(__name__)


class ExecutionState(str, Enum):
    """State of agent execution."""

    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class ToolCall:
    """Represents a tool call extracted from LLM response."""

    tool_name: str
    parameters: Dict[str, Any]
    raw_text: str = ""  # Original text that generated this call


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class ExecutionStep:
    """A single step in the agent execution."""

    step_number: int
    llm_response: str
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]
    thinking: str = ""  # Chain-of-thought reasoning
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentExecutionResult:
    """Complete result of agent execution."""

    task: str
    status: ExecutionState
    steps: List[ExecutionStep]
    final_response: str
    total_duration: float
    token_count: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == ExecutionState.COMPLETED


class ToolExecutor:
    """
    Executes tools on behalf of agents.

    Maps tool calls to actual implementations using ShellSandbox
    and other RAGIX components. Supports parallel execution of
    independent tool calls.
    """

    def __init__(
        self,
        sandbox: ShellSandbox,
        registry: Optional[ToolRegistry] = None,
        index_path: Optional[Path] = None,
        max_parallel: int = 5,
    ):
        """
        Initialize tool executor.

        Args:
            sandbox: ShellSandbox for command execution
            registry: Tool registry (uses global if not provided)
            index_path: Path to semantic index (for semantic_search)
            max_parallel: Maximum parallel tool executions (default 5)
        """
        self.sandbox = sandbox
        self.registry = registry or get_tool_registry()
        self.index_path = index_path
        self.max_parallel = max_parallel
        self._vector_index = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with output or error
        """
        start_time = time.time()

        try:
            # Validate tool call
            is_valid, error = self.registry.validate_tool_call(
                tool_call.tool_name, tool_call.parameters
            )

            if not is_valid:
                return ToolResult(
                    tool_name=tool_call.tool_name,
                    success=False,
                    output="",
                    error=error,
                    duration=time.time() - start_time,
                )

            # Route to appropriate handler
            output = await self._dispatch_tool(tool_call)

            return ToolResult(
                tool_name=tool_call.tool_name,
                success=True,
                output=output,
                duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Tool execution error: {tool_call.tool_name}: {e}")
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output="",
                error=str(e),
                duration=time.time() - start_time,
            )

    async def execute_parallel(
        self, tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls in parallel.

        Independent read-only tools are executed concurrently for better
        performance. Write operations are serialized for safety.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of ToolResult in the same order as input
        """
        if not tool_calls:
            return []

        if len(tool_calls) == 1:
            return [await self.execute(tool_calls[0])]

        # Initialize semaphore if needed
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_parallel)

        # Categorize tools by safety
        read_only_tools = {
            "read_file", "grep_search", "semantic_search", "find_files",
            "git_status", "git_diff", "git_log", "list_directory",
            "project_overview"
        }

        # Separate read-only and write tools
        read_tasks = []
        write_tasks = []
        task_indices = []  # Track original order

        for i, tc in enumerate(tool_calls):
            if tc.tool_name in read_only_tools:
                read_tasks.append((i, tc))
            else:
                write_tasks.append((i, tc))

        results: Dict[int, ToolResult] = {}

        # Execute read-only tools in parallel
        if read_tasks:
            async def bounded_execute(idx: int, tc: ToolCall) -> Tuple[int, ToolResult]:
                async with self._semaphore:
                    result = await self.execute(tc)
                    return (idx, result)

            read_results = await asyncio.gather(
                *[bounded_execute(idx, tc) for idx, tc in read_tasks],
                return_exceptions=True
            )

            for r in read_results:
                if isinstance(r, Exception):
                    logger.error(f"Parallel tool execution error: {r}")
                    continue
                idx, result = r
                results[idx] = result

        # Execute write tools sequentially for safety
        for idx, tc in write_tasks:
            results[idx] = await self.execute(tc)

        # Return results in original order
        return [results[i] for i in range(len(tool_calls)) if i in results]

    async def _dispatch_tool(self, tool_call: ToolCall) -> str:
        """Dispatch tool call to appropriate handler."""
        name = tool_call.tool_name
        params = tool_call.parameters

        # === SHELL TOOLS ===
        if name == "bash":
            result = self.sandbox.run(
                params["command"],
                timeout=params.get("timeout", 60)
            )
            if result.return_code != 0:
                return f"Exit code: {result.return_code}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return result.stdout

        # === FILE TOOLS ===
        elif name == "read_file":
            path = Path(self.sandbox.root) / params["path"]
            if not path.exists():
                raise FileNotFoundError(f"File not found: {params['path']}")

            content = path.read_text(encoding="utf-8")
            lines = content.splitlines()

            start = params.get("start_line")
            end = params.get("end_line")

            if start or end:
                start = (start or 1) - 1  # Convert to 0-indexed
                end = end or len(lines)
                lines = lines[start:end]

                # Add line numbers
                numbered = []
                for i, line in enumerate(lines, start=start + 1):
                    numbered.append(f"{i:4d} | {line}")
                return "\n".join(numbered)

            return content

        elif name == "write_file":
            path = Path(self.sandbox.root) / params["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(params["content"], encoding="utf-8")
            return f"File written: {params['path']} ({len(params['content'])} bytes)"

        elif name == "edit_file":
            path = Path(self.sandbox.root) / params["path"]
            if not path.exists():
                raise FileNotFoundError(f"File not found: {params['path']}")

            content = path.read_text(encoding="utf-8")
            old_text = params["old_text"]
            new_text = params["new_text"]

            if old_text not in content:
                raise ValueError(f"Text not found in file: {old_text[:50]}...")

            # Replace first occurrence
            new_content = content.replace(old_text, new_text, 1)
            path.write_text(new_content, encoding="utf-8")

            return f"File edited: {params['path']}\nReplaced {len(old_text)} chars with {len(new_text)} chars"

        # === SEARCH TOOLS ===
        elif name == "grep_search":
            cmd = f"grep -rn '{params['pattern']}' {params.get('path', '.')}"

            if params.get("include"):
                cmd += f" --include='{params['include']}'"

            max_results = params.get("max_results", 50)
            cmd += f" | head -n {max_results}"

            result = self.sandbox.run(cmd, timeout=30)
            return result.stdout or "(no matches found)"

        elif name == "semantic_search":
            # Load index if available
            if self.index_path and self._vector_index is None:
                try:
                    from .vector_index import NumpyVectorIndex
                    from .embeddings import create_embedding_backend

                    index_dir = self.index_path / "index"
                    if index_dir.exists():
                        self._vector_index = NumpyVectorIndex(384)
                        self._vector_index.load(index_dir)
                        self._embedding_backend = create_embedding_backend()
                except Exception as e:
                    logger.warning(f"Could not load semantic index: {e}")

            if self._vector_index is None:
                return "(semantic search not available - run 'ragix-index' first)"

            # Perform search
            query_embedding = self._embedding_backend.embed_text(params["query"])
            results = self._vector_index.search(query_embedding, k=params.get("k", 10))

            if not results:
                return "(no semantic matches found)"

            output_lines = []
            for r in results:
                output_lines.append(
                    f"{r.file_path}:{r.start_line}-{r.end_line} [{r.chunk_type}] "
                    f"{r.name} (score: {r.score:.3f})"
                )
            return "\n".join(output_lines)

        elif name == "find_files":
            cmd = f"find {params.get('path', '.')} -name '{params['pattern']}' -type f"
            result = self.sandbox.run(cmd, timeout=30)
            return result.stdout or "(no files found)"

        # === GIT TOOLS ===
        elif name == "git_status":
            result = self.sandbox.run("git status --short", timeout=30)
            return result.stdout or "(clean working tree)"

        elif name == "git_diff":
            cmd = "git diff"
            if params.get("staged"):
                cmd += " --staged"
            if params.get("path"):
                cmd += f" -- {params['path']}"
            result = self.sandbox.run(cmd, timeout=30)
            return result.stdout or "(no changes)"

        elif name == "git_log":
            cmd = f"git log --oneline -n {params.get('n', 10)}"
            if params.get("path"):
                cmd += f" -- {params['path']}"
            result = self.sandbox.run(cmd, timeout=30)
            return result.stdout

        # === SYSTEM TOOLS ===
        elif name == "list_directory":
            path = params.get("path", ".")
            if params.get("recursive"):
                result = self.sandbox.run(f"tree -L 3 {path}", timeout=30)
            else:
                result = self.sandbox.run(f"ls -la {path}", timeout=30)
            return result.stdout

        elif name == "project_overview":
            # Generate project overview
            lines = []

            # File count by type
            result = self.sandbox.run(
                "find . -type f -name '*.py' | wc -l && "
                "find . -type f -name '*.js' | wc -l && "
                "find . -type f -name '*.md' | wc -l",
                timeout=30
            )
            counts = result.stdout.strip().split('\n')
            lines.append("File counts:")
            lines.append(f"  Python: {counts[0] if len(counts) > 0 else '?'}")
            lines.append(f"  JavaScript: {counts[1] if len(counts) > 1 else '?'}")
            lines.append(f"  Markdown: {counts[2] if len(counts) > 2 else '?'}")

            # Top-level structure
            result = self.sandbox.run("ls -la", timeout=30)
            lines.append("\nTop-level contents:")
            lines.append(result.stdout)

            # Key files
            result = self.sandbox.run(
                "ls README.md setup.py pyproject.toml package.json 2>/dev/null || true",
                timeout=30
            )
            if result.stdout.strip():
                lines.append(f"\nKey files found: {result.stdout.strip()}")

            return "\n".join(lines)

        else:
            raise ValueError(f"Unknown tool: {name}")


class LLMAgentExecutor:
    """
    Executes agents using LLM reasoning with tool calls.

    Implements the main agent loop:
    1. Send task + context to LLM
    2. Parse LLM response for tool calls
    3. Execute tool calls
    4. Feed results back to LLM
    5. Repeat until task complete or max iterations
    """

    def __init__(
        self,
        llm: OllamaLLM,
        tool_executor: ToolExecutor,
        registry: Optional[ToolRegistry] = None,
        agent_logger: Optional[AgentLogger] = None,
        max_iterations: int = 20,
        system_prompt_template: Optional[str] = None,
        use_parallel_tools: bool = True,
    ):
        """
        Initialize LLM agent executor.

        Args:
            llm: LLM backend for reasoning
            tool_executor: Tool executor for tool calls
            registry: Tool registry (uses global if not provided)
            agent_logger: Logger for agent actions
            max_iterations: Maximum reasoning iterations
            system_prompt_template: Custom system prompt template
            use_parallel_tools: Execute read-only tools in parallel (default True)
        """
        self.llm = llm
        self.tool_executor = tool_executor
        self.registry = registry or get_tool_registry()
        self.agent_logger = agent_logger
        self.max_iterations = max_iterations
        self.system_prompt_template = system_prompt_template or DEFAULT_SYSTEM_PROMPT
        self.use_parallel_tools = use_parallel_tools

    def _build_system_prompt(
        self,
        tools: List[ToolDefinition],
        additional_context: Optional[str] = None
    ) -> str:
        """Build the system prompt with available tools."""
        tools_section = self.registry.generate_tools_prompt(tools)

        prompt = self.system_prompt_template.format(
            tools_section=tools_section,
            additional_context=additional_context or "",
        )

        return prompt

    def _parse_tool_calls(self, response: str) -> Tuple[List[ToolCall], str]:
        """
        Parse tool calls from LLM response.

        Supports multiple formats:
        1. JSON action format: {"action": "tool_name", "param": "value"}
        2. Function call format: tool_name(param="value")
        3. XML-style: <tool_call name="tool_name">...</tool_call>

        Args:
            response: LLM response text

        Returns:
            Tuple of (list of tool calls, remaining text)
        """
        tool_calls = []
        remaining = response

        # Try JSON format first (most reliable)
        json_pattern = r'\{[^{}]*"action"\s*:\s*"([^"]+)"[^{}]*\}'

        for match in re.finditer(json_pattern, response, re.DOTALL):
            try:
                json_str = match.group(0)
                data = json.loads(json_str)

                action = data.get("action")
                if not action:
                    continue

                # Extract parameters (everything except 'action')
                params = {k: v for k, v in data.items() if k != "action"}

                tool_calls.append(ToolCall(
                    tool_name=action,
                    parameters=params,
                    raw_text=json_str
                ))

                remaining = remaining.replace(json_str, "")

            except json.JSONDecodeError:
                continue

        # Try function call format
        func_pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'

        for match in re.finditer(func_pattern, response):
            func_name = match.group(1)
            args_str = match.group(2)

            # Check if this is a known tool
            if not self.registry.get(func_name):
                continue

            # Parse arguments
            try:
                params = {}
                if args_str.strip():
                    # Handle key=value pairs
                    for arg in args_str.split(","):
                        if "=" in arg:
                            key, value = arg.split("=", 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            params[key] = value

                tool_calls.append(ToolCall(
                    tool_name=func_name,
                    parameters=params,
                    raw_text=match.group(0)
                ))
            except Exception:
                continue

        return tool_calls, remaining.strip()

    def _extract_thinking(self, response: str) -> str:
        """Extract chain-of-thought reasoning from response."""
        # Look for thinking tags or sections
        patterns = [
            r'<thinking>(.*?)</thinking>',
            r'\*\*Thinking\*\*:?\s*(.*?)(?=\n\n|\{|$)',
            r'Let me think.*?:\s*(.*?)(?=\n\n|\{|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    async def execute(
        self,
        task: str,
        allowed_tools: Optional[List[str]] = None,
        permissions: Optional[set] = None,
        context: Optional[str] = None,
    ) -> AgentExecutionResult:
        """
        Execute an agent task with LLM reasoning.

        Args:
            task: The task to accomplish
            allowed_tools: List of allowed tool names
            permissions: Agent permission set
            context: Additional context for the task

        Returns:
            AgentExecutionResult with full execution trace
        """
        start_time = time.time()
        steps: List[ExecutionStep] = []

        # Get available tools
        permissions = permissions or {ToolPermission.READ_ONLY, ToolPermission.EXECUTE}
        tools = self.registry.get_tools_for_agent(allowed_tools, permissions)

        # Build system prompt
        system_prompt = self._build_system_prompt(tools, context)

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}"},
        ]

        status = ExecutionState.RUNNING
        final_response = ""
        iteration = 0

        while status == ExecutionState.RUNNING and iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")

            # Get LLM response
            try:
                response = self.llm.generate(messages)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                status = ExecutionState.FAILED
                final_response = f"LLM error: {e}"
                break

            # Parse response
            thinking = self._extract_thinking(response)
            tool_calls, text_response = self._parse_tool_calls(response)

            step = ExecutionStep(
                step_number=iteration,
                llm_response=response,
                tool_calls=tool_calls,
                tool_results=[],
                thinking=thinking,
            )

            # Execute tool calls
            if tool_calls:
                status = ExecutionState.WAITING_FOR_TOOL

                # Log tool calls
                for tc in tool_calls:
                    logger.info(f"Executing tool: {tc.tool_name}")
                    if self.agent_logger:
                        self.agent_logger.log_tool_call(tc.tool_name, tc.parameters)

                # Execute tools (parallel or sequential)
                if self.use_parallel_tools and len(tool_calls) > 1:
                    results = await self.tool_executor.execute_parallel(tool_calls)
                else:
                    results = []
                    for tc in tool_calls:
                        result = await self.tool_executor.execute(tc)
                        results.append(result)

                step.tool_results = results

                # Format tool outputs
                tool_outputs = []
                for tc, result in zip(tool_calls, results):
                    if result.success:
                        tool_outputs.append(
                            f"Tool: {tc.tool_name}\nResult:\n{result.output}"
                        )
                    else:
                        tool_outputs.append(
                            f"Tool: {tc.tool_name}\nError: {result.error}"
                        )

                # Add tool results to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Tool results:\n" + "\n---\n".join(tool_outputs)
                })

                status = ExecutionState.RUNNING

            else:
                # No tool calls - check if task is complete
                if self._is_task_complete(response):
                    status = ExecutionState.COMPLETED
                    final_response = text_response
                else:
                    # Continue reasoning
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": "Please continue with the task or indicate if you're done."
                    })

            steps.append(step)

        # Check for max iterations
        if iteration >= self.max_iterations:
            status = ExecutionState.MAX_ITERATIONS
            final_response = "Maximum iterations reached. Task may be incomplete."

        return AgentExecutionResult(
            task=task,
            status=status,
            steps=steps,
            final_response=final_response,
            total_duration=time.time() - start_time,
        )

    def _is_task_complete(self, response: str) -> bool:
        """Check if the LLM indicates the task is complete."""
        completion_phrases = [
            "task is complete",
            "task completed",
            "i have completed",
            "i've completed",
            "finished the task",
            "done with the task",
            "successfully completed",
            "here is the result",
            "here's the result",
            "in summary",
            "to summarize",
        ]

        response_lower = response.lower()
        return any(phrase in response_lower for phrase in completion_phrases)


def create_agent_executor(
    sandbox_root: Path,
    model: str = "mistral:instruct",
    index_path: Optional[Path] = None,
    max_iterations: int = 20,
) -> LLMAgentExecutor:
    """
    Factory function to create a fully configured agent executor.

    Args:
        sandbox_root: Root directory for sandbox
        model: Ollama model to use
        index_path: Path to semantic search index
        max_iterations: Maximum reasoning iterations

    Returns:
        Configured LLMAgentExecutor
    """
    # Create components
    llm = OllamaLLM(model=model)
    sandbox = ShellSandbox(root=sandbox_root)
    tool_executor = ToolExecutor(sandbox, index_path=index_path)

    return LLMAgentExecutor(
        llm=llm,
        tool_executor=tool_executor,
        max_iterations=max_iterations,
    )


# Default system prompt template
DEFAULT_SYSTEM_PROMPT = """You are a RAGIX agent - an AI assistant specialized in software engineering tasks.

You have access to tools for exploring codebases, editing files, running commands, and searching code.

## Available Tools

{tools_section}

## How to Use Tools

To use a tool, output a JSON object with the action and parameters:

```json
{{"action": "tool_name", "param1": "value1", "param2": "value2"}}
```

For example:
- To read a file: {{"action": "read_file", "path": "src/main.py"}}
- To search code: {{"action": "grep_search", "pattern": "def main", "path": "src/"}}
- To run a command: {{"action": "bash", "command": "python -m pytest tests/"}}

## Guidelines

1. **Explore First**: Use search and read tools to understand the codebase before making changes.
2. **Verify Changes**: After editing files, use git_diff to verify your changes.
3. **Explain Reasoning**: Briefly explain what you're doing and why.
4. **Be Precise**: Make targeted changes rather than large rewrites.
5. **Handle Errors**: If a tool returns an error, try an alternative approach.

## Task Completion

When the task is complete, provide a summary of what was done. Start your summary with "Task completed:" or "In summary:".

{additional_context}

Now, let's work on your task.
"""
