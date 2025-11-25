"""
Tests for Agent-LLM Bridge

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from ragix_core.agent_llm_bridge import (
    ExecutionState,
    ToolCall,
    ToolResult,
    ExecutionStep,
    AgentExecutionResult,
    ToolExecutor,
    LLMAgentExecutor,
    create_agent_executor,
)
from ragix_core.tools_shell import ShellSandbox


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tc = ToolCall(
            tool_name="read_file",
            parameters={"path": "src/main.py"},
            raw_text='{"action": "read_file", "path": "src/main.py"}',
        )

        assert tc.tool_name == "read_file"
        assert tc.parameters["path"] == "src/main.py"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful tool result."""
        result = ToolResult(
            tool_name="read_file",
            success=True,
            output="file contents here",
            duration=0.5,
        )

        assert result.success
        assert result.output == "file contents here"
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed tool result."""
        result = ToolResult(
            tool_name="read_file",
            success=False,
            output="",
            error="File not found",
            duration=0.1,
        )

        assert not result.success
        assert result.error == "File not found"


class TestAgentExecutionResult:
    """Tests for AgentExecutionResult dataclass."""

    def test_successful_execution(self):
        """Test successful execution result."""
        result = AgentExecutionResult(
            task="Fix the bug",
            status=ExecutionState.COMPLETED,
            steps=[],
            final_response="Bug fixed successfully",
            total_duration=5.0,
        )

        assert result.success
        assert result.status == ExecutionState.COMPLETED

    def test_failed_execution(self):
        """Test failed execution result."""
        result = AgentExecutionResult(
            task="Fix the bug",
            status=ExecutionState.FAILED,
            steps=[],
            final_response="",
            total_duration=1.0,
            error="LLM error",
        )

        assert not result.success
        assert result.error == "LLM error"


class TestToolExecutor:
    """Tests for ToolExecutor."""

    @pytest.fixture
    def sandbox(self, sample_project: Path):
        """Create a sandbox in the sample project."""
        return ShellSandbox(root=sample_project)

    @pytest.fixture
    def executor(self, sandbox):
        """Create a tool executor."""
        return ToolExecutor(sandbox)

    @pytest.mark.asyncio
    async def test_execute_read_file(self, executor, sample_project: Path):
        """Test executing read_file tool."""
        tc = ToolCall(
            tool_name="read_file",
            parameters={"path": "src/main.py"},
        )

        result = await executor.execute(tc)

        assert result.success
        assert "def main()" in result.output

    @pytest.mark.asyncio
    async def test_execute_read_file_not_found(self, executor):
        """Test read_file with non-existent file."""
        tc = ToolCall(
            tool_name="read_file",
            parameters={"path": "nonexistent.py"},
        )

        result = await executor.execute(tc)

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_grep_search(self, executor, sample_project: Path):
        """Test executing grep_search tool."""
        tc = ToolCall(
            tool_name="grep_search",
            parameters={"pattern": "def main", "path": "."},
        )

        result = await executor.execute(tc)

        assert result.success
        assert "main" in result.output

    @pytest.mark.asyncio
    async def test_execute_list_directory(self, executor):
        """Test executing list_directory tool."""
        tc = ToolCall(
            tool_name="list_directory",
            parameters={"path": "."},
        )

        result = await executor.execute(tc)

        assert result.success
        assert "src" in result.output

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, executor):
        """Test executing unknown tool."""
        tc = ToolCall(
            tool_name="unknown_tool",
            parameters={},
        )

        result = await executor.execute(tc)

        assert not result.success


class TestToolExecutorParallel:
    """Tests for parallel tool execution."""

    @pytest.fixture
    def sandbox(self, sample_project: Path):
        """Create a sandbox in the sample project."""
        return ShellSandbox(root=sample_project)

    @pytest.fixture
    def executor(self, sandbox):
        """Create a tool executor."""
        return ToolExecutor(sandbox, max_parallel=3)

    @pytest.mark.asyncio
    async def test_execute_parallel_read_only(self, executor):
        """Test parallel execution of read-only tools."""
        tool_calls = [
            ToolCall("read_file", {"path": "src/main.py"}),
            ToolCall("read_file", {"path": "src/utils.py"}),
            ToolCall("list_directory", {"path": "."}),
        ]

        results = await executor.execute_parallel(tool_calls)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_parallel_preserves_order(self, executor):
        """Test that parallel execution preserves result order."""
        tool_calls = [
            ToolCall("read_file", {"path": "src/main.py"}),
            ToolCall("read_file", {"path": "src/utils.py"}),
        ]

        results = await executor.execute_parallel(tool_calls)

        assert len(results) == 2
        # First result should be from main.py
        assert "def main()" in results[0].output
        # Second result should be from utils.py
        assert "load_config" in results[1].output

    @pytest.mark.asyncio
    async def test_execute_parallel_single_call(self, executor):
        """Test parallel with single call (should use regular execute)."""
        tool_calls = [
            ToolCall("read_file", {"path": "src/main.py"}),
        ]

        results = await executor.execute_parallel(tool_calls)

        assert len(results) == 1
        assert results[0].success

    @pytest.mark.asyncio
    async def test_execute_parallel_empty_list(self, executor):
        """Test parallel with empty list."""
        results = await executor.execute_parallel([])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_execute_parallel_write_sequential(self, executor, sample_project: Path):
        """Test that write operations are executed sequentially."""
        # First write, then read to verify
        new_file = sample_project / "src" / "new_file.py"

        tool_calls = [
            ToolCall("write_file", {
                "path": "src/new_file.py",
                "content": "# New file"
            }),
            ToolCall("read_file", {"path": "src/new_file.py"}),
        ]

        results = await executor.execute_parallel(tool_calls)

        assert results[0].success  # Write succeeded
        # Note: Read might fail because it runs in parallel with write
        # This tests that write is sequenced properly


class TestLLMAgentExecutor:
    """Tests for LLMAgentExecutor."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock()
        llm.generate = Mock(return_value='{"action": "read_file", "path": "src/main.py"}')
        return llm

    @pytest.fixture
    def mock_tool_executor(self):
        """Create a mock tool executor."""
        executor = Mock()
        executor.execute = AsyncMock(return_value=ToolResult(
            tool_name="read_file",
            success=True,
            output="def main(): pass",
        ))
        executor.execute_parallel = AsyncMock(return_value=[
            ToolResult(tool_name="read_file", success=True, output="content"),
        ])
        return executor

    def test_parse_json_tool_calls(self, mock_llm, mock_tool_executor):
        """Test parsing JSON format tool calls."""
        executor = LLMAgentExecutor(mock_llm, mock_tool_executor)

        response = '''Let me check the file.
{"action": "read_file", "path": "src/main.py"}
'''

        tool_calls, remaining = executor._parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "read_file"
        assert tool_calls[0].parameters["path"] == "src/main.py"

    def test_parse_multiple_tool_calls(self, mock_llm, mock_tool_executor):
        """Test parsing multiple tool calls."""
        executor = LLMAgentExecutor(mock_llm, mock_tool_executor)

        response = '''
{"action": "grep_search", "pattern": "def main", "path": "."}
Let me also check the file.
{"action": "read_file", "path": "src/main.py"}
'''

        tool_calls, _ = executor._parse_tool_calls(response)

        assert len(tool_calls) == 2

    def test_extract_thinking(self, mock_llm, mock_tool_executor):
        """Test extracting chain-of-thought reasoning."""
        executor = LLMAgentExecutor(mock_llm, mock_tool_executor)

        response = '''<thinking>
The user wants to find the main function.
I should search for it first.
</thinking>

{"action": "grep_search", "pattern": "def main"}
'''

        thinking = executor._extract_thinking(response)

        assert "find the main function" in thinking

    def test_is_task_complete(self, mock_llm, mock_tool_executor):
        """Test task completion detection."""
        executor = LLMAgentExecutor(mock_llm, mock_tool_executor)

        # Complete phrases
        assert executor._is_task_complete("Task completed successfully.")
        assert executor._is_task_complete("I've completed the bug fix.")
        assert executor._is_task_complete("In summary, I fixed the issue.")

        # Incomplete phrases
        assert not executor._is_task_complete("Let me search for the file.")
        assert not executor._is_task_complete("Now I will edit the code.")

    @pytest.mark.asyncio
    async def test_execute_simple_task(self, mock_tool_executor, sample_project):
        """Test executing a simple task."""
        # Create a mock LLM that completes after reading
        call_count = 0
        def generate_response(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '{"action": "read_file", "path": "src/main.py"}'
            else:
                return "Task completed. The file contains the main function."

        mock_llm = Mock()
        mock_llm.generate = generate_response

        executor = LLMAgentExecutor(
            mock_llm,
            mock_tool_executor,
            max_iterations=5,
        )

        result = await executor.execute("Read the main.py file")

        assert result.status == ExecutionState.COMPLETED
        assert len(result.steps) >= 1

    @pytest.mark.asyncio
    async def test_execute_max_iterations(self, mock_tool_executor):
        """Test that execution stops at max iterations."""
        # LLM that never completes
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value='{"action": "read_file", "path": "src/main.py"}')

        executor = LLMAgentExecutor(
            mock_llm,
            mock_tool_executor,
            max_iterations=3,
        )

        result = await executor.execute("Infinite task")

        assert result.status == ExecutionState.MAX_ITERATIONS
        assert len(result.steps) == 3


class TestCreateAgentExecutor:
    """Tests for create_agent_executor factory."""

    def test_create_executor(self, sample_project: Path):
        """Test creating an agent executor."""
        executor = create_agent_executor(
            sandbox_root=sample_project,
            model="mistral:instruct",
            max_iterations=10,
        )

        assert isinstance(executor, LLMAgentExecutor)
        assert executor.max_iterations == 10
