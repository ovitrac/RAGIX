"""
Integration Tests for WASP - Protocol and Executor

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import pytest
from ragix_core import (
    validate_action_schema,
    WaspExecutor,
    WaspExecutionResult,
    get_wasp_executor,
    execute_wasp_action,
)


class TestWaspActionSchema:
    """Tests for wasp_task action schema validation."""

    def test_valid_wasp_task(self):
        action = {
            "action": "wasp_task",
            "tool": "validate_json",
            "inputs": {"content": "{}"}
        }
        valid, error = validate_action_schema(action)
        assert valid is True
        assert error is None

    def test_wasp_task_without_inputs(self):
        # inputs is optional
        action = {
            "action": "wasp_task",
            "tool": "validate_json"
        }
        valid, error = validate_action_schema(action)
        assert valid is True

    def test_wasp_task_missing_tool(self):
        action = {
            "action": "wasp_task",
            "inputs": {"content": "{}"}
        }
        valid, error = validate_action_schema(action)
        assert valid is False
        assert "tool" in error

    def test_wasp_task_invalid_tool_type(self):
        action = {
            "action": "wasp_task",
            "tool": 123
        }
        valid, error = validate_action_schema(action)
        assert valid is False
        assert "string" in error

    def test_wasp_task_invalid_inputs_type(self):
        action = {
            "action": "wasp_task",
            "tool": "validate_json",
            "inputs": "not a dict"
        }
        valid, error = validate_action_schema(action)
        assert valid is False
        assert "dict" in error


class TestWaspExecutor:
    """Tests for WASP executor."""

    @pytest.fixture
    def executor(self):
        return WaspExecutor()

    def test_executor_initialization(self, executor):
        assert len(executor.list_tools()) > 0
        assert executor.has_tool("validate_json")
        assert executor.has_tool("parse_markdown")

    def test_executor_list_tools(self, executor):
        tools = executor.list_tools()
        assert "validate_json" in tools
        assert "extract_headers" in tools
        assert "search_pattern" in tools

    def test_executor_get_tool_info(self, executor):
        info = executor.get_tool_info("validate_json")
        assert info is not None
        assert info["name"] == "validate_json"
        assert "description" in info

    def test_executor_execute_success(self, executor):
        response, result = executor.execute({
            "tool": "validate_json",
            "inputs": {"content": '{"name": "test"}'}
        })
        assert result.success is True
        assert result.tool == "validate_json"
        assert result.result["valid"] is True
        assert result.duration_ms > 0

    def test_executor_execute_failure(self, executor):
        response, result = executor.execute({
            "tool": "validate_json",
            "inputs": {"content": "{invalid json}"}
        })
        assert result.success is False
        assert result.result["valid"] is False

    def test_executor_unknown_tool(self, executor):
        response, result = executor.execute({
            "tool": "nonexistent_tool",
            "inputs": {}
        })
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_executor_missing_tool(self, executor):
        response, result = executor.execute({
            "inputs": {"content": "test"}
        })
        assert result.success is False
        assert "No tool specified" in result.error

    def test_executor_invalid_arguments(self, executor):
        response, result = executor.execute({
            "tool": "validate_json",
            "inputs": {"invalid_param": "value"}
        })
        assert result.success is False
        assert "Invalid arguments" in result.error or result.result is not None

    def test_executor_response_format(self, executor):
        response, result = executor.execute({
            "tool": "validate_json",
            "inputs": {"content": '{}'}
        })
        assert isinstance(response, str)
        assert "validate_json" in response
        assert "successfully" in response or "failed" in response

    def test_executor_register_custom_tool(self, executor):
        def custom_tool(text: str) -> dict:
            return {"result": text.upper()}

        executor.register_tool(
            "custom_upper",
            custom_tool,
            description="Convert text to uppercase",
            category="custom"
        )

        assert executor.has_tool("custom_upper")

        response, result = executor.execute({
            "tool": "custom_upper",
            "inputs": {"text": "hello"}
        })
        assert result.success is True
        assert result.result["result"] == "HELLO"

    def test_executor_unregister_tool(self, executor):
        def temp_tool():
            return {}

        executor.register_tool("temp_tool", temp_tool)
        assert executor.has_tool("temp_tool")

        executor.unregister_tool("temp_tool")
        assert not executor.has_tool("temp_tool")

    def test_executor_tools_prompt(self, executor):
        prompt = executor.get_tools_prompt()
        assert "WASP tools" in prompt
        assert "validate_json" in prompt
        assert "wasp_task" in prompt


class TestWaspExecutionResult:
    """Tests for WaspExecutionResult dataclass."""

    def test_result_to_dict(self):
        result = WaspExecutionResult(
            tool="test_tool",
            success=True,
            result={"data": "value"},
            duration_ms=10.5,
            inputs={"arg": "value"}
        )

        d = result.to_dict()
        assert d["tool"] == "test_tool"
        assert d["success"] is True
        assert d["result"]["data"] == "value"
        assert d["duration_ms"] == 10.5

    def test_result_to_response_success(self):
        result = WaspExecutionResult(
            tool="validate_json",
            success=True,
            result={"valid": True},
            duration_ms=5.0
        )

        response = result.to_response()
        assert "successfully" in response
        assert "validate_json" in response
        assert "5.0ms" in response

    def test_result_to_response_failure(self):
        result = WaspExecutionResult(
            tool="validate_json",
            success=False,
            result=None,
            error="Invalid JSON"
        )

        response = result.to_response()
        assert "failed" in response
        assert "Invalid JSON" in response


class TestGlobalExecutor:
    """Tests for global executor functions."""

    def test_get_wasp_executor(self):
        executor1 = get_wasp_executor()
        executor2 = get_wasp_executor()
        # Should return same instance
        assert executor1 is executor2

    def test_execute_wasp_action(self):
        response, result = execute_wasp_action({
            "action": "wasp_task",
            "tool": "validate_json",
            "inputs": {"content": '{"test": 123}'}
        })

        assert result.success is True
        assert result.result["valid"] is True


class TestToolIntegration:
    """Integration tests for tool execution via executor."""

    @pytest.fixture
    def executor(self):
        return WaspExecutor()

    def test_validation_workflow(self, executor):
        # Validate JSON
        response, result = executor.execute({
            "tool": "validate_json",
            "inputs": {"content": '{"name": "test", "value": 42}'}
        })
        assert result.success is True
        assert result.result["valid"] is True

        # Format the same JSON
        response, result = executor.execute({
            "tool": "format_json",
            "inputs": {"content": '{"name":"test","value":42}', "indent": 4}
        })
        assert result.success is True
        assert result.result["success"] is True

    def test_markdown_workflow(self, executor):
        md = """# Title
## Section 1
Content here.
## Section 2
More content.
"""
        # Extract headers
        response, result = executor.execute({
            "tool": "extract_headers",
            "inputs": {"content": md}
        })
        assert result.success is True
        assert result.result["count"] == 3

        # Generate TOC
        response, result = executor.execute({
            "tool": "generate_toc",
            "inputs": {"content": md, "max_level": 2}
        })
        assert result.success is True
        assert "Title" in result.result["toc"]

    def test_search_workflow(self, executor):
        text = """Line 1: Hello
Line 2: World
Line 3: Hello World"""

        # Count matches
        response, result = executor.execute({
            "tool": "count_matches",
            "inputs": {"pattern": "Hello", "content": text}
        })
        assert result.success is True
        assert result.result["count"] == 2

        # Search and replace
        response, result = executor.execute({
            "tool": "replace_pattern",
            "inputs": {
                "pattern": "Hello",
                "replacement": "Hi",
                "content": text
            }
        })
        assert result.success is True
        assert result.result["replacements"] == 2
        assert "Hi" in result.result["content"]


class TestAllowedTools:
    """Tests for tool allowlist functionality."""

    def test_executor_with_allowed_tools(self):
        executor = WaspExecutor(allowed_tools=["validate_json", "format_json"])

        assert executor.has_tool("validate_json")
        assert executor.has_tool("format_json")
        # Other tools should not be loaded
        assert not executor.has_tool("parse_markdown")
        assert not executor.has_tool("search_pattern")

    def test_register_blocked_tool(self):
        executor = WaspExecutor(allowed_tools=["validate_json"])

        with pytest.raises(ValueError):
            executor.register_tool("custom_tool", lambda: {})
