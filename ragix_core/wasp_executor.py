"""
WASP Executor - Execute WASP tools from agent actions

Handles the wasp_task action type in the RAGIX protocol.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class WaspExecutionResult:
    """Result of a WASP tool execution."""
    tool: str
    success: bool
    result: Any
    error: Optional[str] = None
    duration_ms: float = 0.0
    inputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool": self.tool,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "inputs": self.inputs,
        }

    def to_response(self) -> str:
        """Format as response string for agent context."""
        if self.success:
            result_str = json.dumps(self.result, indent=2, default=str)
            return f"WASP tool '{self.tool}' executed successfully ({self.duration_ms:.1f}ms):\n{result_str}"
        else:
            return f"WASP tool '{self.tool}' failed: {self.error}"


class WaspExecutor:
    """
    Executes WASP tools in response to wasp_task actions.

    Provides a bridge between the RAGIX agent protocol and WASP tools.
    """

    def __init__(
        self,
        tools_module: Optional[str] = "wasp_tools",
        allowed_tools: Optional[List[str]] = None,
        max_output_size: int = 50000,
    ):
        """
        Initialize WASP executor.

        Args:
            tools_module: Module to import tools from
            allowed_tools: Optional whitelist of allowed tool names
            max_output_size: Maximum size of tool output in characters
        """
        self.tools_module = tools_module
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.max_output_size = max_output_size

        # Load tools registry
        self._tools: Dict[str, Callable] = {}
        self._tool_info: Dict[str, Dict[str, Any]] = {}
        self._load_tools()

    def _load_tools(self) -> None:
        """Load tools from the specified module."""
        try:
            import importlib
            module = importlib.import_module(self.tools_module)

            # Get tools registry
            if hasattr(module, "WASP_TOOLS"):
                for name, info in module.WASP_TOOLS.items():
                    if self.allowed_tools is None or name in self.allowed_tools:
                        self._tools[name] = info["func"]
                        self._tool_info[name] = {
                            "description": info.get("description", ""),
                            "category": info.get("category", ""),
                        }

            # Also try to get tools via get_tool function
            if hasattr(module, "get_tool") and hasattr(module, "list_tools"):
                for name in module.list_tools():
                    if name not in self._tools:
                        if self.allowed_tools is None or name in self.allowed_tools:
                            tool = module.get_tool(name)
                            if tool:
                                self._tools[name] = tool

        except ImportError as e:
            # Log but don't fail - tools can be registered later
            pass

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        category: str = "custom",
    ) -> None:
        """
        Register a custom tool.

        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            category: Tool category
        """
        if self.allowed_tools is not None and name not in self.allowed_tools:
            raise ValueError(f"Tool '{name}' is not in the allowed tools list")

        self._tools[name] = func
        self._tool_info[name] = {
            "description": description,
            "category": category,
        }

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            self._tool_info.pop(name, None)
            return True
        return False

    def list_tools(self) -> List[str]:
        """List available tool names."""
        return list(self._tools.keys())

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool information."""
        if name in self._tool_info:
            return {
                "name": name,
                **self._tool_info[name],
            }
        return None

    def has_tool(self, name: str) -> bool:
        """Check if a tool is available."""
        return name in self._tools

    def execute(
        self,
        action: Dict[str, Any],
    ) -> Tuple[Optional[str], WaspExecutionResult]:
        """
        Execute a wasp_task action.

        Args:
            action: Action dict with "tool" and optional "inputs"

        Returns:
            Tuple of (response_for_context, execution_result)
        """
        tool_name = action.get("tool", "")
        inputs = action.get("inputs", {})

        # Validate tool exists
        if not tool_name:
            result = WaspExecutionResult(
                tool="",
                success=False,
                result=None,
                error="No tool specified",
                inputs=inputs,
            )
            return result.to_response(), result

        if tool_name not in self._tools:
            result = WaspExecutionResult(
                tool=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}. Available: {', '.join(self.list_tools()[:10])}",
                inputs=inputs,
            )
            return result.to_response(), result

        # Execute tool
        func = self._tools[tool_name]
        start_time = time.perf_counter()

        try:
            output = func(**inputs)
            duration = (time.perf_counter() - start_time) * 1000

            # Check output size
            output_str = json.dumps(output, default=str)
            if len(output_str) > self.max_output_size:
                # Truncate large outputs
                output = {
                    "_truncated": True,
                    "_original_size": len(output_str),
                    "_preview": output_str[:self.max_output_size] + "...",
                }

            # Determine success based on output
            success = True
            if isinstance(output, dict):
                if output.get("success") is False or output.get("valid") is False:
                    success = False

            result = WaspExecutionResult(
                tool=tool_name,
                success=success,
                result=output,
                error=output.get("error") if isinstance(output, dict) else None,
                duration_ms=duration,
                inputs=inputs,
            )

        except TypeError as e:
            duration = (time.perf_counter() - start_time) * 1000
            result = WaspExecutionResult(
                tool=tool_name,
                success=False,
                result=None,
                error=f"Invalid arguments: {e}",
                duration_ms=duration,
                inputs=inputs,
            )

        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            result = WaspExecutionResult(
                tool=tool_name,
                success=False,
                result=None,
                error=f"Execution error: {type(e).__name__}: {e}",
                duration_ms=duration,
                inputs=inputs,
            )

        return result.to_response(), result

    def get_tools_prompt(self) -> str:
        """
        Generate a prompt snippet describing available WASP tools.

        Useful for including in agent system prompts.
        """
        lines = ["Available WASP tools (use with wasp_task action):"]

        # Group by category
        by_category: Dict[str, List[str]] = {}
        for name, info in self._tool_info.items():
            cat = info.get("category", "other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(name)

        for cat, tools in sorted(by_category.items()):
            lines.append(f"\n[{cat}]")
            for name in sorted(tools):
                desc = self._tool_info[name].get("description", "")
                lines.append(f"  - {name}: {desc}")

        lines.append("\nExample usage:")
        lines.append('{"action": "wasp_task", "tool": "validate_json", "inputs": {"content": "..."}}')

        return "\n".join(lines)


# Global executor instance
_executor: Optional[WaspExecutor] = None


def get_wasp_executor(
    tools_module: str = "wasp_tools",
    allowed_tools: Optional[List[str]] = None,
) -> WaspExecutor:
    """
    Get or create the global WASP executor.

    Args:
        tools_module: Module to load tools from
        allowed_tools: Optional whitelist of allowed tools

    Returns:
        WaspExecutor instance
    """
    global _executor
    if _executor is None:
        _executor = WaspExecutor(
            tools_module=tools_module,
            allowed_tools=allowed_tools,
        )
    return _executor


def execute_wasp_action(action: Dict[str, Any]) -> Tuple[Optional[str], WaspExecutionResult]:
    """
    Convenience function to execute a wasp_task action.

    Args:
        action: Action dict with "tool" and optional "inputs"

    Returns:
        Tuple of (response_for_context, execution_result)
    """
    executor = get_wasp_executor()
    return executor.execute(action)
