#!/usr/bin/env python3
"""
Tool Call Adapter — Generic Interface for Structured Tool-Calling Models
=========================================================================

Problem:
    Some LLMs (e.g., IBM Granite 4) are trained for structured tool calling,
    not free-form bash command generation. They expect:
    - Tool definitions in a specific format (XML, JSON schema)
    - Responses as structured calls: {"name": "tool", "arguments": {...}}

    When asked for raw bash, they output fragments like "grep" or "data/file.txt"
    instead of complete commands — this is Protocol Mismatch, not model failure.

Solution:
    This adapter bridges the gap:
    1. Exposes bash operations as structured tools
    2. Formats prompts with tool schemas for compatible models
    3. Parses structured responses and renders them to executable bash
    4. Falls back gracefully for non-tool-call models

Supported Formats:
    - granite4: IBM's <tools>...</tools> XML with <tool_call> responses
    - openai: OpenAI function calling JSON schema
    - raw: Direct bash (no adaptation)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
Date: 2026-02-03
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


class ToolFormat(Enum):
    """Supported tool calling formats."""
    RAW = "raw"              # Direct bash commands (default)
    GRANITE4 = "granite4"    # IBM Granite 4 XML format
    OPENAI = "openai"        # OpenAI function calling


@dataclass
class ToolDefinition:
    """Definition of a single tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)

    def to_openai_schema(self) -> Dict:
        """Convert to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required
                }
            }
        }

    def to_granite4_schema(self) -> str:
        """Convert to Granite 4 XML format."""
        params_desc = ", ".join(
            f"{k}: {v.get('description', v.get('type', 'string'))}"
            for k, v in self.parameters.items()
        )
        return f'{{"type": "function", "function": {{"name": "{self.name}", "description": "{self.description}", "parameters": {json.dumps({"type": "object", "properties": self.parameters, "required": self.required})}}}}}'


@dataclass
class ToolCall:
    """Parsed tool call from LLM response."""
    name: str
    arguments: Dict[str, Any]
    raw_response: str = ""

    def __post_init__(self):
        """Normalize arguments if they're a JSON-encoded string."""
        if isinstance(self.arguments, str):
            args_stripped = self.arguments.strip()
            if args_stripped.startswith('{'):
                try:
                    self.arguments = json.loads(args_stripped)
                except json.JSONDecodeError:
                    pass

    def __str__(self) -> str:
        return f"{self.name}({json.dumps(self.arguments)})"


# =============================================================================
# Standard Tool Library for Shell Operations
# =============================================================================

SHELL_TOOLS: List[ToolDefinition] = [
    ToolDefinition(
        name="grep",
        description="Search for a pattern in files. Returns matching lines with file:line format. Use extended_regex=true for patterns with | (OR), + (one or more), etc.",
        parameters={
            "pattern": {"type": "string", "description": "The regex pattern to search for"},
            "path": {"type": "string", "description": "File or directory to search in (use . for current dir)"},
            "recursive": {"type": "boolean", "description": "Search recursively in directories"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive search"},
            "line_numbers": {"type": "boolean", "description": "Show line numbers"},
            "extended_regex": {"type": "boolean", "description": "Use extended regex (for |, +, ?, etc.)"}
        },
        required=["pattern"]
    ),
    ToolDefinition(
        name="find",
        description="Find files by name pattern in a directory tree.",
        parameters={
            "path": {"type": "string", "description": "Starting directory (use . for current)"},
            "name": {"type": "string", "description": "Filename pattern (supports wildcards like *.txt)"},
            "type": {"type": "string", "description": "File type: f=file, d=directory"}
        },
        required=["name"]
    ),
    ToolDefinition(
        name="cat",
        description="Display the entire contents of a file.",
        parameters={
            "file": {"type": "string", "description": "Path to the file to read"}
        },
        required=["file"]
    ),
    ToolDefinition(
        name="head",
        description="Display the first N lines of a file.",
        parameters={
            "file": {"type": "string", "description": "Path to the file"},
            "lines": {"type": "integer", "description": "Number of lines to show (default: 10)"}
        },
        required=["file"]
    ),
    ToolDefinition(
        name="tail",
        description="Display the last N lines of a file.",
        parameters={
            "file": {"type": "string", "description": "Path to the file"},
            "lines": {"type": "integer", "description": "Number of lines to show (default: 10)"}
        },
        required=["file"]
    ),
    ToolDefinition(
        name="ls",
        description="List files and directories.",
        parameters={
            "path": {"type": "string", "description": "Directory to list (default: current)"},
            "all": {"type": "boolean", "description": "Include hidden files"},
            "long": {"type": "boolean", "description": "Long format with details"}
        },
        required=[]
    ),
    ToolDefinition(
        name="wc",
        description="Count lines, words, or characters in a file.",
        parameters={
            "file": {"type": "string", "description": "Path to the file"},
            "lines": {"type": "boolean", "description": "Count lines only (-l)"},
            "words": {"type": "boolean", "description": "Count words only (-w)"},
            "chars": {"type": "boolean", "description": "Count characters only (-c)"}
        },
        required=["file"]
    ),
    ToolDefinition(
        name="echo",
        description="Output text or the value of an expression.",
        parameters={
            "text": {"type": "string", "description": "Text to output"}
        },
        required=["text"]
    ),
    ToolDefinition(
        name="count_lines",
        description="Count total lines in files matching a pattern. Use this instead of wc for glob patterns like *.py",
        parameters={
            "pattern": {"type": "string", "description": "Filename pattern (e.g., *.py, *.txt)"},
            "path": {"type": "string", "description": "Starting directory (default: current)"}
        },
        required=["pattern"]
    ),
    ToolDefinition(
        name="egrep",
        description="Search with extended regex (supports |, +, ?, etc.). Use for patterns like 'ERROR|WARN|FATAL'",
        parameters={
            "pattern": {"type": "string", "description": "Extended regex pattern (supports |, +, ?, etc.)"},
            "path": {"type": "string", "description": "File or directory to search"},
            "recursive": {"type": "boolean", "description": "Search recursively"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive search"}
        },
        required=["pattern"]
    ),
    ToolDefinition(
        name="answer",
        description="Report the final answer to the task. Use this when you have found the solution and want to declare the result.",
        parameters={
            "text": {"type": "string", "description": "The final answer or conclusion"}
        },
        required=["text"]
    ),
]


# =============================================================================
# Tool Call Renderers (Tool Call → Bash Command)
# =============================================================================

def render_grep(args: Dict[str, Any]) -> str:
    """Render grep tool call to bash command."""
    pattern = args.get("pattern", "")

    # Auto-detect extended regex need: if pattern contains |, +, ? outside brackets
    needs_extended = args.get("extended_regex", False)
    if not needs_extended and re.search(r'[|+?]', pattern):
        needs_extended = True

    cmd = ["grep"]
    if needs_extended:
        cmd.append("-E")  # Extended regex for |, +, ?, etc.
    if args.get("recursive", True):
        cmd.append("-r")
    if args.get("ignore_case"):
        cmd.append("-i")
    if args.get("line_numbers", False):  # Default off to reduce noise
        cmd.append("-n")
    cmd.append("--")  # End of options
    cmd.append(f'"{pattern}"')
    cmd.append(args.get("path", "."))
    return " ".join(cmd)


def render_find(args: Dict[str, Any]) -> str:
    """Render find tool call to bash command."""
    path = args.get("path", ".")
    cmd = ["find", path]
    if args.get("type"):
        cmd.extend(["-type", args["type"]])
    if args.get("name"):
        cmd.extend(["-name", f'"{args["name"]}"'])
    return " ".join(cmd)


def render_cat(args: Dict[str, Any]) -> str:
    """Render cat tool call to bash command."""
    return f'cat "{args.get("file", "")}"'


def render_head(args: Dict[str, Any]) -> str:
    """Render head tool call to bash command."""
    lines = args.get("lines", 10)
    return f'head -n {lines} "{args.get("file", "")}"'


def render_tail(args: Dict[str, Any]) -> str:
    """Render tail tool call to bash command."""
    lines = args.get("lines", 10)
    return f'tail -n {lines} "{args.get("file", "")}"'


def render_ls(args: Dict[str, Any]) -> str:
    """Render ls tool call to bash command."""
    cmd = ["ls"]
    if args.get("all"):
        cmd.append("-a")
    if args.get("long"):
        cmd.append("-l")
    path = args.get("path", ".")
    cmd.append(f'"{path}"')
    return " ".join(cmd)


def render_wc(args: Dict[str, Any]) -> str:
    """Render wc tool call to bash command."""
    cmd = ["wc"]
    if args.get("lines"):
        cmd.append("-l")
    elif args.get("words"):
        cmd.append("-w")
    elif args.get("chars"):
        cmd.append("-c")
    cmd.append(f'"{args.get("file", "")}"')
    return " ".join(cmd)


def render_echo(args: Dict[str, Any]) -> str:
    """Render echo tool call to bash command."""
    text = args.get("text", "")
    # Escape single quotes in text
    text_escaped = text.replace("'", "'\\''")
    return f"echo '{text_escaped}'"


def render_count_lines(args: Dict[str, Any]) -> str:
    """Render count_lines tool call to bash command using find + wc."""
    pattern = args.get("pattern", "*")
    path = args.get("path", ".")
    # Use find to match pattern, then pipe to wc -l
    # The final wc -l sums all files
    return f'find "{path}" -type f -name "{pattern}" -exec cat {{}} + | wc -l'


def render_egrep(args: Dict[str, Any]) -> str:
    """Render egrep (extended grep) tool call to bash command."""
    cmd = ["grep", "-E"]  # Extended regex
    if args.get("recursive", True):
        cmd.append("-r")
    if args.get("ignore_case"):
        cmd.append("-i")
    cmd.append("--")
    cmd.append(f'"{args.get("pattern", "")}"')
    cmd.append(args.get("path", "."))
    return " ".join(cmd)


def render_answer(args: Dict[str, Any]) -> str:
    """Render answer tool call to echo command."""
    text = args.get("text", "")
    text_escaped = text.replace("'", "'\\''")
    return f"echo 'ANSWER: {text_escaped}'"


TOOL_RENDERERS = {
    "grep": render_grep,
    "find": render_find,
    "cat": render_cat,
    "head": render_head,
    "tail": render_tail,
    "ls": render_ls,
    "wc": render_wc,
    "echo": render_echo,
    "count_lines": render_count_lines,
    "egrep": render_egrep,
    "answer": render_answer,
}


# =============================================================================
# Format-Specific Adapters
# =============================================================================

class BaseToolAdapter(ABC):
    """Abstract base class for tool call adapters."""

    def __init__(self, tools: List[ToolDefinition] = None):
        self.tools = tools or SHELL_TOOLS
        self._tool_map = {t.name: t for t in self.tools}

    @abstractmethod
    def format_prompt(self, base_prompt: str) -> str:
        """Wrap base prompt with tool definitions."""
        pass

    @abstractmethod
    def parse_response(self, response: str) -> Optional[ToolCall]:
        """Parse LLM response to extract tool call."""
        pass

    def render_to_bash(self, tool_call: ToolCall) -> str:
        """Convert tool call to executable bash command."""
        # Normalize arguments to dict if it's a string
        args = tool_call.arguments
        if isinstance(args, str):
            # First, try to parse as JSON (model may output JSON-encoded string)
            args_stripped = args.strip()
            if args_stripped.startswith('{'):
                try:
                    args = json.loads(args_stripped)
                except json.JSONDecodeError:
                    pass

            # If still a string, treat as primary argument for the tool
            if isinstance(args, str):
                if tool_call.name == "grep":
                    args = {"pattern": args}
                elif tool_call.name in ("cat", "head", "tail", "wc"):
                    args = {"file": args}
                elif tool_call.name in ("find", "ls"):
                    args = {"path": args}
                else:
                    args = {"value": args}

        renderer = TOOL_RENDERERS.get(tool_call.name)
        if renderer:
            return renderer(args)
        # Fallback: try to construct a basic command
        if isinstance(args, dict):
            args_str = " ".join(str(v) for v in args.values())
        else:
            args_str = str(args)
        return f"{tool_call.name} {args_str}"

    def process(self, response: str) -> Tuple[str, bool]:
        """
        Process LLM response: parse tool call and render to bash.

        Returns:
            (bash_command, was_tool_call)
        """
        tool_call = self.parse_response(response)
        if tool_call:
            bash = self.render_to_bash(tool_call)
            return bash, True
        # Not a tool call — return as-is
        return response.strip(), False


class RawAdapter(BaseToolAdapter):
    """Pass-through adapter for models that output raw bash."""

    def format_prompt(self, base_prompt: str) -> str:
        return base_prompt

    def parse_response(self, response: str) -> Optional[ToolCall]:
        return None  # Never parse as tool call


class Granite4Adapter(BaseToolAdapter):
    """
    Adapter for IBM Granite 4 tool calling format.

    Expects:
        <tools>
        {"type": "function", "function": {"name": "grep", ...}}
        </tools>

    Responds with:
        <tool_call>
        {"name": "grep", "arguments": {"pattern": "...", "path": "."}}
        </tool_call>
    """

    def format_prompt(self, base_prompt: str) -> str:
        """Wrap prompt with Granite 4 tool definitions."""
        tools_xml = "<tools>\n"
        for tool in self.tools:
            tools_xml += tool.to_granite4_schema() + "\n"
        tools_xml += "</tools>\n\n"

        instruction = (
            "You have access to shell tools. To use a tool, respond with a tool_call block:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
            "</tool_call>\n\n"
            "Available tools:\n"
        )

        return tools_xml + instruction + base_prompt

    def parse_response(self, response: str) -> Optional[ToolCall]:
        """Parse <tool_call>...</tool_call> from Granite 4 response."""
        # Try to extract tool_call block - use greedy match for nested JSON
        match = re.search(r'<tool_call>\s*([\s\S]*?)\s*</tool_call>', response, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            try:
                data = json.loads(json_content)
                return ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                    raw_response=response
                )
            except json.JSONDecodeError:
                # Try to find valid JSON within the block
                # Sometimes there's extra text before/after the JSON
                brace_start = json_content.find('{')
                if brace_start >= 0:
                    # Find matching closing brace
                    depth = 0
                    for i, c in enumerate(json_content[brace_start:], brace_start):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    data = json.loads(json_content[brace_start:i+1])
                                    return ToolCall(
                                        name=data.get("name", ""),
                                        arguments=data.get("arguments", {}),
                                        raw_response=response
                                    )
                                except json.JSONDecodeError:
                                    break

        # Try to parse bare JSON (model might omit XML tags)
        try:
            # Look for JSON object with name and arguments (dict form)
            json_match = re.search(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}', response, re.DOTALL)
            if json_match:
                name = json_match.group(1)
                args = json.loads(json_match.group(2))
                return ToolCall(name=name, arguments=args, raw_response=response)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try to parse JSON with string arguments (e.g., "arguments": "pattern")
        try:
            json_match = re.search(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*"([^"]*)"[^{}]*\}', response, re.DOTALL)
            if json_match:
                name = json_match.group(1)
                args = json_match.group(2)  # String argument
                return ToolCall(name=name, arguments=args, raw_response=response)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try full JSON parse as last resort
        try:
            # Find any JSON object in the response
            json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if "name" in data:
                    return ToolCall(
                        name=data["name"],
                        arguments=data.get("arguments", data.get("args", {})),
                        raw_response=response
                    )
        except (json.JSONDecodeError, AttributeError):
            pass

        # Check for simple tool name output (partial compliance)
        # e.g., "grep" or "find" without structure
        response_stripped = response.strip().lower()
        for tool in self.tools:
            if response_stripped == tool.name:
                # Model output just the tool name — return with empty args
                # This will generate a minimal command
                return ToolCall(name=tool.name, arguments={}, raw_response=response)

        return None


class OpenAIAdapter(BaseToolAdapter):
    """
    Adapter for OpenAI function calling format.

    Uses standard OpenAI tool schema and parses function_call responses.
    """

    def format_prompt(self, base_prompt: str) -> str:
        """Format prompt with OpenAI-style tool instructions."""
        tools_json = json.dumps([t.to_openai_schema() for t in self.tools], indent=2)

        instruction = (
            "You have access to the following tools:\n\n"
            f"{tools_json}\n\n"
            "To use a tool, respond with JSON:\n"
            '{"tool": "tool_name", "args": {"arg1": "value1"}}\n\n'
        )

        return instruction + base_prompt

    def parse_response(self, response: str) -> Optional[ToolCall]:
        """Parse OpenAI-style tool call from response."""
        try:
            # Try to parse as JSON
            data = json.loads(response.strip())
            if "tool" in data:
                return ToolCall(
                    name=data["tool"],
                    arguments=data.get("args", data.get("arguments", {})),
                    raw_response=response
                )
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from response text
        json_match = re.search(r'\{[^{}]*"tool"\s*:\s*"(\w+)".*?\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return ToolCall(
                    name=data["tool"],
                    arguments=data.get("args", data.get("arguments", {})),
                    raw_response=response
                )
            except json.JSONDecodeError:
                pass

        return None


# =============================================================================
# Factory and Model Configuration
# =============================================================================

# Model-to-format mapping
MODEL_TOOL_FORMATS: Dict[str, ToolFormat] = {
    # IBM Granite 4 models use structured tool calling
    "ibm/granite4:32b-a9b-h": ToolFormat.GRANITE4,
    "ibm/granite4:8b": ToolFormat.GRANITE4,
    "ibm/granite4:2b": ToolFormat.GRANITE4,

    # Most models work with raw bash
    "default": ToolFormat.RAW,
}


def get_adapter_for_model(model: str) -> BaseToolAdapter:
    """
    Get the appropriate tool call adapter for a model.

    Args:
        model: Model identifier (e.g., "ibm/granite4:32b-a9b-h")

    Returns:
        Configured adapter instance
    """
    # Check for exact match
    fmt = MODEL_TOOL_FORMATS.get(model)

    # Check for pattern match
    if fmt is None:
        model_lower = model.lower()
        if "granite4" in model_lower or "granite-4" in model_lower:
            fmt = ToolFormat.GRANITE4
        else:
            fmt = ToolFormat.RAW

    # Create adapter
    if fmt == ToolFormat.GRANITE4:
        return Granite4Adapter()
    elif fmt == ToolFormat.OPENAI:
        return OpenAIAdapter()
    else:
        return RawAdapter()


def is_tool_call_model(model: str) -> bool:
    """Check if a model uses structured tool calling."""
    adapter = get_adapter_for_model(model)
    return not isinstance(adapter, RawAdapter)


# =============================================================================
# Integration Helper
# =============================================================================

def adapt_prompt_for_model(model: str, base_prompt: str) -> Tuple[str, BaseToolAdapter]:
    """
    Adapt a prompt for a specific model's tool calling format.

    Args:
        model: Model identifier
        base_prompt: Original prompt asking for bash command

    Returns:
        (adapted_prompt, adapter) tuple
    """
    adapter = get_adapter_for_model(model)
    adapted_prompt = adapter.format_prompt(base_prompt)
    return adapted_prompt, adapter


def process_model_response(model: str, response: str, adapter: BaseToolAdapter = None) -> Tuple[str, bool]:
    """
    Process a model's response, converting tool calls to bash if needed.

    Args:
        model: Model identifier
        response: Raw LLM response
        adapter: Optional pre-created adapter (for efficiency)

    Returns:
        (bash_command, was_tool_call) tuple
    """
    if adapter is None:
        adapter = get_adapter_for_model(model)
    return adapter.process(response)


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Tool Call Adapter — Test Suite")
    print("=" * 60)

    # Test Granite 4 adapter
    print("\n1. Granite 4 Adapter Test")
    print("-" * 40)

    adapter = Granite4Adapter()

    # Test prompt formatting
    base_prompt = "Search for 'error' in the logs directory."
    formatted = adapter.format_prompt(base_prompt)
    print(f"Formatted prompt preview:\n{formatted[:500]}...")

    # Test parsing
    test_responses = [
        '<tool_call>\n{"name": "grep", "arguments": {"pattern": "error", "path": "logs/"}}\n</tool_call>',
        '{"name": "grep", "arguments": {"pattern": "error", "path": "."}}',
        'grep',  # Partial compliance
        'grep -r "error" .',  # Raw bash (should pass through)
    ]

    print("\nParsing test responses:")
    for resp in test_responses:
        bash, was_tool = adapter.process(resp)
        print(f"  Input: {resp[:50]}...")
        print(f"  → Bash: {bash}")
        print(f"  → Was tool call: {was_tool}\n")

    # Test model detection
    print("\n2. Model Format Detection")
    print("-" * 40)

    test_models = [
        "ibm/granite4:32b-a9b-h",
        "gpt-oss-safeguard:120b",
        "deepseek-r1:14b",
        "granite3.1-moe:3b",
    ]

    for model in test_models:
        adapter = get_adapter_for_model(model)
        print(f"  {model}: {type(adapter).__name__}")

    print("\n✓ All tests completed")
