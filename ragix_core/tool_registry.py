"""
Tool Registry - Centralized tool definitions and management for RAGIX agents

Provides type-safe tool definitions, permission checking, and dynamic tool loading.
Unified registry for CLI, Web UI, MCP server, and agent tool access.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolProvider(str, Enum):
    """Source/provider of a tool."""
    BUILTIN = "builtin"       # Core RAGIX tools
    PLUGIN = "plugin"         # Plugin-provided tools
    MCP = "mcp"               # MCP server tools
    CUSTOM = "custom"         # User-defined tools
    WASM = "wasm"             # WASM module tools


class ToolCategory(str, Enum):
    """Categories of tools available to agents."""

    SHELL = "shell"           # Bash command execution
    FILE_READ = "file_read"   # Read file contents
    FILE_WRITE = "file_write" # Write/edit files
    SEARCH = "search"         # Code search (grep, semantic)
    GIT = "git"               # Git operations
    SYSTEM = "system"         # System information


class ToolPermission(str, Enum):
    """Permission levels for tool execution."""

    READ_ONLY = "read_only"     # Can only read, no modifications
    READ_WRITE = "read_write"   # Can read and modify files
    EXECUTE = "execute"         # Can execute commands
    FULL = "full"               # All permissions


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None  # Allowed values


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool available to agents.

    Includes schema for validation, permissions, and execution handler.
    """

    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    required_permissions: Set[ToolPermission] = field(default_factory=set)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    handler: Optional[Callable] = None  # Actual execution function

    # Provider tracking
    provider: ToolProvider = ToolProvider.BUILTIN
    provider_name: str = ""  # e.g., plugin name, MCP server name
    version: str = "1.0.0"
    enabled: bool = True

    # Metadata
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    deprecation_message: str = ""

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema for LLM tool calling."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }

    def to_prompt_description(self) -> str:
        """Generate human-readable description for system prompt."""
        params_desc = []
        for param in self.parameters:
            req = "(required)" if param.required else "(optional)"
            params_desc.append(f"  - {param.name}: {param.description} {req}")

        params_str = "\n".join(params_desc) if params_desc else "  (no parameters)"

        examples_str = ""
        if self.examples:
            examples_str = "\n  Examples:\n"
            for ex in self.examples[:2]:  # Show max 2 examples
                examples_str += f"    {json.dumps(ex)}\n"

        return f"""**{self.name}** [{self.category.value}]
{self.description}
Parameters:
{params_str}{examples_str}"""


class ToolRegistry:
    """
    Central registry for all tools available to RAGIX agents.

    Manages tool definitions, permissions, and provides tool lookup.
    """

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register built-in RAGIX tools."""

        # === SHELL TOOLS ===

        self.register(ToolDefinition(
            name="bash",
            description="Execute a bash command in the sandboxed environment. Use for system commands, file operations, and running scripts.",
            category=ToolCategory.SHELL,
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="The bash command to execute",
                    required=True
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds (default: 60)",
                    required=False,
                    default=60
                ),
            ],
            required_permissions={ToolPermission.EXECUTE},
            examples=[
                {"command": "ls -la src/"},
                {"command": "grep -r 'TODO' . --include='*.py'", "timeout": 30},
                {"command": "python -c 'print(1+1)'"},
            ]
        ))

        # === FILE TOOLS ===

        self.register(ToolDefinition(
            name="read_file",
            description="Read the contents of a file. Use to inspect code, configs, or documentation.",
            category=ToolCategory.FILE_READ,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file (relative to sandbox root)",
                    required=True
                ),
                ToolParameter(
                    name="start_line",
                    type="integer",
                    description="Starting line number (1-indexed, optional)",
                    required=False
                ),
                ToolParameter(
                    name="end_line",
                    type="integer",
                    description="Ending line number (inclusive, optional)",
                    required=False
                ),
            ],
            required_permissions={ToolPermission.READ_ONLY},
            examples=[
                {"path": "src/main.py"},
                {"path": "README.md", "start_line": 1, "end_line": 50},
            ]
        ))

        self.register(ToolDefinition(
            name="write_file",
            description="Write content to a file. Creates the file if it doesn't exist.",
            category=ToolCategory.FILE_WRITE,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file (relative to sandbox root)",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                    required=True
                ),
            ],
            required_permissions={ToolPermission.READ_WRITE},
            examples=[
                {"path": "output.txt", "content": "Hello, world!"},
            ]
        ))

        self.register(ToolDefinition(
            name="edit_file",
            description="Edit a file by replacing a specific text pattern. More precise than full file writes.",
            category=ToolCategory.FILE_WRITE,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file (relative to sandbox root)",
                    required=True
                ),
                ToolParameter(
                    name="old_text",
                    type="string",
                    description="Exact text to find and replace",
                    required=True
                ),
                ToolParameter(
                    name="new_text",
                    type="string",
                    description="Text to replace with",
                    required=True
                ),
            ],
            required_permissions={ToolPermission.READ_WRITE},
            examples=[
                {"path": "src/config.py", "old_text": "DEBUG = True", "new_text": "DEBUG = False"},
            ]
        ))

        # === SEARCH TOOLS ===

        self.register(ToolDefinition(
            name="grep_search",
            description="Search for a pattern in files using grep/ripgrep. Returns matching lines with file paths and line numbers.",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="Search pattern (regex supported)",
                    required=True
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory or file to search in (default: current directory)",
                    required=False,
                    default="."
                ),
                ToolParameter(
                    name="include",
                    type="string",
                    description="File pattern to include (e.g., '*.py')",
                    required=False
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results (default: 50)",
                    required=False,
                    default=50
                ),
            ],
            required_permissions={ToolPermission.READ_ONLY},
            examples=[
                {"pattern": "def main", "path": "src/", "include": "*.py"},
                {"pattern": "TODO|FIXME", "max_results": 20},
            ]
        ))

        self.register(ToolDefinition(
            name="semantic_search",
            description="Search for code using semantic similarity. Finds conceptually related code even without exact matches.",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Natural language description of what to find",
                    required=True
                ),
                ToolParameter(
                    name="k",
                    type="integer",
                    description="Number of results to return (default: 10)",
                    required=False,
                    default=10
                ),
                ToolParameter(
                    name="file_types",
                    type="array",
                    description="File types to search (e.g., ['py', 'js'])",
                    required=False
                ),
            ],
            required_permissions={ToolPermission.READ_ONLY},
            examples=[
                {"query": "function that handles user authentication"},
                {"query": "database connection setup", "k": 5, "file_types": ["py"]},
            ]
        ))

        self.register(ToolDefinition(
            name="find_files",
            description="Find files by name pattern. Use to locate files in the project.",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="Glob pattern for file names (e.g., '*.py', '*test*.py')",
                    required=True
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory to search in (default: current directory)",
                    required=False,
                    default="."
                ),
            ],
            required_permissions={ToolPermission.READ_ONLY},
            examples=[
                {"pattern": "*.py", "path": "src/"},
                {"pattern": "*config*"},
            ]
        ))

        # === GIT TOOLS ===

        self.register(ToolDefinition(
            name="git_status",
            description="Show the working tree status (modified, staged, untracked files).",
            category=ToolCategory.GIT,
            parameters=[],
            required_permissions={ToolPermission.READ_ONLY},
            examples=[{}]
        ))

        self.register(ToolDefinition(
            name="git_diff",
            description="Show changes between commits, working tree, etc.",
            category=ToolCategory.GIT,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Specific file or directory to diff (optional)",
                    required=False
                ),
                ToolParameter(
                    name="staged",
                    type="boolean",
                    description="Show staged changes only",
                    required=False,
                    default=False
                ),
            ],
            required_permissions={ToolPermission.READ_ONLY},
            examples=[
                {},
                {"path": "src/main.py"},
                {"staged": True},
            ]
        ))

        self.register(ToolDefinition(
            name="git_log",
            description="Show commit history.",
            category=ToolCategory.GIT,
            parameters=[
                ToolParameter(
                    name="n",
                    type="integer",
                    description="Number of commits to show (default: 10)",
                    required=False,
                    default=10
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="Show history for specific file/directory",
                    required=False
                ),
            ],
            required_permissions={ToolPermission.READ_ONLY},
            examples=[
                {"n": 5},
                {"path": "src/", "n": 20},
            ]
        ))

        # === SYSTEM TOOLS ===

        self.register(ToolDefinition(
            name="list_directory",
            description="List contents of a directory with details.",
            category=ToolCategory.SYSTEM,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Directory path (default: current directory)",
                    required=False,
                    default="."
                ),
                ToolParameter(
                    name="recursive",
                    type="boolean",
                    description="List recursively (tree view)",
                    required=False,
                    default=False
                ),
            ],
            required_permissions={ToolPermission.READ_ONLY},
            examples=[
                {"path": "src/"},
                {"path": ".", "recursive": True},
            ]
        ))

        self.register(ToolDefinition(
            name="project_overview",
            description="Get an overview of the project structure, key files, and statistics.",
            category=ToolCategory.SYSTEM,
            parameters=[],
            required_permissions={ToolPermission.READ_ONLY},
            examples=[{}]
        ))

    def register(self, tool: ToolDefinition):
        """
        Register a tool in the registry.

        Args:
            tool: ToolDefinition to register
        """
        if tool.name in self.tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")

        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            ToolDefinition or None if not found
        """
        return self.tools.get(name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        permission: Optional[ToolPermission] = None
    ) -> List[str]:
        """
        List available tool names, optionally filtered.

        Args:
            category: Filter by category
            permission: Filter by required permission

        Returns:
            List of tool names
        """
        result = []
        for name, tool in self.tools.items():
            if category and tool.category != category:
                continue
            if permission and permission not in tool.required_permissions:
                continue
            result.append(name)
        return sorted(result)

    def get_tools_for_agent(
        self,
        allowed_tools: Optional[List[str]] = None,
        permissions: Optional[Set[ToolPermission]] = None
    ) -> List[ToolDefinition]:
        """
        Get tools available to an agent based on allowed list and permissions.

        Args:
            allowed_tools: List of allowed tool names (None = all)
            permissions: Agent's permission set

        Returns:
            List of ToolDefinition objects
        """
        result = []

        for name, tool in self.tools.items():
            # Check if tool is in allowed list
            if allowed_tools is not None and name not in allowed_tools:
                continue

            # Check permissions
            if permissions is not None:
                if not tool.required_permissions.issubset(permissions):
                    continue

            result.append(tool)

        return result

    def generate_tools_prompt(
        self,
        tools: List[ToolDefinition]
    ) -> str:
        """
        Generate system prompt section describing available tools.

        Args:
            tools: List of tools to include

        Returns:
            Formatted prompt string
        """
        if not tools:
            return "No tools available."

        sections = []

        # Group by category
        by_category: Dict[ToolCategory, List[ToolDefinition]] = {}
        for tool in tools:
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)

        for category in ToolCategory:
            if category not in by_category:
                continue

            category_tools = by_category[category]
            sections.append(f"\n### {category.value.upper()} TOOLS\n")

            for tool in category_tools:
                sections.append(tool.to_prompt_description())
                sections.append("")

        return "\n".join(sections)

    def generate_json_schemas(
        self,
        tools: List[ToolDefinition]
    ) -> List[Dict[str, Any]]:
        """
        Generate JSON schemas for LLM tool calling.

        Args:
            tools: List of tools to include

        Returns:
            List of JSON schema objects
        """
        return [tool.to_json_schema() for tool in tools]

    def validate_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a tool call.

        Args:
            tool_name: Name of the tool
            parameters: Parameters for the call

        Returns:
            Tuple of (is_valid, error_message)
        """
        tool = self.get(tool_name)

        if not tool:
            return False, f"Unknown tool: {tool_name}"

        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in parameters:
                return False, f"Missing required parameter: {param.name}"

        # Check parameter types (basic validation)
        for param in tool.parameters:
            if param.name in parameters:
                value = parameters[param.name]

                # Type checking
                if param.type == "string" and not isinstance(value, str):
                    return False, f"Parameter {param.name} must be a string"
                elif param.type == "integer" and not isinstance(value, int):
                    return False, f"Parameter {param.name} must be an integer"
                elif param.type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter {param.name} must be a boolean"
                elif param.type == "array" and not isinstance(value, list):
                    return False, f"Parameter {param.name} must be an array"
                elif param.type == "object" and not isinstance(value, dict):
                    return False, f"Parameter {param.name} must be an object"

                # Enum validation
                if param.enum and value not in param.enum:
                    return False, f"Parameter {param.name} must be one of: {param.enum}"

        return True, None


    def register_from_plugin(
        self,
        plugin_name: str,
        tool: ToolDefinition,
    ):
        """
        Register a tool from a plugin.

        Args:
            plugin_name: Name of the source plugin
            tool: Tool definition
        """
        tool.provider = ToolProvider.PLUGIN
        tool.provider_name = plugin_name
        self.register(tool)

    def register_from_mcp(
        self,
        server_name: str,
        tool: ToolDefinition,
    ):
        """
        Register a tool from an MCP server.

        Args:
            server_name: Name of the MCP server
            tool: Tool definition
        """
        tool.provider = ToolProvider.MCP
        tool.provider_name = server_name
        self.register(tool)

    def register_from_wasm(
        self,
        module_name: str,
        tool: ToolDefinition,
    ):
        """
        Register a tool from a WASM module.

        Args:
            module_name: Name of the WASM module
            tool: Tool definition
        """
        tool.provider = ToolProvider.WASM
        tool.provider_name = module_name
        self.register(tool)

    def list_tools_by_provider(
        self,
        provider: Optional[ToolProvider] = None
    ) -> Dict[str, List[str]]:
        """
        List tools grouped by provider.

        Args:
            provider: Filter to specific provider (None = all)

        Returns:
            Dictionary of provider -> list of tool names
        """
        result: Dict[str, List[str]] = {}

        for name, tool in self.tools.items():
            if provider and tool.provider != provider:
                continue

            key = tool.provider.value
            if key not in result:
                result[key] = []
            result[key].append(name)

        return result

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a tool.

        Args:
            name: Tool name

        Returns:
            Tool information dictionary
        """
        tool = self.get(name)
        if not tool:
            return None

        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "provider": tool.provider.value,
            "provider_name": tool.provider_name,
            "version": tool.version,
            "enabled": tool.enabled,
            "deprecated": tool.deprecated,
            "deprecation_message": tool.deprecation_message,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in tool.parameters
            ],
            "permissions": [p.value for p in tool.required_permissions],
            "tags": tool.tags,
            "examples": tool.examples[:2],  # Limit examples
            "has_handler": tool.handler is not None,
        }

    def enable_tool(self, name: str) -> bool:
        """Enable a tool."""
        tool = self.get(name)
        if tool:
            tool.enabled = True
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """Disable a tool."""
        tool = self.get(name)
        if tool:
            tool.enabled = False
            return True
        return False

    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name: Tool name

        Returns:
            True if removed
        """
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def unregister_by_provider(self, provider_name: str) -> int:
        """
        Remove all tools from a specific provider.

        Args:
            provider_name: Provider name (plugin/MCP server name)

        Returns:
            Number of tools removed
        """
        to_remove = [
            name for name, tool in self.tools.items()
            if tool.provider_name == provider_name
        ]

        for name in to_remove:
            del self.tools[name]

        logger.info(f"Unregistered {len(to_remove)} tools from provider: {provider_name}")
        return len(to_remove)

    def export_for_cli(self) -> List[Dict[str, Any]]:
        """
        Export tools for CLI display.

        Returns:
            List of tool summaries
        """
        return [
            {
                "name": t.name,
                "description": t.description[:60] + "..." if len(t.description) > 60 else t.description,
                "category": t.category.value,
                "provider": t.provider.value,
                "enabled": t.enabled,
            }
            for t in self.tools.values()
            if t.enabled
        ]

    def export_for_mcp(self) -> List[Dict[str, Any]]:
        """
        Export tools for MCP server.

        Returns:
            List of MCP-compatible tool definitions
        """
        return [
            tool.to_json_schema()
            for tool in self.tools.values()
            if tool.enabled and not tool.deprecated
        ]


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def sync_plugins_to_registry():
    """
    Sync plugin tools to the global registry.

    Called after plugins are loaded to make their tools available.
    """
    try:
        from .plugin_system import get_plugin_manager

        pm = get_plugin_manager()
        registry = get_tool_registry()

        for plugin_name, plugin in pm.plugins.items():
            if not plugin.enabled:
                continue

            for tool in plugin.manifest.tools:
                if tool.name in registry.tools:
                    continue

                # Create ToolDefinition from plugin tool
                tool_def = ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    category=ToolCategory.SYSTEM,  # Default category
                    parameters=[
                        ToolParameter(
                            name=p.get("name", ""),
                            type=p.get("type", "string"),
                            description=p.get("description", ""),
                            required=p.get("required", False),
                            default=p.get("default"),
                        )
                        for p in tool.parameters
                    ],
                    handler=tool.handler,
                    provider=ToolProvider.PLUGIN,
                    provider_name=plugin_name,
                    version=plugin.manifest.version,
                )

                registry.register(tool_def)

        logger.info(f"Synced plugins to registry")

    except ImportError:
        logger.debug("Plugin system not available")
    except Exception as e:
        logger.error(f"Failed to sync plugins: {e}")
