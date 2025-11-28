"""
WASP Tools - WebAssembly-ready Agentic System Protocol Tools

Deterministic, sandboxed tools for RAGIX agents.
These Python implementations can be compiled to WASM for browser execution.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

# Import version from centralized source
try:
    from ragix_core.version import __version__
except ImportError:
    __version__ = "0.20.0"

from .validate import (
    validate_json,
    validate_yaml,
    format_json,
    format_yaml,
    json_to_yaml,
    yaml_to_json,
)
from .mdparse import (
    parse_markdown,
    extract_headers,
    extract_code_blocks,
    extract_links,
    extract_frontmatter,
    renumber_sections,
    generate_toc,
)
from .search import (
    search_pattern,
    search_lines,
    count_matches,
    extract_matches,
    replace_pattern,
)

# Tool registry for dynamic discovery
WASP_TOOLS = {
    # Validation tools
    "validate_json": {
        "func": validate_json,
        "description": "Validate JSON content and optionally check against schema",
        "category": "validation",
    },
    "validate_yaml": {
        "func": validate_yaml,
        "description": "Validate YAML content and optionally check against schema",
        "category": "validation",
    },
    "format_json": {
        "func": format_json,
        "description": "Format/prettify JSON content",
        "category": "validation",
    },
    "format_yaml": {
        "func": format_yaml,
        "description": "Format/prettify YAML content",
        "category": "validation",
    },
    "json_to_yaml": {
        "func": json_to_yaml,
        "description": "Convert JSON to YAML",
        "category": "validation",
    },
    "yaml_to_json": {
        "func": yaml_to_json,
        "description": "Convert YAML to JSON",
        "category": "validation",
    },
    # Markdown tools
    "parse_markdown": {
        "func": parse_markdown,
        "description": "Parse Markdown to structured AST",
        "category": "markdown",
    },
    "extract_headers": {
        "func": extract_headers,
        "description": "Extract headers from Markdown",
        "category": "markdown",
    },
    "extract_code_blocks": {
        "func": extract_code_blocks,
        "description": "Extract code blocks from Markdown",
        "category": "markdown",
    },
    "extract_links": {
        "func": extract_links,
        "description": "Extract links from Markdown",
        "category": "markdown",
    },
    "extract_frontmatter": {
        "func": extract_frontmatter,
        "description": "Extract YAML frontmatter from Markdown",
        "category": "markdown",
    },
    "renumber_sections": {
        "func": renumber_sections,
        "description": "Renumber Markdown section headers",
        "category": "markdown",
    },
    "generate_toc": {
        "func": generate_toc,
        "description": "Generate table of contents from Markdown",
        "category": "markdown",
    },
    # Search tools
    "search_pattern": {
        "func": search_pattern,
        "description": "Search for regex pattern in content",
        "category": "search",
    },
    "search_lines": {
        "func": search_lines,
        "description": "Search for pattern and return matching lines",
        "category": "search",
    },
    "count_matches": {
        "func": count_matches,
        "description": "Count pattern matches in content",
        "category": "search",
    },
    "extract_matches": {
        "func": extract_matches,
        "description": "Extract all pattern matches with groups",
        "category": "search",
    },
    "replace_pattern": {
        "func": replace_pattern,
        "description": "Replace pattern matches in content",
        "category": "search",
    },
}


def get_tool(name: str):
    """Get a WASP tool by name."""
    if name in WASP_TOOLS:
        return WASP_TOOLS[name]["func"]
    return None


def list_tools(category: str = None) -> list:
    """List available WASP tools, optionally filtered by category."""
    if category:
        return [
            name for name, info in WASP_TOOLS.items()
            if info.get("category") == category
        ]
    return list(WASP_TOOLS.keys())


def get_tool_info(name: str) -> dict:
    """Get tool metadata."""
    if name in WASP_TOOLS:
        return {
            "name": name,
            "description": WASP_TOOLS[name]["description"],
            "category": WASP_TOOLS[name]["category"],
        }
    return None


__all__ = [
    # Version
    "__version__",
    # Validation
    "validate_json",
    "validate_yaml",
    "format_json",
    "format_yaml",
    "json_to_yaml",
    "yaml_to_json",
    # Markdown
    "parse_markdown",
    "extract_headers",
    "extract_code_blocks",
    "extract_links",
    "extract_frontmatter",
    "renumber_sections",
    "generate_toc",
    # Search
    "search_pattern",
    "search_lines",
    "count_matches",
    "extract_matches",
    "replace_pattern",
    # Registry
    "WASP_TOOLS",
    "get_tool",
    "list_tools",
    "get_tool_info",
]
