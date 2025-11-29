"""
WASP CLI - WebAssembly-ready Agentic System Protocol Command Line Interface

Manage and run WASP tools from the command line.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import version from centralized source
try:
    from ragix_core.version import __version__ as RAGIX_VERSION
except ImportError:
    RAGIX_VERSION = "0.21.0"

# Try to import YAML for manifest parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a WASP manifest file."""
    if not manifest_path.exists():
        return None

    try:
        content = manifest_path.read_text()
        if YAML_AVAILABLE:
            return yaml.safe_load(content)
        else:
            # Fallback: try JSON
            return json.loads(content)
    except Exception as e:
        print(f"Error loading manifest: {e}", file=sys.stderr)
        return None


def get_wasp_tools_dir() -> Path:
    """Get the wasp_tools directory."""
    # Look for wasp_tools relative to this file or in parent directories
    current = Path(__file__).parent.parent
    wasp_dir = current / "wasp_tools"
    if wasp_dir.exists():
        return wasp_dir

    # Check current working directory
    wasp_dir = Path.cwd() / "wasp_tools"
    if wasp_dir.exists():
        return wasp_dir

    return None


def cmd_list(args: argparse.Namespace) -> int:
    """List available WASP tools."""
    try:
        from wasp_tools import WASP_TOOLS, list_tools
    except ImportError:
        print("Error: wasp_tools module not found", file=sys.stderr)
        return 1

    # Group by category
    categories: Dict[str, List[str]] = {}
    for name, info in WASP_TOOLS.items():
        cat = info.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)

    print("Available WASP Tools")
    print("=" * 40)
    print()

    for cat, tools in sorted(categories.items()):
        print(f"[{cat}]")
        for tool_name in sorted(tools):
            info = WASP_TOOLS[tool_name]
            desc = info.get("description", "")[:50]
            print(f"  {tool_name:<25} {desc}")
        print()

    print(f"Total: {len(WASP_TOOLS)} tools")

    if args.json:
        # JSON output
        output = {
            "tools": [
                {
                    "name": name,
                    "category": info.get("category"),
                    "description": info.get("description"),
                }
                for name, info in WASP_TOOLS.items()
            ],
            "count": len(WASP_TOOLS),
        }
        print(json.dumps(output, indent=2))

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed information about a tool."""
    tool_name = args.tool

    try:
        from wasp_tools import WASP_TOOLS, get_tool_info
    except ImportError:
        print("Error: wasp_tools module not found", file=sys.stderr)
        return 1

    if tool_name not in WASP_TOOLS:
        print(f"Error: Tool '{tool_name}' not found", file=sys.stderr)
        print("Use 'ragix-wasp list' to see available tools")
        return 1

    info = WASP_TOOLS[tool_name]
    func = info["func"]

    print(f"Tool: {tool_name}")
    print("=" * 40)
    print(f"Category: {info.get('category', 'unknown')}")
    print(f"Description: {info.get('description', 'No description')}")
    print()

    # Try to get function signature and docstring
    if func.__doc__:
        print("Documentation:")
        print("-" * 40)
        print(func.__doc__)
        print()

    # Try to load parameter info from manifest
    wasp_dir = get_wasp_tools_dir()
    if wasp_dir:
        manifest_path = wasp_dir / "manifest.yaml"
        manifest = load_manifest(manifest_path)
        if manifest and "tools" in manifest:
            for tool_def in manifest["tools"]:
                if tool_def.get("name") == tool_name:
                    if "parameters" in tool_def:
                        print("Parameters:")
                        print("-" * 40)
                        for param in tool_def["parameters"]:
                            req = "(required)" if param.get("required") else "(optional)"
                            default = f" [default: {param.get('default')}]" if "default" in param else ""
                            print(f"  {param['name']}: {param.get('type', 'any')} {req}{default}")
                            if param.get("description"):
                                print(f"      {param['description']}")
                        print()
                    break

    if args.json:
        output = {
            "name": tool_name,
            "category": info.get("category"),
            "description": info.get("description"),
            "docstring": func.__doc__,
        }
        print(json.dumps(output, indent=2))

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a WASP tool."""
    tool_name = args.tool

    try:
        from wasp_tools import WASP_TOOLS, get_tool
    except ImportError:
        print("Error: wasp_tools module not found", file=sys.stderr)
        return 1

    func = get_tool(tool_name)
    if not func:
        print(f"Error: Tool '{tool_name}' not found", file=sys.stderr)
        return 1

    # Parse arguments
    kwargs = {}

    # Handle positional content argument
    if args.content:
        kwargs["content"] = args.content
    elif args.file:
        try:
            kwargs["content"] = Path(args.file).read_text()
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            return 1
    elif not sys.stdin.isatty():
        kwargs["content"] = sys.stdin.read()

    # Parse additional arguments from --arg
    if args.arg:
        for arg_str in args.arg:
            if "=" in arg_str:
                key, value = arg_str.split("=", 1)
                # Try to parse as JSON for complex types
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    kwargs[key] = value
            else:
                print(f"Warning: Invalid argument format '{arg_str}', expected key=value", file=sys.stderr)

    # Execute tool
    try:
        result = func(**kwargs)
    except TypeError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Use 'ragix-wasp info {tool_name}' to see required parameters")
        return 1
    except Exception as e:
        print(f"Error executing tool: {e}", file=sys.stderr)
        return 1

    # Output result
    if args.json or isinstance(result, (dict, list)):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result)

    # Return based on success
    if isinstance(result, dict):
        if result.get("success") is False or result.get("valid") is False:
            return 1

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a tool manifest file."""
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)
    if not manifest:
        print("Error: Failed to parse manifest", file=sys.stderr)
        return 1

    errors = []
    warnings = []

    # Check required fields
    required = ["name", "version", "tools"]
    for field in required:
        if field not in manifest:
            errors.append(f"Missing required field: {field}")

    # Validate tools
    if "tools" in manifest:
        for i, tool in enumerate(manifest["tools"]):
            if "name" not in tool:
                errors.append(f"Tool {i}: missing 'name'")
            if "entry" not in tool:
                errors.append(f"Tool {i}: missing 'entry'")

            # Validate parameters
            if "parameters" in tool:
                for j, param in enumerate(tool["parameters"]):
                    if "name" not in param:
                        errors.append(f"Tool {tool.get('name', i)} param {j}: missing 'name'")
                    if "type" not in param:
                        warnings.append(f"Tool {tool.get('name', i)} param {param.get('name', j)}: missing 'type'")

    # Report results
    if errors:
        print("Validation FAILED")
        print()
        print("Errors:")
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("Validation PASSED")

    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    print()
    print(f"Manifest: {manifest.get('name', 'unknown')} v{manifest.get('version', '?')}")
    print(f"Tools defined: {len(manifest.get('tools', []))}")

    return 1 if errors else 0


def cmd_categories(args: argparse.Namespace) -> int:
    """List tool categories."""
    try:
        from wasp_tools import WASP_TOOLS
    except ImportError:
        print("Error: wasp_tools module not found", file=sys.stderr)
        return 1

    categories: Dict[str, int] = {}
    for name, info in WASP_TOOLS.items():
        cat = info.get("category", "other")
        categories[cat] = categories.get(cat, 0) + 1

    print("WASP Tool Categories")
    print("=" * 40)
    for cat, count in sorted(categories.items()):
        print(f"  {cat:<20} {count} tools")

    return 0


def main():
    """Main entry point for WASP CLI."""
    parser = argparse.ArgumentParser(
        prog="ragix-wasp",
        description="WASP - WebAssembly-ready Agentic System Protocol Tools",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {RAGIX_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List available tools")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=cmd_list)

    # info command
    info_parser = subparsers.add_parser("info", help="Show tool information")
    info_parser.add_argument("tool", help="Tool name")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")
    info_parser.set_defaults(func=cmd_info)

    # run command
    run_parser = subparsers.add_parser("run", help="Run a tool")
    run_parser.add_argument("tool", help="Tool name")
    run_parser.add_argument("content", nargs="?", help="Content to process")
    run_parser.add_argument("-f", "--file", help="Read content from file")
    run_parser.add_argument("-a", "--arg", action="append", help="Additional argument (key=value)")
    run_parser.add_argument("--json", action="store_true", help="Force JSON output")
    run_parser.set_defaults(func=cmd_run)

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate manifest file")
    validate_parser.add_argument("manifest", help="Path to manifest file")
    validate_parser.set_defaults(func=cmd_validate)

    # categories command
    cat_parser = subparsers.add_parser("categories", help="List tool categories")
    cat_parser.set_defaults(func=cmd_categories)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
