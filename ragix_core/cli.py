#!/usr/bin/env python3
"""
RAGIX Command Line Interface
=============================

Unified CLI for RAGIX installation, diagnostics, and operations.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26

Usage:
    ragix install          Install/setup RAGIX environment
    ragix doctor           Run diagnostics and health checks
    ragix upgrade          Upgrade RAGIX to latest version
    ragix config           Show current configuration
    ragix status           Show system status
    ragix logs             View recent logs
    ragix verify           Verify log integrity
    ragix mcp              Start MCP server
    ragix web              Start web interface
    ragix run              Start interactive agent
"""

import argparse
import sys
import os
import json
import shutil
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# =============================================================================
# ANSI Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ''
        cls.CYAN = cls.MAGENTA = cls.BOLD = cls.NC = ''


def print_ok(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.NC} {msg}")


def print_warn(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.NC} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}✗{Colors.NC} {msg}")


def print_info(msg: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.NC} {msg}")


def print_header(msg: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{msg}{Colors.NC}")
    print("=" * len(msg))


# =============================================================================
# Diagnostic Checks
# =============================================================================

def check_python_version() -> tuple[bool, str]:
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 10):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (requires 3.10+)"


def check_ollama() -> tuple[bool, str, Optional[List[str]]]:
    """Check Ollama status and available models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return True, f"Running with {len(models)} models", models
    except requests.exceptions.ConnectionError:
        return False, "Not running (connection refused)", None
    except requests.exceptions.Timeout:
        return False, "Not responding (timeout)", None
    except Exception as e:
        return False, f"Error: {e}", None

    return False, "Unknown error", None


def check_gpu() -> tuple[bool, str]:
    """Check for GPU availability."""
    # Check NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(",")
            return True, f"NVIDIA {gpu_info[0].strip()} ({gpu_info[1].strip()})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check AMD
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, f"AMD GPU detected"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False, "No GPU detected (CPU-only mode)"


def check_disk_space(path: Path = Path.cwd()) -> tuple[bool, str]:
    """Check available disk space."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        if free_gb > 10:
            return True, f"{free_gb:.1f} GB free"
        elif free_gb > 2:
            return True, f"{free_gb:.1f} GB free (low)"
        else:
            return False, f"{free_gb:.1f} GB free (critical)"
    except Exception as e:
        return False, f"Error: {e}"


def check_ragix_core() -> tuple[bool, str]:
    """Check ragix_core installation."""
    try:
        from ragix_core import __version__
        return True, f"Version {__version__}"
    except ImportError:
        return False, "Not installed"


def check_config_file() -> tuple[bool, str, Optional[Path]]:
    """Check for ragix.yaml configuration file."""
    try:
        from ragix_core.config import find_config_file
        config_path = find_config_file()
        if config_path:
            return True, str(config_path), config_path
        return False, "Not found", None
    except ImportError:
        # Check manually
        candidates = [
            Path.cwd() / "ragix.yaml",
            Path.cwd() / ".ragix" / "ragix.yaml",
            Path.home() / ".config" / "ragix" / "ragix.yaml",
        ]
        for path in candidates:
            if path.exists():
                return True, str(path), path
        return False, "Not found", None


def check_log_dir() -> tuple[bool, str]:
    """Check log directory."""
    log_dir = Path(".agent_logs")
    if log_dir.exists():
        log_file = log_dir / "commands.log"
        if log_file.exists():
            size_kb = log_file.stat().st_size / 1024
            return True, f"Exists ({size_kb:.1f} KB logs)"
        return True, "Exists (empty)"
    return False, "Not created"


def check_dependencies() -> List[tuple[str, bool, str]]:
    """Check optional dependencies."""
    deps = []

    # sentence-transformers
    try:
        import sentence_transformers
        deps.append(("sentence-transformers", True, sentence_transformers.__version__))
    except ImportError:
        deps.append(("sentence-transformers", False, "Not installed (optional)"))

    # faiss
    try:
        import faiss
        deps.append(("faiss-cpu", True, "Installed"))
    except ImportError:
        deps.append(("faiss-cpu", False, "Not installed (optional)"))

    # streamlit
    try:
        import streamlit
        deps.append(("streamlit", True, streamlit.__version__))
    except ImportError:
        deps.append(("streamlit", False, "Not installed (optional)"))

    # mcp
    try:
        import mcp
        deps.append(("mcp", True, "Installed"))
    except ImportError:
        deps.append(("mcp", False, "Not installed (optional)"))

    # psutil
    try:
        import psutil
        deps.append(("psutil", True, psutil.__version__))
    except ImportError:
        deps.append(("psutil", False, "Not installed (optional)"))

    return deps


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_doctor(args: argparse.Namespace) -> int:
    """Run diagnostic checks."""
    print_header("RAGIX Doctor - System Diagnostics")

    all_ok = True

    # Python version
    ok, msg = check_python_version()
    (print_ok if ok else print_error)(f"Python: {msg}")
    all_ok = all_ok and ok

    # ragix_core
    ok, msg = check_ragix_core()
    (print_ok if ok else print_warn)(f"ragix_core: {msg}")

    # Configuration
    ok, msg, path = check_config_file()
    (print_ok if ok else print_warn)(f"Configuration: {msg}")

    # Log directory
    ok, msg = check_log_dir()
    (print_ok if ok else print_info)(f"Log directory: {msg}")

    # Disk space
    ok, msg = check_disk_space()
    (print_ok if ok else print_warn)(f"Disk space: {msg}")

    # GPU
    ok, msg = check_gpu()
    (print_ok if ok else print_info)(f"GPU: {msg}")

    # Ollama
    print_header("LLM Backend (Ollama)")
    ok, msg, models = check_ollama()
    (print_ok if ok else print_error)(f"Ollama: {msg}")

    if ok and models:
        print(f"\n  Available models ({len(models)}):")
        for model in models[:10]:
            print(f"    🟢 {model}")
        if len(models) > 10:
            print(f"    ... and {len(models) - 10} more")
    elif not ok:
        print_info("  Start Ollama with: ollama serve")
        print_info("  Install a model with: ollama pull mistral")

    # Dependencies
    print_header("Optional Dependencies")
    deps = check_dependencies()
    for name, ok, msg in deps:
        (print_ok if ok else print_info)(f"{name}: {msg}")

    # Summary
    print_header("Summary")
    if all_ok:
        print_ok("All critical checks passed!")
        return 0
    else:
        print_warn("Some checks failed. See above for details.")
        return 1


def cmd_install(args: argparse.Namespace) -> int:
    """Install/setup RAGIX environment."""
    print_header("RAGIX Install")

    # Create directories
    print_info("Creating directories...")
    dirs = [
        Path(".agent_logs"),
        Path(".ragix"),
        Path(".ragix/index"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print_ok(f"  {d}")

    # Create default config if not exists
    config_path = Path("ragix.yaml")
    if not config_path.exists():
        print_info("Creating default configuration...")
        try:
            from ragix_core.config import RAGIXConfig, save_config
            config = RAGIXConfig()
            save_config(config, config_path)
            print_ok(f"  {config_path}")
        except ImportError:
            print_warn("  ragix_core not installed, skipping config generation")

    # Check Ollama
    print_info("Checking Ollama...")
    ok, msg, models = check_ollama()
    if ok:
        print_ok(f"  Ollama running with {len(models or [])} models")
    else:
        print_warn(f"  Ollama not running: {msg}")
        print_info("  Install: curl -fsSL https://ollama.com/install.sh | sh")
        print_info("  Start: ollama serve")
        print_info("  Add model: ollama pull mistral")

    print_header("Installation Complete")
    print_info("Next steps:")
    print(f"  1. Edit {config_path} to customize settings")
    print("  2. Run 'ragix doctor' to verify setup")
    print("  3. Run 'ragix run' to start the agent")

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show current configuration."""
    print_header("RAGIX Configuration")

    ok, msg, path = check_config_file()

    if ok and path:
        print_ok(f"Config file: {path}")
        print()

        try:
            from ragix_core.config import load_config
            config = load_config(path)

            sections = {
                "LLM": {
                    "Backend": f"{config.llm.backend} {'🟢' if config.llm.backend == 'ollama' else '🔴'}",
                    "Model": config.llm.model,
                    "Base URL": config.llm.base_url,
                },
                "Safety": {
                    "Profile": config.safety.profile,
                    "Air-gapped": config.safety.air_gapped,
                    "Log hashing": config.safety.log_hashing,
                },
                "MCP": {
                    "Enabled": config.mcp.enabled,
                    "Port": config.mcp.port,
                },
                "Search": {
                    "Enabled": config.search.enabled,
                    "Fusion": config.search.fusion_strategy,
                },
            }

            for section, items in sections.items():
                print(f"{Colors.BOLD}{section}:{Colors.NC}")
                for key, value in items.items():
                    print(f"  {key}: {value}")
                print()

        except Exception as e:
            print_error(f"Failed to parse config: {e}")
            return 1
    else:
        print_warn("No configuration file found")
        print_info("Run 'ragix install' to create one")
        return 1

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show system status."""
    print_header("RAGIX Status")

    # Ollama
    ok, msg, models = check_ollama()
    print(f"Ollama: {Colors.GREEN if ok else Colors.RED}{'●' if ok else '○'}{Colors.NC} {msg}")

    # Config
    ok, msg, _ = check_config_file()
    print(f"Config: {Colors.GREEN if ok else Colors.YELLOW}{'●' if ok else '○'}{Colors.NC} {msg}")

    # Logs
    ok, msg = check_log_dir()
    print(f"Logs:   {Colors.GREEN if ok else Colors.YELLOW}{'●' if ok else '○'}{Colors.NC} {msg}")

    # Disk
    ok, msg = check_disk_space()
    print(f"Disk:   {Colors.GREEN if ok else Colors.YELLOW}{'●' if ok else '○'}{Colors.NC} {msg}")

    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    """View recent logs."""
    log_file = Path(".agent_logs/commands.log")

    if not log_file.exists():
        print_warn("No log file found")
        return 1

    n = args.lines or 20

    with open(log_file, 'r') as f:
        lines = f.readlines()

    recent = lines[-n:]
    print_header(f"Recent Logs (last {len(recent)} entries)")

    for line in recent:
        line = line.strip()
        if "CMD:" in line:
            print(f"{Colors.GREEN}⚡{Colors.NC} {line}")
        elif "EDIT:" in line:
            print(f"{Colors.BLUE}✏️{Colors.NC} {line}")
        elif "EVENT:" in line:
            print(f"{Colors.YELLOW}📢{Colors.NC} {line}")
        elif "ERROR" in line or "RC: 1" in line:
            print(f"{Colors.RED}❌{Colors.NC} {line}")
        else:
            print(f"  {line}")

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify log integrity."""
    print_header("Log Integrity Verification")

    try:
        from ragix_core.log_integrity import ChainedLogHasher

        hasher = ChainedLogHasher(log_dir=Path(".agent_logs"))
        report = hasher.verify_chain()

        print(f"Log file: {report.log_file}")
        print(f"Total entries: {report.total_entries}")
        print(f"Verified: {report.verified_entries}")
        print(f"Verification time: {report.verification_time}")
        print()

        if report.valid:
            print_ok("Chain integrity verified - no tampering detected")
            return 0
        else:
            print_error(f"Chain broken at entry {report.first_invalid_entry}")
            for error in report.errors[:5]:
                print(f"  - {error}")
            return 1

    except ImportError:
        print_warn("ragix_core.log_integrity not available")
        return 1
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return 1


def cmd_mcp(args: argparse.Namespace) -> int:
    """Start MCP server."""
    print_header("Starting MCP Server")

    mcp_script = Path("MCP/ragix_mcp_server.py")
    if not mcp_script.exists():
        print_error(f"MCP server not found: {mcp_script}")
        return 1

    print_info(f"Running: python3 {mcp_script}")
    os.execvp("python3", ["python3", str(mcp_script)])


def cmd_web(args: argparse.Namespace) -> int:
    """Start web interface."""
    print_header("Starting Web Interface")

    app_script = Path("ragix_app.py")
    if not app_script.exists():
        print_error(f"Web app not found: {app_script}")
        return 1

    port = args.port or 8501
    print_info(f"URL: http://localhost:{port}")
    os.execvp("streamlit", ["streamlit", "run", str(app_script), "--server.port", str(port)])


def cmd_run(args: argparse.Namespace) -> int:
    """Start interactive agent."""
    print_header("Starting RAGIX Agent")

    agent_script = Path("unix-rag-agent.py")
    if not agent_script.exists():
        print_error(f"Agent not found: {agent_script}")
        return 1

    os.execvp("python3", ["python3", str(agent_script)])


def cmd_upgrade(args: argparse.Namespace) -> int:
    """Upgrade RAGIX."""
    print_header("RAGIX Upgrade")

    print_info("Checking for updates...")

    # Check if in git repo
    if Path(".git").exists():
        print_info("Git repository detected")
        print(f"  Run: {Colors.CYAN}git pull origin main{Colors.NC}")
    else:
        print_info("Not a git repository")
        print(f"  Run: {Colors.CYAN}pip install --upgrade ragix{Colors.NC}")

    print()
    print_info("After upgrade, run 'ragix doctor' to verify")

    return 0


# =============================================================================
# Plugin Commands
# =============================================================================

def cmd_plugin_list(args: argparse.Namespace) -> int:
    """List available plugins."""
    print_header("Available Plugins")

    try:
        from ragix_core.plugin_system import get_plugin_manager

        pm = get_plugin_manager()
        discovered = pm.discover()

        if not discovered:
            print_info("No plugins found")
            print_info("Plugin directories:")
            for d in pm.plugin_dirs:
                print(f"  - {d}")
            return 0

        for name, manifest in discovered.items():
            loaded = name in pm.plugins
            enabled = pm.plugins[name].enabled if loaded else False

            status = ""
            if loaded and enabled:
                status = f"{Colors.GREEN}[loaded]{Colors.NC}"
            elif loaded:
                status = f"{Colors.YELLOW}[disabled]{Colors.NC}"
            else:
                status = f"{Colors.BLUE}[available]{Colors.NC}"

            trust_color = {
                "builtin": Colors.GREEN,
                "trusted": Colors.CYAN,
                "untrusted": Colors.YELLOW,
            }.get(manifest.trust_level.value, "")

            print(f"\n{Colors.BOLD}{name}{Colors.NC} v{manifest.version} {status}")
            print(f"  {manifest.description}")
            print(f"  Type: {manifest.plugin_type.value} | "
                  f"Trust: {trust_color}{manifest.trust_level.value}{Colors.NC}")

            if manifest.tools:
                print(f"  Tools: {', '.join(t.name for t in manifest.tools)}")
            if manifest.workflows:
                print(f"  Workflows: {', '.join(w.name for w in manifest.workflows)}")

        return 0

    except ImportError as e:
        print_error(f"Plugin system not available: {e}")
        return 1


def cmd_plugin_info(args: argparse.Namespace) -> int:
    """Show plugin details."""
    plugin_name = args.name

    try:
        from ragix_core.plugin_system import get_plugin_manager

        pm = get_plugin_manager()
        info = pm.get_plugin_info(plugin_name)

        if not info:
            print_error(f"Plugin not found: {plugin_name}")
            return 1

        print_header(f"Plugin: {info['name']}")
        print(f"Version: {info['version']}")
        print(f"Description: {info['description']}")
        print(f"Type: {info['type']}")
        print(f"Trust: {info['trust']}")
        print(f"Author: {info['author'] or 'Unknown'}")
        print(f"Homepage: {info['homepage'] or 'N/A'}")
        print(f"License: {info['license'] or 'Unknown'}")
        print()
        print(f"Loaded: {'Yes' if info['loaded'] else 'No'}")
        print(f"Enabled: {'Yes' if info['enabled'] else 'No'}")

        if info.get('path'):
            print(f"Path: {info['path']}")

        if info.get('tools'):
            print(f"\n{Colors.BOLD}Tools:{Colors.NC}")
            for tool in info['tools']:
                print(f"  - {tool}")

        if info.get('workflows'):
            print(f"\n{Colors.BOLD}Workflows:{Colors.NC}")
            for wf in info['workflows']:
                print(f"  - {wf}")

        if info.get('capabilities'):
            print(f"\n{Colors.BOLD}Capabilities:{Colors.NC}")
            for cap in info['capabilities']:
                print(f"  - {cap}")

        return 0

    except ImportError as e:
        print_error(f"Plugin system not available: {e}")
        return 1


def cmd_plugin_load(args: argparse.Namespace) -> int:
    """Load a plugin."""
    plugin_name = args.name

    try:
        from ragix_core.plugin_system import get_plugin_manager

        pm = get_plugin_manager()
        plugin = pm.load_plugin(plugin_name)

        if plugin:
            print_ok(f"Loaded plugin: {plugin_name}")
            print_info(f"Tools: {list(plugin.loaded_tools.keys())}")
            return 0
        else:
            print_error(f"Failed to load plugin: {plugin_name}")
            return 1

    except Exception as e:
        print_error(f"Error loading plugin: {e}")
        return 1


def cmd_plugin_unload(args: argparse.Namespace) -> int:
    """Unload a plugin."""
    plugin_name = args.name

    try:
        from ragix_core.plugin_system import get_plugin_manager

        pm = get_plugin_manager()

        if pm.unload_plugin(plugin_name):
            print_ok(f"Unloaded plugin: {plugin_name}")
            return 0
        else:
            print_error(f"Plugin not loaded: {plugin_name}")
            return 1

    except Exception as e:
        print_error(f"Error unloading plugin: {e}")
        return 1


def cmd_plugin_create(args: argparse.Namespace) -> int:
    """Create a new plugin scaffold."""
    plugin_name = args.name
    plugin_type = args.type or "tool"

    # Determine location
    if args.global_plugin:
        plugins_dir = Path.home() / ".ragix" / "plugins"
    else:
        plugins_dir = Path.cwd() / "plugins"

    plugin_dir = plugins_dir / plugin_name

    if plugin_dir.exists():
        print_error(f"Plugin directory already exists: {plugin_dir}")
        return 1

    print_header(f"Creating Plugin: {plugin_name}")

    # Create directory structure
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # Create plugin.yaml
    manifest = {
        "name": plugin_name,
        "version": "0.1.0",
        "description": f"RAGIX {plugin_type} plugin",
        "type": plugin_type,
        "trust": "untrusted",
        "author": "",
        "homepage": "",
        "license": "MIT",
        "capabilities": ["file:read"],
    }

    if plugin_type == "tool":
        manifest["tools"] = [
            {
                "name": f"{plugin_name}_example",
                "description": f"Example tool from {plugin_name}",
                "entry": f"{plugin_name}_tools:example_tool",
                "parameters": [
                    {
                        "name": "input",
                        "type": "string",
                        "description": "Input value",
                        "required": True,
                    }
                ],
            }
        ]

        # Create tool module
        tool_code = f'''"""
{plugin_name} Tools - Example plugin tools

Author: [Your Name]
"""


def example_tool(input: str) -> dict:
    """
    Example tool function.

    Args:
        input: Input value

    Returns:
        Result dictionary
    """
    return {{
        "success": True,
        "input": input,
        "output": f"Processed: {{input}}",
    }}
'''

        with open(plugin_dir / f"{plugin_name}_tools.py", 'w') as f:
            f.write(tool_code)

    elif plugin_type == "workflow":
        manifest["workflows"] = [
            {
                "name": f"{plugin_name}_workflow",
                "description": f"Example workflow from {plugin_name}",
                "entry": "workflow.yaml",
                "parameters": [],
            }
        ]

        # Create workflow YAML
        workflow = {
            "name": f"{plugin_name}_workflow",
            "description": "Example workflow",
            "steps": [
                {
                    "name": "step1",
                    "action": "respond",
                    "message": "Workflow step 1 completed",
                }
            ],
        }

        import yaml
        with open(plugin_dir / "workflow.yaml", 'w') as f:
            yaml.safe_dump(workflow, f, default_flow_style=False)

    # Write manifest
    import yaml
    with open(plugin_dir / "plugin.yaml", 'w') as f:
        yaml.safe_dump(manifest, f, default_flow_style=False)

    # Create README
    readme = f"""# {plugin_name}

A RAGIX {plugin_type} plugin.

## Installation

Copy this directory to `~/.ragix/plugins/` or `./plugins/`.

## Usage

```bash
ragix plugin load {plugin_name}
ragix plugin info {plugin_name}
```

## Development

1. Edit `plugin.yaml` to define tools/workflows
2. Implement tool functions in `*_tools.py`
3. Test with `ragix plugin load {plugin_name}`
"""

    with open(plugin_dir / "README.md", 'w') as f:
        f.write(readme)

    print_ok(f"Created plugin directory: {plugin_dir}")
    print_ok(f"Created plugin.yaml")
    print_ok(f"Created {plugin_type} scaffold")
    print()
    print_info("Next steps:")
    print(f"  1. Edit {plugin_dir}/plugin.yaml")
    print(f"  2. Implement your {plugin_type}s")
    print(f"  3. Run: ragix plugin load {plugin_name}")

    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """List available tools."""
    print_header("Available Tools")

    try:
        from ragix_core.tool_registry import get_tool_registry

        registry = get_tool_registry()
        tools_by_provider = registry.list_tools_by_provider()

        for provider, tool_names in sorted(tools_by_provider.items()):
            print(f"\n{Colors.BOLD}{provider.upper()}{Colors.NC}")

            for name in sorted(tool_names):
                info = registry.get_tool_info(name)
                if info:
                    enabled = "✓" if info["enabled"] else "✗"
                    color = Colors.GREEN if info["enabled"] else Colors.RED
                    print(f"  {color}{enabled}{Colors.NC} {name}: {info['description'][:50]}...")

        return 0

    except ImportError as e:
        print_error(f"Tool registry not available: {e}")
        return 1


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main CLI entry point."""
    # Disable colors if not TTY
    if not sys.stdout.isatty():
        Colors.disable()

    parser = argparse.ArgumentParser(
        prog="ragix",
        description="RAGIX - Sovereign AI Development Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ragix doctor          Run system diagnostics
  ragix install         Setup RAGIX environment
  ragix config          Show current configuration
  ragix status          Quick status check
  ragix logs -n 50      View last 50 log entries
  ragix verify          Verify log integrity
  ragix mcp             Start MCP server
  ragix web             Start web interface
  ragix run             Start interactive agent
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # doctor
    sub = subparsers.add_parser("doctor", help="Run system diagnostics")
    sub.set_defaults(func=cmd_doctor)

    # install
    sub = subparsers.add_parser("install", help="Setup RAGIX environment")
    sub.set_defaults(func=cmd_install)

    # config
    sub = subparsers.add_parser("config", help="Show current configuration")
    sub.set_defaults(func=cmd_config)

    # status
    sub = subparsers.add_parser("status", help="Quick status check")
    sub.set_defaults(func=cmd_status)

    # logs
    sub = subparsers.add_parser("logs", help="View recent logs")
    sub.add_argument("-n", "--lines", type=int, default=20, help="Number of lines to show")
    sub.set_defaults(func=cmd_logs)

    # verify
    sub = subparsers.add_parser("verify", help="Verify log integrity")
    sub.set_defaults(func=cmd_verify)

    # mcp
    sub = subparsers.add_parser("mcp", help="Start MCP server")
    sub.set_defaults(func=cmd_mcp)

    # web
    sub = subparsers.add_parser("web", help="Start web interface")
    sub.add_argument("-p", "--port", type=int, default=8501, help="Port number")
    sub.set_defaults(func=cmd_web)

    # run
    sub = subparsers.add_parser("run", help="Start interactive agent")
    sub.set_defaults(func=cmd_run)

    # upgrade
    sub = subparsers.add_parser("upgrade", help="Upgrade RAGIX")
    sub.set_defaults(func=cmd_upgrade)

    # tools
    sub = subparsers.add_parser("tools", help="List available tools")
    sub.set_defaults(func=cmd_tools)

    # plugin commands
    plugin_parser = subparsers.add_parser("plugin", help="Plugin management")
    plugin_sub = plugin_parser.add_subparsers(dest="plugin_command")

    # plugin list
    sub = plugin_sub.add_parser("list", help="List available plugins")
    sub.set_defaults(func=cmd_plugin_list)

    # plugin info
    sub = plugin_sub.add_parser("info", help="Show plugin details")
    sub.add_argument("name", help="Plugin name")
    sub.set_defaults(func=cmd_plugin_info)

    # plugin load
    sub = plugin_sub.add_parser("load", help="Load a plugin")
    sub.add_argument("name", help="Plugin name")
    sub.set_defaults(func=cmd_plugin_load)

    # plugin unload
    sub = plugin_sub.add_parser("unload", help="Unload a plugin")
    sub.add_argument("name", help="Plugin name")
    sub.set_defaults(func=cmd_plugin_unload)

    # plugin create
    sub = plugin_sub.add_parser("create", help="Create a new plugin")
    sub.add_argument("name", help="Plugin name")
    sub.add_argument("-t", "--type", choices=["tool", "workflow"], default="tool",
                     help="Plugin type (default: tool)")
    sub.add_argument("-g", "--global", dest="global_plugin", action="store_true",
                     help="Create in global plugins directory")
    sub.set_defaults(func=cmd_plugin_create)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
