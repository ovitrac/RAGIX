"""
CLI Entry Point for RAGIX Unix Agent

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

try:
    import tomli as tomllib
except ImportError:
    try:
        import tomllib
    except ImportError:
        tomllib = None

from ragix_core import OllamaLLM, ShellSandbox, compute_dry_run_from_profile
from ragix_unix import UnixRAGAgent


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    if not config_path.exists():
        return {}

    if tomllib is None:
        print(f"Warning: tomllib not available, skipping config file", file=sys.stderr)
        return {}

    with config_path.open("rb") as f:
        return tomllib.load(f)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for ragix-unix-agent CLI."""
    parser = argparse.ArgumentParser(
        prog="ragix-unix-agent",
        description="RAGIX Unix-RAG development assistant with local LLM"
    )

    parser.add_argument(
        "--sandbox-root",
        type=str,
        help="Sandbox root directory (default: ~/agent-sandbox)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Ollama model name (default: mistral)"
    )
    parser.add_argument(
        "--profile",
        choices=["strict", "dev", "unsafe"],
        help="Safety profile (default: dev)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enable dry-run mode (no command execution)"
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Disable dry-run mode"
    )
    parser.add_argument(
        "--allow-git-destructive",
        action="store_true",
        help="Allow destructive git commands (git reset --hard, etc.)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (default: ~/.ragix.toml)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shortcut for --log-level DEBUG)"
    )

    return parser


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Resolve configuration with precedence: CLI > ENV > CONFIG > DEFAULTS

    Returns config dict with final values.
    """
    # Step 1: Defaults
    config = {
        "sandbox_root": "~/agent-sandbox",
        "model": "mistral",
        "profile": "dev",
        "dry_run": None,  # None means compute from profile
        "allow_git_destructive": False,
        "log_level": "INFO",
    }

    # Step 2: Config file
    config_path = args.config or Path.home() / ".ragix.toml"
    if config_path.exists():
        file_config = load_config(config_path)
        config.update(file_config.get("unix_agent", {}))

    # Step 3: Environment variables
    if os.environ.get("UNIX_RAG_SANDBOX"):
        config["sandbox_root"] = os.environ["UNIX_RAG_SANDBOX"]
    if os.environ.get("UNIX_RAG_MODEL"):
        config["model"] = os.environ["UNIX_RAG_MODEL"]
    if os.environ.get("UNIX_RAG_PROFILE"):
        config["profile"] = os.environ["UNIX_RAG_PROFILE"].lower()
    if os.environ.get("UNIX_RAG_ALLOW_GIT_DESTRUCTIVE") == "1":
        config["allow_git_destructive"] = True

    # Step 4: CLI arguments (highest priority)
    if args.sandbox_root:
        config["sandbox_root"] = args.sandbox_root
    if args.model:
        config["model"] = args.model
    if args.profile:
        config["profile"] = args.profile
    if args.dry_run:
        config["dry_run"] = True
    elif args.no_dry_run:
        config["dry_run"] = False
    if args.allow_git_destructive:
        config["allow_git_destructive"] = True
    if args.debug:
        config["log_level"] = "DEBUG"
    elif args.log_level:
        config["log_level"] = args.log_level

    # Expand paths
    config["sandbox_root"] = os.path.expanduser(config["sandbox_root"])
    config["sandbox_root"] = os.path.abspath(config["sandbox_root"])

    return config


def main():
    """Main entry point for ragix-unix-agent CLI."""
    parser = build_parser()
    args = parser.parse_args()

    # Resolve configuration
    config = resolve_config(args)

    # Compute dry-run if not explicitly set
    if config["dry_run"] is None:
        config["dry_run"] = compute_dry_run_from_profile(config["profile"], False)

    # Ensure sandbox exists
    os.makedirs(config["sandbox_root"], exist_ok=True)

    # Instantiate components
    try:
        llm = OllamaLLM(config["model"])
        shell = ShellSandbox(
            root=config["sandbox_root"],
            dry_run=config["dry_run"],
            profile=config["profile"],
            allow_git_destructive=config["allow_git_destructive"],
        )
        agent = UnixRAGAgent(llm=llm, shell=shell)

        # Start interactive loop
        agent.interactive_loop()

    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
