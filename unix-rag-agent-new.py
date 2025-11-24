#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unix-RAG Agent (Thin Wrapper for v0.5)
========================================

DEPRECATED: This script is a thin wrapper for backward compatibility.
Use `ragix-unix-agent` CLI or import from ragix_unix package instead.

For new code, use:
    from ragix_unix import UnixRAGAgent
    from ragix_core import OllamaLLM, ShellSandbox, compute_dry_run_from_profile

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import os
from ragix_core import OllamaLLM, ShellSandbox, compute_dry_run_from_profile
from ragix_unix import UnixRAGAgent


# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION (backward compatibility with v0.4)
# ---------------------------------------------------------------------------

# LLM model served by Ollama (e.g. "mistral", "mistral:instruct")
OLLAMA_MODEL = os.environ.get("UNIX_RAG_MODEL", "mistral")

# Sandbox root directory – all shell operations are confined here
SANDBOX_ROOT = os.path.abspath(os.environ.get("UNIX_RAG_SANDBOX", "~/agent-sandbox"))
SANDBOX_ROOT = os.path.expanduser(SANDBOX_ROOT)

# Default dry-run behavior in "dev" profile (can be overridden by profile)
DEFAULT_DRY_RUN = False

# Profiles / modes:
# - "safe-read-only" / "strict": dry-run ON, very conservative behavior
# - "dev": default; dry-run as per DEFAULT_DRY_RUN, strong denylist
# - "unsafe": same as dev but allows git destructive commands if explicitly enabled
AGENT_PROFILE = os.environ.get("UNIX_RAG_PROFILE", "dev").lower()

# Allow destructive git operations if explicitly confirmed via environment variable
ALLOW_GIT_DESTRUCTIVE = os.environ.get("UNIX_RAG_ALLOW_GIT_DESTRUCTIVE", "0") == "1"


def main():
    """Main entry point for backward-compatible unix-rag-agent.py"""
    # Ensure sandbox directory exists
    os.makedirs(SANDBOX_ROOT, exist_ok=True)

    # Configure dry-run based on profile
    dry_run = compute_dry_run_from_profile(AGENT_PROFILE, DEFAULT_DRY_RUN)

    # Instantiate components
    llm = OllamaLLM(OLLAMA_MODEL)
    shell = ShellSandbox(
        root=SANDBOX_ROOT,
        dry_run=dry_run,
        profile=AGENT_PROFILE,
        allow_git_destructive=ALLOW_GIT_DESTRUCTIVE,
    )
    agent = UnixRAGAgent(llm=llm, shell=shell)

    # Start interactive loop
    agent.interactive_loop()


if __name__ == "__main__":
    print("Note: This script is deprecated. Use 'ragix-unix-agent' CLI instead.")
    print("Loading Unix-RAG Agent from modular packages...\n")
    main()
