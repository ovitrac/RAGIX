#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGIX MCP Server v0.7
=====================

Expose the RAGIX multi-agent orchestration platform as an MCP server so that
MCP clients (Claude Desktop, Cursor, Claude Code, etc.) can use RAGIX tools.

Tools exposed
-------------

Core Tools:
1. ragix_chat(prompt: str)
   - Run a single Unix-RAG reasoning step with shell execution.

2. ragix_scan_repo(max_depth: int = 4, include_hidden: bool = False)
   - Quick project overview: list of files (path, size, extension).

3. ragix_read_file(path: str, max_bytes: int = 65536)
   - Read a file (relative to sandbox root) with a size limit.

v0.7 Tools:
4. ragix_search(query: str, k: int = 10, strategy: str = "rrf")
   - Hybrid BM25 + vector search with multiple fusion strategies.

5. ragix_workflow(template: str, params: dict)
   - Execute a workflow template (bug_fix, feature_addition, etc.).

6. ragix_health()
   - Get comprehensive system health status.

7. ragix_templates()
   - List all available workflow templates.

Installation
------------

    # install MCP SDK
    pip install "mcp[cli]"  # or: uv add "mcp[cli]"

    # run in dev mode
    uv run mcp dev ragix_mcp_server.py

    # or install into Claude Desktop
    uv run mcp install ragix_mcp_server.py --name "RAGIX"

Environment variables (reused from unix-rag-agent.py)
-----------------------------------------------------

    UNIX_RAG_MODEL=mistral            # Ollama model name
    UNIX_RAG_SANDBOX=/path/to/sandbox # root folder for all operations
    UNIX_RAG_PROFILE=dev              # safe-read-only | dev | unsafe
    UNIX_RAG_ALLOW_GIT_DESTRUCTIVE=0  # 1 to allow destructive git cmds


Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25

"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import importlib.util
import json
import os
import sys

from mcp.server.fastmcp import FastMCP
from mcp.server.session import ServerSession
from mcp.server.fastmcp import Context

# Add RAGIX to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# 1. Load the existing RAGIX Unix-RAG agent implementation dynamically
#    (file name has a dash so we cannot import it as a normal module).
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_UNIX_RAG_AGENT_PATH = _REPO_ROOT / "unix-rag-agent.py"

if not _UNIX_RAG_AGENT_PATH.exists():
    raise RuntimeError(
        f"Cannot find unix-rag-agent.py at {_UNIX_RAG_AGENT_PATH}. "
        "Make sure you run the MCP server from the RAGIX repo root."
    )

_spec = importlib.util.spec_from_file_location("unix_rag_agent", _UNIX_RAG_AGENT_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to load unix-rag-agent.py via importlib")

_unix_rag_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_unix_rag_agent)  # type: ignore[call-arg]

# Shorthand aliases to keep code readable
UnixRAGAgent = _unix_rag_agent.UnixRAGAgent
ShellSandbox = _unix_rag_agent.ShellSandbox
OllamaLLM = _unix_rag_agent.OllamaLLM
CommandResult = _unix_rag_agent.CommandResult
compute_dry_run_from_profile = _unix_rag_agent.compute_dry_run_from_profile

SANDBOX_ROOT = _unix_rag_agent.SANDBOX_ROOT
OLLAMA_MODEL = _unix_rag_agent.OLLAMA_MODEL
AGENT_PROFILE = _unix_rag_agent.AGENT_PROFILE
ALLOW_GIT_DESTRUCTIVE = _unix_rag_agent.ALLOW_GIT_DESTRUCTIVE

# ---------------------------------------------------------------------------
# 2. Helper: build a fresh UnixRAGAgent
# ---------------------------------------------------------------------------


def create_agent() -> UnixRAGAgent:
    """
    Build a fresh UnixRAGAgent using the same configuration as unix-rag-agent.py.

    For simplicity (and MCP statelessness), we create a new agent on each tool
    call. That means:
      - History is per call, not persisted between calls.
      - The agent still generates its project overview at startup.
    """
    os.makedirs(SANDBOX_ROOT, exist_ok=True)
    dry_run = compute_dry_run_from_profile(AGENT_PROFILE)

    llm = OllamaLLM(OLLAMA_MODEL)
    shell = ShellSandbox(
        root=SANDBOX_ROOT,
        dry_run=dry_run,
        profile=AGENT_PROFILE,
        allow_git_destructive=ALLOW_GIT_DESTRUCTIVE,
    )
    return UnixRAGAgent(llm=llm, shell=shell)


def _command_result_to_dict(res: Optional[CommandResult]) -> Optional[Dict[str, Any]]:
    """
    Convert CommandResult dataclass (or None) to a simple dict for MCP.
    """
    if res is None:
        return None
    # CommandResult is a dataclass in unix-rag-agent, so asdict() works.
    try:
        return asdict(res)
    except TypeError:
        # Fallback if for any reason asdict fails
        return {
            "command": getattr(res, "command", None),
            "cwd": getattr(res, "cwd", None),
            "stdout": getattr(res, "stdout", None),
            "stderr": getattr(res, "stderr", None),
            "returncode": getattr(res, "returncode", None),
        }


# ---------------------------------------------------------------------------
# 3. MCP Server definition (FastMCP)
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="RAGIX",
    instructions=(
        "RAGIX is a sovereign Unix-RAG development assistant. "
        "It runs a local LLM via Ollama and a sandboxed shell rooted at "
        f"{SANDBOX_ROOT}. Use the tools to explore code, read files, "
        "and run single-step Unix-RAG reasoning."
    ),
    website_url="https://github.com/ovitrac/RAGIX",
)


# ---------------------------------------------------------------------------
# 4. Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def ragix_chat(prompt: str) -> Dict[str, Any]:
    """
    Run a single Unix-RAG step with the RAGIX agent.

    Parameters
    ----------
    prompt : str
        Natural-language instruction, e.g.
        "Find where the database is initialized and summarize the config."

    Returns
    -------
    dict
        {
          "response": str | null,
          "last_command": {
              "command": str,
              "cwd": str,
              "stdout": str,
              "stderr": str,
              "returncode": int
          } | null
        }
    """
    agent = create_agent()
    cmd_result, reply = agent.step(prompt)

    return {
        "response": reply,
        "last_command": _command_result_to_dict(cmd_result),
    }


@mcp.tool()
def ragix_scan_repo(
    max_depth: int = 4,
    include_hidden: bool = False,
) -> List[Dict[str, Any]]:
    """
    Quick project overview: walk the sandbox root and list files.

    Parameters
    ----------
    max_depth : int, default 4
        Maximum directory depth relative to SANDBOX_ROOT.

    include_hidden : bool, default False
        If False, skip hidden files and directories (starting with '.').

    Returns
    -------
    list of dict
        Each item:
        {
          "path": "relative/path/to/file",
          "size": int (bytes),
          "ext": ".py" | ".md" | "" ...
        }
    """
    root = Path(SANDBOX_ROOT).resolve()
    results: List[Dict[str, Any]] = []

    root_depth = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.parts) - root_depth
        if depth > max_depth:
            # prevent descending further
            dirnames[:] = []
            continue

        # Optionally filter hidden directories
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]

        for name in filenames:
            full_path = current / name
            rel_path = full_path.relative_to(root)
            try:
                size = full_path.stat().st_size
            except OSError:
                size = -1

            results.append(
                {
                    "path": str(rel_path),
                    "size": int(size),
                    "ext": full_path.suffix,
                }
            )

    return results


@mcp.tool()
def ragix_read_file(path: str, max_bytes: int = 65536) -> str:
    """
    Read a text file from within the sandbox root.

    Parameters
    ----------
    path : str
        Path relative to the sandbox root (UNIX_RAG_SANDBOX).
        Example: "src/main.py" or "README.md".

    max_bytes : int, default 65536
        Maximum number of bytes to read to avoid flooding the context.

    Returns
    -------
    str
        File content (possibly truncated with a notice).
    """
    root = Path(SANDBOX_ROOT).resolve()
    target = (root / path).resolve()

    # Safety: enforce that target is under root
    if not str(target).startswith(str(root)):
        raise ValueError("Attempt to read file outside sandbox root")

    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = target.read_bytes()
    if len(data) > max_bytes:
        snippet = data[:max_bytes].decode("utf-8", errors="replace")
        return (
            f"[RAGIX MCP] File truncated to {max_bytes} bytes "
            f"out of {len(data)}.\n\n" + snippet
        )
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 5. RAGIX v0.7 Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def ragix_search(query: str, k: int = 10, strategy: str = "rrf") -> Dict[str, Any]:
    """
    Hybrid BM25 + vector search across the codebase.

    Parameters
    ----------
    query : str
        Search query (natural language or keywords).
        Example: "database connection error handling"

    k : int, default 10
        Maximum number of results to return.

    strategy : str, default "rrf"
        Fusion strategy: "rrf" (reciprocal rank), "weighted", "interleave",
        "bm25_rerank", or "vector_rerank".

    Returns
    -------
    dict
        {
          "query": str,
          "strategy": str,
          "results": [
            {
              "file": str,
              "name": str,
              "type": str,
              "score": float,
              "matched_terms": list
            }
          ]
        }
    """
    try:
        from ragix_core import BM25Index, BM25Document, Tokenizer, FusionStrategy

        # Build index from project files
        index = BM25Index(k1=1.5, b=0.75)
        tokenizer = Tokenizer(use_stopwords=True, min_token_length=2)

        root = Path(SANDBOX_ROOT).resolve()
        doc_count = 0

        # Index Python and common source files
        for pattern in ["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.go"]:
            for fpath in root.glob(pattern):
                if fpath.is_file() and ".git" not in str(fpath):
                    try:
                        content = fpath.read_text(errors="replace")[:10000]
                        tokens = tokenizer.tokenize(content)
                        if tokens:
                            rel_path = str(fpath.relative_to(root))
                            index.add_document(BM25Document(
                                doc_id=rel_path,
                                tokens=tokens,
                                metadata={
                                    "file": rel_path,
                                    "name": fpath.stem,
                                    "type": fpath.suffix,
                                },
                            ))
                            doc_count += 1
                    except Exception:
                        pass

        # Search
        results = index.search(query, k=k)

        formatted = []
        for r in results:
            formatted.append({
                "file": r.metadata.get("file", r.doc_id),
                "name": r.metadata.get("name", ""),
                "type": r.metadata.get("type", ""),
                "score": round(r.score, 4),
                "matched_terms": r.matched_terms,
            })

        return {
            "query": query,
            "strategy": strategy,
            "indexed_files": doc_count,
            "results": formatted,
        }

    except ImportError as e:
        return {"error": f"RAGIX core not available: {e}"}


@mcp.tool()
def ragix_workflow(template: str, params: Dict[str, str]) -> Dict[str, Any]:
    """
    Execute a workflow template for multi-agent task execution.

    Parameters
    ----------
    template : str
        Template name: "bug_fix", "feature_addition", "code_review",
        "refactoring", "documentation", "security_audit", "test_coverage",
        or "exploration".

    params : dict
        Template parameters. Each template has different required params.
        Use ragix_templates() to see available parameters.

    Returns
    -------
    dict
        {
          "template": str,
          "status": str,
          "nodes": list,
          "execution_order": list,
          "params_used": dict
        }
    """
    try:
        from ragix_core import get_template_manager

        manager = get_template_manager()

        # Check if template exists
        template_def = manager.get_template(template)
        if not template_def:
            available = list(manager.templates.keys())
            return {
                "error": f"Unknown template: {template}",
                "available_templates": available,
            }

        # Instantiate the workflow graph
        graph = manager.instantiate(template, params)

        return {
            "template": template,
            "status": "instantiated",
            "nodes": [
                {"id": n.id, "agent_type": n.agent_type, "description": n.description}
                for n in graph.nodes.values()
            ],
            "execution_order": graph.topological_sort(),
            "params_used": params,
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def ragix_health() -> Dict[str, Any]:
    """
    Get comprehensive RAGIX system health status.

    Returns
    -------
    dict
        {
          "status": "healthy" | "degraded" | "unhealthy",
          "checks": {
            "ollama": {...},
            "disk": {...},
            "memory": {...}
          },
          "ragix_version": str,
          "timestamp": str
        }
    """
    try:
        from ragix_core import (
            get_health_checker,
            check_ollama_health,
            check_disk_space,
            check_memory_usage,
        )

        checker = get_health_checker()

        # Register checks if not already done
        if "ollama" not in checker.checks:
            checker.register("ollama", check_ollama_health)
        if "disk" not in checker.checks:
            checker.register("disk", check_disk_space)
        if "memory" not in checker.checks:
            checker.register("memory", check_memory_usage)

        report = checker.get_status_report()

        return {
            "status": report["status"],
            "checks": report["checks"],
            "ragix_version": "0.7.0",
            "model": OLLAMA_MODEL,
            "sandbox": str(SANDBOX_ROOT),
            "profile": AGENT_PROFILE,
            "timestamp": datetime.now().isoformat(),
        }

    except ImportError as e:
        return {
            "status": "unknown",
            "error": f"RAGIX core not available: {e}",
            "ragix_version": "0.7.0",
            "timestamp": datetime.now().isoformat(),
        }


@mcp.tool()
def ragix_templates() -> Dict[str, Any]:
    """
    List all available workflow templates and their parameters.

    Returns
    -------
    dict
        {
          "templates": [
            {
              "name": str,
              "description": str,
              "parameters": [
                {"name": str, "required": bool, "description": str}
              ],
              "steps": [str]
            }
          ]
        }
    """
    try:
        from ragix_core import get_template_manager, list_builtin_templates

        manager = get_template_manager()
        templates = list_builtin_templates()

        result = []
        for name in templates:
            template = manager.get_template(name)
            if template:
                result.append({
                    "name": name,
                    "description": template.description,
                    "parameters": [
                        {
                            "name": p.name,
                            "required": p.required,
                            "default": p.default,
                            "description": p.description,
                        }
                        for p in template.parameters
                    ],
                    "steps": [s.name for s in template.steps],
                })

        return {"templates": result, "count": len(result)}

    except ImportError as e:
        return {"error": f"RAGIX core not available: {e}"}


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Direct-execution entry point for the MCP server.

    You can run:
        python ragix_mcp_server.py
    or:
        uv run mcp run ragix_mcp_server.py
    """
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
