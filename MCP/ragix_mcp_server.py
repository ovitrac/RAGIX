#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGIX MCP Server v0.8.0
=======================

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

v0.7.1 Tools:
8. ragix_config()
   - Get current RAGIX configuration.

9. ragix_verify_logs()
   - Verify log integrity (SHA256 chain).

10. ragix_logs(n: int = 50)
    - Get recent log entries.

v0.8.0 Tools:
11. ragix_ast_scan(path: str, language: str)
    - Extract AST symbols from source code.

12. ragix_ast_metrics(path: str, language: str)
    - Compute code metrics (complexity, LOC, maintainability).

13. ragix_models_list()
    - List available Ollama models.

14. ragix_model_info(model: str)
    - Get detailed model information.

15. ragix_system_info()
    - Get comprehensive system information (GPU, CPU, memory).

KOAS Tools Enhanced:
- koas_run: Added parallel execution support (parallel=True, workers=4)

Installation
------------

    # install MCP SDK
    pip install "mcp[cli]"  # or: uv add "mcp[cli]"

    # run in dev mode
    uv run mcp dev ragix_mcp_server.py

    # or install into Claude Desktop
    uv run mcp install ragix_mcp_server.py --name "RAGIX"

Configuration
-------------

RAGIX v0.7.1 uses ragix.yaml for unified configuration.
Environment variables override config values:

    RAGIX_LLM_MODEL=mistral
    RAGIX_SANDBOX_ROOT=/path/to/sandbox
    RAGIX_PROFILE=dev
    RAGIX_AIR_GAPPED=false

Legacy environment variables (backward compatibility):
    UNIX_RAG_MODEL=mistral
    UNIX_RAG_SANDBOX=/path/to/sandbox
    UNIX_RAG_PROFILE=dev
    UNIX_RAG_ALLOW_GIT_DESTRUCTIVE=0


Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26

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
            "ragix_version": "0.7.1",
            "model": OLLAMA_MODEL,
            "sandbox": str(SANDBOX_ROOT),
            "profile": AGENT_PROFILE,
            "timestamp": datetime.now().isoformat(),
        }

    except ImportError as e:
        return {
            "status": "unknown",
            "error": f"RAGIX core not available: {e}",
            "ragix_version": "0.7.1",
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
# 6. RAGIX v0.7.1 Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def ragix_config() -> Dict[str, Any]:
    """
    Get current RAGIX configuration.

    Returns
    -------
    dict
        {
          "version": str,
          "llm": {"backend": str, "model": str, ...},
          "safety": {"profile": str, "air_gapped": bool, ...},
          "mcp": {"enabled": bool, "port": int, ...},
          "search": {"enabled": bool, "fusion_strategy": str, ...}
        }
    """
    try:
        from ragix_core import get_config, find_config_file

        config_path = find_config_file()
        config = get_config()

        return {
            "config_file": str(config_path) if config_path else None,
            "version": config.version,
            "sandbox_root": config.sandbox_root,
            "llm": {
                "backend": config.llm.backend,
                "model": config.llm.model,
                "base_url": config.llm.base_url,
                "sovereignty": "sovereign" if config.llm.backend == "ollama" else "cloud",
            },
            "safety": {
                "profile": config.safety.profile,
                "air_gapped": config.safety.air_gapped,
                "log_hashing": config.safety.log_hashing,
                "allow_git_destructive": config.safety.allow_git_destructive,
            },
            "mcp": {
                "enabled": config.mcp.enabled,
                "port": config.mcp.port,
            },
            "search": {
                "enabled": config.search.enabled,
                "fusion_strategy": config.search.fusion_strategy,
                "embedding_model": config.search.embedding_model,
            },
        }

    except ImportError as e:
        return {"error": f"RAGIX config not available: {e}"}


@mcp.tool()
def ragix_verify_logs() -> Dict[str, Any]:
    """
    Verify log integrity using SHA256 chain hashing.

    Returns
    -------
    dict
        {
          "valid": bool,
          "total_entries": int,
          "verified_entries": int,
          "first_invalid_entry": int | null,
          "errors": list[str],
          "verification_time": str
        }
    """
    try:
        from ragix_core import ChainedLogHasher, LogIntegrityReport
        from pathlib import Path

        log_dir = Path(".agent_logs")
        hasher = ChainedLogHasher(log_dir=log_dir)
        report = hasher.verify_chain()

        return {
            "valid": report.valid,
            "total_entries": report.total_entries,
            "verified_entries": report.verified_entries,
            "first_invalid_entry": report.first_invalid_entry,
            "errors": report.errors[:10],  # Limit errors
            "log_file": report.log_file,
            "verification_time": report.verification_time,
        }

    except ImportError as e:
        return {"error": f"RAGIX log_integrity not available: {e}"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def ragix_logs(n: int = 50) -> Dict[str, Any]:
    """
    Get recent log entries.

    Parameters
    ----------
    n : int, default 50
        Number of recent entries to return.

    Returns
    -------
    dict
        {
          "entries": list[str],
          "total": int,
          "log_file": str
        }
    """
    from pathlib import Path

    log_file = Path(".agent_logs/commands.log")

    if not log_file.exists():
        return {
            "entries": [],
            "total": 0,
            "log_file": str(log_file),
            "error": "Log file not found",
        }

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        total = len(lines)
        recent = [line.strip() for line in lines[-n:]]

        return {
            "entries": recent,
            "total": total,
            "returned": len(recent),
            "log_file": str(log_file),
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def ragix_agent_step(prompt: str, persist_history: bool = False) -> Dict[str, Any]:
    """
    Execute a full agent step with config-based instantiation.

    This is an enhanced version of ragix_chat that uses ragix.yaml configuration
    and supports optional history persistence.

    Parameters
    ----------
    prompt : str
        Natural-language instruction for the agent.

    persist_history : bool, default False
        If True, maintain conversation history across calls (experimental).

    Returns
    -------
    dict
        {
          "response": str,
          "last_command": {...},
          "config": {"model": str, "profile": str},
          "sovereignty": str
        }
    """
    try:
        from ragix_core import get_config

        config = get_config()

        # Use config for agent creation
        agent = create_agent()
        cmd_result, reply = agent.step(prompt)

        return {
            "response": reply,
            "last_command": _command_result_to_dict(cmd_result),
            "config": {
                "model": config.llm.model,
                "profile": config.safety.profile,
                "backend": config.llm.backend,
            },
            "sovereignty": "sovereign" if config.llm.backend == "ollama" else "cloud",
        }

    except ImportError:
        # Fall back to legacy behavior
        agent = create_agent()
        cmd_result, reply = agent.step(prompt)

        return {
            "response": reply,
            "last_command": _command_result_to_dict(cmd_result),
            "config": {
                "model": OLLAMA_MODEL,
                "profile": AGENT_PROFILE,
                "backend": "ollama",
            },
            "sovereignty": "sovereign",
        }


# ---------------------------------------------------------------------------
# 7. KOAS Tools (Kernel-Orchestrated Audit System)
# ---------------------------------------------------------------------------

@mcp.tool()
def koas_init(
    project_path: str,
    project_name: str = "Project",
    language: str = "python",
    output_language: str = "en",
) -> Dict[str, Any]:
    """
    Initialize a KOAS audit workspace for a project.

    Parameters
    ----------
    project_path : str
        Path to the project to audit (absolute or relative to sandbox).

    project_name : str
        Human-readable name for the project.

    language : str, default "python"
        Programming language: "python", "java", "typescript", etc.

    output_language : str, default "en"
        Report language: "en" (English) or "fr" (French).

    Returns
    -------
    dict
        {
          "workspace": str,
          "manifest": str,
          "status": "initialized",
          "project": {...}
        }
    """
    try:
        from ragix_kernels.orchestrator import AuditOrchestrator

        # Resolve project path
        project = Path(project_path)
        if not project.is_absolute():
            project = Path(SANDBOX_ROOT) / project_path

        # Create workspace in temp or project-adjacent location
        workspace = project.parent / f".koas_audit_{project_name.replace(' ', '_').lower()}"

        # Initialize
        orchestrator = AuditOrchestrator(
            workspace=workspace,
            project_name=project_name,
            project_path=project,
            language=language,
        )
        orchestrator.init_workspace()

        # Update manifest with output language
        manifest_path = workspace / "manifest.yaml"
        if manifest_path.exists():
            import yaml
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            manifest.setdefault("output", {})["language"] = output_language
            with open(manifest_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False)

        return {
            "workspace": str(workspace),
            "manifest": str(manifest_path),
            "status": "initialized",
            "project": {
                "name": project_name,
                "path": str(project),
                "language": language,
            },
            "output_language": output_language,
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def koas_run(
    workspace: str,
    stage: int = 0,
    kernels: str = "",
    parallel: bool = False,
    workers: int = 4,
) -> Dict[str, Any]:
    """
    Run KOAS audit stages on an initialized workspace.

    Parameters
    ----------
    workspace : str
        Path to the KOAS workspace (containing manifest.yaml).

    stage : int, default 0
        Stage to run: 1 (collection), 2 (analysis), 3 (reporting).
        0 means run all stages.

    kernels : str
        Comma-separated list of specific kernels to run (optional).
        Example: "ast_scan,metrics" or "section_executive,report_assemble"

    parallel : bool, default False
        Enable parallel execution of kernels within stages.
        Uses dependency-aware batching for safe concurrency.

    workers : int, default 4
        Number of parallel workers (only used if parallel=True).

    Returns
    -------
    dict
        {
          "status": "completed" | "failed",
          "stages_run": [int],
          "results": {...},
          "report_path": str (if stage 3 ran),
          "execution_mode": "parallel" | "sequential",
          "duration_seconds": float
        }
    """
    import time
    start_time = time.time()

    try:
        from ragix_kernels.orchestrator import AuditOrchestrator, ManifestConfig
        from ragix_kernels.registry import KernelRegistry

        KernelRegistry.discover()

        ws = Path(workspace)
        manifest = ManifestConfig.from_yaml(ws / "manifest.yaml")

        orchestrator = AuditOrchestrator(
            workspace=ws,
            project_name=manifest.project_name,
            project_path=manifest.project_path,
            language=manifest.language,
        )
        orchestrator.manifest = manifest

        results = {}
        stages_run = []

        # Determine which stages to run
        if stage == 0:
            stages_to_run = [1, 2, 3]
        else:
            stages_to_run = [stage]

        # Parse kernel filter
        kernel_filter = [k.strip() for k in kernels.split(",") if k.strip()] if kernels else None

        for s in stages_to_run:
            if parallel:
                # Use parallel execution with dependency-aware batching
                stage_results = orchestrator._execute_stage_parallel(s, max_workers=workers)
            else:
                stage_results = orchestrator._execute_stage(s)
            results[f"stage{s}"] = {
                "succeeded": sum(1 for r in stage_results if r.success),
                "failed": sum(1 for r in stage_results if not r.success),
                "kernels": [r.kernel_name for r in stage_results],
            }
            stages_run.append(s)

        # Check for report
        report_path = None
        if 3 in stages_run:
            report_file = ws / "stage3" / "audit_report.md"
            if report_file.exists():
                report_path = str(report_file)

        duration = time.time() - start_time

        return {
            "status": "completed",
            "workspace": str(ws),
            "stages_run": stages_run,
            "results": results,
            "report_path": report_path,
            "execution_mode": "parallel" if parallel else "sequential",
            "workers": workers if parallel else 1,
            "duration_seconds": round(duration, 2),
        }

    except Exception as e:
        duration = time.time() - start_time
        return {"status": "failed", "error": str(e), "duration_seconds": round(duration, 2)}


@mcp.tool()
def koas_status(workspace: str) -> Dict[str, Any]:
    """
    Get status of a KOAS audit workspace.

    Parameters
    ----------
    workspace : str
        Path to the KOAS workspace.

    Returns
    -------
    dict
        {
          "workspace": str,
          "manifest": {...},
          "stages": {
            "stage1": {"completed": bool, "kernels": [str]},
            "stage2": {...},
            "stage3": {...}
          },
          "report_available": bool
        }
    """
    try:
        ws = Path(workspace)

        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}"}

        manifest_path = ws / "manifest.yaml"
        if not manifest_path.exists():
            return {"error": "manifest.yaml not found in workspace"}

        import yaml
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        # Check stage directories
        stages = {}
        for i in [1, 2, 3]:
            stage_dir = ws / f"stage{i}"
            if stage_dir.exists():
                json_files = list(stage_dir.glob("*.json"))
                stages[f"stage{i}"] = {
                    "completed": len(json_files) > 0,
                    "kernels": [f.stem for f in json_files if not f.stem.startswith("_")],
                }
            else:
                stages[f"stage{i}"] = {"completed": False, "kernels": []}

        # Check for report
        report_path = ws / "stage3" / "audit_report.md"

        return {
            "workspace": str(ws),
            "manifest": {
                "audit_name": manifest.get("audit", {}).get("name", "Unknown"),
                "project_name": manifest.get("project", {}).get("name", "Unknown"),
                "language": manifest.get("output", {}).get("language", "en"),
            },
            "stages": stages,
            "report_available": report_path.exists(),
            "report_path": str(report_path) if report_path.exists() else None,
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def koas_summary(workspace: str, stage: int = 0) -> Dict[str, Any]:
    """
    Get summaries from KOAS kernel outputs.

    Parameters
    ----------
    workspace : str
        Path to the KOAS workspace.

    stage : int, default 0
        Stage to get summaries from (0 = all stages).

    Returns
    -------
    dict
        {
          "summaries": {
            "kernel_name": "summary text",
            ...
          },
          "stage_summary": str
        }
    """
    try:
        ws = Path(workspace)

        stages_to_check = [1, 2, 3] if stage == 0 else [stage]
        summaries = {}

        for s in stages_to_check:
            stage_dir = ws / f"stage{s}"
            if stage_dir.exists():
                for summary_file in stage_dir.glob("*.summary.txt"):
                    kernel_name = summary_file.stem.replace(".summary", "")
                    summaries[kernel_name] = summary_file.read_text().strip()

        # Get stage summary if available
        stage_summary = None
        for s in stages_to_check:
            summary_path = ws / f"stage{s}" / "_stage_summary.txt"
            if summary_path.exists():
                stage_summary = summary_path.read_text()
                break

        return {
            "summaries": summaries,
            "kernel_count": len(summaries),
            "stage_summary": stage_summary,
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def koas_list_kernels(stage: int = 0, category: str = "") -> Dict[str, Any]:
    """
    List available KOAS kernels.

    Parameters
    ----------
    stage : int, default 0
        Filter by stage (0 = all stages).

    category : str
        Filter by category (empty = all categories).

    Returns
    -------
    dict
        {
          "kernels": [
            {
              "name": str,
              "stage": int,
              "description": str,
              "requires": [str],
              "provides": [str]
            }
          ],
          "total": int
        }
    """
    try:
        from ragix_kernels.registry import KernelRegistry

        KernelRegistry.discover()

        if stage > 0:
            kernel_names = KernelRegistry.list_stage(stage)
        elif category:
            kernel_names = KernelRegistry.list_category(category)
        else:
            kernel_names = KernelRegistry.list_all()

        kernels = []
        for name in kernel_names:
            info = KernelRegistry.get_info(name)
            kernels.append({
                "name": info["name"],
                "stage": info["stage"],
                "category": info["category"],
                "description": info["description"],
                "requires": info["requires"],
                "provides": info["provides"],
            })

        return {
            "kernels": kernels,
            "total": len(kernels),
            "stages": {
                1: len([k for k in kernels if k["stage"] == 1]),
                2: len([k for k in kernels if k["stage"] == 2]),
                3: len([k for k in kernels if k["stage"] == 3]),
            },
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def koas_report(workspace: str, max_chars: int = 10000) -> Dict[str, Any]:
    """
    Get the generated audit report content.

    Parameters
    ----------
    workspace : str
        Path to the KOAS workspace.

    max_chars : int, default 10000
        Maximum characters to return (truncates if larger).

    Returns
    -------
    dict
        {
          "report_path": str,
          "content": str,
          "truncated": bool,
          "total_chars": int
        }
    """
    try:
        ws = Path(workspace)
        report_path = ws / "stage3" / "audit_report.md"

        if not report_path.exists():
            return {"error": "Report not found. Run stage 3 first."}

        content = report_path.read_text()
        total_chars = len(content)
        truncated = total_chars > max_chars

        if truncated:
            content = content[:max_chars] + "\n\n[... truncated ...]"

        return {
            "report_path": str(report_path),
            "content": content,
            "truncated": truncated,
            "total_chars": total_chars,
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 8. AST and Code Analysis Tools (v0.8.0)
# ---------------------------------------------------------------------------

@mcp.tool()
def ragix_ast_scan(
    path: str,
    language: str = "auto",
    include_private: bool = False,
) -> Dict[str, Any]:
    """
    Extract AST symbols from source code files or directories.

    Parameters
    ----------
    path : str
        Path to file or directory to scan (relative to sandbox).

    language : str, default "auto"
        Programming language: "python", "java", "typescript", or "auto".
        "auto" detects based on file extension.

    include_private : bool, default False
        Include private/internal symbols (starting with _).

    Returns
    -------
    dict
        {
          "symbols": [
            {
              "name": str,
              "type": "class" | "method" | "function" | "field",
              "file": str,
              "line": int,
              "visibility": "public" | "private" | "protected"
            }
          ],
          "summary": {
            "classes": int,
            "methods": int,
            "functions": int,
            "files_scanned": int
          }
        }
    """
    try:
        target = Path(SANDBOX_ROOT) / path if not Path(path).is_absolute() else Path(path)

        if not target.exists():
            return {"error": f"Path not found: {path}"}

        # Try to use ragix-ast if available
        try:
            import subprocess
            result = subprocess.run(
                ["ragix-ast", "scan", str(target), "--language", language, "--format", "json"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                symbols = data.get("symbols", [])

                if not include_private:
                    symbols = [s for s in symbols if not s.get("name", "").startswith("_")]

                return {
                    "symbols": symbols[:500],  # Limit for MCP response size
                    "summary": {
                        "classes": sum(1 for s in symbols if s.get("type") == "class"),
                        "methods": sum(1 for s in symbols if s.get("type") == "method"),
                        "functions": sum(1 for s in symbols if s.get("type") == "function"),
                        "fields": sum(1 for s in symbols if s.get("type") == "field"),
                        "total_symbols": len(symbols),
                        "files_scanned": len(set(s.get("file", "") for s in symbols)),
                    },
                    "truncated": len(symbols) > 500,
                }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback: basic Python AST scanning
        if language in ("auto", "python") and target.suffix == ".py" or target.is_dir():
            import ast

            symbols = []
            files_scanned = 0

            def scan_file(file_path: Path):
                nonlocal files_scanned
                try:
                    with open(file_path) as f:
                        tree = ast.parse(f.read())
                    files_scanned += 1

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            symbols.append({
                                "name": node.name,
                                "type": "class",
                                "file": str(file_path.relative_to(target) if target.is_dir() else file_path.name),
                                "line": node.lineno,
                                "visibility": "private" if node.name.startswith("_") else "public",
                            })
                        elif isinstance(node, ast.FunctionDef):
                            symbols.append({
                                "name": node.name,
                                "type": "function",
                                "file": str(file_path.relative_to(target) if target.is_dir() else file_path.name),
                                "line": node.lineno,
                                "visibility": "private" if node.name.startswith("_") else "public",
                            })
                except Exception:
                    pass

            if target.is_file():
                scan_file(target)
            else:
                for py_file in target.rglob("*.py"):
                    scan_file(py_file)

            if not include_private:
                symbols = [s for s in symbols if s.get("visibility") != "private"]

            return {
                "symbols": symbols[:500],
                "summary": {
                    "classes": sum(1 for s in symbols if s.get("type") == "class"),
                    "functions": sum(1 for s in symbols if s.get("type") == "function"),
                    "total_symbols": len(symbols),
                    "files_scanned": files_scanned,
                },
                "truncated": len(symbols) > 500,
                "method": "python_ast_fallback",
            }

        return {"error": f"Unsupported language or file type: {language}"}

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def ragix_ast_metrics(
    path: str,
    language: str = "auto",
) -> Dict[str, Any]:
    """
    Compute code quality metrics for source files.

    Parameters
    ----------
    path : str
        Path to file or directory (relative to sandbox).

    language : str, default "auto"
        Programming language for analysis.

    Returns
    -------
    dict
        {
          "metrics": {
            "total_files": int,
            "total_loc": int,
            "total_classes": int,
            "total_methods": int,
            "avg_complexity": float,
            "maintainability_index": float
          },
          "hotspots": [
            {"file": str, "complexity": int, "loc": int}
          ]
        }
    """
    try:
        target = Path(SANDBOX_ROOT) / path if not Path(path).is_absolute() else Path(path)

        if not target.exists():
            return {"error": f"Path not found: {path}"}

        # Try ragix-ast metrics
        try:
            import subprocess
            result = subprocess.run(
                ["ragix-ast", "metrics", str(target), "--language", language, "--format", "json"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                import json
                return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback: basic metrics from file counting
        total_files = 0
        total_loc = 0
        file_metrics = []

        extensions = {".py", ".java", ".ts", ".js", ".tsx", ".jsx"} if language == "auto" else {f".{language}"}

        for ext in extensions:
            pattern = f"*{ext}"
            for file_path in (target.rglob(pattern) if target.is_dir() else [target]):
                try:
                    content = file_path.read_text()
                    lines = content.splitlines()
                    loc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
                    total_loc += loc
                    total_files += 1
                    file_metrics.append({
                        "file": str(file_path.relative_to(target) if target.is_dir() else file_path.name),
                        "loc": loc,
                    })
                except Exception:
                    pass

        # Sort by LOC for hotspots
        hotspots = sorted(file_metrics, key=lambda x: x["loc"], reverse=True)[:10]

        return {
            "metrics": {
                "total_files": total_files,
                "total_loc": total_loc,
                "avg_loc_per_file": round(total_loc / total_files, 1) if total_files > 0 else 0,
            },
            "hotspots": hotspots,
            "method": "basic_fallback",
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 9. Model Management Tools (v0.8.0)
# ---------------------------------------------------------------------------

@mcp.tool()
def ragix_models_list() -> Dict[str, Any]:
    """
    List available Ollama models.

    Returns
    -------
    dict
        {
          "models": [
            {
              "name": str,
              "size": str,
              "modified": str,
              "family": str
            }
          ],
          "total": int,
          "recommended": str
        }
    """
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return {"error": "Ollama not available or not running"}

        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return {"models": [], "total": 0}

        models = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                size = parts[2] if len(parts) > 2 else "unknown"
                models.append({
                    "name": name,
                    "size": size,
                    "family": name.split(":")[0] if ":" in name else name,
                })

        # Determine recommended model
        recommended = None
        for pref in ["mistral", "qwen", "llama", "granite"]:
            for m in models:
                if pref in m["name"].lower():
                    recommended = m["name"]
                    break
            if recommended:
                break

        return {
            "models": models,
            "total": len(models),
            "recommended": recommended or (models[0]["name"] if models else None),
            "current": OLLAMA_MODEL,
        }

    except FileNotFoundError:
        return {"error": "Ollama CLI not found. Is Ollama installed?"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def ragix_model_info(model: str) -> Dict[str, Any]:
    """
    Get detailed information about an Ollama model.

    Parameters
    ----------
    model : str
        Model name (e.g., "mistral", "qwen2.5:14b").

    Returns
    -------
    dict
        {
          "name": str,
          "parameters": str,
          "quantization": str,
          "context_length": int,
          "capabilities": [str]
        }
    """
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "show", model],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return {"error": f"Model not found: {model}"}

        # Parse output
        info = {"name": model}
        output = result.stdout

        # Extract key info from show output
        for line in output.split("\n"):
            if "parameters" in line.lower():
                info["parameters"] = line.split(":")[-1].strip() if ":" in line else "unknown"
            elif "quantization" in line.lower():
                info["quantization"] = line.split(":")[-1].strip() if ":" in line else "unknown"
            elif "context" in line.lower():
                try:
                    info["context_length"] = int("".join(filter(str.isdigit, line)))
                except ValueError:
                    pass

        # Infer capabilities from model name
        capabilities = ["text_generation"]
        name_lower = model.lower()
        if "instruct" in name_lower or "chat" in name_lower:
            capabilities.append("instruction_following")
        if "code" in name_lower:
            capabilities.append("code_generation")
        if "vision" in name_lower or "vl" in name_lower:
            capabilities.append("vision")

        info["capabilities"] = capabilities

        return info

    except FileNotFoundError:
        return {"error": "Ollama CLI not found"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def ragix_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information for RAGIX deployment.

    Returns
    -------
    dict
        {
          "cpu": {"cores": int, "model": str},
          "memory": {"total_gb": float, "available_gb": float},
          "gpu": {"available": bool, "devices": [...]},
          "disk": {"total_gb": float, "free_gb": float},
          "ragix": {"version": str, "sandbox": str}
        }
    """
    import platform
    import shutil

    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "ragix": {
            "sandbox_root": str(SANDBOX_ROOT),
            "profile": AGENT_PROFILE,
            "model": OLLAMA_MODEL,
        },
    }

    # CPU info
    try:
        import os
        info["cpu"] = {
            "cores": os.cpu_count(),
            "architecture": platform.machine(),
        }
    except Exception:
        info["cpu"] = {"cores": "unknown"}

    # Memory info
    try:
        import subprocess
        result = subprocess.run(["free", "-b"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                mem_parts = lines[1].split()
                if len(mem_parts) >= 3:
                    total = int(mem_parts[1]) / (1024**3)
                    available = int(mem_parts[6]) / (1024**3) if len(mem_parts) > 6 else int(mem_parts[3]) / (1024**3)
                    info["memory"] = {
                        "total_gb": round(total, 1),
                        "available_gb": round(available, 1),
                    }
    except Exception:
        info["memory"] = {"status": "unavailable"}

    # GPU info
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "name": parts[0],
                        "memory_total_mb": int(parts[1]),
                        "memory_free_mb": int(parts[2]),
                    })
            info["gpu"] = {
                "available": True,
                "cuda": True,
                "devices": gpus,
                "count": len(gpus),
            }
        else:
            info["gpu"] = {"available": False}
    except FileNotFoundError:
        info["gpu"] = {"available": False, "cuda": False}
    except Exception:
        info["gpu"] = {"status": "check_failed"}

    # Disk info
    try:
        disk = shutil.disk_usage(SANDBOX_ROOT)
        info["disk"] = {
            "total_gb": round(disk.total / (1024**3), 1),
            "free_gb": round(disk.free / (1024**3), 1),
            "used_percent": round((disk.used / disk.total) * 100, 1),
        }
    except Exception:
        info["disk"] = {"status": "unavailable"}

    # Ollama status
    try:
        import subprocess
        result = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, timeout=5)
        info["ollama"] = {"running": result.returncode == 0}
    except Exception:
        info["ollama"] = {"running": "unknown"}

    return info


# ---------------------------------------------------------------------------
# 10. Entry point
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
