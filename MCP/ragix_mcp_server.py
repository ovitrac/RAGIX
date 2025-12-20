#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGIX MCP Server v0.62.0
========================

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

v0.62.0 KOAS Security Tools (LLM-optimized):
16. koas_security_discover(target, method, timeout, workspace)
    - Network host discovery with nmap/arp-scan.

17. koas_security_scan_ports(target, ports, detect_services, workspace)
    - Port scanning with preset port groups.

18. koas_security_ssl_check(target, check_ciphers, check_vulnerabilities, workspace)
    - SSL/TLS certificate and cipher analysis.

19. koas_security_vuln_scan(target, severity, templates, workspace)
    - Vulnerability assessment with nuclei templates.

20. koas_security_dns_check(domain, check_security, workspace)
    - DNS enumeration and security record analysis.

21. koas_security_compliance(workspace, framework, level)
    - Compliance checking (ANSSI, NIST CSF, CIS Controls).

22. koas_security_risk(workspace, top_hosts)
    - Network security risk scoring.

23. koas_security_report(workspace, format, language)
    - Security assessment report generation.

v0.62.0 KOAS Audit Tools (LLM-optimized):
24. koas_audit_scan(project_path, language, include_tests, workspace)
    - AST scanning and symbol extraction.

25. koas_audit_metrics(workspace, threshold_cc, threshold_loc)
    - Code metrics (complexity, LOC, maintainability).

26. koas_audit_hotspots(workspace, top_n)
    - Complexity and risk hotspot identification.

27. koas_audit_dependencies(workspace, detect_cycles)
    - Dependency analysis and cycle detection.

28. koas_audit_dead_code(workspace)
    - Dead/unused code detection.

29. koas_audit_risk(workspace, include_volumetry)
    - Code risk scoring and assessment.

30. koas_audit_compliance(workspace, standard)
    - Code quality compliance checking.

31. koas_audit_report(workspace, format, language)
    - Audit report generation.

KOAS Base Tools:
- koas_init: Initialize audit workspace
- koas_run: Execute stages with parallel support (parallel=True, workers=4)
- koas_status: Get workspace status
- koas_summary: Get kernel summaries
- koas_list_kernels: List available kernels
- koas_report: Get generated report

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
# 10. KOAS Security Tools (v0.62.0) - Simplified for LLM consumption
# ---------------------------------------------------------------------------

# Import KOAS helpers
try:
    from koas_helpers import (
        get_or_create_workspace,
        simplify_output,
        generate_summary,
        extract_action_items,
        save_results,
        load_stage_results,
        load_stage_dependency_paths,
        resolve_target,
        resolve_ports,
        wrap_kernel_error,
        validate_target,
        generate_security_manifest,
        PORT_PRESETS,
        COMPLIANCE_PRESETS,
    )
    KOAS_HELPERS_AVAILABLE = True
except ImportError:
    KOAS_HELPERS_AVAILABLE = False


@mcp.tool()
def koas_security_discover(
    target: str,
    method: str = "ping",
    timeout: int = 120,
    workspace: str = "",
) -> Dict[str, Any]:
    """
    Discover hosts on a network segment.

    Parameters
    ----------
    target : str
        Target network or host (e.g., "192.168.1.0/24", "10.0.0.1").

    method : str, default "ping"
        Discovery method: "ping", "arp", or "list".

    timeout : int, default 120
        Scan timeout in seconds.

    workspace : str, optional
        Path to workspace. Auto-created if empty.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "hosts": [{ip, mac, hostname}] (max 10),
          "hosts_total": int,
          "action_items": [str],
          "workspace": str,
          "details_file": str
        }

    Example
    -------
    >>> koas_security_discover("192.168.1.0/24")
    {"summary": "Found 12 hosts. Most common: router, server, workstation.",
     "hosts": [...], "hosts_total": 12, ...}
    """
    try:
        # Validate inputs
        is_valid, error_msg = validate_target(target)
        if not is_valid:
            return {"error": error_msg, "summary": f"Invalid target: {error_msg}"}

        # Get or create workspace
        ws = get_or_create_workspace(workspace if workspace else None, "security")

        # Import and run kernel
        from ragix_kernels.security.net_discover import NetDiscoverKernel
        from ragix_kernels.base import KernelInput

        kernel = NetDiscoverKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={
                "targets": [target],
                "methods": [method],
                "timeout": timeout,
            },
            dependencies={},
        )

        result = kernel.compute(kernel_input)

        # Save full results
        details_file = save_results(ws, 1, "net_discover", result)

        # Simplify for LLM
        simplified = simplify_output(result, max_items=10)
        simplified["workspace"] = str(ws)
        simplified["details_file"] = str(details_file)
        simplified["summary"] = generate_summary(result, f"Network discovery on {target}.")

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "net_discover")


@mcp.tool()
def koas_security_scan_ports(
    target: str = "discovered",
    ports: str = "common",
    detect_services: bool = True,
    workspace: str = "",
) -> Dict[str, Any]:
    """
    Scan ports on target hosts.

    Parameters
    ----------
    target : str, default "discovered"
        Target specification. Use "discovered" to scan hosts from net_discover.

    ports : str, default "common"
        Port specification: "common", "web", "database", "admin", "top100", "full",
        or custom (e.g., "22,80,443" or "1-1000").

    detect_services : bool, default True
        Attempt service/version detection.

    workspace : str
        Path to existing workspace (required if target="discovered").

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "ports": [{port, protocol, state, service}] (max 10),
          "ports_total": int,
          "services_found": [str],
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_security_scan_ports("192.168.1.1", ports="web")
    {"summary": "Found 3 open ports: 80/http, 443/https, 8080/http-proxy.",
     "ports": [...], ...}
    """
    try:
        # Get workspace
        ws = get_or_create_workspace(workspace if workspace else None, "security")

        # Resolve target
        targets = resolve_target(target, ws)
        if not targets:
            return {"error": "No targets found", "summary": "No targets to scan."}

        # Resolve ports
        port_spec = resolve_ports(ports)

        # Import and run kernel
        from ragix_kernels.security.port_scan import PortScanKernel
        from ragix_kernels.base import KernelInput

        kernel = PortScanKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={
                "targets": targets,
                "ports": port_spec,
                "service_detection": detect_services,
            },
            dependencies=load_stage_dependency_paths(ws, 1),
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 1, "port_scan", result)

        # Simplify
        simplified = simplify_output(result, max_items=10)
        simplified["workspace"] = str(ws)
        simplified["details_file"] = str(details_file)

        # Extract service names
        if "ports" in result:
            services = list(set(p.get("service", "unknown") for p in result["ports"] if p.get("service")))
            simplified["services_found"] = services[:10]

        simplified["summary"] = generate_summary(result, f"Port scan on {len(targets)} host(s).")

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "port_scan")


@mcp.tool()
def koas_security_ssl_check(
    target: str = "discovered",
    check_ciphers: bool = True,
    check_vulnerabilities: bool = True,
    workspace: str = "",
) -> Dict[str, Any]:
    """
    Analyze SSL/TLS configuration and certificates.

    Parameters
    ----------
    target : str, default "discovered"
        Target specification. Use "discovered" for hosts with HTTPS ports.

    check_ciphers : bool, default True
        Check cipher suite security.

    check_vulnerabilities : bool, default True
        Check for known TLS vulnerabilities (POODLE, BEAST, etc.).

    workspace : str
        Path to existing workspace.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "certificates": [{subject, issuer, expires, valid}] (max 5),
          "weak_ciphers": [str],
          "vulnerabilities": [str],
          "tls_versions": {version: supported},
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_security_ssl_check("example.com")
    {"summary": "TLS 1.2+ supported. Certificate valid until 2025-06-15. 2 weak ciphers found.",
     "certificates": [...], ...}
    """
    try:
        ws = get_or_create_workspace(workspace if workspace else None, "security")
        targets = resolve_target(target, ws)

        if not targets:
            return {"error": "No targets found", "summary": "No targets for SSL analysis."}

        # Import and run kernel
        from ragix_kernels.security.ssl_analysis import SSLAnalysisKernel
        from ragix_kernels.base import KernelInput

        kernel = SSLAnalysisKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={
                "targets": targets,
                "check_ciphers": check_ciphers,
                "check_vulnerabilities": check_vulnerabilities,
            },
            dependencies=load_stage_dependency_paths(ws, 1),
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 2, "ssl_analysis", result)

        # Simplify
        simplified = simplify_output(result, max_items=5)
        simplified["workspace"] = str(ws)
        simplified["details_file"] = str(details_file)

        # Extract key findings
        if "weak_ciphers" in result:
            simplified["weak_ciphers"] = result["weak_ciphers"][:5]
        if "vulnerabilities" in result:
            simplified["vulnerabilities"] = [v.get("name", str(v)) if isinstance(v, dict) else v
                                             for v in result.get("vulnerabilities", [])[:5]]

        simplified["summary"] = generate_summary(result, "SSL/TLS analysis completed.")
        simplified["action_items"] = extract_action_items(result)

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "ssl_analysis")


@mcp.tool()
def koas_security_vuln_scan(
    target: str = "discovered",
    severity: str = "medium",
    templates: str = "default",
    workspace: str = "",
) -> Dict[str, Any]:
    """
    Scan for known vulnerabilities.

    Parameters
    ----------
    target : str, default "discovered"
        Target specification.

    severity : str, default "medium"
        Minimum severity to report: "info", "low", "medium", "high", "critical".

    templates : str, default "default"
        Template set: "default", "cves", "misconfigs", "exposures", or "all".

    workspace : str
        Path to existing workspace.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "vulnerabilities": [{id, severity, title, host}] (max 10),
          "vulnerabilities_total": int,
          "critical_count": int,
          "high_count": int,
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_security_vuln_scan("192.168.1.0/24", severity="high")
    {"summary": "Found 5 vulnerabilities: 1 critical, 4 high. CVE-2024-1234 on 192.168.1.10.",
     "vulnerabilities": [...], ...}
    """
    try:
        ws = get_or_create_workspace(workspace if workspace else None, "security")
        targets = resolve_target(target, ws)

        if not targets:
            return {"error": "No targets found", "summary": "No targets for vulnerability scan."}

        # Map severity to threshold
        severity_order = ["info", "low", "medium", "high", "critical"]
        min_severity = severity.lower()

        # Import and run kernel
        from ragix_kernels.security.vuln_assess import VulnAssessKernel
        from ragix_kernels.base import KernelInput

        kernel = VulnAssessKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={
                "targets": targets,
                "min_severity": min_severity,
                "templates": templates,
            },
            dependencies={**load_stage_dependency_paths(ws, 1), **load_stage_dependency_paths(ws, 2)},
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 2, "vuln_assess", result)

        # Simplify
        simplified = simplify_output(result, max_items=10)
        simplified["workspace"] = str(ws)
        simplified["details_file"] = str(details_file)

        # Count by severity
        vulns = result.get("vulnerabilities", [])
        simplified["critical_count"] = sum(1 for v in vulns if v.get("severity", "").lower() == "critical")
        simplified["high_count"] = sum(1 for v in vulns if v.get("severity", "").lower() == "high")
        simplified["vulnerabilities_total"] = len(vulns)

        simplified["summary"] = generate_summary(result, f"Vulnerability scan on {len(targets)} target(s).")
        simplified["action_items"] = extract_action_items(result)

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "vuln_assess")


@mcp.tool()
def koas_security_dns_check(
    domain: str,
    check_security: bool = True,
    workspace: str = "",
) -> Dict[str, Any]:
    """
    Analyze DNS configuration and security.

    Parameters
    ----------
    domain : str
        Domain name to analyze.

    check_security : bool, default True
        Check security records (SPF, DKIM, DMARC, DNSSEC).

    workspace : str
        Path to workspace.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "records": {A, AAAA, MX, NS, TXT, ...},
          "security": {spf, dkim, dmarc, dnssec},
          "subdomains": [str] (if enumerated),
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_security_dns_check("example.com")
    {"summary": "Domain has 2 MX, 4 NS records. SPF: OK, DMARC: missing, DNSSEC: enabled.",
     "records": {...}, "security": {...}, ...}
    """
    try:
        ws = get_or_create_workspace(workspace if workspace else None, "security")

        # Import and run kernel
        from ragix_kernels.security.dns_enum import DNSEnumKernel
        from ragix_kernels.base import KernelInput

        kernel = DNSEnumKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={
                "domains": [domain],
                "check_security": check_security,
            },
            dependencies={},
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 1, "dns_enum", result)

        # Build simplified response
        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
        }

        # Extract records
        if "records" in result:
            simplified["records"] = result["records"]
        if "security" in result:
            simplified["security"] = result["security"]
        if "subdomains" in result:
            simplified["subdomains"] = result["subdomains"][:20]

        # Generate summary
        summary_parts = [f"DNS analysis for {domain}."]
        if "security" in result:
            sec = result["security"]
            status = []
            if sec.get("spf"):
                status.append("SPF: OK")
            else:
                status.append("SPF: missing")
            if sec.get("dmarc"):
                status.append("DMARC: OK")
            else:
                status.append("DMARC: missing")
            if sec.get("dnssec"):
                status.append("DNSSEC: enabled")
            summary_parts.append(" ".join(status))

        simplified["summary"] = " ".join(summary_parts)[:300]
        simplified["action_items"] = extract_action_items(result)

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "dns_enum")


@mcp.tool()
def koas_security_compliance(
    workspace: str,
    framework: str = "anssi",
    level: str = "standard",
) -> Dict[str, Any]:
    """
    Check compliance against security frameworks.

    Parameters
    ----------
    workspace : str
        Path to workspace with prior scan results.

    framework : str, default "anssi"
        Compliance framework: "anssi", "nist", or "cis".

    level : str, default "standard"
        Compliance level: "essential", "standard", "reinforced" (ANSSI),
        or "IG1", "IG2", "IG3" (CIS).

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "compliance_score": float (0-100),
          "framework": str,
          "passed": int,
          "failed": int,
          "findings": [{rule_id, status, recommendation}] (max 10),
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_security_compliance(workspace="/tmp/koas_123", framework="anssi")
    {"summary": "ANSSI compliance: 75%. 30/40 controls passed. 3 critical gaps.",
     "compliance_score": 75.0, "findings": [...], ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Import and run kernel
        from ragix_kernels.security.compliance import ComplianceKernel
        from ragix_kernels.base import KernelInput

        kernel = ComplianceKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={
                "frameworks": [framework],
                "anssi_level": level if framework == "anssi" else "standard",
                "cis_implementation_group": level if framework == "cis" else "IG1",
            },
            dependencies={**load_stage_dependency_paths(ws, 1), **load_stage_dependency_paths(ws, 2)},
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 2, "compliance", result)

        # Simplify
        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
            "framework": framework.upper(),
        }

        # Extract scores
        if "compliance_scores" in result:
            scores = result["compliance_scores"]
            score_val = scores.get(framework, 0)
            # Handle nested dict or direct value
            if isinstance(score_val, dict):
                simplified["compliance_score"] = score_val.get("score", 0)
            else:
                simplified["compliance_score"] = float(score_val) if score_val else 0

        # Count passed/failed - handle both dict and list formats
        raw_findings = result.get("findings", {})
        if isinstance(raw_findings, dict):
            findings = raw_findings.get(framework, [])
        elif isinstance(raw_findings, list):
            findings = raw_findings
        else:
            findings = []

        simplified["passed"] = sum(1 for f in findings if isinstance(f, dict) and f.get("status") == "pass")
        simplified["failed"] = sum(1 for f in findings if isinstance(f, dict) and f.get("status") == "fail")
        simplified["findings"] = [f for f in findings if isinstance(f, dict) and f.get("status") == "fail"][:10]

        # Summary
        score = simplified.get("compliance_score", 0)
        simplified["summary"] = f"{framework.upper()} compliance: {score:.0f}%. {simplified['passed']}/{simplified['passed']+simplified['failed']} controls passed."

        simplified["action_items"] = extract_action_items(result)

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "compliance")


@mcp.tool()
def koas_security_risk(
    workspace: str,
    top_hosts: int = 5,
) -> Dict[str, Any]:
    """
    Calculate network security risk scores.

    Parameters
    ----------
    workspace : str
        Path to workspace with prior scan results.

    top_hosts : int, default 5
        Number of highest-risk hosts to include.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "risk_score": float (0-10),
          "risk_level": str (LOW/MEDIUM/HIGH/CRITICAL),
          "top_risks": [{host, score, factors}] (top N),
          "risk_breakdown": {category: score},
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_security_risk(workspace="/tmp/koas_123")
    {"summary": "Network risk: HIGH (7.2/10). Top risk: 192.168.1.50 (exposed SSH, outdated OpenSSL).",
     "risk_score": 7.2, "risk_level": "HIGH", ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Import and run kernel
        from ragix_kernels.security.risk_network import RiskNetworkKernel
        from ragix_kernels.base import KernelInput

        kernel = RiskNetworkKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={"top_hosts": top_hosts},
            dependencies={**load_stage_dependency_paths(ws, 1), **load_stage_dependency_paths(ws, 2)},
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 2, "risk_network", result)

        # Simplify
        risk_score = result.get("risk_score", result.get("overall_risk", 0))
        risk_level = "CRITICAL" if risk_score >= 8 else "HIGH" if risk_score >= 6 else "MEDIUM" if risk_score >= 4 else "LOW"

        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
        }

        # Top risks
        if "host_risks" in result:
            simplified["top_risks"] = result["host_risks"][:top_hosts]
        elif "top_risks" in result:
            simplified["top_risks"] = result["top_risks"][:top_hosts]

        # Risk breakdown
        if "risk_breakdown" in result:
            simplified["risk_breakdown"] = result["risk_breakdown"]

        # Summary
        simplified["summary"] = f"Network risk: {risk_level} ({risk_score:.1f}/10)."
        if simplified.get("top_risks"):
            top = simplified["top_risks"][0]
            top_host = top.get("host", top.get("ip", "unknown"))
            simplified["summary"] += f" Top risk: {top_host}."

        simplified["action_items"] = extract_action_items(result)

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "risk_network")


@mcp.tool()
def koas_security_report(
    workspace: str,
    format: str = "summary",
    language: str = "en",
) -> Dict[str, Any]:
    """
    Generate security assessment report.

    Parameters
    ----------
    workspace : str
        Path to workspace with scan results.

    format : str, default "summary"
        Report format: "summary" (concise), "detailed" (full), or "executive".

    language : str, default "en"
        Report language: "en" (English) or "fr" (French).

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "report_file": str (path to full report),
          "key_findings": [str] (top 5),
          "recommendations": [str] (top 5),
          "scores": {compliance, risk, ...}
        }

    Example
    -------
    >>> koas_security_report(workspace="/tmp/koas_123", format="executive")
    {"summary": "Security assessment complete. 3 critical issues, 5 recommendations.",
     "report_file": "/tmp/koas_123/stage3/security_report.md", ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Import and run kernel
        from ragix_kernels.security.section_security import SectionSecurityKernel
        from ragix_kernels.base import KernelInput

        kernel = SectionSecurityKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={
                "format": format,
                "language": language,
                "include_recommendations": True,
            },
            dependencies={
                **load_stage_dependency_paths(ws, 1),
                **load_stage_dependency_paths(ws, 2),
            },
        )

        result = kernel.compute(kernel_input)

        # Save report
        report_file = ws / "stage3" / "security_report.md"
        (ws / "stage3").mkdir(parents=True, exist_ok=True)

        if "report" in result:
            with open(report_file, "w") as f:
                f.write(result["report"])
        elif "content" in result:
            with open(report_file, "w") as f:
                f.write(result["content"])

        # Save JSON results too
        details_file = save_results(ws, 3, "section_security", result)

        # Build response
        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "report_file": str(report_file) if report_file.exists() else None,
            "details_file": str(details_file),
        }

        # Extract key findings and recommendations
        if "key_findings" in result:
            simplified["key_findings"] = result["key_findings"][:5]
        elif "findings" in result:
            simplified["key_findings"] = [f.get("title", str(f)) if isinstance(f, dict) else str(f)
                                          for f in result["findings"][:5]]

        if "recommendations" in result:
            simplified["recommendations"] = [r.get("recommendation", str(r)) if isinstance(r, dict) else str(r)
                                             for r in result["recommendations"][:5]]

        # Scores
        scores = {}
        for key in ["compliance_score", "risk_score", "security_score"]:
            if key in result:
                scores[key.replace("_score", "")] = result[key]
        if scores:
            simplified["scores"] = scores

        # Summary
        findings_count = len(result.get("findings", result.get("key_findings", [])))
        rec_count = len(result.get("recommendations", []))
        simplified["summary"] = f"Security assessment complete. {findings_count} findings, {rec_count} recommendations. Report: {report_file.name}"

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "section_security")


# ---------------------------------------------------------------------------
# 11. KOAS Audit Tools (v0.62.0) - Simplified for LLM consumption
# ---------------------------------------------------------------------------

@mcp.tool()
def koas_audit_scan(
    project_path: str,
    language: str = "auto",
    include_tests: bool = False,
    workspace: str = "",
) -> Dict[str, Any]:
    """
    Scan a codebase and extract AST symbols.

    Parameters
    ----------
    project_path : str
        Path to project directory or source file.

    language : str, default "auto"
        Programming language: "auto", "python", "java", "typescript".

    include_tests : bool, default False
        Include test files in the scan.

    workspace : str, optional
        Path to workspace. Auto-created if empty.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "symbols_total": int,
          "classes": int,
          "methods": int,
          "functions": int,
          "files_scanned": int,
          "top_files": [{file, symbols}] (top 10),
          "workspace": str,
          "details_file": str
        }

    Example
    -------
    >>> koas_audit_scan("/path/to/project", language="python")
    {"summary": "Scanned 45 files. Found 120 classes, 580 methods, 95 functions.",
     "symbols_total": 795, ...}
    """
    try:
        ws = get_or_create_workspace(workspace if workspace else None, "audit")

        # Resolve project path
        project = Path(project_path)
        if not project.is_absolute():
            project = Path(SANDBOX_ROOT) / project_path

        if not project.exists():
            return {"error": f"Project not found: {project_path}", "summary": "Project path not found."}

        # Use existing ragix_ast_scan or kernel
        try:
            from ragix_kernels.audit.ast_scan import ASTScanKernel
            from ragix_kernels.base import KernelInput

            kernel = ASTScanKernel()
            kernel_input = KernelInput(
                workspace=ws,
                config={
                    "source_path": str(project),
                    "language": language,
                    "include_tests": include_tests,
                },
                dependencies={},
            )

            result = kernel.compute(kernel_input)
        except ImportError:
            # Fallback to ragix-ast CLI
            import subprocess
            cmd = ["ragix-ast", "scan", str(project), "--language", language, "--format", "json"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                result = json.loads(proc.stdout)
            else:
                return {"error": f"Scan failed: {proc.stderr}", "summary": "AST scan failed."}

        # Save results
        details_file = save_results(ws, 1, "ast_scan", result)

        # Simplify
        symbols = result.get("symbols", [])
        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
            "symbols_total": len(symbols),
            "classes": sum(1 for s in symbols if s.get("type") == "class"),
            "methods": sum(1 for s in symbols if s.get("type") == "method"),
            "functions": sum(1 for s in symbols if s.get("type") == "function"),
            "files_scanned": len(set(s.get("file", "") for s in symbols)),
        }

        # Top files by symbol count
        from collections import Counter
        file_counts = Counter(s.get("file", "") for s in symbols)
        simplified["top_files"] = [{"file": f, "symbols": c} for f, c in file_counts.most_common(10)]

        # Summary
        simplified["summary"] = (
            f"Scanned {simplified['files_scanned']} files. "
            f"Found {simplified['classes']} classes, {simplified['methods']} methods, "
            f"{simplified['functions']} functions."
        )

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "ast_scan")


@mcp.tool()
def koas_audit_metrics(
    workspace: str,
    threshold_cc: int = 10,
    threshold_loc: int = 300,
) -> Dict[str, Any]:
    """
    Compute code metrics (complexity, LOC, maintainability).

    Parameters
    ----------
    workspace : str
        Path to workspace with prior scan results.

    threshold_cc : int, default 10
        Cyclomatic complexity threshold for flagging.

    threshold_loc : int, default 300
        Lines of code threshold for flagging.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "total_loc": int,
          "avg_complexity": float,
          "maintainability_index": float,
          "high_complexity": [{name, cc, file}] (top 10),
          "large_files": [{file, loc}] (top 10),
          "details_file": str
        }

    Example
    -------
    >>> koas_audit_metrics(workspace="/tmp/koas_123")
    {"summary": "Total: 15,230 LOC, avg CC: 4.2. 8 high-complexity methods flagged.",
     "total_loc": 15230, "avg_complexity": 4.2, ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Import and run kernel
        from ragix_kernels.audit.metrics import MetricsKernel
        from ragix_kernels.base import KernelInput

        kernel = MetricsKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={
                "threshold_cc": threshold_cc,
                "threshold_loc": threshold_loc,
            },
            dependencies=load_stage_dependency_paths(ws, 1),
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 1, "metrics", result)

        # Simplify
        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
        }

        # Extract key metrics
        if "total_loc" in result:
            simplified["total_loc"] = result["total_loc"]
        if "avg_complexity" in result or "average_cc" in result:
            simplified["avg_complexity"] = result.get("avg_complexity", result.get("average_cc", 0))
        if "maintainability_index" in result:
            simplified["maintainability_index"] = round(result["maintainability_index"], 1)

        # High complexity items
        if "high_complexity" in result:
            simplified["high_complexity"] = result["high_complexity"][:10]
        elif "complexity_violations" in result:
            simplified["high_complexity"] = result["complexity_violations"][:10]

        # Large files
        if "large_files" in result:
            simplified["large_files"] = result["large_files"][:10]

        # Summary
        loc = simplified.get("total_loc", 0)
        cc = simplified.get("avg_complexity", 0)
        high_cc = len(simplified.get("high_complexity", []))
        simplified["summary"] = f"Total: {loc:,} LOC, avg CC: {cc:.1f}. {high_cc} high-complexity items flagged."

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "metrics")


@mcp.tool()
def koas_audit_hotspots(
    workspace: str,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Identify complexity and risk hotspots.

    Parameters
    ----------
    workspace : str
        Path to workspace with prior analysis.

    top_n : int, default 20
        Number of hotspots to return.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "hotspots": [{name, file, score, factors}] (top N),
          "risk_distribution": {high, medium, low},
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_audit_hotspots(workspace="/tmp/koas_123", top_n=10)
    {"summary": "10 hotspots identified. Top: PaymentService.process (score: 8.5).",
     "hotspots": [...], ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Import and run kernel
        from ragix_kernels.audit.hotspots import HotspotsKernel
        from ragix_kernels.base import KernelInput

        kernel = HotspotsKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={"top_n": top_n},
            dependencies={**load_stage_dependency_paths(ws, 1), **load_stage_dependency_paths(ws, 2)},
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 2, "hotspots", result)

        # Simplify
        hotspots = result.get("hotspots", [])[:top_n]

        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
            "hotspots": hotspots,
        }

        # Risk distribution
        if "risk_distribution" in result:
            simplified["risk_distribution"] = result["risk_distribution"]
        else:
            simplified["risk_distribution"] = {
                "high": sum(1 for h in hotspots if h.get("score", 0) >= 7),
                "medium": sum(1 for h in hotspots if 4 <= h.get("score", 0) < 7),
                "low": sum(1 for h in hotspots if h.get("score", 0) < 4),
            }

        # Summary
        if hotspots:
            top = hotspots[0]
            top_name = top.get("name", top.get("file", "unknown"))
            top_score = top.get("score", 0)
            simplified["summary"] = f"{len(hotspots)} hotspots identified. Top: {top_name} (score: {top_score:.1f})."
        else:
            simplified["summary"] = "No significant hotspots identified."

        simplified["action_items"] = extract_action_items(result)

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "hotspots")


@mcp.tool()
def koas_audit_dependencies(
    workspace: str,
    detect_cycles: bool = True,
) -> Dict[str, Any]:
    """
    Analyze code dependencies and coupling.

    Parameters
    ----------
    workspace : str
        Path to workspace with prior scan results.

    detect_cycles : bool, default True
        Detect circular dependencies.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "modules": int,
          "dependencies": int,
          "cycles": [[str]] (circular dependency chains),
          "high_coupling": [{module, fan_in, fan_out}],
          "details_file": str
        }

    Example
    -------
    >>> koas_audit_dependencies(workspace="/tmp/koas_123")
    {"summary": "45 modules, 120 dependencies. 2 circular dependencies detected.",
     "modules": 45, "dependencies": 120, "cycles": [...], ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Import and run kernel
        from ragix_kernels.audit.dependency import DependencyKernel
        from ragix_kernels.base import KernelInput

        kernel = DependencyKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={"detect_cycles": detect_cycles},
            dependencies=load_stage_dependency_paths(ws, 1),
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 1, "dependency", result)

        # Simplify
        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
        }

        # Extract counts
        modules = result.get("modules", result.get("nodes", []))
        deps = result.get("dependencies", result.get("edges", []))
        simplified["modules"] = len(modules) if isinstance(modules, list) else modules
        simplified["dependencies"] = len(deps) if isinstance(deps, list) else deps

        # Cycles
        cycles = result.get("cycles", [])
        simplified["cycles"] = cycles[:10]
        simplified["cycles_count"] = len(cycles)

        # High coupling
        if "high_coupling" in result:
            simplified["high_coupling"] = result["high_coupling"][:10]

        # Summary
        cycle_msg = f"{len(cycles)} circular dependencies detected." if cycles else "No circular dependencies."
        simplified["summary"] = f"{simplified['modules']} modules, {simplified['dependencies']} dependencies. {cycle_msg}"

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "dependency")


@mcp.tool()
def koas_audit_dead_code(
    workspace: str,
) -> Dict[str, Any]:
    """
    Detect potentially dead or unused code.

    Parameters
    ----------
    workspace : str
        Path to workspace with prior analysis.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "dead_code": [{name, type, file, reason}] (max 20),
          "dead_code_total": int,
          "by_type": {functions, classes, methods},
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_audit_dead_code(workspace="/tmp/koas_123")
    {"summary": "Found 15 potentially dead code items: 8 functions, 5 methods, 2 classes.",
     "dead_code": [...], ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Import and run kernel
        from ragix_kernels.audit.dead_code import DeadCodeKernel
        from ragix_kernels.base import KernelInput

        kernel = DeadCodeKernel()
        kernel_input = KernelInput(
            workspace=ws,
            config={},
            dependencies={**load_stage_dependency_paths(ws, 1), **load_stage_dependency_paths(ws, 2)},
        )

        result = kernel.compute(kernel_input)

        # Save results
        details_file = save_results(ws, 2, "dead_code", result)

        # Simplify
        dead_items = result.get("dead_code", result.get("unused", []))

        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
            "dead_code": dead_items[:20],
            "dead_code_total": len(dead_items),
        }

        # Count by type
        from collections import Counter
        type_counts = Counter(d.get("type", "unknown") for d in dead_items)
        simplified["by_type"] = dict(type_counts)

        # Summary
        type_str = ", ".join(f"{c} {t}s" for t, c in type_counts.most_common(3))
        simplified["summary"] = f"Found {len(dead_items)} potentially dead code items: {type_str}."

        simplified["action_items"] = [
            {"priority": "low", "action": f"Review {item.get('name', 'item')} for removal"}
            for item in dead_items[:5]
        ]

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "dead_code")


def _calculate_simple_risk(workspace: Path) -> Dict[str, Any]:
    """Calculate simple risk score from available stage results."""
    risk_score = 0.0
    factors = []

    # Load available results
    stage1 = {}
    stage2 = {}

    stage1_dir = workspace / "stage1"
    stage2_dir = workspace / "stage2"

    if stage1_dir.exists():
        for f in stage1_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    stage1[f.stem] = json.load(fp)
            except Exception:
                pass

    if stage2_dir.exists():
        for f in stage2_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    stage2[f.stem] = json.load(fp)
            except Exception:
                pass

    # Calculate risk from available metrics
    if "metrics" in stage1:
        metrics = stage1["metrics"]
        avg_cc = metrics.get("avg_complexity", metrics.get("average_cc", 0))
        if avg_cc > 10:
            risk_score += 3.0
            factors.append({"factor": "complexity", "impact": "high", "value": avg_cc})
        elif avg_cc > 5:
            risk_score += 1.5
            factors.append({"factor": "complexity", "impact": "medium", "value": avg_cc})

    if "hotspots" in stage2:
        hotspots = stage2["hotspots"]
        high_risk = len([h for h in hotspots.get("hotspots", []) if h.get("score", 0) > 7])
        if high_risk > 5:
            risk_score += 2.5
            factors.append({"factor": "hotspots", "impact": "high", "value": high_risk})

    if "dead_code" in stage2:
        dead = stage2["dead_code"]
        dead_count = dead.get("dead_code_total", len(dead.get("dead_code", [])))
        if dead_count > 20:
            risk_score += 1.0
            factors.append({"factor": "dead_code", "impact": "medium", "value": dead_count})

    return {
        "risk_score": min(risk_score, 10.0),
        "factors": factors,
        "method": "simplified_calculation",
    }


@mcp.tool()
def koas_audit_risk(
    workspace: str,
    include_volumetry: bool = False,
) -> Dict[str, Any]:
    """
    Calculate code risk scores.

    Parameters
    ----------
    workspace : str
        Path to workspace with prior analysis.

    include_volumetry : bool, default False
        Include volumetry data in risk calculation.

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "risk_score": float (0-10),
          "risk_level": str (LOW/MEDIUM/HIGH/CRITICAL),
          "top_risks": [{module, score, factors}] (top 10),
          "risk_breakdown": {complexity, coupling, debt},
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_audit_risk(workspace="/tmp/koas_123")
    {"summary": "Code risk: MEDIUM (5.2/10). Top risk: PaymentModule (high complexity, tight coupling).",
     "risk_score": 5.2, "risk_level": "MEDIUM", ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Import and run kernel - try risk_matrix first (more comprehensive)
        try:
            from ragix_kernels.audit.risk_matrix import RiskMatrixKernel
            kernel_class = RiskMatrixKernel
            kernel_name = "risk_matrix"
        except ImportError:
            from ragix_kernels.audit.risk import RiskKernel
            kernel_class = RiskKernel
            kernel_name = "risk"

        from ragix_kernels.base import KernelInput

        kernel = kernel_class()
        kernel_input = KernelInput(
            workspace=ws,
            config={"include_volumetry": include_volumetry},
            dependencies={**load_stage_dependency_paths(ws, 1), **load_stage_dependency_paths(ws, 2)},
        )

        try:
            result = kernel.compute(kernel_input)
        except Exception as e:
            # If risk_matrix fails (e.g., missing module_group), try simpler risk kernel
            if kernel_name == "risk_matrix" and "module_group" in str(e):
                try:
                    from ragix_kernels.audit.risk import RiskKernel
                    kernel = RiskKernel()
                    result = kernel.compute(kernel_input)
                    kernel_name = "risk"
                except Exception:
                    # Fall back to manual risk calculation
                    result = _calculate_simple_risk(ws)
            else:
                raise

        # Save results
        details_file = save_results(ws, 2, kernel_name, result)

        # Simplify
        risk_score = result.get("risk_score", result.get("overall_risk", 0))
        risk_level = "CRITICAL" if risk_score >= 8 else "HIGH" if risk_score >= 6 else "MEDIUM" if risk_score >= 4 else "LOW"

        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
        }

        # Top risks
        if "module_risks" in result:
            simplified["top_risks"] = result["module_risks"][:10]
        elif "top_risks" in result:
            simplified["top_risks"] = result["top_risks"][:10]

        # Risk breakdown
        if "risk_breakdown" in result:
            simplified["risk_breakdown"] = result["risk_breakdown"]

        # Summary
        simplified["summary"] = f"Code risk: {risk_level} ({risk_score:.1f}/10)."
        if simplified.get("top_risks"):
            top = simplified["top_risks"][0]
            top_name = top.get("module", top.get("name", "unknown"))
            simplified["summary"] += f" Top risk: {top_name}."

        simplified["action_items"] = extract_action_items(result)

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "risk")


@mcp.tool()
def koas_audit_compliance(
    workspace: str,
    standard: str = "maintainability",
) -> Dict[str, Any]:
    """
    Check code quality compliance.

    Parameters
    ----------
    workspace : str
        Path to workspace with prior analysis.

    standard : str, default "maintainability"
        Quality standard: "maintainability", "testability", "documentation".

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "compliance_score": float (0-100),
          "standard": str,
          "violations": [{rule, severity, location}] (max 20),
          "metrics": {coverage, documentation, complexity},
          "action_items": [str],
          "details_file": str
        }

    Example
    -------
    >>> koas_audit_compliance(workspace="/tmp/koas_123", standard="maintainability")
    {"summary": "Maintainability compliance: 72%. 15 violations: 8 complexity, 7 documentation.",
     "compliance_score": 72.0, ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Load metrics from workspace
        stage1_results = load_stage_results(ws, 1)
        stage2_results = load_stage_results(ws, 2)

        # Calculate compliance based on standard
        metrics = stage1_results.get("metrics", {})
        hotspots = stage2_results.get("hotspots", {})

        violations = []
        compliance_score = 100.0

        if standard == "maintainability":
            # Check maintainability index
            mi = metrics.get("maintainability_index", 65)
            if mi < 20:
                compliance_score -= 30
                violations.append({"rule": "MI_CRITICAL", "severity": "critical", "location": "global"})
            elif mi < 40:
                compliance_score -= 15
                violations.append({"rule": "MI_LOW", "severity": "high", "location": "global"})

            # Check complexity
            high_cc = metrics.get("high_complexity", [])
            for item in high_cc[:10]:
                compliance_score -= 2
                violations.append({
                    "rule": "CC_HIGH",
                    "severity": "medium",
                    "location": item.get("file", "unknown"),
                })

        elif standard == "testability":
            # Check for test files
            test_coverage = metrics.get("test_coverage", 0)
            if test_coverage < 50:
                compliance_score -= 30
                violations.append({"rule": "COVERAGE_LOW", "severity": "high", "location": "global"})
            elif test_coverage < 80:
                compliance_score -= 10
                violations.append({"rule": "COVERAGE_MEDIUM", "severity": "medium", "location": "global"})

        elif standard == "documentation":
            doc_coverage = metrics.get("javadoc_coverage", metrics.get("docstring_coverage", 0))
            if doc_coverage < 30:
                compliance_score -= 25
                violations.append({"rule": "DOC_MISSING", "severity": "high", "location": "global"})
            elif doc_coverage < 60:
                compliance_score -= 10
                violations.append({"rule": "DOC_LOW", "severity": "medium", "location": "global"})

        compliance_score = max(0, min(100, compliance_score))

        # Save results
        result = {
            "compliance_score": compliance_score,
            "standard": standard,
            "violations": violations,
            "metrics": metrics,
        }
        details_file = save_results(ws, 2, "compliance_audit", result)

        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "details_file": str(details_file),
            "compliance_score": round(compliance_score, 1),
            "standard": standard,
            "violations": violations[:20],
        }

        # Summary
        simplified["summary"] = f"{standard.capitalize()} compliance: {compliance_score:.0f}%. {len(violations)} violations found."

        simplified["action_items"] = [
            {"priority": v["severity"], "action": f"Fix {v['rule']} in {v['location']}"}
            for v in violations[:5]
        ]

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "compliance_audit")


@mcp.tool()
def koas_audit_report(
    workspace: str,
    format: str = "executive",
    language: str = "en",
) -> Dict[str, Any]:
    """
    Generate code audit report.

    Parameters
    ----------
    workspace : str
        Path to workspace with analysis results.

    format : str, default "executive"
        Report format: "executive" (summary), "detailed" (full), or "technical".

    language : str, default "en"
        Report language: "en" (English) or "fr" (French).

    Returns
    -------
    dict
        {
          "summary": str (<300 chars),
          "report_file": str (path to full report),
          "key_findings": [str] (top 5),
          "recommendations": [str] (top 5),
          "scores": {risk, maintainability, ...}
        }

    Example
    -------
    >>> koas_audit_report(workspace="/tmp/koas_123", format="executive")
    {"summary": "Audit complete. Risk: MEDIUM. 5 recommendations for improvement.",
     "report_file": "/tmp/koas_123/stage3/audit_report.md", ...}
    """
    try:
        ws = Path(workspace)
        if not ws.exists():
            return {"error": f"Workspace not found: {workspace}", "summary": "Workspace not found."}

        # Try to use report_assemble kernel
        try:
            from ragix_kernels.audit.report_assemble import ReportAssembleKernel
            from ragix_kernels.base import KernelInput

            kernel = ReportAssembleKernel()
            kernel_input = KernelInput(
                workspace=ws,
                config={
                    "format": format,
                    "language": language,
                },
                dependencies={
                    **load_stage_dependency_paths(ws, 1),
                    **load_stage_dependency_paths(ws, 2),
                },
            )

            result = kernel.compute(kernel_input)
        except ImportError:
            # Manual report generation
            stage1 = load_stage_results(ws, 1)
            stage2 = load_stage_results(ws, 2)

            result = {
                "format": format,
                "language": language,
                "findings": [],
                "recommendations": [],
            }

            # Extract findings from various kernels
            if "hotspots" in stage2:
                for h in stage2["hotspots"].get("hotspots", [])[:5]:
                    result["findings"].append({
                        "title": f"High complexity: {h.get('name', 'unknown')}",
                        "severity": "medium" if h.get("score", 0) < 7 else "high",
                    })

            # Add recommendations
            if "risk" in stage2 or "risk_matrix" in stage2:
                risk_data = stage2.get("risk_matrix", stage2.get("risk", {}))
                if risk_data.get("risk_score", 0) > 5:
                    result["recommendations"].append({
                        "recommendation": "Reduce complexity in high-risk modules",
                        "priority": "high",
                    })

        # Save report
        report_file = ws / "stage3" / "audit_report.md"
        (ws / "stage3").mkdir(parents=True, exist_ok=True)

        # Generate markdown report
        if "report" in result:
            with open(report_file, "w") as f:
                f.write(result["report"])
        elif "content" in result:
            with open(report_file, "w") as f:
                f.write(result["content"])
        else:
            # Generate basic report
            report_content = f"# Code Audit Report\n\n"
            report_content += f"**Format:** {format}\n"
            report_content += f"**Language:** {language}\n\n"
            report_content += "## Findings\n\n"
            for f in result.get("findings", []):
                report_content += f"- **{f.get('severity', 'info').upper()}**: {f.get('title', 'Finding')}\n"
            report_content += "\n## Recommendations\n\n"
            for r in result.get("recommendations", []):
                report_content += f"- {r.get('recommendation', 'Recommendation')}\n"
            with open(report_file, "w") as f:
                f.write(report_content)

        # Save JSON results
        details_file = save_results(ws, 3, "report_assemble", result)

        # Build response
        simplified = {
            "status": "completed",
            "workspace": str(ws),
            "report_file": str(report_file) if report_file.exists() else None,
            "details_file": str(details_file),
        }

        # Extract findings and recommendations
        if "findings" in result:
            simplified["key_findings"] = [
                f.get("title", str(f)) if isinstance(f, dict) else str(f)
                for f in result["findings"][:5]
            ]

        if "recommendations" in result:
            simplified["recommendations"] = [
                r.get("recommendation", str(r)) if isinstance(r, dict) else str(r)
                for r in result["recommendations"][:5]
            ]

        # Scores
        stage2 = load_stage_results(ws, 2)
        scores = {}
        for kernel_name, kernel_data in stage2.items():
            if "risk_score" in kernel_data:
                scores["risk"] = kernel_data["risk_score"]
            if "compliance_score" in kernel_data:
                scores["compliance"] = kernel_data["compliance_score"]
        if scores:
            simplified["scores"] = scores

        # Summary
        findings_count = len(result.get("findings", []))
        rec_count = len(result.get("recommendations", []))
        simplified["summary"] = f"Audit complete. {findings_count} findings, {rec_count} recommendations. Report: {report_file.name}"

        return simplified

    except Exception as e:
        return wrap_kernel_error(e, "report_assemble")


# ---------------------------------------------------------------------------
# 12. Entry point
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
