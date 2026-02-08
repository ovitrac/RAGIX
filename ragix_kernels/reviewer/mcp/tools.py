"""
MCP tools for the KOAS Reviewer kernel family.

Four tools following KOAS MCP conventions:
    review_md_run       Run the full review pipeline
    review_md_status    Query review status and statistics
    review_md_revert    Revert one or more changes
    review_md_show_change  Show details of a specific change

These functions are designed to be registered with a FastMCP server
via @mcp.tool() or called directly as library functions.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_workspace(doc_path: str, workspace: str = "") -> Path:
    """Resolve workspace for a document, auto-creating if needed."""
    if workspace:
        ws = Path(workspace)
    else:
        p = Path(doc_path).resolve()
        from ragix_kernels.reviewer.models import content_hash
        text = p.read_text(encoding="utf-8")
        h = content_hash(text).split(":")[-1][:12]
        ws = p.parent / ".review" / f"{p.stem}_{h}"

    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _wrap_error(error: Exception, tool_name: str) -> Dict[str, Any]:
    """Standardized error response."""
    return {
        "status": "error",
        "error": str(error),
        "tool": tool_name,
        "summary": f"{tool_name} failed: {str(error)[:150]}",
    }


# ---------------------------------------------------------------------------
# Tool: review_md_run
# ---------------------------------------------------------------------------

def review_md_run(
    doc_path: str,
    workspace: str = "",
    backend: str = "ollama",
    model: str = "mistral:instruct",
    tutor_model: str = "",
    endpoint: str = "http://127.0.0.1:11434",
    skip_pyramid: bool = False,
    no_llm: bool = False,
    strict: bool = False,
    in_place: bool = False,
    strict_sovereign: bool = True,
) -> Dict[str, Any]:
    """
    Run the full KOAS Reviewer pipeline on a Markdown document.

    Parameters
    ----------
    doc_path : str
        Path to the Markdown document to review.
    workspace : str, optional
        Workspace directory. Auto-created if empty.
    backend : str, default "ollama"
        LLM backend: "ollama" or "claude".
    model : str, default "mistral:instruct"
        Worker model for edit plan generation.
    tutor_model : str, optional
        Tutor model for validation. Empty = single-model mode.
    endpoint : str, default "http://127.0.0.1:11434"
        Ollama API endpoint.
    skip_pyramid : bool, default False
        Skip pyramidal KB construction.
    no_llm : bool, default False
        Run only deterministic passes (no LLM).
    strict : bool, default False
        Refuse edits on protected regions.
    in_place : bool, default False
        Overwrite the original document.
    strict_sovereign : bool, default True
        Reject non-local LLM backends.

    Returns
    -------
    dict
        {
          "status": "completed|error",
          "summary": str,
          "workspace": str,
          "report_path": str,
          "reviewed_doc_path": str,
          "total_changes": int,
          "attention_changes": int,
        }

    Example
    -------
    >>> review_md_run("docs/spec.md")
    {"status": "completed", "summary": "Review complete: 12 changes", ...}
    """
    try:
        from ragix_kernels.reviewer.cli.reviewctl import (
            _run_kernel,
            _discover_dependencies,
        )

        p = Path(doc_path).resolve()
        if not p.exists():
            return {"status": "error", "error": f"File not found: {doc_path}",
                    "summary": "Document not found."}

        ws = _resolve_workspace(doc_path, workspace)

        config = {
            "doc_path": str(p),
            "reviewer": {
                "in_place": in_place,
                "strict": strict,
                "skip_pyramid": skip_pyramid,
                "no_llm": no_llm,
                "llm": {
                    "backend": backend,
                    "endpoint": endpoint,
                    "edit_model": model,
                    "tutor_model": tutor_model or None,
                    "strict_sovereign": strict_sovereign,
                },
                "style": {},
            },
        }

        # Snapshot
        stage1_dir = ws / "stage1"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        snapshot = stage1_dir / "doc.raw.md"
        if not snapshot.exists():
            import shutil
            shutil.copy2(p, snapshot)

        # Stage 1
        for k in ["md_inventory", "md_structure", "md_protected_regions", "md_chunk"]:
            _run_kernel(k, ws, config, verbose=False)

        # Stage 2
        if not no_llm:
            if not skip_pyramid:
                _run_kernel("md_pyramid", ws, config, verbose=False)
            _run_kernel("md_consistency_scan", ws, config, verbose=False)
            _run_kernel("md_numbering_control", ws, config, verbose=False)
            _run_kernel("md_edit_plan", ws, config, verbose=False)
        else:
            _run_kernel("md_consistency_scan", ws, config, verbose=False)
            _run_kernel("md_numbering_control", ws, config, verbose=False)

        # Stage 3
        _run_kernel("md_apply_ops", ws, config, verbose=False)
        _run_kernel("md_inline_notes_inject", ws, config, verbose=False)
        _run_kernel("md_review_report_assemble", ws, config, verbose=False)

        # Collect results
        report_name = f"REVIEW_{p.stem}.md"
        report_path = ws / "review" / report_name
        reviewed_path = ws / "stage3" / (p.stem + ".REVIEWED.md")
        edited_path = ws / "stage3" / "doc.edited.md"

        # Get change counts from ledger
        from ragix_kernels.reviewer.ledger import Ledger
        ledger_path = ws / "review" / "ledger.jsonl"
        counts = {"total_changes": 0, "attention_changes": 0}
        if ledger_path.exists():
            ledger = Ledger(ledger_path)
            counts = ledger.summary_counts()

        out_doc = str(reviewed_path) if reviewed_path.exists() else str(edited_path)

        return {
            "status": "completed",
            "workspace": str(ws),
            "report_path": str(report_path) if report_path.exists() else "",
            "reviewed_doc_path": out_doc if Path(out_doc).exists() else "",
            "total_changes": counts.get("total_changes", 0),
            "attention_changes": counts.get("attention_changes", 0),
            "silent_minor": counts.get("silent_minor", 0),
            "summary": (
                f"Review complete: {counts.get('total_changes', 0)} changes "
                f"({counts.get('attention_changes', 0)} attention, "
                f"{counts.get('silent_minor', 0)} silent). "
                f"Workspace: {ws}"
            ),
        }

    except Exception as e:
        return _wrap_error(e, "review_md_run")


# ---------------------------------------------------------------------------
# Tool: review_md_status
# ---------------------------------------------------------------------------

def review_md_status(
    doc_path: str,
    workspace: str = "",
) -> Dict[str, Any]:
    """
    Query the review status and statistics for a document.

    Parameters
    ----------
    doc_path : str
        Path to the Markdown document.
    workspace : str, optional
        Workspace directory.

    Returns
    -------
    dict
        {
          "status": "completed|no_review|error",
          "summary": str,
          "workspace": str,
          "counts": dict,
          "stages_completed": list,
          "ledger_entries": int,
        }
    """
    try:
        p = Path(doc_path).resolve()
        ws = _resolve_workspace(doc_path, workspace)

        if not ws.exists():
            return {
                "status": "no_review",
                "summary": f"No review workspace found for {p.name}",
                "workspace": str(ws),
            }

        # Check which stages completed
        stages = []
        for stage in ("stage1", "stage2", "stage3"):
            stage_dir = ws / stage
            if stage_dir.exists() and any(stage_dir.iterdir()):
                stages.append(stage)

        # Load ledger
        from ragix_kernels.reviewer.ledger import Ledger

        ledger_path = ws / "review" / "ledger.jsonl"
        counts = {}
        entries_count = 0
        if ledger_path.exists():
            ledger = Ledger(ledger_path)
            counts = ledger.summary_counts()
            entries_count = ledger.count

        # Check for report
        report_name = f"REVIEW_{p.stem}.md"
        report_exists = (ws / "review" / report_name).exists()

        # Check for edited doc
        edited_exists = (ws / "stage3" / "doc.edited.md").exists()

        return {
            "status": "completed" if "stage3" in stages else "in_progress",
            "workspace": str(ws),
            "stages_completed": stages,
            "ledger_entries": entries_count,
            "counts": counts,
            "report_exists": report_exists,
            "edited_doc_exists": edited_exists,
            "summary": (
                f"Review {'complete' if 'stage3' in stages else 'in progress'}: "
                f"{entries_count} ledger entries, "
                f"stages {', '.join(stages) or 'none'}. "
                f"{counts.get('total_changes', 0)} changes total."
            ),
        }

    except Exception as e:
        return _wrap_error(e, "review_md_status")


# ---------------------------------------------------------------------------
# Tool: review_md_revert
# ---------------------------------------------------------------------------

def review_md_revert(
    doc_path: str,
    change_ids: str,
    workspace: str = "",
) -> Dict[str, Any]:
    """
    Revert one or more changes by their RVW-NNNN IDs.

    Parameters
    ----------
    doc_path : str
        Path to the Markdown document.
    change_ids : str
        Comma-separated change IDs (e.g. "RVW-0001,RVW-0003").
    workspace : str, optional
        Workspace directory.

    Returns
    -------
    dict
        {
          "status": "completed|error",
          "summary": str,
          "total_reverted": int,
          "total_failed": int,
          "results": list,
        }
    """
    try:
        p = Path(doc_path).resolve()
        ws = _resolve_workspace(doc_path, workspace)

        ids = [cid.strip() for cid in change_ids.split(",") if cid.strip()]
        if not ids:
            return {"status": "error", "error": "No change IDs provided",
                    "summary": "Provide comma-separated RVW-NNNN IDs."}

        # Validate IDs
        from ragix_kernels.reviewer.models import ChangeID
        for cid in ids:
            try:
                ChangeID.parse(cid)
            except ValueError:
                return {"status": "error", "error": f"Invalid change ID: {cid}",
                        "summary": f"Invalid format: {cid}. Use RVW-NNNN."}

        # Load document
        edited_path = ws / "stage3" / "doc.edited.md"
        if not edited_path.exists():
            return {"status": "error", "error": "No edited document found",
                    "summary": "Run review_md_run first."}

        text = edited_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        from ragix_kernels.reviewer.ledger import Ledger
        from ragix_kernels.reviewer.kernels.md_revert import revert_single, revert_bulk

        review_dir = ws / "review"
        patches_dir = review_dir / "patches"
        ledger = Ledger(review_dir / "ledger.jsonl")

        if len(ids) == 1:
            result = revert_single(lines, ids[0], ledger, patches_dir, str(p))
            if result["success"]:
                edited_path.write_text("\n".join(result["reverted_lines"]), encoding="utf-8")
                return {
                    "status": "completed",
                    "total_reverted": 1,
                    "total_failed": 0,
                    "results": [{"change_id": ids[0], "revert_id": result["revert_id"]}],
                    "summary": f"Reverted {ids[0]} -> {result['revert_id']}",
                }
            else:
                return {
                    "status": "error",
                    "error": result["error"],
                    "total_reverted": 0,
                    "total_failed": 1,
                    "summary": f"Failed to revert {ids[0]}: {result['error']}",
                }
        else:
            result = revert_bulk(lines, ids, ledger, patches_dir, str(p))
            if result["total_reverted"] > 0:
                edited_path.write_text("\n".join(result["reverted_lines"]), encoding="utf-8")

            simple_results = [
                {
                    "change_id": r.get("reverted_change", "?"),
                    "success": r["success"],
                    "error": r.get("error", ""),
                }
                for r in result["results"]
            ]

            return {
                "status": "completed" if result["total_failed"] == 0 else "partial",
                "total_reverted": result["total_reverted"],
                "total_failed": result["total_failed"],
                "results": simple_results,
                "summary": (
                    f"Reverted {result['total_reverted']}/{len(ids)} changes. "
                    f"{result['total_failed']} failed."
                ),
            }

    except Exception as e:
        return _wrap_error(e, "review_md_revert")


# ---------------------------------------------------------------------------
# Tool: review_md_show_change
# ---------------------------------------------------------------------------

def review_md_show_change(
    doc_path: str,
    change_id: str,
    workspace: str = "",
    include_patch: bool = False,
) -> Dict[str, Any]:
    """
    Show details of a specific change from the ledger.

    Parameters
    ----------
    doc_path : str
        Path to the Markdown document.
    change_id : str
        Change ID (RVW-NNNN).
    workspace : str, optional
        Workspace directory.
    include_patch : bool, default False
        Include the forward patch text in the response.

    Returns
    -------
    dict
        {
          "status": "found|not_found|error",
          "summary": str,
          "change": dict (full ledger entry),
          "patch": str (if include_patch=True),
        }
    """
    try:
        p = Path(doc_path).resolve()
        ws = _resolve_workspace(doc_path, workspace)

        from ragix_kernels.reviewer.ledger import Ledger

        ledger_path = ws / "review" / "ledger.jsonl"
        if not ledger_path.exists():
            return {"status": "error", "error": "No ledger found",
                    "summary": "Run review_md_run first."}

        ledger = Ledger(ledger_path)
        entry = ledger.get_by_id(change_id)

        if entry is None:
            return {
                "status": "not_found",
                "summary": f"Change {change_id} not found in ledger.",
            }

        result: Dict[str, Any] = {
            "status": "found",
            "change": entry.to_dict(),
            "summary": (
                f"{change_id}: {entry.kind} ({entry.severity})"
                f"{' [silent]' if entry.silent else ''}"
                f"{' [REVERT]' if entry.is_revert else ''}"
                f" — {(entry.summary or '')[:100]}"
            ),
        }

        if include_patch:
            patches_dir = ws / "review" / "patches"
            fwd = patches_dir / f"{change_id}.patch"
            if fwd.exists():
                result["patch"] = fwd.read_text(encoding="utf-8")
            else:
                result["patch"] = "(patch file not found)"

        return result

    except Exception as e:
        return _wrap_error(e, "review_md_show_change")


# ---------------------------------------------------------------------------
# Registration helper for FastMCP
# ---------------------------------------------------------------------------

def register_reviewer_tools(mcp_server) -> None:
    """
    Register all reviewer MCP tools with a FastMCP server instance.

    Usage:
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("RAGIX")
        from ragix_kernels.reviewer.mcp.tools import register_reviewer_tools
        register_reviewer_tools(mcp)
    """
    mcp_server.tool()(review_md_run)
    mcp_server.tool()(review_md_status)
    mcp_server.tool()(review_md_revert)
    mcp_server.tool()(review_md_show_change)
    logger.info("Registered 4 KOAS Reviewer MCP tools")
