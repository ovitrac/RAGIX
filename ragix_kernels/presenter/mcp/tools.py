"""
MCP tools for the KOAS Presenter kernel family.

Three tools following KOAS MCP conventions:
    presenter_render    Run the full S1->S2->S3 pipeline
    presenter_export    Export existing workspace to PDF/HTML
    presenter_status    Query workspace state and metadata

These functions are designed to be registered with a FastMCP server
via @mcp.tool() or called directly as library functions.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_error(error: Exception, tool_name: str) -> Dict[str, Any]:
    """Standardized error response."""
    return {
        "status": "error",
        "error": str(error),
        "tool": tool_name,
        "summary": f"{tool_name} failed: {str(error)[:150]}",
    }


# ---------------------------------------------------------------------------
# Tool: presenter_render
# ---------------------------------------------------------------------------

def presenter_render(
    folder_path: str,
    workspace: str = "",
    mode: str = "deterministic",
    model: str = "mistral-small:24b",
    title: str = "",
    author: str = "",
    organization: str = "",
    date: str = "",
    max_slides: int = 60,
    section_order: str = "document",
    export_format: str = "md",
    theme: str = "koas-professional",
) -> Dict[str, Any]:
    """
    Run the full KOAS Presenter pipeline on a document folder.

    Parameters
    ----------
    folder_path : str
        Path to the document folder (input corpus).
    workspace : str, optional
        Workspace directory. Auto-created if empty.
    mode : str, default "deterministic"
        Processing mode: "deterministic" or "llm".
    model : str, default "mistral-small:24b"
        Ollama model for LLM mode.
    title : str, optional
        Presentation title.
    author : str, optional
        Author name.
    organization : str, optional
        Organization name.
    date : str, optional
        Date string (default: today).
    max_slides : int, default 60
        Maximum slide count.
    section_order : str, default "document"
        Section ordering: "document" or "narrative".
    export_format : str, default "md"
        Output format: "md", "pdf", "html", or "both".
    theme : str, default "koas-professional"
        MARP theme name or CSS file path.

    Returns
    -------
    dict
        {
          "status": "completed|error",
          "summary": str,
          "workspace": str,
          "output_dir": str,
          "slide_count": int,
          "presentation_file": str,
          "pdf_file": str,
          "html_file": str,
        }
    """
    try:
        from ragix_kernels.presenter.cli.presenterctl import (
            _run_kernel,
            _resolve_workspace,
        )

        folder = Path(folder_path).resolve()
        if not folder.is_dir():
            return {
                "status": "error",
                "error": f"Not a directory: {folder_path}",
                "summary": "Folder not found.",
            }

        ws = _resolve_workspace(
            folder, Path(workspace) if workspace else None
        )
        ws.mkdir(parents=True, exist_ok=True)

        # Resolve theme
        is_css_path = Path(theme).suffix == ".css"
        config = {
            "folder_path": str(folder),
            "normalizer": {
                "mode": mode,
                "model": model,
            },
            "slide_plan": {
                "max_slides": max_slides,
                "section_order": section_order,
            },
            "theme": {
                "name": theme if not is_css_path else "custom",
                "custom_css_path": theme if is_css_path else None,
            },
            "export": {
                "format": export_format,
            },
            "llm": {
                "backend": "ollama",
                "endpoint": "http://127.0.0.1:11434",
                "model": model,
            },
            "title": title,
            "subtitle": "",
            "author": author,
            "organization": organization,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "lang": "auto",
        }

        # Stage 1
        for k in ["pres_folder_scan", "pres_content_extract", "pres_asset_catalog"]:
            _run_kernel(k, ws, config, verbose=False)

        # Stage 2
        for k in ["pres_semantic_normalize", "pres_slide_plan", "pres_layout_assign"]:
            _run_kernel(k, ws, config, verbose=False)

        # Stage 3
        for k in ["pres_marp_render", "pres_marp_export"]:
            _run_kernel(k, ws, config, verbose=False)

        # Collect results
        output_dir = ws / "output"
        pres_file = output_dir / "presentation.md"

        # Slide count
        slide_count = 0
        render_json = ws / "stage3" / "pres_marp_render.json"
        if render_json.exists():
            try:
                rdata = json.loads(render_json.read_text(encoding="utf-8"))
                slide_count = rdata.get("data", {}).get("slide_count", 0)
            except Exception:
                pass

        result: Dict[str, Any] = {
            "status": "completed",
            "workspace": str(ws),
            "output_dir": str(output_dir),
            "presentation_file": str(pres_file) if pres_file.exists() else "",
            "slide_count": slide_count,
            "summary": (
                f"Presentation complete: {slide_count} slides. "
                f"Workspace: {ws}"
            ),
        }

        # Check for exported files
        for ext in ("pdf", "html"):
            out = output_dir / f"presentation.{ext}"
            if out.exists():
                result[f"{ext}_file"] = str(out)

        return result

    except Exception as e:
        return _wrap_error(e, "presenter_render")


# ---------------------------------------------------------------------------
# Tool: presenter_export
# ---------------------------------------------------------------------------

def presenter_export(
    workspace: str,
    export_format: str = "both",
    theme: str = "koas-professional",
) -> Dict[str, Any]:
    """
    Export an existing presenter workspace to PDF/HTML via marp-cli.

    Parameters
    ----------
    workspace : str
        Path to workspace with output/presentation.md.
    export_format : str, default "both"
        Export format: "pdf", "html", or "both".
    theme : str, default "koas-professional"
        MARP theme name or CSS file path.

    Returns
    -------
    dict
        {
          "status": "completed|error",
          "summary": str,
          "pdf_file": str,
          "html_file": str,
        }
    """
    try:
        from ragix_kernels.presenter.kernels.pres_marp_export import (
            run_marp_cli,
            _resolve_theme_css,
        )

        ws = Path(workspace).resolve()
        pres_file = ws / "output" / "presentation.md"

        if not pres_file.exists():
            return {
                "status": "error",
                "error": f"No presentation.md at {pres_file}",
                "summary": "Run presenter_render first.",
            }

        # Resolve theme
        is_css_path = Path(theme).suffix == ".css"
        if is_css_path:
            theme_css = _resolve_theme_css(None, theme)
        else:
            theme_css = _resolve_theme_css(theme, None)

        formats = ["pdf", "html"] if export_format == "both" else [export_format]
        result: Dict[str, Any] = {"status": "completed"}
        files_created = []

        for fmt in formats:
            marp_result = run_marp_cli(
                presentation_md=pres_file,
                output_format=fmt,
                theme_css=theme_css,
            )

            if marp_result["success"]:
                result[f"{fmt}_file"] = marp_result["output_file"]
                files_created.append(f"{fmt.upper()}: {Path(marp_result['output_file']).name}")
            else:
                result[f"{fmt}_error"] = marp_result.get("stderr", "unknown error")

        result["summary"] = (
            f"Exported: {', '.join(files_created)}" if files_created
            else "No files exported (marp-cli failed)"
        )

        if not files_created:
            result["status"] = "error"

        return result

    except Exception as e:
        return _wrap_error(e, "presenter_export")


# ---------------------------------------------------------------------------
# Tool: presenter_status
# ---------------------------------------------------------------------------

def presenter_status(
    workspace: str,
) -> Dict[str, Any]:
    """
    Query the status and metadata of a presenter workspace.

    Parameters
    ----------
    workspace : str
        Path to the presenter workspace.

    Returns
    -------
    dict
        {
          "status": "completed|in_progress|no_workspace|error",
          "summary": str,
          "workspace": str,
          "stages_completed": list,
          "slide_count": int,
          "output_files": list,
          "metadata": dict,
        }
    """
    try:
        ws = Path(workspace).resolve()

        if not ws.exists():
            return {
                "status": "no_workspace",
                "summary": f"No workspace found at {workspace}",
                "workspace": str(ws),
            }

        # Check which stages completed
        stages = []
        for stage in ("stage1", "stage2", "stage3"):
            stage_dir = ws / stage
            if stage_dir.exists() and any(stage_dir.glob("*.json")):
                stages.append(stage)

        # Output files
        output_dir = ws / "output"
        output_files = []
        if output_dir.exists():
            for f in sorted(output_dir.iterdir()):
                if f.is_file():
                    output_files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                    })

        # Slide count
        slide_count = 0
        render_json = ws / "stage3" / "pres_marp_render.json"
        if render_json.exists():
            try:
                rdata = json.loads(render_json.read_text(encoding="utf-8"))
                slide_count = rdata.get("data", {}).get("slide_count", 0)
            except Exception:
                pass

        # Metadata
        metadata = {}
        meta_file = output_dir / "metadata.json" if output_dir.exists() else None
        if meta_file and meta_file.exists():
            try:
                metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        is_complete = "stage3" in stages and (output_dir / "presentation.md").exists()

        return {
            "status": "completed" if is_complete else "in_progress",
            "workspace": str(ws),
            "stages_completed": stages,
            "slide_count": slide_count,
            "output_files": output_files,
            "metadata": {
                k: metadata.get(k, "")
                for k in ("title", "author", "organization", "date", "generated_at")
            },
            "summary": (
                f"{'Complete' if is_complete else 'In progress'}: "
                f"{slide_count} slides, stages {', '.join(stages) or 'none'}. "
                f"{len(output_files)} output files."
            ),
        }

    except Exception as e:
        return _wrap_error(e, "presenter_status")


# ---------------------------------------------------------------------------
# Registration helper for FastMCP
# ---------------------------------------------------------------------------

def register_presenter_tools(mcp_server) -> None:
    """
    Register all presenter MCP tools with a FastMCP server instance.

    Usage:
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("RAGIX")
        from ragix_kernels.presenter.mcp.tools import register_presenter_tools
        register_presenter_tools(mcp)
    """
    mcp_server.tool()(presenter_render)
    mcp_server.tool()(presenter_export)
    mcp_server.tool()(presenter_status)
    logger.info("Registered 3 KOAS Presenter MCP tools")
