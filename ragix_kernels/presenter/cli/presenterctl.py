"""
presenterctl — CLI for the KOAS Presenter kernel family.

Commands:
    render   Run the full S1->S2->S3 pipeline on a document folder
    export   Export an existing workspace to PDF/HTML via marp-cli
    show     Display workspace info (slide count, metadata, stages)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)

# ANSI color helpers (auto-disabled for non-TTY)
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _c("1", text)


def _green(text: str) -> str:
    return _c("32", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _red(text: str) -> str:
    return _c("31", text)


def _cyan(text: str) -> str:
    return _c("36", text)


def _dim(text: str) -> str:
    return _c("2", text)


# ---------------------------------------------------------------------------
# Kernel map (lazy imports to avoid pulling in all dependencies)
# ---------------------------------------------------------------------------

_KERNEL_MAP = {
    # Stage 1
    "pres_folder_scan": ("ragix_kernels.presenter.kernels.pres_folder_scan", "PresFolderScanKernel"),
    "pres_content_extract": ("ragix_kernels.presenter.kernels.pres_content_extract", "PresContentExtractKernel"),
    "pres_asset_catalog": ("ragix_kernels.presenter.kernels.pres_asset_catalog", "PresAssetCatalogKernel"),
    # Stage 2
    "pres_semantic_normalize": ("ragix_kernels.presenter.kernels.pres_semantic_normalize", "PresSemanticNormalizeKernel"),
    "pres_slide_plan": ("ragix_kernels.presenter.kernels.pres_slide_plan", "PresSlidePlanKernel"),
    "pres_layout_assign": ("ragix_kernels.presenter.kernels.pres_layout_assign", "PresLayoutAssignKernel"),
    # Stage 3
    "pres_marp_render": ("ragix_kernels.presenter.kernels.pres_marp_render", "PresMarpRenderKernel"),
    "pres_marp_export": ("ragix_kernels.presenter.kernels.pres_marp_export", "PresMarpExportKernel"),
}


def _get_presenter_kernel(kernel_name: str):
    """Resolve a presenter kernel by name using direct imports.

    This avoids the global KernelRegistry.discover() which walks all
    ragix_kernels subpackages and may crash on optional dependencies.
    """
    entry = _KERNEL_MAP.get(kernel_name)
    if entry is None:
        return None

    import importlib
    module_path, class_name = entry
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def _resolve_workspace(folder_path: Path, workspace: Optional[Path]) -> Path:
    """Resolve the workspace directory for a folder.

    Default: <folder>/.presenter/<stem>_<hash12>/
    """
    if workspace:
        return workspace

    h = hashlib.sha256(str(folder_path.resolve()).encode()).hexdigest()[:12]
    ws = folder_path / ".presenter" / f"{folder_path.name}_{h}"
    return ws


# ---------------------------------------------------------------------------
# Kernel runner
# ---------------------------------------------------------------------------

def _run_kernel(
    kernel_name: str,
    workspace: Path,
    config: dict,
    verbose: bool = False,
) -> None:
    """Run a single kernel by name."""
    print(f"  [{_cyan(kernel_name)}] ", end="", flush=True)

    try:
        from ragix_kernels.base import KernelInput

        kernel_cls = _get_presenter_kernel(kernel_name)

        if kernel_cls is None:
            print(_yellow("not found (skipped)"))
            return

        kernel = kernel_cls()

        # Build dependencies map from workspace
        deps = _discover_dependencies(kernel.requires, workspace)

        ki = KernelInput(
            workspace=workspace,
            config=config,
            dependencies=deps,
        )

        result = kernel.run(ki)

        if result.success:
            print(_green(result.summary or "done"))
        else:
            errors = "; ".join(result.errors) if result.errors else "unknown error"
            print(_red(f"FAILED: {errors}"))
            if verbose and result.errors:
                for e in result.errors:
                    print(f"    {_dim(e)}")

    except Exception as e:
        print(_red(f"ERROR: {e}"))
        if verbose:
            import traceback
            traceback.print_exc()


def _discover_dependencies(requires: List[str], workspace: Path) -> dict:
    """Find output files from required kernels."""
    deps = {}
    for req in requires:
        for stage in ("stage1", "stage2", "stage3", "output"):
            candidate = workspace / stage / f"{req}.json"
            if candidate.exists():
                deps[req] = candidate
                break
    return deps


# ---------------------------------------------------------------------------
# Command: render
# ---------------------------------------------------------------------------

def cmd_render(args: argparse.Namespace) -> int:
    """Run the full presenter pipeline."""
    folder_path = Path(args.folder).resolve()
    if not folder_path.is_dir():
        print(_red(f"Error: Not a directory: {folder_path}"))
        return 1

    workspace = _resolve_workspace(
        folder_path, Path(args.workspace) if args.workspace else None
    )
    workspace.mkdir(parents=True, exist_ok=True)

    print(_bold(f"KOAS Presenter — {folder_path.name}"))
    print(f"Workspace: {_dim(str(workspace))}")
    if args.compression != "full":
        print(f"Compression: {_yellow(args.compression)}")
    print()

    # Determine mode
    mode = args.mode
    if mode == "auto":
        mode = "deterministic"  # default to deterministic

    # Resolve max_slides default (25 for executive, 60 otherwise)
    max_slides = args.max_slides
    if max_slides is None:
        max_slides = 25 if args.compression == "executive" else 60

    # Build config
    config = {
        "folder_path": str(folder_path),
        "normalizer": {
            "mode": mode,
            "model": args.model,
        },
        "slide_plan": {
            "max_slides": max_slides,
            "section_order": args.section_order,
            "compression": args.compression,
            "toc": {
                "enabled": args.toc,
            },
        },
        "theme": {
            "name": args.theme if not Path(args.theme).suffix else "custom",
            "custom_css_path": args.theme if Path(args.theme).suffix else None,
        },
        "export": {
            "format": args.format,
        },
        "llm": {
            "backend": "ollama",
            "endpoint": "http://127.0.0.1:11434",
            "model": args.model,
        },
        "title": args.title or "",
        "subtitle": "",
        "author": args.author or "",
        "organization": args.organization or "",
        "date": args.date or datetime.now().strftime("%Y-%m-%d"),
        "lang": "auto",
    }

    # Stage 1 — Collection
    print(_bold("Stage 1: Collection"))
    _run_kernel("pres_folder_scan", workspace, config, args.verbose)
    _run_kernel("pres_content_extract", workspace, config, args.verbose)
    _run_kernel("pres_asset_catalog", workspace, config, args.verbose)
    print()

    # Stage 2 — Structuring
    print(_bold("Stage 2: Structuring"))
    _run_kernel("pres_semantic_normalize", workspace, config, args.verbose)
    _run_kernel("pres_slide_plan", workspace, config, args.verbose)
    _run_kernel("pres_layout_assign", workspace, config, args.verbose)
    print()

    # Stage 3 — Rendering
    print(_bold("Stage 3: Rendering"))
    _run_kernel("pres_marp_render", workspace, config, args.verbose)
    _run_kernel("pres_marp_export", workspace, config, args.verbose)
    print()

    # Summary
    output_dir = workspace / "output"
    pres_file = output_dir / "presentation.md"
    if pres_file.exists():
        print(_green(f"Markdown: {pres_file}"))
        print(f"  Size: {_dim(f'{pres_file.stat().st_size:,} bytes')}")

    for fmt_ext in ("pdf", "html"):
        out_file = output_dir / f"presentation.{fmt_ext}"
        if out_file.exists():
            print(_green(f"{fmt_ext.upper()}: {out_file}"))
            print(f"  Size: {_dim(f'{out_file.stat().st_size:,} bytes')}")

    # Count slides from render output
    render_json = workspace / "stage3" / "pres_marp_render.json"
    if render_json.exists():
        try:
            data = json.loads(render_json.read_text(encoding="utf-8"))
            n_slides = data.get("data", {}).get("slide_count", "?")
            print(f"Slides: {_bold(str(n_slides))}")
        except Exception:
            pass

    return 0


# ---------------------------------------------------------------------------
# Command: export
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    """Export an existing workspace to PDF/HTML."""
    workspace = Path(args.workspace).resolve()
    pres_file = workspace / "output" / "presentation.md"

    if not pres_file.exists():
        print(_red(f"Error: No presentation.md found at {pres_file}"))
        print(_dim("Run 'presenterctl render' first to create a presentation."))
        return 1

    print(_bold("KOAS Presenter — Export"))
    print(f"Source: {_dim(str(pres_file))}")
    print()

    from ragix_kernels.presenter.kernels.pres_marp_export import (
        run_marp_cli,
        _resolve_theme_css,
    )

    # Resolve theme
    theme_arg = args.theme
    if Path(theme_arg).suffix:
        theme_css = _resolve_theme_css(None, theme_arg)
    else:
        theme_css = _resolve_theme_css(theme_arg, None)

    # Determine formats
    export_format = args.format
    formats = ["pdf", "html"] if export_format == "both" else [export_format]

    for fmt in formats:
        print(f"  [{_cyan(f'marp-cli {fmt}')}] ", end="", flush=True)

        result = run_marp_cli(
            presentation_md=pres_file,
            output_format=fmt,
            theme_css=theme_css,
        )

        if result["success"]:
            out_path = Path(result["output_file"])
            size = out_path.stat().st_size if out_path.exists() else 0
            print(_green(f"{out_path.name} ({size:,} bytes)"))
        else:
            stderr = result.get("stderr", "unknown error")
            print(_red(f"FAILED (exit {result.get('exit_code', '?')})"))
            if args.verbose:
                print(f"    {_dim(stderr[:300])}")

    print()
    print(_green(f"Output: {workspace / 'output'}"))
    return 0


# ---------------------------------------------------------------------------
# Command: show
# ---------------------------------------------------------------------------

def cmd_show(args: argparse.Namespace) -> int:
    """Display workspace info."""
    workspace = Path(args.workspace).resolve()

    if not workspace.exists():
        print(_red(f"Error: Workspace not found: {workspace}"))
        return 1

    print(_bold("KOAS Presenter — Workspace Info"))
    print(f"Path: {_dim(str(workspace))}")
    print()

    # Check stages
    for stage_name in ("stage1", "stage2", "stage3"):
        stage_dir = workspace / stage_name
        if stage_dir.exists():
            files = sorted(stage_dir.glob("*.json"))
            if files:
                print(f"  {_bold(stage_name)}: {_green(f'{len(files)} files')}")
                for f in files:
                    size = f.stat().st_size
                    print(f"    {f.name} ({_dim(f'{size:,} bytes')})")
            else:
                print(f"  {_bold(stage_name)}: {_dim('empty')}")
        else:
            print(f"  {_bold(stage_name)}: {_dim('not created')}")

    # Output directory
    output_dir = workspace / "output"
    if output_dir.exists():
        print()
        print(f"  {_bold('output')}:")
        pres_file = output_dir / "presentation.md"
        if pres_file.exists():
            print(f"    presentation.md ({_dim(f'{pres_file.stat().st_size:,} bytes')})")
        for ext in ("pdf", "html"):
            out_file = output_dir / f"presentation.{ext}"
            if out_file.exists():
                print(f"    presentation.{ext} ({_dim(f'{out_file.stat().st_size:,} bytes')})")

        # Assets
        assets_dir = output_dir / "assets"
        if assets_dir.exists():
            assets = list(assets_dir.iterdir())
            if assets:
                total = sum(a.stat().st_size for a in assets if a.is_file())
                print(f"    assets/ ({_dim(f'{len(assets)} files, {total:,} bytes')})")

    # Metadata
    meta_file = output_dir / "metadata.json" if output_dir.exists() else None
    if meta_file and meta_file.exists():
        print()
        print(f"  {_bold('Metadata')}:")
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            for key in ("title", "author", "organization", "date", "generated_at"):
                val = meta.get(key, "")
                if val:
                    print(f"    {key}: {val}")
        except Exception:
            print(f"    {_dim('(could not read metadata.json)')}")

    # Slide count from render output
    render_json = workspace / "stage3" / "pres_marp_render.json"
    if render_json.exists():
        try:
            data = json.loads(render_json.read_text(encoding="utf-8"))
            n_slides = data.get("data", {}).get("slide_count", "?")
            n_assets = len(data.get("data", {}).get("asset_refs", []))
            print()
            print(f"  Slides: {_bold(str(n_slides))}")
            print(f"  Asset refs: {n_assets}")
        except Exception:
            pass

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the presenterctl argument parser."""
    parser = argparse.ArgumentParser(
        prog="presenterctl",
        description="KOAS Presenter — folder-to-slides pipeline with MARP export",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- render ---
    p_render = sub.add_parser("render", help="Run the full S1->S2->S3 pipeline")
    p_render.add_argument("folder", help="Path to document folder (input corpus)")
    p_render.add_argument("-w", "--workspace", help="Workspace directory (default: auto)")
    p_render.add_argument(
        "--mode", default="auto",
        choices=["deterministic", "llm", "auto"],
        help="Processing mode (default: auto -> deterministic)"
    )
    p_render.add_argument(
        "--model", default="mistral-small:24b",
        help="Ollama model for LLM mode (default: mistral-small:24b)"
    )
    p_render.add_argument(
        "--max-slides", type=int, default=None,
        help="Maximum slide count (default: 60, or 25 in executive mode)"
    )
    p_render.add_argument(
        "--compression", default="full",
        choices=["full", "compressed", "executive"],
        help="Compression mode: full (archive), compressed (40-60 slides), executive (15-25 slides)"
    )
    p_render.add_argument(
        "--toc", default=True, action="store_true",
        help="Generate a Table of Contents slide (default: enabled)"
    )
    p_render.add_argument(
        "--no-toc", dest="toc", action="store_false",
        help="Disable Table of Contents slide"
    )
    p_render.add_argument("--title", default="", help="Presentation title")
    p_render.add_argument("--author", default="", help="Author name")
    p_render.add_argument("--organization", default="", help="Organization name")
    p_render.add_argument("--date", default="", help="Date (default: today)")
    p_render.add_argument(
        "--section-order", default="document",
        choices=["document", "narrative"],
        help="Section ordering (default: document)"
    )
    p_render.add_argument(
        "-f", "--format", default="md",
        choices=["md", "pdf", "html", "both"],
        help="Output format (default: md)"
    )
    p_render.add_argument(
        "--theme", default="koas-professional",
        help="MARP theme: default|gaia|koas-professional|<path> (default: koas-professional)"
    )
    p_render.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    p_render.set_defaults(func=cmd_render)

    # --- export ---
    p_export = sub.add_parser("export", help="Export existing workspace to PDF/HTML")
    p_export.add_argument("workspace", help="Path to workspace with output/presentation.md")
    p_export.add_argument(
        "-f", "--format", default="both",
        choices=["pdf", "html", "both"],
        help="Export format (default: both)"
    )
    p_export.add_argument(
        "--theme", default="koas-professional",
        help="MARP theme: default|gaia|koas-professional|<path> (default: koas-professional)"
    )
    p_export.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    p_export.set_defaults(func=cmd_export)

    # --- show ---
    p_show = sub.add_parser("show", help="Display workspace info")
    p_show.add_argument("workspace", help="Path to workspace directory")
    p_show.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    p_show.set_defaults(func=cmd_show)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for presenterctl."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
