"""
reviewctl — CLI for the KOAS Reviewer kernel family.

Commands:
    review   Run the full review pipeline on a Markdown document
    report   Generate REVIEW_doc.md from an existing ledger
    revert   Revert one or more changes by their RVW-NNNN IDs
    show     Show details of a specific change from the ledger
    grep     Search ledger entries by kind, severity, or text

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import argparse
import json
import sys
from datetime import datetime, timezone
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
# Workspace resolution
# ---------------------------------------------------------------------------

def _resolve_workspace(doc_path: Path, workspace: Optional[Path]) -> Path:
    """Resolve the workspace directory for a document."""
    if workspace:
        return workspace

    # Default: .review/<doc_stem>_<hash>/ next to the document
    from ragix_kernels.reviewer.models import content_hash
    text = doc_path.read_text(encoding="utf-8")
    h = content_hash(text).split(":")[-1][:12]
    ws = doc_path.parent / ".review" / f"{doc_path.stem}_{h}"
    return ws


# ---------------------------------------------------------------------------
# Command: review
# ---------------------------------------------------------------------------

def cmd_review(args: argparse.Namespace) -> int:
    """Run the full review pipeline."""
    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        print(_red(f"Error: File not found: {doc_path}"))
        return 1

    workspace = _resolve_workspace(doc_path, Path(args.workspace) if args.workspace else None)
    workspace.mkdir(parents=True, exist_ok=True)

    print(_bold(f"KOAS Reviewer — {doc_path.name}"))
    print(f"Workspace: {_dim(str(workspace))}")
    print()

    # Build config
    config = {
        "doc_path": str(doc_path),
        "reviewer": {
            "in_place": args.in_place,
            "strict": args.strict,
            "skip_pyramid": args.skip_pyramid,
            "no_llm": args.no_llm,
            "llm": {
                "backend": args.backend,
                "endpoint": args.endpoint,
                "edit_model": args.model,
                "tutor_model": args.tutor_model,
                "strict_sovereign": args.strict_sovereign,
            },
            "style": {},
        },
    }

    # Copy document to workspace
    stage1_dir = workspace / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    snapshot = stage1_dir / "doc.raw.md"
    if not snapshot.exists():
        import shutil
        shutil.copy2(doc_path, snapshot)
        print(f"  Snapshot: {_green('created')}")
    else:
        print(f"  Snapshot: {_dim('exists (incremental mode)')}")

    # Run kernels in order
    from ragix_kernels.base import KernelInput

    # Stage 1
    _run_kernel("md_inventory", workspace, config, args.verbose)
    _run_kernel("md_structure", workspace, config, args.verbose)
    _run_kernel("md_protected_regions", workspace, config, args.verbose)
    _run_kernel("md_chunk", workspace, config, args.verbose)

    # Stage 2
    if not args.no_llm:
        if not args.skip_pyramid:
            _run_kernel("md_pyramid", workspace, config, args.verbose)
        _run_kernel("md_consistency_scan", workspace, config, args.verbose)
        _run_kernel("md_numbering_control", workspace, config, args.verbose)
        _run_kernel("md_edit_plan", workspace, config, args.verbose)
    else:
        _run_kernel("md_consistency_scan", workspace, config, args.verbose)
        _run_kernel("md_numbering_control", workspace, config, args.verbose)

    # Stage 3
    _run_kernel("md_apply_ops", workspace, config, args.verbose)
    _run_kernel("md_inline_notes_inject", workspace, config, args.verbose)
    _run_kernel("md_review_report_assemble", workspace, config, args.verbose)

    # Summary
    print()
    report_path = workspace / "review"
    report_name = f"REVIEW_{doc_path.stem}.md"
    report_file = report_path / report_name
    if report_file.exists():
        print(_green(f"Report: {report_file}"))

    edited_path = workspace / "stage3" / "doc.edited.md"
    if edited_path.exists():
        # Find reviewed doc
        suffix = ".REVIEWED.md" if not args.in_place else ""
        reviewed = workspace / "stage3" / (doc_path.stem + suffix)
        if reviewed.exists():
            print(_green(f"Reviewed: {reviewed}"))
        else:
            print(_green(f"Edited: {edited_path}"))

    return 0


def _get_reviewer_kernel(kernel_name: str):
    """
    Resolve a reviewer kernel by name using direct imports.

    This avoids the global KernelRegistry.discover() which walks all
    ragix_kernels subpackages (including docs/) and may crash on
    optional dependencies (matplotlib, scipy, etc.).
    """
    # Lazy import map — only reviewer kernels
    _KERNEL_MAP = {
        # Stage 1
        "md_inventory": ("ragix_kernels.reviewer.kernels.md_inventory", "MdInventoryKernel"),
        "md_structure": ("ragix_kernels.reviewer.kernels.md_structure", "MdStructureKernel"),
        "md_protected_regions": ("ragix_kernels.reviewer.kernels.md_protected_regions", "MdProtectedRegionsKernel"),
        "md_chunk": ("ragix_kernels.reviewer.kernels.md_chunk", "MdChunkKernel"),
        # Stage 2
        "md_consistency_scan": ("ragix_kernels.reviewer.kernels.md_consistency_scan", "MdConsistencyScanKernel"),
        "md_numbering_control": ("ragix_kernels.reviewer.kernels.md_numbering_control", "MdNumberingControlKernel"),
        "md_pyramid": ("ragix_kernels.reviewer.kernels.md_pyramid", "MdPyramidKernel"),
        "md_fingerprint_chunk": ("ragix_kernels.reviewer.kernels.md_fingerprint_chunk", "MdFingerprintChunkKernel"),
        "md_edit_plan": ("ragix_kernels.reviewer.kernels.md_edit_plan", "MdEditPlanKernel"),
        # Stage 3
        "md_apply_ops": ("ragix_kernels.reviewer.kernels.md_apply_ops", "MdApplyOpsKernel"),
        "md_inline_notes_inject": ("ragix_kernels.reviewer.kernels.md_inline_notes_inject", "MdInlineNotesInjectKernel"),
        "md_review_report_assemble": ("ragix_kernels.reviewer.kernels.md_review_report_assemble", "MdReviewReportAssembleKernel"),
        "md_revert": ("ragix_kernels.reviewer.kernels.md_revert", "MdRevertKernel"),
    }

    entry = _KERNEL_MAP.get(kernel_name)
    if entry is None:
        return None

    import importlib
    module_path, class_name = entry
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


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

        kernel_cls = _get_reviewer_kernel(kernel_name)

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
        # Check stage1, stage2, stage3 for the output file
        for stage in ("stage1", "stage2", "stage3", "review"):
            candidate = workspace / stage / f"{req}.json"
            if candidate.exists():
                deps[req] = candidate
                break
    return deps


# ---------------------------------------------------------------------------
# Command: report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    """Generate or show the review report."""
    doc_path = Path(args.doc).resolve()
    workspace = _resolve_workspace(doc_path, Path(args.workspace) if args.workspace else None)

    ledger_path = workspace / "review" / "ledger.jsonl"
    if not ledger_path.exists():
        print(_red(f"No ledger found at {ledger_path}"))
        return 1

    report_name = f"REVIEW_{doc_path.stem}.md"
    report_path = workspace / "review" / report_name
    if report_path.exists():
        print(report_path.read_text(encoding="utf-8"))
    else:
        # Generate report
        config = {"doc_path": str(doc_path), "reviewer": {}}
        _run_kernel("md_review_report_assemble", workspace, config, args.verbose)
        if report_path.exists():
            print(report_path.read_text(encoding="utf-8"))
        else:
            print(_yellow("Report generation produced no output."))

    return 0


# ---------------------------------------------------------------------------
# Command: revert
# ---------------------------------------------------------------------------

def cmd_revert(args: argparse.Namespace) -> int:
    """Revert one or more changes."""
    doc_path = Path(args.doc).resolve()
    workspace = _resolve_workspace(doc_path, Path(args.workspace) if args.workspace else None)

    change_ids = args.change_ids

    # Validate change IDs
    from ragix_kernels.reviewer.models import ChangeID
    for cid in change_ids:
        try:
            ChangeID.parse(cid)
        except ValueError:
            print(_red(f"Invalid change ID: {cid}"))
            return 1

    print(_bold(f"Reverting {len(change_ids)} change(s)..."))

    # Load current document
    edited_path = workspace / "stage3" / "doc.edited.md"
    if not edited_path.exists():
        print(_red("No edited document found. Run 'review' first."))
        return 1

    text = edited_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    from ragix_kernels.reviewer.ledger import Ledger
    from ragix_kernels.reviewer.kernels.md_revert import revert_single, revert_bulk

    review_dir = workspace / "review"
    patches_dir = review_dir / "patches"
    ledger = Ledger(review_dir / "ledger.jsonl")

    if len(change_ids) == 1:
        result = revert_single(lines, change_ids[0], ledger, patches_dir, str(doc_path))
        if result["success"]:
            edited_path.write_text("\n".join(result["reverted_lines"]), encoding="utf-8")
            print(_green(f"Reverted {change_ids[0]} -> {result['revert_id']}"))
        else:
            print(_red(f"Failed: {result['error']}"))
            return 1
    else:
        result = revert_bulk(lines, change_ids, ledger, patches_dir, str(doc_path))
        if result["total_reverted"] > 0:
            edited_path.write_text("\n".join(result["reverted_lines"]), encoding="utf-8")
        print(
            f"Reverted: {_green(str(result['total_reverted']))}, "
            f"Failed: {_red(str(result['total_failed']))}"
        )
        for r in result["results"]:
            status = _green("OK") if r["success"] else _red(r.get("error", "failed"))
            cid = r.get("reverted_change", r.get("error", "?"))
            print(f"  {cid}: {status}")

        if result["total_failed"] > 0:
            return 1

    return 0


# ---------------------------------------------------------------------------
# Command: show
# ---------------------------------------------------------------------------

def cmd_show(args: argparse.Namespace) -> int:
    """Show details of a specific change."""
    doc_path = Path(args.doc).resolve()
    workspace = _resolve_workspace(doc_path, Path(args.workspace) if args.workspace else None)

    from ragix_kernels.reviewer.ledger import Ledger

    ledger_path = workspace / "review" / "ledger.jsonl"
    if not ledger_path.exists():
        print(_red("No ledger found. Run 'review' first."))
        return 1

    ledger = Ledger(ledger_path)
    entry = ledger.get_by_id(args.change_id)

    if entry is None:
        print(_red(f"Change {args.change_id} not found in ledger"))
        return 1

    # Display entry details
    print(_bold(f"Change: {entry.id}"))
    print(f"  Document:   {entry.doc}")
    print(f"  Timestamp:  {entry.timestamp}")
    print(f"  Kind:       {entry.kind}")
    print(f"  Severity:   {_severity_color(entry.severity)}")
    print(f"  Silent:     {'yes' if entry.silent else 'no'}")
    print(f"  Is revert:  {'yes' if entry.is_revert else 'no'}")
    if entry.reverted_id:
        print(f"  Reverts:    {entry.reverted_id}")
    print(f"  Summary:    {entry.summary}")
    if entry.rationale:
        print(f"  Rationale:  {entry.rationale[:200]}")
    print(f"  Hash before: {_dim(entry.doc_hash_before[:30])}...")
    print(f"  Hash after:  {_dim(entry.doc_hash_after[:30])}...")
    if entry.scope:
        print(f"  Scope:      {json.dumps(entry.scope)}")
    if entry.review_note:
        print(f"  Review note: {json.dumps(entry.review_note)}")

    # Show patch if requested
    if args.patch:
        patches_dir = workspace / "review" / "patches"
        fwd = patches_dir / f"{entry.id}.patch"
        if fwd.exists():
            print()
            print(_bold("Forward patch:"))
            print(fwd.read_text(encoding="utf-8"))

    return 0


def _severity_color(severity: str) -> str:
    mapping = {
        "minor": _dim(severity),
        "attention": _yellow(severity),
        "deletion": _red(severity),
        "critical": _red(_bold(severity)),
    }
    return mapping.get(severity, severity)


# ---------------------------------------------------------------------------
# Command: grep
# ---------------------------------------------------------------------------

def cmd_grep(args: argparse.Namespace) -> int:
    """Search ledger entries."""
    doc_path = Path(args.doc).resolve()
    workspace = _resolve_workspace(doc_path, Path(args.workspace) if args.workspace else None)

    from ragix_kernels.reviewer.ledger import Ledger

    ledger_path = workspace / "review" / "ledger.jsonl"
    if not ledger_path.exists():
        print(_red("No ledger found. Run 'review' first."))
        return 1

    ledger = Ledger(ledger_path)
    entries = ledger.filter(
        kind=args.kind,
        severity=args.severity,
        silent=args.silent if args.silent is not None else None,
        is_revert=args.reverts if args.reverts is not None else None,
    )

    # Text search if pattern given
    if args.pattern:
        import re
        try:
            pat = re.compile(args.pattern, re.IGNORECASE)
        except re.error as e:
            print(_red(f"Invalid regex: {e}"))
            return 1

        entries = [
            e for e in entries
            if pat.search(e.summary or "") or pat.search(e.rationale or "") or pat.search(e.id)
        ]

    if not entries:
        print(_dim("No matching entries."))
        return 0

    # Display
    print(f"{'ID':<16} {'Kind':<14} {'Severity':<10} {'Silent':<7} Summary")
    print("-" * 80)
    for entry in entries:
        silent_tag = "yes" if entry.silent else "no"
        revert_tag = " (R)" if entry.is_revert else ""
        summary = (entry.summary or "")[:50]
        print(
            f"{entry.id + revert_tag:<16} {entry.kind:<14} "
            f"{_severity_color(entry.severity):<10} {silent_tag:<7} {summary}"
        )

    print(f"\n{_dim(f'{len(entries)} entries')}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the reviewctl argument parser."""
    parser = argparse.ArgumentParser(
        prog="reviewctl",
        description="KOAS Reviewer — traceable Markdown review and editing tool",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- review ---
    p_review = sub.add_parser("review", help="Run the full review pipeline")
    p_review.add_argument("doc", help="Path to Markdown document")
    p_review.add_argument("-w", "--workspace", help="Workspace directory (default: auto)")
    p_review.add_argument("--in-place", action="store_true", help="Overwrite original document")
    p_review.add_argument("--strict", action="store_true", help="Refuse edits on protected regions")
    p_review.add_argument("--skip-pyramid", action="store_true", help="Skip pyramidal KB construction")
    p_review.add_argument("--no-llm", action="store_true", help="Deterministic passes only (no LLM)")
    p_review.add_argument("--backend", default="ollama", choices=["ollama", "claude"],
                          help="LLM backend (default: ollama)")
    p_review.add_argument("--endpoint", default="http://127.0.0.1:11434",
                          help="Ollama endpoint URL")
    p_review.add_argument("--model", default="mistral:instruct",
                          help="Worker model (default: mistral:instruct)")
    p_review.add_argument("--tutor-model", default=None,
                          help="Tutor model for validation (default: none = single-model)")
    p_review.add_argument("--strict-sovereign", action="store_true", default=True,
                          help="Reject non-local backends (default: true)")
    p_review.add_argument("--no-strict-sovereign", action="store_false", dest="strict_sovereign",
                          help="Allow non-local backends")
    p_review.set_defaults(func=cmd_review)

    # --- report ---
    p_report = sub.add_parser("report", help="Generate or display the review report")
    p_report.add_argument("doc", help="Path to Markdown document")
    p_report.add_argument("-w", "--workspace", help="Workspace directory")
    p_report.set_defaults(func=cmd_report)

    # --- revert ---
    p_revert = sub.add_parser("revert", help="Revert one or more changes")
    p_revert.add_argument("doc", help="Path to Markdown document")
    p_revert.add_argument("change_ids", nargs="+", help="Change IDs to revert (RVW-NNNN)")
    p_revert.add_argument("-w", "--workspace", help="Workspace directory")
    p_revert.set_defaults(func=cmd_revert)

    # --- show ---
    p_show = sub.add_parser("show", help="Show details of a change")
    p_show.add_argument("doc", help="Path to Markdown document")
    p_show.add_argument("change_id", help="Change ID (RVW-NNNN)")
    p_show.add_argument("--patch", action="store_true", help="Show the forward patch")
    p_show.add_argument("-w", "--workspace", help="Workspace directory")
    p_show.set_defaults(func=cmd_show)

    # --- grep ---
    p_grep = sub.add_parser("grep", help="Search ledger entries")
    p_grep.add_argument("doc", help="Path to Markdown document")
    p_grep.add_argument("pattern", nargs="?", help="Regex pattern to search")
    p_grep.add_argument("-k", "--kind", help="Filter by kind (e.g. typo, logic_flow)")
    p_grep.add_argument("-s", "--severity", help="Filter by severity (minor|attention|deletion|critical)")
    p_grep.add_argument("--silent", type=_parse_bool, default=None, help="Filter by silent (true|false)")
    p_grep.add_argument("--reverts", type=_parse_bool, default=None, help="Show only reverts (true|false)")
    p_grep.add_argument("-w", "--workspace", help="Workspace directory")
    p_grep.set_defaults(func=cmd_grep)

    return parser


def _parse_bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for reviewctl."""
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
