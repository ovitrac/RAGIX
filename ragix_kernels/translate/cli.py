"""
KOAS-Translate pipeline runner (`ragix-translate`).

Chains the six translate kernels over a workspace translation memory, in
dependency order. Unlike the audit-oriented `ragix-koas` orchestrator, this is a
small dedicated runner for the translate family: every kernel operates on the
shared workspace artifacts (`out/source.md`, `out/tm.sqlite`), and the
`requires` chain provides ordering.

Usage::

    ragix-translate run    -w PROJECT [--stages all] [--model M] [--glossary CSV] \\
                           [--lang-pair en-fr] [--limit N] [--src-dir DIR]
    ragix-translate status -w PROJECT

Each stage is idempotent/resumable, so re-running only does outstanding work.
LLM stages use the local Ollama backend by default (run on a machine with the
models); the deterministic stages (extract/segment/rebuild) need no LLM.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ragix_kernels.base import KernelInput

from . import tm_store
from .extract import TranslateExtractKernel
from .segment import TranslateSegmentKernel
from .draft import TranslateDraftKernel
from .qa import TranslateQAKernel
from .harmonize import TranslateHarmonizeKernel
from .rebuild import TranslateRebuildKernel

#: The translate pipeline, in dependency order.
PIPELINE = [
    TranslateExtractKernel,
    TranslateSegmentKernel,
    TranslateDraftKernel,
    TranslateQAKernel,
    TranslateHarmonizeKernel,
    TranslateRebuildKernel,
]
STAGE_NAMES: List[str] = [K.name for K in PIPELINE]
_STAGE_OF: Dict[str, int] = {K.name: K.stage for K in PIPELINE}


def run_pipeline(
    workspace: Path | str,
    config: Optional[Dict[str, Any]] = None,
    stages: Optional[List[str]] = None,
    *,
    backend: Optional[Callable[[str], str]] = None,
    extractor: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Run the selected translate kernels over *workspace*, in order.

    Dependencies are wired to each prior kernel's persisted JSON output
    (``workspace/stage{N}/{name}.json``); the kernels read their real inputs from
    the workspace, so the dependency only enforces ordering. Stops at the first
    failed kernel. *backend* / *extractor* inject deterministic stubs for tests.

    Returns one ``{kernel, success, summary, errors}`` dict per executed stage.
    """
    workspace = Path(workspace)
    config = dict(config or {})
    selected = set(stages) if stages else set(STAGE_NAMES)
    results: List[Dict[str, Any]] = []

    for K in PIPELINE:
        if K.name not in selected:
            continue
        deps: Dict[str, Path] = {}
        for req in K.requires:
            p = workspace / f"stage{_STAGE_OF[req]}" / f"{req}.json"
            if p.exists():
                deps[req] = p

        kernel = K()
        if backend is not None and hasattr(kernel, "backend"):
            kernel.backend = backend
        if extractor is not None and hasattr(kernel, "extractor"):
            kernel.extractor = extractor

        out = kernel.run(KernelInput(workspace=workspace, config=config, dependencies=deps))
        results.append({
            "kernel": K.name,
            "success": out.success,
            "summary": out.summary,
            "errors": out.errors,
        })
        if not out.success:
            break

    return results


def status(workspace: Path | str, tm_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """TM progress counts (segments / translated / qa / final / chapter_revisions)."""
    tm = Path(tm_path) if tm_path else Path(workspace) / "out" / "tm.sqlite"
    if not tm.exists():
        return {"tm_path": str(tm), "exists": False}
    with tm_store.connect(tm) as conn:
        q = lambda sql: conn.execute(sql).fetchone()[0]
        return {
            "tm_path": str(tm),
            "exists": True,
            "segments": q("SELECT COUNT(*) FROM segments"),
            "translated": q("SELECT COUNT(*) FROM segments WHERE raw_translation IS NOT NULL"),
            "qa": q("SELECT COUNT(*) FROM segments WHERE qa_report IS NOT NULL"),
            "final": q("SELECT COUNT(*) FROM segments WHERE final_translation IS NOT NULL"),
            "chapter_revisions": q("SELECT COUNT(*) FROM chapter_revisions"),
        }


def _config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    raw = {
        "model": args.model,
        "glossary_path": args.glossary,
        "lang_pair": args.lang_pair,
        "src_dir": args.src_dir,
        "tm_path": args.tm_path,
        "limit": args.limit,
        "max_pages": args.max_pages,
        "ollama_url": args.ollama_url,
    }
    return {k: v for k, v in raw.items() if v is not None}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="ragix-translate",
                                description="KOAS-Translate pipeline runner.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run translate stages over a workspace.")
    pr.add_argument("-w", "--workspace", required=True)
    pr.add_argument("--stages", default="all",
                    help=f"'all' or comma list of: {','.join(STAGE_NAMES)}")
    pr.add_argument("--model", help="LLM model (overrides per-stage defaults).")
    pr.add_argument("--glossary", help="Glossary CSV path (EN,FR,rule).")
    pr.add_argument("--lang-pair", default="en-fr")
    pr.add_argument("--src-dir", help="Source PDFs dir (extract).")
    pr.add_argument("--tm-path", help="Translation-memory SQLite path.")
    pr.add_argument("--limit", type=int, help="Cap new chunks translated (draft).")
    pr.add_argument("--max-pages", type=int, help="Cap pages extracted.")
    pr.add_argument("--ollama-url")

    ps = sub.add_parser("status", help="Show TM progress.")
    ps.add_argument("-w", "--workspace", required=True)
    ps.add_argument("--tm-path")

    args = p.parse_args(argv)

    if args.cmd == "status":
        st = status(args.workspace, args.tm_path)
        if not st["exists"]:
            print(f"[translate] no TM at {st['tm_path']}", file=sys.stderr)
            return 2
        print(f"segments={st['segments']}  translated={st['translated']}  "
              f"qa={st['qa']}  final={st['final']}  "
              f"chapter_revisions={st['chapter_revisions']}")
        return 0

    stages = STAGE_NAMES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    unknown = [s for s in stages if s not in STAGE_NAMES]
    if unknown:
        print(f"[translate] unknown stage(s): {unknown}; valid: {STAGE_NAMES}", file=sys.stderr)
        return 2

    results = run_pipeline(args.workspace, _config_from_args(args), stages)
    ok = True
    for r in results:
        mark = "✓" if r["success"] else "✗"
        print(f"  {mark} {r['summary']}")
        if not r["success"]:
            ok = False
            for e in r["errors"]:
                print(f"      {e}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
