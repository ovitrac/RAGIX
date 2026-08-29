"""
saqqaractl — CLI for the saqqara document-substrate kernel.

Commands:
    run     Read documents into typed trees and recognise their structure
    status  Read back what a previous run found, without re-reading the corpus

Usage:
    python -m ragix_kernels.saqqara.cli.saqqaractl run ./documents -w ./work
    python -m ragix_kernels.saqqara.cli.saqqaractl run ./documents --formats .docx,.xlsx
    python -m ragix_kernels.saqqara.cli.saqqaractl status ./work

The two commands mirror the MCP surface exactly (koas_saqqara_run /
koas_saqqara_status): same configuration keys, same stored result, same roots. A
CLI that answered differently from the tool would be a second implementation of
the envelope, and the first divergence would show up as a citation nobody could
reproduce.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from ragix_kernels.base import KernelInput
from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel

# ANSI colour helpers — silent when the output is not a terminal.
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(text: str) -> str:
    return _c("1", text)


def _green(text: str) -> str:
    return _c("32", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _red(text: str) -> str:
    return _c("31", text)


def _abstentions(data: Dict[str, Any]) -> list[tuple[str, str, str]]:
    """Every abstention in a result, as (document, analyzer, reason).

    Abstention is an object here, not a missing value, so it is reported rather
    than counted: a run that decided nothing about forty documents and a run that
    decided everything both have the same number of documents.
    """
    found = []
    for document in data.get("documents", []) or []:
        for name, trace in (document.get("traces") or {}).items():
            if not isinstance(trace, dict):
                continue
            for entry in trace.get("abstentions", []) or []:
                reason = entry.get("reason", "unstated") if isinstance(entry, dict) else str(entry)
                found.append((document.get("path", "?"), name, reason))
    return found


#: Report keys this reporter knows how to render. Anything else is printed raw
#: rather than ignored: a reporter that quietly skips a key it does not recognise
#: is how a new count becomes invisible, and the first version of this function
#: guessed "refused"/"dropped" and silently printed nothing for either.
_KNOWN_REPORT_KEYS = ("counts", "refusals", "duplicates")


def _drops(data: Dict[str, Any]) -> list[tuple[str, str, int]]:
    """Every counted drop, as (document, analyzer, count). Drops live in traces."""
    found = []
    for document in data.get("documents", []) or []:
        for name, trace in (document.get("traces") or {}).items():
            if isinstance(trace, dict) and trace.get("dropped"):
                found.append((document.get("path", "?"), name, int(trace["dropped"])))
    return found


def _report(data: Dict[str, Any], verbose: bool) -> None:
    documents = data.get("documents", []) or []
    report = data.get("report") or {}
    counts = report.get("counts") or {}

    print(
        f"{_bold('read')} {counts.get('read', len(documents))}  "
        f"{_bold('refused')} {counts.get('refused', 0)}  "
        f"{_bold('duplicate')} {counts.get('duplicate', 0)}"
    )

    for document in documents:
        nodes = SaqqaraKernel._count(document["tree"]["root"])
        print(f"  {document['format']:6s} {nodes:6d} nodes  {document['path']}")

    refusals = report.get("refusals") or []
    if refusals:
        print(f"\n{_yellow('refusals')}  {len(refusals)}")
        for entry in refusals:
            print(f"  {entry.get('reason', 'unstated')}: {entry.get('path', '?')}")

    duplicates = report.get("duplicates") or []
    if duplicates:
        print(f"\n{_yellow('duplicates')}  {len(duplicates)}")
        for entry in duplicates:
            print(f"  {entry}")

    drops = _drops(data)
    if drops:
        print(f"\n{_yellow('drops')}  {sum(n for _, _, n in drops)}")
        for path, analyzer, n in drops:
            print(f"  {analyzer}: {n}  ({path})")

    abstentions = _abstentions(data)
    if abstentions:
        print(f"\n{_bold('abstentions')}  {len(abstentions)}")
        shown = abstentions if verbose else abstentions[:10]
        for path, analyzer, reason in shown:
            print(f"  {analyzer}: {reason}  ({path})")
        if len(shown) < len(abstentions):
            print(f"  … {len(abstentions) - len(shown)} more (use -v)")

    unknown = [k for k in report if k not in _KNOWN_REPORT_KEYS]
    if unknown:
        print(f"\n{_yellow('report keys this CLI does not render')}")
        for key in sorted(unknown):
            print(f"  {key} = {json.dumps(report[key], ensure_ascii=False)[:200]}")

    print(f"\n{_bold('merkle_root')}  {data.get('merkle_root')}")
    print(f"{_bold('source_root')}  {data.get('source_root')}")


def cmd_run(args) -> int:
    source = Path(args.source)
    if not source.exists():
        print(_red(f"no such source: {source}"), file=sys.stderr)
        return 2

    workspace = Path(args.workspace) if args.workspace else source.resolve().parent
    # The envelope validates that the workspace exists and refuses otherwise. On a
    # CLI that refusal is unhelpful: `-w ./work` on a fresh directory is the normal
    # first invocation, and failing it teaches nothing about the documents.
    workspace.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {"source": {"path": str(source)}}
    if args.formats:
        config["formats"] = [f.strip() for f in args.formats.split(",") if f.strip()]
    if args.promote_outline:
        config["promote_outline"] = True

    output = SaqqaraKernel().run(KernelInput(workspace=workspace, config=config))

    if args.json:
        print(json.dumps(output.data, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if output.success else 1

    if not output.success:
        # Do not print a report for a run that did not happen: "documents 0" and
        # two null roots read as a finding about the corpus rather than as a
        # failure to read it.
        print(_red(output.summary), file=sys.stderr)
        for error in output.errors or []:
            print(_red(f"error: {error}"), file=sys.stderr)
        return 1

    print(_green(output.summary))
    _report(output.data or {}, args.verbose)
    print(f"\nwritten to {output.output_file}")
    return 0


def cmd_status(args) -> int:
    stored = Path(args.workspace) / "stage1" / "saqqara.json"
    if not stored.is_file():
        print(_red(f"no saqqara result in {args.workspace}"), file=sys.stderr)
        return 2

    data = json.loads(stored.read_text(encoding="utf-8")).get("data", {})
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    _report(data, args.verbose)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saqqaractl",
        description="Read documents into typed trees with provenance on every node.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="read a file or directory into typed trees")
    p_run.add_argument("source", help="file or directory to read")
    p_run.add_argument("-w", "--workspace", default=None,
                       help="where to write the result (default: beside the source)")
    p_run.add_argument("--formats", default="",
                       help="comma-separated extensions to narrow the scan, e.g. .docx,.xlsx")
    p_run.add_argument("--promote-outline", action="store_true",
                       help="also run the opt-in typed-outline pass, which ADDS inferred "
                            "headings; off by default because a promotion is an inference")
    p_run.add_argument("--json", action="store_true", help="emit the raw result as JSON")
    p_run.add_argument("-v", "--verbose", action="store_true", help="list every abstention")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="read back a previous run")
    p_status.add_argument("workspace", help="the workspace a previous run wrote to")
    p_status.add_argument("--json", action="store_true", help="emit the raw result as JSON")
    p_status.add_argument("-v", "--verbose", action="store_true", help="list every abstention")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
