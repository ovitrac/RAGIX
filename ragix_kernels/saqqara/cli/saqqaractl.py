"""
saqqaractl — CLI for the saqqara document-substrate kernel.

Commands:
    run     Read documents into typed trees and recognise their structure
    status  Read back what a previous run found, without re-reading the corpus
    index   Chunk what was read into a store and embed what is missing
    search  Query that store, showing both lane ranks and every hit's citation

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
from ragix_kernels.saqqara.kernels.saqqara_index import SaqqaraIndexKernel
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

    **Read from `report.abstentions`, never re-derived (K4.3).** This function
    used to search each trace for a key named `abstentions`, which only
    `header_bands` writes — so it listed that analyzer's abstentions and showed
    none of `grid_tables`', while the summary line counted them all. Two surfaces
    deriving the same fact by two rules is two surfaces that disagree; the
    register is the one place both now read.
    """
    register = (data.get("report") or {}).get("abstentions")
    if register is None:
        return []
    return [(entry.get("path", "?"), entry.get("analyzer", "?"),
             entry.get("reason") or "unstated")
            for entry in register]


#: Report keys this reporter knows how to render. Anything else is printed raw
#: rather than ignored: a reporter that quietly skips a key it does not recognise
#: is how a new count becomes invisible, and the first version of this function
#: guessed "refused"/"dropped" and silently printed nothing for either.
_KNOWN_REPORT_KEYS = ("counts", "refusals", "duplicates", "abstentions")


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


def cmd_index(args) -> int:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    config: Dict[str, Any] = {}
    if args.config:
        config["config"] = args.config

    read = workspace / "stage1" / "saqqara.json"
    if not read.is_file():
        print(_red(f"nothing has been read into {workspace}; run `run` first"),
              file=sys.stderr)
        return 2
    output = SaqqaraIndexKernel().run(KernelInput(
        workspace=workspace, config=config, dependencies={"document_tree": read}))

    if args.json:
        print(json.dumps(output.data, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if output.success else 1
    if not output.success:
        print(_red(output.summary), file=sys.stderr)
        for error in output.errors or []:
            print(_red(f"error: {error}"), file=sys.stderr)
        return 1

    print(_green(output.summary))
    status = (output.data or {}).get("status", {})
    for key in ("documents", "chunks", "objects", "edges", "embeddings", "embeddings_parked"):
        print(f"  {key:18s} {status.get(key, 0)}")
    print(f"  {'dense':18s} {status.get('dense', 'unknown')}")
    for drop in status.get("drops", []):
        print(_yellow(f"  dropped: {drop}"))
    return 0


def _hit_payload(hit, tree) -> Dict[str, Any]:
    """One hit as data — the same shape the MCP tool returns (K7.16)."""
    from ragix_kernels.saqqara.store.retrieve import provenance_of

    payload = hit.to_dict()
    payload["provenance"] = provenance_of(tree, hit.chunk) if tree else []
    return payload


def cmd_search(args) -> int:
    from ragix_kernels.saqqara.store.config import load_config
    from ragix_kernels.saqqara.store.embed import build_embedder
    from ragix_kernels.saqqara.store.ports import build_store
    from ragix_kernels.saqqara.store.retrieve import Retriever

    config = load_config(args.config or None)
    store_section = config.section("store")
    path = Path(store_section.get("path", ".ragix/saqqara.db"))
    if not path.is_absolute():
        path = Path(args.workspace) / path
    if not path.is_file():
        print(_red(f"no store at {path}; run `index` first"), file=sys.stderr)
        return 2
    store = build_store({**store_section, "path": str(path)})

    embedder_section = config.section("embedder")
    provider = embedder_section.get("provider", "none")
    model = embedder_section.get("model", "") or provider
    embedder = build_embedder(provider, model=embedder_section.get("model", ""))
    vector = embedder.embed_batch([args.query])[0] if embedder else None

    retrieval = config.section("retrieval")
    hits = Retriever(store, model=model if embedder else "",
                     backend=config.get("index.backend", "numpy"),
                     rrf_k=retrieval.get("rrf_k", 60)).search(
        query=args.query, vector=vector,
        top_k=args.top_k or retrieval.get("top_k", 10),
        dense_k=retrieval.get("dense_k", 40), lexical_k=retrieval.get("lexical_k", 40))

    trees = {d.doc_id: d.tree for d in store.list_documents()}
    payload = [_hit_payload(h, trees.get(h.chunk.doc_id)) for h in hits]

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    if embedder is None:
        print(_yellow("dense: disabled (no embedder) — lexical lane only"))
    if not hits:
        print("no hit")
        return 0
    for hit in hits:
        ranks = f"dense={hit.dense_rank} lexical={hit.lexical_rank} final={hit.final_rank}"
        print(f"\n{_bold(ranks)}")
        print(f"  {hit.chunk.text[:160]}")
        for entry in _hit_payload(hit, trees.get(hit.chunk.doc_id))["provenance"]:
            print(f"    {entry['kind']:10s} {entry['source_path']} {entry['chain']}")
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

    p_index = sub.add_parser("index", help="chunk what was read into a store")
    p_index.add_argument("workspace", help="the workspace a previous run wrote to")
    p_index.add_argument("-c", "--config", default=None, help="path to a saqqara.yaml")
    p_index.add_argument("--json", action="store_true", help="emit the raw result as JSON")
    p_index.add_argument("-v", "--verbose", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="query the store")
    p_search.add_argument("workspace", help="the workspace holding the store")
    p_search.add_argument("query", help="what to look for")
    p_search.add_argument("-c", "--config", default=None, help="path to a saqqara.yaml")
    p_search.add_argument("-k", "--top-k", type=int, default=0, help="how many hits")
    p_search.add_argument("--json", action="store_true", help="emit hits as JSON")
    p_search.add_argument("-v", "--verbose", action="store_true")
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
