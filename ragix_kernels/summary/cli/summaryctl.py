"""
summaryctl — CLI for the KOAS Summary kernel family.

Commands:
    ingest     Run Stage 1: corpus collection + rule extraction + memory storage
    summarize  Run Stage 2-3: budgeted recall + generation + verification + assembly
    show       Display workspace info (memory stats, domain coverage)
    run        Run full pipeline (ingest + summarize)
    drift      Cross-corpus drift detection
    viz        Generate static HTML visualizations (graph, memory explorer, drift report)
    query      Search memory items by text query (fulltext or scored)

Usage:
    python -m ragix_kernels.summary.cli.summaryctl ingest /path/to/corpus --scope grdf-rie -v
    python -m ragix_kernels.summary.cli.summaryctl summarize workspace/ --model ibm/granite4:32b-a9b-h -v
    python -m ragix_kernels.summary.cli.summaryctl run /path/to/corpus --scope grdf-rie --format md -v

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ANSI color helpers
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


def _cyan(text: str) -> str:
    return _c("36", text)


# ---------------------------------------------------------------------------
# Kernel map (lazy imports)
# ---------------------------------------------------------------------------

_KERNEL_MAP = {
    # Stage 1
    "summary_collect": (
        "ragix_kernels.summary.kernels.summary_collect",
        "SummaryCollectKernel",
    ),
    "summary_ingest": (
        "ragix_kernels.summary.kernels.summary_ingest",
        "SummaryIngestKernel",
    ),
    # Stage 2
    "summary_extract_entities": (
        "ragix_kernels.summary.kernels.summary_extract_entities",
        "SummaryExtractEntitiesKernel",
    ),
    "summary_build_graph": (
        "ragix_kernels.summary.kernels.summary_build_graph",
        "SummaryBuildGraphKernel",
    ),
    "summary_consolidate": (
        "ragix_kernels.summary.kernels.summary_consolidate",
        "SummaryConsolidateKernel",
    ),
    "summary_budgeted_recall": (
        "ragix_kernels.summary.kernels.summary_budgeted_recall",
        "SummaryBudgetedRecallKernel",
    ),
    "summary_drift": (
        "ragix_kernels.summary.kernels.summary_drift",
        "SummaryDriftKernel",
    ),
    # Stage 3
    "summary_generate": (
        "ragix_kernels.summary.kernels.summary_generate",
        "SummaryGenerateKernel",
    ),
    "summary_verify": (
        "ragix_kernels.summary.kernels.summary_verify",
        "SummaryVerifyKernel",
    ),
    "summary_redact": (
        "ragix_kernels.summary.kernels.summary_redact",
        "SummaryRedactKernel",
    ),
    "summary_capabilities": (
        "ragix_kernels.summary.kernels.summary_capabilities",
        "SummaryCapabilitiesKernel",
    ),
    "summary_report": (
        "ragix_kernels.summary.kernels.summary_report",
        "SummaryReportKernel",
    ),
}


def _get_summary_kernel(kernel_name: str):
    """Resolve a summary kernel by name using direct imports."""
    if kernel_name not in _KERNEL_MAP:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    mod_path, cls_name = _KERNEL_MAP[kernel_name]
    import importlib
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls()


def _run_kernel(
    kernel_name: str,
    workspace: Path,
    config: Dict[str, Any],
    dependencies: Optional[Dict[str, Path]] = None,
    verbose: bool = False,
):
    """Run a kernel and return its output."""
    from ragix_kernels.base import KernelInput

    kernel = _get_summary_kernel(kernel_name)
    if verbose:
        print(f"  {_cyan('RUN')} {kernel.name} v{kernel.version} (stage {kernel.stage})")

    ki = KernelInput(
        workspace=workspace,
        config=config,
        dependencies=dependencies or {},
    )
    output = kernel.run(ki)

    if verbose:
        status = _green("OK") if output.success else _red("FAIL")
        print(f"  {status} {kernel.name}: {output.summary}")
        if output.warnings:
            for w in output.warnings:
                print(f"    {_yellow('WARN')}: {w}")

    return output


def _discover_dependencies(workspace: Path, kernel_name: str) -> Dict[str, Path]:
    """Auto-discover dependency files from previous kernel outputs."""
    kernel = _get_summary_kernel(kernel_name)
    deps = {}
    for dep_name in kernel.requires:
        dep_kernel = _get_summary_kernel(dep_name)
        dep_file = workspace / f"stage{dep_kernel.stage}" / f"{dep_name}.json"
        if dep_file.exists():
            deps[dep_name] = dep_file
    return deps


def _resolve_fts_tokenizer(args) -> Optional[str]:
    """Resolve FTS tokenizer from CLI args. Returns None for default."""
    raw = getattr(args, "fts_tokenizer", None)
    if raw is None:
        return None
    from ragix_core.memory.store import FTS_TOKENIZER_PRESETS
    return FTS_TOKENIZER_PRESETS.get(raw, raw)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_ingest(args):
    """Run Stage 1: collection + ingestion."""
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    verbose = args.verbose

    delta_mode = getattr(args, "delta", False)

    corpus_id = getattr(args, "corpus_id", None)

    fts_tok = _resolve_fts_tokenizer(args)

    config = {
        "input_folder": str(Path(args.corpus).resolve()),
        "scope": args.scope,
        "model": args.model,
        "strategy": args.strategy,
        "max_chunk_tokens": args.max_chunk_tokens,
        "db_path": str(workspace / "memory.db"),
        "embedder_backend": args.embedder,
        "embedder_model": args.embedder_model,
        "ollama_url": args.ollama_url,
        "delta": delta_mode,
        "corpus_id": corpus_id,
    }
    if fts_tok:
        config["fts_tokenizer"] = fts_tok

    if verbose:
        print(_bold(f"\n=== KOAS Summary: Ingest ==="))
        print(f"  Corpus:    {args.corpus}")
        print(f"  Workspace: {workspace}")
        print(f"  Scope:     {args.scope}")
        print(f"  Model:     {args.model}")
        print(f"  Strategy:  {args.strategy}")
        if corpus_id:
            print(f"  Corpus ID: {_cyan(corpus_id)}")
        if delta_mode:
            print(f"  Delta:     {_cyan('enabled')} (only new/modified files)")
        if fts_tok:
            print(f"  FTS:       {_cyan(fts_tok)}")
        print()

    start = time.monotonic()

    # Stage 1a: Collect
    out_collect = _run_kernel("summary_collect", workspace, config, verbose=verbose)
    if not out_collect.success:
        print(_red(f"FAILED: {out_collect.errors}"))
        return 1

    # Stage 1b: Ingest
    out_ingest = _run_kernel("summary_ingest", workspace, config, verbose=verbose)
    if not out_ingest.success:
        print(_red(f"FAILED: {out_ingest.errors}"))
        return 1

    elapsed = time.monotonic() - start
    if verbose:
        print(f"\n{_green('Ingestion complete')} in {elapsed:.1f}s")
        ing = out_ingest.data.get("ingestion", {})
        st = out_ingest.data.get("store", {})
        print(f"  Files:    {ing.get('files_processed', 0)}")
        print(f"  Chunks:   {ing.get('chunks_processed', 0)}")
        print(f"  Accepted: {ing.get('items_accepted', 0)}")
        print(f"  Store:    {st.get('total_items', 0)} items")

    return 0


def cmd_summarize(args):
    """Run Stage 2-3: consolidation + recall + generation + verification + assembly."""
    workspace = Path(args.workspace)
    verbose = args.verbose

    db_path = str(workspace / "memory.db")
    if not Path(db_path).exists():
        print(_red(f"No memory.db in {workspace}. Run 'ingest' first."))
        return 1

    # Graph, secrecy, and delta config
    use_graph = getattr(args, "graph", True)
    secrecy_tier = getattr(args, "secrecy", "S3")
    delta_mode = getattr(args, "delta", False)

    fts_tok = _resolve_fts_tokenizer(args)

    config = {
        "scope": args.scope,
        "model": args.model,
        "db_path": db_path,
        "max_inject_tokens": args.max_tokens,
        "min_items_per_domain": args.min_per_domain,
        "max_items_per_domain": args.max_per_domain,
        "embedder_backend": args.embedder,
        "embedder_model": args.embedder_model,
        "ollama_url": args.ollama_url,
        "language": args.language,
        "graph": {"enabled": use_graph},
        "secrecy": {"tier": secrecy_tier},
        "delta": delta_mode,
        "domains": getattr(args, "domains", None),
    }
    if fts_tok:
        config["fts_tokenizer"] = fts_tok

    if verbose:
        print(_bold(f"\n=== KOAS Summary: Summarize ==="))
        print(f"  Workspace: {workspace}")
        print(f"  Scope:     {args.scope}")
        print(f"  Model:     {args.model}")
        print(f"  Graph:     {'enabled' if use_graph else 'disabled'}")
        print(f"  Secrecy:   {secrecy_tier}")
        if delta_mode:
            print(f"  Delta:     {_cyan('enabled')} (neighborhood-scoped consolidation)")
        if config.get("domains"):
            print(f"  Domains:   {', '.join(config['domains'])}")
        if fts_tok:
            print(f"  FTS:       {_cyan(fts_tok)}")
        print()

    start = time.monotonic()

    # Stage 2a: Extract entities (deterministic backfill)
    if use_graph and not args.skip_consolidation:
        deps_entities = _discover_dependencies(workspace, "summary_extract_entities")
        _run_kernel("summary_extract_entities", workspace, config, deps_entities, verbose=verbose)

    # Stage 2b: Build graph (optional)
    if use_graph and not args.skip_consolidation:
        deps_graph = _discover_dependencies(workspace, "summary_build_graph")
        _run_kernel("summary_build_graph", workspace, config, deps_graph, verbose=verbose)

    # Stage 2c: Consolidate (optional)
    if not args.skip_consolidation:
        # V3.0: Pass new_item_ids for delta consolidation
        if delta_mode:
            ingest_json = workspace / "stage1" / "summary_ingest.json"
            if ingest_json.exists():
                with open(ingest_json) as f:
                    ingest_data = json.load(f).get("data", {})
                config["new_item_ids"] = ingest_data.get("new_item_ids", [])
                if verbose and config["new_item_ids"]:
                    print(f"  {_cyan('DELTA')} {len(config['new_item_ids'])} new items for neighborhood consolidation")

        deps_cons = _discover_dependencies(workspace, "summary_consolidate")
        _run_kernel("summary_consolidate", workspace, config, deps_cons, verbose=verbose)

    # Stage 2d: Budgeted recall
    deps_recall = _discover_dependencies(workspace, "summary_budgeted_recall")
    out_recall = _run_kernel("summary_budgeted_recall", workspace, config, deps_recall, verbose=verbose)
    if not out_recall.success:
        print(_red(f"FAILED: {out_recall.errors}"))
        return 1

    # Stage 3a: Generate
    deps_gen = _discover_dependencies(workspace, "summary_generate")
    out_gen = _run_kernel(
        "summary_generate", workspace, config, deps_gen, verbose=verbose
    )
    if not out_gen.success:
        print(_red(f"FAILED: {out_gen.errors}"))
        return 1

    # Stage 3b: Verify
    deps_verify = _discover_dependencies(workspace, "summary_verify")
    out_verify = _run_kernel(
        "summary_verify", workspace, config, deps_verify, verbose=verbose
    )

    # Stage 3c: Report (assemble summary.md + summary.json)
    deps_report = _discover_dependencies(workspace, "summary_report")
    out_report = _run_kernel(
        "summary_report", workspace, config, deps_report, verbose=verbose
    )

    # Stage 3d: Redact (must run after report — operates on summary.md)
    if secrecy_tier != "S3":
        deps_redact = _discover_dependencies(workspace, "summary_redact")
        _run_kernel("summary_redact", workspace, config, deps_redact, verbose=verbose)

    # Stage 3e: Capabilities manifest
    deps_caps = _discover_dependencies(workspace, "summary_capabilities")
    _run_kernel("summary_capabilities", workspace, config, deps_caps, verbose=verbose)

    elapsed = time.monotonic() - start
    if verbose:
        print(f"\n{_green('Summary complete')} in {elapsed:.1f}s")
        if out_report.success:
            arts = out_report.data.get("artifacts", {})
            for name, path in arts.items():
                print(f"  {name}: {path}")

    return 0


def cmd_run(args):
    """Run full pipeline: ingest + summarize."""
    # Set workspace if not provided
    if not args.workspace:
        args.workspace = str(Path(args.corpus).parent / "summary_workspace")

    ret = cmd_ingest(args)
    if ret != 0:
        return ret

    return cmd_summarize(args)


def cmd_drift(args):
    """Run cross-corpus drift detection."""
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    verbose = args.verbose

    config = {
        "scope": args.scope,
    }

    # Determine comparison mode
    if args.corpus_a and args.corpus_b:
        # Same-DB mode: two corpus_ids in one DB
        config["corpus_a"] = args.corpus_a
        config["corpus_b"] = args.corpus_b
        config["db_path"] = str(workspace / "memory.db")
    elif args.workspace_a and args.workspace_b:
        # Two-workspace mode: separate DBs
        config["db_path_a"] = str(Path(args.workspace_a) / "memory.db")
        config["db_path_b"] = str(Path(args.workspace_b) / "memory.db")
        config["corpus_a"] = args.workspace_a
        config["corpus_b"] = args.workspace_b
    else:
        print(_red("Drift requires --corpus-a/--corpus-b OR --workspace-a/--workspace-b"))
        return 1

    if verbose:
        print(_bold(f"\n=== KOAS Summary: Drift Detection ==="))
        print(f"  A: {config.get('corpus_a') or config.get('db_path_a')}")
        print(f"  B: {config.get('corpus_b') or config.get('db_path_b')}")
        print(f"  Scope: {args.scope}")
        print()

    start = time.monotonic()
    out = _run_kernel("summary_drift", workspace, config, verbose=verbose)

    elapsed = time.monotonic() - start
    if verbose:
        if out.success:
            d = out.data
            print(f"\n{_green('Drift detection complete')} in {elapsed:.1f}s")
            print(f"  Added:     {d.get('added', 0)}")
            print(f"  Removed:   {d.get('removed', 0)}")
            print(f"  Modified:  {d.get('modified', 0)}")
            print(f"  Unchanged: {d.get('unchanged', 0)}")
            print(f"  Drift:     {d.get('drift_pct', 0)}%")
            if d.get("artifact"):
                print(f"  Report:    {d['artifact']}")
        else:
            print(_red(f"FAILED: {out.errors}"))
            return 1

    return 0


def cmd_viz(args):
    """Generate static HTML visualizations for a workspace."""
    workspace = Path(args.workspace)
    db_path = workspace / "memory.db"
    verbose = args.verbose

    if not db_path.exists():
        print(_red(f"No memory.db in {workspace}. Run 'ingest' first."))
        return 1

    from ragix_core.memory.store import MemoryStore
    from ragix_core.memory.graph_store import GraphStore
    from ragix_kernels.summary.visualization.render_html import render_all

    store = MemoryStore(str(db_path))
    graph_store = GraphStore(str(db_path))

    secrecy_tier = getattr(args, "secrecy", "S3")
    scope = getattr(args, "scope", None)
    corpus_id = getattr(args, "corpus_id", None)

    if verbose:
        print(_bold(f"\n=== KOAS Summary: Visualize ==="))
        print(f"  Workspace: {workspace}")
        print(f"  Secrecy:   {secrecy_tier}")
        if scope:
            print(f"  Scope:     {scope}")
        if corpus_id:
            print(f"  Corpus:    {corpus_id}")
        print()

    # V3.1: View selection
    selected_views = None
    views_arg = getattr(args, "views", None)
    if views_arg:
        selected_views = set(v.strip() for v in views_arg.split(","))
        if verbose:
            print(f"  Views:     {', '.join(sorted(selected_views))}")

    outputs = render_all(
        workspace=workspace,
        store=store,
        graph_store=graph_store,
        tier=secrecy_tier,
        scope=scope,
        corpus_id=corpus_id,
        views=selected_views,
    )

    if verbose:
        print(f"\n{_green('Visualization complete')}: {len(outputs)} files")
        for name, path in outputs.items():
            size = path.stat().st_size
            print(f"  {name}: {path} ({size:,} bytes)")
    else:
        for name, path in outputs.items():
            print(f"{name}: {path}")

    return 0


def cmd_query(args):
    """Search memory items by text query."""
    workspace = Path(args.workspace)
    db_path = workspace / "memory.db"
    verbose = args.verbose

    if not db_path.exists():
        print(_red(f"No memory.db in {workspace}. Run 'ingest' first."))
        return 1

    query_text = " ".join(args.query)
    if not query_text.strip():
        print(_red("Empty query. Provide search terms."))
        return 1

    from ragix_core.memory.store import MemoryStore
    store = MemoryStore(str(db_path))

    limit = args.limit
    tier = getattr(args, "tier", None)
    type_filter = getattr(args, "type", None)
    scope = getattr(args, "scope", None)
    domain_filter = getattr(args, "domain", None)
    use_scored = getattr(args, "scored", False)
    json_output = getattr(args, "json", False)

    if use_scored:
        # Hybrid scoring via RecallEngine
        from ragix_core.memory.recall import RecallEngine
        from ragix_core.memory.embedder import MockEmbedder, OllamaEmbedder

        embedder_backend = getattr(args, "embedder", "mock")
        if embedder_backend == "ollama":
            embedder = OllamaEmbedder()
        else:
            embedder = MockEmbedder()

        engine = RecallEngine(store=store, embedder=embedder)
        scored = engine.search_with_scores(
            query=query_text, limit=limit, tier=tier,
        )
        # scored is list of (MemoryItem, float)
        results = [(item, score) for item, score in scored]
        mode_label = "scored (RecallEngine)"
    else:
        # Fulltext search via SQL LIKE
        items = store.search_fulltext(
            query=query_text, tier=tier, type_filter=type_filter,
            scope=scope, limit=limit,
        )
        results = [(_item, _text_score(query_text, _item)) for _item in items]
        mode_label = "fulltext (SQL LIKE)"

    # Post-filter by domain
    if domain_filter:
        from ragix_kernels.summary.visualization.domain_utils import extract_domain
        results = [
            (item, score) for item, score in results
            if extract_domain(item) == domain_filter.lower()
        ]

    if json_output:
        import json as _json
        out = {
            "query": query_text,
            "mode": mode_label,
            "count": len(results),
            "results": [
                {
                    "score": round(score, 3),
                    "id": item.id,
                    "title": item.title,
                    "tier": item.tier,
                    "type": item.type,
                    "domain": _get_domain(item),
                    "tags": item.tags,
                    "content_preview": (item.content or "")[:200],
                }
                for item, score in results
            ],
        }
        print(_json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # Tabular output
    if verbose:
        print(f"\n{_bold('Query')}: {query_text}")
        print(f"{_bold('Mode')}:  {mode_label}")
        print(f"Found {_cyan(str(len(results)))} result(s)\n")

    if not results:
        print("  No results.")
        return 0

    # Table header
    print(f"  {'Score':>6}  {'Tier':<4}  {'Type':<14}  {'Domain':<16}  Title")
    print(f"  {'─'*6}  {'─'*4}  {'─'*14}  {'─'*16}  {'─'*40}")

    for item, score in results:
        domain = _get_domain(item)
        title = (item.title or "")[:50]
        tier_str = (item.tier or "").upper()
        print(f"  {score:6.3f}  {tier_str:<4}  {item.type:<14}  {domain:<16}  {title}")

    return 0


def _text_score(query: str, item) -> float:
    """Simple term-frequency relevance score (0.0 – 1.0)."""
    terms = query.lower().split()
    if not terms:
        return 0.0
    haystack = " ".join([
        item.title or "",
        item.content or "",
        " ".join(item.tags or []),
        " ".join(item.entities or []),
    ]).lower()
    hits = sum(1 for t in terms if t in haystack)
    return hits / len(terms)


def _get_domain(item) -> str:
    """Extract domain from item (lazy import to avoid circular deps)."""
    try:
        from ragix_kernels.summary.visualization.domain_utils import extract_domain
        return extract_domain(item)
    except Exception:
        return "unknown"


def cmd_show(args):
    """Display workspace info."""
    workspace = Path(args.workspace)
    db_path = workspace / "memory.db"
    output_format = getattr(args, "format", "table")

    if not db_path.exists():
        print(_red(f"No memory.db in {workspace}"))
        return 1

    from ragix_core.memory.store import MemoryStore
    store = MemoryStore(str(db_path))
    stats = store.stats()

    if output_format == "json":
        import json as _json
        result = {
            "workspace": str(workspace),
            "items": stats.get("total_items", 0),
            "by_tier": stats.get("by_tier", {}),
            "by_type": stats.get("by_type", {}),
            "events": stats.get("events_count", 0),
            "corpora": [],
            "stages": {},
        }
        for cm in store.list_corpora():
            result["corpora"].append({
                "corpus_id": cm.corpus_id,
                "item_count": cm.item_count,
                "doc_count": cm.doc_count,
                "parent": cm.parent_corpus_id,
            })
        for stage in (1, 2, 3):
            stage_dir = workspace / f"stage{stage}"
            if stage_dir.exists():
                result["stages"][f"stage{stage}"] = [f.name for f in stage_dir.glob("*.json")]
        summary_md = workspace / "stage3" / "summary.md"
        if summary_md.exists():
            result["report"] = {"path": str(summary_md), "size_bytes": summary_md.stat().st_size}
        print(_json.dumps(result, indent=2))
        return 0

    print(_bold(f"\n=== KOAS Summary: Workspace ==="))
    print(f"  Path:     {workspace}")
    print(f"  Items:    {stats.get('total_items', 0)}")
    print(f"  By tier:  {stats.get('by_tier', {})}")
    print(f"  By type:  {stats.get('by_type', {})}")
    print(f"  Events:   {stats.get('events_count', 0)}")

    # V3.0: Show registered corpora
    corpora = store.list_corpora()
    if corpora:
        print(f"\n  {_bold('Corpora')} ({len(corpora)} registered):")
        for cm in corpora:
            parent = f" (parent: {cm.parent_corpus_id})" if cm.parent_corpus_id else ""
            print(f"    {cm.corpus_id}: {cm.item_count} items, {cm.doc_count} docs{parent}")

    # Check for stage outputs
    for stage in (1, 2, 3):
        stage_dir = workspace / f"stage{stage}"
        if stage_dir.exists():
            files = list(stage_dir.glob("*.json"))
            print(f"  Stage {stage}: {len(files)} outputs")

    # Check for final report
    summary_md = workspace / "stage3" / "summary.md"
    if summary_md.exists():
        size = summary_md.stat().st_size
        print(f"\n  {_green('Report')}: {summary_md} ({size:,} bytes)")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build argparse parser with all summaryctl subcommands."""
    parser = argparse.ArgumentParser(
        prog="summaryctl",
        description="KOAS Summary — Balanced, citeable technical summaries",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- ingest --
    p_ingest = sub.add_parser("ingest", help="Ingest corpus into memory")
    p_ingest.add_argument("corpus", help="Path to document folder")
    p_ingest.add_argument("-w", "--workspace", default=None,
                          help="Workspace directory (default: corpus_parent/summary_workspace)")
    p_ingest.add_argument("--scope", default="project", help="Memory scope label")
    p_ingest.add_argument("--model", default="ibm/granite4:32b-a9b-h",
                          help="LLM model for rule extraction")
    p_ingest.add_argument("--strategy", default="pages",
                          choices=["pages", "headings", "windows"],
                          help="Chunking strategy")
    p_ingest.add_argument("--max-chunk-tokens", type=int, default=800,
                          help="Max tokens per chunk")
    p_ingest.add_argument("--embedder", default="mock",
                          choices=["mock", "ollama", "sentence-transformers"],
                          help="Embedding backend")
    p_ingest.add_argument("--embedder-model", default="nomic-embed-text",
                          help="Embedding model name")
    p_ingest.add_argument("--ollama-url", default="http://localhost:11434",
                          help="Ollama API URL")
    p_ingest.add_argument("--fts-tokenizer", default=None,
                          help="FTS5 tokenizer: fr (accent-insensitive), "
                               "en (porter stemming), raw (unicode61), "
                               "or a raw tokenizer string")
    p_ingest.add_argument("--delta", action="store_true",
                          help="Delta mode: only process new/modified files")
    p_ingest.add_argument("--corpus-id", default=None,
                          help="Corpus version ID (e.g. grdf-rie-2026Q1) for cross-corpus tracking")
    p_ingest.add_argument("-v", "--verbose", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    # -- summarize --
    p_sum = sub.add_parser("summarize", help="Generate summary from memory")
    p_sum.add_argument("workspace", help="Workspace directory with memory.db")
    p_sum.add_argument("--scope", default="project", help="Memory scope")
    p_sum.add_argument("--model", default="ibm/granite4:32b-a9b-h",
                       help="LLM model for summary generation")
    p_sum.add_argument("--max-tokens", type=int, default=12000,
                       help="Max injection tokens budget")
    p_sum.add_argument("--min-per-domain", type=int, default=3,
                       help="Min items per domain")
    p_sum.add_argument("--max-per-domain", type=int, default=25,
                       help="Max items per domain")
    p_sum.add_argument("--embedder", default="mock",
                       choices=["mock", "ollama", "sentence-transformers"])
    p_sum.add_argument("--embedder-model", default="nomic-embed-text")
    p_sum.add_argument("--ollama-url", default="http://localhost:11434")
    p_sum.add_argument("--fts-tokenizer", default=None,
                       help="FTS5 tokenizer: fr/en/raw or raw string")
    p_sum.add_argument("--language", default="French",
                       help="Summary output language")
    p_sum.add_argument("--skip-consolidation", action="store_true",
                       help="Skip consolidation step")
    p_sum.add_argument("--graph", dest="graph", action="store_true", default=True,
                       help="Enable graph-assisted consolidation (default)")
    p_sum.add_argument("--no-graph", dest="graph", action="store_false",
                       help="Disable graph-assisted consolidation")
    p_sum.add_argument("--secrecy", choices=["S0", "S2", "S3"], default="S3",
                       help="Secrecy tier for report redaction (default: S3=full detail)")
    p_sum.add_argument("--delta", action="store_true",
                       help="Delta mode: neighborhood-scoped consolidation only")
    p_sum.add_argument("--domains", nargs="+", default=None,
                       help="Only regenerate specified domains (e.g. --domains oracle rhel)")
    p_sum.add_argument("--corpus-id", default=None,
                       help="Corpus version ID for filtering")
    p_sum.add_argument("-v", "--verbose", action="store_true")
    p_sum.set_defaults(func=cmd_summarize)

    # -- run (full pipeline) --
    p_run = sub.add_parser("run", help="Full pipeline: ingest + summarize")
    p_run.add_argument("corpus", help="Path to document folder")
    p_run.add_argument("-w", "--workspace", default=None)
    p_run.add_argument("--scope", default="project")
    p_run.add_argument("--model", default="ibm/granite4:32b-a9b-h")
    p_run.add_argument("--strategy", default="pages")
    p_run.add_argument("--max-chunk-tokens", type=int, default=800)
    p_run.add_argument("--max-tokens", type=int, default=12000)
    p_run.add_argument("--min-per-domain", type=int, default=3)
    p_run.add_argument("--max-per-domain", type=int, default=25)
    p_run.add_argument("--embedder", default="mock")
    p_run.add_argument("--embedder-model", default="nomic-embed-text")
    p_run.add_argument("--ollama-url", default="http://localhost:11434")
    p_run.add_argument("--fts-tokenizer", default=None,
                       help="FTS5 tokenizer: fr/en/raw or raw string")
    p_run.add_argument("--language", default="French")
    p_run.add_argument("--skip-consolidation", action="store_true")
    p_run.add_argument("--graph", dest="graph", action="store_true", default=True,
                       help="Enable graph-assisted consolidation (default)")
    p_run.add_argument("--no-graph", dest="graph", action="store_false",
                       help="Disable graph-assisted consolidation")
    p_run.add_argument("--secrecy", choices=["S0", "S2", "S3"], default="S3",
                       help="Secrecy tier for report redaction")
    p_run.add_argument("--delta", action="store_true",
                       help="Delta mode: process only new/modified files")
    p_run.add_argument("--domains", nargs="+", default=None,
                       help="Only regenerate specified domains (e.g. --domains oracle rhel)")
    p_run.add_argument("--corpus-id", default=None,
                       help="Corpus version ID (e.g. grdf-rie-2026Q1) for cross-corpus tracking")
    p_run.add_argument("-v", "--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # -- drift --
    p_drift = sub.add_parser("drift", help="Cross-corpus drift detection")
    p_drift.add_argument("workspace", help="Output workspace directory")
    p_drift.add_argument("--corpus-a", default=None,
                         help="Baseline corpus ID (same-DB mode)")
    p_drift.add_argument("--corpus-b", default=None,
                         help="Current corpus ID (same-DB mode)")
    p_drift.add_argument("--workspace-a", default=None,
                         help="Baseline workspace path (two-workspace mode)")
    p_drift.add_argument("--workspace-b", default=None,
                         help="Current workspace path (two-workspace mode)")
    p_drift.add_argument("--scope", default="project", help="Memory scope")
    p_drift.add_argument("-v", "--verbose", action="store_true")
    p_drift.set_defaults(func=cmd_drift)

    # -- viz --
    p_viz = sub.add_parser("viz", help="Generate HTML visualizations (graph, memory, drift)")
    p_viz.add_argument("workspace", help="Workspace directory with memory.db")
    p_viz.add_argument("--scope", default=None, help="Filter by memory scope")
    p_viz.add_argument("--corpus-id", default=None, help="Filter by corpus ID")
    p_viz.add_argument("--secrecy", choices=["S0", "S2", "S3"], default="S3",
                       help="Secrecy tier for visualization (default: S3=full detail)")
    p_viz.add_argument("--views", default=None,
                       help="Comma-separated views to render (default: all). "
                            "Valid: graph,memory,query,drift,timeline,clusters,heatmap,metrics")
    p_viz.add_argument("-v", "--verbose", action="store_true")
    p_viz.set_defaults(func=cmd_viz)

    # -- query --
    p_query = sub.add_parser("query", help="Search memory items by text query")
    p_query.add_argument("workspace", help="Workspace directory with memory.db")
    p_query.add_argument("query", nargs="+", help="Search terms")
    p_query.add_argument("--tier", default=None, choices=["stm", "mtm", "ltm"],
                         help="Filter by tier")
    p_query.add_argument("--type", default=None, help="Filter by item type")
    p_query.add_argument("--scope", default=None, help="Filter by scope")
    p_query.add_argument("--domain", default=None,
                         help="Filter by document domain (e.g. rhel, oracle)")
    p_query.add_argument("-k", "--limit", type=int, default=20,
                         help="Max results (default: 20)")
    p_query.add_argument("--scored", action="store_true",
                         help="Use hybrid RecallEngine scoring (tags+embeddings)")
    p_query.add_argument("--embedder", default="mock",
                         choices=["mock", "ollama"],
                         help="Embedding backend for --scored mode")
    p_query.add_argument("--json", action="store_true",
                         help="Machine-readable JSON output")
    p_query.add_argument("-v", "--verbose", action="store_true")
    p_query.set_defaults(func=cmd_query)

    # -- show --
    p_show = sub.add_parser("show", help="Show workspace info")
    p_show.add_argument("workspace", help="Workspace directory")
    p_show.add_argument("--format", choices=["table", "json"], default="table",
                        help="Output format (default: table)")
    p_show.set_defaults(func=cmd_show)

    return parser


def main():
    """Parse arguments, configure logging, and dispatch to subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Fix workspace default for ingest/run
    if args.command in ("ingest", "run") and args.workspace is None:
        args.workspace = str(Path(args.corpus).resolve().parent / "summary_workspace")

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
