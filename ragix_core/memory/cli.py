"""
Memory CLI Utilities — Unix-Composable Memory Commands

Commands:
    ragix-memory init [path]                      Create memory workspace
    ragix-memory push "query" --source files...   Ingest + recall (alias: pipe)
    ragix-memory pull --tags t --title "T"        Store LLM output from stdin
    ragix-memory search "query" --tier ltm --k 10
    ragix-memory show <id>
    ragix-memory stats
    ragix-memory consolidate --scope project
    ragix-memory export --format jsonl
    ragix-memory import memory.jsonl
    ragix-memory report -w <workspace> -s <scenario> [-o report.md]
    ragix-memory recall "query" --budget 1500 -w <workspace>
    ragix-memory ingest --source file1.md file2.md --tags doc -w default
    ragix-memory pipe "query" --source files... --budget 2000
    ragix-memory serve --fts-tokenizer fr         Start MCP server

Environment variables:
    RAGIX_MEMORY_DB      Path to SQLite database (overrides last-db cache)
    RAGIX_MEMORY_BUDGET  Token budget default for pipe/recall

Usage:
    ragix-memory <command> [args]

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-18
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Parse integer env var with fallback. Never raises on bad input."""
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default

_LAST_DB_FILE = Path.home() / ".cache" / "ragix" / "last_memory_db"


def _save_last_db(db_path: str) -> None:
    """Persist the last-used DB path for implicit reuse."""
    try:
        _LAST_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_DB_FILE.write_text(str(Path(db_path).resolve()), encoding="utf-8")
    except OSError:
        pass  # best-effort


def _load_last_db() -> Optional[str]:
    """Return last-used DB path, or None if unavailable."""
    try:
        text = _LAST_DB_FILE.read_text(encoding="utf-8").strip()
        if text and Path(text).exists():
            return text
    except OSError:
        pass
    return None


def _get_dispatcher(db_path: str = "memory.db"):
    """Create a MemoryToolDispatcher with default config."""
    from ragix_core.memory.config import MemoryConfig, StoreConfig, EmbedderConfig
    from ragix_core.memory.tools import create_dispatcher

    config = MemoryConfig(
        store=StoreConfig(db_path=db_path),
        embedder=EmbedderConfig(backend="mock"),  # CLI default: mock embedder
    )
    return create_dispatcher(config)


def cmd_search(args: argparse.Namespace) -> None:
    """Search memory items."""
    dispatcher = _get_dispatcher(args.db)
    result = dispatcher.dispatch("search", {
        "query": args.query,
        "tier": args.tier,
        "type": args.type,
        "k": args.k,
    })

    if result["status"] != "ok":
        print(f"Error: {result.get('message', 'unknown')}", file=sys.stderr)
        return

    items = result.get("items", [])
    if not items:
        print("No results found.")
        return

    print(f"Found {len(items)} item(s):\n")
    for item in items:
        print(f"  [{item['tier'].upper()}] {item['id']}  {item['type']:12s}  {item['title']}")
        if item.get("tags"):
            print(f"    tags: {', '.join(item['tags'])}")
        print()


def cmd_show(args: argparse.Namespace) -> None:
    """Show a memory item by ID."""
    dispatcher = _get_dispatcher(args.db)
    result = dispatcher.dispatch("read", {"ids": [args.id]})

    if result["status"] != "ok" or not result.get("items"):
        print(f"Item not found: {args.id}", file=sys.stderr)
        return

    item = result["items"][0]
    print(json.dumps(item, indent=2, ensure_ascii=False))


def cmd_stats(args: argparse.Namespace) -> None:
    """Show memory store statistics."""
    dispatcher = _get_dispatcher(args.db)
    stats = dispatcher.store.stats()

    print("Memory Store Statistics")
    print("=" * 40)
    print(f"  Total items (active): {stats['total_items']}")
    print(f"  By tier:")
    for tier, count in sorted(stats.get("by_tier", {}).items()):
        print(f"    {tier.upper():4s}: {count}")
    print(f"  By type:")
    for typ, count in sorted(stats.get("by_type", {}).items()):
        print(f"    {typ:12s}: {count}")
    print(f"  Embeddings:  {stats['embeddings_count']}")
    print(f"  Audit events: {stats['events_count']}")

    # V3.0: Show registered corpora
    corpora = dispatcher.store.list_corpora()
    if corpora:
        print(f"\n  Corpora ({len(corpora)} registered):")
        for cm in corpora:
            parent = f" <- {cm.parent_corpus_id}" if cm.parent_corpus_id else ""
            print(f"    {cm.corpus_id}: {cm.item_count} items, {cm.doc_count} docs{parent}")


def cmd_consolidate(args: argparse.Namespace) -> None:
    """Run consolidation pipeline."""
    dispatcher = _get_dispatcher(args.db)
    result = dispatcher.dispatch("consolidate", {
        "scope": args.scope,
        "tiers": args.tiers.split(",") if args.tiers else ["stm"],
        "promote": not args.no_promote,
    })

    if result["status"] != "ok":
        print(f"Error: {result.get('message', 'unknown')}", file=sys.stderr)
        return

    print("Consolidation complete:")
    for key, val in result.items():
        if key != "status":
            print(f"  {key}: {val}")


def cmd_export(args: argparse.Namespace) -> None:
    """Export memory items to JSONL, optionally applying secrecy-tier redaction."""
    dispatcher = _get_dispatcher(args.db)
    tier = getattr(args, "tier", "S3") or "S3"

    if tier == "S3":
        # No redaction — direct export
        data = dispatcher.store.export_jsonl()
    else:
        # Apply redaction before export
        from ragix_kernels.summary.kernels.summary_redact import (
            redact_for_storage,
        )
        items = dispatcher.store.list_items(exclude_archived=False, limit=100000)
        redacted_count = 0
        lines = []
        for item in items:
            d = item.to_dict()
            orig_title = d.get("title", "")
            orig_content = d.get("content", "")
            d["title"] = redact_for_storage(orig_title, tier)
            d["content"] = redact_for_storage(orig_content, tier)
            if d["title"] != orig_title or d["content"] != orig_content:
                redacted_count += 1
            lines.append(json.dumps(d, ensure_ascii=False))
        data = "\n".join(lines)
        if args.verbose if hasattr(args, "verbose") else False:
            print(f"Redacted {redacted_count}/{len(items)} items at tier {tier}")

    if args.output:
        Path(args.output).write_text(data, encoding="utf-8")
        print(f"Exported to {args.output}" + (f" (tier {tier})" if tier != "S3" else ""))
    else:
        print(data)


def cmd_import(args: argparse.Namespace) -> None:
    """Import memory items from JSONL."""
    dispatcher = _get_dispatcher(args.db)
    data = Path(args.file).read_text(encoding="utf-8")
    count = dispatcher.store.import_jsonl(data)
    print(f"Imported {count} item(s)")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate a memory report using a named scenario."""
    from ragix_core.memory.reporting.api import generate_report, list_scenarios
    from ragix_core.memory.reporting.io import parse_set_args

    # List mode
    if getattr(args, "list_scenarios", False):
        for name in list_scenarios():
            print(f"  {name}")
        return

    if not args.workspace:
        print("Error: --workspace is required for report generation", file=sys.stderr)
        sys.exit(1)

    scenario = getattr(args, "scenario", None) or "summarize_content"

    # Parse --set overrides
    overrides = parse_set_args(args.set) if args.set else None

    try:
        md = generate_report(
            db_path=args.db,
            workspace=args.workspace,
            scenario=scenario,
            config_path=args.config,
            out_path=args.output,
            overrides=overrides,
            embedder=args.embedder or "mock",
            scope=args.scope or "audit",
            corpus_id=args.corpus,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Report failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print to stdout if no --output
    if not args.output:
        print(md)
    else:
        lines = md.count("\n") + 1
        print(f"Report written: {args.output} ({lines} lines)")


def cmd_recall(args: argparse.Namespace) -> None:
    """Token-budgeted recall — produces an injection block on stdout."""
    dispatcher = _get_dispatcher(args.db)

    # Workspace resolution
    scope, corpus_id = None, None
    if args.workspace:
        from ragix_core.memory.mcp.workspace import WorkspaceRouter
        router = WorkspaceRouter(dispatcher.store._conn)
        try:
            scope, corpus_id = router.resolve(args.workspace)
        except KeyError:
            print(f"Error: unknown workspace {args.workspace!r}", file=sys.stderr)
            sys.exit(1)

    # Search (large k, then budget-trim in formatting)
    params: dict = {"query": args.query, "k": 50}
    if args.tier:
        params["tier"] = args.tier
    if scope:
        params["scope"] = scope
    result = dispatcher.dispatch("search", params)

    if result["status"] != "ok":
        print(f"Error: {result.get('message', 'unknown')}", file=sys.stderr)
        sys.exit(1)

    items = result.get("items", [])
    if not items:
        print("No matching items.", file=sys.stderr)
        sys.exit(0)

    # Enrich with full content
    ids = [it["id"] for it in items]
    full = dispatcher.dispatch("read", {"ids": ids})
    if full["status"] == "ok":
        items = full.get("items", items)

    # Filter non-injectable
    items = [it for it in items if it.get("injectable", True)]

    if not items:
        print("No injectable items matched.", file=sys.stderr)
        sys.exit(0)

    # Format injection block
    from ragix_core.memory.mcp.formatting import format_injection_block
    text = format_injection_block(items, args.budget, len(items))

    # Optional: strip header
    if args.no_header:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if not line.strip() and i > 0:
                text = "\n".join(lines[i + 1:])
                break

    print(text)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest files (or stdin) into memory."""
    from ragix_core.memory.ingest import ingest_file, ingest_stdin, IngestResult, resolve_sources

    dispatcher = _get_dispatcher(args.db)
    store = dispatcher.store
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    total = IngestResult()

    if args.source == ["-"]:
        # Stdin mode
        total = ingest_stdin(
            store,
            workspace=args.workspace or "default",
            scope=args.scope,
            corpus_id=args.corpus,
            max_tokens=args.chunk_tokens,
            tags=tags,
            injectable=args.injectable,
        )
    else:
        for path in resolve_sources(args.source):
            r = ingest_file(
                store,
                path,
                workspace=args.workspace or "default",
                scope=args.scope,
                corpus_id=args.corpus,
                max_tokens=args.chunk_tokens,
                tags=tags,
                format_mode=args.format,
                injectable=args.injectable,
            )
            total.files_processed += r.files_processed
            total.files_skipped += r.files_skipped
            total.chunks_created += r.chunks_created
            total.item_ids.extend(r.item_ids)

    print(
        f"Ingested {total.chunks_created} chunks from "
        f"{total.files_processed} file(s) "
        f"({total.files_skipped} skipped, already in store)"
    )


def cmd_pipe(args: argparse.Namespace) -> None:
    """One-shot: ingest sources (if provided) + recall + print injection block."""
    from ragix_core.memory.ingest import ingest_file, IngestResult, resolve_sources
    from ragix_core.memory.mcp.formatting import format_injection_block

    dispatcher = _get_dispatcher(args.db)
    store = dispatcher.store

    # --- Phase 1: Ingest (optional) ---
    if args.source:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        resolved = resolve_sources(args.source)
        total = IngestResult()
        for path in resolved:
            r = ingest_file(
                store, path,
                workspace=args.workspace or "default",
                scope=args.scope,
                max_tokens=args.chunk_tokens,
                tags=tags,
                format_mode="auto",
                injectable=True,  # pipe mode: injectable by design
            )
            total.files_processed += r.files_processed
            total.files_skipped += r.files_skipped
            total.chunks_created += r.chunks_created
            total.item_ids.extend(r.item_ids)

        print(
            f"[pipe] Ingested {total.chunks_created} chunks from "
            f"{total.files_processed} file(s) "
            f"({total.files_skipped} skipped)",
            file=sys.stderr,
        )

    # --- Phase 2: Recall via FTS5 ---
    params: dict = {"query": args.query, "k": 50}
    if args.tier:
        params["tier"] = args.tier
    result = dispatcher.dispatch("search", params)

    if result["status"] != "ok":
        print(f"Error: {result.get('message', 'unknown')}", file=sys.stderr)
        sys.exit(1)

    items = result.get("items", [])
    if not items:
        print("[pipe] No matching items.", file=sys.stderr)
        sys.exit(0)

    # Enrich with full content
    ids = [it["id"] for it in items]
    full = dispatcher.dispatch("read", {"ids": ids})
    if full["status"] == "ok":
        items = full.get("items", items)

    # Filter non-injectable
    items = [it for it in items if it.get("injectable", True)]

    if not items:
        print("[pipe] No injectable items matched.", file=sys.stderr)
        sys.exit(0)

    # --- Phase 3: Format + print ---
    text = format_injection_block(items, args.budget, len(items))
    print(text)


def cmd_palace(args: argparse.Namespace) -> None:
    """Browse memory palace."""
    dispatcher = _get_dispatcher(args.db)
    from ragix_core.memory.palace import MemoryPalace

    palace = MemoryPalace(dispatcher.store)
    result = palace.list_path(args.path or "")

    print(f"Palace: {result.get('path', '/')} ({result.get('level', '')})")
    print(f"  {result.get('count', 0)} entries\n")
    for child in result.get("children", []):
        name = child.get("name") or child.get("title") or child.get("item_id", "")
        extra = ""
        if "item_count" in child:
            extra = f"  ({child['item_count']} items)"
        elif "tier" in child:
            extra = f"  [{child.get('tier', '').upper()}] {child.get('type', '')}"
        print(f"  {name}{extra}")


# ---------------------------------------------------------------------------
# Default config for init command
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_YAML = """\
# RAGIX Memory workspace configuration
# See: https://github.com/ovitrac/RAGIX
store:
  wal_mode: true
  fts_tokenizer: "unicode61 remove_diacritics 2"
policy:
  secret_patterns_enabled: true
  injection_patterns_enabled: true
  instructional_content_enabled: true
  max_content_length: 2000
"""


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize a new memory workspace directory."""
    target = Path(args.path).resolve()
    target.mkdir(parents=True, exist_ok=True)

    db_path = target / "memory.db"
    config_path = target / "config.yaml"
    gitignore_path = target / ".gitignore"

    if db_path.exists() and not args.force:
        print(f"Workspace already exists: {target}", file=sys.stderr)
        print(f"  Use --force to reinitialize.", file=sys.stderr)
        sys.exit(1)

    # Create database with schema + FTS5
    from ragix_core.memory.store import MemoryStore
    store = MemoryStore(db_path=str(db_path))
    store.close()

    # Write default config (if absent)
    if not config_path.exists():
        config_path.write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")

    # Write .gitignore (if absent)
    if not gitignore_path.exists():
        gitignore_path.write_text(
            "*.db\n*.db-wal\n*.db-shm\n", encoding="utf-8"
        )

    # Save as last-used DB
    _save_last_db(str(db_path))

    print(f"Memory workspace initialized: {target}")
    print(f"  Database:  {db_path}")
    print(f"  Config:    {config_path}")
    print(f"  Gitignore: {gitignore_path}")
    print(f"\nSet environment variable for db-free commands:")
    print(f"  export RAGIX_MEMORY_DB=\"{db_path}\"")


def cmd_pull(args: argparse.Namespace) -> None:
    """Capture LLM output from stdin and store as memory items."""
    text = sys.stdin.read()
    if not text.strip():
        print("[pull] No input received on stdin.", file=sys.stderr)
        sys.exit(1)

    dispatcher = _get_dispatcher(args.db)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    # Attempt structured proposal extraction
    proposals = []
    try:
        from ragix_core.memory.proposer import MemoryProposer
        proposer = MemoryProposer()
        _, proposals = proposer.parse_response_text(text)
    except Exception as e:
        logger.debug(f"Proposer parse failed: {e}")

    if proposals:
        # Structured: multiple atomic items via governance pipeline
        items_data = [p.to_dict() if hasattr(p, "to_dict") else p for p in proposals]
        for it in items_data:
            if tags:
                it.setdefault("tags", []).extend(tags)
            it.setdefault("scope", args.scope)
        result = dispatcher.dispatch("propose", {"items": items_data})
        accepted = result.get("accepted", 0)
        rejected = result.get("rejected", 0)
        print(
            f"[pull] Stored {accepted} item(s) from proposals "
            f"({rejected} rejected by policy)",
            file=sys.stderr,
        )
    else:
        # Fallback: store entire text as a single note
        print("[pull] no structured proposals found \u2014 storing as single note",
              file=sys.stderr)
        title = args.title or f"LLM capture {datetime.now():%Y-%m-%d %H:%M}"
        content = text.strip()
        if len(content) > 2000:
            # Split into chunks if too long for a single item
            from ragix_core.memory.ingest import chunk_paragraphs
            chunks = chunk_paragraphs(content, max_tokens=1800)
            for i, (chunk_text, _, _) in enumerate(chunks):
                chunk_title = f"{title} \u00b7 part {i+1}/{len(chunks)}" if len(chunks) > 1 else title
                dispatcher.dispatch("write", {
                    "title": chunk_title,
                    "content": chunk_text,
                    "tags": tags,
                    "tier": "stm",
                    "type": "note",
                    "scope": args.scope,
                })
            print(
                f"[pull] Stored as {len(chunks)} note(s): {title}",
                file=sys.stderr,
            )
        else:
            result = dispatcher.dispatch("write", {
                "title": title,
                "content": content,
                "tags": tags,
                "tier": "stm",
                "type": "note",
                "scope": args.scope,
            })
            item_id = result.get("id", "unknown")
            print(f"[pull] Stored as note {item_id}: {title}", file=sys.stderr)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the RAGIX Memory MCP server in foreground."""
    try:
        from ragix_core.memory.mcp.server import create_server, build_parser
    except ImportError:
        print(
            "MCP dependencies not installed. Run: pip install ragix[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build server args from CLI args
    server_argv = ["--db", args.db]
    if hasattr(args, "fts_tokenizer") and args.fts_tokenizer:
        server_argv.extend(["--fts-tokenizer", args.fts_tokenizer])
    if args.verbose:
        server_argv.append("--verbose")

    server_args = build_parser().parse_args(server_argv)
    mcp, _ = create_server(server_args)

    print(f"Starting RAGIX Memory MCP server (db={args.db})", file=sys.stderr)
    print("Press Ctrl+C to stop.", file=sys.stderr)
    mcp.run()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ragix-memory",
        description="RAGIX Memory Subsystem CLI",
    )
    # Precedence: CLI --db > RAGIX_MEMORY_DB env > last-db cache > memory.db
    _env_db = os.environ.get("RAGIX_MEMORY_DB")
    _last = _load_last_db()
    _db_default = _env_db or _last or "memory.db"
    parser.add_argument(
        "--db", default=_db_default,
        help=f"Path to SQLite database (default: {_db_default})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # search
    p_search = sub.add_parser("search", help="Search memory items")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--tier", help="Filter by tier (stm/mtm/ltm)")
    p_search.add_argument("--type", help="Filter by type")
    p_search.add_argument("--k", type=int, default=10, help="Max results")
    p_search.set_defaults(func=cmd_search)

    # show
    p_show = sub.add_parser("show", help="Show item details")
    p_show.add_argument("id", help="Memory item ID")
    p_show.set_defaults(func=cmd_show)

    # stats
    p_stats = sub.add_parser("stats", help="Store statistics")
    p_stats.add_argument("--corpus", default=None,
                         help="Filter stats by corpus ID")
    p_stats.set_defaults(func=cmd_stats)

    # consolidate
    p_cons = sub.add_parser("consolidate", help="Run consolidation")
    p_cons.add_argument("--scope", default="project", help="Scope filter")
    p_cons.add_argument("--tiers", default="stm", help="Comma-separated tiers")
    p_cons.add_argument("--no-promote", action="store_true", help="Skip promotion")
    p_cons.set_defaults(func=cmd_consolidate)

    # export
    p_export = sub.add_parser("export", help="Export to JSONL")
    p_export.add_argument("--output", "-o", help="Output file (stdout if omitted)")
    p_export.add_argument(
        "--tier", choices=["S0", "S2", "S3"], default="S3",
        help="Secrecy tier for redaction (default: S3 = no redaction)",
    )
    p_export.set_defaults(func=cmd_export)

    # import
    p_import = sub.add_parser("import", help="Import from JSONL")
    p_import.add_argument("file", help="JSONL file path")
    p_import.set_defaults(func=cmd_import)

    # report
    p_report = sub.add_parser("report", help="Generate memory report")
    p_report.add_argument("--workspace", "-w", default=None,
                          help="Named workspace (e.g. rie-grdf)")
    p_report.add_argument("--scenario", "-s", default="summarize_content",
                          help="Scenario name (default: summarize_content)")
    p_report.add_argument("--config", "-c", default=None,
                          help="Custom YAML config path (default: builtin)")
    p_report.add_argument("--set", action="append", default=[],
                          help="Override config value (e.g. recall.budgets=500,1500)")
    p_report.add_argument("--output", "-o", default=None,
                          help="Output file (stdout if omitted)")
    p_report.add_argument("--embedder", default="mock",
                          help="Embedding backend (default: mock)")
    p_report.add_argument("--scope", default="audit",
                          help="Workspace scope (default: audit)")
    p_report.add_argument("--corpus", default=None,
                          help="Corpus ID for workspace registration")
    p_report.add_argument("--list", dest="list_scenarios", action="store_true",
                          help="List available scenarios and exit")
    p_report.set_defaults(func=cmd_report)

    # recall
    p_recall = sub.add_parser("recall", help="Token-budgeted recall (injection block)")
    p_recall.add_argument("query", help="Natural language query")
    _recall_budget = _env_int("RAGIX_MEMORY_BUDGET", 1500)
    p_recall.add_argument("--budget", type=int, default=_recall_budget,
                          help=f"Token budget (default: {_recall_budget})")
    p_recall.add_argument("-w", "--workspace", default=None,
                          help="Named workspace")
    p_recall.add_argument("--tier", default=None,
                          help="Filter by tier (stm/mtm/ltm)")
    p_recall.add_argument("--no-header", action="store_true",
                          help="Strip metadata header from output")
    p_recall.set_defaults(func=cmd_recall)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest files into memory")
    p_ingest.add_argument("--source", nargs="+", required=True,
                          help="Files, directories, globs, or '-' for stdin")
    p_ingest.add_argument("-w", "--workspace", default=None,
                          help="Named workspace")
    p_ingest.add_argument("--chunk-tokens", type=int, default=1800,
                          help="Max tokens per chunk (default: 1800)")
    p_ingest.add_argument("--tags", default=None,
                          help="Comma-separated tags")
    p_ingest.add_argument("--format", default="text",
                          choices=["text", "auto"],
                          help="text=plain, auto=infer type/tags/title")
    p_ingest.add_argument("--injectable", action="store_true",
                          help="Mark chunks as injectable (default: false)")
    p_ingest.add_argument("--scope", default="audit",
                          help="Item scope (default: audit)")
    p_ingest.add_argument("--corpus", default=None,
                          help="Corpus ID")
    p_ingest.set_defaults(func=cmd_ingest)

    # Shared argument extractor for pipe/push (prevents silent drift)
    _budget_default = _env_int("RAGIX_MEMORY_BUDGET", 2000)

    def _add_pipe_arguments(p: argparse.ArgumentParser) -> None:
        """Register pipe/push arguments on a parser. Single source of truth."""
        p.add_argument("query", help="Natural language query")
        p.add_argument("--source", nargs="+", default=None,
                       help="Files, directories, or globs to ingest (skipped if unchanged)")
        p.add_argument("--budget", type=int, default=_budget_default,
                       help=f"Token budget (default: {_budget_default})")
        p.add_argument("-w", "--workspace", default=None,
                       help="Named workspace")
        p.add_argument("--tier", default=None,
                       help="Filter by tier (stm/mtm/ltm)")
        p.add_argument("--chunk-tokens", type=int, default=1800,
                       help="Max tokens per chunk (default: 1800)")
        p.add_argument("--tags", default=None,
                       help="Comma-separated tags for ingested files")
        p.add_argument("--scope", default="audit",
                       help="Item scope (default: audit)")
        p.set_defaults(func=cmd_pipe)

    # pipe (unified: ingest + recall)
    p_pipe = sub.add_parser("pipe", help="Ingest + recall in one shot")
    _add_pipe_arguments(p_pipe)

    # push (alias for pipe — same handler, same args, zero drift)
    p_push = sub.add_parser("push", help="Ingest + recall in one shot (alias for pipe)")
    _add_pipe_arguments(p_push)

    # init
    p_init = sub.add_parser("init", help="Initialize a memory workspace")
    p_init.add_argument("path", nargs="?", default=".memory",
                        help="Workspace directory (default: .memory)")
    p_init.add_argument("--force", action="store_true",
                        help="Reinitialize existing workspace")
    p_init.set_defaults(func=cmd_init)

    # pull
    p_pull = sub.add_parser("pull", help="Store LLM output from stdin into memory")
    p_pull.add_argument("--tags", default=None,
                        help="Comma-separated tags for stored items")
    p_pull.add_argument("--title", default=None,
                        help="Title for the stored note (fallback mode)")
    p_pull.add_argument("--scope", default="audit",
                        help="Item scope (default: audit)")
    p_pull.set_defaults(func=cmd_pull)

    # serve
    p_serve = sub.add_parser("serve", help="Start MCP server")
    p_serve.add_argument("--fts-tokenizer", default=None,
                         help="FTS5 tokenizer preset (fr/en/raw)")
    p_serve.set_defaults(func=cmd_serve)

    # palace
    p_palace = sub.add_parser("palace", help="Browse memory palace")
    p_palace.add_argument("path", nargs="?", default="", help="Palace path (e.g. default/architecture)")
    p_palace.set_defaults(func=cmd_palace)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Persist last-used DB for implicit reuse
    _save_last_db(args.db)

    args.func(args)


if __name__ == "__main__":
    main()
