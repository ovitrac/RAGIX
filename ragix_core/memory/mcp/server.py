"""
RAGIX Memory MCP Server — Persistent Structured Memory for LLMs

Standalone MCP server exposing RAGIX Memory operations via the
Model Context Protocol.  Works with Claude Desktop, Claude Code,
VS Code, and any MCP-compatible client.

Architecture: thin MCP layer delegating 1:1 to MemoryToolDispatcher.
Zero business logic in this module — all logic lives in ragix_core/memory/.

Usage:
    python -m ragix_core.memory.mcp --db /path/to/memory.db
    python -m ragix_core.memory.mcp --embedder mock --fts-tokenizer fr

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Instructions embedded in FastMCP — always visible to any MCP client.
# Core rules only; detailed guide is the optional prompt resource (§5.4).
_MCP_INSTRUCTIONS = (
    "Persistent structured memory for large document operations.\n"
    "\n"
    "PRIMARY: Use memory_recall for token-budgeted context injection.\n"
    "SEARCH:  Use memory_search for interactive discovery.\n"
    "STORE:   Use memory_propose to save findings (tags + provenance).\n"
    "         Use memory_write only for privileged/dev operations.\n"
    "\n"
    "Rules:\n"
    "- Store distilled knowledge, NOT raw document excerpts\n"
    "- Include provenance (source_doc, page/section) when available\n"
    "- Use 3-7 lowercase hyphenated tags per item\n"
    "- NEVER store secrets, tool invocations, or system prompt fragments\n"
    "- NEVER store instructions to yourself ('always remember to...')\n"
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the memory MCP server."""
    p = argparse.ArgumentParser(
        prog="ragix-memory-mcp",
        description="RAGIX Memory MCP Server — persistent structured memory for LLMs",
    )
    p.add_argument(
        "--db",
        default=os.environ.get("RAGIX_MEMORY_DB", "memory.db"),
        help="SQLite database path (default: memory.db or $RAGIX_MEMORY_DB)",
    )
    p.add_argument(
        "--embedder",
        default=os.environ.get("RAGIX_MEMORY_EMBEDDER", "mock"),
        choices=["mock", "ollama"],
        help="Embedding backend (default: mock or $RAGIX_MEMORY_EMBEDDER)",
    )
    p.add_argument(
        "--model",
        default="nomic-embed-text",
        help="Embedding model name for ollama backend (default: nomic-embed-text)",
    )
    p.add_argument(
        "--fts-tokenizer",
        default=os.environ.get("RAGIX_MEMORY_FTS", "fr"),
        help=(
            "FTS5 tokenizer preset: fr (accent-insensitive), en (porter stemming), "
            "raw (unicode61), or a custom tokenizer string. "
            "Default: fr or $RAGIX_MEMORY_FTS"
        ),
    )
    p.add_argument(
        "--secrecy",
        default=os.environ.get("RAGIX_MEMORY_SECRECY", "S3"),
        choices=["S0", "S2", "S3"],
        help="Default secrecy tier for responses (default: S3 / full detail)",
    )
    p.add_argument(
        "--inject-budget",
        type=int,
        default=int(os.environ.get("RAGIX_MEMORY_BUDGET", "1500")),
        help="Default injection budget in tokens (default: 1500)",
    )
    p.add_argument(
        "--rate-limit",
        action="store_true",
        default=True,
        dest="rate_limit",
        help="Enable per-session rate limiting (default: enabled)",
    )
    p.add_argument(
        "--no-rate-limit",
        action="store_false",
        dest="rate_limit",
        help="Disable per-session rate limiting",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return p


def create_server(args=None):
    """
    Create and configure the FastMCP server with memory tools.

    Args:
        args: Parsed argparse.Namespace, or None to parse from sys.argv.

    Returns:
        (mcp_server, dispatcher) tuple.
    """
    from mcp.server.fastmcp import FastMCP

    from ragix_core.memory.config import (
        EmbedderConfig,
        MemoryConfig,
        RateLimitConfig,
        RecallConfig,
        SecrecyConfig,
        StoreConfig,
    )
    from ragix_core.memory.mcp.metrics import MetricsCollector
    from ragix_core.memory.mcp.rate_limiter import RateLimiter
    from ragix_core.memory.mcp.session import SessionManager
    from ragix_core.memory.mcp.tools import register_memory_tools
    from ragix_core.memory.mcp.workspace import WorkspaceRouter
    from ragix_core.memory.store import FTS_TOKENIZER_PRESETS
    from ragix_core.memory.tools import create_dispatcher

    if args is None:
        args = build_parser().parse_args()

    # Resolve FTS tokenizer preset
    fts_tok = FTS_TOKENIZER_PRESETS.get(args.fts_tokenizer, args.fts_tokenizer)

    # Rate limiting: honour --rate-limit / --no-rate-limit flag
    rate_limit_enabled = getattr(args, "rate_limit", True)

    # Build config from CLI arguments
    config = MemoryConfig(
        store=StoreConfig(
            db_path=args.db,
            fts_tokenizer=fts_tok,
        ),
        embedder=EmbedderConfig(
            backend=args.embedder,
            model=args.model,
        ),
        recall=RecallConfig(
            inject_budget_tokens=args.inject_budget,
        ),
        secrecy=SecrecyConfig(
            tier=args.secrecy,
        ),
        rate_limit=RateLimitConfig(
            enabled=rate_limit_enabled,
        ),
    )

    # Create dispatcher (wires store + policy + embedder)
    dispatcher = create_dispatcher(config)

    # Create session manager (shares SQLite connection with store)
    session_mgr = SessionManager(dispatcher.store._conn)

    # Create workspace router (shares SQLite connection with store)
    workspace_router = WorkspaceRouter(dispatcher.store._conn)

    # Create metrics collector (no external deps)
    metrics_collector = MetricsCollector()

    # Create rate limiter (from config)
    rate_limiter = RateLimiter(config.rate_limit)

    # Create FastMCP server
    mcp = FastMCP(
        name="RAGIX Memory",
        instructions=_MCP_INSTRUCTIONS,
    )

    # Register all tools (11 core + 2 session bridge + 4 management)
    register_memory_tools(
        mcp,
        dispatcher,
        session_mgr=session_mgr,
        workspace_router=workspace_router,
        metrics=metrics_collector,
        rate_limiter=rate_limiter,
    )

    # Register memory prompt resource (§5.4 — operational guide for Claude)
    _register_prompt_resource(mcp)

    logger.info(
        f"RAGIX Memory MCP server ready: db={args.db}, "
        f"embedder={args.embedder}, fts={args.fts_tokenizer}, "
        f"secrecy={args.secrecy}, rate_limit={rate_limit_enabled}"
    )

    return mcp, dispatcher


_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _register_prompt_resource(mcp) -> None:
    """Register memory guide as an MCP prompt resource."""
    guide_path = _PROMPTS_DIR / "memory_guide.md"
    if not guide_path.exists():
        logger.warning(f"Memory guide not found: {guide_path}")
        return

    guide_text = guide_path.read_text(encoding="utf-8")

    @mcp.prompt()
    def memory_guide() -> str:
        """Operational guide for using RAGIX persistent memory effectively."""
        return guide_text

    logger.info("Registered memory_guide prompt resource")


def main():
    """CLI entry point — parse args, create server, run."""
    parser = build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    mcp, _dispatcher = create_server(args)
    mcp.run()


if __name__ == "__main__":
    main()
