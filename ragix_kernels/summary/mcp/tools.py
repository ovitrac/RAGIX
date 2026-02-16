"""
KOAS Summary MCP Tools

Provides FastMCP tool registration for:
    summary_ingest     - Run corpus ingestion (Stage 1)
    summary_run        - Run full pipeline (Stage 1+2+3)
    summary_status     - Check workspace status
    summary_query      - Search memory items by text query
    summary_drift      - Compute drift between corpus versions
    summary_viz        - Generate HTML visualizations
    summary_summarize  - Generate summary from existing memory

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def register_summary_tools(mcp_server):
    """Register summary tools with a FastMCP server instance."""

    @mcp_server.tool()
    async def summary_ingest(
        corpus_folder: str,
        workspace: Optional[str] = None,
        scope: str = "project",
        model: str = "ibm/granite4:32b-a9b-h",
        strategy: str = "pages",
    ) -> str:
        """
        Ingest a document corpus into memory.

        Reads all PDF/MD/TXT files, extracts rules via LLM,
        and stores as policy-governed memory items.

        Args:
            corpus_folder: Path to folder containing documents
            workspace: Output workspace (default: auto)
            scope: Memory scope label (e.g., "grdf-rie")
            model: LLM model for rule extraction
            strategy: Chunking strategy (pages/headings/windows)
        """
        try:
            from ragix_kernels.summary.cli.summaryctl import cmd_ingest
            import argparse

            ws = workspace or str(
                Path(corpus_folder).resolve().parent / "summary_workspace"
            )
            args = argparse.Namespace(
                corpus=corpus_folder,
                workspace=ws,
                scope=scope,
                model=model,
                strategy=strategy,
                max_chunk_tokens=800,
                embedder="mock",
                embedder_model="nomic-embed-text",
                ollama_url="http://localhost:11434",
                verbose=True,
            )
            ret = cmd_ingest(args)
            if ret == 0:
                return json.dumps({"status": "ok", "workspace": ws})
            return json.dumps({"status": "error", "code": ret})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp_server.tool()
    async def summary_run(
        corpus_folder: str,
        workspace: Optional[str] = None,
        scope: str = "project",
        model: str = "ibm/granite4:32b-a9b-h",
        language: str = "French",
        max_tokens: int = 12000,
    ) -> str:
        """
        Run the full KOAS Summary pipeline.

        Stage 1: Ingest corpus -> extract rules -> store in memory
        Stage 2: Consolidate -> budgeted recall
        Stage 3: Generate sections -> verify citations -> assemble report

        Returns the path to the final summary.md.

        Args:
            corpus_folder: Path to folder containing documents
            workspace: Output workspace (default: auto)
            scope: Memory scope label
            model: LLM model for extraction and generation
            language: Output language (French/English)
            max_tokens: Token budget for memory injection
        """
        try:
            from ragix_kernels.summary.cli.summaryctl import cmd_run
            import argparse

            ws = workspace or str(
                Path(corpus_folder).resolve().parent / "summary_workspace"
            )
            args = argparse.Namespace(
                corpus=corpus_folder,
                workspace=ws,
                scope=scope,
                model=model,
                strategy="pages",
                max_chunk_tokens=800,
                max_tokens=max_tokens,
                min_per_domain=3,
                max_per_domain=25,
                embedder="mock",
                embedder_model="nomic-embed-text",
                ollama_url="http://localhost:11434",
                language=language,
                skip_consolidation=False,
                verbose=True,
            )
            ret = cmd_run(args)

            summary_path = Path(ws) / "stage3" / "summary.md"
            if ret == 0 and summary_path.exists():
                return json.dumps({
                    "status": "ok",
                    "workspace": ws,
                    "summary": str(summary_path),
                    "size_bytes": summary_path.stat().st_size,
                })
            return json.dumps({"status": "error", "code": ret})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp_server.tool()
    async def summary_status(workspace: str) -> str:
        """
        Check the status of a KOAS Summary workspace.

        Returns memory store stats, stage completion, and report info.

        Args:
            workspace: Path to the summary workspace directory
        """
        try:
            ws = Path(workspace)
            db_path = ws / "memory.db"

            result: Dict[str, Any] = {"workspace": str(ws)}

            if db_path.exists():
                from ragix_core.memory.store import MemoryStore
                store = MemoryStore(str(db_path))
                result["memory"] = store.stats()
            else:
                result["memory"] = None

            # Check stages
            stages = {}
            for stage in (1, 2, 3):
                stage_dir = ws / f"stage{stage}"
                if stage_dir.exists():
                    files = list(stage_dir.glob("*.json"))
                    stages[f"stage{stage}"] = [f.name for f in files]
            result["stages"] = stages

            # Check report
            summary_md = ws / "stage3" / "summary.md"
            if summary_md.exists():
                result["report"] = {
                    "path": str(summary_md),
                    "size_bytes": summary_md.stat().st_size,
                }

            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    # ------------------------------------------------------------------
    # V3.1: New tools
    # ------------------------------------------------------------------

    @mcp_server.tool()
    async def summary_query(
        workspace: str,
        query: str,
        tier: Optional[str] = None,
        type_filter: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """
        Search memory items by text query.

        Full-text search across titles, content, and tags.
        Returns scored results with optional tier/type/domain filters.

        Args:
            workspace: Path to the summary workspace directory
            query: Search terms (space-separated, AND logic)
            tier: Filter by memory tier (stm/mtm/ltm)
            type_filter: Filter by item type
            domain: Filter by document domain
            limit: Maximum results to return (default 20)
        """
        try:
            ws = Path(workspace)
            db_path = ws / "memory.db"
            if not db_path.exists():
                return json.dumps({"status": "error", "message": "No memory.db found"})

            from ragix_core.memory.store import MemoryStore
            store = MemoryStore(str(db_path))

            items = store.search_fulltext(
                query=query,
                tier=tier,
                type_filter=type_filter,
                limit=limit * 2,  # over-fetch for domain post-filter
            )

            # Post-filter by domain if requested
            if domain:
                from ragix_kernels.summary.visualization.domain_utils import extract_domain
                items = [it for it in items if extract_domain(it).lower() == domain.lower()]

            items = items[:limit]

            rows = []
            for item in items:
                rows.append({
                    "id": item.id,
                    "title": item.title,
                    "type": item.type,
                    "tier": item.tier,
                    "tags": item.tags,
                    "confidence": item.confidence,
                    "source_id": item.provenance.source_id,
                    "content_preview": (item.content or "")[:300],
                })

            return json.dumps({
                "status": "ok",
                "query": query,
                "count": len(rows),
                "items": rows,
            }, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp_server.tool()
    async def summary_drift(
        workspace_a: str,
        workspace_b: str,
    ) -> str:
        """
        Compute drift between two corpus versions.

        Compares memory stores from two workspaces to identify
        added, removed, and modified items.

        Args:
            workspace_a: Path to the baseline workspace
            workspace_b: Path to the updated workspace
        """
        try:
            from ragix_kernels.summary.cli.summaryctl import cmd_drift
            import argparse

            args = argparse.Namespace(
                workspace=None,
                workspace_a=workspace_a,
                workspace_b=workspace_b,
                corpus_a=None,
                corpus_b=None,
                verbose=True,
            )
            ret = cmd_drift(args)

            # Find the drift report
            ws_b = Path(workspace_b)
            drift_files = list(ws_b.glob("**/drift_report.json"))
            if ret == 0 and drift_files:
                with open(drift_files[-1]) as f:
                    report = json.load(f)
                return json.dumps({
                    "status": "ok",
                    "report_path": str(drift_files[-1]),
                    "summary": {
                        "added": report.get("added", 0),
                        "removed": report.get("removed", 0),
                        "modified": report.get("modified", 0),
                        "unchanged": report.get("unchanged", 0),
                    },
                }, indent=2)
            return json.dumps({"status": "error", "code": ret})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp_server.tool()
    async def summary_viz(
        workspace: str,
        secrecy: str = "S3",
        views: Optional[str] = None,
    ) -> str:
        """
        Generate HTML visualizations for a workspace.

        Produces interactive dashboards: graph, memory explorer,
        query explorer, timeline, clusters, heatmap, metrics.

        Args:
            workspace: Path to the summary workspace directory
            secrecy: Secrecy tier for redaction (S0/S2/S3)
            views: Comma-separated view names to generate (default: all)
        """
        try:
            ws = Path(workspace)
            db_path = ws / "memory.db"
            if not db_path.exists():
                return json.dumps({"status": "error", "message": "No memory.db found"})

            from ragix_core.memory.store import MemoryStore
            from ragix_kernels.summary.visualization.render_html import render_all

            store = MemoryStore(str(db_path))

            # Optional graph store
            graph_store = None
            graph_db = ws / "graph.db"
            if graph_db.exists():
                from ragix_core.memory.graph_store import GraphStore
                graph_store = GraphStore(str(graph_db))

            selected = set(views.split(",")) if views else None

            outputs = render_all(
                workspace=ws,
                store=store,
                graph_store=graph_store,
                tier=secrecy,
                views=selected,
            )

            result = {
                "status": "ok",
                "secrecy": secrecy,
                "files": {
                    name: {"path": str(path), "size_bytes": path.stat().st_size}
                    for name, path in outputs.items()
                },
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp_server.tool()
    async def summary_summarize(
        workspace: str,
        model: str = "ibm/granite4:32b-a9b-h",
        language: str = "French",
        max_tokens: int = 12000,
        secrecy: str = "S3",
    ) -> str:
        """
        Generate summary from existing memory (Stage 2+3 only).

        Skips ingestion — uses the existing memory store to consolidate,
        recall, and generate a summary report. Useful for re-generating
        with different parameters.

        Args:
            workspace: Path to workspace with existing memory.db
            model: LLM model for generation
            language: Output language (French/English)
            max_tokens: Token budget for memory injection
            secrecy: Secrecy tier for redaction (S0/S2/S3)
        """
        try:
            from ragix_kernels.summary.cli.summaryctl import cmd_summarize
            import argparse

            ws = Path(workspace)
            if not (ws / "memory.db").exists():
                return json.dumps({"status": "error", "message": "No memory.db found"})

            args = argparse.Namespace(
                workspace=str(ws),
                scope=None,
                model=model,
                max_tokens=max_tokens,
                min_per_domain=3,
                max_per_domain=25,
                embedder="mock",
                embedder_model="nomic-embed-text",
                ollama_url="http://localhost:11434",
                language=language,
                skip_consolidation=False,
                graph=True,
                secrecy=secrecy,
                delta=False,
                verbose=True,
            )
            ret = cmd_summarize(args)

            summary_path = ws / "stage3" / "summary.md"
            if ret == 0 and summary_path.exists():
                return json.dumps({
                    "status": "ok",
                    "workspace": str(ws),
                    "summary": str(summary_path),
                    "size_bytes": summary_path.stat().st_size,
                })
            return json.dumps({"status": "error", "code": ret})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
