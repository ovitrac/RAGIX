"""
summary_build_graph — Stage 2: Build Deterministic Graph from Provenance

Constructs a lightweight graph over the memory store to constrain
consolidation merge candidates. All edges are deterministic (except
optional 'similar' edges which require embeddings).

This kernel is idempotent: it clears the graph and rebuilds from scratch
each time. Build time is typically <30s for ~1,500 items.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class SummaryBuildGraphKernel(Kernel):
    name = "summary_build_graph"
    version = "1.0.0"
    category = "summary"
    stage = 2
    description = "Build deterministic graph edges from memory provenance"
    requires = ["summary_ingest"]
    provides = ["graph_stats"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Build deterministic graph edges from memory provenance and optional similarity."""
        from ragix_core.memory.embedder import create_embedder
        from ragix_core.memory.graph_store import GraphStore
        from ragix_core.memory.store import MemoryStore

        cfg = input.config
        db_path = cfg.get("db_path", str(input.workspace / "memory.db"))
        scope = cfg.get("scope", "project")
        embedder_backend = cfg.get("embedder_backend", "mock")
        embedder_model = cfg.get("embedder_model", "nomic-embed-text")
        graph_cfg = cfg.get("graph", {})
        sim_threshold = graph_cfg.get("similarity_edge_threshold", 0.85)
        similarity_top_k = graph_cfg.get("similarity_top_k", 0)
        max_edges_per_node = graph_cfg.get("max_edges_per_node", 0)

        fts_tokenizer = cfg.get("fts_tokenizer")
        store = MemoryStore(db_path, fts_tokenizer=fts_tokenizer)
        graph = GraphStore(db_path)

        # Full rebuild (graph is cheap)
        graph.clear()

        # Create embedder for optional similarity edges
        embedder = None
        if embedder_backend != "mock":
            embedder = create_embedder(
                backend=embedder_backend, model=embedder_model,
            )

        # Build all edges
        edge_counts = graph.build_from_store(
            store, scope, embedder=embedder, sim_threshold=sim_threshold,
            similarity_top_k=similarity_top_k,
            max_edges_per_node=max_edges_per_node,
        )

        stats = graph.stats()

        # Write artifact
        artifact = {
            "edge_counts": edge_counts,
            "stats": stats,
            "scope": scope,
            "sim_threshold": sim_threshold if embedder else None,
            "similarity_top_k": similarity_top_k,
            "max_edges_per_node": max_edges_per_node,
            "embedder": embedder_backend,
        }
        artifact_path = input.workspace / "stage2" / "graph_stats.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        logger.info(
            f"Graph built: {stats['total_nodes']} nodes, "
            f"{stats['total_edges']} edges → {artifact_path}"
        )

        return artifact

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of graph construction results."""
        s = data.get("stats", {})
        ec = data.get("edge_counts", {})
        return (
            f"Graph: {s.get('total_nodes', 0)} nodes, "
            f"{s.get('total_edges', 0)} edges "
            f"(contains={ec.get('contains', 0)}, adjacent={ec.get('adjacent', 0)}, "
            f"extracted={ec.get('extracted_from', 0)}, mentions={ec.get('mentions', 0)}, "
            f"similar={ec.get('similar', 0)})"
        )
