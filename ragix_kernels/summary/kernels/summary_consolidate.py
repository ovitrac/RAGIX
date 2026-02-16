"""
summary_consolidate — Stage 2: Memory Consolidation

Runs the consolidation pipeline with real embeddings to merge
near-duplicate rules and promote items STM -> MTM -> LTM.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

from typing import Any, Dict

from ragix_kernels.base import Kernel, KernelInput


class SummaryConsolidateKernel(Kernel):
    name = "summary_consolidate"
    version = "1.0.0"
    category = "summary"
    stage = 2
    description = "Consolidate memory items (cluster, merge, promote)"
    requires = ["summary_ingest"]
    provides = ["consolidation_stats"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Embed items, build graph, cluster near-duplicates, and promote tiers.

        Run consolidation pipeline with real or mock embeddings. Supports
        delta mode (neighborhood-scoped) and graph-assisted clustering.
        """
        import json
        import logging
        from ragix_core.memory.config import ConsolidateConfig
        from ragix_core.memory.consolidate import ConsolidationPipeline
        from ragix_core.memory.embedder import create_embedder
        from ragix_core.memory.store import MemoryStore

        log = logging.getLogger(__name__)

        cfg = input.config
        db_path = cfg.get("db_path", str(input.workspace / "memory.db"))
        scope = cfg.get("scope", "project")
        embedder_backend = cfg.get("embedder_backend", "mock")
        embedder_model = cfg.get("embedder_model", "nomic-embed-text")
        similarity_threshold = cfg.get("similarity_threshold", 0.85)

        fts_tokenizer = cfg.get("fts_tokenizer")
        store = MemoryStore(db_path, fts_tokenizer=fts_tokenizer)
        embedder = create_embedder(
            backend=embedder_backend, model=embedder_model,
        )

        consolidate_config = ConsolidateConfig(
            cluster_distance_threshold=1.0 - similarity_threshold,
            fallback_to_deterministic=True,
        )

        # Embed all items first if using real embedder
        embed_stats = {"embedded": 0, "skipped": 0, "failed": 0}
        if embedder_backend != "mock":
            items = store.list_items(scope=scope, exclude_archived=True, limit=5000)
            log.info(f"Embedding {len(items)} items with {embedder_model}...")
            # Collect un-embedded items for batch processing
            to_embed_ids = []
            to_embed_texts = []
            for item in items:
                existing = store.read_embedding(item.id)
                if existing is None:
                    to_embed_ids.append(item.id)
                    to_embed_texts.append(f"{item.title} {item.content}")
                else:
                    embed_stats["skipped"] += 1
            # Batch embed when possible (reduces API round-trips)
            if to_embed_texts:
                batch_size = 50
                for start in range(0, len(to_embed_texts), batch_size):
                    batch_texts = to_embed_texts[start:start + batch_size]
                    batch_ids = to_embed_ids[start:start + batch_size]
                    try:
                        vectors = embedder.embed_batch(batch_texts)
                        for iid, vec in zip(batch_ids, vectors):
                            store.write_embedding(
                                iid, vec,
                                embedder.model_name, embedder.dimension,
                            )
                            embed_stats["embedded"] += 1
                    except Exception as e:
                        # Fall back to sequential on batch failure
                        for iid, text in zip(batch_ids, batch_texts):
                            try:
                                vec = embedder.embed_text(text)
                                store.write_embedding(
                                    iid, vec,
                                    embedder.model_name, embedder.dimension,
                                )
                                embed_stats["embedded"] += 1
                            except Exception as e2:
                                embed_stats["failed"] += 1
                                if embed_stats["failed"] <= 3:
                                    log.warning(f"Embed failed for {iid}: {e2}")
                    if start + batch_size < len(to_embed_texts):
                        log.info(f"  Embedded {min(start + batch_size, len(to_embed_texts))}/{len(to_embed_texts)} items...")
            log.info(
                f"Embedding complete: {embed_stats['embedded']} new, "
                f"{embed_stats['skipped']} cached, {embed_stats['failed']} failed"
            )

        # Build/load graph if enabled
        graph = None
        graph_cfg = cfg.get("graph", {})
        if graph_cfg.get("enabled", True):
            from ragix_core.memory.graph_store import GraphStore
            graph = GraphStore(db_path)
            gs = graph.stats()
            if gs.get("total_nodes", 0) == 0:
                log.info("No graph found, building from store...")
                graph.build_from_store(
                    store, scope, embedder if embedder_backend != "mock" else None,
                    sim_threshold=graph_cfg.get("similarity_edge_threshold", 0.85),
                )
            else:
                log.info(
                    f"Graph loaded: {gs['total_nodes']} nodes, "
                    f"{gs['total_edges']} edges"
                )

        secrecy_cfg = cfg.get("secrecy", {})
        secrecy_tier = secrecy_cfg.get("tier", "S3")

        pipeline = ConsolidationPipeline(
            store, embedder, consolidate_config,
            graph=graph, secrecy_tier=secrecy_tier,
        )

        # Run consolidation (V3.0: delta-aware wiring)
        delta_mode = cfg.get("delta", False)
        new_item_ids = cfg.get("new_item_ids", [])

        if delta_mode and new_item_ids:
            log.info(
                f"Delta consolidation: {len(new_item_ids)} new items "
                f"(neighborhood-scoped)"
            )
            stats = pipeline.run_delta(
                new_item_ids, scope=scope, promote=True,
            )
        else:
            if delta_mode and not new_item_ids:
                log.info("Delta mode but no new_item_ids — falling back to full consolidation")
            stats = pipeline.run(scope=scope, tiers=["stm"], promote=True)

        # Post-consolidation: compact graph (rewire superseded nodes)
        if graph is not None:
            compact_stats = graph.compact(store)
            log.info(
                f"Graph compacted: {compact_stats['nodes_removed']} nodes removed, "
                f"{compact_stats['rewired']} rewired"
            )

        # Write merge chain artifact (P0 requirement)
        merge_chains = stats.get("merge_chains", [])
        if merge_chains:
            artifact_path = input.workspace / "stage2" / "merge_chains.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            with open(artifact_path, "w") as f:
                json.dump({
                    "total_chains": len(merge_chains),
                    "total_items_merged": stats.get("items_merged", 0),
                    "chains": merge_chains,
                }, f, indent=2)
            log.info(
                f"Merge chain artifact: {len(merge_chains)} chains "
                f"({stats.get('items_merged', 0)} items merged) → {artifact_path}"
            )

        # Merge quality metrics (dispersion, same-doc ratio, etc.)
        merge_metrics = pipeline.compute_merge_metrics(merge_chains)

        # Post-consolidation store stats
        store_stats = store.stats()

        return {
            "consolidation": {
                k: v for k, v in stats.items() if k != "merge_chains"
            },
            "merge_summary": {
                "total_chains": len(merge_chains),
                "total_items_merged": stats.get("items_merged", 0),
                "longest_chain": max(
                    (c["source_count"] for c in merge_chains), default=0
                ),
            },
            "merge_metrics": merge_metrics,
            "embed_stats": embed_stats,
            "store_after": store_stats,
            "embedder": embedder_backend,
            "model": embedder_model if embedder_backend != "mock" else "mock",
            "graph_enabled": graph is not None,
            "secrecy_tier": secrecy_tier,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of consolidation results with merge metrics."""
        c = data.get("consolidation", {})
        s = data.get("store_after", {})
        m = data.get("merge_summary", {})
        e = data.get("embed_stats", {})
        mm = data.get("merge_metrics", {})
        parts = [
            f"Consolidated {c.get('items_processed', 0)} items: "
            f"{c.get('items_merged', 0)} merged in {m.get('total_chains', 0)} chains "
            f"(max={mm.get('max_chain_size', m.get('longest_chain', '?'))}, "
            f"median={mm.get('median_chain_size', '?')}), "
            f"{c.get('items_promoted', 0)} promoted.",
        ]
        if mm.get("same_doc_pct"):
            parts.append(f"Same-doc: {mm['same_doc_pct']}%.")
        if mm.get("high_dispersion_merges", 0) > 0:
            parts.append(
                f"WARNING: {mm['high_dispersion_merges']} high-dispersion merges detected."
            )
        if e.get("embedded", 0) > 0:
            parts.append(
                f"Embeddings: {e['embedded']} new, "
                f"{e.get('skipped', 0)} cached, {e.get('failed', 0)} failed."
            )
        parts.append(f"Store: {s.get('total_items', 0)} active items.")
        if data.get("graph_enabled"):
            parts.append("Graph: enabled.")
        return " ".join(parts)
