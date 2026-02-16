"""
Consolidation Pipeline — STM -> MTM -> LTM Promotion

Clusters STM items by type+tags (coarse) then by embedding distance (fine).
For each cluster, produces a merged canonical item via:
  - Granite 3.2B summarizer (if Ollama available)
  - Deterministic fallback (longest content wins, tags/entities merged)

Hard requirement: consolidation never deletes; it supersedes/archives originals.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from ragix_core.memory.config import ConsolidateConfig
from ragix_core.memory.embedder import MemoryEmbedder, cosine_similarity
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import (
    MemoryItem,
    MemoryLink,
    MemoryProvenance,
    _generate_id,
    _now_iso,
)

logger = logging.getLogger(__name__)


class ConsolidationPipeline:
    """
    Consolidation: cluster, merge, promote, and assign palace locations.

    Triggered manually, periodically, or when STM count exceeds threshold.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: MemoryEmbedder,
        config: Optional[ConsolidateConfig] = None,
        graph=None,
        secrecy_tier: str = "S3",
    ):
        """Initialize consolidation pipeline with store, embedder, and optional graph."""
        self._store = store
        self._embedder = embedder
        self._config = config or ConsolidateConfig()
        self._graph = graph  # Optional[GraphStore]
        self._secrecy_tier = secrecy_tier

    def run(
        self,
        scope: str = "project",
        tiers: Optional[List[str]] = None,
        promote: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the consolidation pipeline.

        Steps:
          1. Collect items from target tiers
          2. Cluster by type+tags (coarse) then embedding (fine)
          3. Merge each cluster into a canonical item
          4. Promote eligible items to higher tiers
          5. Auto-assign palace locations

        Returns summary dict.
        """
        tiers = tiers or ["stm"]
        stats = {
            "items_processed": 0,
            "clusters_found": 0,
            "items_merged": 0,
            "items_promoted": 0,
            "palace_assigned": 0,
            "merge_chains": [],   # P0: [{merged_id, source_ids, source_titles}]
        }

        # Step 1: Collect
        items = []
        for tier in tiers:
            items.extend(
                self._store.list_items(
                    tier=tier, scope=scope, exclude_archived=True, limit=5000,
                )
            )
        stats["items_processed"] = len(items)

        if len(items) < 2:
            logger.info(f"Consolidation: only {len(items)} item(s), skipping")
            return stats

        # Step 2: Cluster (graph-assisted if available)
        if self._graph is not None:
            clusters = self._cluster_items_graph(items, self._graph)
        else:
            clusters = self._cluster_items(items)
        stats["clusters_found"] = len(clusters)

        # Step 3: Merge + promote
        # Track IDs already handled (merged or promoted) to avoid double-promotion
        handled_ids: Set[str] = set()

        for cluster in clusters:
            if len(cluster) < 2:
                # Single-item clusters may still be promoted
                if promote:
                    item = cluster[0]
                    if item.id not in handled_ids:
                        if self._try_promote(item):
                            stats["items_promoted"] += 1
                        handled_ids.add(item.id)
                continue

            merged = self._merge_cluster(cluster)
            if merged is not None:
                # Write-time secrecy: redact content before storing
                if self._secrecy_tier != "S3":
                    try:
                        from ragix_kernels.summary.kernels.summary_redact import (
                            redact_for_storage,
                        )
                        merged.content = redact_for_storage(
                            merged.content, self._secrecy_tier,
                        )
                        merged.title = redact_for_storage(
                            merged.title, self._secrecy_tier,
                        )
                    except ImportError:
                        logger.debug("summary_redact not available for write-time redaction")

                # Store merged item — promoted items are injectable
                target_tier = "mtm" if promote else "stm"
                merged.tier = target_tier
                merged.injectable = promote  # MTM = injectable; STM stays non-injectable
                self._store.write_item(merged, reason="consolidation_merge")

                # Embed merged item
                try:
                    vec = self._embedder.embed_text(f"{merged.title} {merged.content}")
                    self._store.write_embedding(
                        merged.id, vec,
                        self._embedder.model_name, self._embedder.dimension,
                    )
                except Exception as e:
                    logger.warning(f"Embedding failed for merged {merged.id}: {e}")

                # Supersede originals + track merge chain
                source_ids = []
                source_titles = []
                for orig in cluster:
                    self._store.supersede_item(orig.id, merged.id)
                    self._store.write_link(MemoryLink(
                        src_id=merged.id, dst_id=orig.id, rel="supersedes",
                    ))
                    handled_ids.add(orig.id)
                    source_ids.append(orig.id)
                    source_titles.append(orig.title[:80])

                stats["items_merged"] += len(cluster)
                stats["merge_chains"].append({
                    "merged_id": merged.id,
                    "merged_title": merged.title[:120],
                    "source_count": len(cluster),
                    "source_ids": source_ids,
                    "source_titles": source_titles,
                })

                # Palace assignment
                try:
                    from ragix_core.memory.palace import MemoryPalace
                    palace = MemoryPalace(self._store)
                    palace.auto_assign(merged)
                    stats["palace_assigned"] += 1
                except Exception as e:
                    logger.warning(f"Palace assignment failed: {e}")

        # Step 4: Promote remaining standalone items not already handled
        if promote:
            for item in items:
                if item.id in handled_ids:
                    continue
                if not item.archived and item.superseded_by is None:
                    if self._try_promote(item):
                        stats["items_promoted"] += 1
                    handled_ids.add(item.id)

        logger.info(
            f"Consolidation complete: {stats['items_processed']} items, "
            f"{stats['clusters_found']} clusters, "
            f"{stats['items_merged']} merged, "
            f"{stats['items_promoted']} promoted"
        )
        return stats

    def run_delta(
        self,
        new_item_ids: List[str],
        scope: str = "project",
        promote: bool = True,
    ) -> Dict[str, Any]:
        """
        V2.4: Neighborhood-scoped re-consolidation for delta mode.

        Instead of re-clustering all items, only process items in the
        graph neighborhoods of newly added/modified items.

        Steps:
          1. For each new item, find BFS neighborhood in graph
          2. Collect all items in affected neighborhoods
          3. Run standard clustering + merge only on this subset
          4. Promote as usual

        Returns summary dict (same shape as run()).
        """
        stats = {
            "items_processed": 0,
            "clusters_found": 0,
            "items_merged": 0,
            "items_promoted": 0,
            "palace_assigned": 0,
            "merge_chains": [],
            "delta_mode": True,
            "new_items": len(new_item_ids),
            "affected_items": 0,
        }

        if not new_item_ids:
            logger.info("Delta consolidation: no new items, skipping")
            return stats

        # Step 1: Find affected neighborhoods via graph BFS
        affected_ids: set = set(new_item_ids)
        if self._graph is not None:
            for iid in new_item_ids:
                nbrs = self._graph.neighborhood_items(
                    iid, depth=2, max_size=50,
                )
                affected_ids.update(nbrs)

        logger.info(
            f"Delta consolidation: {len(new_item_ids)} new items → "
            f"{len(affected_ids)} affected (including neighbors)"
        )
        stats["affected_items"] = len(affected_ids)

        # Step 2: Load affected items from store
        items = []
        for iid in affected_ids:
            item = self._store.read_item(iid)
            if item is not None and not item.archived and item.superseded_by is None:
                items.append(item)

        stats["items_processed"] = len(items)
        if len(items) < 2:
            logger.info(f"Delta consolidation: only {len(items)} active item(s), skipping merge")
            # Still promote singletons
            if promote:
                for item in items:
                    if self._try_promote(item):
                        stats["items_promoted"] += 1
            return stats

        # Step 3: Cluster affected items (graph-assisted if available)
        if self._graph is not None:
            clusters = self._cluster_items_graph(items, self._graph)
        else:
            clusters = self._cluster_items(items)
        stats["clusters_found"] = len(clusters)

        # Step 4: Merge + promote (reuse main pipeline logic)
        handled_ids: set = set()
        for cluster in clusters:
            if len(cluster) < 2:
                if promote:
                    item = cluster[0]
                    if item.id not in handled_ids:
                        if self._try_promote(item):
                            stats["items_promoted"] += 1
                        handled_ids.add(item.id)
                continue

            merged = self._merge_cluster(cluster)
            if merged is not None:
                # Write-time secrecy
                if self._secrecy_tier != "S3":
                    try:
                        from ragix_kernels.summary.kernels.summary_redact import (
                            redact_for_storage,
                        )
                        merged.content = redact_for_storage(
                            merged.content, self._secrecy_tier,
                        )
                        merged.title = redact_for_storage(
                            merged.title, self._secrecy_tier,
                        )
                    except ImportError:
                        pass

                target_tier = "mtm" if promote else "stm"
                merged.tier = target_tier
                merged.injectable = promote  # MTM = injectable; STM stays non-injectable
                self._store.write_item(merged, reason="delta_consolidation_merge")

                try:
                    vec = self._embedder.embed_text(f"{merged.title} {merged.content}")
                    self._store.write_embedding(
                        merged.id, vec,
                        self._embedder.model_name, self._embedder.dimension,
                    )
                except Exception as e:
                    logger.warning(f"Embedding failed for merged {merged.id}: {e}")

                source_ids = []
                source_titles = []
                for orig in cluster:
                    self._store.supersede_item(orig.id, merged.id)
                    self._store.write_link(MemoryLink(
                        src_id=merged.id, dst_id=orig.id, rel="supersedes",
                    ))
                    handled_ids.add(orig.id)
                    source_ids.append(orig.id)
                    source_titles.append(orig.title[:80])

                stats["items_merged"] += len(cluster)
                stats["merge_chains"].append({
                    "merged_id": merged.id,
                    "merged_title": merged.title[:120],
                    "source_count": len(cluster),
                    "source_ids": source_ids,
                    "source_titles": source_titles,
                })

                try:
                    from ragix_core.memory.palace import MemoryPalace
                    palace = MemoryPalace(self._store)
                    palace.auto_assign(merged)
                    stats["palace_assigned"] += 1
                except Exception:
                    pass

        # Promote remaining standalone items
        if promote:
            for item in items:
                if item.id in handled_ids:
                    continue
                if not item.archived and item.superseded_by is None:
                    if self._try_promote(item):
                        stats["items_promoted"] += 1
                    handled_ids.add(item.id)

        logger.info(
            f"Delta consolidation complete: {stats['items_processed']} affected items, "
            f"{stats['clusters_found']} clusters, "
            f"{stats['items_merged']} merged, "
            f"{stats['items_promoted']} promoted"
        )
        return stats

    def should_trigger(self, scope: str = "project") -> bool:
        """Check if automatic consolidation should be triggered."""
        stm_count = self._store.count_items(tier="stm")
        return stm_count >= self._config.stm_threshold

    # -- Clustering --------------------------------------------------------

    def _cluster_items(
        self, items: List[MemoryItem]
    ) -> List[List[MemoryItem]]:
        """
        Two-pass clustering:
          1. Coarse: group by type + primary tag
          2. Fine: split groups by embedding distance
        """
        # Coarse pass: type + primary tag
        coarse: Dict[str, List[MemoryItem]] = defaultdict(list)
        for item in items:
            key = item.type
            if item.tags:
                key += ":" + item.tags[0].lower()
            coarse[key].append(item)

        # Fine pass: split by embedding distance
        clusters: List[List[MemoryItem]] = []
        threshold = self._config.cluster_distance_threshold

        for group_items in coarse.values():
            if len(group_items) <= 1:
                clusters.append(group_items)
                continue

            # Get embeddings
            embs: Dict[str, List[float]] = {}
            for item in group_items:
                data = self._store.read_embedding(item.id)
                if data is not None:
                    embs[item.id] = data[0]

            if not embs:
                clusters.append(group_items)
                continue

            # Simple agglomerative: greedily group items within threshold
            assigned: Set[str] = set()
            for item in group_items:
                if item.id in assigned:
                    continue
                cluster = [item]
                assigned.add(item.id)
                if item.id not in embs:
                    clusters.append(cluster)
                    continue
                for other in group_items:
                    if other.id in assigned or other.id not in embs:
                        continue
                    sim = cosine_similarity(embs[item.id], embs[other.id])
                    if sim >= (1.0 - threshold):  # high sim = low distance
                        cluster.append(other)
                        assigned.add(other.id)
                clusters.append(cluster)

        return clusters

    @staticmethod
    def _extract_doc(item: MemoryItem) -> str:
        """Extract source document name from provenance (empty if unknown)."""
        sid = item.provenance.source_id or ""
        if ":" in sid and "chunk" in sid:
            return sid.rsplit(":", 1)[0]
        return sid

    def _cluster_items_graph(
        self,
        items: List[MemoryItem],
        graph,
        depth: int = 2,
        max_size: int = 50,
    ) -> List[List[MemoryItem]]:
        """
        Graph-neighborhood-aware clustering with locality constraints.

        Two items are merged into the same component only if they are
        BFS neighbors AND share at least one locality signal:
          - same source document, OR
          - same primary tag, OR
          - embedding cosine >= 0.90

        This prevents cross-document contamination through entity hubs.
        Graph constrains candidates — it does not define clusters by itself.
        """
        item_map = {item.id: item for item in items}
        item_ids = list(item_map.keys())
        item_set = set(item_ids)

        # Pre-compute per-item metadata for locality checks
        item_doc: Dict[str, str] = {}
        item_tag: Dict[str, str] = {}
        for iid, item in item_map.items():
            item_doc[iid] = self._extract_doc(item)
            item_tag[iid] = item.tags[0].lower() if item.tags else ""

        # Pre-load embeddings for high-cosine locality check
        item_embs: Dict[str, List[float]] = {}
        for iid in item_ids:
            data = self._store.read_embedding(iid)
            if data is not None:
                item_embs[iid] = data[0]

        # Pre-compute BFS neighborhoods (intersected with our item set)
        neighborhoods: Dict[str, Set[str]] = {}
        for iid in item_ids:
            neighbor_ids = set(graph.neighborhood_items(iid, depth=depth, max_size=max_size))
            neighborhoods[iid] = neighbor_ids & item_set

        # Union-find with locality gate
        parent: Dict[str, str] = {iid: iid for iid in item_ids}

        def find(x: str) -> str:
            """Find root of x with path compression."""
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            """Merge components containing a and b."""
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def _shares_locality(a: str, b: str) -> bool:
            """Two items may merge only if they share a locality signal."""
            # Same source document
            da, db = item_doc.get(a, ""), item_doc.get(b, "")
            if da and db and da == db:
                return True
            # Same primary tag
            ta, tb = item_tag.get(a, ""), item_tag.get(b, "")
            if ta and tb and ta == tb:
                return True
            # High embedding cosine (>= 0.90)
            ea, eb = item_embs.get(a), item_embs.get(b)
            if ea is not None and eb is not None:
                sim = cosine_similarity(ea, eb)
                if sim >= 0.90:
                    return True
            return False

        # Build components: BFS neighbor + locality signal required
        for iid, nbrs in neighborhoods.items():
            for nbr in nbrs:
                if nbr in parent and _shares_locality(iid, nbr):
                    union(iid, nbr)

        # Group items by component
        components: Dict[str, List[MemoryItem]] = defaultdict(list)
        for iid in item_ids:
            components[find(iid)].append(item_map[iid])

        # Compute metrics
        comp_sizes = [len(c) for c in components.values()]
        singletons = sum(1 for s in comp_sizes if s == 1)
        same_doc_merges = 0
        multi_item_comps = [c for c in components.values() if len(c) > 1]
        for comp in multi_item_comps:
            docs = {self._extract_doc(i) for i in comp}
            docs.discard("")
            if len(docs) <= 1:
                same_doc_merges += 1

        logger.info(
            f"Graph clustering: {len(items)} items → "
            f"{len(components)} components "
            f"(largest={max(comp_sizes) if comp_sizes else 0}, "
            f"singletons={singletons}, "
            f"same_doc_ratio={same_doc_merges}/{len(multi_item_comps) or 1})"
        )

        # Within each component, apply standard coarse+fine clustering
        all_clusters: List[List[MemoryItem]] = []
        for comp_items in components.values():
            if len(comp_items) <= 1:
                all_clusters.append(comp_items)
            else:
                sub_clusters = self._cluster_items(comp_items)
                all_clusters.extend(sub_clusters)

        return all_clusters

    # -- Merging -----------------------------------------------------------

    def _merge_cluster(self, cluster: List[MemoryItem]) -> Optional[MemoryItem]:
        """
        Merge a cluster of items into a single canonical item.

        Tries LLM summarizer first, falls back to deterministic merge.
        """
        if not cluster:
            return None

        # Try LLM-based merge
        if self._config.enabled and not self._config.fallback_to_deterministic:
            merged = self._llm_merge(cluster)
            if merged is not None:
                return merged

        if self._config.fallback_to_deterministic or not self._config.enabled:
            return self._deterministic_merge(cluster)

        # Try LLM, fall back to deterministic
        merged = self._llm_merge(cluster)
        if merged is not None:
            return merged
        return self._deterministic_merge(cluster)

    def _compute_merge_confidence(self, cluster: List[MemoryItem]) -> float:
        """
        V31-3: Weighted blend of four confidence signals.

        Formula:
          merged = w_max * max_conf
                 + w_overlap * jaccard_overlap
                 + w_recency * recency_decay
                 + w_validation * validation_bonus

        Components:
          - max_conf: highest confidence among cluster members (backward compat)
          - jaccard_overlap: fraction of tags shared by >=2 members / total unique tags
          - recency_decay: exp(-lambda * days_old) using most recent updated_at
          - validation_bonus: 1.0 if any "verified", 0.0 if any "retracted", else 0.5

        Result is clamped to [0.0, 1.0].
        """
        cfg = self._config

        # --- Component 1: max confidence (legacy) ---
        max_conf = max(i.confidence for i in cluster)

        # --- Component 2: Jaccard tag overlap ---
        # Count how many unique tags appear in >=2 items
        tag_counts: Dict[str, int] = defaultdict(int)
        for item in cluster:
            seen_in_item: Set[str] = set()
            for tag in item.tags:
                key = tag.lower()
                if key not in seen_in_item:
                    tag_counts[key] += 1
                    seen_in_item.add(key)
        total_unique = len(tag_counts)
        if total_unique > 0:
            shared = sum(1 for count in tag_counts.values() if count >= 2)
            jaccard_overlap = shared / total_unique
        else:
            jaccard_overlap = 0.0

        # --- Component 3: recency decay ---
        now = datetime.now(timezone.utc)
        most_recent_dt = None
        for item in cluster:
            try:
                dt = datetime.fromisoformat(item.updated_at)
                if most_recent_dt is None or dt > most_recent_dt:
                    most_recent_dt = dt
            except (ValueError, TypeError):
                continue
        if most_recent_dt is not None:
            days_old = max((now - most_recent_dt).total_seconds() / 86400.0, 0.0)
            recency_decay = math.exp(-cfg.recency_lambda * days_old)
            recency_decay = min(recency_decay, 1.0)
        else:
            recency_decay = 0.5  # unknown age → neutral

        # --- Component 4: validation bonus ---
        validations = {i.validation for i in cluster}
        if "retracted" in validations:
            validation_bonus = 0.0
        elif "verified" in validations:
            validation_bonus = 1.0
        else:
            validation_bonus = 0.5  # all unverified (or contested)

        # --- Blend ---
        merged_conf = (
            cfg.merge_w_max * max_conf
            + cfg.merge_w_overlap * jaccard_overlap
            + cfg.merge_w_recency * recency_decay
            + cfg.merge_w_validation * validation_bonus
        )
        return max(0.0, min(1.0, merged_conf))

    def _deterministic_merge(self, cluster: List[MemoryItem]) -> MemoryItem:
        """
        Deterministic merge: pick best content, union tags/entities/provenance.
        """
        # Sort by confidence (desc), then content length (desc)
        ranked = sorted(
            cluster,
            key=lambda i: (i.confidence, len(i.content)),
            reverse=True,
        )
        best = ranked[0]

        # Merge tags, entities, provenance
        all_tags: List[str] = []
        all_entities: List[str] = []
        all_chunk_ids: List[str] = []
        all_hashes: List[str] = []
        seen_tags: Set[str] = set()
        seen_entities: Set[str] = set()

        # Find the best real source_id (not consolidation-sourced)
        best_source_id = ""
        for item in ranked:
            sid = item.provenance.source_id or ""
            if sid and not sid.startswith("consolidation:"):
                best_source_id = sid
                break
        if not best_source_id:
            best_source_id = best.provenance.source_id or f"consolidation:{_now_iso()}"

        for item in cluster:
            for tag in item.tags:
                if tag.lower() not in seen_tags:
                    all_tags.append(tag)
                    seen_tags.add(tag.lower())
            for ent in item.entities:
                if ent.lower() not in seen_entities:
                    all_entities.append(ent)
                    seen_entities.add(ent.lower())
            all_chunk_ids.extend(item.provenance.chunk_ids)
            all_hashes.extend(item.provenance.content_hashes)

        merged = MemoryItem(
            tier=best.tier,
            type=best.type,
            title=best.title,
            content=best.content,
            tags=all_tags,
            entities=all_entities,
            provenance=MemoryProvenance(
                source_kind="mixed",
                source_id=best_source_id,
                chunk_ids=list(set(all_chunk_ids)),
                content_hashes=list(set(all_hashes)),
            ),
            confidence=self._compute_merge_confidence(cluster),
            validation=best.validation,
            scope=best.scope,
            usage_count=sum(i.usage_count for i in cluster),
        )

        return merged

    def _llm_merge(self, cluster: List[MemoryItem]) -> Optional[MemoryItem]:
        """
        LLM-based merge using Granite 3.2B via Ollama.

        Returns None if LLM not available.
        """
        try:
            import requests
        except ImportError:
            return None

        items_text = "\n---\n".join(
            f"[{i.type}] {i.title}: {i.content}" for i in cluster
        )
        prompt = (
            "Merge the following memory items into a single concise canonical entry. "
            "Keep the most important information. Return JSON with keys: "
            "title, content, type (one of: fact, decision, definition, constraint, pattern, todo, pointer, note).\n\n"
            f"{items_text}"
        )

        try:
            resp = requests.post(
                f"{self._config.ollama_url}/api/generate",
                json={
                    "model": self._config.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")

            # Try to parse JSON from response
            import re
            json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                merged = self._deterministic_merge(cluster)
                merged.title = data.get("title", merged.title)
                merged.content = data.get("content", merged.content)
                if data.get("type") in {
                    "fact", "decision", "definition", "constraint",
                    "pattern", "todo", "pointer", "note",
                }:
                    merged.type = data["type"]
                return merged

        except Exception as e:
            logger.warning(f"LLM merge failed: {e}")

        return None

    # -- Merge quality metrics ----------------------------------------------

    def compute_merge_metrics(
        self, merge_chains: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Post-merge quality metrics:
          - avg/median/max component size
          - singleton count
          - same-doc merge ratio
          - dispersion: for each canonical, average graph distance to originals
          - high_dispersion_merges: merges with avg distance > threshold

        The dispersion metric detects over-merges where a canonical
        spans items with high graph diameter.
        """
        if not merge_chains:
            return {"chains": 0}

        sizes = [c["source_count"] for c in merge_chains]
        sorted_sizes = sorted(sizes)
        n = len(sorted_sizes)
        median_size = (
            sorted_sizes[n // 2]
            if n % 2 == 1
            else (sorted_sizes[n // 2 - 1] + sorted_sizes[n // 2]) / 2
        )

        # Same-doc analysis
        same_doc_count = 0
        for chain in merge_chains:
            docs = set()
            for sid in chain.get("source_ids", []):
                item = self._store.read_item(sid)
                if item:
                    doc = self._extract_doc(item)
                    if doc:
                        docs.add(doc)
            if len(docs) <= 1:
                same_doc_count += 1

        # Dispersion analysis (graph distance from canonical to sources)
        high_dispersion = []
        if self._graph is not None:
            for chain in merge_chains:
                merged_id = chain["merged_id"]
                source_ids = chain.get("source_ids", [])
                if len(source_ids) < 2:
                    continue

                # Compute distances from canonical to each source
                distances = []
                merged_node = f"item:{merged_id}"
                merged_nbrs = self._graph.neighbors(merged_node, depth=4, max_size=500)
                for sid in source_ids:
                    src_node = f"item:{sid}"
                    if src_node in merged_nbrs:
                        # BFS to find actual distance
                        for d in range(1, 5):
                            nbrs_d = self._graph.neighbors(merged_node, depth=d, max_size=500)
                            if src_node in nbrs_d:
                                distances.append(d)
                                break
                    else:
                        distances.append(5)  # unreachable = max penalty

                if distances:
                    avg_dist = sum(distances) / len(distances)
                    if avg_dist > 2.5:
                        high_dispersion.append({
                            "merged_id": merged_id,
                            "merged_title": chain.get("merged_title", "")[:80],
                            "source_count": len(source_ids),
                            "avg_distance": round(avg_dist, 2),
                        })

        return {
            "chains": len(merge_chains),
            "avg_chain_size": round(sum(sizes) / len(sizes), 1) if sizes else 0,
            "median_chain_size": median_size,
            "max_chain_size": max(sizes) if sizes else 0,
            "same_doc_ratio": f"{same_doc_count}/{len(merge_chains)}",
            "same_doc_pct": round(100 * same_doc_count / len(merge_chains), 1) if merge_chains else 0,
            "high_dispersion_merges": len(high_dispersion),
            "high_dispersion_details": high_dispersion[:10],  # cap for artifact size
        }

    # -- Promotion ---------------------------------------------------------

    def _try_promote(self, item: MemoryItem) -> bool:
        """
        Try to promote an item to a higher tier based on rules.

        Promotion to LTM requires:
          - usage_count >= threshold, OR
          - type in auto-promote list, OR
          - explicitly pinned (validation="verified")
        """
        if item.tier == "ltm":
            return False  # Already at highest tier

        eligible = False

        if item.usage_count >= self._config.usage_count_for_ltm:
            eligible = True
        elif item.type in self._config.auto_promote_types:
            eligible = True
        elif item.validation == "verified":
            eligible = True

        if not eligible:
            # STM -> MTM promotion (less strict)
            if item.tier == "stm" and item.usage_count >= 2:
                self._store.update_item(item.id, {"tier": "mtm", "injectable": True})
                return True
            return False

        target_tier = "ltm" if item.tier == "mtm" else "mtm"

        # Policy check: provenance required for MTM/LTM
        if target_tier in ("mtm", "ltm") and not item.provenance.source_id:
            logger.info(
                f"Cannot promote {item.id} to {target_tier}: missing provenance"
            )
            return False

        self._store.update_item(item.id, {"tier": target_tier, "injectable": True})
        logger.info(f"Promoted {item.id}: {item.tier} -> {target_tier}")
        return True
