"""
Cluster API — Deterministic community detection for topic clustering view.

Uses connected-component analysis on similarity edges, then refines with
label propagation on the same edge set. No external dependencies.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

from ragix_kernels.summary.visualization.domain_utils import extract_domain


def _make_redactor(tier: str) -> Callable[[str], str]:
    """Return a redaction function for the given secrecy tier."""
    if tier == "S3":
        return lambda text: text
    from ragix_kernels.summary.kernels.summary_redact import redact_for_storage
    return lambda text: redact_for_storage(text, tier)


def _redact_dict_keys(d: Dict[str, int], redact: Callable[[str], str]) -> Dict[str, int]:
    """Redact dictionary keys while preserving values."""
    out: Dict[str, int] = {}
    for k, v in d.items():
        rk = redact(k)
        out[rk] = out.get(rk, 0) + v
    return out


def detect_communities(
    graph_store,
    store=None,
    edge_kinds: Optional[Set[str]] = None,
    min_weight: float = 0.3,
    max_iterations: int = 20,
    secrecy_tier: str = "S3",
) -> Dict[str, Any]:
    """
    Detect topic clusters via label propagation on graph edges.

    Args:
        graph_store: GraphStore instance.
        store: Optional MemoryStore for enriching cluster data.
        edge_kinds: Edge types to consider (default: similarity + mentions).
        min_weight: Minimum edge weight threshold.
        max_iterations: Label propagation iterations.
        secrecy_tier: Secrecy level (S0/S2/S3). Non-S3 redacts labels/domains.

    Returns:
        {"clusters": [...], "nodes": [...], "metadata": {...}}
    """
    _redact = _make_redactor(secrecy_tier)

    if edge_kinds is None:
        edge_kinds = {"similar", "mentions", "same_rule", "evolved_from"}

    raw = graph_store.export_graph(tier=secrecy_tier)
    nodes = raw.get("nodes", [])
    edges = raw.get("edges", [])

    # Filter to item nodes only (topics are about items, not chunks/docs)
    item_nodes = {n["id"]: n for n in nodes if n["kind"] == "item"}
    if not item_nodes:
        return {"clusters": [], "nodes": [], "metadata": {"total_clusters": 0}}

    # Pre-load store items for enrichment
    item_map = {}
    if store:
        all_items = store.list_items(exclude_archived=True, limit=5000)
        for it in all_items:
            item_map[it.id] = it

    # Build adjacency list (item→item) from relevant edges
    adj: Dict[str, List[tuple]] = defaultdict(list)
    for e in edges:
        if e["kind"] not in edge_kinds:
            continue
        w = e.get("weight", 1.0)
        if w < min_weight:
            continue
        src, dst = e["src"], e["dst"]
        if src in item_nodes and dst in item_nodes:
            adj[src].append((dst, w))
            adj[dst].append((src, w))

    # Label propagation
    labels = {nid: i for i, nid in enumerate(item_nodes)}
    for _ in range(max_iterations):
        changed = False
        for nid in item_nodes:
            if nid not in adj:
                continue
            # Weighted vote
            votes: Dict[int, float] = defaultdict(float)
            for neighbor, weight in adj[nid]:
                votes[labels[neighbor]] += weight
            if votes:
                best = max(votes, key=votes.get)
                if best != labels[nid]:
                    labels[nid] = best
                    changed = True
        if not changed:
            break

    # Group by label
    clusters_map: Dict[int, List[str]] = defaultdict(list)
    for nid, label in labels.items():
        clusters_map[label].append(nid)

    # Build output — enrich from store
    clusters = []
    for label, members in sorted(clusters_map.items(), key=lambda x: -len(x[1])):
        cluster_tags: Dict[str, int] = defaultdict(int)
        cluster_domains: Dict[str, int] = defaultdict(int)

        for nid in members:
            node = item_nodes[nid]
            item = item_map.get(node.get("item_id"))
            if item:
                for t in (item.tags or []):
                    cluster_tags[t] += 1
                cluster_domains[extract_domain(item)] += 1
            else:
                # Fallback: use node label words
                for word in (node.get("label", "")).split()[:3]:
                    cluster_tags[word.lower()] += 1

        # Determine cluster name: prefer top domain, fallback top tag
        if cluster_domains:
            top_name = max(cluster_domains, key=cluster_domains.get)
        elif cluster_tags:
            top_name = max(cluster_tags, key=cluster_tags.get)
        else:
            top_name = f"cluster-{label}"

        clusters.append({
            "id": label,
            "name": _redact(top_name),
            "size": len(members),
            "members": members,
            "tags": _redact_dict_keys(dict(sorted(cluster_tags.items(), key=lambda x: -x[1])[:10]), _redact),
            "domains": _redact_dict_keys(dict(sorted(cluster_domains.items(), key=lambda x: -x[1])[:5]), _redact),
        })

    # Build enriched node list
    node_list = []
    for nid, node in item_nodes.items():
        item = item_map.get(node.get("item_id"))
        entry = {
            "id": nid,
            "cluster": labels[nid],
            "label": _redact(node.get("label", "")),
            "item_id": node.get("item_id"),
        }
        if item:
            entry.update({
                "title": _redact(item.title),
                "tags": [_redact(t) for t in (item.tags or [])],
                "type": item.type,
                "domain": _redact(extract_domain(item)),
            })
        node_list.append(entry)

    # Filter out singletons for the output (keep metadata count)
    singleton_count = sum(1 for c in clusters if c["size"] == 1)
    visible_clusters = [c for c in clusters if c["size"] > 1]

    return {
        "clusters": visible_clusters,
        "nodes": node_list,
        "metadata": {
            "total_clusters": len(visible_clusters),
            "total_items": len(item_nodes),
            "singletons": singleton_count,
            "iterations": max_iterations,
            "edge_kinds": list(edge_kinds),
            "min_weight": min_weight,
        },
    }
