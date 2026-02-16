"""
Metrics API — Smart metrics dashboard data.

Computes canonical stability, graph centrality, orphan count,
merge confidence, and per-domain health indicators.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from ragix_kernels.summary.visualization.domain_utils import extract_domain


def _make_redactor(tier: str) -> Callable[[str], str]:
    """Return a redaction function for the given secrecy tier."""
    if tier == "S3":
        return lambda text: text
    from ragix_kernels.summary.kernels.summary_redact import redact_for_storage
    return lambda text: redact_for_storage(text, tier)


def compute_metrics(
    store,
    graph_store=None,
    scope: Optional[str] = None,
    corpus_id: Optional[str] = None,
    secrecy_tier: str = "S3",
) -> Dict[str, Any]:
    """
    Compute smart metrics for the dashboard.

    Args:
        secrecy_tier: Secrecy level (S0/S2/S3). Non-S3 redacts domain/label fields.

    Returns:
        {
            "summary": {...},           # top-level KPIs
            "per_domain": [...],        # domain health rows
            "centrality": [...],        # top-N central items
            "tier_distribution": {...}, # stm/mtm/ltm counts
            "type_distribution": {...}, # by item type
            "metadata": {...},
        }
    """
    _redact = _make_redactor(secrecy_tier)

    items = store.list_items(
        scope=scope,
        corpus_id=corpus_id,
        exclude_archived=True,
        limit=5000,
    )

    stats = store.stats()

    # -- Per-domain health --
    domain_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0, "ltm": 0, "mtm": 0, "stm": 0,
        "verified": 0, "total_confidence": 0.0, "total_usage": 0,
        "entity_count": 0,
    })

    orphan_count = 0  # Items with no tags
    for item in items:
        domain = extract_domain(item)
        d = domain_data[domain]
        d["count"] += 1
        d[item.tier] = d.get(item.tier, 0) + 1
        if item.validation == "verified":
            d["verified"] += 1
        d["total_confidence"] += (item.confidence or 0)
        d["total_usage"] += (item.usage_count or 0)
        d["entity_count"] += len(item.entities or [])
        if not item.tags:
            orphan_count += 1

    domain_rows = []
    for domain in sorted(domain_data.keys()):
        d = domain_data[domain]
        count = d["count"]
        avg_conf = d["total_confidence"] / max(count, 1)
        stability = d["ltm"] / max(count, 1) * 100  # % promoted to LTM
        domain_rows.append({
            "domain": _redact(domain),
            "count": count,
            "ltm": d["ltm"],
            "mtm": d["mtm"],
            "stm": d["stm"],
            "verified": d["verified"],
            "avg_confidence": round(avg_conf, 3),
            "stability_pct": round(stability, 1),
            "avg_usage": round(d["total_usage"] / max(count, 1), 1),
            "entity_density": round(d["entity_count"] / max(count, 1), 2),
        })

    # -- Graph centrality (degree) --
    centrality_list = []
    if graph_store is not None:
        raw = graph_store.export_graph(tier=secrecy_tier)
        g_nodes = raw.get("nodes", [])
        g_edges = raw.get("edges", [])

        degree: Dict[str, int] = defaultdict(int)
        for e in g_edges:
            degree[e["src"]] += 1
            degree[e["dst"]] += 1

        # Map node_id → item info
        item_nodes = {n["id"]: n for n in g_nodes if n["kind"] == "item"}

        # Graph-level stats
        total_graph_nodes = len(g_nodes)
        total_graph_edges = len(g_edges)

        # Orphan nodes (degree 0 among items)
        connected_items = {nid for nid in item_nodes if degree.get(nid, 0) > 0}
        graph_orphans = len(item_nodes) - len(connected_items)

        # Top-N by degree
        sorted_items = sorted(
            item_nodes.keys(),
            key=lambda nid: degree.get(nid, 0),
            reverse=True,
        )
        for nid in sorted_items[:20]:
            node = item_nodes[nid]
            centrality_list.append({
                "node_id": nid,
                "label": _redact(node.get("label", "")),
                "item_id": node.get("item_id"),
                "degree": degree.get(nid, 0),
            })
    else:
        total_graph_nodes = 0
        total_graph_edges = 0
        graph_orphans = 0

    # -- Merge confidence (from consolidation events) --
    consol_events = store.read_events(action="consolidate", limit=200)
    merge_count = len(consol_events)

    # -- Summary KPIs --
    total = len(items)
    ltm_count = sum(1 for i in items if i.tier == "ltm")
    verified_count = sum(1 for i in items if i.validation == "verified")
    avg_confidence = sum(i.confidence or 0 for i in items) / max(total, 1)
    entity_coverage = sum(1 for i in items if i.entities) / max(total, 1) * 100

    summary = {
        "total_items": total,
        "total_domains": len(domain_data),
        "ltm_ratio": round(ltm_count / max(total, 1) * 100, 1),
        "verified_ratio": round(verified_count / max(total, 1) * 100, 1),
        "avg_confidence": round(avg_confidence, 3),
        "entity_coverage_pct": round(entity_coverage, 1),
        "orphan_items": orphan_count,
        "graph_nodes": total_graph_nodes,
        "graph_edges": total_graph_edges,
        "graph_orphans": graph_orphans,
        "merge_count": merge_count,
    }

    return {
        "summary": summary,
        "per_domain": domain_rows,
        "centrality": centrality_list,
        "tier_distribution": dict(stats.get("by_tier", {})),
        "type_distribution": dict(stats.get("by_type", {})),
        "metadata": {
            "scope": scope,
            "corpus_id": corpus_id,
        },
    }
