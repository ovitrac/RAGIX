"""
Graph API — JSON export for D3.js visualization.

Converts GraphStore data into D3-compatible node-link format with
tier-aware filtering and layer toggles.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set


def export_graph_d3(
    graph_store,
    tier: str = "S3",
    node_kinds: Optional[Set[str]] = None,
    edge_kinds: Optional[Set[str]] = None,
    max_nodes: int = 2000,
) -> Dict[str, Any]:
    """
    Export graph as D3.js node-link JSON.

    Args:
        graph_store: GraphStore instance.
        tier: Secrecy tier for redaction (S0/S2/S3).
        node_kinds: Filter to these node kinds (default: all).
        edge_kinds: Filter to these edge kinds (default: all).
        max_nodes: Maximum nodes to export (prevent browser overload).

    Returns:
        D3 node-link format: {"nodes": [...], "links": [...], "metadata": {...}}
    """
    raw = graph_store.export_graph(tier=tier)

    nodes = raw.get("nodes", [])
    edges = raw.get("edges", [])

    # Filter by kind
    if node_kinds:
        nodes = [n for n in nodes if n["kind"] in node_kinds]
    if max_nodes and len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]

    node_ids = {n["id"] for n in nodes}

    if edge_kinds:
        edges = [e for e in edges if e["kind"] in edge_kinds]

    # Filter edges to only include connected nodes
    links = [
        e for e in edges
        if e["src"] in node_ids and e["dst"] in node_ids
    ]

    # Remove isolated nodes (no edges) to reduce clutter
    connected_ids = set()
    for e in links:
        connected_ids.add(e["src"])
        connected_ids.add(e["dst"])
    nodes = [n for n in nodes if n["id"] in connected_ids]
    node_ids = {n["id"] for n in nodes}

    # Convert to D3 format
    d3_nodes = []
    for n in nodes:
        d3_nodes.append({
            "id": n["id"],
            "kind": n["kind"],
            "label": n["label"],
            "item_id": n.get("item_id"),
            "group": _kind_to_group(n["kind"]),
        })

    d3_links = []
    for e in links:
        d3_links.append({
            "source": e["src"],
            "target": e["dst"],
            "kind": e["kind"],
            "weight": e.get("weight", 1.0),
        })

    stats = raw.get("stats", {})

    return {
        "nodes": d3_nodes,
        "links": d3_links,
        "metadata": {
            "tier": tier,
            "total_nodes": len(d3_nodes),
            "total_links": len(d3_links),
            "node_kinds": list(stats.get("nodes_by_kind", {}).keys()),
            "edge_kinds": list(stats.get("edges_by_kind", {}).keys()),
            "filtered_node_kinds": list(node_kinds) if node_kinds else None,
            "filtered_edge_kinds": list(edge_kinds) if edge_kinds else None,
        },
    }


def _kind_to_group(kind: str) -> int:
    """Map node kind to D3 color group index."""
    return {"doc": 0, "chunk": 1, "item": 2, "entity": 3}.get(kind, 4)
