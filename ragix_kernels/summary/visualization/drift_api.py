"""
Drift API — Load and format drift reports for visualization.

Reads drift_report.json and provides Sankey-compatible flow data
for the Drift Report view, plus item-level diff data for the
graphical diff viewer (V3.2).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_drift_report(workspace: Path) -> Optional[Dict[str, Any]]:
    """
    Load drift report from workspace.

    Returns None if no drift report exists.
    """
    path = workspace / "stage2" / "drift_report.json"
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def format_drift_sankey(drift: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert drift data to Sankey diagram format.

    Nodes: domains + classification categories
    Links: domain → classification with value = count
    """
    per_domain = drift.get("per_domain", {})
    if not per_domain:
        return {"nodes": [], "links": []}

    # Build node list: domains (left) + categories (right)
    categories = ["UNCHANGED", "MODIFIED", "ADDED", "REMOVED"]
    domains = sorted(per_domain.keys())

    nodes = []
    node_idx = {}

    for d in domains:
        node_idx[f"domain:{d}"] = len(nodes)
        nodes.append({"name": d.upper(), "group": "domain"})

    for c in categories:
        node_idx[f"cat:{c}"] = len(nodes)
        nodes.append({"name": c, "group": "category"})

    # Build links: domain → category with value
    links = []
    for domain, counts in per_domain.items():
        for cat in categories:
            val = counts.get(cat.lower(), 0)
            if val > 0:
                links.append({
                    "source": node_idx[f"domain:{domain}"],
                    "target": node_idx[f"cat:{cat}"],
                    "value": val,
                })

    return {"nodes": nodes, "links": links}


def format_drift_summary(drift: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format drift for the summary display (counts + per-domain table).
    """
    counts = drift.get("counts", {})
    per_domain = drift.get("per_domain", {})

    domain_rows = []
    for domain, dc in sorted(per_domain.items()):
        total = sum(dc.values())
        drift_pct = (
            (dc.get("added", 0) + dc.get("removed", 0) + dc.get("modified", 0))
            / max(total, 1) * 100
        )
        domain_rows.append({
            "domain": domain,
            "added": dc.get("added", 0),
            "removed": dc.get("removed", 0),
            "modified": dc.get("modified", 0),
            "unchanged": dc.get("unchanged", 0),
            "drift_pct": round(drift_pct, 1),
        })

    return {
        "counts": counts,
        "drift_pct": drift.get("drift_pct", 0),
        "domains": domain_rows,
    }


def _compute_line_diff(text_a: str, text_b: str) -> List[Dict[str, Any]]:
    """
    Compute line-level diff between two texts.

    Returns a list of diff operations:
      {"op": "equal"|"insert"|"delete"|"replace", "lines_a": [...], "lines_b": [...]}
    """
    lines_a = text_a.splitlines(keepends=True) if text_a else []
    lines_b = text_b.splitlines(keepends=True) if text_b else []
    sm = difflib.SequenceMatcher(None, lines_a, lines_b)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ops.append({
            "op": tag,
            "lines_a": [l.rstrip("\n\r") for l in lines_a[i1:i2]],
            "lines_b": [l.rstrip("\n\r") for l in lines_b[j1:j2]],
        })
    return ops


def format_modified_items(drift: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format modified items with line-level diffs for the graphical diff viewer.

    Each item includes:
      - id_a, id_b, title_a, title_b, domain, rule_id
      - diff: list of diff operations (equal/insert/delete/replace)
      - change_summary: {added_lines, removed_lines, changed_lines}
    """
    modified = drift.get("modified", [])
    items = []
    for m in modified:
        content_a = m.get("content_a", "")
        content_b = m.get("content_b", "")
        diff_ops = _compute_line_diff(content_a, content_b)

        # Compute summary metrics
        added = sum(len(op["lines_b"]) for op in diff_ops if op["op"] == "insert")
        removed = sum(len(op["lines_a"]) for op in diff_ops if op["op"] == "delete")
        replaced_a = sum(len(op["lines_a"]) for op in diff_ops if op["op"] == "replace")
        replaced_b = sum(len(op["lines_b"]) for op in diff_ops if op["op"] == "replace")

        items.append({
            "id_a": m.get("id_a", ""),
            "id_b": m.get("id_b", ""),
            "title_a": m.get("title_a", ""),
            "title_b": m.get("title_b", ""),
            "type_a": m.get("type_a", ""),
            "type_b": m.get("type_b", ""),
            "domain": m.get("domain", "unknown"),
            "rule_id": m.get("rule_id", ""),
            "diff": diff_ops,
            "change_summary": {
                "added_lines": added + replaced_b,
                "removed_lines": removed + replaced_a,
                "total_ops": len([op for op in diff_ops if op["op"] != "equal"]),
            },
        })
    return items
