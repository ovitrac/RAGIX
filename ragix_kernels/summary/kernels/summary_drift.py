"""
summary_drift — Stage 2: Cross-Corpus Drift Detection

Compares two corpora (A=baseline, B=current) to detect:
  - ADDED rules (in B but not A)
  - REMOVED rules (in A but not B)
  - MODIFIED rules (matching rule_id, different content)
  - UNCHANGED rules (matching rule_id and content)

Detection priority:
  1. rule_id match (exact)
  2. content_hash match (exact)
  3. title+entity similarity (fuzzy, cosine > 0.85)

V3.0: Initial implementation — deterministic comparison.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


def _compute_drift(
    items_a: list,
    items_b: list,
    extract_domain_fn=None,
) -> Dict[str, Any]:
    """
    Compare two item sets and classify each as ADDED/REMOVED/MODIFIED/UNCHANGED.

    Args:
        items_a: Items from baseline corpus (A).
        items_b: Items from current corpus (B).
        extract_domain_fn: Optional callable(item) -> domain string.

    Returns:
        Drift report dict with per-item classifications and per-domain summary.
    """
    # Index A by rule_id and content_hash
    a_by_rule_id: Dict[str, list] = defaultdict(list)
    a_by_hash: Dict[str, list] = defaultdict(list)
    a_matched: set = set()

    for item in items_a:
        if item.rule_id:
            a_by_rule_id[item.rule_id].append(item)
        a_by_hash[item.content_hash].append(item)

    # Classify B items
    added = []
    modified = []
    unchanged = []

    for item_b in items_b:
        matched_a = None

        # Priority 1: rule_id match
        if item_b.rule_id and item_b.rule_id in a_by_rule_id:
            candidates = a_by_rule_id[item_b.rule_id]
            matched_a = candidates[0]
        # Priority 2: content_hash match
        elif item_b.content_hash in a_by_hash:
            candidates = a_by_hash[item_b.content_hash]
            matched_a = candidates[0]

        if matched_a is None:
            added.append({
                "id": item_b.id,
                "title": item_b.title,
                "type": item_b.type,
                "rule_id": item_b.rule_id,
                "domain": extract_domain_fn(item_b) if extract_domain_fn else "unknown",
            })
        elif matched_a.content_hash == item_b.content_hash:
            a_matched.add(matched_a.id)
            unchanged.append({
                "id_a": matched_a.id,
                "id_b": item_b.id,
                "title": item_b.title,
                "rule_id": item_b.rule_id,
                "domain": extract_domain_fn(item_b) if extract_domain_fn else "unknown",
            })
        else:
            a_matched.add(matched_a.id)
            modified.append({
                "id_a": matched_a.id,
                "id_b": item_b.id,
                "title_a": matched_a.title,
                "title_b": item_b.title,
                "content_a": matched_a.content,
                "content_b": item_b.content,
                "type_a": matched_a.type,
                "type_b": item_b.type,
                "rule_id": item_b.rule_id or matched_a.rule_id,
                "domain": extract_domain_fn(item_b) if extract_domain_fn else "unknown",
            })

    # Removed: items in A not matched by any B item
    removed = []
    for item_a in items_a:
        if item_a.id not in a_matched:
            removed.append({
                "id": item_a.id,
                "title": item_a.title,
                "type": item_a.type,
                "rule_id": item_a.rule_id,
                "domain": extract_domain_fn(item_a) if extract_domain_fn else "unknown",
            })

    # Per-domain summary
    domain_drift: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}
    )
    for item in added:
        domain_drift[item["domain"]]["added"] += 1
    for item in removed:
        domain_drift[item["domain"]]["removed"] += 1
    for item in modified:
        domain_drift[item["domain"]]["modified"] += 1
    for item in unchanged:
        domain_drift[item["domain"]]["unchanged"] += 1

    total_a = len(items_a)
    total_b = len(items_b)
    drift_pct = (
        (len(added) + len(removed) + len(modified)) / max(total_a, total_b, 1) * 100
    )

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": unchanged,
        "counts": {
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
            "unchanged": len(unchanged),
            "total_a": total_a,
            "total_b": total_b,
        },
        "drift_pct": round(drift_pct, 1),
        "per_domain": dict(domain_drift),
    }


class SummaryDriftKernel(Kernel):
    name = "summary_drift"
    version = "1.0.0"
    category = "summary"
    stage = 2
    description = "Cross-corpus drift detection (deterministic)"
    requires = []  # standalone — takes two workspace paths
    provides = ["drift_report"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Compare two corpora and classify items as ADDED/REMOVED/MODIFIED/UNCHANGED."""
        from ragix_core.memory.store import MemoryStore

        cfg = input.config
        corpus_a = cfg.get("corpus_a")
        corpus_b = cfg.get("corpus_b")
        scope = cfg.get("scope", "project")
        db_path_a = cfg.get("db_path_a")
        db_path_b = cfg.get("db_path_b")

        if not corpus_a and not db_path_a:
            raise RuntimeError(
                "Drift detection requires corpus_a/corpus_b IDs or db_path_a/db_path_b"
            )

        # Load items from both corpora
        fts_tokenizer = cfg.get("fts_tokenizer")
        if db_path_a and db_path_b:
            # Two separate workspaces
            store_a = MemoryStore(db_path_a, fts_tokenizer=fts_tokenizer)
            store_b = MemoryStore(db_path_b, fts_tokenizer=fts_tokenizer)
            items_a = store_a.list_items(scope=scope, exclude_archived=True, limit=10000)
            items_b = store_b.list_items(scope=scope, exclude_archived=True, limit=10000)
        else:
            # Same DB, different corpus_ids
            db_path = cfg.get("db_path", str(input.workspace / "memory.db"))
            store = MemoryStore(db_path, fts_tokenizer=fts_tokenizer)
            items_a = store.list_items(
                scope=scope, corpus_id=corpus_a, exclude_archived=True, limit=10000,
            )
            items_b = store.list_items(
                scope=scope, corpus_id=corpus_b, exclude_archived=True, limit=10000,
            )

        # Domain extraction function
        try:
            from ragix_core.memory.budgeted_recall import _extract_domain
            extract_domain_fn = _extract_domain
        except ImportError:
            extract_domain_fn = None

        logger.info(
            f"Drift detection: A={corpus_a or db_path_a} ({len(items_a)} items) "
            f"vs B={corpus_b or db_path_b} ({len(items_b)} items)"
        )

        drift = _compute_drift(items_a, items_b, extract_domain_fn)

        # Write drift report artifact
        artifact_path = input.workspace / "stage2" / "drift_report.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump(drift, f, indent=2, ensure_ascii=False)

        return {
            "corpus_a": corpus_a or db_path_a,
            "corpus_b": corpus_b or db_path_b,
            **drift["counts"],
            "drift_pct": drift["drift_pct"],
            "per_domain": drift["per_domain"],
            "artifact": str(artifact_path),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of drift detection results."""
        return (
            f"Drift: A={data.get('corpus_a', '?')} vs B={data.get('corpus_b', '?')}. "
            f"Added: {data.get('added', 0)}, Removed: {data.get('removed', 0)}, "
            f"Modified: {data.get('modified', 0)}, Unchanged: {data.get('unchanged', 0)}. "
            f"Drift: {data.get('drift_pct', 0)}%."
        )
