"""
Memory API — Item listing with filters for visualization.

Converts MemoryStore data into table-friendly JSON for the
Memory Explorer view.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Secrecy-tier redaction helper
# ---------------------------------------------------------------------------

def _redact_field(text: str, tier: str) -> str:
    """Apply secrecy-tier redaction to a single text field."""
    if tier == "S3" or not text:
        return text
    from ragix_kernels.summary.kernels.summary_redact import redact_for_storage
    return redact_for_storage(text, tier)


def _redact_list(items: List[str], tier: str) -> List[str]:
    """Apply secrecy-tier redaction to a list of strings."""
    if tier == "S3" or not items:
        return items
    from ragix_kernels.summary.kernels.summary_redact import redact_for_storage
    return [redact_for_storage(s, tier) for s in items]


def export_memory_items(
    store,
    scope: Optional[str] = None,
    corpus_id: Optional[str] = None,
    tier: Optional[str] = None,
    type_filter: Optional[str] = None,
    secrecy_tier: str = "S3",
    limit: int = 2000,
) -> Dict[str, Any]:
    """
    Export memory items as a filterable JSON structure.

    Args:
        secrecy_tier: Secrecy level (S0/S2/S3). Non-S3 redacts sensitive fields.

    Returns:
        {"items": [...], "metadata": {...}}
    """
    items = store.list_items(
        scope=scope,
        corpus_id=corpus_id,
        tier=tier,
        type_filter=type_filter,
        exclude_archived=True,
        limit=limit,
    )

    rows = []
    for item in items:
        preview = item.content[:200] if item.content else ""
        rows.append({
            "id": item.id,
            "title": _redact_field(item.title, secrecy_tier),
            "type": item.type,
            "tier": item.tier,
            "scope": item.scope,
            "corpus_id": item.corpus_id,
            "rule_id": item.rule_id,
            "tags": _redact_list(item.tags, secrecy_tier),
            "entities": _redact_list(item.entities, secrecy_tier),
            "confidence": item.confidence,
            "validation": item.validation,
            "usage_count": item.usage_count,
            "source_id": _redact_field(item.provenance.source_id, secrecy_tier),
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "content_preview": _redact_field(preview, secrecy_tier),
        })

    stats = store.stats()

    return {
        "items": rows,
        "metadata": {
            "total_returned": len(rows),
            "total_active": stats.get("total_items", 0),
            "filters": {
                "scope": scope,
                "corpus_id": corpus_id,
                "tier": tier,
                "type": type_filter,
            },
            "by_tier": stats.get("by_tier", {}),
            "by_type": stats.get("by_type", {}),
        },
    }


def export_memory_query(
    store,
    scope: Optional[str] = None,
    corpus_id: Optional[str] = None,
    secrecy_tier: str = "S3",
    limit: int = 5000,
) -> Dict[str, Any]:
    """
    Export memory items for the Query Explorer.

    Args:
        secrecy_tier: Secrecy level (S0/S2/S3). Non-S3 redacts sensitive fields.

    Differences from export_memory_items():
        - Includes ``domain`` field per item (via domain_utils.extract_domain)
        - Includes full ``content`` (not just 200-char preview)
        - Adds ``metadata.domains`` list for populating dropdown
        - Adds ``metadata.tier_counts`` / ``metadata.type_counts`` for KPI cards
    """
    from ragix_kernels.summary.visualization.domain_utils import extract_domain

    items = store.list_items(
        scope=scope,
        corpus_id=corpus_id,
        exclude_archived=True,
        limit=limit,
    )

    rows = []
    domain_counter: Counter = Counter()
    tier_counter: Counter = Counter()
    type_counter: Counter = Counter()

    for item in items:
        domain = extract_domain(item)
        domain_counter[domain] += 1
        tier_counter[item.tier] += 1
        type_counter[item.type] += 1

        rows.append({
            "id": item.id,
            "title": _redact_field(item.title, secrecy_tier),
            "type": item.type,
            "tier": item.tier,
            "scope": item.scope,
            "domain": _redact_field(domain, secrecy_tier),
            "corpus_id": item.corpus_id,
            "rule_id": item.rule_id,
            "tags": _redact_list(item.tags, secrecy_tier),
            "entities": _redact_list(item.entities, secrecy_tier),
            "confidence": item.confidence,
            "validation": item.validation,
            "usage_count": item.usage_count,
            "source_id": _redact_field(item.provenance.source_id, secrecy_tier),
            "content": _redact_field(item.content or "", secrecy_tier),
            "created_at": item.created_at,
            "updated_at": item.updated_at,
        })

    return {
        "items": rows,
        "metadata": {
            "total": len(rows),
            "domains": sorted(domain_counter.keys()),
            "tier_counts": dict(tier_counter),
            "type_counts": dict(type_counter),
            "domain_counts": dict(domain_counter),
        },
    }
