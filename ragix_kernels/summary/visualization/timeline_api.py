"""
Timeline API — Temporal corpus evolution data for the Timeline Slider view.

Queries items by creation time and events to produce an animatable timeline.

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


def build_timeline(
    store,
    scope: Optional[str] = None,
    corpus_id: Optional[str] = None,
    bin_by: str = "day",
    secrecy_tier: str = "S3",
) -> Dict[str, Any]:
    """
    Build timeline data from memory items and events.

    Args:
        store: MemoryStore instance.
        scope: Optional scope filter.
        corpus_id: Optional corpus filter.
        bin_by: Temporal bin size ("day", "week", "month").
        secrecy_tier: Secrecy level (S0/S2/S3). Non-S3 redacts domain labels.

    Returns:
        {"bins": [...], "events": [...], "metadata": {...}}
    """
    _redact = _make_redactor(secrecy_tier)

    items = store.list_items(
        scope=scope,
        corpus_id=corpus_id,
        exclude_archived=True,
        limit=5000,
    )

    # Bin items by creation date
    bins = defaultdict(lambda: {"added": 0, "types": defaultdict(int), "domains": defaultdict(int)})

    for item in items:
        ts = item.created_at or ""
        key = _bin_key(ts, bin_by)
        if not key:
            continue
        bins[key]["added"] += 1
        bins[key]["types"][item.type] += 1
        bins[key]["domains"][_redact(extract_domain(item))] += 1

    # Sort bins chronologically
    sorted_keys = sorted(bins.keys())
    cumulative = 0
    bin_list = []
    for key in sorted_keys:
        b = bins[key]
        cumulative += b["added"]
        bin_list.append({
            "date": key,
            "added": b["added"],
            "cumulative": cumulative,
            "types": dict(b["types"]),
            "domains": dict(b["domains"]),
        })

    # Get recent events for annotation
    events = store.read_events(limit=200)
    event_list = []
    for ev in events:
        event_list.append({
            "timestamp": getattr(ev, "timestamp", ""),
            "action": getattr(ev, "action", ""),
            "item_id": getattr(ev, "item_id", ""),
            "details": _redact(str(getattr(ev, "details", ""))),
        })

    return {
        "bins": bin_list,
        "events": event_list[:100],  # Cap for UI
        "metadata": {
            "total_items": len(items),
            "bin_by": bin_by,
            "date_range": [sorted_keys[0], sorted_keys[-1]] if sorted_keys else [],
            "scope": scope,
            "corpus_id": corpus_id,
        },
    }


def _bin_key(iso_ts: str, bin_by: str) -> str:
    """Extract bin key from ISO timestamp."""
    if len(iso_ts) < 10:
        return ""
    date_part = iso_ts[:10]  # "YYYY-MM-DD"
    if bin_by == "month":
        return date_part[:7]  # "YYYY-MM"
    elif bin_by == "week":
        # Round to Monday
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(date_part)
            monday = dt.toordinal() - dt.weekday()
            return datetime.fromordinal(monday).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return ""
    return date_part  # day
