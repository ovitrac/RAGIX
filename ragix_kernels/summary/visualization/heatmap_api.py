"""
Heatmap API — Domain x Version drift intensity matrix.

Builds a heatmap-compatible matrix from drift reports across corpora.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ragix_kernels.summary.visualization.domain_utils import extract_domain


def _make_redactor(tier: str) -> Callable[[str], str]:
    """Return a redaction function for the given secrecy tier."""
    if tier == "S3":
        return lambda text: text
    from ragix_kernels.summary.kernels.summary_redact import redact_for_storage
    return lambda text: redact_for_storage(text, tier)


def build_drift_heatmap(
    workspace: Path,
    store=None,
    secrecy_tier: str = "S3",
) -> Dict[str, Any]:
    """
    Build domain x corpus drift heatmap from available drift reports.

    If multiple corpus versions exist, computes per-domain drift intensity
    across versions. Falls back to single-version domain density if only
    one corpus is available.

    Args:
        secrecy_tier: Secrecy level (S0/S2/S3). Non-S3 redacts domain labels.

    Returns:
        {"rows": [...], "columns": [...], "cells": [...], "metadata": {...}}
    """
    _redact = _make_redactor(secrecy_tier)

    # Collect all drift reports in workspace
    drift_files = sorted(workspace.glob("**/drift_report.json"))
    domains_set: set = set()
    corpora_set: set = set()
    drift_matrix: Dict[str, Dict[str, float]] = {}  # domain → {corpus → drift_pct}

    for df in drift_files:
        try:
            with open(df) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        corpus_label = data.get("corpus_b", data.get("corpus_a", df.parent.name))
        corpora_set.add(corpus_label)

        per_domain = data.get("per_domain", {})
        for domain, counts in per_domain.items():
            domains_set.add(domain)
            total = sum(counts.values())
            drift_pct = (
                (counts.get("added", 0) + counts.get("removed", 0) + counts.get("modified", 0))
                / max(total, 1) * 100
            )
            if domain not in drift_matrix:
                drift_matrix[domain] = {}
            drift_matrix[domain][corpus_label] = round(drift_pct, 1)

    # If no drift data, fall back to item density heatmap
    if not drift_matrix and store:
        return _build_density_heatmap(store, _redact)

    rows = sorted(domains_set)
    columns = sorted(corpora_set)

    cells = []
    for r, domain in enumerate(rows):
        for c, corpus in enumerate(columns):
            val = drift_matrix.get(domain, {}).get(corpus, 0)
            cells.append({
                "row": r,
                "col": c,
                "domain": _redact(domain),
                "corpus": corpus,
                "value": val,
            })

    return {
        "rows": [_redact(d) for d in rows],
        "columns": columns,
        "cells": cells,
        "metadata": {
            "type": "drift",
            "total_domains": len(rows),
            "total_corpora": len(columns),
        },
    }


def _build_density_heatmap(store, _redact: Callable[[str], str]) -> Dict[str, Any]:
    """
    Fallback: domain x type item density heatmap.

    When no drift data exists, shows distribution of items across
    domains and types as a heatmap.
    """
    items = store.list_items(exclude_archived=True, limit=5000)

    domains_set: set = set()
    types_set: set = set()
    density: Dict[str, Dict[str, int]] = {}

    for item in items:
        domain = extract_domain(item)
        domains_set.add(domain)
        types_set.add(item.type)
        if domain not in density:
            density[domain] = {}
        density[domain][item.type] = density[domain].get(item.type, 0) + 1

    rows = sorted(domains_set)
    columns = sorted(types_set)

    cells = []
    for r, domain in enumerate(rows):
        for c, item_type in enumerate(columns):
            val = density.get(domain, {}).get(item_type, 0)
            cells.append({
                "row": r,
                "col": c,
                "domain": _redact(domain),
                "type": item_type,
                "value": val,
            })

    return {
        "rows": [_redact(d) for d in rows],
        "columns": columns,
        "cells": cells,
        "metadata": {
            "type": "density",
            "total_domains": len(rows),
            "total_types": len(columns),
        },
    }
