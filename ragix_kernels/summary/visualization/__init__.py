"""
KOAS Summary Visualization — Static HTML exports for memory exploration.

V3.0-B modules:
    graph_api    — JSON graph export with tier-aware filtering
    memory_api   — Item listing with filters
    drift_api    — Drift report JSON + Sankey format

V3.0-C modules:
    timeline_api — Temporal corpus evolution data
    cluster_api  — Deterministic community detection (label propagation)
    heatmap_api  — Domain x version drift intensity matrix
    metrics_api  — Smart metrics dashboard data

    render_html  — Jinja2 → self-contained static HTML rendering

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""
