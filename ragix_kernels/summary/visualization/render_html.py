"""
HTML Renderer — Jinja2 → self-contained static HTML files.

Generates standalone HTML pages (no server required) with embedded
D3.js visualizations. All data is inlined as JSON in <script> tags.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_template(name: str) -> str:
    """Load a Jinja2 template from the templates directory."""
    path = _TEMPLATES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_template(template_str: str, context: Dict[str, Any]) -> str:
    """Render a Jinja2 template with context."""
    try:
        from jinja2 import Environment, BaseLoader
        env = Environment(loader=BaseLoader())
        template = env.from_string(template_str)
        return template.render(**context)
    except ImportError:
        # Fallback: simple string substitution for key variables
        result = template_str
        for key, value in context.items():
            if isinstance(value, str):
                result = result.replace(f"{{{{ {key} }}}}", value)
        return result


def render_graph_view(
    graph_data: Dict[str, Any],
    output_path: Path,
    title: str = "KOAS Memory Graph",
) -> Path:
    """
    Render interactive graph view as self-contained HTML.

    Args:
        graph_data: D3 node-link format from graph_api.export_graph_d3().
        output_path: Where to write the HTML file.
        title: Page title.

    Returns:
        Path to the generated HTML file.
    """
    template = _load_template("graph_view.html")
    html = _render_template(template, {
        "title": title,
        "graph_data_json": json.dumps(graph_data, ensure_ascii=False),
        "metadata": graph_data.get("metadata", {}),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Graph view: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def render_memory_explorer(
    memory_data: Dict[str, Any],
    output_path: Path,
    title: str = "KOAS Memory Explorer",
) -> Path:
    """
    Render memory explorer as self-contained HTML.

    Args:
        memory_data: From memory_api.export_memory_items().
        output_path: Where to write the HTML file.
        title: Page title.

    Returns:
        Path to the generated HTML file.
    """
    template = _load_template("memory_explorer.html")
    html = _render_template(template, {
        "title": title,
        "memory_data_json": json.dumps(memory_data, ensure_ascii=False),
        "metadata": memory_data.get("metadata", {}),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Memory explorer: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def render_query_explorer(
    query_data: Dict[str, Any],
    output_path: Path,
    title: str = "KOAS Memory Query",
) -> Path:
    """
    Render search-first query explorer as self-contained HTML.

    Args:
        query_data: From memory_api.export_memory_query().
        output_path: Where to write the HTML file.
        title: Page title.

    Returns:
        Path to the generated HTML file.
    """
    template = _load_template("query_explorer.html")
    html = _render_template(template, {
        "title": title,
        "query_data_json": json.dumps(query_data, ensure_ascii=False),
        "metadata": query_data.get("metadata", {}),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Query explorer: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def render_drift_report(
    drift_data: Dict[str, Any],
    sankey_data: Dict[str, Any],
    output_path: Path,
    title: str = "KOAS Drift Report",
    modified_items: Optional[list] = None,
) -> Path:
    """
    Render drift report as self-contained HTML.

    Args:
        drift_data: From drift_api.format_drift_summary().
        sankey_data: From drift_api.format_drift_sankey().
        output_path: Where to write the HTML file.
        title: Page title.
        modified_items: From drift_api.format_modified_items() — item-level diffs.

    Returns:
        Path to the generated HTML file.
    """
    template = _load_template("drift_report.html")
    html = _render_template(template, {
        "title": title,
        "drift_data_json": json.dumps(drift_data, ensure_ascii=False),
        "sankey_data_json": json.dumps(sankey_data, ensure_ascii=False),
        "modified_items_json": json.dumps(modified_items or [], ensure_ascii=False),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Drift report: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def render_timeline(
    timeline_data: Dict[str, Any],
    output_path: Path,
    title: str = "KOAS Corpus Timeline",
) -> Path:
    """Render timeline slider as self-contained HTML."""
    template = _load_template("timeline.html")
    html = _render_template(template, {
        "title": title,
        "timeline_data_json": json.dumps(timeline_data, ensure_ascii=False),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Timeline: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def render_cluster_view(
    cluster_data: Dict[str, Any],
    output_path: Path,
    title: str = "KOAS Topic Clusters",
) -> Path:
    """Render topic clustering bubble chart as self-contained HTML."""
    template = _load_template("cluster_view.html")
    html = _render_template(template, {
        "title": title,
        "cluster_data_json": json.dumps(cluster_data, ensure_ascii=False),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Cluster view: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def render_heatmap(
    heatmap_data: Dict[str, Any],
    output_path: Path,
    title: str = "KOAS Drift Heatmap",
) -> Path:
    """Render domain x version drift heatmap as self-contained HTML."""
    template = _load_template("heatmap.html")
    html = _render_template(template, {
        "title": title,
        "heatmap_data_json": json.dumps(heatmap_data, ensure_ascii=False),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Heatmap: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def render_metrics_dashboard(
    metrics_data: Dict[str, Any],
    output_path: Path,
    title: str = "KOAS Smart Metrics",
) -> Path:
    """Render smart metrics dashboard as self-contained HTML."""
    template = _load_template("metrics_dashboard.html")
    html = _render_template(template, {
        "title": title,
        "metrics_data_json": json.dumps(metrics_data, ensure_ascii=False),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Metrics dashboard: {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


_ALL_VIEWS = {"graph", "memory", "query", "drift", "timeline", "clusters", "heatmap", "metrics"}


def render_all(
    workspace: Path,
    store=None,
    graph_store=None,
    tier: str = "S3",
    scope: Optional[str] = None,
    corpus_id: Optional[str] = None,
    views: Optional[Set[str]] = None,
) -> Dict[str, Path]:
    """
    Render visualization outputs for a workspace.

    Args:
        views: Set of view names to render (default: all).
               Valid names: graph, memory, query, drift, timeline,
               clusters, heatmap, metrics.

    Returns dict mapping view name → output path.
    """
    selected = views if views else _ALL_VIEWS
    from ragix_kernels.summary.visualization.graph_api import export_graph_d3
    from ragix_kernels.summary.visualization.memory_api import export_memory_items, export_memory_query
    from ragix_kernels.summary.visualization.drift_api import (
        load_drift_report,
        format_drift_sankey,
        format_drift_summary,
        format_modified_items,
    )
    from ragix_kernels.summary.visualization.timeline_api import build_timeline
    from ragix_kernels.summary.visualization.cluster_api import detect_communities
    from ragix_kernels.summary.visualization.heatmap_api import build_drift_heatmap
    from ragix_kernels.summary.visualization.metrics_api import compute_metrics

    output_dir = workspace / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # V3.0-B: Graph view
    if graph_store is not None and "graph" in selected:
        graph_data = export_graph_d3(graph_store, tier=tier)
        path = render_graph_view(graph_data, output_dir / "graph.html")
        outputs["graph"] = path

    # V3.0-B: Memory explorer
    if store is not None and "memory" in selected:
        memory_data = export_memory_items(
            store, scope=scope, corpus_id=corpus_id,
            secrecy_tier=tier,
        )
        path = render_memory_explorer(memory_data, output_dir / "memory.html")
        outputs["memory"] = path

    # V3.0-D: Query explorer
    if store is not None and "query" in selected:
        query_data = export_memory_query(
            store, scope=scope, corpus_id=corpus_id,
            secrecy_tier=tier,
        )
        path = render_query_explorer(query_data, output_dir / "query.html")
        outputs["query"] = path

    # V3.0-B: Drift report (V3.2: graphical diff viewer)
    drift_raw = load_drift_report(workspace)
    if drift_raw is not None and "drift" in selected:
        summary = format_drift_summary(drift_raw)
        sankey = format_drift_sankey(drift_raw)
        mod_items = format_modified_items(drift_raw)
        path = render_drift_report(
            summary, sankey, output_dir / "drift.html",
            modified_items=mod_items,
        )
        outputs["drift"] = path

    # V3.0-C: Timeline
    if store is not None and "timeline" in selected:
        timeline_data = build_timeline(
            store, scope=scope, corpus_id=corpus_id,
            secrecy_tier=tier,
        )
        if timeline_data.get("bins"):
            path = render_timeline(timeline_data, output_dir / "timeline.html")
            outputs["timeline"] = path

    # V3.0-C: Topic clustering
    if graph_store is not None and "clusters" in selected:
        cluster_data = detect_communities(
            graph_store, store=store,
            secrecy_tier=tier,
        )
        if cluster_data.get("clusters"):
            path = render_cluster_view(cluster_data, output_dir / "clusters.html")
            outputs["clusters"] = path

    # V3.0-C: Drift heatmap
    if "heatmap" not in selected:
        heatmap_data = {"cells": []}
    else:
        heatmap_data = build_drift_heatmap(
            workspace, store=store,
            secrecy_tier=tier,
        )
    if heatmap_data.get("cells"):
        path = render_heatmap(heatmap_data, output_dir / "heatmap.html")
        outputs["heatmap"] = path

    # V3.0-C: Smart metrics
    if store is not None and "metrics" in selected:
        metrics_data = compute_metrics(
            store, graph_store=graph_store,
            scope=scope, corpus_id=corpus_id,
            secrecy_tier=tier,
        )
        path = render_metrics_dashboard(metrics_data, output_dir / "metrics.html")
        outputs["metrics"] = path

    logger.info(f"Rendered {len(outputs)} visualizations → {output_dir}")
    return outputs
