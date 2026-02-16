"""
BenchmarkScenario — performance benchmark report.

Sections: timed search batch, timed recall batch, consolidation, stats, metrics.
"""

from __future__ import annotations

import time

from ragix_core.memory.reporting.renderer import MarkdownRenderer
from ragix_core.memory.reporting.scenarios import ScenarioBase, register_scenario

if False:  # TYPE_CHECKING
    from ragix_core.memory.reporting.engine import ReportEngine


@register_scenario
class BenchmarkScenario(ScenarioBase):

    @property
    def id(self) -> str:
        return "benchmarks"

    @property
    def default_config_name(self) -> str:
        return "benchmarks.yml"

    def run(self, engine: "ReportEngine", config: dict) -> str:
        meta = config.get("meta", {})
        md = MarkdownRenderer(
            title=meta.get("title", "RAGIX Memory — Benchmark Report"),
            author="Olivier Vitrac, PhD, HDR | Adservio Innovation Lab",
        )
        md.header(meta, engine)

        search_cfg = config.get("search", {})
        recall_cfg = config.get("recall", {})
        thresholds = config.get("thresholds", {})

        # §1 Search benchmark
        queries = search_cfg.get("queries", [])
        k = search_cfg.get("k", 5)
        search_rows = []
        search_times = []

        for q in queries:
            t0 = time.perf_counter()
            result = engine.tool(
                "memory_search",
                query=q,
                workspace=engine.workspace,
                k=k,
            )
            dt = (time.perf_counter() - t0) * 1000
            search_times.append(dt)

            items = result.get("items", [])
            top_title = items[0].get("title", "")[:50] if items else ""
            search_rows.append({
                "Query": q,
                "Results": result.get("count", 0),
                "Top Hit": top_title,
                "ms": f"{dt:.1f}",
            })

        md.section_benchmark_table(
            engine,
            title=f"Search Benchmark ({len(queries)} queries)",
            columns=["Query", "Results", "Top Hit", "ms"],
            rows=search_rows,
        )

        if search_times:
            avg_search = sum(search_times) / len(search_times)
            md.w(f"**Avg search latency**: {avg_search:.1f} ms")
            md.w()
            engine._timings["avg_search"] = avg_search

        # §2 Recall benchmark
        recall_runs = recall_cfg.get("runs", [])
        recall_rows = []
        recall_times = []

        for run in recall_runs:
            q = run.get("query", "")
            budget = run.get("budget_tokens", 1500)

            t0 = time.perf_counter()
            result = engine.tool(
                "memory_recall",
                query=q,
                budget_tokens=budget,
                workspace=engine.workspace,
            )
            dt = (time.perf_counter() - t0) * 1000
            recall_times.append(dt)

            inject = result.get("inject_text", "")
            recall_rows.append({
                "Query": q[:50],
                "Budget": budget,
                "Used": result.get("tokens_used", 0),
                "Matched": result.get("matched", 0),
                "Inject": f"{len(inject)}c",
                "ms": f"{dt:.1f}",
            })

            # Assert format version
            fmt_ver = thresholds.get("format_version", 1)
            if inject:
                engine.assert_format_version(inject, fmt_ver)

        md.section_benchmark_table(
            engine,
            title=f"Recall Benchmark ({len(recall_runs)} budgets)",
            columns=["Query", "Budget", "Used", "Matched", "Inject", "ms"],
            rows=recall_rows,
        )

        if recall_times:
            avg_recall = sum(recall_times) / len(recall_times)
            md.w(f"**Avg recall latency**: {avg_recall:.1f} ms")
            md.w()
            engine._timings["avg_recall"] = avg_recall

        # §3 Consolidation
        with engine.timed("consolidation"):
            engine.tool("memory_consolidate", workspace=engine.workspace)

        # §4 Stats
        md.section_stats(engine)

        # §5 Timing summary
        md.w("## Timing Summary")
        md.w()
        md.w("| Phase | ms |")
        md.w("|-------|:--:|")
        for label, ms in engine.timings().items():
            md.w(f"| {label} | {ms:.1f} |")
        md.w()

        # Threshold assertions
        max_search = thresholds.get("max_avg_search_ms")
        if max_search and search_times:
            engine.assert_max_latency("avg_search", max_search)

        max_recall = thresholds.get("max_avg_recall_ms")
        if max_recall and recall_times:
            engine.assert_max_latency("avg_recall", max_recall)

        engine.assert_no_errors()

        md.footer(engine)
        return md.render()
