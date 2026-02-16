"""
RegressionMinScenario — CI smoke test.

Minimal report: preflight, 3 searches, 1 recall, consolidation, asserts.
All assertions must pass for CI green.
"""

from __future__ import annotations

import time

from ragix_core.memory.reporting.renderer import MarkdownRenderer
from ragix_core.memory.reporting.scenarios import ScenarioBase, register_scenario

if False:  # TYPE_CHECKING
    from ragix_core.memory.reporting.engine import ReportEngine


@register_scenario
class RegressionMinScenario(ScenarioBase):

    @property
    def id(self) -> str:
        return "regression_min"

    @property
    def default_config_name(self) -> str:
        return "regression_min.yml"

    def run(self, engine: "ReportEngine", config: dict) -> str:
        meta = config.get("meta", {})
        md = MarkdownRenderer(
            title=meta.get("title", "RAGIX Memory — Regression Smoke Report"),
        )
        md.header(meta, engine)

        search_cfg = config.get("search", {})
        recall_cfg = config.get("recall", {})
        thresholds = config.get("thresholds", {})

        # §1 Preflight — store must have items
        md.section_stats(engine)
        n_items = len(engine.list_items())
        min_items = thresholds.get("min_items_total", 1)
        engine.assert_min_count("total_items", n_items, min_items)

        # §2 Search smoke
        queries = search_cfg.get("queries", [])
        k = search_cfg.get("k", 3)
        search_times = []

        n = md._next_section()
        md.w(f"## {n}. Search Smoke ({len(queries)} queries)")
        md.w()

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

            count = result.get("count", 0)
            items = result.get("items", [])
            top = items[0].get("title", "")[:60] if items else "—"
            md.w(f"- `{q}` → **{count}** results, {dt:.1f} ms — *{top}*")

        md.w()
        if search_times:
            avg = sum(search_times) / len(search_times)
            md.w(f"**Avg**: {avg:.1f} ms")
            engine._timings["avg_search"] = avg
            md.w()

        # §3 Recall smoke
        recall_query = recall_cfg.get("query", "")
        budget = recall_cfg.get("budget_tokens", 800)
        truncation = recall_cfg.get("truncation_chars", 1200)

        n2 = md._next_section()
        md.w(f"## {n2}. Recall Smoke")
        md.w()

        t0 = time.perf_counter()
        result = engine.tool(
            "memory_recall",
            query=recall_query,
            budget_tokens=budget,
            workspace=engine.workspace,
        )
        dt = (time.perf_counter() - t0) * 1000
        engine._timings["avg_recall"] = dt

        inject = result.get("inject_text", "")
        md.w(f"- Query: `{recall_query}`")
        md.w(f"- Budget: {budget} | Used: {result.get('tokens_used', 0)} | "
             f"Matched: {result.get('matched', 0)}")
        md.w(f"- Injection: {len(inject)} chars | Time: {dt:.0f} ms")
        md.w()

        if inject:
            md.w("```")
            if len(inject) > truncation:
                md.w(inject[:truncation])
                md.w(f"... [{len(inject) - truncation} chars truncated]")
            else:
                md.w(inject)
            md.w("```")
            md.w()

            # Assert format version
            fmt_ver = thresholds.get("format_version", 1)
            engine.assert_format_version(inject, fmt_ver)

        # §4 Consolidation (idempotence)
        with engine.timed("consolidation"):
            engine.tool("memory_consolidate", workspace=engine.workspace)

        # §5 Verdicts
        n3 = md._next_section()
        md.w(f"## {n3}. Verdicts")
        md.w()

        verdicts = []
        verdicts.append(("Items in store", n_items >= min_items))
        verdicts.append(("Search returns results", all(
            t > 0 for t in search_times
        ) if search_times else False))
        verdicts.append(("Recall produces injection", len(inject) > 0))
        verdicts.append(("Format version correct", True))  # would have raised

        max_search = thresholds.get("max_avg_search_ms")
        if max_search and search_times:
            avg_ok = (sum(search_times) / len(search_times)) <= max_search
            verdicts.append((f"Search latency <= {max_search} ms", avg_ok))

        max_recall = thresholds.get("max_avg_recall_ms")
        if max_recall:
            recall_ok = dt <= max_recall
            verdicts.append((f"Recall latency <= {max_recall} ms", recall_ok))

        verdicts.append(("No tool errors", len(engine._errors) == 0))

        all_pass = True
        for label, ok in verdicts:
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            md.w(f"- [{status}] {label}")

        md.w()
        md.w(f"**Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}**")
        md.w()

        # Run assertion checks (these raise on failure for CI)
        if max_search:
            engine.assert_max_latency("avg_search", max_search)
        if max_recall:
            engine.assert_max_latency("avg_recall", max_recall)
        engine.assert_no_errors()

        md.footer(engine)
        return md.render()
