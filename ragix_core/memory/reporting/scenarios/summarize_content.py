"""
SummarizeContentScenario — content summary report.

Sections: inventory, audit queries, cross-references, recall injection,
filtered discovery, link graph, footer.
"""

from __future__ import annotations

from ragix_core.memory.reporting.renderer import MarkdownRenderer
from ragix_core.memory.reporting.schema import (
    parse_crossref_chains,
    parse_query_items,
)
from ragix_core.memory.reporting.scenarios import ScenarioBase, register_scenario

if False:  # TYPE_CHECKING
    from ragix_core.memory.reporting.engine import ReportEngine


@register_scenario
class SummarizeContentScenario(ScenarioBase):

    @property
    def id(self) -> str:
        return "summarize_content"

    @property
    def default_config_name(self) -> str:
        return "summarize_content.yml"

    def run(self, engine: "ReportEngine", config: dict) -> str:
        meta = config.get("meta", {})
        md = MarkdownRenderer(
            title=meta.get("title", "RAGIX Memory — Content Summary Report"),
            author="Olivier Vitrac, PhD, HDR | Adservio Innovation Lab",
        )

        # Header
        md.header(meta, engine)

        # §1 Inventory
        inv = config.get("inventory", {})
        md.section_inventory(
            engine,
            by_type=inv.get("by_type", True),
            by_tags=inv.get("by_tags", True),
            top_tags_k=inv.get("top_tags_k", 15),
        )

        # §2 Audit queries
        q_cfg = config.get("queries", {})
        queries = parse_query_items(q_cfg.get("items", []))
        if queries:
            md.section_queries(
                engine,
                queries,
                default_k=q_cfg.get("k", 5),
                read_full_top_n=q_cfg.get("read_full_top_n", 3),
            )

        # §3 Cross-references
        xref_cfg = config.get("crossrefs", {})
        chains = parse_crossref_chains(xref_cfg.get("chains", []))
        if chains:
            md.section_crossrefs(
                engine, chains, default_k=xref_cfg.get("k", 2),
            )

        # §4 Recall injection
        recall_cfg = config.get("recall", {})
        recall_query = recall_cfg.get("query", "")
        recall_budgets = recall_cfg.get("budgets", [500, 1500, 4000])
        if recall_query:
            md.section_recall(
                engine,
                recall_query,
                recall_budgets,
                truncation_chars=recall_cfg.get("truncation_chars", 3000),
            )

        # §5 Filtered discovery
        views = config.get("filtered_views", [])
        if views:
            md.section_filtered(engine, views)

        # §6 Link graph
        links_cfg = config.get("links", {})
        if links_cfg.get("create") or links_cfg.get("traverse"):
            md.section_link_graph(engine, links_cfg)

        # §7 Summary + footer
        md.w("## Summary")
        md.w()
        n_items = len(engine.list_items())
        md.w(f"- **{n_items} findings** in memory")
        if queries:
            md.w(f"- **{len(queries)} audit queries** with full content retrieval")
        if chains:
            md.w(f"- **{len(chains)} cross-reference chains**")
        if recall_budgets:
            budgets_str = "/".join(str(b) for b in recall_budgets)
            md.w(f"- **{len(recall_budgets)} budget levels** ({budgets_str} tokens)")
        if views:
            md.w(f"- **{len(views)} filtered views**")
        if links_cfg.get("create"):
            md.w(f"- **{len(links_cfg['create'])} link specifications**")
        md.w()

        md.footer(engine)
        return md.render()
