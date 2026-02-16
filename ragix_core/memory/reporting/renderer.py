"""
MarkdownRenderer — reusable section methods for memory reports.

Each method takes an engine (for tool calls) and config data,
appends formatted Markdown to an internal buffer.
"""

from __future__ import annotations

import textwrap
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ragix_core.memory.reporting.schema import (
    CrossrefChain,
    LinkSpec,
    QueryItem,
)

# Type alias for the engine (avoids circular import)
if False:  # TYPE_CHECKING
    from ragix_core.memory.reporting.engine import ReportEngine


class MarkdownRenderer:
    """Consistent Markdown formatting for memory report sections."""

    def __init__(self, title: str = "", author: str = "", corpus: str = ""):
        self._lines: List[str] = []
        self.title = title
        self.author = author
        self.corpus = corpus
        self._section_counter = 0

    def w(self, line: str = ""):
        self._lines.append(line)

    def _next_section(self) -> int:
        self._section_counter += 1
        return self._section_counter

    # ── Header / Footer ────────────────────────────────────────────────

    def header(self, meta: dict, engine: "ReportEngine"):
        n_items = len(engine.list_items())
        self.w(f"# {self.title or meta.get('title', 'Memory Report')}")
        self.w()
        self.w(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if self.author:
            self.w(f"**Auteur**: {self.author}")
        if self.corpus:
            self.w(f"**Corpus**: {self.corpus}")
        self.w(f"**Items in memory**: {n_items}")
        self.w(f"**Outil**: RAGIX Memory MCP v2.0 ({engine.n_tools} tools)")
        self.w()
        self.w("---")
        self.w()

    def footer(self, engine: "ReportEngine"):
        metrics = engine.metrics_summary()
        total_calls = metrics.get("total_calls", 0)
        total_errors = metrics.get("total_errors", 0)
        avg_lat = metrics.get("avg_latency_ms", 0)
        self.w(f"**MCP calls**: {total_calls} | "
               f"**Errors**: {total_errors} | "
               f"**Avg latency**: {avg_lat:.1f} ms")
        self.w()
        self.w("---")
        self.w()
        self.w(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
               f"| RAGIX Memory MCP v2.0*")

    # ── §Inventory ─────────────────────────────────────────────────────

    def section_inventory(
        self,
        engine: "ReportEngine",
        *,
        by_type: bool = True,
        by_tags: bool = True,
        top_tags_k: int = 15,
    ):
        n = self._next_section()
        self.w(f"## {n}. Inventory Overview")
        self.w()

        all_items = engine.list_items()

        if by_type:
            type_counts: Dict[str, int] = {}
            type_desc = {
                "constraint": "Hard compliance requirements",
                "decision": "Aggregated verdicts and scores",
                "definition": "Reference definitions",
                "note": "Document summaries and observations",
            }
            for it in all_items:
                t = getattr(it, "type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

            self.w("### By type")
            self.w()
            self.w("| Type | Count | Description |")
            self.w("|------|:-----:|-------------|")
            for t in sorted(type_counts, key=type_counts.get, reverse=True):
                self.w(f"| {t} | {type_counts[t]} | {type_desc.get(t, '')} |")
            self.w()

        if by_tags:
            tag_counts: Dict[str, int] = {}
            for it in all_items:
                for tag in getattr(it, "tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            top = sorted(tag_counts.items(), key=lambda x: -x[1])[:top_tags_k]

            self.w("### Top tags")
            self.w()
            self.w("| Tag | Occurrences |")
            self.w("|-----|:-----------:|")
            for tag, count in top:
                self.w(f"| `{tag}` | {count} |")
            self.w()

    # ── §Queries ───────────────────────────────────────────────────────

    def section_queries(
        self,
        engine: "ReportEngine",
        queries: List[QueryItem],
        *,
        default_k: int = 5,
        read_full_top_n: int = 3,
    ):
        n = self._next_section()
        self.w(f"## {n}. Audit Queries — Full Content Retrieval")
        self.w()

        for qi, q in enumerate(queries, 1):
            self.w(f"### Q{qi}. {q.question}")
            self.w()
            self.w(f"> **Query**: `{q.query}`")
            if q.type_filter:
                self.w(f"> **Filter**: type={q.type_filter}")
            if q.tags:
                self.w(f"> **Tags**: {q.tags}")
            self.w()

            params: Dict[str, Any] = {
                "query": q.query,
                "workspace": engine.workspace,
                "k": q.k or default_k,
            }
            if q.type_filter:
                params["type_filter"] = q.type_filter
            if q.tags:
                params["tags"] = q.tags

            t0 = time.perf_counter()
            result = engine.tool("memory_search", **params)
            dt = (time.perf_counter() - t0) * 1000

            items = result.get("items", [])
            count = result.get("count", 0)
            self.w(f"**{count} result(s)** in {dt:.1f} ms")
            self.w()

            for i, it in enumerate(items, 1):
                title = it.get("title", "?")
                itype = it.get("type", "?")
                tier = it.get("tier", "?")
                tags = it.get("tags", [])

                full_content = None
                if i <= read_full_top_n and it.get("id"):
                    read_r = engine.tool("memory_read", ids=it["id"])
                    full_items = read_r.get("items", [])
                    if full_items:
                        full_content = full_items[0].get("content", "")

                self.w(f"**{i}. {title}**")
                tag_str = ", ".join(f"`{t}`" for t in tags[:5])
                self.w(f"   - Type: `{itype}` | Tier: `{tier}` | Tags: {tag_str}")
                if full_content:
                    for line in textwrap.fill(full_content, width=90).split("\n"):
                        self.w(f"   > {line}")
                self.w()

            self.w("---")
            self.w()

    # ── §Crossrefs ─────────────────────────────────────────────────────

    def section_crossrefs(
        self,
        engine: "ReportEngine",
        chains: List[CrossrefChain],
        *,
        default_k: int = 2,
    ):
        n = self._next_section()
        self.w(f"## {n}. Cross-Reference Discovery")
        self.w()

        for chain in chains:
            self.w(f"### {chain.name}")
            self.w()

            chain_titles = []
            for facet in chain.facets:
                params: Dict[str, Any] = {
                    "query": facet.query,
                    "workspace": engine.workspace,
                    "k": default_k,
                }
                if facet.type_filter:
                    params["type_filter"] = facet.type_filter

                result = engine.tool("memory_search", **params)
                top = result.get("items", [])

                if top:
                    it = top[0]
                    full_content = ""
                    if it.get("id"):
                        read_r = engine.tool("memory_read", ids=it["id"])
                        full_items = read_r.get("items", [])
                        if full_items:
                            full_content = full_items[0].get("content", "")

                    self.w(f"**{facet.label}** → `{it.get('title', '?')}`")
                    if full_content:
                        for line in textwrap.fill(full_content, width=90).split("\n"):
                            self.w(f"> {line}")
                    self.w()
                    chain_titles.append(
                        f"{facet.label}: *{it.get('title', '?')[:60]}*"
                    )

            if chain_titles:
                self.w("**Chain**: " + " → ".join(chain_titles))
                self.w()

            self.w("---")
            self.w()

    # ── §Recall ────────────────────────────────────────────────────────

    def section_recall(
        self,
        engine: "ReportEngine",
        query: str,
        budgets: List[int],
        *,
        truncation_chars: int = 3000,
    ):
        n = self._next_section()
        self.w(f"## {n}. Token-Budgeted Recall — LLM Context Injection")
        self.w()
        self.w("The `memory_recall` tool retrieves items within a token budget,")
        self.w("formatted for direct injection into an LLM prompt.")
        self.w()

        for budget in budgets:
            self.w(f"### Budget: {budget} tokens")
            self.w()

            t0 = time.perf_counter()
            result = engine.tool(
                "memory_recall",
                query=query,
                budget_tokens=budget,
                workspace=engine.workspace,
            )
            dt = (time.perf_counter() - t0) * 1000

            tokens_used = result.get("tokens_used", 0)
            matched = result.get("matched", 0)
            inject = result.get("inject_text", "")

            self.w(f"- **Matched**: {matched} items | "
                   f"**Tokens used**: {tokens_used}/{budget} | "
                   f"**Time**: {dt:.0f} ms")
            self.w(f"- **Injection size**: {len(inject)} chars")
            self.w()

            if inject:
                self.w("```")
                if len(inject) > truncation_chars:
                    self.w(inject[:truncation_chars])
                    self.w(f"... [{len(inject) - truncation_chars} chars truncated]")
                else:
                    self.w(inject)
                self.w("```")
            self.w()

        self.w("---")
        self.w()

    # ── §Filtered Discovery ────────────────────────────────────────────

    def section_filtered(
        self,
        engine: "ReportEngine",
        views: List[Dict[str, Any]],
    ):
        n = self._next_section()
        self.w(f"## {n}. Filtered Discovery")
        self.w()

        for view in views:
            self.w(f"### {view.get('name', 'View')}")
            self.w()

            params: Dict[str, Any] = {
                "query": view.get("query", "*"),
                "workspace": engine.workspace,
                "k": view.get("k", 10),
            }
            if view.get("type_filter"):
                params["type_filter"] = view["type_filter"]
            if view.get("tags"):
                params["tags"] = view["tags"]

            result = engine.tool("memory_search", **params)
            items = result.get("items", [])

            if items:
                self.w("| # | Title | Type | Tags |")
                self.w("|:-:|-------|------|------|")
                for i, it in enumerate(items, 1):
                    title = it.get("title", "?")[:70]
                    itype = it.get("type", "?")
                    tags = ", ".join(it.get("tags", [])[:4])
                    self.w(f"| {i} | {title} | `{itype}` | {tags} |")
                self.w()

                # Expand first item
                if items[0].get("id"):
                    read_r = engine.tool("memory_read", ids=items[0]["id"])
                    full_items = read_r.get("items", [])
                    if full_items:
                        content = full_items[0].get("content", "")
                        self.w(f"**Detail** — *{items[0].get('title', '')}*:")
                        for line in textwrap.fill(content, width=90).split("\n"):
                            self.w(f"> {line}")
                        self.w()
            else:
                self.w("*No results*")
                self.w()

        self.w("---")
        self.w()

    # ── §Link Graph ────────────────────────────────────────────────────

    def section_link_graph(
        self,
        engine: "ReportEngine",
        links_config: Dict[str, Any],
    ):
        n = self._next_section()
        self.w(f"## {n}. Relationship Graph")
        self.w()

        # Build item index by title pattern
        all_items = engine.list_items()
        by_title: Dict[str, dict] = {}
        for it in all_items:
            title = getattr(it, "title", "")
            by_title[title] = {
                "id": it.id, "title": it.title, "type": it.type,
                "tier": it.tier, "tags": it.tags, "content": it.content,
            }

        def find(pattern: str) -> Optional[dict]:
            for title, it in by_title.items():
                if pattern.lower() in title.lower():
                    return it
            return None

        # Create links
        link_specs = links_config.get("create", [])
        created = []
        for spec in link_specs:
            src = find(spec.get("src_pattern", ""))
            dst = find(spec.get("dst_pattern", ""))
            if src and dst:
                try:
                    result = engine.tool(
                        "memory_link",
                        src_id=src["id"],
                        dst_id=dst["id"],
                        relation=spec.get("relation", "supports"),
                    )
                    if result.get("status") == "ok":
                        created.append({
                            "src": src["title"][:50],
                            "dst": dst["title"][:50],
                            "relation": spec.get("relation", "supports"),
                        })
                except Exception:
                    pass

        self.w(f"**{len(created)} links created**")
        self.w()

        if created:
            self.w("| Source | Relation | Destination |")
            self.w("|--------|:--------:|-------------|")
            for lk in created:
                self.w(f"| {lk['src']} | `{lk['relation']}` | {lk['dst']} |")
            self.w()

            # Mermaid graph
            self.w("### Dependency Graph (Mermaid)")
            self.w()
            self.w("```mermaid")
            self.w("graph TD")
            node_ids: Dict[str, str] = {}
            counter = [0]

            def nid(title):
                if title not in node_ids:
                    counter[0] += 1
                    node_ids[title] = f"N{counter[0]}"
                return node_ids[title]

            for lk in created:
                s, d = nid(lk["src"]), nid(lk["dst"])
                self.w(f'    {s}["{lk["src"]}"] -->|{lk["relation"]}| '
                       f'{d}["{lk["dst"]}"]')

            for title, node in node_ids.items():
                it = find(title)
                if it:
                    itype = it.get("type", "note")
                    if itype == "constraint":
                        self.w(f"    style {node} fill:#ffcccc,stroke:#cc0000")
                    elif itype == "decision":
                        self.w(f"    style {node} fill:#cce5ff,stroke:#0066cc")
                    elif itype == "definition":
                        self.w(f"    style {node} fill:#e6ffe6,stroke:#009900")
            self.w("```")
            self.w()

        # Traverse from start item
        start_pattern = links_config.get("traverse", {}).get("start_pattern")
        if start_pattern:
            start = find(start_pattern)
            if start:
                self.w("### Graph Traversal")
                self.w()
                self.w(f"Starting from: **{start['title']}** (`{start['id']}`)")
                self.w()
                self.w(f"- **Content**: {start.get('content', '')}")
                self.w(f"- **Tags**: {', '.join(start.get('tags', []))}")

                outgoing = engine.query_links(start["id"], "outgoing")
                incoming = engine.query_links(start["id"], "incoming")
                self.w(f"- **Outgoing links**: {len(outgoing)}")
                self.w(f"- **Incoming links**: {len(incoming)}")
                self.w()

                if outgoing:
                    self.w("**Outgoing**:")
                    self.w()
                    for dst_id, rel in outgoing:
                        tr = engine.tool("memory_read", ids=dst_id)
                        titems = tr.get("items", [])
                        if titems:
                            t = titems[0]
                            self.w(f"  - **{rel}** → *{t.get('title', '?')}* "
                                   f"(`{t.get('type', '?')}`)")
                            self.w(f"    > {t.get('content', '')[:200]}")
                            self.w()

                if incoming:
                    self.w("**Incoming**:")
                    self.w()
                    for src_id, rel in incoming:
                        sr = engine.tool("memory_read", ids=src_id)
                        sitems = sr.get("items", [])
                        if sitems:
                            s = sitems[0]
                            self.w(f"  - *{s.get('title', '?')}* **{rel}** → this")
                            self.w()

        self.w("---")
        self.w()

    # ── §Benchmark Table ───────────────────────────────────────────────

    def section_benchmark_table(
        self,
        engine: "ReportEngine",
        title: str,
        columns: List[str],
        rows: List[Dict[str, Any]],
    ):
        n = self._next_section()
        self.w(f"## {n}. {title}")
        self.w()
        if rows:
            header = " | ".join(columns)
            sep = " | ".join(":---:" if c in ("ms", "k", "budget", "used",
                                                "matched", "count")
                             else "---" for c in columns)
            self.w(f"| {header} |")
            self.w(f"| {sep} |")
            for row in rows:
                vals = " | ".join(str(row.get(c, "")) for c in columns)
                self.w(f"| {vals} |")
            self.w()
        self.w("---")
        self.w()

    # ── §Stats ─────────────────────────────────────────────────────────

    def section_stats(self, engine: "ReportEngine"):
        result = engine.tool("memory_stats", workspace=engine.workspace)
        n = self._next_section()
        self.w(f"## {n}. Store State")
        self.w()
        self.w(f"- Items: **{result.get('total_items', 0)}**")
        by_tier = result.get("by_tier", {})
        self.w(f"- Tiers: {', '.join(f'{k}={v}' for k, v in by_tier.items())}")
        by_type = result.get("by_type", {})
        self.w(f"- Types: {', '.join(f'{k}={v}' for k, v in by_type.items())}")
        self.w(f"- FTS5: {result.get('fts5_available', False)}")
        self.w()

    # ── Render ─────────────────────────────────────────────────────────

    def render(self) -> str:
        return "\n".join(self._lines)
