"""
Kernel: pres_marp_render
Stage: 3 (Rendering)

Converts a SlideDeck (with layout directives) into a MARP-compatible
Markdown string. Produces the final .md content ready for marp-cli export.

MARP format:
- YAML frontmatter with marp: true
- Slide separators: ---
- Layout directives: <!-- _class: ... -->, <!-- _paginate: false -->, etc.
- Speaker notes: <!-- notes -->

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.presenter.models import (
    Slide,
    SlideCode,
    SlideContent,
    SlideDeck,
    SlideImage,
    SlideLayout,
    SlideTable,
    SlideType,
)
from ragix_kernels.shared.md_toc import expand_toc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MARP Markdown rendering
# ---------------------------------------------------------------------------

def render_frontmatter(deck: SlideDeck, config: Dict[str, Any]) -> str:
    """Render MARP YAML frontmatter block."""
    theme_cfg = config.get("theme", {})
    theme_name = theme_cfg.get("name", deck.theme.name or "default")
    size = theme_cfg.get("size", deck.theme.size or "16:9")
    math = theme_cfg.get("math", deck.theme.math or "katex")
    paginate = theme_cfg.get("paginate", True)

    # Header/footer with template substitution
    header = theme_cfg.get("header", "{organization}")
    footer = theme_cfg.get("footer", "{title} -- {date}")

    meta = deck.metadata
    header = header.replace("{organization}", meta.organization or "")
    header = header.replace("{title}", meta.title or "")
    header = header.replace("{author}", meta.author or "")
    footer = footer.replace("{title}", meta.title or "")
    footer = footer.replace("{date}", meta.date or "")
    footer = footer.replace("{organization}", meta.organization or "")
    # Clean up template placeholders that had no value
    header = header.replace("{", "").replace("}", "").strip()
    footer = footer.replace("{", "").replace("}", "").strip()

    lines = [
        "---",
        "marp: true",
        f"theme: {theme_name}",
        f"paginate: {str(paginate).lower()}",
        f"math: {math}",
        f"size: {size}",
    ]
    if header:
        lines.append(f"header: '{header}'")
    if footer:
        lines.append(f"footer: '{footer}'")
    lines.append("---")

    return "\n".join(lines)


def render_layout_directives(layout: Optional[SlideLayout]) -> str:
    """Render MARP HTML comment directives from a SlideLayout."""
    if layout is None:
        return ""

    parts: List[str] = []
    if layout.css_class:
        parts.append(f"<!-- _class: {layout.css_class} -->")
    if not layout.paginate:
        parts.append("<!-- _paginate: false -->")
    if layout.background_color:
        parts.append(f"<!-- _backgroundColor: {layout.background_color} -->")
    if layout.background_image:
        parts.append(f"<!-- _backgroundImage: url('{layout.background_image}') -->")

    return "\n".join(parts)


def render_slide(slide: Slide) -> str:
    """Render a single slide to MARP Markdown."""
    parts: List[str] = []

    # Layout directives
    directives = render_layout_directives(slide.layout)
    if directives:
        parts.append(directives)

    c = slide.content

    # --- TITLE ---
    if slide.type == SlideType.TITLE:
        if c.heading:
            parts.append(f"# {c.heading}")
        if c.subheading:
            parts.append(f"## {c.subheading}")
        if c.body:
            parts.append("")
            for line in c.body:
                parts.append(line)

    # --- SECTION ---
    elif slide.type == SlideType.SECTION:
        if c.heading:
            parts.append(f"# {c.heading}")

    # --- CONTENT ---
    elif slide.type == SlideType.CONTENT:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.body:
            parts.append("")
            for line in c.body:
                parts.append(line)
        if c.bullets:
            parts.append("")
            for item in c.bullets:
                parts.append(f"- {item}")
        if c.numbered:
            parts.append("")
            for i, item in enumerate(c.numbered, 1):
                parts.append(f"{i}. {item}")

    # --- CODE ---
    elif slide.type == SlideType.CODE:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.code:
            parts.append("")
            lang = c.code.language or ""
            parts.append(f"```{lang}")
            parts.append(c.code.text)
            parts.append("```")

    # --- EQUATION ---
    elif slide.type == SlideType.EQUATION:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.equation:
            parts.append("")
            parts.append("$$")
            parts.append(c.equation)
            parts.append("$$")

    # --- TABLE ---
    elif slide.type == SlideType.TABLE:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.table:
            parts.append("")
            table_class = slide.layout.table_class if slide.layout else ""
            if table_class:
                parts.append(f'<div class="{table_class}">')
                parts.append("")
            parts.append(_render_table(c.table))
            if table_class:
                parts.append("")
                parts.append('</div>')

    # --- IMAGE_TEXT ---
    elif slide.type == SlideType.IMAGE_TEXT:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.image and c.image.path:
            use_inline = slide.layout and slide.layout.inline_image
            if use_inline:
                alt = c.image.alt or c.image.caption or ""
                img_cls = (slide.layout.image_class
                           if slide.layout and slide.layout.image_class
                           else "figure")
                parts.append("")
                parts.append(f'<div class="{img_cls}">')
                parts.append(f'<img src="{c.image.path}" alt="{alt}" />')
                parts.append('</div>')
            else:
                bg_spec = "bg left:40%"
                if slide.layout and slide.layout.bg_position:
                    bg_spec = f"bg {slide.layout.bg_position}"
                    if slide.layout.bg_size:
                        bg_spec += f":{slide.layout.bg_size}"
                parts.append("")
                parts.append(f"![{bg_spec}]({c.image.path})")
        if c.body:
            parts.append("")
            for line in c.body:
                parts.append(line)

    # --- IMAGE_FULL ---
    elif slide.type == SlideType.IMAGE_FULL:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.image and c.image.path:
            use_inline = slide.layout and slide.layout.inline_image
            if use_inline:
                alt = c.image.alt or c.image.caption or ""
                img_cls = (slide.layout.image_class
                           if slide.layout and slide.layout.image_class
                           else "figure-full")
                parts.append("")
                parts.append(f'<div class="{img_cls}">')
                parts.append(f'<img src="{c.image.path}" alt="{alt}" />')
                parts.append('</div>')
            else:
                bg_spec = "bg contain"
                if slide.layout and slide.layout.bg_size:
                    bg_spec = f"bg {slide.layout.bg_size}"
                parts.append("")
                parts.append(f"![{bg_spec}]({c.image.path})")
        elif c.code and c.code.language == "mermaid":
            # Mermaid diagrams rendered as code block
            parts.append("")
            parts.append("```mermaid")
            parts.append(c.code.text)
            parts.append("```")

    # --- QUOTE ---
    elif slide.type == SlideType.QUOTE:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.body:
            parts.append("")
            for line in c.body:
                for subline in line.splitlines():
                    parts.append(f"> {subline}")

    # --- TWO_COLUMN ---
    elif slide.type == SlideType.TWO_COLUMN:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.columns and len(c.columns) >= 2:
            parts.append("")
            parts.append('<div class="columns">')
            parts.append('<div class="column">')
            parts.append("")
            for line in c.columns[0].body:
                parts.append(line)
            for item in c.columns[0].bullets:
                parts.append(f"- {item}")
            parts.append("")
            parts.append('</div>')
            parts.append('<div class="column">')
            parts.append("")
            for line in c.columns[1].body:
                parts.append(line)
            for item in c.columns[1].bullets:
                parts.append(f"- {item}")
            parts.append("")
            parts.append('</div>')
            parts.append('</div>')

    # --- SUMMARY ---
    elif slide.type == SlideType.SUMMARY:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.bullets:
            parts.append("")
            for item in c.bullets:
                parts.append(f"- {item}")

    # --- BLANK ---
    elif slide.type == SlideType.BLANK:
        pass  # empty slide

    # Speaker notes
    if slide.notes:
        parts.append("")
        parts.append(f"<!-- {slide.notes} -->")

    return "\n".join(parts)


def _render_table(table: SlideTable) -> str:
    """Render a SlideTable to Markdown pipe-table format."""
    lines: List[str] = []
    if table.headers:
        lines.append("| " + " | ".join(table.headers) + " |")
        lines.append("| " + " | ".join("---" for _ in table.headers) + " |")
    for row in table.rows:
        lines.append("| " + " | ".join(row) + " |")
    if table.caption:
        lines.append("")
        lines.append(f"*{table.caption}*")
    return "\n".join(lines)


def render_deck(deck: SlideDeck, config: Dict[str, Any]) -> str:
    """
    Render a full SlideDeck to MARP Markdown.

    Args:
        deck: SlideDeck with layout directives.
        config: Pipeline configuration dict.

    Returns:
        Complete MARP Markdown string.
    """
    parts: List[str] = []

    # Frontmatter
    parts.append(render_frontmatter(deck, config))

    # Slides
    for i, slide in enumerate(deck.slides):
        parts.append("")
        rendered = render_slide(slide)
        parts.append(rendered)
        # Slide separator (not after last slide)
        if i < len(deck.slides) - 1:
            parts.append("")
            parts.append("---")

    parts.append("")  # trailing newline
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class PresMarpRenderKernel(Kernel):
    """SlideDeck → MARP Markdown string."""

    name = "pres_marp_render"
    version = "1.0.0"
    category = "presenter"
    stage = 3
    description = "Render SlideDeck to MARP-compatible Markdown"

    requires: List[str] = ["pres_layout_assign", "pres_asset_catalog"]
    provides: List[str] = ["marp_markdown"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load layout-assigned deck
        layout_path = input.dependencies["pres_layout_assign"]
        layout_data = json.loads(layout_path.read_text(encoding="utf-8"))
        deck = SlideDeck.from_dict(layout_data["data"])

        # Render
        marp_md = render_deck(deck, input.config)

        # Expand [TOC] markers if present
        toc_cfg = input.config.get("slide_plan", {}).get("toc", {})
        toc_enabled = toc_cfg.get("enabled", True)
        toc_pages = 0
        if toc_enabled and "[TOC]" in marp_md:
            items_per_page = toc_cfg.get("items_per_page", 12)
            min_level = toc_cfg.get("min_level", 1)
            max_level = toc_cfg.get("max_level", 1)
            continuation = toc_cfg.get(
                "continuation_title",
                "Table of Contents (continued)",
            )
            # Use ## for continuation pages (matches MARP section style)
            cont_heading = f"## {continuation}"
            marp_md, toc_report = expand_toc(
                marp_md,
                min_level=min_level,
                max_level=max_level,
                items_per_page=items_per_page,
                page_separator="\n\n---\n\n",
                toc_continuation_title=cont_heading,
                exclude_before_toc=True,
            )
            toc_pages = toc_report.pages
            if toc_report.toc_markers_expanded > 0:
                logger.info(
                    "TOC expanded: %d entries on %d page(s)",
                    toc_report.headings_included,
                    toc_report.pages,
                )

        # Slide count: original deck slides + extra pages from TOC pagination
        slide_count = len(deck.slides)
        if toc_pages > 1:
            slide_count += toc_pages - 1  # first page replaces the TOC slide

        # Collect asset references
        asset_refs: List[str] = []
        for slide in deck.slides:
            if slide.content.image and slide.content.image.path:
                asset_refs.append(slide.content.image.path)

        return {
            "marp_markdown": marp_md,
            "slide_count": slide_count,
            "asset_refs": sorted(set(asset_refs)),
            "markdown_bytes": len(marp_md.encode("utf-8")),
            "toc_pages": toc_pages,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        n_slides = data.get("slide_count", 0)
        n_bytes = data.get("markdown_bytes", 0)
        n_assets = len(data.get("asset_refs", []))
        return (
            f"MARP render: {n_slides} slides, "
            f"{n_bytes} bytes, {n_assets} asset refs"
        )
