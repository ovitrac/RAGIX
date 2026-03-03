"""
MARP Post-Processor — deterministic post-processing for MARP presentations.

Two-step pipeline (v2.0):

  **Step 1 — Layout preprocess** (``layout_preprocess``):
    Scans slides for bare ``<img>`` tags, probes image dimensions (SVG viewBox
    or raster via PIL), classifies the shape (tall / wide / square), and wraps
    the slide content in layout directive comments.  Can be run standalone
    (``--layout-only``) to produce an intermediate file for manual tuning.

  **Step 2 — Full postprocess** (``postprocess_marp``):
    Expands layout directives into inline-style HTML, then applies all remaining
    transforms (heading cleanup, table classification, progress bars, etc.).

Transforms applied in order:
  -1. **strip_postprocess_artifacts** — idempotency guard (strip injected bars, nav, styles)
  0. **rewrite_title_slide** — replace ``Chapitre 0`` heading with real title
  1. **strip_heading_numbers** — remove N.M.K prefixes from ``## `` headings
  1a. **strip_heading_pipes** — remove ``## | Title`` pipe artifacts
  1b. **normalize_section_dividers** — ``# N. Title`` → ``# Chapitre N — Title``
  1b2. **strip_source_comments** — remove pipeline-generated comments
  1b3. **strip_navigation_rapide** — remove Navigation rapide blockquotes
  1b4. **fix_singleton_numbered_lists** — convert lone ``1.`` items to bullets
  1c. **auto_classify_tables** — inject per-slide ``<style>`` for dense tables
  1d. **remove_trailing_sommaire** — trim redundant Sommaire slides at end
  1d2. **remove_garbled_sommaire** — remove lead+content garbled Sommaire block
  1e. **clean_toc_slide** — fix TOC heading, remove non-chapter entries
  1f. **remove_empty_chapter_dividers** — drop section dividers with no content after
  1f2. **layout_preprocess** — auto-detect image shape → inject ``[I,T]``/``[I;T]`` directives
  1g. **expand_layout_directives** — ``[I,T]``, ``[T,I]``, ``[I;T]``, ``[I,I;t,t]`` → inline HTML
  2. **inject_progress_bar** — visual slide-position indicator (bottom bar)
  3. **inject_chapter_nav** — clickable chapter navigation strip (top-right)
  4. **inject_chapter_footer** — per-slide footer reflecting current chapter

SVG asset transforms (standalone, applied to directory):
  - **clean_svg_title** — strip ``Fig. N —`` prefixes from matplotlib chart titles
  - **boost_mermaid_contrast** — increase stroke widths, darken fills/text for projection

Usage::

    from ragix_kernels.shared.marp_postprocess import postprocess_marp

    # Full pipeline (auto-layout + all transforms)
    new_content = postprocess_marp(content, base_dir="output/")

    # Skip auto-layout (use explicit directives only)
    new_content = postprocess_marp(content, auto_layout=False)

    # Layout preprocess only (for manual tuning)
    from ragix_kernels.shared.marp_postprocess import layout_preprocess
    layout_content = layout_preprocess(content, base_dir="output/")

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-20
"""

from __future__ import annotations

import getpass
import logging
import os
import platform
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MARP parsing helpers
# ---------------------------------------------------------------------------

_SLIDE_SEP_RE = re.compile(r"\n\n?---\n\n?")
_SLIDE_SEP_OUT = "\n\n---\n\n"


def _parse_marp(content: str) -> Tuple[str, List[str]]:
    """Split MARP content into (frontmatter_block, [slide_content, ...]).

    The frontmatter block includes the opening ``---`` but not the closing
    separator.  Each slide string is the raw text between ``---`` separators.

    Handles both ``\\n---\\n`` and ``\\n\\n---\\n\\n`` patterns.
    """
    parts = _SLIDE_SEP_RE.split(content)
    if len(parts) < 2:
        return "", parts
    return parts[0], parts[1:]


def _join_marp(frontmatter: str, slides: List[str]) -> str:
    """Rejoin frontmatter and slides into MARP content.

    Always uses ``\\n\\n---\\n\\n`` — the canonical MARP separator format
    with blank lines on both sides.
    """
    return _SLIDE_SEP_OUT.join([frontmatter] + slides)


# ---------------------------------------------------------------------------
# Transform -1 — Idempotency guard (strip previously injected artifacts)
# ---------------------------------------------------------------------------

# Progress bar div: ``<div class="progress-track"><div ...></div></div>``
# IMPORTANT: only consume surrounding blank lines, NOT slide separators (---).
_PROGRESS_DIV_RE = re.compile(
    r'\n*<div class="progress-track"><div[^>]*></div></div>[^\S\n]*\n?',
)
# Chapter nav div: ``<div class="chapter-nav">...</div>`` (always single line)
_CHAPNAV_DIV_RE = re.compile(
    r'\n*<div class="chapter-nav">.*?</div>[^\S\n]*\n?',
)
# Footer directives: ``<!-- footer: "..." -->``
_FOOTER_DIR_RE = re.compile(
    r'^<!-- footer:\s*["\'].*?["\']\s*-->\n?', re.MULTILINE,
)
# Injected CSS blocks from the postprocessor (progress CSS, chapter nav CSS)
_INJECTED_CSS_RE = re.compile(
    r'<style>\s*/\*\s*MARP post-processor:.*?</style>\n?',
    re.DOTALL,
)
# Scoped style blocks injected by auto_shrink / auto_constrain / auto_classify / TOC
_SCOPED_STYLE_RE = re.compile(
    r'<style scoped>\s*(?:section[\s.][^}]*\{[^}]*\}\s*)+</style>\n?',
    re.DOTALL,
)
# Scoped style blocks injected by auto_classify_tables (non-scoped <style>)
_TABLE_STYLE_RE = re.compile(
    r'<style>\s*table\s*\{[^}]*\}\s*table\s+th.*?</style>\n?',
    re.DOTALL,
)


def strip_postprocess_artifacts(content: str) -> str:
    """Strip previously injected post-processing artifacts for idempotent reruns.

    Removes progress bars, chapter navigation, footer directives, and
    injected CSS blocks so the pipeline can be re-applied cleanly on
    content that was already postprocessed (e.g. when ``.bak`` was created
    from a postprocessed version).
    """
    # Remove injected CSS blocks (progress bar CSS, chapter nav CSS)
    content = _INJECTED_CSS_RE.sub("", content)
    # Remove progress bar divs
    content = _PROGRESS_DIV_RE.sub("", content)
    # Remove chapter nav divs
    content = _CHAPNAV_DIV_RE.sub("", content)
    # Remove footer directives
    content = _FOOTER_DIR_RE.sub("", content)
    # Remove scoped style blocks (auto_shrink, auto_constrain)
    content = _SCOPED_STYLE_RE.sub("", content)
    # Remove table density style blocks
    content = _TABLE_STYLE_RE.sub("", content)
    # Collapse multiple blank lines left behind
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content


# ---------------------------------------------------------------------------
# Chapter detection
# ---------------------------------------------------------------------------

@dataclass
class ChapterInfo:
    """A chapter section-divider detected in the deck."""
    slide_idx: int       # 0-based index in the slides list
    chapter_num: int     # chapter number (0, 1, …, 12)
    title: str           # full chapter title
    short_title: str     # truncated for tooltip (≤24 chars)


# ``# Chapitre 3 — Méthodologie``
_CHAPTER_RE = re.compile(
    r"^#\s+Chapitre\s+(\d+)\s*[—–-]\s*(.+)$", re.MULTILINE,
)
# ``# 10. Recommandations consolidees`` (inconsistent format)
_CHAPTER_NUM_RE = re.compile(
    r"^#\s+(\d+)\.\s+(.+)$", re.MULTILINE,
)


def _detect_chapters(
    slides: List[str], *, skip_zero: bool = True,
) -> List[ChapterInfo]:
    """Detect chapter section-dividers across all slides.

    Args:
        skip_zero: If True, skip Chapter 0 (typically the title slide).
    """
    chapters: List[ChapterInfo] = []
    for i, slide in enumerate(slides):
        m = _CHAPTER_RE.search(slide)
        if not m:
            m = _CHAPTER_NUM_RE.search(slide)
        if m:
            num = int(m.group(1))
            if skip_zero and num == 0:
                continue
            title = m.group(2).strip()
            short = title[:22] + "…" if len(title) > 24 else title
            chapters.append(ChapterInfo(i, num, title, short))
    return chapters


def _current_chapter(
    slide_idx: int, chapters: List[ChapterInfo],
) -> Optional[int]:
    """Return the chapter number that *slide_idx* belongs to."""
    current = None
    for ch in chapters:
        if slide_idx >= ch.slide_idx:
            current = ch.chapter_num
        else:
            break
    return current


# ---------------------------------------------------------------------------
# Transform 0 — Rewrite title slide
# ---------------------------------------------------------------------------

# Non-chapter h1 on a lead slide: ``# Audit d'architecture et de dette...``
_PLAIN_H1_RE = re.compile(r"^#\s+(?!Chapitre\s+\d)(?!\d+\.\s)(.+)$", re.MULTILINE)


def rewrite_title_slide(content: str) -> str:
    """Replace ``# Chapitre 0 — …`` on the title slide with the real title.

    Scans for the first non-chapter ``# Title`` on a lead slide and uses
    it to replace the Chapter 0 heading on slide 1.  Also updates the
    frontmatter ``footer:`` to remove the Chapter 0 reference.
    """
    frontmatter, slides = _parse_marp(content)
    if not slides:
        return content

    # Only act if slide 1 has a Chapter 0 heading
    ch0 = _CHAPTER_RE.search(slides[0])
    if not ch0 or int(ch0.group(1)) != 0:
        return content

    # Find the real document title: first non-chapter h1 on a lead slide
    real_title = None
    for slide in slides[1:]:
        if "<!-- _class: lead -->" in slide:
            m = _PLAIN_H1_RE.search(slide)
            if m:
                real_title = m.group(1).strip()
                break

    if not real_title:
        return content

    # Replace the heading on slide 1
    slides[0] = slides[0].replace(
        ch0.group(0), f"# {real_title}"
    )

    # Update frontmatter footer default — preserve date suffix, use
    # double quotes to avoid breaking on apostrophes (e.g. "d'architecture").
    escaped_title = real_title.replace('"', '\\"')
    date_str = _extract_frontmatter_footer(frontmatter) or ""
    date_part = f" -- {date_str}" if date_str else ""
    frontmatter = re.sub(
        r"^footer:\s*['\"].*?['\"]$",
        f'footer: "{escaped_title}{date_part}"',
        frontmatter,
        flags=re.MULTILINE,
    )

    return _join_marp(frontmatter, slides)


# ---------------------------------------------------------------------------
# Transform 1 — Strip heading numbers
# ---------------------------------------------------------------------------

# Matches: ``## 3.6.4 Auditabilité`` or ``### 10.2 Title``
#           ``## 6.2bis Security Hotspots`` (French legal suffixes)
# Does NOT match: ``# Chapitre 3 — ...`` (h1 handled separately)
_HEADING_NUM_RE = re.compile(
    r"^(#{2,6})\s+\d+(?:\.\d+)*(?:bis|ter|quater)?\.?\s+(.+)$", re.MULTILINE,
)


def strip_heading_numbers(content: str) -> str:
    """Remove N.M.K prefixes from ``##``-level headings.

    Transforms::

        ## 3.6.4 Auditabilité           →  ## Auditabilité
        ## 1.3 Constats majeurs (2)      →  ## Constats majeurs (2)
        ### 10.2.1 Détail                →  ### Détail

    Preserves h1 section dividers and body text unchanged.
    """
    return _HEADING_NUM_RE.sub(r"\1 \2", content)


# ---------------------------------------------------------------------------
# Transform 1a — Strip pipe artifacts from headings
# ---------------------------------------------------------------------------

# Matches: ``## | Title`` or ``## | **Title**``
_HEADING_PIPE_RE = re.compile(
    r"^(#{1,6})\s+\|\s+(.+)$", re.MULTILINE,
)


def strip_heading_pipes(content: str) -> str:
    """Remove ``|`` pipe separators from heading lines.

    Transforms::

        ## | Title                 →  ## Title
        ## | **Bold Title**       →  ## **Bold Title**
        # 1 | Vue générale        →  (handled by normalize_section_dividers)

    This targets the ``## |`` pattern left by SIAS-style presenter output
    where pipes were used as separators in the source document headings.
    """
    return _HEADING_PIPE_RE.sub(r"\1 \2", content)


# ---------------------------------------------------------------------------
# Transform 1b — Normalize section dividers (optional)
# ---------------------------------------------------------------------------

_BARE_NUM_HEADING_RE = re.compile(
    r"^#\s+(\d+)(?:\.|\s*\|)\s+(.+)$", re.MULTILINE,
)

# Matches already-normalized headings: ``# Chapitre N — Title``
_CHAPITRE_HEADING_RE = re.compile(
    r"^(#\s+Chapitre\s+)\d+(\s*—\s*.+)$", re.MULTILINE,
)


def normalize_section_dividers(content: str) -> str:
    """Normalize ``# N. Title`` or ``# N | Title`` to ``# Chapitre N — Title``.

    Ensures consistent format for all chapter section-divider slides.
    Leaves non-numbered h1 headings untouched.
    """
    def _repl(m: re.Match) -> str:
        num, title = m.group(1), m.group(2).strip()
        # Strip bold markers from title if present
        title = title.strip("*").strip()
        return f"# Chapitre {num} — {title}"
    return _BARE_NUM_HEADING_RE.sub(_repl, content)


def renumber_chapters(content: str) -> str:
    """Renumber ``# Chapitre N — Title`` headings sequentially (1, 2, 3...).

    Also updates TOC entries that reference ``Chapitre N``.
    Removes slides with generic ``# Other`` section dividers.
    """
    # Remove "# Other" section divider slides (uses _parse_marp for robustness)
    frontmatter, slides = _parse_marp(content)
    cleaned = []
    for slide in slides:
        lines = slide.strip().splitlines()
        # Skip slides that contain "# Other" (section divider)
        if any(re.match(r"^#\s+Other\s*$", l.strip()) for l in lines):
            continue
        cleaned.append(slide)
    content = _join_marp(frontmatter, cleaned)

    # Build old→new chapter number mapping
    chapter_map: dict = {}  # old_num_str -> new_num
    counter = 0
    for m in _CHAPITRE_HEADING_RE.finditer(content):
        # Extract old number from the matched text
        old_text = m.group(0)
        old_num = re.search(r"Chapitre\s+(\d+)", old_text)
        if old_num:
            old_str = old_num.group(1)
            if old_str not in chapter_map:
                counter += 1
                chapter_map[old_str] = counter

    if not chapter_map:
        return content

    # Single-pass renumber: replace ALL "Chapitre N" references at once
    # (avoids cascading replacements when doing sequential str.replace)
    _chapitre_any_re = re.compile(r"Chapitre\s+(\d+)")

    def _renum_any(m: re.Match) -> str:
        old_str = m.group(1)
        if old_str in chapter_map:
            return f"Chapitre {chapter_map[old_str]}"
        return m.group(0)

    content = _chapitre_any_re.sub(_renum_any, content)

    return content


# ---------------------------------------------------------------------------
# Transform 1b2 — Strip source comments and figure numbering from captions
# ---------------------------------------------------------------------------

_SOURCE_COMMENT_RE = re.compile(
    r"^<!--\s*Source:\s*.+?-->\s*$", re.MULTILINE,
)
_SECTION_COMMENT_RE = re.compile(
    r"^<!--\s*Section:\s*.+?-->\s*$", re.MULTILINE,
)
_TITLE_COMMENT_RE = re.compile(
    r"^<!--\s*Title slide\s*-->\s*$", re.MULTILINE,
)
# Caption prefixes: "Fig. 4 — ..." or "Figure 12 — ..." or "Tableau 3 — ..."
_CAPTION_PREFIX_RE = re.compile(
    r"^(?:Fig\.?\s*|Figure\s+|Diagram(?:me)?\s+|Sch[eé]ma\s+|Scheme\s+|Table(?:au)?\s+)\d+\s*[—–:\-\.]\s*",
    re.MULTILINE,
)


def strip_source_comments(content: str) -> str:
    """Remove pipeline-generated comments and figure numbering from captions.

    Strips:
      - ``<!-- Source: 04_ARCHITECTURE.md:L140-L140 -->``
      - ``<!-- Section: Chapitre 3 — Méthodologie -->``
      - ``<!-- Title slide -->``
      - ``Fig. N —`` / ``Figure N —`` / ``Tableau N —`` prefixes from captions
    """
    content = _SOURCE_COMMENT_RE.sub("", content)
    content = _SECTION_COMMENT_RE.sub("", content)
    content = _TITLE_COMMENT_RE.sub("", content)
    content = _CAPTION_PREFIX_RE.sub("", content)
    return content


# ---------------------------------------------------------------------------
# Transform 1b2 — Strip "Navigation rapide" blockquotes
# ---------------------------------------------------------------------------

_NAV_RAPIDE_RE = re.compile(
    r"^>\s*\*{0,2}Navigation rapide\*{0,2}\s*[:：].*$",
    re.MULTILINE,
)


def strip_navigation_rapide(content: str) -> str:
    """Remove ``> **Navigation rapide** : …`` blockquotes.

    These are document-level cross-reference aids that have no meaning
    in a slide presentation.  Removes the blockquote line and any
    immediately following continuation lines (``> …``).  If a slide
    becomes empty after removal, the entire slide is dropped.
    """
    frontmatter, slides = _parse_marp(content)
    result: List[str] = []

    for slide in slides:
        # Check if this slide has a nav-rapide blockquote
        had_nav = bool(_NAV_RAPIDE_RE.search(slide))
        # Remove the nav-rapide blockquote (may span multiple > lines)
        cleaned = _NAV_RAPIDE_RE.sub("", slide)
        # Also strip any orphan continuation lines that immediately follow
        cleaned = re.sub(r"\n>\s*\n", "\n\n", cleaned)
        # Collapse multiple blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if not had_nav:
            # Slide was not modified — keep as-is
            result.append(cleaned)
            continue
        # Slide had nav-rapide removed — check if anything substantive remains
        text = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL).strip()
        text = re.sub(r"<style\b.*?</style>", "", text, flags=re.DOTALL).strip()
        # Strip progress-bar divs (line-level, safe for nested figures)
        text_lines = [
            l for l in text.splitlines()
            if l.strip() and 'class="progress-' not in l
        ]
        if not text_lines:
            continue  # slide is empty after removal
        # Drop if only a heading remains (no body content)
        if len(text_lines) == 1 and text_lines[0].strip().startswith("#"):
            continue
        result.append(cleaned)

    return _join_marp(frontmatter, result)


# ---------------------------------------------------------------------------
# Transform 1b3 — Fix singleton numbered lists (1. → bullet)
# ---------------------------------------------------------------------------

_NUMBERED_ITEM_RE = re.compile(r"^(\d+)\.\s+", re.MULTILINE)


def fix_singleton_numbered_lists(content: str) -> str:
    """Convert lone ``1.`` list items to bullet points.

    When the presenter pipeline splits a numbered list across slides,
    individual slides end up with a single ``1. Text`` which renders as
    a numbered list with one item.  This transform converts such singletons
    to ``- Text`` (unordered) while leaving multi-item lists untouched.
    """
    frontmatter, slides = _parse_marp(content)
    result: List[str] = []

    for slide in slides:
        nums = _NUMBERED_ITEM_RE.findall(slide)
        if len(nums) == 1 and nums[0] == "1":
            # Single "1." with no 2., 3., etc. → convert to bullet
            slide = re.sub(r"^1\.\s+", "- ", slide, count=1, flags=re.MULTILINE)
        result.append(slide)

    return _join_marp(frontmatter, result)


# ---------------------------------------------------------------------------
# Transform 1c — Auto-classify table density
# ---------------------------------------------------------------------------

_TABLE_ROW_RE = re.compile(r"^\|.+\|$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|$", re.MULTILINE)


def _count_table_data_rows(slide: str) -> int:
    """Count data rows in a markdown table (header + separator excluded)."""
    all_rows = _TABLE_ROW_RE.findall(slide)
    sep_rows = _TABLE_SEP_RE.findall(slide)
    return max(0, len(all_rows) - len(sep_rows) - 1)  # -1 for header


_TABLE_SCALE = {
    "small": "font-size: 0.65em; line-height: 1.3;",
    "tiny":  "font-size: 0.50em; line-height: 1.25;",
}

_TABLE_PAD = {
    "small": "padding: 3px 8px;",
    "tiny":  "padding: 2px 5px;",
}


def auto_classify_tables(content: str) -> str:
    """Inject per-slide ``<style scoped>`` for dense tables.

    Thresholds (data rows, excluding header):
      - 5-7 rows → ``small`` scaling
      - 8+ rows  → ``tiny`` scaling
      - <5 rows  → leave unchanged

    Slides already containing ``table-small`` or ``table-tiny`` (set by
    the presenter pipeline) are skipped.  Uses MARP's ``<style scoped>``
    which applies only to the current slide.
    """
    frontmatter, slides = _parse_marp(content)
    result: List[str] = []

    for slide in slides:
        # Skip if already classified or no table present
        if "table-small" in slide or "table-tiny" in slide or "|" not in slide:
            result.append(slide)
            continue

        rows = _count_table_data_rows(slide)
        if rows < 5:
            result.append(slide)
            continue

        tier = "tiny" if rows >= 8 else "small"
        style_block = (
            f"<style>\n"
            f"table {{ {_TABLE_SCALE[tier]} }}\n"
            f"table th, table td {{ {_TABLE_PAD[tier]} }}\n"
            f"</style>"
        )
        # Prepend scoped style before slide content
        slide = style_block + "\n\n" + slide
        result.append(slide)

    return _join_marp(frontmatter, result)


# ---------------------------------------------------------------------------
# Transform 1d — Remove trailing Sommaire / duplicate TOC slides
# ---------------------------------------------------------------------------

_SOMMAIRE_RE = re.compile(r"^#\s+Sommaire\s*$", re.MULTILINE)
_TOC_CONTINUED_RE = re.compile(
    r"^##\s+Table of Contents\s*\(continued\)", re.MULTILINE,
)


def remove_trailing_sommaire(content: str) -> str:
    """Remove redundant Sommaire / TOC slides at the end of the deck.

    The presenter pipeline sometimes appends a Sommaire section (lead +
    content + continuation) that duplicates the TOC near the beginning.
    This transform trims those trailing slides.
    """
    frontmatter, slides = _parse_marp(content)
    if len(slides) < 5:
        return content

    # Walk backwards from the end and remove slides that are part of
    # a trailing Sommaire block (lead, TOC list, or TOC continued).
    cut_from = len(slides)
    for i in range(len(slides) - 1, max(len(slides) - 6, -1), -1):
        slide = slides[i]
        is_sommaire_lead = (
            _SOMMAIRE_RE.search(slide) and "<!-- _class: lead -->" in slide
        )
        is_sommaire_content = (
            "## Sommaire" in slide
            and "[Chapitre" in slide
        )
        is_toc_continued = bool(_TOC_CONTINUED_RE.search(slide))
        if is_sommaire_lead or is_sommaire_content or is_toc_continued:
            cut_from = i
        else:
            break

    if cut_from < len(slides):
        removed = len(slides) - cut_from
        slides = slides[:cut_from]
        # Silently removed — summary will show updated slide count
    return _join_marp(frontmatter, slides)


# ---------------------------------------------------------------------------
# Transform 1d2 — Remove garbled Sommaire block
# ---------------------------------------------------------------------------


def remove_garbled_sommaire(content: str) -> str:
    """Remove garbled Sommaire lead+content slide block.

    The presenter pipeline sometimes produces a redundant Sommaire block
    consisting of a lead ``# Sommaire`` slide followed by a garbled
    ``## Sommaire`` content slide with mixed formats (numbered entries,
    pipe artifacts, self-referencing ``[Sommaire](#sommaire)``).

    This transform detects and removes the block if:
      - A ``# Sommaire`` lead slide exists outside the first 3 slides
      - The lead slide has no substantive content besides the heading
      - Optionally followed by a ``## Sommaire`` slide with mixed entries

    A clean ``## Sommaire`` (from ``clean_toc_slide``) near slide 2 is preserved.
    """
    frontmatter, slides = _parse_marp(content)
    if len(slides) < 5:
        return content

    to_remove: set = set()
    for i, slide in enumerate(slides):
        # Skip the clean TOC (first 3 slides)
        if i < 3:
            continue
        # Detect lead Sommaire slides (section divider)
        if _SOMMAIRE_RE.search(slide) and "<!-- _class: lead -->" in slide:
            to_remove.add(i)
        # Detect garbled Sommaire content slides with mixed formats
        elif "## Sommaire" in slide and i >= 3:
            # Garbled if: has self-reference, "Other" entry, or pipe-prefixed entries
            if (
                "[Sommaire](#sommaire)" in slide
                or "[Other]" in slide
                or re.search(r"^\d+\.\s+\[\|", slide, re.MULTILINE)
            ):
                to_remove.add(i)

    if not to_remove:
        return content

    slides = [s for i, s in enumerate(slides) if i not in to_remove]
    return _join_marp(frontmatter, slides)


# ---------------------------------------------------------------------------
# Transform 1e — Clean Table of Contents slide
# ---------------------------------------------------------------------------

_TOC_TITLE_ENTRY_RE = re.compile(
    r"^-\s+\[(?!Chapitre\s)(?!\d+\.\s)(.+?)\]\(#.+?\)\s*$", re.MULTILINE,
)
_TOC_NUMBERED_ENTRY_RE = re.compile(
    r"^\d+\.\s+\[(.+?)\]\(#(.+?)\)\s*$", re.MULTILINE,
)


_CHAPTER_ENTRY_RE = re.compile(
    r"^-\s+\[Chapitre\s+\d+\s*[—–-]\s*.+?\]\(#.+?\)\s*$", re.MULTILINE,
)


def clean_toc_slide(content: str) -> str:
    """Clean the Table of Contents slide near the beginning.

    Fixes:
      - Merges chapter entries from ``Table of Contents (continued)`` slides.
      - Removes non-chapter entries (the document title link, Sommaire link).
      - Normalizes ``10. [Title](#anchor)`` → ``- [Chapitre 10 — Title](#anchor)``.
      - Renames heading from ``Table of Contents`` to ``Sommaire``.
    """
    frontmatter, slides = _parse_marp(content)
    if not slides:
        return content

    # Find the main TOC slide (one of the first 5 slides)
    # Supports both English and French headings (idempotent re-runs)
    toc_idx = None
    for i in range(min(5, len(slides))):
        has_toc = (
            "## Table of Contents" in slides[i]
            or "## Sommaire" in slides[i]
        )
        if has_toc and "(continued)" not in slides[i]:
            toc_idx = i
            break
    if toc_idx is None:
        return content

    slide = slides[toc_idx]

    # 1. Collect chapter entries from "Table of Contents (continued)" slides
    #    and merge them into the main TOC before those slides are removed.
    #    Deduplicate by anchor to avoid double-merging from multiple continued pages.
    extra_entries: List[str] = []
    seen_anchors: set = set()
    for i in range(len(slides)):
        if i == toc_idx:
            continue
        if _TOC_CONTINUED_RE.search(slides[i]):
            for m in _CHAPTER_ENTRY_RE.finditer(slides[i]):
                # Extract anchor for dedup
                anchor_m = re.search(r"\(#(.+?)\)", m.group(0))
                anchor = anchor_m.group(1) if anchor_m else m.group(0)
                if anchor not in seen_anchors:
                    seen_anchors.add(anchor)
                    extra_entries.append(m.group(0))

    # 2. Rename heading to French
    slide = slide.replace("## Table of Contents", "## Sommaire")

    # 2b. Strip pipe artifacts and bold markers from TOC entry text (v1.5)
    #     ``[Chapitre 1 — | **Vue générale**]`` → ``[Chapitre 1 — Vue générale]``
    slide = re.sub(
        r"(\[Chapitre\s+\d+\s*—)\s*\|\s*\*{0,2}(.*?)\*{0,2}\]",
        r"\1 \2]",
        slide,
    )

    # 3. Remove non-chapter entries (e.g. the document title link)
    slide = _TOC_TITLE_ENTRY_RE.sub("", slide)

    # 4. Normalize numbered entries: ``10. [Recommandations...]`` →
    #    ``- [Chapitre 10 — Recommandations...]``
    def _fix_numbered(m: re.Match) -> str:
        title = m.group(1).strip()
        anchor = m.group(2).strip()
        num_m = re.match(r"^(\d+)", m.group(0).strip())
        num = num_m.group(1) if num_m else "?"
        return f"- [Chapitre {num} — {title}](#{anchor})"

    slide = _TOC_NUMBERED_ENTRY_RE.sub(_fix_numbered, slide)

    # 5. Append merged chapter entries from "continued" slides
    #    (only entries not already present in the main TOC)
    if extra_entries:
        existing_anchors = {
            m.group(1)
            for m in re.finditer(r"\(#(.+?)\)", slide)
        }
        new_entries = [
            e for e in extra_entries
            if not any(a in e for a in existing_anchors)
        ]
        if new_entries:
            lines = slide.split("\n")
            last_entry_idx = -1
            for idx, line in enumerate(lines):
                if line.strip().startswith("- [Chapitre"):
                    last_entry_idx = idx
            if last_entry_idx >= 0:
                for entry in new_entries:
                    last_entry_idx += 1
                    lines.insert(last_entry_idx, entry)
                slide = "\n".join(lines)

    # 6. Remove "Table of Contents (continued)" slides that are now merged
    continued_indices = [
        i for i in range(len(slides))
        if i != toc_idx and _TOC_CONTINUED_RE.search(slides[i])
    ]
    if continued_indices:
        slides = [s for i, s in enumerate(slides) if i not in continued_indices]

    # 7. Recalculate toc_idx after removal (slides before it may have been removed)
    removed_before = sum(1 for ci in continued_indices if ci < toc_idx)
    toc_idx -= removed_before

    # 8. Remove the TOC-expansion comment if present
    slide = slide.replace(
        "<!-- Table of contents — expanded by pres_marp_render -->", ""
    )

    # 7. Collapse multiple blank lines
    slide = re.sub(r"\n{3,}", "\n\n", slide)

    # 9b. Synchronize TOC entries with actual chapter headings (v1.5)
    #     Rebuild TOC from chapters detected in the slides to ensure
    #     numbering matches after empty chapter removal + renumbering.
    actual_chapters = _detect_chapters(slides)
    if actual_chapters:
        toc_lines = []
        for ch in actual_chapters:
            anchor = re.sub(
                r"[^\w\s-]", "",
                ch.title.lower().replace(" ", "-").replace("—", ""),
            ).strip("-")
            toc_lines.append(
                f"- [Chapitre {ch.chapter_num} — {ch.title}](#{anchor})"
            )
        # Replace existing entries with rebuilt ones
        slide_lines = slide.splitlines()
        new_lines = []
        skipping_entries = False
        for line in slide_lines:
            if line.strip().startswith("- [Chapitre"):
                if not skipping_entries:
                    # Insert all rebuilt entries here
                    new_lines.extend(toc_lines)
                    skipping_entries = True
                # Skip old entry
                continue
            else:
                skipping_entries = False
                new_lines.append(line)
        slide = "\n".join(new_lines)

    # 10. Inject TOC-specific styling for comfortable spacing (v1.3)
    toc_style = (
        '<style scoped>\n'
        'section { font-size: 0.85em; }\n'
        'section li { margin-bottom: 0.35em; line-height: 1.5; }\n'
        'section li a { text-decoration: none; color: var(--koas-primary, #0066cc); }\n'
        '</style>'
    )
    slide = toc_style + "\n\n" + slide

    slides[toc_idx] = slide
    return _join_marp(frontmatter, slides)


# ---------------------------------------------------------------------------
# Transform 1f — Remove empty chapter dividers
# ---------------------------------------------------------------------------


def remove_empty_chapter_dividers(content: str) -> str:
    """Remove section-divider slides that have no content slides following them.

    When the presenter generates chapters that contain no content (only a
    ``# Chapitre N — Title`` lead slide), this results in consecutive
    section dividers with no information between them.  This transform
    detects and removes such empty chapters, then re-triggers chapter
    renumbering so the remaining chapters are sequential.

    A section divider is considered "empty" if the very next slide is also
    a section divider (lead with ``# Chapitre``) or the slide is the last
    non-traceability slide.
    """
    frontmatter, slides = _parse_marp(content)
    if len(slides) < 3:
        return content

    def _is_chapter_divider(s: str) -> bool:
        return (
            "<!-- _class: lead -->" in s
            and bool(_CHAPTER_RE.search(s))
        )

    # Mark slides for removal: chapter dividers whose NEXT slide is also a divider
    to_remove: set = set()
    for i in range(len(slides) - 1):
        if _is_chapter_divider(slides[i]) and _is_chapter_divider(slides[i + 1]):
            to_remove.add(i)

    # Also check if the last chapter divider has nothing after it
    # (or only traceability)
    for i in range(len(slides) - 1, -1, -1):
        if "Traçabilité de génération" in slides[i]:
            continue
        if _is_chapter_divider(slides[i]):
            to_remove.add(i)
        break

    if not to_remove:
        return content

    slides = [s for i, s in enumerate(slides) if i not in to_remove]
    result = _join_marp(frontmatter, slides)

    # Renumber remaining chapters to be sequential
    result = renumber_chapters(result)
    return result


# ---------------------------------------------------------------------------
# Transform 2 — Progress bar
# ---------------------------------------------------------------------------

_PROGRESS_CSS = """\
<style>
/* MARP post-processor: slide overflow guard */
section {
  overflow: hidden;
}
/* Base font — !important required to defeat MARP default-theme cascade.
 * MARP re-emits `@import 'default'` rules with identical specificity after
 * our custom theme rules, so the last declaration wins. !important is the
 * only reliable override mechanism. */
section {
  font-size: 22px !important;
}
section h2 {
  font-size: 1.5em !important;
}
/* Page numbers — discrete, bottom-right (belt-and-suspenders with theme) */
section::after {
  font-size: 12px !important;
  color: #999 !important;
}
/* Blockquote — smaller than body for callout emphasis (reinforces theme) */
section blockquote {
  font-size: 0.88em !important;
}
/* Image centering is handled by center_images_in_html() which runs as
 * HTML post-processing AFTER marp-cli. MARP's CSS sanitizer strips display,
 * margin, and text-align properties from all image-related CSS rules in both
 * theme CSS and <style> blocks, even with !important. */
/* Reduce image heights to account for heading + footer + nav + progress.
 * MARP 16:9 = 720px high. Overhead budget: top-pad(50) + bottom-pad(60)
 * + heading(45) + footer(15) + nav(28) + progress(4) = 202px.
 * Available = 518px = ~72vh. With body text below: ~66vh.
 * Conservative defaults (auto_constrain tightens further when needed). */
section .figure-landscape img {
  max-height: 44vh;
  max-width: 100%;
}
section .figure img {
  max-height: 48vh;
  max-width: 100%;
}
section .figure-full img {
  max-height: 62vh;
  max-width: 100%;
}
/* Image alt-text caption — keep it compact */
section .figure-landscape,
section .figure,
section .figure-full {
  margin-bottom: 0;
}
/* Figure captions — class-based matching (v1.3) */
section .fig-caption {
  font-size: 0.58em;
  color: #636e72;
  font-style: italic;
  margin-top: 0.2em;
  line-height: 1.3;
}
/* Footer — smaller and more compact */
section footer {
  font-size: 9px;
  opacity: 0.7;
}
/* Table centering — center tables horizontally when alone on slide */
section table {
  width: fit-content;
  margin-left: auto;
  margin-right: auto;
}
/* Tables inside layout cells: MARP strips `table table` selectors,
 * class/data-* attributes on <table>, <div style="..."> wrappers,
 * and `table` rules from <style scoped>. No CSS-based fix survives.
 * Tables in layout cells remain auto-centered by the global rule above.
 * For explicit left-alignment, use inline HTML tables in the source. */
/* Table overflow guard — compact classified tables (harmonized with theme v1.3) */
section .table-small table {
  font-size: max(0.65em, 13px);
}
section .table-small th,
section .table-small td {
  padding: 3px 8px;
  line-height: 1.3;
}
section .table-tiny table {
  font-size: max(0.50em, 11px);
}
section .table-tiny th,
section .table-tiny td {
  padding: 2px 5px;
  line-height: 1.25;
}
/* MARP post-processor: progress bar */
.progress-track {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: rgba(0, 0, 0, 0.06);
  z-index: 100;
  pointer-events: none;
}
.progress-fill {
  height: 100%;
  background: var(--koas-primary, #0066cc);
}
/* Lead / section slides: light bar */
section.lead .progress-track,
section.section .progress-track {
  background: rgba(255, 255, 255, 0.12);
}
section.lead .progress-fill,
section.section .progress-fill {
  background: rgba(255, 255, 255, 0.55);
}
</style>"""


def inject_progress_bar(content: str) -> str:
    """Inject a thin progress bar at the bottom of every slide.

    Width is proportional to slide position: slide 1 → 1/N, slide N → 100%.
    Uses ``--koas-primary`` colour; adapts on lead/section slides.
    """
    frontmatter, slides = _parse_marp(content)
    total = len(slides)
    if total == 0:
        return content

    result: List[str] = []
    for i, slide in enumerate(slides):
        pct = round(100 * (i + 1) / total, 1)
        bar = (
            f'<div class="progress-track">'
            f'<div class="progress-fill" style="width:{pct}%"></div>'
            f'</div>'
        )
        # CSS injected once on first slide
        if i == 0:
            slide = _PROGRESS_CSS + "\n" + slide.lstrip("\n")
        # Bar appended at end of slide content
        slide = slide.rstrip("\n") + "\n\n" + bar
        result.append(slide)

    return _join_marp(frontmatter, result)


# ---------------------------------------------------------------------------
# Transform 3 — Chapter navigation
# ---------------------------------------------------------------------------

_CHAPNAV_CSS = """\
<style>
/* MARP post-processor: chapter navigation */
.chapter-nav {
  position: absolute;
  top: 10px;
  right: 16px;
  display: flex;
  gap: 3px;
  z-index: 100;
}
.chapter-nav a {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  height: 20px;
  padding: 0 3px;
  border-radius: 10px;
  font-size: 9px;
  font-weight: 600;
  color: #aaa;
  background: #f0f0f0;
  text-decoration: none;
  line-height: 1;
}
.chapter-nav a.active {
  background: var(--koas-primary, #0066cc);
  color: #fff;
}
.chapter-nav a:hover {
  background: #ddd;
}
.chapter-nav a.active:hover {
  background: #0055aa;
}
/* Lead / section slides: adapt to dark background */
section.lead .chapter-nav a,
section.section .chapter-nav a {
  color: rgba(255, 255, 255, 0.35);
  background: rgba(255, 255, 255, 0.10);
}
section.lead .chapter-nav a.active,
section.section .chapter-nav a.active {
  color: #fff;
  background: rgba(255, 255, 255, 0.35);
}
section.lead .chapter-nav a:hover,
section.section .chapter-nav a:hover {
  background: rgba(255, 255, 255, 0.22);
}
</style>"""


def inject_chapter_nav(content: str) -> str:
    """Inject a chapter-navigation strip at top-right of every slide.

    Each chapter appears as a small numbered circle.  The current chapter
    is highlighted.  In MARP HTML output the circles are anchor-links
    that jump to the chapter's section-divider slide.
    """
    frontmatter, slides = _parse_marp(content)
    chapters = _detect_chapters(slides)
    if not chapters:
        return content

    result: List[str] = []
    for i, slide in enumerate(slides):
        current = _current_chapter(i, chapters)

        # Build nav links — MARP HTML viewer uses 1-based #N anchors
        parts: List[str] = []
        for ch in chapters:
            # +2 because slide_idx is 0-based in the slides list,
            # but slide 0 in the list is slide 2 in the file
            # (slide 1 = frontmatter).
            # Actually MARP counts from 1 and frontmatter is not a slide,
            # so slides[0] = MARP slide 1.
            slide_anchor = ch.slide_idx + 1
            cls = ' class="active"' if ch.chapter_num == current else ""
            title_attr = f"Ch.{ch.chapter_num} — {ch.short_title}"
            # Escape quotes in title
            title_attr = title_attr.replace('"', "&quot;")
            parts.append(
                f'<a href="#{slide_anchor}"{cls}'
                f' title="{title_attr}">'
                f"{ch.chapter_num}</a>"
            )

        nav_html = '<div class="chapter-nav">' + "".join(parts) + "</div>"

        # CSS injected once on first slide; nav prepended at top.
        # IMPORTANT: blank line after HTML block so markdown-it resumes
        # markdown parsing for headings, lists, etc.
        if i == 0:
            slide = _CHAPNAV_CSS + "\n" + nav_html + "\n\n" + slide.lstrip("\n")
        else:
            slide = nav_html + "\n\n" + slide
        result.append(slide)

    return _join_marp(frontmatter, result)


# ---------------------------------------------------------------------------
# Transform 4 — Per-slide chapter footer
# ---------------------------------------------------------------------------

_FOOTER_DIRECTIVE_RE = re.compile(
    r"^<!--\s*footer:\s*['\"]?.*?['\"]?\s*-->$", re.MULTILINE,
)


def _extract_frontmatter_footer(frontmatter: str) -> Optional[str]:
    """Extract the date portion from the frontmatter footer line.

    Parses ``footer: 'Chapitre 0 — Index général -- 2026-02-13'`` and
    returns ``'2026-02-13'``.  Returns None if no footer or no date found.
    """
    m = re.search(r"^footer:\s*['\"]?.*?--\s*(\d{4}-\d{2}-\d{2})['\"]?\s*$",
                  frontmatter, re.MULTILINE)
    return m.group(1) if m else None


def inject_chapter_footer(content: str) -> str:
    """Inject per-slide ``<!-- footer: ... -->`` directives.

    Each slide's footer shows the current chapter title and the date
    extracted from the frontmatter.  Section-divider and lead slides
    get ``<!-- footer: '' -->`` (hidden footer, matching the CSS rule
    that already hides header/footer on those slides).
    """
    frontmatter, slides = _parse_marp(content)
    chapters = _detect_chapters(slides)
    if not chapters:
        return content

    date_str = _extract_frontmatter_footer(frontmatter) or ""
    date_suffix = f" -- {date_str}" if date_str else ""

    result: List[str] = []
    for i, slide in enumerate(slides):
        # Remove any existing footer directive
        slide = _FOOTER_DIRECTIVE_RE.sub("", slide)

        current = _current_chapter(i, chapters)

        # Find the chapter info for the current chapter
        ch_info = None
        for ch in chapters:
            if ch.chapter_num == current:
                ch_info = ch
                break

        # Section-divider / lead slides: hide footer
        is_special = "<!-- _class: lead -->" in slide or "<!-- _class: section -->" in slide
        if is_special:
            footer = '<!-- footer: "" -->'
        elif ch_info:
            text = f"Chapitre {ch_info.chapter_num} — {ch_info.title}{date_suffix}"
            footer = f'<!-- footer: "{text}" -->'
        else:
            # Pre-chapter slides: no directive → inherit frontmatter default
            footer = ""

        if footer:
            slide = footer + "\n" + slide
        result.append(slide)

    return _join_marp(frontmatter, result)


# ---------------------------------------------------------------------------
# Transform 5 — Traceability slide (audit provenance)
# ---------------------------------------------------------------------------

def _collect_trace_info() -> Dict[str, str]:
    """Gather system and environment info for the traceability slide."""
    # User and host
    try:
        user = getpass.getuser()
    except Exception:
        user = os.environ.get("USER", "unknown")

    hostname = socket.gethostname()
    # Get real network IP (not 127.0.0.1) via UDP connect trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        try:
            ip = socket.gethostbyname(hostname)
        except Exception:
            ip = "N/A"

    # RAGIX version
    try:
        from ragix_core.version import __version__ as ragix_version
    except ImportError:
        ragix_version = "N/A"

    # Timestamp (UTC + local)
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now()
    tz_name = datetime.now().astimezone().tzname() or "local"

    # Python + OS
    py_version = platform.python_version()
    os_info = f"{platform.system()} {platform.release()}"

    return {
        "user": user,
        "hostname": hostname,
        "ip": ip,
        "ragix_version": ragix_version,
        "timestamp_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "timestamp_local": now_local.strftime("%Y-%m-%d %H:%M:%S") + f" {tz_name}",
        "python": py_version,
        "os": os_info,
        "platform": platform.platform(),
    }


def inject_traceability_slide(
    content: str,
    *,
    transforms_applied: Optional[List[str]] = None,
) -> str:
    """Append a traceability slide with provenance metadata.

    The slide records who generated the deck, when, where, and how,
    enabling full reproducibility of the MARP export.
    """
    info = _collect_trace_info()

    transforms_list = transforms_applied or []
    if transforms_list:
        pipeline_str = " → ".join(transforms_list)
    else:
        pipeline_str = "postprocess_marp (default pipeline)"

    slide = (
        '<!-- _paginate: false -->\n'
        '<!-- footer: "" -->\n'
        "\n"
        "## Traçabilité de génération\n"
        "\n"
        "| | |\n"
        "|:---|:---|\n"
        f"| **Générateur** | RAGIX KOAS Presenter v{info['ragix_version']} |\n"
        f"| **Utilisateur** | `{info['user']}@{info['hostname']}` |\n"
        f"| **Adresse IP** | `{info['ip']}` |\n"
        f"| **Horodatage** | {info['timestamp_local']} |\n"
        f"| **Horodatage UTC** | {info['timestamp_utc']} |\n"
        f"| **Système** | {info['os']} — Python {info['python']} |\n"
        f"| **Pipeline** | {pipeline_str} |\n"
        "\n"
        "<style scoped>\n"
        "section { font-size: 20px; }\n"
        "table { width: fit-content; margin: 0 auto; }\n"
        "table th { display: none; }\n"
        "table td { padding: 5px 14px; }\n"
        "table td:first-child { font-size: 0.85em; white-space: nowrap; }\n"
        "table td:last-child { font-family: 'JetBrains Mono', monospace; font-size: 0.82em; }\n"
        "</style>"
    )

    frontmatter, slides = _parse_marp(content)
    slides.append(slide)
    return _join_marp(frontmatter, slides)


# ---------------------------------------------------------------------------
# Transform 5b — Auto-constrain figure slides with body text (v1.3)
# ---------------------------------------------------------------------------

_FIGURE_DIV_RE = re.compile(
    r'<div\s+class="(figure-landscape|figure-full|figure)">', re.IGNORECASE
)

# Matches inline <img> tags (not inside figure divs)
_INLINE_IMG_RE = re.compile(r"<img\s[^>]*>", re.IGNORECASE)


def auto_constrain_figure_slides(content: str) -> str:
    """Constrain figure max-height on slides that have both a figure and body text.

    When a slide contains a figure div AND additional visible text after it
    (body paragraph, caption, description), the figure needs a tighter
    max-height to prevent vertical overflow into the footer zone.

    Also handles slides with multiple inline ``<img>`` tags (not inside KOAS
    figure divs) — e.g. raw ``<p><img src=...></p>`` stacks from source docs.
    When 2+ images are stacked, each gets a height cap to prevent overflow.

    Slides with a figure as the sole content keep the default (generous) limits
    set in ``_PROGRESS_CSS``.

    MARP 16:9 canvas = 720px high.  With heading + body text + footer + nav +
    progress + padding, roughly 240px is consumed by non-figure elements.
    Available for image: ~480px = ~66vh.  We use ~38-55vh with safety margin.
    """
    frontmatter, slides = _parse_marp(content)
    result: List[str] = []

    for slide in slides:
        m = _FIGURE_DIV_RE.search(slide)
        if m:
            # --- Path A: KOAS figure-div slides ---
            fig_class = m.group(1)  # figure-landscape | figure-full | figure

            # Find text content AFTER the last </div> that closes the figure
            last_div_pos = slide.rfind("</div>")
            if last_div_pos < 0:
                result.append(slide)
                continue

            after_figure = slide[last_div_pos + len("</div>"):]
            cleaned = re.sub(r"<!--.*?-->", "", after_figure, flags=re.DOTALL)
            cleaned = re.sub(r"<style.*?</style>", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'<p class="fig-caption">.*?</p>', "", cleaned, flags=re.DOTALL)
            cleaned = cleaned.strip()

            if len(cleaned) < 10:
                result.append(slide)
                continue

            if fig_class == "figure-landscape":
                rule = "section .figure-landscape img { max-height: 38vh !important; }"
            elif fig_class == "figure-full":
                rule = "section .figure-full img { max-height: 55vh !important; }"
            else:
                rule = "section .figure img { max-height: 42vh !important; }"

            style = f"<style scoped>\n{rule}\n</style>"
            slide = style + "\n\n" + slide
            result.append(slide)
            continue

        # --- Path B: inline <img> stacks (not in KOAS figure divs) ---
        img_count = len(_INLINE_IMG_RE.findall(slide))
        if img_count >= 2:
            # Two+ stacked images: cap each to share vertical space.
            # Use px (MARP canvas = 720px); vh resolves against browser
            # viewport in SVG foreignObject, not the slide canvas.
            # Budget: 720px - heading(80) - caption(40) - footer(60) - pad(40) = 500px
            px = 240 if img_count == 2 else 155
            rule = f"section img {{ max-height: {px}px !important; }}"
            style = f"<style scoped>\n{rule}\n</style>"
            slide = style + "\n\n" + slide

        result.append(slide)

    return _join_marp(frontmatter, result)


# ---------------------------------------------------------------------------
# Transform 6 — Auto-shrink dense slides
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


_IMG_MAX_HEIGHT_RE = re.compile(
    r"max-height\s*:\s*(\d+)\s*px", re.IGNORECASE,
)


def _estimate_image_height_px(slide: str) -> int:
    """Return total image height (px) from inline <img> max-height styles.

    When a slide contains ``<img ... style="max-height:250px">`` or similar,
    the image consumes vertical space that reduces the text budget.  This
    helper extracts the declared pixel heights so the shrink heuristic can
    account for them.

    Returns 0 if no sized images are found.
    """
    total = 0
    for m in _INLINE_IMG_RE.finditer(slide):
        hm = _IMG_MAX_HEIGHT_RE.search(m.group(0))
        if hm:
            total += int(hm.group(1))
    return total


def _visible_text_length(slide: str) -> int:
    """Estimate visible text character count (strip HTML, comments, markup)."""
    text = _COMMENT_RE.sub("", slide)
    text = _HTML_TAG_RE.sub("", text)
    # Strip markdown heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Strip bold/italic markers
    text = text.replace("**", "").replace("__", "")
    # Collapse whitespace
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return len(text)


def _count_list_items(slide: str) -> int:
    """Count bullet and numbered list items in visible content."""
    text = _COMMENT_RE.sub("", slide)
    text = _HTML_TAG_RE.sub("", text)
    return len(re.findall(r"^\s*[-*+]\s+|\s*\d+\.\s+", text, re.MULTILINE))


def auto_shrink_dense_slides(content: str) -> str:
    """Inject per-slide font reduction for slides with overflowing content.

    Estimates visible text density and injects a scoped ``<style>`` block
    when content is likely to overflow the MARP 16:9 slide area.

    The effective length combines:
      - Visible text chars (excluding HTML, comments, headings)
      - List-item wrap penalty (+40 per item)
      - Image height penalty (+1.5 chars per px of declared ``max-height``)

    Thresholds (effective chars):
      - ≤ 900:  no change (normal 22px base)
      - 901–1200: 0.88em (~19px) — slight reduction
      - 1201–1600: 0.78em (~17px) — moderate reduction
      - > 1600: 0.68em (~15px) — compact
    """
    frontmatter, slides = _parse_marp(content)
    result = []

    for i, slide in enumerate(slides):
        # Skip lead/section divider slides (no content overflow risk)
        if "<!-- _class: lead -->" in slide or "<!-- _class: section -->" in slide:
            result.append(slide)
            continue

        # Skip TOC slides — they have their own styling (v1.3)
        if "## Sommaire" in slide or "## Table of Contents" in slide:
            result.append(slide)
            continue

        vlen = _visible_text_length(slide)
        items = _count_list_items(slide)
        img_h = _estimate_image_height_px(slide)

        # In side-by-side layout tables ([I,T]/[T,I]), the image shares
        # horizontal space with text — it does NOT consume vertical budget.
        # Detect by checking if the <img> sits inside a layout <td>.
        is_side_by_side = (
            "display:table !important" in slide
            and bool(re.search(r"<td[^>]*><img\s", slide))
        )
        img_penalty = 0 if is_side_by_side else int(img_h * 1.5)

        # Heuristic: long items wrap more, so boost effective length.
        # Image penalty: each px of declared image height displaces ~1.5 chars
        # of text capacity (720px canvas, ~22px/line, ~80 chars/line).
        effective = vlen + items * 40 + img_penalty

        if effective <= 900:
            result.append(slide)
            continue

        # Cap font reduction when slide contains classified tables
        # to prevent cascading below readable threshold (v1.3)
        has_table_density = "table-small" in slide or "table-tiny" in slide

        if effective <= 1200:
            scale = "0.88em"
        elif effective <= 1600:
            scale = "0.78em" if not has_table_density else "0.88em"
        else:
            scale = "0.68em" if not has_table_density else "0.88em"

        # Build scoped style rules
        rules = [
            f"section {{ font-size: {scale}; }}",
            "section li { margin-bottom: 0.15em; line-height: 1.35; }",
        ]

        # When an image is stacked above/below a table, the vertical budget
        # is severely reduced — compact table padding and line-height too.
        # Skip for side-by-side layouts where image and table share width.
        has_table = bool(re.search(r"^\|.*\|.*\|", slide, re.MULTILINE))
        if img_h > 0 and has_table and not has_table_density and not is_side_by_side:
            rules.append(
                "section table { font-size: 0.85em; line-height: 1.25; }"
            )
            rules.append(
                "section table th, section table td "
                "{ padding: 3px 8px; }"
            )

        style = (
            "<style scoped>\n"
            + "\n".join(rules)
            + "\n</style>"
        )

        # When images are present and the slide overflows, reduce image
        # max-height proportionally.  The inline style on <img> survives
        # MARP rendering (no specificity war).
        # Skip for side-by-side layouts — image height is independent.
        if img_h > 0 and not is_side_by_side:
            # Target: reclaim ~20% of image height per shrink tier
            if scale == "0.88em":
                cap_ratio = 0.80
            elif scale == "0.78em":
                cap_ratio = 0.70
            else:
                cap_ratio = 0.60

            def _cap_img_height(m: "re.Match[str]") -> str:
                tag = m.group(0)
                hm = _IMG_MAX_HEIGHT_RE.search(tag)
                if not hm:
                    return tag
                old_h = int(hm.group(1))
                new_h = max(120, int(old_h * cap_ratio))
                if new_h >= old_h:
                    return tag
                return tag[:hm.start(1)] + str(new_h) + tag[hm.end(1):]

            slide = _INLINE_IMG_RE.sub(_cap_img_height, slide)

        # Insert style at the beginning (before any content)
        slide = style + "\n\n" + slide
        result.append(slide)

    return _join_marp(frontmatter, result)


# ---------------------------------------------------------------------------
# Transform 7 — Logo injection (programmatic, folder-based)
# ---------------------------------------------------------------------------

_LOGO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def inject_logos(
    content: str,
    logos_dir: str,
    *,
    assets_subdir: str = "assets",
) -> str:
    """Inject logos on title slide (horizontal) and last content slide (vertical).

    Logos are discovered from *logos_dir*, sorted alphabetically, and
    referenced via *assets_subdir* (relative to the MARP file).

    Args:
        content: MARP markdown content.
        logos_dir: Path to directory containing logo image files.
        assets_subdir: Subdirectory name where logos are accessible from MARP.

    Returns:
        Modified MARP content with logos injected.
    """
    from pathlib import Path as P
    import shutil

    d = P(logos_dir)
    if not d.is_dir():
        return content

    # Discover logos, sorted alphabetically (case-insensitive)
    logos = sorted(
        [f for f in d.iterdir() if f.suffix.lower() in _LOGO_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )
    if not logos:
        return content

    frontmatter, slides = _parse_marp(content)
    if len(slides) < 3:
        return content  # need at least title + content + traceability

    # Build logo references (relative to MARP file)
    logo_refs = [f"{assets_subdir}/{logo.name}" for logo in logos]

    # --- Title slide: horizontal logo bar at bottom (white badges for visibility) ---
    title_imgs = " ".join(
        f'<div style="background: white; border-radius: 8px; padding: 6px 14px; '
        f'display: flex; align-items: center;">'
        f'<img src="{ref}" alt="{P(ref).stem}" style="height: 40px; object-fit: contain;" />'
        f'</div>'
        for ref in logo_refs
    )
    title_logo_html = (
        "\n\n"
        '<div style="position: absolute; bottom: 40px; left: 0; right: 0; '
        'display: flex; justify-content: center; align-items: center; gap: 30px;">\n'
        f"  {title_imgs}\n"
        "</div>"
    )
    slides[0] = slides[0] + title_logo_html

    # --- Last content slide (before traceability): vertical logo column ---
    # Find the last non-traceability slide
    trace_idx = len(slides) - 1
    if "Traçabilité de génération" in slides[trace_idx]:
        last_content_idx = trace_idx - 1
    else:
        last_content_idx = trace_idx

    vert_imgs = "\n  ".join(
        f'<div style="background: white; border-radius: 6px; padding: 5px 12px; '
        f'display: flex; align-items: center;">'
        f'<img src="{ref}" alt="{P(ref).stem}" style="max-height: 60px; max-width: 180px; object-fit: contain;" />'
        f'</div>'
        for ref in logo_refs
    )
    vert_logo_html = (
        "\n\n"
        '<div style="position: absolute; bottom: 40px; right: 50px; '
        'display: flex; flex-direction: column; align-items: flex-end; gap: 14px;">\n'
        f"  {vert_imgs}\n"
        "</div>"
    )
    slides[last_content_idx] = slides[last_content_idx] + vert_logo_html

    return _join_marp(frontmatter, slides)


# ---------------------------------------------------------------------------
# Transform 7.5 — Layout preprocess  (auto-detect image layout from shape)
# ---------------------------------------------------------------------------
#
# Scans each slide for bare <img> tags.  When found, probes image dimensions,
# classifies the shape (tall / wide / square), and wraps the slide content in
# layout directives that Transform 8 will expand into inline HTML.
#
# Slides that already contain ``<!-- layout: ... -->`` are skipped.

# Soft dependency on Pillow for raster image probing
try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _PILImage = None  # type: ignore[assignment]
    _HAS_PIL = False

# SVG dimension patterns (same as pres_asset_catalog, duplicated to avoid import)
_SVG_W_RE = re.compile(r'<svg[^>]*\bwidth\s*=\s*["\']([^"\']+)["\']', re.I)
_SVG_H_RE = re.compile(r'<svg[^>]*\bheight\s*=\s*["\']([^"\']+)["\']', re.I)
_SVG_VB_RE = re.compile(
    r'<svg[^>]*\bviewBox\s*=\s*["\']'
    r'\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*["\']',
    re.I,
)

# Bare <img> tag (with or without inline style)
_BARE_IMG_RE = re.compile(
    r'<img\s+src="(?P<src>[^"]+)"'
    r'(?:\s+alt="(?P<alt>[^"]*)")?'
    r'[^>]*/?>',
)

# Markdown image reference: ![alt](path)
_MD_IMG_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)',
)


def _probe_image_dimensions(
    img_path: str, *, base_dir: Optional[Path] = None,
) -> Optional[Tuple[int, int]]:
    """Return (width, height) for an image file.  Returns None if unavailable.

    Supports SVG (viewBox / width+height attributes) and raster (via PIL).
    *base_dir* is prepended to relative paths.
    """
    p = Path(img_path)
    if not p.is_absolute() and base_dir:
        p = base_dir / p
    if not p.exists():
        return None

    ext = p.suffix.lower()
    if ext == ".svg":
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:4096]
            # Try explicit width/height first
            wm, hm = _SVG_W_RE.search(head), _SVG_H_RE.search(head)
            if wm and hm:
                try:
                    w = int(float(wm.group(1).replace("px", "").replace("pt", "")))
                    h = int(float(hm.group(1).replace("px", "").replace("pt", "")))
                    if w > 0 and h > 0:
                        return (w, h)
                except ValueError:
                    pass
            # Fallback to viewBox
            vb = _SVG_VB_RE.search(head)
            if vb:
                w = int(float(vb.group(3)))
                h = int(float(vb.group(4)))
                if w > 0 and h > 0:
                    return (w, h)
        except Exception:
            pass
        return None

    # Raster via PIL
    if _HAS_PIL and ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        try:
            with _PILImage.open(p) as img:
                return img.size  # (width, height)
        except Exception:
            pass
    return None


def _classify_shape(w: int, h: int) -> str:
    """Classify aspect ratio: ``'tall'``, ``'wide'``, or ``'square'``."""
    ratio = w / h if h > 0 else 1.0
    if ratio < 0.9:
        return "tall"
    elif ratio > 1.5:
        return "wide"
    return "square"


def _auto_height(shape: str, has_text: bool) -> str:
    """Pick a default max-height based on image shape and content presence."""
    if not has_text:
        return "460px"   # full-slide: use most of the 720px viewport
    if shape == "tall":
        return "420px"   # side-by-side: tall image needs vertical space
    if shape == "wide":
        return "280px"   # stacked: wide image sits on top, text below
    return "380px"       # square in side-by-side


def _auto_width(shape: str) -> str:
    """Pick image-cell width for side-by-side layouts."""
    if shape == "square":
        return "45%"
    return "38%"         # tall: narrower column is enough


def _detect_slide_images(slide: str) -> List[Dict[str, str]]:
    """Find bare ``<img>`` tags and ``![alt](src)`` refs in a slide.

    Returns a list of dicts with keys: ``src``, ``alt``, ``full_match``.
    """
    results: List[Dict[str, str]] = []
    for m in _BARE_IMG_RE.finditer(slide):
        results.append({
            "src": m.group("src"),
            "alt": m.group("alt") or "",
            "full_match": m.group(0),
        })
    for m in _MD_IMG_RE.finditer(slide):
        results.append({
            "src": m.group("src"),
            "alt": m.group("alt") or "",
            "full_match": m.group(0),
        })
    return results


def _has_text_content(slide: str, image_matches: List[Dict[str, str]]) -> bool:
    """Check whether a slide has meaningful text/tables beyond the image(s).

    Removes images, headings, style blocks, and MARP directives, then checks
    if significant content remains.
    """
    text = slide
    for img in image_matches:
        text = text.replace(img["full_match"], "")
    # Strip headings, HTML comments, style blocks, progress divs
    text = re.sub(r"^##?\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r'<div class="progress-track">.*?</div>', "", text, flags=re.DOTALL)
    text = text.strip()
    # Meaningful if there's a table, blockquote, list, or 20+ chars of text
    return len(text) > 20


def layout_preprocess(
    content: str,
    *,
    base_dir: Optional[str] = None,
) -> str:
    """Auto-detect image layout for slides with bare ``<img>`` tags.

    For each slide that contains one or more bare images (``<img>`` or
    ``![]()``) and does **not** already have a ``<!-- layout: ... -->``
    directive, this function:

    1. Probes image dimensions (SVG viewBox or raster via PIL).
    2. Classifies the shape: tall (W/H < 0.9), wide (W/H ≥ 1.5), square.
    3. Picks a layout: ``[I,T]`` for tall/square, ``[I;T]`` for wide.
    4. Wraps the slide content in layout directive comments.

    Slides with existing ``<!-- layout: ... -->`` directives are untouched.
    Slides with no images or images that cannot be probed are untouched.

    Args:
        content: MARP markdown (with frontmatter).
        base_dir: Directory for resolving relative image paths.  Defaults to
            the current working directory.

    Returns:
        MARP content with layout directives injected where appropriate.
    """
    if base_dir is None:
        _base = Path.cwd()
    else:
        _base = Path(base_dir)

    frontmatter, slides = _parse_marp(content)
    changed = 0

    for i, slide in enumerate(slides):
        # Skip slides that already have layout directives
        if "<!-- layout:" in slide:
            continue

        images = _detect_slide_images(slide)
        if not images:
            continue

        # For now: handle 1-image slides (most common case)
        if len(images) == 1:
            img = images[0]
            dims = _probe_image_dimensions(img["src"], base_dir=_base)
            if dims is None:
                logger.debug("layout_preprocess: cannot probe %s, skipping", img["src"])
                continue

            w, h = dims
            shape = _classify_shape(w, h)
            has_text = _has_text_content(slide, images)

            if not has_text:
                # Image-only slide: just center it, no layout directive needed
                centered = (
                    f'<div style="text-align:center">'
                    f'<img src="{img["src"]}" alt="{img["alt"]}" '
                    f'style="max-height:460px;max-width:95%;object-fit:contain" />'
                    f'</div>'
                )
                slides[i] = slide.replace(img["full_match"], centered)
                changed += 1
                continue

            # Build layout directive
            auto_h = _auto_height(shape, has_text)
            alt_text = img["alt"] or img["src"].rsplit("/", 1)[-1]

            if shape == "wide":
                layout_type = "[I;T]"
            else:
                layout_type = "[I,T]"

            img_directive = f'<!-- I: {img["src"]} | alt: {alt_text} | h: {auto_h} -->'

            # Split slide into: before-image (heading + style), after-image (content)
            # The image is replaced by the layout block
            before_img, _, after_img = slide.partition(img["full_match"])

            # Build new slide: heading/style, then layout block wrapping content
            new_slide = (
                f'{before_img.rstrip()}\n\n'
                f'<!-- layout: {layout_type} -->\n'
                f'{img_directive}\n\n'
                f'{after_img.strip()}\n\n'
                f'<!-- /layout -->'
            )
            slides[i] = new_slide
            changed += 1

        elif len(images) == 2:
            # 2 images: probe both, use [I,I;t,t] grid
            dims1 = _probe_image_dimensions(images[0]["src"], base_dir=_base)
            dims2 = _probe_image_dimensions(images[1]["src"], base_dir=_base)
            if dims1 is None or dims2 is None:
                continue

            alt1 = images[0]["alt"] or images[0]["src"].rsplit("/", 1)[-1]
            alt2 = images[1]["alt"] or images[1]["src"].rsplit("/", 1)[-1]
            dir1 = f'<!-- I: {images[0]["src"]} | alt: {alt1} | h: 240px -->'
            dir2 = f'<!-- I: {images[1]["src"]} | alt: {alt2} | h: 240px -->'

            # Remove both images from slide, keep remaining text
            text = slide
            for img in images:
                text = text.replace(img["full_match"], "")

            # Extract heading (before first image position)
            heading_end = slide.find(images[0]["full_match"])
            heading_part = slide[:heading_end].rstrip() if heading_end > 0 else ""
            text_part = text
            # Remove heading from text_part to avoid duplication
            if heading_part:
                text_part = text_part.replace(heading_part, "", 1)
            text_part = text_part.strip()

            new_slide = (
                f'{heading_part}\n\n'
                f'<!-- layout: [I,I;t,t] -->\n'
                f'{dir1}\n'
                f'{dir2}\n\n'
                f'{text_part}\n\n'
                f'<!-- /layout -->'
            )
            slides[i] = new_slide
            changed += 1

    logger.info("layout_preprocess: %d slides auto-wrapped with layout directives", changed)
    return _join_marp(frontmatter, slides)


# ---------------------------------------------------------------------------
# Transform 8 — Layout directives  (expand shorthand → inline HTML)
# ---------------------------------------------------------------------------

# Matches a layout block:  <!-- layout: [I,T] --> ... <!-- /layout -->
_LAYOUT_BLOCK_RE = re.compile(
    r"<!-- layout:\s*(\[[^\]]+\])\s*-->\n"
    r"(.*?)"
    r"<!-- /layout -->",
    re.DOTALL,
)

# Matches an image directive:  <!-- I: path | alt: text | h: NNNpx | w: NN% -->
_IMG_DIRECTIVE_RE = re.compile(
    r"<!-- I:\s*(?P<path>[^\s|]+)"
    r"(?:\s*\|\s*alt:\s*(?P<alt>[^|]*?))?"
    r"(?:\s*\|\s*h:\s*(?P<h>[^|]*?))?"
    r"(?:\s*\|\s*w:\s*(?P<w>[^|]*?))?"
    r"\s*-->"
)

# Default layout dimensions (from koas-typography.yaml)
_LAYOUT_IMG_WIDTH = "38%"
_LAYOUT_CONTENT_WIDTH = "62%"
_LAYOUT_GAP = "18px"
_DEFAULT_H_SIDE = "420px"      # [I,T] portrait image default max-height
_DEFAULT_H_TOP = "280px"       # [I;T] landscape image default max-height


def _parse_img_directive(text: str, default_h: str = _DEFAULT_H_SIDE) -> Optional[Dict[str, str]]:
    """Extract image path, alt, height, width from an ``<!-- I: ... -->`` comment."""
    m = _IMG_DIRECTIVE_RE.search(text)
    if not m:
        return None
    path = m.group("path").strip()
    alt = (m.group("alt") or "").strip() or path.rsplit("/", 1)[-1]
    h = (m.group("h") or "").strip() or default_h
    w = (m.group("w") or "").strip()
    return {"path": path, "alt": alt, "h": h, "w": w, "full_match": m.group(0)}


def _img_tag(info: Dict[str, str], *, max_w: str = "100%") -> str:
    """Build an ``<img>`` tag from parsed image info.

    MARP strips ``display``, ``margin``, and ``<div>`` wrappers from inline
    HTML during rendering — only ``max-height``, ``max-width``, and
    ``object-fit`` survive on ``<img>`` elements.  Centering is handled by
    the theme CSS rule ``section > img { display:block; margin:0 auto }``.
    """
    return (
        f'<img src="{info["path"]}" alt="{info["alt"]}" '
        f'style="max-height:{info["h"]};max-width:{max_w};object-fit:contain" />'
    )


# Invisible outer table cell style (no borders, no background)
_TD_BASE = "border:none;padding:0;vertical-align:top;background:transparent"
_TR_BASE = "background:transparent !important;border:none !important"
_TABLE_BASE = (
    "display:table !important;"
    "border:none;border-collapse:collapse;width:100%;margin:0;background:transparent"
)


def _expand_side_by_side(body: str, *, image_left: bool = True) -> str:
    """Expand ``[I,T]`` or ``[T,I]`` layout into inline-style HTML table.

    Parses the first ``<!-- I: ... -->`` directive from *body*, uses it as the
    image cell, and places remaining content in the text cell.
    """
    img_info = _parse_img_directive(body, default_h=_DEFAULT_H_SIDE)
    if not img_info:
        return body  # no image directive found — pass through unchanged

    # Remove the image directive from body to get the text content
    text_content = body.replace(img_info["full_match"], "").strip()

    img_w = img_info["w"] or _LAYOUT_IMG_WIDTH
    txt_w = _LAYOUT_CONTENT_WIDTH if not img_info["w"] else f"calc(100% - {img_info['w']})"
    gap_pad = f"padding:0 0 0 {_LAYOUT_GAP}" if image_left else f"padding:0 {_LAYOUT_GAP} 0 0"

    img_cell = (
        f'<td style="width:{img_w};{_TD_BASE};text-align:center">'
        f'{_img_tag(img_info)}'
        f'</td>'
    )
    txt_cell = (
        f'<td style="width:{txt_w};{_TD_BASE};{gap_pad}">\n\n'
        f'{text_content}\n\n'
        f'</td>'
    )

    if image_left:
        cells = img_cell + "\n" + txt_cell
    else:
        cells = txt_cell + "\n" + img_cell

    return (
        f'<table style="{_TABLE_BASE}">'
        f'<tr style="{_TR_BASE}">\n'
        f'{cells}\n'
        f'</tr></table>'
    )


def _expand_vertical_stack(body: str) -> str:
    """Expand ``[I;T]`` layout — image on top, text below."""
    img_info = _parse_img_directive(body, default_h=_DEFAULT_H_TOP)
    if not img_info:
        return body

    text_content = body.replace(img_info["full_match"], "").strip()

    # No <div> wrapper needed — MARP strips styled divs during rendering.
    # The <img> self-centers via display:block;margin:0 auto in its style.
    img_html = _img_tag(img_info)

    return f'{img_html}\n\n{text_content}'


def _expand_grid_2x2(body: str) -> str:
    """Expand ``[I,I;t,t]`` layout — 2 images top, 2 text blocks bottom.

    Expects exactly 2 image directives.  Text content is split at the first
    blank line after the second image directive.
    """
    matches = list(_IMG_DIRECTIVE_RE.finditer(body))
    if len(matches) < 2:
        return body

    img1 = _parse_img_directive(matches[0].group(0), default_h="240px")
    img2 = _parse_img_directive(matches[1].group(0), default_h="240px")
    if not img1 or not img2:
        return body

    # Remove both image directives to get remaining text
    remaining = body
    for m in reversed(matches[:2]):
        remaining = remaining[:m.start()] + remaining[m.end():]
    remaining = remaining.strip()

    # Split remaining text in half at blank line
    parts = re.split(r"\n\n", remaining, maxsplit=1)
    text1 = parts[0].strip() if len(parts) > 0 else ""
    text2 = parts[1].strip() if len(parts) > 1 else ""

    w = "50%"
    return (
        f'<table style="{_TABLE_BASE}">\n'
        f'<tr style="{_TR_BASE}">\n'
        f'<td style="width:{w};{_TD_BASE};text-align:center">{_img_tag(img1)}</td>\n'
        f'<td style="width:{w};{_TD_BASE};text-align:center">{_img_tag(img2)}</td>\n'
        f'</tr>\n'
        f'<tr style="{_TR_BASE}">\n'
        f'<td style="width:{w};{_TD_BASE}">\n\n{text1}\n\n</td>\n'
        f'<td style="width:{w};{_TD_BASE}">\n\n{text2}\n\n</td>\n'
        f'</tr>\n'
        f'</table>'
    )


# Map layout names to expander functions
_LAYOUT_EXPANDERS = {
    "[I,T]": lambda b: _expand_side_by_side(b, image_left=True),
    "[T,I]": lambda b: _expand_side_by_side(b, image_left=False),
    "[I;T]": _expand_vertical_stack,
    "[I,I;t,t]": _expand_grid_2x2,
}


def expand_layout_directives(content: str) -> str:
    """Expand layout shorthand comments into inline-style HTML.

    Supported layouts::

        <!-- layout: [I,T] -->  — image left, text right
        <!-- layout: [T,I] -->  — text left, image right
        <!-- layout: [I;T] -->  — image top, text below
        <!-- layout: [I,I;t,t] --> — 2x2 grid

    Image directives inside the block::

        <!-- I: assets/arch05.svg | alt: Architecture | h: 400px -->

    The block is closed with ``<!-- /layout -->``.

    This transform is idempotent: already-expanded HTML is not re-expanded
    because the directive comments are consumed during expansion.
    """
    def _replace(m: re.Match) -> str:
        layout_type = m.group(1).strip()
        body = m.group(2)
        expander = _LAYOUT_EXPANDERS.get(layout_type)
        if expander is None:
            return m.group(0)  # unknown layout — leave unchanged
        return expander(body)

    return _LAYOUT_BLOCK_RE.sub(_replace, content)


def compact_layout_slides(content: str) -> str:
    """Reduce vertical padding and expand images on side-by-side layout slides.

    MARP's default ``section`` padding-top is 50px and ``h2`` has ~18px of
    top margin.  For layout slides these consume ~85px above the first
    content line, which is wasted space — the layout table needs every
    available pixel.

    This transform detects slides containing the ``display:table !important``
    layout-table signature (injected by ``expand_layout_directives``) and:

    1. Injects a scoped style that reduces section padding and heading margins
    2. Increases ``max-height`` on images inside layout ``<td>`` cells to
       fill the available vertical space (~500px after compact padding)

    MARP 16:9 canvas = 720px.  After compact padding (20px top + 60px bottom),
    heading (~35px), footer/progress (~30px) → ~575px for layout table.
    """
    # Target max-height for images in layout tables (side-by-side)
    _LAYOUT_IMG_MAX_H = 550

    frontmatter, slides = _parse_marp(content)
    result: List[str] = []

    for slide in slides:
        # Only compact slides with side-by-side layout tables
        if "display:table !important" not in slide:
            result.append(slide)
            continue

        # Skip if already compacted (idempotency)
        if "padding-top:" in slide and "20px" in slide:
            result.append(slide)
            continue

        style = (
            "<style scoped>\n"
            "section { padding-top: 20px; }\n"
            "section h2 { margin-top: 0; margin-bottom: 0.3em; }\n"
            "</style>"
        )

        # Increase max-height on images inside layout <td> cells.
        # The inline style has absolute specificity, so we must modify
        # the tag directly (CSS cannot override inline styles).
        def _boost_layout_img(m: "re.Match[str]") -> str:
            tag = m.group(0)
            # Only boost images inside layout table cells (<td>...<img)
            hm = _IMG_MAX_HEIGHT_RE.search(tag)
            if not hm:
                return tag
            old_h = int(hm.group(1))
            if old_h >= _LAYOUT_IMG_MAX_H:
                return tag  # already large enough
            return tag[:hm.start(1)] + str(_LAYOUT_IMG_MAX_H) + tag[hm.end(1):]

        # Only modify <img> tags inside <td> (layout cell), not standalone
        slide = re.sub(
            r"(<td[^>]*>)\s*(<img\s[^>]*>)",
            lambda m: m.group(1) + _boost_layout_img(re.match(r"<img\s[^>]*>", m.group(2)) and m.group(2) and re.search(r"<img\s[^>]*>", m.group(2)) or m).replace(m.group(2), _boost_layout_img(type("M", (), {"group": lambda self, n=0: m.group(2)})())),
            slide,
        ) if False else slide  # placeholder — use simpler approach below

        # Simpler: find <td...><img...max-height:NNNpx...> and boost
        def _boost_td_img(m: "re.Match[str]") -> str:
            td_open = m.group(1)
            img_tag = m.group(2)
            hm = _IMG_MAX_HEIGHT_RE.search(img_tag)
            if not hm:
                return m.group(0)
            old_h = int(hm.group(1))
            if old_h >= _LAYOUT_IMG_MAX_H:
                return m.group(0)
            new_img = img_tag[:hm.start(1)] + str(_LAYOUT_IMG_MAX_H) + img_tag[hm.end(1):]
            return td_open + new_img

        slide = re.sub(
            r"(<td[^>]*>)\s*(<img\s[^>]*>)",
            _boost_td_img,
            slide,
        )

        slide = style + "\n\n" + slide
        result.append(slide)

    return _join_marp(frontmatter, result)


# ---------------------------------------------------------------------------
# Combined post-processor
# ---------------------------------------------------------------------------

def postprocess_marp(
    content: str,
    *,
    rewrite_title: bool = True,
    strip_numbers: bool = True,
    normalize_dividers: bool = True,
    clean_toc: bool = True,
    remove_sommaire: bool = True,
    progress_bar: bool = True,
    chapter_nav: bool = True,
    chapter_footer: bool = True,
    traceability: bool = True,
    auto_layout: bool = True,
    base_dir: Optional[str] = None,
    logos_dir: Optional[str] = None,
) -> str:
    """Apply all MARP post-processing transforms.

    Args:
        content: Raw MARP markdown (with frontmatter).
        rewrite_title: Replace ``Chapitre 0`` title with actual document title.
        strip_numbers: Remove N.M.K prefixes from slide headings.
        normalize_dividers: Convert ``# N. Title`` → ``# Chapitre N — Title``.
        clean_toc: Fix TOC slide (rename heading, remove non-chapter entries).
        remove_sommaire: Remove redundant trailing Sommaire slides.
        progress_bar: Inject visual progress bar at bottom.
        chapter_nav: Inject chapter navigation strip at top-right.
        chapter_footer: Inject per-slide footer with current chapter title.
        traceability: Append a provenance/traceability slide at the end.
        auto_layout: Auto-detect image layouts from dimensions (tall→[I,T],
            wide→[I;T]).  Runs before ``expand_layout_directives``.
        base_dir: Directory for resolving relative image paths (for
            ``auto_layout``).  Defaults to the current working directory.
        logos_dir: Path to directory with logo images (injected on title
            and last content slides). Logos are sorted alphabetically.

    Returns:
        Processed MARP content.
    """
    # Track which transforms are applied for the traceability slide
    applied = []

    # -1. Idempotency guard — strip previously injected artifacts (v1.5)
    content = strip_postprocess_artifacts(content)
    applied.append("strip_artifacts")
    # 0. Rewrite title
    if rewrite_title:
        content = rewrite_title_slide(content)
        applied.append("rewrite_title")
    # 1. Strip heading numbers
    if strip_numbers:
        content = strip_heading_numbers(content)
        applied.append("strip_numbers")
    # 1a. Strip pipe artifacts from headings (v1.5)
    content = strip_heading_pipes(content)
    applied.append("strip_pipes")
    # 1b. Normalize section dividers
    if normalize_dividers:
        content = normalize_section_dividers(content)
        content = renumber_chapters(content)
        applied.append("normalize_dividers")
    # 1b2. Strip source comments and caption numbering
    content = strip_source_comments(content)
    # 1b3. Strip Navigation rapide blockquotes (v1.4)
    content = strip_navigation_rapide(content)
    applied.append("strip_comments")
    # 1b4. Fix singleton numbered lists (1. → bullet)
    content = fix_singleton_numbered_lists(content)
    applied.append("fix_singletons")
    # 1d. Remove trailing Sommaire
    if remove_sommaire:
        content = remove_trailing_sommaire(content)
        applied.append("remove_sommaire")
    # 1d2. Remove garbled Sommaire block (v1.5)
    content = remove_garbled_sommaire(content)
    applied.append("remove_garbled_sommaire")
    # 1f. Remove empty chapter dividers BEFORE TOC rebuild (v1.5)
    content = remove_empty_chapter_dividers(content)
    applied.append("remove_empty_chapters")
    # 1e. Clean TOC slide (AFTER chapter removal so TOC reflects final chapters)
    if clean_toc:
        content = clean_toc_slide(content)
        applied.append("clean_toc")
    # 1f2. Auto-detect image layouts from dimensions (v2.0)
    if auto_layout:
        content = layout_preprocess(content, base_dir=base_dir)
        applied.append("auto_layout")
    # 1g. Expand layout directives (before table classification)
    content = expand_layout_directives(content)
    applied.append("expand_layouts")
    # 1g2. Compact vertical padding on layout slides (v2.0)
    content = compact_layout_slides(content)
    applied.append("compact_layouts")
    # 1c. Auto-classify tables (deterministic, no flag needed)
    content = auto_classify_tables(content)
    applied.append("auto_tables")
    # Auto-constrain figures on slides with body text (v1.3)
    content = auto_constrain_figure_slides(content)
    applied.append("auto_constrain_figures")
    # Auto-shrink dense slides (before nav/footer injection to measure clean text)
    content = auto_shrink_dense_slides(content)
    applied.append("auto_shrink")
    # 2. Progress bar
    if progress_bar:
        content = inject_progress_bar(content)
        applied.append("progress_bar")
    # 3. Chapter navigation
    if chapter_nav:
        content = inject_chapter_nav(content)
        applied.append("chapter_nav")
    # 4. Chapter footer
    if chapter_footer:
        content = inject_chapter_footer(content)
        applied.append("chapter_footer")
    # 5. Traceability slide
    if traceability:
        content = inject_traceability_slide(content, transforms_applied=applied)
    # 7. Logos
    if logos_dir:
        content = inject_logos(content, logos_dir)
        applied.append("logos")
    return content


# ---------------------------------------------------------------------------
# File-based API
# ---------------------------------------------------------------------------

def postprocess_file(
    path: str,
    *,
    dry_run: bool = False,
    rewrite_title: bool = True,
    strip_numbers: bool = True,
    normalize_dividers: bool = True,
    clean_toc: bool = True,
    remove_sommaire: bool = True,
    progress_bar: bool = True,
    chapter_nav: bool = True,
    chapter_footer: bool = True,
) -> str:
    """Post-process a MARP file in place.

    Args:
        path: Path to MARP ``.md`` file.
        dry_run: If True, compute result but don't overwrite.
        rewrite_title: Replace Chapitre 0 with actual document title.
        strip_numbers: Remove N.M.K prefixes from slide headings.
        normalize_dividers: Normalize section divider format.
        clean_toc: Fix TOC slide (French heading, remove non-chapter entries).
        remove_sommaire: Remove redundant trailing Sommaire slides.
        progress_bar: Inject progress bar.
        chapter_nav: Inject chapter navigation.
        chapter_footer: Inject per-slide chapter footer.

    Returns:
        Human-readable summary of changes applied.
    """
    from pathlib import Path as P

    p = P(path)
    content = p.read_text(encoding="utf-8")
    original = content

    new_content = postprocess_marp(
        content,
        rewrite_title=rewrite_title,
        strip_numbers=strip_numbers,
        normalize_dividers=normalize_dividers,
        clean_toc=clean_toc,
        remove_sommaire=remove_sommaire,
        progress_bar=progress_bar,
        chapter_nav=chapter_nav,
        chapter_footer=chapter_footer,
    )

    # Compute summary
    orig_lines = original.split("\n")
    new_lines = new_content.split("\n")
    changed_lines = sum(
        1 for a, b in zip(orig_lines, new_lines) if a != b
    )
    added_lines = abs(len(new_lines) - len(orig_lines))

    _, orig_slides = _parse_marp(original)
    _, new_slides = _parse_marp(new_content)
    slides_removed = len(orig_slides) - len(new_slides)

    summary_parts = [
        f"MARP Post-Processor Summary",
        f"  File: {p.name}",
        f"  Slides: {len(orig_slides)}"
        + (f" → {len(new_slides)} ({slides_removed} removed)" if slides_removed else ""),
    ]
    if strip_numbers:
        stripped = len(_HEADING_NUM_RE.findall(original))
        summary_parts.append(f"  Heading numbers stripped: {stripped}")
    if normalize_dividers:
        normalized = len(_BARE_NUM_HEADING_RE.findall(original))
        summary_parts.append(f"  Section dividers normalized: {normalized}")
    if clean_toc:
        summary_parts.append(f"  TOC cleaned: heading → Sommaire, non-chapter entries removed")
    if remove_sommaire and slides_removed > 0:
        summary_parts.append(f"  Trailing Sommaire: {slides_removed} slides removed")
    if progress_bar:
        summary_parts.append(f"  Progress bar: injected on {len(new_slides)} slides")
    if chapter_nav:
        chapters = _detect_chapters(_parse_marp(new_content)[1])
        summary_parts.append(
            f"  Chapter nav: {len(chapters)} chapters, "
            f"injected on {len(new_slides)} slides"
        )
    if chapter_footer:
        summary_parts.append(f"  Chapter footer: injected on {len(new_slides)} slides")
    summary_parts.append(f"  Lines: {len(orig_lines)} → {len(new_lines)} (+{added_lines})")

    if not dry_run:
        p.write_text(new_content, encoding="utf-8")
        summary_parts.append("  Status: WRITTEN")
    else:
        summary_parts.append("  Status: DRY RUN (not written)")

    return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# SVG title cleaner — strip "Fig. N — " prefixes from matplotlib chart titles
# ---------------------------------------------------------------------------

# Matches matplotlib title comments: <!-- Fig. 11 — Distribution ... -->
_SVG_TITLE_COMMENT_RE = re.compile(
    r"<!--\s*(?:Fig\.?\s*|Figure\s+|Diagram(?:me)?\s+|Sch[eé]ma\s+|Scheme\s+|Table(?:au)?\s+)"
    r"(\d+)\s*[—–:\-\.]\s*(.+?)\s*-->",
)

# Matches <use> elements with optional transform
_SVG_USE_RE = re.compile(
    r'<use\s+xlink:href="[^"]+?"'
    r'(?:\s+transform="translate\(([0-9.]+)\s+[0-9.]+\)")?'
    r"\s*/?>",
)

# ---------------------------------------------------------------------------
# HTML post-processing (runs AFTER marp-cli on the final .html output)
# ---------------------------------------------------------------------------

# Match <img> tags that have object-fit:contain in their style (layout images).
# Emoji and icon images don't have object-fit and are left untouched.
_IMG_OBJFIT_RE = re.compile(
    r'(<img\b[^>]*\sstyle=")([^"]*object-fit\s*:\s*contain[^"]*)"'
)


def center_images_in_html(html_path: str) -> int:
    """Add ``display:block;margin:0 auto`` to layout images in rendered HTML.

    MARP's CSS sanitizer strips ``display``, ``margin``, and ``text-align``
    properties from **all** image-related CSS rules (theme CSS, ``<style>``
    blocks, inline ``style`` attributes on ``<img>``), even with
    ``!important``.  The only reliable way to center images is to modify
    the HTML **after** marp-cli has finished rendering.

    This function targets only ``<img>`` elements whose ``style`` contains
    ``object-fit:contain`` — i.e., layout-generated images.  Small inline
    images (emoji, icons) are not affected.

    Args:
        html_path: Path to the marp-cli output HTML file (modified in place).

    Returns:
        Number of images patched.
    """
    path = Path(html_path)
    html = path.read_text()
    count = 0

    def _patch(m: re.Match) -> str:
        nonlocal count
        prefix = m.group(1)  # '<img ... style="'
        style = m.group(2)   # existing style value (without closing quote)
        # Avoid double-patching
        if "display:block" in style:
            return m.group(0)
        count += 1
        return f'{prefix}display:block;margin:0 auto;{style}"'

    html = _IMG_OBJFIT_RE.sub(_patch, html)

    if count > 0:
        path.write_text(html)
    return count


# Signature of a layout table's opening tag (from _TABLE_BASE constant).
_LAYOUT_TABLE_SIG = "border:none;border-collapse:collapse;width:100%"

# Match any <table...> tag (captures attributes).
_ANY_TABLE_RE = re.compile(r'<table\b([^>]*)>')


def _find_matching_close(html: str, open_pos: int, tag: str) -> int:
    """Find the closing tag matching the opener at *open_pos* (nesting-aware).

    Scans forward from *open_pos* counting ``<tag`` opens and ``</tag>``
    closes.  Returns the index of the ``>`` in the matching ``</tag>``, or
    -1 if not found.
    """
    depth = 0
    i = open_pos
    open_re = re.compile(rf'<{tag}\b', re.IGNORECASE)
    close_re = re.compile(rf'</{tag}\s*>', re.IGNORECASE)
    while i < len(html):
        om = open_re.search(html, i)
        cm = close_re.search(html, i)
        if cm is None:
            return -1
        if om is not None and om.start() < cm.start():
            depth += 1
            i = om.end()
        else:
            depth -= 1
            if depth == 0:
                return cm.end()
            i = cm.end()
    return -1


def fix_layout_tables_in_html(html_path: str) -> int:
    """Fix layout tables and their content tables in rendered HTML.

    MARP sets ``display: block`` on **all** ``<table>`` elements via its
    default theme.  This breaks the CSS table layout model: the ``<tr>``
    inside a ``display:block`` table creates an anonymous table box that
    shrinks to content instead of spanning the full block width.  As a
    result, ``<td width="38%">`` + ``<td width="62%">`` fills only ~77%
    of the outer table.

    This function:

    1. Injects ``display:table`` on **layout tables** (identified by
       ``border:none;border-collapse:collapse;width:100%``) so the ``<tr>``
       fills the full slide width.
    2. Injects ``width:100%`` on **content tables** inside layout cells so
       they span the full cell width.

    Args:
        html_path: Path to the marp-cli output HTML file (modified in place).

    Returns:
        Number of tables patched (layout + content).
    """
    path = Path(html_path)
    html = path.read_text()
    total = 0
    offset = 0

    while True:
        # Find next layout table
        sig_pos = html.find(_LAYOUT_TABLE_SIG, offset)
        if sig_pos == -1:
            break
        # Walk back to the <table that contains this signature
        table_open = html.rfind("<table", 0, sig_pos)
        if table_open == -1:
            offset = sig_pos + 1
            continue
        # Find the matching </table>
        table_close = _find_matching_close(html, table_open, "table")
        if table_close == -1:
            offset = sig_pos + 1
            continue

        # Extract the layout table block
        block = html[table_open:table_close]

        # Patch all <table> tags inside this block
        def _patch(m: re.Match) -> str:
            nonlocal total
            attrs = m.group(1)
            if _LAYOUT_TABLE_SIG in attrs:
                # Layout table: inject display:table to override MARP's
                # display:block (which breaks <tr> width inheritance).
                if "display:table" in attrs:
                    return m.group(0)
                patched = attrs.replace('style="', 'style="display:table;')
                total += 1
                return f'<table{patched}>'
            else:
                # Content table: inject width:100% to fill the cell.
                if "width:100%" in attrs:
                    return m.group(0)
                if 'style="' in attrs:
                    patched = attrs.replace('style="', 'style="width:100%;')
                else:
                    patched = f' style="width:100%"' + attrs
                total += 1
                return f'<table{patched}>'

        new_block = _ANY_TABLE_RE.sub(_patch, block)
        html = html[:table_open] + new_block + html[table_close:]
        offset = table_open + len(new_block)

    if total > 0:
        path.write_text(html)
    return total


# ---------------------------------------------------------------------------
# Image embedding (self-contained HTML)
# ---------------------------------------------------------------------------

# Match <img src="relative/path"> (not http://, data:, or /absolute).
_LOCAL_IMG_RE = re.compile(
    r'(<img\b[^>]*\bsrc=")([^"]+)(")'
)

_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}


def embed_images_in_html(html_path: str, *, max_dim: int = 2000,
                         jpeg_quality: int = 85) -> dict:
    """Replace local ``<img src>`` paths with base64 data URIs.

    Produces a **self-contained HTML** file that can be transferred without
    its ``assets/`` directory.

    Raster images (PNG, JPG, WebP) larger than *max_dim* pixels on any side
    are downscaled (Lanczos) to fit, then re-encoded.  PNGs with transparency
    stay PNG; opaque images are converted to JPEG at *jpeg_quality* for smaller
    size.  SVGs are embedded as-is (vector — resolution-independent).

    The *max_dim* default of 2000 px gives ~200 DPI for a 10″-wide printed
    slide (MARP default: 1280×720 viewport ≈ 10″ × 5.6″).

    Args:
        html_path: Path to the HTML file (modified in place).
        max_dim:   Max pixel dimension (width or height) for raster images.
        jpeg_quality: JPEG compression quality (1–95).

    Returns:
        Dict with ``embedded`` (count), ``skipped`` (count),
        ``original_kb`` and ``embedded_kb`` (file sizes).
    """
    import base64
    from io import BytesIO

    path = Path(html_path)
    html_dir = path.parent
    html = path.read_text()
    original_size = path.stat().st_size

    embedded = 0
    skipped = 0

    def _embed(m: re.Match) -> str:
        nonlocal embedded, skipped
        prefix = m.group(1)   # '<img ... src="'
        src = m.group(2)      # the path
        suffix = m.group(3)   # '"'

        # Skip non-local sources
        if src.startswith(("http://", "https://", "data:", "/")):
            skipped += 1
            return m.group(0)

        img_path = html_dir / src
        if not img_path.is_file():
            skipped += 1
            return m.group(0)

        ext = img_path.suffix.lower()
        mime = _MIME_MAP.get(ext)
        if mime is None:
            skipped += 1
            return m.group(0)

        if ext == ".svg":
            # SVG — embed as-is (vector, no resize)
            svg_bytes = img_path.read_bytes()
            b64 = base64.b64encode(svg_bytes).decode("ascii")
        else:
            # Raster — resize if needed, then encode
            try:
                from PIL import Image
            except ImportError:
                skipped += 1
                return m.group(0)

            img = Image.open(img_path)
            w, h = img.size

            # Downscale if exceeding max_dim
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            buf = BytesIO()
            if img.mode == "RGBA" or ext == ".png":
                # Keep PNG for transparency
                img.save(buf, format="PNG", optimize=True)
                mime = "image/png"
            else:
                # Convert to JPEG for smaller size
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(buf, format="JPEG", quality=jpeg_quality)
                mime = "image/jpeg"

            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        embedded += 1
        return f'{prefix}data:{mime};base64,{b64}{suffix}'

    html = _LOCAL_IMG_RE.sub(_embed, html)
    path.write_text(html)
    final_size = path.stat().st_size

    return {
        "embedded": embedded,
        "skipped": skipped,
        "original_kb": original_size // 1024,
        "embedded_kb": final_size // 1024,
    }


# ---------------------------------------------------------------------------
# SVG asset transforms
# ---------------------------------------------------------------------------

# Matches the parent <g transform="translate(X Y) scale(S -S)">
_SVG_TITLE_G_RE = re.compile(
    r'(<g\s+transform="translate\()([0-9.]+)(\s+[0-9.]+\)\s+scale\()([0-9.]+)(\s+-?[0-9.]+\)">)',
)


def clean_svg_title(svg_content: str) -> Optional[str]:
    """Strip 'Fig. N — ' prefix from a matplotlib SVG chart title.

    Removes the prefix glyph ``<use>`` elements and re-centers the remaining
    title text by adjusting the parent ``<g transform="translate(...)">`` offset.

    Args:
        svg_content: Raw SVG text.

    Returns:
        Modified SVG text with prefix stripped, or *None* if no title comment
        matched (i.e. the SVG has no figure-numbered title).
    """
    m_comment = _SVG_TITLE_COMMENT_RE.search(svg_content)
    if not m_comment:
        return None

    # Build the prefix string: "Fig. NN — " (what we want to remove)
    prefix_in_comment = svg_content[m_comment.start():m_comment.end()]
    # Count characters in prefix (before actual title)
    full_comment_text = m_comment.group(0)
    # Extract just "Fig. 11 — " from the comment text
    title_text = m_comment.group(2)  # the actual title after prefix
    # Everything before title_text in the comment content is the prefix
    inner_start = full_comment_text.index("<!--") + 4
    inner_end = full_comment_text.index("-->")
    inner = full_comment_text[inner_start:inner_end].strip()
    # prefix_chars = everything in inner before title_text
    title_pos = inner.find(title_text)
    if title_pos < 0:
        return None
    prefix_str = inner[:title_pos]
    n_prefix_chars = len(prefix_str)

    if n_prefix_chars == 0:
        return None

    # Find the <g id="text_*"> that contains this comment
    # We look for the comment position and work within its parent group
    comment_pos = m_comment.start()

    # Find the <g transform="translate(X Y) scale(S -S)"> after the comment
    m_parent_g = _SVG_TITLE_G_RE.search(svg_content, comment_pos)
    if not m_parent_g:
        return None

    # Find </defs> after the comment — <use> elements follow it
    defs_end = svg_content.find("</defs>", m_parent_g.start())
    if defs_end < 0:
        return None
    defs_end += len("</defs>")

    # Find closing </g> for the transform group
    close_g = svg_content.find("</g>", defs_end)
    if close_g < 0:
        return None

    # Extract the <use> block
    use_block = svg_content[defs_end:close_g]
    uses = list(_SVG_USE_RE.finditer(use_block))

    if len(uses) <= n_prefix_chars:
        return None  # not enough glyphs — something is wrong

    # The first n_prefix_chars <use> elements are the prefix
    # Get x-offset of the first character AFTER the prefix (the title start)
    first_title_use = uses[n_prefix_chars]
    shift_x = float(first_title_use.group(1)) if first_title_use.group(1) else 0.0

    # Build new <use> block with prefix removed and x-offsets shifted
    new_uses = []
    for use_match in uses[n_prefix_chars:]:
        original = use_match.group(0)
        if use_match.group(1) is not None:
            old_x = float(use_match.group(1))
            new_x = old_x - shift_x
            # Replace the x coordinate in the transform
            original = original.replace(
                f"translate({use_match.group(1)} ",
                f"translate({new_x:.6f} ",
            )
        new_uses.append(original)

    new_use_block = "\n     ".join(new_uses)

    # Adjust parent <g> translate to re-center
    old_translate_x = float(m_parent_g.group(2))
    scale = float(m_parent_g.group(4))
    # Shift right by half the removed prefix width (in pixel coords)
    new_translate_x = old_translate_x + (shift_x * scale) / 2.0

    # Rebuild SVG
    result = svg_content[:comment_pos]
    # Update comment
    result += f"<!-- {title_text} -->"
    # Skip to parent <g> and update its translate
    result += svg_content[m_comment.end():m_parent_g.start()]
    result += (
        m_parent_g.group(1)
        + f"{new_translate_x:.6f}"
        + m_parent_g.group(3)
        + m_parent_g.group(4)
        + m_parent_g.group(5)
    )
    # Keep <defs> block intact
    result += svg_content[m_parent_g.end():defs_end]
    # Insert new <use> block
    result += "\n     " + new_use_block + "\n    "
    # Skip old <use> block, continue from </g>
    result += svg_content[close_g:]

    return result


def clean_svg_titles_in_dir(
    assets_dir: str,
    *,
    dry_run: bool = False,
) -> str:
    """Strip 'Fig. N — ' prefixes from all SVG chart titles in a directory.

    Args:
        assets_dir: Path to the assets directory containing SVG files.
        dry_run: If True, report changes without writing.

    Returns:
        Human-readable summary of changes.
    """
    from pathlib import Path as P

    d = P(assets_dir)
    if not d.is_dir():
        return f"ERROR: {assets_dir} is not a directory"

    svgs = sorted(d.glob("*.svg"))
    cleaned = 0
    skipped = 0
    errors = []

    for svg_path in svgs:
        try:
            content = svg_path.read_text(encoding="utf-8")
            result = clean_svg_title(content)
            if result is None:
                skipped += 1
                continue
            if not dry_run:
                svg_path.write_text(result, encoding="utf-8")
            cleaned += 1
        except Exception as exc:
            errors.append(f"  {svg_path.name}: {exc}")

    parts = [
        f"SVG Title Cleaner Summary",
        f"  Directory: {d}",
        f"  Total SVGs: {len(svgs)}",
        f"  Cleaned: {cleaned}",
        f"  Skipped (no prefix): {skipped}",
    ]
    if errors:
        parts.append(f"  Errors: {len(errors)}")
        parts.extend(errors)
    if dry_run:
        parts.append("  Status: DRY RUN")
    else:
        parts.append("  Status: WRITTEN")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Mermaid SVG contrast booster — increase visibility for slide projection
# ---------------------------------------------------------------------------

# CSS substitution rules: (pattern, replacement) applied to <style> block
_MERMAID_CSS_RULES: List[Tuple[str, str]] = [
    # Node shapes: thicker stroke, darker fill & border
    ("fill:#ECECFF", "fill:#D8D8FF"),
    ("stroke:#9370DB", "stroke:#6A3DB5"),
    # Edge paths: thicker, darker
    ("stroke:#333333", "stroke:#111111"),
    ("stroke-width:2.0px", "stroke-width:3px"),
    # Markers / arrows: darker
    ("fill:#333333", "fill:#111111"),
    # Text: near-black for readability at distance
    ("fill:#333;", "fill:#111;"),
    ("fill:#333}", "fill:#111}"),
    ("color:#333;", "color:#111;"),
    ("color:#333}", "color:#111}"),
    # Cluster borders: thicker, darker (also catches tooltip border)
    ("#aaaa33", "#777722"),
    # Node stroke-width (1px → 2px) — must be specific to node shapes
    # Applied via regex below to avoid hitting edge-thickness-normal
]

# Inline style substitutions for cluster rect pastel fills
_MERMAID_CLUSTER_FILLS: List[Tuple[str, str]] = [
    ("fill:#ffffde", "fill:#fff9b0"),     # yellow → richer yellow
    ("fill:#e3f2fd", "fill:#bbdefb"),     # light blue → medium blue
    ("fill:#e8f5e9", "fill:#c8e6c9"),     # light green → medium green
    ("fill:#ffebee", "fill:#ffcdd2"),     # light red/pink → medium pink
]

# Regex: node-shape stroke-width in CSS (not edge-thickness classes)
_MERMAID_NODE_SW_RE = re.compile(
    r"(\.node\s+rect.*?stroke-width:)1px",
)


def _is_mermaid_svg(content: str) -> bool:
    """Detect if SVG content was generated by Mermaid."""
    return (
        'id="mermaid-svg"' in content
        or 'class="flowchart"' in content
        or "#mermaid-svg" in content
    )


def boost_mermaid_contrast(svg_content: str) -> Optional[str]:
    """Increase visual contrast of a Mermaid-generated SVG for slide projection.

    Modifications:
      - **Node shapes**: darker fill (#D8D8FF), stronger stroke (#6A3DB5, 2px)
      - **Edges**: thicker (3px), darker (#111111)
      - **Text**: near-black (#111) instead of dark-gray (#333)
      - **Cluster rects**: saturated pastel fills, thicker/darker borders
      - **Markers**: darker arrow fills

    Args:
        svg_content: Raw SVG text.

    Returns:
        Modified SVG with boosted contrast, or *None* if not a Mermaid SVG.
    """
    if not _is_mermaid_svg(svg_content):
        return None

    result = svg_content

    # 1. CSS substitutions in <style> block
    for old, new in _MERMAID_CSS_RULES:
        result = result.replace(old, new)

    # 2. Node-specific stroke-width: 1px → 2px
    #    Target the CSS rule for .node rect/circle/ellipse/polygon/path
    result = re.sub(
        r"(\.node\s+rect,.*?stroke-width:)1px",
        r"\g<1>2px",
        result,
    )

    # 3. Cluster stroke-width in CSS: 1px → 2px
    result = re.sub(
        r"(\.cluster\s+rect\{.*?stroke-width:)1px",
        r"\g<1>2px",
        result,
    )

    # 4. Inline style cluster rect fills (more saturated pastels)
    for old, new in _MERMAID_CLUSTER_FILLS:
        result = result.replace(old, new)

    # 5. Marker sizes: increase markerWidth/markerHeight for better arrow visibility
    result = re.sub(
        r'(class="marker flowchart-v2"[^>]*markerWidth=")8(")',
        r'\g<1>10\2',
        result,
    )
    result = re.sub(
        r'(class="marker flowchart-v2"[^>]*markerHeight=")8(")',
        r'\g<1>10\2',
        result,
    )

    return result


def boost_mermaid_contrast_in_dir(
    assets_dir: str,
    *,
    dry_run: bool = False,
) -> str:
    """Boost contrast of all Mermaid SVGs in a directory.

    Args:
        assets_dir: Path to the assets directory containing SVG files.
        dry_run: If True, report changes without writing.

    Returns:
        Human-readable summary of changes.
    """
    from pathlib import Path as P

    d = P(assets_dir)
    if not d.is_dir():
        return f"ERROR: {assets_dir} is not a directory"

    svgs = sorted(d.glob("*.svg"))
    boosted = 0
    skipped = 0
    errors = []

    for svg_path in svgs:
        try:
            content = svg_path.read_text(encoding="utf-8")
            result = boost_mermaid_contrast(content)
            if result is None:
                skipped += 1
                continue
            if not dry_run:
                svg_path.write_text(result, encoding="utf-8")
            boosted += 1
        except Exception as exc:
            errors.append(f"  {svg_path.name}: {exc}")

    parts = [
        f"Mermaid Contrast Booster Summary",
        f"  Directory: {d}",
        f"  Total SVGs: {len(svgs)}",
        f"  Boosted: {boosted}",
        f"  Skipped (not Mermaid): {skipped}",
    ]
    if errors:
        parts.append(f"  Errors: {len(errors)}")
        parts.extend(errors)
    if dry_run:
        parts.append("  Status: DRY RUN")
    else:
        parts.append("  Status: WRITTEN")

    return "\n".join(parts)
