"""
Markdown Table of Contents — expand ``[TOC]`` markers into linked ToC blocks.

Scans the document for headings, builds a linked table of contents using
GitHub-style anchor slugs, and replaces every ``[TOC]`` marker with the
generated block.  For long documents the ToC can be **paginated** (split
across MARP ``---`` page breaks) to fit slide decks.

Supports:
  - All ATX heading levels (``#`` through ``######``)
  - Numbered headings: ``## 3. Title`` and unnumbered: ``## Title``
  - GitHub-style slug generation (lowercase, spaces→hyphens, strip punctuation)
  - Configurable ``min_level`` / ``max_level`` to limit which headings appear
  - ``items_per_page`` pagination with ``---`` page separators
  - Preserves existing content around the ``[TOC]`` marker
  - Multiple ``[TOC]`` markers in a single document

Usage::

    from ragix_kernels.shared.md_toc import expand_toc, expand_toc_file

    new_content, report = expand_toc(content)
    print(report.summary())

    # With pagination (for MARP slide decks)
    new_content, report = expand_toc(content, items_per_page=12)

    # Limit to h2-h3 headings only
    new_content, report = expand_toc(content, min_level=2, max_level=3)

    # File-based (with optional dry_run)
    report = expand_toc_file("docs/GUIDE.md", items_per_page=15, dry_run=True)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-13
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TocHeading:
    """A heading found in the document."""
    level: int          # 1-6 (number of '#')
    title: str          # raw heading text (without leading ##)
    slug: str           # GitHub-style anchor slug
    line_num: int       # 1-based line number
    number: str = ""    # e.g. "3", "3.1" — empty if unnumbered


@dataclass
class TocReport:
    """Report of TOC expansion."""
    headings_found: int = 0
    headings_included: int = 0
    toc_markers_found: int = 0
    toc_markers_expanded: int = 0
    pages: int = 1
    items_per_page: int = 0
    min_level: int = 1
    max_level: int = 6

    def summary(self) -> str:
        lines = [
            "TOC Expansion Report",
            f"  Headings scanned: {self.headings_found}",
            f"  Headings included: {self.headings_included} "
            f"(levels {self.min_level}-{self.max_level})",
            f"  [TOC] markers found: {self.toc_markers_found}",
            f"  [TOC] markers expanded: {self.toc_markers_expanded}",
        ]
        if self.pages > 1:
            lines.append(
                f"  Pages: {self.pages} "
                f"({self.items_per_page} items/page)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heading regex
# ---------------------------------------------------------------------------

# ATX heading: ## Optional-number. Title
_HEADING_RE = re.compile(
    r'^(#{1,6})\s+(?:(\d+(?:\.\d+)*\.?)\s+)?(.+?)\s*$'
)

# [TOC] marker — alone on a line (possibly with surrounding whitespace)
_TOC_MARKER_RE = re.compile(r'^\s*\[TOC\]\s*$', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Slug generation (GitHub-compatible)
# ---------------------------------------------------------------------------

# Characters to strip from slugs (keep letters, digits, hyphens, spaces)
_SLUG_STRIP_RE = re.compile(r'[^\w\s-]', re.UNICODE)
_SLUG_SPACE_RE = re.compile(r'[\s]+')


def _make_slug(title: str) -> str:
    """
    Generate a GitHub-style anchor slug from a heading title.

    Rules (matching GitHub's algorithm):
      - Strip inline markdown formatting (bold, italic, code, links)
      - Lowercase
      - Replace spaces with hyphens
      - Remove punctuation except hyphens
      - Collapse consecutive hyphens
    """
    # Strip markdown inline formatting
    text = title
    # Remove inline code: `code`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove links: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove bold/italic: **text**, *text*, __text__, _text_
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)

    # Lowercase
    text = text.lower()
    # Strip non-word characters (keep Unicode letters, digits, spaces, hyphens)
    text = _SLUG_STRIP_RE.sub('', text)
    # Replace spaces with hyphens
    text = _SLUG_SPACE_RE.sub('-', text.strip())
    # Collapse consecutive hyphens
    text = re.sub(r'-{2,}', '-', text)
    # Strip leading/trailing hyphens
    text = text.strip('-')

    return text


# ---------------------------------------------------------------------------
# Slug deduplication
# ---------------------------------------------------------------------------

def _dedupe_slugs(headings: List[TocHeading]) -> None:
    """
    Deduplicate slugs in-place by appending ``-1``, ``-2``, etc.
    (matches GitHub behavior for duplicate headings).
    """
    seen: dict[str, int] = {}
    for h in headings:
        base = h.slug
        if base in seen:
            seen[base] += 1
            h.slug = f"{base}-{seen[base]}"
        else:
            seen[base] = 0


# ---------------------------------------------------------------------------
# Core: scan headings
# ---------------------------------------------------------------------------

def _scan_headings(
    lines: List[str],
    min_level: int = 1,
    max_level: int = 6,
) -> List[TocHeading]:
    """
    Scan all ATX headings in the document.

    Returns headings with levels in [min_level, max_level].
    Skips headings inside fenced code blocks.
    """
    headings: List[TocHeading] = []
    in_fence = False

    for i, line in enumerate(lines):
        # Track fenced code blocks
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        m = _HEADING_RE.match(line)
        if not m:
            continue

        level = len(m.group(1))
        number = (m.group(2) or "").rstrip(".")
        title = m.group(3).strip()

        if level < min_level or level > max_level:
            continue

        slug = _make_slug(title)

        headings.append(TocHeading(
            level=level,
            title=title,
            slug=slug,
            line_num=i + 1,
            number=number,
        ))

    _dedupe_slugs(headings)
    return headings


# ---------------------------------------------------------------------------
# Core: format ToC block
# ---------------------------------------------------------------------------

def _format_toc_entry(heading: TocHeading, base_level: int) -> str:
    """
    Format a single ToC entry as a Markdown list item with link.

    Indentation is computed relative to ``base_level`` (the minimum heading
    level in the ToC).  Each level adds 2 spaces of indentation.
    """
    indent = "  " * (heading.level - base_level)
    # Use number prefix if the heading was numbered
    if heading.number:
        return f"{indent}{heading.number}. [{heading.title}](#{heading.slug})"
    else:
        return f"{indent}- [{heading.title}](#{heading.slug})"


def _build_toc_block(
    headings: List[TocHeading],
    items_per_page: int = 0,
    page_separator: str = "\n---\n",
    toc_continuation_title: str = "## Table of Contents (continued)",
) -> str:
    """
    Build a full ToC block from headings.

    Parameters
    ----------
    headings : list
        Filtered headings to include.
    items_per_page : int
        Max items per page.  0 = no pagination (single block).
    page_separator : str
        String inserted between pages (default: MARP slide break).
    toc_continuation_title : str
        Heading inserted at the top of continuation pages.

    Returns
    -------
    str
        The formatted ToC block (may contain page separators).
    """
    if not headings:
        return ""

    base_level = min(h.level for h in headings)
    entries = [_format_toc_entry(h, base_level) for h in headings]

    if items_per_page <= 0 or len(entries) <= items_per_page:
        # Single block, no pagination
        return "\n".join(entries)

    # Paginate
    pages: List[str] = []
    for start in range(0, len(entries), items_per_page):
        chunk = entries[start : start + items_per_page]
        if pages:
            # Continuation pages get a title
            pages.append(
                toc_continuation_title + "\n\n" + "\n".join(chunk)
            )
        else:
            pages.append("\n".join(chunk))

    return page_separator.join(pages)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_toc(
    content: str,
    *,
    min_level: int = 1,
    max_level: int = 6,
    items_per_page: int = 0,
    page_separator: str = "\n---\n",
    toc_continuation_title: str = "## Table of Contents (continued)",
    exclude_before_toc: bool = False,
) -> Tuple[str, TocReport]:
    """
    Expand all ``[TOC]`` markers in *content* with a linked table of contents.

    Parameters
    ----------
    content : str
        Markdown document content.
    min_level : int
        Minimum heading level to include (default 1 = ``#``).
    max_level : int
        Maximum heading level to include (default 6 = ``######``).
    items_per_page : int
        Max ToC entries per page.  0 = no pagination.
    page_separator : str
        Separator between pages (default ``---`` for MARP slides).
    toc_continuation_title : str
        Title for continuation pages.
    exclude_before_toc : bool
        If True, exclude headings that appear before the first ``[TOC]``
        marker.  Useful for slide decks where the title slide heading
        should not appear in the ToC.

    Returns
    -------
    (new_content, report) : tuple
        The modified content and a TocReport with statistics.
    """
    lines = content.split("\n")
    report = TocReport(
        min_level=min_level,
        max_level=max_level,
        items_per_page=items_per_page,
    )

    # Find [TOC] markers first (needed to exclude ToC title headings)
    toc_marker_lines: List[int] = []
    for i, line in enumerate(lines):
        if _TOC_MARKER_RE.match(line):
            toc_marker_lines.append(i)
    report.toc_markers_found = len(toc_marker_lines)

    # Identify ToC title headings: any heading within 3 lines before a [TOC]
    # marker whose slug contains "table-of-contents" or "toc" or "sommaire"
    _TOC_TITLE_SLUGS = {"table-of-contents", "toc", "sommaire",
                        "table-des-matières", "table-des-matieres"}
    toc_title_lines: set[int] = set()
    for marker_idx in toc_marker_lines:
        for offset in range(1, 4):
            check_line = marker_idx - offset
            if check_line >= 0:
                hm = _HEADING_RE.match(lines[check_line])
                if hm:
                    slug = _make_slug(hm.group(3).strip())
                    if slug in _TOC_TITLE_SLUGS:
                        toc_title_lines.add(check_line + 1)  # 1-based
                    break  # stop at first heading found above marker

    # Scan all headings
    all_headings = _scan_headings(lines, min_level=1, max_level=6)
    report.headings_found = len(all_headings)

    # Determine the earliest line to include
    first_toc_line = toc_marker_lines[0] + 1 if toc_marker_lines else 0  # 0-based → 1-based

    # Filter to requested levels, excluding ToC title headings
    # and optionally headings before the first [TOC] marker
    toc_headings = [
        h for h in all_headings
        if min_level <= h.level <= max_level
        and h.line_num not in toc_title_lines
        and (not exclude_before_toc or h.line_num > first_toc_line)
    ]
    report.headings_included = len(toc_headings)

    if not toc_marker_lines or not toc_headings:
        return content, report

    # Build the ToC block
    toc_block = _build_toc_block(
        toc_headings,
        items_per_page=items_per_page,
        page_separator=page_separator,
        toc_continuation_title=toc_continuation_title,
    )

    # Compute page count for reporting
    if items_per_page > 0 and len(toc_headings) > items_per_page:
        report.pages = (
            (len(toc_headings) + items_per_page - 1) // items_per_page
        )

    # Replace [TOC] markers (process in reverse to preserve line numbers)
    for marker_idx in reversed(toc_marker_lines):
        lines[marker_idx] = toc_block
        report.toc_markers_expanded += 1

    return "\n".join(lines), report


def expand_toc_file(
    path: str,
    *,
    min_level: int = 1,
    max_level: int = 6,
    items_per_page: int = 0,
    page_separator: str = "\n---\n",
    toc_continuation_title: str = "## Table of Contents (continued)",
    exclude_before_toc: bool = False,
    dry_run: bool = False,
) -> TocReport:
    """
    Expand ``[TOC]`` markers in a file.

    Parameters
    ----------
    path : str
        Path to the Markdown file.
    min_level, max_level, items_per_page, page_separator, toc_continuation_title, exclude_before_toc
        See :func:`expand_toc`.
    dry_run : bool
        If True, compute the report but do not write changes.

    Returns
    -------
    TocReport
        Statistics about the expansion.
    """
    p = Path(path)
    content = p.read_text(encoding="utf-8")

    new_content, report = expand_toc(
        content,
        min_level=min_level,
        max_level=max_level,
        items_per_page=items_per_page,
        page_separator=page_separator,
        toc_continuation_title=toc_continuation_title,
        exclude_before_toc=exclude_before_toc,
    )

    if not dry_run and report.toc_markers_expanded > 0:
        p.write_text(new_content, encoding="utf-8")

    return report
