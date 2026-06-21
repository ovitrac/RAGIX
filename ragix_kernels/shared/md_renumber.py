"""
Markdown Auto-Renumbering — deterministic section/figure/table renumbering.

Three-phase pipeline:
  1. **Inventory** — scan and collect all numbered elements
  2. **Renumber** — compute correct sequential numbers based on document order
  3. **Apply** — substitute old numbers with new ones (elements + cross-references)

Supports:
  - Section headings: ``## N. Title`` (any heading level, dot-delimited hierarchy)
  - Table of Contents: ``N. [Title](#anchor)`` blocks
  - Figure references: ``Figure N``, ``Fig. N``, ``fig. N`` (definition + cross-ref)
  - HTML figure captions: ``<b>Fig. XX</b>``, ``<b>Figure XX</b>``
  - HTML image sources: ``src="assets/figXX_name.svg"``
  - Figure listing tables: markdown tables with ``| Fig. XX | description |``
  - Table references: ``Table N``, ``Tableau N``, ``Tab. N`` (definition + cross-ref)
  - Cross-references: ``Section N``, ``section N.M``, ``§N``, ``§N.M``
  - Number padding preservation (2-digit inputs produce 2-digit zero-padded outputs)
  - SVG file renaming (optional, via ``rename_figure_files()``)
  - Bilingual FR/EN patterns

Usage::

    from ragix_kernels.shared.md_renumber import renumber_markdown

    new_content, report = renumber_markdown(content)
    print(report.summary())

    # Selective scope
    new_content, report = renumber_markdown(content, scope=["sections", "toc"])

    # With figure table rebuild (reorder rows by new number)
    new_content, report = renumber_markdown(content, scope=["figures", "figure_table"])

    # Rename SVG files after renumbering
    from ragix_kernels.shared.md_renumber import rename_figure_files
    renames = rename_figure_files(figure_map, assets_dir, dry_run=False)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-13
Updated: 2026-03-04 — HTML figure patterns, figure table blocks, padding, SVG renaming
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NumberedElement:
    """A numbered element found in the document."""
    kind: str           # "section", "toc", "figure", "table", "xref", "fig_table"
    line_num: int       # 1-based line number
    old_number: str     # e.g. "3", "3.1", "12", "04"
    new_number: str = ""  # computed after renumbering
    col_start: int = 0  # column start of the number in the line
    col_end: int = 0    # column end of the number in the line
    context: str = ""   # original line text (for reporting)
    label: str = ""     # "Figure", "Table", "Tableau", "Section", "§", etc.

    @property
    def changed(self) -> bool:
        return self.new_number != "" and self.new_number != self.old_number


@dataclass
class RenumberReport:
    """Report of all renumbering operations."""
    elements: List[NumberedElement] = field(default_factory=list)
    changes: int = 0
    sections_found: int = 0
    toc_entries_found: int = 0
    figure_refs_found: int = 0
    table_refs_found: int = 0
    xrefs_found: int = 0
    fig_table_rows: int = 0
    figure_map: Dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Markdown Renumber Report",
            f"  Sections: {self.sections_found} found",
            f"  ToC entries: {self.toc_entries_found} found",
            f"  Figure refs: {self.figure_refs_found} found",
            f"  Table refs: {self.table_refs_found} found",
            f"  Cross-refs: {self.xrefs_found} found",
            f"  Figure table rows: {self.fig_table_rows} found",
            f"  Changes applied: {self.changes}",
        ]
        if self.figure_map:
            changed = {k: v for k, v in self.figure_map.items() if k != v}
            if changed:
                lines.append(f"  Figure mapping ({len(changed)} renumbered):")
                for old, new in sorted(changed.items()):
                    lines.append(f"    Fig {old} -> Fig {new}")
        if self.changes > 0:
            lines.append("  Details:")
            for el in self.elements:
                if el.changed:
                    lines.append(
                        f"    L{el.line_num} [{el.kind}] "
                        f"{el.old_number} -> {el.new_number}"
                    )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Section heading: ## N. Title  or  ## N.M. Title  or  ## N.M.K Title
_SECTION_RE = re.compile(
    r'^(#{1,6})\s+(\d+(?:\.\d+)*\.?)\s+(.+)$'
)

# ToC entry: N. [Title](#anchor) — with optional leading whitespace
_TOC_ENTRY_RE = re.compile(
    r'^(\s*)(\d+)\.\s+\[(.+?)\]\(#(.+?)\)\s*$'
)

# Figure definitions and references (bilingual, plain text)
_FIGURE_REF_RE = re.compile(
    r'(?i)\b(Figure|Fig\.|fig\.)\s+(\d+)\b'
)

# HTML figure captions: <b>Fig. XX</b> or <b>Figure XX</b>
_HTML_CAPTION_RE = re.compile(
    r'<b>(Fig\.|Figure)\s+(\d+)</b>'
)

# HTML img src: src="assets/figXX_whatever.svg"
_IMG_SRC_RE = re.compile(
    r'(src="assets/fig)(\d+)(_[^"]*\.svg")'
)

# Figure listing table row: | Fig. XX | description | §Y |
_FIG_TABLE_ROW_RE = re.compile(
    r'^\|\s*(Fig\.)\s+(\d+)\s*\|(.+)$'
)

# Table definitions and references (bilingual)
_TABLE_REF_RE = re.compile(
    r'(?i)\b(Table|Tableau|Tab\.)\s+(\d+)\b'
)

# Cross-references to sections
_SECTION_XREF_RE = re.compile(
    r'(?i)\b(Section|section|§)\s*(\d+(?:\.\d+)*)\b'
)


# ---------------------------------------------------------------------------
# Phase 1: Inventory
# ---------------------------------------------------------------------------

def _detect_toc_block(lines: List[str]) -> Tuple[int, int]:
    """
    Detect a ToC block: a contiguous sequence of lines matching _TOC_ENTRY_RE.

    Returns (start, end) 0-based line indices, or (-1, -1) if not found.
    A ToC must have >= 3 consecutive matching lines.
    """
    best_start = -1
    best_end = -1
    cur_start = -1
    cur_count = 0

    for i, line in enumerate(lines):
        if _TOC_ENTRY_RE.match(line):
            if cur_start < 0:
                cur_start = i
                cur_count = 1
            else:
                cur_count += 1
        else:
            if cur_count >= 3 and cur_count > (best_end - best_start):
                best_start = cur_start
                best_end = cur_start + cur_count
            cur_start = -1
            cur_count = 0

    # Check trailing block
    if cur_count >= 3 and cur_count > (best_end - best_start):
        best_start = cur_start
        best_end = cur_start + cur_count

    return best_start, best_end


def _detect_figure_table_block(lines: List[str]) -> Tuple[int, int]:
    """
    Detect a figure listing table block (e.g. "Table des Figures").

    Looks for a contiguous markdown table containing rows matching
    ``| Fig. XX | ... |``.  Returns (start, end) 0-based line indices
    including header/separator rows, or (-1, -1) if not found.
    A valid block needs >= 2 Fig. rows.
    """
    start = end = -1
    in_table = False
    fig_count = 0

    for i, line in enumerate(lines):
        if _FIG_TABLE_ROW_RE.match(line):
            if not in_table:
                # Look back for table header rows (| Header | ... |)
                start = i
                for j in range(i - 1, max(i - 5, -1), -1):
                    if lines[j].strip().startswith('|'):
                        start = j
                    else:
                        break
            in_table = True
            fig_count += 1
            end = i + 1
        elif in_table and not line.strip().startswith('|'):
            break

    if fig_count < 2:
        return -1, -1

    return start, end


def _inventory_sections(
    lines: List[str],
) -> Tuple[List[NumberedElement], Dict[str, str]]:
    """
    Collect all numbered section headings and compute renumbering map.

    Returns:
        (elements, old_to_new) where old_to_new maps old section numbers
        to new sequential ones.
    """
    elements: List[NumberedElement] = []
    counters = [0] * 7  # h1..h6 + buffer

    for i, line in enumerate(lines):
        m = _SECTION_RE.match(line)
        if m:
            hashes, old_num, title = m.group(1), m.group(2), m.group(3)
            level = len(hashes) - 1  # 0-indexed (h1=0, h2=1, ...)

            # Increment counter for this level, reset deeper
            counters[level] += 1
            for j in range(level + 1, 7):
                counters[j] = 0

            # Build new number from counters
            new_num = ".".join(
                str(counters[k]) for k in range(level + 1)
                if counters[k] > 0
            )

            # Normalize old_num (strip trailing dot for comparison)
            old_clean = old_num.rstrip(".")

            col_start = len(hashes) + 1  # position after "## "
            col_end = col_start + len(old_num)

            elements.append(NumberedElement(
                kind="section",
                line_num=i + 1,
                old_number=old_clean,
                new_number=new_num,
                col_start=col_start,
                col_end=col_end,
                context=line,
            ))

    # Build mapping
    old_to_new: Dict[str, str] = {}
    for el in elements:
        if el.old_number != el.new_number:
            old_to_new[el.old_number] = el.new_number

    return elements, old_to_new


def _inventory_toc(
    lines: List[str],
    toc_start: int,
    toc_end: int,
    section_elements: List[NumberedElement],
) -> List[NumberedElement]:
    """
    Collect ToC entries and pair them to sections by ordinal position.

    The N-th ToC entry maps to the N-th section heading, so renumbering
    is correct even when old numbers are mismatched between ToC and headings.
    """
    elements: List[NumberedElement] = []

    toc_idx = 0
    for i in range(toc_start, toc_end):
        m = _TOC_ENTRY_RE.match(lines[i])
        if m:
            indent, old_num, title, anchor = (
                m.group(1), m.group(2), m.group(3), m.group(4),
            )
            # Pair by position: toc_idx-th ToC entry -> toc_idx-th section
            if toc_idx < len(section_elements):
                new_num = section_elements[toc_idx].new_number
            else:
                new_num = old_num  # no matching section
            col_start = len(indent)
            col_end = col_start + len(old_num)
            elements.append(NumberedElement(
                kind="toc",
                line_num=i + 1,
                old_number=old_num,
                new_number=new_num,
                col_start=col_start,
                col_end=col_end,
                context=lines[i],
            ))
            toc_idx += 1

    return elements


def _format_figure_number(old_str: str, new_int: int) -> str:
    """
    Format a new figure number preserving the padding style of the original.

    If the original was 2+ digits (e.g. "04", "12"), output is zero-padded
    to the same width.  Single-digit originals ("4") stay unpadded.
    """
    width = len(old_str)
    if width >= 2:
        return f"{new_int:0{width}d}"
    return str(new_int)


def _inventory_figure_table_refs(
    lines: List[str],
    toc_start: int,
    toc_end: int,
    fig_table_start: int,
    fig_table_end: int,
) -> Tuple[List[NumberedElement], List[NumberedElement], Dict[int, int]]:
    """
    Collect figure and table references throughout the document.

    Scans for figure definitions (captions) in body order, skipping
    both the ToC block and figure listing table block.  Supports both
    plain-text patterns (``Figure N``, ``Fig. N``) and HTML patterns
    (``<b>Fig. XX</b>``, ``src="assets/figXX_..."``).

    Returns:
        (fig_elements, tbl_elements, fig_int_map)
        where fig_int_map maps old_int -> new_int for figures.
    """
    fig_elements: List[NumberedElement] = []
    tbl_elements: List[NumberedElement] = []

    # ---------- First pass: collect figure/table numbers in body order ----------
    fig_numbers_seen: List[int] = []
    tbl_numbers_seen: List[str] = []

    def _in_skip_zone(idx: int) -> bool:
        if toc_start <= idx < toc_end:
            return True
        if fig_table_start <= idx < fig_table_end:
            return True
        return False

    for i, line in enumerate(lines):
        if _in_skip_zone(i):
            continue

        # Figures — collect from all pattern types
        fig_nums_in_line: List[int] = []

        for m in _HTML_CAPTION_RE.finditer(line):
            fig_nums_in_line.append(int(m.group(2)))

        for m in _FIGURE_REF_RE.finditer(line):
            fig_nums_in_line.append(int(m.group(2)))

        for m in _IMG_SRC_RE.finditer(line):
            fig_nums_in_line.append(int(m.group(2)))

        for num in fig_nums_in_line:
            if num not in fig_numbers_seen:
                fig_numbers_seen.append(num)

        # Tables
        for m in _TABLE_REF_RE.finditer(line):
            num = m.group(2)
            if num not in tbl_numbers_seen:
                tbl_numbers_seen.append(num)

    # ---------- Build renumbering maps ----------
    fig_int_map: Dict[int, int] = {}
    for idx, old_int in enumerate(fig_numbers_seen, 1):
        fig_int_map[old_int] = idx

    tbl_map: Dict[str, str] = {}
    for idx, num in enumerate(tbl_numbers_seen, 1):
        tbl_map[num] = str(idx)

    # ---------- Second pass: collect all references with positions ----------
    for i, line in enumerate(lines):
        if _in_skip_zone(i):
            continue

        # Collect substitutions for this line to deduplicate overlaps
        fig_subs: List[Tuple[int, int, str, str, str]] = []
        # Each: (col_start, col_end, old_str, new_str, label)

        # HTML captions: <b>Fig. XX</b>
        for m in _HTML_CAPTION_RE.finditer(line):
            old_str = m.group(2)
            old_int = int(old_str)
            new_int = fig_int_map.get(old_int, old_int)
            new_str = _format_figure_number(old_str, new_int)
            fig_subs.append((m.start(2), m.end(2), old_str, new_str, m.group(1)))

        # Img src: src="assets/figXX_..."
        for m in _IMG_SRC_RE.finditer(line):
            old_str = m.group(2)
            old_int = int(old_str)
            new_int = fig_int_map.get(old_int, old_int)
            new_str = _format_figure_number(old_str, new_int)
            fig_subs.append((m.start(2), m.end(2), old_str, new_str, "img_src"))

        # Plain text: Figure N, Fig. N
        for m in _FIGURE_REF_RE.finditer(line):
            old_str = m.group(2)
            old_int = int(old_str)
            new_int = fig_int_map.get(old_int, old_int)
            new_str = _format_figure_number(old_str, new_int)
            fig_subs.append((m.start(2), m.end(2), old_str, new_str, m.group(1)))

        # Deduplicate overlapping matches (keep first = most specific)
        fig_subs.sort(key=lambda x: x[0])
        deduped: List[Tuple[int, int, str, str, str]] = []
        last_end = -1
        for start, end, old_s, new_s, lbl in fig_subs:
            if start >= last_end:
                deduped.append((start, end, old_s, new_s, lbl))
                last_end = end

        for start, end, old_s, new_s, lbl in deduped:
            fig_elements.append(NumberedElement(
                kind="figure",
                line_num=i + 1,
                old_number=old_s,
                new_number=new_s,
                col_start=start,
                col_end=end,
                context=line,
                label=lbl,
            ))

        # Tables
        for m in _TABLE_REF_RE.finditer(line):
            label, num = m.group(1), m.group(2)
            new_num = tbl_map.get(num, num)
            tbl_elements.append(NumberedElement(
                kind="table",
                line_num=i + 1,
                old_number=num,
                new_number=new_num,
                col_start=m.start(2),
                col_end=m.end(2),
                context=line,
                label=label,
            ))

    return fig_elements, tbl_elements, fig_int_map


def _inventory_section_xrefs(
    lines: List[str],
    section_map: Dict[str, str],
    toc_start: int,
    toc_end: int,
    section_lines: Set[int],
) -> List[NumberedElement]:
    """Collect cross-references to sections (Section N, §N.M)."""
    elements: List[NumberedElement] = []

    for i, line in enumerate(lines):
        # Skip ToC and section heading lines (already handled)
        if toc_start <= i < toc_end:
            continue
        if (i + 1) in section_lines:
            continue

        for m in _SECTION_XREF_RE.finditer(line):
            label, old_num = m.group(1), m.group(2)
            new_num = section_map.get(old_num, old_num)
            elements.append(NumberedElement(
                kind="xref",
                line_num=i + 1,
                old_number=old_num,
                new_number=new_num,
                col_start=m.start(2),
                col_end=m.end(2),
                context=line,
                label=label,
            ))

    return elements


# ---------------------------------------------------------------------------
# Phase 2 + 3: Renumber and Apply
# ---------------------------------------------------------------------------

def _apply_line_substitutions(
    line: str,
    subs: List[Tuple[int, int, str]],
) -> str:
    """
    Apply multiple column-based substitutions to a line.
    subs is [(col_start, col_end, new_text), ...], applied right-to-left
    to preserve column positions.
    """
    # Sort by col_start descending to apply from right to left
    for start, end, new_text in sorted(subs, key=lambda s: s[0], reverse=True):
        line = line[:start] + new_text + line[end:]
    return line


def _apply_section_renumber(
    lines: List[str],
    elements: List[NumberedElement],
) -> List[str]:
    """Apply section heading renumbering."""
    section_by_line: Dict[int, NumberedElement] = {
        el.line_num: el for el in elements if el.changed
    }

    result = []
    for i, line in enumerate(lines):
        line_num = i + 1
        if line_num in section_by_line:
            el = section_by_line[line_num]
            m = _SECTION_RE.match(line)
            if m:
                hashes = m.group(1)
                title = m.group(3)
                # Preserve trailing dot style from original
                dot = "." if el.context and _SECTION_RE.match(el.context) else ""
                line = f"{hashes} {el.new_number}. {title}"
        result.append(line)

    return result


def _apply_toc_renumber(
    lines: List[str],
    elements: List[NumberedElement],
) -> List[str]:
    """Apply ToC entry renumbering."""
    toc_by_line: Dict[int, NumberedElement] = {
        el.line_num: el for el in elements if el.changed
    }

    result = []
    for i, line in enumerate(lines):
        line_num = i + 1
        if line_num in toc_by_line:
            el = toc_by_line[line_num]
            m = _TOC_ENTRY_RE.match(line)
            if m:
                indent = m.group(1)
                title = m.group(3)
                anchor = m.group(4)
                line = f"{indent}{el.new_number}. [{title}](#{anchor})"
        result.append(line)

    return result


def _apply_inline_renumber(
    lines: List[str],
    elements: List[NumberedElement],
) -> List[str]:
    """Apply figure/table/xref renumbering (inline substitutions)."""
    # Group elements by line number
    by_line: Dict[int, List[NumberedElement]] = {}
    for el in elements:
        if el.changed:
            by_line.setdefault(el.line_num, []).append(el)

    result = []
    for i, line in enumerate(lines):
        line_num = i + 1
        if line_num in by_line:
            subs = [
                (el.col_start, el.col_end, el.new_number)
                for el in by_line[line_num]
            ]
            line = _apply_line_substitutions(line, subs)
        result.append(line)

    return result


def _rebuild_figure_table(
    lines: List[str],
    fig_table_start: int,
    fig_table_end: int,
    fig_int_map: Dict[int, int],
) -> List[str]:
    """
    Rebuild a figure listing table: reorder rows by new figure number
    and update figure numbers within each row.

    The figure listing table (e.g. "Table des Figures") is rebuilt
    with rows sorted by their new figure number.
    """
    if fig_table_start < 0 or fig_table_end < 0:
        return lines

    headers = []
    fig_rows = []

    for i in range(fig_table_start, fig_table_end):
        m = _FIG_TABLE_ROW_RE.match(lines[i])
        if m:
            old_num = int(m.group(2))
            rest = m.group(3)
            fig_rows.append((old_num, rest))
        else:
            headers.append(lines[i])

    # Compute new numbers and sort
    rows_with_new_num = []
    for old_num, rest in fig_rows:
        new_num = fig_int_map.get(old_num, old_num)
        rows_with_new_num.append((new_num, rest))
    rows_with_new_num.sort(key=lambda x: x[0])

    # Rebuild block
    new_block = list(headers)
    for new_num, rest in rows_with_new_num:
        new_block.append(f"| Fig. {new_num:02d} |{rest}")

    return lines[:fig_table_start] + new_block + lines[fig_table_end:]


# ---------------------------------------------------------------------------
# SVG file renaming
# ---------------------------------------------------------------------------

def rename_figure_files(
    fig_int_map: Dict[int, int],
    assets_dir: str | Path,
    dry_run: bool = False,
    pattern: str = "fig[0-9][0-9]_*.svg",
) -> List[Tuple[str, str]]:
    """
    Rename figure SVG files using a 2-pass approach to avoid collisions.

    Args:
        fig_int_map: Mapping of old figure int to new figure int.
        assets_dir: Directory containing SVG files.
        dry_run: If True, compute renames but don't execute.
        pattern: Glob pattern for figure files.

    Returns:
        List of (old_name, new_name) pairs.
    """
    import re as _re

    assets = Path(assets_dir)
    renames: List[Tuple[str, str]] = []
    fig_files = sorted(assets.glob(pattern))

    file_map: Dict[Path, Tuple[Path, Path]] = {}
    for f in fig_files:
        m = _re.match(r'fig(\d{2})_(.+)', f.name)
        if m:
            old_num = int(m.group(1))
            rest = m.group(2)
            if old_num in fig_int_map and fig_int_map[old_num] != old_num:
                new_num = fig_int_map[old_num]
                temp_name = f"__tmp_fig{new_num:02d}_{rest}"
                final_name = f"fig{new_num:02d}_{rest}"
                file_map[f] = (assets / temp_name, assets / final_name)
                renames.append((f.name, final_name))

    if not dry_run and file_map:
        # Pass 1: rename to temp (avoids collisions)
        for src, (temp, _) in file_map.items():
            shutil.move(str(src), str(temp))
        # Pass 2: rename to final
        for _, (temp, final) in file_map.items():
            shutil.move(str(temp), str(final))

    return renames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ALL_SCOPES = {"sections", "toc", "figures", "tables", "xrefs"}


def renumber_markdown(
    content: str,
    scope: Optional[List[str]] = None,
) -> Tuple[str, RenumberReport]:
    """
    Renumber a markdown document.

    Three-phase pipeline:
      1. Inventory — scan all numbered elements
      2. Renumber — compute correct sequential numbers
      3. Apply — substitute old with new

    Args:
        content: Markdown text.
        scope: Which element types to renumber.
            Options: "sections", "toc", "figures", "tables", "xrefs",
                     "figure_table" (reorder figure listing table).
            Default (None): all types except figure_table.

    Returns:
        (new_content, report)
    """
    active_scopes: Set[str] = set(scope) if scope else _ALL_SCOPES
    lines = content.split("\n")
    report = RenumberReport()

    # --- Phase 1: Inventory ---

    # Sections
    section_elements: List[NumberedElement] = []
    section_map: Dict[str, str] = {}
    section_lines: Set[int] = set()

    if "sections" in active_scopes or "toc" in active_scopes or "xrefs" in active_scopes:
        section_elements, section_map = _inventory_sections(lines)
        section_lines = {el.line_num for el in section_elements}
        report.sections_found = len(section_elements)

    # ToC
    toc_start, toc_end = _detect_toc_block(lines)
    toc_elements: List[NumberedElement] = []
    if "toc" in active_scopes and toc_start >= 0:
        toc_elements = _inventory_toc(
            lines, toc_start, toc_end, section_elements,
        )
        report.toc_entries_found = len(toc_elements)

    # Figure listing table
    fig_table_start, fig_table_end = _detect_figure_table_block(lines)

    # Figures and Tables
    fig_elements: List[NumberedElement] = []
    tbl_elements: List[NumberedElement] = []
    fig_int_map: Dict[int, int] = {}
    if "figures" in active_scopes or "tables" in active_scopes:
        fig_elements, tbl_elements, fig_int_map = _inventory_figure_table_refs(
            lines, toc_start, toc_end, fig_table_start, fig_table_end,
        )
        if "figures" not in active_scopes:
            fig_elements = []
            fig_int_map = {}
        if "tables" not in active_scopes:
            tbl_elements = []
        report.figure_refs_found = len(fig_elements)
        report.table_refs_found = len(tbl_elements)
        report.figure_map = fig_int_map

    # Cross-references
    xref_elements: List[NumberedElement] = []
    if "xrefs" in active_scopes and section_map:
        xref_elements = _inventory_section_xrefs(
            lines, section_map, toc_start, toc_end, section_lines,
        )
        report.xrefs_found = len(xref_elements)

    # --- Phase 2 + 3: Apply ---

    # Sections first (rewrites whole heading lines)
    if "sections" in active_scopes and section_elements:
        lines = _apply_section_renumber(lines, section_elements)

    # ToC (rewrites whole ToC lines)
    if "toc" in active_scopes and toc_elements:
        lines = _apply_toc_renumber(lines, toc_elements)

    # Figure listing table rebuild (reorder + renumber rows)
    if "figure_table" in active_scopes and fig_int_map and fig_table_start >= 0:
        lines = _rebuild_figure_table(
            lines, fig_table_start, fig_table_end, fig_int_map,
        )
        # Count fig table rows for report
        report.fig_table_rows = sum(
            1 for i in range(fig_table_start, fig_table_end)
            if _FIG_TABLE_ROW_RE.match(lines[i]) if i < len(lines)
        )
        # Recalculate bounds (may have shifted)
        fig_table_start, fig_table_end = _detect_figure_table_block(lines)

    # Inline substitutions (figures, tables, xrefs) — column-based
    inline_elements = fig_elements + tbl_elements + xref_elements
    if inline_elements:
        lines = _apply_inline_renumber(lines, inline_elements)

    # --- Report ---
    all_elements = (
        section_elements + toc_elements
        + fig_elements + tbl_elements + xref_elements
    )
    report.elements = all_elements
    report.changes = sum(1 for el in all_elements if el.changed)

    return "\n".join(lines), report


def renumber_file(
    path: str,
    scope: Optional[List[str]] = None,
    dry_run: bool = False,
    assets_dir: Optional[str] = None,
) -> RenumberReport:
    """
    Renumber a markdown file in place.

    Args:
        path: Path to markdown file.
        scope: Which element types to renumber (see renumber_markdown).
        dry_run: If True, compute changes but don't write.
        assets_dir: If provided and "figures" in scope, rename SVG files too.

    Returns:
        RenumberReport with all changes.
    """
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    new_content, report = renumber_markdown(content, scope=scope)

    if not dry_run and report.changes > 0:
        p.write_text(new_content, encoding="utf-8")

    # SVG file renaming
    if assets_dir and report.figure_map and "figures" in (scope or _ALL_SCOPES):
        renames = rename_figure_files(
            report.figure_map, assets_dir, dry_run=dry_run,
        )
        if renames:
            print(f"  SVG renames: {len(renames)} files")
            for old_name, new_name in renames:
                print(f"    {old_name} -> {new_name}")

    return report
