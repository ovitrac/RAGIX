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
  - Table references: ``Table N``, ``Tableau N``, ``Tab. N`` (definition + cross-ref)
  - Cross-references: ``Section N``, ``section N.M``, ``§N``, ``§N.M``
  - Bilingual FR/EN patterns

Usage::

    from ragix_kernels.shared.md_renumber import renumber_markdown

    new_content, report = renumber_markdown(content)
    print(report.summary())

    # Selective scope
    new_content, report = renumber_markdown(content, scope=["sections", "toc"])

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-13
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NumberedElement:
    """A numbered element found in the document."""
    kind: str           # "section", "toc", "figure", "table", "xref"
    line_num: int       # 1-based line number
    old_number: str     # e.g. "3", "3.1", "12"
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

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Markdown Renumber Report",
            f"  Sections: {self.sections_found} found",
            f"  ToC entries: {self.toc_entries_found} found",
            f"  Figure refs: {self.figure_refs_found} found",
            f"  Table refs: {self.table_refs_found} found",
            f"  Cross-refs: {self.xrefs_found} found",
            f"  Changes applied: {self.changes}",
        ]
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

# Figure definitions and references (bilingual)
_FIGURE_REF_RE = re.compile(
    r'(?i)\b(Figure|Fig\.|fig\.)\s+(\d+)\b'
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
            # Pair by position: toc_idx-th ToC entry → toc_idx-th section
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


def _inventory_figure_table_refs(
    lines: List[str],
    toc_start: int,
    toc_end: int,
) -> Tuple[List[NumberedElement], List[NumberedElement]]:
    """
    Collect figure and table references throughout the document.

    Counts definitions in order of appearance and builds renumbering.
    Skips ToC block to avoid false matches.
    """
    fig_elements: List[NumberedElement] = []
    tbl_elements: List[NumberedElement] = []

    # First pass: collect all figure/table numbers in order of appearance
    fig_numbers_seen: List[str] = []
    tbl_numbers_seen: List[str] = []

    for i, line in enumerate(lines):
        # Skip ToC block
        if toc_start <= i < toc_end:
            continue

        for m in _FIGURE_REF_RE.finditer(line):
            num = m.group(2)
            if num not in fig_numbers_seen:
                fig_numbers_seen.append(num)

        for m in _TABLE_REF_RE.finditer(line):
            num = m.group(2)
            if num not in tbl_numbers_seen:
                tbl_numbers_seen.append(num)

    # Build renumbering maps (sequential 1, 2, 3, ...)
    fig_map: Dict[str, str] = {}
    for idx, num in enumerate(fig_numbers_seen, 1):
        fig_map[num] = str(idx)

    tbl_map: Dict[str, str] = {}
    for idx, num in enumerate(tbl_numbers_seen, 1):
        tbl_map[num] = str(idx)

    # Second pass: collect all references with new numbers
    for i, line in enumerate(lines):
        if toc_start <= i < toc_end:
            continue

        for m in _FIGURE_REF_RE.finditer(line):
            label, num = m.group(1), m.group(2)
            new_num = fig_map.get(num, num)
            fig_elements.append(NumberedElement(
                kind="figure",
                line_num=i + 1,
                old_number=num,
                new_number=new_num,
                col_start=m.start(2),
                col_end=m.end(2),
                context=line,
                label=label,
            ))

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

    return fig_elements, tbl_elements


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
            Options: "sections", "toc", "figures", "tables", "xrefs".
            Default (None): all types.

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

    # Figures and Tables
    fig_elements: List[NumberedElement] = []
    tbl_elements: List[NumberedElement] = []
    if "figures" in active_scopes or "tables" in active_scopes:
        fig_elements, tbl_elements = _inventory_figure_table_refs(
            lines, toc_start, toc_end,
        )
        if "figures" not in active_scopes:
            fig_elements = []
        if "tables" not in active_scopes:
            tbl_elements = []
        report.figure_refs_found = len(fig_elements)
        report.table_refs_found = len(tbl_elements)

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
) -> RenumberReport:
    """
    Renumber a markdown file in place.

    Args:
        path: Path to markdown file.
        scope: Which element types to renumber (see renumber_markdown).
        dry_run: If True, compute changes but don't write.

    Returns:
        RenumberReport with all changes.
    """
    from pathlib import Path as P
    p = P(path)
    content = p.read_text(encoding="utf-8")
    new_content, report = renumber_markdown(content, scope=scope)

    if not dry_run and report.changes > 0:
        p.write_text(new_content, encoding="utf-8")

    return report
