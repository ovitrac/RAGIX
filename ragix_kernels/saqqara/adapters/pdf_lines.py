"""
saqqara.adapters.pdf_lines — the fragments of one visual line, joined on request.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Opt-in: the `pdf.line_join` option, off by default, and off means this module is never called.

A page's text reaches the pdf reader in the pieces its content stream shows it in, one
text-showing operation at a time, and a producer may cut a line wherever it changes font,
kerning or position — inside a word, and inside a number: a date set as three operations
arrives as `2`, `7 mars 203`, `1`. Each piece is an honest observation of the stream and a poor
unit of text: a search for the word misses it, and an extractor that reads across the pieces
joins what may have been two table cells.

This module joins consecutive pieces that sit on one baseline and abut, and does three things a
repair would not:

- **the raw pieces stay on the joined node**, each with its own facts, and the separator the join
  put before each piece is recorded beside it — the reading the join replaced is still there;
- **a join is decided by geometry the file states** — the baseline, where the next piece starts,
  where the previous one measurably ends — and where the end is not known (a font that declares
  no widths) the pieces stay apart and the undecided boundary is counted on the page;
- **every boundary where a digit meets a digit is marked for review.** The join may have made one
  number of two, and whether it did is a question for a person holding the page, never for this
  module: a joined value is to be verified before it is used.

Joins are made along upright lines only. Type turned on the page is left as it was read.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

__all__ = [
    "ABUT_EM",
    "BASELINE_EM",
    "JOIN_PAGE_FACTS",
    "JOIN_TEXT_FACTS",
    "REVIEW_REASONS",
    "WORD_GAP_EM",
    "join_lines",
]

#: Two pieces share a baseline when their origins differ vertically by at most this
#: fraction of the larger size. A tenth of an em: a producer repeats the coordinate
#: for the pieces of one line, while a superscript sits about a third of an em above
#: it — a raised exponent is not joined to its base.
BASELINE_EM = 0.1

#: Where the next piece starts, measured from where the previous one ends, in ems of
#: the previous piece's size. Within ABUT_EM either way the glyphs touch: the cut fell
#: inside a word and the join inserts nothing. Beyond it and up to WORD_GAP_EM the gap
#: is a word space — a quarter to a third of an em, stretched in justified text — and
#: the join inserts one space. Beyond WORD_GAP_EM the pieces are further apart than a
#: word space ever is: a tab stop, a column gutter, the next cell of a table. They stay
#: apart, which is what they were before this module existed.
ABUT_EM = 0.1
WORD_GAP_EM = 1.0

#: Why a joined node is marked. Closed, and produced: a digit met a digit at a join.
REVIEW_REASONS = ("digit-run-joined",)

#: What a text record carries in addition when the join is on, and what a page carries.
#: Every text record carries both facts, `None` where the record is not a join, so the
#: vocabulary of the option is the same on every record it produces.
JOIN_TEXT_FACTS = ("fragments", "review")
JOIN_PAGE_FACTS = ("line_join",)

#: One placement as the pdf reader holds it: text (stripped), x, y, font size, font, width.
Placement = tuple
#: What the join needs beside it, in page space: origin x, origin y, upright, and whether
#: the raw text began or ended with white space. None where the reader could not say.
Geometry = Optional[tuple]


def _separator(a: Placement, ga: Geometry, b: Placement, gb: Geometry,
               counts: dict[str, int]) -> Optional[str]:
    """What joins `b` to `a`, or None when they are not one line.

    None is the answer this module gives by default: every condition below is one
    more thing that must be read from the file before two pieces become one.
    """
    if ga is None or gb is None:
        counts["undecided"] += 1                        # no geometry: cannot say
        return None
    ax, ay, a_upright, _a_lead, a_trail = ga
    bx, by, b_upright, b_lead, _b_trail = gb
    if not (a_upright and b_upright):
        return None
    size = max(float(a[3] or 0), float(b[3] or 0))
    if size <= 0:
        counts["undecided"] += 1
        return None
    if abs(by - ay) > BASELINE_EM * size:
        return None                                     # another line
    width = a[5]
    if width is None:
        counts["undecided"] += 1                        # its end is not known
        return None
    em = float(a[3]) if a[3] else size
    gap = bx - (ax + float(width))
    if gap < -ABUT_EM * em or gap > WORD_GAP_EM * em:
        return None
    if a_trail or b_lead or gap > ABUT_EM * em:
        return " "
    return ""


def _fragment(p: Placement, separator: Optional[str]) -> dict[str, Any]:
    text, x, y, size, font, width = p
    return {"text": text, "x": round(x, 2), "y": round(y, 2),
            "font_size": round(size, 2), "font": font,
            "width": None if width is None else round(width, 2),
            "sep": separator}


def join_lines(placements: Sequence[Placement], geometry: Sequence[Geometry]):
    """Join the pieces of each visual line.

    Returns ``(lines, extras, counts)``: the placements after the join, in reading
    order; per line, the facts the option adds (``fragments`` and ``review``, both
    ``None`` for a line of one piece); and the page's counts — lines joined, pieces
    absorbed into them, joins marked for review, boundaries left undecided.
    """
    if len(placements) != len(geometry):
        raise ValueError("every placement needs its geometry, and none may be extra")
    counts = {"joined": 0, "absorbed": 0, "review": 0, "undecided": 0}
    groups: list[list[int]] = []
    separators: list[list[str]] = []
    for index in range(len(placements)):
        if groups:
            last = groups[-1][-1]
            sep = _separator(placements[last], geometry[last],
                             placements[index], geometry[index], counts)
            if sep is not None:
                groups[-1].append(index)
                separators[-1].append(sep)
                continue
        groups.append([index])
        separators.append([])

    lines: list[Placement] = []
    extras: list[dict[str, Any]] = []
    for members, between in zip(groups, separators):
        first = placements[members[0]]
        if len(members) == 1:
            lines.append(first)
            extras.append({"fragments": None, "review": None})
            continue
        pieces = [placements[m] for m in members]
        text = pieces[0][0] + "".join(s + p[0] for s, p in zip(between, pieces[1:]))
        # The extent is measured in page space, from the first origin to the end of
        # the last piece — unknown if the last piece's width is.
        last, g_first, g_last = pieces[-1], geometry[members[0]], geometry[members[-1]]
        width = (None if last[5] is None
                 else g_last[0] + float(last[5]) - g_first[0])
        lines.append((text, first[1], first[2], first[3], first[4], width))
        digits = [k for k in range(1, len(pieces))
                  if pieces[k - 1][0][-1:].isdigit() and pieces[k][0][:1].isdigit()]
        extras.append({
            "fragments": [_fragment(p, None if k == 0 else between[k - 1])
                          for k, p in enumerate(pieces)],
            "review": ({"reason": REVIEW_REASONS[0], "boundaries": digits}
                       if digits else None),
        })
        counts["joined"] += 1
        counts["absorbed"] += len(pieces)
        counts["review"] += 1 if digits else 0
    return lines, extras, counts
