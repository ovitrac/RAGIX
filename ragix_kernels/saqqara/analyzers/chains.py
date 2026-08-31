"""
saqqara.analyzers.chains — what heads this column, and what labels this row.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.8-K3.10 and K3.21-K3.24 of SPEC.md.

A value in a grid means nothing alone. `12` is an answer only once you know it sits under
"Technical means" inside "Resources committed", on the row labelled "Lot B". Reconstructing those
two ancestries is what turns a spreadsheet into evidence somebody can check.

Three things this module refuses to do.

**It does not read a chain depth-first.** A column chain runs broadest first, by interval
containment: the widest span that covers the column, then the next widest inside it, down to the
narrowest. That ordering is the document's own hierarchy, and reversing it inverts the meaning.

**It does not skip a blank tile.** Where a label tile exists but was left empty, the rung is
reported as blank rather than omitted. Somebody left it blank; that is a fact about the document,
and a chain that quietly shortened itself would report an ancestry the document does not have.

**It does not drop where a rung was read.** A chain says what heads a column; a rung says it and
names the cell it was read from. The two travel together, because a citation that cannot be
followed back to a position is a claim about a document rather than a reading of one — and the
address costs nothing to keep: resolving the covering cell is how a rung is found in the first
place.

**It does not treat a header as a value.** Asking for the chain of a position inside the header
band or the label zone raises. Those regions are not value candidates: a header is not an answer
to itself, and a reader that let one be addressed as a value would produce a citation pointing at
its own question.

Where the block abstained, every result says so and carries the reason, so a caller cannot mistake
a fallback for a reading.
"""

from __future__ import annotations

from typing import Any

from ..model import Node
from .geometry import Rect, coord, parse_range
from .grid import GridCell, grid_cells

__all__ = ["BLANK_RUNG", "SelfReferenceError", "anchors", "rung"]

#: A rung whose tile exists and holds nothing.
BLANK_RUNG = "(blank-label)"


def rung(text: str, ref: str) -> dict[str, str]:
    """One step of a chain: what it says, and the cell that says it.

    A mapping of primitives rather than an object, deliberately. Everything
    `anchors` returns today is JSON-native, and a caller that stores a chain
    beside the answer it explains would otherwise meet the one value in the
    result that will not serialise. `ref` is an A1 reference, and a merged tile
    reports its whole extent — the rung is the tile, not a corner of it.
    """
    return {"text": text, "ref": ref}


class SelfReferenceError(ValueError):
    """A header or label position was addressed as if it held a value."""


def _covering(cells: dict[tuple[int, int], GridCell], row: int, col: int):
    """The cell whose own position or merged extent covers (row, col)."""
    direct = cells.get((row, col))
    if direct is not None:
        # A cell that anchors a merge covers the whole extent. Returning its own
        # square instead makes one tile look like two rungs when it spans two
        # label columns.
        return direct, direct.extent
    for cell in cells.values():
        if cell.merged and cell.extent.contains(row, col):
            return cell, cell.extent
    return None, None


def anchors(block: Node, ref: str) -> dict[str, Any]:
    """The column chain and the row chain of one position inside a block."""
    header = block.facts.get("header")
    if header is None:
        raise ValueError("run the header_bands analyzer before reading chains")

    row, col = coord(ref)
    # The block's format is read once, to pick a mapping. Below this line the
    # chain rules see grid cells and never learn which format produced them.
    cells_list, _mapping = grid_cells(block)
    cells = {(c.row, c.col): c for c in cells_list}

    if header["uncertain"]:
        # An abstention is not a licence to guess: the caller is told, in every
        # result, that no band was read and that this is a fallback.
        return {
            "cell": ref,
            "col_chain": [],
            "col_rungs": [],
            "row_chain": [],
            "row_rungs": [],
            "col_header": None,
            "uncertain": True,
            "abstention": header["abstention"],
            "how": "fallback-no-band",
        }

    label_cols = [coord(f"{name}1")[1] for name in header["label_cols"]]
    header_rows = header["header_rows"]

    if row in header_rows:
        raise SelfReferenceError(f"{ref} is inside the header band: it is not a value position")
    if col in label_cols:
        raise SelfReferenceError(f"{ref} is inside the label zone: it is not a value position")

    col_rungs = _column_chain(cells, header_rows, col)
    row_rungs = _row_chain(cells, label_cols, row)
    # The texts are a view of the rungs, derived here and nowhere else: two
    # lists built independently would be two things that can disagree about one
    # chain, and the disagreement would be found by whoever cited the wrong one.
    col_chain = [step["text"] for step in col_rungs]
    row_chain = [step["text"] for step in row_rungs]

    return {
        "cell": ref,
        "col_chain": col_chain,
        # The same rungs with the cell each was read from, for a caller that has
        # to show its work.
        "col_rungs": col_rungs,
        # Kept for callers that only ever wanted the deepest rung. It is the last
        # element of the chain, never the reference itself.
        "col_header": col_chain[-1] if col_chain else None,
        "row_chain": row_chain,
        "row_rungs": row_rungs,
        "uncertain": False,
        "how": "lattice",
    }


def _column_chain(cells, header_rows: list[int], col: int) -> list[dict[str, str]]:
    """Broadest first, by containment. A merge across tiers is one rung."""
    rungs: list[tuple[Rect, str]] = []
    for row in header_rows:
        cell, extent = _covering(cells, row, col)
        if cell is None or cell.text is None:
            continue
        if rungs and rungs[-1][0].to_a1() == extent.to_a1():
            continue                      # the same merge seen again on the next tier
        rungs.append((extent, cell.text))

    rungs.sort(key=lambda pair: (-pair[0].width, pair[0].top))
    return [rung(text, extent.to_a1()) for extent, text in rungs]


def _row_chain(cells, label_cols: list[int], row: int) -> list[dict[str, str]]:
    """Left to right through the label zone, blank tiles included as blank."""
    chain: list[dict[str, str]] = []
    last: str | None = None
    for col in sorted(label_cols):
        cell, extent = _covering(cells, row, col)
        if extent is not None and extent.to_a1() == last:
            continue                      # one tile spanning two label columns
        if cell is None:
            continue
        last = extent.to_a1() if extent is not None else None
        # A blank tile is a rung with an address: where the label is missing is
        # exactly the fact a reviewer needs, and it is unaddressable without it.
        chain.append(rung(cell.text if cell.text is not None else BLANK_RUNG,
                          last if last is not None else extent.to_a1()))
    return chain
