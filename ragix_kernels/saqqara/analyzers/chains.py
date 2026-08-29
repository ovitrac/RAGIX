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

__all__ = ["BLANK_RUNG", "SelfReferenceError", "anchors"]

#: A rung whose tile exists and holds nothing.
BLANK_RUNG = "(blank-label)"


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
            "row_chain": [],
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

    col_chain = _column_chain(cells, header_rows, col)
    row_chain = _row_chain(cells, label_cols, row)

    return {
        "cell": ref,
        "col_chain": col_chain,
        # Kept for callers that only ever wanted the deepest rung. It is the last
        # element of the chain, never the reference itself.
        "col_header": col_chain[-1] if col_chain else None,
        "row_chain": row_chain,
        "uncertain": False,
        "how": "lattice",
    }


def _column_chain(cells, header_rows: list[int], col: int) -> list[str]:
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
    return [text for _, text in rungs]


def _row_chain(cells, label_cols: list[int], row: int) -> list[str]:
    """Left to right through the label zone, blank tiles included as blank."""
    chain: list[str] = []
    last: str | None = None
    for col in sorted(label_cols):
        cell, extent = _covering(cells, row, col)
        if extent is not None and extent.to_a1() == last:
            continue                      # one tile spanning two label columns
        if cell is None:
            continue
        last = extent.to_a1() if extent is not None else None
        chain.append(cell.text if cell.text is not None else BLANK_RUNG)
    return chain
