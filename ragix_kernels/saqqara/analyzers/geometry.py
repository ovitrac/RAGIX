"""
saqqara.analyzers.geometry — the shared grid vocabulary.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Supports K3.a-K3.f. The analyzers reason about rectangles, intervals and containment; putting
that vocabulary in one place is what lets the same recognition run over a spreadsheet and over a
word-processing table without either format reaching the rules (K3.34).

A1 notation is used on the way in and out because it is what a human reader can check against the
file. Inside, coordinates are 1-based (row, column) integers.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Rect", "a1", "coord", "is_laminar", "parse_range", "rng"]

_LETTERS = 26


def a1(row: int, col: int) -> str:
    """(3, 2) -> 'B3'."""
    letters = ""
    while col > 0:
        col, remainder = divmod(col - 1, _LETTERS)
        letters = chr(ord("A") + remainder) + letters
    return f"{letters}{row}"


def coord(ref: str) -> tuple[int, int]:
    """'B3' -> (3, 2)."""
    letters = "".join(c for c in ref if c.isalpha())
    digits = "".join(c for c in ref if c.isdigit())
    col = 0
    for char in letters.upper():
        col = col * _LETTERS + (ord(char) - ord("A") + 1)
    return int(digits), col


def rng(top: int, left: int, bottom: int, right: int) -> str:
    """A rectangle as a human-checkable A1 range."""
    return f"{a1(top, left)}:{a1(bottom, right)}"


def parse_range(ref: str) -> "Rect":
    """'B3:D8' -> Rect. A single reference is a one-cell rectangle."""
    if ":" in ref:
        start, end = ref.split(":", 1)
        top, left = coord(start)
        bottom, right = coord(end)
    else:
        top, left = coord(ref)
        bottom, right = top, left
    return Rect(top=top, left=left, bottom=bottom, right=right)


@dataclass(frozen=True)
class Rect:
    """A rectangle of grid positions, inclusive on every side."""

    top: int
    left: int
    bottom: int
    right: int

    @property
    def rows(self) -> range:
        return range(self.top, self.bottom + 1)

    @property
    def cols(self) -> range:
        return range(self.left, self.right + 1)

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    def positions(self):
        for row in self.rows:
            for col in self.cols:
                yield row, col

    def contains(self, row: int, col: int) -> bool:
        return self.top <= row <= self.bottom and self.left <= col <= self.right

    def to_a1(self) -> str:
        return rng(self.top, self.left, self.bottom, self.right)


def is_laminar(intervals: list[tuple[int, int]]) -> bool:
    """Do these column intervals nest, or are they merely overlapping?

    A header band is a nesting of spans: each tier refines the one above it, so
    any two spans are either disjoint or one contains the other. Two spans that
    cross — [A,B] against [B,C] — describe no tree at all, and there is nothing
    to read a chain from. This is the whole of the laminar-or-abstain rule.
    """
    for i, (a_start, a_end) in enumerate(intervals):
        for b_start, b_end in intervals[i + 1:]:
            disjoint = a_end < b_start or b_end < a_start
            nested = (a_start <= b_start and b_end <= a_end) or (
                b_start <= a_start and a_end <= b_end
            )
            if not disjoint and not nested:
                return False
    return True
