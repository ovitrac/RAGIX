"""ragix_kernels.harvest.fr.cut — values the store's text layer cut in two: one definition, three readers.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

A register of a scanned contract found this upstream of every model: the store's leaves can fall
inside a digit run, so a grammar matching from a word boundary reads « 0,00 € » where the page prints
250,00 €, « 8 heures » for 48 heures, « 000 heures » for 1 000 heures. The span is byte-exact, its
hash verifies, and the value is wrong — the one failure this system exists to prevent, and the one a
span check cannot see. Such a value is `cut`: never quoted, always listed, always counted.

Rejoining is not the fix: "150\\n50" is two cells of a table, not 15050. The fix is upstream in the
leaf join, and it is not taken here.

This module is deliberately small and has no dependency: the harvest offers values to a model, the
brief quotes claims, and the checker verifies a brief — three readers who must not drift apart on
what "cut" means. The version string (``tender.cut 1.1``) is kept as first published.
"""
from __future__ import annotations

import re

VERSION = "tender.cut 1.1"


def is_cut(text: str, start: int) -> bool:
    """Does the span at `start` begin on the far side of a digit run the text layer split?

    True when the characters before it are, in order: a digit, optional spaces or tabs, a newline,
    optional spaces or tabs. A space alone is not a leaf boundary, and a newline after a letter or a
    full stop is ordinary prose.
    """
    if start <= 0 or start >= len(text):
        return False
    # the tail of a split number starts with a digit. Without this guard the rule flagged about a quarter
    # of the values an early harvest offered — every one letter-initial, "Article I", "Page 4 sur 8"
    # after a line ending in a page number — and none with it.
    if not text[start:].lstrip(" \t")[:1].isdigit():
        return False
    head = text[:start].rstrip(" \t")
    if not head.endswith("\n"):
        return False
    before = head.rstrip(" \t\n\r")
    return bool(before) and before[-1].isdigit()


def cut_spans(text: str, spans) -> set[int]:
    """The indices of those spans that are cut. `spans` is any iterable of objects carrying `start`."""
    return {i for i, span in enumerate(spans) if is_cut(text, getattr(span, "start", span))}


#: the opposite failure of a cut. The grammar's number pattern matches `\s`, so it JOINS a digit run
#: across a line break — "150\n50 €" reads 15050.00 EUR, and a scanned "18" can be `1\n8` in the store. A join is sometimes right (OCR splitting one printed
#: token) and sometimes not (two table cells), and the store cannot say which, so a joined critical
#: value is needs_review until a human confirms it against the page.
JOINED = re.compile(r"\d+(?:[ \t]*\n[ \t]*\d+)+")


def joined_runs(raw: str) -> list[str]:
    """The digit runs this raw span joins across line breaks, quoted exactly as the store has them."""
    return JOINED.findall(raw or "")
