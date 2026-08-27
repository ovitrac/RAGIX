"""
saqqara.services — the three questions every consumer asks of a tree.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries K3.35-K3.39 of SPEC.md.

*What is this document called? Where in it am I? Where does this phrase occur?*

These belong to no analyzer. They are asked by every consumer, they are answered from the tree
rather than from the file, and each of them is a place where a plausible answer is worse than an
honest refusal.

**The title cascade** runs in a declared order and stops at the first rung that answers. Metadata
outranks content, because a title the source declared is a statement by its author while a title
read off the first heading is an inference by us. When the answer comes from a node, that node is
returned with it: a title without provenance is a claim nobody can check.

**The page policy differs by format, and says which one it used.** A laid-out document has pages
and they are exact; a presentation has slides and a reader counts them from one; a spreadsheet has
sheets. A word-processing document has none of these — it has a flow — so it is cut into windows
of a declared number of words. That last one is an approximation and it is labelled as one. A
service that quietly returned window numbers as though they were pages would produce citations
that look precise and cannot be followed.

**Lookup honours the page restriction under every policy**, including the approximate one. A
restriction that silently did nothing under one format would be worse than no restriction at all.

Every service reports what it skipped. A node with no usable coordinate is counted, not dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Pattern

from .model import Node, Tree

__all__ = [
    "PAGE_WORD_WINDOW",
    "TITLE_RUNGS",
    "LookupHit",
    "PageMap",
    "TitleResult",
    "doc_title",
    "lookup",
    "page_nodes",
]

#: How many words make one window where a format has no pages of its own.
PAGE_WORD_WINDOW = 400

#: The cascade, in order. The first rung that answers wins.
TITLE_RUNGS = ("metadata", "heading", "first-text")

#: Which locator field carries the page-like coordinate, per format, and what to
#: call it. Declared as data so a reader can see the policy without reading code.
_PAGE_POLICY = {
    "pdf": ("page", "exact-page"),
    "pptx": ("slide", "one-based-slide"),
    "xlsx": ("sheet_index", "sheet-index"),
}
_WINDOW_FORMATS = ("docx", "md")


@dataclass
class TitleResult:
    """The document's title, the rung that produced it, and where it came from."""

    title: str | None
    rung: str | None
    node: Node | None = None
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageMap:
    """Nodes grouped by page-like coordinate, and the policy that grouped them."""

    policy: str
    pages: dict[Any, list[Node]]
    skipped: int = 0
    approximate: bool = False

    @property
    def keys(self) -> list:
        return sorted(self.pages)


@dataclass(frozen=True)
class LookupHit:
    """One match, with the page it sits on under the policy in force."""

    node: Node
    page: Any
    text: str


# ------------------------------------------------------------------- title

def doc_title(tree: Tree) -> TitleResult:
    """Walk the rungs in order and stop at the first that answers."""
    trace: dict[str, Any] = {"rungs": []}

    declared = (tree.meta or {}).get("title")
    trace["rungs"].append({"rung": "metadata", "answered": bool(declared)})
    if declared:
        # A title the source declared outranks one we inferred: the first is a
        # statement by the author, the second is a guess by us.
        return TitleResult(title=str(declared), rung="metadata", node=None, trace=trace)

    heading = next(
        (n for n in tree.walk() if n.kind == "heading" and n.text), None
    )
    trace["rungs"].append({"rung": "heading", "answered": heading is not None})
    if heading is not None:
        # Returned WITH its node: a title nobody can trace back is a claim.
        return TitleResult(title=heading.text, rung="heading", node=heading, trace=trace)

    first = next(
        (n for n in tree.walk() if n.kind != "document" and n.text and n.text.strip()), None
    )
    trace["rungs"].append({"rung": "first-text", "answered": first is not None})
    if first is not None:
        return TitleResult(title=first.text, rung="first-text", node=first, trace=trace)

    trace["exhausted"] = True
    return TitleResult(title=None, rung=None, node=None, trace=trace)


# -------------------------------------------------------------------- pages

def page_nodes(tree: Tree) -> PageMap:
    """Group the tree's nodes by the page-like coordinate its format offers."""
    fmt = tree.root.provenance.source_format

    if fmt in _PAGE_POLICY:
        field_name, policy = _PAGE_POLICY[fmt]
        pages: dict[Any, list[Node]] = {}
        skipped = 0
        for node in tree.walk():
            key = getattr(node.provenance.leaf, field_name, None)
            if key is None:
                skipped += 1                 # counted, never dropped
                continue
            pages.setdefault(key, []).append(node)
        return PageMap(policy=policy, pages=pages, skipped=skipped)

    if fmt in _WINDOW_FORMATS:
        return _windows(tree)

    raise ValueError(f"no page policy declares the format {fmt!r}")


def _windows(tree: Tree) -> PageMap:
    """Cut a flow into windows of a declared word count.

    Marked approximate, because it is. A window is not a page, and a citation
    that presented one as the other would look precise and be unfollowable.
    """
    pages: dict[Any, list[Node]] = {}
    skipped = 0
    window = 1
    words = 0

    for node in tree.walk():
        if node.kind == "document":
            skipped += 1
            continue
        pages.setdefault(window, []).append(node)
        words += len((node.text or "").split())
        if words >= PAGE_WORD_WINDOW:
            window += 1
            words = 0

    return PageMap(policy="word-window", pages=pages, skipped=skipped, approximate=True)


# ------------------------------------------------------------------- lookup

def _matcher(pattern, regex: bool, ignore_case: bool):
    """Accept a literal, a list of alternatives, a regular expression, or a compiled one."""
    if isinstance(pattern, re.Pattern):
        return pattern.search, "precompiled"
    flags = re.IGNORECASE if ignore_case else 0
    if isinstance(pattern, (list, tuple, set)):
        joined = "|".join(re.escape(str(p)) for p in pattern)
        return re.compile(joined, flags).search, "alternation"
    if regex:
        return re.compile(str(pattern), flags).search, "regex"
    return re.compile(re.escape(str(pattern)), flags).search, "literal"


def lookup(
    tree: Tree,
    pattern,
    *,
    regex: bool = False,
    ignore_case: bool = False,
    pages: Iterable[Any] | None = None,
) -> tuple[list[LookupHit], dict[str, Any]]:
    """Find a phrase, optionally restricted to given pages under the format's policy."""
    search, kind = _matcher(pattern, regex, ignore_case)
    page_map = page_nodes(tree)
    wanted = set(pages) if pages is not None else None

    page_of: dict[int, Any] = {}
    for key, nodes in page_map.pages.items():
        for node in nodes:
            page_of[id(node)] = key

    hits: list[LookupHit] = []
    out_of_range = 0
    for node in tree.walk():
        if not node.text:
            continue
        key = page_of.get(id(node))
        if wanted is not None and key not in wanted:
            out_of_range += 1
            continue
        if search(node.text):
            hits.append(LookupHit(node=node, page=key, text=node.text))

    trace = {
        "pattern_kind": kind,
        "ignore_case": ignore_case,
        "policy": page_map.policy,
        "approximate_pages": page_map.approximate,
        "restricted_to": sorted(wanted) if wanted is not None else None,
        "excluded_by_page": out_of_range,
        "hits": len(hits),
    }
    return hits, trace
