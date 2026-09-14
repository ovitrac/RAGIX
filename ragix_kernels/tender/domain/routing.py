"""tender.routing — the routing class of a document, read off the kernel's facts.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

`DocumentFacts.routing_class` is the field the retrieval policy filters on, and
until this module existed nothing wrote it: the lane's tests set it by hand.
This computes it.

**Re-specification, not a port.** The classes and their meanings are the
DIGESTION_STRATEGY §3 decision matrix, unchanged:

    P1  a laid-out document that DECLARES an outline      L2 reads the bookmarks
    P2  a laid-out document that declares none            L2 reconstructs it
    P3  a laid-out document with no text layer at all     OCR pending, fail closed
    D1  a word-processing document with real heading styles
    D2  a word-processing document that is structurally flat
    X1  a presentation      (slide-structured, no document outline)
    M1  markdown            (parsed directly)
    E1  a spreadsheet       (sheet-structured)

What changed is the *address* of every signal, and two of the three addresses
changed CHANNEL, not merely name (`NOTE_K5A_INTERPRETATION_20260830.md` §1.3):

  - **a declared heading is a style, not a node kind.** A styled word-processing
    document yields `kind="heading"` for nothing at all; the declaration is read
    into `facts["style"]`. A classifier keyed on the kind would call every such
    document flat — D2 — and report it as a fact about the corpus;
  - **a declared outline IS a node kind** — but only for the laid-out format,
    where the reader promotes each bookmark into a heading node. A classifier
    keyed on `facts["style"]` would call every outlined document P2.

The two branches therefore read two different channels on purpose. Reading one
channel for both formats is the mistake this module is shaped to refuse, and
`tests/test_tender_routing.py` asserts each direction.

**`origin == "read"` is load-bearing on the P1 branch.** The kernel's promoting
analyzers add heading nodes inferred from size and weight contrast — measured:
a flat three-page document read with `promote=True` gains three of them. They
carry `origin="inferred"` (a two-value closed vocabulary the model enforces),
so counting only read nodes is what keeps a promoted tree from being classified
P1. Without that word this module would answer differently depending on how its
caller happened to read the file.

**Thresholds and predicates are held, not revisited** — they are the scientific
lead's (CLAUDE.md §8.2, RED):

  - `D1_MIN_HEADINGS = 3`, the value of `RouterCfg.d1_min_headings`. Several
    corpus documents carry a single stray heading style while being flat;
  - the heading-style pattern is the one both substrates use, unchanged. The old
    intake matched a style *ID* (`Heading1`), the kernel reports a style *NAME*
    (`Heading 1`); the `\\s*` already in the pattern reads both;
  - P3 is "**no** page has a text layer", which is the old `total_chars == 0`
    over the whole document. The kernel reports `has_text` per page, so a
    partially scanned document is P1/P2 with its blank pages counted as
    evidence — the same answer the old whole-document test gave, now with the
    number visible instead of collapsed.

**The evidence does not go into `DocumentFacts.quality`.** That field is the
quality axis (`debris_score` et al.), K8 territory and declared out of phase
upstream; filling it with this module's counts would occupy a signed field with
a different referent. The counts travel on the `Routing` object, for the caller
to trace or log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .records import DocumentFacts

__all__ = [
    "D1_MIN_HEADINGS",
    "HEADING_STYLE",
    "Routing",
    "classify",
    "heading_style_level",
]

#: EN 'Heading 2' / 'Heading2', DE 'Überschrift 2'. Identical to the pattern the
#: old intake matched style IDs with; `\s*` is why one pattern reads both the ID
#: and the kernel's style name.
HEADING_STYLE = re.compile(r"(?:heading|berschrift)\s*(\d)?", re.IGNORECASE)

#: Styled headings a word-processing document needs before it is D1 rather than
#: D2. Lead's threshold (`RouterCfg.d1_min_headings`), held at its value.
D1_MIN_HEADINGS = 3

#: Format -> the class it decides on its own. The laid-out and word-processing
#: formats are absent because their class is not decided by format alone.
BY_FORMAT = {"xlsx": "E1", "pptx": "X1", "md": "M1"}


@dataclass
class Routing:
    """One document's routing class, the branch that decided it, and the numbers.

    `rule` names the branch so a verdict can be argued with rather than trusted,
    the way `doc_filter` names R1-R4.
    """

    routing_class: str
    rule: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def facts(self, *, document_date: Optional[str] = None) -> DocumentFacts:
        """The signed document facts this routing contributes to.

        `quality` stays empty on purpose (see the module docstring), and
        `document_date` is the caller's to supply: it is a different extraction
        and absence of one is `undated`, never timeless.
        """
        return DocumentFacts(routing_class=self.routing_class,
                             document_date=document_date)


def heading_style_level(node: Any) -> Optional[int]:
    """The level a node's DECLARED heading style implies, else None.

    This is what replaces `kind == "heading"` for word-processing documents. A
    style with no digit (`Heading`, `Titre`) is level 1, which is what the style
    means; a node with no style, or one that is not a heading style, is None.
    """
    style = (getattr(node, "facts", None) or {}).get("style") or ""
    match = HEADING_STYLE.match(style)
    if not match:
        return None
    return int(match.group(1)) if match.group(1) else 1


def classify(tree: Any) -> Routing:
    """Classify one analyzed saqqara tree into its routing class.

    Args:
        tree: a `Tree` as `tender.substrate.read_tree` returns it. The format is
            taken from the root's provenance — the adapter that actually read the
            file — not from the filename, which is a claim about the file rather
            than a reading of it.

    Returns:
        A `Routing` carrying the class, the branch that decided it, and the
        counts that branch weighed.

    Raises:
        ValueError: on a format this matrix does not classify. Fail closed: an
            unclassified document must not acquire a class by default, because
            the retrieval policy filters on this field.
    """
    fmt = tree.root.provenance.source_format

    if fmt in BY_FORMAT:
        return Routing(BY_FORMAT[fmt], rule=f"format:{fmt}")

    if fmt == "docx":
        styled = sum(1 for node in _walk(tree.root)
                     if heading_style_level(node) is not None)
        evidence = {"heading_style_paras": styled, "threshold": D1_MIN_HEADINGS}
        if styled >= D1_MIN_HEADINGS:
            return Routing("D1", rule="docx:declared-styles", evidence=evidence)
        return Routing("D2", rule="docx:flat", evidence=evidence)

    if fmt == "pdf":
        pages = [n for n in _walk(tree.root) if n.kind == "page"]
        with_text = sum(1 for n in pages if (n.facts or {}).get("has_text"))
        # `origin == "read"` and not merely `kind == "heading"`: the promoting
        # analyzers infer headings from size and weight contrast, and an inferred
        # heading is not a declaration by the document.
        declared = sum(1 for n in _walk(tree.root)
                       if n.kind == "heading" and n.origin == "read")
        evidence = {"outline_entries": declared,
                    "pages": len(pages), "pages_with_text": with_text}
        # `pages and` is the old `doc.page_count > 0`: a document with no page
        # at all has not been shown to need OCR, it has been shown to be empty.
        # P3 is tested before the outline for the old order's reason — a scanned
        # document that still carries bookmarks is unreadable, not outlined.
        if pages and with_text == 0:
            return Routing("P3", rule="pdf:no-text-layer", evidence=evidence)
        if declared:
            return Routing("P1", rule="pdf:declared-outline", evidence=evidence)
        return Routing("P2", rule="pdf:no-outline", evidence=evidence)

    raise ValueError(f"no routing class for format {fmt!r}")   # fail closed


def _walk(node: Any) -> Iterator[Any]:
    yield node
    for child in node.children:
        yield from _walk(child)
