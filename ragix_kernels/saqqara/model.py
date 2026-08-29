"""
saqqara.model — the tree: nodes, kinds, locators, provenance, stable JSON.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Specified by K1 in SPEC.md (K1.1-K1.8). Nothing here exists that no proposition asks for.

Four ideas, and the reason each is shaped the way it is:

`Locator` — one class per format, because a citation is only meaningful in the coordinate system
of the thing cited: a page and a box for a laid-out document, a sheet and a cell for a
spreadsheet, a paragraph and a run for a word processor. Locators of one format order among
themselves; comparing across formats raises rather than inventing an order that means nothing.

`Provenance` — an ordered chain of locators, from the broadest coordinate to the narrowest, plus
the kernel that produced the node and its version. It is required at construction (K1.1). Making
a citation an argument rather than an attribute is the whole point: a node without provenance
cannot be built, so there is no window in which one exists and no later pass that can forget to
fill it in.

`KindRegistry` — the standard kinds are declared, not wired in. An adapter that needs a kind
nobody anticipated registers it instead of patching the core, and an unregistered kind is refused
at construction (K1.3). This is what makes the vocabulary closed without making it fixed.

`Tree` — canonical JSON in both directions (K1.2). Keys are sorted, separators fixed, text left as
text: a serialisation that varies between two runs cannot support a hash, and a hash is what makes
a citation checkable.

The P1 interface has been reopened twice, deliberately, both on 2026-08-27, and both times to make
a coordinate able to name a place it could already be asked about. First `PptxLocator` gained `notes`.
A slide and its speaker notes shared one coordinate, so a locator was not a unique address inside
its own format — a defect in the central promise of this package, costing one field to fix before
publication and a breaking change to fix after it (K1.9). Then it gained `row` and `col`, so that a
cell inside a table on a slide can be cited rather than only the table containing it.

Every node also declares whether it was **read** or **inferred** (K1.4). A node read from a file
carries full confidence; an inferred node must carry less. The model refuses both contradictions,
so a kernel cannot present a guess with the authority of a reading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Iterator, Optional

__all__ = [
    "CANONICAL_JSON",
    "DocumentLocator",
    "DocxLocator",
    "KindError",
    "Locator",
    "LocatorError",
    "MdLocator",
    "Node",
    "PdfLocator",
    "PptxLocator",
    "Provenance",
    "ProvenanceError",
    "STANDARD_KINDS",
    "Tree",
    "XlsxLocator",
    "kind_registry",
    "locator_from_dict",
    "register_locator",
]


# --------------------------------------------------------------------- errors

class KindError(ValueError):
    """A node kind that the registry does not know."""


class ProvenanceError(ValueError):
    """A node built without a usable citation."""


class LocatorError(ValueError):
    """A locator that cannot be rebuilt or compared."""


# ------------------------------------------------------------------- locators

@dataclass(frozen=True)
class Locator:
    """A position inside a source, in that source's own coordinate system.

    Subclasses declare `format` and implement `key()`. Two locators of the same
    format compare by their keys; two of different formats do not compare at
    all — an order between a page and a spreadsheet cell would be arbitrary,
    and an arbitrary order is worse than a refusal.
    """

    format: ClassVar[str] = ""

    def key(self) -> tuple:
        """Ordering key within this format. Sortable, hence never None."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        out = {"format": self.format}
        for name, value in sorted(vars(self).items()):
            if value is not None:
                out[name] = list(value) if isinstance(value, tuple) else value
        return out

    def __lt__(self, other: "Locator") -> bool:
        if not isinstance(other, Locator):
            return NotImplemented
        if other.format != self.format:
            raise LocatorError(
                f"cannot order a {self.format!r} locator against a {other.format!r} one"
            )
        return self.key() < other.key()


#: format name -> locator class, for rebuilding from JSON.
_LOCATORS: dict[str, type[Locator]] = {}


def register_locator(cls: type[Locator]) -> type[Locator]:
    """Declare a locator class for its format. Refuses to redefine one."""
    if not cls.format:
        raise LocatorError(f"{cls.__name__} declares no format")
    known = _LOCATORS.get(cls.format)
    if known is not None and known is not cls:
        raise LocatorError(f"format {cls.format!r} is already served by {known.__name__}")
    _LOCATORS[cls.format] = cls
    return cls


def locator_from_dict(data: dict[str, Any]) -> Locator:
    """Rebuild a locator. An unknown format is refused, never approximated."""
    fmt = data.get("format")
    cls = _LOCATORS.get(fmt)
    if cls is None:
        raise LocatorError(f"unknown locator format: {fmt!r}")
    fields = {k: v for k, v in data.items() if k != "format"}
    if "bbox" in fields and isinstance(fields["bbox"], list):
        fields["bbox"] = tuple(fields["bbox"])
    return cls(**fields)


@register_locator
@dataclass(frozen=True)
class DocumentLocator(Locator):
    """The document as a whole — the coordinate a tree root cites."""

    format: ClassVar[str] = "document"

    def key(self) -> tuple:
        return ()


@register_locator
@dataclass(frozen=True)
class PdfLocator(Locator):
    """A page, optionally a box on it (left, top, right, bottom)."""

    format: ClassVar[str] = "pdf"
    page: int = 0
    bbox: Optional[tuple[float, float, float, float]] = None
    #: The resource name an image was drawn under. Addressing belongs to the
    #: locator; the facts describe the object, not where it was found.
    xobject: Optional[str] = None

    def key(self) -> tuple:
        return (self.page, self.bbox or (0.0, 0.0, 0.0, 0.0), self.xobject or "")


@register_locator
@dataclass(frozen=True)
class XlsxLocator(Locator):
    """A sheet, optionally a cell on it, optionally the merged extent it anchors."""

    format: ClassVar[str] = "xlsx"
    sheet: str = ""
    sheet_index: int = 0
    cell: Optional[str] = None
    row: Optional[int] = None
    col: Optional[int] = None
    merged_range: Optional[str] = None
    #: Where a picture is anchored. Addressing belongs to the locator.
    anchor: Optional[str] = None

    def key(self) -> tuple:
        return (self.sheet_index, self.row or 0, self.col or 0)


@register_locator
@dataclass(frozen=True)
class DocxLocator(Locator):
    """A position in a word-processing flow.

    `flow` names the stream — the body, a page header, a nested table — because
    an index only means something within its own stream (K2.10).
    """

    format: ClassVar[str] = "docx"
    flow: str = "body"
    paragraph: Optional[int] = None
    run: Optional[int] = None
    #: The relationship a picture part was reached through.
    relationship: Optional[str] = None
    table_index: Optional[int] = None
    row: Optional[int] = None
    col: Optional[int] = None

    def key(self) -> tuple:
        return (
            self.flow,
            self.paragraph if self.paragraph is not None else -1,
            self.table_index if self.table_index is not None else -1,
            self.row or 0,
            self.col or 0,
            self.run or 0,
        )


@register_locator
@dataclass(frozen=True)
class PptxLocator(Locator):
    """A slide, one-based as a reader counts them; a shape on it, or its notes.

    `notes` exists because a slide and the speaker notes attached to it are two
    different places, and a coordinate that cannot tell them apart is not an
    address. Ordering puts the notes after the slide's shapes: they are read
    last because they are said last.

    `row` and `col` address a cell inside a table on a slide. Without them a
    table shape could be cited but nothing inside it could, which is the same
    defect `notes` was added to fix, one level further down.
    """

    format: ClassVar[str] = "pptx"
    slide: int = 1
    shape: Optional[int] = None
    notes: bool = False
    #: The shape's own identifier, which the format states and the index does not.
    shape_id: Optional[int] = None
    row: Optional[int] = None
    col: Optional[int] = None

    def key(self) -> tuple:
        return (self.slide, 1 if self.notes else 0,
                self.shape if self.shape is not None else -1,
                self.row if self.row is not None else -1,
                self.col if self.col is not None else -1)


@register_locator
@dataclass(frozen=True)
class MdLocator(Locator):
    """A line, one-based."""

    format: ClassVar[str] = "md"
    line: int = 1

    def key(self) -> tuple:
        return (self.line,)


# ----------------------------------------------------------------- provenance

@dataclass(frozen=True)
class Provenance:
    """Where a node came from, and what produced it.

    `chain` runs broadest first: sheet then cell, page then box. It must not be
    empty — a node whose citation is an empty list cites nothing.
    """

    source_path: str
    source_format: str
    chain: tuple[Locator, ...]
    kernel: str
    kernel_version: str
    source_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        missing = [
            name
            for name in ("source_path", "source_format", "kernel", "kernel_version")
            if not getattr(self, name)
        ]
        if missing:
            raise ProvenanceError(f"provenance is missing: {', '.join(missing)}")
        if not self.chain:
            raise ProvenanceError("provenance carries an empty locator chain")
        for loc in self.chain:
            if not isinstance(loc, Locator):
                raise ProvenanceError(f"not a locator: {loc!r}")

    @property
    def leaf(self) -> Locator:
        """The narrowest coordinate — what a citation points a reader at."""
        return self.chain[-1]

    def to_dict(self) -> dict[str, Any]:
        out = {
            "source_path": self.source_path,
            "source_format": self.source_format,
            "chain": [loc.to_dict() for loc in self.chain],
            "kernel": self.kernel,
            "kernel_version": self.kernel_version,
        }
        if self.source_sha256:
            out["source_sha256"] = self.source_sha256
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Provenance":
        return cls(
            source_path=data["source_path"],
            source_format=data["source_format"],
            chain=tuple(locator_from_dict(d) for d in data["chain"]),
            kernel=data["kernel"],
            kernel_version=data["kernel_version"],
            source_sha256=data.get("source_sha256"),
        )


# ----------------------------------------------------------------------- kinds

STANDARD_KINDS = (
    "document",
    "section",
    "heading",
    "paragraph",
    "list",
    "list_item",
    "table",
    "figure",
    "caption",
)


class KindRegistry:
    """The kinds a node may declare.

    Closed but not fixed: the standard kinds are declared here, and an adapter
    that genuinely needs another registers it rather than editing this file. An
    unregistered kind is refused at construction, so the vocabulary of a tree is
    always something someone decided on.
    """

    def __init__(self, kinds: tuple[str, ...] = STANDARD_KINDS) -> None:
        self._kinds = set(kinds)

    def register(self, kind: str) -> str:
        if not kind or not isinstance(kind, str):
            raise KindError(f"not a usable kind: {kind!r}")
        self._kinds.add(kind)
        return kind

    def check(self, kind: str) -> str:
        if kind not in self._kinds:
            raise KindError(
                f"unregistered kind: {kind!r}. Register it deliberately "
                "(kind_registry.register) rather than widening the core."
            )
        return kind

    def known(self) -> tuple[str, ...]:
        return tuple(sorted(self._kinds))

    def __contains__(self, kind: object) -> bool:
        return kind in self._kinds


#: The process-wide registry. Adapters register their kinds at import time.
kind_registry = KindRegistry()


# ----------------------------------------------------------------------- nodes

@dataclass
class Node:
    """One element of a document, with its citation.

    `facts` holds what the reader observed and did not interpret — a data type,
    a boldness, a number format. `span` is the extent the node covers in its
    source's own terms; it stays an opaque serialisable value until an adapter
    needs it typed (P2), because inventing a geometry no proposition asks for
    would be a shape decided by a guess.
    """

    kind: str
    provenance: Provenance
    text: Optional[str] = None
    level: Optional[int] = None
    span: Optional[Any] = None
    facts: dict[str, Any] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    origin: str = "read"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        kind_registry.check(self.kind)
        if not isinstance(self.provenance, Provenance):
            raise ProvenanceError(f"{self.kind!r} node built without provenance")
        if self.origin not in ("read", "inferred"):
            raise ValueError(f"origin must be 'read' or 'inferred', not {self.origin!r}")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(f"confidence must lie in (0, 1], not {self.confidence!r}")
        if self.origin == "inferred" and self.confidence == 1.0:
            raise ValueError(
                "an inferred node may not claim full confidence: it was not read, "
                "and presenting a guess with the authority of a reading is the "
                "failure this check exists to prevent"
            )
        if self.origin == "read" and self.confidence < 1.0:
            raise ValueError(
                "a node read from the file carries full confidence; lower it only "
                "by declaring origin='inferred'"
            )

    def walk(self) -> Iterator["Node"]:
        """This node, then its descendants, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "provenance": self.provenance.to_dict(),
            "origin": self.origin,
            "confidence": self.confidence,
        }
        if self.text is not None:
            out["text"] = self.text
        if self.level is not None:
            out["level"] = self.level
        if self.span is not None:
            out["span"] = self.span
        if self.facts:
            out["facts"] = self.facts
        if self.children:
            out["children"] = [c.to_dict() for c in self.children]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        return cls(
            kind=data["kind"],
            provenance=Provenance.from_dict(data["provenance"]),
            text=data.get("text"),
            level=data.get("level"),
            span=data.get("span"),
            facts=data.get("facts", {}),
            children=[cls.from_dict(c) for c in data.get("children", [])],
            origin=data.get("origin", "read"),
            confidence=data.get("confidence", 1.0),
        )


# ------------------------------------------------------------------------ tree

#: Canonical JSON settings. Fixed so that two runs produce the same bytes.
CANONICAL_JSON = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": False}


@dataclass
class Tree:
    """A document: its root node and whatever metadata the source declared."""

    root: Node
    meta: dict[str, Any] = field(default_factory=dict)

    def walk(self) -> Iterator[Node]:
        return self.root.walk()

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tree":
        return cls(root=Node.from_dict(data["root"]), meta=data.get("meta", {}))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), **CANONICAL_JSON)

    @classmethod
    def from_json(cls, text: str) -> "Tree":
        return cls.from_dict(json.loads(text))

    def replace_source_path(self, path: str) -> "Tree":
        """The same tree read from another path — used to prove path-invariance."""

        def rewrite(node: Node) -> Node:
            return Node(
                kind=node.kind,
                provenance=replace(node.provenance, source_path=path),
                text=node.text,
                level=node.level,
                span=node.span,
                facts=dict(node.facts),
                children=[rewrite(c) for c in node.children],
                origin=node.origin,
                confidence=node.confidence,
            )

        return Tree(root=rewrite(self.root), meta=dict(self.meta))
