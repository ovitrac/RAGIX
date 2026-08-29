"""
saqqara.adapters.contract — what a reader is, and what it is allowed to say.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Specified by K2 in SPEC.md.

A reader answers one question — *what does this file say?* — and is forbidden the next one,
*what does it mean?* The separation is the point of this layer. A `Mastaba` is one raw observation:
a cell and the facts the file records about it, a table cell and its merge state, a slide and its
notes. Nothing here decides that a bold numeric cell is a header, or that an empty cell is an
answer slot; those are judgements, they belong to the analyzers, and they are wrong often enough
that the evidence they rest on has to survive them unaltered.

The practical consequence is that a new fact can be added without disturbing a single rule that
does not use it, and a rule can be rewritten without re-reading a single file.

Failure is explicit. `read` raises on a file it cannot claim; `read_paths` catches, counts, and
carries on, so a corpus with one unreadable file yields a report with one refusal in it rather
than a silently shorter list of results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..model import Locator

__all__ = [
    "Adapter",
    "FIGURE_FACTS",
    "FIGURE_SOURCES",
    "PART_SKIPS",
    "GRID_CELL_FACTS",
    "GRID_TABLE_FACTS",
    "Mastaba",
    "OpenVocabulary",
    "ReadReport",
    "Refusal",
    "UnreadableFile",
    "UnsupportedFormat",
    "adapter_for",
    "read_path",
    "read_corpus",
    "read_paths",
    "register_adapter",
    "registered_adapters",
]


#: The vocabulary of a figure, shared by every reader that finds one.
#:
#: A picture is the same object whichever format carried it, so one set is
#: declared here and four readers point at it. `width`/`height` are the stored
#: image's own pixels; `x`/`y`/`w`/`h` are the box it occupied, in points, and a
#: format that does not state one emits it as None rather than omitting it or
#: guessing — a fact present with an unknown value keeps the shared pin
#: checkable, while a fact absent from one reader and present in another would
#: break it. Where the picture came from lives in `source`; WHERE IT WAS FOUND
#: lives in the locator, never here.
FIGURE_FACTS: tuple[str, ...] = (
    "asset", "source", "media_type", "width", "height",
    "x", "y", "w", "h", "colorspace", "bits", "smask",
)

#: How a picture was held. Every value is produced by something: an image stored
#: as its own object in a laid-out document, and an image stored as a part by the
#: office formats. A declared value nothing emits is the defect K2.20 prevents.
FIGURE_SOURCES: tuple[str, ...] = ("xobject", "part")

#: What an office reader may decline, and why. One vocabulary, three readers.
#:
#: A picture is declined only where its bytes cannot be reached at all. A
#: picture whose bytes are readable but whose *header* will not parse is not
#: declined: it is emitted with its dimensions unknown (K6.9), because the
#: evidence is the picture, not the library's opinion of its format.
PART_SKIPS: tuple[str, ...] = ("image-part-empty", "image-part-unreadable")


#: The vocabulary of a grid, shared by every reader that finds one.
#:
#: A table in a word-processing document and a table on a slide are the same
#: object seen through two file formats, so they are described by one set of
#: facts and not by two that happen to agree. Declared once here rather than
#: repeated per reader: repeated, the two copies would be free to drift, and the
#: grid core downstream would be reading a coincidence instead of a contract.
GRID_TABLE_FACTS: tuple[str, ...] = ("n_rows", "n_grid_cols", "ragged", "style")
GRID_CELL_FACTS: tuple[str, ...] = (
    "span", "vmerge", "empty", "fillable", "marker", "bold", "shaded",
)


@dataclass(frozen=True)
class OpenVocabulary:
    """A record kind whose fact names come from the document, not from the reader.

    Front matter is the case that forces this: the keys are whatever the author
    wrote, so declaring the kind as a closed set would pin the fixture rather
    than the reader. Open is itself a declaration — the reader says the names
    are the document's, and lists the ones it reserves for itself — not the
    absence of one.
    """

    reserved: tuple[str, ...] = ()


class UnsupportedFormat(ValueError):
    """No reader claims this file."""


class UnreadableFile(ValueError):
    """A reader claimed the file and could not read it."""


@dataclass(frozen=True)
class Mastaba:
    """One raw observation about a source, with the coordinate it was seen at.

    `kind` is the reader's own vocabulary — what it saw, not what the tree will
    call it. The mapping from observation to node kind happens later, when there
    is enough context to make it, and is recorded there rather than assumed here.
    """

    kind: str
    locator: Locator
    text: str | None = None
    facts: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "locator": self.locator.to_dict()}
        if self.text is not None:
            out["text"] = self.text
        if self.facts:
            out["facts"] = dict(self.facts)
        return out


@dataclass(frozen=True)
class Refusal:
    """A file that was not read, and why. Counted, never silent."""

    path: str
    reason: str
    detail: str = ""


@dataclass
class ReadReport:
    """What happened across a set of files, including what did not happen."""

    read: list[str] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "read": len(self.read),
            "refused": len(self.refusals),
            "duplicate": len(self.duplicates),
        }


class Adapter:
    """A reader for one family of files.

    Subclasses declare `format`, `version`, `extensions` and `fact_sets` — the
    exact facts they emit, *per record kind*. Declared rather than inferred, so
    that a change to it is a visible edit next to a version number, which is
    what makes K2.5 checkable at all.

    Per kind, and not one flat set for the reader, because a reader speaks more
    than one vocabulary: a cell and a paragraph are not described by the same
    facts, and a single set either merges them — losing which fact belongs to
    which — or, as it did here, declares one kind and leaves the others
    unguarded. A kind emitted with no declaration is a failure, not a default.
    """

    format: str = ""
    version: str = "0.0.0"
    extensions: tuple[str, ...] = ()
    fact_sets: Mapping[str, tuple[str, ...] | OpenVocabulary] = {}

    #: The reasons this reader may decline something. Closed, per reader, and
    #: checked at the point of counting so the vocabulary cannot drift.
    skip_reasons: tuple[str, ...] = ()

    def __init__(self) -> None:
        #: Where extracted bytes go. A reader with no store reads no objects: it
        #: has nowhere to put them, and putting them in the tree is what K6.2
        #: forbids. So the objects layer is opt-in at the call site.
        self.store = None
        #: What this reader declined, by name. Counted, never silent.
        self.skips: dict[str, int] = {}

    def _skip(self, reason: str) -> None:
        """Count something this reader declined. Named, and never silent."""
        if reason not in self.skip_reasons:
            raise ValueError(f"{self.format}: undeclared skip reason {reason!r}")
        self.skips[reason] = self.skips.get(reason, 0) + 1

    def read(self, path: Path) -> Iterator[Mastaba]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Adapter {self.format}@{self.version}>"


#: extension -> reader.
_ADAPTERS: dict[str, Adapter] = {}


def register_adapter(adapter: Adapter) -> Adapter:
    """Claim a set of extensions. Refuses to take one already claimed."""
    if not adapter.format or not adapter.extensions:
        raise ValueError(f"{adapter!r} declares no format or no extension")
    for ext in adapter.extensions:
        held = _ADAPTERS.get(ext)
        if held is not None and held.format != adapter.format:
            raise ValueError(f"{ext!r} is already claimed by {held.format!r}")
        _ADAPTERS[ext] = adapter
    return adapter


def registered_adapters() -> dict[str, Adapter]:
    return dict(_ADAPTERS)


def adapter_for(path: Path) -> Adapter | None:
    return _ADAPTERS.get(path.suffix.lower())


def read_path(path: Path, store=None) -> list[Mastaba]:
    """Read one file. Raises rather than returning an empty result.

    `store` is where extracted bytes go. A reader given none reads no objects:
    it has nowhere to put them, and putting them in the tree is exactly what
    K6.2 forbids. So the objects layer is opt-in at the call site, and a caller
    that wants only structure pays nothing for pictures it will not read.
    """
    adapter = adapter_for(path)
    if adapter is None:
        raise UnsupportedFormat(f"no reader claims {path.suffix!r}: {path.name}")
    if not path.is_file():
        raise UnreadableFile(f"not a file: {path}")
    if store is not None and hasattr(adapter, "store"):
        adapter.store = store
    try:
        return list(adapter.read(path))
    except (UnsupportedFormat, UnreadableFile):
        raise
    except Exception as exc:                      # the reader claimed it and failed
        raise UnreadableFile(f"{adapter.format} reader failed on {path.name}: {exc}") from exc


def read_corpus(paths: Iterable[Path]) -> tuple[dict[str, list[Mastaba]], ReadReport]:
    """Read a set of files, keeping each observation with the file it came from.

    A Mastaba carries a position inside a document, not the identity of the
    document — that belongs to the tree built from it. So reading many files at
    once has to keep the association here, or the caller is left holding a pile
    of observations with no way to tell whose they are.
    """
    report = ReadReport()
    seen: dict[str, str] = {}
    by_path: dict[str, list[Mastaba]] = {}

    for path in paths:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            report.refusals.append(Refusal(str(path), "unreadable-file", str(exc)))
            continue

        first = seen.get(digest)
        if first is not None:
            report.duplicates.append(str(path))
            continue

        try:
            found = read_path(path)
        except UnsupportedFormat as exc:
            report.refusals.append(Refusal(str(path), "unsupported-format", str(exc)))
            continue
        except UnreadableFile as exc:
            report.refusals.append(Refusal(str(path), "unreadable-file", str(exc)))
            continue

        seen[digest] = str(path)
        report.read.append(str(path))
        by_path[str(path)] = found

    return by_path, report


def read_paths(paths: Iterable[Path]) -> tuple[list[Mastaba], ReadReport]:
    """Read a set of files, counting what was refused and what was a duplicate.

    Duplicates are decided by the bytes, not by the name: two identical files
    are one document seen twice, and counting them twice would let a copy pass
    for corroboration.
    """
    by_path, report = read_corpus(paths)
    return [fact for facts in by_path.values() for fact in facts], report
