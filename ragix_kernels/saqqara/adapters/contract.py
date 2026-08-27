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
    "Mastaba",
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

    Subclasses declare `format`, `version`, `extensions` and `fact_set` — the
    exact facts they emit. `fact_set` is declared rather than inferred so that a
    change to it is a visible edit next to a version number, which is what makes
    K2.5 checkable at all.
    """

    format: str = ""
    version: str = "0.0.0"
    extensions: tuple[str, ...] = ()
    fact_set: tuple[str, ...] = ()

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


def read_path(path: Path) -> list[Mastaba]:
    """Read one file. Raises rather than returning an empty result."""
    adapter = adapter_for(path)
    if adapter is None:
        raise UnsupportedFormat(f"no reader claims {path.suffix!r}: {path.name}")
    if not path.is_file():
        raise UnreadableFile(f"not a file: {path}")
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
