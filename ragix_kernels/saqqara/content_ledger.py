"""Literal source-character coverage over canonical members and context groups.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

An association to a source span is not proof that its characters survived.
Mappings are checked against both the current carrier and the original span.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..harvest.region_types import RegionRefused
from ..harvest.regions import _inside
from .census import page_lines
from .context_groups import _members, _observed, _refused_candidates, _source, cell_inventory

if TYPE_CHECKING:
    from ..harvest.regions import RegionIndex
    from .context_groups import ContextGroup
    from .explorer import ExplorerResult

STATUSES = ("CARRIED", "CARRIED_EXACT", "PARTIAL", "NOT_CARRIED", "EXCLUDED")


@dataclass(frozen=True)
class TextCarrier:
    kind: str
    carrier_id: str
    how: str

    def __post_init__(self):
        if (
            self.kind not in ("MEMBER", "CELL")
            or not self.carrier_id
            or self.how not in ("MAPPED", "EXACT")
        ):
            raise RegionRefused("INVALID_TEXT_CARRIER")
        if self.kind == "CELL" and self.how != "EXACT":
            raise RegionRefused("INVALID_TEXT_CARRIER")


@dataclass(frozen=True)
class TextEntry:
    span_id: str
    page: int
    span_text: str
    span_length: int
    status: str
    carriers: tuple[TextCarrier, ...]
    missing: tuple[tuple[int, int], ...] = ()
    rule: str | None = None

    def __post_init__(self):
        if (
            not self.span_id
            or type(self.page) is not int
            or self.page < 1
            or not isinstance(self.span_text, str)
            or type(self.span_length) is not int
            or self.span_length != len(self.span_text)
            or self.status not in STATUSES
            or not isinstance(self.carriers, tuple)
            or any(not isinstance(c, TextCarrier) for c in self.carriers)
        ):
            raise RegionRefused("INVALID_TEXT_ENTRY")
        if bool(self.rule) != (self.status == "EXCLUDED"):
            raise RegionRefused("INVALID_TEXT_EXCLUSION")
        if bool(self.missing) != (self.status == "PARTIAL"):
            raise RegionRefused("INVALID_TEXT_COVERAGE")
        previous = -1
        for start, end in self.missing:
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= self.span_length
                or start <= previous
            ):
                raise RegionRefused("INVALID_TEXT_COVERAGE")
            previous = end
        if self.status in ("CARRIED", "CARRIED_EXACT", "PARTIAL") and not self.carriers:
            raise RegionRefused("TEXT_CARRIER_REQUIRED")


@dataclass(frozen=True)
class TextLedger:
    source_id: str
    entries: tuple[TextEntry, ...]
    rule: str = "text-occurrence-ledger/1"

    def __post_init__(self):
        if (
            not self.source_id
            or self.rule != "text-occurrence-ledger/1"
            or not isinstance(self.entries, tuple)
            or any(not isinstance(e, TextEntry) for e in self.entries)
            or len({e.span_id for e in self.entries}) != len(self.entries)
        ):
            raise RegionRefused("INVALID_TEXT_LEDGER")

    @property
    def passes(self) -> bool:
        return not any(entry.status in ("PARTIAL", "NOT_CARRIED") for entry in self.entries)

    def summary(self) -> dict:
        pages = {}
        for page in sorted({entry.page for entry in self.entries}):
            counts = Counter(entry.status for entry in self.entries if entry.page == page)
            pages[page] = {status: counts[status] for status in STATUSES}
        counts = Counter(entry.status for entry in self.entries)
        return {
            "rule": self.rule,
            "source_id": self.source_id,
            "passes": self.passes,
            "counts": {status: counts[status] for status in STATUSES},
            "pages": pages,
        }


def _missing(length, offsets):
    ranges, start = [], None
    for position in range(length + 1):
        absent = position < length and position not in offsets
        if absent and start is None:
            start = position
        elif not absent and start is not None:
            ranges.append((start, position))
            start = None
    return tuple(ranges)


def text_ledger(
    result: ExplorerResult,
    index: RegionIndex,
    groups: tuple[ContextGroup, ...],
    *,
    source_id: str,
) -> TextLedger:
    """Measure literal retention; expose gaps rather than reconstruct missing text.

    Furniture exclusions reuse the reader's line classification. A span partly
    excluded and partly missing is not excused as an entirely excluded span.
    A cell counts exactly only within observed geometry and on the same page.
    """
    groups = tuple(groups)
    pages, tables, native = _source(result, source_id, index)
    members = _members(index)
    refused = _refused_candidates(index, tables, native)
    if {g.candidate_id for g in groups} != set(refused):
        raise RegionRefused("CONTEXT_REFUSAL_NOT_GROUPED")
    for group in groups:
        if any(mid not in members for mid in group.member_ids):
            raise RegionRefused("CONTEXT_MEMBER_SOURCE_MISSING")
    inventory = cell_inventory(result, groups, source_id=source_id)
    spans = {s.span_id: s for page in pages.values() for s in page.spans}
    views = {v.view_id: v for page in pages.values() for v in page_lines(page)}
    furniture = set(result.reading.furniture)
    if not furniture <= views.keys():
        raise RegionRefused("FURNITURE_SOURCE_LINE_MISSING")
    coverage = {sid: set() for sid in spans}
    excluded = {sid: set() for sid in spans}
    nonfurniture = {sid: set() for sid in spans}
    carriers = {sid: set() for sid in spans}
    for vid, view in views.items():
        if len(view.text) != len(view.mapping):
            raise RegionRefused("INVALID_CONTEXT_CHARACTER_MAPPING")
        member = members.get(vid)
        if member is not None and (member.text != view.text or member.page != view.page):
            raise RegionRefused("STALE_CONTEXT_MEMBER_TEXT")
        if member is not None and (
            member.bbox != view.bbox
            or set(member.source_spans) != {r.span_id for r in view.mapping if r is not None}
        ):
            raise RegionRefused("STALE_CONTEXT_MEMBER_GEOMETRY_OR_REFS")
        covered_here = set()
        for i, ref in enumerate(view.mapping):
            if ref is None:
                continue  # Inserted whitespace is not a source character.
            span = spans.get(ref.span_id)
            if span is None or span.page != view.page:
                raise RegionRefused("CONTEXT_MAPPING_SOURCE_MISSING")
            offset = ref.offset - span.source_offset
            if (
                type(offset) is not int
                or not 0 <= offset < len(span.text)
                or view.text[i] != span.text[offset]
            ):
                raise RegionRefused("CONTEXT_MAPPING_CHARACTER_MISMATCH")
            (excluded if vid in furniture else nonfurniture)[span.span_id].add(offset)
            if member is not None:
                coverage[span.span_id].add(offset)
                covered_here.add(span.span_id)
        for sid in covered_here:
            carriers[sid].add(("MEMBER", vid, "MAPPED"))
    # Existing accepted CELL members and new literal cell observations both count.
    # A native cell with missing geometry must never borrow its container's box.
    exact = []
    for mid, member in members.items():
        if mid in views:
            continue
        if member.kind != "CELL":
            raise RegionRefused("CONTEXT_MEMBER_MAPPING_UNAVAILABLE")
        physical = native.get(mid)
        if physical is None or member.text != physical[1].text:
            raise RegionRefused("CONTEXT_CELL_SOURCE_MISMATCH")
        table, cell = physical
        if member.page != table.page or member.bbox != cell.bbox:
            raise RegionRefused("CONTEXT_CELL_SOURCE_MISMATCH")
        if _observed(cell):
            exact.append(("MEMBER", mid, table.page, cell.text, cell.bbox))
    exact.extend(
        ("CELL", e.cell_id, e.page, e.text, e.observed_bbox)
        for e in inventory
        if e.observed_bbox is not None
    )
    exact_by_page = {}
    for carrier in exact:
        exact_by_page.setdefault(carrier[2], []).append(carrier)
    entries = []
    for span in sorted(spans.values(), key=lambda s: (s.page, s.bbox[1], s.bbox[0], s.span_id)):
        sid, length = span.span_id, len(span.text)
        offsets = coverage[sid]
        if length and len(offsets) == length:
            status = "CARRIED"
        elif length and len(excluded[sid]) == length and not nonfurniture[sid]:
            status = "EXCLUDED"
        else:
            matches = {
                (kind, cid, "EXACT")
                for kind, cid, page, text, box in exact_by_page.get(span.page, ())
                if span.text
                and isinstance(text, str)
                and span.text in text
                and _inside(span.bbox, box)
            }
            if matches:
                carriers[sid].update(matches)
                status = "CARRIED_EXACT"
            elif offsets:
                status = "PARTIAL"
            else:
                status = "NOT_CARRIED"
        entries.append(
            TextEntry(
                sid,
                span.page,
                span.text,
                length,
                status,
                tuple(TextCarrier(*c) for c in sorted(carriers[sid])),
                _missing(length, offsets) if status == "PARTIAL" else (),
                "reader-furniture/1" if status == "EXCLUDED" else None,
            )
        )
    return TextLedger(source_id, tuple(entries))
