"""Literal source-character coverage over canonical members and context groups.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

An association to a source span is not proof that its characters survived.
Mappings are checked against both the current carrier and the original span.
Whitespace normalisation is accepted only when the retained representation
preserves separation across source-span seams.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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

NORMALISATION_RULE = "whitespace-normalisation/1"
LEDGER_RULE = "text-occurrence-ledger/2"
STATUSES = (
    "CARRIED",
    "CARRIED_EXACT",
    "CARRIED_NORMALISED",
    "PARTIAL",
    "NOT_CARRIED",
    "EXCLUDED",
)


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
class NormalisedRange:
    start: int
    end: int
    rule: str = NORMALISATION_RULE

    def __post_init__(self):
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
            or self.rule != NORMALISATION_RULE
        ):
            raise RegionRefused("INVALID_TEXT_NORMALISATION")


@dataclass(frozen=True)
class OrderBreak:
    span_id: str
    carrier_id: str
    positions: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if (
            not self.span_id
            or not self.carrier_id
            or len(self.positions) < 2
            or any(
                type(carrier_position) is not int
                or type(source_offset) is not int
                or min(carrier_position, source_offset) < 0
                for carrier_position, source_offset in self.positions
            )
            or all(
                before[1] < after[1] for before, after in zip(self.positions, self.positions[1:])
            )
        ):
            raise RegionRefused("INVALID_TEXT_ORDER_BREAK")


def _range_offsets(ranges):
    offsets = set()
    previous = -1
    for start, end in ranges:
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
            or start <= previous
        ):
            raise RegionRefused("INVALID_TEXT_COVERAGE")
        offsets.update(range(start, end))
        previous = end
    return offsets


@dataclass(frozen=True)
class TextEntry:
    span_id: str
    page: int
    span_text: str
    span_length: int
    status: str
    carriers: tuple[TextCarrier, ...]
    carried_count: int
    normalised: tuple[NormalisedRange, ...] = ()
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
            or any(not isinstance(carrier, TextCarrier) for carrier in self.carriers)
            or type(self.carried_count) is not int
            or not 0 <= self.carried_count <= self.span_length
            or not isinstance(self.normalised, tuple)
            or any(not isinstance(item, NormalisedRange) for item in self.normalised)
            or not isinstance(self.missing, tuple)
        ):
            raise RegionRefused("INVALID_TEXT_ENTRY")
        if bool(self.rule) != (self.status == "EXCLUDED"):
            raise RegionRefused("INVALID_TEXT_EXCLUSION")
        normalised = _range_offsets(tuple((item.start, item.end) for item in self.normalised))
        missing = _range_offsets(self.missing)
        if (
            any(offset >= self.span_length for offset in normalised | missing)
            or normalised & missing
            or self.carried_count + len(normalised) + len(missing) != self.span_length
        ):
            raise RegionRefused("INVALID_TEXT_PARTITION")
        if self.carried_count and not self.carriers and self.status != "EXCLUDED":
            raise RegionRefused("TEXT_CARRIER_REQUIRED")
        if self.status == "CARRIED" and (
            self.carried_count != self.span_length
            or normalised
            or missing
            or any(carrier.how == "EXACT" for carrier in self.carriers)
        ):
            raise RegionRefused("INVALID_TEXT_STATUS")
        if self.status == "CARRIED_EXACT" and (
            self.carried_count != self.span_length
            or normalised
            or missing
            or not any(carrier.how == "EXACT" for carrier in self.carriers)
        ):
            raise RegionRefused("INVALID_TEXT_STATUS")
        if self.status == "CARRIED_NORMALISED" and (
            not normalised or missing or self.carried_count + len(normalised) != self.span_length
        ):
            raise RegionRefused("INVALID_TEXT_STATUS")
        if self.status == "PARTIAL" and (not missing or self.carried_count + len(normalised) == 0):
            raise RegionRefused("INVALID_TEXT_STATUS")
        if self.status == "NOT_CARRIED" and (
            self.carried_count or normalised or len(missing) != self.span_length
        ):
            raise RegionRefused("INVALID_TEXT_STATUS")
        if self.status == "EXCLUDED" and (
            missing or self.carried_count + len(normalised) != self.span_length
        ):
            raise RegionRefused("INVALID_TEXT_STATUS")


@dataclass(frozen=True)
class TextLedger:
    source_id: str
    entries: tuple[TextEntry, ...]
    order_breaks: tuple[OrderBreak, ...] = ()
    rule: str = LEDGER_RULE

    def __post_init__(self):
        entries = {entry.span_id: entry for entry in self.entries}
        if (
            not self.source_id
            or self.rule != LEDGER_RULE
            or not isinstance(self.entries, tuple)
            or any(not isinstance(entry, TextEntry) for entry in self.entries)
            or len(entries) != len(self.entries)
            or not isinstance(self.order_breaks, tuple)
            or any(not isinstance(item, OrderBreak) for item in self.order_breaks)
            or len({(item.span_id, item.carrier_id) for item in self.order_breaks})
            != len(self.order_breaks)
            or any(
                item.span_id not in entries
                or item.carrier_id
                not in {carrier.carrier_id for carrier in entries[item.span_id].carriers}
                or any(
                    source_offset >= entries[item.span_id].span_length
                    for _, source_offset in item.positions
                )
                for item in self.order_breaks
            )
        ):
            raise RegionRefused("INVALID_TEXT_LEDGER")

    @property
    def passes(self) -> bool:
        return not self.order_breaks and not any(
            entry.status in ("PARTIAL", "NOT_CARRIED") for entry in self.entries
        )

    @staticmethod
    def _categories(counts):
        return {
            "passed": counts["CARRIED"] + counts["CARRIED_EXACT"],
            "accepted_normalisation": counts["CARRIED_NORMALISED"],
            "missing_content": counts["PARTIAL"] + counts["NOT_CARRIED"],
            "excluded": counts["EXCLUDED"],
        }

    def summary(self) -> dict:
        pages = {}
        page_categories = {}
        for page in sorted({entry.page for entry in self.entries}):
            counts = Counter(entry.status for entry in self.entries if entry.page == page)
            pages[page] = {status: counts[status] for status in STATUSES}
            page_categories[page] = self._categories(counts)
        counts = Counter(entry.status for entry in self.entries)
        return {
            "rule": self.rule,
            "source_id": self.source_id,
            "passes": self.passes,
            "counts": {status: counts[status] for status in STATUSES},
            "categories": self._categories(counts),
            "pages": pages,
            "page_categories": page_categories,
            "order_breaks": len(self.order_breaks),
        }


@dataclass(frozen=True)
class _Occurrence:
    kind: str
    carrier_id: str
    structure_kind: str
    position: int
    text: str

    @property
    def carrier_key(self):
        return self.kind, self.carrier_id

    @property
    def structure_key(self):
        return self.structure_kind, self.carrier_id


@dataclass(frozen=True)
class _ExactCarrier:
    kind: str
    carrier_id: str
    page: int
    text: str
    bbox: tuple[float, float, float, float]


def _ranges(offsets):
    offsets = sorted(set(offsets))
    if not offsets:
        return ()
    ranges = []
    start = previous = offsets[0]
    for offset in offsets[1:]:
        if offset != previous + 1:
            ranges.append((start, previous + 1))
            start = offset
        previous = offset
    ranges.append((start, previous + 1))
    return tuple(ranges)


def _trimmed(text):
    start, end = 0, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end, text[start:end]


def _substring_occurrences(text, literal):
    start = 0
    while literal and start <= len(text) - len(literal):
        found = text.find(literal, start)
        if found < 0:
            break
        yield found, found + len(literal)
        start = found + 1


def _overlaps(interval, used):
    return any(max(interval[0], other[0]) < min(interval[1], other[1]) for other in used)


def _separation_preserved(left, right):
    pairs = [(a, b) for a in left for b in right]
    same = [(a, b) for a, b in pairs if a.carrier_key == b.carrier_key]
    if same:
        return all(
            a.position < b.position
            and any(character.isspace() for character in a.text[a.position + 1 : b.position])
            for a, b in same
        )
    return bool(pairs) and all(a.structure_key != b.structure_key for a, b in pairs)


def _exact_carriers(members, native, inventory):
    records = {}
    for member_id, member in members.items():
        if member.kind != "CELL":
            continue
        physical = native.get(member_id)
        if physical is None or member.text != physical[1].text:
            raise RegionRefused("CONTEXT_CELL_SOURCE_MISMATCH")
        table, cell = physical
        if member.page != table.page or member.bbox != cell.bbox:
            raise RegionRefused("CONTEXT_CELL_SOURCE_MISMATCH")
        if _observed(cell) and isinstance(cell.text, str):
            records[(table.page, member_id)] = _ExactCarrier(
                "MEMBER", member_id, table.page, cell.text, cell.bbox
            )
    for entry in inventory:
        if entry.observed_bbox is None or not isinstance(entry.text, str):
            continue
        key = (entry.page, entry.cell_id)
        candidate = _ExactCarrier(
            "CELL", entry.cell_id, entry.page, entry.text, entry.observed_bbox
        )
        previous = records.get(key)
        if previous is not None:
            if previous.text != candidate.text or previous.bbox != candidate.bbox:
                raise RegionRefused("CONTEXT_CELL_SOURCE_MISMATCH")
            continue
        records[key] = candidate
    return tuple(
        sorted(
            records.values(),
            key=lambda item: (
                item.page,
                item.bbox[1],
                item.bbox[0],
                item.carrier_id,
            ),
        )
    )


def _allocate_exact(spans, carriers, coverage, occurrences, text_carriers):
    used = defaultdict(list)
    exact_offsets = defaultdict(set)
    carriers_by_page = defaultdict(list)
    for carrier_index, carrier in enumerate(carriers):
        carriers_by_page[carrier.page].append((carrier_index, carrier))
    for span in spans:
        start, end, literal = _trimmed(span.text)
        needed = set(range(start, end)) - coverage[span.span_id]
        if not literal or not needed:
            continue
        candidates = []
        for carrier_index, carrier in carriers_by_page[span.page]:
            if not _inside(span.bbox, carrier.bbox):
                continue
            for occurrence in _substring_occurrences(carrier.text, literal):
                if not _overlaps(occurrence, used[(carrier.page, carrier.carrier_id)]):
                    candidates.append((carrier_index, occurrence, carrier))
        if not candidates:
            continue
        _, interval, carrier = min(candidates, key=lambda item: (item[0], item[1][0]))
        used[(carrier.page, carrier.carrier_id)].append(interval)
        text_carriers[span.span_id].add((carrier.kind, carrier.carrier_id, "EXACT"))
        for offset in range(start, end):
            position = interval[0] + offset - start
            coverage[span.span_id].add(offset)
            exact_offsets[span.span_id].add(offset)
            occurrences[span.span_id][offset].add(
                _Occurrence(
                    carrier.kind,
                    carrier.carrier_id,
                    "CELL",
                    position,
                    carrier.text,
                )
            )
    return exact_offsets


def _source_sequence(spans):
    return tuple(
        (span.span_id, offset, character)
        for span in spans
        for offset, character in enumerate(span.text)
    )


def _normalised_whitespace(spans, coverage, occurrences):
    sequence = _source_sequence(spans)
    left, right = {}, {}
    previous = None
    for position in sequence:
        span_id, offset, character = position
        if not character.isspace() and offset in coverage[span_id]:
            previous = span_id, offset
        elif character.isspace() and offset not in coverage[span_id]:
            left[span_id, offset] = previous
    following = None
    for position in reversed(sequence):
        span_id, offset, character = position
        if not character.isspace() and offset in coverage[span_id]:
            following = span_id, offset
        elif character.isspace() and offset not in coverage[span_id]:
            right[span_id, offset] = following
    normalised = defaultdict(set)
    for key in sorted(set(left) | set(right)):
        before, after = left.get(key), right.get(key)
        if before is None or after is None:
            normalised[key[0]].add(key[1])
            continue
        if _separation_preserved(
            occurrences[before[0]][before[1]],
            occurrences[after[0]][after[1]],
        ):
            normalised[key[0]].add(key[1])
    return normalised


def text_ledger(
    result: ExplorerResult,
    index: RegionIndex,
    groups: tuple[ContextGroup, ...],
    *,
    source_id: str,
) -> TextLedger:
    """Measure literal retention; expose gaps without repairing source text.

    Mapped characters must equal their source characters and retain their order
    within a carrier. Exact cell fallback consumes distinct text occurrences.
    Unmapped whitespace is accepted only under whitespace-normalisation/1.
    """
    groups = tuple(groups)
    pages, tables, native = _source(result, source_id, index)
    members = _members(index)
    refused = _refused_candidates(index, tables, native)
    if {group.candidate_id for group in groups} != set(refused):
        raise RegionRefused("CONTEXT_REFUSAL_NOT_GROUPED")
    for group in groups:
        if any(member_id not in members for member_id in group.member_ids):
            raise RegionRefused("CONTEXT_MEMBER_SOURCE_MISSING")
    inventory = cell_inventory(result, groups, source_id=source_id)
    spans = {span.span_id: span for page in pages.values() for span in page.spans}
    ordered_spans = tuple(
        sorted(
            spans.values(),
            key=lambda span: (span.page, span.bbox[1], span.bbox[0], span.span_id),
        )
    )
    views = {view.view_id: view for page in pages.values() for view in page_lines(page)}
    furniture = set(result.reading.furniture)
    if not furniture <= views.keys():
        raise RegionRefused("FURNITURE_SOURCE_LINE_MISSING")
    coverage = {span_id: set() for span_id in spans}
    excluded = {span_id: set() for span_id in spans}
    nonfurniture = {span_id: set() for span_id in spans}
    text_carriers = {span_id: set() for span_id in spans}
    occurrences = {span_id: defaultdict(set) for span_id in spans}
    furniture_occurrences = {span_id: defaultdict(set) for span_id in spans}
    order_breaks = []
    for view_id, view in views.items():
        if len(view.text) != len(view.mapping):
            raise RegionRefused("INVALID_CONTEXT_CHARACTER_MAPPING")
        member = members.get(view_id)
        if member is not None and (member.text != view.text or member.page != view.page):
            raise RegionRefused("STALE_CONTEXT_MEMBER_TEXT")
        if member is not None and (
            member.bbox != view.bbox
            or set(member.source_spans) != {ref.span_id for ref in view.mapping if ref is not None}
        ):
            raise RegionRefused("STALE_CONTEXT_MEMBER_GEOMETRY_OR_REFS")
        positions = defaultdict(list)
        covered_here = set()
        for carrier_position, ref in enumerate(view.mapping):
            if ref is None:
                continue
            span = spans.get(ref.span_id)
            if span is None or span.page != view.page:
                raise RegionRefused("CONTEXT_MAPPING_SOURCE_MISSING")
            offset = ref.offset - span.source_offset
            if (
                type(offset) is not int
                or not 0 <= offset < len(span.text)
                or view.text[carrier_position] != span.text[offset]
            ):
                raise RegionRefused("CONTEXT_MAPPING_CHARACTER_MISMATCH")
            positions[span.span_id].append((carrier_position, offset))
            (excluded if view_id in furniture else nonfurniture)[span.span_id].add(offset)
            if view_id in furniture:
                furniture_occurrences[span.span_id][offset].add(
                    _Occurrence(
                        "MEMBER",
                        view_id,
                        "LINE",
                        carrier_position,
                        view.text,
                    )
                )
            if member is not None:
                coverage[span.span_id].add(offset)
                covered_here.add(span.span_id)
                occurrences[span.span_id][offset].add(
                    _Occurrence(
                        "MEMBER",
                        view_id,
                        "LINE",
                        carrier_position,
                        view.text,
                    )
                )
        if member is not None:
            for span_id, sequence in positions.items():
                if any(before[1] >= after[1] for before, after in zip(sequence, sequence[1:])):
                    order_breaks.append(OrderBreak(span_id, view_id, tuple(sequence)))
            for span_id in covered_here:
                text_carriers[span_id].add(("MEMBER", view_id, "MAPPED"))
    exact_offsets = _allocate_exact(
        ordered_spans,
        _exact_carriers(members, native, inventory),
        coverage,
        occurrences,
        text_carriers,
    )
    normalised = _normalised_whitespace(ordered_spans, coverage, occurrences)
    furniture_normalised = _normalised_whitespace(ordered_spans, excluded, furniture_occurrences)
    entries = []
    for span in ordered_spans:
        span_id = span.span_id
        length = len(span.text)
        offsets = set(range(length))
        non_whitespace = {
            offset for offset, character in enumerate(span.text) if not character.isspace()
        }
        whitespace = offsets - non_whitespace
        furniture_normalised_offsets = furniture_normalised[span_id] - excluded[span_id]
        excluded_with_normalisation = (
            bool(excluded[span_id])
            and non_whitespace <= excluded[span_id]
            and not (non_whitespace & nonfurniture[span_id])
            and whitespace <= excluded[span_id] | furniture_normalised_offsets
        )
        if excluded_with_normalisation:
            status = "EXCLUDED"
            carried_count = len(excluded[span_id])
            normalised_ranges = tuple(
                NormalisedRange(start, end) for start, end in _ranges(furniture_normalised_offsets)
            )
            missing_ranges = ()
        else:
            normalised_offsets = normalised[span_id] - coverage[span_id]
            missing_offsets = set(range(length)) - coverage[span_id] - normalised_offsets
            carried_count = len(coverage[span_id])
            normalised_ranges = tuple(
                NormalisedRange(start, end) for start, end in _ranges(normalised_offsets)
            )
            missing_ranges = _ranges(missing_offsets)
            if missing_offsets:
                status = "PARTIAL" if carried_count or normalised_offsets else "NOT_CARRIED"
            elif normalised_offsets:
                status = "CARRIED_NORMALISED"
            elif exact_offsets[span_id]:
                status = "CARRIED_EXACT"
            else:
                status = "CARRIED"
        entries.append(
            TextEntry(
                span_id=span_id,
                page=span.page,
                span_text=span.text,
                span_length=length,
                status=status,
                carriers=tuple(TextCarrier(*carrier) for carrier in sorted(text_carriers[span_id])),
                carried_count=carried_count,
                normalised=normalised_ranges,
                missing=missing_ranges,
                rule="reader-furniture/1" if status == "EXCLUDED" else None,
            )
        )
    return TextLedger(
        source_id,
        tuple(entries),
        tuple(sorted(order_breaks, key=lambda item: (item.span_id, item.carrier_id))),
    )
