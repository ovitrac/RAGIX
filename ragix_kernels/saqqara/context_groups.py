"""Bounded source-context overlays, independent of table connectivity.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

These functions read observations and region indexes without changing either.
Cell observations remain literal records, not assertions of table structure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

from ..harvest.region_types import RegionRefused, box_union, identity
from ..harvest.regions import _inside

if TYPE_CHECKING:
    from ..harvest.regions import RegionIndex
    from .explorer import ExplorerResult

Box = tuple[float, float, float, float]
RULE = "context-group/1"


@dataclass(frozen=True)
class ContextGroupPolicy:
    max_group_members: int = 400
    max_group_cells: int = 400
    page_scale_fraction: float = 0.5

    def __post_init__(self):
        if (
            any(
                type(n) is not int or n <= 0 for n in (self.max_group_members, self.max_group_cells)
            )
            or type(self.page_scale_fraction) not in (int, float)
            or not math.isfinite(self.page_scale_fraction)
            or not 0 < self.page_scale_fraction <= 1
        ):
            raise RegionRefused("INVALID_CONTEXT_GROUP_POLICY")


@dataclass(frozen=True)
class ContextGroup:
    group_id: str
    rule: str
    source_id: str
    pages: tuple[int, ...]
    envelope_box: Box | None
    envelope_rule: str
    origin: str
    candidate_id: str
    refusal_code: str
    chunk_index: int
    chunk_count: int
    prev_group_id: str | None
    next_group_id: str | None
    region_ids: tuple[str, ...]
    member_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    order: tuple[str, ...]
    flags: tuple[str, ...]
    policy: ContextGroupPolicy


@dataclass(frozen=True)
class CellEntry:
    cell_id: str
    candidate_id: str
    page: int
    group_id: str
    text: str | None
    readability: str
    geometry_kind: str
    flags: tuple[str, ...]
    observed_bbox: Box | None
    container_bbox: Box | None
    source_spans: tuple[str, ...]


def _observed(cell) -> bool:
    return cell.geometry_kind == "cell_box" and "MISSING_CELL_GEOMETRY" not in cell.flags


def _source(result, source_id, index=None):
    """Validate copy/page identity before deriving any overlay or inventory."""
    if not isinstance(source_id, str) or not source_id or result.document.source_id != source_id:
        raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
    if result.reading is None or getattr(result.report, "status", None) == "FAILED":
        raise RegionRefused("EXPLORER_RESULT_UNREADABLE")
    if result.reading.source_id != source_id:
        raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
    pages, tables, cells, spans = {}, {}, {}, set()
    for page in result.document.pages:
        if page.page in pages:
            raise RegionRefused("DUPLICATE_CONTEXT_PAGE")
        pages[page.page] = page
        for span in page.spans:
            if span.source_id != source_id or span.page != page.page:
                raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
            if span.span_id in spans:
                raise RegionRefused("DUPLICATE_CONTEXT_SPAN")
            spans.add(span.span_id)
        for table in page.tables:
            if table.table_id in tables:
                raise RegionRefused("DUPLICATE_CONTEXT_CANDIDATE")
            if table.page != page.page or any(
                e.source_id != source_id or e.page != page.page for e in table.evidence
            ):
                raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
            tables[table.table_id] = table
            for row in table.cell_rows:
                for cell in row:
                    if cell.cell_id in cells:
                        raise RegionRefused("DUPLICATE_CONTEXT_CELL")
                    cells[cell.cell_id] = (table, cell)
    if index is not None:
        if index.source_id != source_id:
            raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
        for region in index.regions:
            if region.source_id != source_id or not set(region.pages) <= pages.keys():
                raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
            for member in region.members:
                if member.source_id != source_id or member.page not in pages:
                    raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
        for refusal in index.refusals:
            if refusal.source_id != source_id or not set(refusal.pages) <= pages.keys():
                raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
    return pages, tables, cells


def _members(index):
    """Canonical ids may be shared by regions, but not disagree on content."""
    members = {}
    for region in index.regions:
        for member in region.members:
            if member.member_id in members:
                previous = members[member.member_id]
                # A source line can appear as both LINE and CAPTION. Those are
                # presentations; text, location and source references must agree.
                fields = ("source_id", "page", "text", "bbox", "source_spans")
                if any(getattr(previous, name) != getattr(member, name) for name in fields):
                    raise RegionRefused("CONFLICTING_CONTEXT_MEMBER")
                if (previous.kind == "CELL") != (member.kind == "CELL"):
                    raise RegionRefused("CONFLICTING_CONTEXT_MEMBER")
                continue
            members[member.member_id] = member
    return members


def _refused_candidates(index, tables, cells):
    candidates = {}
    for refusal in index.refusals:
        if not set(refusal.member_ids) <= cells.keys():
            raise RegionRefused("REFUSAL_SOURCE_MEMBER_MISSING")
        # Recovered-table refusals may name a composite id: project their exact
        # source members back to native observations, without inventing cells.
        ids = {cells[mid][0].table_id for mid in refusal.member_ids}
        if refusal.table_id in tables:
            ids.add(refusal.table_id)
        if not ids:
            raise RegionRefused("REFUSAL_CANDIDATE_UNAVAILABLE")
        for ident in sorted(ids):
            table = tables[ident]
            if table.page not in refusal.pages:
                raise RegionRefused("REFUSAL_PAGE_MISMATCH")
            if ident in candidates and candidates[ident] != refusal.code:
                raise RegionRefused("CONFLICTING_CONTEXT_REFUSAL")
            candidates[ident] = refusal.code
    return candidates


def _member_key(member):
    return member.page, member.bbox[1], member.bbox[0], member.member_id


def _region_key(region):
    return region.page, region.bbox[1], region.bbox[0], region.region_id


def _pack_regions(regions, eligible, members, limit):
    """Prefer region boundaries; split an oversized region without dropping ids."""
    chunks, current, seen = [], [], set()
    for region in regions:
        ids = sorted(
            {m.member_id for m in region.members} & eligible - seen,
            key=lambda ident: _member_key(members[ident]),
        )
        seen.update(ids)
        if not ids:
            continue
        if len(ids) > limit:
            if current:
                chunks.append((tuple(current), False))
                current = []
            chunks.extend((tuple(ids[i : i + limit]), True) for i in range(0, len(ids), limit))
        else:
            if len(current) + len(ids) > limit:
                chunks.append((tuple(current), False))
                current = []
            current.extend(ids)
    if current:
        chunks.append((tuple(current), False))
    if seen != eligible:
        raise RegionRefused("CONTEXT_MEMBER_NOT_IN_REGION")
    return chunks


def context_groups(
    result: ExplorerResult,
    index: RegionIndex,
    *,
    source_id: str,
    policy: ContextGroupPolicy = ContextGroupPolicy(),
) -> tuple[ContextGroup, ...]:
    """Group literal observations of refused candidates; assert no cell topology.

    Source order is the existing page/geometry order, with ids breaking ties.
    It is a deterministic presentation order, not a semantic reading assertion.
    """
    if not isinstance(policy, ContextGroupPolicy):
        raise RegionRefused("INVALID_CONTEXT_GROUP_POLICY")
    pages, tables, cells = _source(result, source_id, index)
    refused = _refused_candidates(index, tables, cells)
    members = _members(index)
    regions = sorted(index.regions, key=_region_key)
    groups = []
    for ident in sorted(refused, key=lambda t: (tables[t].page, t)):
        table = tables[ident]
        native = sorted(
            (c for row in table.cell_rows for c in row),
            key=lambda c: (c.bbox[1], c.bbox[0], c.cell_id),
        )
        located = [c for c in native if _observed(c)]
        envelope = box_union(c.bbox for c in located) if located else None
        flags = set()
        if envelope is None:
            flags.add("GEOMETRY_UNAVAILABLE")
        elif len(located) != len(native):
            flags.add("PARTIAL_GEOMETRY")
        if envelope is not None:
            page = pages[table.page]
            area = (envelope[2] - envelope[0]) * (envelope[3] - envelope[1])
            if area >= policy.page_scale_fraction * page.width * page.height:
                flags.add("PAGE_SCALE_CANDIDATE")
        eligible = {
            mid
            for mid, member in members.items()
            if member.kind != "CELL"
            and member.page == table.page
            and envelope is not None
            and _inside(member.bbox, envelope)
        }
        member_chunks = _pack_regions(regions, eligible, members, policy.max_group_members)
        cell_chunks = [
            native[i : i + policy.max_group_cells]
            for i in range(0, len(native), policy.max_group_cells)
        ]
        count = max(len(member_chunks), len(cell_chunks), 1)
        candidate_groups = []
        for k in range(count):
            mids, split = member_chunks[k] if k < len(member_chunks) else ((), False)
            chunk_cells = cell_chunks[k] if k < len(cell_chunks) else ()
            cids = tuple(c.cell_id for c in chunk_cells)
            chunk_flags = flags | ({"REGION_SPLIT"} if split else set())
            if any(c.text is None for c in chunk_cells):
                chunk_flags.add("CONTAINS_UNREADABLE")
            candidate_groups.append(
                ContextGroup(
                    identity(RULE, source_id, ident, k, mids, cids),
                    RULE,
                    source_id,
                    (table.page,),
                    envelope,
                    "observed-cell-union/1",
                    "REFUSED_TABLE_CANDIDATE",
                    ident,
                    refused[ident],
                    k,
                    count,
                    None,
                    None,
                    tuple(
                        r.region_id for r in regions if set(mids) & {m.member_id for m in r.members}
                    ),
                    mids,
                    cids,
                    tuple(sorted(mids, key=lambda mid: _member_key(members[mid]))),
                    tuple(sorted(chunk_flags)),
                    policy,
                )
            )
        groups.extend(
            replace(
                group,
                prev_group_id=candidate_groups[k - 1].group_id if k else None,
                next_group_id=candidate_groups[k + 1].group_id if k + 1 < count else None,
            )
            for k, group in enumerate(candidate_groups)
        )
    shared = Counter(mid for group in groups for mid in group.member_ids)
    return tuple(
        (
            replace(group, flags=tuple(sorted(set(group.flags) | {"SHARES_MEMBERS"})))
            if any(shared[mid] > 1 for mid in group.member_ids)
            else group
        )
        for group in groups
    )


def _group_assignments(groups, source_id, tables, cells):
    owners, ids, candidates = {}, set(), set()
    for group in groups:
        if (
            group.source_id != source_id
            or group.rule != RULE
            or group.origin != "REFUSED_TABLE_CANDIDATE"
        ):
            raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
        if group.group_id in ids or group.candidate_id not in tables:
            raise RegionRefused("INVALID_CONTEXT_GROUP")
        ids.add(group.group_id)
        candidates.add(group.candidate_id)
        if group.pages != (tables[group.candidate_id].page,):
            raise RegionRefused("CONTEXT_SOURCE_MISMATCH")
        expected_id = identity(
            RULE, source_id, group.candidate_id, group.chunk_index, group.member_ids, group.cell_ids
        )
        if group.group_id != expected_id:
            raise RegionRefused("STALE_CONTEXT_GROUP")
        for cid in group.cell_ids:
            if cid not in cells or cells[cid][0].table_id != group.candidate_id:
                raise RegionRefused("CONTEXT_CELL_SOURCE_MISMATCH")
            if cid in owners:
                raise RegionRefused("DUPLICATE_CONTEXT_CELL_ASSIGNMENT")
            owners[cid] = group.group_id
    expected = {cid for cid, (table, cell) in cells.items() if table.table_id in candidates}
    if owners.keys() != expected:
        raise RegionRefused("CONTEXT_CELL_NOT_ASSIGNED")
    return owners


def cell_inventory(
    result: ExplorerResult,
    groups: tuple[ContextGroup, ...],
    *,
    source_id: str,
) -> tuple[CellEntry, ...]:
    """Keep every grouped native cell once, including unreadable/unlocated cells.

    Refusal coverage is validated by context_groups/text_ledger, which receive the
    index. This function validates all cells of every candidate named by groups.
    """
    groups = tuple(groups)
    _, tables, cells = _source(result, source_id)
    owners = _group_assignments(groups, source_id, tables, cells)
    entries = []
    for cid in sorted(
        owners,
        key=lambda cid: (cells[cid][0].page, cells[cid][1].bbox[1], cells[cid][1].bbox[0], cid),
    ):
        table, cell = cells[cid]
        entries.append(
            CellEntry(
                cid,
                table.table_id,
                table.page,
                owners[cid],
                cell.text,
                (
                    "UNREADABLE"
                    if cell.text is None
                    else "READABLE_EMPTY" if cell.text == "" else "READABLE"
                ),
                cell.geometry_kind,
                cell.flags,
                cell.bbox if _observed(cell) else None,
                None if _observed(cell) else cell.bbox,
                cell.source_spans,
            )
        )
    return tuple(entries)
