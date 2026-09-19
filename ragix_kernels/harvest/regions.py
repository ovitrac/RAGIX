"""Lookup source-exact prose, tables, lists and figures as structured JSON.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from collections import defaultdict
from dataclasses import asdict, replace
from .region_types import (
    RULE,
    KINDS,
    RegionRefused,
    TableRegionRefusal,
    RegionMember,
    PageGeometry,
    BoundaryPolicy,
    RegionWindow,
    RegionLimits,
    FigureImage,
    FigureInput,
    Region,
    Neighbourhood,
    Anchor,
    RenderableRegion,
    box_union,
    identity,
    json_data,
)
from .region_boundaries import group_lines, order
from .region_images import image_payload, image_from_store, validate_image
from .table_context import CellSpan


def region(kind, members, *, source_id, policy, flags=(), figure=None):
    members = tuple(
        sorted(
            members,
            key=lambda m: (
                m.page,
                m.row if m.kind == "CELL" else m.bbox[1],
                m.column if m.kind == "CELL" else m.bbox[0],
                m.kind,
                m.member_id,
            ),
        )
    )
    if kind not in KINDS or any(m.source_id != source_id for m in members):
        raise RegionRefused("MIXED_REGION_SOURCE")
    if len({m.member_id for m in members}) != len(members):
        raise RegionRefused("DUPLICATE_REGION_MEMBER")
    boxes = defaultdict(list)
    for m in members:
        boxes[m.page].append(m.bbox)
    if figure:
        boxes[figure.page].append(figure.bbox)
    if not boxes:
        raise RegionRefused("EMPTY_REGION")
    pages = tuple(sorted(boxes))
    page_boxes = tuple((p, box_union(boxes[p])) for p in pages)
    fs = set(flags) | {f for m in members for f in m.flags}
    if len(pages) > 1:
        fs.add("CROSSES_PAGE")
    if kind == "TABLE" and not any(m.kind == "CELL" and m.is_header for m in members):
        fs.add("NO_COLUMN_HEADER")
    if figure and figure.image is None:
        fs.add("FIGURE_IMAGE_UNAVAILABLE")
    rid = identity(
        RULE,
        source_id,
        pages,
        kind,
        tuple(m.member_id for m in members),
        figure.figure_id if figure else None,
        asdict(policy),
    )
    return Region(
        rid,
        source_id,
        pages[0],
        page_boxes[0][1],
        kind,
        RULE,
        members,
        pages,
        page_boxes,
        tuple(sorted(fs)),
        figure.image if figure else None,
        figure.image_reason if figure else None,
        figure.figure_id if figure else None,
        figure.bbox if figure else None,
    )


def _inside(box, container):
    return all(
        (
            box[0] >= container[0],
            box[1] >= container[1],
            box[2] <= container[2],
            box[3] <= container[3],
        )
    )


class RegionIndex:
    def __init__(
        self,
        source_id,
        pages,
        lines,
        *,
        tables=(),
        figures=(),
        policy=BoundaryPolicy(),
        limits=RegionLimits(),
        continuations=(),
        refusals=(),
    ):
        self.source_id = source_id
        self.policy = policy
        self.limits = limits
        pages = tuple(pages)
        lines = tuple(lines)
        tables = tuple(tuple(t) for t in tables)
        figures = tuple(figures)
        if not isinstance(source_id, str) or not source_id:
            raise RegionRefused("SOURCE_ID_REQUIRED")
        if len({p.page for p in pages}) != len(pages):
            raise RegionRefused("DUPLICATE_PAGE")
        known_pages = {p.page for p in pages}
        self.refusals = tuple(refusals)
        if any(
            not isinstance(r, TableRegionRefusal)
            or r.source_id != source_id
            or not set(r.pages) <= known_pages
            for r in self.refusals
        ):
            raise RegionRefused("INVALID_TABLE_REGION_REFUSAL")
        all_members = lines + tuple(m for table in tables for m in table)
        if len(all_members) > limits.max_members:
            raise RegionRefused("REGION_MEMBER_LIMIT")
        if any(m.source_id != source_id or m.page not in known_pages for m in all_members):
            raise RegionRefused("MEMBER_SOURCE_OR_PAGE_MISMATCH")
        if any(f.source_id != source_id or f.page not in known_pages for f in figures):
            raise RegionRefused("FIGURE_SOURCE_OR_PAGE_MISMATCH")
        if len({m.member_id for m in all_members}) != len(all_members):
            raise RegionRefused("DUPLICATE_REGION_MEMBER")
        lookup = {m.member_id: m for m in lines}
        claimed = defaultdict(list)
        seeds = []
        for table in tables:
            if not table or any(m.kind != "CELL" for m in table):
                raise RegionRefused("TABLE_REQUIRES_OBSERVED_CELLS")
            tids = {m.table_id for m in table}
            if len(tids) != 1:
                raise RegionRefused("MIXED_TABLE_TOPOLOGY")
            boxes = {
                p: box_union(m.bbox for m in table if m.page == p) for p in {m.page for m in table}
            }
            aliases = []
            for line in lines:
                if line.page in boxes and _inside(line.bbox, boxes[line.page]):
                    aliases.append(
                        replace(line, flags=tuple(sorted(set(line.flags) | {"SOURCE_LINE_ALIAS"})))
                    )
                    claimed[line.member_id].append(len(seeds))
            seeds.append(region("TABLE", (*table, *aliases), source_id=source_id, policy=policy))
        for figure in figures:
            ids = set(figure.member_ids) | set(figure.caption_ids)
            if not ids <= set(lookup):
                raise RegionRefused("UNKNOWN_FIGURE_MEMBER")
            ids.update(
                m.member_id for m in lines if m.page == figure.page and _inside(m.bbox, figure.bbox)
            )
            members = []
            for ident in ids:
                m = lookup[ident]
                if m.page != figure.page:
                    raise RegionRefused("FIGURE_MEMBER_PAGE_MISMATCH")
                if ident in figure.caption_ids:
                    m = replace(m, kind="CAPTION")
                members.append(m)
                claimed[ident].append(len(seeds))
            seeds.append(
                region(
                    "FIGURE",
                    members,
                    source_id=source_id,
                    policy=policy,
                    flags=figure.flags,
                    figure=figure,
                )
            )
        self._ambiguous = {ident for ident, owners in claimed.items() if len(owners) > 1}
        barriers = tuple((p, b) for seed in seeds for p, b in seed.page_boxes)
        remaining = tuple(m for m in lines if m.member_id not in claimed)
        seeds.extend(
            region(g.kind, g.members, source_id=source_id, policy=policy, flags=g.flags)
            for g in group_lines(
                remaining, pages, policy, barriers=barriers, continuations=continuations
            )
        )
        self.regions = tuple(
            sorted(seeds, key=lambda r: (r.page, r.bbox[1], r.bbox[0], r.region_id))
        )
        self._member = {}
        for r in self.regions:
            for m in r.members:
                if m.member_id not in self._ambiguous:
                    self._member[m.member_id] = (r, m)
        self._validated_images = set()

    def refusal_report(self):
        """A serializable count and scope for every omitted table."""
        return {
            "source_id": self.source_id,
            "rule": "table-region/1",
            "count": len(self.refusals),
            "tables": [asdict(r) for r in self.refusals],
        }

    def get(self, anchor, *, window=RegionWindow()):
        if window.before + window.after > self.limits.max_members:
            raise RegionRefused("REGION_WINDOW_LIMIT")
        ident = anchor.cell_id if isinstance(anchor, CellSpan) else anchor
        if not isinstance(ident, str):
            raise RegionRefused("INVALID_ANCHOR")
        if ident in self._ambiguous:
            raise RegionRefused("AMBIGUOUS_ENCLOSING_REGION")
        if ident not in self._member:
            raise RegionRefused("ANCHOR_NOT_FOUND")
        target, member = self._member[ident]
        if member.text is None:
            raise RegionRefused("ANCHOR_TEXT_UNAVAILABLE")
        if isinstance(anchor, CellSpan):
            if (
                member.kind != "CELL"
                or anchor.source_id != self.source_id
                or anchor.table_id != member.table_id
                or anchor.page != member.page
                or tuple(anchor.bbox) != member.bbox
                or not 0 <= anchor.start <= anchor.end <= len(member.text)
                or member.text[anchor.start : anchor.end] != anchor.raw
            ):
                raise RegionRefused("STALE_OR_FOREIGN_CELL_SPAN")
            selected = Anchor(ident, member.page, anchor.start, anchor.end, anchor.raw, member.bbox)
        else:
            selected = Anchor(ident, member.page, 0, len(member.text), member.text, member.bbox)
        siblings = [r for r in self.regions if not window.same_page or member.page in r.pages]
        index = siblings.index(target)
        before = tuple(siblings[max(0, index - window.before) : index])
        after = tuple(siblings[index + 1 : index + 1 + window.after])
        flags = set(target.flags)
        if len(before) != window.before or len(after) != window.after:
            flags.add("TRUNCATED_AT_WINDOW")
        neighbourhood = Neighbourhood(
            before,
            after,
            window.before,
            window.after,
            member.page if window.same_page else None,
            window.rule,
        )
        for r in (*before, target, *after):
            if r.image is not None and r.image not in self._validated_images:
                validate_image(r.image, self.limits)
                self._validated_images.add(r.image)
        result = RenderableRegion(
            target, selected, neighbourhood, self.policy, tuple(sorted(flags))
        )
        if len(result.to_json().encode("utf-8")) > self.limits.max_output_bytes:
            raise RegionRefused("REGION_OUTPUT_LIMIT")
        return result


def cell_member(cell, *, is_header=False, is_row_label=False):
    """Wrap the existing Cell contract without changing its spans or topology."""
    return RegionMember(
        cell.cell_id,
        cell.source_id,
        cell.page,
        "CELL",
        cell.text,
        cell.bbox,
        cell.source_spans,
        cell.flags,
        cell.table_id,
        cell.row,
        cell.column,
        cell.row_span,
        cell.column_span,
        is_header,
        is_row_label,
    )
