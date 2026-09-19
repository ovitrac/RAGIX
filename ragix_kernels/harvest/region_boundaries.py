"""Declared geometry and source-structure rules for prose and list boundaries.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from collections import defaultdict
from dataclasses import dataclass, replace
import re
from .region_types import RegionRefused, BoundaryPolicy


@dataclass(frozen=True)
class MemberGroup:
    kind: str
    members: tuple
    flags: tuple[str, ...] = ()


def order(member):
    return (member.page, member.bbox[1], member.bbox[0], member.member_id)


def _compatible(a, b, policy):
    if a.kind == "HEADING" or b.kind == "HEADING":
        return False
    if a.section_id != b.section_id or a.column_id != b.column_id:
        return False
    height = max(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])
    return min(a.bbox[2], b.bbox[2]) > max(a.bbox[0], b.bbox[0]) and (
        abs(a.bbox[0] - b.bbox[0]) <= policy.max_indent_ratio * height
        or (_MARKER.match(b.text or "") and b.bbox[0] <= a.bbox[0])
        or (_MARKER.match(a.text or "") and _MARKER.match(b.text or ""))
    )


def _blocked(a, b, barriers):
    left, right = min(a.bbox[0], b.bbox[0]), max(a.bbox[2], b.bbox[2])
    return any(
        page == a.page
        and box[1] < b.bbox[1]
        and box[3] > a.bbox[3]
        and box[0] < right
        and box[2] > left
        for page, box in barriers
    )


def _edges(lines, pages, policy, barriers, continuations):
    by_page = defaultdict(list)
    flags = defaultdict(set)
    for line in lines:
        by_page[line.page].append(line)
    successors = {}
    predecessors = defaultdict(list)
    for page, items in by_page.items():
        items.sort(key=order)
        for index, a in enumerate(items):
            options = []
            for b in items[index + 1 :]:
                if b.bbox[1] <= a.bbox[1]:
                    continue
                gap = b.bbox[1] - a.bbox[3]
                # Candidate heights are not used to extend the search window.
                if gap > policy.max_gap_ratio * (a.bbox[3] - a.bbox[1]):
                    break
                if _compatible(a, b, policy) and not _blocked(a, b, barriers):
                    options.append(b)
            if not options:
                continue
            top = min(b.bbox[1] for b in options)
            best = [b for b in options if b.bbox[1] == top]
            if len(best) == 1:
                successors[a.member_id] = best[0].member_id
                predecessors[best[0].member_id].append(a)
            else:
                flags[a.member_id].add("AMBIGUOUS_PROSE_BOUNDARY")
    # One successor can have only its nearest unique predecessor.
    for target, previous in predecessors.items():
        bottom = max(p.bbox[3] for p in previous)
        best = [p for p in previous if p.bbox[3] == bottom]
        for p in previous:
            if len(best) != 1 or p is not best[0]:
                successors.pop(p.member_id, None)
        if len(best) != 1:
            flags[target].add("AMBIGUOUS_PROSE_BOUNDARY")
    if policy.cross_page:
        for page in sorted(by_page):
            if page + 1 not in by_page or page not in pages or page + 1 not in pages:
                continue
            ends = [
                a
                for a in by_page[page]
                if a.member_id not in successors
                and a.bbox[3] >= pages[page].height * (1 - policy.page_edge_fraction)
            ]
            targeted = set(successors.values())
            starts = [
                b
                for b in by_page[page + 1]
                if b.member_id not in targeted
                and b.bbox[1] <= pages[page + 1].height * policy.page_edge_fraction
            ]
            candidates = [
                (a, b)
                for a in ends
                for b in starts
                if _compatible(a, b, policy)
                and not any(
                    p == a.page
                    and box[3] > a.bbox[3]
                    and min(box[2], a.bbox[2]) > max(box[0], a.bbox[0])
                    for p, box in barriers
                )
                and not any(
                    p == b.page
                    and box[1] < b.bbox[1]
                    and min(box[2], b.bbox[2]) > max(box[0], b.bbox[0])
                    for p, box in barriers
                )
            ]
            for a, b in candidates:
                unique = (
                    sum(x.member_id == a.member_id for x, y in candidates) == 1
                    and sum(y.member_id == b.member_id for x, y in candidates) == 1
                )
                if unique and a.text and not re.search(r"[.!?:;]\s*$", a.text):
                    successors[a.member_id] = b.member_id
                    flags[a.member_id].add("PAGE_CONTINUATION_DERIVED")
                else:
                    flags[a.member_id].add("PAGE_BOUNDARY_UNRESOLVED")
                    flags[b.member_id].add("PAGE_BOUNDARY_UNRESOLVED")
    lookup = {m.member_id: m for m in lines}
    for left, right in continuations:
        if left not in lookup or right not in lookup:
            raise RegionRefused("UNKNOWN_CONTINUATION_MEMBER")
        a, b = lookup[left], lookup[right]
        if (
            not order(a) < order(b)
            or b.page - a.page not in (0, 1)
            or a.kind == "HEADING"
            or b.kind == "HEADING"
        ):
            raise RegionRefused("INVALID_CONTINUATION")
        for predecessor, target in tuple(successors.items()):
            if target == right and predecessor != left:
                del successors[predecessor]
        successors[left] = right
        flags[left].add("DECLARED_CONTINUATION")
        flags[left].discard("PAGE_BOUNDARY_UNRESOLVED")
        flags[right].discard("PAGE_BOUNDARY_UNRESOLVED")
    return successors, flags


_MARKER = re.compile(r"^\s*(?:(?P<bullet>[•◦▪‣*\-])\s+(?=\S)|(?P<number>\d+)[.)]\s+(?=\S))")


def _list_start(line, chain, index):
    match = _MARKER.match(line.text or "")
    if not match:
        return False
    if match["bullet"]:
        return not (match["bullet"] == "-" and re.match(r"\s*\d", line.text[match.end() :]))
    return any(
        (m := _MARKER.match(other.text or ""))
        and m["number"]
        and int(m["number"]) == int(match["number"]) + 1
        for other in chain[index + 1 :]
    )


def _split_chain(chain, flags):
    output = []
    current = []
    kind = "PROSE"
    base = 0

    def finish():
        if current:
            fs = {f for m in current for f in flags[m.member_id]}
            fs.add("LIST_STRUCTURE_DERIVED" if kind == "LIST" else "PARAGRAPH_BOUNDARY_DERIVED")
            output.append(MemberGroup(kind, tuple(current), tuple(sorted(fs))))

    item = -1
    indents = []
    for index, line in enumerate(chain):
        marker = _MARKER.match(line.text or "")
        begins = _list_start(line, chain, index) or bool(kind == "LIST" and marker)
        continues = kind == "LIST" and not begins and line.bbox[0] > base
        desired = "LIST" if begins or continues else "PROSE"
        if current and (desired != kind or line.kind == "HEADING" or current[-1].kind == "HEADING"):
            finish()
            current = []
            item = -1
            indents = []
        kind = desired
        if kind == "LIST":
            if not current:
                base = line.bbox[0]
            if begins:
                item += 1
                while indents and line.bbox[0] < indents[-1]:
                    indents.pop()
                if not indents or line.bbox[0] > indents[-1]:
                    indents.append(line.bbox[0])
            line = replace(line, list_item=max(item, 0), list_depth=max(len(indents) - 1, 0))
        current.append(line)
    finish()
    return output


def group_lines(lines, pages, policy=BoundaryPolicy(), *, barriers=(), continuations=()):
    lines = tuple(sorted(lines, key=order))
    pages = {p.page: p for p in pages}
    if policy.mode == "line":
        return tuple(MemberGroup("PROSE", (m,), ("LINE_BASELINE",)) for m in lines)
    if policy.mode == "section":
        groups = defaultdict(list)
        for m in lines:
            groups[(m.section_id, m.member_id if m.kind == "HEADING" else None)].append(m)
        return tuple(
            MemberGroup("PROSE", tuple(ms), ("SECTION_BASELINE",)) for ms in groups.values()
        )
    barriers = tuple(barriers) + tuple((m.page, m.bbox) for m in lines if m.kind == "HEADING")
    edges, flags = _edges(lines, pages, policy, barriers, continuations)
    lookup = {m.member_id: m for m in lines}
    targeted = set(edges.values())
    groups = []
    visited = set()
    for line in lines:
        if line.member_id in targeted:
            continue
        chain = []
        node = line.member_id
        while node is not None:
            if node in visited:
                raise RegionRefused("CYCLIC_REGION_MEMBERSHIP")
            visited.add(node)
            chain.append(lookup[node])
            node = edges.get(node)
        groups.extend(_split_chain(chain, flags))
    if len(visited) != len(lines):
        raise RegionRefused("CYCLIC_REGION_MEMBERSHIP")
    return tuple(groups)
