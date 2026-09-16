"""Shared, bounded label-value windows over immutable line observations.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass, replace
from collections import Counter
import math
import re
from .field_views import TextView, stable_id, union_box, view_from_dict


@dataclass(frozen=True)
class ContinuationPolicy:
    max_gap_ratio: float = 2.5
    max_lines: int = 8
    version: str = "value-window/0.2"
    source: str = "default"
    gap_histogram: tuple[tuple[float, int], ...] = ()
    derivation_reason: str | None = None

    def __post_init__(self):
        if (
            isinstance(self.max_gap_ratio, bool)
            or not math.isfinite(self.max_gap_ratio)
            or self.max_gap_ratio <= 0
            or type(self.max_lines) is not int
            or self.max_lines < 1
            or self.version != "value-window/0.2"
            or self.source not in {"derived", "default", "amended"}
        ):
            raise ValueError("invalid continuation policy")


DEFAULT_CONTINUATION = ContinuationPolicy()


def derive_continuation(page_line_sets, fallback=DEFAULT_CONTINUATION):
    """Use the valley between two compact, repeatedly observed gap modes.

    Histogram bins are milliline-height ratios. A split needs at least two
    observations on each side and an empty interval wider than either mode's
    spread; otherwise the evidence does not establish bimodality. No fitting to
    a supplied template or semantic label occurs.
    """
    page_line_sets = tuple(tuple(lines) for lines in page_line_sets)
    ratios = []
    for lines in page_line_sets:
        for before, after in zip(lines, lines[1:]):
            if before.page != after.page or _same_row(before, after):
                continue
            if before.bbox[2] < after.bbox[0] or after.bbox[2] < before.bbox[0]:
                continue
            gap = after.bbox[1] - before.bbox[3]
            height = before.bbox[3] - before.bbox[1]
            if gap >= 0 and height > 0:
                ratios.append(round(gap / height, 3))
    histogram = tuple(sorted(Counter(ratios).items()))
    if fallback.source == "amended":
        return replace(fallback, gap_histogram=histogram, derivation_reason=None)
    ordered = sorted(ratios)
    choices = []
    for i in range(2, len(ordered) - 1):
        gap = round(ordered[i] - ordered[i - 1], 3)
        if gap > max(round(ordered[i - 1] - ordered[0], 3), round(ordered[-1] - ordered[i], 3)):
            choices.append((gap, i))
    if not choices:
        return replace(
            fallback,
            source="default",
            gap_histogram=histogram,
            derivation_reason=(
                "no_body_lines"
                if not any(page_line_sets)
                else "too_few_gaps" if len(ratios) < 4 else "unimodal"
            ),
        )
    gap, i = max(choices)
    return replace(
        fallback,
        max_gap_ratio=(ordered[i - 1] + ordered[i]) / 2,
        source="derived",
        gap_histogram=histogram,
        derivation_reason=None,
    )


@dataclass(frozen=True)
class ValueWindow:
    window_id: str
    label: str
    label_end: int
    value_start: int
    separator: str
    views: tuple[TextView, ...]
    policy: ContinuationPolicy
    value_position: str
    stop_reason: str
    stop_view_id: str | None
    flags: tuple[str, ...] = ()
    needs_review: bool = False
    grid_cells: tuple[tuple[float, float, float, float], ...] = ()

    def __post_init__(self):
        if (
            not self.views
            or not self.label
            or not self.window_id
            or len({(v.source_id, v.page) for v in self.views}) != 1
            or not 0 < self.label_end <= self.value_start <= len(self.views[0].text)
            or self.value_position not in {"same_line", "next_line", "next_cell", "none"}
            or len(self.views) - 1 > self.policy.max_lines
            or self.needs_review != bool(self.flags)
            or (
                self.stop_reason in {"gap_bound", "line_bound"}
                and "WINDOW_BOUND_HIT" not in self.flags
            )
            or self.stop_reason
            not in {
                "page_end",
                "label",
                "table_header",
                "table_caption",
                "numbered_heading",
                "line_bound",
                "page_break",
                "horizontal_rule",
                "gap_bound",
                "column_break",
                "cell_boundary",
            }
        ):
            raise ValueError("invalid label value window")

    @property
    def following_text(self):
        return "\n".join(
            (self.views[0].text[self.value_start :], *(v.text for v in self.views[1:]))
        )


def _same_row(a, b):
    return min(a.bbox[3], b.bbox[3]) > max(a.bbox[1], b.bbox[1])


def _next_cell(a, b, vertical_rules):
    return (
        _same_row(a, b)
        and b.bbox[0] >= a.bbox[2]
        and sum(
            a.bbox[2] <= r.x <= b.bbox[0]
            and r.top < min(a.bbox[3], b.bbox[3])
            and r.bottom > max(a.bbox[1], b.bbox[1])
            for r in vertical_rules
        )
        == 1
    )


def _boundary(anchor, previous, candidate, policy, vertical_rules, horizontal_rules):
    if (candidate.source_id, candidate.page) != (anchor.source_id, anchor.page):
        return "page_break"
    if any(r.between(previous.bbox, candidate.bbox) for r in horizontal_rules):
        return "horizontal_rule"
    if _next_cell(previous, candidate, vertical_rules):
        return None
    height = max(anchor.bbox[3] - anchor.bbox[1], 1e-9)
    if candidate.bbox[1] - previous.bbox[3] > policy.max_gap_ratio * height:
        return "gap_bound"
    if (
        candidate.bbox[1] < previous.bbox[1]
        or candidate.bbox[2] < previous.bbox[0]
        or candidate.bbox[0] > previous.bbox[2] + height
        or any(r.crosses(union_box((previous.bbox, candidate.bbox))) for r in vertical_rules)
    ):
        return "column_break"
    if _same_row(previous, candidate):
        return "column_break"
    return None


def _heading(text):
    match = re.match(r"^\s*\d+(?:\.\d+)*[.)]?\s+([^\W\d_]+)", text)
    return bool(match and match[1].casefold() not in {"à", "to", "through", "et", "and"})


def build_value_windows(
    lines,
    *,
    identifiers,
    policy=DEFAULT_CONTINUATION,
    vertical_rules=(),
    horizontal_rules=(),
    table_headers=(),
    edge_ids=frozenset(),
    findings=None,
):
    """Build once at census time; every reader uses these same sealed objects.

    Colon labels are literal observations. Existing plain-label candidacy remains
    restricted to a locally adjacent identifier; no new ranking rule is added.
    """
    lines = tuple(v for v in lines if v.text.strip())
    headers = set(table_headers)
    labels = {}
    for index, line in enumerate(lines):
        if line.text.strip() in headers:
            continue
        match = re.match(r"([^:\n]{0,100}):\s*", line.text)
        if match and not match[1].strip():
            if findings is not None:
                from .failures import construct_finding

                findings.append(
                    construct_finding(
                        line.source_id,
                        "census",
                        "EMPTY_LABEL",
                        page=line.page,
                        span_id=line.view_id,
                    )
                )
            continue
        if match:
            labels[index] = (match[1].strip(), match.end(1), match.end(), ":")
        elif (
            index + 1 < len(lines)
            and line.view_id not in edge_ids
            and not identifiers(line.text)
            and not re.search(r"\d", line.text)
            and identifiers(lines[index + 1].text)
            and not _boundary(
                line, line, lines[index + 1], policy, vertical_rules, horizontal_rules
            )
        ):
            labels[index] = (line.text.strip(), len(line.text), len(line.text), "newline")
    windows = []
    for index, (label, label_end, value_start, separator) in labels.items():
        anchor = lines[index]
        cell_data = _grid_members(anchor, lines, vertical_rules, horizontal_rules)
        grid_cells = () if cell_data is None else cell_data[0]
        views = [anchor]
        flags = []
        stop = "page_end"
        stopped = None
        if cell_data is not None:
            stop = "cell_boundary"
            for candidate in cell_data[1]:
                stopped = candidate.view_id
                candidate_index = next(
                    i for i, v in enumerate(lines) if v.view_id == candidate.view_id
                )
                if candidate_index in labels or candidate.text.strip() in headers:
                    stop = "label" if candidate_index in labels else "table_header"
                    break
                if len(views) - 1 >= policy.max_lines:
                    stop = "line_bound"
                    break
                views.append(candidate)
                stopped = None
        else:
            for candidate_index in range(index + 1, len(lines)):
                candidate = lines[candidate_index]
                stopped = candidate.view_id
                if candidate_index in labels:
                    stop = "label"
                    break
                if candidate.text.strip() in headers:
                    stop = "table_header"
                    break
                if re.match(r"^\s*(?:table|tableau|figure|fig\.)\s+\d", candidate.text, re.I):
                    stop = "table_caption"
                    break
                if _heading(candidate.text):
                    stop = "numbered_heading"
                    break
                if len(views) - 1 >= policy.max_lines:
                    stop = "line_bound"
                    break
                boundary = _boundary(
                    anchor, views[-1], candidate, policy, vertical_rules, horizontal_rules
                )
                if boundary:
                    stop = boundary
                    break
                ids = identifiers(candidate.text)
                if ids and re.search(r"[^\W\d_]", candidate.text[: ids[0].start()]):
                    flags.append("ROLE_LINE_UNDECIDABLE")
                views.append(candidate)
                stopped = None
            if (
                len(views) == 1
                and not anchor.text[value_start:].strip()
                and stop == "page_end"
                and any(
                    r.y >= anchor.bbox[3] and r.left < anchor.bbox[2] and r.right > anchor.bbox[0]
                    for r in horizontal_rules
                )
            ):
                if findings is not None:
                    from .failures import construct_finding

                    findings.append(
                        construct_finding(
                            anchor.source_id,
                            "census",
                            "INVALID_LABEL_WINDOW",
                            page=anchor.page,
                            span_id=anchor.view_id,
                        )
                    )
        first_value = next((v for v in views[1:] if v.text.strip()), None)
        position = (
            "same_line"
            if anchor.text[value_start:].strip()
            else (
                "next_cell"
                if first_value and (grid_cells or _next_cell(anchor, first_value, vertical_rules))
                else "next_line" if first_value else "none"
            )
        )
        if stop in {"gap_bound", "line_bound"}:
            flags.append("WINDOW_BOUND_HIT")
        identity = stable_id(
            "value-window/0.2",
            asdict(policy),
            label,
            label_end,
            value_start,
            [v.view_id for v in views],
            stop,
            stopped,
        )
        windows.append(
            ValueWindow(
                identity,
                label,
                label_end,
                value_start,
                separator,
                tuple(views),
                policy,
                position,
                stop,
                stopped,
                tuple(sorted(set(flags))),
                bool(flags),
                tuple(grid_cells),
            )
        )
    return tuple(windows)


def join_window(window):
    text = ""
    mapping = []
    for view in window.views:
        if text:
            text += "\n"
            mapping.append(None)
        text += view.text
        mapping.extend(view.mapping)
    return TextView(
        window.window_id,
        window.views[0].source_id,
        window.views[0].page,
        text,
        tuple(mapping),
        "UNKNOWN" if any(v.state == "UNKNOWN" for v in window.views) else "CONTENT",
        tuple(sorted({*window.flags, *(f for v in window.views for f in v.flags)})),
        union_box(v.bbox for v in window.views),
        producer="value-window/0.2",
    )


def window_from_dict(data, policy):
    supplied = policy_from_dict(data["policy"])
    if supplied != policy:
        raise ValueError("window/census policy mismatch")
    return ValueWindow(
        **{
            **data,
            "views": tuple(view_from_dict(v) for v in data["views"]),
            "policy": policy,
            "flags": tuple(data["flags"]),
            "grid_cells": tuple(tuple(b) for b in data.get("grid_cells", ())),
        }
    )


def policy_from_dict(data):
    return ContinuationPolicy(
        **{**data, "gap_histogram": tuple(tuple(v) for v in data.get("gap_histogram", ()))}
    )


def _cell_at(x, y, vertical, horizontal):
    xs = sorted({r.x for r in vertical if r.top <= y <= r.bottom})
    ys = sorted({r.y for r in horizontal if r.left <= x <= r.right})
    left = [v for v in xs if v < x]
    right = [v for v in xs if v > x]
    top = [v for v in ys if v < y]
    bottom = [v for v in ys if v > y]
    if not all((left, right, top, bottom)):
        return None
    box = (left[-1], top[-1], right[0], bottom[0])
    if not all(
        any(r.y == yy and r.left <= box[0] and r.right >= box[2] for r in horizontal)
        for yy in (box[1], box[3])
    ):
        return None
    return box


def _grid_members(anchor, lines, vertical, horizontal):
    box = anchor.bbox
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    cell = _cell_at(x, y, vertical, horizontal)
    if cell is None or not (
        cell[0] <= box[0] <= box[2] <= cell[2] and cell[1] <= box[1] <= box[3] <= cell[3]
    ):
        return None
    cells = [cell]
    right_edges = sorted({r.x for r in vertical if r.x > cell[2] and r.top <= y <= r.bottom})
    if right_edges:
        adjacent = _cell_at((cell[2] + right_edges[0]) / 2, y, vertical, horizontal)
        if adjacent and adjacent[0] == cell[2]:
            cells.append(adjacent)
    lower_edges = sorted({r.y for r in horizontal if r.y > cell[3] and r.left <= x <= r.right})
    if lower_edges:
        adjacent = _cell_at(x, (cell[3] + lower_edges[0]) / 2, vertical, horizontal)
        if adjacent and adjacent[1] == cell[3] and adjacent not in cells:
            cells.append(adjacent)
    members = []
    seen = {anchor.view_id}
    for region in cells:
        for line in lines:
            b = line.bbox
            if (
                line.view_id not in seen
                and region[0] <= b[0] <= b[2] <= region[2]
                and region[1] <= b[1] <= b[3] <= region[3]
            ):
                members.append(line)
                seen.add(line.view_id)
    return tuple(cells), tuple(members)


def unresolved_reason(window):
    if "WINDOW_BOUND_HIT" in window.flags:
        return "WINDOW_BOUND_HIT"
    if window.stop_reason in {
        "label",
        "table_header",
        "table_caption",
        "numbered_heading",
        "horizontal_rule",
        "column_break",
    }:
        return "RULE_STOP"
    if window.grid_cells:
        return "EMPTY_CELL"
    return "NO_TEXT"
