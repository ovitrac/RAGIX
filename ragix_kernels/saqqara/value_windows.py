"""Shared, bounded label-value windows over immutable line observations.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass, replace
from collections import Counter
import math
import re
import unicodedata
from .field_views import TextView, stable_id, union_box, view_from_dict


def fold(text):
    """Case- and accent-insensitive text, with the source index of each character."""
    folded, index = [], []
    for i, char in enumerate(text):
        for c in unicodedata.normalize("NFKD", char):
            if not unicodedata.combining(c):
                for low in c.casefold():
                    folded.append(low)
                    index.append(i)
    return "".join(folded), index


@dataclass(frozen=True)
class ContinuationPolicy:
    max_gap_ratio: float = 2.5
    max_lines: int = 8
    version: str = "value-window/0.3"
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
            or self.version != "value-window/0.3"
            or self.source not in {"derived", "default", "amended"}
        ):
            raise ValueError("invalid continuation policy")


DEFAULT_CONTINUATION = ContinuationPolicy()


GENERIC_TYPE_WORDS = ("specification", "specifications", "spécification", "spécifications")
GENERIC_ROLE_WORDS = (
    "procedure",
    "procedures",
    "procédure",
    "procédures",
    "report",
    "reports",
    "rapport",
    "rapports",
    "document",
    "documents",
    "instruction",
    "instructions",
    "form",
    "forms",
    "formulaire",
    "formulaires",
    "see",
    "voir",
    "cf",
)


@dataclass(frozen=True)
class ReferencePolicy:
    """Declared numbers of the label-value reader; each is provenance, none a layout.

    `rule_tolerance` (points): parallel painted rules no farther apart than this are
    one edge. Producers paint each cell's own borders, so a shared border arrives
    as two rules a fraction of a point apart; the sliver between them is no cell.

    `underline_margin` (line heights): a rule in the lower half of a line's box, or
    on its bottom edge, that runs no farther than this beyond the line's ends
    underlines that line. An underline is emphasis, never an edge: it closes no
    window and bounds no cell. A table or cell edge runs past the text it bounds.

    `labels`: label phrases declared by the consumer. None is shipped: a label is
    data of the consumer's documents, and a document also supplies its own, the
    phrases it shows with a colon.

    `type_words`: words that, directly before an identifier, belong to the
    reference (a document-type noun). The shipped ones are language-generic; a
    domain's acronyms come from the consumer.

    `role_words`: words that, opening a line directly before an identifier, name
    another document's role (a procedure, a report, a cross-reference). Whether
    such a line belongs to the field cannot be decided from the page: it is listed
    and never read. A word cannot be both a type word and a role word.
    """

    rule_tolerance: float = 1.0
    underline_margin: float = 0.5
    labels: tuple[str, ...] = ()
    type_words: tuple[str, ...] = GENERIC_TYPE_WORDS
    role_words: tuple[str, ...] = GENERIC_ROLE_WORDS
    version: str = "reference-policy/0.1"

    def __post_init__(self):
        if (
            any(
                isinstance(v, bool) or not math.isfinite(v) or v < 0
                for v in (self.rule_tolerance, self.underline_margin)
            )
            or self.version != "reference-policy/0.1"
            or any(
                type(words) is not tuple
                or any(not isinstance(w, str) or not w.strip() for w in words)
                for words in (self.labels, self.type_words, self.role_words)
            )
            or {fold(w.strip())[0] for w in self.type_words}
            & {fold(w.strip())[0] for w in self.role_words}
        ):
            raise ValueError("invalid reference policy")


DEFAULT_REFERENCE = ReferencePolicy()


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
    undecidable: tuple[TextView, ...] = ()

    def __post_init__(self):
        if (
            not self.views
            or bool(self.undecidable) != ("ROLE_LINE_UNDECIDABLE" in self.flags)
            or any(
                (v.source_id, v.page) != (self.views[0].source_id, self.views[0].page)
                for v in self.undecidable
            )
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
    """Two boxes share a row when the shorter one's vertical centre lies in the taller.

    Overlap alone does not: the boxes of consecutive lines routinely overlap by a
    couple of points, and reading that as one row turns a wrapped value into a
    column break.
    """
    short, tall = sorted((a.bbox, b.bbox), key=lambda box: box[3] - box[1])
    return tall[1] < (short[1] + short[3]) / 2 < tall[3]


def underlines(rule, view, reference):
    """The rule is emphasis under this line, not an edge between lines or cells."""
    left, top, right, bottom = view.bbox
    margin = reference.underline_margin * (bottom - top)
    return (
        (top + bottom) / 2 < rule.y <= bottom + reference.rule_tolerance
        and rule.left >= left - margin
        and rule.right <= right + margin
    )


def _edges(values, tolerance):
    """Painted rules within the tolerance of an edge's first face are that edge.

    Anchoring on the first face keeps a run of close rules from drifting into one
    wide edge. An edge keeps both faces: (low, high).
    """
    edges = []
    for value in sorted(values):
        if edges and value - edges[-1][0] <= tolerance:
            edges[-1] = (edges[-1][0], value)
        else:
            edges.append((value, value))
    return edges


def _next_cell(a, b, vertical_rules, tolerance):
    return (
        _same_row(a, b)
        and b.bbox[0] >= a.bbox[2]
        and len(
            _edges(
                (
                    r.x
                    for r in vertical_rules
                    if a.bbox[2] <= r.x <= b.bbox[0]
                    and r.top < min(a.bbox[3], b.bbox[3])
                    and r.bottom > max(a.bbox[1], b.bbox[1])
                ),
                tolerance,
            )
        )
        == 1
    )


def _boundary(anchor, previous, candidate, policy, vertical_rules, horizontal_rules, tolerance):
    if (candidate.source_id, candidate.page) != (anchor.source_id, anchor.page):
        return "page_break"
    if any(r.between(previous.bbox, candidate.bbox) for r in horizontal_rules):
        return "horizontal_rule"
    if _next_cell(previous, candidate, vertical_rules, tolerance):
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


COLON_LABEL = re.compile(r"([^:\n]{0,100}):\s*")
BULLETS = " \t•●○◦▪▫■□►▶‣⁃∙·*-–—"


def colon_labels(lines, table_headers=()):
    """The label phrases a page shows with their colon: the document's own lexicon."""
    headers = set(table_headers)
    return tuple(
        match[1].strip()
        for line in lines
        if line.text.strip() not in headers
        for match in [COLON_LABEL.match(line.text)]
        if match and match[1].strip()
    )


def begins_with_reference(text, identifiers, type_words=()):
    """Reference content opens the text.

    An identifier, a declared type word directly before one, a section sign or a
    dotted number, or a quoted title. A label introduces a value; label words
    followed by anything else are prose.
    """
    text = text.lstrip()
    if re.match(r"§\s*\d|\d+(?:\.\d+)+|[«“„\"]", text):
        return True
    found = identifiers(text)
    if found and found[0].start() == 0:
        return True
    folded, index = fold(text)
    for word in type_words:
        key = fold(word)[0]
        if folded.startswith(key) and not folded[len(key) : len(key) + 1].isalnum():
            rest = re.sub(
                r"^\s*(?:n\s*[°º]|no\.?|#)?\s*[:\-–]?\s*", "", text[index[len(key) - 1] + 1 :]
            )
            found = identifiers(rest)
            if found and found[0].start() == 0:
                return True
    return False


def role_line(text, identifiers, role_words):
    """A declared role word opens the line, directly before an identifier."""
    text = text.lstrip(BULLETS)
    folded, index = fold(text)
    for word in role_words:
        key = fold(word.strip())[0]
        if folded.startswith(key) and not folded[len(key) : len(key) + 1].isalnum():
            rest = re.sub(
                r"^\.?\s*(?:[:：]|n\s*[°º]|no\.?|#)?\s*", "", text[index[len(key) - 1] + 1 :]
            )
            found = identifiers(rest)
            if found and found[0].start() == 0:
                return True
    return False


def _known_label(text, known):
    """(label, end offset) when a known label opens the line, bullets aside."""
    lead = len(text) - len(text.lstrip(BULLETS))
    folded, index = fold(text[lead:])
    for key, label in known:
        if folded.startswith(key) and not folded[len(key) : len(key) + 1].isalnum():
            return label, lead + index[len(key) - 1] + 1
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
    reference=DEFAULT_REFERENCE,
    known_labels=(),
    mentions=None,
):
    """Build once at census time; every reader uses these same sealed objects.

    Colon labels are literal observations. Existing plain-label candidacy remains
    restricted to a locally adjacent identifier; no new ranking rule is added.

    A label must introduce a value. `known_labels` are the phrases the document
    shows with a colon and those the consumer declares: without its colon, such a
    phrase opening a line is a label only when reference content follows; otherwise
    it is recorded in `mentions` as label words in prose and makes no window.
    """
    lines = tuple(v for v in lines if v.text.strip())
    # An underline is never an edge: it is set aside once, for every later test.
    horizontal_rules = tuple(
        r for r in horizontal_rules if not any(underlines(r, v, reference) for v in lines)
    )
    headers = set(table_headers)
    known = sorted(
        {(fold(label.strip())[0], label.strip()) for label in (*reference.labels, *known_labels)},
        key=lambda pair: (-len(pair[0]), pair),
    )
    labels = {}

    def role(view):
        return role_line(view.text, identifiers, reference.role_words)

    for index, line in enumerate(lines):
        if line.text.strip() in headers or role(line):
            continue
        match = COLON_LABEL.match(line.text)
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
            continue
        opening = None if line.view_id in edge_ids else _known_label(line.text, known)
        if opening and line.text[opening[1] :].strip():
            rest = line.text[opening[1] :]
            if begins_with_reference(rest, identifiers, reference.type_words):
                start = opening[1] + len(rest) - len(rest.lstrip())
                labels[index] = (opening[0], opening[1], start, "space")
            elif mentions is not None:
                mentions.append((line, opening[0], opening[1], "NO_REFERENCE_CONTENT"))
            continue
        if (
            index + 1 < len(lines)
            and line.view_id not in edge_ids
            and not identifiers(line.text)
            and not re.search(r"\d", line.text)
            and identifiers(lines[index + 1].text)
            and not role(lines[index + 1])
            and not _boundary(
                line,
                line,
                lines[index + 1],
                policy,
                vertical_rules,
                horizontal_rules,
                reference.rule_tolerance,
            )
        ):
            labels[index] = (
                opening[0] if opening else line.text.strip(),
                len(line.text),
                len(line.text),
                "newline",
            )
    windows = []
    for index, (label, label_end, value_start, separator) in labels.items():
        anchor = lines[index]
        cell_data = _grid_members(
            anchor, lines, vertical_rules, horizontal_rules, reference.rule_tolerance
        )
        grid_cells = () if cell_data is None else cell_data[0]
        views = [anchor]
        undecidable = []
        flags = []
        stop = "page_end"
        stopped = None

        def listed(candidate):
            """A role line, and every line after it up to the next stop, is never read."""
            if not undecidable and not role(candidate):
                return False
            if not undecidable:
                flags.append("ROLE_LINE_UNDECIDABLE")
            undecidable.append(candidate)
            return True

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
                if len(views) - 1 + len(undecidable) >= policy.max_lines:
                    stop = "line_bound"
                    break
                if not listed(candidate):
                    views.append(candidate)
                stopped = None
        else:
            previous = anchor
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
                if len(views) - 1 + len(undecidable) >= policy.max_lines:
                    stop = "line_bound"
                    break
                boundary = _boundary(
                    anchor,
                    previous,
                    candidate,
                    policy,
                    vertical_rules,
                    horizontal_rules,
                    reference.rule_tolerance,
                )
                if boundary:
                    stop = boundary
                    break
                if not listed(candidate):
                    views.append(candidate)
                previous = candidate
                stopped = None
            if (
                len(views) == 1
                and not undecidable
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
                if first_value
                and (
                    grid_cells
                    or _next_cell(anchor, first_value, vertical_rules, reference.rule_tolerance)
                )
                else "next_line" if first_value else "none"
            )
        )
        if stop in {"gap_bound", "line_bound"}:
            flags.append("WINDOW_BOUND_HIT")
        identity = stable_id(
            "value-window/0.3",
            asdict(policy),
            asdict(reference),
            label,
            label_end,
            value_start,
            [v.view_id for v in views],
            [v.view_id for v in undecidable],
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
                tuple(undecidable),
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
        producer="value-window/0.3",
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
            "undecidable": tuple(view_from_dict(v) for v in data.get("undecidable", ())),
        }
    )


def policy_from_dict(data):
    return ContinuationPolicy(
        **{**data, "gap_histogram": tuple(tuple(v) for v in data.get("gap_histogram", ()))}
    )


def reference_from_dict(data):
    return ReferencePolicy(
        **{
            **data,
            "labels": tuple(data["labels"]),
            "type_words": tuple(data["type_words"]),
            "role_words": tuple(data["role_words"]),
        }
    )


def _cell_at(x, y, vertical, horizontal, tolerance):
    """The ruled cell around a point: its inner box and the outer faces of its edges."""
    xs = _edges((r.x for r in vertical if r.top <= y <= r.bottom), tolerance)
    ys = _edges((r.y for r in horizontal if r.left <= x <= r.right), tolerance)
    left = [e for e in xs if e[1] < x]
    right = [e for e in xs if e[0] > x]
    top = [e for e in ys if e[1] < y]
    bottom = [e for e in ys if e[0] > y]
    if not all((left, right, top, bottom)):
        return None
    box = (left[-1][1], top[-1][1], right[0][0], bottom[0][0])
    if not all(
        any(
            low <= r.y <= high and r.left <= box[0] + tolerance and r.right >= box[2] - tolerance
            for r in horizontal
        )
        for low, high in (top[-1], bottom[0])
    ):
        return None
    return box, (left[-1][0], top[-1][0], right[0][1], bottom[0][1])


def _grid_members(anchor, lines, vertical, horizontal, tolerance):
    box = anchor.bbox
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    found = _cell_at(x, y, vertical, horizontal, tolerance)
    if found is None:
        return None
    cell, outer = found
    if not (cell[0] <= box[0] <= box[2] <= cell[2] and cell[1] <= box[1] <= box[3] <= cell[3]):
        return None
    cells = [cell]
    # A neighbour shares an edge: its inner face is the outer face of this cell's edge.
    right_edges = [
        e
        for e in _edges((r.x for r in vertical if r.top <= y <= r.bottom), tolerance)
        if e[0] > outer[2]
    ]
    if right_edges:
        adjacent = _cell_at((outer[2] + right_edges[0][0]) / 2, y, vertical, horizontal, tolerance)
        if adjacent and adjacent[0][0] == outer[2]:
            cells.append(adjacent[0])
    lower_edges = [
        e
        for e in _edges((r.y for r in horizontal if r.left <= x <= r.right), tolerance)
        if e[0] > outer[3]
    ]
    if lower_edges:
        adjacent = _cell_at(x, (outer[3] + lower_edges[0][0]) / 2, vertical, horizontal, tolerance)
        if adjacent and adjacent[0][1] == outer[3] and adjacent[0] not in cells:
            cells.append(adjacent[0])
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
    if window.undecidable:
        return "ROLE_LINE_UNDECIDABLE"
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
