"""Deterministic notation observations; no semantic roles or interpretation fields.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import re
import json
from .field_views import TextSpan, VerticalRule, HorizontalRule, line_views, stable_id, union_box

from .value_windows import (
    ContinuationPolicy,
    DEFAULT_CONTINUATION,
    ValueWindow,
    build_value_windows,
    window_from_dict,
    derive_continuation,
    policy_from_dict,
)

from ..harvest.numeric_locale import physical_numbers

from .failures import ConstructFinding

VERSION = "census/0.5"
IDENTIFIER = re.compile(r"(?<!\w)[A-Za-z0-9]+(?:[-/]+[A-Za-z0-9]+)+(?!\w)")
NUMBERING = re.compile(r"(?<![\w.])\d+(?:\.\s*\d+)+(?![\w.])")
CATEGORIES = frozenset(
    {
        "identifier",
        "numbering",
        "connector",
        "label",
        "table_header",
        "recurrence",
        "geometry",
        "notation",
        "language_token",
        "title",
        "page",
        "stamp",
    }
)
LANGUAGE_WORDS = {
    "fr": frozenset("le la les des une dans avec pour entre et est doit".split()),
    "en": frozenset("the a an with for between and is shall must from".split()),
}
NUMBER_WORDS = frozenset(
    "one two three four five six seven eight nine ten un une deux trois quatre cinq six sept huit neuf dix".split()
)


def identifiers(text):
    return tuple(
        m
        for m in IDENTIFIER.finditer(text)
        if re.search(r"[A-Za-z]", m[0]) and re.search(r"\d", m[0])
    )


def shape(raw):
    return re.sub(
        r"[A-Za-z]+|\d+",
        lambda m: ("L" if m[0].isalpha() else "D") + "{" + str(len(m[0])) + "}",
        raw,
    )


@dataclass(frozen=True)
class Evidence:
    source_id: str
    page: int
    span_id: str
    start: int
    end: int
    literal: str
    bbox: tuple[float, float, float, float]

    def __post_init__(self):
        if (
            not self.source_id
            or not self.span_id
            or self.page < 1
            or self.start < 0
            or self.end < self.start
            or self.end - self.start != len(self.literal)
        ):
            raise ValueError("invalid exact evidence span")


@dataclass(frozen=True)
class TableObservation:
    table_id: str
    page: int
    headers: tuple[str, ...]
    rows: tuple[tuple[str | None, ...], ...]
    evidence: tuple[Evidence, ...]
    geometry: str = "drawn_grid"

    def __post_init__(self):
        if (
            not self.table_id
            or not self.headers
            or not self.evidence
            or self.geometry not in {"drawn_grid", "whitespace_alignment"}
            or any(len(row) != len(self.headers) for row in self.rows)
        ):
            raise ValueError("table topology/evidence required")


@dataclass(frozen=True)
class PageDigest:
    page: int
    width: float
    height: float
    spans: tuple[TextSpan, ...]
    rules: tuple[VerticalRule, ...] = ()
    tables: tuple[TableObservation, ...] = ()
    drawing_count: int = 0
    horizontal_rules: tuple[HorizontalRule, ...] = ()

    def __post_init__(self):
        if (
            self.page < 1
            or self.width <= 0
            or self.height <= 0
            or self.drawing_count < 0
            or any(s.page != self.page for s in self.spans)
            or any(t.page != self.page for t in self.tables)
        ):
            raise ValueError("invalid page digest")


@dataclass(frozen=True)
class DocumentDigest:
    source_id: str
    pages: tuple[PageDigest, ...]
    extractor: str
    extractor_version: str

    def __post_init__(self):
        if (
            not self.source_id
            or not self.extractor
            or not self.extractor_version
            or not self.pages
            or tuple(p.page for p in self.pages) != tuple(range(1, len(self.pages) + 1))
            or any(s.source_id != self.source_id for p in self.pages for s in p.spans)
        ):
            raise ValueError("complete ordered page inventory and extractor identity required")


@dataclass(frozen=True)
class CensusRecord:
    candidate_id: str
    category: str
    literal: str
    count: int
    evidence: tuple[Evidence, ...]
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if (
            self.category not in CATEGORIES
            or type(self.count) is not int
            or self.count < 1
            or self.count != len(self.evidence)
        ):
            raise ValueError("census counts require every occurrence")
        allowed = {
            "shape",
            "zone",
            "adjacent",
            "revision_tail",
            "depth",
            "marker",
            "split",
            "follow",
            "separator",
            "pattern",
            "rotated",
            "large",
            "edge",
            "date_capitals",
            "header",
            "geometry",
            "empty",
            "unreadable",
            "columns",
            "rows",
            "id_columns",
            "text_columns",
            "kind",
            "code",
            "text_layer",
            "drawings",
            "edge_fraction",
            "large_points",
            "rotation_threshold",
            "value_position",
            "window_line_count",
            "basis",
            "personal_data_suspected",
            "matched_spans",
        }
        if any(k not in allowed or not isinstance(v, str) for k, v in self.attributes):
            raise ValueError("census interpretation fields forbidden")


@dataclass(frozen=True)
class Census:
    source_id: str
    records: tuple[CensusRecord, ...]
    pages: int
    digest_id: str
    continuation_policy: ContinuationPolicy = DEFAULT_CONTINUATION
    windows: tuple[ValueWindow, ...] = ()
    version: str = VERSION
    construct_findings: tuple[ConstructFinding, ...] = ()
    geometry_policy: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.version != VERSION or not self.source_id or not self.digest_id or self.pages < 1:
            raise ValueError("invalid census version/identity")
        if len({r.candidate_id for r in self.records}) != len(self.records):
            raise ValueError("duplicate census id")


@dataclass(frozen=True)
class CensusConfig:
    edge_fraction: float = 0.08
    large_points: float = 20
    adjacency_chars: int = 32
    rotation_threshold: float = 0.1
    recurrence_fraction: float = 0.5
    continuation_policy: ContinuationPolicy = DEFAULT_CONTINUATION

    def __post_init__(self):
        if (
            not 0 < self.edge_fraction < 0.5
            or self.large_points <= 0
            or self.adjacency_chars < 0
            or not 0 < self.rotation_threshold <= 1
            or not 0 < self.recurrence_fraction <= 1
        ):
            raise ValueError("invalid census configuration")


def page_lines(page):
    """Geometry views remain UNKNOWN until profile classification; no hidden policy."""
    return line_views(page.spans, page.rules)


def census(document: DocumentDigest, config=CensusConfig()) -> Census:
    buckets = defaultdict(list)
    windows = []
    physical_evidence = []
    construct_findings = []
    observed_lines = []
    policy = derive_continuation(
        [
            [
                v
                for v in page_lines(p)
                if v.bbox[1] >= p.height * config.edge_fraction
                and v.bbox[3] <= p.height * (1 - config.edge_fraction)
            ]
            for p in document.pages
        ],
        config.continuation_policy,
    )

    def emit(category, literal, evidence, **attrs):
        buckets[(category, literal, tuple(sorted((k, str(v)) for k, v in attrs.items())))].append(
            evidence
        )

    for page in document.pages:
        page_ev = Evidence(
            document.source_id,
            page.page,
            f"page:{page.page}",
            0,
            0,
            "",
            (0, 0, page.width, page.height),
        )
        emit("page", "page", page_ev, text_layer=bool(page.spans), drawings=page.drawing_count)
        lines = page_lines(page)
        edge_ids = {
            v.view_id
            for v in lines
            if v.bbox[1] < page.height * config.edge_fraction
            or v.bbox[3] > page.height * (1 - config.edge_fraction)
        }
        page_windows = build_value_windows(
            lines,
            identifiers=identifiers,
            policy=policy,
            vertical_rules=page.rules,
            horizontal_rules=page.horizontal_rules,
            table_headers=tuple(h for t in page.tables for h in (*t.headers, " ".join(t.headers))),
            edge_ids=edge_ids,
            findings=construct_findings,
        )
        windows.extend(page_windows)
        window_by_label = {w.views[0].view_id: w for w in page_windows}
        for index, line in enumerate(lines):
            text = line.text

            def ev(start=0, end=None):
                end = len(text) if end is None else end
                return Evidence(
                    document.source_id,
                    page.page,
                    line.view_id,
                    start,
                    end,
                    text[start:end],
                    line.bbox,
                )

            edge = line.bbox[1] < page.height * config.edge_fraction or line.bbox[
                3
            ] > page.height * (1 - config.edge_fraction)
            observed_lines.append((ev(), edge))
            emit(
                "recurrence",
                re.sub(r"\d+", "#", text.strip()),
                ev(),
                edge=edge,
                edge_fraction=config.edge_fraction,
                large_points=config.large_points,
                rotation_threshold=config.rotation_threshold,
            )
            for m in identifiers(text):
                emit(
                    "identifier",
                    m[0],
                    ev(m.start(), m.end()),
                    shape=shape(m[0]),
                    zone="edge" if edge else "body",
                    adjacent=text[max(0, m.start() - config.adjacency_chars) : m.start()].strip(),
                    revision_tail=text[m.end() : m.end() + config.adjacency_chars].strip(),
                )
            physical = physical_numbers(text)
            numbers = [
                m
                for m in NUMBERING.finditer(text)
                if not any(m.start() < p.end and m.end() > p.start for p in physical)
            ]
            for m in numbers:
                prefix = text[: m.start()]
                marker = (
                    re.search(r"(?:[^\W\d_]+|§)\s*$", prefix)
                    if not any(n.end() <= m.start() for n in numbers)
                    else None
                )
                emit(
                    "numbering",
                    m[0],
                    ev(m.start(), m.end()),
                    depth=m[0].count(".") + 1,
                    marker=marker[0].strip() if marker else "",
                    split=bool(re.search(r"\.\s+", m[0])),
                )
            for left, right in zip(numbers, numbers[1:]):
                gap = text[left.end() : right.start()].strip()
                emit("connector", gap, ev(left.end(), right.start()), pattern="between_numbering")
            window = window_by_label.get(line.view_id)
            if window:
                following = window.following_text
                follow = (
                    "identifier-bearing"
                    if identifiers(following)
                    else (
                        "number-bearing"
                        if re.search(r"\d", following)
                        else "free-text" if following.strip() else "empty"
                    )
                )
                emit(
                    "label",
                    window.label,
                    ev(0, window.label_end),
                    follow=follow,
                    separator=window.separator,
                    value_position=window.value_position,
                    window_line_count=len(window.views),
                )
            for m in re.finditer(
                r"\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?|[±≤≥<>]|\+/-|[^\W\d_]+", text
            ):
                raw = m[0]
                kind = None
                if re.search(r"\d[.,]\d", raw):
                    kind = "decimal_token"
                elif re.search(r"\d[ \u00a0\u202f]\d", raw):
                    kind = "grouping"
                elif raw in {"±", "+/-", "≤", "≥", "<", ">"}:
                    kind = "operator"
                elif raw.casefold() in NUMBER_WORDS:
                    kind = "number_word"
                if kind:
                    emit("notation", raw, ev(m.start(), m.end()), kind=kind)
                for code, words in LANGUAGE_WORDS.items():
                    if raw.casefold() in words:
                        emit("language_token", raw, ev(m.start(), m.end()), code=code)
            # Unit/comparator vocabulary is observation, not a role assignment.
            from ..harvest.quantitative import SCALAR, PREFIX

            for observed in physical:
                evidence = ev(observed.start, observed.end)
                evidence = Evidence(
                    evidence.source_id,
                    evidence.page,
                    evidence.span_id,
                    evidence.start,
                    evidence.end,
                    evidence.literal,
                    union_box(r.bbox for r in line.source_refs(observed.start, observed.end)),
                )
                physical_evidence.append(evidence)
                emit(
                    "notation",
                    observed.raw,
                    evidence,
                    kind="decimal" if any(c in observed.raw for c in ".,") else "physical_integer",
                    basis=observed.basis,
                )
            for m in SCALAR.finditer(text):
                if not any(
                    m.start("number") == p.start and m.end("number") == p.end for p in physical
                ):
                    continue
                emit("notation", m["unit"], ev(m.start("unit"), m.end("unit")), kind="unit")
                op = PREFIX.search(text[: m.start()])
                if op:
                    emit("notation", op["op"], ev(op.start(), op.end()), kind="operator")
        for span in page.spans:
            rotated = abs(span.direction[1]) > config.rotation_threshold
            large = span.font_size >= config.large_points
            edge = span.bbox[1] < page.height * config.edge_fraction or span.bbox[
                3
            ] > page.height * (1 - config.edge_fraction)
            stamp = bool(
                re.search(r"\d{2}[/.-]\d{2}[/.-]\d{4}", span.text)
                and re.search(r"\b[A-Z][a-z]+", span.text)
            )
            evidence = Evidence(
                document.source_id, page.page, span.span_id, 0, len(span.text), span.text, span.bbox
            )
            if rotated or large or stamp:
                emit(
                    "geometry",
                    span.text,
                    evidence,
                    rotated=rotated,
                    large=large,
                    edge=edge,
                    date_capitals=stamp,
                )
            if page.page == 1 and large:
                emit("title", span.text, evidence)
        for table in page.tables:
            empty = tuple(
                sum(row[i] == "" for row in table.rows) for i in range(len(table.headers))
            )
            unreadable = tuple(
                sum(row[i] is None for row in table.rows) for i in range(len(table.headers))
            )
            id_columns = [
                i
                for i in range(len(table.headers))
                if table.rows
                and all(
                    cell is not None
                    and (bool(identifiers(cell)) or bool(NUMBERING.fullmatch(cell)))
                    for row in table.rows
                    for cell in [row[i]]
                    if cell != ""
                )
                and any(row[i] not in (None, "") for row in table.rows)
            ]
            for row_index, row in enumerate(table.rows):
                for col, cell in enumerate(row):
                    if col in id_columns or cell is None:
                        continue
                    cell_evidence = table_cell_evidence(document.source_id, table, row_index, col)
                    for observed in physical_numbers(cell, table_cell=True):
                        box = cell_evidence.bbox
                        duplicate = any(
                            e.page == table.page
                            and e.literal == observed.raw
                            and min(e.bbox[2], box[2]) > max(e.bbox[0], box[0])
                            and min(e.bbox[3], box[3]) > max(e.bbox[1], box[1])
                            for e in physical_evidence
                        )
                        if duplicate:
                            continue
                        evidence = Evidence(
                            document.source_id,
                            table.page,
                            cell_evidence.span_id,
                            observed.start,
                            observed.end,
                            observed.raw,
                            box,
                        )
                        emit(
                            "notation",
                            observed.raw,
                            evidence,
                            kind=(
                                "decimal"
                                if any(c in observed.raw for c in ".,")
                                else "physical_integer"
                            ),
                            basis="table_cell",
                        )
            text_columns = [
                i
                for i in range(len(table.headers))
                if i not in id_columns
                and any(
                    isinstance(row[i], str) and re.search(r"[^\W\d_]", row[i]) for row in table.rows
                )
            ]
            # Each table occurrence counts once; all cell provenance stays in the digest.

            emit(
                "table_header",
                json.dumps(table.headers, ensure_ascii=False),
                table.evidence[0],
                geometry=table.geometry,
                empty=json.dumps(empty),
                unreadable=json.dumps(unreadable),
                columns=len(table.headers),
                rows=len(table.rows),
                id_columns=json.dumps(id_columns),
                text_columns=json.dumps(text_columns),
            )
    from .privacy import stamp_matches

    recurrence_pages = defaultdict(set)
    for evidence, _ in observed_lines:
        recurrence_pages[(re.sub(r"\d+", "#", evidence.literal), round(evidence.bbox[1], 3))].add(
            evidence.page
        )
    for evidence, edge in observed_lines:
        repeated = len(
            recurrence_pages[(re.sub(r"\d+", "#", evidence.literal), round(evidence.bbox[1], 3))]
        ) >= max(2, len(document.pages) * config.recurrence_fraction)
        matches = stamp_matches(evidence.literal, edge=edge, recurring=repeated)
        if matches:
            emit(
                "stamp",
                evidence.literal,
                evidence,
                personal_data_suspected=True,
                matched_spans=json.dumps(matches, ensure_ascii=False, sort_keys=True),
            )
    records = []
    for (category, literal, attrs), evidence in sorted(buckets.items()):
        evidence = tuple(sorted(evidence, key=lambda e: (e.page, e.span_id, e.start, e.end)))
        ident = stable_id(
            VERSION,
            document.source_id,
            category,
            literal,
            attrs,
            [(e.page, e.span_id, e.start, e.end) for e in evidence],
        )
        records.append(CensusRecord(ident, category, literal, len(evidence), evidence, attrs))
    from ..harvest.report import replay_digest

    return Census(
        document.source_id,
        tuple(records),
        len(document.pages),
        replay_digest([document]),
        policy,
        tuple(windows),
        construct_findings=tuple(construct_findings),
        geometry_policy={
            key: getattr(config, key)
            for key in (
                "edge_fraction",
                "large_points",
                "rotation_threshold",
                "recurrence_fraction",
            )
        },
    )


def digest_from_dict(data):
    pages = []
    for p in data["pages"]:
        if "horizontal_rules" not in p:
            raise ValueError("digest lacks horizontal-rule observations; re-extract before Slice 2")
        spans = tuple(
            TextSpan(
                **{
                    **s,
                    "bbox": tuple(s["bbox"]),
                    "glyph_boxes": tuple(tuple(b) for b in s.get("glyph_boxes", ())),
                    "origin": tuple(s.get("origin", (0, 0))),
                    "direction": tuple(s.get("direction", (1, 0))),
                    "flags": tuple(s.get("flags", ())),
                }
            )
            for s in p["spans"]
        )
        tables = tuple(
            TableObservation(
                **{
                    **t,
                    "headers": tuple(t["headers"]),
                    "rows": tuple(tuple(r) for r in t["rows"]),
                    "evidence": tuple(Evidence(**e) for e in t["evidence"]),
                }
            )
            for t in p.get("tables", ())
        )
        pages.append(
            PageDigest(
                **{
                    **p,
                    "spans": spans,
                    "tables": tables,
                    "rules": tuple(VerticalRule(**r) for r in p.get("rules", ())),
                    "horizontal_rules": tuple(
                        HorizontalRule(**r) for r in p.get("horizontal_rules", ())
                    ),
                }
            )
        )
    return DocumentDigest(**{**data, "pages": tuple(pages)})


def census_from_dict(data):
    policy = policy_from_dict(data["continuation_policy"])
    return Census(
        **{
            **data,
            "continuation_policy": policy,
            "construct_findings": tuple(
                ConstructFinding(**f) for f in data.get("construct_findings", ())
            ),
            "windows": tuple(window_from_dict(w, policy) for w in data["windows"]),
            "records": tuple(
                CensusRecord(
                    **{
                        **r,
                        "evidence": tuple(Evidence(**e) for e in r["evidence"]),
                        "attributes": tuple(tuple(a) for a in r["attributes"]),
                    }
                )
                for r in data["records"]
            ),
        }
    )


def table_cell_evidence(source_id, table, row_index, column):
    """Reuse exact cell provenance when supplied; otherwise retain table scope."""
    raw = table.rows[row_index][column]
    if raw is None:
        raise ValueError("unreadable cell has no numeric literal")
    suffix = f":{row_index+1}:{column}"
    found = [e for e in table.evidence if e.span_id.endswith(suffix) and e.literal == raw]
    if len(found) == 1:
        return found[0]
    return Evidence(
        source_id,
        table.page,
        f"{table.table_id}:cell:{row_index}:{column}",
        0,
        len(raw),
        raw,
        union_box(e.bbox for e in table.evidence),
    )
