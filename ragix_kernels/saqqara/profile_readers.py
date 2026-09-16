"""Profile-driven fields, generic identifiers and literal section expressions.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass, replace
import re
from .census import DocumentDigest, Census, identifiers, shape, NUMBERING, page_lines
from .field_views import TextView, stable_id, union_box
from .profile import DocumentProfile, derive_census_id
from ..harvest.quantitative import harvest
from ..harvest.report import replay_digest
from .value_windows import join_window, unresolved_reason
from .census import table_cell_evidence


@dataclass(frozen=True)
class UnknownTemplate:
    record_id: str
    source_id: str
    field: str
    census_id: str
    inspected_count: int
    reason: str = "UNKNOWN_TEMPLATE"
    page: int | None = None
    span_id: str | None = None

    def __post_init__(self):
        if (
            not all((self.record_id, self.source_id, self.field, self.census_id))
            or self.inspected_count < 1
        ):
            raise ValueError("unknown template requires identity, scope and inspected count")


@dataclass(frozen=True)
class SectionExpression:
    raw: str
    start: int
    end: int
    kind: str
    key_from: str | None
    key_to: str | None = None
    connector: str | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Target:
    raw: str
    start: int
    end: int
    family_id: str
    sections: tuple[SectionExpression, ...]
    revision_raw: str | None = None
    revision_start: int | None = None
    revision_end: int | None = None


@dataclass(frozen=True)
class FieldReading:
    record_id: str
    label: str
    view: TextView
    targets: tuple[Target, ...]
    status: str
    flags: tuple[str, ...] = ()
    needs_review: bool = False


@dataclass(frozen=True)
class ReaderResult:
    source_id: str
    profile_id: str
    fields: tuple[FieldReading, ...]
    quantities: tuple[dict, ...]
    tables: tuple[dict, ...]
    furniture: tuple[str, ...]
    findings: tuple[UnknownTemplate, ...]


def section_expressions(text, style, offset=0):
    """Keep range relations without expansion or invented interior keys.

    Only observed connectors are operators. Unknown connectors and split numbering
    remain uncertain. Digit suffixes cannot be disambiguated without an inventory.
    """
    matches = list(NUMBERING.finditer(text))
    result = []
    i = 0
    ranges = set(style["range_connectors"])
    lists = set(style["list_connectors"])
    markers = sorted((m for m in style["markers"] if m), key=len, reverse=True)

    def strip_marker(gap):
        for marker in markers:
            if gap.endswith(marker):
                return gap[: -len(marker)].strip()
        return gap

    while i < len(matches):
        m = matches[i]
        raw = m[0]
        split = bool(re.search(r"\.\s+", raw))
        key = None if split else raw
        if i + 1 < len(matches):
            n = matches[i + 1]
            gap = strip_marker(text[m.end() : n.start()].strip())
            if gap in ranges:
                other_split = bool(re.search(r"\.\s+", n[0]))
                flags = ("SPLIT_NUMBERING",) if split or other_split else ()
                result.append(
                    SectionExpression(
                        text[m.start() : n.end()],
                        offset + m.start(),
                        offset + n.end(),
                        "split" if flags else "range",
                        key,
                        None if other_split else n[0],
                        gap,
                        flags,
                    )
                )
                i += 2
                continue
            if gap not in lists and gap:
                result.append(
                    SectionExpression(
                        raw,
                        offset + m.start(),
                        offset + m.end(),
                        "single",
                        key,
                        flags=("CONNECTOR_UNRESOLVED",),
                    )
                )
                i += 1
                continue
        kind = "split" if split else "single"
        flags = ("SPLIT_NUMBERING",) if split else ()
        tail = text[m.end() :].strip()
        if tail in ranges:
            kind = "open_range"
            flags += ("OPEN_RANGE",)
        # A directly attached alphabetic fragment is an observed glue, never fixed.
        if m.end() < len(text) and text[m.end() : m.end() + 1].isalpha():
            kind = "glued"
            flags += ("GLUED_NUMBERING",)
            key = None
        result.append(
            SectionExpression(raw, offset + m.start(), offset + m.end(), kind, key, flags=flags)
        )
        i += 1
    return tuple(result)


def read_document(
    document: DocumentDigest, census: Census, profile: DocumentProfile
) -> ReaderResult:
    if (
        not document.source_id == census.source_id == profile.source_id
        or derive_census_id(census) != profile.census_id
        or replay_digest([document]) != census.digest_id
    ):
        raise ValueError("source/profile/census mismatch")
    known_ids = {r.candidate_id for r in census.records}
    if any(not set(f.evidence) <= known_ids for f in profile.fields.values()):
        raise ValueError("profile evidence outside census")
    fields = []
    quantities = []
    tables = []
    furniture = []
    findings = [
        UnknownTemplate(
            f.record_id,
            document.source_id,
            "reference_fields",
            profile.census_id,
            f.inspected_count,
            f.reason,
            f.page,
            f.span_id,
        )
        for f in census.construct_findings
    ]

    def unknown(field):
        if not any(f.field == field for f in findings):
            findings.append(
                UnknownTemplate(
                    stable_id("unknown", profile.census_id, field),
                    document.source_id,
                    field,
                    profile.census_id,
                    len(document.pages),
                )
            )

    def value(field):
        entry = profile.fields[field]
        if entry.status == "UNKNOWN":
            unknown(field)
            return None
        return entry.value

    reference = profile.fields["reference_fields"].value
    if not census.windows and not census.construct_findings and reference is None:
        unknown("reference_fields")
    numbering = value("numbering_style")
    families = value("identifier_families")
    furniture_rules = value("furniture")
    locale = profile.fields["numeric_locale"].value
    table_profile = profile.fields["id_row_tables"].value
    for page in document.pages:
        source_by_id = {s.span_id: s for s in page.spans}
        lines = []
        for line in page_lines(page):
            pattern = re.sub(r"\d+", "#", line.text.strip())
            refs = {r.span_id for r in line.mapping if r is not None}
            spans = [source_by_id[sid] for sid in refs]
            # Profiles never authorize deleting matching body text: enforce the
            # geometry condition again at the observation being classified.
            edge_fraction = furniture_rules["edge_fraction"] if furniture_rules else 0
            edge = line.bbox[1] < page.height * edge_fraction or line.bbox[3] > page.height * (
                1 - edge_fraction
            )
            mark = furniture_rules and any(
                abs(s.direction[1]) > furniture_rules["rotation_threshold"]
                and s.font_size >= furniture_rules["large_points"]
                for s in spans
            )
            is_furniture = furniture_rules and (
                (edge and pattern in furniture_rules["running_patterns"])
                or (mark and line.text in furniture_rules["mark_literals"])
            )
            if is_furniture:
                furniture.append(line.view_id)
                continue
            # A body position alone does not establish a content class.
            lines.append(line)
            quantities.extend(
                {**asdict(c), "needs_review": c.needs_review}
                for c in harvest(
                    line.text,
                    source_id=document.source_id,
                    node_id=line.view_id,
                    classification=line.state,
                    uncertainty=line.flags,
                    token_locale=True,
                    locale_prior=locale,
                )
            )
        if census.windows:
            if profile.continuation_policy != census.continuation_policy:
                raise ValueError("window policy differs from sealed census; rebuild census")
            for window in census.windows:
                if window.views[0].page != page.page:
                    continue
                if not identifiers(window.following_text):
                    # Invalid terminal windows already have a construct finding.
                    if not any(
                        f.span_id == window.views[0].view_id for f in census.construct_findings
                    ):
                        reason = (
                            unresolved_reason(window)
                            if not window.following_text.strip()
                            else "NON_IDENTIFIER_VALUE"
                        )
                        findings.append(
                            UnknownTemplate(
                                stable_id("reference-occurrence", window.window_id, reason),
                                document.source_id,
                                "reference_fields",
                                profile.census_id,
                                1,
                                reason,
                                page.page,
                                window.views[0].view_id,
                            )
                        )
                    continue
                view = join_window(window)
                label = window.label
                matches = identifiers(view.text)
                targets = []
                for j, m in enumerate(matches):
                    stop = matches[j + 1].start() if j + 1 < len(matches) else len(view.text)
                    start = m.end()
                    revision = None
                    revision_start = None
                    revision_end = None
                    styles = [
                        marker
                        for family in (families or [])
                        if family["shape"] == shape(m[0])
                        for marker in family["revision_markers"]
                    ]
                    if styles:
                        revision_match = re.match(
                            r"\s*(?:"
                            + "|".join(re.escape(s) for s in styles)
                            + r")\s*[:.]?\s*(\d+(?:\.\d+)*)",
                            view.text[start:stop],
                            re.I,
                        )
                        if revision_match:
                            revision = revision_match[1]
                            revision_start = start + revision_match.start(1)
                            revision_end = start + revision_match.end(1)
                            start += revision_match.end()
                    segments = (
                        section_expressions(view.text[start:stop], numbering, start)
                        if numbering
                        else ()
                    )
                    targets.append(
                        Target(
                            m[0],
                            m.start(),
                            m.end(),
                            stable_id("family", shape(m[0])),
                            segments,
                            revision,
                            revision_start,
                            revision_end,
                        )
                    )
                flags = window.flags + (
                    ()
                    if reference and window.label in reference["labels"]
                    else ("REFERENCE_CLASS_UNKNOWN",)
                )
                undecidable = bool(flags)
                fields.append(
                    FieldReading(
                        stable_id("reading", view.view_id),
                        label,
                        view,
                        tuple(targets),
                        "UNDECIDABLE" if undecidable else "READ",
                        flags,
                        bool(flags),
                    )
                )

        for table in page.tables:
            if table.cell_rows:
                continue  # Reconstructed below, after recurrence filtering.
            # Infer id column from all nonempty row cells; header language and
            # column order are never part of the identifier grammar.
            matches = (
                [t for t in table_profile["tables"] if t["headers"] == list(table.headers)]
                if table_profile
                else []
            )
            column_sets = {tuple(t["id_columns"]) for t in matches}
            id_cols = list(next(iter(column_sets))) if len(column_sets) == 1 else []
            if len(set(table.headers)) != len(table.headers):
                unknown("id_row_tables")
                continue
            if len(id_cols) != 1:
                unknown("id_row_tables")
                continue
            col = id_cols[0]
            for index, row in enumerate(table.rows):
                for column, cell in enumerate(row):
                    if column == col or cell is None:
                        continue
                    evidence = table_cell_evidence(document.source_id, table, index, column)
                    # Unitless cell readings are only added when not already read
                    # from the physical text layer in this exact cell scope.
                    cell_candidates = harvest(
                        cell,
                        source_id=document.source_id,
                        node_id=evidence.span_id,
                        classification="UNKNOWN",
                        token_locale=True,
                        locale_prior=locale,
                        table_cell=True,
                    )
                    duplicates = []
                    for c in cell_candidates:
                        duplicate = False
                        for q in quantities:
                            if q["raw"] != c.raw:
                                continue
                            matching = next((v for v in lines if v.view_id == q["node_id"]), None)
                            if (
                                matching
                                and min(matching.bbox[2], evidence.bbox[2])
                                > max(matching.bbox[0], evidence.bbox[0])
                                and min(matching.bbox[3], evidence.bbox[3])
                                > max(matching.bbox[1], evidence.bbox[1])
                            ):
                                duplicate = True
                                break
                        duplicates.append(duplicate)
                    # Keep a cell batch whole: dropping a duplicate child alone
                    # would leave a composite referencing an absent candidate.
                    if cell_candidates and not all(duplicates):
                        quantities.extend(
                            {**asdict(c), "needs_review": c.needs_review} for c in cell_candidates
                        )
                tables.append(
                    {
                        "record_id": stable_id(table.table_id, index),
                        "table_id": table.table_id,
                        "page": table.page,
                        "key": row[col],
                        "cells": {h: row[i] for i, h in enumerate(table.headers)},
                        "flags": ["UNREADABLE_CELL"] if any(c is None for c in row) else [],
                        "evidence": [asdict(e) for e in table.evidence],
                    }
                )
    for table in census.table_analysis.tables:
        id_cols = [i for i, role in enumerate(table.roles) if role == "id"]
        for row in table.rows:
            flags = set(row.flags) | set(table.flags)
            if len(id_cols) != 1:
                flags.add("AMBIGUOUS_ID_COLUMN")
            duplicate_headers = len(set(table.headers)) != len(table.headers)
            columns = tuple(
                {
                    "column_id": stable_id("table-column", table.table_id, i),
                    "header": h,
                    "role": table.roles[i],
                }
                for i, h in enumerate(table.headers)
            )
            tables.append(
                {
                    "record_id": row.record_id,
                    "table_id": table.table_id,
                    "page": row.page,
                    "key": row.cells[id_cols[0]] if len(id_cols) == 1 else None,
                    "cells": {
                        columns[i]["column_id"] if duplicate_headers else h: row.cells[i]
                        for i, h in enumerate(table.headers)
                    },
                    "columns": columns,
                    "members": row.members,
                    "roles": table.roles,
                    "flags": sorted(flags),
                    "bbox": row.bbox,
                    "continued_on": table.continued_on,
                    "source_tables": table.fragments,
                    "evidence_ids": tuple(cid for members in row.members for cid in members),
                }
            )
    for failure in census.table_analysis.findings:
        findings.append(
            UnknownTemplate(
                failure.record_id,
                document.source_id,
                "id_row_tables",
                profile.census_id,
                failure.inspected_count,
                "TABLE_UNRESOLVED:" + failure.reason,
                failure.page,
                failure.table_id,
            )
        )
    return ReaderResult(
        document.source_id,
        replay_digest([profile]),
        tuple(fields),
        tuple(quantities),
        tuple(tables),
        tuple(furniture),
        tuple(findings),
    )
