"""Evidence-scoped negative records and a self-contained, escaped HTML report.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass, is_dataclass
from html import escape
import hashlib
import json
import math
import unicodedata


def canonical_data(value, key=""):
    """Semantic replay: NFC, geometry at millipoints, explicit volatile omissions.

    List order is significant (characters, rows, expression operands). Only maps
    and top-level record sets are sorted, never characters inside a mapping.
    """
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        if len({unicodedata.normalize("NFC", k) for k in value}) != len(value):
            raise ValueError("NFC key collision")
        return {
            unicodedata.normalize("NFC", k): canonical_data(v, k)
            for k, v in sorted(value.items())
            if not (
                key in {"", "_meta", "coverage", "provenance", "manifest"}
                and k in {"timestamp", "run_id", "wall_time", "latency_ms", "stage_times"}
            )
        }
    if isinstance(value, (list, tuple)):
        return [canonical_data(v, key) for v in value]
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite replay number")
        return (
            round(value, 3)
            if key in {"bbox", "glyph_boxes", "origin", "width", "height", "x", "top", "bottom"}
            else value
        )
    return value


def canonical_json(value):
    return json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def replay_digest(records):
    rows = sorted((canonical_json(row) for row in records))
    return hashlib.sha256(canonical_json(rows).encode()).hexdigest()


@dataclass(frozen=True)
class InsufficientEvidence:
    record_id: str
    question: str
    searched_scope: tuple[str, ...]
    rule_id: str
    inspected_count: int
    outcome: str = "NOT_ESTABLISHED"

    def __post_init__(self):
        if (
            not self.record_id
            or not self.question
            or not self.searched_scope
            or not all(self.searched_scope)
            or not self.rule_id
            or type(self.inspected_count) is not int
            or self.inspected_count < 1
            or self.outcome != "NOT_ESTABLISHED"
        ):
            raise ValueError(
                "negative record needs searched scope, rule and positive inspected count"
            )


@dataclass(frozen=True)
class ReadingCoverage:
    record_id: str
    source_id: str
    pages: int
    text_layer_pages: int
    tables_recovered: int
    tables_not_recovered: int
    unreadable_cells: int
    contaminated_lines: int
    unknown_template_fields: int
    model_calls_made: int = 0
    model_calls_cached: int = 0
    model_calls_refused: int = 0
    stage_times: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if not self.record_id or not self.source_id:
            raise ValueError("coverage identity required")
        for key, value in asdict(self).items():
            if key not in {"record_id", "source_id", "stage_times"} and (
                type(value) is not int or value < 0
            ):
                raise ValueError("nonnegative integer coverage required")
        if self.pages < 1 or self.text_layer_pages > self.pages:
            raise ValueError("invalid page coverage")


@dataclass(frozen=True)
class Report:
    source_id: str
    profile: dict
    findings: tuple[dict, ...]
    insufficient: tuple[InsufficientEvidence, ...]
    coverage: ReadingCoverage
    provenance: dict
    replay_digest: str


def build_report(document, census, profile, reading, provenance=None):
    from ..saqqara.field_views import stable_id

    negative = tuple(
        InsufficientEvidence(
            f.record_id,
            f.field,
            (document.source_id,),
            "profile/required-field/1",
            f.inspected_count,
        )
        for f in reading.findings
    )
    negative += tuple(
        InsufficientEvidence(
            stable_id("no-text", document.source_id, p.page),
            "text-layer reading",
            (document.source_id + ":" + str(p.page),),
            "coverage/text-layer/1",
            1,
        )
        for p in document.pages
        if not p.spans
    )
    all_tables = [t for p in document.pages for t in p.tables]
    negative += tuple(
        InsufficientEvidence(
            stable_id("unreadable", document.source_id, t.table_id),
            "table-cell reading",
            (document.source_id, t.table_id),
            "coverage/unreadable-cell/1",
            sum(len(r) for r in t.rows),
        )
        for t in all_tables
        if any(c is None for r in t.rows for c in r)
    )
    recovered = {t["table_id"] for t in reading.tables}
    coverage = ReadingCoverage(
        stable_id("coverage", document.source_id),
        document.source_id,
        len(document.pages),
        sum(bool(p.spans) for p in document.pages),
        len(recovered),
        len(all_tables) - len(recovered),
        sum(c is None for t in all_tables for r in t.rows for c in r),
        sum(bool(s.flags) for p in document.pages for s in p.spans),
        len(reading.findings),
    )
    findings = tuple(
        {"record_id": f.record_id, "kind": "field", "data": asdict(f)} for f in reading.fields
    )
    findings += tuple(
        {"record_id": q["candidate_id"], "kind": "quantity", "data": q} for q in reading.quantities
    )
    findings += tuple(
        {"record_id": t["record_id"], "kind": "table_row", "data": t} for t in reading.tables
    )
    provenance = {
        **(provenance or {}),
        "source_id": document.source_id,
        "extractor": document.extractor,
        "extractor_version": document.extractor_version,
    }
    payload = [
        asdict(profile),
        *findings,
        *[asdict(n) for n in negative],
        asdict(coverage),
        provenance,
    ]
    return Report(
        document.source_id,
        asdict(profile),
        findings,
        negative,
        coverage,
        provenance,
        replay_digest(payload),
    )


def render_report(report: Report, label_map: dict[str, str]) -> str:
    """All displayed numerical content is inside an element resolving to a record.

    Labels for found relation kinds are caller supplied. No inferred relation is
    reworded into proof. Untrusted source text is always HTML escaped.
    """
    kinds = {r["kind"] for r in report.findings}
    if not kinds <= set(label_map):
        raise ValueError("caller label map incomplete")

    def record(ident, value):
        return (
            '<pre id="'
            + escape(ident, quote=True)
            + '" data-record-id="'
            + escape(ident, quote=True)
            + '">'
            + escape(canonical_json(value))
            + "</pre>"
        )

    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Document reading report</title>',
        "<style>body{font:1rem system-ui;max-width:90rem;margin:2rem auto;padding:1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid #ccc;padding:1rem}h2{margin-top:2rem}</style>",
        "<header><h1>Document reading report</h1></header><main><h2>How the document was read</h2>",
        record("profile", report.profile),
        "<h2>What was found</h2>",
    ]
    for finding in report.findings:
        body.extend(
            (
                "<h3>" + escape(label_map[finding["kind"]]) + "</h3>",
                record(finding["record_id"], finding),
            )
        )
    body.append("<h2>What could not be established</h2>")
    for item in report.insufficient:
        body.append(record(item.record_id, asdict(item)))
    body.extend(
        (
            "<h2>Reading coverage</h2>",
            record(report.coverage.record_id, asdict(report.coverage)),
            "<h2>Provenance and replay</h2>",
            record("provenance", report.provenance),
            record("replay", report.replay_digest),
            "</main></html>",
        )
    )
    return "\n".join(body)
