"""Generated notation fixtures and adversarial Explorer gates X1–X12.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import subprocess
import sys
import pytest
from ragix_kernels.saqqara.census import (
    CensusRecord,
    Census,
    DocumentDigest,
    PageDigest,
    Evidence,
    TableObservation,
    census,
    digest_from_dict,
)
from ragix_kernels.saqqara.field_views import TextSpan, line_views
from ragix_kernels.saqqara.profile import ProfileField, ProfileReview, derive_profile, apply_reviews
from ragix_kernels.saqqara.profile_readers import read_document, section_expressions
from ragix_kernels.saqqara.explorer import explore, imported_provenance
from ragix_kernels.harvest.report import (
    InsufficientEvidence,
    canonical_json,
    replay_digest,
    render_report,
)
from ragix_kernels.harvest.profile_classify import (
    ClassificationConfig,
    classify_profile,
    validate_decisions,
    VERSION,
)


def line(text, page, index, y=100, x=40, size=10, direction=(1, 0)):
    boxes = tuple((x + 4 * i, y, x + 4 * (i + 1), y + size) for i in range(len(text)))
    return TextSpan(
        "synthetic",
        f"s:{page}:{index}",
        page,
        text,
        (x, y, x + 4 * len(text), y + size),
        boxes,
        (x, y + size),
        direction,
        size,
    )


def fixture(
    label="Renvoi",
    marker="§",
    separator=",",
    family="XZ-AB",
    mark="ARCHIVED",
    order=(0, 1, 2),
    header="Running header",
):
    pages = []
    for p in range(1, 4):
        texts = [
            f"{label}: {family}-641 {marker} 4.2; 4.3 à 4.6",
            f"{family}-975 {marker} 7.1, 7.4",
            f"Value: 6{separator}25 V",
            "the part is in the box",
        ]
        spans = [line(t, p, i, 100 + i * 16) for i, t in enumerate(texts)]
        spans.extend((line(header, p, 8, 12), line(mark, p, 9, 400, size=28, direction=(0, 1))))
        headers = ("Key", "Text", "Result")
        rows = (("2.1", "First row", ""), ("2.2", "Second row", ""))
        evidence = (Evidence("synthetic", p, f"table:{p}", 0, 3, "Key", (40, 200, 350, 280)),)
        table = TableObservation(
            f"table:{p}",
            p,
            tuple(headers[i] for i in order),
            tuple(tuple(r[i] for i in order) for r in rows),
            evidence,
        )
        pages.append(PageDigest(p, 600, 800, tuple(spans), tables=(table,)))
    return DocumentDigest("synthetic", tuple(pages), "synthetic", "1")


def semantics(result):
    return [
        (
            tuple(
                (t.raw, tuple((s.kind, s.key_from, s.key_to) for s in t.sections))
                for t in f.targets
            ),
            f.status,
        )
        for f in result.reading.fields
    ]


def changed_values(a, b):
    return {k for k in a.profile.fields if a.profile.fields[k].value != b.profile.fields[k].value}


def test_census_exact_spans_every_page_and_closed_schema():
    document = fixture()
    observed = census(document)
    assert observed == census(document)
    assert len([r for r in observed.records if r.category == "page"]) == 1
    assert sum(r.count for r in observed.records if r.category == "page") == 3
    views = {v.view_id: v for p in document.pages for v in line_views(p.spans)}
    for record in observed.records:
        assert record.count == len(record.evidence)
        for e in record.evidence:
            if e.span_id in views:
                assert views[e.span_id].text[e.start : e.end] == e.literal
    with pytest.raises(ValueError):
        CensusRecord(
            "id",
            "label",
            "word",
            1,
            (Evidence("s", 1, "x", 0, 4, "word", (0, 0, 1, 1)),),
            (("role", "reference_field"),),
        )
    with pytest.raises(TypeError):
        CensusRecord("id", "label", "word", 1, (), role="reference_field")


def test_x1_label_mutation_and_baseline():
    a = explore(fixture())
    b = explore(fixture(label="Link field"))
    assert len(a.reading.fields) == 3
    assert all(len(f.targets) == 2 for f in a.reading.fields)
    assert semantics(a) == semantics(b)
    assert changed_values(a, b) == {"reference_fields"}
    assert a.profile.template_signature != b.profile.template_signature
    # A literal-label baseline fails the translated fixture; profile reader does not.
    baseline = lambda d: sum(s.text.startswith("Renvoi:") for p in d.pages for s in p.spans)
    assert baseline(fixture()) == 3 and baseline(fixture(label="Link field")) == 0


@pytest.mark.parametrize("marker", ["Section", ""])
def test_x2_marker_mutation(marker):
    a = explore(fixture())
    b = explore(fixture(marker=marker))
    assert semantics(a) == semantics(b)
    assert changed_values(a, b) == {"numbering_style"}


def test_x3_locale_mutation_values_and_grouping():
    a = explore(fixture())
    b = explore(fixture(separator="."))
    assert changed_values(a, b) == {"numeric_locale"}
    assert [q["number"] for q in a.reading.quantities if q["kind"] == "scalar"] == [
        q["number"] for q in b.reading.quantities if q["kind"] == "scalar"
    ]
    from ragix_kernels.harvest.quantitative import harvest

    assert (
        "GROUPING_ASSUMED"
        in harvest("7 350 kg", source_id="s", node_id="n", classification="CONTENT")[0].flags
    )


def test_x4_columns_reordered():
    a = explore(fixture())
    b = explore(fixture(order=(2, 0, 1)))
    assert changed_values(a, b) == {"id_row_tables"}
    assert [(r["key"], r["cells"]) for r in a.reading.tables] == [
        (r["key"], r["cells"]) for r in b.reading.tables
    ]


def test_x5_identifier_family_no_whitelist():
    a = explore(fixture())
    b = explore(fixture(family="LONG/FAMILY"))
    assert changed_values(a, b) == {"identifier_families"}
    assert (
        sum(len(f.targets) for f in a.reading.fields)
        == sum(len(f.targets) for f in b.reading.fields)
        == 6
    )
    assert all(t.raw.startswith("LONG/FAMILY") for f in b.reading.fields for t in f.targets)


def test_x6_furniture_mutation_and_body_negative_control():
    a = explore(fixture())
    b = explore(fixture(mark="RETIRED", header="Different header"))
    assert changed_values(a, b) == {"furniture"}
    assert len(a.reading.furniture) == len(b.reading.furniture) == 6
    d = fixture()
    pages = tuple(
        replace(p, spans=(*p.spans, line("Running header", p.page, 99, 350))) for p in d.pages
    )
    result = explore(replace(d, pages=pages))
    assert len(result.reading.furniture) == 6


def test_x7_unknown_template_and_textless_page_accounting():
    d = DocumentDigest("synthetic", (PageDigest(1, 600, 800, ()),), "synthetic", "1")
    result = explore(d)
    assert result.reading.findings and not result.reading.fields
    assert result.report.coverage.pages == 1 and result.report.coverage.text_layer_pages == 0
    assert all(f.inspected_count == 1 for f in result.reading.findings)
    assert all(f.value is None for k, f in result.profile.fields.items() if k != "reading_coverage")


def test_profile_unknown_evidence_and_append_only_amendments():
    a = explore(fixture())
    entry = a.profile.fields["reference_fields"]
    with pytest.raises(ValueError):
        ProfileField("guess", 0.7, (), "rule", "PROBED")
    with pytest.raises(ValueError):
        ProfileField("guess", 0, (), "rule", "UNKNOWN")
    r = ProfileReview(
        "r1",
        "synthetic",
        "reference_fields",
        replace(entry, status="CONFIRMED"),
        "reviewer",
        "explicit check",
    )
    amended = apply_reviews(a.profile, [r], a.census)
    assert amended.reviews == (r,) and a.profile.reviews == ()
    with pytest.raises(ValueError):
        apply_reviews(amended, [r], a.census)
    bad = replace(r, review_id="r2")
    with pytest.raises(ValueError):
        apply_reviews(amended, [bad], a.census)


class Cache:
    def __init__(self):
        self.entries = {}

    def get(self, model, prompt, temperature, model_digest):
        return self.entries.get((model, prompt, model_digest))

    def put(self, model, prompt, response, temperature, model_digest):
        self.entries[model, prompt, model_digest] = response


class Port:
    def __init__(self, mutate=None):
        self.calls = 0
        self.mutate = mutate

    def generate(self, **kw):
        self.calls += 1
        assert kw["temperature"] == 0
        ids = kw["schema"]["properties"]["decisions"]["items"]["properties"]["candidate_id"]["enum"]
        data = {
            "version": VERSION,
            "decisions": [
                {"candidate_id": i, "role": None, "reason": "insufficient_evidence"} for i in ids
            ],
        }
        if self.mutate:
            self.mutate(data)
        return json.dumps(data)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["decisions"][0].update(literal="invented"),
        lambda d: d["decisions"][0].update(candidate_id="unknown"),
        lambda d: d["decisions"].pop(),
        lambda d: d["decisions"].append(d["decisions"][0]),
    ],
)
def test_x8_classifier_refusals_are_cached_and_never_change_profile(mutate):
    a = explore(fixture())
    port = Port(mutate)
    cache = Cache()
    config = ClassificationConfig("local-model", "digest", True, packet_bytes=1000000)
    first = classify_profile(a.census, a.profile, config, port=port, cache=cache)
    second = classify_profile(a.census, a.profile, config, port=port, cache=cache)
    assert first.manifest["outcome"] == "refused" and second.manifest["cached"] and port.calls == 1
    assert first.profile == second.profile == a.profile


def test_x9_model_off_parity_and_valid_accounting():
    a = explore(fixture())
    off = classify_profile(a.census, a.profile, ClassificationConfig("", ""))
    assert off.profile is a.profile and not off.decisions
    port = Port()
    cache = Cache()
    on = classify_profile(
        a.census,
        a.profile,
        ClassificationConfig("local", "digest", True, packet_bytes=1000000),
        port=port,
        cache=cache,
    )
    assert on.profile is a.profile and len(on.decisions) == len(a.census.records)
    assert on.manifest["outcome"] == "accepted"
    with pytest.raises(ValueError, match="budget"):
        classify_profile(
            a.census,
            a.profile,
            ClassificationConfig("local", "digest", True, packet_bytes=20),
            port=port,
            cache=cache,
        )
    assert port.calls == 1


def test_x10_replay_unicode_and_character_geometry():
    a = explore(fixture())
    b = explore(digest_from_dict(asdict(fixture())))
    assert canonical_json(a) == canonical_json(b)
    assert replay_digest(
        [{"mapping": [{"bbox": [1.000001, 2.0, 3.0, 4.0]}], "text": "e\u0301", "timestamp": "a"}]
    ) == replay_digest(
        [{"mapping": [{"bbox": [1.000002, 2.0, 3.0, 4.0]}], "text": "é", "timestamp": "b"}]
    )
    assert replay_digest([{"mapping": ["a", "b"]}]) != replay_digest([{"mapping": ["b", "a"]}])


def test_x11_ranges_preserved_without_inventory_expansion():
    a = explore(fixture())
    ranges = [
        s for f in a.reading.fields for t in f.targets for s in t.sections if s.kind == "range"
    ]
    assert len(ranges) == 3 and all((s.key_from, s.key_to) == ("4.3", "4.6") for s in ranges)
    assert all("expanded" not in asdict(s) for s in ranges)
    style = a.profile.fields["numbering_style"].value
    split = section_expressions("4. 3 à 4.6", style)
    assert split[0].kind == "split" and split[0].key_from is None


@pytest.mark.parametrize(
    "field,value",
    [("searched_scope", ()), ("rule_id", ""), ("inspected_count", 0), ("inspected_count", True)],
)
def test_x12_negative_schema(field, value):
    data = dict(
        record_id="n",
        question="link",
        searched_scope=("source",),
        rule_id="rule",
        inspected_count=1,
    )
    data[field] = value
    with pytest.raises(ValueError):
        InsufficientEvidence(**data)


def test_report_escapes_source_and_requires_label_map():
    result = explore(fixture(label="<script>bad</script>"))
    with pytest.raises(ValueError):
        render_report(result.report, {})
    html = render_report(
        result.report, {"field": "Observed field", "quantity": "Quantity", "table_row": "Row"}
    )
    assert "<script>bad</script>" not in html and "&lt;script&gt;" in html
    assert "data-record-id=" in html and "https://" not in html


def test_library_and_kernel_chain_no_network(tmp_path, monkeypatch):
    from ragix_kernels.base import KernelInput
    from ragix_kernels.saqqara.kernels.explorer import (
        CensusKernel,
        ProfileKernel,
        ReadKernel,
        ReportKernel,
    )

    def forbidden(*a, **k):
        raise AssertionError("network call in deterministic chain")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    doc = fixture()
    last = None
    for cls in (CensusKernel, ProfileKernel, ReadKernel, ReportKernel):
        kernel = cls()
        config = {"documents": [asdict(doc)]} if last is None else {}
        deps = {} if last is None else {kernel.requires[0]: last.output_file}
        last = kernel.run(KernelInput(tmp_path, config, deps))
        assert last.success, last.errors
    assert canonical_json(last.data["reports"][0]) == canonical_json(explore(doc).report)


def test_cli_library_parity(tmp_path):
    doc = fixture()
    path = tmp_path / "digest.json"
    path.write_text(json.dumps(asdict(doc)))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ragix_kernels.saqqara.cli.explorer",
            str(path),
            "--output",
            str(tmp_path / "output"),
        ],
        check=True,
    )
    output = next((tmp_path / "output").glob("*.json"))
    assert canonical_json(json.loads(output.read_text())) == canonical_json(explore(doc))


def test_reader_code_has_no_fixture_label_family_or_marker_literal():
    import ragix_kernels.saqqara.profile_readers as module

    text = Path(module.__file__).read_text()
    for literal in ("Renvoi", "XZ-AB", "§"):
        assert literal not in text


def test_orchestrator_manifest_chain_and_cache(tmp_path, monkeypatch):
    import yaml
    from ragix_kernels.saqqara.kernels.explorer import register_explorer_kernels
    from ragix_kernels.orchestrator import Orchestrator

    register_explorer_kernels()
    data = {
        "audit": {"name": "Synthetic explorer"},
        "project": {"path": "."},
        "stage1": {
            "explorer_census": {"enabled": True, "options": {"documents": [asdict(fixture())]}}
        },
        "stage2": {"explorer_profile": {"enabled": True}, "explorer_read": {"enabled": True}},
        "stage3": {"explorer_report": {"enabled": True}},
    }
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(data))

    def forbidden(*a, **k):
        raise AssertionError("network in orchestrated kernel")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    engine = Orchestrator(tmp_path)
    for stage, names in [
        (1, ["explorer_census"]),
        (2, ["explorer_profile", "explorer_read"]),
        (3, ["explorer_report"]),
    ]:
        outputs = engine.run_stage(stage, names)
        assert outputs and all(o.success for o in outputs), [o.errors for o in outputs]
    report = json.loads((tmp_path / "stage3" / "explorer_report.json").read_text())["data"][
        "reports"
    ][0]
    assert canonical_json(report) == canonical_json(explore(fixture()).report)
    again = engine.run_stage(3, ["explorer_report"], use_cache=True)
    assert all(o.success for o in again)


def test_ambiguous_numeric_locale_does_not_guess():
    d = fixture()
    pages = tuple(
        replace(
            p,
            spans=tuple(
                (
                    replace(s, text=s.text.replace("6,25", "6,250"), glyph_boxes=(), bbox=s.bbox)
                    if "6,25" in s.text
                    else s
                )
                for s in p.spans
            ),
        )
        for p in d.pages
    )
    result = explore(replace(d, pages=pages))
    assert result.profile.fields["numeric_locale"].status == "UNKNOWN"
    assert any(f.field == "numeric_locale" for f in result.reading.findings)


def test_profile_evidence_cannot_be_forged():
    a = explore(fixture())
    fields = dict(a.profile.fields)
    fields["reference_fields"] = replace(fields["reference_fields"], evidence=("unknown",))
    with pytest.raises(ValueError, match="evidence"):
        read_document(a.document, a.census, replace(a.profile, fields=fields))


def test_unreadable_and_duplicate_headers_cannot_silently_overwrite_cells():
    d = fixture()
    pages = tuple(
        replace(p, tables=(replace(p.tables[0], headers=("Key", "Text", "Text")),)) for p in d.pages
    )
    result = explore(replace(d, pages=pages))
    assert not result.reading.tables and any(
        f.field == "id_row_tables" for f in result.reading.findings
    )


def test_revision_is_not_a_section_member():
    d = fixture()
    pages = []
    for p in d.pages:
        spans = []
        for s in p.spans:
            text = s.text.replace("XZ-AB-641 §", "XZ-AB-641 rev. 2.0 §")
            if text != s.text:
                s = line(text, p.page, 0, 100)
            spans.append(s)
        pages.append(replace(p, spans=tuple(spans)))
    result = explore(replace(d, pages=tuple(pages)))
    for field in result.reading.fields:
        target = field.targets[0]
        assert target.revision_raw == "2.0"
        assert field.view.text[target.revision_start : target.revision_end] == "2.0"
        assert all(s.key_from != "2.0" for s in target.sections)


def test_profile_value_schema_is_closed():
    a = explore(fixture())
    fields = dict(a.profile.fields)
    fields["reference_fields"] = replace(fields["reference_fields"], value={"invented": "policy"})
    with pytest.raises(ValueError, match="schema"):
        replace(a.profile, fields=fields)


def test_stale_digest_profile_is_refused_even_with_same_source_label():
    a = explore(fixture())
    page = a.document.pages[0]
    span = page.spans[0]
    changed = replace(span, text=span.text.replace("641", "642"))
    altered = replace(
        a.document, pages=(replace(page, spans=(changed, *page.spans[1:])), *a.document.pages[1:])
    )
    with pytest.raises(ValueError, match="mismatch"):
        read_document(altered, a.census, a.profile)


def test_spec_explorer_propositions_are_contiguous_and_declared():
    import re

    text = (Path(__file__).resolve().parents[2] / "ragix_kernels/saqqara/SPEC.md").read_text()
    assert re.findall(r"\*\*K8\.(\d+)", text) == [str(i) for i in range(1, 5)]
    assert re.findall(r"\*\*K9\.(\d+)", text) == [str(i) for i in range(1, 11)]
