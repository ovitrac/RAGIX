"""Synthetic X16/X17 and threshold sensitivity: shared windows, local numbers.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, replace
from decimal import Decimal
from itertools import product
import inspect
import pytest

from ragix_kernels.saqqara.field_views import TextSpan, VerticalRule, HorizontalRule
from ragix_kernels.saqqara.census import (
    DocumentDigest,
    PageDigest,
    CensusConfig,
    TableObservation,
    Evidence,
    census,
    census_from_dict,
)
from ragix_kernels.saqqara.profile import ProfileConfig, derive_profile
from ragix_kernels.saqqara.profile_readers import read_document
from ragix_kernels.saqqara.value_windows import ContinuationPolicy
from ragix_kernels.harvest.numeric_locale import resolve_number, physical_numbers
from ragix_kernels.harvest.quantitative import harvest


def line(text, page, ident, y, x=40):
    boxes = tuple((x + i * 4, y, x + (i + 1) * 4, y + 10) for i in range(len(text)))
    return TextSpan(
        "synthetic-slice2",
        f"s:{page}:{ident}",
        page,
        text,
        (x, y, x + 4 * len(text), y + 10),
        boxes,
        (x, y + 10),
        font_size=10,
    )


def document(*texts):
    return DocumentDigest(
        "synthetic-slice2",
        (PageDigest(1, 600, 800, tuple(line(t, 1, i, 100 + i * 16) for i, t in enumerate(texts))),),
        "synthetic",
        "1",
    )


def run(doc, policy=ContinuationPolicy(source="amended"), profile_config=ProfileConfig()):
    observed = census(doc, CensusConfig(continuation_policy=policy))
    profile = derive_profile(observed, profile_config)
    return observed, profile, read_document(doc, observed, profile)


def window_fixture(position="same_line", label="Link", stop="Notes:", numeric=()):
    pages = []
    for p in (1, 2):
        target = "LM-K-582 §4.2; 4.3 à 4.6"
        rules = ()
        if position == "same_line":
            spans = [line(label + ": " + target, p, 0, 100)]
        else:
            spans = [line(label + ":", p, 0, 100)]
            if position == "next_cell":
                spans.append(line(target, p, 1, 100, 220))
                rules = (VerticalRule(180, 80, 135),)
            else:
                y = {"next_line": 116, "two_lines": 132, "beyond": 160}[position]
                spans.append(line(target, p, 1, y))
        spans.append(line(stop, p, 2, 180))
        spans.extend(line(t, p, i + 3, 220 + i * 16) for i, t in enumerate(numeric))
        pages.append(PageDigest(p, 600, 800, tuple(spans), rules))
    return DocumentDigest("synthetic-slice2", tuple(pages), "synthetic", "1")


def field_semantics(reading):
    return [
        (
            f.label,
            f.status,
            f.flags,
            f.needs_review,
            [(t.raw, [(s.kind, s.key_from, s.key_to) for s in t.sections]) for t in f.targets],
        )
        for f in reading.fields
    ]


@pytest.mark.parametrize("label", ["Link", "Renvoi"])
@pytest.mark.parametrize("position", ["same_line", "next_line", "next_cell", "two_lines", "beyond"])
def test_e7_1_value_window_positions(label, position):
    observed, profile, reading = run(window_fixture(position, label))
    records = [r for r in observed.records if r.category == "label" and r.literal == label]
    assert records
    assert {dict(r.attributes)["follow"] for r in records} == (
        {"empty"} if position == "beyond" else {"identifier-bearing"}
    )
    if position != "beyond":
        assert len(reading.fields) == 2 and all(len(f.targets) == 1 for f in reading.fields)
    else:
        windows = [w for w in observed.windows if w.label == label]
        assert all("WINDOW_BOUND_HIT" in w.flags and w.needs_review for w in windows)


def test_x16_same_profile_and_readings_when_value_moves_line():
    a = run(window_fixture("same_line"))
    b = run(window_fixture("next_line"))
    assert a[1].fields["reference_fields"].value == b[1].fields["reference_fields"].value
    assert a[1].fields["reference_fields"].confidence == b[1].fields["reference_fields"].confidence
    assert field_semantics(a[2]) == field_semantics(b[2])
    assert {w.value_position for w in a[0].windows if w.label == "Link"} == {"same_line"}
    assert {w.value_position for w in b[0].windows if w.label == "Link"} == {"next_line"}


def test_e7_3_reader_consumes_census_window_object(monkeypatch):
    import ragix_kernels.saqqara.profile_readers as readers

    observed, profile, _ = run(window_fixture("next_cell"))
    original = readers.join_window
    consumed = []

    def checked(window):
        assert any(window is item for item in observed.windows)
        assert window.policy is observed.continuation_policy
        consumed.append(window.window_id)
        return original(window)

    monkeypatch.setattr(readers, "join_window", checked)
    read_document(window_fixture("next_cell"), observed, profile)
    assert len(consumed) == 2
    source = inspect.getsource(readers.read_document)
    assert "max_gap_ratio" not in source and "max_lines" not in source
    replay = census_from_dict(asdict(observed))
    assert all(w.policy is replay.continuation_policy for w in replay.windows)


@pytest.mark.parametrize(
    "stop,reason",
    [("Notes:", "label"), ("4.2 Overview", "numbered_heading"), ("Table 7", "table_caption")],
)
def test_structural_stops_are_retained_without_guard_flag(stop, reason):
    d = document("Link:", stop, "LM-K-582")
    observed, _, _ = run(d)
    w = next(w for w in observed.windows if w.label == "Link")
    assert w.stop_reason == reason and not w.needs_review
    assert stop not in w.following_text and w.stop_view_id is not None


def test_horizontal_rule_and_column_break_stop_windows():
    d = document("Link:", "LM-K-582")
    p = replace(d.pages[0], horizontal_rules=(HorizontalRule(113, 0, 600),))
    c, _, _ = run(replace(d, pages=(p,)))
    assert next(w for w in c.windows if w.label == "Link").stop_reason == "horizontal_rule"
    p = replace(d.pages[0], spans=(d.pages[0].spans[0], line("LM-K-582", 1, 1, 116, 450)))
    c, _, _ = run(replace(d, pages=(p,)))
    assert next(w for w in c.windows if w.label == "Link").stop_reason == "column_break"


def test_guard_hit_marks_field_and_unrelated_header_is_not_label():
    d = document("Link: LM-K-582", *("4.2" for _ in range(10)))
    c, _, _ = run(d, ContinuationPolicy(max_lines=2, source="amended"))
    w = next(w for w in c.windows if w.label == "Link")
    assert w.stop_reason == "line_bound" and w.needs_review and "WINDOW_BOUND_HIT" in w.flags
    d = DocumentDigest(
        "synthetic-slice2",
        (
            PageDigest(
                1, 600, 800, (line("Recurring header", 1, 0, 100), line("LM-K-582", 1, 1, 300))
            ),
        ),
        "synthetic",
        "1",
    )
    assert not [r for r in run(d)[0].records if r.category == "label"]


def test_bimodal_gap_histogram_is_derived_not_default():
    ys = (100, 116, 180, 196, 260, 276)
    d = DocumentDigest(
        "synthetic-slice2",
        (PageDigest(1, 600, 800, tuple(line("body", 1, i, y) for i, y in enumerate(ys))),),
        "synthetic",
        "1",
    )
    c = census(d)
    assert c.continuation_policy.source == "derived"
    assert c.continuation_policy.gap_histogram == ((0.6, 3), (5.4, 2))
    assert c.continuation_policy.max_gap_ratio == 3
    assert derive_profile(c).continuation_policy is c.continuation_policy
    amended = census(
        d, CensusConfig(continuation_policy=ContinuationPolicy(1.75, 6, source="amended"))
    )
    assert (
        amended.continuation_policy.source == "amended"
        and amended.continuation_policy.max_gap_ratio == 1.75
    )
    assert census(document("a", "b", "c")).continuation_policy.source == "default"


@pytest.mark.parametrize(
    "raw,value,separator",
    [
        ("12,75", "12.75", ","),
        ("12.75", "12.75", "."),
        ("0.125", "0.125", "."),
        ("1,234.50", "1234.50", "."),
        ("1.234,50", "1234.50", ","),
        ("1 234,50", "1234.50", ","),
        ("1,234,567", "1234567", "."),
    ],
)
def test_per_token_resolution(raw, value, separator):
    result = resolve_number(raw)
    assert result.value == Decimal(value) and result.separator == separator


@pytest.mark.parametrize("separator", [".", ","])
def test_x17_only_physical_evidence_votes(separator):
    texts = ["Section 9.4", "rev. 8.7", "2032-05-17", "Page 12", "AB-7/92", "§4.2 m"]
    physical = [f"Voltage {n}{separator}25 V" for n in range(2, 8)]
    d = document(*texts, *physical)
    c, p, r = run(d)
    locale = p.fields["numeric_locale"]
    assert (
        locale.value["decimal_separator"] == separator
        and locale.value["prior_strength"] == "dominant"
    )
    assert locale.value["n"] == 6 and locale.value["separator_counts"][separator] == 6
    selected = [x for x in c.records if x.candidate_id in locale.evidence]
    assert {x.literal for x in selected} == {f"{n}{separator}25" for n in range(2, 8)}
    for x in selected:
        assert all(
            e.literal in physical[e.page - 1] or e.literal in "\n".join(physical)
            for e in x.evidence
        )
    scalars = [q for q in r.quantities if q["kind"] == "scalar"]
    assert len(scalars) == 6
    assert {q["number"] for q in scalars} == {f"{n}.25" for n in range(2, 8)}


def test_weak_prior_and_no_circular_vote_from_ambiguous_tokens():
    _, p, r = run(document("5,25 V", "6,75 V", "1.234 V"))
    assert p.fields["numeric_locale"].value["n"] == 2
    assert p.fields["numeric_locale"].value["prior_strength"] == "weak"
    assert p.fields["numeric_locale"].confidence <= 0.5
    candidate = next(q for q in r.quantities if q["raw"] == "1.234 V")
    assert candidate["number"] == "1234" and candidate["needs_review"]
    assert {"SEPARATOR_AMBIGUOUS", "LOCALE_PRIOR_USED", "LOCALE_PRIOR_WEAK"} <= set(
        candidate["flags"]
    )
    _, p, r = run(document("1.234 V", "2.345 V", "3.456 V"))
    assert p.fields["numeric_locale"].value["n"] == 0
    assert all(q["number"] is None for q in r.quantities if q["kind"] == "scalar")


def test_mixed_tokens_are_read_and_only_prior_contradictions_flagged():
    _, p, r = run(document("2,50 V", "3.75 V"))
    assert p.fields["numeric_locale"].status == "PROBED"
    assert p.fields["numeric_locale"].value["decimal_separator"] is None
    assert [q["number"] for q in r.quantities if q["kind"] == "scalar"] == ["2.50", "3.75"]
    lines = [f"{n}.25 V" for n in range(2, 12)] + ["7,50 V"]
    _, p, r = run(document(*lines))
    assert p.fields["numeric_locale"].value["decimal_separator"] == "."
    q = next(q for q in r.quantities if q["raw"] == "7,50 V")
    assert q["number"] == "7.50" and "SEPARATOR_AMBIGUOUS" in q["flags"] and q["needs_review"]
    assert all(
        "SEPARATOR_AMBIGUOUS" not in q["flags"]
        for q in r.quantities
        if q["raw"] != "7,50 V" and q["kind"] == "scalar"
    )


def test_unitless_comparator_and_nonidentifier_cells_are_physical():
    assert [x.raw for x in physical_numbers("limit >= 8,25")] == ["8,25"]
    c = harvest(
        "limit >= 8,25", source_id="s", node_id="n", classification="CONTENT", token_locale=True
    )
    assert c[0].number == "8.25" and c[0].comparator_normalized == ">="
    table = TableObservation(
        "t",
        1,
        ("Key", "Value"),
        (("AB-K-17", "8,25"), ("AB-K-18", "9,75")),
        (Evidence("synthetic-slice2", 1, "table", 0, 3, "Key", (30, 300, 450, 360)),),
    )
    d = replace(document("text"), pages=(replace(document("text").pages[0], tables=(table,)),))
    observed, profile, reading = run(d)
    assert profile.fields["numeric_locale"].value["n"] == 2
    assert {q["number"] for q in reading.quantities if q["kind"] == "scalar"} == {"8.25", "9.75"}


@pytest.mark.parametrize(
    "gap,cap,dominance,n", list(product((1.5, 2.5, 4), (6, 8, 16), (0.8, 0.9, 1), (2, 3, 4, 5)))
)
def test_e7_7_e8_7_sensitivity(gap, cap, dominance, n):
    d = window_fixture("next_line", numeric=tuple(f"Value: {i},25 V" for i in range(2, 8)))
    observed, p, r = run(
        d,
        ContinuationPolicy(gap, cap),
        ProfileConfig(locale_dominance_ratio=dominance, locale_minimum_n=n),
    )
    base = run(d)[2]
    assert field_semantics(r) == field_semantics(base)
    semantic = lambda x: [
        (q["raw"], q["start"], q["end"], q["kind"], q["number"], q["flags"], q["needs_review"])
        for q in x.quantities
    ]
    assert semantic(r) == semantic(base)
    assert p.continuation_policy.max_gap_ratio == gap and p.continuation_policy.max_lines == cap
    assert p.fields["numeric_locale"].value["dominance_ratio"] == dominance
    assert p.fields["numeric_locale"].value["minimum_n"] == n


def test_table_composite_never_loses_a_member_during_deduplication():
    d = document("8,25 V")
    table = TableObservation(
        "t",
        1,
        ("Key", "Value"),
        (("AB-K-17", "8,25 V à 9,75 V"),),
        (Evidence("synthetic-slice2", 1, "table", 0, 3, "Key", (30, 90, 450, 150)),),
    )
    _, _, reading = run(replace(d, pages=(replace(d.pages[0], tables=(table,)),)))
    ids = {q["candidate_id"] for q in reading.quantities}
    assert any(q["kind"] == "interval" for q in reading.quantities)
    assert all(m["candidate_id"] in ids for q in reading.quantities for m in q["members"])


def test_literal_offsets_and_comparators_survive_token_resolution():
    text = "at ≥ 2,75 V and 1.234 V; limit jusqu'à 8,25"
    candidates = harvest(
        text, source_id="s", node_id="n", classification="CONTENT", token_locale=True
    )
    assert all(text[c.start : c.end] == c.raw for c in candidates)
    ambiguous = next(c for c in candidates if c.raw == "1.234 V")
    assert ambiguous.number is None and ambiguous.needs_review
    direction = next(c for c in candidates if c.raw == "jusqu'à 8,25")
    assert direction.direction_status == "unresolved" and direction.comparator_normalized is None


def test_old_profile_schema_and_forged_prior_counts_are_refused():
    _, p, _ = run(document("2.50 V", "3.75 V"))
    with pytest.raises(ValueError, match="version"):
        replace(p, version="document-profile/0.1")
    fields = dict(p.fields)
    fields["numeric_locale"] = replace(
        fields["numeric_locale"], value={**fields["numeric_locale"].value, "n": 99}
    )
    with pytest.raises(ValueError, match="prior schema"):
        replace(p, fields=fields)


def test_table_header_is_a_structural_stop():
    d = document("Link:", "Key", "LM-K-582")
    table = TableObservation(
        "t",
        1,
        ("Key", "Value"),
        (),
        (Evidence("synthetic-slice2", 1, "table", 0, 3, "Key", (40, 116, 450, 180)),),
    )
    c, _, _ = run(replace(d, pages=(replace(d.pages[0], tables=(table,)),)))
    window = next(w for w in c.windows if w.label == "Link")
    assert window.stop_reason == "table_header" and not window.needs_review


def test_page_break_cannot_attach_another_pages_identifier():
    d = DocumentDigest(
        "synthetic-slice2",
        (
            PageDigest(1, 600, 800, (line("Link:", 1, 0, 100),)),
            PageDigest(2, 600, 800, (line("LM-K-582", 2, 0, 100),)),
        ),
        "synthetic",
        "1",
    )
    c, _, _ = run(d)
    window = next(w for w in c.windows if w.label == "Link")
    assert window.stop_reason == "page_end" and len(window.views) == 1 and not window.needs_review


def test_native_pdf_horizontal_rule_is_persisted(tmp_path):
    import importlib.util, json, subprocess, sys

    if importlib.util.find_spec("pymupdf") is None:
        pytest.skip("optional PDF reader")
    code = """import json,sys,pymupdf
from pathlib import Path
from ragix_kernels.saqqara.explorer import digest_pdf
from ragix_kernels.saqqara.census import census
p=Path(sys.argv[1]); d=pymupdf.open(); page=d.new_page()
page.insert_text((40,100),"Link:",fontsize=10)
page.insert_text((40,140),"LM-K-582",fontsize=10)
page.draw_line((30,120),(500,120))
d.save(p,no_new_id=True); d.close()
digest=digest_pdf(p); c=census(digest)
w=next(w for w in c.windows if w.label=="Link")
print(json.dumps({"horizontal":len(digest.pages[0].horizontal_rules),"stop":w.stop_reason,"review":w.needs_review}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "synthetic.pdf")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "horizontal": 1,
        "stop": "horizontal_rule",
        "review": False,
    }


def test_guard_limited_reference_field_cannot_be_ready():
    pages = tuple(
        PageDigest(
            p,
            600,
            800,
            tuple(
                line(t, p, i, 100 + i * 16)
                for i, t in enumerate(("Link: LM-K-582", *("4.2" for _ in range(10))))
            ),
        )
        for p in (1, 2)
    )
    d = DocumentDigest("synthetic-slice2", pages, "synthetic", "1")
    _, _, reading = run(d, ContinuationPolicy(max_lines=2, source="amended"))
    assert len(reading.fields) == 2
    assert all(
        f.status == "UNDECIDABLE" and f.needs_review and "WINDOW_BOUND_HIT" in f.flags
        for f in reading.fields
    )


def test_opt_in_quantity_producer_distinguishes_new_rules():
    kwargs = dict(source_id="s", node_id="n", classification="CONTENT")
    legacy = harvest("2.50 V", **kwargs)
    local = harvest("2.50 V", token_locale=True, **kwargs)
    assert legacy[0].producer == "quantitative/1.0"
    assert local[0].producer == "quantitative/1.1" and local[0].number == legacy[0].number


def test_serialized_guard_stop_cannot_drop_its_review_flag():
    observed, _, _ = run(window_fixture("beyond"))
    window = next(w for w in observed.windows if w.label == "Link")
    assert window.stop_reason == "gap_bound"
    with pytest.raises(ValueError, match="invalid label value window"):
        replace(window, flags=(), needs_review=False)


def test_legacy_digest_without_horizontal_observations_is_refused():
    from ragix_kernels.saqqara.census import digest_from_dict

    payload = asdict(document("Link:", "LM-K-582"))
    payload["pages"][0].pop("horizontal_rules")
    with pytest.raises(ValueError, match="re-extract"):
        digest_from_dict(payload)


def test_horizontal_rule_geometry_uses_the_replay_precision():
    from ragix_kernels.harvest.report import replay_digest

    a = {"horizontal_rules": [{"y": 120.00001, "left": 40.00001, "right": 500.00001}]}
    b = {"horizontal_rules": [{"y": 120.00002, "left": 40.00002, "right": 500.00002}]}
    assert replay_digest([a]) == replay_digest([b])
