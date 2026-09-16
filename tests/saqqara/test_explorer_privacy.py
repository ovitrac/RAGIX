"""X18: generated stamp shapes, masking surfaces and immutable observations.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, replace
import json
import pytest
from ragix_kernels.saqqara.census import DocumentDigest, PageDigest
from ragix_kernels.saqqara.field_views import TextSpan
from ragix_kernels.saqqara.explorer import explore
from ragix_kernels.saqqara.privacy import MaskPolicy, PresentationMasker, stamp_matches
from ragix_kernels.harvest.report import (
    render_report,
    render_page_view,
    render_report_json,
    replay_digest,
)

DATES = ("2033-07-19", "19/07/2033", "19.07.2033", "19 Jul 2033", "Jul 19, 2033", "20330719")
NAMES = ("Aster Quorin", "Élodine Vaurèle")
LABELS = {"field": "Observed field", "quantity": "Literal quantity", "table_row": "Observed row"}


def span(text, page, ident, y=12):
    boxes = tuple((30 + 4 * i, y, 34 + 4 * i, y + 10) for i in range(len(text)))
    return TextSpan(
        "synthetic-privacy",
        f"span:{page}:{ident}",
        page,
        text,
        (30, y, 30 + 4 * len(text), y + 10),
        boxes,
        (30, y + 10),
        font_size=10,
    )


def fixture():
    return DocumentDigest(
        "synthetic-privacy",
        tuple(
            PageDigest(
                p,
                700,
                800,
                (
                    span(f"Export: {DATES[p-1]} {NAMES[(p-1)%2]}", p, 0),
                    span("The device was inspected on 2033-07-19.", p, 1, 130),
                    span("Link: AB-K-752", p, 2, 200),
                    span("Value: 3,25 V", p, 3, 220),
                ),
            )
            for p in range(1, 7)
        ),
        "synthetic",
        "1",
    )


@pytest.mark.parametrize("date", DATES)
@pytest.mark.parametrize("name", NAMES)
def test_e10_1_dates_and_exact_subspans(date, name):
    text = f"Export: {date} {name}"
    matches = stamp_matches(text, edge=True)
    assert any(m["raw"] == name for m in matches)
    assert all(text[m["start"] : m["end"]] == m["raw"] for m in matches)


@pytest.mark.parametrize(
    "text",
    [
        "14:35 Aster Quorin",
        "14h35 Élodine Vaurèle",
        "14:35 aq731",
        "14:35 aster.quorin37@example.invalid",
    ],
)
def test_time_email_and_user_shapes(text):
    assert stamp_matches(text, edge=True)
    assert stamp_matches(text, recurring=True)
    assert not stamp_matches(text)


@pytest.mark.parametrize(
    "text,edge",
    [
        ("The device was inspected on 2033-07-19.", False),
        ("Premium Module", True),
        ("4.2 Inspection Overview", False),
    ],
)
def test_e10_2_negative_controls(text, edge):
    assert not stamp_matches(text, edge=edge)


def test_body_date_and_name_is_flagged():
    assert stamp_matches("2033-07-19 Aster Quorin", edge=False)


def test_e10_3_every_presentation_surface_masks_and_keeps_coordinates():
    result = explore(fixture())
    stamps = result.profile.fields["furniture"].value["stamp_lines"]
    assert len(stamps) == 6 and result.report.provenance["masked_lines"] == 6
    outputs = [render_report(result.report, LABELS), render_report_json(result.report)]
    outputs.extend(render_page_view(result.report, p) for p in range(1, 7))
    for output in outputs:
        for name in NAMES:
            for token in name.split():
                assert token not in output
        assert "masked_lines" in output and "[MASKED]" in output
    assert "data-bbox=" in outputs[2]
    assert stamps[0]["text"].endswith(NAMES[0])
    assert all(s["personal_data_suspected"] for s in stamps)


def test_e10_4_explicit_unmask_policy_and_reason_required():
    result = explore(fixture())
    with pytest.raises(ValueError):
        MaskPolicy(unmask=True)
    with pytest.raises(TypeError):
        render_report(result.report, LABELS, policy={"unmask": True})
    policy = MaskPolicy(unmask=True, reason="Authorised source inspection")
    text = render_report(result.report, LABELS, policy=policy)
    assert NAMES[0] in text and policy.reason in text and policy.digest in text
    assert (
        json.loads(render_report_json(result.report, policy=policy))["provenance"]["masked_lines"]
        == 0
    )


def test_e10_5_source_census_profile_and_report_are_not_mutated():
    result = explore(fixture())
    before = replay_digest([result.document, result.census, result.profile, result.report])
    render_report(result.report, LABELS)
    render_report(result.report, LABELS, policy=MaskPolicy(unmask=True, reason="Controlled test"))
    assert replay_digest([result.document, result.census, result.profile, result.report]) == before


def test_private_policy_is_effective_but_not_disclosed():
    result = explore(fixture())
    policy = MaskPolicy(extra_literals=("device",), extra_patterns=(r"AB-K-\d+",))
    output = render_report_json(result.report, policy=policy)
    assert "device" not in output and "AB-K-752" not in output
    assert policy.digest in output and "extra_literals" not in output
    assert json.loads(output)["provenance"]["masked_lines"] == 18


def test_split_name_fragments_and_masked_recurrence_do_not_leak():
    matches = stamp_matches("14:35 Aster Quorin", edge=True)
    masker = PresentationMasker([{"matches": matches}])
    result = masker.data({"parts": ["Aster", "Quorin"], "repeat": "##:## Aster Quorin"})
    assert "Aster" not in json.dumps(result) and "Quorin" not in json.dumps(result)


def test_unsafe_policy_or_masked_key_collision_fails_closed():
    with pytest.raises(ValueError):
        MaskPolicy(extra_patterns=(".*",))
    masker = PresentationMasker([], MaskPolicy(extra_literals=("one", "two")))
    with pytest.raises(ValueError):
        masker.data({"one": 1, "two": 2})


def test_recurring_body_time_stamp_is_flagged_with_shared_policy():
    doc = DocumentDigest(
        "synthetic-privacy",
        tuple(PageDigest(p, 700, 800, (span("14:35 Aster Quorin", p, 0, 200),)) for p in (1, 2, 3)),
        "synthetic",
        "1",
    )
    result = explore(doc)
    assert result.profile.fields["furniture"].value["stamp_count"] == 3


def test_mask_token_cannot_be_a_backdoor_to_unmasking():
    with pytest.raises(ValueError):
        MaskPolicy(mask_token="Aster Quorin")


def test_profile_amendment_cannot_disable_upstream_privacy():
    from ragix_kernels.harvest.report import build_report

    result = explore(fixture())
    fields = dict(result.profile.fields)
    fields["furniture"] = replace(
        fields["furniture"],
        value={**fields["furniture"].value, "stamp_lines": [], "stamp_count": 0},
    )
    report = build_report(
        result.document, result.census, replace(result.profile, fields=fields), result.reading
    )
    assert NAMES[0] not in render_report(report, LABELS)
    assert report.provenance["masked_lines"] == 6


def test_cli_exports_only_masked_presentations_by_default(tmp_path):
    from ragix_kernels.saqqara.cli.explorer import main

    source = tmp_path / "source.json"
    source.write_text(json.dumps(asdict(fixture())))
    output = tmp_path / "export"
    assert main([str(source), "--output", str(output)]) == 0
    assert len(list(output.glob("*.page-*.html"))) == 6
    for path in output.iterdir():
        text = path.read_text()
        assert all(token not in text for name in NAMES for token in name.split())
    assert not list(output.glob("*.observations.json"))


def test_decomposed_unicode_names_keep_exact_offsets_and_mask_variants():
    import unicodedata

    name = unicodedata.normalize("NFD", NAMES[1])
    text = "2033-07-19 " + name
    matches = stamp_matches(text, edge=True)
    assert any(m["raw"] == name for m in matches)
    assert all(text[m["start"] : m["end"]] == m["raw"] for m in matches)
    masker = PresentationMasker([{"matches": matches}])
    assert NAMES[1] not in masker.text(NAMES[1]) and name not in masker.text(name)


def test_explicit_observation_export_preserves_original_unicode(tmp_path):
    import unicodedata
    from ragix_kernels.saqqara.cli.explorer import main

    text = "2033-07-19 " + unicodedata.normalize("NFD", NAMES[1])
    document = DocumentDigest(
        "synthetic-privacy", (PageDigest(1, 700, 800, (span(text, 1, 0),)),), "synthetic", "1"
    )
    source = tmp_path / "source.json"
    source.write_text(json.dumps(asdict(document)))
    output = tmp_path / "out"
    main([str(source), "--output", str(output), "--observations"])
    original = json.loads(next(output.glob("*.observations.json")).read_text())
    assert original["document"]["pages"][0]["spans"][0]["text"] == text


def test_failure_reports_remain_renderable_with_masking_enabled():
    from ragix_kernels.saqqara.failures import failure_report, construct_finding

    report = failure_report(
        construct_finding("synthetic-privacy", "census", "INVALID_CONSTRUCT", page=1)
    )
    assert "INVALID_CONSTRUCT" in render_report(report, {})
    assert "FAILED" in render_report_json(report)
    assert "reading failed" in render_page_view(report, 1)
