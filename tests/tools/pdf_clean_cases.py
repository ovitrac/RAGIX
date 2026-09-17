"""Synthetic PDF-operation falsifiers; no controlled document literals.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import io
import json
from pathlib import Path
import pytest

pikepdf = pytest.importorskip("pikepdf")
pymupdf = pytest.importorskip("pymupdf")
from ragix_core.pdf_watermark import Limits, Refused, inspect_pdf, apply_plan


def document(path, *, form=False, mixed=False, opaque=False, split=False, marker="SAMPLE", pages=2):
    with pikepdf.Pdf.new() as pdf:
        font = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font,
                Subtype=pikepdf.Name.Type1,
                BaseFont=pikepdf.Name.Helvetica,
                Encoding=pikepdf.Name.WinAnsiEncoding,
            )
        )
        gs = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.ExtGState, ca=0.25, CA=0.25))
        encoded = marker.encode("cp1252").hex().encode()
        wm = (
            b"q /Fade gs BT /F1 42 Tf 0.7071 0.7071 -0.7071 0.7071 60 75 Tm <"
            + encoded
            + b"> Tj ET Q\n"
        )
        if opaque:
            wm = b"q BT /F1 12 Tf 1 0 0 1 35 160 Tm <" + encoded + b"> Tj ET Q\n"
        if mixed:
            wm = wm.replace(b"ET Q", b"0 -25 Td (Body must survive 31.75) Tj ET Q")
        for index in range(pages):
            page = pdf.add_blank_page(page_size=(320, 260))
            page.Resources = pikepdf.Dictionary(
                Font=pikepdf.Dictionary(F1=font), ExtGState=pikepdf.Dictionary(Fade=gs)
            )
            body = b"q 0 G 1 w 20 80 280 100 re S 20 130 m 300 130 l S Q\nq BT /F1 12 Tf 1 0 0 1 30 145 Tm (Body 31.75 V remains) Tj ET Q\n"
            if form:
                if index == 0:
                    watermark = pdf.make_stream(wm)
                    watermark.Type = pikepdf.Name.XObject
                    watermark.Subtype = pikepdf.Name.Form
                    watermark.BBox = pikepdf.Array([0, 0, 320, 260])
                    watermark.Resources = page.Resources
                page.Resources.XObject = pikepdf.Dictionary(Watermark=watermark)
                wm_content = b"q /Watermark Do Q\n"
            else:
                wm_content = wm
            if split:
                page.Contents = pikepdf.Array(
                    [
                        pdf.make_stream(body),
                        pdf.make_stream(wm_content[:2]),
                        pdf.make_stream(wm_content[2:]),
                    ]
                )
            else:
                page.Contents = pdf.make_stream(body + wm_content)
        pdf.save(path, deterministic_id=True)
    return path


def clean(path, out, **kwargs):
    plan = inspect_pdf(path, kwargs.pop("marker", "SAMPLE"), kwargs.pop("limits", Limits()))
    assert len(plan["candidates"]) == 2, plan
    audit = apply_plan(
        path, plan, [c["candidate_id"] for c in plan["candidates"]], out, out.with_suffix(".json")
    )
    return plan, audit


@pytest.mark.parametrize("form", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("dpi", [72, 144])
def test_marker_removal_preserves_overlapped_body_and_rules(tmp_path, form, split, dpi):
    path = document(tmp_path / "original.pdf", form=form, split=split)
    original = path.read_bytes()
    _, audit = clean(path, tmp_path / "derived.pdf", limits=Limits(dpi=dpi))
    assert path.read_bytes() == original
    assert all(
        c["removed_glyphs"] == 6 and c["outside_marker_changed_pixels"] == 0
        for c in audit["checks"]
    )
    with pymupdf.open(tmp_path / "derived.pdf") as pdf:
        assert all("SAMPLE" not in p.get_text() and "31.75 V" in p.get_text() for p in pdf)
        assert all(len(p.get_drawings()) == 2 for p in pdf)


def test_nonascii_marker_and_single_shared_form_invocation(tmp_path):
    path = document(tmp_path / "original.pdf", form=True, marker="ÉTALON")
    plan = inspect_pdf(path, "ÉTALON")
    assert len(plan["candidates"]) == 2
    selected = [c["candidate_id"] for c in plan["candidates"] if c["page"] == 1]
    apply_plan(path, plan, selected, tmp_path / "derived.pdf", tmp_path / "audit.json")
    with pymupdf.open(tmp_path / "derived.pdf") as pdf:
        assert "ÉTALON" not in pdf[0].get_text()
        assert "ÉTALON" in pdf[1].get_text()


@pytest.mark.parametrize("option", ["mixed", "opaque"])
def test_body_or_ordinary_heading_is_not_a_removable_marker(tmp_path, option):
    path = document(tmp_path / "original.pdf", **{option: True})
    plan = inspect_pdf(path, "SAMPLE")
    assert plan["candidates"] == []


def test_source_selection_and_output_guards(tmp_path):
    path = document(tmp_path / "original.pdf")
    plan = inspect_pdf(path, "SAMPLE")
    ids = [c["candidate_id"] for c in plan["candidates"]]
    with pytest.raises(Refused, match="OUTPUT_ALIASES_INPUT"):
        apply_plan(path, plan, ids, path, tmp_path / "audit.json")
    with pytest.raises(Refused, match="UNKNOWN_CANDIDATE"):
        apply_plan(path, plan, ["missing"], tmp_path / "out.pdf", tmp_path / "audit.json")
    before = path.read_bytes()
    path.write_bytes(before + b"\n")
    with pytest.raises(Refused, match="STALE_INPUT"):
        apply_plan(path, plan, ids, tmp_path / "out.pdf", tmp_path / "audit.json")
    path.write_bytes(before)
    out = tmp_path / "out.pdf"
    out.write_bytes(b"owned content")
    with pytest.raises(Refused, match="OUTPUT_EXISTS"):
        apply_plan(path, plan, ids, out, tmp_path / "audit.json")
    assert out.read_bytes() == b"owned content" and not (tmp_path / "audit.json").exists()


def test_collateral_failure_publishes_nothing(tmp_path, monkeypatch):
    import ragix_core.pdf_watermark as wm

    path = document(tmp_path / "original.pdf")
    plan = inspect_pdf(path, "SAMPLE")

    def refuse(*args):
        raise Refused("RETAINED_GLYPHS_CHANGED")

    monkeypatch.setattr(wm, "_validate", refuse)
    with pytest.raises(Refused, match="RETAINED_GLYPHS_CHANGED"):
        apply_plan(
            path,
            plan,
            [c["candidate_id"] for c in plan["candidates"]],
            tmp_path / "out.pdf",
            tmp_path / "audit.json",
        )
    assert not (tmp_path / "out.pdf").exists() and not (tmp_path / "audit.json").exists()


def test_malformed_graphics_state_is_refused(tmp_path):
    path = document(tmp_path / "original.pdf")
    with pikepdf.open(path) as pdf:
        pdf.pages[0].Contents = pdf.make_stream(b"Q\n")
        pdf.save(tmp_path / "bad.pdf")
    plan = inspect_pdf(tmp_path / "bad.pdf", "SAMPLE")
    assert plan["refusals"] == [{"page": 1, "reason": "UNBALANCED_GRAPHICS_STATE"}]
    assert all(c["page"] != 1 for c in plan["candidates"])


def test_signature_field_is_not_rewritten(tmp_path):
    path = document(tmp_path / "original.pdf")
    with pikepdf.open(path) as pdf:
        field = pdf.make_indirect(pikepdf.Dictionary(FT=pikepdf.Name.Sig, T="Signature"))
        pdf.Root.AcroForm = pikepdf.Dictionary(Fields=pikepdf.Array([field]))
        pdf.save(tmp_path / "signed.pdf")
    with pytest.raises(Refused, match="SIGNATURE_FIELD"):
        inspect_pdf(tmp_path / "signed.pdf", "SAMPLE")


@pytest.mark.parametrize("corruption", ["tokens", "font", "rule", "image"])
def test_validator_detects_actual_collateral_edit(tmp_path, corruption):
    import ragix_core.pdf_watermark as wm

    path = document(tmp_path / "original.pdf")
    data = path.read_bytes()
    plan = inspect_pdf(path, "SAMPLE")
    selected = {
        c["page"] - 1: [wm.Region(c["start_token"], c["end_token"], c["kind"])]
        for c in plan["candidates"]
    }
    with pikepdf.open(path) as pdf:
        for n, regions in selected.items():
            tokens = wm._tokens(pdf.pages[n], Limits())
            pdf.pages[n].Contents = pdf.make_stream(wm._filtered(tokens, (), regions))
        page = pdf.pages[0]
        if corruption == "font":
            page.Resources.Font.F1.BaseFont = pikepdf.Name.Courier
        else:
            raw = page.Contents.read_bytes()
            addition = {
                "tokens": b"\n% collateral rewrite",
                "rule": b"\n0 0 m 200 200 l S",
                "image": b"\nq 40 0 0 40 200 20 cm /Im Do Q",
            }[corruption]
            if corruption == "image":
                im = pdf.make_stream(bytes([255, 0, 0]))
                im.Type = pikepdf.Name.XObject
                im.Subtype = pikepdf.Name.Image
                im.Width = im.Height = 1
                im.BitsPerComponent = 8
                im.ColorSpace = pikepdf.Name.DeviceRGB
                page.Resources.XObject = pikepdf.Dictionary(Im=im)
            page.Contents = pdf.make_stream(raw + addition)
        buffer = io.BytesIO()
        pdf.save(buffer)
    with pikepdf.open(path) as original:
        with pytest.raises(Refused, match="RETAINED_(TOKENS|GLYPHS)_CHANGED"):
            wm._validate(data, buffer.getvalue(), original, selected, Limits())


@pytest.mark.parametrize("unsupported", ["image", "clip", "unknown", "type3"])
def test_unsupported_marker_structures_are_not_selected(tmp_path, unsupported):
    path = document(tmp_path / "original.pdf", pages=1)
    with pikepdf.open(path) as pdf:
        page = pdf.pages[0]
        raw = page.Contents.read_bytes()
        if unsupported == "type3":
            page.Resources.Font.F1.Subtype = pikepdf.Name.Type3
        elif unsupported == "image":
            raw = b"q 10 0 0 10 20 20 cm BI /W 1 /H 1 /BPC 8 /CS /RGB ID \xff\x00\x00 EI Q"
        elif unsupported == "clip":
            raw = raw.replace(b"/Fade gs BT", b"/Fade gs BT 7 Tr")
        else:
            raw = raw.replace(b"/Fade gs BT", b"/Fade gs mysterious BT")
        page.Contents = pdf.make_stream(raw)
        pdf.save(tmp_path / "unsupported.pdf")
    assert not inspect_pdf(tmp_path / "unsupported.pdf", "SAMPLE")["candidates"]


def test_encryption_refused(tmp_path):
    path = document(tmp_path / "original.pdf")
    with pikepdf.open(path) as pdf:
        pdf.save(
            tmp_path / "encrypted.pdf",
            encryption=pikepdf.Encryption(owner="owner", user="open-password"),
        )
    with pytest.raises(Refused, match="OPEN_PASSWORD_REQUIRED"):
        inspect_pdf(tmp_path / "encrypted.pdf", "SAMPLE")


def test_image_under_marker_and_optional_content_survive(tmp_path):
    path = document(tmp_path / "original.pdf")
    with pikepdf.open(path) as pdf:
        group = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.OCG, Name="Layer"))
        pdf.Root.OCProperties = pikepdf.Dictionary(
            OCGs=pikepdf.Array([group]),
            D=pikepdf.Dictionary(ON=pikepdf.Array([group]), Order=pikepdf.Array([group])),
        )
        for page in pdf.pages:
            im = pdf.make_stream(bytes([0, 0, 255]))
            im.Type = pikepdf.Name.XObject
            im.Subtype = pikepdf.Name.Image
            im.Width = im.Height = 1
            im.BitsPerComponent = 8
            im.ColorSpace = pikepdf.Name.DeviceRGB
            page.Resources.XObject = pikepdf.Dictionary(Im=im)
            page.Resources.Properties = pikepdf.Dictionary(Layer=group)
            raw = page.Contents.read_bytes()
            raw = raw.replace(b"q /Fade", b"/OC /Layer BDC q /Fade") + b" EMC\n"
            page.Contents = pdf.make_stream(b"q 60 0 0 60 80 80 cm /Im Do Q\n" + raw)
        pdf.save(tmp_path / "layered.pdf")
    clean(tmp_path / "layered.pdf", tmp_path / "derived.pdf")
    with pymupdf.open(tmp_path / "derived.pdf") as pdf:
        assert all(len(p.get_image_info()) == 1 for p in pdf)
        assert len(pdf.get_ocgs()) == 1


def test_empty_open_password_preserves_encryption_and_permissions(tmp_path):
    path = document(tmp_path / "original.pdf")
    with pikepdf.open(path) as pdf:
        pdf.save(
            tmp_path / "restricted.pdf",
            encryption=pikepdf.Encryption(
                owner="owner", user="", allow=pikepdf.Permissions(extract=False, modify_other=False)
            ),
        )
    clean(tmp_path / "restricted.pdf", tmp_path / "derived.pdf")
    with (
        pikepdf.open(tmp_path / "restricted.pdf") as before,
        pikepdf.open(tmp_path / "derived.pdf") as after,
    ):
        assert after.is_encrypted and after.allow == before.allow
        assert after.encryption.R == before.encryption.R


@pytest.mark.parametrize("opacity", [0.01, 0.125, 0.5, 1.0])
def test_low_alpha_support_does_not_require_pixel_tolerance(tmp_path, opacity):
    path = document(tmp_path / "original.pdf", form=True)
    with pikepdf.open(path) as pdf:
        for page in pdf.pages:
            page.Resources.ExtGState.Fade.ca = opacity
            page.Resources.ExtGState.Fade.CA = opacity
        pdf.save(tmp_path / "faint.pdf")
    _, audit = clean(tmp_path / "faint.pdf", tmp_path / "derived.pdf")
    assert all(c["outside_marker_changed_pixels"] == 0 for c in audit["checks"])
