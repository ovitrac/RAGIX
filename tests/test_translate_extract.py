"""
Tests for the KOAS-Translate extract kernel (ragix_kernels/translate/extract).

The orchestration (multi-PDF concat, ordering, source.md writing) is tested via
an injected stub extractor — no PDF dependency. A separate skipif-guarded test
exercises the real pymupdf4llm path when the [translate] extra is installed.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import sys
from pathlib import Path

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.translate import extract as extract_module
from ragix_kernels.translate.extract import PERMISSIVE_MARK, TranslateExtractKernel


def _stub(pdf, max_pages, out_dir):
    return f"# {pdf.stem}\n\nContent of {pdf.name}."


def _run(root, **cfg):
    kernel = TranslateExtractKernel()
    kernel.extractor = _stub
    return kernel.run(KernelInput(workspace=root, config=cfg, dependencies={}))


class TestExtractKernel:
    def test_concatenates_pdfs_with_headers(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.pdf").write_bytes(b"%PDF-1.4")
        (src / "b.pdf").write_bytes(b"%PDF-1.4")
        out = _run(tmp_path)
        assert out.success, out.errors
        assert out.data["n_pdfs"] == 2
        assert out.data["pdfs"] == ["a.pdf", "b.pdf"]
        text = Path(out.data["source_md"]).read_text(encoding="utf-8")
        assert "<!-- source: a.pdf -->" in text
        assert "<!-- source: b.pdf -->" in text
        assert "\n---\n" in text                       # separator between sources

    def test_lexicographic_order(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        for n in ["02.pdf", "01.pdf", "10.pdf"]:
            (src / n).write_bytes(b"%PDF")
        out = _run(tmp_path)
        assert out.data["pdfs"] == ["01.pdf", "02.pdf", "10.pdf"]

    def test_no_pdfs_fails_cleanly(self, tmp_path):
        (tmp_path / "src").mkdir()
        out = _run(tmp_path)
        assert not out.success
        assert "no PDFs" in (out.data.get("error", "") + " ".join(out.errors))


def test_real_pymupdf4llm_extraction(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("pymupdf4llm")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World from a real PDF")
    src = tmp_path / "src"
    src.mkdir()
    doc.save(str(src / "x.pdf"))
    out = TranslateExtractKernel().run(
        KernelInput(workspace=tmp_path, config={}, dependencies={}))
    assert out.success, out.errors
    assert "Hello World" in Path(out.data["source_md"]).read_text(encoding="utf-8")


# -- the permissive fallback (2026-09-14): used only where PyMuPDF is not installed ------

def _text_pdf(path: Path, sentence: str) -> Path:
    """One synthetic page, written byte by byte — no PDF library needed to make it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "saqqara"))
    from generators import _pdf_assemble, _pdf_escape, _pdf_stream

    payload = b"BT /F1 12 Tf 72 700 Td (" + _pdf_escape(sentence) + b") Tj ET\n"
    path.write_bytes(_pdf_assemble([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
         b"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>"),
        _pdf_stream(payload),
    ]))
    return path


def _absent(monkeypatch):
    """PyMuPDF made unimportable for the duration of one test, installed or not."""
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "pymupdf4llm", None)


def test_absence_is_detected_without_importing(monkeypatch):
    _absent(monkeypatch)
    assert extract_module._pymupdf_installed() is False


def test_permissive_fallback_when_pymupdf_is_absent(tmp_path, monkeypatch):
    pytest.importorskip("pypdf")
    _absent(monkeypatch)
    src = tmp_path / "src"
    src.mkdir()
    _text_pdf(src / "x.pdf", "Rapport synthetique de test")
    out = TranslateExtractKernel().run(
        KernelInput(workspace=tmp_path, config={}, dependencies={}))
    assert out.success, out.errors
    text = Path(out.data["source_md"]).read_text(encoding="utf-8")
    assert PERMISSIVE_MARK in text, "the reader that produced it is on the page"
    assert "Rapport synthetique de test" in text


def test_the_installed_route_is_the_one_it_always_was(tmp_path, monkeypatch):
    """With PyMuPDF installed the fallback is never consulted."""
    calls = []
    monkeypatch.setattr(extract_module, "_pymupdf_installed", lambda: True)
    monkeypatch.setattr(extract_module, "_has_text_layer",
                        lambda p, max_pages=None: calls.append("layer") or True)
    monkeypatch.setattr(extract_module, "_extract_pymupdf",
                        lambda p, max_pages=None: calls.append("pymupdf") or "md")
    monkeypatch.setattr(extract_module, "_extract_permissive",
                        lambda *a: calls.append("permissive") or "fallback")
    assert extract_module.extract_pdf(tmp_path / "x.pdf", None, tmp_path) == "md"
    assert calls == ["layer", "pymupdf"]
