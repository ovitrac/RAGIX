"""
Tests for the KOAS-Translate extract kernel (ragix_kernels/translate/extract).

The orchestration (multi-PDF concat, ordering, source.md writing) is tested via
an injected stub extractor — no PDF dependency. A separate skipif-guarded test
exercises the real pymupdf4llm path when the [translate] extra is installed.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from pathlib import Path

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.translate.extract import TranslateExtractKernel


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
