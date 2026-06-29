"""
KOAS-Translate · stage 1 — extract (PDF → Markdown).

Tries ``pymupdf4llm`` first (fast, text layer); if the PDF has no usable text
layer (scanned/image-typeset books) it falls back to ``marker-pdf`` (ML OCR +
layout). Multiple PDFs in the source dir are concatenated in lexicographic order
with a per-source header. ``max_pages`` caps each PDF (bake-off runs).

The PDF engines are heavy optional deps (the ``[translate]`` extra) and are
imported lazily inside the extractor functions, so this module imports cleanly
for registry discovery on machines without them. The kernel exposes an
``extractor`` seam so its orchestration (multi-PDF concat, source.md writing) is
unit-testable without any PDF dependency.

Ported behaviour-unchanged from the pipeline's ``extract.py``.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput

#: An extractor: (pdf_path, max_pages, out_dir) -> markdown text.
Extractor = Callable[[Path, Optional[int], Path], str]


def _has_text_layer(pdf_path: Path, max_pages: Optional[int] = None) -> bool:
    """True iff any (capped) page yields non-empty extractable text."""
    import pymupdf  # lazy
    doc = pymupdf.open(str(pdf_path))
    try:
        limit = min(len(doc), max_pages) if max_pages else len(doc)
        return any(doc[i].get_text("text").strip() for i in range(limit))
    finally:
        doc.close()


def _extract_pymupdf(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    import pymupdf  # lazy
    import pymupdf4llm  # lazy
    if max_pages is None:
        return pymupdf4llm.to_markdown(str(pdf_path))
    try:
        return pymupdf4llm.to_markdown(str(pdf_path), pages=list(range(max_pages)))
    except TypeError:  # older API — slice to a temp PDF
        import tempfile
        src = pymupdf.open(str(pdf_path))
        dst = pymupdf.open()
        dst.insert_pdf(src, from_page=0, to_page=max_pages - 1)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
            dst.save(t.name)
            tmp = Path(t.name)
        try:
            return pymupdf4llm.to_markdown(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)


def _extract_marker(pdf_path: Path, max_pages: Optional[int], out_dir: Path) -> str:
    """Fallback for image-only PDFs. Heavy ML deps — imported lazily."""
    from marker.converters.pdf import PdfConverter  # type: ignore
    from marker.models import create_model_dict  # type: ignore
    from marker.output import text_from_rendered  # type: ignore

    cfg: Dict[str, object] = {"output_format": "markdown"}
    if max_pages is not None:
        cfg["page_range"] = list(range(max_pages))
    converter = PdfConverter(artifact_dict=create_model_dict(), config=cfg)
    text, _, images = text_from_rendered(converter(str(pdf_path)))
    for name, img in (images or {}).items():
        try:
            img.save(out_dir / name)
        except Exception:  # noqa: BLE001 — image save is best-effort
            pass
    return text


def extract_pdf(pdf_path: Path, max_pages: Optional[int], out_dir: Path) -> str:
    """Convert one PDF to Markdown with the best engine for it."""
    if _has_text_layer(pdf_path, max_pages=max_pages):
        return _extract_pymupdf(pdf_path, max_pages=max_pages)
    return _extract_marker(pdf_path, max_pages, out_dir)


class TranslateExtractKernel(Kernel):
    """Stage 1 — concatenate source PDFs into out/source.md."""

    name = "translate_extract"
    version = "1.0.0"
    category = "translate"
    stage = 1
    description = "Extract source PDF(s) to a single Markdown document (pymupdf4llm, marker fallback)."
    requires: List[str] = []

    #: Optional injected extractor (tests / programmatic); else the real engine.
    extractor: Optional[Extractor] = None

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        cfg = input.config or {}
        src_dir = Path(cfg.get("src_dir", input.workspace / "src"))
        out_dir = Path(cfg.get("out_dir", input.workspace / "out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        source_md = Path(cfg.get("source_md", out_dir / "source.md"))
        max_pages = cfg.get("max_pages")
        extractor = self.extractor or extract_pdf

        pdfs = sorted(src_dir.glob("*.pdf"))
        if not pdfs:
            raise RuntimeError(f"no PDFs found in {src_dir}")

        parts: List[str] = []
        for pdf in pdfs:
            md = extractor(pdf, max_pages, out_dir)
            parts.append(f"<!-- source: {pdf.name} -->\n\n" + md.strip())

        full = "\n\n---\n\n".join(parts) + "\n"
        source_md.write_text(full, encoding="utf-8")
        return {
            "n_pdfs": len(pdfs),
            "pdfs": [p.name for p in pdfs],
            "n_chars": len(full),
            "n_words": len(full.split()),
            "max_pages": max_pages,
            "source_md": str(source_md),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        if "error" in data:
            return f"translate_extract failed: {data['error']}"
        return (f"translate_extract: {data['n_pdfs']} PDF(s) → "
                f"{Path(data['source_md']).name} ({data['n_chars']:,} chars, "
                f"~{data['n_words']:,} words)")
