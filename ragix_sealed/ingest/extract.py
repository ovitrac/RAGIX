"""
RAGIX-Sealed — pluggable text extraction (WP §8, Sprint 2).

TXT/MD extraction is dependency-free. PDF and DOCX use lazy imports of the optional
``sealed`` extra (``pip install .[sealed]`` → pypdf / pdfminer.six / python-docx). If the
dependency is absent, ``ExtractorUnavailable`` is raised with a precise install hint —
deps stay strictly opt-in and scoped to the sealed subsystem (per project CLAUDE.md).

IMPORTANT: all extraction runs INSIDE the sealed ingestion enclave on already-decrypted
bytes; the extracted text is INTERNAL and never returned across the LLM boundary.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

import io

class ExtractorUnavailable(RuntimeError):
    """Raised when extraction for a source kind needs an uninstalled optional dependency."""


# Source kinds handled with the standard library, no external dependency.
_TEXT_KINDS = {"txt", "text", "md", "markdown"}


def normalize_kind(kind: str) -> str:
    return kind.lower().lstrip(".").strip()


#: The PDF readers a caller may name instead of the default order. PyMuPDF is
#: AGPL-3.0 (README, "Optional extras and their licences"): it is imported only when
#: named here, never as a fallback, and naming it without it installed is refused.
OPT_IN_PDF_READERS = ("pymupdf",)


def _extract_pdf_opt_in(data: bytes, reader: str) -> str:
    """Extract PDF text with a reader the caller chose explicitly."""
    if reader not in OPT_IN_PDF_READERS:
        raise ValueError(
            f"unknown PDF reader {reader!r}: the default order needs none, "
            f"and the opt-in readers are {OPT_IN_PDF_READERS}"
        )
    try:
        import pymupdf  # AGPL-3.0 -- opt-in only, see OPT_IN_PDF_READERS
    except ImportError as exc:
        raise ExtractorUnavailable(
            "the PDF reader 'pymupdf' was requested and is not installed; it is "
            "AGPL-3.0 (the 'saqqara-mupdf' extra). The default order (pdfminer.six, "
            "then pypdf) needs no choice."
        ) from exc
    document = pymupdf.open(stream=data, filetype="pdf")
    try:
        return "\n".join(page.get_text("text") for page in document)
    finally:
        document.close()


def _extract_pdf(data: bytes, reader: str | None = None) -> str:
    """Extract PDF text via pdfminer.six (preferred) or pypdf, both from the sealed extra.

    ``reader`` names an opt-in reader instead (``OPT_IN_PDF_READERS``); ``None``, the
    default, is the order above, unchanged.
    """
    if reader is not None:
        return _extract_pdf_opt_in(data, reader)
    try:
        from pdfminer.high_level import extract_text as _pdfminer_extract
    except ImportError:
        pass
    else:
        return _pdfminer_extract(io.BytesIO(data))
    try:
        import pypdf
    except ImportError as exc:
        raise ExtractorUnavailable(
            "PDF extraction requires the 'sealed' extra (pip install .[sealed]): "
            "neither pdfminer.six nor pypdf is available."
        ) from exc
    reader = pypdf.PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(data: bytes) -> str:
    """Extract DOCX paragraph text via python-docx (from the sealed/docs extra)."""
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise ExtractorUnavailable(
            "DOCX extraction requires the 'sealed' extra (pip install .[sealed]): "
            "python-docx is not available."
        ) from exc
    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def extract_text(data: bytes, kind: str, pdf_reader: str | None = None) -> str:
    """Extract UTF-8 text from ``data`` for the given source ``kind``.

    ``pdf_reader`` names an opt-in PDF reader (``OPT_IN_PDF_READERS``); ``None`` keeps
    the default order. It is read for PDF sources only.

    Raises:
        ExtractorUnavailable: for an unknown kind, or a PDF/DOCX without the optional dep.
        ValueError: for a ``pdf_reader`` that is not an opt-in reader.
        UnicodeDecodeError: if a text source is not valid UTF-8 (caller decides policy).
    """
    k = normalize_kind(kind)
    if k in _TEXT_KINDS:
        return data.decode("utf-8")
    if k == "pdf":
        return _extract_pdf(data, pdf_reader)
    if k == "docx":
        return _extract_docx(data)
    raise ExtractorUnavailable(f"unsupported source kind: {kind!r}")
