"""
RAGIX-Sealed — PDF extraction: the default order unchanged, PyMuPDF an explicit opt-in.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The opt-in is AGPL-3.0, so the claim about the default is the strong one: a process that
extracted by the default order has not loaded it. Both extraction routes therefore run in a fresh
interpreter — loading the library in this one would make the claim depend on test order.

Synthetic fixture only: one invented sentence, written byte by byte.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

from generators import _pdf_assemble, _pdf_escape, _pdf_stream  # noqa: E402

from ragix_sealed.contracts import load_contracts  # noqa: E402
from ragix_sealed.ingest import (  # noqa: E402
    ExtractorUnavailable,
    SealedIngestor,
    extract_text,
    new_case_context,
)
from ragix_sealed.ingest import pipeline as pipeline_module  # noqa: E402
from ragix_sealed.vault import RAGIXSealedVaultBackend  # noqa: E402

HAS_DEFAULT = find_spec("pdfminer") is not None or find_spec("pypdf") is not None
HAS_MUPDF = find_spec("pymupdf") is not None

SENTENCE = "Rapport synthetique de test"

PROBE = """
import json, sys
from ragix_sealed.ingest.extract import extract_text
data = open(sys.argv[1], "rb").read()
reader = sys.argv[2] or None
text = (extract_text(data, "pdf") if reader is None
        else extract_text(data, "pdf", pdf_reader=reader))
print(json.dumps({"text": text,
                  "loaded": [m for m in ("pymupdf", "fitz", "pymupdf4llm") if m in sys.modules]}))
"""


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory) -> Path:
    payload = b"BT /F1 12 Tf 72 700 Td (" + _pdf_escape(SENTENCE) + b") Tj ET\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
         b"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>"),
        _pdf_stream(payload),
    ]
    path = tmp_path_factory.mktemp("sealed_pdf") / "synthetic.pdf"
    path.write_bytes(_pdf_assemble(objects))
    return path


def _probe(path: Path, reader: str) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, "-c", PROBE, str(path), reader],
                          capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not HAS_DEFAULT, reason="the 'sealed' extra is not installed")
def test_the_default_order_does_not_load_pymupdf(pdf_path):
    result = _probe(pdf_path, "")
    assert SENTENCE in " ".join(result["text"].split())
    assert result["loaded"] == [], result


@pytest.mark.skipif(not HAS_MUPDF, reason="the 'saqqara-mupdf' extra is not installed")
def test_pymupdf_reads_when_named(pdf_path):
    result = _probe(pdf_path, "pymupdf")
    assert SENTENCE in " ".join(result["text"].split())
    assert "pymupdf" in result["loaded"], "the opt-in is loaded when, and because, it is named"


def test_an_unknown_reader_is_refused():
    with pytest.raises(ValueError, match="unknown PDF reader"):
        extract_text(b"%PDF-1.4", "pdf", pdf_reader="fitz")


def test_a_named_reader_that_is_not_installed_is_refused(monkeypatch):
    """Named and absent is a refusal, never a quiet return to the default order."""
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    with pytest.raises(ExtractorUnavailable, match="pymupdf"):
        extract_text(b"%PDF-1.4", "pdf", pdf_reader="pymupdf")


def test_the_ingestor_passes_the_choice_and_records_it(monkeypatch):
    calls = []

    def stub(data, kind, pdf_reader=None):
        calls.append(pdf_reader)
        return "A synthetic memo with nothing to detect."

    monkeypatch.setattr(pipeline_module, "extract_text", stub)
    ingestor = SealedIngestor(load_contracts(), RAGIXSealedVaultBackend(), pdf_reader="pymupdf")
    status = ingestor.ingest(new_case_context("case_synth_pdf"), b"%PDF-1.4 synthetic", "pdf",
                             ingestion_counter=1)
    assert calls == ["pymupdf"]
    assert ingestor.get_cooled(status.doc_id).sanitized_metadata["pdf_reader"] == "pymupdf"


def test_the_default_ingestor_calls_extraction_as_it_always_did(monkeypatch):
    """Two positional arguments and nothing else: a keyword would break this stub."""
    def stub(data, kind):
        return "A synthetic memo with nothing to detect."

    monkeypatch.setattr(pipeline_module, "extract_text", stub)
    ingestor = SealedIngestor(load_contracts(), RAGIXSealedVaultBackend())
    status = ingestor.ingest(new_case_context("case_synth_pdf"), b"%PDF-1.4 synthetic", "pdf",
                             ingestion_counter=1)
    assert "pdf_reader" not in ingestor.get_cooled(status.doc_id).sanitized_metadata
