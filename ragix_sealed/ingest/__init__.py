"""
RAGIX-Sealed warm ingestion (WP §8, Sprint 2).

Dependency-free TXT/MD ingestion pipeline that drives the contract state machine from
RECEIVED to COOLED_INDEXABLE, sealing originals and placeholderizing via the vault.
PDF/DOCX/OCR/multimodal are deferred (Sprint 2bis).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from .detect import Detection, detect_entities
from .extract import ExtractorUnavailable, extract_text
from .ids import CaseContext, new_case_context, raw_sha256
from .leak_scan import LeakVerdict, scan
from .pipeline import CooledDocument, IngestError, IngestStatus, SealedIngestor

__all__ = [
    "CaseContext",
    "new_case_context",
    "raw_sha256",
    "extract_text",
    "ExtractorUnavailable",
    "Detection",
    "detect_entities",
    "LeakVerdict",
    "scan",
    "SealedIngestor",
    "IngestStatus",
    "CooledDocument",
    "IngestError",
]
