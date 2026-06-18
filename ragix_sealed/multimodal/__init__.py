"""
RAGIX-Sealed multimodal support (WP §8bis, Sprint 2bis).

Source-class detection and a derivative provenance graph. Images, video, OCR, captions,
and transcripts are first-class controlled derivatives; the provenance graph is the
traceability backbone (opaque ids, parent links) and carries no raw content.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from .provenance import (
    ProvenanceError,
    ProvenanceGraph,
    ProvenanceNode,
    asset_id,
    caption_id,
    frame_id,
    ocr_id,
    segment_id,
    transcript_id,
)
from .sources import SourceClass, classify_source

__all__ = [
    "SourceClass",
    "classify_source",
    "ProvenanceGraph",
    "ProvenanceNode",
    "ProvenanceError",
    "asset_id",
    "frame_id",
    "segment_id",
    "ocr_id",
    "caption_id",
    "transcript_id",
]
