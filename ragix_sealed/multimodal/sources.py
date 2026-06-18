"""
RAGIX-Sealed — source-class detection (WP §8bis, Sprint 2bis).

Classifies a source into a controlled set so the pipeline knows whether it needs
extraction, OCR, frame extraction, or transcription. Deterministic: extension first,
magic-byte sniffing as a fallback/confirmation. No external dependency.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from enum import Enum


class SourceClass(Enum):
    """Top-level source classes (WP §8bis.1)."""

    TEXT_DOCUMENT = "TEXT_DOCUMENT"        # extractable text (txt/md/docx)
    SCANNED_DOCUMENT = "SCANNED_DOCUMENT"  # pdf/image likely needing OCR
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    UNKNOWN = "UNKNOWN"


_EXT = {
    "txt": SourceClass.TEXT_DOCUMENT, "text": SourceClass.TEXT_DOCUMENT,
    "md": SourceClass.TEXT_DOCUMENT, "markdown": SourceClass.TEXT_DOCUMENT,
    "docx": SourceClass.TEXT_DOCUMENT, "odt": SourceClass.TEXT_DOCUMENT,
    "rtf": SourceClass.TEXT_DOCUMENT,
    "pdf": SourceClass.SCANNED_DOCUMENT,  # may carry text; OCR-need decided downstream
    "png": SourceClass.IMAGE, "jpg": SourceClass.IMAGE, "jpeg": SourceClass.IMAGE,
    "tif": SourceClass.IMAGE, "tiff": SourceClass.IMAGE, "heic": SourceClass.IMAGE,
    "gif": SourceClass.IMAGE, "bmp": SourceClass.IMAGE, "webp": SourceClass.IMAGE,
    "mp4": SourceClass.VIDEO, "mov": SourceClass.VIDEO, "avi": SourceClass.VIDEO,
    "mkv": SourceClass.VIDEO, "webm": SourceClass.VIDEO,
    "wav": SourceClass.AUDIO, "mp3": SourceClass.AUDIO, "m4a": SourceClass.AUDIO,
    "flac": SourceClass.AUDIO, "ogg": SourceClass.AUDIO,
}

# (magic prefix, class) — used only when the extension is unknown/empty.
_MAGIC = [
    (b"%PDF", SourceClass.SCANNED_DOCUMENT),
    (b"\x89PNG\r\n\x1a\n", SourceClass.IMAGE),
    (b"\xff\xd8\xff", SourceClass.IMAGE),               # JPEG
    (b"GIF87a", SourceClass.IMAGE), (b"GIF89a", SourceClass.IMAGE),
    (b"II*\x00", SourceClass.IMAGE), (b"MM\x00*", SourceClass.IMAGE),  # TIFF
    (b"OggS", SourceClass.AUDIO),
    (b"RIFF", SourceClass.UNKNOWN),                     # WAV/AVI — needs sub-tag, leave UNKNOWN
    (b"PK\x03\x04", SourceClass.TEXT_DOCUMENT),         # zip container (docx/odt)
]


def classify_source(kind: str, data: bytes = b"") -> SourceClass:
    """Classify a source by extension; fall back to magic bytes when kind is unknown."""
    k = (kind or "").lower().lstrip(".").strip()
    if k in _EXT:
        return _EXT[k]
    for prefix, cls in _MAGIC:
        if data.startswith(prefix):
            return cls
    return SourceClass.UNKNOWN
