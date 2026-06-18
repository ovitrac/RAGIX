"""
RAGIX-Sealed — cooled-text chunking (WP §13 Sprint 3).

Splits a cooled (placeholderized, leak-scanned) document into chunks with opaque,
case-bound chunk ids. Operates only on already-cooled text — never raw content.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..ingest.ids import CaseContext


@dataclass(frozen=True)
class CooledChunk:
    """A placeholderized text chunk with an opaque chunk id. SANITIZED_LLM_SAFE."""

    chunk_id: str
    doc_id: str
    page: int
    block: int
    text: str


def _split_by_length(text: str, max_chars: int) -> List[str]:
    """Split a long block on whitespace boundaries into <= max_chars pieces."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: List[str] = []
    cur = ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            pieces.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_cooled(ctx: CaseContext, doc_id: str, text: str, max_chars: int = 600) -> List[CooledChunk]:
    """Chunk cooled text into blocks, assigning opaque `chunk_id`s (WP §5.2).

    Blocks split on blank lines first, then by length. Page is 1 for the text MVP
    (multimodal/paged sources set it per source).
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks and text.strip():
        blocks = [text.strip()]

    chunks: List[CooledChunk] = []
    block_no = 0
    for raw_block in blocks:
        for piece in _split_by_length(raw_block, max_chars):
            block_no += 1
            cid = ctx.chunk_id(doc_id, page=1, block=block_no)
            chunks.append(CooledChunk(chunk_id=cid, doc_id=doc_id, page=1, block=block_no, text=piece))
    return chunks
