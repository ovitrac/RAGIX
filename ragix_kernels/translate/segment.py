"""
KOAS-Translate · stage 1 — segment.

Splits the extracted Markdown into LLM-sized translation units, masks
non-translatable spans with the shared protected-span codec, and records each
segment in the SQLite translation memory.

Chunking (ported from the translation pipeline's ``segment.py``, behaviour
unchanged) boundaries, in priority order:
  1. **Chapter** (``# Heading``) — always flushes (chapters are the unit of
     harmonization).
  2. **Soft target** (``target_words``) — flush once the buffer reaches it.
  3. **Hard cap** (``max_words``) — flush before exceeding it; a single
     paragraph over the cap becomes its own chunk.
Subsection (``##``+) boundaries do *not* force a flush, so micro-sections
coalesce instead of paying full prompt overhead on tiny payloads.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.shared.protected_spans import SpanCounter, protect

from . import tm_store

DEFAULT_TARGET_WORDS = 1500
DEFAULT_MAX_WORDS = 2500

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _word_count(s: str) -> int:
    return len(s.split())


def _split_paragraphs(block: str) -> List[str]:
    """Split by blank lines; preserve paragraph boundaries."""
    paras = re.split(r"\n\s*\n", block.strip())
    return [p for p in paras if p.strip()]


@dataclass
class Chunk:
    segment_id: str
    chapter: str | None
    section: str | None
    order_idx: int
    source_text: str
    protected_map: Dict[str, str]


def _iter_heading_blocks(md: str) -> Iterable[tuple[str | None, str | None, str]]:
    """Yield ``(chapter, section, block_text)`` triples.

    A *chapter* is the most recent ``# Heading``; a *section* is the most recent
    heading at level 2+. The block is everything from one heading line up to (but
    not including) the next.
    """
    chapter: str | None = None
    section: str | None = None
    matches = list(HEADING_RE.finditer(md))
    if not matches:
        yield None, None, md
        return

    if matches[0].start() > 0:
        pre = md[: matches[0].start()].strip()
        if pre:
            yield None, None, pre

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        if level == 1:
            chapter, section = title, None
        else:
            section = title
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        yield chapter, section, md[start:end]


def _finalize(paras: List[str], chapter: str | None, section: str | None,
              order: int, counter: SpanCounter) -> Chunk:
    raw = "\n\n".join(paras)
    protected, mapping = protect(raw, counter=counter)
    return Chunk(
        segment_id=f"seg-{order:05d}",
        chapter=chapter,
        section=section,
        order_idx=order,
        source_text=protected,
        protected_map=mapping,
    )


def chunk_markdown(
    md: str,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> List[Chunk]:
    """Chunk *md* into translation units; mask protected spans per chunk.

    A single ``SpanCounter`` spans the whole document so ``⟦P####⟧`` token names
    never collide across chunks (rebuild unions per-chunk maps per chapter).
    """
    counter = SpanCounter()
    chunks: List[Chunk] = []
    order = 0

    buf: List[str] = []
    buf_words = 0
    buf_chapter: str | None = None
    buf_section: str | None = None

    def _emit() -> None:
        nonlocal buf, buf_words, order
        if not buf:
            return
        chunks.append(_finalize(buf, buf_chapter, buf_section, order, counter))
        order += 1
        buf, buf_words = [], 0

    for chapter, section, block in _iter_heading_blocks(md):
        # Chapter boundary — always flush whatever is buffered.
        if buf and chapter != buf_chapter:
            _emit()

        paras = _split_paragraphs(block)
        if not paras:
            continue

        if not buf:
            buf_chapter = chapter
            buf_section = section

        for p in paras:
            pw = _word_count(p)

            # A single paragraph already over the cap → emit buffer, then it alone.
            if pw > max_words:
                _emit()
                chunks.append(_finalize([p], chapter, section, order, counter))
                order += 1
                continue

            # Would adding p exceed the cap? Flush, restart metadata here.
            if buf_words + pw > max_words and buf:
                _emit()
                buf_chapter = chapter
                buf_section = section

            buf.append(p)
            buf_words += pw

            if buf_words >= target_words:
                _emit()
        # No flush at end of heading block — subsections merge within a chapter.

    _emit()
    return chunks


class TranslateSegmentKernel(Kernel):
    """Stage 1 — Markdown → chunks + translation-memory rows (protected)."""

    name = "translate_segment"
    version = "1.0.0"
    category = "translate"
    stage = 1
    description = "Segment extracted Markdown into protected translation units and record them in the TM."
    requires = ["translate_extract"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        cfg = input.config or {}
        # Source markdown: prefer the extract kernel's declared dependency.
        src = input.dependencies.get("translate_extract")
        if src is None:
            src = Path(cfg.get("source_md", input.workspace / "out" / "source.md"))
        md = Path(src).read_text(encoding="utf-8")

        target = int(cfg.get("target_words", DEFAULT_TARGET_WORDS))
        max_w = int(cfg.get("max_words", DEFAULT_MAX_WORDS))
        chunks = chunk_markdown(md, target_words=target, max_words=max_w)

        out_dir = Path(cfg.get("out_dir", input.workspace / "out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        chunks_jsonl = out_dir / "chunks.jsonl"
        with chunks_jsonl.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({
                    "segment_id": c.segment_id,
                    "chapter": c.chapter,
                    "section": c.section,
                    "order_idx": c.order_idx,
                    "source_text": c.source_text,
                    "protected_map": c.protected_map,
                    "word_count": _word_count(c.source_text),
                }, ensure_ascii=False) + "\n")

        tm_path = Path(cfg.get("tm_path", out_dir / "tm.sqlite"))
        with tm_store.connect(tm_path) as conn:
            for c in chunks:
                tm_store.upsert_source_segment(
                    conn,
                    segment_id=c.segment_id,
                    chapter=c.chapter,
                    section=c.section,
                    order_idx=c.order_idx,
                    source_text=c.source_text,
                    protected_map=c.protected_map,
                )

        protected_total = sum(len(c.protected_map) for c in chunks)
        chapters = sorted({c.chapter for c in chunks if c.chapter is not None})
        return {
            "n_chunks": len(chunks),
            "n_protected_spans": protected_total,
            "n_chapters": len(chapters),
            "target_words": target,
            "max_words": max_w,
            "chunks_jsonl": str(chunks_jsonl),
            "tm_path": str(tm_path),
            "segment_ids": [c.segment_id for c in chunks],
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        if "error" in data:
            return f"translate_segment failed: {data['error']}"
        return (f"translate_segment: {data['n_chunks']} chunks across "
                f"{data['n_chapters']} chapter(s), {data['n_protected_spans']} "
                f"protected spans → {Path(data['tm_path']).name}")
