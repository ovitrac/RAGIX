"""
RAGIX-Sealed — cooled corpus index + safe search (WP §13 Sprint 3).

Wraps the pure-Python ``ragix_core.bm25_index.BM25Index`` over placeholderized chunks.
Boundary discipline:

- only opaque ids ever enter the index (the BM25 ``file_path`` field carries the opaque
  parent ``doc_id``, never a filename — K3);
- a defensive leak re-check runs on every chunk before indexing (deny-by-default — K6/K7);
- search results expose placeholderized snippets and `doc_id/page/chunk_id` citations
  only (K2/K4), never raw content.

A vector index (`+ optional vector`) is a later add behind the `retrieval` extra; this
MVP is BM25-only and dependency-free.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ragix_core.bm25_index import BM25Document, BM25Index

from ..ingest.ids import CaseContext
from ..ingest.leak_scan import scan as leak_scan
from .chunking import CooledChunk, chunk_cooled


class CorpusError(Exception):
    """Raised when content failing the boundary check is offered to the index."""


@dataclass(frozen=True)
class SafeSearchResult:
    """A safe, SANITIZED_LLM_SAFE search hit. No raw content, no filenames."""

    chunk_id: str
    doc_id: str
    page: int
    score: float
    snippet: str               # placeholderized chunk text
    matched_terms: List[str] = field(default_factory=list)

    def citation(self) -> str:
        """K4 citation form: `doc_id/page/chunk_id`."""
        return f"{self.doc_id}/page_{self.page}/{self.chunk_id}"

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page": self.page,
            "score": round(self.score, 4),
            "snippet": self.snippet,
            "citation": self.citation(),
            "matched_terms": list(self.matched_terms),
        }


class CooledCorpusIndex:
    """BM25 index over cooled (placeholderized) chunks with a safe search API."""

    def __init__(self, contracts: Any) -> None:
        self._schema = contracts.placeholder_schema
        self._bm25 = BM25Index()
        self._chunks: Dict[str, CooledChunk] = {}

    def add_document(self, ctx: CaseContext, doc_id: str, cooled_text: str) -> List[str]:
        """Chunk and index cooled text. Returns the chunk ids added.

        Raises ``CorpusError`` if any chunk fails the defensive leak re-check — cooled
        text should already be clean, so this is defense-in-depth at the index boundary.
        """
        chunks = chunk_cooled(ctx, doc_id, cooled_text)
        added: List[str] = []
        for ch in chunks:
            verdict = leak_scan(ch.text, [], self._schema)
            if verdict.verdict != "PASS":
                raise CorpusError(
                    f"chunk {ch.chunk_id} failed boundary leak check ({verdict.verdict}); not indexed"
                )
            # Opaque ids only: doc_id=chunk_id, file_path=opaque parent doc_id, name=chunk_id.
            self._bm25.add_document(BM25Document(
                doc_id=ch.chunk_id,
                file_path=ch.doc_id,
                start_line=ch.page,
                end_line=ch.block,
                chunk_type="cooled_text",
                name=ch.chunk_id,
                tokens=self._bm25.tokenizer.tokenize(ch.text),
            ))
            self._chunks[ch.chunk_id] = ch
            added.append(ch.chunk_id)
        return added

    def search(self, query: str, k: int = 10) -> List[SafeSearchResult]:
        """Safe keyword search over the cooled corpus."""
        results: List[SafeSearchResult] = []
        for r in self._bm25.search(query, k=k):
            ch = self._chunks.get(r.doc_id)
            if ch is None:
                continue
            results.append(SafeSearchResult(
                chunk_id=ch.chunk_id, doc_id=ch.doc_id, page=ch.page,
                score=r.score, snippet=ch.text, matched_terms=r.matched_terms,
            ))
        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
