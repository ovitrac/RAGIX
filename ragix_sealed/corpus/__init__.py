"""
RAGIX-Sealed cooled corpus index (WP §13 Sprint 3).

BM25 search over placeholderized chunks with opaque ids and a safe, SANITIZED_LLM_SAFE
result surface. Vector search is a later add behind the `retrieval` extra.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from .chunking import CooledChunk, chunk_cooled
from .cooled_index import CooledCorpusIndex, CorpusError, SafeSearchResult

__all__ = [
    "CooledChunk",
    "chunk_cooled",
    "CooledCorpusIndex",
    "CorpusError",
    "SafeSearchResult",
]
