"""
Text similarity for loop fixed-point detection — RAGIX memory subsystem.

Provides two tiers of text similarity:
  - **Tier A (stdlib):** Token Jaccard + difflib.SequenceMatcher — zero dependencies.
  - **Tier B (RAGIX-native):** Cosine similarity via existing ``embedder.py`` backends.

The ``compute_similarity()`` dispatcher selects the tier automatically (``auto``),
or the caller can force ``lexical`` (Tier A) or ``embedding`` (Tier B).

Query-cycle detection is also provided for the loop controller.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-18
"""

from __future__ import annotations

import difflib
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from ragix_core.memory.embedder import MemoryEmbedder

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Tier A — stdlib lexical similarity
# ---------------------------------------------------------------------------


def token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two normalized texts.

    Returns 1.0 when both strings are empty (vacuously identical).
    """
    tokens_a = set(normalize_text(a).split())
    tokens_b = set(normalize_text(b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def sequence_ratio(a: str, b: str) -> float:
    """``difflib.SequenceMatcher`` ratio between normalized texts.

    Returns 1.0 when both strings are empty.
    """
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def lexical_similarity(
    a: str,
    b: str,
    w_jaccard: float = 0.5,
    w_sequence: float = 0.5,
) -> float:
    """Tier A: weighted blend of Token Jaccard and SequenceMatcher ratio.

    Default weights give equal importance to set overlap and sequential
    ordering — a good default for answer-level fixed-point detection.
    """
    j = token_jaccard(a, b)
    s = sequence_ratio(a, b)
    return w_jaccard * j + w_sequence * s


# ---------------------------------------------------------------------------
# Tier B — embedding cosine similarity (via existing embedder.py)
# ---------------------------------------------------------------------------


def embedding_similarity(a: str, b: str, embedder: "MemoryEmbedder") -> float:
    """Tier B: cosine similarity of embeddings produced by *embedder*.

    Raises ``ValueError`` if the embedder produces vectors of different
    dimensions (should never happen with a well-behaved backend).
    """
    from ragix_core.memory.embedder import cosine_similarity

    vec_a = embedder.embed_text(a)
    vec_b = embedder.embed_text(b)
    return cosine_similarity(vec_a, vec_b)


# ---------------------------------------------------------------------------
# Dispatcher — auto / lexical / embedding
# ---------------------------------------------------------------------------


def compute_similarity(
    a: str,
    b: str,
    mode: str = "auto",
    embedder: Optional["MemoryEmbedder"] = None,
    w_jaccard: float = 0.5,
    w_sequence: float = 0.5,
) -> Tuple[float, str]:
    """Compute similarity between two texts.

    Returns ``(score, method)`` where *method* is ``"lexical"`` or
    ``"cosine"``.

    Parameters
    ----------
    mode : ``"auto"`` | ``"lexical"`` | ``"embedding"``
        - ``auto`` — embedding cosine if *embedder* is available, else lexical.
        - ``lexical`` — Tier A only (stdlib).
        - ``embedding`` — Tier B only (requires *embedder*).
    embedder : optional MemoryEmbedder
        Any object implementing ``embed_text(str) -> List[float]``.
    """
    if mode not in ("auto", "lexical", "embedding"):
        raise ValueError(f"Unknown similarity mode: {mode!r}")

    if mode == "lexical":
        return lexical_similarity(a, b, w_jaccard, w_sequence), "lexical"

    if mode == "embedding":
        if embedder is None:
            raise ValueError("embedding mode requires an embedder instance")
        return embedding_similarity(a, b, embedder), "cosine"

    # --- auto: try embedding first, fall back to lexical ---
    if embedder is not None:
        try:
            score = embedding_similarity(a, b, embedder)
            return score, "cosine"
        except Exception:
            pass  # graceful fallback

    return lexical_similarity(a, b, w_jaccard, w_sequence), "lexical"


# ---------------------------------------------------------------------------
# Query-cycle detection
# ---------------------------------------------------------------------------


def detect_query_cycle(
    query: str,
    previous_queries: List[str],
    threshold: float = 0.90,
    mode: str = "auto",
    embedder: Optional["MemoryEmbedder"] = None,
) -> Tuple[bool, Optional[str]]:
    """Check whether *query* is a repeat or near-duplicate of earlier queries.

    Returns ``(is_cycle, reason)`` — *reason* is ``None`` when no cycle.

    Cycle triggers (in priority order):

    1. Empty query.
    2. Exact match (after normalization) with any previous query.
    3. Similarity >= *threshold* with the most recent previous query.
    """
    if not query or not query.strip():
        return True, "empty query"

    norm_q = normalize_text(query)

    # Exact-match cycle (normalized)
    for i, prev in enumerate(previous_queries):
        if normalize_text(prev) == norm_q:
            return True, f"exact repeat of query {i + 1}"

    # Similarity to most-recent query
    if previous_queries:
        sim, method = compute_similarity(
            query, previous_queries[-1], mode=mode, embedder=embedder
        )
        if sim >= threshold:
            return True, (
                f"similarity {sim:.3f} >= {threshold} ({method})"
                f" vs previous query"
            )

    return False, None
