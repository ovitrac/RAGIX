"""
Tests for ragix_core.memory.similarity — Tier A/B similarity and cycle detection.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-19
"""

import pytest

from ragix_core.memory.similarity import (
    compute_similarity,
    detect_query_cycle,
    embedding_similarity,
    lexical_similarity,
    normalize_text,
    sequence_ratio,
    token_jaccard,
)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strip_punctuation(self):
        assert normalize_text("foo, bar! baz?") == "foo bar baz"

    def test_collapse_whitespace(self):
        assert normalize_text("  foo   bar  ") == "foo bar"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_only_punctuation(self):
        assert normalize_text("...!!!") == ""

    def test_unicode_preserved(self):
        # Unicode word characters are kept
        assert "é" in normalize_text("café")


# ---------------------------------------------------------------------------
# token_jaccard
# ---------------------------------------------------------------------------


class TestTokenJaccard:
    def test_identical(self):
        assert token_jaccard("the quick brown fox", "the quick brown fox") == 1.0

    def test_disjoint(self):
        assert token_jaccard("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap(self):
        # {"the", "cat"} & {"the", "dog"} = {"the"} => 1/3
        assert token_jaccard("the cat", "the dog") == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert token_jaccard("", "") == 1.0

    def test_one_empty(self):
        assert token_jaccard("hello", "") == 0.0
        assert token_jaccard("", "hello") == 0.0

    def test_case_insensitive(self):
        assert token_jaccard("Hello World", "hello world") == 1.0

    def test_punctuation_ignored(self):
        assert token_jaccard("hello, world!", "hello world") == 1.0

    def test_duplicate_tokens(self):
        # set-based: duplicates don't matter
        assert token_jaccard("a a a b", "a b") == 1.0


# ---------------------------------------------------------------------------
# sequence_ratio
# ---------------------------------------------------------------------------


class TestSequenceRatio:
    def test_identical(self):
        assert sequence_ratio("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert sequence_ratio("aaa", "zzz") == 0.0

    def test_both_empty(self):
        assert sequence_ratio("", "") == 1.0

    def test_one_empty(self):
        assert sequence_ratio("hello", "") == 0.0

    def test_partial_match(self):
        r = sequence_ratio("the quick brown fox", "the slow brown fox")
        assert 0.5 < r < 1.0

    def test_order_matters(self):
        # SequenceMatcher is order-sensitive — swapping words drops ratio
        r1 = sequence_ratio("alpha beta gamma", "alpha beta gamma")
        r2 = sequence_ratio("alpha beta gamma", "gamma beta alpha")
        assert r1 > r2


# ---------------------------------------------------------------------------
# lexical_similarity
# ---------------------------------------------------------------------------


class TestLexicalSimilarity:
    def test_identical(self):
        assert lexical_similarity("hello world", "hello world") == 1.0

    def test_both_empty(self):
        assert lexical_similarity("", "") == 1.0

    def test_range_zero_one(self):
        s = lexical_similarity("the quick brown fox", "something entirely different")
        assert 0.0 <= s <= 1.0

    def test_custom_weights(self):
        a, b = "the cat sat", "the dog sat"
        s_j = lexical_similarity(a, b, w_jaccard=1.0, w_sequence=0.0)
        s_s = lexical_similarity(a, b, w_jaccard=0.0, w_sequence=1.0)
        s_half = lexical_similarity(a, b, w_jaccard=0.5, w_sequence=0.5)
        assert s_half == pytest.approx(0.5 * s_j + 0.5 * s_s)


# ---------------------------------------------------------------------------
# embedding_similarity (Tier B — with MockEmbedder)
# ---------------------------------------------------------------------------


class TestEmbeddingSimilarity:
    @pytest.fixture()
    def mock_embedder(self):
        from ragix_core.memory.embedder import MockEmbedder
        return MockEmbedder(dimension=32, seed=42)

    def test_identical_text(self, mock_embedder):
        # Same text → same embedding → cosine = 1.0
        s = embedding_similarity("hello world", "hello world", mock_embedder)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_different_text(self, mock_embedder):
        s = embedding_similarity("hello world", "goodbye universe", mock_embedder)
        assert -1.0 <= s <= 1.0
        assert s < 1.0  # different texts ≠ identical

    def test_empty_text(self, mock_embedder):
        # MockEmbedder hashes the text; empty string is still a valid hash
        s = embedding_similarity("", "", mock_embedder)
        assert s == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_similarity — dispatcher
# ---------------------------------------------------------------------------


class TestComputeSimilarity:
    @pytest.fixture()
    def mock_embedder(self):
        from ragix_core.memory.embedder import MockEmbedder
        return MockEmbedder(dimension=32, seed=42)

    def test_lexical_mode(self):
        score, method = compute_similarity("hello world", "hello world", mode="lexical")
        assert method == "lexical"
        assert score == pytest.approx(1.0)

    def test_embedding_mode(self, mock_embedder):
        score, method = compute_similarity(
            "hello world", "hello world", mode="embedding", embedder=mock_embedder
        )
        assert method == "cosine"
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_embedding_mode_no_embedder_raises(self):
        with pytest.raises(ValueError, match="requires an embedder"):
            compute_similarity("a", "b", mode="embedding", embedder=None)

    def test_auto_with_embedder(self, mock_embedder):
        score, method = compute_similarity(
            "hello", "hello", mode="auto", embedder=mock_embedder
        )
        assert method == "cosine"

    def test_auto_without_embedder(self):
        score, method = compute_similarity("hello", "hello", mode="auto")
        assert method == "lexical"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown similarity mode"):
            compute_similarity("a", "b", mode="invalid")

    def test_auto_fallback_on_embedder_error(self):
        """If the embedder raises, auto mode falls back to lexical."""

        class BrokenEmbedder:
            def embed_text(self, text):
                raise RuntimeError("embedder unavailable")

        score, method = compute_similarity(
            "hello world", "hello world", mode="auto", embedder=BrokenEmbedder()
        )
        assert method == "lexical"
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# detect_query_cycle
# ---------------------------------------------------------------------------


class TestDetectQueryCycle:
    def test_empty_query(self):
        is_cycle, reason = detect_query_cycle("", ["prev"])
        assert is_cycle is True
        assert "empty" in reason

    def test_whitespace_only_query(self):
        is_cycle, reason = detect_query_cycle("   ", [])
        assert is_cycle is True

    def test_exact_repeat(self):
        is_cycle, reason = detect_query_cycle(
            "What is RAGIX?", ["What is RAGIX?"]
        )
        assert is_cycle is True
        assert "exact repeat" in reason

    def test_exact_repeat_case_insensitive(self):
        is_cycle, reason = detect_query_cycle(
            "WHAT IS RAGIX?", ["what is ragix"]
        )
        assert is_cycle is True

    def test_near_duplicate(self):
        is_cycle, reason = detect_query_cycle(
            "What is RAGIX used for?",
            ["What is RAGIX used for today?"],
            threshold=0.80,
        )
        assert is_cycle is True
        assert "similarity" in reason

    def test_no_cycle(self):
        is_cycle, reason = detect_query_cycle(
            "How does ingestion work?",
            ["What is RAGIX?", "Explain the safety model"],
            threshold=0.90,
        )
        assert is_cycle is False
        assert reason is None

    def test_no_previous_queries(self):
        is_cycle, reason = detect_query_cycle("first query", [])
        assert is_cycle is False
        assert reason is None

    def test_exact_repeat_deep_in_history(self):
        """Exact match triggers even if query is not the most recent."""
        is_cycle, reason = detect_query_cycle(
            "first query",
            ["first query", "second query", "third query"],
        )
        assert is_cycle is True
        assert "query 1" in reason

    def test_similarity_only_checks_most_recent(self):
        """Near-duplicate check is only against the last query, not all."""
        is_cycle, reason = detect_query_cycle(
            "How does ingestion work exactly?",
            [
                "How does ingestion work?",          # similar but NOT last
                "Explain the safety model in RAGIX",  # last — dissimilar
            ],
            threshold=0.90,
        )
        # Should NOT cycle: exact match fails (different text),
        # and sim check is only vs "Explain the safety model..."
        assert is_cycle is False
