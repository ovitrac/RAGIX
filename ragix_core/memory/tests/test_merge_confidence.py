"""
Tests for merge confidence blending in consolidate.py.

Validates:
- Single item cluster -> confidence = that item's confidence
- Two items with high tag overlap -> higher confidence
- Recency decay: recent items get higher scores
- Validation bonus: verified > unverified > retracted
- Result always in [0.0, 1.0]
- Edge case: all items have confidence=0

Uses MockEmbedder for deterministic embeddings.
"""

import math
import pytest
from datetime import datetime, timezone, timedelta

from ragix_core.memory.config import ConsolidateConfig
from ragix_core.memory.consolidate import ConsolidationPipeline
from ragix_core.memory.embedder import MockEmbedder
from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryProvenance, _now_iso


@pytest.fixture
def store():
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def embedder():
    return MockEmbedder(dimension=32, seed=42)


@pytest.fixture
def pipeline(store, embedder):
    config = ConsolidateConfig(fallback_to_deterministic=True)
    return ConsolidationPipeline(store=store, embedder=embedder, config=config)


def _make_item(
    item_id: str,
    confidence: float = 0.5,
    tags: list = None,
    validation: str = "unverified",
    updated_at: str = None,
) -> MemoryItem:
    """Helper to create a MemoryItem with controlled parameters."""
    return MemoryItem(
        id=item_id,
        tier="stm",
        type="fact",
        title=f"Item {item_id}",
        content=f"Content for {item_id}.",
        tags=tags or ["tag-a"],
        provenance=MemoryProvenance(source_kind="doc", source_id=f"doc_{item_id}"),
        confidence=confidence,
        validation=validation,
        updated_at=updated_at or _now_iso(),
    )


# ---------------------------------------------------------------------------
# Single item cluster
# ---------------------------------------------------------------------------

class TestMergeSingleItem:
    def test_single_item_confidence_dominates(self, pipeline):
        """With a single item, max_conf component = that item's confidence."""
        item = _make_item("S1", confidence=0.8, validation="unverified")
        conf = pipeline._compute_merge_confidence([item])
        # max_conf * 0.3 + overlap(0) * 0.3 + recency(~1) * 0.2 + validation(0.5) * 0.2
        # = 0.24 + 0 + ~0.2 + 0.1 = ~0.54
        assert 0.0 <= conf <= 1.0
        # Should reflect the item's confidence contribution
        assert conf > 0.3  # non-trivial

    def test_single_high_confidence_item(self, pipeline):
        """A single item with confidence=1.0 should produce a high merged conf."""
        item = _make_item("S2", confidence=1.0, validation="verified")
        conf = pipeline._compute_merge_confidence([item])
        assert conf > 0.6  # high individual + verified bonus


# ---------------------------------------------------------------------------
# Tag overlap
# ---------------------------------------------------------------------------

class TestMergeTagOverlap:
    def test_high_overlap_boosts_confidence(self, pipeline):
        """Items sharing many tags should produce higher merged confidence."""
        item_a = _make_item("A1", confidence=0.5, tags=["db", "sql", "storage"])
        item_b = _make_item("A2", confidence=0.5, tags=["db", "sql", "cache"])
        conf_high = pipeline._compute_merge_confidence([item_a, item_b])

        item_c = _make_item("A3", confidence=0.5, tags=["network", "tcp"])
        item_d = _make_item("A4", confidence=0.5, tags=["graphics", "gpu"])
        conf_low = pipeline._compute_merge_confidence([item_c, item_d])

        # High overlap (db, sql shared) should produce higher confidence
        assert conf_high > conf_low

    def test_identical_tags_max_overlap(self, pipeline):
        """Items with identical tags should have jaccard_overlap = 1.0."""
        item_a = _make_item("B1", confidence=0.5, tags=["alpha", "beta"])
        item_b = _make_item("B2", confidence=0.5, tags=["alpha", "beta"])
        conf = pipeline._compute_merge_confidence([item_a, item_b])
        # overlap=1.0 * 0.3 + rest => should be notably high
        assert conf > 0.4

    def test_no_tag_overlap(self, pipeline):
        """Items with zero shared tags should have jaccard_overlap = 0."""
        item_a = _make_item("C1", confidence=0.5, tags=["alpha"])
        item_b = _make_item("C2", confidence=0.5, tags=["beta"])
        conf_no_overlap = pipeline._compute_merge_confidence([item_a, item_b])

        item_c = _make_item("C3", confidence=0.5, tags=["alpha"])
        item_d = _make_item("C4", confidence=0.5, tags=["alpha"])
        conf_full_overlap = pipeline._compute_merge_confidence([item_c, item_d])

        assert conf_full_overlap > conf_no_overlap

    def test_empty_tags(self, pipeline):
        """Items with no tags should still produce a valid confidence."""
        item_a = _make_item("D1", confidence=0.5, tags=[])
        item_b = _make_item("D2", confidence=0.5, tags=[])
        conf = pipeline._compute_merge_confidence([item_a, item_b])
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------

class TestMergeRecency:
    def test_recent_items_score_higher(self, pipeline):
        """Items updated recently should yield higher recency decay."""
        now = datetime.now(timezone.utc)
        recent_time = now.isoformat()
        old_time = (now - timedelta(days=365)).isoformat()

        item_recent = _make_item("R1", confidence=0.5, updated_at=recent_time)
        conf_recent = pipeline._compute_merge_confidence([item_recent])

        item_old = _make_item("R2", confidence=0.5, updated_at=old_time)
        conf_old = pipeline._compute_merge_confidence([item_old])

        assert conf_recent > conf_old

    def test_very_old_item_low_recency(self, pipeline):
        """An item updated years ago should have low recency decay."""
        old_time = (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()
        item = _make_item("R3", confidence=0.5, updated_at=old_time)
        conf = pipeline._compute_merge_confidence([item])
        # recency_decay = exp(-0.01 * 1000) = exp(-10) ≈ 4.5e-5 → near zero
        assert 0.0 <= conf <= 1.0
        # Confidence should be lower than a recent item
        recent = _make_item("R4", confidence=0.5)
        conf_recent = pipeline._compute_merge_confidence([recent])
        assert conf < conf_recent


# ---------------------------------------------------------------------------
# Validation bonus
# ---------------------------------------------------------------------------

class TestMergeValidation:
    def test_verified_beats_unverified(self, pipeline):
        """Verified items should produce higher confidence than unverified."""
        item_v = _make_item("V1", confidence=0.5, validation="verified")
        conf_v = pipeline._compute_merge_confidence([item_v])

        item_u = _make_item("V2", confidence=0.5, validation="unverified")
        conf_u = pipeline._compute_merge_confidence([item_u])

        assert conf_v > conf_u

    def test_retracted_gives_zero_bonus(self, pipeline):
        """Retracted items should get validation_bonus = 0.0."""
        item_r = _make_item("V3", confidence=0.5, validation="retracted")
        conf_r = pipeline._compute_merge_confidence([item_r])

        item_u = _make_item("V4", confidence=0.5, validation="unverified")
        conf_u = pipeline._compute_merge_confidence([item_u])

        assert conf_r < conf_u

    def test_mixed_validation_cluster(self, pipeline):
        """A cluster with one verified and one retracted item."""
        item_v = _make_item("V5", confidence=0.5, tags=["x"], validation="verified")
        item_r = _make_item("V6", confidence=0.5, tags=["x"], validation="retracted")
        conf = pipeline._compute_merge_confidence([item_v, item_r])
        # retracted in cluster => validation_bonus = 0.0
        assert 0.0 <= conf <= 1.0

    def test_verified_in_cluster_gives_bonus(self, pipeline):
        """A cluster with at least one verified gets validation_bonus=1.0 if no retracted."""
        item_v = _make_item("V7", confidence=0.5, tags=["x"], validation="verified")
        item_u = _make_item("V8", confidence=0.5, tags=["x"], validation="unverified")
        conf = pipeline._compute_merge_confidence([item_v, item_u])
        # No retracted => validation_bonus=1.0
        assert conf > 0.4


# ---------------------------------------------------------------------------
# Range invariant
# ---------------------------------------------------------------------------

class TestMergeConfidenceRange:
    def test_always_in_unit_interval(self, pipeline):
        """Merge confidence must always be in [0.0, 1.0]."""
        configs = [
            (0.0, "retracted", []),
            (1.0, "verified", ["a", "b"]),
            (0.5, "unverified", ["x"]),
            (0.0, "unverified", []),
            (1.0, "retracted", ["a", "b", "c"]),
        ]
        for conf, val, tags in configs:
            item = _make_item(f"RANGE-{conf}-{val}", confidence=conf, tags=tags, validation=val)
            merged_conf = pipeline._compute_merge_confidence([item])
            assert 0.0 <= merged_conf <= 1.0, (
                f"Out of range: conf={conf}, val={val}, tags={tags} -> {merged_conf}"
            )

    def test_all_zero_confidence(self, pipeline):
        """All items with confidence=0 should still produce a valid result."""
        items = [
            _make_item("Z1", confidence=0.0, tags=["z"]),
            _make_item("Z2", confidence=0.0, tags=["z"]),
        ]
        conf = pipeline._compute_merge_confidence(items)
        assert 0.0 <= conf <= 1.0
        # max_conf=0 * 0.3 = 0, but overlap + recency + validation contribute
        assert conf > 0.0  # not all-zero because recency and validation contribute

    def test_large_cluster(self, pipeline):
        """A large cluster should not produce out-of-range confidence."""
        items = [
            _make_item(f"LC-{i}", confidence=0.5 + 0.01*i, tags=["common"])
            for i in range(20)
        ]
        conf = pipeline._compute_merge_confidence(items)
        assert 0.0 <= conf <= 1.0
