"""
Tests for summary_drift.py — cross-corpus drift detection.
"""

import pytest

from ragix_core.memory.types import (
    MemoryItem,
    MemoryProvenance,
    _generate_id,
    _now_iso,
)
from ragix_kernels.summary.kernels.summary_drift import _compute_drift


def _make_item(
    title: str,
    content: str = "test content",
    rule_id: str = None,
    corpus_id: str = None,
) -> MemoryItem:
    return MemoryItem(
        id=_generate_id(),
        title=title,
        content=content,
        type="constraint",
        tier="stm",
        scope="test",
        tags=["test"],
        rule_id=rule_id,
        corpus_id=corpus_id,
        provenance=MemoryProvenance(source_kind="document"),
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )


class TestComputeDrift:
    """Test the drift computation algorithm."""

    def test_identical_corpora(self):
        """Same items → all UNCHANGED, 0% drift."""
        items = [_make_item("Rule A", "content A", rule_id="R-001")]
        drift = _compute_drift(items, items)
        assert drift["counts"]["unchanged"] == 1
        assert drift["counts"]["added"] == 0
        assert drift["counts"]["removed"] == 0
        assert drift["counts"]["modified"] == 0
        assert drift["drift_pct"] == 0.0

    def test_all_added(self):
        """Items only in B → all ADDED."""
        items_a = []
        items_b = [_make_item("New rule", "new content")]
        drift = _compute_drift(items_a, items_b)
        assert drift["counts"]["added"] == 1
        assert drift["counts"]["removed"] == 0
        assert drift["drift_pct"] == 100.0

    def test_all_removed(self):
        """Items only in A → all REMOVED."""
        items_a = [_make_item("Old rule", "old content")]
        items_b = []
        drift = _compute_drift(items_a, items_b)
        assert drift["counts"]["removed"] == 1
        assert drift["counts"]["added"] == 0
        assert drift["drift_pct"] == 100.0

    def test_modified_by_rule_id(self):
        """Same rule_id, different content → MODIFIED."""
        item_a = _make_item("Rule X v1", "version 1 content", rule_id="R-100")
        item_b = _make_item("Rule X v2", "version 2 content", rule_id="R-100")
        drift = _compute_drift([item_a], [item_b])
        assert drift["counts"]["modified"] == 1
        assert drift["counts"]["unchanged"] == 0

    def test_unchanged_by_content_hash(self):
        """No rule_id, same content → UNCHANGED via content_hash."""
        content = "identical content across both corpora"
        item_a = _make_item("Rule Y", content)
        item_b = _make_item("Rule Y", content)
        drift = _compute_drift([item_a], [item_b])
        assert drift["counts"]["unchanged"] == 1

    def test_mixed_drift(self):
        """Complex scenario: 1 unchanged, 1 modified, 1 added, 1 removed."""
        shared_content = "shared content"
        a1 = _make_item("Unchanged", shared_content, rule_id="R-001")
        a2 = _make_item("Will be modified", "old", rule_id="R-002")
        a3 = _make_item("Will be removed", "removed content", rule_id="R-003")

        b1 = _make_item("Unchanged", shared_content, rule_id="R-001")
        b2 = _make_item("Was modified", "new", rule_id="R-002")
        b4 = _make_item("Newly added", "new content")

        drift = _compute_drift([a1, a2, a3], [b1, b2, b4])
        assert drift["counts"]["unchanged"] == 1
        assert drift["counts"]["modified"] == 1
        assert drift["counts"]["added"] == 1
        assert drift["counts"]["removed"] == 1

    def test_per_domain_summary(self):
        """Drift report includes per-domain breakdown."""
        item_a = _make_item("Rule", "content A", rule_id="R-001")
        item_b = _make_item("Rule New", "content B")

        def _domain(item):
            return "rhel"

        drift = _compute_drift([item_a], [item_b], extract_domain_fn=_domain)
        assert "rhel" in drift["per_domain"]
        domain = drift["per_domain"]["rhel"]
        assert domain["removed"] == 1
        assert domain["added"] == 1

    def test_returns_expected_keys(self):
        drift = _compute_drift([], [])
        assert "added" in drift
        assert "removed" in drift
        assert "modified" in drift
        assert "unchanged" in drift
        assert "counts" in drift
        assert "drift_pct" in drift
        assert "per_domain" in drift

    def test_empty_corpora(self):
        drift = _compute_drift([], [])
        assert drift["drift_pct"] == 0.0
        assert drift["counts"]["total_a"] == 0
        assert drift["counts"]["total_b"] == 0
