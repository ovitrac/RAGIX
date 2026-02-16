"""
Tests for graph_store.py — GraphStore, _is_valid_entity, BFS, compact, export.
"""

import json
import os
import tempfile

import pytest

from ragix_core.memory.graph_store import (
    GraphStore,
    _is_valid_entity,
    _PRODUCT_VOCABULARY,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite DB path."""
    return str(tmp_path / "test_graph.db")


@pytest.fixture
def graph(tmp_db):
    """Create a fresh GraphStore."""
    g = GraphStore(tmp_db)
    yield g
    g.close()


# ── _is_valid_entity ─────────────────────────────────────────────────────

class TestIsValidEntity:
    """Test the controlled vocabulary filter."""

    def test_cve_valid(self):
        assert _is_valid_entity("CVE-2024-1234") is True

    def test_cve_case_insensitive(self):
        assert _is_valid_entity("cve-2024-5678") is True

    def test_version_3component(self):
        assert _is_valid_entity("9.1.2") is True

    def test_version_4component(self):
        assert _is_valid_entity("1.2.3.4") is True

    def test_version_2component(self):
        # 2-component matches the version regex pattern ^\d+(?:\.\d+){1,3}
        assert _is_valid_entity("8.4") is True

    def test_product_rhel(self):
        assert _is_valid_entity("rhel") is True

    def test_product_case_insensitive(self):
        # Vocabulary comparison is .lower(), so uppercase matches too
        assert _is_valid_entity("RHEL") is True
        assert _is_valid_entity("postgresql") is True

    def test_product_multiword(self):
        assert _is_valid_entity("active directory") is True
        assert _is_valid_entity("sql server") is True

    def test_compliance_must(self):
        assert _is_valid_entity("MUST") is True

    def test_compliance_should(self):
        assert _is_valid_entity("SHOULD") is True

    def test_config_path(self):
        assert _is_valid_entity("/etc/ssh/sshd_config") is True
        assert _is_valid_entity("/var/log/messages") is True

    def test_port_number(self):
        assert _is_valid_entity("22/tcp") is True
        assert _is_valid_entity("443/tcp") is True

    def test_short_rejected(self):
        assert _is_valid_entity("a") is False

    def test_long_rejected(self):
        assert _is_valid_entity("x" * 61) is False

    def test_freetext_rejected(self):
        assert _is_valid_entity("hello world") is False
        assert _is_valid_entity("configure the server") is False

    def test_review_rule_id(self):
        assert _is_valid_entity("RVW-0001") is True

    def test_memory_item_id(self):
        assert _is_valid_entity("MEM-abcdef01") is True

    def test_corpus_driven_products(self):
        """V2.2 additions should be in vocabulary."""
        for prod in ("crowdstrike", "sql server", "proftpd", "openssh", "angular"):
            assert prod in _PRODUCT_VOCABULARY, f"{prod} missing from vocabulary"
            assert _is_valid_entity(prod) is True


# ── GraphStore CRUD ──────────────────────────────────────────────────────

class TestGraphStoreCRUD:
    """Test basic node/edge operations."""

    def test_add_and_get_node(self, graph):
        graph.add_node("item:MEM-001", "item", label="test item", item_id="MEM-001")
        node = graph.get_node("item:MEM-001")
        assert node is not None
        assert node["kind"] == "item"
        assert node["label"] == "test item"
        assert node["item_id"] == "MEM-001"

    def test_get_nonexistent_node(self, graph):
        assert graph.get_node("item:NOPE") is None

    def test_add_edge(self, graph):
        graph.add_node("item:A", "item")
        graph.add_node("item:B", "item")
        graph.add_edge("item:A", "item:B", "similar", weight=0.95)
        stats = graph.stats()
        assert stats["total_edges"] == 1
        assert stats["edges_by_kind"].get("similar", 0) == 1

    def test_stats_empty(self, graph):
        stats = graph.stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0


# ── BFS Neighbors ────────────────────────────────────────────────────────

class TestBFSNeighbors:
    """Test BFS traversal."""

    def _build_chain(self, graph, n=5):
        """Build a linear chain: A -> B -> C -> D -> E."""
        ids = [f"item:{chr(65 + i)}" for i in range(n)]
        for nid in ids:
            graph.add_node(nid, "item")
        for i in range(n - 1):
            graph.add_edge(ids[i], ids[i + 1], "similar")
        return ids

    def test_depth_1(self, graph):
        ids = self._build_chain(graph)
        nbrs = graph.neighbors("item:A", depth=1)
        assert "item:B" in nbrs
        assert "item:C" not in nbrs

    def test_depth_2(self, graph):
        ids = self._build_chain(graph)
        nbrs = graph.neighbors("item:A", depth=2)
        assert "item:B" in nbrs
        assert "item:C" in nbrs
        assert "item:D" not in nbrs

    def test_excludes_start(self, graph):
        ids = self._build_chain(graph)
        nbrs = graph.neighbors("item:A", depth=3)
        assert "item:A" not in nbrs

    def test_edge_kind_filter(self, graph):
        graph.add_node("item:X", "item")
        graph.add_node("entity:rhel", "entity")
        graph.add_node("item:Y", "item")
        graph.add_edge("item:X", "entity:rhel", "mentions")
        graph.add_edge("item:X", "item:Y", "similar")
        # Only follow 'mentions' edges
        nbrs = graph.neighbors("item:X", depth=1, edge_kinds=["mentions"])
        assert "entity:rhel" in nbrs
        assert "item:Y" not in nbrs

    def test_max_size_cap(self, graph):
        # Build a 2-level tree to test max_size across multiple BFS iterations
        # Hub → 10 level-1 nodes → each with 5 level-2 nodes = 60 total
        graph.add_node("hub", "item")
        for i in range(10):
            l1 = f"l1:{i}"
            graph.add_node(l1, "item")
            graph.add_edge("hub", l1, "similar")
            for j in range(5):
                l2 = f"l2:{i}:{j}"
                graph.add_node(l2, "item")
                graph.add_edge(l1, l2, "similar")

        # With max_size=20, BFS should stop before visiting all 60 nodes
        nbrs = graph.neighbors("hub", depth=2, max_size=20)
        assert len(nbrs) <= 20


class TestNeighborhoodItems:
    """Test neighborhood_items — returns only item IDs."""

    def test_returns_item_ids_only(self, graph):
        graph.add_node("item:MEM-001", "item", item_id="MEM-001")
        graph.add_node("entity:rhel", "entity")
        graph.add_node("item:MEM-002", "item", item_id="MEM-002")
        graph.add_edge("item:MEM-001", "entity:rhel", "mentions")
        graph.add_edge("item:MEM-002", "entity:rhel", "mentions")

        items = graph.neighborhood_items("MEM-001", depth=2)
        assert "MEM-002" in items
        assert "rhel" not in items  # entity, not item

    def test_excludes_self(self, graph):
        graph.add_node("item:MEM-001", "item", item_id="MEM-001")
        graph.add_node("item:MEM-002", "item", item_id="MEM-002")
        graph.add_edge("item:MEM-001", "item:MEM-002", "similar")

        items = graph.neighborhood_items("MEM-001", depth=1)
        assert "MEM-001" not in items
        assert "MEM-002" in items


# ── Clear ────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_removes_all(self, graph):
        graph.add_node("item:A", "item")
        graph.add_edge("item:A", "item:A", "self")
        graph.clear()
        stats = graph.stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0


# ── Edge Cap ─────────────────────────────────────────────────────────────

class TestEnforceEdgeCap:
    def test_cap_removes_excess(self, graph):
        graph.add_node("hub", "item")
        for i in range(10):
            nid = f"spoke:{i}"
            graph.add_node(nid, "item")
            graph.add_edge("hub", nid, "similar", weight=float(i) / 10)

        removed = graph.enforce_edge_cap(max_edges_per_node=5)
        assert removed > 0
        # After cap, hub should have at most 5 edges
        stats = graph.stats()
        assert stats["total_edges"] <= 5

    def test_cap_noop_under_limit(self, graph):
        graph.add_node("A", "item")
        graph.add_node("B", "item")
        graph.add_edge("A", "B", "similar")
        removed = graph.enforce_edge_cap(max_edges_per_node=100)
        assert removed == 0


# ── Export (secrecy-aware) ───────────────────────────────────────────────

class TestExportGraph:
    def test_s3_no_redaction(self, graph):
        graph.add_node("item:MEM-001", "item", label="RHEL 8.4 config")
        graph.add_node("entity:rhel", "entity", label="rhel")
        graph.add_edge("item:MEM-001", "entity:rhel", "mentions")

        export = graph.export_graph(tier="S3")
        assert export["tier"] == "S3"
        assert len(export["nodes"]) == 2
        assert len(export["edges"]) == 1
        # Labels unchanged
        labels = {n["label"] for n in export["nodes"]}
        assert "RHEL 8.4 config" in labels
        assert "rhel" in labels

    def test_export_has_stats(self, graph):
        graph.add_node("item:X", "item")
        export = graph.export_graph(tier="S3")
        assert "stats" in export
        assert export["stats"]["total_nodes"] == 1
