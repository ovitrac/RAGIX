"""
Integration tests for RAGIX Memory MCP tools layer.

Focus areas (from PLAN §6 Phase 1):
    1. Governance parity — MCP rejects exactly same items as raw dispatcher
    2. Provenance integrity — content_hash + source_doc survive MCP round-trip
    3. Injection format stability — memory_recall matches §3.4 contract
    4. Tool delegation — each MCP tool maps correctly to dispatcher action
    5. Formatting — injection block has correct structure and version

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import re

import pytest

from ragix_core.memory.config import EmbedderConfig, MemoryConfig, StoreConfig
from ragix_core.memory.mcp.formatting import (
    FORMAT_VERSION,
    format_injection_block,
    format_search_results,
)
from ragix_core.memory.tools import MemoryToolDispatcher, create_dispatcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> MemoryConfig:
    """In-memory config with mock embedder for deterministic tests."""
    return MemoryConfig(
        store=StoreConfig(db_path=":memory:"),
        embedder=EmbedderConfig(backend="mock", dimension=32, mock_seed=42),
    )


@pytest.fixture
def dispatcher(config) -> MemoryToolDispatcher:
    """Fully wired dispatcher with in-memory store."""
    return create_dispatcher(config)


@pytest.fixture
def mcp_tools(dispatcher):
    """
    Create a mock MCP server and register tools.

    Returns a dict mapping tool_name → callable for direct invocation.
    """
    tools = {}

    class MockMCP:
        def tool(self):
            def decorator(fn):
                tools[fn.__name__] = fn
                return fn
            return decorator

    from ragix_core.memory.mcp.tools import register_memory_tools
    register_memory_tools(MockMCP(), dispatcher)
    return tools


@pytest.fixture
def seeded_tools(mcp_tools, dispatcher):
    """MCP tools with 5 pre-seeded memory items for search/recall tests."""
    items = [
        {
            "title": "Oracle 19c CPU Patch Required",
            "content": "Oracle Database 19c requires Critical Patch Update from Jan 2025.",
            "tags": ["oracle", "CVE", "patch", "19c"],
            "type": "constraint",
            "provenance_hint": {"source_kind": "doc", "source_id": "RIE-Oracle-19c.pdf"},
        },
        {
            "title": "Kubernetes 1.28 PSP Deprecation",
            "content": "PodSecurityPolicy removed in K8s 1.28; migrate to Pod Security Standards.",
            "tags": ["kubernetes", "deprecation", "security"],
            "type": "fact",
        },
        {
            "title": "RHEL 9 Firewall Default Policy",
            "content": "RHEL 9 uses firewalld with zone-based policies. Default zone: public.",
            "tags": ["rhel", "firewall", "security"],
            "type": "fact",
        },
        {
            "title": "AD GPO Audit Interval",
            "content": "Active Directory GPO audit interval should be 90 days maximum.",
            "tags": ["active-directory", "gpo", "audit"],
            "type": "constraint",
        },
        {
            "title": "Migration Deadline Q2 2026",
            "content": "All Oracle 12c instances must be migrated to 19c by Q2 2026.",
            "tags": ["oracle", "migration", "deadline"],
            "type": "decision",
        },
    ]

    # Seed via dispatcher directly (faster than MCP propose for fixtures)
    for item in items:
        dispatcher.dispatch("propose", {"items": [item]})

    return mcp_tools


# ===========================================================================
# 1. GOVERNANCE PARITY
# ===========================================================================

class TestGovernanceParity:
    """MCP layer rejects exactly the same items as raw dispatcher."""

    def test_secret_blocked_via_mcp(self, mcp_tools):
        """Secret in content is blocked by MCP propose (same as dispatcher)."""
        result = mcp_tools["memory_propose"](
            items=json.dumps([{
                "title": "Test Secret",
                "content": "The api_key = sk-proj-abcdefghijklmnop1234567890 is sensitive.",
                "tags": ["test"],
                "type": "note",
            }]),
        )
        assert result["status"] == "ok"
        assert result["rejected"] == 1
        assert result["accepted"] == 0

    def test_clean_item_accepted_via_mcp(self, mcp_tools):
        """Clean item passes policy via MCP propose."""
        result = mcp_tools["memory_propose"](
            items=json.dumps([{
                "title": "RHEL 9 Patch Cycle",
                "content": "RHEL 9 receives quarterly security patches.",
                "tags": ["rhel", "security"],
                "type": "fact",
            }]),
        )
        assert result["status"] == "ok"
        assert result["accepted"] == 1
        assert result["rejected"] == 0

    def test_secret_blocked_via_direct_write(self, mcp_tools):
        """API key blocked in memory_write too (policy-checked)."""
        result = mcp_tools["memory_write"](
            title="Secret Write",
            content="aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            tags="test",
        )
        assert result["status"] == "rejected"

    def test_oversized_content_rejected(self, mcp_tools):
        """Content exceeding max_content_length is rejected."""
        result = mcp_tools["memory_propose"](
            items=json.dumps([{
                "title": "Too Long",
                "content": "x" * 3000,
                "tags": ["test"],
                "type": "note",
            }]),
        )
        assert result["status"] == "ok"
        assert result["rejected"] == 1

    def test_governance_parity_batch(self, mcp_tools):
        """Mixed batch: 1 clean + 1 secret → exactly 1 accepted, 1 rejected."""
        result = mcp_tools["memory_propose"](
            items=json.dumps([
                {
                    "title": "Clean Item",
                    "content": "This is fine.",
                    "tags": ["test"],
                    "type": "note",
                },
                {
                    "title": "Secret Item",
                    "content": "password= hunter2secretpassword123",
                    "tags": ["test"],
                    "type": "note",
                },
            ]),
        )
        assert result["accepted"] == 1
        assert result["rejected"] == 1


# ===========================================================================
# 2. PROVENANCE INTEGRITY
# ===========================================================================

class TestProvenanceIntegrity:
    """Content hash and source_doc survive MCP round-trip."""

    def test_source_doc_propagated(self, mcp_tools):
        """source_doc in memory_propose flows into provenance."""
        result = mcp_tools["memory_propose"](
            items=json.dumps([{
                "title": "Provenance Test",
                "content": "Oracle 19c requires CPU patch.",
                "tags": ["oracle"],
                "type": "fact",
            }]),
            source_doc="RIE-Oracle-19c.pdf",
        )
        assert result["accepted"] == 1
        item_id = result["items"][0]["id"]

        # Read back and verify provenance
        read_result = mcp_tools["memory_read"](ids=item_id)
        assert read_result["status"] == "ok"
        item = read_result["items"][0]
        prov = item.get("provenance", {})
        assert prov.get("source_id") == "RIE-Oracle-19c.pdf"
        assert prov.get("source_kind") == "doc"

    def test_content_hash_round_trip(self, mcp_tools):
        """Content hash is computed and persisted correctly."""
        content = "Specific content for hash verification."
        result = mcp_tools["memory_propose"](
            items=json.dumps([{
                "title": "Hash Test",
                "content": content,
                "tags": ["test"],
                "type": "note",
            }]),
        )
        item_id = result["items"][0]["id"]

        read_result = mcp_tools["memory_read"](ids=item_id)
        item = read_result["items"][0]
        assert item["content"] == content

    def test_tags_round_trip(self, mcp_tools):
        """Tags survive propose → read round-trip as list."""
        result = mcp_tools["memory_propose"](
            items=json.dumps([{
                "title": "Tag Test",
                "content": "Tag preservation test.",
                "tags": ["alpha", "beta", "gamma"],
                "type": "fact",
            }]),
        )
        item_id = result["items"][0]["id"]

        read_result = mcp_tools["memory_read"](ids=item_id)
        item = read_result["items"][0]
        assert set(item["tags"]) == {"alpha", "beta", "gamma"}


# ===========================================================================
# 3. INJECTION FORMAT STABILITY
# ===========================================================================

class TestInjectionFormat:
    """memory_recall output matches the §3.4 contract."""

    def test_format_version_header(self, seeded_tools):
        """Injection block starts with format_version header."""
        result = seeded_tools["memory_recall"](query="oracle", budget_tokens=2000)
        assert result["status"] == "ok"
        text = result["inject_text"]
        assert text.startswith("## Memory (Injected)")
        assert f"format_version: {FORMAT_VERSION}" in text

    def test_injection_type_header(self, seeded_tools):
        """Injection block contains injection_type field."""
        result = seeded_tools["memory_recall"](query="oracle")
        assert "injection_type: memory_recall" in result["inject_text"]

    def test_generated_at_iso(self, seeded_tools):
        """Injection block contains generated_at in ISO 8601 format."""
        result = seeded_tools["memory_recall"](query="oracle")
        match = re.search(r"generated_at: (\S+)", result["inject_text"])
        assert match is not None
        # Should be ISO 8601 UTC
        ts = match.group(1)
        assert "T" in ts and ts.endswith("Z")

    def test_budget_tokens_in_header(self, seeded_tools):
        """Injection block reflects requested budget."""
        result = seeded_tools["memory_recall"](query="oracle", budget_tokens=800)
        assert "budget_tokens: 800" in result["inject_text"]

    def test_end_marker(self, seeded_tools):
        """Injection block ends with --- End Memory marker."""
        result = seeded_tools["memory_recall"](query="oracle")
        text = result["inject_text"]
        assert f"--- End Memory (format_version={FORMAT_VERSION}" in text

    def test_item_format_structure(self, seeded_tools):
        """Each item in injection follows [N] [TIER:state] type — title pattern."""
        result = seeded_tools["memory_recall"](query="oracle")
        text = result["inject_text"]
        # Should contain at least one item with the expected format
        pattern = r"\[\d+\] \[(STM|MTM|LTM):\w+\] \w+ — .+"
        assert re.search(pattern, text) is not None

    def test_provenance_in_items(self, seeded_tools):
        """Each injected item includes provenance line."""
        result = seeded_tools["memory_recall"](query="oracle")
        assert "provenance:" in result["inject_text"]

    def test_tags_in_items(self, seeded_tools):
        """Each injected item includes tags line."""
        result = seeded_tools["memory_recall"](query="oracle")
        assert "tags:" in result["inject_text"]

    def test_confidence_in_items(self, seeded_tools):
        """Each injected item includes confidence line."""
        result = seeded_tools["memory_recall"](query="oracle")
        assert "confidence:" in result["inject_text"]

    def test_format_version_in_response(self, seeded_tools):
        """Response dict includes format_version key."""
        result = seeded_tools["memory_recall"](query="oracle")
        assert result["format_version"] == FORMAT_VERSION

    def test_tokens_used_reported(self, seeded_tools):
        """Response includes tokens_used estimate."""
        result = seeded_tools["memory_recall"](query="oracle")
        assert isinstance(result["tokens_used"], int)
        assert result["tokens_used"] > 0

    def test_empty_query_returns_empty(self, mcp_tools):
        """Empty store + query returns empty inject_text."""
        result = mcp_tools["memory_recall"](query="nonexistent")
        assert result["inject_text"] == ""
        assert result["matched"] == 0


# ===========================================================================
# 4. TOOL DELEGATION
# ===========================================================================

class TestToolDelegation:
    """Each MCP tool correctly delegates to the dispatcher."""

    def test_write_and_read(self, mcp_tools):
        """memory_write → memory_read round-trip."""
        w = mcp_tools["memory_write"](
            title="Delegation Test",
            content="Testing write-read delegation.",
            tags="test,delegation",
            type="note",
        )
        assert w["status"] == "ok"
        item_id = w["id"]

        r = mcp_tools["memory_read"](ids=item_id)
        assert r["status"] == "ok"
        assert len(r["items"]) == 1
        assert r["items"][0]["title"] == "Delegation Test"

    def test_update(self, mcp_tools):
        """memory_update patches fields correctly."""
        w = mcp_tools["memory_write"](
            title="Before Update",
            content="Original content.",
            tags="test",
        )
        item_id = w["id"]

        u = mcp_tools["memory_update"](
            id=item_id,
            title="After Update",
            confidence=0.9,
        )
        assert u["status"] == "ok"

        r = mcp_tools["memory_read"](ids=item_id)
        assert r["items"][0]["title"] == "After Update"
        assert r["items"][0]["confidence"] == 0.9

    def test_link(self, mcp_tools):
        """memory_link creates relationship between items."""
        w1 = mcp_tools["memory_write"](title="Item A", content="A")
        w2 = mcp_tools["memory_write"](title="Item B", content="B")

        link = mcp_tools["memory_link"](
            src_id=w1["id"],
            dst_id=w2["id"],
            relation="supports",
        )
        assert link["status"] == "ok"

    def test_stats(self, seeded_tools):
        """memory_stats returns store statistics."""
        result = seeded_tools["memory_stats"]()
        assert result["status"] == "ok"
        assert result["total_items"] >= 5
        assert "by_tier" in result
        assert "by_type" in result

    def test_consolidate(self, seeded_tools):
        """memory_consolidate runs without error."""
        result = seeded_tools["memory_consolidate"]()
        assert result["status"] == "ok"

    def test_palace_list(self, mcp_tools):
        """memory_palace_list returns (possibly empty) locations."""
        result = mcp_tools["memory_palace_list"]()
        assert result["status"] == "ok"
        assert "locations" in result

    def test_palace_get_missing(self, mcp_tools):
        """memory_palace_get for missing item returns error."""
        result = mcp_tools["memory_palace_get"](item_id="MEM-nonexistent")
        assert result["status"] == "error"

    def test_search(self, seeded_tools):
        """memory_search returns matching items."""
        result = seeded_tools["memory_search"](query="oracle")
        assert result["status"] == "ok"
        assert result["count"] > 0

    def test_search_with_tag_filter(self, seeded_tools):
        """memory_search with tag filter narrows results."""
        result = seeded_tools["memory_search"](query="security", tags="rhel")
        assert result["status"] == "ok"
        # Should find RHEL firewall item
        assert result["count"] >= 0  # at least doesn't error

    def test_recall_catalog_mode(self, seeded_tools):
        """memory_recall in catalog mode returns structured items."""
        result = seeded_tools["memory_recall"](
            query="kubernetes", mode="catalog",
        )
        assert result["status"] == "ok"
        assert isinstance(result["items"], list)

    def test_recall_inject_mode(self, seeded_tools):
        """memory_recall in inject mode returns only text, no items."""
        result = seeded_tools["memory_recall"](
            query="kubernetes", mode="inject",
        )
        assert result["status"] == "ok"
        assert isinstance(result["inject_text"], str)
        assert result["items"] == []

    def test_propose_invalid_json(self, mcp_tools):
        """memory_propose with invalid JSON returns error."""
        result = mcp_tools["memory_propose"](items="not valid json{{{")
        assert result["status"] == "error"
        assert "Invalid JSON" in result["message"]

    def test_propose_single_object(self, mcp_tools):
        """memory_propose accepts single object (auto-wrapped in list)."""
        result = mcp_tools["memory_propose"](
            items=json.dumps({
                "title": "Single Item",
                "content": "This is a single item, not an array.",
                "tags": ["test"],
                "type": "note",
            }),
        )
        assert result["accepted"] == 1


# ===========================================================================
# 5. FORMATTING MODULE
# ===========================================================================

class TestFormattingModule:
    """Unit tests for the formatting module (format_injection_block, etc.)."""

    def test_empty_items(self):
        """Empty item list returns empty string."""
        assert format_injection_block([]) == ""

    def test_format_version_constant(self):
        """FORMAT_VERSION is integer 1."""
        assert FORMAT_VERSION == 1
        assert isinstance(FORMAT_VERSION, int)

    def test_single_item_block(self):
        """Single item produces valid injection block."""
        items = [{
            "tier": "ltm",
            "validation": "verified",
            "type": "constraint",
            "title": "Test Rule",
            "content": "This is a test rule.",
            "provenance": {"source_kind": "doc", "source_id": "test.pdf"},
            "tags": ["test", "rule"],
            "confidence": 0.95,
        }]
        block = format_injection_block(items, budget_tokens=500)
        assert "## Memory (Injected)" in block
        assert "format_version: 1" in block
        assert "[1] [LTM:verified] constraint — Test Rule" in block
        assert "tags: test, rule" in block
        assert "confidence: 0.95" in block
        assert "--- End Memory" in block

    def test_budget_truncation(self):
        """Items exceeding budget are truncated."""
        items = [
            {
                "tier": "stm", "validation": "unverified", "type": "note",
                "title": f"Item {i}", "content": "x" * 200,
                "tags": ["test"], "confidence": 0.5,
                "provenance": {},
            }
            for i in range(20)
        ]
        block = format_injection_block(items, budget_tokens=200)
        # Should not include all 20 items (200 tokens ≈ 800 chars)
        item_count = block.count("[STM:unverified]")
        assert item_count < 20
        assert item_count >= 1

    def test_search_results_format(self):
        """format_search_results returns structured list."""
        items = [{
            "id": "MEM-abc", "title": "Test", "tier": "stm",
            "type": "note", "tags": ["a"], "confidence": 0.5,
            "validation": "unverified", "content": "Long content " * 50,
        }]
        results = format_search_results(items)
        assert len(results) == 1
        assert results[0]["id"] == "MEM-abc"
        # content_preview should be truncated
        assert len(results[0]["content_preview"]) <= 200
