"""
Tests for KOAS Summary MCP tool registration and basic interfaces.

Validates:
- All 7 tools are registered (check function names)
- summary_query returns expected result structure
- summary_viz accepts --views parameter
- Tool signatures match expected arguments

Does NOT require a running Ollama or real workspace — tests use
a minimal mock MCP server to verify registration and argument parsing.
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

from ragix_core.memory.store import MemoryStore
from ragix_core.memory.types import MemoryItem, MemoryProvenance


# ---------------------------------------------------------------------------
# Minimal MCP server mock
# ---------------------------------------------------------------------------

class MockMCPServer:
    """Minimal mock that captures registered tools."""

    def __init__(self):
        self._tools = {}

    def tool(self):
        """Decorator that captures the async function."""
        def decorator(func):
            self._tools[func.__name__] = func
            return func
        return decorator

    @property
    def registered_names(self):
        return set(self._tools.keys())

    def get_tool(self, name: str):
        return self._tools.get(name)


@pytest.fixture
def mcp_server():
    return MockMCPServer()


@pytest.fixture
def registered_server(mcp_server):
    """MCP server with all summary tools registered."""
    from ragix_kernels.summary.mcp.tools import register_summary_tools
    register_summary_tools(mcp_server)
    return mcp_server


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_all_seven_tools_registered(self, registered_server):
        """All 7 MCP tools should be registered."""
        expected = {
            "summary_ingest",
            "summary_run",
            "summary_status",
            "summary_query",
            "summary_drift",
            "summary_viz",
            "summary_summarize",
        }
        assert registered_server.registered_names == expected

    def test_tools_are_callable(self, registered_server):
        """Each registered tool should be a callable."""
        for name in registered_server.registered_names:
            tool = registered_server.get_tool(name)
            assert callable(tool), f"Tool '{name}' is not callable"

    def test_tools_are_async(self, registered_server):
        """Each registered tool should be an async function."""
        import asyncio
        for name in registered_server.registered_names:
            tool = registered_server.get_tool(name)
            assert asyncio.iscoroutinefunction(tool), (
                f"Tool '{name}' is not async"
            )

    def test_registration_is_idempotent(self, mcp_server):
        """Registering twice should not duplicate tools."""
        from ragix_kernels.summary.mcp.tools import register_summary_tools
        register_summary_tools(mcp_server)
        first_names = set(mcp_server.registered_names)
        # Re-register (overwrites same keys)
        register_summary_tools(mcp_server)
        assert mcp_server.registered_names == first_names


# ---------------------------------------------------------------------------
# summary_query tool
# ---------------------------------------------------------------------------

class TestSummaryQueryTool:
    @pytest.mark.asyncio
    async def test_query_missing_db_returns_error(self, registered_server, tmp_path):
        """Query against a workspace without memory.db returns error."""
        tool = registered_server.get_tool("summary_query")
        result_str = await tool(
            workspace=str(tmp_path),
            query="test",
        )
        result = json.loads(result_str)
        assert result["status"] == "error"
        assert "memory.db" in result["message"].lower() or "no memory" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_query_returns_valid_structure(self, registered_server, tmp_path):
        """Query with a valid workspace returns expected JSON structure."""
        # Create a minimal workspace with memory.db
        store = MemoryStore(str(tmp_path / "memory.db"))
        store.write_item(MemoryItem(
            id="QRY-001", tier="stm", type="fact",
            title="Test database item",
            content="This item is about database testing.",
            tags=["test", "database"],
            provenance=MemoryProvenance(source_kind="chat", source_id="t1"),
        ))
        store.close()

        tool = registered_server.get_tool("summary_query")
        result_str = await tool(
            workspace=str(tmp_path),
            query="database",
        )
        result = json.loads(result_str)
        assert result["status"] == "ok"
        assert "count" in result
        assert "items" in result
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_query_with_tier_filter(self, registered_server, tmp_path):
        """Query with tier filter passes the parameter through."""
        store = MemoryStore(str(tmp_path / "memory.db"))
        store.write_item(MemoryItem(
            id="QRY-002", tier="stm", type="fact",
            title="STM item", content="Short-term memory.",
            tags=["test"],
            provenance=MemoryProvenance(source_kind="chat", source_id="t1"),
        ))
        store.write_item(MemoryItem(
            id="QRY-003", tier="mtm", type="fact",
            title="MTM item", content="Medium-term memory.",
            tags=["test"],
            provenance=MemoryProvenance(source_kind="doc", source_id="d1"),
        ))
        store.close()

        tool = registered_server.get_tool("summary_query")
        result_str = await tool(
            workspace=str(tmp_path),
            query="memory",
            tier="stm",
        )
        result = json.loads(result_str)
        assert result["status"] == "ok"
        # Should only contain STM items
        for item in result["items"]:
            assert item["tier"] == "stm"


# ---------------------------------------------------------------------------
# summary_viz tool
# ---------------------------------------------------------------------------

class TestSummaryVizTool:
    @pytest.mark.asyncio
    async def test_viz_missing_db_returns_error(self, registered_server, tmp_path):
        """Viz against a workspace without memory.db returns error."""
        tool = registered_server.get_tool("summary_viz")
        result_str = await tool(
            workspace=str(tmp_path),
        )
        result = json.loads(result_str)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_viz_accepts_views_parameter(self, registered_server):
        """The summary_viz tool should accept a 'views' parameter."""
        tool = registered_server.get_tool("summary_viz")
        import inspect
        sig = inspect.signature(tool)
        param_names = set(sig.parameters.keys())
        assert "views" in param_names
        assert "secrecy" in param_names
        assert "workspace" in param_names


# ---------------------------------------------------------------------------
# summary_status tool
# ---------------------------------------------------------------------------

class TestSummaryStatusTool:
    @pytest.mark.asyncio
    async def test_status_empty_workspace(self, registered_server, tmp_path):
        """Status of empty workspace returns valid JSON."""
        tool = registered_server.get_tool("summary_status")
        result_str = await tool(workspace=str(tmp_path))
        result = json.loads(result_str)
        assert "workspace" in result
        # No memory.db => memory should be None
        assert result.get("memory") is None

    @pytest.mark.asyncio
    async def test_status_with_db(self, registered_server, tmp_path):
        """Status with a valid memory.db returns stats."""
        store = MemoryStore(str(tmp_path / "memory.db"))
        store.write_item(MemoryItem(
            id="ST-001", tier="stm", type="note",
            title="Status test", content="Testing status tool.",
            tags=["test"],
            provenance=MemoryProvenance(source_kind="chat", source_id="t1"),
        ))
        store.close()

        tool = registered_server.get_tool("summary_status")
        result_str = await tool(workspace=str(tmp_path))
        result = json.loads(result_str)
        assert result.get("memory") is not None
        assert result["memory"]["total_items"] >= 1
