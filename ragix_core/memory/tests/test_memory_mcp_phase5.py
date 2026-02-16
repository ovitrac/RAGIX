"""
Phase 5 integration tests for RAGIX Memory MCP — Wiring.

Focus areas:
    1. Workspace resolution via MCP tool params (workspace → scope)
    2. Rate limiting blocks tool calls when exhausted
    3. Metrics collector records tool call latency and errors
    4. Management tools: workspace_list, workspace_register, workspace_remove, metrics
    5. Proposal rate limiting (per-turn cap)
    6. Session bridge tools respect rate limiter with session_id

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from ragix_core.memory.config import (
    EmbedderConfig,
    MemoryConfig,
    RateLimitConfig,
    StoreConfig,
)
from ragix_core.memory.mcp.metrics import MetricsCollector
from ragix_core.memory.mcp.rate_limiter import RateLimiter
from ragix_core.memory.mcp.session import SessionManager
from ragix_core.memory.mcp.workspace import WorkspaceRouter
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
def workspace_router(dispatcher) -> WorkspaceRouter:
    """WorkspaceRouter sharing the dispatcher's SQLite connection."""
    return WorkspaceRouter(dispatcher.store._conn)


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Fresh MetricsCollector."""
    return MetricsCollector()


@pytest.fixture
def rate_limiter() -> RateLimiter:
    """Rate limiter with low limits for testing (5 calls/min, 3 proposals/turn)."""
    return RateLimiter(RateLimitConfig(
        enabled=True,
        calls_per_minute=5,
        proposals_per_turn=3,
        burst_multiplier=1.0,
    ))


@pytest.fixture
def session_mgr(dispatcher) -> SessionManager:
    """Session manager sharing the dispatcher's SQLite connection."""
    return SessionManager(dispatcher.store._conn)


class MockMCP:
    """Minimal mock MCP server that collects registered tools."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator

    def prompt(self):
        def decorator(fn):
            return fn
        return decorator


@pytest.fixture
def full_mcp(dispatcher, session_mgr, workspace_router, metrics_collector, rate_limiter):
    """
    Register all tools with workspace, metrics, and rate limiter wired in.

    Returns (tools_dict, workspace_router, metrics_collector, rate_limiter).
    """
    from ragix_core.memory.mcp.tools import register_memory_tools

    mock = MockMCP()
    register_memory_tools(
        mock,
        dispatcher,
        session_mgr=session_mgr,
        workspace_router=workspace_router,
        metrics=metrics_collector,
        rate_limiter=rate_limiter,
    )
    return mock.tools, workspace_router, metrics_collector, rate_limiter


@pytest.fixture
def tools(full_mcp):
    """Just the tools dict for convenience."""
    return full_mcp[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_item(tools, title="Test item", content="Test content", tags=None, scope="project"):
    """Write a single item and return the result."""
    return tools["memory_write"](
        title=title,
        content=content,
        tags=",".join(tags) if tags else "test",
        scope=scope,
    )


# ---------------------------------------------------------------------------
# Test 1: Tool Registration Count
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Verify all expected tools are registered."""

    def test_full_registration_count(self, tools):
        """17 tools: 11 core + 2 session + 3 workspace + 1 metrics."""
        assert len(tools) == 17

    def test_management_tools_present(self, tools):
        """Workspace and metrics management tools exist."""
        assert "memory_workspace_list" in tools
        assert "memory_workspace_register" in tools
        assert "memory_workspace_remove" in tools
        assert "memory_metrics" in tools

    def test_core_tools_still_present(self, tools):
        """All 11 core tools + 2 session bridge remain registered."""
        core = [
            "memory_recall", "memory_search", "memory_propose",
            "memory_write", "memory_read", "memory_update",
            "memory_link", "memory_consolidate", "memory_stats",
            "memory_palace_list", "memory_palace_get",
            "memory_session_inject", "memory_session_store",
        ]
        for name in core:
            assert name in tools, f"Missing tool: {name}"


# ---------------------------------------------------------------------------
# Test 2: Workspace Resolution in Tools
# ---------------------------------------------------------------------------

class TestWorkspaceResolution:
    """Verify workspace parameter resolves to scope in tool calls."""

    def test_write_with_workspace(self, tools, full_mcp):
        _, ws_router, _, _ = full_mcp
        ws_router.register("audit-sias", scope="sias-audit", corpus_id="sias-v7")

        result = tools["memory_write"](
            title="Test via workspace",
            content="Item written via workspace param",
            scope="project",
            workspace="audit-sias",
        )
        assert result.get("status") == "ok"

    def test_search_with_workspace(self, tools, full_mcp):
        _, ws_router, _, _ = full_mcp
        ws_router.register("my-ws", scope="my-scope")

        # Write an item in my-scope
        tools["memory_write"](
            title="Scoped item",
            content="Item in my-scope",
            scope="my-scope",
        )

        result = tools["memory_search"](query="scoped", workspace="my-ws")
        assert result.get("status") == "ok"

    def test_unknown_workspace_returns_error(self, tools):
        result = tools["memory_search"](query="test", workspace="nonexistent")
        assert result["status"] == "error"
        assert "Unknown workspace" in result["message"]

    def test_recall_with_workspace(self, tools, full_mcp):
        _, ws_router, _, _ = full_mcp
        ws_router.register("recall-ws", scope="recall-scope")

        result = tools["memory_recall"](query="anything", workspace="recall-ws")
        assert result.get("status") == "ok"

    def test_consolidate_with_workspace(self, tools, full_mcp):
        _, ws_router, _, _ = full_mcp
        ws_router.register("cons-ws", scope="cons-scope")

        result = tools["memory_consolidate"](workspace="cons-ws")
        # Consolidation runs (even if nothing to consolidate)
        assert result.get("status") == "ok"

    def test_stats_with_workspace(self, tools, full_mcp):
        _, ws_router, _, _ = full_mcp
        ws_router.register("stats-ws", scope="stats-scope", corpus_id="c1")

        result = tools["memory_stats"](workspace="stats-ws")
        assert result.get("status") == "ok"

    def test_no_workspace_uses_scope(self, tools):
        """When workspace is None, scope is used directly (backward compat)."""
        result = tools["memory_write"](
            title="Direct scope",
            content="No workspace param",
            scope="direct-scope",
        )
        assert result.get("status") == "ok"

    def test_propose_with_workspace(self, tools, full_mcp):
        _, ws_router, _, _ = full_mcp
        ws_router.register("prop-ws", scope="prop-scope")

        items = json.dumps([{
            "title": "Proposal via ws",
            "content": "Testing workspace in propose",
            "tags": ["test"],
            "type": "note",
        }])
        result = tools["memory_propose"](items=items, workspace="prop-ws")
        assert result.get("status") == "ok"


# ---------------------------------------------------------------------------
# Test 3: Rate Limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Verify rate limiter blocks tool calls when exhausted."""

    def test_calls_within_limit_succeed(self, tools):
        """5 calls should succeed (limit is 5/min with burst=1.0)."""
        for i in range(5):
            result = tools["memory_stats"]()
            assert result["status"] == "ok", f"Call {i+1} should succeed"

    def test_exceeding_limit_returns_error(self, tools):
        """6th call should be rate-limited."""
        for _ in range(5):
            tools["memory_stats"]()

        result = tools["memory_stats"]()
        assert result["status"] == "error"
        assert "Rate limit" in result["message"]
        assert "retry_after_ms" in result

    def test_rate_limit_error_recorded_in_metrics(self, tools, full_mcp):
        _, _, mc, _ = full_mcp
        # Exhaust the rate limit
        for _ in range(5):
            tools["memory_search"](query="test")

        # 6th call should be blocked
        tools["memory_search"](query="test")

        m = mc.get_metrics("memory_search")
        assert m["error_count"] >= 1

    def test_different_tools_share_rate_limit(self, tools):
        """All non-session tools share the _DEFAULT_SESSION bucket."""
        tools["memory_stats"]()
        tools["memory_stats"]()
        tools["memory_stats"]()
        tools["memory_search"](query="a")
        tools["memory_search"](query="b")

        # 6th total call should be blocked regardless of tool name
        result = tools["memory_stats"]()
        assert result["status"] == "error"

    def test_metrics_tool_not_rate_limited(self, tools):
        """memory_metrics has no rate guard — always works."""
        # Exhaust the bucket
        for _ in range(5):
            tools["memory_stats"]()

        # metrics should still work
        result = tools["memory_metrics"]()
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Test 4: Proposal Rate Limiting
# ---------------------------------------------------------------------------

class TestProposalLimit:
    """Verify per-turn proposal cap."""

    def test_proposals_within_limit_succeed(self, tools):
        """3 proposals per turn (limit is 3)."""
        for i in range(3):
            items = json.dumps([{
                "title": f"Item {i}",
                "content": f"Content {i}",
                "tags": ["test"],
                "type": "note",
            }])
            result = tools["memory_propose"](items=items)
            assert result.get("status") == "ok", f"Proposal {i+1} should succeed"

    def test_exceeding_proposal_limit(self, tools):
        """4th proposal in same turn should be blocked."""
        items = json.dumps([{
            "title": "Test",
            "content": "Content",
            "tags": ["t"],
            "type": "note",
        }])
        for _ in range(3):
            tools["memory_propose"](items=items)

        result = tools["memory_propose"](items=items)
        assert result["status"] == "error"
        assert "Proposal limit" in result["message"]


# ---------------------------------------------------------------------------
# Test 5: Metrics Recording
# ---------------------------------------------------------------------------

class TestMetricsRecording:
    """Verify metrics collector records tool call latency."""

    def test_successful_call_recorded(self, tools, full_mcp):
        _, _, mc, _ = full_mcp
        tools["memory_stats"]()

        m = mc.get_metrics("memory_stats")
        assert m["call_count"] == 1
        assert m["total_latency_ms"] > 0
        assert m["error_count"] == 0

    def test_multiple_tools_tracked(self, tools, full_mcp):
        _, _, mc, _ = full_mcp
        tools["memory_stats"]()
        tools["memory_search"](query="test")
        tools["memory_write"](title="t", content="c")

        summary = mc.get_summary()
        assert summary["total_calls"] == 3
        assert summary["tools_used"] == 3

    def test_write_records_latency(self, tools, full_mcp):
        _, _, mc, _ = full_mcp
        tools["memory_write"](title="Test", content="Content", tags="a,b")

        m = mc.get_metrics("memory_write")
        assert m["call_count"] == 1
        assert m["avg_latency_ms"] > 0

    def test_workspace_tools_recorded(self, tools, full_mcp):
        _, _, mc, _ = full_mcp
        tools["memory_workspace_list"]()

        m = mc.get_metrics("memory_workspace_list")
        assert m["call_count"] == 1


# ---------------------------------------------------------------------------
# Test 6: Management Tools
# ---------------------------------------------------------------------------

class TestManagementTools:
    """Verify workspace and metrics management tools."""

    def test_workspace_list_default(self, tools):
        result = tools["memory_workspace_list"]()
        assert result["status"] == "ok"
        names = [w["name"] for w in result["workspaces"]]
        assert "default" in names

    def test_workspace_register_and_list(self, tools):
        result = tools["memory_workspace_register"](
            name="test-ws",
            scope="test-scope",
            corpus_id="c42",
            description="Test workspace",
        )
        assert result["status"] == "ok"
        assert result["workspace"]["name"] == "test-ws"
        assert result["workspace"]["scope"] == "test-scope"
        assert result["workspace"]["corpus_id"] == "c42"

        # Verify it appears in list
        listing = tools["memory_workspace_list"]()
        names = [w["name"] for w in listing["workspaces"]]
        assert "test-ws" in names

    def test_workspace_register_empty_name_error(self, tools):
        result = tools["memory_workspace_register"](name="")
        assert result["status"] == "error"

    def test_workspace_remove(self, tools):
        tools["memory_workspace_register"](name="to-remove")
        result = tools["memory_workspace_remove"](name="to-remove")
        assert result["status"] == "ok"
        assert result["removed"] is True

    def test_workspace_remove_default_rejected(self, tools):
        result = tools["memory_workspace_remove"](name="default")
        assert result["status"] == "ok"
        assert result["removed"] is False

    def test_workspace_remove_nonexistent(self, tools):
        result = tools["memory_workspace_remove"](name="ghost")
        assert result["status"] == "ok"
        assert result["removed"] is False

    def test_metrics_summary(self, tools, full_mcp):
        _, _, _, rl = full_mcp
        # Make some calls first
        tools["memory_stats"]()
        tools["memory_search"](query="x")

        result = tools["memory_metrics"]()
        assert result["status"] == "ok"
        assert result["total_calls"] >= 2
        assert result["tools_used"] >= 2
        assert "rate_limiter" in result  # rate limiter status appended

    def test_metrics_per_tool(self, tools, full_mcp):
        tools["memory_stats"]()

        result = tools["memory_metrics"](tool_name="memory_stats")
        assert result["status"] == "ok"
        assert result["metrics"]["call_count"] == 1

    def test_metrics_unknown_tool(self, tools):
        result = tools["memory_metrics"](tool_name="memory_nonexistent")
        assert result["status"] == "ok"
        assert result["metrics"]["call_count"] == 0


# ---------------------------------------------------------------------------
# Test 7: Session Bridge with Rate Limiter
# ---------------------------------------------------------------------------

class TestSessionBridgeRateLimiting:
    """Session bridge tools use their explicit session_id for rate limiting."""

    def test_session_inject_succeeds(self, tools):
        result = tools["memory_session_inject"](
            query="test query",
            session_id="proj:conv1",
        )
        assert result.get("status") == "ok"

    def test_session_store_succeeds(self, tools):
        result = tools["memory_session_store"](
            response_text="This is a response with no proposals.",
            session_id="proj:conv1",
        )
        assert result.get("status") == "ok"

    def test_session_inject_rate_limited_separately(self, tools):
        """Session tools use their own session_id, not _DEFAULT_SESSION."""
        # Exhaust the default session bucket (5 calls)
        for _ in range(5):
            tools["memory_stats"]()

        # Default bucket exhausted, but session bridge uses different session_id
        result = tools["memory_session_inject"](
            query="test",
            session_id="proj:conv99",
        )
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Test 8: No Modules Graceful Degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """When modules are None, tools still work (Phase 1 backward compat)."""

    def test_no_modules_core_tools_work(self, dispatcher):
        """Register with all modules=None — core tools still function."""
        mock = MockMCP()
        from ragix_core.memory.mcp.tools import register_memory_tools
        register_memory_tools(mock, dispatcher)

        tools = mock.tools
        # Only 11 core tools (no session, no workspace, no metrics)
        assert len(tools) == 11

        result = tools["memory_stats"]()
        assert result["status"] == "ok"

    def test_workspace_param_ignored_without_router(self, dispatcher):
        """workspace= param silently falls back to scope when no router."""
        mock = MockMCP()
        from ragix_core.memory.mcp.tools import register_memory_tools
        register_memory_tools(mock, dispatcher)

        tools = mock.tools
        result = tools["memory_write"](
            title="Test",
            content="No router",
            workspace="anything",
            scope="project",
        )
        assert result.get("status") == "ok"


# ---------------------------------------------------------------------------
# Test 9: Disabled Rate Limiting
# ---------------------------------------------------------------------------

class TestDisabledRateLimiting:
    """When rate_limit.enabled=False, all calls are allowed."""

    @pytest.fixture
    def unlimited_tools(self, dispatcher):
        mock = MockMCP()
        from ragix_core.memory.mcp.tools import register_memory_tools
        register_memory_tools(
            mock,
            dispatcher,
            rate_limiter=RateLimiter(RateLimitConfig(enabled=False)),
            metrics=MetricsCollector(),
        )
        return mock.tools

    def test_unlimited_calls(self, unlimited_tools):
        """100 calls should all succeed when rate limiting is disabled."""
        for _ in range(100):
            result = unlimited_tools["memory_stats"]()
            assert result["status"] == "ok"

    def test_unlimited_proposals(self, unlimited_tools):
        items = json.dumps([{
            "title": "T",
            "content": "C",
            "tags": ["t"],
            "type": "note",
        }])
        for _ in range(50):
            result = unlimited_tools["memory_propose"](items=items)
            assert result.get("status") == "ok"
