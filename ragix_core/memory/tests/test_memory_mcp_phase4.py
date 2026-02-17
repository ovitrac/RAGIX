"""
Phase 4 integration tests for RAGIX Memory MCP — Production Hardening.

Focus areas:
    1. WorkspaceRouter: register, resolve, list, remove, persistence
    2. MetricsCollector: record, timed_call, summary, reset
    3. RateLimiter: token bucket, proposal limits, turn reset, disabled mode
    4. RateLimitConfig integration with MemoryConfig
    5. Edge cases and thread safety

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from ragix_core.memory.config import (
    MemoryConfig,
    RateLimitConfig,
)
from ragix_core.memory.mcp.metrics import MetricsCollector, ToolMetrics
from ragix_core.memory.mcp.rate_limiter import (
    RateLimiter,
    RateLimitResult,
    SessionBucket,
)
from ragix_core.memory.mcp.workspace import (
    WorkspaceInfo,
    WorkspaceRouter,
    _DEFAULT_NAME,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory SQLite connection with Row factory (mimics MemoryStore)."""
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


@pytest.fixture
def router(conn) -> WorkspaceRouter:
    """WorkspaceRouter backed by the in-memory connection."""
    return WorkspaceRouter(conn)


@pytest.fixture
def metrics() -> MetricsCollector:
    """Fresh MetricsCollector."""
    return MetricsCollector()


@pytest.fixture
def limiter() -> RateLimiter:
    """Default RateLimiter."""
    return RateLimiter()


# ===========================================================================
# 1. WorkspaceRouter Tests
# ===========================================================================

class TestWorkspaceRouter:
    """Test workspace registration, resolution, and lifecycle."""

    def test_default_workspace_exists(self, router):
        """The 'default' workspace is auto-created on init."""
        ws = router.get_workspace("default")
        assert ws is not None
        assert ws.scope == "project"
        assert ws.corpus_id == ""

    def test_resolve_none_returns_default(self, router):
        """resolve(None) returns the default workspace."""
        scope, corpus_id = router.resolve(None)
        assert scope == "project"
        assert corpus_id == ""

    def test_resolve_empty_returns_default(self, router):
        """resolve('') returns the default workspace."""
        scope, corpus_id = router.resolve("")
        assert scope == "project"

    def test_register_auto_scope(self, router):
        """When scope is None, workspace name is used as scope."""
        info = router.register("audit-msg_hub", description="MSG-HUB audit")
        assert info.scope == "audit-msg_hub"
        assert info.corpus_id == ""

    def test_register_explicit_scope(self, router):
        """Explicit scope and corpus_id are honored."""
        info = router.register(
            "acme_erp", scope="audit", corpus_id="CORP-001", description="ACME-ERP"
        )
        assert info.scope == "audit"
        assert info.corpus_id == "CORP-001"

    def test_resolve_registered(self, router):
        """resolve() returns the scope and corpus_id of a registered workspace."""
        router.register("test-ws", scope="test-scope", corpus_id="C1")
        scope, cid = router.resolve("test-ws")
        assert scope == "test-scope"
        assert cid == "C1"

    def test_resolve_unknown_raises(self, router):
        """resolve() raises KeyError for unknown workspace names."""
        with pytest.raises(KeyError, match="nosuch"):
            router.resolve("nosuch")

    def test_list_workspaces(self, router):
        """list_workspaces returns all registered workspaces."""
        router.register("ws1")
        router.register("ws2")
        names = [w["name"] for w in router.list_workspaces()]
        assert "default" in names
        assert "ws1" in names
        assert "ws2" in names

    def test_register_update_preserves_created_at(self, router):
        """Updating a workspace preserves its created_at timestamp."""
        info1 = router.register("upd-test", scope="v1")
        info2 = router.register("upd-test", scope="v2")
        assert info2.scope == "v2"
        ws = router.get_workspace("upd-test")
        assert ws.created_at == info1.created_at

    def test_remove_workspace(self, router):
        """remove() deletes a workspace and returns True."""
        router.register("to-delete")
        assert router.remove("to-delete") is True
        assert router.get_workspace("to-delete") is None

    def test_remove_nonexistent(self, router):
        """remove() returns False for non-existent workspace."""
        assert router.remove("nope") is False

    def test_remove_default_rejected(self, router):
        """Cannot remove the 'default' workspace."""
        assert router.remove("default") is False
        assert router.get_workspace("default") is not None

    def test_register_empty_name_raises(self, router):
        """register('') raises ValueError."""
        with pytest.raises(ValueError):
            router.register("")

    def test_concurrent_registration(self, router):
        """Multiple threads can register workspaces concurrently."""
        errors = []

        def worker(i):
            try:
                router.register(f"thread-{i}", description=f"worker {i}")
                router.resolve(f"thread-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        ws_list = router.list_workspaces()
        assert len(ws_list) >= 11  # default + 10 threads

    def test_persistence_across_instances(self, conn):
        """Workspaces persist across WorkspaceRouter instances."""
        r1 = WorkspaceRouter(conn)
        r1.register("persistent-ws", scope="saved")

        r2 = WorkspaceRouter(conn)
        ws = r2.get_workspace("persistent-ws")
        assert ws is not None
        assert ws.scope == "saved"


# ===========================================================================
# 2. MetricsCollector Tests
# ===========================================================================

class TestMetricsCollector:
    """Test per-tool metrics collection."""

    def test_record_and_retrieve(self, metrics):
        """record_call stores metrics retrievable via get_metrics."""
        metrics.record_call("memory.search", latency_ms=42.5)
        m = metrics.get_metrics("memory.search")
        assert m["call_count"] == 1
        assert m["total_latency_ms"] == 42.5
        assert m["error_count"] == 0

    def test_record_error(self, metrics):
        """Error calls increment error_count."""
        metrics.record_call("memory.write", latency_ms=10.0, error=True)
        m = metrics.get_metrics("memory.write")
        assert m["error_count"] == 1
        assert m["call_count"] == 1

    def test_avg_latency(self, metrics):
        """avg_latency_ms computed from total_latency / call_count."""
        metrics.record_call("memory.search", latency_ms=40.0)
        metrics.record_call("memory.search", latency_ms=60.0)
        m = metrics.get_metrics("memory.search")
        assert m["avg_latency_ms"] == 50.0

    def test_unknown_tool_returns_zeros(self, metrics):
        """get_metrics for unknown tool returns zero values."""
        m = metrics.get_metrics("nonexistent")
        assert m["call_count"] == 0
        assert m["avg_latency_ms"] == 0.0

    def test_get_all_metrics(self, metrics):
        """get_metrics(None) returns all tracked tools."""
        metrics.record_call("tool_a", latency_ms=1.0)
        metrics.record_call("tool_b", latency_ms=2.0)
        all_m = metrics.get_metrics()
        assert "tool_a" in all_m
        assert "tool_b" in all_m

    def test_summary_aggregates(self, metrics):
        """get_summary aggregates across all tools."""
        metrics.record_call("a", latency_ms=10.0)
        metrics.record_call("a", latency_ms=20.0, error=True)
        metrics.record_call("b", latency_ms=30.0)
        s = metrics.get_summary()
        assert s["total_calls"] == 3
        assert s["total_errors"] == 1
        assert s["tools_used"] == 2
        assert s["uptime_seconds"] >= 0

    def test_summary_sorted_by_count(self, metrics):
        """by_tool in summary is sorted by call_count descending."""
        metrics.record_call("less", latency_ms=1.0)
        for _ in range(5):
            metrics.record_call("more", latency_ms=1.0)
        s = metrics.get_summary()
        assert s["by_tool"][0]["tool_name"] == "more"

    def test_timed_call_success(self, metrics):
        """timed_call context manager records latency on success."""
        with metrics.timed_call("timed_tool"):
            time.sleep(0.01)  # ~10ms
        m = metrics.get_metrics("timed_tool")
        assert m["call_count"] == 1
        assert m["error_count"] == 0
        assert m["total_latency_ms"] >= 8.0  # at least ~10ms

    def test_timed_call_error(self, metrics):
        """timed_call records error=True when exception is raised."""
        with pytest.raises(ValueError):
            with metrics.timed_call("error_tool"):
                raise ValueError("boom")
        m = metrics.get_metrics("error_tool")
        assert m["call_count"] == 1
        assert m["error_count"] == 1

    def test_reset(self, metrics):
        """reset clears all metrics."""
        metrics.record_call("tool", latency_ms=5.0)
        metrics.reset()
        s = metrics.get_summary()
        assert s["total_calls"] == 0
        assert s["tools_used"] == 0

    def test_tool_metrics_dataclass(self):
        """ToolMetrics avg_latency_ms property works correctly."""
        tm = ToolMetrics(tool_name="test", call_count=0)
        assert tm.avg_latency_ms == 0.0
        tm.call_count = 4
        tm.total_latency_ms = 100.0
        assert tm.avg_latency_ms == 25.0

    def test_concurrent_recording(self, metrics):
        """Multiple threads can record calls concurrently."""
        errors = []

        def worker(i):
            try:
                for _ in range(20):
                    metrics.record_call(f"tool-{i % 3}", latency_ms=1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        s = metrics.get_summary()
        assert s["total_calls"] == 100  # 5 threads * 20 calls


# ===========================================================================
# 3. RateLimiter Tests
# ===========================================================================

class TestRateLimiter:
    """Test token bucket rate limiting."""

    def test_fresh_session_allowed(self, limiter):
        """A fresh session should be allowed."""
        result = limiter.check_rate("sess1")
        assert result.allowed is True
        assert result.remaining > 0

    def test_consume_deducts_tokens(self, limiter):
        """consume() reduces the token count."""
        assert limiter.consume("sess1", 1) is True
        status = limiter.get_status("sess1")
        assert status["tokens"] < 60.0

    def test_exhausted_bucket_blocks(self):
        """When all tokens consumed, check_rate returns allowed=False."""
        rl = RateLimiter(RateLimitConfig(calls_per_minute=2, burst_multiplier=1.0))
        assert rl.consume("s", 2) is True
        result = rl.check_rate("s")
        assert result.allowed is False
        assert result.retry_after_ms > 0

    def test_consume_returns_false_when_insufficient(self):
        """consume() returns False when not enough tokens."""
        rl = RateLimiter(RateLimitConfig(calls_per_minute=3, burst_multiplier=1.0))
        assert rl.consume("s", 3) is True
        assert rl.consume("s", 1) is False

    def test_proposal_limit(self, limiter):
        """Proposal limit enforced per turn."""
        # Default: 10 proposals per turn
        result = limiter.check_proposal_limit("sess", turn=1, count=8)
        assert result.allowed is True
        assert result.remaining == 2

        result = limiter.check_proposal_limit("sess", turn=1, count=3)
        assert result.allowed is False

    def test_proposal_turn_reset(self, limiter):
        """Proposal counter resets on new turn."""
        limiter.check_proposal_limit("sess", turn=1, count=10)
        # New turn should reset
        result = limiter.check_proposal_limit("sess", turn=2, count=1)
        assert result.allowed is True
        assert result.remaining == 9

    def test_reset_session(self, limiter):
        """reset_session clears all state for a session."""
        limiter.consume("sess", 30)
        limiter.reset_session("sess")
        status = limiter.get_status("sess")
        assert status["tokens"] == 60.0  # fresh bucket

    def test_disabled_mode_allows_all(self):
        """When enabled=False, all checks pass."""
        rl = RateLimiter(RateLimitConfig(enabled=False))
        assert rl.check_rate("x").allowed is True
        assert rl.consume("x", 99999) is True
        assert rl.check_proposal_limit("x", turn=1, count=99999).allowed is True

    def test_burst_multiplier(self):
        """Bucket can hold calls_per_minute * burst_multiplier tokens."""
        cfg = RateLimitConfig(calls_per_minute=10, burst_multiplier=2.0)
        rl = RateLimiter(cfg)
        status = rl.get_status("sess")
        assert status["max_tokens"] == 20.0

    def test_refill_over_time(self):
        """Tokens refill based on elapsed time."""
        cfg = RateLimitConfig(calls_per_minute=60, burst_multiplier=1.0)
        rl = RateLimiter(cfg)
        rl.consume("sess", 60)  # drain completely

        # Manually advance the last_refill timestamp to simulate time passing
        with rl._lock:
            bucket = rl._get_bucket("sess")
            bucket.last_refill -= 1.0  # pretend 1 second elapsed

        result = rl.check_rate("sess")
        assert result.allowed is True  # should have ~1 token refilled

    def test_get_status_fields(self, limiter):
        """get_status returns expected fields."""
        status = limiter.get_status("sess")
        assert "tokens" in status
        assert "max_tokens" in status
        assert "proposals_this_turn" in status
        assert "proposals_limit" in status
        assert "current_turn" in status
        assert "enabled" in status

    def test_concurrent_consume(self):
        """Multiple threads can consume tokens concurrently."""
        cfg = RateLimitConfig(calls_per_minute=1000, burst_multiplier=1.0)
        rl = RateLimiter(cfg)
        successes = [0]
        lock = threading.Lock()

        def worker():
            for _ in range(20):
                if rl.consume("shared", 1):
                    with lock:
                        successes[0] += 1

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have consumed exactly 200 tokens (1000 available, 200 requested)
        assert successes[0] == 200

    def test_rate_limit_result_fields(self):
        """RateLimitResult has expected fields."""
        r = RateLimitResult(allowed=True, reason="ok", remaining=5, retry_after_ms=0)
        assert r.allowed is True
        assert r.reason == "ok"
        assert r.remaining == 5


# ===========================================================================
# 4. Config Integration Tests
# ===========================================================================

class TestConfigIntegration:
    """Test RateLimitConfig in MemoryConfig."""

    def test_default_rate_limit_in_config(self):
        """MemoryConfig includes RateLimitConfig with defaults."""
        cfg = MemoryConfig()
        assert isinstance(cfg.rate_limit, RateLimitConfig)
        assert cfg.rate_limit.calls_per_minute == 60
        assert cfg.rate_limit.enabled is True

    def test_from_dict_rate_limit(self):
        """from_dict correctly parses rate_limit section."""
        cfg = MemoryConfig.from_dict({
            "rate_limit": {
                "calls_per_minute": 120,
                "enabled": False,
            }
        })
        assert cfg.rate_limit.calls_per_minute == 120
        assert cfg.rate_limit.enabled is False

    def test_from_dict_empty_rate_limit(self):
        """from_dict with no rate_limit section uses defaults."""
        cfg = MemoryConfig.from_dict({})
        assert cfg.rate_limit.calls_per_minute == 60

    def test_rate_limit_config_from_rate_limiter(self):
        """RateLimiter can be constructed from MemoryConfig.rate_limit."""
        cfg = MemoryConfig.from_dict({"rate_limit": {"calls_per_minute": 30}})
        rl = RateLimiter(cfg.rate_limit)
        status = rl.get_status("sess")
        # Max tokens = 30 * 1.5 = 45
        assert status["max_tokens"] == 45.0


# ===========================================================================
# 5. Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Edge case tests for Phase 4 modules."""

    def test_workspace_info_dataclass(self):
        """WorkspaceInfo can be constructed with minimal args."""
        ws = WorkspaceInfo(name="test")
        assert ws.scope == "project"
        assert ws.corpus_id == ""
        assert ws.created_at != ""

    def test_session_bucket_defaults(self):
        """SessionBucket initializes with sensible defaults."""
        sb = SessionBucket()
        assert sb.tokens == 0.0
        assert sb.proposals_this_turn == 0
        assert sb.current_turn == 0

    def test_metrics_empty_summary(self, metrics):
        """Summary with no recorded calls returns zeros."""
        s = metrics.get_summary()
        assert s["total_calls"] == 0
        assert s["avg_latency_ms"] == 0.0
        assert s["by_tool"] == []

    def test_workspace_default_name_constant(self):
        """The default workspace name constant is 'default'."""
        assert _DEFAULT_NAME == "default"

    def test_limiter_multiple_sessions(self, limiter):
        """Different sessions have independent buckets."""
        limiter.consume("sess-a", 30)
        status_a = limiter.get_status("sess-a")
        status_b = limiter.get_status("sess-b")
        assert status_a["tokens"] < status_b["tokens"]
