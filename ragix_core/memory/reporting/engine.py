"""
ReportEngine — MCP tool adapter, timing, and assertion helpers.

Wraps MemoryStore + RecallEngine + MemoryToolDispatcher behind a simple
``engine.tool(name, **kwargs)`` interface, with timing and assertions.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from ragix_core.memory.store import MemoryStore
from ragix_core.memory.embedder import MockEmbedder
from ragix_core.memory.recall import RecallEngine
from ragix_core.memory.tools import MemoryToolDispatcher
from ragix_core.memory.mcp.workspace import WorkspaceRouter
from ragix_core.memory.mcp.metrics import MetricsCollector


class _MockMCP:
    """Captures ``@mcp.tool()`` registrations for direct invocation."""

    def __init__(self):
        self.tools: Dict[str, Any] = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator

    def resource(self, uri):
        def decorator(fn):
            return fn
        return decorator


class ReportEngine:
    """Orchestrates MCP tool calls, timing, and assertions for reporting."""

    def __init__(
        self,
        db_path: str,
        workspace: str,
        *,
        embedder: str = "mock",
        scope: str = "audit",
        corpus_id: Optional[str] = None,
    ):
        self.db_path = db_path
        self.workspace = workspace
        self.scope = scope
        self.corpus_id = corpus_id

        # Core stack
        self.store = MemoryStore(db_path)
        self.embedder = MockEmbedder()
        self.recall = RecallEngine(self.store, self.embedder)
        self.dispatcher = MemoryToolDispatcher(
            self.store, self.embedder, self.recall,
        )
        self.workspace_router = WorkspaceRouter(self.store._conn)
        self.metrics = MetricsCollector()

        # Register MCP tools
        self._mcp = _MockMCP()
        from ragix_core.memory.mcp.tools import register_memory_tools
        register_memory_tools(
            self._mcp,
            self.dispatcher,
            workspace_router=self.workspace_router,
            metrics=self.metrics,
        )

        # Ensure workspace exists
        try:
            self.workspace_router.resolve(workspace)
        except KeyError:
            if scope and corpus_id:
                self.workspace_router.register(workspace, scope, corpus_id)

        # Timing storage
        self._timings: Dict[str, float] = {}
        self._errors: list = []

    # ── Tool dispatch ──────────────────────────────────────────────────

    def tool(self, name: str, **kwargs) -> dict:
        """Call an MCP tool by name, return its result dict."""
        if name not in self._mcp.tools:
            raise KeyError(f"Unknown tool: {name}")
        try:
            return self._mcp.tools[name](**kwargs)
        except Exception as exc:
            self._errors.append({"tool": name, "error": str(exc)})
            raise

    @property
    def n_tools(self) -> int:
        return len(self._mcp.tools)

    # ── Timing ─────────────────────────────────────────────────────────

    @contextmanager
    def timed(self, label: str):
        """Context manager that records elapsed ms under ``label``."""
        t0 = time.perf_counter()
        yield
        dt_ms = (time.perf_counter() - t0) * 1000
        self._timings[label] = dt_ms

    def timings(self) -> Dict[str, float]:
        """Return all recorded timings as {label: ms}."""
        return dict(self._timings)

    # ── Metrics ────────────────────────────────────────────────────────

    def metrics_summary(self) -> dict:
        """Return MCP metrics summary from MetricsCollector."""
        return self.metrics.get_summary()

    # ── Store access (for inventory/links) ─────────────────────────────

    def list_items(self, **kwargs):
        """List items from store (returns MemoryItem dataclass list)."""
        if "scope" not in kwargs:
            kwargs["scope"] = self.scope
        return self.store.list_items(**kwargs)

    def query_links(self, item_id: str, direction: str = "outgoing"):
        """Query links table directly for an item."""
        conn = self.store._conn
        if direction == "outgoing":
            return conn.execute(
                "SELECT dst_id, rel FROM memory_links WHERE src_id = ?",
                (item_id,),
            ).fetchall()
        else:
            return conn.execute(
                "SELECT src_id, rel FROM memory_links WHERE dst_id = ?",
                (item_id,),
            ).fetchall()

    # ── Assert helpers ─────────────────────────────────────────────────

    def assert_format_version(self, inject_text: str, expected: int = 1):
        """Assert injection block contains expected format_version."""
        match = re.search(r"format_version:\s*(\d+)", inject_text)
        if not match:
            raise AssertionError(
                "Injection block missing format_version header"
            )
        actual = int(match.group(1))
        if actual != expected:
            raise AssertionError(
                f"format_version mismatch: expected {expected}, got {actual}"
            )

    def assert_min_count(self, label: str, actual: int, minimum: int):
        """Assert count >= minimum."""
        if actual < minimum:
            raise AssertionError(
                f"{label}: expected >= {minimum}, got {actual}"
            )

    def assert_max_latency(self, label: str, max_ms: float):
        """Assert recorded timing for label <= max_ms."""
        actual = self._timings.get(label)
        if actual is None:
            raise AssertionError(f"No timing recorded for '{label}'")
        if actual > max_ms:
            raise AssertionError(
                f"{label}: {actual:.1f} ms exceeds threshold {max_ms:.1f} ms"
            )

    def assert_no_errors(self):
        """Assert no tool call errors occurred."""
        if self._errors:
            msgs = "; ".join(
                f"{e['tool']}: {e['error']}" for e in self._errors
            )
            raise AssertionError(f"Tool errors: {msgs}")
