"""
Per-tool call metrics collector for Memory MCP Server.

Tracks call_count, latency, error_count per tool name.
Thread-safe via explicit lock. All output is JSON-serializable
(datetimes emitted as ISO-8601 strings).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _monotonic_ms() -> float:
    """Current monotonic clock value in milliseconds."""
    return time.monotonic() * 1000.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ToolMetrics:
    """Per-tool accumulated metrics."""

    tool_name: str
    call_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    last_call_at: str = ""

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds (0.0 if no calls recorded)."""
        if self.call_count == 0:
            return 0.0
        return self.total_latency_ms / self.call_count

    def to_dict(self) -> Dict:
        """JSON-serializable dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "call_count": self.call_count,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "error_count": self.error_count,
            "last_call_at": self.last_call_at,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
        }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Thread-safe collector for per-tool call metrics.

    Usage::

        collector = MetricsCollector()

        # Manual recording
        collector.record_call("memory.search", latency_ms=42.5)
        collector.record_call("memory.write", latency_ms=18.1, error=True)

        # Auto-timed recording via context manager
        with collector.timed_call("memory.read"):
            result = do_expensive_work()

        # Inspect
        print(collector.get_summary())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tool_metrics: Dict[str, ToolMetrics] = {}
        self._start_time: float = time.monotonic()
        self._start_iso: str = _now_iso()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_call(
        self,
        tool_name: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """
        Record a single tool call.

        Parameters
        ----------
        tool_name : str
            Identifier of the tool (e.g. ``"memory.search"``).
        latency_ms : float
            Wall-clock latency of the call in milliseconds.
        error : bool
            Whether the call resulted in an error.
        """
        now = _now_iso()
        with self._lock:
            tm = self._tool_metrics.get(tool_name)
            if tm is None:
                tm = ToolMetrics(tool_name=tool_name)
                self._tool_metrics[tool_name] = tm
            tm.call_count += 1
            tm.total_latency_ms += latency_ms
            if error:
                tm.error_count += 1
            tm.last_call_at = now

    # ------------------------------------------------------------------
    # Context manager for auto-timed recording
    # ------------------------------------------------------------------

    @contextmanager
    def timed_call(self, tool_name: str) -> Iterator[None]:
        """
        Context manager that measures wall-clock latency and records it.

        If the wrapped block raises an exception the call is recorded with
        ``error=True`` and the exception is re-raised.

        Usage::

            with collector.timed_call("memory.write"):
                perform_write()
        """
        t0 = _monotonic_ms()
        error = False
        try:
            yield
        except Exception:
            error = True
            raise
        finally:
            elapsed = _monotonic_ms() - t0
            self.record_call(tool_name, latency_ms=elapsed, error=error)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_metrics(self, tool_name: Optional[str] = None) -> Dict:
        """
        Return metrics for a single tool or all tools.

        Parameters
        ----------
        tool_name : str or None
            If provided, return metrics for that tool only.
            If ``None``, return a dict keyed by tool name.

        Returns
        -------
        dict
            JSON-serializable metrics dictionary.
        """
        with self._lock:
            if tool_name is not None:
                tm = self._tool_metrics.get(tool_name)
                if tm is None:
                    return {
                        "tool_name": tool_name,
                        "call_count": 0,
                        "total_latency_ms": 0.0,
                        "error_count": 0,
                        "last_call_at": "",
                        "avg_latency_ms": 0.0,
                    }
                return tm.to_dict()

            return {
                name: tm.to_dict()
                for name, tm in self._tool_metrics.items()
            }

    def get_summary(self) -> Dict:
        """
        Return an aggregate summary across all tools.

        Returns
        -------
        dict
            Keys: ``total_calls``, ``total_errors``, ``avg_latency_ms``,
            ``tools_used``, ``uptime_seconds``, ``started_at``, ``by_tool``
            (list sorted by call_count descending).
        """
        with self._lock:
            total_calls = 0
            total_errors = 0
            total_latency = 0.0

            for tm in self._tool_metrics.values():
                total_calls += tm.call_count
                total_errors += tm.error_count
                total_latency += tm.total_latency_ms

            avg_latency = (total_latency / total_calls) if total_calls > 0 else 0.0
            uptime = time.monotonic() - self._start_time

            by_tool = sorted(
                [tm.to_dict() for tm in self._tool_metrics.values()],
                key=lambda d: d["call_count"],
                reverse=True,
            )

            return {
                "total_calls": total_calls,
                "total_errors": total_errors,
                "avg_latency_ms": round(avg_latency, 3),
                "tools_used": len(self._tool_metrics),
                "uptime_seconds": round(uptime, 3),
                "started_at": self._start_iso,
                "by_tool": by_tool,
            }

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all collected metrics and reset the uptime clock."""
        with self._lock:
            self._tool_metrics.clear()
            self._start_time = time.monotonic()
            self._start_iso = _now_iso()
        logger.debug("MetricsCollector reset")
