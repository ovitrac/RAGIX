"""
Monitoring - Production metrics, health checks, and observability

Provides monitoring infrastructure for RAGIX agents and workflows.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import asyncio
import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Deque
import statistics

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """A single metric measurement."""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
        }


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


class MetricsCollector:
    """
    Collects and aggregates metrics.

    Thread-safe metrics collection with support for counters,
    gauges, histograms, and timers.
    """

    def __init__(self, retention_minutes: int = 60):
        """
        Initialize metrics collector.

        Args:
            retention_minutes: How long to keep metrics
        """
        self.retention = timedelta(minutes=retention_minutes)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=1000))
        self._timers: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value for a histogram."""
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    def record_time(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._timers[key].append(duration)

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._gauges.get(key)

    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        with self._lock:
            values = list(self._histograms.get(key, []))

        if not values:
            return {}

        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values),
            "p99": sorted(values)[int(len(values) * 0.99)] if len(values) >= 100 else max(values),
        }

    def get_timer_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics."""
        return self.get_histogram_stats(name, labels)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self.get_histogram_stats(k) for k in self._histograms
                },
                "timers": {
                    k: self.get_timer_stats(k) for k in self._timers
                },
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()


class HealthChecker:
    """
    Runs health checks for system components.

    Supports custom health check functions and aggregates results.
    """

    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._results: Dict[str, HealthCheck] = {}
        self._lock = threading.RLock()

    def register(self, name: str, check_fn: Callable[[], HealthCheck]):
        """
        Register a health check.

        Args:
            name: Unique name for the check
            check_fn: Function that returns HealthCheck
        """
        with self._lock:
            self._checks[name] = check_fn

    def unregister(self, name: str):
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
            self._results.pop(name, None)

    def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        with self._lock:
            check_fn = self._checks.get(name)

        if not check_fn:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Check not found",
            )

        start = time.time()
        try:
            result = check_fn()
            result.duration_ms = (time.time() - start) * 1000
        except Exception as e:
            result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

        with self._lock:
            self._results[name] = result

        return result

    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        with self._lock:
            check_names = list(self._checks.keys())

        results = {}
        for name in check_names:
            results[name] = self.run_check(name)

        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.run_all_checks()

        if not results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    def get_status_report(self) -> Dict[str, Any]:
        """Get a complete status report."""
        results = self.run_all_checks()
        overall = self.get_overall_status()

        return {
            "status": overall.value,
            "checks": {name: result.to_dict() for name, result in results.items()},
            "timestamp": time.time(),
        }


class AgentMonitor:
    """
    Monitors agent execution and collects metrics.

    Provides observability for agent workflows including:
    - Execution counts and durations
    - Success/failure rates
    - Tool call statistics
    - Error tracking
    """

    def __init__(self, metrics: Optional[MetricsCollector] = None):
        """
        Initialize agent monitor.

        Args:
            metrics: MetricsCollector instance (creates new if not provided)
        """
        self.metrics = metrics or MetricsCollector()
        self._active_executions: Dict[str, float] = {}
        self._errors: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._lock = threading.RLock()

    def start_execution(self, execution_id: str, task: str, agent_type: str = "unknown"):
        """Record start of agent execution."""
        with self._lock:
            self._active_executions[execution_id] = time.time()

        self.metrics.increment("agent_executions_started", labels={"agent": agent_type})
        self.metrics.set_gauge("agent_executions_active", len(self._active_executions))

        logger.debug(f"Started execution {execution_id}: {task[:50]}...")

    def end_execution(
        self,
        execution_id: str,
        success: bool,
        agent_type: str = "unknown",
        error: Optional[str] = None,
    ):
        """Record end of agent execution."""
        with self._lock:
            start_time = self._active_executions.pop(execution_id, None)

        if start_time:
            duration = time.time() - start_time
            self.metrics.record_time(
                "agent_execution_duration_seconds",
                duration,
                labels={"agent": agent_type},
            )

        status = "success" if success else "failure"
        self.metrics.increment(
            "agent_executions_completed",
            labels={"agent": agent_type, "status": status},
        )
        self.metrics.set_gauge("agent_executions_active", len(self._active_executions))

        if not success and error:
            self.record_error(execution_id, error, agent_type)

        logger.debug(f"Ended execution {execution_id}: {status}")

    def record_tool_call(self, tool_name: str, success: bool, duration: float):
        """Record a tool call."""
        status = "success" if success else "failure"
        self.metrics.increment(
            "tool_calls_total",
            labels={"tool": tool_name, "status": status},
        )
        self.metrics.record_time(
            "tool_call_duration_seconds",
            duration,
            labels={"tool": tool_name},
        )

    def record_llm_call(self, model: str, tokens: int, duration: float):
        """Record an LLM call."""
        self.metrics.increment("llm_calls_total", labels={"model": model})
        self.metrics.observe("llm_tokens_total", tokens, labels={"model": model})
        self.metrics.record_time("llm_call_duration_seconds", duration, labels={"model": model})

    def record_error(self, execution_id: str, error: str, agent_type: str = "unknown"):
        """Record an error."""
        error_entry = {
            "execution_id": execution_id,
            "error": error,
            "agent_type": agent_type,
            "timestamp": time.time(),
        }

        with self._lock:
            self._errors.append(error_entry)

        self.metrics.increment("agent_errors_total", labels={"agent": agent_type})
        logger.error(f"Agent error [{agent_type}]: {error}")

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        with self._lock:
            errors = list(self._errors)

        return errors[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "metrics": self.metrics.get_all_metrics(),
            "active_executions": len(self._active_executions),
            "recent_errors": self.get_recent_errors(5),
        }


class RateLimiter:
    """
    Rate limiter for API calls and tool execution.

    Uses token bucket algorithm.
    """

    def __init__(self, rate: float, burst: int = 10):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False otherwise
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def acquire_async(self, tokens: int = 1, timeout: float = 10.0) -> bool:
        """
        Async acquire with waiting.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait

        Returns:
            True if acquired within timeout
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self.acquire(tokens):
                return True
            await asyncio.sleep(0.1)

        return False

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class CircuitBreaker:
    """
    Circuit breaker for handling failures gracefully.

    Prevents cascading failures by stopping requests when
    failure rate is too high.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying again
            half_open_requests: Requests allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._failures = 0
        self._successes = 0
        self._state = "closed"  # closed, open, half-open
        self._last_failure_time: Optional[float] = None
        self._half_open_count = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self):
        """Check and perform state transitions."""
        if self._state == "open":
            if (
                self._last_failure_time
                and time.time() - self._last_failure_time > self.recovery_timeout
            ):
                self._state = "half-open"
                self._half_open_count = 0
                logger.info("Circuit breaker: open -> half-open")

    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        with self._lock:
            self._check_state_transition()

            if self._state == "closed":
                return True
            elif self._state == "half-open":
                if self._half_open_count < self.half_open_requests:
                    self._half_open_count += 1
                    return True
                return False
            else:  # open
                return False

    def record_success(self):
        """Record a successful request."""
        with self._lock:
            self._successes += 1

            if self._state == "half-open":
                if self._successes >= self.half_open_requests:
                    self._state = "closed"
                    self._failures = 0
                    self._successes = 0
                    logger.info("Circuit breaker: half-open -> closed")

    def record_failure(self):
        """Record a failed request."""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._state == "closed":
                if self._failures >= self.failure_threshold:
                    self._state = "open"
                    logger.warning("Circuit breaker: closed -> open")
            elif self._state == "half-open":
                self._state = "open"
                self._successes = 0
                logger.warning("Circuit breaker: half-open -> open")

    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._successes = 0
            self._half_open_count = 0
            self._last_failure_time = None


# Built-in health checks

def check_ollama_health(base_url: str = "http://localhost:11434") -> HealthCheck:
    """Check Ollama server health."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                return HealthCheck(
                    name="ollama",
                    status=HealthStatus.HEALTHY,
                    message="Ollama is running",
                )
    except urllib.error.URLError as e:
        return HealthCheck(
            name="ollama",
            status=HealthStatus.UNHEALTHY,
            message=f"Cannot connect to Ollama: {e}",
        )
    except Exception as e:
        return HealthCheck(
            name="ollama",
            status=HealthStatus.UNHEALTHY,
            message=f"Ollama check failed: {e}",
        )

    return HealthCheck(
        name="ollama",
        status=HealthStatus.UNHEALTHY,
        message="Unexpected response",
    )


def check_disk_space(path: str = ".", min_gb: float = 1.0) -> HealthCheck:
    """Check available disk space."""
    import shutil

    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)

        if free_gb >= min_gb:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.HEALTHY,
                message=f"{free_gb:.1f} GB free",
                details={"free_gb": free_gb, "total_gb": usage.total / (1024**3)},
            )
        elif free_gb >= min_gb * 0.5:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.DEGRADED,
                message=f"Low disk space: {free_gb:.1f} GB",
                details={"free_gb": free_gb},
            )
        else:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Critical: {free_gb:.1f} GB free",
                details={"free_gb": free_gb},
            )
    except Exception as e:
        return HealthCheck(
            name="disk_space",
            status=HealthStatus.UNKNOWN,
            message=f"Check failed: {e}",
        )


def check_memory_usage(max_percent: float = 90.0) -> HealthCheck:
    """Check system memory usage."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        used_percent = mem.percent

        if used_percent < max_percent * 0.8:
            status = HealthStatus.HEALTHY
        elif used_percent < max_percent:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        return HealthCheck(
            name="memory",
            status=status,
            message=f"{used_percent:.1f}% used",
            details={
                "used_percent": used_percent,
                "available_mb": mem.available / (1024**2),
            },
        )
    except ImportError:
        return HealthCheck(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message="psutil not installed",
        )
    except Exception as e:
        return HealthCheck(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message=f"Check failed: {e}",
        )


# Global instances

_global_metrics: Optional[MetricsCollector] = None
_global_health_checker: Optional[HealthChecker] = None
_global_agent_monitor: Optional[AgentMonitor] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
        # Register built-in checks
        _global_health_checker.register("ollama", check_ollama_health)
        _global_health_checker.register("disk_space", check_disk_space)
        _global_health_checker.register("memory", check_memory_usage)
    return _global_health_checker


def get_agent_monitor() -> AgentMonitor:
    """Get global agent monitor."""
    global _global_agent_monitor
    if _global_agent_monitor is None:
        _global_agent_monitor = AgentMonitor(get_metrics())
    return _global_agent_monitor
