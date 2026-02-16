"""
Per-session rate limiter using a token bucket algorithm.

Enforces per-session call rate limits and per-turn proposal caps for the
memory MCP server.  Thread-safe via a single ``threading.Lock``.

Token bucket refill
-------------------
On every ``check_rate`` call the bucket is topped-up based on elapsed wall
time since the last refill:

    refill = elapsed_seconds * (calls_per_minute / 60)

The bucket is capped at ``calls_per_minute * burst_multiplier`` to allow
short bursts while still converging to the steady-state rate.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from ragix_core.memory.config import RateLimitConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SessionBucket:
    """Internal per-session token bucket state."""
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)
    proposals_this_turn: int = 0
    current_turn: int = 0


@dataclass
class RateLimitResult:
    """Outcome of a rate-limit check."""
    allowed: bool
    reason: str
    remaining: int
    retry_after_ms: int = 0


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter with per-session state.

    Parameters
    ----------
    config : RateLimitConfig or None
        If *None*, a default :class:`RateLimitConfig` is used.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        self._config = config or RateLimitConfig()
        self._sessions: Dict[str, SessionBucket] = {}
        self._lock = threading.Lock()

    # -- internal helpers ---------------------------------------------------

    def _get_bucket(self, session_id: str) -> SessionBucket:
        """Return the bucket for *session_id*, creating one if necessary.

        Must be called while holding ``self._lock``.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionBucket(
                tokens=float(self._config.calls_per_minute),
                last_refill=time.monotonic(),
            )
        return self._sessions[session_id]

    def _refill(self, bucket: SessionBucket) -> None:
        """Top-up *bucket* based on elapsed wall time.

        Must be called while holding ``self._lock``.
        """
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        if elapsed <= 0:
            return
        rate = self._config.calls_per_minute / 60.0  # tokens per second
        bucket.tokens += elapsed * rate
        max_tokens = self._config.calls_per_minute * self._config.burst_multiplier
        if bucket.tokens > max_tokens:
            bucket.tokens = max_tokens
        bucket.last_refill = now

    # -- public API ---------------------------------------------------------

    def check_rate(self, session_id: str) -> RateLimitResult:
        """Check whether *session_id* is within the call-rate budget.

        The bucket is refilled before the check.  No tokens are consumed;
        use :meth:`consume` to actually deduct tokens.

        Returns
        -------
        RateLimitResult
            ``allowed=True`` when at least one token is available.
        """
        if not self._config.enabled:
            return RateLimitResult(
                allowed=True,
                reason="rate_limiting_disabled",
                remaining=self._config.calls_per_minute,
            )

        with self._lock:
            bucket = self._get_bucket(session_id)
            self._refill(bucket)

            if bucket.tokens >= 1.0:
                return RateLimitResult(
                    allowed=True,
                    reason="ok",
                    remaining=int(bucket.tokens),
                )

            # Compute wait time until one token is available.
            rate = self._config.calls_per_minute / 60.0
            deficit = 1.0 - bucket.tokens
            retry_seconds = deficit / rate if rate > 0 else 60.0
            return RateLimitResult(
                allowed=False,
                reason="rate_limit_exceeded",
                remaining=0,
                retry_after_ms=int(retry_seconds * 1000) + 1,
            )

    def check_proposal_limit(
        self,
        session_id: str,
        turn: int,
        count: int = 1,
    ) -> RateLimitResult:
        """Check whether *count* proposals can be made in the current *turn*.

        If *turn* differs from the stored ``current_turn`` the per-turn
        counter is automatically reset.

        Returns
        -------
        RateLimitResult
            ``allowed=True`` when the proposal budget has room.
        """
        if not self._config.enabled:
            return RateLimitResult(
                allowed=True,
                reason="rate_limiting_disabled",
                remaining=self._config.proposals_per_turn,
            )

        with self._lock:
            bucket = self._get_bucket(session_id)

            # Auto-reset on new turn.
            if bucket.current_turn != turn:
                bucket.current_turn = turn
                bucket.proposals_this_turn = 0

            headroom = self._config.proposals_per_turn - bucket.proposals_this_turn
            if count <= headroom:
                bucket.proposals_this_turn += count
                return RateLimitResult(
                    allowed=True,
                    reason="ok",
                    remaining=headroom - count,
                )

            return RateLimitResult(
                allowed=False,
                reason="proposal_limit_exceeded",
                remaining=max(headroom, 0),
            )

    def consume(self, session_id: str, n: int = 1) -> bool:
        """Consume *n* tokens from the bucket.

        The bucket is refilled before consumption.

        Returns
        -------
        bool
            ``True`` if there were enough tokens and they were consumed.
        """
        if not self._config.enabled:
            return True

        with self._lock:
            bucket = self._get_bucket(session_id)
            self._refill(bucket)

            if bucket.tokens >= n:
                bucket.tokens -= n
                return True
            return False

    def reset_session(self, session_id: str) -> None:
        """Clear all state for *session_id*."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def get_status(self, session_id: str) -> Dict:
        """Return a snapshot of the current bucket for *session_id*.

        Returns
        -------
        dict
            Keys: ``tokens``, ``max_tokens``, ``proposals_this_turn``,
            ``proposals_limit``, ``current_turn``, ``enabled``.
        """
        with self._lock:
            bucket = self._get_bucket(session_id)
            self._refill(bucket)
            max_tokens = self._config.calls_per_minute * self._config.burst_multiplier
            return {
                "tokens": round(bucket.tokens, 2),
                "max_tokens": max_tokens,
                "proposals_this_turn": bucket.proposals_this_turn,
                "proposals_limit": self._config.proposals_per_turn,
                "current_turn": bucket.current_turn,
                "enabled": self._config.enabled,
            }
