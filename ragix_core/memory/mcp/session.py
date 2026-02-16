"""
Session State Management for Memory MCP Server.

Tracks per-session state: turn counter, scope, project identity.
Session state is stored in the same SQLite database as memory items
(lightweight, no external dependencies).

Session identity:
    session_id = <project_id>:<conversation_id>

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ragix_core.memory.types import _generate_id, _now_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL (added to existing memory database)
# ---------------------------------------------------------------------------

_SESSION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_sessions (
    session_id   TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL DEFAULT '',
    scope        TEXT NOT NULL DEFAULT 'project',
    turn_count   INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""


@dataclass
class SessionState:
    """Represents the state of a single memory session."""

    session_id: str = ""
    project_id: str = ""
    scope: str = "project"
    turn_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def conversation_id(self) -> str:
        """Extract conversation_id from session_id (after the colon)."""
        parts = self.session_id.split(":", 1)
        return parts[1] if len(parts) > 1 else self.session_id


class SessionManager:
    """
    Manages per-session state in the memory SQLite database.

    Thread-safe via explicit lock (shares database with MemoryStore).
    """

    def __init__(self, conn: sqlite3.Connection):
        """Initialize session manager using an existing SQLite connection."""
        self._conn = conn
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create session table if it doesn't exist."""
        with self._lock:
            self._conn.executescript(_SESSION_SCHEMA_SQL)
            self._conn.commit()

    def get_or_create(
        self,
        session_id: str,
        scope: str = "project",
    ) -> SessionState:
        """Get existing session or create a new one."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()

            if row is not None:
                return SessionState(
                    session_id=row["session_id"],
                    project_id=row["project_id"],
                    scope=row["scope"],
                    turn_count=row["turn_count"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    metadata=_safe_json_loads(row["metadata_json"]),
                )

            # Parse project_id from session_id
            parts = session_id.split(":", 1)
            project_id = parts[0] if len(parts) > 1 else ""

            now = _now_iso()
            state = SessionState(
                session_id=session_id,
                project_id=project_id,
                scope=scope,
                turn_count=0,
                created_at=now,
                updated_at=now,
            )
            self._conn.execute(
                """INSERT INTO memory_sessions
                   (session_id, project_id, scope, turn_count,
                    created_at, updated_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    state.session_id, state.project_id, state.scope,
                    state.turn_count, state.created_at, state.updated_at,
                    "{}",
                ),
            )
            self._conn.commit()
            return state

    def increment_turn(self, session_id: str) -> int:
        """Increment turn counter and return the new value."""
        with self._lock:
            now = _now_iso()
            self._conn.execute(
                """UPDATE memory_sessions
                   SET turn_count = turn_count + 1, updated_at = ?
                   WHERE session_id = ?""",
                (now, session_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT turn_count FROM memory_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            return row["turn_count"] if row else 0

    def get_turn_count(self, session_id: str) -> int:
        """Get current turn count for a session."""
        with self._lock:
            row = self._conn.execute(
                "SELECT turn_count FROM memory_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            return row["turn_count"] if row else 0

    def update_metadata(
        self, session_id: str, metadata: Dict[str, Any]
    ) -> None:
        """Merge metadata keys into session state."""
        import json

        with self._lock:
            row = self._conn.execute(
                "SELECT metadata_json FROM memory_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                return

            existing = _safe_json_loads(row["metadata_json"])
            existing.update(metadata)
            now = _now_iso()
            self._conn.execute(
                """UPDATE memory_sessions
                   SET metadata_json = ?, updated_at = ?
                   WHERE session_id = ?""",
                (json.dumps(existing, ensure_ascii=False), now, session_id),
            )
            self._conn.commit()

    def list_sessions(self, scope: Optional[str] = None) -> list:
        """List all sessions, optionally filtered by scope."""
        with self._lock:
            if scope:
                rows = self._conn.execute(
                    "SELECT * FROM memory_sessions WHERE scope=? ORDER BY updated_at DESC",
                    (scope,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memory_sessions ORDER BY updated_at DESC"
                ).fetchall()

            return [
                {
                    "session_id": r["session_id"],
                    "project_id": r["project_id"],
                    "scope": r["scope"],
                    "turn_count": r["turn_count"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Parse JSON string, returning empty dict on failure."""
    import json
    try:
        return json.loads(text) if text else {}
    except (json.JSONDecodeError, TypeError):
        return {}
