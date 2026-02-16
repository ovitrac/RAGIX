"""
Workspace Router — Named Workspace Management for Memory MCP Server.

Tracks named workspaces, each mapping to a (scope, corpus_id, description)
tuple.  Workspaces provide a lightweight namespace layer so that a single
MCP session can address multiple corpora or scopes without re-configuring
the server.

Persistence uses the same SQLite database as the memory store (shared
connection, separate table ``memory_workspaces``).

Thread safety: all mutable operations are serialized via ``threading.Lock``.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from ragix_core.memory.types import _now_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL (added to existing memory database)
# ---------------------------------------------------------------------------

_WORKSPACE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_workspaces (
    name        TEXT PRIMARY KEY,
    scope       TEXT NOT NULL DEFAULT 'project',
    corpus_id   TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceInfo:
    """Describes a single named workspace."""

    name: str
    scope: str = "project"
    corpus_id: str = ""
    description: str = ""
    created_at: str = field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

# Name of the built-in default workspace (cannot be removed).
_DEFAULT_NAME = "default"


class WorkspaceRouter:
    """
    Named workspace registry backed by SQLite.

    Each workspace maps a human-friendly *name* to a ``(scope, corpus_id)``
    pair consumed by the memory recall / search subsystem.

    A ``"default"`` workspace always exists (scope="project", corpus_id="").

    Thread-safe: all read and write operations acquire ``self._lock``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """
        Initialise the router using an existing SQLite connection.

        Parameters
        ----------
        conn:
            A ``sqlite3.Connection`` that is already opened in WAL mode
            with ``row_factory = sqlite3.Row``.  Typically the same
            connection owned by :class:`MemoryStore`.
        """
        self._conn = conn
        self._lock = threading.Lock()
        self._ensure_schema()
        self._ensure_default()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create the workspace table if it does not yet exist."""
        with self._lock:
            self._conn.executescript(_WORKSPACE_SCHEMA_SQL)
            self._conn.commit()

    def _ensure_default(self) -> None:
        """Guarantee the ``"default"`` workspace is present."""
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM memory_workspaces WHERE name = ?",
                (_DEFAULT_NAME,),
            ).fetchone()
            if row is None:
                now = _now_iso()
                self._conn.execute(
                    """INSERT INTO memory_workspaces
                       (name, scope, corpus_id, description, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (_DEFAULT_NAME, "project", "", "Default workspace", now),
                )
                self._conn.commit()
                logger.debug("Created default workspace")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        scope: Optional[str] = None,
        corpus_id: Optional[str] = None,
        description: str = "",
    ) -> WorkspaceInfo:
        """
        Register (or update) a named workspace.

        Parameters
        ----------
        name:
            Unique workspace name.
        scope:
            Memory scope (e.g. ``"project"``, ``"audit"``).
            Defaults to *name* when ``None``.
        corpus_id:
            Corpus identifier for cross-corpus recall.
            Defaults to ``""`` when ``None``.
        description:
            Free-text description shown by :meth:`list_workspaces`.

        Returns
        -------
        WorkspaceInfo
            The workspace record (newly created or updated).
        """
        if not name or not isinstance(name, str):
            raise ValueError("Workspace name must be a non-empty string")

        resolved_scope = scope if scope is not None else name
        resolved_corpus = corpus_id if corpus_id is not None else ""
        now = _now_iso()

        with self._lock:
            existing = self._conn.execute(
                "SELECT name FROM memory_workspaces WHERE name = ?",
                (name,),
            ).fetchone()

            if existing is not None:
                # Update in place
                self._conn.execute(
                    """UPDATE memory_workspaces
                       SET scope = ?, corpus_id = ?, description = ?
                       WHERE name = ?""",
                    (resolved_scope, resolved_corpus, description, name),
                )
                # Preserve original created_at
                row = self._conn.execute(
                    "SELECT created_at FROM memory_workspaces WHERE name = ?",
                    (name,),
                ).fetchone()
                created = row["created_at"] if row else now
            else:
                self._conn.execute(
                    """INSERT INTO memory_workspaces
                       (name, scope, corpus_id, description, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (name, resolved_scope, resolved_corpus, description, now),
                )
                created = now

            self._conn.commit()

        info = WorkspaceInfo(
            name=name,
            scope=resolved_scope,
            corpus_id=resolved_corpus,
            description=description,
            created_at=created,
        )
        logger.info("Registered workspace %r → scope=%s, corpus=%s", name, resolved_scope, resolved_corpus)
        return info

    def resolve(self, workspace: Optional[str] = None) -> Tuple[str, str]:
        """
        Resolve a workspace name to ``(scope, corpus_id)``.

        Parameters
        ----------
        workspace:
            Workspace name.  If ``None`` or empty, resolves to the
            ``"default"`` workspace.

        Returns
        -------
        tuple[str, str]
            ``(scope, corpus_id)`` for the resolved workspace.

        Raises
        ------
        KeyError
            If *workspace* is not registered.
        """
        target = workspace if workspace else _DEFAULT_NAME

        with self._lock:
            row = self._conn.execute(
                "SELECT scope, corpus_id FROM memory_workspaces WHERE name = ?",
                (target,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Unknown workspace: {target!r}")

        return (row["scope"], row["corpus_id"])

    def list_workspaces(self) -> List[Dict[str, str]]:
        """
        Return a list of all registered workspaces as plain dicts.

        Returns
        -------
        list[dict]
            Each dict contains keys: ``name``, ``scope``, ``corpus_id``,
            ``description``, ``created_at``.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_workspaces ORDER BY created_at"
            ).fetchall()

        return [
            {
                "name": r["name"],
                "scope": r["scope"],
                "corpus_id": r["corpus_id"],
                "description": r["description"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def get_workspace(self, name: str) -> Optional[WorkspaceInfo]:
        """
        Retrieve a single workspace record.

        Parameters
        ----------
        name:
            Workspace name to look up.

        Returns
        -------
        WorkspaceInfo or None
            ``None`` when no workspace with that name exists.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_workspaces WHERE name = ?",
                (name,),
            ).fetchone()

        if row is None:
            return None

        return WorkspaceInfo(
            name=row["name"],
            scope=row["scope"],
            corpus_id=row["corpus_id"],
            description=row["description"],
            created_at=row["created_at"],
        )

    def remove(self, name: str) -> bool:
        """
        Remove a workspace by name.

        The ``"default"`` workspace is protected and cannot be removed.

        Parameters
        ----------
        name:
            Workspace to delete.

        Returns
        -------
        bool
            ``True`` if a workspace was actually deleted, ``False`` if
            *name* was ``"default"`` (rejected) or did not exist.
        """
        if name == _DEFAULT_NAME:
            logger.warning("Cannot remove the default workspace")
            return False

        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memory_workspaces WHERE name = ?",
                (name,),
            )
            self._conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Removed workspace %r", name)
        return deleted
