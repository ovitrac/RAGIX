"""
SQLite-backed translation memory (TM) for the KOAS-Translate family.

Every segment carries its source hash, raw and final translations, QA report,
model identifier, prompt version, and timestamps. Re-runs are idempotent: the
``upsert_*`` writers only touch the relevant columns and the ``needs_*``
predicates let stages skip work already on record for the current source. When a
segment's source text changes, its downstream artefacts (translation / QA /
final) are invalidated so the pipeline retranslates.

Ported verbatim (schema unchanged) from the standalone translation pipeline's
``store.py`` and decoupled from its global ``config`` — :func:`connect` takes an
explicit database path so a kernel can point it at its workspace. The
``lang_pair`` / ``glossary_version`` columns from the design are deferred to P3
(generalization) to keep P1 byte-for-byte reproducible.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    segment_id        TEXT    PRIMARY KEY,
    chapter           TEXT,
    section           TEXT,
    order_idx         INTEGER NOT NULL,
    source_hash       TEXT    NOT NULL,
    source_text       TEXT    NOT NULL,
    protected_map     TEXT    NOT NULL,          -- JSON: { "⟦P0001⟧": "$\\nabla u = 0$", ... }
    raw_translation   TEXT,
    qa_report         TEXT,                      -- JSON
    final_translation TEXT,
    model             TEXT,
    prompt_version    TEXT,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS segments_order ON segments(order_idx);
CREATE INDEX IF NOT EXISTS segments_chapter ON segments(chapter, order_idx);

CREATE TABLE IF NOT EXISTS chapter_revisions (
    chapter         TEXT PRIMARY KEY,
    revised_text    TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_hash(text: str) -> str:
    """SHA256 of the segment source — the key for idempotent re-runs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open (creating if needed) the TM database at *path*, schema ensured.

    Commits on clean exit, always closes. WAL journaling for concurrent reads.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Segment writers
# ---------------------------------------------------------------------------


def upsert_source_segment(
    conn: sqlite3.Connection,
    *,
    segment_id: str,
    chapter: str | None,
    section: str | None,
    order_idx: int,
    source_text: str,
    protected_map: dict[str, str],
) -> None:
    """Insert or refresh a source segment.

    If the stored ``source_hash`` matches, translations/QA are preserved (only
    navigational metadata is refreshed). Otherwise stale downstream artefacts are
    cleared so the pipeline retranslates.
    """
    h = source_hash(source_text)
    now = _now()
    row = conn.execute(
        "SELECT source_hash FROM segments WHERE segment_id = ?", (segment_id,)
    ).fetchone()
    if row is None:
        conn.execute(
            """INSERT INTO segments
               (segment_id, chapter, section, order_idx, source_hash, source_text,
                protected_map, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (segment_id, chapter, section, order_idx, h, source_text,
             json.dumps(protected_map, ensure_ascii=False), now, now),
        )
    elif row["source_hash"] != h:
        # Source changed → invalidate everything downstream.
        conn.execute(
            """UPDATE segments
               SET chapter=?, section=?, order_idx=?, source_hash=?, source_text=?,
                   protected_map=?, raw_translation=NULL, qa_report=NULL,
                   final_translation=NULL, model=NULL, prompt_version=NULL,
                   updated_at=?
               WHERE segment_id=?""",
            (chapter, section, order_idx, h, source_text,
             json.dumps(protected_map, ensure_ascii=False), now, segment_id),
        )
    else:
        # Source identical → only refresh navigational metadata.
        conn.execute(
            """UPDATE segments
               SET chapter=?, section=?, order_idx=?, updated_at=?
               WHERE segment_id=?""",
            (chapter, section, order_idx, now, segment_id),
        )


def save_translation(
    conn: sqlite3.Connection,
    segment_id: str,
    raw_translation: str,
    *,
    model: str,
    prompt_version: str,
) -> None:
    conn.execute(
        """UPDATE segments
           SET raw_translation=?, model=?, prompt_version=?, updated_at=?
           WHERE segment_id=?""",
        (raw_translation, model, prompt_version, _now(), segment_id),
    )


def save_qa(conn: sqlite3.Connection, segment_id: str, qa_report: dict[str, Any]) -> None:
    conn.execute(
        "UPDATE segments SET qa_report=?, updated_at=? WHERE segment_id=?",
        (json.dumps(qa_report, ensure_ascii=False), _now(), segment_id),
    )


def save_final(conn: sqlite3.Connection, segment_id: str, final_translation: str) -> None:
    conn.execute(
        "UPDATE segments SET final_translation=?, updated_at=? WHERE segment_id=?",
        (final_translation, _now(), segment_id),
    )


# ---------------------------------------------------------------------------
# Readers / predicates
# ---------------------------------------------------------------------------


def iter_segments(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM segments ORDER BY order_idx ASC")
    yield from cur


def get_segment(conn: sqlite3.Connection, segment_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
    ).fetchone()


def needs_translation(row: sqlite3.Row) -> bool:
    return row["raw_translation"] is None


def needs_qa(row: sqlite3.Row) -> bool:
    return row["raw_translation"] is not None and row["qa_report"] is None


def needs_final(row: sqlite3.Row) -> bool:
    return row["raw_translation"] is not None and row["final_translation"] is None


# ---------------------------------------------------------------------------
# Chapter revisions
# ---------------------------------------------------------------------------


def save_chapter_revision(
    conn: sqlite3.Connection, chapter: str, revised_text: str,
    *, model: str, prompt_version: str,
) -> None:
    conn.execute(
        """INSERT INTO chapter_revisions
                (chapter, revised_text, model, prompt_version, created_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(chapter) DO UPDATE SET
             revised_text=excluded.revised_text,
             model=excluded.model,
             prompt_version=excluded.prompt_version,
             created_at=excluded.created_at""",
        (chapter, revised_text, model, prompt_version, _now()),
    )


def get_chapter_revision(conn: sqlite3.Connection, chapter: str) -> str | None:
    row = conn.execute(
        "SELECT revised_text FROM chapter_revisions WHERE chapter = ?", (chapter,)
    ).fetchone()
    return row["revised_text"] if row else None
