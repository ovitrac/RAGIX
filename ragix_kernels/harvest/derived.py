"""ragix_kernels.harvest.derived — the sidecar derived store, option (b) of §2.1.1 (D-0023).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The lead ruled "go, option 2": the multiresolution layer lives **beside** saqqara's store, not inside it.
Saqqara is not touched at all — not its schema, not its seam, not its files — and this module owns the
derived area instead:

    nodes          the identity and topology of the resolution objects
    knowledge      the derived S, E(S), K, with their provenance and production metadata
    derived_edges  the resolution map and the other relations

Every row is keyed by saqqara's own identifiers (document, chunk, node) **plus the source hash it was
computed from**, so *source changed ⇒ row stale* rather than silently valid. Knowledge is **append-only and
versioned**: a new harvest writes revision r+1 and never overwrites r. Nothing here calls a model; the
model's name is recorded as a fact about a row, which is what lets the model change without touching
saqqara or destroying earlier results.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

HARVEST_SCHEMA = "harvest/0.1"
DETERMINISTIC = "none (deterministic)"

#: the sidecar as block A created it (41c9b0e), kept verbatim so a migration test can build an old one
SCHEMA_V0 = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id       TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,              -- article | section | document | family | dce | community
    parent_id     TEXT,
    doc_id        TEXT,                       -- saqqara's document id, when the node has one
    members_json  TEXT NOT NULL DEFAULT '[]', -- child node ids, chunk ids or claim ids
    source_root   TEXT NOT NULL,              -- the store this was computed from
    source_sha256 TEXT NOT NULL,              -- the source hash: the staleness guard
    run_id        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge (
    knowledge_id    TEXT PRIMARY KEY,
    subject_id      TEXT NOT NULL,            -- a saqqara chunk id, or a node id of this store
    subject_kind    TEXT NOT NULL,            -- chunk | node
    revision        INTEGER NOT NULL,         -- append-only: K^(r1), K^(r2), ...
    summary         TEXT,
    summary_map_json TEXT NOT NULL DEFAULT '[]',
    k_json          TEXT NOT NULL,
    vector          BLOB,
    vector_dim      INTEGER,
    source_root     TEXT NOT NULL,
    source_sha256   TEXT NOT NULL,
    harvest_schema  TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (subject_id, revision, harvest_schema, model, prompt_version)
);
CREATE TABLE IF NOT EXISTS derived_edges (
    edge_id      TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,               -- contains | summarises | refers_to | governed_by | ...
    source_id    TEXT NOT NULL,
    target_id    TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    run_id       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_subject ON knowledge(subject_id, revision);
CREATE INDEX IF NOT EXISTS idx_edges_source ON derived_edges(source_id, kind);
"""
#: WP §8.4, approved by the lead under D-0023 on 2026-09-11: one vector per (node, pass, embedder, abstract).
#: §8.4 proposed the key (node_id, pass, embedder); the abstract's hash is added because under the
#: append-only law a changed abstract must be a new row, never an overwrite of the old vector.
SCHEMA = SCHEMA_V0 + """
CREATE TABLE IF NOT EXISTS node_embeddings (
    embedding_id    TEXT PRIMARY KEY,
    node_id         TEXT NOT NULL,
    pass            INTEGER NOT NULL,
    embedder        TEXT NOT NULL,                -- the exact tag the executor served
    digest          TEXT,
    dim             INTEGER NOT NULL,             -- declared, never inferred from the blob
    made_on         TEXT NOT NULL,
    vector          BLOB NOT NULL,                -- float32, little-endian
    abstract_sha256 TEXT NOT NULL,                -- what was embedded
    source_sha256   TEXT,                         -- what the abstract was written from
    run_id          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (node_id, pass, embedder, abstract_sha256)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_node ON node_embeddings(node_id, pass, embedder);
"""
#: the knowledge columns §8.4 names, added to an existing sidecar by migration, never by a rebuild
KNOWLEDGE_V1 = (("pass", "INTEGER"), ("prompt_sha256", "TEXT"), ("made_on", "TEXT"),
                ("children_json", "TEXT"), ("highlights_json", "TEXT"), ("selector", "TEXT"))
SELECTORS = ("map", "register", "direction")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NodeRow:
    node_id: str
    kind: str
    parent_id: Optional[str]
    doc_id: Optional[str]
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeRow:
    subject_id: str
    subject_kind: str            # chunk | node
    k: dict[str, Any]
    summary: Optional[str] = None
    summary_map: tuple[dict[str, Any], ...] = ()
    vector: Optional[bytes] = None
    vector_dim: Optional[int] = None
    model: str = DETERMINISTIC
    prompt_version: str = "-"
    harvest_schema: str = HARVEST_SCHEMA
    pass_: Optional[int] = None          # WP §8.4: 1 is orientation, 2 is evidence — the column is `pass`
    prompt_sha256: Optional[str] = None
    made_on: Optional[str] = None
    children: tuple[str, ...] = ()
    highlights: tuple[str, ...] = ()
    selector: Optional[str] = None       # pass 2 only: which rule selected this node

    def __post_init__(self) -> None:
        if self.pass_ not in (None, 1, 2):
            raise ValueError(f"pass {self.pass_!r}: a knowledge row is pass 1 or pass 2")
        if self.selector is not None and (self.selector not in SELECTORS or self.pass_ != 2):
            raise ValueError(f"selector {self.selector!r}: pass 2 only, one of {SELECTORS}")
        if self.subject_kind not in ("chunk", "node"):
            raise ValueError(f"subject_kind {self.subject_kind!r} outside vocabulary")
        if not self.subject_id:
            raise ValueError("knowledge without a subject")
        for entry in self.summary_map:
            if not entry.get("children"):
                raise ValueError("a summary sentence maps to no child: refused (the provenance tree is the point)")


class DerivedStore:
    """The sidecar. It never opens saqqara's database, and saqqara never learns it exists."""

    def __init__(self, path: str, source_root: str, source_sha256: str, run_id: str) -> None:
        self.path = path
        self.source_root = source_root
        self.source_sha256 = source_sha256
        self.run_id = run_id
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add §8.4's knowledge columns to a sidecar that predates them. Idempotent; nothing is rebuilt."""
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(knowledge)")}
        for name, sqltype in KNOWLEDGE_V1:
            if name not in have:
                self.conn.execute(f'ALTER TABLE knowledge ADD COLUMN "{name}" {sqltype}')

    # ------------------------------------------------------------------ write
    def write_node(self, node: NodeRow, created_at: str) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO nodes(node_id, kind, parent_id, doc_id, members_json, source_root, "
            "source_sha256, run_id, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (node.node_id, node.kind, node.parent_id, node.doc_id,
             json.dumps(list(node.members), ensure_ascii=False, sort_keys=True),
             self.source_root, self.source_sha256, self.run_id, created_at))
        return node.node_id

    def write_knowledge(self, row: KnowledgeRow, created_at: str) -> str:
        """Append: the next revision for this subject, never an overwrite."""
        cur = self.conn.execute("SELECT COALESCE(MAX(revision), 0) FROM knowledge WHERE subject_id = ?",
                                (row.subject_id,))
        revision = int(cur.fetchone()[0]) + 1
        knowledge_id = sha256_text(f"{row.subject_id}|{revision}|{row.harvest_schema}|{row.model}|"
                                   f"{row.prompt_version}")[:32]
        self.conn.execute(
            "INSERT INTO knowledge(knowledge_id, subject_id, subject_kind, revision, summary, summary_map_json, "
            "k_json, vector, vector_dim, source_root, source_sha256, harvest_schema, model, prompt_version, "
            'run_id, created_at, "pass", prompt_sha256, made_on, children_json, highlights_json, selector) '
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (knowledge_id, row.subject_id, row.subject_kind, revision, row.summary,
             json.dumps(list(row.summary_map), ensure_ascii=False, sort_keys=True),
             json.dumps(row.k, ensure_ascii=False, sort_keys=True), row.vector, row.vector_dim,
             self.source_root, self.source_sha256, row.harvest_schema, row.model, row.prompt_version,
             self.run_id, created_at, row.pass_, row.prompt_sha256, row.made_on,
             json.dumps(list(row.children), ensure_ascii=False), json.dumps(list(row.highlights), ensure_ascii=False),
             row.selector))
        return knowledge_id

    def write_embedding(self, *, node_id: str, pass_: int, embedder: str, digest: Optional[str], dim: int,
                        made_on: str, vector: bytes, abstract_sha256: str, source_sha256: Optional[str],
                        created_at: str) -> bool:
        """Append one vector. The same abstract under the same embedder is written once; a second call with
        other bytes is refused, because it means the embedder did not reproduce itself."""
        if len(vector) != dim * 4:
            raise ValueError(f"vector has {len(vector)} bytes, {dim * 4} expected for {dim} float32")
        key = (node_id, pass_, embedder, abstract_sha256)
        found = self.conn.execute("SELECT vector FROM node_embeddings WHERE node_id=? AND pass=? AND embedder=? "
                                  "AND abstract_sha256=?", key).fetchone()
        if found is not None:
            if bytes(found[0]) != vector:
                raise ValueError(f"{node_id}: the same abstract under the same embedder, and a vector that is not "
                                 "byte-identical — the embedder did not reproduce itself")
            return False
        self.conn.execute(
            "INSERT INTO node_embeddings(embedding_id, node_id, pass, embedder, digest, dim, made_on, vector, "
            "abstract_sha256, source_sha256, run_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (sha256_text("|".join(map(str, key)))[:32], node_id, pass_, embedder, digest, dim, made_on, vector,
             abstract_sha256, source_sha256, self.run_id, created_at))
        return True

    def write_edge(self, kind: str, source_id: str, target_id: str, payload: Optional[dict] = None) -> str:
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
        edge_id = sha256_text(f"{kind}|{source_id}|{target_id}|{payload_json}")[:32]
        self.conn.execute(
            # append-only: re-deriving an edge that exists changes nothing, and the run that first wrote it
            # stays its run. OR REPLACE silently re-stamped it with every later run.
            "INSERT OR IGNORE INTO derived_edges(edge_id, kind, source_id, target_id, payload_json, run_id) "
            "VALUES (?,?,?,?,?,?)", (edge_id, kind, source_id, target_id, payload_json, self.run_id))
        return edge_id

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.close()

    # ------------------------------------------------------------------- read
    def node(self, node_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        return dict(row) if row else None

    def latest_knowledge(self, subject_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM knowledge WHERE subject_id = ? ORDER BY revision DESC LIMIT 1", (subject_id,)).fetchone()
        return dict(row) if row else None

    def children(self, node_id: str, kind: str = "contains") -> list[str]:
        return [r["target_id"] for r in self.conn.execute(
            "SELECT target_id FROM derived_edges WHERE source_id = ? AND kind = ? ORDER BY target_id",
            (node_id, kind))]

    def stale(self, subject_id: str, current_source_sha256: str) -> Optional[bool]:
        """True when the source moved under a knowledge row. None when nothing is known of the subject."""
        row = self.latest_knowledge(subject_id)
        if row is None:
            return None
        return row["source_sha256"] != current_source_sha256

    def counts(self) -> dict[str, int]:
        return {t: int(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                for t in ("nodes", "knowledge", "derived_edges")}
