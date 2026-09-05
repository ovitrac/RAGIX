"""
saqqara.store.sqlite — the one-file hybrid store.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

Gate K7.1, K7.6, K7.12, K7.13.

One SQLite file holds the trees, the objects and edges read from them, the chunks
cut from them, and the vectors computed over those chunks. Keeping embeddings in
the same file as the text they describe is the whole point: a vector store beside
a document store is two things that can disagree about what exists, and the
disagreement is discovered at query time by a user.

**Delete is trash, not loss.** `delete_document` marks the row and parks its
embeddings in `embeddings_trash`, so a restore returns the same vectors rather
than recomputing them — recomputation is not restoration, it is a second opinion
wearing the first one's name. `purge` is opt-in and counted, never a side effect
of deleting.

**Every drop is counted.** The store keeps a drop trace in the same spirit as the
builder's: what it removed, why, and how many. A purge that reports nothing is
indistinguishable from a purge that did nothing.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any, Iterable, Optional

# Reading a stored tree reconstructs its nodes, and Node refuses a kind the
# registry does not know. The adapters register their kinds at import time, so a
# process that only imports the store — the CLI, an example script — could not
# deserialise a tree it had written itself: KindError: unregistered kind: 'page'.
# The test suite could not see it, because importing an adapter anywhere registers
# for everything after. Same shape as the provider-registration defect in ports.py.
from .. import analyzers as _analyzers  # noqa: F401  (registers "block")
from .. import builder as _builder      # noqa: F401  (registers cell, shape, page, …)
from ..model import CANONICAL_JSON, Tree
from .records import (ChunkRecord, DocumentRecord, EdgeRecord, EmbeddingRecord,
                      EmbeddingRefusalRecord, Hit, ObjectRecord)
from .ports import register_store

__all__ = ["SqliteDocumentStore", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

#: FTS5 tokenizer. Diacritics are folded because a query typed without accents
#: must find text written with them; this corpus is not only English.
FTS_TOKENIZER = "unicode61 remove_diacritics 2"

_SCHEMA = f"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    corpus          TEXT NOT NULL,
    doc_class       TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    source_sha256   TEXT NOT NULL,
    meta_json       TEXT NOT NULL DEFAULT '{{}}',
    tree_json       TEXT,
    digested_at     TEXT,
    kernel          TEXT NOT NULL,
    kernel_version  TEXT NOT NULL,
    trashed         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS documents_source ON documents(source_sha256);

CREATE TABLE IF NOT EXISTS document_paths (
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    seen_at     TEXT,
    PRIMARY KEY (doc_id, source_path)
);

CREATE TABLE IF NOT EXISTS objects (
    doc_id     TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    node_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    page       INTEGER,
    bbox_json  TEXT,
    caption    TEXT,
    asset_ref  TEXT,
    cells_json TEXT,
    meta_json  TEXT NOT NULL DEFAULT '{{}}',
    PRIMARY KEY (doc_id, node_id)
);

CREATE TABLE IF NOT EXISTS edges (
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    src    TEXT NOT NULL,
    dst    TEXT NOT NULL,
    type   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS edges_doc ON edges(doc_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    seq               INTEGER NOT NULL,
    text              TEXT NOT NULL,
    level             INTEGER NOT NULL,
    parent_id         TEXT,
    section_path_json TEXT NOT NULL DEFAULT '[]',
    node_ids_json     TEXT NOT NULL,
    pages_json        TEXT NOT NULL DEFAULT '[]',
    lang              TEXT,
    object_refs_json  TEXT NOT NULL DEFAULT '[]',
    meta_json         TEXT NOT NULL DEFAULT '{{}}'
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id   TEXT NOT NULL,
    model      TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    indexed_at TEXT,
    PRIMARY KEY (chunk_id, model)
);

CREATE TABLE IF NOT EXISTS embedding_refusals (
    chunk_id    TEXT NOT NULL,
    model       TEXT NOT NULL,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    reason      TEXT NOT NULL,
    signals_json TEXT NOT NULL DEFAULT '{{}}',
    refused_at  TEXT,
    PRIMARY KEY (chunk_id, model)
);
CREATE INDEX IF NOT EXISTS embedding_refusals_doc ON embedding_refusals(doc_id);

CREATE TABLE IF NOT EXISTS embeddings_trash (
    chunk_id   TEXT NOT NULL,
    model      TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    indexed_at TEXT,
    doc_id     TEXT NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    section,
    tokenize = '{FTS_TOKENIZER}'
);
"""


def _pack(vector: tuple[float, ...]) -> bytes:
    """float32 little-endian, so a stored vector is the same bytes everywhere."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // 4}f", blob)


class SqliteDocumentStore:
    """The document store, in one file. Implements `DocumentStore`."""

    def __init__(self, path: str = ".ragix/saqqara.db", corpus: str = "default") -> None:
        self.path = Path(path)
        self.corpus = corpus
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._db.commit()
        #: What the store removed, and why. Read by the CLI and the kernel report.
        self.drops: list[dict[str, Any]] = []
        #: Dense indexes, one per model, built on demand from the rows above.
        self._retrievers: dict[str, Any] = {}

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "SqliteDocumentStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------- documents

    def upsert_document(self, doc: DocumentRecord,
                        objects: Iterable[ObjectRecord] = (),
                        edges: Iterable[EdgeRecord] = ()) -> None:
        tree_json = json.dumps(doc.tree.to_dict(), **CANONICAL_JSON) if doc.tree else None
        self._db.execute(
            """INSERT INTO documents
                 (doc_id, corpus, doc_class, source_path, source_sha256, meta_json,
                  tree_json, digested_at, kernel, kernel_version, trashed)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 corpus=excluded.corpus, doc_class=excluded.doc_class,
                 source_path=excluded.source_path, meta_json=excluded.meta_json,
                 tree_json=excluded.tree_json, digested_at=excluded.digested_at,
                 kernel=excluded.kernel, kernel_version=excluded.kernel_version,
                 trashed=excluded.trashed""",
            (doc.doc_id, doc.corpus or self.corpus, doc.doc_class, doc.source_path,
             doc.source_sha256, json.dumps(doc.meta, **CANONICAL_JSON), tree_json,
             doc.digested_at, doc.kernel, doc.kernel_version, int(doc.trashed)),
        )
        # Every path the same bytes have been seen at. The document is one row;
        # where it was found is a history, and losing it loses the answer to
        # "where did this come from" for every copy but the last.
        self._db.execute(
            "INSERT OR IGNORE INTO document_paths(doc_id, source_path, seen_at) VALUES (?,?,?)",
            (doc.doc_id, doc.source_path, doc.digested_at),
        )
        self._db.execute("DELETE FROM objects WHERE doc_id=?", (doc.doc_id,))
        for obj in objects:
            self._db.execute(
                """INSERT INTO objects
                     (doc_id, node_id, kind, page, bbox_json, caption, asset_ref,
                      cells_json, meta_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (obj.doc_id, obj.node_id, obj.kind, obj.page,
                 json.dumps(obj.bbox) if obj.bbox is not None else None, obj.caption,
                 obj.asset_ref, json.dumps(obj.cells) if obj.cells is not None else None,
                 json.dumps(obj.meta, **CANONICAL_JSON)),
            )
        self._db.execute("DELETE FROM edges WHERE doc_id=?", (doc.doc_id,))
        for edge in edges:
            self._db.execute(
                "INSERT INTO edges(doc_id, src, dst, type) VALUES (?,?,?,?)",
                (edge.doc_id, edge.src, edge.dst, edge.type),
            )
        self._db.commit()

    def _document_from_row(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"], corpus=row["corpus"], doc_class=row["doc_class"],
            source_path=row["source_path"], source_sha256=row["source_sha256"],
            kernel=row["kernel"], kernel_version=row["kernel_version"],
            tree=Tree.from_dict(json.loads(row["tree_json"])) if row["tree_json"] else None,
            meta=json.loads(row["meta_json"]), digested_at=row["digested_at"],
            trashed=bool(row["trashed"]),
        )

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        row = self._db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self, include_trashed: bool = False) -> list[DocumentRecord]:
        sql = "SELECT * FROM documents"
        if not include_trashed:
            sql += " WHERE trashed=0"
        sql += " ORDER BY doc_id"
        return [self._document_from_row(r) for r in self._db.execute(sql)]

    def find_doc_by_source(self, sha256: str) -> Optional[DocumentRecord]:
        row = self._db.execute(
            "SELECT * FROM documents WHERE source_sha256=?", (sha256,)
        ).fetchone()
        return self._document_from_row(row) if row else None

    def source_paths(self, doc_id: str) -> list[str]:
        """Every path these bytes have been seen at, oldest insertion first."""
        return [r["source_path"] for r in self._db.execute(
            "SELECT source_path FROM document_paths WHERE doc_id=? ORDER BY source_path",
            (doc_id,))]

    def delete_document(self, doc_id: str, purge: bool = False) -> dict[str, int]:
        """Trash by default; purge only when asked, and counted either way."""
        if self.get_document(doc_id) is None:
            return {"documents": 0, "chunks": 0, "embeddings": 0, "parked": 0}

        chunk_ids = [r["chunk_id"] for r in self._db.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))]

        if not purge:
            parked = 0
            if chunk_ids:
                marks = ",".join("?" * len(chunk_ids))
                parked = self._db.execute(
                    f"""INSERT OR REPLACE INTO embeddings_trash
                          (chunk_id, model, dimensions, vector, indexed_at, doc_id)
                        SELECT chunk_id, model, dimensions, vector, indexed_at, ?
                          FROM embeddings WHERE chunk_id IN ({marks})""",
                    (doc_id, *chunk_ids),
                ).rowcount
                self._db.execute(
                    f"DELETE FROM embeddings WHERE chunk_id IN ({marks})", chunk_ids)
            self._db.execute("UPDATE documents SET trashed=1 WHERE doc_id=?", (doc_id,))
            self._db.commit()
            counted = {"documents": 1, "chunks": 0, "embeddings": 0, "parked": max(parked, 0)}
            self.drops.append({"reason": "document-trashed", "doc_id": doc_id, **counted})
            return counted

        removed_embeddings = 0
        if chunk_ids:
            marks = ",".join("?" * len(chunk_ids))
            removed_embeddings = self._db.execute(
                f"DELETE FROM embeddings WHERE chunk_id IN ({marks})", chunk_ids).rowcount
            self._db.execute(
                f"DELETE FROM embeddings_trash WHERE chunk_id IN ({marks})", chunk_ids)
            self._db.execute(
                f"DELETE FROM chunks_fts WHERE chunk_id IN ({marks})", chunk_ids)
        removed_chunks = self._db.execute(
            "DELETE FROM chunks WHERE doc_id=?", (doc_id,)).rowcount
        self._db.execute("DELETE FROM objects WHERE doc_id=?", (doc_id,))
        self._db.execute("DELETE FROM edges WHERE doc_id=?", (doc_id,))
        self._db.execute("DELETE FROM document_paths WHERE doc_id=?", (doc_id,))
        self._db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        self._db.commit()
        counted = {"documents": 1, "chunks": max(removed_chunks, 0),
                   "embeddings": max(removed_embeddings, 0), "parked": 0}
        self.drops.append({"reason": "document-purged", "doc_id": doc_id, **counted})
        return counted

    def restore_document(self, doc_id: str) -> bool:
        """Bring a trashed document back, with the vectors it had — not new ones."""
        row = self._db.execute(
            "SELECT trashed FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if row is None or not row["trashed"]:
            return False
        self._db.execute(
            """INSERT OR REPLACE INTO embeddings
                 (chunk_id, model, dimensions, vector, indexed_at)
               SELECT chunk_id, model, dimensions, vector, indexed_at
                 FROM embeddings_trash WHERE doc_id=?""",
            (doc_id,),
        )
        self._db.execute("DELETE FROM embeddings_trash WHERE doc_id=?", (doc_id,))
        self._db.execute("UPDATE documents SET trashed=0 WHERE doc_id=?", (doc_id,))
        self._db.commit()
        return True

    # -------------------------------------------------------------- chunks

    def replace_chunks(self, doc_id: str, chunks: Iterable[ChunkRecord]) -> dict[str, int]:
        """Write only what changed, so an unchanged tree costs nothing (K7.5)."""
        incoming = {c.chunk_id: c for c in chunks}
        existing = {r["chunk_id"] for r in self._db.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))}

        to_insert = [c for cid, c in incoming.items() if cid not in existing]
        to_delete = sorted(existing - set(incoming))

        if to_insert or to_delete:
            # The cached index describes rows that just changed, so it is dropped
            # rather than patched: an index patched in parallel with its source is
            # a second answer waiting to diverge.
            self._retrievers.clear()

        for chunk in to_insert:
            self._db.execute(
                """INSERT INTO chunks
                     (chunk_id, doc_id, seq, text, level, parent_id, section_path_json,
                      node_ids_json, pages_json, lang, object_refs_json, meta_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (chunk.chunk_id, chunk.doc_id, chunk.seq, chunk.text, chunk.level,
                 chunk.parent_id, json.dumps(chunk.section_path),
                 json.dumps(chunk.node_ids), json.dumps(chunk.pages), chunk.lang,
                 json.dumps(chunk.object_refs), json.dumps(chunk.meta, **CANONICAL_JSON)),
            )
            self._db.execute(
                "INSERT INTO chunks_fts(chunk_id, text, section) VALUES (?,?,?)",
                (chunk.chunk_id, chunk.text, " / ".join(chunk.section_path)),
            )
        if to_delete:
            marks = ",".join("?" * len(to_delete))
            self._db.execute(f"DELETE FROM chunks WHERE chunk_id IN ({marks})", to_delete)
            self._db.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({marks})", to_delete)
            self._db.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({marks})", to_delete)
            self.drops.append({"reason": "chunk-superseded", "doc_id": doc_id,
                               "chunks": len(to_delete)})
        self._db.commit()
        return {"inserted": len(to_insert), "deleted": len(to_delete),
                "unchanged": len(incoming) - len(to_insert)}

    def _chunk_from_row(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"], seq=row["seq"],
            text=row["text"], level=row["level"], parent_id=row["parent_id"],
            section_path=json.loads(row["section_path_json"]),
            node_ids=json.loads(row["node_ids_json"]),
            pages=json.loads(row["pages_json"]), lang=row["lang"],
            object_refs=json.loads(row["object_refs_json"]),
            meta=json.loads(row["meta_json"]),
        )

    def get_chunks(self, doc_id: Optional[str] = None) -> list[ChunkRecord]:
        if doc_id is None:
            rows = self._db.execute("SELECT * FROM chunks ORDER BY doc_id, seq")
        else:
            rows = self._db.execute(
                "SELECT * FROM chunks WHERE doc_id=? ORDER BY seq", (doc_id,))
        return [self._chunk_from_row(r) for r in rows]

    def get_objects(self, doc_id: str) -> list[ObjectRecord]:
        return [ObjectRecord(
            doc_id=r["doc_id"], node_id=r["node_id"], kind=r["kind"], page=r["page"],
            bbox=json.loads(r["bbox_json"]) if r["bbox_json"] else None,
            caption=r["caption"], asset_ref=r["asset_ref"],
            cells=json.loads(r["cells_json"]) if r["cells_json"] else None,
            meta=json.loads(r["meta_json"]),
        ) for r in self._db.execute(
            "SELECT * FROM objects WHERE doc_id=? ORDER BY node_id", (doc_id,))]

    def get_edges(self, doc_id: str) -> list[EdgeRecord]:
        return [EdgeRecord(doc_id=r["doc_id"], src=r["src"], dst=r["dst"], type=r["type"])
                for r in self._db.execute(
                    "SELECT * FROM edges WHERE doc_id=? ORDER BY src, dst, type", (doc_id,))]

    # ---------------------------------------------------------- embeddings

    def existing_embeddings(self, chunk_ids: Iterable[str], model: str) -> set[str]:
        ids = list(chunk_ids)
        if not ids:
            return set()
        marks = ",".join("?" * len(ids))
        return {r["chunk_id"] for r in self._db.execute(
            f"SELECT chunk_id FROM embeddings WHERE model=? AND chunk_id IN ({marks})",
            (model, *ids))}

    def upsert_embeddings(self, records: Iterable[EmbeddingRecord]) -> int:
        written = 0
        for rec in records:
            self._db.execute(
                """INSERT INTO embeddings (chunk_id, model, dimensions, vector, indexed_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(chunk_id, model) DO UPDATE SET
                     dimensions=excluded.dimensions, vector=excluded.vector,
                     indexed_at=excluded.indexed_at""",
                (rec.chunk_id, rec.model, rec.dimensions, _pack(rec.vector), rec.indexed_at),
            )
            written += 1
        self._db.commit()
        return written

    def replace_embedding_refusals(
        self, doc_id: str, records: Iterable[EmbeddingRefusalRecord]
    ) -> int:
        """This document's refusals, as of this run. Always called, empty included.

        Replacement rather than accumulation, and unconditional rather than only
        when there is something to write: a document that stopped refusing must
        stop being listed, and a register that only ever grows describes every run
        except the last one.
        """
        rows = list(records)
        self._db.execute("DELETE FROM embedding_refusals WHERE doc_id=?", (doc_id,))
        for record in rows:
            self._db.execute(
                """INSERT INTO embedding_refusals
                     (chunk_id, model, doc_id, reason, signals_json, refused_at)
                   VALUES (?,?,?,?,?,?)""",
                (record.chunk_id, record.model, record.doc_id, record.reason,
                 json.dumps(record.signals, **CANONICAL_JSON), record.refused_at),
            )
        self._db.commit()
        return len(rows)

    def get_embedding_refusals(
        self, doc_id: Optional[str] = None
    ) -> list[EmbeddingRefusalRecord]:
        """The register the store holds, whole or for one document."""
        sql = ("SELECT * FROM embedding_refusals"
               + (" WHERE doc_id=?" if doc_id else "")
               + " ORDER BY doc_id, chunk_id")
        rows = self._db.execute(sql, (doc_id,) if doc_id else ())
        return [EmbeddingRefusalRecord(
            chunk_id=r["chunk_id"], doc_id=r["doc_id"], model=r["model"],
            reason=r["reason"], signals=json.loads(r["signals_json"]),
            refused_at=r["refused_at"],
        ) for r in rows]

    def get_embeddings(self, model: str) -> list[EmbeddingRecord]:
        """Live vectors for one model, excluding anything whose document is trashed."""
        return [EmbeddingRecord(
            chunk_id=r["chunk_id"], model=r["model"], dimensions=r["dimensions"],
            vector=_unpack(r["vector"]), indexed_at=r["indexed_at"],
        ) for r in self._db.execute(
            """SELECT e.* FROM embeddings e
                 JOIN chunks c ON c.chunk_id = e.chunk_id
                 JOIN documents d ON d.doc_id = c.doc_id
                WHERE e.model=? AND d.trashed=0
                ORDER BY e.chunk_id""", (model,))]

    # ------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        def count(sql: str, *args: Any) -> int:
            return int(self._db.execute(sql, args).fetchone()[0])

        models = [r["model"] for r in self._db.execute(
            "SELECT DISTINCT model FROM embeddings ORDER BY model")]
        return {
            "path": str(self.path),
            "corpus": self.corpus,
            "schema_version": SCHEMA_VERSION,
            "documents": count("SELECT COUNT(*) FROM documents WHERE trashed=0"),
            "trashed": count("SELECT COUNT(*) FROM documents WHERE trashed=1"),
            "objects": count("SELECT COUNT(*) FROM objects"),
            "edges": count("SELECT COUNT(*) FROM edges"),
            "chunks": count("SELECT COUNT(*) FROM chunks"),
            "embeddings": count("SELECT COUNT(*) FROM embeddings"),
            "embeddings_parked": count("SELECT COUNT(*) FROM embeddings_trash"),
            # Holdings and refusals are read together or not at all: a vector count
            # alone reads as completeness, and this lane is complete only when the
            # second number is zero.
            "embedding_refusals": count("SELECT COUNT(*) FROM embedding_refusals"),
            "models": models,
            "drops": list(self.drops),
        }

    # ---------------------------------------------------------- retrieval

    def lexical_search(self, query: str, top_k: int) -> list[Hit]:
        """The FTS5 lane. Ranks are 1-based and lane-local, never cross-lane.

        A trashed document's chunks are excluded here rather than filtered by the
        caller: a store that returns rows it considers deleted has two answers to
        the question of what it holds.
        """
        if not query.strip():
            return []
        rows = self._db.execute(
            """SELECT c.* FROM chunks_fts f
                 JOIN chunks c    ON c.chunk_id = f.chunk_id
                 JOIN documents d ON d.doc_id   = c.doc_id
                WHERE chunks_fts MATCH ? AND d.trashed = 0
                ORDER BY bm25(chunks_fts)
                LIMIT ?""",
            (query, int(top_k)),
        ).fetchall()
        return [Hit(chunk=self._chunk_from_row(r), lexical_rank=i)
                for i, r in enumerate(rows, 1)]

    def search(self, vector: Iterable[float], top_k: int, model: str) -> list[Hit]:
        """The dense lane, over vectors read from this database.

        The index is built by retrieve.py and cached per model on this store. It is
        a cache and never an artefact: it is rebuilt from the rows here, so it
        cannot outlive or contradict them.
        """
        from .retrieve import Retriever

        retriever = self._retrievers.get(model)
        if retriever is None:
            retriever = Retriever(self, model=model)
            self._retrievers[model] = retriever
        return retriever.dense(vector, top_k)

    def invalidate_index(self, model: Optional[str] = None) -> None:
        """Drop a cached index, or all of them. The next search rebuilds from here."""
        for name, retriever in list(self._retrievers.items()):
            if model is None or name == model:
                retriever.invalidate()
                self._retrievers.pop(name, None)


register_store("sqlite", SqliteDocumentStore)
