"""Synthetic document stores for the harvest family's tests — built through the store's own API.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The harvest reads what a `saqqara` store holds in a known shape: per document, level-0 leaves; a
level-1 roll-up with no parent whose text is the leaves joined by newlines; and, for a long roll-up,
level-1 windows whose parent is the roll-up and whose `meta.part.span` gives their character range in
it. A workbook holds one roll-up per sheet. This module builds exactly that shape, with invented
French text, through `SqliteDocumentStore` — the schema is the store's, never a copy of it — so each
test states the text it needs instead of depending on what a reader happens to produce.

Nothing here is read from a corpus. Every document, sentence and file name is written for the test.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ragix_kernels.saqqara.store.records import ChunkRecord, DocumentRecord
from ragix_kernels.saqqara.store.sqlite import SqliteDocumentStore


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Sheet:
    """One roll-up: its leaves, and optionally windows given as character spans of the roll-up."""

    leaves: list[str]
    windows: list[tuple[int, int]] = field(default_factory=list)
    pages: Optional[list[list[int]]] = None          # one list per leaf; None leaves pages empty
    section: list[str] = field(default_factory=list)


@dataclass
class Doc:
    name: str                                         # a short handle used by the tests
    source_path: str                                  # the path the store records, invented
    sheets: list[Sheet]
    doc_class: str = "pdf"
    tree: Optional[dict[str, Any]] = None             # written as tree_json when given


@dataclass
class Built:
    doc_id: str
    rollups: list[str]
    leaves: list[list[str]]
    windows: list[list[str]]
    texts: list[str]

    @property
    def rollup(self) -> str:
        return self.rollups[0]


def build_store(path: Path, docs: list[Doc]) -> dict[str, Built]:
    """Write `docs` into a new store at `path`; return, per handle, the ids the store holds."""
    store = SqliteDocumentStore(str(path))
    out: dict[str, Built] = {}
    for doc in docs:
        doc_id = sha(f"{doc.name}|{doc.source_path}")
        store.upsert_document(DocumentRecord(
            doc_id=doc_id, corpus="default", doc_class=doc.doc_class, source_path=doc.source_path,
            source_sha256=doc_id, kernel="saqqara", kernel_version="synthetic"))
        chunks: list[ChunkRecord] = []
        built = Built(doc_id, [], [], [], [])
        seq = 0
        node = 0
        for s_index, sheet in enumerate(doc.sheets):
            roll_text = "\n".join(sheet.leaves)
            roll_id = sha(f"{doc_id}|rollup|{s_index}")
            leaf_ids, leaf_nodes = [], []
            for l_index, text in enumerate(sheet.leaves):
                leaf_id = sha(f"{doc_id}|leaf|{s_index}|{l_index}")
                pages = sheet.pages[l_index] if sheet.pages else []
                chunks.append(ChunkRecord(chunk_id=leaf_id, doc_id=doc_id, seq=seq, text=text, level=0,
                                          node_ids=[str(node)], parent_id=roll_id,
                                          section_path=list(sheet.section), pages=list(pages)))
                leaf_ids.append(leaf_id)
                leaf_nodes.append(str(node))
                seq += 1
                node += 1
            all_pages = sorted({p for ps in (sheet.pages or []) for p in ps})
            chunks.append(ChunkRecord(chunk_id=roll_id, doc_id=doc_id, seq=seq, text=roll_text, level=1,
                                      node_ids=leaf_nodes or [str(node)], parent_id=None,
                                      section_path=list(sheet.section), pages=all_pages))
            seq += 1
            window_ids = []
            for w_index, (c0, c1) in enumerate(sheet.windows):
                window_id = sha(f"{doc_id}|window|{s_index}|{w_index}")
                chunks.append(ChunkRecord(chunk_id=window_id, doc_id=doc_id, seq=seq,
                                          text=roll_text[c0:c1], level=1, node_ids=leaf_nodes or ["0"],
                                          parent_id=roll_id, section_path=list(sheet.section),
                                          pages=all_pages, meta={"part": {"span": [c0, c1], "index": w_index}}))
                window_ids.append(window_id)
                seq += 1
            built.rollups.append(roll_id)
            built.leaves.append(leaf_ids)
            built.windows.append(window_ids)
            built.texts.append(roll_text)
        store.replace_chunks(doc_id, chunks)
        out[doc.name] = built
    store.close()
    if any(doc.tree is not None for doc in docs):
        con = sqlite3.connect(str(path))
        for doc in docs:
            if doc.tree is not None:
                con.execute("update documents set tree_json=? where doc_id=?",
                            (json.dumps(doc.tree, ensure_ascii=False), out[doc.name].doc_id))
        con.commit()
        con.close()
    return out


def windows_of(text: str, size: int) -> list[tuple[int, int]]:
    """Consecutive character spans of `text`, cut at the first newline after each `size` characters."""
    spans, start = [], 0
    while start < len(text):
        end = text.find("\n", start + size)
        end = len(text) if end == -1 else end
        spans.append((start, end))
        start = end + 1
    return spans


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
