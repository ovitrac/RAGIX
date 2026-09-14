"""The sidecar gains the columns the lead approved under D-0023, and a table for abstract embeddings.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

`knowledge` gains pass, prompt_sha256, made_on, children_json, highlights_json and selector (WP §8.4);
a `node_embeddings` table holds one vector per (node, pass, embedder, abstract), append-only like the
rest. An existing sidecar is migrated on open, never rebuilt, and opening it twice changes nothing.
"""
from __future__ import annotations

import sqlite3
import struct

import pytest

from ragix_kernels.harvest import derived
from ragix_kernels.harvest.derived import DerivedStore, KnowledgeRow

NEW = {"pass", "prompt_sha256", "made_on", "children_json", "highlights_json", "selector"}


def _store(path):
    return DerivedStore(str(path), source_root="/store", source_sha256="a" * 64, run_id="test")


def _cols(path, table):
    return {r[1] for r in sqlite3.connect(path).execute(f"pragma table_info({table})")}


def _vec(dim, seed=0.0):
    return struct.pack(f"<{dim}f", *[seed + i / dim for i in range(dim)])


def test_a_fresh_sidecar_has_the_columns_and_the_table(tmp_path):
    _store(tmp_path / "s.db").close()
    assert NEW <= _cols(tmp_path / "s.db", "knowledge")
    assert {"node_id", "pass", "embedder", "dim", "made_on", "vector", "abstract_sha256"} <= _cols(tmp_path / "s.db", "node_embeddings")


def test_an_existing_sidecar_is_migrated_not_rebuilt(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(derived.SCHEMA_V0)                     # the schema as block A created it
    con.execute("INSERT INTO knowledge(knowledge_id, subject_id, subject_kind, revision, k_json, source_root, "
                "source_sha256, harvest_schema, model, prompt_version, run_id, created_at) "
                "VALUES ('k1','n1','node',1,'{}','/s','x','h','m','p','r','t')")
    con.commit(); con.close()
    for _ in range(2):                                       # twice: the migration is idempotent
        _store(path).close()
    assert NEW <= _cols(path, "knowledge")
    assert sqlite3.connect(path).execute("select count(*) from knowledge").fetchone()[0] == 1


def test_the_new_knowledge_fields_are_written_and_held_to_their_vocabulary(tmp_path):
    s = _store(tmp_path / "s.db")
    s.write_knowledge(KnowledgeRow(subject_id="n1", subject_kind="node", k={}, pass_=1, prompt_sha256="b" * 64,
                                   made_on="executor", children=("c1", "c2"), highlights=("maintenance",)),
                      created_at="t")
    s.commit()
    row = s.latest_knowledge("n1")
    assert row["pass"] == 1 and row["made_on"] == "executor" and '"c1"' in row["children_json"]
    with pytest.raises(ValueError):
        KnowledgeRow(subject_id="n1", subject_kind="node", k={}, pass_=3)
    with pytest.raises(ValueError):
        KnowledgeRow(subject_id="n1", subject_kind="node", k={}, pass_=2, selector="guess")


def test_an_embedding_is_append_only_and_guards_its_own_determinism(tmp_path):
    s = _store(tmp_path / "s.db")
    kw = dict(node_id="n1", pass_=1, embedder="snowflake-arctic-embed2:latest", digest="5de93a84837d", dim=8,
              made_on="executor", abstract_sha256="c" * 64, source_sha256="d" * 64, created_at="t")
    assert s.write_embedding(vector=_vec(8), **kw) is True
    assert s.write_embedding(vector=_vec(8), **kw) is False            # the same abstract, the same bytes: nothing
    assert s.write_embedding(vector=_vec(8, 1.0), **{**kw, "abstract_sha256": "e" * 64}) is True   # a new abstract: a new row
    with pytest.raises(ValueError, match="not byte-identical"):
        s.write_embedding(vector=_vec(8, 2.0), **kw)                   # same key, other bytes: non-determinism
    with pytest.raises(ValueError, match="bytes"):
        s.write_embedding(vector=_vec(7), **{**kw, "abstract_sha256": "f" * 64})
