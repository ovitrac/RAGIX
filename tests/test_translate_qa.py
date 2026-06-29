"""
Tests for the KOAS-Translate qa kernel (ragix_kernels/translate/qa).

Deterministic stub backend exercises Pass B: ok→promote, revise→hold,
prose-wrapped JSON, unparseable→synthetic revise, backend exception→synthetic
revise, idempotent skip, glossary injection.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.translate import tm_store
from ragix_kernels.translate.qa import TranslateQAKernel

OK = '{"status": "ok", "issues": []}'
REVISE = '{"status": "revise", "issues": [{"type": "omission", "source": "x", "problem": "p", "suggested_fix": "f"}]}'


class QAStub:
    def __init__(self, responses):
        self.responses = responses        # list (per-call) or single value/Exception
        self.prompts = []

    def __call__(self, prompt):
        i = len(self.prompts)
        self.prompts.append(prompt)
        r = self.responses[i] if isinstance(self.responses, list) else self.responses
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def tm(tmp_path):
    """TM with two translated segments awaiting QA."""
    db = tmp_path / "out" / "tm.sqlite"
    with tm_store.connect(db) as conn:
        for sid, order in [("seg-00000", 0), ("seg-00001", 1)]:
            tm_store.upsert_source_segment(
                conn, segment_id=sid, chapter="A", section=None, order_idx=order,
                source_text=f"Source {order}.", protected_map={})
            tm_store.save_translation(conn, sid, f"FR {order}.",
                                      model="m", prompt_version="v2")
    return tmp_path, db


def _run(root, db, stub, **cfg):
    kernel = TranslateQAKernel()
    kernel.backend = stub
    inp = KernelInput(workspace=root, config={"tm_path": str(db), **cfg},
                      dependencies={"translate_draft": db})
    return kernel.run(inp)


class TestQAKernel:
    def test_ok_promotes_raw_to_final(self, tm):
        root, db = tm
        out = _run(root, db, QAStub(OK))
        assert out.data["n_ok"] == 2 and out.data["n_revise"] == 0
        with tm_store.connect(db) as conn:
            rows = list(tm_store.iter_segments(conn))
        assert all(r["final_translation"] == r["raw_translation"] for r in rows)

    def test_revise_holds_final_and_counts_issue(self, tm):
        root, db = tm
        out = _run(root, db, QAStub(REVISE))
        assert out.data["n_revise"] == 2
        assert out.data["issues_by_type"].get("omission") == 2
        with tm_store.connect(db) as conn:
            rows = list(tm_store.iter_segments(conn))
        assert all(r["final_translation"] is None for r in rows)

    def test_prose_wrapped_json_parsed(self, tm):
        root, db = tm
        out = _run(root, db, QAStub([f"Here: {OK} done", f"```json\n{OK}\n```"]))
        assert out.data["n_ok"] == 2

    def test_unparseable_becomes_synthetic_revise(self, tm):
        root, db = tm
        out = _run(root, db, QAStub("definitely not json"))
        assert out.data["n_revise"] == 2
        with tm_store.connect(db) as conn:
            row = tm_store.get_segment(conn, "seg-00000")
        assert row["qa_report"] is not None             # recorded, not lost
        assert row["final_translation"] is None

    def test_backend_exception_becomes_synthetic_revise(self, tm):
        root, db = tm
        out = _run(root, db, QAStub(RuntimeError("timeout")))
        assert out.data["n_revise"] == 2
        assert out.success                               # whole run did not abort

    def test_idempotent_skips_cached_qa(self, tm):
        root, db = tm
        _run(root, db, QAStub(OK))
        out2 = _run(root, db, QAStub(OK))
        assert out2.data["n_skipped"] == 2
        assert out2.data["n_ok"] == 0

    def test_glossary_injected_into_prompt(self, tmp_path, tm):
        root, db = tm
        gloss = tmp_path / "g.csv"
        gloss.write_text("EN,FR,rule\nbias,biais,preferred\n", encoding="utf-8")
        stub = QAStub(OK)
        _run(root, db, stub, glossary_path=str(gloss))
        assert "bias → biais" in stub.prompts[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
