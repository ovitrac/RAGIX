"""
Tests for the KOAS-Translate draft kernel (ragix_kernels/translate/draft).

Uses a deterministic stub backend (no Ollama) to exercise the Pass A flow:
translate-and-save, idempotent skip, --limit, empty/exception handling,
continuity context + chapter reset, and glossary injection.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.translate import tm_store
from ragix_kernels.translate.draft import TranslateDraftKernel, PROMPT_VERSION


class StubBackend:
    """Records prompts; echoes the source chunk as 'FR[…]' (tokens preserved).

    ``empty_on`` / ``raise_on`` are sets of 0-based call indices to simulate the
    silent-empty and exception failure modes.
    """

    def __init__(self, empty_on=None, raise_on=None):
        self.prompts = []
        self.empty_on = set(empty_on or ())
        self.raise_on = set(raise_on or ())

    def __call__(self, prompt: str) -> str:
        idx = len(self.prompts)
        self.prompts.append(prompt)
        if idx in self.raise_on:
            raise RuntimeError("backend boom")
        if idx in self.empty_on:
            return ""
        src = prompt.split("Source to translate:\n", 1)[1].strip()
        return f"FR[{src}]"


@pytest.fixture
def tm(tmp_path):
    db = tmp_path / "out" / "tm.sqlite"
    segs = [
        ("seg-00000", "A", 0, "Alpha sentence."),
        ("seg-00001", "A", 1, "Beta sentence."),
        ("seg-00002", "B", 2, "Gamma sentence."),
    ]
    with tm_store.connect(db) as conn:
        for sid, ch, order, text in segs:
            tm_store.upsert_source_segment(
                conn, segment_id=sid, chapter=ch, section=None,
                order_idx=order, source_text=text, protected_map={})
    return tmp_path, db


def _run(root, db, stub, **cfg):
    kernel = TranslateDraftKernel()
    kernel.backend = stub                      # injected off-config (config stays JSON)
    inp = KernelInput(
        workspace=root,
        config={"tm_path": str(db), **cfg},
        dependencies={"translate_segment": db},
    )
    return kernel.run(inp)


class TestDraftKernel:
    def test_translates_all_and_saves(self, tm):
        root, db = tm
        out = _run(root, db, StubBackend())
        assert out.success, out.errors
        assert out.data["n_translated"] == 3
        assert out.data["n_failures"] == 0
        with tm_store.connect(db) as conn:
            rows = list(tm_store.iter_segments(conn))
        assert all(r["raw_translation"].startswith("FR[") for r in rows)
        assert all(r["prompt_version"] == PROMPT_VERSION for r in rows)
        assert all(r["model"] == "granite4.1-translate" for r in rows)

    def test_idempotent_skips_cached(self, tm):
        root, db = tm
        _run(root, db, StubBackend())
        out2 = _run(root, db, StubBackend())
        assert out2.data["n_translated"] == 0
        assert out2.data["n_cached"] == 3

    def test_limit_caps_new_chunks(self, tm):
        root, db = tm
        out = _run(root, db, StubBackend(), limit=1)
        assert out.data["n_translated"] == 1
        with tm_store.connect(db) as conn:
            untranslated = [r for r in tm_store.iter_segments(conn)
                            if tm_store.needs_translation(r)]
        assert len(untranslated) == 2

    def test_empty_output_is_skipped_not_saved(self, tm):
        root, db = tm
        out = _run(root, db, StubBackend(empty_on={0}))
        assert out.data["n_failures"] == 1
        assert out.data["failures"][0]["reason"] == "empty"
        with tm_store.connect(db) as conn:
            row = tm_store.get_segment(conn, "seg-00000")
        assert row["raw_translation"] is None      # empty never committed

    def test_backend_exception_is_recorded_and_continues(self, tm):
        root, db = tm
        out = _run(root, db, StubBackend(raise_on={0}))
        assert out.data["n_failures"] == 1
        assert out.data["n_translated"] == 2       # other two still done
        assert "boom" in out.data["failures"][0]["reason"]

    def test_continuity_resets_across_chapters(self, tm):
        root, db = tm
        stub = StubBackend()
        _run(root, db, stub)
        # prompt 0 (seg A0): no previous context
        assert "(none)" in stub.prompts[0].split("Source to translate:")[0]
        # prompt 1 (seg A1, same chapter): carries seg A0's FR as context
        ctx1 = stub.prompts[1].split("Source to translate:")[0]
        assert "FR[Alpha sentence.]" in ctx1
        # prompt 2 (seg B2, new chapter): context reset
        ctx2 = stub.prompts[2].split("Source to translate:")[0]
        assert "FR[Beta sentence.]" not in ctx2
        assert "(none)" in ctx2

    def test_lang_pair_recorded_default(self, tm):
        root, db = tm
        out = _run(root, db, StubBackend())
        assert out.data["lang_pair"] == "en-fr"        # default, surfaced in output

    def test_glossary_is_injected(self, tmp_path, tm):
        root, db = tm
        gloss = tmp_path / "glossary.csv"
        gloss.write_text("EN,FR,rule\nbehavior,comportement,preferred\n", encoding="utf-8")
        stub = StubBackend()
        _run(root, db, stub, glossary_path=str(gloss))
        assert "behavior → comportement" in stub.prompts[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
