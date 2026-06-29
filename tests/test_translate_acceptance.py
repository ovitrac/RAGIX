"""
KOAS-Translate acceptance — port-fidelity against the live original pipeline.

This is the P1 acceptance check: the kernelized stages must reproduce the
original translation pipeline's output **exactly** on real input. It compares the
deterministic kernels (segment, rebuild) to the original ``pipeline/*`` functions
run on the *same* source / translation memory.

Copyright & data policy
-----------------------
The source material (a copyrighted book) is **never** committed and never appears
in this file. The test is gated on the ``RAGIX_TRANSLATE_FIXTURE`` environment
variable pointing at a *local* translation project (containing both the original
``pipeline/`` code and ``out/snapshot-30pages/``). When it is unset or absent —
i.e. on the public repo, CI, or any other machine — every test here is skipped.
Nothing book-derived is read into the repo; comparisons happen in memory.

    RAGIX_TRANSLATE_FIXTURE=~/Documents/Adservio/draft/translation \
        python -m pytest tests/test_translate_acceptance.py -v

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

_FIX = os.environ.get("RAGIX_TRANSLATE_FIXTURE")
_FIXTURE = Path(_FIX).expanduser() if _FIX else None


def _have_fixture() -> bool:
    return (
        _FIXTURE is not None
        and (_FIXTURE / "pipeline" / "segment.py").exists()
        and (_FIXTURE / "out" / "snapshot-30pages" / "source.md").exists()
        and (_FIXTURE / "out" / "snapshot-30pages" / "tm.sqlite").exists()
    )


pytestmark = pytest.mark.skipif(
    not _have_fixture(),
    reason="set RAGIX_TRANSLATE_FIXTURE to a local translation project "
           "(copyrighted book data — never committed)",
)


@pytest.fixture(scope="module")
def orig():
    """Import the original (local-only) pipeline modules for comparison."""
    sys.path.insert(0, str(_FIXTURE))
    try:
        import pipeline.config as config
        import pipeline.segment as segment
        import pipeline.rebuild as rebuild
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not import original pipeline: {e}")
    return config, segment, rebuild


SNAP = lambda name: _FIXTURE / "out" / "snapshot-30pages" / name


def test_segment_reproduces_original(orig):
    config, segment, _ = orig
    from ragix_kernels.translate.segment import chunk_markdown

    md = SNAP("source.md").read_text(encoding="utf-8")
    o = segment.chunk_markdown(md)
    m = chunk_markdown(md, target_words=config.CHUNK_TARGET_WORDS,
                       max_words=config.CHUNK_MAX_WORDS)
    assert len(m) == len(o), f"chunk count differs: {len(m)} vs {len(o)}"
    for i, (a, b) in enumerate(zip(o, m)):
        assert b.segment_id == a.segment_id, f"seg {i} id"
        assert b.chapter == a.chapter and b.order_idx == a.order_idx, f"seg {i} meta"
        assert b.source_text == a.source_text, f"seg {i} source_text"
        assert b.protected_map == a.protected_map, f"seg {i} protected_map"


def test_rebuild_reproduces_original(orig, tmp_path, monkeypatch):
    config, _, rebuild = orig
    from ragix_kernels.translate import tm_store
    from ragix_kernels.translate.rebuild import assemble

    work = tmp_path / "tm.sqlite"
    shutil.copy(SNAP("tm.sqlite"), work)
    orig_out = tmp_path / "final_orig.md"

    # Point the original at our scratch copy and run it without CLI args.
    monkeypatch.setattr(config, "TM_SQLITE", work)
    monkeypatch.setattr(config, "FINAL_MD", orig_out)
    monkeypatch.setattr(sys, "argv", ["rebuild"])
    rc = rebuild.main()
    assert rc in (0, 1)                       # 1 = wrote file but reported problems
    expected = orig_out.read_text(encoding="utf-8")

    with tm_store.connect(work) as conn:
        mine, _problems, _stats = assemble(conn)
    assert mine == expected, "rebuild output diverges from the original pipeline"
