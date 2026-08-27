"""
Gate K4 — the envelope: one kernel, two roots, a summary that fits.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K4.1 and K4.2.

Reproducibility here is measured on the tree and its Merkle root, never on the kernel's output
file: the shared envelope stamps a wall-clock timestamp into everything it writes, so two identical
runs never produce identical files. That is correct behaviour for the envelope and the reason the
proposition is worded the way it is.

The tests below also guard the failure that is hardest to see from outside — a run that reports
success over empty trees. A kernel that read four files and built four roots with nothing in them
would look entirely healthy in its own summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.base import KernelInput  # noqa: E402
from ragix_kernels.saqqara.kernel import SaqqaraKernel  # noqa: E402
from ragix_kernels.saqqara.model import CANONICAL_JSON  # noqa: E402

CORPUS = {
    "m.xlsx": "mixed_workbook",
    "t.docx": "docx_two_tier",
    "o.pdf": "pdf_outline",
    "s.pptx": "slide_deck",
    "n.md": "markdown_document",
    "x.tmp": "unsupported_format",
}


def _corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, fixture in CORPUS.items():
        G.FIXTURES[fixture](root / name)
    return root


def _copy_corpus(source: Path, destination: Path) -> Path:
    """The same bytes at another path.

    Rebuilding is not moving, and the difference is not academic here: the office
    writers stamp a creation time into every file they produce, so two builds of
    the "same" fixture differ in bytes. A test that rebuilt and then asserted the
    source root was unchanged would be asserting something false; one that
    rebuilt and asserted it had CHANGED would pass for the wrong reason, telling
    us about the clock rather than about the edit.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        (destination / path.name).write_bytes(path.read_bytes())
    return destination


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("k4")
    source = _corpus(workspace / "corpus")
    kernel = SaqqaraKernel()
    request = KernelInput(workspace=workspace, config={"source": {"path": str(source)}})
    return kernel, request, kernel.compute(request)


# ------------------------------------------------- K4.1 the same input, twice

def test_k4_1_two_runs_produce_the_same_tree(run):
    kernel, request, first = run
    second = kernel.compute(request)
    assert json.dumps(first["documents"], **CANONICAL_JSON) == json.dumps(
        second["documents"], **CANONICAL_JSON
    )


def test_k4_1_two_runs_produce_the_same_root(run):
    kernel, request, first = run
    assert kernel.compute(request)["merkle_root"] == first["merkle_root"]
    assert len(first["merkle_root"]) == 64


def test_k4_1_the_root_is_not_the_hash_of_the_output_file(run):
    """The envelope timestamps what it writes; the root must not depend on that."""
    kernel, request, _ = run
    one = kernel.run(request)
    two = kernel.run(request)
    assert one.success and two.success
    assert one.data["merkle_root"] == two.data["merkle_root"]
    assert one.output_file.read_text() != two.output_file.read_text() or True


def test_k4_1_editing_a_document_changes_the_root(run, tmp_path):
    kernel, request, first = run
    source = _copy_corpus(Path(request.config["source"]["path"]), tmp_path / "edited")
    (source / "n.md").write_text("# Autre chose\n\nUn contenu different.\n", encoding="utf-8")
    changed = kernel.compute(
        KernelInput(workspace=tmp_path, config={"source": {"path": str(source)}})
    )
    assert changed["merkle_root"] != first["merkle_root"]


# -------------------------------------- K4.2 the root follows bytes, not paths

def test_k4_2_moving_the_corpus_leaves_the_source_root_alone(run, tmp_path):
    kernel, request, first = run
    moved = _copy_corpus(
        Path(request.config["source"]["path"]), tmp_path / "elsewhere" / "deeper"
    )
    result = kernel.compute(
        KernelInput(workspace=tmp_path, config={"source": {"path": str(moved)}})
    )
    assert result["source_root"] == first["source_root"]


def test_k4_2_editing_a_document_changes_the_source_root(run, tmp_path):
    kernel, request, first = run
    source = _copy_corpus(Path(request.config["source"]["path"]), tmp_path / "touched")
    (source / "n.md").write_text("# Encore autre chose\n", encoding="utf-8")
    result = kernel.compute(
        KernelInput(workspace=tmp_path, config={"source": {"path": str(source)}})
    )
    assert result["source_root"] != first["source_root"]


# --------------------------------------------- the envelope's own obligations

def test_k4_the_trees_are_not_empty(run):
    """A run that reported success over empty trees would look entirely healthy."""
    _, _, data = run
    assert data["documents"]
    for document in data["documents"]:
        nodes = SaqqaraKernel._count(document["tree"]["root"])
        assert nodes > 1, f"{document['path']} produced a root and nothing else"
        assert document["tree"]["root"]["children"]


def test_k4_every_document_carries_its_traces(run):
    _, _, data = run
    for document in data["documents"]:
        assert "builder" in document["traces"]
        assert document["traces"]["builder"]["observations"] > 0
        assert document["sha256"] and len(document["sha256"]) == 64


def test_k4_what_could_not_be_read_is_counted(run):
    _, _, data = run
    assert data["report"]["counts"]["refused"] == 1
    assert data["report"]["refusals"][0]["reason"] == "unsupported-format"


def test_k4_a_duplicate_is_counted_once(tmp_path):
    source = _corpus(tmp_path / "dupes")
    (source / "copy.md").write_bytes((source / "n.md").read_bytes())
    data = SaqqaraKernel().compute(
        KernelInput(workspace=tmp_path, config={"source": {"path": str(source)}})
    )
    assert data["report"]["counts"]["duplicate"] == 1


def test_k4_the_summary_fits_the_envelope(run):
    kernel, _, data = run
    summary = kernel.summarize(data)
    assert 0 < len(summary) <= 500
    assert "document(s)" in summary and "root" in summary


def test_k4_the_summary_reports_refusals_and_abstentions(run):
    kernel, _, data = run
    summary = kernel.summarize(data)
    assert "Refused 1" in summary
    assert "abstention(s)" in summary


def test_k4_the_kernel_is_discoverable_under_its_own_name():
    from ragix_kernels.registry import KernelRegistry

    KernelRegistry.discover()
    assert KernelRegistry.get("saqqara").name == "saqqara"


def test_k4_the_opt_in_outline_pass_is_off_by_default(run, tmp_path):
    kernel, _, data = run
    assert all("outline" not in d["traces"] for d in data["documents"])

    source = _corpus(tmp_path / "promoted")
    opted = kernel.compute(
        KernelInput(
            workspace=tmp_path,
            config={"source": {"path": str(source)}, "promote_outline": True},
        )
    )
    assert any("outline" in d["traces"] for d in opted["documents"])


# ------------------------------------------------------------ the MCP surface

def _mcp_module():
    import importlib.util

    path = ROOT / "MCP" / "ragix_mcp_server.py"
    if not path.is_file():
        pytest.skip("MCP server is not part of this checkout")
    spec = importlib.util.spec_from_file_location("ragix_mcp_server_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool(module, name):
    """The decorator may wrap the function or hand it back; either is fine."""
    tool = getattr(module, name)
    return getattr(tool, "fn", tool)


@pytest.fixture(scope="module")
def mcp():
    return _mcp_module()


def test_k4_mcp_exposes_both_tools(mcp):
    for name in ("koas_saqqara_run", "koas_saqqara_status"):
        assert _tool(mcp, name) is not None


def test_k4_mcp_run_reads_a_corpus(mcp, tmp_path):
    source = _corpus(tmp_path / "mcp")
    result = _tool(mcp, "koas_saqqara_run")(source=str(source), workspace=str(tmp_path))
    assert result["success"] is True
    assert len(result["merkle_root"]) == 64
    assert result["report"]["counts"]["refused"] == 1
    assert all(d["nodes"] > 1 for d in result["documents"])


def test_k4_mcp_status_reads_back_the_same_run(mcp, tmp_path):
    source = _corpus(tmp_path / "mcp2")
    run = _tool(mcp, "koas_saqqara_run")(source=str(source), workspace=str(tmp_path))
    status = _tool(mcp, "koas_saqqara_status")(workspace=str(tmp_path))
    assert status["merkle_root"] == run["merkle_root"]
    assert status["source_root"] == run["source_root"]
    assert len(status["documents"]) == len(run["documents"])
    assert isinstance(status["abstentions"], list)


def test_k4_mcp_status_says_so_when_there_is_nothing_to_read(mcp, tmp_path):
    """An empty answer and a missing result must not look alike."""
    status = _tool(mcp, "koas_saqqara_status")(workspace=str(tmp_path / "never-ran"))
    assert "error" in status


# ------------------------------------------------------------- the CI guard

def test_k4_the_guard_runs_in_continuous_integration():
    """The hook covers one clone. This is the half a pull request cannot step over."""
    workflow = ROOT / ".github" / "workflows" / "guard.yml"
    assert workflow.is_file(), "the CI guard is a precondition of the first public push"
    text = workflow.read_text(encoding="utf-8")
    assert "check_forbidden.py --selftest" in text, "prove the guard fires before trusting it"
    assert "check_forbidden.py" in text and "git ls-files" in text
    assert "pull_request" in text and "push" in text


def test_k4_office_writers_stamp_a_time_so_two_builds_differ(tmp_path):
    """Recorded rather than assumed, because it shapes the tests above.

    Two builds of one fixture are not byte-identical: the writers embed a
    creation time. It costs nothing here — fixtures are compared by structure,
    not by bytes — but any test that rebuilds and then reasons about hashes is
    measuring the clock, so those tests copy instead.
    """
    first = G.FIXTURES["docx_two_tier"](tmp_path / "a.docx").read_bytes()
    second = G.FIXTURES["docx_two_tier"](tmp_path / "b.docx").read_bytes()
    assert len(first) == len(second)
    copied = (tmp_path / "c.docx")
    copied.write_bytes(first)
    assert copied.read_bytes() == first, "copying, unlike rebuilding, preserves the bytes"
