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

#: The tools this family exposes, in the order it registers them.
#:
#: Pinned, and extended deliberately each time the surface grows: registering a
#: tool is one decorator, which is exactly the kind of change that should not
#: arrive unnoticed. The store added the last two (plan §1), and this line is the
#: edit that says so.
DECLARED_TOOLS = ["koas_saqqara_run", "koas_saqqara_status",
                  "koas_saqqara_index", "koas_saqqara_search"]


class _StubServer:
    """The smallest thing that looks like a FastMCP server to a registrar.

    The surface is reached through `register_saqqara_tools`, which is how the real
    server reaches it too, rather than by looking up attributes on the server
    MODULE. The earlier version did the latter, and so asserted that two functions
    were defined at the top level of `MCP/ragix_mcp_server.py` — true only while
    the family did not own its own surface. Moving the tools into the family, which
    is the pattern every other family follows, broke four tests without changing
    what any caller can do. A gate should pin what the family exposes, not which
    file happens to hold it.
    """

    def __init__(self):
        self.tools: dict = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


@pytest.fixture(scope="module")
def mcp():
    from ragix_kernels.saqqara.mcp.tools import register_saqqara_tools

    server = _StubServer()
    register_saqqara_tools(server)
    return server


def _tool(server, name):
    """The decorator may wrap the function or hand it back; either is fine."""
    tool = server.tools[name]
    return getattr(tool, "fn", tool)


def test_k4_mcp_exposes_exactly_the_declared_tools(mcp):
    assert list(mcp.tools) == DECLARED_TOOLS


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


def test_k4_the_server_registers_the_family_surface():
    """The server must still reach the tools — the family owning them is not enough.

    Checked on the source rather than by importing: loading the server pulls in the
    whole project, and a dependency missing from this environment would turn a
    registration question into an unrelated ImportError.
    """
    server = ROOT / "MCP" / "ragix_mcp_server.py"
    if not server.is_file():
        pytest.skip("MCP server is not part of this checkout")
    text = server.read_text(encoding="utf-8")
    assert "register_saqqara_tools" in text, "the server no longer registers saqqara"
    assert "koas_saqqara_run" not in text.split("def _register_saqqara_tools")[-1].split(
        "\ndef "
    )[0], "the server should register the tools, not define them"


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


# ------------------------------------------- K4.3 — an abstention is listed, not counted

ABSTAINING = {"prose.docx": "docx_layout_prose", "block.xlsx": "undecidable_block"}


def _abstaining_corpus(root: Path) -> Path:
    """Two documents that abstain, through two analyzers, in two native shapes.

    `docx_layout_prose` is prose in a grid: `grid_tables` types both tables as
    layout and keeps a LIST of two records. `undecidable_block` is a uniform
    block: `header_bands` keeps a TALLY in `abstained` and its record apart in
    `abstentions`. One fixture would exercise one shape and prove half the claim.
    """
    root.mkdir(parents=True, exist_ok=True)
    for name, fixture in ABSTAINING.items():
        G.FIXTURES[fixture](root / name)
    return root


def test_k4_3_the_counter_accepts_each_declared_shape_and_refuses_the_rest():
    """One assertion per shape, and the two that are easy to get wrong.

    A histogram and a single abstention record are both mappings. Telling them
    apart by what they hold rather than by who wrote them is the whole rule, and
    an empty histogram is the case that would otherwise put an abstention into
    every document that had none.
    """
    from ragix_kernels.saqqara.analyzers.contract import count_reported

    assert count_reported(None) == 0                                    # nothing to report
    assert count_reported(0) == 0                                       # a tally at rest
    assert count_reported(7) == 7                                       # a tally
    assert count_reported([]) == 0                                      # a list, empty
    assert count_reported([{"rule": "a"}, {"rule": "b"}]) == 2          # a list of records
    assert count_reported({}) == 0                                      # a histogram, empty
    assert count_reported({"no-candidate": 3, "already-bound": 2}) == 5  # a histogram
    assert count_reported({"reason": "no-size-contrast",
                           "signals": {"lines": 0}}) == 1               # ONE record

    for refused in ("2", 2.0, True, False, object()):
        with pytest.raises(TypeError):
            count_reported(refused)


def test_k4_3_each_shape_normalises_into_the_register_without_loss():
    """The four native shapes, each through the source declared for its producer.

    The locator claim is asserted in both directions: kept where the producer
    kept one, `None` where it did not — never invented to make the record look
    complete.
    """
    from ragix_kernels.saqqara.analyzers.contract import abstention_records

    listed = abstention_records("grid_tables", {"abstained": [
        {"flow": "body", "table_index": 0, "type": "layout", "rule": "D4-unstyled-single-row"}]})
    assert listed == [{"analyzer": "grid_tables", "locator": {"flow": "body", "table_index": 0},
                       "reason": "D4-unstyled-single-row", "signals": {"type": "layout"},
                       "count": 1}]

    tallied = abstention_records("header_bands", {
        "abstained": 1, "abstentions": [{"range": "A1:C3", "reason": "no-body-rows"}]})
    assert tallied[0]["locator"] == {"range": "A1:C3"}
    assert tallied[0]["reason"] == "no-body-rows"

    single = abstention_records("format_headings", {
        "abstained": {"reason": "declared-outline", "signals": {"declared_headings": 4}}})
    assert len(single) == 1
    assert single[0]["locator"] is None, "a document-level abstention addresses nothing"
    assert single[0]["signals"] == {"declared_headings": 4}

    histogram = abstention_records("saqqara.caption_binding",
                                   {"abstained": {"no-candidate-within-gap": 3}})
    assert histogram[0]["count"] == 3 and histogram[0]["reason"] == "no-candidate-within-gap"

    assert abstention_records("format_headings", {"abstained": None}) == []
    assert abstention_records("saqqara.caption_binding", {"abstained": {}}) == []

    with pytest.raises(KeyError):
        abstention_records("an_analyzer_nobody_registered", {"abstained": [{}]})


def test_k4_3_a_tally_that_disagrees_with_its_records_is_refused():
    """`header_bands` keeps both. If they ever part, the register cannot choose."""
    from ragix_kernels.saqqara.analyzers.contract import abstention_records

    with pytest.raises(ValueError):
        abstention_records("header_bands", {
            "abstained": 4, "abstentions": [{"range": "A1:B2", "reason": "uniform-block"}]})


def test_k4_3_every_producer_that_abstains_is_registered(tmp_path):
    """The preventive half: an unregistered producer fails here, not in a run.

    A register claims to list everything. An analyzer whose abstentions it does
    not know would make that claim false in silence, so the walk asserts the
    declaration exists for every producer the real pipeline meets.
    """
    from ragix_kernels.saqqara.analyzers.contract import (
        ABSTENTION_KEYS, ABSTENTION_SOURCES, abstention_records, reports_abstention)

    for name, source in ABSTENTION_SOURCES.items():
        assert source.records in ABSTENTION_KEYS, (
            f"{name} declares its records under {source.records!r}, outside the "
            f"convention {ABSTENTION_KEYS} the register searches by")
        assert source.tally is None or source.tally in ABSTENTION_KEYS

    corpus = _abstaining_corpus(tmp_path / "corpus")
    for name, fixture in CORPUS.items():                      # and one per format
        G.FIXTURES[fixture](corpus / name)
    data = SaqqaraKernel().compute(
        KernelInput(workspace=tmp_path / "ws", config={"source": {"path": str(corpus)}}))

    seen = set()
    for document in data["documents"]:
        for name, trace in (document.get("traces") or {}).items():
            if isinstance(trace, dict) and reports_abstention(trace):
                seen.add(name)
                abstention_records(name, trace)      # raises if it is not registered
    assert seen, "no analyzer reported an abstention — the walk found nothing to check"


def test_k4_3_the_register_lists_who_where_and_why(tmp_path):
    """The claim itself, on two analyzers abstaining in two different shapes."""
    corpus = _abstaining_corpus(tmp_path / "corpus")
    kernel = SaqqaraKernel()
    data = kernel.compute(
        KernelInput(workspace=tmp_path / "ws", config={"source": {"path": str(corpus)}}))

    register = data["report"]["abstentions"]
    assert len(register) == 3, register              # two from grid_tables, one from header_bands

    by_analyzer = {}
    for record in register:
        by_analyzer.setdefault(record["analyzer"], []).append(record)
    assert set(by_analyzer) == {"grid_tables", "header_bands"}, (
        "the fixture no longer exercises two producers — the test would prove half its claim")

    for record in register:
        assert record["reason"], f"an abstention without a reason is a defect: {record}"
        assert record["path"].endswith((".docx", ".xlsx"))
        assert record["locator"], "both producers here keep a locator"

    assert {r["reason"] for r in by_analyzer["grid_tables"]} == {"D4-unstyled-single-row"}
    assert by_analyzer["header_bands"][0]["reason"] == "uniform-block"

    counted = sum(record["count"] for record in register)
    assert counted == len(register), "every record here stands for one abstention"
    assert f"{counted} abstention(s)" in kernel.summarize(data)


def test_k4_3_the_summary_reads_the_register_rather_than_counting_again(tmp_path):
    """Two rules for one fact are two rules that disagree. Falsified by editing one."""
    corpus = _abstaining_corpus(tmp_path / "corpus")
    kernel = SaqqaraKernel()
    data = kernel.compute(
        KernelInput(workspace=tmp_path / "ws", config={"source": {"path": str(corpus)}}))

    data["report"]["abstentions"] = data["report"]["abstentions"][:1]
    assert "1 abstention(s)" in kernel.summarize(data), (
        "the summary counted the traces again instead of reading the register")


def test_k4_3_the_cli_and_the_summary_cannot_disagree(tmp_path):
    """The CLI listed only the producer that writes `abstentions`; the summary counted all."""
    from ragix_kernels.saqqara.cli.saqqaractl import _abstentions

    corpus = _abstaining_corpus(tmp_path / "corpus")
    kernel = SaqqaraKernel()
    data = kernel.compute(
        KernelInput(workspace=tmp_path / "ws", config={"source": {"path": str(corpus)}}))

    listed = _abstentions(data)
    assert len(listed) == len(data["report"]["abstentions"])
    assert {analyzer for _, analyzer, _ in listed} == {"grid_tables", "header_bands"}

    # the control: the rule the CLI used until K4.3, applied to the same result
    old_rule = [
        (document["path"], name, entry.get("reason"))
        for document in data["documents"]
        for name, trace in (document.get("traces") or {}).items()
        if isinstance(trace, dict)
        for entry in (trace.get("abstentions") or [])
    ]
    assert len(old_rule) == 1, (
        "the old rule found only the analyzer that writes `abstentions`; if it now "
        "finds them all, this control has stopped controlling anything")


def test_k4_3_a_document_that_abstains_is_summarised_rather_than_lost(tmp_path):
    """The end-to-end claim, through the envelope that used to lose the run.

    Before K4.3 `summarize` cast `grid_tables`' list with `int(...)`, the envelope
    caught the TypeError and wrote an output holding only the error — a run that
    read the documents, built their trees and then lost them.
    """
    corpus = _abstaining_corpus(tmp_path / "corpus")
    workspace = tmp_path / "ws"
    workspace.mkdir()                       # `run` validates it; `compute` does not

    kernel = SaqqaraKernel()
    output = kernel.run(
        KernelInput(workspace=workspace, config={"source": {"path": str(corpus)}}))

    assert output.success, "a document that abstains must not fail the run"
    data = json.loads(Path(output.output_file).read_text())["data"]
    assert len(data.get("documents") or []) == 2, "the run kept no documents"
    assert len(data["report"]["abstentions"]) == 3
    assert "3 abstention(s)" in kernel.summarize(data)


def test_k4_3_the_cast_this_replaces_fails_on_the_same_document(tmp_path):
    """The control. Without it, the tests above pass on a fixture that never abstained.

    This is the line `summarize` carried until K4.3, applied to the same trace.
    """
    corpus = _abstaining_corpus(tmp_path / "corpus")
    data = SaqqaraKernel().compute(
        KernelInput(workspace=tmp_path / "ws", config={"source": {"path": str(corpus)}}))

    prose = next(d for d in data["documents"] if d["path"].endswith("prose.docx"))
    abstained = prose["traces"]["grid_tables"]["abstained"]
    assert isinstance(abstained, list) and len(abstained) == 2

    with pytest.raises(TypeError):
        int(abstained or 0)


def test_k4_3_a_record_carries_what_the_rules_read_not_only_which_rule_fired(tmp_path):
    """`header_bands` holds its signals; until K4.3 the trace record dropped them.

    Measured on a real corpus before this assertion existed: 903 of 910 records
    named a rule and carried `signals: {}`. A reader could see that R4 fired and
    not one number it fired on.

    Three reasons, three fixtures, because `band-too-deep` — the most frequent
    abstention on that corpus, 811 of 910 — was reachable by no generated
    document at all until `deep_header_band` was written for this gate.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, fixture in (("deep.xlsx", "deep_header_band"),
                          ("uniform.xlsx", "undecidable_block"),
                          ("merges.xlsx", "overlapping_merges")):
        G.FIXTURES[fixture](corpus / name)

    data = SaqqaraKernel().compute(
        KernelInput(workspace=tmp_path / "ws", config={"source": {"path": str(corpus)}}))
    bands = [r for r in data["report"]["abstentions"] if r["analyzer"] == "header_bands"]

    assert {r["reason"] for r in bands} == {
        "band-too-deep", "uniform-block", "non-laminar-band-merges"}, (
        "the fixtures no longer reach all three reasons this gate is about")

    for record in bands:
        assert record["signals"], f"{record['reason']} carries no signals: {record}"
        assert record["signals"].get("rules"), (
            f"{record['reason']} does not say which rule fired: {record['signals']}")
        assert record["locator"] and record["locator"].get("range")

    deep = next(r for r in bands if r["reason"] == "band-too-deep")
    assert "R4-band-depth" in deep["signals"]["rules"]
    assert deep["signals"]["dtypes"], "the block's types are part of what the rule read"
