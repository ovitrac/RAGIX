"""
Gate K2' — the pdf reader's text port, and its opt-in line join.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The default reading is pypdf's, and every other test of this package runs on it. What is proved
here is what the options add and what they leave alone: a reader configured at the defaults is the
registered reader itself; a reader chosen by name says so in its provenance; a line join keeps the
fragments it joined, separates a word space from a column gap, leaves a piece whose end it cannot
measure where it was and counts it, and marks every join where a digit meets a digit for a person to
verify. The PyMuPDF reader runs in a fresh process, so that loading it here can never disturb the
K6.16 check that no default route loaded it, whatever order the tests run in.

The fixture is written byte by byte below, from invented text, in the manner of generators.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

from generators import _pdf_assemble, _pdf_escape, _pdf_stream  # noqa: E402

from ragix_kernels.base import KernelInput  # noqa: E402
from ragix_kernels.saqqara.adapters import read_path, registered_adapters  # noqa: E402
from ragix_kernels.saqqara.adapters.pdf import (  # noqa: E402
    OPTIONS,
    PAGE_FACTS,
    TEXT_FACTS,
)
from ragix_kernels.saqqara.adapters.pdf_lines import (  # noqa: E402
    JOIN_PAGE_FACTS,
    JOIN_TEXT_FACTS,
    REVIEW_REASONS,
)
from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel  # noqa: E402
from ragix_kernels.saqqara.store.config import load_config  # noqa: E402

HAS_MUPDF = find_spec("pymupdf") is not None

#: One text-showing operation per entry: text, x, y, font. `FW` declares its widths,
#: every character half an em, so at 10 points a piece of n characters ends 5n points
#: after it starts and every gap below is checkable by hand. `FN` declares none.
PIECES = [
    ("Livraison le 2", 100, 700, "FW"),   # 14 chars: ends at 170
    ("7 mars 2031.", 170, 700, "FW"),     # starts where it ends: one word, a digit meets a digit
    ("Article", 100, 650, "FW"),          # 7 chars: ends at 135
    ("douze", 138, 650, "FW"),            # 0.3 em on: a word space
    ("Montant 150", 100, 600, "FW"),      # 11 chars: ends at 155
    ("50", 300, 600, "FW"),               # 14.5 em on: another column, never one number
    ("10", 100, 550, "FW"),               # 2 chars: ends at 110
    ("3", 110, 554, "FW"),                # 0.4 em higher: an exponent, not the same baseline
    ("Remise", 100, 500, "FN"),           # no widths: its end is not known
    ("finale", 140, 500, "FN"),           # so this boundary is undecided, and counted
]

JOINED = {
    "Livraison le 27 mars 2031.": ([None, ""], {"reason": "digit-run-joined", "boundaries": [1]}),
    "Article douze": ([None, " "], None),
}
APART = ["Montant 150", "50", "10", "3", "Remise", "finale"]


def _pieces_pdf(path: Path) -> Path:
    payload = b"".join(
        b"BT /" + font.encode("ascii") + b" 10 Tf " + f"{x} {y}".encode("ascii")
        + b" Td (" + _pdf_escape(text) + b") Tj ET\n"
        for text, x, y, font in PIECES
    )
    widths = " ".join(["500"] * 95)                  # codes 32..126, half an em each
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [5 0 R] /Count 1 >>",
        ("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /FirstChar 32 "
         "/LastChar 126 /Widths [" + widths + "] >>").encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << "
         b"/Font << /FW 3 0 R /FN 4 0 R >> >> /Contents 6 0 R >>"),
        _pdf_stream(payload),
    ]
    path.write_bytes(_pdf_assemble(objects))
    return path


@pytest.fixture(scope="module")
def pieces(tmp_path_factory) -> Path:
    return _pieces_pdf(tmp_path_factory.mktemp("k2p_lines") / "pieces.pdf")


@pytest.fixture(scope="module")
def joined(pieces):
    reader = registered_adapters()[".pdf"].configured({"line_join": True})
    return reader, read_path(pieces, readers={"pdf": reader})


def _of(records, kind):
    return [r for r in records if r.kind == kind]


# ------------------------------------------------- the defaults are the registered reader

def test_k2p_the_default_options_configure_nothing():
    """At the defaults the registered reader reads — not a copy that behaves like it."""
    registered = registered_adapters()[".pdf"]
    assert registered.configured(None) is registered
    assert registered.configured({}) is registered
    assert registered.configured(dict(OPTIONS)) is registered


def test_k2p_the_options_are_declared_once_in_the_store_defaults():
    """The reader's defaults and the configuration's shape cannot drift apart."""
    assert load_config().section("pdf") == OPTIONS


def test_k2p_by_default_every_piece_is_its_own_record(pieces):
    records = _of(read_path(pieces), "text")
    assert [r.text for r in records] == [p[0] for p in PIECES]
    for record in records:
        assert set(record.facts) == set(TEXT_FACTS)


def test_k2p_an_unknown_option_or_value_is_refused():
    registered = registered_adapters()[".pdf"]
    for bad in ({"text_reader": "pdfminer"}, {"line_join": "yes"}, {"linejoin": True}):
        with pytest.raises(ValueError):
            registered.configured(bad)


def test_k2p_the_kernel_refuses_bad_options_before_reading(pieces, tmp_path):
    output = SaqqaraKernel().run(KernelInput(
        workspace=tmp_path,
        config={"source": {"path": str(pieces)}, "pdf": {"text_reader": "pdfminer"}}))
    assert output.success is False
    assert any("pdf.text_reader" in str(e) for e in output.errors), output.errors


# ------------------------------------------------------------------------- the line join

def test_k2p_a_join_keeps_its_fragments_and_marks_a_digit_meeting_a_digit(joined):
    _, records = joined
    lines = {r.text: r for r in _of(records, "text")}
    for text, (separators, review) in JOINED.items():
        facts = lines[text].facts
        assert [f["sep"] for f in facts["fragments"]] == separators, text
        assert facts["review"] == review, text
    fragments = lines["Livraison le 27 mars 2031."].facts["fragments"]
    assert [f["text"] for f in fragments] == ["Livraison le 2", "7 mars 2031."]
    assert [(f["x"], f["y"]) for f in fragments] == [(100.0, 700.0), (170.0, 700.0)]
    assert lines["Livraison le 27 mars 2031."].facts["review"]["reason"] in REVIEW_REASONS


def test_k2p_what_is_not_one_line_stays_as_read(joined):
    """A column gap, a raised exponent and an unmeasured end are not joins."""
    _, records = joined
    lines = {r.text: r for r in _of(records, "text")}
    for text in APART:
        assert lines[text].facts["fragments"] is None, text
        assert lines[text].facts["review"] is None, text
    assert "Montant 15050" not in lines and "103" not in lines


def test_k2p_the_page_counts_the_joins_and_what_it_could_not_decide(joined):
    _, records = joined
    (page,) = _of(records, "page")
    assert page.facts["line_join"] == {"joined": 2, "absorbed": 4, "review": 1, "undecided": 1}


def test_k2p_every_record_carries_the_vocabulary_its_reader_declares(joined):
    reader, records = joined
    for kind in ("text", "page"):
        for record in _of(records, kind):
            assert set(record.facts) == set(reader.fact_sets[kind]), kind
    assert reader.fact_sets["text"] == TEXT_FACTS + JOIN_TEXT_FACTS
    assert reader.fact_sets["page"] == PAGE_FACTS + JOIN_PAGE_FACTS
    registered = registered_adapters()[".pdf"]
    assert registered.fact_sets["text"] == TEXT_FACTS, "the registered vocabulary is untouched"
    assert registered.version == "0.8.0"


def test_k2p_the_reader_used_is_in_the_provenance(pieces, tmp_path):
    def version(options, where):
        config = {"source": {"path": str(pieces)}}
        if options is not None:
            config["pdf"] = options
        output = SaqqaraKernel().run(KernelInput(workspace=where, config=config))
        assert output.success, output.errors
        root = output.data["documents"][0]["tree"]["root"]
        return root["provenance"]["kernel_version"]

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert version(None, tmp_path / "a") == "0.8.0"
    assert version({"line_join": True}, tmp_path / "b") == "0.8.0+line-join"


# ---------------------------------------------------- the opt-in reader, in a fresh process

PROBE = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    from ragix_kernels.base import KernelInput
    from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel
    from ragix_kernels.saqqara.render.guard import loaded_agpl_modules

    path, workspace, options = sys.argv[1], Path(sys.argv[2]), json.loads(sys.argv[3])
    output = SaqqaraKernel().run(KernelInput(
        workspace=workspace, config={"source": {"path": path}, "pdf": options}))
    root = output.data["documents"][0]["tree"]["root"] if output.success else {}
    texts = []
    def walk(node):
        if node.get("kind") == "paragraph":
            texts.append((node.get("text"), node.get("facts", {}).get("review")))
        for child in node.get("children", []):
            walk(child)
    walk(root)
    print(json.dumps({"success": output.success, "errors": [str(e) for e in output.errors],
                      "version": root.get("provenance", {}).get("kernel_version"),
                      "texts": texts, "loaded": loaded_agpl_modules()}))
""")


def _probe(path: Path, workspace: Path, options: dict) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    workspace.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([sys.executable, "-c", PROBE, str(path), str(workspace),
                           json.dumps(options)],
                          capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not HAS_MUPDF, reason="the saqqara-mupdf extra is not installed")
def test_k2p_the_opt_in_reader_reads_and_says_so(pieces, tmp_path):
    result = _probe(pieces, tmp_path / "w", {"text_reader": "pymupdf"})
    assert result["success"], result["errors"]
    assert result["version"].startswith("0.8.0+pymupdf-")
    assert "pymupdf" in result["loaded"], "the runtime check sees the opt-in when it is chosen"
    words = " ".join(text for text, _ in result["texts"])
    for word in ("Livraison", "Article", "douze", "Remise", "finale"):
        assert word in words


@pytest.mark.skipif(not HAS_MUPDF, reason="the saqqara-mupdf extra is not installed")
def test_k2p_the_opt_in_reader_takes_the_join(pieces, tmp_path):
    result = _probe(pieces, tmp_path / "w", {"text_reader": "pymupdf", "line_join": True})
    assert result["success"], result["errors"]
    assert result["version"].startswith("0.8.0+pymupdf-")
    assert result["version"].endswith(".line-join")
    texts = [text for text, _ in result["texts"]]
    assert "Montant 150" in texts and "50" in texts, "two columns never become one number"
    assert "Montant 15050" not in texts and "Montant 150 50" not in texts


def test_k2p_the_opt_in_reader_refuses_when_absent(pieces, tmp_path, monkeypatch):
    """Chosen and not installed is a refusal, never a quiet return to the default."""
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    output = SaqqaraKernel().run(KernelInput(
        workspace=tmp_path,
        config={"source": {"path": str(pieces)}, "pdf": {"text_reader": "pymupdf"}}))
    assert output.success is False
    assert any("saqqara-mupdf" in str(e) for e in output.errors), output.errors
