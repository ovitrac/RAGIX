"""
Gate K2' — the laid-out-document reader.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Carries SPEC.md K2.13 and K2.14, the two propositions test_k2_adapters.py could only name while
this reader did not exist.

Both fixtures are written byte by byte by the generators, so the ground truth is not a library's
opinion about what it wrote: PDF_OUTLINE_ENTRIES below is the outline the file declares, because
it is the list the builder wrote into it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

from generators import (  # noqa: E402
    PDF_OUTLINE_ENTRIES,
    pdf_no_text_layer,
    pdf_outline,
)

from ragix_kernels.saqqara.adapters import read_path, read_paths  # noqa: E402


@pytest.fixture(scope="module")
def outlined(tmp_path_factory):
    return read_path(pdf_outline(tmp_path_factory.mktemp("k2p") / "outlined.pdf"))


@pytest.fixture(scope="module")
def scanned_path(tmp_path_factory):
    return pdf_no_text_layer(tmp_path_factory.mktemp("k2p_scan") / "scanned.pdf")


def _of(records, kind):
    return [r for r in records if r.kind == kind]


# ------------------------------------------- K2.13 a declared outline is read

def test_k2_13_one_entry_per_declared_entry(outlined):
    assert len(_of(outlined, "outline_entry")) == len(PDF_OUTLINE_ENTRIES)


def test_k2_13_titles_pages_and_depths_are_exact(outlined):
    read = [
        (r.text, r.locator.page, r.facts["level"]) for r in _of(outlined, "outline_entry")
    ]
    assert read == PDF_OUTLINE_ENTRIES


def test_k2_13_a_nested_entry_keeps_its_depth(outlined):
    """A reader that flattens the tree reports the right count at the wrong depths."""
    depths = [r.facts["level"] for r in _of(outlined, "outline_entry")]
    assert set(depths) == {1, 2}, "this fixture nests, by construction"
    nested = next(r for r in _of(outlined, "outline_entry") if r.facts["level"] == 2)
    assert nested.text == "Moyens humains"
    assert nested.locator.page == 2


def test_k2_13_the_reader_does_not_invent_an_outline(scanned_path):
    """Where none is declared, the reader says nothing about structure."""
    assert _of(read_path(scanned_path), "outline_entry") == []


def test_k2_13_text_carries_the_page_it_sits_on(outlined):
    text = _of(outlined, "text")
    assert text
    assert {r.locator.page for r in text} == {1, 2, 3}
    for record in text:
        assert record.facts["font_size"] > 0
        assert isinstance(record.facts["x"], float)


def test_k2_13_no_bounding_box_is_claimed(outlined):
    """A box invented from a font size is a measurement nobody made."""
    for record in _of(outlined, "text"):
        assert record.locator.bbox is None


# ---------------------------------------- K2.14 a page with no text layer

def test_k2_14_a_scanned_page_is_declared_pending(scanned_path):
    pages = _of(read_path(scanned_path), "page")
    assert len(pages) == 1
    facts = pages[0].facts
    assert facts["has_text"] is False
    assert facts["needs_ocr"] is True
    assert facts["image_count"] == 1


def test_k2_14_it_is_not_reported_as_empty_content(scanned_path):
    """The page must exist in the output. Absence would read as 'nothing there'."""
    records = read_path(scanned_path)
    assert _of(records, "page"), "the page is reported, not omitted"
    assert _of(records, "text") == [], "and it carries no text, because it has none"


def test_k2_14_a_page_with_text_is_not_marked_pending(outlined):
    pages = _of(outlined, "page")
    assert len(pages) == 3
    assert all(p.facts["needs_ocr"] is False for p in pages)
    assert all(p.facts["has_text"] is True for p in pages)


def test_k2_14_a_scanned_page_is_not_a_refusal(scanned_path):
    """Pending is a state of a readable document, not a failure to read it."""
    facts, report = read_paths([scanned_path])
    assert report.counts == {"read": 1, "refused": 0, "duplicate": 0}
    assert facts


def test_k2_14_the_pending_state_survives_serialisation(scanned_path):
    payload = [r.to_dict() for r in read_path(scanned_path)]
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    assert json.loads(text) == payload
    assert '"needs_ocr":true' in text.replace(", ", ",").replace(": ", ":")
