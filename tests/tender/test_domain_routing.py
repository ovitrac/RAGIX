"""The routing classifier — and the two channels it must not confuse.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

`routing_class` is what the retrieval policy filters on, so a classifier that
answers plausibly-but-wrongly does not fail here, it filters there. The two
tests that matter most are the ones asserting the *silent flip* of
`NOTE_K5A_INTERPRETATION_20260830.md` §1.3 in both directions: a styled
word-processing document carries no heading node, and an outlined laid-out
document carries no heading style. A classifier reading one channel for both
formats passes half of this file and answers D2/P2 for everything else.

Every fixture is proved non-empty before the verdict is read: a control over a
document the reader saw nothing in would pass and say nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ragix_kernels.tender.domain.routing import D1_MIN_HEADINGS, classify, heading_style_level  # noqa: E402
from ragix_kernels.tender.domain.substrate import read_tree                                     # noqa: E402


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


# ── fixtures, each built by code ─────────────────────────────────────────────

def _docx(tmp_path: Path, n_headings: int) -> Path:
    docx = pytest.importorskip("docx", reason="python-docx")
    document = docx.Document()
    for index in range(n_headings):
        document.add_paragraph(f"Titre {index + 1}", style="Heading 1")
    document.add_paragraph("Un paragraphe de corps.")
    out = tmp_path / f"styled_{n_headings}.docx"
    document.save(out)
    return out


def _pdf(tmp_path: Path, *, name: str, text: bool = True, toc=None,
         pages: int = 3, blank_first: bool = False) -> Path:
    fitz = pytest.importorskip("fitz", reason="pdf-fitz extra")
    document = fitz.open()
    for number in range(pages):
        page = document.new_page()
        if not text or (blank_first and number == 0):
            continue
        page.insert_text((72, 700), f"{number + 1}. Titre de chapitre", fontsize=20)
        y = 660
        for line in range(10):
            page.insert_text((72, y), f"Ligne de corps {line} page {number + 1}.",
                             fontsize=10)
            y -= 16
    if toc:
        document.set_toc(toc)
    out = tmp_path / name
    document.save(out)
    document.close()
    return out


# ── the flip, both directions ────────────────────────────────────────────────

def test_a_styled_docx_is_D1_although_its_tree_holds_no_heading_node(tmp_path):
    """The channel that moved. The second assertion is the point: D1 is reached
    with zero heading nodes in the tree, so it cannot have been reached by
    counting them."""
    tree = read_tree(_docx(tmp_path, D1_MIN_HEADINGS)).tree

    assert classify(tree).routing_class == "D1"
    assert not [n for n in _walk(tree.root) if n.kind == "heading"], (
        "the fixture must carry no heading node, or this test proves nothing")


def test_an_outlined_pdf_is_P1_although_no_node_declares_a_heading_style(tmp_path):
    """The same flip, the other way. P1 is reached with no heading style
    anywhere, so it cannot have been reached by reading facts['style']."""
    path = _pdf(tmp_path, name="outlined.pdf",
                toc=[[1, "Chapitre A", 1], [2, "Section A.1", 2], [1, "Chapitre B", 3]])
    tree = read_tree(path).tree

    assert classify(tree).routing_class == "P1"
    assert not [n for n in _walk(tree.root) if heading_style_level(n) is not None], (
        "the fixture must declare no heading style, or this test proves nothing")


# ── the D1/D2 threshold ──────────────────────────────────────────────────────

def test_one_stray_heading_style_below_the_threshold_is_still_D2(tmp_path):
    """The control on D1. Corpus documents carry a single stray styled heading
    while being structurally flat; the threshold is what separates them."""
    result = classify(read_tree(_docx(tmp_path, D1_MIN_HEADINGS - 1)).tree)

    assert result.routing_class == "D2"
    assert result.evidence["heading_style_paras"] == D1_MIN_HEADINGS - 1


# ── the promoted tree ────────────────────────────────────────────────────────

def test_promotion_does_not_turn_a_flat_pdf_into_P1(tmp_path):
    """The trap `origin == "read"` exists for. The promoting analyzers infer
    headings from size contrast; those are the kernel's conclusion, not the
    document's declaration, and the class must not depend on how the caller
    happened to read the file."""
    path = _pdf(tmp_path, name="flat.pdf")

    plain = read_tree(path, promote=False).tree
    promoted = read_tree(path, promote=True).tree

    inferred = [n for n in _walk(promoted.root)
                if n.kind == "heading" and n.origin == "inferred"]
    assert inferred, ("the fixture must actually provoke a promotion, or this "
                      "control cannot fire")

    assert classify(plain).routing_class == "P2"
    assert classify(promoted).routing_class == "P2"
    assert classify(promoted).evidence["outline_entries"] == 0


# ── the text layer ───────────────────────────────────────────────────────────

def test_a_pdf_with_no_text_layer_anywhere_is_P3(tmp_path):
    result = classify(read_tree(_pdf(tmp_path, name="scan.pdf", text=False)).tree)

    assert result.routing_class == "P3"
    assert result.evidence["pages_with_text"] == 0
    assert result.evidence["pages"] == 3


def test_one_blank_page_does_not_make_a_document_P3(tmp_path):
    """P3 is the whole document having no text, which is what the old
    `total_chars == 0` said. The kernel reports the text layer per page, so the
    predicate has to be stated over all of them — and the blank page is counted
    rather than collapsed."""
    path = _pdf(tmp_path, name="mixed.pdf", blank_first=True)
    result = classify(read_tree(path).tree)

    assert result.routing_class == "P2"
    assert result.evidence["pages"] == 3
    assert result.evidence["pages_with_text"] == 2


# ── the classes a format decides on its own ──────────────────────────────────

def test_the_spreadsheet_presentation_and_markdown_classes(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    book.active["A1"] = "Question"
    xlsx = tmp_path / "book.xlsx"
    book.save(xlsx)

    md = tmp_path / "note.md"
    md.write_text("# Titre\n\nUn paragraphe.\n", encoding="utf-8")

    assert classify(read_tree(xlsx).tree).routing_class == "E1"
    assert classify(read_tree(md).tree).routing_class == "M1"

    pptx_mod = pytest.importorskip("pptx", reason="python-pptx")
    deck = pptx_mod.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Titre de la diapositive"
    deck_path = tmp_path / "deck.pptx"
    deck.save(deck_path)

    assert classify(read_tree(deck_path).tree).routing_class == "X1"


# ── fail closed ──────────────────────────────────────────────────────────────

def test_an_unclassified_format_raises_rather_than_defaulting(tmp_path):
    """A document with no class must not acquire one by default: the retrieval
    policy filters on this field, and a default would filter on a guess."""
    from dataclasses import replace

    tree = read_tree(_docx(tmp_path, 1)).tree
    # Provenance is frozen by the kernel, which is why this goes through
    # `replace` rather than an assignment: the tree stays a real tree.
    tree.root.provenance = replace(tree.root.provenance, source_format="rtf")

    with pytest.raises(ValueError, match="rtf"):
        classify(tree)


def test_the_facts_it_produces_carry_the_class_and_leave_quality_empty(tmp_path):
    """`quality` is the quality axis, declared out of phase; this lane does not
    occupy it with its own counts."""
    facts = classify(read_tree(_docx(tmp_path, D1_MIN_HEADINGS)).tree).facts()

    assert facts.routing_class == "D1"
    assert facts.quality == {}
    assert facts.dated is False
