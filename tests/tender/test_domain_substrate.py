"""The lab's single way to obtain a saqqara tree.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

`tender.substrate.read_tree` exists because eight lab instruments need the same
three calls and the same two opt-ins, and an opt-in forgotten in one of eight
copies is invisible: the tree still builds, it is simply missing what nobody
asked for. These tests hold the promoting opt-in and the drop accounting, which are the
parts that fail silently.

The object-store opt-in is **not** tested here, deliberately: no lab fixture
contains an image, so an assertion that a store-less read finds no objects
would pass whether or not the parameter were wired at all. A test that cannot
fail guards nothing. It is owned upstream by K6 on generated fixtures that do
carry figures, and it comes back here the day the lab has such a fixture.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ragix_kernels.tender.domain.substrate import TreeRead, read_tree

from tests.tender.fixtures import synthetic_crt, synthetic_docx_tables


@pytest.fixture(scope="module")
def spreadsheet(tmp_path_factory) -> Path:
    return synthetic_crt.build(tmp_path_factory.mktemp("substrate") / "crt.xlsx")


@pytest.fixture(scope="module")
def document(tmp_path_factory) -> Path:
    return synthetic_docx_tables.build(
        tmp_path_factory.mktemp("substrate") / "tables.docx")


def test_reads_a_spreadsheet_into_an_analyzed_tree(spreadsheet):
    read = read_tree(spreadsheet)

    assert read.source_format == "xlsx"
    assert read.reader_version                      # the reader states its version
    assert read.tree.root.provenance.source_format == "xlsx"

    kinds = {node.kind for node in read.tree.walk()}
    # `cell` is the property that matters: a lookup blind to spreadsheet cells
    # was the defect the tree services were introduced to fix (F-RD4).
    assert "cell" in kinds
    assert "table" in kinds


def test_reads_a_word_processing_document(document):
    read = read_tree(document)

    assert read.source_format == "docx"
    assert {node.kind for node in read.tree.walk()} >= {"paragraph", "table", "cell"}


def test_the_promoting_analyzers_are_opt_in(spreadsheet):
    """Not a preference: a reading comparable with the old structure kernel needs them."""
    assert read_tree(spreadsheet).promoted is False
    assert read_tree(spreadsheet, promote=True).promoted is True


def test_an_unclaimed_suffix_is_refused(tmp_path):
    from ragix_kernels.saqqara.adapters import UnsupportedFormat

    orphan = tmp_path / "note.rtf"
    orphan.write_text("nothing claims this")
    with pytest.raises(UnsupportedFormat):
        read_tree(orphan)


def test_what_the_builder_refused_to_place_is_returned_not_discarded():
    """The accounting itself — a drop nobody can read is a silent drop."""
    read = TreeRead(
        tree=None, source_format="xlsx", reader_version="0",
        drops=Counter({("unmapped-observation", "figure"): 2,
                       ("no-parent", "cell"): 1}),
    )
    assert read.dropped == 3
    assert read.drops[("unmapped-observation", "figure")] == 2


def test_a_clean_fixture_drops_nothing(spreadsheet, document):
    """The fixtures are read whole; a non-zero drop here is a regression, not noise."""
    assert read_tree(spreadsheet).dropped == 0
    assert read_tree(document).dropped == 0
