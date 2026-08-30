"""
The vector index's result shape, and who it excludes.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

`SearchResult` was written for code chunks and requires `file_path`, `start_line`,
`end_line`, `chunk_type` and `name`. `NumpyVectorIndex.search` reads them as direct
subscripts, so any caller whose chunks are not code raises `KeyError: 'file_path'`
— a document has no start line, and filling one in with 0 would be inventing data
rather than adapting to it.

Two tests, in this order and for this reason:

1. the existing caller's results must not move by one byte. The change below is
   additive or it is a regression, and only a fixed expected shape can tell those
   apart;
2. a caller with no code fields must search and receive the empty defaults rather
   than an exception.

The first passes before and after. The second is the one that was red.
"""

from __future__ import annotations

import pytest

from ragix_core.vector_index import NumpyVectorIndex, SearchResult

#: One code chunk, exactly as an existing caller supplies it.
CODE_METADATA = {
    "chunk_id": "chunk-1",
    "file_path": "src/app/Service.java",
    "start_line": 10,
    "end_line": 42,
    "chunk_type": "method",
    "name": "doTheThing",
    "metadata": {"language": "java"},
}

#: What an existing caller must keep receiving, field by field.
EXPECTED_CODE_RESULT = {
    "chunk_id": "chunk-1",
    "file_path": "src/app/Service.java",
    "start_line": 10,
    "end_line": 42,
    "chunk_type": "method",
    "name": "doTheThing",
    "metadata": {"language": "java"},
}


def _fields(result: SearchResult) -> dict:
    """Every field but the score, which is a float and compared separately."""
    return {
        "chunk_id": result.chunk_id,
        "file_path": result.file_path,
        "start_line": result.start_line,
        "end_line": result.end_line,
        "chunk_type": result.chunk_type,
        "name": result.name,
        "metadata": result.metadata,
    }


def test_an_existing_code_caller_gets_exactly_what_it_got_before():
    """The additivity check: this must pass before AND after the change.

    Falsified by: any field of a code caller's result changing value, name or type.
    """
    index = NumpyVectorIndex(dimension=3)
    index.add([[1.0, 0.0, 0.0]], [dict(CODE_METADATA)])

    results = index.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 1
    assert _fields(results[0]) == EXPECTED_CODE_RESULT
    assert results[0].score == pytest.approx(1.0)


def test_two_code_chunks_keep_their_order_and_their_fields():
    """Ranking is untouched: the change is about absent fields, not about scoring."""
    index = NumpyVectorIndex(dimension=3)
    index.add(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [dict(CODE_METADATA), {**CODE_METADATA, "chunk_id": "chunk-2", "name": "other"}],
    )
    results = index.search([1.0, 0.0, 0.0], k=2)
    assert [r.chunk_id for r in results] == ["chunk-1", "chunk-2"]
    assert results[0].score > results[1].score


def test_a_caller_with_no_code_fields_searches_and_gets_empty_defaults():
    """A document chunk has no line numbers, so it supplies none.

    It must receive the empty defaults rather than an exception, and the defaults
    must be empty rather than invented: "" and 0 say "this caller has no such
    field", where a fabricated path or line number would read as a fact.

    Falsified by: a KeyError, or a default that is not empty.
    """
    index = NumpyVectorIndex(dimension=3)
    index.add([[0.0, 0.0, 1.0]], [{"chunk_id": "doc-chunk-1"}])

    results = index.search([0.0, 0.0, 1.0], k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "doc-chunk-1"
    assert results[0].file_path == ""
    assert results[0].start_line == 0
    assert results[0].end_line == 0
    assert results[0].chunk_type == ""
    assert results[0].name == ""
    assert results[0].metadata == {}


def test_a_caller_may_carry_its_own_metadata_without_the_code_fields():
    """The generic `metadata` dict stays available to whoever is not code."""
    index = NumpyVectorIndex(dimension=2)
    index.add([[1.0, 0.0]], [{"chunk_id": "doc-2", "metadata": {"doc_id": "abc", "level": 1}}])

    result = index.search([1.0, 0.0], k=1)[0]
    assert result.metadata == {"doc_id": "abc", "level": 1}
    assert result.file_path == ""
