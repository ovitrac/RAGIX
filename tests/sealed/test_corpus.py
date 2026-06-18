"""
Tests for the Sprint 3 cooled corpus index (WP §13).

End-to-end ingest → index → safe search, opaque/stable chunk ids, no-raw snippets,
citation form, and the defensive leak re-check at the index boundary. Synthetic only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

import json

import pytest

from ragix_sealed.contracts import load_contracts
from ragix_sealed.corpus import CooledCorpusIndex, CorpusError, chunk_cooled
from ragix_sealed.ingest import SealedIngestor, new_case_context
from ragix_sealed.vault import RAGIXSealedVaultBackend

SYNTH = (
    "Memo dated 2024-03-15 about the project schedule.\n\n"
    "Contact jane.synthetic@example.com regarding the milestone review.\n\n"
    "Budget total 1 250,00 EUR under reference n° 12-3456."
).encode("utf-8")

RAW_BITS = ["jane.synthetic@example.com", "2024-03-15", "12-3456"]


def _cooled_text():
    ing = SealedIngestor(load_contracts(), RAGIXSealedVaultBackend())
    ctx = new_case_context("case_idx_001")
    status = ing.ingest(ctx, SYNTH, "txt", ingestion_counter=1)
    return ctx, status.doc_id, ing.get_cooled(status.doc_id).text


# -- chunking ----------------------------------------------------------------------------

def test_chunk_ids_opaque_and_stable():
    ctx = new_case_context("case_idx_001")
    text = "Block one about [PERSON_001].\n\nBlock two about [ORG_001]."
    c1 = chunk_cooled(ctx, "doc_xyz", text)
    c2 = chunk_cooled(ctx, "doc_xyz", text)
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]   # stable
    assert all(c.chunk_id.startswith("chunk_") for c in c1)
    assert all("doc_xyz" not in c.chunk_id for c in c1)           # opaque


# -- index + search ----------------------------------------------------------------------

def test_index_and_search_returns_placeholderized_snippet():
    c = load_contracts()
    ctx, doc_id, cooled = _cooled_text()
    idx = CooledCorpusIndex(c)
    added = idx.add_document(ctx, doc_id, cooled)
    assert added and idx.chunk_count == len(added)

    hits = idx.search("milestone review", k=5)
    assert hits, "expected at least one hit"
    top = hits[0]
    assert top.doc_id == doc_id
    assert top.chunk_id.startswith("chunk_")
    # snippet is placeholderized; no raw values present
    for raw in RAW_BITS:
        assert raw not in top.snippet
    assert "[EMAIL_" in cooled  # sanity: the cooled text did get placeholderized


def test_citation_form():
    c = load_contracts()
    ctx, doc_id, cooled = _cooled_text()
    idx = CooledCorpusIndex(c)
    idx.add_document(ctx, doc_id, cooled)
    hit = idx.search("budget", k=3)[0]
    assert hit.citation() == f"{doc_id}/page_{hit.page}/{hit.chunk_id}"
    # public dict carries no raw values
    blob = json.dumps(hit.to_public_dict())
    for raw in RAW_BITS:
        assert raw not in blob


def test_empty_query_returns_nothing():
    c = load_contracts()
    ctx, doc_id, cooled = _cooled_text()
    idx = CooledCorpusIndex(c)
    idx.add_document(ctx, doc_id, cooled)
    assert idx.search("", k=5) == []


# -- defensive boundary check ------------------------------------------------------------

def test_index_rejects_chunk_with_residual_raw():
    """If un-cooled text reaches the index (raw email present), it is rejected (K6/K7)."""
    c = load_contracts()
    ctx = new_case_context("case_idx_001")
    leaky = "This block still contains raw@example.com which must never be indexed."
    idx = CooledCorpusIndex(c)
    with pytest.raises(CorpusError, match="boundary leak check"):
        idx.add_document(ctx, "doc_leak", leaky)
    assert idx.chunk_count == 0
