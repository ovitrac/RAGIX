"""
Tests for the Sprint 5 analysis kernels v0 (WP §18).

Deterministic timeline / entity-role graph / commitment / gap detection over cooled
chunks, citation form, no-raw findings, and the defensive boundary check. Synthetic only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

import json

import pytest

from ragix_sealed.contracts import load_contracts
from ragix_sealed.corpus.chunking import CooledChunk
from ragix_sealed.kernels import (
    AnalysisError,
    UnknownAnalysisKernelError,
    analysis_kernel_list,
    run_analysis,
    run_analysis_kernel,
)

SCHEMA = load_contracts().placeholder_schema


def _chunk(block, text):
    return CooledChunk(chunk_id=f"chunk_{block:02d}", doc_id="doc_x", page=1, block=block, text=text)


COOLED_CHUNKS = [
    _chunk(1, "On [DATE_017] [PERSON_001 | primary_party] notified [ORG_001 | organization]."),
    _chunk(2, "[ORG_001] shall deliver [DOCUMENT_004] no later than [DATE_018]."),
    _chunk(3, "Reference to [CONTRACT_002] appears here with [PERSON_001]."),
]


def test_timeline_extracts_dates_with_citations():
    res = run_analysis_kernel("timeline", COOLED_CHUNKS, SCHEMA)
    dates = {f["date"] for f in res.findings}
    assert {"[DATE_017]", "[DATE_018]"} <= dates
    assert all(f["citation"].startswith("doc_x/page_1/chunk_") for f in res.findings)


def test_entity_role_graph_nodes_and_edges():
    res = run_analysis_kernel("entity_role_graph", COOLED_CHUNKS, SCHEMA)
    graph = res.findings[0]
    node_ids = {n["placeholder"] for n in graph["nodes"]}
    assert "[PERSON_001]" in node_ids and "[ORG_001]" in node_ids
    # role captured from the labelled occurrence
    person = next(n for n in graph["nodes"] if n["placeholder"] == "[PERSON_001]")
    assert person["role"] == "primary_party"
    # co-occurrence edge between PERSON_001 and ORG_001 (chunk 1)
    pairs = {(e["source"], e["target"]) for e in graph["edges"]}
    assert ("[ORG_001]", "[PERSON_001]") in pairs or ("[PERSON_001]", "[ORG_001]") in pairs


def test_commitment_detects_cue_and_entity():
    res = run_analysis_kernel("commitment_v0", COOLED_CHUNKS, SCHEMA)
    assert res.findings, "expected a commitment candidate"
    f = res.findings[0]
    assert "shall" in f["cues"] or "no later than" in f["cues"]
    assert f["requires_review"] is True
    assert f["citation"].startswith("doc_x/page_1/")


def test_gap_detection_flags_document_and_contract_refs():
    res = run_analysis_kernel("gap_v0", COOLED_CHUNKS, SCHEMA)
    refs = {f["reference"] for f in res.findings}
    assert "[DOCUMENT_004]" in refs and "[CONTRACT_002]" in refs


def test_findings_have_no_raw_and_serialize():
    results = run_analysis(COOLED_CHUNKS, SCHEMA)
    assert set(results.keys()) == set(analysis_kernel_list())
    blob = json.dumps(results)
    # snippets are placeholderized; no raw email/name patterns should appear
    assert "@" not in blob


def test_defensive_boundary_rejects_raw_snippet():
    leaky = [_chunk(9, "Contact raw@example.com on [DATE_001].")]
    with pytest.raises(AnalysisError, match="leak check"):
        run_analysis_kernel("timeline", leaky, SCHEMA)


def test_unknown_analysis_kernel_raises():
    with pytest.raises(UnknownAnalysisKernelError):
        run_analysis_kernel("contradiction", COOLED_CHUNKS, SCHEMA)  # deferred, not registered
