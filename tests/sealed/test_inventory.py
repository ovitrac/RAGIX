"""
Tests for the Sprint 4 inventory kernels (WP §16).

Metrics correctness over a synthetic CorpusView (cooled / blocked / review mix),
run_inventory over all kernels, and ORCHESTRATOR_METRICS-safety (no raw, JSON-serializable).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

import json

import pytest

from ragix_sealed.contracts import load_contracts
from ragix_sealed.ingest import IngestStatus
from ragix_sealed.kernels import (
    CorpusView,
    UnknownKernelError,
    kernel_list,
    run_inventory,
    run_inventory_kernel,
)


def _status(doc_id, state, kind="txt", entities=None, verdict="PASS", review=False, pages=1):
    return IngestStatus(
        doc_id=doc_id, state=state, source_kind=kind,
        entity_counts=entities or {}, leak_verdict=verdict,
        human_review_required=review, pages=pages,
    )


def _view():
    docs = [
        _status("doc_a", "COOLED_INDEXABLE", "txt", {"EMAIL": 2, "DATE": 1}, "PASS", False, 1),
        _status("doc_b", "COOLED_INDEXABLE", "md", {"PHONE": 1}, "UNCERTAIN", True, 3),
        _status("doc_c", "BLOCKED", "txt", {}, "FAIL", False, 2),
    ]
    return CorpusView.from_statuses(docs, contracts=load_contracts(), chunk_count=7)


def test_corpus_metrics():
    m = run_inventory_kernel("corpus_metrics", _view()).metrics
    assert m["documents_total"] == 3
    assert m["documents_indexable"] == 2
    assert m["documents_blocked"] == 1
    assert m["pages_total"] == 6
    assert m["human_review_required"] == 1
    assert m["chunks_total"] == 7
    assert m["source_kinds"] == {"txt": 2, "md": 1}


def test_entity_inventory_aggregates():
    m = run_inventory_kernel("entity_inventory", _view()).metrics
    assert m["entity_counts"] == {"EMAIL": 2, "DATE": 1, "PHONE": 1}


def test_quality_risk():
    m = run_inventory_kernel("quality_risk", _view()).metrics
    assert m["leak_verdicts"] == {"PASS": 1, "UNCERTAIN": 1, "FAIL": 1}
    assert m["blocked"] == 1
    assert m["review_required"] == 1
    assert m["uncertain"] == 1


def test_review_queue_lists_opaque_ids_only():
    m = run_inventory_kernel("review_queue", _view()).metrics
    assert m["review_required_docs"] == ["doc_b"]
    assert m["blocked_docs"] == ["doc_c"]


def test_run_inventory_runs_all_and_is_serializable():
    results = run_inventory(_view())
    assert set(results.keys()) == set(kernel_list())
    # ORCHESTRATOR_METRICS-safe: serializable, numbers/ids only.
    json.dumps(results)


def test_unknown_kernel_raises():
    with pytest.raises(UnknownKernelError):
        run_inventory_kernel("does_not_exist", _view())
