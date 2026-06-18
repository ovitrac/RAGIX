"""
Tests for Sprint 2bis multimodal support (WP §8bis).

Source classification + the provenance graph (opaque ids, parent links, schema validity,
no-raw boundary, lineage). Synthetic only.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

import json

import pytest

from ragix_sealed.contracts import load_contracts
from ragix_sealed.ingest import new_case_context
from ragix_sealed.multimodal import (
    ProvenanceError,
    ProvenanceGraph,
    ProvenanceNode,
    SourceClass,
    asset_id,
    caption_id,
    classify_source,
)


# -- source classification ---------------------------------------------------------------

def test_classify_by_extension():
    assert classify_source("txt") is SourceClass.TEXT_DOCUMENT
    assert classify_source("png") is SourceClass.IMAGE
    assert classify_source("mp4") is SourceClass.VIDEO
    assert classify_source("wav") is SourceClass.AUDIO
    assert classify_source("pdf") is SourceClass.SCANNED_DOCUMENT


def test_classify_by_magic_when_kind_unknown():
    assert classify_source("", b"%PDF-1.7 ...") is SourceClass.SCANNED_DOCUMENT
    assert classify_source("", b"\x89PNG\r\n\x1a\n") is SourceClass.IMAGE
    assert classify_source("", b"nope") is SourceClass.UNKNOWN


# -- provenance ids ----------------------------------------------------------------------

def test_derivative_ids_opaque_and_stable():
    ctx = new_case_context("case_mm_001")
    a1 = asset_id(ctx, "doc_abc", "image")
    a2 = asset_id(ctx, "doc_abc", "image")
    assert a1 == a2 and a1.startswith("asset_")
    cap = caption_id(ctx, "doc_abc", a1, "vlm", "v1")
    assert cap.startswith("caption_") and "doc_abc" not in cap


# -- provenance graph --------------------------------------------------------------------

def _node(**kw):
    base = dict(
        artifact_id="caption_001", artifact_type="DERIVED_CAPTION_PLACEHOLDERIZED",
        case_id="case_mm_001", source_id="doc_abc", state="DERIVED_COOLED_DESCRIPTOR",
        hash="sha256:deadbeef", aad={"case_id": "case_mm_001", "source_id": "doc_abc"},
    )
    base.update(kw)
    return ProvenanceNode(**base)


def test_graph_add_and_lineage():
    c = load_contracts()
    g = ProvenanceGraph(c)
    g.add(_node(artifact_id="img_001", artifact_type="IMAGE", state="DERIVED_RAW_INTERNAL"))
    g.add(_node(artifact_id="caption_001", parent_artifact_id="img_001"))
    assert g.lineage("caption_001") == ["caption_001", "img_001"]


def test_graph_rejects_unknown_parent():
    c = load_contracts()
    g = ProvenanceGraph(c)
    with pytest.raises(ProvenanceError, match="unknown parent"):
        g.add(_node(parent_artifact_id="ghost"))


def test_graph_public_node_has_no_raw_fields():
    c = load_contracts()
    g = ProvenanceGraph(c)
    node = _node()
    g.add(node)
    forbidden = set(c.provenance["forbidden_fields"])
    assert set(node.to_public_dict().keys()).isdisjoint(forbidden)
    # serializable, no raw markers
    json.dumps(node.to_public_dict())
