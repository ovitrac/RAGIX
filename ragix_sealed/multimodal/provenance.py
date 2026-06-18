"""
RAGIX-Sealed — derivative provenance graph (WP §8bis.2/8bis.3, Sprint 2bis).

Every source and derivative (frames, OCR, captions, transcripts) gets an opaque,
case-bound id and a provenance node that points to its parent. The graph replaces
human-readable filenames as the traceability backbone — and stays public-facing-safe:
nodes carry no raw content, validated against `provenance.schema.yaml`.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ingest.ids import CaseContext


class ProvenanceError(Exception):
    """Raised on an invalid or inconsistent provenance node."""


# Convenience id derivations (WP §8bis.2). All opaque, case-bound, stable.
def asset_id(ctx: CaseContext, source_id: str, asset_type: str, version: str = "1") -> str:
    return ctx.derive_id("asset", source_id, asset_type, version)


def frame_id(ctx: CaseContext, source_id: str, timestamp: float, index: int, profile: str) -> str:
    return ctx.derive_id("frame", source_id, timestamp, index, profile)


def segment_id(ctx: CaseContext, source_id: str, t_start: float, t_end: float, profile: str) -> str:
    return ctx.derive_id("segment", source_id, t_start, t_end, profile)


def ocr_id(ctx: CaseContext, source_id: str, page_or_frame: str, profile: str) -> str:
    return ctx.derive_id("ocr", source_id, page_or_frame, profile)


def caption_id(ctx: CaseContext, source_id: str, asset: str, model: str, profile: str) -> str:
    return ctx.derive_id("caption", source_id, asset, model, profile)


def transcript_id(ctx: CaseContext, source_id: str, audio: str, model: str, profile: str) -> str:
    return ctx.derive_id("transcript", source_id, audio, model, profile)


@dataclass(frozen=True)
class ProvenanceNode:
    """A public-facing provenance node. Carries no raw content (WP §8bis.3)."""

    artifact_id: str
    artifact_type: str            # e.g. IMAGE, DERIVED_OCR, DERIVED_CAPTION_PLACEHOLDERIZED
    case_id: str
    source_id: str
    state: str                    # a derivative_flow state
    hash: str                     # sha256 of the (sealed) artifact, for audit
    aad: Dict[str, Any]           # binding object (no raw values)
    parent_artifact_id: Optional[str] = None
    derivation: Optional[Dict[str, Any]] = None  # {tool, model, profile, created_at}
    citations: List[Dict[str, Any]] = field(default_factory=list)

    def to_public_dict(self) -> Dict[str, Any]:
        d = {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "case_id": self.case_id,
            "source_id": self.source_id,
            "state": self.state,
            "hash": self.hash,
            "aad": self.aad,
        }
        if self.parent_artifact_id is not None:
            d["parent_artifact_id"] = self.parent_artifact_id
        if self.derivation is not None:
            d["derivation"] = self.derivation
        if self.citations:
            d["citations"] = self.citations
        return d


class ProvenanceGraph:
    """In-memory provenance graph, validated against the provenance contract."""

    def __init__(self, contracts: Any) -> None:
        schema = contracts.provenance
        self._required = list(schema.get("artifact_required_fields", []))
        self._forbidden = set(schema.get("forbidden_fields", []))
        self._nodes: Dict[str, ProvenanceNode] = {}

    def add(self, node: ProvenanceNode) -> None:
        public = node.to_public_dict()
        # Required fields present.
        missing = [f for f in self._required if f not in public]
        if missing:
            raise ProvenanceError(f"node {node.artifact_id} missing required fields: {missing}")
        # No forbidden (raw) field names.
        leak = set(public.keys()) & self._forbidden
        if leak:
            raise ProvenanceError(f"node {node.artifact_id} exposes forbidden fields: {sorted(leak)}")
        # Parent must exist if referenced.
        if node.parent_artifact_id and node.parent_artifact_id not in self._nodes:
            raise ProvenanceError(
                f"node {node.artifact_id} references unknown parent {node.parent_artifact_id}"
            )
        if node.artifact_id in self._nodes:
            raise ProvenanceError(f"duplicate artifact_id: {node.artifact_id}")
        self._nodes[node.artifact_id] = node

    def get(self, artifact_id: str) -> Optional[ProvenanceNode]:
        return self._nodes.get(artifact_id)

    def lineage(self, artifact_id: str) -> List[str]:
        """Return the chain of artifact_ids from this node up to its root source."""
        chain: List[str] = []
        cur: Optional[str] = artifact_id
        seen = set()
        while cur and cur in self._nodes and cur not in seen:
            chain.append(cur)
            seen.add(cur)
            cur = self._nodes[cur].parent_artifact_id
        return chain
