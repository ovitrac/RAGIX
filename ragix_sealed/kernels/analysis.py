"""
RAGIX-Sealed — Level-2 analysis kernels, deterministic v0 (WP §17-§18, Sprint 5).

These reason over COOLED, placeholderized chunks only and cite `doc_id/page/chunk_id`
(never filenames — K3/K4). v0 is deterministic (regex/co-occurrence/cue-words): it never
invents facts and needs no model. Model-router-backed analysis (incl. the contradiction
kernel, which needs semantics) is a clean follow-up.

Every finding is SANITIZED_LLM_SAFE: placeholders + citations only. A defensive leak
re-check runs on every emitted snippet (deny-by-default — K6/K7).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List

from ..corpus.chunking import CooledChunk
from ..ingest.leak_scan import scan as leak_scan

# Matches "[PERSON_001]" or "[DATE_017 | month external]" -> class, number, optional role.
PLACEHOLDER_RE = re.compile(r"\[([A-Z][A-Z0-9_]*?)_(\d+)(?:\s*\|\s*(?P<role>[^\]]+?))?\]")

# Deterministic commitment cue words (v0).
_COMMITMENT_CUES = [
    "shall", "must", "agrees to", "agree to", "undertakes", "undertake",
    "no later than", "deadline", "within", "is required to", "responsible for",
]
_COMMITMENT_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in _COMMITMENT_CUES) + r")\b", re.IGNORECASE)

# Referenced-artifact classes for gap detection (v0).
_REFERENCE_CLASSES = {"DOCUMENT", "CONTRACT"}


class AnalysisError(Exception):
    """Raised when a finding would carry non-cooled content (defensive boundary)."""


def _citation(ch: CooledChunk) -> str:
    return f"{ch.doc_id}/page_{ch.page}/{ch.chunk_id}"


def _placeholders(text: str):
    """Yield (canonical_token, entity_type, role) for each placeholder in text."""
    for m in PLACEHOLDER_RE.finditer(text):
        cls, num = m.group(1), m.group(2)
        yield f"[{cls}_{num}]", cls, (m.group("role") or None)


def _safe_snippet(schema: Dict[str, Any], text: str) -> str:
    """Return the snippet only if it passes the leak re-check; else raise (deny)."""
    if leak_scan(text, [], schema).verdict != "PASS":
        raise AnalysisError("analysis snippet failed boundary leak check")
    return text


@dataclass(frozen=True)
class AnalysisResult:
    """Level-2 result — SANITIZED_LLM_SAFE findings (placeholders + citations only)."""

    kernel: str
    findings: List[Dict[str, Any]] = field(default_factory=list)

    def to_public_dict(self) -> Dict[str, Any]:
        return {"kernel": self.kernel, "findings": self.findings}


class AnalysisKernel(ABC):
    name: str = "analysis"

    @abstractmethod
    def run(self, chunks: List[CooledChunk], schema: Dict[str, Any]) -> AnalysisResult:
        ...


class TimelineKernel(AnalysisKernel):
    """Lists DATE placeholders in order of appearance, with citations."""

    name = "timeline"

    def run(self, chunks, schema) -> AnalysisResult:
        events: List[Dict[str, Any]] = []
        for ch in chunks:
            for token, cls, _role in _placeholders(ch.text):
                if cls == "DATE":
                    events.append({
                        "date": token,
                        "citation": _citation(ch),
                        "snippet": _safe_snippet(schema, ch.text),
                    })
        return AnalysisResult(self.name, events)


class EntityRoleGraphKernel(AnalysisKernel):
    """Nodes (placeholders + roles) and co-occurrence edges within chunks."""

    name = "entity_role_graph"

    def run(self, chunks, schema) -> AnalysisResult:
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: Dict[tuple, Dict[str, Any]] = {}
        for ch in chunks:
            present = []
            for token, cls, role in _placeholders(ch.text):
                present.append(token)
                node = nodes.setdefault(token, {"placeholder": token, "entity_type": cls, "role": role})
                if role and not node.get("role"):
                    node["role"] = role
            for a, b in combinations(sorted(set(present)), 2):
                key = (a, b)
                edge = edges.setdefault(key, {"source": a, "target": b, "relation": "co_mentioned",
                                              "weight": 0, "citations": []})
                edge["weight"] += 1
                edge["citations"].append(_citation(ch))
        findings = [{"nodes": list(nodes.values()), "edges": list(edges.values())}]
        return AnalysisResult(self.name, findings)


class CommitmentKernelV0(AnalysisKernel):
    """Heuristic commitment candidates: a cue word + at least one entity. requires_review."""

    name = "commitment_v0"

    def run(self, chunks, schema) -> AnalysisResult:
        out: List[Dict[str, Any]] = []
        for ch in chunks:
            cues = sorted({m.group(0).lower() for m in _COMMITMENT_RE.finditer(ch.text)})
            entities = [t for t, _c, _r in _placeholders(ch.text)]
            if cues and entities:
                out.append({
                    "candidate": _safe_snippet(schema, ch.text),
                    "cues": cues,
                    "entities": sorted(set(entities)),
                    "citation": _citation(ch),
                    "confidence": 0.4,
                    "requires_review": True,
                })
        return AnalysisResult(self.name, out)


class GapDetectionKernelV0(AnalysisKernel):
    """Flags referenced DOCUMENT/CONTRACT placeholders for presence verification."""

    name = "gap_v0"

    def run(self, chunks, schema) -> AnalysisResult:
        refs: Dict[str, Dict[str, Any]] = {}
        for ch in chunks:
            for token, cls, _role in _placeholders(ch.text):
                if cls in _REFERENCE_CLASSES:
                    r = refs.setdefault(token, {"reference": token, "entity_type": cls,
                                                "citations": [], "requires_review": True})
                    r["citations"].append(_citation(ch))
        return AnalysisResult(self.name, list(refs.values()))


ALL_ANALYSIS_KERNELS = [
    TimelineKernel(),
    EntityRoleGraphKernel(),
    CommitmentKernelV0(),
    GapDetectionKernelV0(),
]
_REGISTRY = {k.name: k for k in ALL_ANALYSIS_KERNELS}

# Contradiction analysis needs semantics; deferred to a model-router-backed kernel.
DEFERRED_KERNELS = ["contradiction"]


class UnknownAnalysisKernelError(Exception):
    pass


def analysis_kernel_list() -> List[str]:
    return list(_REGISTRY.keys())


def run_analysis_kernel(name: str, chunks: List[CooledChunk], schema: Dict[str, Any]) -> AnalysisResult:
    kernel = _REGISTRY.get(name)
    if kernel is None:
        raise UnknownAnalysisKernelError(f"unknown analysis kernel: {name!r}")
    return kernel.run(chunks, schema)


def run_analysis(chunks: List[CooledChunk], schema: Dict[str, Any], names=None) -> Dict[str, list]:
    selected = names if names is not None else list(_REGISTRY.keys())
    return {name: run_analysis_kernel(name, chunks, schema).findings for name in selected}
