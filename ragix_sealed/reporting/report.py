"""
RAGIX-Sealed — report builders (WP §19-§20, Sprint 6).

Assembles reviewer-facing reports from inventory metrics + analysis findings. Reports are
built in their SANITIZED form (placeholders + citations only); export mode is applied
separately (see ``export.py``). No builder ever emits raw content.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ReportSection:
    heading: str
    lines: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SealedReport:
    """A sanitized report: placeholders + citations only."""

    title: str
    sections: List[ReportSection] = field(default_factory=list)

    def to_markdown(self) -> str:
        out = [f"# {self.title}", ""]
        for sec in self.sections:
            out.append(f"## {sec.heading}")
            out.extend(sec.lines if sec.lines else ["_(none)_"])
            out.append("")
        return "\n".join(out).rstrip() + "\n"


def build_sanitized_memo(inventory: Dict[str, dict], analysis: Dict[str, list]) -> SealedReport:
    """Sanitized case memo from inventory metrics + analysis findings (WP §20.1)."""
    cm = inventory.get("corpus_metrics", {})
    sections: List[ReportSection] = []

    sections.append(ReportSection("Corpus Summary", [
        f"- Documents total: {cm.get('documents_total', 0)}",
        f"- Indexable (cooled): {cm.get('documents_indexable', 0)}",
        f"- Blocked: {cm.get('documents_blocked', 0)}",
        f"- Pages total: {cm.get('pages_total', 0)}",
        f"- Documents needing review: {cm.get('human_review_required', 0)}",
    ]))

    graph = (analysis.get("entity_role_graph") or [{}])[0]
    actors = [
        f"- {n['placeholder']}" + (f" — {n['role']}" if n.get("role") else "")
        for n in graph.get("nodes", [])
    ]
    sections.append(ReportSection("Key Actors by Role", actors))

    chrono = [f"- {e['date']} ({e['citation']})" for e in analysis.get("timeline", [])]
    sections.append(ReportSection("Chronology", chrono))

    commitments = [
        f"- candidate ({c['citation']}); cues: {', '.join(c['cues'])}; review required"
        for c in analysis.get("commitment_v0", [])
    ]
    sections.append(ReportSection("Commitments (candidates — review required)", commitments))

    gaps = [f"- {g['reference']} referenced — verify presence" for g in analysis.get("gap_v0", [])]
    sections.append(ReportSection("Referenced Artifacts (verify presence)", gaps))

    rq = inventory.get("review_queue", {})
    review = [f"- review: {d}" for d in rq.get("review_required_docs", [])] + \
             [f"- blocked: {d}" for d in rq.get("blocked_docs", [])]
    sections.append(ReportSection("Review Points", review))

    return SealedReport(title="Sanitized Case Memo", sections=sections)


def build_commitment_matrix(analysis: Dict[str, list]) -> SealedReport:
    """A simple commitment matrix (WP §20.4) from commitment_v0 findings."""
    rows = ["| Candidate | Cues | Entities | Source | Review |", "|---|---|---|---|---|"]
    for c in analysis.get("commitment_v0", []):
        rows.append(
            f"| candidate | {', '.join(c['cues'])} | {', '.join(c['entities'])} | "
            f"{c['citation']} | yes |"
        )
    return SealedReport(title="Commitment Matrix", sections=[ReportSection("Commitments", rows)])


def build_audit_attestation(
    case_id: str,
    inventory: Dict[str, dict],
    hash_chain_tip: str,
    policy_version: str,
    placeholder_schema_version: str,
    kernel_versions: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Audit attestation (WP §20.5): provenance/metrics/versions only — no content."""
    cm = inventory.get("corpus_metrics", {})
    return {
        "case_id": case_id,
        "documents_total": cm.get("documents_total", 0),
        "documents_indexable": cm.get("documents_indexable", 0),
        "documents_blocked": cm.get("documents_blocked", 0),
        "policy_version": policy_version,
        "placeholder_schema_version": placeholder_schema_version,
        "kernel_versions": kernel_versions or {},
        "hash_chain_tip": hash_chain_tip,
    }
