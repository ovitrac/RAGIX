"""tender.pyramid — the resolution objects, their K, their deterministic summary, zoom and rollup.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Block A of the demo's day: the deadline at three resolutions, built from claims that already exist and
**with no model at all**. The summary here is the extractive-deterministic floor of §5 — sentences composed
from the claims' own typed values, every number a claim reference, every sentence mapped to the children it
came from. A generated summary replaces it later without changing anything below.

  K_b = A(K_children) + I_b        aggregation is a reference join; interpretation adds no critical value.

`zoom` walks a summary sentence down to its children and, at the leaf, to the exact spans; `rollup` walks
back. Nothing here writes to saqqara.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .claims import ClaimRecord

CONTAINS = "contains"
SUMMARISES = "summarises"


@dataclass(frozen=True)
class Sentence:
    """One sentence of a node's summary, with the children and claims it was composed from."""

    index: int
    text: str
    children: tuple[str, ...]
    claims: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"sentence": self.index, "text": self.text, "children": list(self.children),
                "claims": list(self.claims)}


def _span(claim: ClaimRecord) -> dict[str, Any]:
    """The span a claim's own text was read from: its first source, which is the one `raw` may not match."""
    s = claim.provenance.sources[0]
    return {"doc_id": s.doc_id, "chunk_id": s.chunk_id, "char_start": s.char_start,
            "char_end": s.char_end, "span_sha256": s.span_sha256}


def _sources(claim: ClaimRecord) -> list[dict[str, Any]]:
    """Every operand's span. A derived claim's `raw` belongs to one of these, not necessarily the first."""
    return [{"doc_id": s.doc_id, "chunk_id": s.chunk_id, "char_start": s.char_start,
             "char_end": s.char_end, "span_sha256": s.span_sha256} for s in claim.provenance.sources]


def aggregate(claims: Iterable[ClaimRecord]) -> dict[str, Any]:
    """A(K_children): every child claim kept, by reference, grouped by field and value. Nothing is dropped."""
    fields: dict[str, dict[str, list[str]]] = {}
    for claim in claims:
        fields.setdefault(claim.field, {}).setdefault(claim.value.normalized, []).append(claim.claim_id)
    return {"fields": {f: {v: sorted(ids) for v, ids in sorted(values.items())}
                       for f, values in sorted(fields.items())},
            "claims": sorted(c.claim_id for c in claims)}


def document_node(doc_id: str, piece: str, claims: list[ClaimRecord]) -> dict[str, Any]:
    """One document's K: its own claims aggregated, and whether its value stands alone in the corpus."""
    k = aggregate(claims)
    k["piece"] = piece
    k["interpreted"] = {}
    return k


def dce_node(children: dict[str, dict[str, Any]], resolution: dict[str, Any],
             dependents: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    """The DCE's K: the children joined by reference, plus the branch layer — and no critical value of its own.

    The interpreted part carries the conflict, the authority that settles it and the verdict; the resolved
    value itself is a **reference** to the governing claims, not a value written here.
    """
    joined: dict[str, dict[str, list[str]]] = {}
    for child in children.values():
        for field, values in child["fields"].items():
            for value, ids in values.items():
                joined.setdefault(field, {}).setdefault(value, []).extend(ids)
    fields = {f: {v: sorted(set(ids)) for v, ids in sorted(values.items())} for f, values in sorted(joined.items())}
    offer = fields.get("offer_deadline", {})
    governing = sorted(resolution.get("governing", []))
    displaced = sorted({cid for value, ids in offer.items() for cid in ids if cid not in set(governing)})
    questions = [e["claim_id"] for e in (dependents.get("questions_deadline") or [])]
    window = dependents.get("visit_window") or {}

    # derived: computed from operands, with the expression and the operands' claim ids
    derived: dict[str, Any] = {}
    if window.get("days_end_after_questions") is not None:
        derived["days_visit_end_after_questions"] = {
            "value": window["days_end_after_questions"],
            "expression": "visit_window_end − questions_deadline, in days",
            "operands": {"questions_deadline": questions[:1], "visit_window_end": "see the leaf claims"},
        }

    # interpreted: flags, labels and references only — never a date, an amount or a quantity (ClaimRecord 1.1)
    interpreted = {
        "conflict": len(offer) > 1,
        "conflicting_value_count": len(offer),
        "governing_claims": governing,
        "displaced_claims": displaced,
        "verdict": resolution.get("verdict"),
        "rendering": resolution.get("rendering"),
        "authority": {"surfaced": sorted(a["label"] for a in authority.get("authorities", [])),
                      "ranks": authority.get("ranks", {}),
                      "coverage_sufficient": authority.get("sufficient")},
        "dependent_claims": {"questions_deadline": questions},
    }
    return {"fields": fields,
            "claims": sorted({i for v in fields.values() for ids in v.values() for i in ids}),
            "derived": derived, "interpreted": interpreted}


def _value_of(claim_ids: list[str], by_id: dict[str, ClaimRecord]) -> Optional[str]:
    for cid in claim_ids:
        if cid in by_id:
            return by_id[cid].value.normalized
    return None


def dce_sentences(k: dict[str, Any], by_id: dict[str, ClaimRecord], children: list[str]) -> list[Sentence]:
    """The DCE summary, composed from the claims — no number is typed, every sentence maps to children."""
    interpreted, derived = k["interpreted"], k.get("derived", {})
    governing = list(interpreted["governing_claims"])
    displaced = list(interpreted["displaced_claims"])
    resolved = _value_of(governing, by_id)
    out: list[Sentence] = []
    if resolved:
        out.append(Sentence(
            len(out) + 1,
            f"La date limite de remise des plis retenue est le {resolved}, établie par l'autorité applicable "
            f"({', '.join(interpreted['authority']['surfaced'])}) ; {len(displaced)} autres pièces portent une "
            f"valeur différente, conservée et déclassée.",
            tuple(children), tuple(governing + displaced[:3])))
    questions = interpreted["dependent_claims"]["questions_deadline"]
    question_value = _value_of(questions, by_id)
    if question_value:
        out.append(Sentence(
            len(out) + 1,
            f"Les questions à l'acheteur sont dues le {question_value}, dérivé de cette date limite.",
            tuple(children), tuple(questions[:1])))
    days = (derived.get("days_visit_end_after_questions") or {}).get("value")
    end_claims = [c for c in k["fields"].get("visit_window_end", {}).values()]
    end_ids = sorted({i for ids in end_claims for i in ids})
    end_value = _value_of(end_ids, by_id)
    if days is not None and end_value:
        out.append(Sentence(
            len(out) + 1,
            f"La visite de site se ferme le {end_value}, soit {days} jours après l'échéance des questions : "
            f"une visite faite ces jours-là ne peut plus donner lieu à une question.",
            tuple(children), tuple(questions[:1] + end_ids[:1])))
    return out


def document_sentences(k: dict[str, Any], by_id: dict[str, ClaimRecord]) -> list[Sentence]:
    out: list[Sentence] = []
    for field, values in k["fields"].items():
        for value, ids in values.items():
            out.append(Sentence(len(out) + 1,
                                f"{k['piece']} : {field} = {value}.", tuple(ids), tuple(ids)))
    return out


def zoom(node_id: str, sentence: Sentence, nodes: dict[str, dict[str, Any]],
         by_id: dict[str, ClaimRecord]) -> dict[str, Any]:
    """One summary sentence down to its children, their claims, and the exact spans underneath."""
    steps = [{"level": "dce", "node": node_id, "sentence": sentence.index, "text": sentence.text}]
    for child in sentence.children:
        child_k = nodes[child]
        steps.append({"level": "document", "node": child, "piece": child_k["piece"],
                      "claims": sorted({i for v in child_k["fields"].values() for ids in v.values() for i in ids})})
    spans = []
    for cid in sentence.claims:
        claim = by_id[cid]
        spans.append({"claim_id": cid, "field": claim.field, "value": claim.value.normalized,
                      "origin": claim.provenance.origin, "span": _span(claim),
                      "sources": _sources(claim), "raw": claim.value.raw})
    steps.append({"level": "leaf", "spans": spans})
    return {"steps": steps}


def rollup(claim_ids: Iterable[str], nodes: dict[str, dict[str, Any]]) -> list[str]:
    """The nodes whose K references these claims: the walk back up."""
    wanted = set(claim_ids)
    return sorted(node for node, k in nodes.items() if wanted & set(k.get("claims", [])))
