"""tender.authority — the deadline slice's authority records, its two gates, its resolution (WP-3D 3.4a-D).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

§I.6 of base/WP_TENDER_LAYERS_20260910.md and nothing wider. Two authority records, each an
Applicability 1.1 authority with the span that establishes it:

  RC 6    "Les plis devront parvenir … indiquées sur la page de garde du présent document." The piece
          carrying this clause states that its own cover governs the offer deadline.
  CCAP 2  "… les pièces contractuelles … prévalent dans cet ordre de priorité : …" The order of the
          pieces; rank 1 governs.

The invariant — no conclusion while an unresolved applicable authority can alter the value — is two
gates:

  authority discovery   the field's authorities are sought across the pieces; what is found is kept
                        with its span, and the coverage is recorded: every piece carrying a claim of
                        the field must be ranked by a found authority;
  conclusion blocking   a missing authority, an unranked piece or two authorities that disagree yield
                        an abstention, never a value. Coverage that cannot be shown sufficient is not
                        permission to conclude.

A piece's kind is read from its own cover title, span kept, never from a file name — the ranking
clause names other pieces before its own header, so only the cover can say which piece a document is.
A title the ranking does not name leaves that piece unranked, which abstains. No model here, and no
sixth verdict: the vocabulary is tender.contract's.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Optional

from .claims import ClaimRecord, sha256_text
from .contract import HUMAN_RENDERING

FIELD = "offer_deadline"
RC6, CCAP2 = "RC 6", "CCAP 2"
EXPECTED_AUTHORITIES = (RC6, CCAP2)
RANKING_WINDOW = 3000

PIECE_KINDS = (("RC", "Règlement de la Consultation"),
               ("AE", "Acte d'Engagement"),
               ("CCAP", "Cahier des Clauses Administratives Particulières"),
               ("CCTP", "Cahier des Clauses Techniques Particulières"))

_CLASSES = {"é": "[ée]", "è": "[èe]", "à": "[àa]", "û": "[ûu]", "ç": "[çc]", "ê": "[êe]", "'": "['’]"}


def _loose(phrase: str) -> str:
    """The phrase with whitespace (line breaks included) allowed between any two characters."""
    parts = [r"\s+" if ch == " " else _CLASSES.get(ch, re.escape(ch)) for ch in phrase]
    out = parts[0]
    for left, right in zip(parts, parts[1:]):
        out += ("" if r"\s+" in (left, right) else r"\s*") + right
    return out


_PIECE_RE = {kind: re.compile(_loose(name), re.IGNORECASE) for kind, name in PIECE_KINDS}
_RANKING_CONTEXT = re.compile(_loose("pièces contractuelles"), re.IGNORECASE)
_RANKING_HEAD = re.compile(_loose("ordre de priorité") + r"\s*:", re.IGNORECASE)
_RC6 = re.compile(_loose("Les plis devront parvenir") + r"[\s\S]{0,240}?"
                  + _loose("page de garde du présent document"), re.IGNORECASE)


def authority_span(chunk, start: int, end: int) -> dict[str, Any]:
    """A provenance span in the shape Applicability 1.1's authority keeps (no node ids, §I.7)."""
    return {"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "char_start": start, "char_end": end,
            "span_sha256": sha256_text(chunk.text[start:end])}


@dataclass(frozen=True)
class AuthorityRecord:
    label: str                              # RC 6 | CCAP 2
    kind: str                               # own_cover_governs | piece_ranking
    field: str
    span: dict[str, Any]                    # the span that establishes it
    ranks: dict[str, int]                   # piece kind -> rank (1 governs); empty for RC 6
    rank_spans: dict[str, dict[str, Any]]   # per ranked kind, the span of its name in the list

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "kind": self.kind, "field": self.field, "span": self.span,
                "ranks": self.ranks, "rank_spans": self.rank_spans}


@dataclass(frozen=True)
class Discovery:
    field: str
    authorities: tuple[AuthorityRecord, ...]
    missing: tuple[str, ...]                     # expected authorities not surfaced
    pieces: dict[str, str]                       # doc_id -> piece kind, from its own cover
    piece_spans: dict[str, dict[str, Any]]       # doc_id -> the cover title's span
    ranks: dict[str, int]                        # piece kind -> rank
    ranked: dict[str, int]                       # doc_id -> rank
    unranked: tuple[str, ...]                    # doc_ids carrying a claim and no rank
    disagreements: tuple[str, ...]
    sufficient: bool

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "authorities": [a.as_dict() for a in self.authorities],
                "missing": list(self.missing), "pieces": self.pieces, "piece_spans": self.piece_spans,
                "ranks": self.ranks, "ranked": self.ranked, "unranked": list(self.unranked),
                "disagreements": list(self.disagreements), "sufficient": self.sufficient}


@dataclass(frozen=True)
class Conflict:
    field: str
    values: tuple[str, ...]
    claims_by_value: dict[str, tuple[str, ...]]
    flagged: bool

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "values": list(self.values), "flagged": self.flagged,
                "claims_by_value": {v: list(ids) for v, ids in self.claims_by_value.items()}}


@dataclass(frozen=True)
class Resolution:
    field: str
    verdict: str
    rendering: str
    value: Optional[str]
    governing: tuple[str, ...]
    displaced: dict[str, list[int]]
    reason: str
    evidence: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "verdict": self.verdict, "rendering": self.rendering,
                "value": self.value, "governing": list(self.governing), "displaced": self.displaced,
                "reason": self.reason, "evidence": list(self.evidence)}


def piece_kinds(covers) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Each document's piece kind, read from the earliest title on its own cover, with the span."""
    kinds, spans = {}, {}
    for cover in covers:
        best = None
        for kind, _ in PIECE_KINDS:
            m = _PIECE_RE[kind].search(cover.text)
            if m and (best is None or m.start() < best[1].start()):
                best = (kind, m)
        if best is not None:
            kinds[cover.doc_id] = best[0]
            spans[cover.doc_id] = authority_span(cover, best[1].start(), best[1].end())
    return kinds, spans


def find_authorities(chunks) -> tuple[AuthorityRecord, ...]:
    """Seek the field's authorities across the pieces. Nothing is assumed present."""
    found = []
    for c in chunks:
        m = _RC6.search(c.text)
        if m:
            found.append(AuthorityRecord(RC6, "own_cover_governs", FIELD,
                                         authority_span(c, m.start(), m.end()), {}, {}))
        head = _RANKING_HEAD.search(c.text)
        if head and _RANKING_CONTEXT.search(c.text[:head.start()]):
            window = c.text[head.end():head.end() + RANKING_WINDOW]
            hits = []
            for kind, _ in PIECE_KINDS:
                mm = _PIECE_RE[kind].search(window)
                if mm:
                    hits.append((mm.start(), mm.end(), kind))
            ranks, rank_spans = {}, {}
            for rank, (start, end, kind) in enumerate(sorted(hits), 1):
                ranks[kind] = rank
                rank_spans[kind] = authority_span(c, head.end() + start, head.end() + end)
            if ranks:
                found.append(AuthorityRecord(CCAP2, "piece_ranking", FIELD,
                                             authority_span(c, head.start(), head.end()), ranks, rank_spans))
    return tuple(sorted(found, key=lambda a: (a.label, a.span["chunk_id"], a.span["char_start"])))


def discover(claims: list[ClaimRecord], covers, chunks, field: str = FIELD) -> Discovery:
    """The authority discovery gate: what was found, and whether the coverage is sufficient."""
    authorities = find_authorities(chunks)
    kinds, spans = piece_kinds(covers)
    labels = {a.label for a in authorities}
    missing = tuple(label for label in EXPECTED_AUTHORITIES if label not in labels)

    ranks: dict[str, int] = {}
    disagreements = []
    for a in authorities:
        if a.kind != "piece_ranking":
            continue
        for kind, rank in a.ranks.items():
            if ranks.setdefault(kind, rank) != rank:
                disagreements.append(f"{a.label}: {kind} ranked {rank} and {ranks[kind]}")
    governing_kind = min(ranks, key=lambda k: ranks[k]) if ranks else None
    for a in authorities:
        if a.kind == "own_cover_governs" and governing_kind is not None:
            carrier = kinds.get(a.span["doc_id"])
            if carrier is None:
                disagreements.append(f"{a.label}: the piece carrying it has no cover title")
            elif carrier != governing_kind:
                disagreements.append(f"{a.label}: carried by {carrier}, while rank 1 is {governing_kind}")

    ranked, unranked = {}, []
    for c in claims:
        if c.field != field:
            continue
        doc = c.provenance.sources[0]["doc_id"] if isinstance(c.provenance.sources[0], dict) \
            else c.provenance.sources[0].doc_id
        kind = kinds.get(doc)
        if kind is None or kind not in ranks:
            if doc not in unranked:
                unranked.append(doc)
        else:
            ranked[doc] = ranks[kind]
    sufficient = not missing and not unranked and not disagreements
    return Discovery(field, authorities, missing, kinds, spans, ranks, ranked,
                     tuple(sorted(unranked)), tuple(disagreements), sufficient)


def same_field_conflict(claims: list[ClaimRecord], field: str = FIELD) -> Conflict:
    """Same field, overlapping applicability: distinct normalised values are a conflict."""
    by_value: dict[str, list[str]] = {}
    for c in claims:
        if c.field == field:
            by_value.setdefault(c.value.normalized, []).append(c.claim_id)
    values = tuple(sorted(by_value))
    return Conflict(field, values, {v: tuple(sorted(by_value[v])) for v in values}, len(values) > 1)


def conclude(claims: list[ClaimRecord], discovery: Discovery, conflict: Conflict) -> Resolution:
    """The conclusion blocking gate, and the resolution it permits. Fails closed, always."""
    field = discovery.field
    if not discovery.sufficient:
        why = []
        if discovery.missing:
            why.append("authority not surfaced: " + ", ".join(discovery.missing))
        if discovery.unranked:
            why.append(f"{len(discovery.unranked)} piece(s) carrying a claim are not ranked")
        if discovery.disagreements:
            why.append("authorities disagree: " + "; ".join(discovery.disagreements))
        return Resolution(field, "abstain_conflict", HUMAN_RENDERING["abstain_conflict"], None, (), {},
                          "no conclusion while an applicable authority is unresolved — " + "; ".join(why),
                          tuple(a.span for a in discovery.authorities))

    if not discovery.ranked:
        return Resolution(field, "abstain_no_evidence", HUMAN_RENDERING["abstain_no_evidence"], None, (), {},
                          f"no claim of {field} carries a ranked piece", ())
    best = min(discovery.ranked.values())
    governing, values = [], {}
    for c in claims:
        if c.field != field:
            continue
        source = c.provenance.sources[0]
        doc = source["doc_id"] if isinstance(source, dict) else source.doc_id
        rank = discovery.ranked[doc]
        if rank == best:
            governing.append(c)
        values.setdefault(c.value.normalized, set()).add(rank)
    at_best = {c.value.normalized for c in governing}
    evidence = [a.span for a in discovery.authorities]
    for a in discovery.authorities:
        for kind, span in sorted(a.rank_spans.items()):
            if a.ranks.get(kind) == best:
                evidence.append(span)
    for c in sorted(governing, key=lambda c: c.claim_id):
        source = c.provenance.sources[0]
        evidence.append(source if isinstance(source, dict) else
                        {k: getattr(source, k) for k in ("doc_id", "chunk_id", "char_start", "char_end",
                                                         "span_sha256")})
    if len(at_best) != 1:
        return Resolution(field, "abstain_conflict", HUMAN_RENDERING["abstain_conflict"], None, (), {},
                          f"rank {best} carries {len(at_best)} values: the authority does not separate them",
                          tuple(evidence))
    value = at_best.pop()
    displaced = {v: sorted(r) for v, r in sorted(values.items()) if v != value}
    verdict = "supported_with_caveats" if conflict.flagged else "supported"
    reason = (f"rank {best} governs ({min(discovery.ranks, key=lambda k: discovery.ranks[k])}); "
              f"{len(governing)} claim(s) carry {value}")
    if displaced:
        reason += "; displaced: " + ", ".join(f"{v} at rank(s) {r}" for v, r in displaced.items())
    return Resolution(field, verdict, HUMAN_RENDERING[verdict], value,
                      tuple(sorted(c.claim_id for c in governing)), displaced, reason, tuple(evidence))


def dependents(claims: list[ClaimRecord], resolution: Resolution) -> dict[str, Any]:
    """The dates that depend on the resolved value, re-derived from it, and the interval it decides."""
    out: dict[str, Any] = {"questions_deadline": None, "displaced": [], "visit_window": None}
    if resolution.value is None:
        out["blocked"] = "no resolved value: nothing is re-derived"
        return out
    kept, displaced = [], []
    from_governing = _derived_from(claims, resolution)
    for c in claims:
        if c.field != "questions_deadline":
            continue
        source = c.provenance.sources[0]
        operand = source["char_start"] if isinstance(source, dict) else source.char_start
        chunk = source["chunk_id"] if isinstance(source, dict) else source.chunk_id
        entry = {"claim_id": c.claim_id, "value": c.value.normalized, "derivation": c.provenance.derivation,
                 "operand": {"chunk_id": chunk, "char_start": operand}}
        (kept if c.claim_id in from_governing else displaced).append(entry)
    out["questions_deadline"] = sorted(kept, key=lambda e: e["claim_id"])
    out["displaced"] = sorted({e["value"] for e in displaced})
    ends = [c for c in claims if c.field == "visit_window_end"]
    if kept and ends:
        questions = dt.datetime.fromisoformat(kept[0]["value"])
        end = dt.date.fromisoformat(ends[0].value.normalized)
        days = (end - questions.date()).days
        out["visit_window"] = {
            "end": ends[0].value.normalized, "questions_deadline": kept[0]["value"],
            "days_end_after_questions": days,
            "note": ("the visit window closes after the questions deadline: a visit in its last "
                     f"{days} day(s) can no longer be followed by a question")
            if days > 0 else "the questions deadline falls after the visit window closes"}
    return out


def _derived_from(claims: list[ClaimRecord], resolution: Resolution) -> set[str]:
    """The derived claims whose first operand is a governing claim's span."""
    spans = set()
    for c in claims:
        if c.claim_id in resolution.governing:
            s = c.provenance.sources[0]
            spans.add((s["chunk_id"], s["char_start"]) if isinstance(s, dict) else (s.chunk_id, s.char_start))
    out = set()
    for c in claims:
        if c.provenance.origin != "derived":
            continue
        s = c.provenance.sources[0]
        key = (s["chunk_id"], s["char_start"]) if isinstance(s, dict) else (s.chunk_id, s.char_start)
        if key in spans:
            out.add(c.claim_id)
    return out
