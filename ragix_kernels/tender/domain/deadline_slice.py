"""tender.deadline_slice — the deadline slice's claims (WP-3D 3.2a-D).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Over the text of a declared scope of chunks: the grammar's readings (tender.dates_fr), then a
deterministic classification by cue into the four fields of ClaimRecord 1.0, then the two
derivations the slice needs, each keeping its expression and every operand's span. No model.

- offer_deadline, observed: a complete datetime reading preceded by "limites de réception des offres :".
- visit_window_end, observed, and visit_window_start: the period "du … au …" preceded by "visites sur
  site pourront s'effectuer"; the start is observed when its year is written, and otherwise derived,
  its year taken from the end.
- questions_deadline, derived: the clause "demande doit intervenir au plus tard N jours avant la date
  limite de remise des plis", applied to every observed offer_deadline: offer_deadline − N days.

Every claim carries an Applicability 1.1 scoped to the whole consultation and all lots, with its
authority left unresolved (None): 3.4a-D establishes it. A reading no cue claims carries no field. A
cue whose value cannot be typed yields no claim and is counted as a drop, never completed.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from .claims import ClaimRecord, ClaimValue, Provenance, SourceSpan, sha256_text
from .records import Applicability

CUE_WINDOW = 160
CUE_OFFER = re.compile(r"limites\s+de\s+r[ée]ception\s+des\s+offres\s*:\s*$", re.IGNORECASE)
CUE_VISIT = re.compile(r"visites\s+sur\s+site\s+pourront\s+s['’]\s*effectuer\s+du\s+$", re.IGNORECASE)
CUE_QUESTIONS = re.compile(r"demande\s+doit\s+intervenir\s+(?P<clause>au\s+plus\s+tard\s+(?P<n>\d+)\s+jours\s+"
                           r"avant\s+la\s+date\s+limite\s+de\s+remise\s+des\s+plis)", re.IGNORECASE)
CLAUSE_SCOPE = {"pieces": "consultation", "lots": "all"}
VISIT_START_DERIVATION = "day and month of the start as written; year of visit_window_end"


@dataclass(frozen=True)
class Chunk:
    """A chunk of the store with its nodes: its text is its nodes' texts joined by line breaks."""

    doc_id: str
    chunk_id: str
    text: str
    node_ids: tuple[str, ...]
    node_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.node_ids) != len(self.node_texts):
            raise ValueError(f"chunk {self.chunk_id[:12]}: node ids and texts differ in number")
        if "\n".join(self.node_texts) != self.text:
            raise ValueError(f"chunk {self.chunk_id[:12]}: its text is not its nodes' texts joined")

    def span(self, start: int, end: int) -> SourceSpan:
        nodes, pos = [], 0
        for nid, text in zip(self.node_ids, self.node_texts):
            if start < pos + len(text) and pos < end:
                nodes.append(nid)
            pos += len(text) + 1
        return SourceSpan(self.doc_id, self.chunk_id, tuple(nodes), start, end,
                          sha256_text(self.text[start:end]))


def _applicability(project: str) -> Applicability:
    return Applicability(project=project, clause_scope=dict(CLAUSE_SCOPE), authority=None)


def extract(chunks: list[Chunk], project: str) -> tuple[list[dict[str, Any]], list[ClaimRecord], list[dict[str, Any]]]:
    """The readings of every chunk, the slice's claims, and the drops (a cue whose value cannot be typed)."""
    # The harvest family's French date grammar, imported where it is used: the tender family's
    # package walk imports every module, and importing this one must not require another family.
    from ragix_kernels.harvest.fr.dates import CHANNEL, periods, read
    readings: list[dict[str, Any]] = []
    claims: list[ClaimRecord] = []
    drops: list[dict[str, Any]] = []
    offers: list[tuple[ClaimRecord, SourceSpan]] = []

    def drop(chunk: Chunk, start: int, cue: str, why: str) -> None:
        drops.append({"chunk_id": chunk.chunk_id, "char_start": start, "cue": cue, "why": why})

    for c in chunks:
        rs = read(c.text)
        for r in rs:
            readings.append({"doc_id": c.doc_id, "chunk_id": c.chunk_id, "char_start": r.start,
                             "char_end": r.end, "raw": r.raw, "span_sha256": sha256_text(r.raw),
                             "type": r.type, "normalized": r.normalized,
                             "complete": r.normalized is not None, "reason": r.reason})
        for r in rs:
            if not CUE_OFFER.search(c.text[max(0, r.start - CUE_WINDOW):r.start]):
                continue
            if r.type != "datetime" or r.normalized is None:
                drop(c, r.start, "offer_deadline", r.reason or "not a datetime")
                continue
            s = c.span(r.start, r.end)
            claim = ClaimRecord("offer_deadline", ClaimValue("datetime", r.raw, r.normalized),
                                Provenance("observed", (s,), None, CHANNEL), _applicability(project))
            claims.append(claim)
            offers.append((claim, s))
        for a, b in periods(c.text, rs):
            if not CUE_VISIT.search(c.text[max(0, a.start - CUE_WINDOW):a.start]):
                continue
            if b.type != "date" or b.normalized is None:
                drop(c, b.start, "visit_window_end", b.reason or "not a date")
                continue
            sa, sb = c.span(a.start, a.end), c.span(b.start, b.end)
            claims.append(ClaimRecord("visit_window_end", ClaimValue("date", b.raw, b.normalized),
                                      Provenance("observed", (sb,), None, CHANNEL), _applicability(project)))
            if a.type == "date" and a.normalized is not None:
                claims.append(ClaimRecord("visit_window_start", ClaimValue("date", a.raw, a.normalized),
                                          Provenance("observed", (sa,), None, CHANNEL), _applicability(project)))
            elif a.type == "date" and a.year is None:
                try:
                    start = dt.date(b.year, a.month, a.day)
                except ValueError:
                    drop(c, a.start, "visit_window_start", "not a calendar date with the end's year")
                    continue
                claims.append(ClaimRecord("visit_window_start", ClaimValue("date", a.raw, start.isoformat()),
                                          Provenance("derived", (sa, sb), VISIT_START_DERIVATION, CHANNEL),
                                          _applicability(project)))
            else:
                drop(c, a.start, "visit_window_start", a.reason or "not a date")

    for c in chunks:
        for m in CUE_QUESTIONS.finditer(c.text):
            n = int(m["n"])
            clause = c.span(m.start("clause"), m.end("clause"))
            for offer, source in offers:
                value = dt.datetime.fromisoformat(offer.value.normalized) - dt.timedelta(days=n)
                claims.append(ClaimRecord("questions_deadline",
                                          ClaimValue("datetime", m["clause"], value.strftime("%Y-%m-%dT%H:%M")),
                                          Provenance("derived", (source, clause), f"offer_deadline − {n} days",
                                                     CHANNEL),
                                          _applicability(project)))
    return readings, claims, drops
