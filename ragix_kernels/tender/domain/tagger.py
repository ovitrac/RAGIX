"""tender.tagger — deterministic coarse-axis tagger (the MANDATORY baseline).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Standing rule of the project: every advanced method ships with the simple
method it must beat. The LLM extraction lane of the requirement lane
(``DESIGN_REQUIREMENT_LANE_20260821`` §6) is judged against this tagger — if it
does not beat it measurably (gate GR3-1), it does not earn its place.

Mechanism: ORDERED NAMED SIGNALS, never a blended score. The signals, in the
order they are tried, and the reason for that order:

  1. ``ancestry-exact``     — an ancestry element IS an axis name (or one of
                              its aliases). Structure prescribed by the buyer
                              beats anything read from free text.
  2. ``ancestry-contains``  — an axis name (>= 2 fold tokens) appears as a
                              contiguous token run inside an ancestry element
                              ("5.2 Pénalités de retard" -> `penalites`).
  3. ``text-contains``      — same test on the question/answer text itself.
  4. ``ancestry-token``     — single-token axes (`rgpd`, `dqe`, `résiliation`)
                              matched as a whole token of an ancestry element.
                              Last, and on STRUCTURE ONLY: a one-word axis is
                              too weak a signal to be trusted inside prose.

Within a signal, the deepest ancestry element decides; within one element the
most specific axis (most fold tokens) wins; an exact tie between two axes is an
ABSTENTION (``unknown``, signal ``ambiguous``) with both competitors traced —
fail closed, never a coin flip.

Every tag carries its trace: signal, matched evidence (verbatim), ancestry
depth, competitors. ``unknown`` is a valid, counted outcome.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from . import vocabulary

UNKNOWN = vocabulary.UNKNOWN

#: declared filter for signal 4 — a single-token axis is only trusted against a
#: HEADING. Some ancestry elements are whole sentences ("Les candidats sont
#: invités à remplir cette annexe financière…"), where isolated words like
#: `annexe`, `prix`, `objet` collide and mean nothing (measured: 38 ambiguous
#: abstentions, all on such elements). Above this length the element is prose.
HEADING_MAX_TOKENS = 6


@dataclass
class Tag:
    theme: str                                  # axis_id or "unknown"
    signal: str                                 # named signal / ambiguous / none
    evidence: str = ""                          # verbatim matched string
    depth: Optional[int] = None                 # 0 = deepest ancestry element
    competitors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"theme": self.theme, "signal": self.signal,
                "evidence": self.evidence, "depth": self.depth,
                "competitors": self.competitors}


@lru_cache(maxsize=4)
def _forms(path: Optional[str] = None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """(axis_id, fold tokens) for every canonical name and alias."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for a in vocabulary.load_axes(path)["axes"]:
        for name in [a["canonical"], *a.get("aliases", [])]:
            toks = tuple(vocabulary.fold(name).split())
            if toks:
                out.append((a["axis_id"], toks))
    return tuple(sorted(set(out), key=lambda t: (-len(t[1]), t[0])))


def ancestry_chain(trace: dict[str, Any]) -> list[str]:
    """Ancestry elements, DEEPEST FIRST, from a Question/AnswerRecord trace.

    Two shapes exist in the corpus (qa_corpus_extract): xlsx records carry
    sheet / block title / nearest section row, pdf records carry an outline
    chain built shallow -> deep.
    """
    anc = (trace or {}).get("ancestry") or {}
    if "headings" in anc:
        return [h for h in reversed(anc.get("headings") or []) if h]
    return [x for x in (anc.get("section_row_text"), anc.get("block_title"),
                        anc.get("sheet")) if x]


def fold_spans(s: str) -> tuple[tuple[str, ...], list[tuple[int, int]]]:
    """Fold ``s`` while keeping each surviving token's span in the RAW string.

    The evidence a tagger reports must be a verbatim substring of what it read
    (rule 9) — reporting the folded form would be a reformulation, and
    ``RequirementRecord.from_extraction`` rightly refuses it. Keeping the raw
    offsets makes the guard pass by construction instead of by luck.
    """
    toks: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"[^\W_]+(?:['’][^\W_]+)*", s, flags=re.UNICODE):
        f = vocabulary.fold(m.group(0))
        if f:                          # stopwords fold to "" and are skipped
            for part in f.split():     # an elided form can yield one token
                toks.append(part)
                spans.append((m.start(), m.end()))
    return tuple(toks), spans


def _find_run(hay: tuple[str, ...], needle: tuple[str, ...]) -> Optional[int]:
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return None


def _decide(hits: list[tuple[str, tuple[str, ...], str]], signal: str,
            depth: Optional[int]) -> Optional[Tag]:
    """Most specific axis wins; an exact tie abstains (fail closed)."""
    if not hits:
        return None
    best = max(len(t) for _, t, _ in hits)
    top = {aid for aid, t, _ in hits if len(t) == best}
    if len(top) > 1:
        return Tag(UNKNOWN, "ambiguous", depth=depth,
                   competitors=sorted(top))
    aid = top.pop()
    ev = next(e for a, t, e in hits if a == aid and len(t) == best)
    return Tag(aid, signal, evidence=ev, depth=depth,
               competitors=sorted({a for a, _, _ in hits} - {aid}))


def tag(text: str, ancestry: list[str], *,
        path: Optional[str] = None) -> Tag:
    """Assign one coarse axis to a question/answer/section. Never guesses."""
    forms = _forms(path)
    multi = [(a, t) for a, t in forms if len(t) >= 2]
    single = [(a, t) for a, t in forms if len(t) == 1]
    anc = [(el, *fold_spans(el)) for el in ancestry]

    def _verbatim(raw: str, spans: list[tuple[int, int]],
                  i: int, n: int) -> str:
        return raw[spans[i][0]:spans[i + n - 1][1]]

    # 1 — ancestry-exact
    for d, (el, toks, _sp) in enumerate(anc):
        hits = [(a, t, el) for a, t in forms if t == toks]
        got = _decide(hits, "ancestry-exact", d)
        if got:
            return got

    # 2 — ancestry-contains (>= 2 tokens)
    for d, (el, toks, sp) in enumerate(anc):
        hits = []
        for a, t in multi:
            i = _find_run(toks, t)
            if i is not None:
                hits.append((a, t, _verbatim(el, sp, i, len(t))))
        got = _decide(hits, "ancestry-contains", d)
        if got:
            return got

    # 3 — text-contains (>= 2 tokens)
    ttoks, tsp = fold_spans(text)
    hits = []
    for a, t in multi:
        i = _find_run(ttoks, t)
        if i is not None:
            hits.append((a, t, _verbatim(text, tsp, i, len(t))))
    got = _decide(hits, "text-contains", None)
    if got:
        return got

    # 4 — ancestry-token (single-token axes, SHORT structural elements only)
    for d, (el, toks, sp) in enumerate(anc):
        if len(toks) > HEADING_MAX_TOKENS:
            continue
        hits = []
        for a, t in single:
            i = _find_run(toks, t)
            if i is not None:
                hits.append((a, t, _verbatim(el, sp, i, 1)))
        got = _decide(hits, "ancestry-token", d)
        if got:
            return got

    return Tag(UNKNOWN, "none")
