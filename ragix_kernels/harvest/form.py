"""ragix_kernels.harvest.form — the harvest form, its refusals, and the extractive floor.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

One call per node returns **one JSON document**: the level's summary, the interpreted layer and the
provenance tree. This module is the contract around that call. It never calls a model; it validates what a
model returned, and it fails closed.

The refusals the design rests on:

  1. **a critical value written as text**, rather than referenced. The model does not write dates, amounts,
     percentages, durations or quantities at all: it writes ``{{claim:<id>}}`` where one belongs, and the
     builder substitutes the claim's own typed value afterwards. A digit-bearing critical pattern anywhere
     outside a placeholder is a refusal, not a repair;
  2. **a summary sentence that maps to no child**, or to a child the node does not have. That is what
     "preserve the provenance tree" means operationally, and it is what makes `zoom` exact.

  3. **a span the model proposes that is not byte-exact.** A person, an organisation or a place is the one
     thing no grammar finds, so the model may propose those spans — and the span *is* the value, verified
     character for character against the object's own text. A span that is not a substring is refused.

Plus the ordinary ones: malformed JSON, a missing field, an unknown claim reference, a value outside a closed
vocabulary. Nothing is repaired, every refusal is counted and carries its reason.

The extractive floor (`extractive_summary`) produces the same shape with **no model at all**: sentences
selected from the children, never generated. It is the baseline a bake-off measures against and the
fallback under every run.

The form version names the validator that judged each record. Version 0.9 extends the
literal guard to the physical/count/composite candidate reader; replay must not reuse 0.8 verdicts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: 0.7: the SCHEMA is 0.6's, unchanged — what moved is how the form asks for it, and a record must not
#: claim the same instrument for two different askings. A first run refused a fifth of its windows on the
#: vocabularies alone, and the details said why: 'injunction', 'reference' and 'condition' arrived in
#: `relevance` (they are act values) and 'background' in `act` (it is a relevance value). The model filled
#: the two fields the wrong way round, and the prompt listed both vocabularies in adjacent sentences with
#: no example. 0.7 names each field with its own list, says the lists share no word, and shows one filled
#: answer. A form that invites a swap is a form defect.
#: 0.8: the repeated-phrase rule as a non-author review read it back — inside one SENTENCE (an entry
#: holds several), on the RENDERED line (the marker-stripped one invents phrases), two content words in
#: the span, a span that repeats in the window accepted as faithful quotation, and the exact rule beside
#: it: the same value cited twice.
FORM_VERSION = "harvest-form/0.9"
RELEVANCE = ("critical", "important", "informative", "background")
TYPES = ("informative", "injunction", "prohibition", "condition", "exception", "definition", "reference",
         "penalty")
LINK_KINDS = ("refers_to", "continues", "supersedes", "binds", "governed_by")
#: the node-prose role must write prose, not a string of placeholders. The floor sits below a writer's
#: observed median and above the hollow mode, where a summary is a list of markers and nothing else.
#: Below the floor is a contract failure for that role, not a soft score.
MIN_PROSE_WORDS = 30          # 30 content words at window grain (it was 25 before the grains were split)
#: form 0.6: a pilot opened its summary with ONE sentence carrying eighty-five consecutive markers — it
#: cited everything, committed to nothing and passed the contract. A summary is prose that cites, not a
#: list of citations, so a sentence may carry at most four markers and must stand on at least eight
#: content words outside them. Below either is `hollow sentence`, counted like any other refusal.
MAX_REFS_PER_SENTENCE = 4
MIN_SENTENCE_WORDS = 8
#: the numbers per grain: a window and a roll-up do not owe the same prose. The exported names above
#: keep their roll-up values for the callers that import them.
GRAIN = {"window": {"min_words": MIN_PROSE_WORDS, "max_refs": 4, "min_sentence_words": 6},
         "rollup": {"min_words": 80, "max_refs": 4, "min_sentence_words": MIN_SENTENCE_WORDS}}


def grain_rules(grain: str) -> dict:
    """validate()'s floor and hollow rule for one grain, as keyword arguments."""
    if grain not in GRAIN:
        raise HarvestRefusal("unknown grain", grain)
    return dict(GRAIN[grain])
ENTITY_KINDS = ("person", "organisation", "place")

#: The units a figure may carry, LONGEST ALTERNATIVE FIRST. Alternation takes the first that matches,
#: and "h" before "heures" made a first implementation replace "1 000 h" inside "1 000 heures" and
#: leave "eures" behind — kept here as the reason for the order.
_UNITS = ("pour cent", "pour-cent", "euros", "euro", "EUR", "€", "%", "heures", "heure", "minutes",
          "minute", "journées", "journée", "journees", "journee", "semaines", "semaine", "années",
          "année", "annees", "annee", "jours", "jour", "mois", "fois", "ans", "an", "min", "h")
FIGURE = re.compile(r"\d+(?:[.,\s  ]\d+)*(?:\s*(?:" +
                    "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)) +
                    r"))?", re.IGNORECASE)


def _fold(text: str) -> str:
    """The only form in which two figures are compared: whitespace gone, case folded."""
    return re.sub(r"\s+", "", text or "").casefold()


def substitute(sentences: Iterable[str], offered: Iterable[dict[str, Any]]) -> tuple[list[str], list[dict]]:
    """Protect the values the grammar read, rather than trusting a model to cite them: deterministic
    extraction, protection, verbatim reinsertion, verification.

    Every figure in a sentence that equals one of the offered values — folded, or by the normalised
    form both sides carry through `fr.grammars` — becomes that value's `{{claim:<value_id>}}`.
    The comparison is EXACT: a figure that merely contains, or is contained by, an offered value is
    not that value. The looser rule was tried on recorded answers and it is how « 50 heures » would claim the id
    of « 1 000 heures ».

    A figure matching no offered value is a refusal, `value not offered`: the pipeline may protect
    what the grammar read and may not invent a citation for a figure with nothing behind it. Every
    such figure is named in the refusal's detail, so a caller counts them instead of guessing.

    Returns the substituted sentences and the substitutions made. The grain rules are NOT applied
    here — the caller passes `grain_rules("window")` or `grain_rules("rollup")` to `validate()`,
    because one implementation cannot know which grain it is serving, and guessing that is how a
    roll-up's floor of 80 words once came to refuse three window answers.
    """
    from .fr.grammars import read_values                      # local: the grammar, read-only

    def normalised(text: str) -> Optional[str]:
        values = [v.normalized for v in read_values(text) if v.normalized]
        return values[0] if len(values) == 1 else None

    pool = []
    for value in offered:
        raw = str(value.get("raw") or "")
        pool.append({"value_id": value.get("value_id"), "raw": raw, "folded": _fold(raw),
                     "normalized": value.get("normalized") or normalised(raw)})

    out: list[str] = []
    made: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for sentence in sentences:
        # WHAT COUNTS AS A FIGURE IS THE GRAMMAR'S ANSWER, not a second regex living here. It gives
        # the exact span, so the replacement is a slice and « 1 000 h » can never be substituted
        # inside « 1 000 heures »; and a bare number the grammar does not type — an article number
        # in « articles 10 et 11 » — is not a claimable value, so it is neither cited nor refused.
        text = sentence
        for value in sorted(read_values(sentence), key=lambda v: v.start, reverse=True):
            if value.kind == "reference" or not any(c.isdigit() for c in value.raw):
                continue
            folded = _fold(value.raw)
            match = next((v for v in pool if v["folded"] == folded), None)
            if match is None and value.normalized:
                match = next((v for v in pool if v["normalized"] == value.normalized), None)
            if match is None:
                unmatched.append(value.raw.strip())
                continue
            text = text[:value.start] + marker("claim", str(match["value_id"])) + text[value.end:]
            made.append({"was": value.raw.strip(), "value_id": match["value_id"],
                         "raw": match["raw"], "normalized": match["normalized"]})
        out.append(text)
    if unmatched:
        raise HarvestRefusal("value not offered",
                             "; ".join(sorted(set(unmatched))[:8]) +
                             (f" (+{len(set(unmatched)) - 8} more)" if len(set(unmatched)) > 8 else ""))
    return out, made

#: ONE source for the marker syntax. Twice in one day a prompt typed its own braces and passed them
#: through str.format: `{{claim:v1}}` in a template renders as `{claim:v1}`, the model wrote what it was
#: shown, and the validator refused it — lost verdicts in one run, a voided coverage measurement in the
#: next. Nothing types braces now; a prompt calls marker() and the pattern is built from the same
#: constants, so the two cannot drift apart. tests/harvest/test_form_markers.py renders the real prompt
#: and checks it.
_OPEN, _CLOSE = "{{", "}}"
#: a marker names a claim id (hex) or a value id ("v1") — the runner passes the grammar's value ids,
#: and a hex-only pattern silently matched none of them, leaving the citation rule toothless
_REFS = {"claim": r"[A-Za-z0-9_.:-]{1,64}", "k": r"[A-Za-z0-9_.:-]{1,64}", "trap": r"T\d{2}"}


def marker(kind: str, ref: str) -> str:
    """The literal marker a prompt shows and a model must write. The only place the braces exist."""
    if kind not in _REFS:
        raise HarvestRefusal("unknown marker", kind)
    return f"{_OPEN}{kind}:{ref}{_CLOSE}"


PATTERNS = {kind: re.compile(re.escape(_OPEN) + kind + ":(" + ref + ")" + re.escape(_CLOSE))
            for kind, ref in _REFS.items()}
PLACEHOLDER = PATTERNS["claim"]
#: what a model may never write as literal text: a date, a clock time, an amount, a percentage, a duration
CRITICAL_TEXT = re.compile(
    r"\d{4}-\d{2}-\d{2}"                                   # an ISO date
    r"|\d{1,2}\s*[/.]\s*\d{1,2}\s*[/.]\s*\d{2,4}"          # a numeric date
    r"|\d{1,2}\s*(?:h|heures?)\s*\d{0,2}"                  # a clock time
    r"|\d[\d\s.,]*\s*(?:€|%|euros?|pour\s*cent)"           # an amount or a percentage
    r"|\b\d+\s*(?:jours?|mois|semaines?|ans?|heures?)\b",  # a duration
    re.IGNORECASE)


class HarvestRefusal(ValueError):
    """A call that broke the contract. Counted, never repaired."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _critical_literal(text: str) -> str | None:
    from .quantitative import harvest

    hit = CRITICAL_TEXT.search(text)
    if hit:
        return hit.group(0).strip()
    return next((c.raw for c in harvest(text, source_id="literal-guard", node_id="literal-guard",
                                        classification="CONTENT") if c.quantitative), None)


@dataclass(frozen=True)
class HarvestResult:
    node_id: str
    summary: str                      # still carrying its placeholders
    summary_map: tuple[dict[str, Any], ...]
    interpreted: dict[str, Any]
    references: dict[str, list[str]]
    form_version: str = FORM_VERSION
    #: the grammar's values, classified by the model — never re-written by it
    values: tuple[dict[str, Any], ...] = ()
    #: spans the model proposes for what no grammar can find; the span IS the value
    entities: tuple[dict[str, Any], ...] = ()

    def render(self, values: dict[str, str]) -> str:
        """The summary with every placeholder replaced by its claim's own value. No number was typed."""
        def swap(m: re.Match) -> str:
            claim_id = m.group(1)
            if claim_id in values:
                return values[claim_id]
            # a prefix that matches two claims was resolved by dict order, which could substitute the
            # wrong critical value into the prose and look right. It is a refusal.
            matches = sorted(full for full in values if full.startswith(claim_id))
            if len(matches) > 1:
                raise HarvestRefusal("ambiguous claim reference",
                                     f"{claim_id} matches {len(matches)}: {', '.join(m[:12] for m in matches[:3])}")
            if not matches:
                raise HarvestRefusal("unknown claim reference", claim_id)
            return values[matches[0]]
        return PLACEHOLDER.sub(swap, self.summary)


def speech_act(entry: dict[str, Any]) -> Any:
    """form 0.5 asks for `act`; `type` is accepted as the deprecated alias it used to be.

    The old name sat beside a list of *kinds* and invited the confusion it got: a model answered
    `type: duration`, which is a kind, where a speech act was asked for.
    """
    return entry.get("act", entry.get("type"))


def _locate_loose(span: str, text: Optional[str]) -> Optional[tuple[str, int, int]]:
    """Find what the model named, tolerating the whitespace a PDF inserts inside a word.

    Returns the source's exact substring with its offsets, so the record keeps the bytes that are there
    rather than the tidy form the model wrote. Nothing is accepted that is not present.
    """
    if not text or not span or not span.strip():
        return None
    if span in text:
        start = text.index(span)
        return span, start, start + len(span)
    pattern = r"\s*".join(re.escape(ch) for ch in re.sub(r"\s+", "", span))
    m = re.search(pattern, text)
    if m is None:
        return None
    return m.group(0), m.start(), m.end()


def _check_vocabulary(interpreted: dict[str, Any]) -> None:
    # A FIELD THAT IS ABSENT IS MISSING, not "outside vocabulary: None". The difference is the whole
    # diagnosis: a run once showed refusals reading « relevance outside vocabulary: None », which looks
    # like a model writing a bad word and was in fact a field that was never there — or, worse, an
    # entity being asked for a field it cannot have.
    if "relevance" not in interpreted:
        raise HarvestRefusal("missing field", "relevance")
    if interpreted.get("relevance") not in RELEVANCE:
        raise HarvestRefusal("relevance outside vocabulary", str(interpreted.get("relevance")))
    if not any(k in interpreted for k in ("act", "speech_act", "type")):
        raise HarvestRefusal("missing field", "act")
    if speech_act(interpreted) not in TYPES:
        raise HarvestRefusal("act outside vocabulary", str(speech_act(interpreted)))
    for link in interpreted.get("links") or []:
        if link.get("kind") not in LINK_KINDS:
            raise HarvestRefusal("link kind outside vocabulary", str(link.get("kind")))


def validate(raw: str, *, node_id: str, allowed_children: Iterable[str],
             allowed_claims: Iterable[str], value_ids: Iterable[str] = (),
             text: Optional[str] = None, min_words: int = 0, max_refs: int = 0,
             min_sentence_words: int = 0) -> HarvestResult:
    """Parse and check one call's JSON. Every failure is a refusal with its reason; nothing is repaired."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HarvestRefusal("strict JSON", str(exc)) from None
    if not isinstance(payload, dict):
        raise HarvestRefusal("strict JSON", "the form is an object")
    for key in ("node_id", "summary", "summary_map", "interpreted"):
        if key not in payload:
            raise HarvestRefusal("missing field", key)
    # `references` was required here and never asked for in the prompt: most refusals of an early
    # bake-off. summary_map and values already carry the provenance, so it is optional and defaults to empty.
    references = payload.get("references") or {}
    if payload["node_id"] != node_id:
        raise HarvestRefusal("wrong node", f"{payload['node_id']!r} for {node_id!r}")

    children, claims, known = set(allowed_children), set(allowed_claims), set(value_ids)
    body = payload["summary"]
    if isinstance(body, list):                       # a model may return the sentences as a list
        sentences_given = [str(s).strip() for s in body if str(s).strip()]
        summary = " ".join(sentences_given)
    elif isinstance(body, str):
        summary = body
        sentences_given = [s for s in re.split(r"(?<=[.!?])\s+", summary.strip()) if s]
    else:
        raise HarvestRefusal("missing field", "summary is neither text nor a list of sentences")
    stripped = PLACEHOLDER.sub(" ", summary)
    hit = _critical_literal(stripped)
    if hit:
        raise HarvestRefusal("critical value written by the model", hit)
    if min_words:
        prose = [w for w in re.split(r"[^\w'’-]+", stripped) if len(w) > 2]
        if len(prose) < min_words:
            raise HarvestRefusal("summary below the minimum",
                                 f"{len(prose)} content words, {min_words} asked")
    for claim_id in PLACEHOLDER.findall(summary):
        if not any(c.startswith(claim_id) for c in claims):
            raise HarvestRefusal("unknown claim reference", claim_id)

    # form 0.6: a sentence that is a string of markers is not a summary. Opt-in — the rule was declared
    # for the node roll-ups; the generic form's window-level callers keep their behaviour (0 = off).
    every = re.compile("|".join(p.pattern for p in PATTERNS.values()))
    for i, sentence in enumerate(sentences_given if (max_refs or min_sentence_words) else [], start=1):
        refs = len(every.findall(sentence))
        bare = every.sub(" ", sentence)
        words = [w for w in re.split(r"\s+", bare) if any(c.isalpha() for c in w)]
        # RE-DECLARED after reading recorded answers: counting markers alone refused a true sentence. A
        # visit ladder cites every rung — twelve markers on fifteen content words — and that is prose
        # that cites, not a list of citations. What the rule is really about is a sentence whose markers
        # outnumber its words: eighty-five on twenty. So a ceiling is not enough on its own; it must be
        # crossed AND the markers must outweigh the prose carrying them.
        if max_refs and refs > max_refs and refs > len(words):
            raise HarvestRefusal("hollow sentence",
                                 f"sentence {i} carries {refs} markers, more than the {max_refs} "
                                 f"allowed and more than its {len(words)} content words")
        if min_sentence_words and refs and len(words) < min_sentence_words:
            raise HarvestRefusal("hollow sentence",
                                 f"sentence {i} cites {refs} value(s) on {len(words)} content word(s), "
                                 f"{min_sentence_words} required")

    mapping = payload["summary_map"]
    if not isinstance(mapping, list) or not mapping:
        raise HarvestRefusal("empty provenance tree", "the summary maps to nothing")
    seen = set()
    own = next((c for c in children if str(c).endswith("#leaf")), None)
    for entry in mapping:
        index = entry.get("sentence")
        targets = list(entry.get("children") or [])
        # form 0.5. A sentence that cites a value must name it; a sentence that cites none has nothing to
        # name at window resolution, and is attributed to the object itself. Form 0.4 refused the second
        # case, which pushed a model toward inventing a citation — the one thing this design forbids.
        cited = PLACEHOLDER.findall(sentences_given[index - 1]) if isinstance(index, int) \
            and 1 <= index <= len(sentences_given) else []
        if not targets:
            if cited:
                raise HarvestRefusal("unsupported sentence",
                                     f"sentence {index} cites a value and names no child")
            if own is None:
                raise HarvestRefusal("unsupported sentence", f"sentence {index} maps to no child")
            targets = [own]
            entry["children"] = targets
        # a child is a child node, or — at window resolution, where an object IS its own child — one of
        # the grammar's values. A model answered with value ids against the first form, and it was right.
        unknown = [c for c in targets if c not in children and c not in set(value_ids)]
        if unknown:
            raise HarvestRefusal("unknown child", ", ".join(sorted(unknown)[:3]))
        seen.add(index)
    if len(seen) != len(sentences_given):
        raise HarvestRefusal("unsupported sentence",
                             f"{len(sentences_given)} sentence(s), {len(seen)} mapped")

    interpreted = payload["interpreted"]
    if not isinstance(interpreted, dict):
        raise HarvestRefusal("missing field", "interpreted")
    _check_vocabulary(interpreted)
    blob = json.dumps(interpreted, ensure_ascii=False)
    hit = _critical_literal(PLACEHOLDER.sub(" ", blob))
    if hit:
        raise HarvestRefusal("critical value written by the model", f"interpreted: {hit}")

    if not isinstance(references, dict):
        raise HarvestRefusal("missing field", "references is not an object")
    for claim_id in references.get("claims") or []:
        if claim_id not in claims:
            raise HarvestRefusal("unknown claim reference", claim_id)
    for child in references.get("children") or []:
        if child not in children:
            raise HarvestRefusal("unknown child", child)
    values = payload.get("values") or []
    if not isinstance(values, list):
        raise HarvestRefusal("missing field", "values")
    for entry in values:
        if entry.get("value_id") not in known:
            raise HarvestRefusal("unknown value reference", str(entry.get("value_id")))
        _check_vocabulary(entry)
        for key in ("value", "normalized", "raw"):
            if key in entry:
                raise HarvestRefusal("critical value written by the model",
                                     f"{entry['value_id']} carries {key}")

    entities = payload.get("entities") or []
    if not isinstance(entities, list):
        raise HarvestRefusal("missing field", "entities")
    for entity in entities:
        if entity.get("kind") not in ENTITY_KINDS:
            raise HarvestRefusal("entity kind outside vocabulary", str(entity.get("kind")))
        span = entity.get("span")
        if not isinstance(span, str) or not span.strip():
            raise HarvestRefusal("entity span missing", str(entity.get("kind")))
        located = _locate_loose(span, text)
        if located is None:
            raise HarvestRefusal("entity span not verbatim", span[:40])
        # the span stays the source's own bytes, line breaks included — the model named it, it did not
        # rewrite it. A PDF splits "Acte d'Engage\nment", and no model can echo that back by hand.
        entity["span"] = located[0]
        entity["char_start"], entity["char_end"] = located[1], located[2]
        # NO vocabulary check here. An entity is {kind, span} — it has no relevance and no act, so an
        # earlier line asked it for two fields it cannot have, read None, and refused: every answer that
        # named a place or an organisation was refused for it, and the entity layer produced nothing for a
        # whole run while the count read "relevance outside vocabulary: None". A non-author review of the
        # vocabulary refusals found it. The entity's own two checks are above: its kind is in
        # ENTITY_KINDS, and its span is verbatim in the text.

    return HarvestResult(node_id, summary, tuple(mapping), interpreted,
                         {"claims": sorted(references.get("claims") or []),
                          "children": sorted(references.get("children") or [])},
                         FORM_VERSION, tuple(values), tuple(entities))


def extractive_summary(children: list[dict[str, Any]], budget_sentences: int = 3) -> HarvestResult:
    """The floor: sentences selected from the children, never generated, with no model in the loop.

    Deterministic by construction — the children in their given order, the first sentence of each, up to the
    budget — so it is also the baseline the generated summaries are measured against.
    """
    if budget_sentences < 1:
        raise ValueError("a budget of at least one sentence")
    picked, mapping = [], []
    for child in children:
        if len(picked) >= budget_sentences:
            break
        text = (child.get("text") or "").strip()
        if not text:
            continue
        sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
        if not sentence:
            continue
        picked.append(sentence)
        mapping.append({"sentence": len(picked), "children": [child["id"]], "selected": True})
    if not picked:
        raise HarvestRefusal("empty provenance tree", "no child carried a sentence to select")
    return HarvestResult("", " ".join(picked), tuple(mapping),
                         {"relevance": "informative", "type": "informative", "links": []},
                         {"claims": [], "children": [c["id"] for c in children[:len(picked)]]},
                         form_version=FORM_VERSION + "+extractive")
