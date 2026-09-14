#!/usr/bin/env python3
"""pass1 — the two rules pass 1's text must satisfy, in one place for the smoke and the core (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE FIRST RUN OF THE CORE (rule T4), on the coordinating seat's word of 2026-09-11, which
carries the lead's mandate. Both rules replace what `smoke.py` declared this afternoon, and both were
written because the smoke measured their absence:

R1 the budget by construction. The abstract is the **longest prefix of whole sentences** whose word count
   (whitespace-separated) is at most WORDS (60). Generation is capped at NUM_PREDICT (140) tokens, near
   the budget rather than far past it. A sentence beyond the prefix is `trimmed`, and counted. When the
   first sentence alone exceeds the budget it is **kept and the abstract recorded `over_budget`** — never
   silently cut, so an over-long node stays visible instead of becoming a clean-looking fragment.
   Why: in the smoke all twenty abstracts exceeded 60 words (65 to 139) although the prompt asked for 60,
   and nine hit the old 256-token cap mid-sentence. A budget asked for is not a budget.

R2 the digit rule, upgraded, whitespace-insensitive on both sides. For each sentence:
   (a) **a figure travels with its unit.** Every match of QUANTITY — a number with its immediately
       following unit or month word (`4 heures`, `300,00 €`, `3 jours`, `11 septembre`, `12h00`, `5 %`) —
       must appear in the source with all whitespace removed and case folded. The store's text layer
       breaks digit runs, so the comparison cannot be literal; but the unit must travel with the number,
       because a bare run of one or two digits is present in almost any 4 000-character object and proves
       nothing.
   (b) **a date must exist as a date.** Every reading the kernel's own grammar (`tender.dates_fr.read`,
       used read-only) parses in the sentence with a normalised value must equal, normalised, a reading
       the same grammar finds in the source. Nothing else may be a date.
   A sentence failing (a) or (b) is **dropped and counted**, with the offending spans recorded; the
   abstract keeps its other sentences; an abstract that loses every sentence is a refusal, never an
   empty abstract.
   Why (b): in the smoke `granite3.1-moe:3b` wrote « Les délais de réception des offres sont fixés au
   11 septembre 2027 à 12h00 » over an object whose digits were all present separately — a date composed
   from pieces. The afternoon's substring test passed it. This is the rule that catches it, and the
   coordinating seat's reading of the abstracts is what found it: the check that matters here is a human
   reading twenty short texts, and the rule is what makes the finding reusable.

Nothing in this module calls a model, writes a file or judges French. Pass-1 text is orientation and never
evidence: no abstract it accepts is a source, and none may be cited in a brief (WP §8.1).

AMENDED 2026-09-11 ~19:0x, on the lead's word ("150, 350, 800 - go") relayed by the coordinating seat
while this seat was away — the agent's edit, left for this seat's review, since no kernel is gated by the
author of its change. R1's ceiling is per level: LADDER window 150, sheet 150, document 350, dce 800,
the model choosing the length under it by the content; `abstract_of` takes it as `ceiling`, defaulting to
WORDS (60) so the smoke, the rule check and the map test keep row 22's behaviour. The generation cap is
`num_predict_for(ceiling)`, about 2.2 tokens a French word plus a quarter, because two of row 24's
fourteen windows ended at the old 140 with `done_reason: length` — truncation must never be what binds.
The rule itself is unchanged: the longest prefix of whole sentences under the ceiling, a first sentence
over it kept and recorded `over_budget`. R2 is unchanged.

REVIEWED AND KEPT by the testing seat on the merge of 2026-09-11 ~19:4x, with its own amendments added
below. Three things of the agent's version are better than what this seat had written on its branch and
are kept as they are: `sheet` is an explicit rung rather than a default; the ceiling is a **parameter** of
`abstract_of` with WORDS as its default, so the smoke, this module's own checks and the map test keep
row 22's behaviour instead of silently changing what they measure; and `num_predict_for` at about 2.2
tokens a French word is better founded than this seat's 6.7, which divided generated tokens by the words
of the **kept** prefix — the trimmed and dropped sentences were in the numerator and not the denominator,
so the ratio was inflated. Four rules are added:

R1b truncation is a refusal, not an abstract. Row 22 shows why: `num_predict` 140 bound in **207 of its
   244 calls**, every one at `eval_count` exactly 140; 17 cards were stored ending mid-word
   (« technicien ≤30 »), and the document-level call that read one of those cards invented « 30 jours »
   from « ≤30 ». The agent's per-level count of `length` in the summary is kept beside it: refused **and**
   counted.
R1b **refined**, 2026-09-11 ~22:5x, on the coordinating seat's relay, measured on row 29
   (`outputs_core150/core_abstracts.jsonl`, sha256 `ecf7c859cb59619c…`): 142 of its 244 calls stopped on
   `length`, **37 of them ending on an unfinished tail and 105 on a whole sentence**, and all 142 were
   admitted. Refusing all 142 would throw away 105 sound abstracts; admitting all 142 is what stored the
   mid-word cards. So the cap's ending is removed rather than judged: when `done_reason` is `length`, a
   final fragment carrying no terminal punctuation (`. ! ? … » "`) is **dropped before R1's prefix rule**,
   and the record is clean, marked `trimmed_after_length` with the fragment it lost. **Only an empty
   remainder is a refusal** — row 22's 17 of 207 are exactly that case, and row 29's 37 of 142 the other.
R2.1 canonical comparison, on the bench's measurement of row 24 (61 of its first 199 nodes lost a
   sentence, 108 offences, all quantities and 78 % of them a unit the model spelled out where the source
   writes « 48h », « 3j » or « 4 H »). A number with its unit and a date are compared as
   `tender.grammars_fr` normalises them — PT48H against PT48H, 250.00 EUR against 250.00 EUR — and only
   fall back to R2's literal test when the grammar cannot read the hit. Measured limit, reported not
   fixed: grammars_fr 1.2 reads `48h`, `48 H`, `4 H`, `3 jours`, `250,00 €`, and does **not** read `3j`,
   `30 min`, `½ journée` or `2 fois`, so R2.1 rescues part of that 78 % and the rest stays dropped.
R3 substitution, so prose never carries a critical value (the lead's doctrine as relayed: row 19's plain
   summaries state « 11 septembre 2026 » from a cover while the RC says 18 septembre under authority, and
   no sentence can say which). A figure the grammar offers for that node becomes its
   `{{claim:<value_id>}}`, built with `tender.harvest.marker`, and the card stores the values it used.
   The comparison is **exact folded equality, never containment**: K1 of `check_rules.py` caught a
   containment fallback matching the fabricated « 11 septembre 2027 » to the offered « 11 septembre » and
   laundering it into a placeholder. A figure that matches nothing offered stays literal and R2/R2.1
   decide its sentence.
R4 the trap flags: every span of `traps.json` inside the node's byte range is put on the card with its
   French question, because the abstracts reproduce the corpus's copy-paste defects faithfully (the UPS
   piece's abstract lists doors and gates) and a reader must see the register's warning beside the prose.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from .fr.dates import read as read_dates          # the kernel's grammar, read-only
from .fr.grammars import VERSION as GRAMMAR_VERSION   # recorded per run (R2.1)
from .fr.grammars import read_values              # the kernel's grammar, read-only
from .form import PLACEHOLDER, marker          # the form's pattern and its only brace maker
from .form import HarvestRefusal               # R3's refusal, handled per sentence
from .form import substitute as harvest_substitute   # R3 itself, since 2026-09-12

WORDS = 60
NUM_PREDICT = 140
#: the lead's ladder: a ceiling per level, the model choosing the length under it by the content
LADDER = {"window": 150, "sheet": 150, "document": 350, "dce": 800}
#: 2.2 tokens a French word is the density MEASURED on row 29 (generated tokens over the words of the
#: whole answer, trimmed and dropped sentences included in both). It is kept here as the measurement and
#: is no longer the budget: at 2.2 × 1.25 the cap still bound 142 of 244 calls, so truncation was still
#: deciding where an abstract ended. Every run after row 29 budgets at 3.0 — the measured density plus
#: the room a model needs to finish a sentence — and the quarter in num_predict_for stays the margin.
MEASURED_TOKENS_PER_WORD = 2.2
TOKENS_PER_WORD = 3.0
#: the hollow rule at window grain, the lead's word relayed 2026-09-11: four markers, six content words.
#: Measured against row 23's eleven substituted sentences, whose thinnest carries seven.
MAX_MARKERS = 4
MIN_SENTENCE_CONTENT = 6
#: below this a card is recorded `thin`, never refused: a floor on generated prose is a padding incentive
THIN_CONTENT = 12
#: units LONGEST FIRST — alternation takes the first that matches, and `h` before `heures` made an earlier
#: version replace "1 000 h" inside "1 000 heures". A French thousands group is part of the number:
#: "1 000 heures" is one quantity, not "1" and "000 heures". Space and the two no-break spaces only,
#: never a newline, which would glue two table cells. Both found by K4 of check_rules.py.
UNITS = (r"pour\s*cent|%|euros?|EUR|€|heures?|h|minutes?|min|journées?|jours?|semaines?|mois|années?|"
         r"ans?|fois|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre")
QUANTITY = re.compile(r"\d+(?:[ \u202f\u00a0]\d{3})*(?:[.,]\d+)?"
                      r"(?:\s*(?:" + UNITS + r"))?(?![^\W\d_])", re.IGNORECASE)
#: R2.2's fallback reader. `QUANTITY` is anchored on the right and not on the left, so it starts inside
#: an alphanumeric reference: FOSTER EP1440H gave 1440H, NW12H1 gave 12H, GHF031,2H17ANW50F gave 031,2H.
#: The left boundary is only correct BEHIND the grammar — on its own it merely shifts the match to 440H.
QUANTITY_BOUNDED = re.compile(r"(?<!\w)" + QUANTITY.pattern, QUANTITY.flags)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WS = re.compile(r"\s+")


#: How a parent's source is built from its children's abstracts — **the one definition**, imported by
#: the driver that writes the parent and by the audit that rebuilds it, so the two can never drift. A
#: recipe is versioned: if the joining ever changes, the token changes with it and an old record still
#: says how it was made. Added 2026-09-12, because the re-scorer found that a parent's source is
#: recorded nowhere and a parent's card could therefore be audited against nothing.
ROLLUP = "rollup/1"                 # "- " + the child's abstract, one per line
DCE = "dce/1"                       # "- " + the child's piece name + " : " + its abstract


class ParentDrift(Exception):
    """A parent's recorded provenance does not hold: a refusal, never a silent re-assembly."""


def assemble(recipe: str, parts: list[tuple[str, str]]) -> str:
    """`parts` is (name, text) per child, in the order the driver used them."""
    if recipe == ROLLUP:
        return "\n".join(f"- {text}" for _, text in parts)
    if recipe == DCE:
        return "\n".join(f"- {name} : {text}" for name, text in parts)
    raise ValueError(f"unknown assembly recipe {recipe!r}")


def rebuild_parent(record: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str | None:
    """X1 for a parent: rebuild the source it was written from, or say why it cannot be rebuilt.

    Returns the rebuilt source when every child's abstract still hashes to what the parent recorded
    and the assembled text hashes to the parent's `source_sha256`; `None` when the record carries no
    assembly (the three runs committed before 2026-09-12 — they cannot be audited at all, and that
    impossibility is a fact about the record, not about the parent); and raises `ParentDrift` when the
    provenance is there and does not hold, which is a refusal and never a silent re-assembly.
    """
    recipe = record.get("assembly")
    if not recipe:
        return None
    used = record.get("children_used") or []
    hashes = record.get("children_sha256") or []
    if len(used) != len(hashes):
        raise ParentDrift(f"{len(used)} children used, {len(hashes)} hashes recorded")
    parts: list[tuple[str, str]] = []
    for child_id, recorded in zip(used, hashes):
        child = by_id.get(child_id)
        if child is None:
            raise ParentDrift(f"child {child_id[:16]} is not in the run")
        text = child.get("abstract") or ""
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != recorded:
            raise ParentDrift(f"child {child_id[:16]}'s abstract is not the one the parent read")
        parts.append((child.get("piece") or "", text))
    source = assemble(recipe, parts)
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != (record.get("source_sha256") or "").split(":")[-1]:
        raise ParentDrift("the reassembled source does not hash to the parent's source_sha256")
    return source


def num_predict_for(ceiling: int) -> int:
    """The generation cap for a ceiling in words: truncation must never be what binds.

    3.0 tokens a word with a quarter of margin, since row 29 showed that 2.2 — the density
    actually measured there — still let the cap bind 142 of 244 calls. `MEASURED_TOKENS_PER_WORD`
    keeps that measurement where it belongs: in the record, not in the arithmetic."""
    return int(ceiling * TOKENS_PER_WORD * 1.25 + 0.5)


#: terminal punctuation of a finished French sentence, closing quotation marks included
TERMINAL = ".!?…»\""
#: the context lengths a server may quietly fall back to; used only when a record carries no num_ctx
POWERS_OF_TWO = {2 ** k for k in range(9, 21)}
#: French prose, characters per token — the floor under which a reading cannot have been whole
CHARS_PER_TOKEN_FR = 4


def ends_whole(text: str) -> bool:
    """Whether a text ends where a sentence ends. The test R1b turns on."""
    stripped = (text or "").rstrip()
    return bool(stripped) and stripped[-1] in TERMINAL


def input_truncation(record: dict[str, Any]) -> dict[str, Any]:
    """R1c, decided from a record alone — and saying what it had to assume.

    A server that cuts the prompt to `num_ctx` still answers, and answers well enough to look
    sound: row 24's DCE node was given 707 085 characters, 198 853 tokens were read as 8 192, and
    the record said `ok`. Where `num_ctx` is recorded the rule is arithmetic. Where it is not —
    the three runs committed before core.py recorded it — the rule must not quietly assume 8 192:
    a reading that sits exactly on a power-of-two boundary while its source is far larger than
    that reading can hold is truncation-suspect, and suspect is refused, with the assumption named
    in the verdict so a reader can reject it."""
    read = record.get("prompt_eval_count") or 0
    ctx = record.get("num_ctx")
    if ctx:
        return {"refused": read >= ctx, "num_ctx": ctx, "prompt_eval_count": read,
                "basis": f"num_ctx {ctx} recorded with the call"}
    chars = record.get("source_chars") or 0
    # a token of French prose is about four characters, and the prompt adds its instructions on top,
    # so a source longer than four characters per token read cannot have been read whole. Row 29's
    # DCE node is the case that sets it: 58 860 characters read as exactly 8 192 tokens, 7.2 per
    # token, which no tokeniser of French produces.
    suspect = read in POWERS_OF_TWO and chars > read * CHARS_PER_TOKEN_FR
    return {"refused": bool(suspect), "num_ctx": None, "prompt_eval_count": read,
            "basis": "no num_ctx in the record; ASSUMED: a power-of-two reading under a far larger "
                     "source is truncation-suspect" if suspect else
                     "no num_ctx in the record; the reading is not at a power-of-two boundary"}


def fold(text: str) -> str:
    """Whitespace removed, case folded: the only form in which both sides are compared."""
    return WS.sub("", text).casefold()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]


def content_words(text: str) -> list[str]:
    r"""The form's own counting rule, `tender.harvest.validate` at its `min_words` check:

        prose = [w for w in re.split(r"[^\w'\u2019-]+", PLACEHOLDER.sub(" ", summary)) if len(w) > 2]

    It lives here because the kernel does not export it; the moment it does, both passes import that one.
    """
    return [w for w in re.split(r"[^\w'\u2019-]+", PLACEHOLDER.sub(" ", text)) if len(w) > 2]


def canonical(text: str) -> set[str]:
    """R2.1: every canonical form the kernel's grammars read in `text`."""
    return ({v.normalized for v in read_values(text) if v.normalized}
            | {r.normalized for r in read_dates(text) if r.normalized})


def offered_values(source: str) -> list[dict[str, Any]]:
    """R3: the grammar's values for this node, numbered as the harvest numbers them."""
    return [{"value_id": f"v{i + 1}", "kind": v.kind, "raw": v.raw, "normalized": v.normalized,
             "start": v.start, "end": v.end}
            for i, v in enumerate(read_values(source))]


def substitute(sentence: str, values: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """R3 — **delegated to `tender.harvest.substitute` since 2026-09-12**, on the coordinating seat's
    arbitration, now that stage 2 has landed. Declared here so nobody compares row 24's two halves
    across the change: rows 22, 24 and 29 were written by this file's own implementation, and every
    run after this line was written by the kernel's.

    The kernel does what this seat's version did and says why in its own docstring: the grammar gives
    the figure and its exact span, the replacement is a slice so « 1 000 h » can never be substituted
    inside « 1 000 heures », the comparison is exact (folded, or by the normalised form both sides
    carry), and a `reference` kind is never claimable. Two differences are real and both are handled
    here rather than hidden:

    1. **The kernel matches on the normalised form as well as the folded raw.** That is strictly better
       — « 48 heures » cites the value the grammar read as « 48h » — and it is kept.
    2. **The kernel raises `HarvestRefusal("value not offered")` for a figure with nothing behind it,
       and pass 1 must not.** At the harvest's grain a card that cites an unoffered figure is void; at
       pass 1's grain the sentence is R2/R2.1's business — it is dropped if the figure is absent from
       the source and **kept if the figure is there but the grammar cannot read it**, which is exactly
       the « une visite » class of FINDINGS §12.5. Refusing the sentence here would turn a grammar gap
       into a lost sentence. So the call is made per sentence and a refusal leaves that sentence
       unsubstituted, for the digit rule to judge.
    """
    try:
        done, made = harvest_substitute([sentence], values)
    except HarvestRefusal:
        return sentence, []                            # difference 2: R2/R2.1's to judge, not R3's
    by_id = {v.get("value_id"): v for v in values}
    used = [{**m, "kind": by_id.get(m.get("value_id"), {}).get("kind")} for m in made]
    return done[0], used


def check_sentence(sentence: str, source_folded: str, source_dates: set[str],
                   source_canonical: set[str] | None = None) -> list[dict[str, Any]]:
    """R2, R2.1 and R2.2: the offences of one sentence — empty when it may be kept.

    R2.2 (2026-09-12, approved by the coordinating seat on the review's §5): **the sentence is read
    with `read_values`, the same reader the source is read with.** Until today the source was read by
    `read_values` and the sentence by `QUANTITY`, and two readers on one figure convict a correct
    quotation: `read_values` declines a clock form (`24h00`, `7h30`, `14 h 30`) and reads
    « Vendre\ndi\n11\nseptembre 202\n6\nà 12 heures 00 » as ONE datetime, while `QUANTITY` split the
    sentence's copy into a bare `24h` or `12 heures` and called it a duration. Of the 45 sentences the
    recorded runs kept and the rule then dropped, **45 were the instrument's and 0 the model's**.
    The digit fallback stays for what the grammar declines — without it an unreadable invention such as
    « 4321 heures ouvrées » would pass unseen — but it runs only OUTSIDE the spans the grammar covered
    and through `QUANTITY_BOUNDED`, which a part number cannot start inside (`EP1440H` → nothing).
    """
    bad: list[dict[str, Any]] = []
    canon = source_canonical if source_canonical is not None else set()
    covered: list[tuple[int, int]] = []
    for value in read_values(sentence):                 # R2.2: one reader, the source's
        covered.append((value.start, value.end))
        span = value.raw.strip()
        if value.normalized:
            if value.normalized not in canon:
                bad.append({"kind": "quantity", "span": span,
                            "canonical": [value.normalized], "rule": "R2.2"})
        elif fold(span) not in source_folded:           # read, but not normalised: compare literally
            bad.append({"kind": "quantity", "span": span, "rule": "R2"})
    for m in QUANTITY_BOUNDED.finditer(sentence):       # the fallback, outside what the grammar read
        if any(m.start() < end and start < m.end() for start, end in covered):
            continue
        span = m.group(0).strip()
        if not any(c.isdigit() for c in span):
            continue
        if fold(span) not in source_folded:             # R2: the grammar cannot read it
            bad.append({"kind": "quantity", "span": span, "rule": "R2"})
    for reading in read_dates(sentence):
        if not reading.normalized:
            continue
        if reading.normalized not in source_dates and reading.normalized not in canon:
            bad.append({"kind": "date", "span": reading.raw.strip(),
                        "normalized": reading.normalized, "rule": "R2.1"})
    return bad


def abstract_of(raw: str, source: str, ceiling: int = WORDS, done_reason: str | None = None,
                values: list[dict[str, Any]] | None = None,
                traps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """R1, R1b, R2, R2.1, R3 and R4 applied to one model answer, under the level's ceiling."""
    # R1b: the cap wrote this ending, not the model — so the ending is removed, not judged. The
    # fragment goes BEFORE R2 reads the sentences and before R1's prefix rule, so a half-written
    # figure never reaches the digit rule and never reaches a parent's source.
    truncated = done_reason == "length"
    offered_sentences = sentences(raw)
    fragment: str | None = None
    if truncated and offered_sentences and not ends_whole(offered_sentences[-1]):
        fragment = offered_sentences.pop()
    #: nothing whole was written at all — the only refusal the cap is answerable for. An abstract
    #: emptied later by the digit rule is R2's refusal, and saying otherwise would blame the budget
    #: for a fabrication. K7 of check_rules.py found this exact confusion in the first draft.
    nothing_whole = truncated and not offered_sentences
    source_folded = fold(source)
    source_dates = {r.normalized for r in read_dates(source) if r.normalized}
    source_canonical = canonical(source)
    offered = values or []
    kept: list[str] = []
    dropped: list[dict[str, Any]] = []
    used: list[dict[str, Any]] = []
    for sentence in offered_sentences:
        placed, mine = substitute(sentence, offered) if offered else (sentence, [])   # R3 first
        bad = check_sentence(placed, source_folded, source_dates, source_canonical)
        if bad:
            dropped.append({"sentence": sentence, "offences": bad})
        else:
            kept.append(placed)
            used += mine
    prefix: list[str] = []
    trimmed: list[str] = []
    words = 0
    for sentence in kept:
        n = len(sentence.split())
        if prefix and words + n > ceiling:
            trimmed.append(sentence)
            continue
        if not prefix and n > ceiling:                      # the first sentence alone is over budget
            prefix.append(sentence)
            words = n
            continue
        prefix.append(sentence)
        words += n
    if trimmed and len(prefix) < len(kept):
        trimmed = kept[len(prefix):]
    abstract = " ".join(prefix)
    content = len(content_words(abstract))
    in_card = [u for u in used if marker("claim", u["value_id"]) in abstract]
    return {"abstract": abstract, "words": words, "over_budget": words > ceiling, "ceiling": ceiling,
            "content_words": content, "thin": bool(abstract) and content < THIN_CONTENT,
            "values_used": in_card, "placeholders": len(in_card), "traps": traps or [],
            "sentences_kept": len(prefix), "sentences_trimmed": len(trimmed),
            "trimmed": trimmed, "sentences_dropped": len(dropped), "dropped": dropped,
            # counted over the sentences R2 actually read: a fragment the cap wrote is not a
            # quantity this node offered, and counting it would inflate both totals
            "quantities_checked": sum(len(QUANTITY.findall(s)) for s in offered_sentences),
            "dates_checked": sum(1 for s in offered_sentences for r in read_dates(s) if r.normalized),
            "trimmed_after_length": truncated, "fragment_dropped": fragment,
            "ok": bool(abstract),
            "refusal": None if abstract else
                       ("truncated: the cap stopped the generation before a whole sentence"
                        if nothing_whole else "every sentence dropped by the digit rule")}
