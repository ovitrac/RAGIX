"""ragix_kernels.harvest.fr.grammars — the deterministic value grammars beside the dates.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

"Grammars first" only held for dates until now. A model asked to classify an amount it cannot write needs
the amount to exist already, so these are the other critical kinds a tender response needs:

    amount      250,00 €    1 234,56 EUR    40 € HT
    percentage  35 %        1,5%
    duration    10 jours    6 mois    2 ans    48 heures
    reference   article 4.1 du CCAG    annexe 4 du CCTP    R.2132-11    RC 8.1
    quantity    12 lots     3 familles          (a number and a counted noun, closed list)

Same discipline as `fr.dates`: whitespace — line breaks included — is accepted between any two
characters of a token, because a PDF splits "25" from "0,00 €"; the span returned is the exact slice it was
read from; a value that cannot be typed is returned incomplete rather than guessed. `person`,
`organisation` and `place` are deliberately absent: no grammar finds them, so they arrive as spans the
model proposes and the harvest verifies byte-exact against the text — the span is the value.

The name and version (``tender.grammars_fr 1.5``) are kept as first published: value ids and records
carry them, and a version is what says which reading produced a value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Optional

from .dates import Reading as DateReading
from .dates import read as read_dates

GRAMMAR = "tender.grammars_fr"
VERSION = "1.5"
CHANNEL = f"{GRAMMAR} {VERSION}"

KINDS = ("date", "datetime", "period", "amount", "percentage", "duration", "reference", "quantity")
#: numbers written in words, which tender prose uses for delays ("tous les six mois")
WORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
         "neuf": 9, "dix": 10, "onze": 11, "douze": 12, "quinze": 15, "vingt": 20, "trente": 30}
COUNTED = ("lots", "lot", "familles", "famille", "établissements", "etablissements", "visites", "visite",
           "pièces", "pieces", "candidats", "opérateurs", "operateurs", "fois")    # 1.3: « 2 fois »
#: unit -> the ISO 8601 duration slot it fills (NORMALISATION.md: PnYnMnDTnHnM, weeks as days)
UNITS = {"jour": "D", "jours": "D", "mois": "M", "an": "Y", "ans": "Y", "année": "Y", "années": "Y",
         "annee": "Y", "annees": "Y", "heure": "H", "heures": "H", "semaine": "W", "semaines": "W",
         # 1.3 — spellings scanned tender text uses, found behind sentences the pass-1 digit rule dropped:
         # « 3j », « 3 j », « ½ journée », « 30 min ». Minutes get their own slot: PT30M, never P30M (months).
         "j": "D", "journée": "D", "journées": "D", "journee": "D", "journees": "D",
         "min": "MIN", "minute": "MIN", "minutes": "MIN"}
#: 1.3 — a fraction written as one character (« ½ journée » → P0.5D)
FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75}

#: 1.4 — the decimal group takes NO whitespace: French writes 24,7 as one number and « 24, 7 » as
#: two. 1.3's `\s*,\s*` read « 24 heures sur 24, 7 jours sur 7 » as 24,7 days, so the canonical set
#: of that source held P24.7D and never P7D, and the pass-1 digit rule refused a sentence verbatim in its
#: own source — found on a recorded answer, not on a fixture. The integer group
#: keeps its spaces: a thousands separator in French is a space, « 1 000 heures ».
_WS = r"\s*"
_NUM = r"\d(?:[\d\s.  ]*\d)?(?:,\d+)?"


@dataclass(frozen=True)
class Value:
    """One typed value as written: its exact span, its kind, its normalised form or the reason it has none."""

    start: int
    end: int
    raw: str
    kind: str
    normalized: Optional[str]
    reason: Optional[str] = None
    unit: Optional[str] = None


def _number(text: str) -> Optional[float]:
    cleaned = re.sub(r"[\s  .]", "", text).replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if value % 1 else str(int(value))


_AMOUNT = re.compile(rf"(?<![\w,.])(?P<n>{_NUM})\s*(?P<cur>€|EUR\b|euros?\b)", re.IGNORECASE)
_PERCENT = re.compile(rf"(?<![\w,.])(?P<n>{_NUM})\s*(?P<cur>%|pour\s*cent\b)", re.IGNORECASE)
_DURATION = re.compile(rf"(?<![\w,.])(?P<n>{_NUM}|" + "|".join(sorted(list(WORDS) + list(FRACTIONS), key=len,
                                                                        reverse=True))
                       + r")\s*(?P<unit>" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")\b",
                       re.IGNORECASE)
#: 1.2 — hours written with the abbreviation. Most "critical value written" refusals of an early harvest
#: were unsatisfiable: the text says "24h" or "50 h", the models wrote it so, and this grammar read
#: nothing there, so no marker existed to cite instead. "h" is a duration only when no digit follows:
#: "14h30" and "14h 30" are clock times and stay out.
_HOURS = re.compile(rf"(?<![\w,.])(?P<n>{_NUM})\s*h\b(?!\s*\d)", re.IGNORECASE)
_QUANTITY = re.compile(rf"(?<![\w,.])(?P<n>{_NUM})\s*(?P<noun>" + "|".join(sorted(COUNTED, key=len, reverse=True))
                       + r")\b", re.IGNORECASE)
#: a reference is emitted in the convention's pointer form: <PIECE> <number>, <PIECE> annexe <n>, page n/m
_PIECES = ("RC", "CCAP", "CCTP", "CCAG-FCS", "CCAG", "AE", "BPU", "DUME")
_REFERENCE = re.compile(
    r"(?<!\w)(?:"
    r"(?P<art>(?:article|articles)\s+(?P<artnum>[\w.\-]+)(?:\s+(?:du|de\s+la|des)\s+"
    r"(?P<artpiece>CCAG\s*-\s*FCS|CCAG|CCAP|CCTP|RC|AE))?)"
    r"|(?P<ann>(?:annexe|annexes)\s+(?P<annnum>[\w.\-]+)(?:\s+(?:à|a|du|de\s+la|des)\s+"
    r"(?:l['’]\s*)?(?P<annpiece>Acte\s+d['’]\s*Engagement|CCTP|CCAP|CCAG|RC|AE))?)"
    r"|(?P<code>(?:R|L|D)\s*\.\s*\d{3,4}\s*-\s*\d{1,3})"
    r"|(?P<short>(?:RC|CCAP|CCTP|CCAG|AE|BPU|DUME|DC\d)\s*\d+(?:\.\d+)*)"
    r"|(?P<page>[Pp]age\s*(?P<pnum>\d+)\s*sur\s*(?P<ptot>\d+))"
    r")", re.IGNORECASE)


def _piece(text: str) -> str:
    folded = re.sub(r"[\s.]", "", text or "").upper().replace("-", "")
    if folded.startswith("ACTED"):
        return "AE"
    if folded.startswith("CCAGFCS"):
        return "CCAG-FCS"
    for piece in _PIECES:
        if folded.startswith(piece.replace("-", "")):
            return piece
    return folded


def _reference(m: re.Match) -> tuple:
    if m.group("page"):
        return (f"{m.group('pnum')}/{m.group('ptot')}", None, None)
    if m.group("code"):
        return (re.sub(r"\s+", "", m.group("code")).upper(), None, None)
    if m.group("short"):
        raw = re.sub(r"\s+", " ", m.group("short")).strip()
        head, _, num = raw.partition(" ")
        return (f"{_piece(head)} {num}".strip(), None, None)
    if m.group("art"):
        num = re.sub(r"\s+", "", m.group("artnum"))
        piece = _piece(m.group("artpiece")) if m.group("artpiece") else None
        return (f"{piece} {num}" if piece else f"article {num}", None, None)
    num = re.sub(r"\s+", "", m.group("annnum"))
    piece = _piece(m.group("annpiece")) if m.group("annpiece") else None
    return (f"{piece} annexe {num}" if piece else f"annexe {num}", None, None)


def _scan(text: str, pattern: re.Pattern, kind: str, normalise) -> list[Value]:
    out = []
    for m in pattern.finditer(text):
        normalized, reason, unit = normalise(m)
        out.append(Value(m.start(), m.end(), m.group(0), kind, normalized, reason, unit))
    return out


def _duration(m: re.Match) -> tuple:
    raw = re.sub(r"\s+", "", m["n"]).lower()
    n = WORDS.get(raw, FRACTIONS.get(raw, _number(m["n"])))
    if n is None:
        return (None, "not a number", None)
    slot = UNITS[re.sub(r"\s+", "", m["unit"]).lower()]
    if slot == "W":                       # weeks become days, so two writings compare (NORMALISATION.md)
        n, slot = n * 7, "D"
    count = _fmt(n)
    iso = f"PT{count}H" if slot == "H" else f"PT{count}M" if slot == "MIN" else f"P{count}{slot}"
    return (iso, None, slot)


def _hours(m: re.Match) -> tuple:
    n = _number(m["n"])
    if n is None:
        return (None, "not a number", None)
    return (f"PT{_fmt(n)}H", None, "H")


#: 1.5 — the head of a decimal whose tail was pushed onto the next line: a digit, the separator, then a
#: LINE BREAK before the reading starts. « 250,\n00 € » must never be read « 0.00 EUR »: that is not a
#: missing value but a wrong one, small where the true one is large.
#: The line break is the signature, and it has to be. « 24 heures sur 24, 7 jours sur 7 » is the same
#: shape with a space, it is an enumeration, and it is attested in recorded answers — declining it
#: would undo what 1.4 fixed. A comma and a space therefore stay two readings; a comma and a newline are
#: declined. The limit is recorded in tests/harvest/test_fr_grammars.py rather than hidden: a decimal split
#: by a SPACE is still read as its tail, and the reference corpus had no instance of it.
_SPLIT_DECIMAL_HEAD = re.compile(r"\d[ \t  ]*,[ \t]*\n\s*\Z")


def _is_split_decimal_tail(text: str, start: int) -> bool:
    return bool(_SPLIT_DECIMAL_HEAD.search(text[:start]))


def read_values(text: str, kinds: tuple[str, ...] = KINDS) -> list[Value]:
    """Every typed value of the declared kinds, in order, with overlaps resolved by the longest span."""
    found: list[Value] = []
    if "date" in kinds or "datetime" in kinds:
        for r in read_dates(text):
            found.append(Value(r.start, r.end, r.raw, r.type, r.normalized, r.reason))
    if "amount" in kinds:
        found += _scan(text, _AMOUNT, "amount", lambda m: (
            (f"{_number(m['n']):.2f} EUR", None, "EUR") if _number(m["n"]) is not None
            else (None, "not a number", "EUR")))
    if "percentage" in kinds:
        found += _scan(text, _PERCENT, "percentage", lambda m: (
            (_fmt(_number(m["n"])) + " %", None, "%") if _number(m["n"]) is not None
            else (None, "not a number", "%")))
    if "duration" in kinds:
        found += _scan(text, _DURATION, "duration", _duration)
        found += _scan(text, _HOURS, "duration", _hours)
    if "quantity" in kinds:
        found += _scan(text, _QUANTITY, "quantity", lambda m: (
            (f"{_fmt(_number(m['n']))} {re.sub(r'\\s+', '', m['noun']).lower()}", None, None)
            if _number(m["n"]) is not None else (None, "not a number", None)))
    if "reference" in kinds:
        found += _scan(text, _REFERENCE, "reference", _reference)
    if "period" in kinds:
        found += _periods(text, found)
    # 1.5: a reading that begins where a split decimal's tail begins is DECLINED — kept in the list
    # with its reason so a trace counts the refusal, never normalised so nothing downstream can use it.
    found = [v if not (v.raw[:1].isdigit() and _is_split_decimal_tail(text, v.start))
             else replace(v, normalized=None,
                          reason="declined: the tail of a decimal split by a line break "
                                 "(grammars_fr 1.5); the head is on the previous line")
             for v in found]
    found.sort(key=lambda v: (v.start, -(v.end - v.start)))
    kept: list[Value] = []
    for value in found:
        if kept and value.start < kept[-1].end:      # an overlap: the longer span wins, deterministically
            if (value.end - value.start) <= (kept[-1].end - kept[-1].start):
                continue
            kept.pop()
        kept.append(value)
    return kept


def _periods(text: str, found: list[Value]) -> list[Value]:
    """"du <date> au <date>" is ONE period (NORMALISATION.md rule 3), not two dates.

    When the start carries no year it takes the end's, and the value records that the year was borrowed:
    the completion is visible rather than silent, which is the one exception rule 2 allows.
    """
    import datetime as _dt

    from .dates import periods as date_periods
    from .dates import read as read_dates

    readings = read_dates(text)
    out = []
    for a, b in date_periods(text, readings):
        start_value, reason = a.normalized, None
        if start_value is None and a.year is None and b.year is not None:
            try:
                start_value = _dt.date(b.year, a.month, a.day).isoformat()
                reason = "the start's year is borrowed from the end"
            except ValueError:
                start_value = None
        if start_value and b.normalized:
            out.append(Value(a.start, b.end, text[a.start:b.end], "period",
                             f"{start_value}/{b.normalized}", reason))
        else:
            out.append(Value(a.start, b.end, text[a.start:b.end], "period", None,
                             "an endpoint could not be typed"))
    return out
