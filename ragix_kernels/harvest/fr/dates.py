"""ragix_kernels.harvest.fr.dates — the French date and clock-time grammar over exact spans.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

A deterministic grammar, no model. Text extracted from a PDF can arrive with a token split by line
breaks ("Vendre\\ndi", "202\\n6", "1\\n8" for 18, "jui\\nllet"), so whitespace, line breaks included,
is accepted between any two characters of a token; the span returned is always the exact slice of
the text it was read from. Every value is typed and normalised to ISO 8601, local time as written.

A date whose year is not written is returned incomplete (normalized None, reason "year absent"):
nothing is completed from context here. Completion is a derivation, recorded on a claim.

Families:
  - "[weekday] day month [year] [à HH heures MM | à HHhMM]", the day possibly "1er";
  - "DD/MM/YYYY [à ...]".
One period rule, "du <date> au <date>" (``periods``), which a visit window needs.

The grammar keeps the name and version it was first published under (``tender.dates_fr 1.0``): records
already written carry that string as their channel, and a renamed channel would split one instrument in two.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Optional

GRAMMAR = "tender.dates_fr"
VERSION = "1.0"
CHANNEL = f"{GRAMMAR} {VERSION}"

MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
          "octobre", "novembre", "décembre")
WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")

_ACCENTS = {"é": "[ée]", "û": "[ûu]"}


def _spread(word: str) -> str:
    """The word's letters, with any whitespace (line breaks included) allowed between two of them."""
    return r"\s*".join(_ACCENTS.get(c, re.escape(c)) for c in word)


def _fold(text: str) -> str:
    return re.sub(r"\s+", "", text).lower().replace("é", "e").replace("û", "u")


_MONTH_OF = {_fold(m): i for i, m in enumerate(MONTHS, 1)}

_TIME = (r"(?:\s+à\s+(?P<hour>[0-2]?\s*\d)\s*(?:" + _spread("heure") + r"(?:\s*s)?|h)"
         r"(?:\s*(?P<minute>[0-5]\s*\d))?(?!\d))?")

_NAMED = re.compile(
    r"(?<!\w)"
    r"(?:(?P<weekday>" + "|".join(_spread(w) for w in WEEKDAYS) + r")\s+)?"
    r"(?P<day>1\s*er|[0-3]?\s*\d)"
    r"\s+(?P<month>" + "|".join(_spread(m) for m in sorted(MONTHS, key=len, reverse=True)) + r")(?!\w)"
    r"(?:\s+(?P<year>2\s*0\s*\d\s*\d)(?!\w))?"
    + _TIME,
    re.IGNORECASE,
)
_NUMERIC = re.compile(
    r"(?<![\w/])(?P<day>[0-3]?\d)\s*/\s*(?P<month>[01]?\d)\s*/\s*(?P<year>\d{4})(?![\w/])" + _TIME,
    re.IGNORECASE,
)
_DU_BEFORE = re.compile(r"(?<!\w)du\s+$", re.IGNORECASE)
_AU_BETWEEN = re.compile(r"\s+au\s+", re.IGNORECASE)


@dataclass(frozen=True)
class Reading:
    """One date expression as written: its exact span, its type, its value or the reason it has none."""

    start: int
    end: int
    raw: str
    type: str                      # date | datetime
    normalized: Optional[str]      # ISO 8601, or None when the value cannot be typed as written
    reason: Optional[str]          # why normalized is None
    day: int
    month: int
    year: Optional[int]
    hour: Optional[int]
    minute: Optional[int]


def _digits(text: str) -> int:
    return int(re.sub(r"\s+", "", text))


def _reading(m: re.Match, day: int, month: int, year: Optional[int],
             hour: Optional[int], minute: Optional[int]) -> Reading:
    rtype = "date" if hour is None else "datetime"
    normalized, reason = None, None
    if year is None:
        reason = "year absent"
    else:
        try:
            date = dt.date(year, month, day)
        except ValueError:
            reason = "not a calendar date"
        else:
            if hour is None:
                normalized = date.isoformat()
            elif hour > 23 or minute > 59:
                reason = "not a clock time"
            else:
                normalized = f"{date.isoformat()}T{hour:02d}:{minute:02d}"
    return Reading(m.start(), m.end(), m.group(0), rtype, normalized, reason, day, month, year, hour, minute)


def read(text: str) -> list[Reading]:
    """Every date expression of ``text``, in order. Overlapping readings are refused, never arbitrated."""
    found = []
    for family, pattern in (("named", _NAMED), ("numeric", _NUMERIC)):
        for m in pattern.finditer(text):
            day_text = re.sub(r"\s+", "", m["day"]).lower()
            day = 1 if day_text == "1er" else int(day_text)
            month = _MONTH_OF[_fold(m["month"])] if family == "named" else int(m["month"])
            year = _digits(m["year"]) if m["year"] else None
            hour = _digits(m["hour"]) if m["hour"] else None
            minute = _digits(m["minute"]) if m["minute"] else (0 if hour is not None else None)
            found.append(_reading(m, day, month, year, hour, minute))
    found.sort(key=lambda r: (r.start, r.end))
    for a, b in zip(found, found[1:]):
        if b.start < a.end:
            raise ValueError(f"two readings overlap: {a.start}..{a.end} and {b.start}..{b.end}")
    return found


def periods(text: str, readings: list[Reading]) -> list[tuple[Reading, Reading]]:
    """The one period rule: two consecutive readings written "du <date> au <date>"."""
    return [(a, b) for a, b in zip(readings, readings[1:])
            if _AU_BETWEEN.fullmatch(text[a.end:b.start])
            and _DU_BEFORE.search(text[max(0, a.start - 8):a.start])]
