"""Immutable literal candidates and deterministic composites, without semantic policy.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15

This is a separate contract from harvest-form and the recurrence pass1 module.
No unit conversion, cross-document comparison, model, store or KOAS dependency.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re

from .fr.numbers import parse_decimal
from .fr.grammars import read_values

VERSION = "quantitative/1.0"
QUANTITATIVE_KINDS = frozenset({"scalar", "duration", "cardinality", "percentage", "interval",
                              "tolerance", "inequality", "symbolic_bound", "rate"})
UNITS = {
    "°C": ("degC", "temperature"), "°F": ("degF", "temperature"), "K": ("K", "temperature"),
    "V": ("V", "voltage"), "mV": ("mV", "voltage"), "W": ("W", "power"), "kW": ("kW", "power"),
    "Hz": ("Hz", "frequency"), "kHz": ("kHz", "frequency"), "Pa": ("Pa", "pressure"),
    "kPa": ("kPa", "pressure"), "bar": ("bar", "pressure"),
    **{u: (u, "length") for u in ("nm", "um", "µm", "mm", "cm", "m")},
    **{u: (u, "volume") for u in ("L", "mL", "uL", "µL")},
    **{u: (u, "mass") for u in ("kg", "g", "mg")}, "%": ("%", "percentage"),
    **{u: ("h", "time") for u in ("h", "heure", "heures")},
    **{u: ("min", "time") for u in ("min", "minute", "minutes")},
    **{u: ("s", "time") for u in ("s", "seconde", "secondes")},
    **{u: ("day", "time") for u in ("jour", "jours", "day", "days")},
    **{u: ("count", "count") for u in ("essai", "essais", "fois", "cycle", "cycles", "repetitions")},
}
NUMBER = r"(?:[+\u2212]|(?<!\+/)-)?\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?"
UNIT = "(?:" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True)) + ")"
SCALAR = re.compile(rf"(?<![\w.,])(?P<number>{NUMBER})\s*(?P<unit>{UNIT})(?!\w)")
PREFIX = re.compile(r"(?P<op><=|>=|≤|≥|<|>|=|au moins|au plus|minimum|maximum|"
                    r"(?:allant\s+)?jusqu['’][aà])\s*$", re.I)
NORMAL_OP = {"≤": "<=", "≥": ">=", "au moins": ">=", "au plus": "<=", "minimum": ">=", "maximum": "<="}


@dataclass(frozen=True)
class Member:
    candidate_id: str
    role: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source_id: str
    node_id: str
    start: int
    end: int
    raw: str
    kind: str
    number: str | None = None
    unit_raw: str | None = None
    unit_start: int | None = None
    unit_end: int | None = None
    unit: str | None = None
    dimension: str | None = None
    comparator_raw: str | None = None
    comparator_normalized: str | None = None
    direction_status: str = "not_applicable"
    normalization_status: str = "parsed"
    members: tuple[Member, ...] = ()
    lower: str | None = None
    upper: str | None = None
    nominal: str | None = None
    tolerance: str | None = None
    lower_inclusive: bool | None = None
    upper_inclusive: bool | None = None
    flags: tuple[str, ...] = ()
    producer: str = VERSION

    @property
    def quantitative(self):
        return self.kind in QUANTITATIVE_KINDS


def _make(source_id, node_id, text, start, end, kind, flags=(), **values):
    candidate = Candidate("", source_id, node_id, start, end, text[start:end], kind,
                          flags=tuple(sorted(set(flags))), **values)
    raw = json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return replace(candidate, candidate_id=hashlib.sha256(raw.encode()).hexdigest())


def harvest(text: str, *, source_id: str, node_id: str, classification: str,
            uncertainty=(), decimal_separator=None) -> tuple[Candidate, ...]:
    """Read classified evidence; return children as well as deterministic composites.

    Furniture produces no claims. UNKNOWN is retained with a review flag. Kind
    routing must use quantitative roots, not the count of all numeric-looking ids.
    """
    if not source_id or not node_id or classification not in {"CONTENT", "UNKNOWN", "FURNITURE"}:
        raise ValueError("source/node identity and explicit classification required")
    if classification == "FURNITURE":
        return ()
    flags = tuple(uncertainty) + (("CLASSIFICATION_UNKNOWN",) if classification == "UNKNOWN" else ())
    found, scalars = [], []
    identities = [(m.start(), m.end()) for m in re.finditer(r"\b[A-Z][A-Z0-9]*(?:-+[A-Z0-9]+)+\b", text)]
    for match in SCALAR.finditer(text):
        if any(a <= match.start() < match.end() <= b for a, b in identities):
            continue
        value = parse_decimal(match["number"], decimal_separator=decimal_separator)
        unit, dimension = UNITS[match["unit"]]
        kind = {"time": "duration", "count": "cardinality", "percentage": "percentage"}.get(dimension, "scalar")
        start, op, normalized, direction = match.start(), None, None, "not_applicable"
        prefix = PREFIX.search(text[:start])
        if prefix:
            start, op = prefix.start(), prefix["op"]
            direction = "unresolved" if "jusqu" in op.lower() else "resolved"
            normalized = NORMAL_OP.get(op.lower(), op) if direction == "resolved" else None
            kind = "inequality"
        extra = flags + (("DIRECTION_UNRESOLVED",) if direction == "unresolved" else ())
        extra += ("NUMBER_UNPARSED",) if value is None else ()
        candidate = _make(source_id, node_id, text, start, match.end(), kind, extra,
                          number=format(value, "f") if value is not None else None,
                          unit_raw=match["unit"], unit_start=match.start("unit"), unit_end=match.end("unit"),
                          unit=unit, dimension=dimension,
                          comparator_raw=op, comparator_normalized=normalized,
                          direction_status=direction, normalization_status="parsed" if value is not None else "unparsed")
        found.append(candidate)
        scalars.append((candidate, match))
    for (left, lm), (right, rm) in zip(scalars, scalars[1:]):
        gap = text[lm.end():rm.start()].strip()
        members, kind, values = (), None, {}
        if re.fullmatch(r"±|\+/-", gap):
            kind = "tolerance"
            members = (Member(left.candidate_id, "nominal"), Member(right.candidate_id, "tolerance"))
            values = {"nominal": left.number, "tolerance": right.number}
            compatible = (left.unit == right.unit or right.unit == "%") and (
                right.number is not None and parse_decimal(right.number, decimal_separator=".") >= 0)
        else:
            chain = re.fullmatch(r"(?P<a><=|>=|≤|≥|<|>)\s*[A-Za-z_°][\w°]*\s*(?P<b><=|>=|≤|≥|<|>)", gap)
            plain = re.fullmatch(r"à|a|to|\.\.|…|–|—|-", gap, re.I)
            if not (chain or plain):
                continue
            kind = "interval"
            lower, upper = left, right
            inclusive = (True, True)
            compatible = left.unit == right.unit
            if chain:
                a, b = (NORMAL_OP.get(chain[k], chain[k]) for k in ("a", "b"))
                compatible &= a[0] == b[0]
                if a[0] == ">":
                    lower, upper = right, left
                    inclusive = ("=" in b, "=" in a)
                else:
                    inclusive = ("=" in a, "=" in b)
            members = (Member(lower.candidate_id, "lower"), Member(upper.candidate_id, "upper"))
            values = {"lower": lower.number, "upper": upper.number,
                      "lower_inclusive": inclusive[0], "upper_inclusive": inclusive[1]}
            if lower.number is not None and upper.number is not None:
                compatible &= parse_decimal(lower.number, decimal_separator=".") <= parse_decimal(upper.number, decimal_separator=".")
        valid = compatible and left.normalization_status == right.normalization_status == "parsed"
        extra = flags + left.flags + right.flags + (() if valid else ("COMPOSITE_UNRESOLVED",))
        found.append(_make(source_id, node_id, text, left.start, right.end, kind, extra,
                           unit_raw=left.unit_raw, unit=left.unit, dimension=left.dimension,
                           comparator_raw=gap, comparator_normalized="bounded" if kind == "interval" else "tolerance",
                           direction_status="resolved" if valid else "unresolved",
                           normalization_status="parsed" if valid else "unparsed", members=members, **values))
    # An omitted left unit may be inherited only inside an explicit composite.
    shared = re.compile(rf"(?<![\w.,])(?P<left>{NUMBER})\s*"
                        rf"(?P<op>±|\+/-|à|to|\.\.|…|–|—|\s-\s)\s*"
                        rf"(?P<right>{NUMBER})\s*(?P<unit>{UNIT})(?!\w)", re.I)
    for match in shared.finditer(text):
        if any(match.start() < c.end and match.start("right") > c.start for c in scalars_only(found)):
            continue
        right = next((c for c in found if c.start == match.start("right") and c.end == match.end()), None)
        if match["unit"] not in UNITS:
            continue
        if right is None:
            number = parse_decimal(match["right"], decimal_separator=decimal_separator)
            unit, dimension = UNITS[match["unit"]]
            right = _make(source_id, node_id, text, match.start("right"), match.end(), "scalar", flags,
                          number=format(number, "f") if number is not None else None,
                          unit_raw=match["unit"], unit_start=match.start("unit"), unit_end=match.end("unit"),
                          unit=unit, dimension=dimension,
                          normalization_status="parsed" if number is not None else "unparsed")
            found.append(right)
        value = parse_decimal(match["left"], decimal_separator=decimal_separator)
        left = _make(source_id, node_id, text, match.start("left"), match.end("left"), "scalar", flags,
                     number=format(value, "f") if value is not None else None,
                     unit_raw=right.unit_raw, unit_start=right.unit_start, unit_end=right.unit_end,
                     unit=right.unit, dimension=right.dimension,
                     normalization_status="parsed" if value is not None else "unparsed")
        kind = "tolerance" if match["op"].strip() in {"±", "+/-"} else "interval"
        valid = left.number is not None and right.number is not None
        if valid:
            bound = parse_decimal(right.number, decimal_separator=".")
            valid = bound >= 0 if kind == "tolerance" else value <= bound
        roles = ("nominal", "tolerance") if kind == "tolerance" else ("lower", "upper")
        values = dict(zip(roles, (left.number, right.number)))
        if kind == "interval":
            values.update(lower_inclusive=True, upper_inclusive=True)
        found.extend((left, _make(source_id, node_id, text, match.start(), match.end(), kind,
                      flags + (() if valid else ("COMPOSITE_UNRESOLVED",)),
                      unit_raw=right.unit_raw, unit_start=right.unit_start, unit_end=right.unit_end,
                      unit=right.unit, dimension=right.dimension, comparator_raw=match["op"],
                      comparator_normalized="bounded" if kind == "interval" else "tolerance",
                      direction_status="resolved" if valid else "unresolved",
                      normalization_status="parsed" if valid else "unparsed",
                      members=(Member(left.candidate_id, roles[0]), Member(right.candidate_id, roles[1])), **values)))
    occupied = [(c.start, c.end) for c in found]
    symbolic = re.compile(r"(?P<op><=|>=|≤|≥)\s*(?P<symbol>[A-Za-zÀ-ÿ_][\wÀ-ÿ]*(?:[ \t]+[A-Za-zÀ-ÿ_][\wÀ-ÿ]*){0,2})")
    for match in symbolic.finditer(text):
        if any(match.start() < b and match.end() > a for a, b in occupied):
            continue
        found.append(_make(source_id, node_id, text, match.start(), match.end(), "symbolic_bound", flags,
                           comparator_raw=match["op"], comparator_normalized=NORMAL_OP.get(match["op"], match["op"]),
                           direction_status="resolved"))
    rate = re.compile(rf"(?<![\w.,])(?P<n>{NUMBER})\s+(?P<event>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ /-]{{0,60}}?)"
                      r"\s+(?:par|per)\s+(?P<period>jour|day|heure|hour|minute|seconde|second)s?\b", re.I)
    for match in rate.finditer(text):
        overlaps = [c for c in found if match.start() < c.end and match.end() > c.start]
        if any(c.kind != "cardinality" or not match.start() <= c.start < c.end <= match.end() for c in overlaps):
            continue
        number = parse_decimal(match["n"], decimal_separator=decimal_separator)
        found.append(_make(source_id, node_id, text, match.start(), match.end(), "rate",
                           flags + (() if number is not None else ("NUMBER_UNPARSED",)),
                           number=format(number, "f") if number is not None else None,
                           unit_raw=text[match.end("n"):match.end()].strip(), dimension="rate",
                           members=tuple(Member(c.candidate_id, "event_count") for c in overlaps),
                           normalization_status="parsed" if number is not None else "unparsed"))
    for value in read_values(text, kinds=("date", "datetime", "period", "reference")):
        if not any(value.start < b and value.end > a for a, b in occupied):
            found.append(_make(source_id, node_id, text, value.start, value.end, value.kind, flags))
    for pattern, kind in ((r"\b[A-Z][A-Z0-9]*(?:-+[A-Z0-9]+)+\b", "identifier"),
                          (r"\b(?:version|rev(?:ision)?|v)\s*[:.]?\s*\d+(?:\.\d+)*", "version"),
                          (r"\b(?:edition|édition)\s*[:.]?\s*[\w.-]+", "edition"),
                          (r"\b(?:part number|p/n)\s*[:.]?\s*[\w.-]+", "part_number")):
        for match in re.finditer(pattern, text, re.I if kind != "identifier" else 0):
            if not any(match.start() < c.end and match.end() > c.start for c in found):
                found.append(_make(source_id, node_id, text, match.start(), match.end(), kind, flags))
    return tuple(sorted(found, key=lambda c: (c.start, c.end, c.kind, c.candidate_id)))


def roots(candidates):
    candidates = tuple(candidates)
    members = {m.candidate_id for c in candidates for m in c.members}
    return tuple(c for c in candidates if c.candidate_id not in members)


def scalars_only(candidates):
    return tuple(c for c in candidates if c.quantitative and not c.members)


def route(candidates, *, semantic_cue: bool) -> str:
    return "Q+" if any(c.quantitative for c in roots(candidates)) else ("S+/Q-" if semantic_cue else "S-/Q-")
