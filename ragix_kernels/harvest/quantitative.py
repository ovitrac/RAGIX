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
TOKEN_LOCALE_VERSION = "quantitative/1.1"
QUANTITATIVE_KINDS = frozenset(
    {
        "scalar",
        "duration",
        "cardinality",
        "percentage",
        "interval",
        "tolerance",
        "inequality",
        "equality",
        "symbolic_bound",
        "rate",
    }
)
UNITS = {
    "L/min": ("L/min", "volume_rate"),
    "°C": ("degC", "temperature"),
    "°F": ("degF", "temperature"),
    "K": ("K", "temperature"),
    "V": ("V", "voltage"),
    "mV": ("mV", "voltage"),
    "W": ("W", "power"),
    "kW": ("kW", "power"),
    "Hz": ("Hz", "frequency"),
    "kHz": ("kHz", "frequency"),
    "Pa": ("Pa", "pressure"),
    "kPa": ("kPa", "pressure"),
    "bar": ("bar", "pressure"),
    **{u: (u, "length") for u in ("nm", "um", "µm", "mm", "cm", "m")},
    **{u: (u, "volume") for u in ("L", "mL", "uL", "µL")},
    **{u: (u, "mass") for u in ("kg", "g", "mg")},
    "%": ("%", "percentage"),
    **{u: ("h", "time") for u in ("h", "heure", "heures")},
    **{u: ("min", "time") for u in ("min", "minute", "minutes")},
    **{u: ("s", "time") for u in ("s", "seconde", "secondes")},
    **{u: ("day", "time") for u in ("jour", "jours", "day", "days")},
    **{
        u: ("count", "count") for u in ("essai", "essais", "fois", "cycle", "cycles", "repetitions")
    },
}
NUMBER = r"(?:[+\u2212]|(?<!\+/)-)?\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?"
UNIT = "(?:" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True)) + ")"
#: a currency sign after a unit letter is money (« 10 K€ »), never a physical unit
UNIT_END = r"(?![\w€$£])"
SCALAR = re.compile(rf"(?<![\w.,])(?P<number>{NUMBER})\s*(?P<unit>{UNIT}){UNIT_END}")
PREFIX = re.compile(
    r"(?P<op><=|>=|≤|≥|<|>|=|au moins|au plus|minimum|maximum|"
    r"(?<!\w)(?:at least|at most|not more than|not less than|minimal|maximal)|"
    r"(?:allant\s+)?jusqu['’][aà])\s*$",
    re.I,
)
NORMAL_OP = {
    "≤": "<=",
    "≥": ">=",
    "au moins": ">=",
    "au plus": "<=",
    "minimum": ">=",
    "maximum": "<=",
    "maximal": "<=",
    "minimal": ">=",
    "at least": ">=",
    "at most": "<=",
    "not more than": "<=",
    "not less than": ">=",
}


def notation_flags(raw, decimal_separator=None):
    flags = []
    if re.search(r"\d[ \u00a0\u202f]\d{3}", raw):
        flags.append("GROUPING_ASSUMED")
    if decimal_separator is None and re.fullmatch(r"[+−-]?[1-9]\d{0,2}[.,]\d{3}", raw):
        flags.append("GROUPING_AMBIGUOUS")
    return tuple(flags)


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
    def needs_review(self):
        return bool(
            self.flags
            or self.normalization_status != "parsed"
            or self.direction_status == "unresolved"
        )

    @property
    def quantitative(self):
        return self.kind in QUANTITATIVE_KINDS


def _make(source_id, node_id, text, start, end, kind, flags=(), **values):
    candidate = Candidate(
        "",
        source_id,
        node_id,
        start,
        end,
        text[start:end],
        kind,
        flags=tuple(sorted(set(flags))),
        **values,
    )
    raw = json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return replace(candidate, candidate_id=hashlib.sha256(raw.encode()).hexdigest())


def harvest(
    text: str,
    *,
    source_id: str,
    node_id: str,
    classification: str,
    uncertainty=(),
    decimal_separator=None,
    token_locale=False,
    locale_prior=None,
    table_cell=False,
    context=None,
) -> tuple[Candidate, ...]:
    """Read classified evidence; return children as well as deterministic composites.

    Furniture produces no claims. UNKNOWN is retained with a review flag. Kind
    routing must use quantitative roots, not the count of all numeric-looking ids.
    """
    if context is not None:
        from .contextual_quantitative import harvest_cell

        return harvest_cell(
            text,
            context=context,
            source_id=source_id,
            node_id=node_id,
            classification=classification,
            uncertainty=uncertainty,
            decimal_separator=decimal_separator,
            token_locale=token_locale,
            locale_prior=locale_prior,
        )
    if not source_id or not node_id or classification not in {"CONTENT", "UNKNOWN", "FURNITURE"}:
        raise ValueError("source/node identity and explicit classification required")
    if classification == "FURNITURE":
        return ()
    flags = tuple(uncertainty) + (
        ("CLASSIFICATION_UNKNOWN",) if classification == "UNKNOWN" else ()
    )

    def make(*args, **kwargs):
        return _make(*args, producer=TOKEN_LOCALE_VERSION if token_locale else VERSION, **kwargs)

    found, scalars = [], []
    from .numeric_locale import (
        resolve_number,
        nonphysical_spans,
        physical_numbers,
        NUMBER as TOKEN_NUMBER,
    )

    def parse_literal(raw):
        return (
            resolve_number(raw, locale_prior).value
            if token_locale
            else parse_decimal(raw, decimal_separator=decimal_separator)
        )

    def literal_flags(raw):
        return (
            resolve_number(raw, locale_prior).flags
            if token_locale
            else notation_flags(raw, decimal_separator)
        )

    number_pattern = (
        TOKEN_NUMBER.replace("[+−-]?", r"(?:[+−]|(?<!\+/)-)?", 1) if token_locale else NUMBER
    )
    scalar_pattern = (
        re.compile(rf"(?<![\w.,])(?P<number>{number_pattern})\s*(?P<unit>{UNIT}){UNIT_END}")
        if token_locale
        else SCALAR
    )
    excluded = nonphysical_spans(text, table_cell=table_cell) if token_locale else ()
    identities = [
        (m.start(), m.end()) for m in re.finditer(r"\b[A-Z][A-Z0-9]*(?:-+[A-Z0-9]+)+\b", text)
    ]
    for match in scalar_pattern.finditer(text):
        if any(match.start() < b and match.end() > a for a, b, _ in excluded):
            continue
        if any(a <= match.start() < match.end() <= b for a, b in identities):
            continue
        value = parse_literal(match["number"])
        unit, dimension = UNITS[match["unit"]]
        kind = {"time": "duration", "count": "cardinality", "percentage": "percentage"}.get(
            dimension, "scalar"
        )
        if kind == "percentage" and match["number"].startswith(("-", "−", "+")):
            kind = "scalar"
        end = unit_end = match.end()
        if dimension == "voltage":
            qualifier = re.match(r"[ \t]+(?:AC|DC)\b", text[end:])
            if qualifier:
                end += qualifier.end()
                unit_end = end
        start, op, normalized, direction = match.start(), None, None, "not_applicable"
        prefix = PREFIX.search(text[:start])
        if prefix:
            start, op = prefix.start(), prefix["op"]
            direction = "unresolved" if "jusqu" in op.lower() else "resolved"
            normalized = NORMAL_OP.get(op.lower(), op) if direction == "resolved" else None
            kind = "equality" if normalized == "=" else "inequality"
        postfix = re.match(r"\s+(?P<op>maximal|minimal|maximum|minimum)\b", text[end:], re.I)
        if postfix and op is None:
            op = postfix["op"]
            normalized, direction, kind = NORMAL_OP[op.lower()], "resolved", "inequality"
            end += postfix.end()
        extra = flags + (("DIRECTION_UNRESOLVED",) if direction == "unresolved" else ())
        extra += ("NUMBER_UNPARSED",) if value is None else ()
        extra += literal_flags(match["number"])
        if match["unit"] == "K" and match.end("number") == match.start("unit"):
            extra += ("UNIT_AMBIGUOUS_K",)
        candidate = make(
            source_id,
            node_id,
            text,
            start,
            end,
            kind,
            extra,
            number=format(value, "f") if value is not None else None,
            unit_raw=text[match.start("unit") : unit_end],
            unit_start=match.start("unit"),
            unit_end=unit_end,
            unit=unit,
            dimension=dimension,
            comparator_raw=op,
            comparator_normalized=normalized,
            direction_status=direction,
            normalization_status="parsed" if value is not None else "unparsed",
        )
        found.append(candidate)
        scalars.append((candidate, match))
    for (left, lm), (right, rm) in zip(scalars, scalars[1:]):
        raw_gap = text[left.end : rm.start()]
        gap = raw_gap.strip()
        members, kind, values = (), None, {}
        if re.fullmatch(r"±|\+/-", gap):
            kind = "tolerance"
            members = (
                Member(left.candidate_id, "nominal"),
                Member(right.candidate_id, "tolerance"),
            )
            values = {"nominal": left.number, "tolerance": right.number}
            compatible = (left.unit == right.unit or right.unit == "%") and (
                right.number is not None and parse_decimal(right.number, decimal_separator=".") >= 0
            )
        else:
            chain = re.fullmatch(
                r"(?P<a><=|>=|≤|≥|<|>)\s*[A-Za-z_°][\w°]*\s*(?P<b><=|>=|≤|≥|<|>)", gap
            )
            plain = re.fullmatch(r"à|a|to|\.\.|…|–|—|-", gap, re.I)
            if not (chain or plain):
                continue
            if plain and gap in {"-", "–", "—"} and left.unit != right.unit:
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
            values = {
                "lower": lower.number,
                "upper": upper.number,
                "lower_inclusive": inclusive[0],
                "upper_inclusive": inclusive[1],
            }
            if lower.number is not None and upper.number is not None:
                compatible &= parse_decimal(lower.number, decimal_separator=".") <= parse_decimal(
                    upper.number, decimal_separator="."
                )
        valid = compatible and left.normalization_status == right.normalization_status == "parsed"
        extra = flags + left.flags + right.flags + (() if valid else ("COMPOSITE_UNRESOLVED",))
        if kind == "tolerance" and raw_gap == gap:
            extra += ("GLUED",)
        found.append(
            make(
                source_id,
                node_id,
                text,
                left.start,
                right.end,
                kind,
                extra,
                unit_raw=left.unit_raw,
                unit=left.unit,
                dimension=left.dimension,
                comparator_raw=gap,
                comparator_normalized="bounded" if kind == "interval" else "tolerance",
                direction_status="resolved" if valid else "unresolved",
                normalization_status="parsed" if valid else "unparsed",
                members=members,
                **values,
            )
        )
    # An omitted left unit may be inherited only inside an explicit composite, and is flagged:
    # « Test 3 à 37 °C » has the shape of « 8 à 19 °C », so an inherited unit is never ready as read.
    shared = re.compile(
        rf"(?<![\w.,])(?P<left>{number_pattern})\s*"
        rf"(?P<op>±|\+/-|à|to|et|and|\.\.|…|–|—|-)\s*"
        rf"(?P<right>{number_pattern})\s*(?P<unit>{UNIT}){UNIT_END}",
        re.I,
    )
    for match in shared.finditer(text):
        if any(match.start() < b and match.end() > a for a, b, _ in excluded):
            continue
        intro = re.search(r"\b(?:entre|between|de|from)\s*$", text[: match.start()], re.I)
        if match["op"].lower() in {"et", "and"} and not intro:
            continue
        if any(
            match.start() < c.end and match.start("right") > c.start for c in scalars_only(found)
        ):
            continue
        right = next(
            (c for c in found if c.start == match.start("right") and c.end == match.end()), None
        )
        if match["unit"] not in UNITS:
            continue
        if right is None:
            number = parse_literal(match["right"])
            unit, dimension = UNITS[match["unit"]]
            right = make(
                source_id,
                node_id,
                text,
                match.start("right"),
                match.end(),
                "scalar",
                flags + literal_flags(match["right"]),
                number=format(number, "f") if number is not None else None,
                unit_raw=match["unit"],
                unit_start=match.start("unit"),
                unit_end=match.end("unit"),
                unit=unit,
                dimension=dimension,
                normalization_status="parsed" if number is not None else "unparsed",
            )
            found.append(right)
        value = parse_literal(match["left"])
        shared_flags = ("UNIT_INHERITED",) + literal_flags(match["left"])
        if intro and match["op"] not in {"±", "+/-"}:
            shared_flags += ("UNIT_SHARED",)
        if (
            match["op"] in {"±", "+/-"}
            and match.end("left") == match.start("op")
            and match.end("op") == match.start("right")
        ):
            shared_flags += ("GLUED",)
        if match["op"] == "-":
            shared_flags += ("SIGN_RANGE_AMBIGUOUS",)
        if (
            not intro
            and re.search(r"[^\W\d_]\s+$", text[: match.start()])
            and re.fullmatch(r"\d+", match["left"])
        ):
            shared_flags += ("LABEL_NUMBER_SUSPECTED",)
        left = make(
            source_id,
            node_id,
            text,
            match.start("left"),
            match.end("left"),
            "scalar",
            flags + shared_flags,
            number=format(value, "f") if value is not None else None,
            unit_raw=right.unit_raw,
            unit_start=right.unit_start,
            unit_end=right.unit_end,
            unit=right.unit,
            dimension=right.dimension,
            normalization_status="parsed" if value is not None else "unparsed",
        )
        kind = "tolerance" if match["op"].strip() in {"±", "+/-"} else "interval"
        valid = left.number is not None and right.number is not None
        if valid:
            bound = parse_decimal(right.number, decimal_separator=".")
            valid = bound >= 0 if kind == "tolerance" else value <= bound
        roles = ("nominal", "tolerance") if kind == "tolerance" else ("lower", "upper")
        values = dict(zip(roles, (left.number, right.number)))
        if kind == "interval":
            values.update(lower_inclusive=True, upper_inclusive=True)
        found.extend(
            (
                left,
                make(
                    source_id,
                    node_id,
                    text,
                    intro.start() if intro else match.start(),
                    match.end(),
                    kind,
                    flags + left.flags + right.flags + (() if valid else ("COMPOSITE_UNRESOLVED",)),
                    unit_raw=right.unit_raw,
                    unit_start=right.unit_start,
                    unit_end=right.unit_end,
                    unit=right.unit,
                    dimension=right.dimension,
                    comparator_raw=match["op"],
                    comparator_normalized="bounded" if kind == "interval" else "tolerance",
                    direction_status="resolved" if valid else "unresolved",
                    normalization_status="parsed" if valid else "unparsed",
                    members=(
                        Member(left.candidate_id, roles[0]),
                        Member(right.candidate_id, roles[1]),
                    ),
                    **values,
                ),
            )
        )
    # A trailing tolerance operand can borrow the nominal unit. Its own exact
    # span remains unitless and the inheritance flag reaches the composite.
    for nominal, match in scalars:
        tail = re.match(
            rf"\s*(?P<op>±|\+/-)\s*(?P<n>{number_pattern})(?![\w.,])", text[match.end() :]
        )
        if not tail:
            continue
        start, end = match.end() + tail.start("n"), match.end() + tail.end("n")
        if any(c.start <= start < c.end for c in found):
            continue
        value = parse_literal(tail["n"])
        extra = flags + ("UNIT_INHERITED",) + literal_flags(tail["n"])
        operand = make(
            source_id,
            node_id,
            text,
            start,
            end,
            "scalar",
            extra,
            number=format(value, "f") if value is not None else None,
            unit=nominal.unit,
            dimension=nominal.dimension,
            unit_raw=nominal.unit_raw,
            unit_start=nominal.unit_start,
            unit_end=nominal.unit_end,
            normalization_status="parsed" if value is not None else "unparsed",
        )
        valid = value is not None and value >= 0 and nominal.number is not None
        found.extend(
            (
                operand,
                make(
                    source_id,
                    node_id,
                    text,
                    nominal.start,
                    end,
                    "tolerance",
                    extra + nominal.flags + (() if valid else ("COMPOSITE_UNRESOLVED",)),
                    nominal=nominal.number,
                    tolerance=operand.number,
                    unit=nominal.unit,
                    unit_raw=nominal.unit_raw,
                    dimension=nominal.dimension,
                    comparator_raw=tail["op"],
                    comparator_normalized="tolerance",
                    direction_status="resolved" if valid else "unresolved",
                    normalization_status="parsed" if valid else "unparsed",
                    members=(
                        Member(nominal.candidate_id, "nominal"),
                        Member(operand.candidate_id, "tolerance"),
                    ),
                ),
            )
        )
    found.extend(_span_composites(text, found, make, flags))
    if token_locale:
        for observed in physical_numbers(text, table_cell=table_cell):
            if any(observed.start < c.end and observed.end > c.start for c in found):
                continue
            value = parse_literal(observed.raw)
            unresolved_direction = bool(
                observed.comparator and "jusqu" in observed.comparator.casefold()
            )
            start = (
                observed.comparator_start
                if observed.comparator_start is not None
                else observed.start
            )
            found.append(
                make(
                    source_id,
                    node_id,
                    text,
                    start,
                    observed.end,
                    "inequality" if observed.comparator else "scalar",
                    flags
                    + literal_flags(observed.raw)
                    + (("DIRECTION_UNRESOLVED",) if unresolved_direction else ())
                    + (() if value is not None else ("NUMBER_UNPARSED",)),
                    number=format(value, "f") if value is not None else None,
                    normalization_status="parsed" if value is not None else "unparsed",
                    comparator_raw=observed.comparator,
                    comparator_normalized=(
                        None
                        if unresolved_direction
                        else (
                            NORMAL_OP.get(observed.comparator.casefold(), observed.comparator)
                            if observed.comparator
                            else None
                        )
                    ),
                    direction_status=(
                        "unresolved"
                        if unresolved_direction
                        else "resolved" if observed.comparator else "not_applicable"
                    ),
                )
            )
    occupied = [(c.start, c.end) for c in found]
    symbolic = re.compile(
        r"(?P<op><=|>=|≤|≥)\s*(?P<symbol>[A-Za-zÀ-ÿ_][\wÀ-ÿ]*(?:[ \t]+[A-Za-zÀ-ÿ_][\wÀ-ÿ]*){0,2})"
    )
    for match in symbolic.finditer(text):
        if any(match.start() < b and match.end() > a for a, b in occupied):
            continue
        found.append(
            make(
                source_id,
                node_id,
                text,
                match.start(),
                match.end(),
                "symbolic_bound",
                flags,
                comparator_raw=match["op"],
                comparator_normalized=NORMAL_OP.get(match["op"], match["op"]),
                direction_status="resolved",
            )
        )
    rate = re.compile(
        rf"(?<![\w.,])(?:(?P<n>{number_pattern})\s+(?P<event>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ /-]{{0,60}}?)|(?P<word>once|twice))"
        r"\s+(?:par|per)\s+(?P<period>jour|day|heure|hour|minute|seconde|second|semaine|week|mois|month|année|year)s?\b",
        re.I,
    )
    periods = {
        "jour": "day",
        "day": "day",
        "heure": "hour",
        "hour": "hour",
        "minute": "minute",
        "seconde": "second",
        "second": "second",
        "semaine": "week",
        "week": "week",
        "mois": "month",
        "month": "month",
        "année": "year",
        "year": "year",
    }
    for match in rate.finditer(text):
        overlaps = [c for c in found if match.start() < c.end and match.end() > c.start]
        if any(
            c.kind != "cardinality" or not match.start() <= c.start < c.end <= match.end()
            for c in overlaps
        ):
            continue
        word = match["word"]
        number = (
            parse_literal(str({"once": 1, "twice": 2}[word.lower()]))
            if word
            else parse_literal(match["n"])
        )
        unit_start = match.end("word" if word else "n")
        unit_start += len(text[unit_start : match.end()]) - len(
            text[unit_start : match.end()].lstrip()
        )
        found.append(
            make(
                source_id,
                node_id,
                text,
                match.start(),
                match.end(),
                "rate",
                flags
                + (literal_flags(match["n"]) if token_locale and not word else ())
                + (() if number is not None else ("NUMBER_UNPARSED",)),
                number=format(number, "f") if number is not None else None,
                unit_raw=text[unit_start : match.end()],
                unit_start=unit_start,
                unit_end=match.end(),
                unit="count/" + periods[match["period"].lower()],
                dimension="rate",
                members=tuple(Member(c.candidate_id, "event_count") for c in overlaps),
                normalization_status="parsed" if number is not None else "unparsed",
            )
        )
    for value in read_values(text, kinds=("date", "datetime", "period", "reference")):
        if not any(value.start < b and value.end > a for a, b in occupied):
            found.append(make(source_id, node_id, text, value.start, value.end, value.kind, flags))
    for pattern, kind in (
        (r"\b[A-Z][A-Z0-9]*(?:-+[A-Z0-9]+)+\b", "identifier"),
        (r"\b(?:version|rev(?:ision)?|v)\s*[:.]?\s*\d+(?:\.\d+)*", "version"),
        (r"\b(?:edition|édition)\s*[:.]?\s*[\w.-]+", "edition"),
        (r"\b(?:part number|p/n)\s*[:.]?\s*[\w.-]+", "part_number"),
    ):
        for match in re.finditer(pattern, text, re.I if kind != "identifier" else 0):
            if not any(match.start() < c.end and match.end() > c.start for c in found):
                found.append(
                    make(source_id, node_id, text, match.start(), match.end(), kind, flags)
                )
    return tuple(sorted(found, key=lambda c: (c.start, c.end, c.kind, c.candidate_id)))


def _span_composites(text, candidates, make, flags):
    extra = []
    used = {m.candidate_id for c in candidates for m in c.members}
    leaves = [
        c for c in candidates if c.quantitative and not c.members and c.candidate_id not in used
    ]
    for c in leaves:
        leading = re.search(r"(?P<op>±|\+/-)\s*$", text[: c.start])
        if not leading or c.comparator_raw is not None:
            continue
        valid = c.number is not None and parse_decimal(c.number, decimal_separator=".") >= 0
        extra.append(
            make(
                c.source_id,
                c.node_id,
                text,
                leading.start(),
                c.end,
                "tolerance",
                flags
                + c.flags
                + ("NOMINAL_ABSENT",)
                + (() if valid else ("COMPOSITE_UNRESOLVED",)),
                tolerance=c.number,
                nominal=None,
                unit_raw=c.unit_raw,
                unit_start=c.unit_start,
                unit_end=c.unit_end,
                unit=c.unit,
                dimension=c.dimension,
                comparator_raw=leading["op"],
                comparator_normalized="tolerance",
                normalization_status="parsed" if valid else "unparsed",
                members=(Member(c.candidate_id, "tolerance"),),
            )
        )
        used.add(c.candidate_id)
    group = []
    for c in sorted(leaves, key=lambda value: value.start) + [None]:
        eligible = c is not None and c.kind == "duration" and c.candidate_id not in used
        if group and (
            not eligible
            or text[group[-1].end : c.start].strip()
            or c.unit in {m.unit for m in group}
        ):
            if len(group) > 1:
                first, last = group[0], group[-1]
                valid = all(member.normalization_status == "parsed" for member in group)
                extra.append(
                    make(
                        first.source_id,
                        first.node_id,
                        text,
                        first.start,
                        last.end,
                        "duration",
                        flags + tuple(f for member in group for f in member.flags),
                        dimension="time",
                        normalization_status="parsed" if valid else "unparsed",
                        members=tuple(Member(member.candidate_id, "component") for member in group),
                    )
                )
            group = []
        if eligible:
            group.append(c)
    return extra


def roots(candidates):
    candidates = tuple(candidates)
    members = {m.candidate_id for c in candidates for m in c.members}
    return tuple(c for c in candidates if c.candidate_id not in members)


def scalars_only(candidates):
    return tuple(c for c in candidates if c.quantitative and not c.members)


def route(candidates, *, semantic_cue: bool) -> str:
    return (
        "Q+"
        if any(c.quantitative for c in roots(candidates))
        else ("S+/Q-" if semantic_cue else "S-/Q-")
    )
