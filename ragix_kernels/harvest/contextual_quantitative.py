"""Quantities with exact table-context unit provenance; no parameter semantics.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from .quantitative import Candidate, Member, UNITS, UNIT, UNIT_END, _make, notation_flags, NORMAL_OP
from .table_context import CellContext, CellSpan
from .numeric_locale import physical_numbers, resolve_number
from .fr.numbers import parse_decimal

VERSION = "quantitative-context/1"


@dataclass(frozen=True)
class UnitMention:
    unit: str
    dimension: str
    span: CellSpan
    association: str


@dataclass(frozen=True)
class ContextualCandidate(Candidate):
    unit_source: str = "NONE"
    unit_evidence: tuple[UnitMention, ...] = ()
    context_evidence: tuple[CellSpan, ...] = ()
    unit_reason: str | None = None


def identify(candidate):
    candidate = replace(candidate, candidate_id="")
    from .report import canonical_json

    payload = canonical_json(asdict(candidate))
    return replace(candidate, candidate_id=hashlib.sha256(payload.encode()).hexdigest())


CONTEXT_UNIT = rf"(?:°[ \t]*[CF]|{UNIT})"
BRACKET_UNIT = re.compile(rf"[\[(]\s*(?P<unit>{CONTEXT_UNIT})\s*[\])]")
TRAILING_UNIT = re.compile(rf"(?<!\w)(?P<unit>{CONTEXT_UNIT}){UNIT_END}\s*$")


def unit_mentions(context):
    result = []
    for role, cells in [
        ("column_header", context.column_headers),
        ("row_label", context.row_labels),
    ]:
        for cell in cells:
            matches = list(BRACKET_UNIT.finditer(cell.text))
            tail = TRAILING_UNIT.search(cell.text)
            if tail and not any(m.start("unit") == tail.start("unit") for m in matches):
                prefix = cell.text[: tail.start()].strip()
                # A numeric value in another cell is not a unit declaration.
                if not prefix or (re.search(r"[^\W\d_]", prefix) and not re.search(r"\d", prefix)):
                    matches.append(tail)
            residual = cell.text
            for match in reversed(matches):
                residual = residual[: match.start()] + residual[match.end() :]
            if re.search(r"\d", residual) and not re.search(r"[^\W\d_]", residual):
                continue
            for match in matches:
                raw = match["unit"]
                key = re.sub(r"[ \t]+", "", raw) if raw.startswith("°") else raw
                unit, dimension = UNITS[key]
                result.append(
                    UnitMention(
                        unit, dimension, cell.span(match.start("unit"), match.end("unit")), role
                    )
                )
    return tuple(sorted(result, key=lambda m: (m.span.page, m.span.cell_id, m.span.start, m.unit)))


def bind_unit(candidate, context):
    mentions = unit_mentions(context)
    choices = {(m.unit, m.dimension) for m in mentions}
    flags = set(candidate.flags)
    evidence = tuple(c.span() for c in context.column_headers + context.row_labels)
    values = asdict(candidate)
    values["members"] = candidate.members
    source = "NONE"
    reason = None
    proof = ()
    if candidate.unit is not None:
        source = "INHERITED" if "UNIT_INHERITED" in candidate.flags else "INLINE"
        if candidate.unit_start is not None and candidate.unit_end is not None:
            proof = (
                UnitMention(
                    candidate.unit,
                    candidate.dimension,
                    context.value.span(candidate.unit_start, candidate.unit_end),
                    "value",
                ),
            )
        if choices and choices != {(candidate.unit, candidate.dimension)}:
            flags.add("UNIT_CONTEXT_CONFLICT")
            reason = "INLINE_UNIT_CONFLICTS_WITH_CONTEXT"
    elif re.match(r"\s*(?:[€$£]|[^\W\d_])", context.value.text[candidate.end :]) and not re.match(
        r"\s*(?:à|au|to)\s+[+−-]?\d", context.value.text[candidate.end :]
    ):
        reason = "UNSUPPORTED_VALUE_SUFFIX"
        flags.add("UNIT_CONTEXT_UNRESOLVED")
    elif candidate.members and candidate.dimension is not None:
        reason = "COMPOSITE_COMPONENT_UNITS"
    elif len(choices) == 1 and not any(
        c.flags for c in context.column_headers + context.row_labels
    ):
        unit, dimension = next(iter(choices))
        values.update(
            unit=unit,
            dimension=dimension,
            unit_raw=mentions[0].span.raw,
            unit_start=None,
            unit_end=None,
        )
        source = "INHERITED"
        proof = mentions
        flags.add("UNIT_INHERITED")
    else:
        reason = (
            "AMBIGUOUS_ASSOCIATED_UNITS"
            if len(choices) > 1
            else "UNCERTAIN_CONTEXT" if choices else "NO_UNIT_IN_ASSOCIATED_CELLS"
        )
        flags.add("UNIT_CONTEXT_AMBIGUOUS" if len(choices) > 1 else "UNIT_CONTEXT_UNRESOLVED")
        proof = mentions
    values.update(flags=tuple(sorted(flags)), producer=VERSION)
    return identify(
        ContextualCandidate(
            **values,
            unit_source=source,
            unit_evidence=proof,
            context_evidence=evidence,
            unit_reason=reason,
        )
    )


def harvest_cell(
    text,
    *,
    context,
    source_id,
    node_id,
    classification,
    uncertainty=(),
    decimal_separator=None,
    token_locale=False,
    locale_prior=None,
    **kwargs,
):
    from .quantitative import harvest

    if not isinstance(context, CellContext) or (source_id, node_id, text) != (
        context.value.source_id,
        context.value.cell_id,
        context.value.text,
    ):
        raise ValueError("context does not describe the harvested source cell")
    if classification == "FURNITURE":
        return ()
    if classification not in {"CONTENT", "UNKNOWN"}:
        raise ValueError("explicit classification required")
    from .relative_quantitative import harvest_relative

    relative = harvest_relative(
        context,
        classification=classification,
        uncertainty=uncertainty,
        token_locale=token_locale,
        decimal_separator=decimal_separator,
        locale_prior=locale_prior,
    )
    if relative is not None:
        return (relative,)
    base = list(
        harvest(
            text,
            source_id=source_id,
            node_id=node_id,
            classification=classification,
            uncertainty=tuple(uncertainty) + context.value.flags,
            decimal_separator=decimal_separator,
            token_locale=token_locale,
            locale_prior=locale_prior,
            table_cell=True,
        )
    )
    flags = (
        tuple(uncertainty)
        + context.value.flags
        + (("CLASSIFICATION_UNKNOWN",) if classification == "UNKNOWN" else ())
    )
    for number in physical_numbers(text, table_cell=True):
        if any(number.start < c.end and number.end > c.start for c in base):
            continue
        value = (
            resolve_number(number.raw, locale_prior).value
            if token_locale
            else parse_decimal(number.raw, decimal_separator=decimal_separator)
        )
        nf = (
            resolve_number(number.raw, locale_prior).flags
            if token_locale
            else notation_flags(number.raw, decimal_separator)
        )
        base.append(
            _make(
                source_id,
                node_id,
                text,
                number.comparator_start if number.comparator_start is not None else number.start,
                number.end,
                "inequality" if number.comparator else "scalar",
                flags
                + nf
                + (
                    ("DIRECTION_UNRESOLVED",)
                    if number.comparator and "jusqu" in number.comparator.casefold()
                    else ()
                ),
                comparator_raw=number.comparator,
                comparator_normalized=(
                    None
                    if number.comparator and "jusqu" in number.comparator.casefold()
                    else NORMAL_OP.get(number.comparator, number.comparator)
                ),
                direction_status=(
                    "unresolved"
                    if number.comparator and "jusqu" in number.comparator.casefold()
                    else "resolved" if number.comparator else "not_applicable"
                ),
                number=format(value, "f") if value is not None else None,
                normalization_status="parsed" if value is not None else "unparsed",
            )
        )
    originals = {c.candidate_id: c for c in base}
    changed = {}

    def annotate(candidate):
        if candidate.candidate_id in changed:
            return changed[candidate.candidate_id]
        children = [annotate(originals[m.candidate_id]) for m in candidate.members]
        c = bind_unit(candidate, context) if candidate.quantitative else candidate
        if children:
            proof = list(c.unit_evidence) if isinstance(c, ContextualCandidate) else []
            for child in children:
                for mention in getattr(child, "unit_evidence", ()):
                    if mention not in proof:
                        proof.append(mention)
            c = replace(
                c,
                members=tuple(
                    Member(child.candidate_id, member.role)
                    for child, member in zip(children, candidate.members)
                ),
            )
            if isinstance(c, ContextualCandidate):
                c = replace(c, unit_evidence=tuple(proof))
            c = identify(c)
        changed[candidate.candidate_id] = c
        return c

    result = [annotate(c) for c in base]
    # Unitless range/tolerance cells gain the same explicit composite shape as
    # inline-unit expressions, without appending a fictitious unit to their text.
    leaves = sorted((c for c in result if c.quantitative and not c.members), key=lambda c: c.start)
    occupied = [(c.start, c.end) for c in result if c.members]
    for left, right in zip(leaves, leaves[1:]):
        if any(left.start < b and right.end > a for a, b in occupied):
            continue
        gap = text[left.end : right.start].strip()
        if gap not in {"à", "au", "to", "..", "…", "±", "+/-"}:
            continue
        tolerance = gap in {"±", "+/-"}
        valid = (
            left.number is not None
            and right.number is not None
            and left.unit is not None
            and left.unit == right.unit
        )
        if valid:
            a, b = parse_decimal(left.number, decimal_separator="."), parse_decimal(
                right.number, decimal_separator="."
            )
            valid = b >= 0 if tolerance else a <= b
        roles = ("nominal", "tolerance") if tolerance else ("lower", "upper")
        child = _make(
            source_id,
            node_id,
            text,
            left.start,
            right.end,
            "tolerance" if tolerance else "interval",
            tuple(
                sorted(set(left.flags + right.flags + (() if valid else ("COMPOSITE_UNRESOLVED",))))
            ),
            members=(Member(left.candidate_id, roles[0]), Member(right.candidate_id, roles[1])),
            normalization_status="parsed" if valid else "unparsed",
            direction_status="resolved" if valid else "unresolved",
            comparator_raw=gap,
            comparator_normalized="tolerance" if tolerance else "bounded",
            **dict(zip(roles, (left.number, right.number))),
        )
        result.append(bind_unit(child, context))
    return tuple(sorted(result, key=lambda c: (c.start, c.end, c.kind, c.candidate_id)))
