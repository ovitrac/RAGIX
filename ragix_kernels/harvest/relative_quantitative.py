"""Uninterpreted anchored offset shapes with exact source spans.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import dataclass, replace, fields
import re
import unicodedata
from .quantitative import _make, UNITS, notation_flags
from .numeric_locale import resolve_number, physical_numbers
from .fr.numbers import parse_decimal
from .table_context import CellSpan
from .contextual_quantitative import (
    ContextualCandidate,
    UnitMention,
    CONTEXT_UNIT,
    bind_unit,
    identify,
    unit_mentions,
)

VERSION = "relative-quantity/1"
WORD = r"[^\W\d_]\w*"
ANCHOR = rf"{WORD}(?:(?:[ \t]+|[’\x27-]){WORD})*?"
MAGNITUDE = r"\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?"
SIGN = r"(?:\+/-|[+−±-])"


def operand(suffix, *, anchored):
    anchor = rf"(?P<anchor{suffix}>{ANCHOR})[ \t]+" if anchored else ""
    return (
        anchor
        + rf"(?P<offset{suffix}>(?P<sign{suffix}>{SIGN})[ \t]*(?P<number{suffix}>{MAGNITUDE}))"
        + rf"(?:[ \t]*(?P<unit{suffix}>{CONTEXT_UNIT})(?![\w€$£]))?"
    )


def expression(anchored):
    return re.compile(
        operand("1", anchored=anchored)
        + rf"(?:[ \t]+(?P<connector>au|à|to)[ \t]+"
        + operand("2", anchored=anchored)
        + r")?",
        re.I,
    )


INLINE = expression(True)
CONTEXT = expression(False)
HEADER_ANCHOR = re.compile(rf"(?:relative\s+to|par\s+rapport\s+à)\s+(?P<anchor>{ANCHOR})", re.I)


@dataclass(frozen=True)
class RelativeOffset:
    operator: str
    magnitude: str | None
    number: str | None
    span: CellSpan
    anchor: CellSpan | None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelativeQuantityCandidate(ContextualCandidate):
    offsets: tuple[RelativeOffset, ...] = ()
    anchor_candidates: tuple[CellSpan, ...] = ()
    anchor_source: str = "VALUE"
    offset_lower: str | None = None
    offset_upper: str | None = None

    @property
    def quantitative(self):
        return True


def _key(raw):
    return " ".join(unicodedata.normalize("NFC", raw).casefold().split())


def _header_anchors(context):
    mentions = unit_mentions(context)
    found = []
    for cell in context.column_headers + context.row_labels:
        ends = [m.span.start for m in mentions if m.span.cell_id == cell.cell_id]
        text = cell.text[: min(ends)].rstrip(" ([\t") if ends else cell.text.rstrip()
        beginning = len(text) - len(text.lstrip())
        match = HEADER_ANCHOR.fullmatch(text.strip())
        if match:
            found.append(
                cell.span(beginning + match.start("anchor"), beginning + match.end("anchor"))
            )
    return tuple(found)


def harvest_relative(
    context,
    *,
    classification,
    uncertainty=(),
    token_locale=False,
    decimal_separator=None,
    locale_prior=None,
):
    text = context.value.text
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    body = text[start:end]
    if body.startswith("(") and body.endswith(")"):
        start += 1
        end -= 1
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        body = text[start:end]
    match = INLINE.fullmatch(body)
    anchors = ()
    source = "VALUE"
    if not match:
        anchors = _header_anchors(context)
        match = CONTEXT.fullmatch(body) if anchors else None
        source = "CONTEXT"
    if not match:
        prefix = INLINE.match(body)
        if prefix is None and not (anchors and physical_numbers(body, table_cell=True)):
            return None
        if prefix:
            anchors = tuple(
                context.value.span(
                    start + prefix.start("anchor" + suffix), start + prefix.end("anchor" + suffix)
                )
                for suffix in ("1", "2")
                if prefix["anchor" + suffix] is not None
            )
            source = "VALUE"
        base = _make(
            context.value.source_id,
            context.value.cell_id,
            text,
            start,
            end,
            "relative_unparsed",
            tuple(uncertainty)
            + context.value.flags
            + ("RELATIVE_SHAPE_UNPARSED", "RELATIVE_REFERENCE_UNRESOLVED")
            + (("CLASSIFICATION_UNKNOWN",) if classification == "UNKNOWN" else ()),
            normalization_status="unparsed",
            direction_status="unresolved",
        )
        values = {field.name: getattr(base, field.name) for field in fields(type(base))}
        values["producer"] = VERSION
        return identify(
            RelativeQuantityCandidate(
                **values,
                anchor_candidates=anchors,
                anchor_source=source,
                context_evidence=tuple(
                    c.span() for c in context.column_headers + context.row_labels
                ),
                unit_reason="UNSUPPORTED_RELATIVE_SHAPE",
            )
        )
    offsets = []
    inline_units = []
    flags = set(uncertainty) | set(context.value.flags) | {"RELATIVE_REFERENCE_UNRESOLVED"}
    if classification == "UNKNOWN":
        flags.add("CLASSIFICATION_UNKNOWN")
    if source == "CONTEXT" and len(anchors) != 1:
        flags.add("ANCHOR_CONTEXT_AMBIGUOUS")
    value_anchors = []
    for suffix in ("1", "2"):
        raw = match["number" + suffix]
        if raw is None:
            continue
        if token_locale:
            parsed = resolve_number(raw, locale_prior)
            magnitude = parsed.value
            nf = parsed.flags
        else:
            magnitude = parse_decimal(raw, decimal_separator=decimal_separator)
            nf = notation_flags(raw, decimal_separator)
        sign = match["sign" + suffix]
        number = None
        if magnitude is not None and sign not in {"±", "+/-"}:
            number = format(-magnitude if sign in {"-", "−"} else magnitude, "f")
        if sign in {"±", "+/-"}:
            nf += ("RELATIVE_BRANCHING",)
        if magnitude is None:
            nf += ("NUMBER_UNPARSED",)
        flags.update(nf)
        if source == "VALUE":
            anchor = context.value.span(
                start + match.start("anchor" + suffix), start + match.end("anchor" + suffix)
            )
            value_anchors.append(anchor)
        else:
            anchor = anchors[0] if len(anchors) == 1 else None
        offsets.append(
            RelativeOffset(
                sign,
                format(magnitude, "f") if magnitude is not None else None,
                number,
                context.value.span(
                    start + match.start("offset" + suffix), start + match.end("offset" + suffix)
                ),
                anchor,
                tuple(sorted(nf)),
            )
        )
        raw_unit = match["unit" + suffix]
        if raw_unit:
            key = re.sub(r"[ \t]+", "", raw_unit) if raw_unit.startswith("°") else raw_unit
            # Units remain case-sensitive even though the range connector is not.
            if key not in UNITS:
                return None
            unit, dimension = UNITS[key]
            inline_units.append(
                UnitMention(
                    unit,
                    dimension,
                    context.value.span(
                        start + match.start("unit" + suffix), start + match.end("unit" + suffix)
                    ),
                    "value",
                )
            )
    anchors = tuple(value_anchors) if source == "VALUE" else anchors
    same_anchor = len({_key(a.raw) for a in anchors}) == 1 and all(
        o.anchor is not None for o in offsets
    )
    if not same_anchor:
        flags.add("RELATIVE_ANCHOR_MISMATCH")
    choices = {(m.unit, m.dimension) for m in inline_units}
    if len(choices) > 1:
        flags.add("RELATIVE_UNITS_CONFLICT")
    known_unit = inline_units[0] if len(choices) == 1 else None
    if known_unit and len(inline_units) < len(offsets):
        flags.add("UNIT_INHERITED")
    base = _make(
        context.value.source_id,
        context.value.cell_id,
        text,
        start,
        end,
        "relative_interval" if len(offsets) == 2 else "relative_offset",
        tuple(sorted(flags)),
        normalization_status="parsed" if all(o.number is not None for o in offsets) else "unparsed",
        direction_status="unresolved",
        comparator_raw=match["connector"],
        unit=known_unit.unit if known_unit else None,
        dimension=known_unit.dimension if known_unit else None,
        unit_raw=known_unit.span.raw if known_unit else None,
        unit_start=known_unit.span.start if known_unit else None,
        unit_end=known_unit.span.end if known_unit else None,
    )
    bound = bind_unit(base, context)
    if inline_units:
        bound = replace(bound, unit_evidence=tuple(inline_units))
    if len(choices) > 1:
        bound = replace(
            bound,
            unit=None,
            dimension=None,
            unit_raw=None,
            unit_start=None,
            unit_end=None,
            unit_source="NONE",
            unit_reason="RELATIVE_UNITS_CONFLICT",
            normalization_status="unparsed",
        )
    lower = upper = None
    if (
        len(offsets) == 2
        and same_anchor
        and len(choices) <= 1
        and bound.unit is not None
        and "UNIT_CONTEXT_CONFLICT" not in bound.flags
        and all(o.number is not None for o in offsets)
    ):
        values = [parse_decimal(o.number, decimal_separator=".") for o in offsets]
        lower, upper = format(min(values), "f"), format(max(values), "f")
    values = {field.name: getattr(bound, field.name) for field in fields(ContextualCandidate)}
    values["producer"] = VERSION
    return identify(
        RelativeQuantityCandidate(
            **values,
            offsets=tuple(offsets),
            anchor_candidates=anchors,
            anchor_source=source,
            offset_lower=lower,
            offset_upper=upper,
        )
    )
