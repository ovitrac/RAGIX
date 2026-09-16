"""Token-first numeric notation and evidence-scoped document priors.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import dataclass
from decimal import Decimal
import re
from .fr.numbers import parse_decimal

NUMBER = r"[+−-]?\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)*"
TOKEN = re.compile(rf"(?<![\w.,])(?P<number>{NUMBER})(?![\d.,])")
IDENTIFIER = re.compile(r"(?<!\w)[A-Za-z0-9]+(?:[-/]+[A-Za-z0-9]+)+(?!\w)")
DOTTED = re.compile(r"(?<![\w.])\d+(?:\.\s*\d+)+(?![\w.])")


@dataclass(frozen=True)
class NumberReading:
    value: Decimal | None
    separator: str | None
    basis: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhysicalNumber:
    start: int
    end: int
    raw: str
    basis: str
    comparator: str | None = None
    comparator_start: int | None = None


def resolve_number(raw, prior=None):
    """A prior never overrides an unambiguous local interpretation.

    A three-digit single-separator tail remains ambiguous without independent
    prior evidence. Prior-dependent interpretations always remain reviewable.
    """
    text = raw.strip().replace("−", "-")
    prior_separator = (prior or {}).get("decimal_separator")
    if prior_separator not in {None, ".", ","}:
        raise ValueError("invalid numeric prior")
    separator = None
    flags = []
    basis = "integer"
    local = True
    punctuation = [c for c in text if c in ".,"]
    spaced = bool(re.search(r"\d[ \u00a0\u202f]\d", text))
    if spaced:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?", text):
            return NumberReading(None, None, "unresolved", ("SEPARATOR_AMBIGUOUS",))
        separator = punctuation[-1] if punctuation else None
        basis = "grouping_pattern"
        flags.append("GROUPING_ASSUMED")
    elif "." in text and "," in text:
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+\.\d+", text):
            separator = "."
        elif re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+,\d+", text):
            separator = ","
        else:
            return NumberReading(None, None, "unresolved", ("SEPARATOR_AMBIGUOUS",))
        basis = "grouping_pattern"
        flags.append("GROUPING_ASSUMED")
    elif len(punctuation) > 1:
        mark = punctuation[0]
        if not re.fullmatch(r"[+-]?\d{1,3}(?:" + re.escape(mark) + r"\d{3})+", text):
            return NumberReading(None, None, "unresolved", ("SEPARATOR_AMBIGUOUS",))
        separator = "," if mark == "." else "."
        basis = "grouping_pattern"
        flags.append("GROUPING_ASSUMED")
    elif punctuation:
        mark = punctuation[0]
        head, tail = text.lstrip("+-").split(mark)
        if len(tail) != 3 or head == "0" or len(head) > 3:
            separator = mark
            basis = "token_decimal"
        else:
            local = False
            separator = prior_separator
            basis = "document_prior"
            flags.extend(("SEPARATOR_AMBIGUOUS", "LOCALE_PRIOR_USED"))
            if separator is None:
                return NumberReading(None, None, "unresolved", ("SEPARATOR_AMBIGUOUS",))
            if prior.get("prior_strength") == "weak":
                flags.append("LOCALE_PRIOR_WEAK")
            if separator != mark:
                flags.append("GROUPING_ASSUMED")
    if local and prior_separator and separator and separator != prior_separator:
        flags.append("SEPARATOR_AMBIGUOUS")
    value = parse_decimal(text, decimal_separator=separator or ".")
    if value is None:
        flags.append("SEPARATOR_AMBIGUOUS")
    return NumberReading(value, separator, basis, tuple(sorted(set(flags))))


def _cue(text, start, end):
    from .quantitative import UNIT, UNIT_END, PREFIX

    unit = re.match(rf"\s*{UNIT}{UNIT_END}", text[end:])
    prefix = PREFIX.search(text[:start])
    if unit:
        return "unit", prefix
    if prefix:
        return "comparator", prefix
    return None, None


def nonphysical_spans(text, *, table_cell=False):
    """Explicit identities/dates/revisions/pages and structural numbering own spans.

    An unmarked dotted token with attached physical context is not claimed as a
    section. A non-identifier table cell supplies context for otherwise bare
    numbers, but never overrides explicit section/date/revision evidence.
    """
    spans = []
    for match in IDENTIFIER.finditer(text):
        if re.search(r"[A-Za-z]", match[0]) and re.search(r"\d", match[0]):
            spans.append((match.start(), match.end(), "identifier"))
    patterns = {
        "date": r"(?<!\w)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?!\w)",
        "version": r"\b(?:version|rev(?:ision)?|v)\s*[:.]?\s*\d+(?:\.\d+)*",
        "page": r"\b(?:page|p\.)\s*\d+(?:\s*(?:/|of|sur)\s*\d+)?",
        "numbering": r"(?:§|\bsection\b|\bchapter\b|\bchapitre\b)\s*\d+(?:\.\s*\d+)*",
    }
    for kind, pattern in patterns.items():
        spans.extend((m.start(), m.end(), kind) for m in re.finditer(pattern, text, re.I))
    for match in DOTTED.finditer(text):
        cue, _ = _cue(text, match.start(), match.end())
        heading = (
            bool(re.match(r"\s*[^\W\d_]", text[match.end() :]))
            and not text[: match.start()].strip()
        )
        if not cue and (not table_cell or heading):
            spans.append((match.start(), match.end(), "numbering"))
    return tuple(sorted(set(spans)))


def physical_numbers(text, *, table_cell=False, identifier_column=False):
    if identifier_column:
        return ()
    exclusions = nonphysical_spans(text, table_cell=table_cell)
    result = []
    for match in TOKEN.finditer(text):
        if any(match.start() < b and match.end() > a for a, b, _ in exclusions):
            continue
        cue, prefix = _cue(text, match.start(), match.end())
        if not cue and (
            not table_cell or (match.end() < len(text) and text[match.end()].isalnum())
        ):
            continue
        result.append(
            PhysicalNumber(
                match.start(),
                match.end(),
                match[0],
                cue or "table_cell",
                prefix["op"] if prefix else None,
                prefix.start() if prefix else None,
            )
        )
    return tuple(result)
