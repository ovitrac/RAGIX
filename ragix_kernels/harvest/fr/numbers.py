"""Exact decimal literals; ambiguous separators are declined, never guessed.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15
"""

from decimal import Decimal, InvalidOperation
import re


def parse_decimal(raw: str, *, decimal_separator: str | None = None,
                  allow_split_digits: bool = False) -> Decimal | None:
    """Read signed decimal text, with an optional explicitly declared separator.

    With no declaration a comma is decimal; a lone dot with three following digits
    and a nonzero 1-3 digit head is ambiguous. Spaces may group thousands. The
    legacy reader can explicitly permit split PDF digits; callers own join review.
    """
    if decimal_separator not in (None, ".", ","):
        raise ValueError("decimal_separator must be '.', ',' or None")
    text = raw.strip().replace("\u2212", "-")
    if re.search(r"\s", text):
        if not allow_split_digits and not re.fullmatch(
                r"[+-]?\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?", text):
            return None
        text = re.sub(r"\s", "", text)
    if decimal_separator == ".":
        if "," in text:
            if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
                return None
            text = text.replace(",", "")
    elif decimal_separator == "," or "," in text:
        if "." in text:
            if not re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+(?:,\d+)?", text):
                return None
            text = text.replace(".", "")
        text = text.replace(",", ".")
    elif re.fullmatch(r"[+-]?[1-9]\d{0,2}\.\d{3}", text):
        return None
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def format_decimal(value, *, minimum_places: int = 0) -> str:
    """Canonical decimal digits without precision loss or binary-float arithmetic."""
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    if not number.is_finite() or minimum_places < 0:
        raise ValueError("finite decimal and nonnegative minimum_places required")
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if minimum_places:
        head, _, tail = text.partition(".")
        text = head + "." + tail.ljust(minimum_places, "0")
    return text
