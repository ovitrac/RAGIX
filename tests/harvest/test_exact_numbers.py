"""Synthetic decimal/sign/precision regressions; no source-corpus fixtures.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15
"""

from decimal import Decimal
import pytest

from ragix_kernels.harvest.fr.numbers import parse_decimal, format_decimal
from ragix_kernels.harvest.fr.grammars import read_values, VERSION
from ragix_kernels.harvest.form import substitute, HarvestRefusal


@pytest.mark.parametrize("raw, expected", [
    ("0.2", "0.2"), ("0,2", "0.2"), ("-7", "-7"), ("+7", "7"),
    ("\u22127.25", "-7.25"), ("0,0007", "0.0007"), ("12 345,67", "12345.67"),
    ("2.345,67", "2345.67"), ("12.25", "12.25"), ("0.007", "0.007"),
])
def test_exact_decimal(raw, expected):
    assert parse_decimal(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", ["1.234", "1.2.3", "1,2,3", "NaN", "inf", "25\n0", "12, 7"])
def test_ambiguous_or_malformed_refused(raw):
    assert parse_decimal(raw) is None


def test_explicit_locale_and_precision():
    assert parse_decimal("1.234", decimal_separator=".") == Decimal("1.234")
    assert parse_decimal("1.234", decimal_separator=",") == Decimal("1234")
    assert format_decimal(Decimal("0.00007")) == "0.00007"
    assert format_decimal(Decimal("5"), minimum_places=2) == "5.00"
    assert format_decimal(Decimal("5.123"), minimum_places=2) == "5.123"


@pytest.mark.parametrize("raw, expected", [
    ("0.2 %", "0.2 %"), ("-7 %", "-7 %"), ("0,0007 %", "0.0007 %"),
    ("0,0007 EUR", "0.0007 EUR"),
])
def test_legacy_reader_never_changes_magnitude(raw, expected):
    value, = read_values(raw)
    assert value.raw == raw and value.start == 0 and value.end == len(raw)
    assert value.normalized == expected
    assert VERSION == "1.6"


@pytest.mark.parametrize("offered, requested", [("0.2 %", "2 %"), ("2 %", "0.2 %"),
                                               ("-7 %", "7 %"), ("0 %", "0,0007 %")])
def test_unequal_values_cannot_share_an_offered_claim(offered, requested):
    with pytest.raises(HarvestRefusal):
        substitute([requested], [{"value_id": "synthetic-value", "raw": offered}])
