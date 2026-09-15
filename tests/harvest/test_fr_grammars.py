"""fr.grammars — the value grammars beside the dates, on synthetic text, version by version.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Each version of the grammar arrived with a case it had to read and a case it had to leave alone;
both halves are pinned here, oldest first, so that no later version can quietly undo an earlier
one. Every sentence is invented for the purpose.

  1.2  hours written with the abbreviation (« 24h », « 50 h ») are durations; a clock time is not;
  1.3  « 3j », « 30 min », « ½ journée », « 2 fois »;
  1.4  a decimal separator admits no whitespace around it: « 24, 7 » is two numbers;
  1.5  a decimal split by a line break is declined, never read as its tail.
"""

from __future__ import annotations

import pytest

from ragix_kernels.harvest.fr.grammars import VERSION, read_values


def kinds(text, **kw):
    return [(v.kind, v.normalized) for v in read_values(text, **kw)]


def normalised(text: str) -> list[str]:
    return [v.normalized for v in read_values(text) if v.normalized]


def _read(text, kind):
    return [(v.raw, v.normalized) for v in read_values(text) if v.kind == kind]


def _durations(text: str) -> list[tuple[str, str]]:
    return [(v.raw, v.normalized) for v in read_values(text) if v.kind == "duration"]


# ----------------------------------------------------------------------- the kinds

def test_amounts_keep_their_cents_and_survive_a_split():
    assert kinds("le montant est 25\n0,00 € HT") == [("amount", "250.00 EUR")]
    assert kinds("pénalité de 1 000,00 € par jour")[0] == ("amount", "1000.00 EUR")
    assert kinds("40 EUR") == [("amount", "40.00 EUR")]


def test_percentages_durations_quantities_and_references():
    assert kinds("une remise de 35 %") == [("percentage", "35 %")]
    assert kinds("au plus tard 10 jours avant") == [("duration", "P10D")]
    assert kinds("2 ans ferme") == [("duration", "P2Y")]
    assert kinds("tous les six mois") == [("duration", "P6M")]      # a number written in words
    assert kinds("85 lots") == [("quantity", "85 lots")]
    assert kinds("article 4.1 du CCAG") == [("reference", "CCAG 4.1")]
    assert kinds("R.2132-11 du code") == [("reference", "R.2132-11")]
    assert kinds("RC 8.1 précise") == [("reference", "RC 8.1")]
    assert kinds("l’annexe 1 à l’Acte d’Engagement") == [("reference", "AE annexe 1")]


def test_a_datetime_absorbs_its_clock_time_rather_than_leaving_a_duration():
    assert kinds("offres : Jeu\ndi 14 octobre 203\n1 à 10 heures 00") == [("datetime", "2031-10-14T10:00")]


def test_a_period_is_one_value_and_the_start_may_borrow_the_end_s_year():
    """One period, not two dates; and the one exception to completion: the borrowing is recorded."""
    values = read_values("du 3 mars au 17 avril 2031")
    assert [(v.kind, v.normalized) for v in values] == [("period", "2031-03-03/2031-04-17")]
    assert values[0].reason == "the start's year is borrowed from the end"
    assert kinds("jusqu'au 31 décembre") == [("date", None)]          # no period, no year: still incomplete


def test_what_is_not_a_value():
    for text in ("Tél : 01 23 45 67 89", "2031XY-4"):
        assert read_values(text) == [], text
    # a page marker IS a reference under the convention, and is labelled as one
    assert kinds("Page 3 sur 7") == [("reference", "3/7")]


def test_the_kinds_asked_for_are_the_kinds_returned():
    text = "10 jours avant le 14 octobre 2031, pour 250,00 €"
    assert {k for k, _ in kinds(text, kinds=("amount",))} == {"amount"}
    assert {k for k, _ in kinds(text, kinds=("duration", "date"))} == {"duration", "date"}


def test_spans_are_exact():
    text = "le montant est 25\n0,00 € HT et la remise 35 %"
    for value in read_values(text):
        assert text[value.start:value.end] == value.raw


# ------------------------------------------------------------------ 1.2, the hours

@pytest.mark.parametrize("text, raw, iso", [
    ("délai de 24h", "24h", "PT24H"),
    ("sous 24 h", "24 h", "PT24H"),
    ("une visite aux 50 h d'utilisation", "50 h", "PT50H"),
    ("toutes les 1 000 h", "1 000 h", "PT1000H"),
])
def test_the_abbreviation_is_read(text, raw, iso):
    found = _durations(text)
    assert (raw, iso) in found, f"{text!r} gave {found}"


@pytest.mark.parametrize("text", ["la visite de 14h30", "rendez-vous à 14h 30", "à 9h45 au plus tard"])
def test_a_clock_time_is_not_a_duration(text):
    assert _durations(text) == [], f"{text!r} was read as a duration"


def test_the_spelled_form_still_wins_its_span():
    """« 48 heures » must keep reading as one value, not as « 48 h » plus a tail."""
    assert ("48 heures", "PT48H") in _durations("un délai de 48 heures ouvrées")


# ---------------------------------------------------------- 1.3, the source spellings

def test_the_version_is_at_least_1_3():
    assert tuple(int(p) for p in VERSION.split(".")) >= (1, 3)


@pytest.mark.parametrize("text, raw, iso", [
    ("un délai de 3j", "3j", "P3D"), ("sous 3 j ouvrés", "3 j", "P3D"),
    ("une intervention de 30 min", "30 min", "PT30M"), ("en 30 minutes", "30 minutes", "PT30M"),
    ("une ½ journée de formation", "½ journée", "P0.5D"), ("deux sessions de 2 journées", "2 journées", "P2D"),
])
def test_the_new_durations(text, raw, iso):
    assert (raw, iso) in _read(text, "duration"), f"{text!r} gave {_read(text, 'duration')}"


def test_fois_is_a_quantity_in_the_convention_s_form():
    assert ("2 fois", "2 fois") in _read("contrôlé 2 fois par an", "quantity")


@pytest.mark.parametrize("text", ["le 3 juillet", "un minimum de garantie", "la visite de 14h30"])
def test_what_must_not_become_a_duration(text):
    assert _read(text, "duration") == [], f"{text!r} was read as {_read(text, 'duration')}"


def test_a_date_that_starts_like_a_day_count_stays_a_date():
    assert [v.kind for v in read_values("le 3 juillet")] == ["date"]


def test_what_1_2_read_still_reads():
    assert ("1 000 heures", "PT1000H") in _read("toutes les 1 000 heures", "duration")
    assert ("48h", "PT48H") in _read("sous 48h", "duration")


# ------------------------------------------------------- 1.4, no space in a decimal

def test_the_version_is_at_least_1_4():
    assert tuple(int(x) for x in VERSION.split(".")) >= (1, 4)


def test_an_enumeration_is_not_a_decimal():
    """1.3 read « 24, 7 jours » as 24,7 days: the source then held P24.7D and never P7D, and a
    sentence quoting it verbatim was refused against its own text."""
    read = normalised("un service de garde joignable 24 heures sur 24, 7 jours sur 7")
    assert "PT24H" in read and "P7D" in read
    assert "P24.7D" not in read


@pytest.mark.parametrize("text, iso", [
    ("un délai de 24,7 jours", "P24.7D"),          # a real French decimal still reads
    ("une tolérance de 0,5 heure", "PT0.5H"),
])
def test_a_decimal_written_without_a_space_still_reads(text, iso):
    assert iso in normalised(text)


@pytest.mark.parametrize("text, iso", [
    ("un montant de 250,00 €", "250.00 EUR"),
    ("un forfait de 2 500,50 euros", "2500.50 EUR"),
    ("une visite de 1 000 heures", "PT1000H"),      # the thousands group keeps its space
])
def test_the_shapes_that_must_not_move(text, iso):
    assert iso in normalised(text)


def test_a_comma_followed_by_a_space_separates_two_quantities():
    read = normalised("des pénalités de 150 €, 300 € et 450 €")
    assert read.count("150.00 EUR") == 1 and "300.00 EUR" in read and "450.00 EUR" in read
    assert not any(v.startswith("150.3") or v.startswith("300.4") for v in read)


# ----------------------------------------------- 1.5, the tail of a split decimal

def test_the_version_is_1_6():
    assert VERSION == "1.6"


@pytest.mark.parametrize("text, phantom", [
    ("un montant de 250,\n00 €", "0.00 EUR"),
    ("une retenue de 1 500,\n50 euros", "50.00 EUR"),
    ("sous 48,\n5 heures", "PT5H"),
    ("une remise de 12,\n5 %", "12.5 %"),
])
def test_the_tail_of_a_split_decimal_is_not_read_as_a_value(text, phantom):
    assert phantom not in normalised(text), "the tail of a split decimal was read as a value"


def test_the_limit_of_this_rule_measured_and_left_standing():
    """The line break is the signature. The same shape with a space is an enumeration, which 1.4
    must keep reading; so a decimal split by a SPACE is still read as its tail. Recorded, not hidden."""
    assert "P7D" in normalised("une permanence 24 heures sur 24, 7 jours sur 7")   # 1.4 stands
    assert "0.00 EUR" in normalised("un montant de 250, 00 €")                     # the limit, named


def test_the_decline_is_carried_in_the_reading_with_its_reason():
    """Declined, not dropped: a refusal that leaves no trace cannot be counted."""
    declined = [v for v in read_values("un montant de 250,\n00 €") if v.normalized is None]
    assert declined and any("decimal" in (v.reason or "") for v in declined)


@pytest.mark.parametrize("text, expected", [
    ("des pénalités de 150 €, 300 € et 450 €", ["150.00 EUR", "300.00 EUR", "450.00 EUR"]),
    ("un montant de 250,00 €", ["250.00 EUR"]),
    ("un délai de 24,7 jours", ["P24.7D"]),
    ("une visite de 1 000 heures", ["PT1000H"]),
])
def test_what_must_not_move(text, expected):
    for value in expected:
        assert value in normalised(text)
