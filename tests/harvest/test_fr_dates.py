"""fr.dates — the French date grammar on synthetic text.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

No corpus text: the fragment breaks a PDF text layer makes are imitated, not copied.
"""

from __future__ import annotations

from ragix_kernels.harvest.fr.dates import periods, read


def one(text):
    rs = read(text)
    assert len(rs) == 1, rs
    r = rs[0]
    assert text[r.start:r.end] == r.raw
    return r


def test_split_weekday_day_year_and_clock():
    r = one("Date limite :\nMerc\nredi\n2\n4\njuin\n202\n7\nà 1\n0 heures 3\n0\nSuite")
    assert r.raw.startswith("Merc") and r.raw.endswith("3\n0")
    assert (r.type, r.normalized) == ("datetime", "2027-06-24T10:30")


def test_clock_forms():
    assert one("le 3 mars 2025 à 9h30").normalized == "2025-03-03T09:30"
    assert one("le 3 mars 2025 à 9h ").normalized == "2025-03-03T09:00"
    assert one("le 3 mars 2025 à 14 heures 05\n41000").normalized == "2025-03-03T14:05"


def test_month_split_letter_by_letter_and_first_of_month():
    assert one("au 7\no\nc\ntobre 2026.").normalized == "2026-10-07"
    assert one("du 1er avril 2029,").normalized == "2029-04-01"
    assert one("le 1\n9 d\né\ncembre 2026").normalized == "2026-12-19"


def test_year_absent_is_incomplete_not_completed():
    r = one("jusqu'au 30 avril\nSommaire")
    assert (r.type, r.normalized, r.reason, r.year) == ("date", None, "year absent", None)


def test_year_as_written_is_kept():
    assert one("Période du 1er janvier 202\n5\n").normalized == "2025-01-01"


def test_not_a_calendar_date_or_clock_time():
    assert one("le 30 février 2026").reason == "not a calendar date"
    assert one("le 3 mars 2026 à 25h00").reason == "not a clock time"


def test_numeric_family():
    assert one("remise le 05/10/2026 à 16h00").normalized == "2026-10-05T16:00"


def test_not_dates():
    for text in ("(GMT+01:00)", "Visite des 1 000 heures", "Page\n3\nsur\n7", "du lundi au vendredi",
                 "ouvert\n–\n203\n1XY\n-\n4", "Tél : 01 23 45 67 89", "4\nMaintenance préventive",
                 "au plus tard 10 jours avant la date limite"):
        assert read(text) == [], text


def test_periods():
    t = "pourront se tenir du\n2\n0\nma\ni\nau 1\n2\njuin\n202\n7\npuis"
    rs = read(t)
    assert [r.normalized for r in rs] == [None, "2027-06-12"]
    assert periods(t, rs) == [(rs[0], rs[1])]
    t = "Période du 1er janvier 2028 au 31 décembre 2029\n"
    rs = read(t)
    assert [r.normalized for r in rs] == ["2028-01-01", "2029-12-31"] and len(periods(t, rs)) == 1
    t = "Reconductible jusqu'au 31 décembre 2030"
    assert periods(t, read(t)) == []
