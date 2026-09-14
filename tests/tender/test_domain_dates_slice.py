"""tender.dates_fr and tender.deadline_slice — the grammar and the slice's claims on synthetic text.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

No corpus text: the fragment breaks are imitated, not copied.
"""

from __future__ import annotations

import pytest

# the slice reads its dates with the harvest family's French grammar, ported on its own branch
pytest.importorskip("ragix_kernels.harvest.fr.dates",
                    reason="ragix_kernels.harvest.fr.dates is not on this branch")

from ragix_kernels.harvest.fr.dates import periods, read
from ragix_kernels.tender.domain.deadline_slice import Chunk, extract


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
                 "ouvert\n–\n202\n7AB\n-\n1", "Tél : 02 54 55 64 78", "4\nMaintenance préventive",
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


def chunk(doc, cid, lines):
    return Chunk(doc, cid, "\n".join(lines), tuple(f"0.{i}" for i in range(len(lines))), tuple(lines))


COVER_A = chunk("docA", "chunkA", ["Date et heure limites de réception des offres", ":", "Jeudi", "1", "5",
                                   "octobre 202", "7", "à 12 heures 00",
                                   "Période du 1er janvier 2028 au 31 décembre 2029"])
COVER_B = chunk("docB", "chunkB", ["Date et heure limites de réception des offres :",
                                   "Vendredi 8 octobre 2027 à 12 heures 00"])
VISIT = chunk("docA", "chunkV", ["Les visites sur site pourront s’effectuer du", "1", "sep", "tembre",
                                 "au 2", "8", "septembre", "202", "7", "du lundi au vendredi."])
QUESTIONS = chunk("docA", "chunkQ", [
    "Cette demande doit intervenir au plus tard 10 jours avant la date limite de remise des plis.",
    "Une réponse sera adressée 6 jours au plus tard avant la date limite de remise des plis."])


def test_the_slice_claims():
    readings, claims, drops = extract([COVER_A, COVER_B, VISIT, QUESTIONS], project="P")
    by = {}
    for c in claims:
        by.setdefault(c.field, []).append(c)
    assert drops == [] and len(readings) == 6
    assert [c.value.normalized for c in by["offer_deadline"]] == ["2027-10-15T12:00", "2027-10-08T12:00"]
    assert {c.provenance.origin for c in by["offer_deadline"]} == {"observed"}
    assert by["offer_deadline"][0].provenance.sources[0].node_ids == ("0.2", "0.3", "0.4", "0.5", "0.6", "0.7")
    assert [c.value.normalized for c in by["questions_deadline"]] == ["2027-10-05T12:00", "2027-09-28T12:00"]
    q = by["questions_deadline"][0]
    assert q.provenance.origin == "derived" and q.provenance.derivation == "offer_deadline − 10 days"
    assert q.value.raw.startswith("au plus tard 10 jours") and len(q.provenance.sources) == 2
    (end,) = by["visit_window_end"]
    (start,) = by["visit_window_start"]
    assert (end.value.normalized, end.provenance.origin) == ("2027-09-28", "observed")
    assert (start.value.normalized, start.provenance.origin) == ("2027-09-01", "derived")
    assert start.value.raw == "1\nsep\ntembre" and len(start.provenance.sources) == 2
    assert all(c.applicability.clause_scope == {"pieces": "consultation", "lots": "all"}
               and c.applicability.authority is None for c in claims)
    assert {r["normalized"] for r in readings} >= {"2028-01-01", "2029-12-31"}


def test_a_cue_with_an_untypable_value_is_a_drop():
    cover = chunk("docC", "chunkC", ["Date et heure limites de réception des offres :",
                                     "Vendredi 8 octobre à 12 heures 00"])
    readings, claims, drops = extract([cover], project="P")
    assert claims == [] and len(drops) == 1 and drops[0]["why"] == "year absent"
