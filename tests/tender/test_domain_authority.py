"""tender.authority — the two gates, the ranking, the resolution and its controls, on synthetic text.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

No corpus text: the clauses and the fragment breaks are imitated, not copied.
"""

from __future__ import annotations

import pytest

# the slice reads its dates with the harvest family's French grammar, ported on its own branch
pytest.importorskip("ragix_kernels.harvest.fr.dates",
                    reason="ragix_kernels.harvest.fr.dates is not on this branch")

from ragix_kernels.tender.domain.authority import CCAP2, RC6, conclude, dependents, discover, find_authorities, same_field_conflict
from ragix_kernels.tender.domain.deadline_slice import Chunk, extract


def chunk(doc, cid, lines):
    return Chunk(doc, cid, "\n".join(lines), tuple(f"0.{i}" for i in range(len(lines))), tuple(lines))


COVER_RC = chunk("docRC", "cRC", ["Règlement de la Consultation", "Appel d'offres ouvert",
                                  "Date et heure limites de réception des offres", ":",
                                  "Jeudi 2 octobre 2027 à 12 heures 00"])
COVER_CCAP = chunk("docCCAP", "cCCAP", ["Cahier des Clauses Administratives Particulières",
                                        "Date et heure limites de réception des offres :",
                                        "Vendredi 8 octobre 2027 à 12 heures 00"])
# the ranking clause, with one piece name split mid-word as the extractor finds them
RANKING = chunk("docCCAP", "cRANK", ["2", "-", "Pièces contractuelles",
                                     "Par dérogation à l'article 4.1 du CCAG, les pièces contractuelles",
                                     "prévalent dans cet ordre de priorité :", "-",
                                     "Le Règlement de la Consultation (RC) et son annexe", "-",
                                     "L'Acte d'Engagement (AE) et ses annexes", "-",
                                     "Le Cahier des Clauses Administratives P", "articulières (CCAP)", "-",
                                     "Le Cahier des Clauses Techniques Particulières (CCTP)"])
RC6_CLAUSE = chunk("docRC", "cRC6", ["6", "-", "Conditions d'envoi ou de", "remise des plis",
                                     "Les plis devront parvenir à destination avant la date et l'heure limites",
                                     "de réception des offres indiquées sur la page de",
                                     "garde du présent document."])
DEPENDENTS = chunk("docRC", "cDEP", ["Les visites sur site pourront s’effectuer du", "1", "septembre",
                                     "au 2", "8", "septembre", "202", "7", "du lundi au vendredi.",
                                     "Cette demande doit intervenir au plus tard 10 jours avant la date "
                                     "limite de remise des plis."])
COVERS = [COVER_RC, COVER_CCAP]
CHUNKS = [COVER_RC, COVER_CCAP, RANKING, RC6_CLAUSE, DEPENDENTS]


def claims_of(chunks=None):
    _, claims, _ = extract(chunks or [COVER_RC, COVER_CCAP, DEPENDENTS], project="P")
    return claims


def test_the_ranking_is_read_with_its_spans_even_when_a_name_is_split():
    (ccap2,) = [a for a in find_authorities(CHUNKS) if a.label == CCAP2]
    assert ccap2.ranks == {"RC": 1, "AE": 2, "CCAP": 3, "CCTP": 4}
    assert set(ccap2.rank_spans) == {"RC", "AE", "CCAP", "CCTP"}
    assert len(ccap2.rank_spans["RC"]["span_sha256"]) == 64


def test_both_authorities_are_surfaced_and_the_coverage_is_sufficient():
    d = discover(claims_of(), COVERS, CHUNKS)
    assert sorted(a.label for a in d.authorities) == [CCAP2, RC6]
    assert d.missing == () and d.disagreements == () and d.unranked == ()
    assert d.ranked == {"docRC": 1, "docCCAP": 3} and d.sufficient


def test_the_conflict_is_flagged_with_both_values():
    c = same_field_conflict(claims_of())
    assert c.flagged and c.values == ("2027-10-02T12:00", "2027-10-08T12:00")
    assert all(len(ids) == 1 for ids in c.claims_by_value.values())


def test_the_rank_one_value_governs_and_the_other_is_displaced():
    claims = claims_of()
    r = conclude(claims, discover(claims, COVERS, CHUNKS), same_field_conflict(claims))
    assert (r.verdict, r.rendering, r.value) == ("supported_with_caveats", "PRÊT AVEC RÉSERVES",
                                                 "2027-10-02T12:00")
    assert len(r.governing) == 1 and r.displaced == {"2027-10-08T12:00": [3]}
    assert len(r.evidence) >= 4 and all(len(e["span_sha256"]) == 64 for e in r.evidence)


def test_a_withheld_authority_blocks_the_conclusion():
    claims = claims_of()
    for withheld, dropped in ((CCAP2, "cRANK"), (RC6, "cRC6")):
        kept = [c for c in CHUNKS if c.chunk_id != dropped]
        r = conclude(claims, discover(claims, COVERS, kept), same_field_conflict(claims))
        assert r.verdict == "abstain_conflict" and r.value is None
        assert withheld in r.reason


def test_authorities_that_disagree_block_the_conclusion():
    bad = chunk("docCCAP", "cBAD", ["Pièces contractuelles", "prévalent dans cet ordre de priorité :",
                                    "Le Cahier des Clauses Techniques Particulières (CCTP)",
                                    "Le Règlement de la Consultation (RC)"])
    claims = claims_of()
    d = discover(claims, COVERS, [COVER_RC, COVER_CCAP, bad, RC6_CLAUSE])
    assert d.disagreements and not d.sufficient
    r = conclude(claims, d, same_field_conflict(claims))
    assert r.verdict == "abstain_conflict" and r.value is None


def test_a_piece_the_ranking_does_not_name_is_unranked_and_abstains():
    other = chunk("docBPU", "cBPU", ["Bordereau des Prix Unitaires",
                                     "Date et heure limites de réception des offres :",
                                     "Lundi 4 octobre 2027 à 12 heures 00"])
    claims = claims_of([COVER_RC, COVER_CCAP, other, DEPENDENTS])
    d = discover(claims, COVERS + [other], CHUNKS + [other])
    assert d.unranked == ("docBPU",) and not d.sufficient
    assert conclude(claims, d, same_field_conflict(claims)).value is None


def test_without_a_conflict_the_verdict_asserts_plainly():
    claims = claims_of([COVER_RC, DEPENDENTS])
    d = discover(claims, [COVER_RC], CHUNKS)
    r = conclude(claims, d, same_field_conflict(claims))
    assert (r.verdict, r.value) == ("supported", "2027-10-02T12:00")


def test_the_dependent_dates_are_re_derived_from_the_resolved_value():
    claims = claims_of()
    r = conclude(claims, discover(claims, COVERS, CHUNKS), same_field_conflict(claims))
    deps = dependents(claims, r)
    assert [e["value"] for e in deps["questions_deadline"]] == ["2027-09-22T12:00"]
    assert deps["displaced"] == ["2027-09-28T12:00"]
    assert deps["visit_window"]["days_end_after_questions"] == 6
    assert "can no longer be followed by a question" in deps["visit_window"]["note"]


def test_nothing_is_re_derived_without_a_resolved_value():
    claims = claims_of()
    blocked = conclude(claims, discover(claims, COVERS, [COVER_RC, COVER_CCAP]), same_field_conflict(claims))
    assert blocked.value is None
    assert dependents(claims, blocked)["blocked"].startswith("no resolved value")
