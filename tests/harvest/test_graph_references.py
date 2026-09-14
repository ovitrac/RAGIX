"""The graph's references: an article of the CCTP is never a pointer to a lot.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The reference grammar normalises « article 5 du CCTP » to « CCTP 5 » — the same form as the lot
pointer « CCTP 05 » — and the resolver read both as lot 5, drawing an arrow from a CCTP to another
lot's CCTP for what was an article of its own. « du présent CCTP » is outside the grammar's match
and was left as an article without its piece. Both are references inside the citing CCTP, and
nothing else: read outside a CCTP, an article of the CCTP names no lot and is counted unresolved.

Every sentence below is invented for this test.
"""
from __future__ import annotations

from ragix_kernels.harvest.build_edges import resolve, resolve_reference
from ragix_kernels.harvest.fr.grammars import read_values

DOCS = {"05": "doc-cctp-05", "07": "doc-cctp-07", "CCAP": "doc-ccap", "RC": "doc-rc", "AE": "doc-ae"}


def _one_reference(text: str):
    refs = [v for v in read_values(text) if v.kind == "reference"]
    assert len(refs) == 1, [v.raw for v in refs]
    return refs[0]


def test_an_article_of_the_cctp_read_in_a_cctp_is_that_cctp_not_lot_n():
    text = "Les essais de réception suivent la procédure décrite à l'article 5 du CCTP."
    v = _one_reference(text)
    # the defect, pinned: the grammar's form is the lot pointer's, and the bare resolver takes lot 5
    assert v.normalized == "CCTP 5"
    assert resolve(v.normalized, DOCS) == ("doc-cctp-05", "")
    # the fix: the citing CCTP itself, which the builder counts as `self`
    assert resolve_reference(v, text, "07", DOCS) == ("doc-cctp-07", "")


def test_a_numbered_article_of_the_present_cctp_is_the_citing_cctp():
    text = "Le rapport de visite est remis dans le délai fixé à l'article 5.2 du présent CCTP."
    v = _one_reference(text)
    assert v.normalized == "article 5.2"            # the grammar stops before « du présent CCTP »
    assert resolve(v.normalized, DOCS)[0] is None
    assert resolve_reference(v, text, "07", DOCS) == ("doc-cctp-07", "")


def test_the_same_article_in_the_lot_s_own_cctp_is_still_self():
    text = "Les pièces de rechange sont stockées selon l'article 5 du CCTP."
    v = _one_reference(text)
    assert resolve_reference(v, text, "05", DOCS) == ("doc-cctp-05", "")


def test_an_article_of_the_cctp_read_outside_a_cctp_names_no_lot():
    for text in ("Les pénalités ne s'appliquent pas aux cas prévus à l'article 5 du CCTP.",
                 "Le titulaire respecte l'article 5 du présent CCTP."):
        v = _one_reference(text)
        target, why = resolve_reference(v, text, "CCAP", DOCS)
        assert target is None and why == "ambiguous: an article of a CCTP without its lot", text


def test_a_lot_pointer_still_resolves_to_its_lot():
    text = "Les quantités du CCTP 05 servent de base au bordereau."
    v = _one_reference(text)
    assert v.normalized == "CCTP 05"
    assert resolve_reference(v, text, "CCAP", DOCS) == ("doc-cctp-05", "")
    assert resolve_reference(v, text, "07", DOCS) == ("doc-cctp-05", "")


def test_an_article_of_another_piece_is_unchanged():
    text = "La résiliation suit l'article 12 du CCAP."
    v = _one_reference(text)
    assert resolve_reference(v, text, "07", DOCS) == resolve(v.normalized, DOCS) == ("doc-ccap", "")
    text = "La visite est organisée comme le prévoit l'article 3 du présent document."
    v = _one_reference(text)
    assert resolve_reference(v, text, "07", DOCS) == resolve(v.normalized, DOCS)
