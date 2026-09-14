"""Gate GR3-1a — the deterministic coarse tagger is a HONEST baseline.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The LLM lane of the requirement lane must beat this tagger (gate GR3-1). A
baseline is only usable if it is itself falsifiable, so this gate pins the
properties the comparison relies on:

  1. totality      — always returns a tag; the theme is inside the closed
                     vocabulary or ``unknown``;
  2. determinism   — same input, same tag (no RNG, no dict-order luck);
  3. ordered signals — ancestry beats free text, exact beats containment,
                     specificity decides inside one element;
  4. fail closed   — an exact tie abstains and names its competitors;
  5. verbatim evidence — the reported evidence is a substring of what the
                     tagger actually read. ``RequirementRecord.from_extraction``
                     depends on this: a tagger that reports a normalized form
                     would be dropped by the span guard (it was, before this
                     property was enforced — the guard caught it).

No corpus is touched here: the gate runs on synthetic inputs plus a sweep over
the shipped vocabulary.
"""

from __future__ import annotations

from ragix_kernels.tender.domain import tagger, vocabulary
from ragix_kernels.tender.domain.records import Applicability
from ragix_kernels.tender.domain.requirements import RequirementRecord

VOCAB = vocabulary.load_axes()
APP = Applicability(project="PROJECT-A")


def test_totality_and_closed_output():
    for text, anc in [("", []), ("n'importe quoi", []),
                      ("Quelles sont les modalités ?", ["Feuille 1"]),
                      ("x", ["", "  "])]:
        t = tagger.tag(text, [a for a in anc])
        assert t.theme == "unknown" or t.theme in vocabulary.axis_ids()
        assert t.signal in ("ancestry-exact", "ancestry-contains",
                            "text-contains", "ancestry-token", "ambiguous",
                            "none")


def test_determinism():
    args = ("Les pénalités de retard sont-elles plafonnées ?",
            ["Pénalités de retard", "CCAP"])
    assert tagger.tag(*args).to_dict() == tagger.tag(*args).to_dict()


def test_ancestry_beats_free_text():
    t = tagger.tag("Décrivez vos mesures de sécurité applicatives.",
                   ["Propriété intellectuelle"])
    assert t.theme == vocabulary.resolve("propriété intellectuelle")
    assert t.signal.startswith("ancestry")


def test_exact_beats_containment_and_deepest_wins():
    deep = tagger.tag("q", ["Réversibilité", "Mesures de sécurité du marché"])
    assert deep.theme == vocabulary.resolve("réversibilité")
    assert deep.depth == 0


def test_specificity_decides_inside_one_element():
    # two axes of the vocabulary nest: "lieu d'exécution" is contained in
    # "lieu d'exécution des prestations". The more specific one must win, and
    # the loser must be traced — a beaten candidate is never silently dropped.
    t = tagger.tag("q", ["Article 4 Lieu d’exécution des prestations"])
    assert t.theme == vocabulary.resolve("lieu d’execution des prestations")
    assert vocabulary.resolve("lieu d’exécution") in t.competitors


def test_a_weaker_signal_never_overrides_a_stronger_one():
    # `confidentialité` (one token, signal 4) cannot beat
    # `obligation de confidentialité` (two tokens, signal 2) — the ordering of
    # signals decides before specificity does.
    t = tagger.tag("q", ["5.2 Obligation de confidentialité du titulaire"])
    assert t.theme == vocabulary.resolve("obligation de confidentialité")
    assert t.signal == "ancestry-contains"


def test_tie_abstains_and_names_competitors():
    hits = [("AX-a", ("x", "y"), "x y"), ("AX-b", ("z", "w"), "z w")]
    t = tagger._decide(hits, "ancestry-contains", 0)
    assert t.theme == "unknown" and t.signal == "ambiguous"
    assert t.competitors == ["AX-a", "AX-b"]


def test_single_token_axes_never_fire_on_free_text():
    # `rgpd` is a one-token axis: allowed from a heading, refused from prose
    assert tagger.tag("Le RGPD est respecté.", []).theme == "unknown"
    assert tagger.tag("q", ["RGPD"]).theme == vocabulary.resolve("rgpd")


def test_single_token_axes_refused_on_a_prose_ancestry_element():
    prose = ("Les candidats sont invités à remplir cette annexe financière "
             "en indiquant le prix par profil et par objet")
    assert len(vocabulary.fold(prose).split()) > tagger.HEADING_MAX_TOKENS
    assert tagger.tag("PRIX € HT par jour", [prose]).theme == "unknown"


def test_evidence_is_verbatim_for_every_axis_in_the_vocabulary():
    """Sweep: each axis name planted in a heading must yield a substring."""
    checked = 0
    for axis in VOCAB["axes"]:
        el = f"Article 7 {axis['canonical']}"
        t = tagger.tag("question quelconque", [el])
        if t.theme == "unknown":
            continue                      # abstention is a valid outcome
        assert t.evidence and t.evidence in el, axis["axis_id"]
        checked += 1
    assert checked > 40                   # the sweep must actually exercise it


def test_tagger_evidence_survives_the_record_span_guard():
    text = "Quelles sont les mesures de sécurité mises en place ?"
    anc = ["Hébergement hors site"]
    t = tagger.tag(text, anc)
    rec = RequirementRecord.from_extraction(
        payload="\n".join([text, *anc]), scale="coarse",
        evidence_span=t.evidence, source_kind="question",
        source_id="q_test", source_path="PROJECT-A/x.xlsx",
        applicability=APP, channel="vocab-tagger", theme=t.theme)
    assert rec.trace["span_guard"] == "verified"


def test_ancestry_chain_reads_both_corpus_shapes():
    xlsx = {"ancestry": {"sheet": "Maturité sécurité", "block_title": "SAAS",
                         "section_row_text": "Gestion des vulnérabilités"}}
    assert tagger.ancestry_chain(xlsx) == ["Gestion des vulnérabilités",
                                           "SAAS", "Maturité sécurité"]
    pdf = {"ancestry": {"headings": ["Titre I", "Chapitre 2", "2.1 Pénalités"],
                        "resolution": "page"}}
    assert tagger.ancestry_chain(pdf)[0] == "2.1 Pénalités"   # deepest first
    assert tagger.ancestry_chain({}) == []
