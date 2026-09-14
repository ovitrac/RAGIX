"""fr.cut — the two opposite failures of a text layer that splits digits across a line break.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

A value is CUT when its span begins on the far side of a digit run the layer split: « 0,00 € » read
where the page prints 250,00 €. A value is JOINED when the grammar reads a digit run across a line
break — « 150\\n50 » read as 15050 — which is sometimes one printed token and sometimes two table
cells, and the store cannot say which. Invented text throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragix_kernels.harvest.fr.cut import VERSION, cut_spans, is_cut, joined_runs


def test_runs_are_quoted_as_the_store_has_them():
    assert joined_runs("Jeudi\n1\n4\noctobre\n203\n1\nà 10 heures 00") == ["1\n4", "203\n1"]
    assert joined_runs("150\n50 €") == ["150\n50"]
    assert joined_runs("15\njui\nllet") == []          # letters split, no digit run joined
    assert joined_runs("le 14 octobre 2031") == []
    assert joined_runs("") == [] and joined_runs(None) == []


def test_the_tail_of_a_number_split_by_a_newline_is_cut():
    text = "Forfait de 25\n0,00 € par jour"
    assert is_cut(text, text.index("0,00"))
    assert is_cut("Montant 25 \n  0,00 €", len("Montant 25 \n  "))   # spaces and tabs around the break


def test_a_newline_after_a_letter_or_a_space_alone_is_not_a_cut():
    assert not is_cut("Forfait\n250,00 €", len("Forfait\n"))          # a letter before the break
    assert not is_cut("Montant 25 0,00 €", len("Montant 25 "))       # a space is not a leaf boundary
    assert not is_cut("fin de page.\n12 heures", len("fin de page.\n"))


def test_a_tail_that_does_not_start_with_a_digit_is_not_a_cut():
    """The guard that removed every false cut on letter-initial spans: « Article I » or « Page 4 sur
    8 » after a line that ends in a page number is ordinary prose."""
    assert not is_cut("Page 4\nArticle I", len("Page 4\n"))
    assert not is_cut("12\n  Page 4 sur 8", len("12\n  "))


def test_the_edges_of_the_text_are_never_cut():
    assert not is_cut("250,00 €", 0)
    assert not is_cut("12\n3", 99)


@dataclass
class _Span:
    start: int


def test_cut_spans_reads_offsets_or_objects_carrying_a_start():
    text = "Forfait de 25\n0,00 € et 40 €"
    spans = [_Span(text.index("0,00")), _Span(text.index("40"))]
    assert cut_spans(text, spans) == {0}
    assert cut_spans(text, [text.index("0,00"), 0]) == {0}
    assert VERSION == "tender.cut 1.1"
