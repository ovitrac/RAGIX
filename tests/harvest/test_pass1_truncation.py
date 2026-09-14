"""R1b refined and R1c auditable — pass 1's two truncation rules, on the real records.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Written in the testing seat's kernel (`demoE2E/verify/pass1/`) by the coding seat on the
coordinating seat's relay of 2026-09-11 ~22:5x, while that seat's quota was spent, and flagged
for its review — the same arrangement as the ladder.

The rules these gates hold:

**R1b, refined.** A `length` stop is not a refusal by itself. The cap wrote the ending, so the
final fragment is dropped when it carries no terminal punctuation, and what remains is an
abstract like any other. Only an empty remainder is a refusal. Two real fixtures decide it:
row 22's 17 records of 207 whose text ends mid-word (`…d`, `…e`, `…é`, `…€`) and row 29's 37 of
142 — both counted here from the committed files rather than quoted.

**R1c, auditable.** A record whose `prompt_eval_count` reaches its `num_ctx` was cut by the
server before the model read it. `core.py` records `num_ctx` since the pass-1 merge, but the
three committed runs predate that, so the rule must DECLARE what it assumes instead of assuming
8 192 quietly: a power-of-two read from a source far larger than it is truncation-suspect, and
suspect is refused.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
ROW22 = LAB / "demoE2E/verify/pass1/outputs/core_abstracts.jsonl"
ROW24 = LAB / "demoE2E/verify/pass1/outputs_rest150/core_abstracts.jsonl"
ROW29 = LAB / "demoE2E/verify/pass1/outputs_core150/core_abstracts.jsonl"

SOURCE = ("Le présent cahier décrit les opérations de maintenance préventive des équipements de "
          "cuisine. Les prestations sont réalisées sur site par le titulaire. Les interventions "
          "de dépannage sont effectuées dans un délai convenu avec l'établissement.")
WHOLE = ("Le présent cahier décrit les opérations de maintenance préventive des équipements de "
         "cuisine. Les prestations sont réalisées sur site par le titulaire.")
FRAGMENT = " Les interventions de dépannage sont effectuées dans un dél"


@pytest.fixture(scope="module")
def P():
    from ragix_kernels.harvest import pass1
    return pass1


def records(path: Path) -> list[dict]:
    if not path.is_file():
        pytest.skip(f"{path.name} is not in this commit")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- R1b

def test_a_length_stop_that_ends_on_a_whole_sentence_is_an_abstract(P):
    """The cap fell between sentences: nothing is lost, and the record says the cap was reached."""
    v = P.abstract_of(WHOLE, SOURCE, ceiling=150, done_reason="length")
    assert v["ok"] and v["refusal"] is None
    assert v["abstract"] == WHOLE
    assert v["trimmed_after_length"] is True and v["fragment_dropped"] is None


def test_a_length_stop_drops_its_unfinished_tail_and_keeps_the_rest(P):
    """Row 29's case, 37 of 142: the tail ends mid-word, and only the tail goes."""
    v = P.abstract_of(WHOLE + FRAGMENT, SOURCE, ceiling=150, done_reason="length")
    assert v["ok"] and v["refusal"] is None
    assert v["abstract"] == WHOLE
    assert v["trimmed_after_length"] is True
    assert v["fragment_dropped"] == FRAGMENT.strip()


def test_a_length_stop_with_nothing_whole_left_is_still_a_refusal(P):
    """Row 22's case, 17 of 207 read the other way: drop the fragment and nothing remains."""
    v = P.abstract_of(FRAGMENT.strip(), SOURCE, ceiling=150, done_reason="length")
    assert v["ok"] is False and v["abstract"] == ""
    assert "truncat" in (v["refusal"] or "")
    assert v["fragment_dropped"] == FRAGMENT.strip()


def test_an_abstract_emptied_by_the_digit_rule_does_not_blame_the_cap(P):
    """K7 found this in the first draft of the refinement: under a `length` stop, a record whose
    sentences were all dropped by R2 was reported as truncated. The cap is answerable only when
    nothing whole was written at all — blaming the budget for a fabrication hides the fabrication."""
    invented = "Le délai d'intervention est de 30 jours ouvrés."      # no such figure in SOURCE
    v = P.abstract_of(invented, SOURCE, ceiling=150, done_reason="length")
    assert v["ok"] is False and v["abstract"] == ""
    assert v["refusal"] == "every sentence dropped by the digit rule"
    assert v["dropped"] and v["dropped"][0]["offences"]


def test_an_unfinished_tail_is_kept_when_the_model_stopped_by_itself(P):
    """The rule is about the cap, not about punctuation: a model that ends badly on its own
    terms is judged by R2, and its sentence stands or falls on its digits."""
    v = P.abstract_of(WHOLE + FRAGMENT, SOURCE, ceiling=150, done_reason="stop")
    assert v["abstract"] == (WHOLE + FRAGMENT).strip()
    assert v["trimmed_after_length"] is False and v["fragment_dropped"] is None


def test_the_two_committed_runs_split_exactly_where_the_rule_says(P):
    """No new run: the rule is counted against what is already on disk."""
    r22 = [r for r in records(ROW22) if r.get("done_reason") == "length"]
    unfinished = [r for r in r22 if not P.ends_whole(r.get("abstract") or "")]
    assert (len(r22), len(unfinished)) == (207, 17)
    r29 = [r for r in records(ROW29) if r.get("done_reason") == "length"]
    assert (len(r29), sum(1 for r in r29 if not P.ends_whole(r.get("abstract") or ""))) == (142, 37)
    assert all(r.get("ok") for r in r29), "row 29 admitted every one of them, which is what R1b refines"


def test_a_fragment_count_is_not_a_record_count(P):
    """The testing seat's reading of K7, 2026-09-12: 17 fragments in row 22 are not 17 refused
    records — 7 of them keep a whole sentence and are admitted, and 10 are refused. Row 29's 37
    fragments cost exactly 1 record. Both numbers must be printed side by side, or a clean rate
    gets read off the wrong line."""
    for path, expect in ((ROW22, (207, 17, 10, 7)), (ROW29, (142, 37, 1, 36))):
        rows = records(path)
        stops = [r for r in rows if r.get("done_reason") == "length"]
        fragments = [r for r in stops if not P.ends_whole(r.get("abstract") or "")]
        verdicts = [P.abstract_of(r["abstract"], r["abstract"], 150, "length", [], []) for r in fragments]
        refused = [v for v in verdicts if not v["ok"]]
        assert (len(stops), len(fragments), len(refused), len(fragments) - len(refused)) == expect
        # and every one of them lost its tail: the fragment count is about tails, not about records
        assert all(v["fragment_dropped"] for v in verdicts)


# --------------------------------------------------------------------------- R1c

def test_a_recorded_num_ctx_decides_by_itself(P):
    verdict = P.input_truncation({"prompt_eval_count": 8192, "num_ctx": 8192, "source_chars": 707085})
    assert verdict["refused"] and verdict["num_ctx"] == 8192 and "recorded" in verdict["basis"]
    assert P.input_truncation({"prompt_eval_count": 2736, "num_ctx": 8192,
                               "source_chars": 40000})["refused"] is False


def test_without_a_recorded_num_ctx_the_assumption_is_declared_not_made(P):
    """The three committed runs carry no num_ctx. A power-of-two read from a source far larger
    than it can be is truncation-suspect, and suspect is refused — with the assumption named."""
    suspect = P.input_truncation({"prompt_eval_count": 8192, "source_chars": 707085})
    assert suspect["refused"] and suspect["num_ctx"] is None
    assert "assum" in suspect["basis"] or "suspect" in suspect["basis"]
    # a reading that is not at a power-of-two boundary is not suspected on size alone
    assert P.input_truncation({"prompt_eval_count": 2736, "source_chars": 707085})["refused"] is False


def test_the_dce_records_of_the_three_runs_are_judged_as_the_record_shows_them(P):
    """Row 22's DCE was read whole; rows 24 and 29 were cut at 8 192 and admitted anyway —
    R1c is what refuses them, and it must do so from the record alone."""
    def dce(path):
        return next(r for r in records(path) if r.get("level") == "dce")
    r22 = dce(ROW22)
    assert "num_ctx" not in r22                       # the gate's own blind spot, named here
    assert P.input_truncation(r22)["refused"] is False
    for path in (ROW24, ROW29):
        r = dce(path)
        assert r.get("ok") is True                    # the run admitted it
        assert P.input_truncation(r)["refused"] is True   # and R1c refuses it from the record


def test_every_record_written_from_now_on_carries_its_num_ctx():
    """The blind spot closed at the source: core.py sets num_ctx on the record before the call,
    so a transport failure carries it too."""
    from ragix_kernels.harvest import core as core_module
    core = Path(core_module.__file__).read_text(encoding="utf-8")
    assert 'record["num_ctx"] = num_ctx' in core
    assert core.index('record["num_ctx"] = num_ctx') < core.index('post(a.host, "/api/generate"')


# --------------------------------------------------------------------------- the budget

def test_the_generation_cap_is_three_tokens_a_word_with_a_quarter_of_margin(P):
    assert P.TOKENS_PER_WORD == 3.0
    assert [P.num_predict_for(c) for c in (150, 350, 800)] == [563, 1313, 3000]
    # and the density that was measured stays in the header, not in the arithmetic
    assert "2.2" in (P.num_predict_for.__doc__ or "") + P.__doc__
