"""Percentage kind follows the observed unit independently of its numeric sign.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from decimal import Decimal
import pytest
from ragix_kernels.harvest.quantitative import harvest, roots


@pytest.mark.parametrize("token_locale", [False, True])
@pytest.mark.parametrize("sign", ["", "+", "-", "−"])
@pytest.mark.parametrize("literal", ["5", "0.1", "0,1", "0,001"])
def test_sign_does_not_change_percentage_kind(sign, literal, token_locale):
    text = sign + literal + " %"
    candidates = harvest(
        text,
        source_id="fixture",
        node_id="node",
        classification="CONTENT",
        token_locale=token_locale,
    )
    (candidate,) = roots(candidates)
    assert candidate.kind == "percentage"
    assert candidate.dimension == "percentage" and candidate.unit == "%"
    assert candidate.raw == text and (candidate.start, candidate.end) == (0, len(text))
    assert text[candidate.unit_start : candidate.unit_end] == candidate.unit_raw == "%"
    assert Decimal(candidate.number) == Decimal(
        (sign + literal).replace("−", "-").replace(",", ".")
    )
    assert candidate.normalization_status == "parsed"
    assert not candidate.members


@pytest.mark.parametrize("token_locale", [False, True])
@pytest.mark.parametrize("text", ["230 V +/-10%", "230 V ± 10 %", "± 5 %"])
def test_percentage_operand_does_not_break_tolerance(text, token_locale):
    candidates = harvest(
        text,
        source_id="fixture",
        node_id="node",
        classification="CONTENT",
        token_locale=token_locale,
    )
    (root,) = roots(candidates)
    assert root.kind == "tolerance" and root.raw == text
    by_id = {c.candidate_id: c for c in candidates}
    (operand,) = (by_id[m.candidate_id] for m in root.members if m.role == "tolerance")
    assert operand.kind == "percentage" and operand.unit == "%"
    assert operand.raw == text[operand.start : operand.end]
