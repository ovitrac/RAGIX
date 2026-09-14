"""Pass 1's ceiling is per level, and every prompt states its own (the lead: "150, 350, 800 - go").

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Row 22 ran one budget, 60 words, for a window, a document and the whole DCE alike. The lead's ladder is
150 for a window or a sheet, 350 for a document, 800 for the DCE, each a ceiling the model chooses under
by the content. This renders the prompt each level actually receives and asserts it states that level's
ceiling and no other — and that the generation cap scales with it, since two of row 24's fourteen
windows ended at `num_predict` under the old one.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

def _core():
    from ragix_kernels.harvest import core
    return core


@pytest.mark.parametrize("level, ceiling", [("window", 150), ("document", 350), ("dce", 800)])
def test_each_prompt_states_its_own_ceiling(level, ceiling):
    core = _core()
    shown = core.render_prompt(level, "Un texte.", core.P.LADDER)
    assert f"{ceiling} mots" in shown, f"the {level} prompt does not state {ceiling} words"
    assert "60 mots" not in shown, f"the {level} prompt still states the old 60"


def test_the_generation_cap_scales_with_the_ceiling():
    core = _core()
    caps = {level: core.P.num_predict_for(words) for level, words in core.P.LADDER.items()}
    assert caps["window"] > core.P.NUM_PREDICT and caps["dce"] > caps["document"] > caps["window"]
    assert all(cap >= 2.2 * core.P.LADDER[level] for level, cap in caps.items())


def test_r1_keeps_its_rule_under_a_larger_ceiling():
    core = _core()
    text = " ".join(["Une phrase de dix mots pour mesurer la règle ici."] * 20)
    under = core.P.abstract_of(text, text, ceiling=150)
    assert under["words"] <= 150 and not under["over_budget"]
    assert core.P.abstract_of(text, text)["words"] <= 60          # the default is still row 22's
