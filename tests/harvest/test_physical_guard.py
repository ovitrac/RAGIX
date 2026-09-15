"""Synthetic narrative-literal bypass regressions; parameter digits remain legal.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15
"""

import json
import pytest
from ragix_kernels.harvest.form import HarvestRefusal, validate


def packet(sentence):
    return {"node_id": "node", "summary": sentence, "summary_map": [{"sentence": 1, "children": ["child"]}],
            "interpreted": {"relevance": "critical", "type": "condition", "links": []}}


@pytest.mark.parametrize("literal", ["-7 °C", "12 V", "6 essais", "0.2 %", "4 jours", "8 à 19 °C", "<= ambient"])
@pytest.mark.parametrize("location", ["summary", "interpreted"])
def test_no_unreferenced_physical_literals(literal, location):
    p = packet("La mesure est documentaire.")
    if location == "summary":
        p["summary"] = "La mesure est " + literal + "."
    else:
        p["interpreted"]["note"] = literal
    with pytest.raises(HarvestRefusal, match="critical value written by the model"):
        validate(json.dumps(p), node_id="node", allowed_children=["child"], allowed_claims=[])


@pytest.mark.parametrize("name", ["CO2", "O2", "H2O2"])
def test_scientific_parameter_is_not_a_quantity(name):
    p = packet("La mesure porte sur " + name + ".")
    validate(json.dumps(p), node_id="node", allowed_children=["child"], allowed_claims=[])
