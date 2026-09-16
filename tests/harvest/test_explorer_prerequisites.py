"""Synthetic E0 regressions for bounded PR24 correctness follow-ups.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""
from dataclasses import replace
import pytest
from ragix_kernels.harvest.quantitative import harvest,roots
from ragix_kernels.harvest.bindings import guard_free_text,BindingRefusal
from ragix_kernels.saqqara.field_views import TextSpan,line_views


def read(text): return harvest(text,source_id="synthetic",node_id="n",classification="CONTENT")


@pytest.mark.parametrize("text,kind,a,b,flag",[
    ("entre 5 et 17 V","interval","5","17","UNIT_INHERITED"),
    ("between 5 and 17 V","interval","5","17","UNIT_INHERITED"),
    ("5-17 V","interval","5","17","SIGN_RANGE_AMBIGUOUS"),
    ("11 V ±3","tolerance","11","3","UNIT_INHERITED"),
    ("Step 6 à 21 V","interval","6","21","LABEL_NUMBER_SUSPECTED"),
])
def test_missing_composites_stay_uncertain(text,kind,a,b,flag):
    cs=read(text); c=next(c for c in cs if c.kind==kind)
    assert (c.lower,c.upper)==(a,b) if kind=="interval" else (c.nominal,c.tolerance)==(a,b)
    assert flag in c.flags and all(text[c.start:c.end]==c.raw for c in cs)
    assert all(m.candidate_id in {x.candidate_id for x in cs} for m in c.members)


@pytest.mark.parametrize("text,flag",[("> 83K","UNIT_AMBIGUOUS_K"),("7 350 kg","GROUPING_ASSUMED"),("7.350 kg","GROUPING_AMBIGUOUS")])
def test_ambiguity_is_not_ready(text,flag): assert flag in read(text)[0].flags


@pytest.mark.parametrize("value",["above 15",{"scope":[{"limit":"4 V"}]},{"range":12},"1.2e3","at −7",{"level 8":"warm"}])
def test_numeric_free_text_refused(value):
    with pytest.raises(BindingRefusal): guard_free_text(value)


def test_scientific_names_are_not_numeric_literals(): guard_free_text({"gas":"CO2","parameter":"H2O"})


def fragment(text,x,y,ident):
    return TextSpan("s",ident,1,text,(x,y-8,x+len(text)*5,y+2),tuple((x+i*5,y-8,x+(i+1)*5,y+2) for i in range(len(text))),
                    (x,y),font_size=10,state="CONTENT")


def test_baseline_jitter_orders_x_without_transitive_drift():
    a=fragment("AB",10,20.07,"a"); b=fragment("CD",23,20,"b")
    assert [v.text for v in line_views([a,b])]==["AB CD"]
    cs=[fragment("A",10,20,"a"),fragment("B",16,20.8,"b"),fragment("C",22,21.6,"c")]
    assert len(line_views(cs))==2
