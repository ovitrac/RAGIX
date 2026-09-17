# E11 composite-span validation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

The independently specified fixture set contains 24 FR/EN rows and three
additional candidates. All 27 are checked with legacy and token-local notation:
54 exact-span assertions. Additional controls retain child provenance and
uncertainty, keep compound-duration units without conversion, and keep mixed-unit
dash neighbors separate. Missing nominals are flagged, never replaced with zero.
No new numerical decision threshold is introduced.

The changes cover leading tolerances, equality, word comparators, shared-unit
word ranges, compound durations, signed percentages, electrical qualifiers,
glued tolerances and calendar-period rates. Cardinalities remain separate from
neighboring measurements. Explicit incompatible word ranges retain unresolved
composites; a dash between incompatible units leaves two scalar candidates.
Literal signs, source text, offsets and observed units are preserved. Composite
members refer only to immutable child ids. Reader/report kernel versions advance
to 0.6.1; the census/profile table versions remain 0.6.0.

## Existing-fixture audit

The same 423 existing Explorer tests pass before and after the change. Capture on
the untouched freeze and the candidate yields **1,534 identical-input calls**,
**zero changed outputs** and **zero unmatched inputs**. Comparison includes all
candidate fields and ids, not just normalized values. Thus the changed-candidate
list for those existing inputs is empty. This covers the existing X1–X13 and
X16–X17 constructions; it does not claim completion of the deferred E12/X14 work.

The opt-in plugin `tests/harvest/explorer_quantity_audit.py` makes this audit
reproducible. Put that plugin directory and the target checkout on `PYTHONPATH`,
run from the target checkout, and use:

```bash
python -m pytest tests/saqqara/test_explorer.py \
  tests/saqqara/test_explorer_slice2.py tests/saqqara/test_explorer_tables.py \
  -p explorer_quantity_audit --quantity-audit-output /path/to/audit.json
```

Compare output dictionaries by identical input keys. The new-fixture replay is
`python -m tests.harvest.explorer_slice4_quantity_replay --output PATH` from a
clean checkout. Architecture replay, full-suite records and independent consumer
root-fidelity confirmation are separate gates; fixture success is not a claim
that all residual composite forms have been recovered.
