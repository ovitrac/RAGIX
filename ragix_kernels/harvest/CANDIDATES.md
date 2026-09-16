# Deterministic Candidate Libraries

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15

These APIs are ordinary Python libraries. They neither execute models nor require a
KOAS kernel. Classification and semantic policy are caller inputs, never inferred
authority or compliance decisions.

## Contracts

- `fr.numbers.parse_decimal`: signed exact decimals; ambiguous dot/group separators
  return `None` unless a separator is explicitly declared. No binary-float rounding.
- `quantitative.harvest`: classified text -> frozen literal candidates, including
  deterministic interval/tolerance members and raw/normalized comparator direction.
  Shared units retain their own source offsets. This is a bounded grammar, not a
  claim of exhaustive extraction. `roots` separates composites from their members;
  `route` counts quantitative kinds, not identifiers, revisions, editions or dates.
- `bindings.validate_bindings`: strict candidate-id-only constraints, injected
  semantic validation and exhaustive accounting. Every candidate has one or more
  accepted claims XOR exactly one abstention. A composite accounts for its members.
  Review flags survive; the model cannot write review states or numeric constraints.
- `bindings.validate_unparsed`: exact-span missed-expression observations, without
  normalized values. These are audit findings, not new authoritative candidates.
- `bindings.record_batch` / `validate_and_record`: reuse `DerivedStore.knowledge`
  under the `candidate-binding/1.0` namespace, retaining exact output and hashes.
  Identical replay is idempotent; changed inputs under the same batch id are refused.
  The caller owns the transaction and commits accepted or refused audit records.

```python
from ragix_kernels.harvest.quantitative import harvest, roots, route

items = harvest("8 to 19 °C", source_id="copy-digest", node_id="view-digest",
                classification="UNKNOWN")
assert roots(items)[0].kind == "interval"
assert route(items, semantic_cue=False) == "Q+"
assert "CLASSIFICATION_UNKNOWN" in roots(items)[0].flags
```

`source_id` identifies an immutable copy and `node_id` a versioned text view.
Offsets are Python character offsets into exactly the supplied text. Candidate ids
include the producer version and complete deterministic record. Callers must retain
source/view manifests and must not substitute different text under an existing id.
An accepted binding envelope is not an expert-reviewed semantic conclusion.

## Documentary Views

`saqqara.field_views` is independent of PDF packages. It preserves per-character
source references and explicit inserted separators. `line_views` checks drawn
vertical rules before joining and partitions native coalesced spans from glyph
geometry. Missing geometry, glyph/rule intersections and digit joins remain flagged.
Furniture classification precedes assembly. Rotated text is not automatically furniture.

The optional `MuPdfTextReader.page_geometry` adapter provides glyph boxes, direction,
font size and drawn vertical rules in unrotated, top-left page points. Existing PDF
placement APIs are unchanged. The caller supplies source identity and classifier.
There is no general table recognizer or automatic cross-cell field association here.

## Compatibility And Tests

Grammar version **1.6** fixes signed/decimal normalization and preserves precision.
Form version **0.9** extends the existing narrative guard to physical quantities and
composites, while retaining scientific names such as CO2. Do not reuse cached 1.5
normalizations or 0.8 form verdicts under these versions. These contracts remain
separate from `candidate-binding/1.0` and the existing recurrence `pass1` module.

The base Saqqara reader remains unchanged. `saqqara-mupdf` is optional; `ragix[all]`
also installs PyMuPDF through `translate`. Optional-reader tests run in subprocesses
so default-route import guards still exercise an uncontaminated interpreter.

```sh
python -m pytest tests/harvest tests/saqqara -q
```

All fixtures in these new public tests are independently synthetic. No external
corpus, model endpoint, network access or document-specific policy is required.

## Bounded notation safeguards

Shared-unit intervals accept explicit French/English pairs and hyphen connectors.
Borrowed units carry `UNIT_INHERITED`; a hyphen connector also carries
`SIGN_RANGE_AMBIGUOUS`. A bare integer after a label word carries
`LABEL_NUMBER_SUSPECTED`. Trailing unitless tolerances retain their own literal
span and cite the nominal unit's span. Glued uppercase K is never a thousands
multiplier and carries `UNIT_AMBIGUOUS_K`. Grouped numbers carry
`GROUPING_ASSUMED`; an undeclared ambiguous three-digit fractional tail carries
`GROUPING_AMBIGUOUS`. These flags survive binding as review reasons.

The generic binding envelope rejects numeric literals recursively in semantic
free text, including nested applicability objects and their keys. Typed candidate
and node id slots are separate. Scientific tokens such as CO2 remain admissible;
this structural guard does not replace the caller's mandatory semantic policy.
Synthetic regressions: `tests/harvest/test_explorer_prerequisites.py`.
