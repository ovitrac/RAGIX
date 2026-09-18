# Table context for quantitative readers

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Structural context is distinct from interpretation. `harvest.table_context.Cell`
records a located source cell, its row/column extent and source span ids.
`associate(value, cells, header_rows=..., row_label_columns=...)` returns a
`CellContext` containing separate column-header and row-label cells. No text is
concatenated and no unit or parameter is interpreted by this layer.

Header rows and stub columns are explicit structural inputs. No declared header
means no column-header association. Multiple declared header rows are retained;
a spanning header must cover the whole value-column extent. Cross-copy, cross-table
and incorrect column/row links are rejected.

The Explorer adapters expose `reading.cell_contexts` and `report.cell_contexts`
records. Recovered tables use physical member ids and the observed header
source retained by continuation. Native cell grids can supply context even when
the separate ID-row reader cannot classify their columns. Unlocated cells and
headers whose topology cannot be verified produce scoped findings. They never
borrow nearby prose or promote the first data row to a missing header.

For native grids the first column is exposed as a positional stub by default;
`native_contexts(..., row_label_columns=...)` permits an explicit alternative or
an empty tuple. This is a structural convention, not a claim about parameter
meaning. Recovered ID-row tables default to no stub columns; callers may provide
an explicit stub selection to `recovered_contexts`.

Use `Cell.span(start, end)` for an exact cell-local source slice. Offsets from a
header belong to that header, never to the value cell. Context findings remain
separate from the unchanged source observations.

## Inherited units

Pass a `CellContext` to `harvest(..., context=context)` with the value cell's
source id, cell id and unchanged text. The result uses `quantitative-context/1`
and records `unit_source` (`INLINE`, `INHERITED`, `NONE`), `unit_evidence` with exact
cell-local spans, `context_evidence`, and an explicit `unit_reason` when unresolved.
Calls without context retain the existing candidate contract and identifiers.

Bracketed units and their spaced degree notation are recognized in associated
headers/labels. Conflicting units remain alternatives with no chosen unit. An
inline unit remains the literal observation; a contradictory context flags it.
Uncertain context, numeric neighboring values and unsupported value suffixes
cannot supply a silently assumed unit. Inherited units always retain the
`UNIT_INHERITED` review flag. No unit is appended to the source text; an inherited
unit's offsets belong to its evidence cell, so the value's `unit_start` and
`unit_end` stay null.

The cell reader retains composite member graphs and exact value spans. It replaces
a previous text-view batch only when all its character provenance lies inside
cells being re-read; it never drops one child from a retained composite. Original
observations are unchanged. Units and numeric shapes confer no parameter role,
applicability, comparison direction or documentary authority.

## Relative expressions

With cell context, complete expressions such as `datum -13 to datum -31` produce a
`RelativeQuantityCandidate` (`relative-quantity/1`). Each offset retains its
operator, magnitude, signed number when unambiguous, exact source span and anchor
span. The anchor remains text: the reader assigns no parameter name or role.
`offset_lower` and `offset_upper` describe offsets only, and remain null when the
anchor, unit or signs do not support an ordered range. The ordinary absolute
`number`, `lower`, `upper` and `nominal` fields stay null.

The unit can be inline or inherited with the same exact provenance rules. A header
that explicitly says `relative to datum [mm]` can supply the anchor for a signed
value cell; an ordinary header such as `Travel [mm]` cannot. Anchor evidence then
points to the header's actual character range. Multiple or inconsistent anchors,
conflicting units and branching `±` signs stay unresolved. Bare arithmetic,
identifier hyphens and en/em dashes do not become anchored offsets. Partial relative
matches produce `relative_unparsed` with the exact source and no normalized number;
they cannot fall back to apparently absolute scalar quantities.

Every relative record retains `RELATIVE_REFERENCE_UNRESOLVED` and requires review.
There is no reference-parameter assignment, applicability decision, absolute-range
derivation or cross-document comparison in this module.

Stored contexts can be validated and re-read without rebuilding their text:

```python
from ragix_kernels.harvest.table_context import context_from_dict
from ragix_kernels.harvest.quantitative import harvest

context = context_from_dict(stored_context)
candidates = harvest(
    context.value.text,
    source_id=context.value.source_id,
    node_id=context.value.cell_id,
    classification="UNKNOWN",
    context=context,
    token_locale=True,
)
```

Explorer's read/report kernels use version 0.9.0 for this combined capability;
existing census/profile versions and environment locks are unchanged. Run the full
suite and `python -m tests.harvest.header_units_replay --output PATH` from a clean
checkout on both supported architectures. Compare `cases`, not host provenance.
