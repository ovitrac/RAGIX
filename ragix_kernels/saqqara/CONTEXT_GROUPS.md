# Context groups and literal source retention

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

A rejected table layout does not make its source text unusable. Context groups
provide a bounded overlay over existing line members and native cell observations.
They make no table, header, applicability or semantic reading-order assertion.
The canonical region reader remains unchanged.

## API

```python
from ragix_kernels.saqqara.context_groups import (
    ContextGroupPolicy, context_groups, cell_inventory,
)
from ragix_kernels.saqqara.content_ledger import text_ledger
from ragix_kernels.saqqara.renderable_regions import regions_from_explorer

# result is an existing ExplorerResult; no extraction happens in these functions.
index = regions_from_explorer(result)
source_id = result.document.source_id
policy = ContextGroupPolicy(max_group_members=400, max_group_cells=400,
                            page_scale_fraction=0.5)
groups = context_groups(result, index, source_id=source_id, policy=policy)
cells = cell_inventory(result, groups, source_id=source_id)
ledger = text_ledger(result, index, groups, source_id=source_id)
summary = ledger.summary()
# ledger.passes describes observed text retention only.
```

The outputs are frozen dataclasses, serializable using `dataclasses.asdict`.
No function modifies `result`, `index`, source observations or their text.
`RegionRefused` reports malformed identities, inconsistent observations and stale
mapping or member text. A measured retention gap returns a failing ledger instead
of repairing or discarding the source text.

## Context and chunk identity

Each group records its source, native candidate, strict refusal code, envelope,
policy, members and cell ids. Recovered-table refusals are traced to their native
candidate observations by the exact member ids in the refusal record.

Envelopes are unions of observed cell boxes. A substituted container box never
establishes membership. With no observed box, the group has no line members and
still retains its native cells. Partially known and page-scale envelopes are
flagged. Equal text at different positions remains distinct.

Groups pack regions in page/geometry order, with ids breaking ties. An oversized
region splits at member boundaries with `REGION_SPLIT`. Cells are independently
sorted and chunked, then assigned to matching chunk indices; extra cell chunks
can have no line members. Thus a chunk need not contain all line and cell readings
of the same text. Consumers can follow its candidate and neighbouring links.
The union of chunks retains both complete eligible inventories. Source ids,
candidate ids, chunk indices, member ids and cell ids determine group identity.

A canonical member shared by two candidate envelopes is explicitly shared, not
owned twice by the canonical reader. Consumers should not count those groups as
independent evidence. Member/cell counts do not bound bytes or model tokens;
packet limits and presentation policy belong to the consumer.

## Two separate inventories

The cell inventory preserves literal text, readability, source-span associations,
geometry kind and flags. Missing geometry is exposed as `observed_bbox=None` with
a labelled `container_bbox`. `None` means unreadable; only `""` means readable-empty.
The inventory validates every cell in every candidate named by the supplied groups.
The text ledger additionally checks that every refusal has its context group, so
run both functions when evaluating the document gate.

The `text-occurrence-ledger/2` record has one entry per native source span.
Version 2 adds normalisation, order and occurrence accounting to the version-1
literal coverage record:

- `CARRIED`: all original character offsets are represented by checked mappings.
  Mapped order is assessed separately; any `order_breaks` make the ledger fail.
- `CARRIED_EXACT`: a native cell or canonical cell member contains the whole literal
  span within observed geometry on the same page. Distinct, non-overlapping text
  occurrences are consumed in geometric order; a shared span id is insufficient.
- `CARRIED_NORMALISED`: all content is retained after a declared
  `whitespace-normalisation/1` transformation. Unmapped Unicode whitespace is
  accepted only when the actual carrier keeps whitespace between the adjacent
  source content or retains a line/cell boundary. Edge trimming is never enough
  when it merges tokens across a source-span seam.
- `PARTIAL`: carried or accepted-normalised characters survive, with every
  missing half-open offset range explicit.
- `NOT_CARRIED`: no sufficient literal representation was established.
- `EXCLUDED`: every non-whitespace source character belongs only to lines already
  classified as furniture by the reader, while remaining whitespace is carried
  or accepted under `whitespace-normalisation/1`. The ledger adds no exclusion rule.

Each entry records `carried_count`, accepted-normalised ranges with their rule,
and missing ranges. These form a disjoint partition of the source offsets; bad
arithmetic or overlap refuses construction. `order_breaks` records mapped
characters that do not strictly increase within one carrier. `passes` requires
zero missing-content entries and zero order breaks. The summary separates passed,
accepted-normalised, missing and excluded populations by page and document.

The ledger establishes character retention only. It does not establish row/column
association, table structure, completeness of extraction or absence of unreadable
cells. Original source text in the ledger is observation data; use the consumer's
existing presentation controls when exposing it.

## Verification

Run `python -m pytest tests/saqqara/test_context_groups.py
tests/saqqara/test_content_ledger_v22.py -q`, then the repository's full test
suite and forbidden-content guard. The synthetic suite includes mapped sign loss,
same-id text substitution, cells without source spans, missing geometry, cross-span
seams, order reversal, occurrence consumption, explicit partition arithmetic,
oversized regions, cell-only chunks and canonical snapshots. The connectivity
view is separate and is not implemented by this slice.
