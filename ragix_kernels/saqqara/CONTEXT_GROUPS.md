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

The text ledger has one entry per native source span:

- `CARRIED`: all original character offsets are represented by checked mappings.
- `CARRIED_EXACT`: a native cell or canonical cell member contains the whole literal
  span within observed geometry on the same page. A shared span id is insufficient.
- `PARTIAL`: some mapped characters survive; the missing half-open offset ranges
  are explicit.
- `NOT_CARRIED`: no sufficient literal representation was established.
- `EXCLUDED`: every source character belongs only to lines already classified as
  furniture by the reader. The ledger adds no exclusion rule.

`passes` requires zero `PARTIAL` and zero `NOT_CARRIED` entries. The summary reports
all statuses by page and for the document. It establishes retention of observations,
not completeness of extraction, correctness of a table or absence of unreadable
cells. Original source text in the ledger is observation data; use the consumer's
existing presentation controls when exposing it.

## Verification

Run `python -m pytest tests/saqqara/test_context_groups.py -q`, then the repository's
full test suite and forbidden-content guard. The synthetic suite includes mapped
sign loss, same-id text substitution, cells without source spans, missing geometry,
oversized regions, cell-only chunks, repeated occurrences and canonical snapshots.
The connectivity view is separate and is not implemented by this slice.
