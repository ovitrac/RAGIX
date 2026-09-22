# Native-cell connectivity view

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

`grid_view` is a pure, geometry-only view over one native `TableObservation`.
It does not change the canonical region reader, admit a refused table, infer a
semantic table, establish a header, or assign a value to a model or requirement.

## API

```python
from dataclasses import asdict
from ragix_kernels.saqqara.grid_view import grid_view

view = grid_view(observation, source_id=document.source_id)
payload = asdict(view)
```

The caller supplies the immutable source identity. Every `Evidence` record on the
observation must name that source and page, and every cell must have matching
evidence. Identity disagreement, duplicate cells and missing evidence fail closed.

## Geometry and states

Coordinates are rounded once to 0.001 point. Original and rounded boxes remain
separate. Consecutive x and y edges form elementary half-open tiles, exposed as
`view.tiles[y][x]`. Their owners are sorted cell ids:

- `UNRESOLVED`: no observed cell rectangle covers the tile;
- `OWNED`: exactly one cell covers it;
- `CONFLICT`: several distinct cells cover it.

Missing cell geometry and non-cell geometry own no tile. Those observations remain
in `cells` and are referenced by `unlocated_cells`. A positive-sized original
box that collapses after rounding likewise remains in `cells` and is referenced by
`degenerate_cells`. `None` is unreadable; `""` is readable-empty. Neither
state is discarded.

A merged cell remains one observation with half-open x/y index ranges spanning
all of its tiles. Equal strings and equal boxes do not merge distinct cell ids.

## Components and header proposals

Components are maximal four-connected sets of `OWNED` tiles. They may be
nonrectangular. Conflict and unresolved tiles never connect components. Each
component records the part of each cell scope that intersects it; the cell remains
global and whole. `EXTENDS_BEYOND_COMPONENT` reports that its full scope reaches
a conflict or another component.

`header-proposal/1` inspects only a component's geometric first tile row.
It emits a hypothesis only when every tile in that row has a readable, nonempty,
single-x-tile owner wholly contained in the component. It never falls through to
a later row. A successful proposal is still not documentary or semantic authority.

View status is deliberately narrow:

- `COMPLETE`: every defined tile is singly owned, one component covers them,
  and no cell is unlocated or degenerate;
- `PARTIAL`: some usable connectivity exists but a hole, conflict, multiple
  component, unlocated cell or degenerate cell remains;
- `NONE`: no singly owned component exists.

These states describe geometric occupancy only. They say nothing about extraction
completeness, table semantics, readability, applicability, evidence sufficiency,
qualification or compliance.

## Verification

Run:

```bash
PYTHONPATH=. python -m pytest tests/saqqara/test_grid_view.py -q
```

The synthetic suite covers merged scopes, cross-row conflicts, holes, unreadable
and empty cells, missing geometry, identical boxes, connected L/T shapes, cells
crossing a conflict, header non-fallthrough, coordinate collapse and deterministic
replay. Canonical region/refusal snapshots must remain byte-identical.
