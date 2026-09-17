# Explorer Slice 4 validation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Base: `e3da09991300c00b33117cc18077722d4a68ac63`. The demo freeze is unchanged.
This record separates implementation checks from independent consumer acceptance.

## Item 0 — split table recovery

`recover_tables` is a 25-line orchestrator. Candidacy, geometry, row assembly,
collapse, continuation and support have separate functions; none exceeds 68
lines. Internal candidate, geometry and fragment records replace positional
continuation tuples. No public record, threshold, rule or cache version changes.

The 236 existing table tests pass in the aligned local environment. A direct
comparison with the untouched freeze over 24 cases gives identical census,
profile, reading and report digests: twelve older table layouts and twelve
realistic-table/control/missing-page variants. The realistic fixture still
returns 98 of its 108 planted rows at this stage; the two headerless fragments
account for the ten missing rows. Repairing that behavior belongs to item 1.

The generic fixture uses a 32-page inventory and the independently specified
108-row sequence. Native column counts, widths, row heights, watermark size and
angle are parameters bounded by declared ranges. Measured geometry is not a
production default. Five-column control pages, two headerless fragments,
furniture, an approval table, and a missing-fragment variant are present.

The RAGIX gate environment is `ragix-env`, matching the frozen platform locks:
Python 3.12.12, SQLite 3.51.0 (x86_64) / 3.51.2 (ARM), and the PyMuPDF
release declared in `tools/explorer-lock-linux-64.json` and its ARM counterpart.
The separate application environment is not used for RAGIX commit gates. A probe
there failed on missing development extras; it is not a code regression or an
acceptance run. The sole missing local Explorer dependency is restored from its
existing declared pin; no dependency requirement or application environment changes.

## Item 1 — realistic table, provenance and privacy

The frozen reader returns 98/108 planted rows, losing the two headerless
fragments. The new path treats identifier-bearing first rows as data. It uses
only unambiguous adjacent profiles with at least two observed headers; it never
manufactures header text. Collapsing precedes continuation and support. The
logical table records the inherited header source for each affected fragment.
Conflicting neighbors, unmatched geometry and missing pages cannot create a
continuation bridge.

Fixture T yields one five-column table with all 108 rows for native partitions
12/13/14/15 and five-column controls. Geometry controls span column widths
80–120 points, row heights 24–40, watermark sizes 50–90 and angles 30–60 degrees.
Every overlapped physical cell carries its own lifecycle flag with its source
member id. Text is preserved. The missing-page variant produces two tables.

Approval rows remain outside id-row readings. The bounded privacy hook explicitly
associates cells from the same observed row, retaining per-character source
references. It applies the existing date/name stamp matcher and default masking;
it never associates a name on one row with a timestamp on another. All three
planted approval rows are flagged and their names masked. Source observations
are untouched. The new table-row observations are additive; old analyses still
load. Table-sensitive kernel versions advance to 0.6.0.

The integration gate is `python -m tests.saqqara.explorer_slice4_replay --output
PATH`: thirteen clean-checkout synthetic cases, including padding/control layouts,
range endpoints, the gap refusal and masked presentations. Independent consumer
confirmation remains required. No private inventory, source literal or field-reader
change belongs to this package.
