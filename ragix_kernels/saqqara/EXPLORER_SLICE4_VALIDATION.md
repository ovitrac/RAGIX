# Explorer Slice 4 validation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Base: `e3da09991300c00b33117cc18077722d4a68ac63`. The demo freeze is unchanged.
This record separates implementation checks from independent consumer acceptance.
Current state: the table-row masking extension is withdrawn. The earlier privacy
results below are historical; the split record at the end supersedes them.

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

## Measured commit gates

All gates use detached checkouts and pinned imports in `ragix-env`.

| Commit | Scope | x86_64 full suite | ARM full suite |
|---|---|---|---|
| `af036e0` | Stage split | 2,377 passed / 38 skipped | 2,377 passed / 38 skipped |
| `fdcfa2e` | Tables, cell evidence, bounded privacy hook | 2,403 passed / 38 skipped | 2,403 passed / 38 skipped |
| `1cd625b` | E11 integration | 2,468 passed / 38 skipped | 2,468 passed / 38 skipped |

The refactor's 28 legacy replay cases and privacy outputs match the freeze and
both architectures. The table package's 13 table/gap/masking cases match across
architectures. E11's 54 exact-span replay cases also match, and the table replay
remains green at the E11 commit. No model calls were used.

During setup, the declared Explorer dependency was absent locally; the declared
PDF renderer and PDF-mining extras were absent on ARM. They were restored only
in the RAGIX gate environments. The separate demo environment was not modified,
and neither source tests nor platform locks were weakened.

### Fixture slot-projection correction

The first executable fixture padded wide-header placeholders at the end of its
raw matrix. It now projects each physical cell anchor onto the observed native
grid, leaving the spanned slots empty. This corrects the fixture representation,
not production behavior. Four explicit anchor-position tests accompany the
correction, and its synthetic intake version is `slice4/1`. The corrected table
fixture still yields the specified 108 rows and passes the 30 table-specific
tests. Its replay must be regenerated from the final fixture commit.

Independent confirmation on the seven documents is still required before merge.
Each item has had zero private confirmation strikes in this run. E12 remains
held until the reference-field owner releases its shared files. No change to the
frozen main branch, private reference-field implementation, or decision registry
is part of this work.


## Split — withdraw table-row masking

Independent confirmation rejected the table-row masking extension. Associating
name-shaped fragments across cell boundaries and promoting their component words
to global masking terms can over-mask ordinary body text. The table recovery,
per-cell glyph evidence and E11 changes are retained.

`privacy.py` is restored byte-for-byte to the freeze. Table-row stamp observation,
serialization and its two hook tests are removed. The table replay retains its
presentation digests but now asserts the freeze's zero stamp count on fixture T.
Existing line-stamp privacy tests remain in force. Approval-row masking is deferred
to a separately specified change; this split does not implement that redesign.
The earlier zero-strike statement is superseded: row masking has one failed
independent confirmation. Consumer counts and corpus findings stay outside this
repository.

The kernel adapter contains no separate stamp wiring: its only differences from
the freeze are the version values. They remain 0.6.0 / 0.6.1 as ruled; an integration
version bump is required separately. E12 remains held for the reference-field
branch's merge and path release. This split does not move the demo freeze.


## Integrated release

The reference-field and table branches are integrated together. All four Explorer
kernels now use version **0.7.0**: census, profile, read and report. Their cache
identities therefore differ from both standalone branches and the earlier reader.
The runtime sources match the independently confirmed combined implementation.
The separate PDF-cleaner utility remains additive and is not called by Explorer.

The executable contracts are stated in SPEC K9.11–K9.14. Reproduce the three
synthetic replays from a clean checkout in the declared gate environment:

```bash
python -m tests.saqqara.explorer_slice4_fields_replay --output fields.json
python -m tests.saqqara.explorer_slice4_replay --output tables.json
python -m tests.harvest.explorer_slice4_quantity_replay --output quantities.json
```

Compare the `cases` objects across architectures; provenance identifies each host
and checkout separately. Run the full repository suite as well. Consumer source
material and confirmation measurements remain outside this repository.

The strict expected failure for weak gap-bound derivation remains visible. Slash
revision parsing, the implicit field-end rule, labels intersected by a rule with
values on another row, identifier-family expansion and table-row privacy redesign
remain outside this release. No environment lock or extractor pin is changed.
