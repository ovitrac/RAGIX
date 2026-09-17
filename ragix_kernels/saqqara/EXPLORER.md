# Document Explorer

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Explorer discovers observed notation before applying a reader. It exposes an
explicit UNKNOWN_TEMPLATE result when required notation cannot be supported.
It does not infer document identity equivalence or resolve section references.

```python
import json
from pathlib import Path
from ragix_kernels.saqqara.explorer import digest_pdf, explore
from ragix_kernels.harvest.report import render_report

# Select the lock for the current platform.
lock = json.loads(Path("tools/explorer-lock-linux-64.json").read_text())
# PyMuPDF is an explicit opt-in dependency with its existing licence conditions.
digest = digest_pdf(path, expected_pymupdf=lock["pymupdf"])
result = explore(digest)
html = render_report(result.report, {
    "field": "Declared field", "quantity": "Literal quantity",
    "table_row": "Observed row",
})
```

For an existing typed digest, construct `DocumentDigest` or use
`census.digest_from_dict`. The deterministic library does not import KOAS or call
a model. Raw words, glyph boxes, rules and table observations remain separate
from census counts, profile derivation and reader results.

The CLI accepts a typed digest JSON, a PDF, or a folder of PDFs:

```bash
python -m ragix_kernels.saqqara.cli.explorer input.json --output /tmp/explorer-report
python -m ragix_kernels.saqqara.cli.explorer documents/ --pdf-reader pymupdf \
  --extractor-version "$EXPLORER_PYMUPDF_VERSION" --provenance --gate --output /tmp/explorer-report
```

Set `EXPLORER_PYMUPDF_VERSION` from the platform lock before the PDF command.
A folder run deduplicates byte-identical copies. `--gate` refuses a dirty or
unidentified imported checkout. `--provenance` records the development hash,
commit, clean status, Python, SQLite and extractor versions. Paths are local
inputs and do not become document identifiers or report provenance.

Consumers register adapters with
`saqqara.kernels.explorer.register_explorer_kernels()`. The manifest supplies
serialized `DocumentDigest` records under
`stage1.explorer_census.options.documents`; stages 2 and 3 enable
`explorer_profile`, `explorer_read`, and `explorer_report`. See the runnable
manifest test in `tests/saqqara/test_explorer.py`.

The optional classification job takes an explicit `StructuredPort` and existing
`LLMCache`. `OllamaStructuredPort` supports JSON-schema generation at temperature
zero. Model identity, thinking configuration, packet/output hashes, validity,
cache use and latency are recorded. Oversized packets and incomplete responses
are refused. Classification annotations never overwrite a deterministic profile;
explicit amendments go through `apply_reviews`.

## Reproducible validation

```bash
python -m pytest tests/ -q -p no:cacheprovider
python tools/explorer_gate.py --output /tmp/explorer-gate.json
```

The gate tool uses only generated fixtures. Its output contains per-category
census counts, profile and report digests, imported-code provenance and a
PDF round-trip result. Use the same committed source, Python and extractor
versions on both architectures. The optional model benchmark runs only against
an explicitly selected local endpoint and uses a generated, sealed packet.

## Limits made explicit

- Language inference is a small French/English function-word baseline; other or
  insufficient language evidence stays UNKNOWN.
- Deterministic range semantics cover known generic connector words; unfamiliar
  connectors remain observations and carry reader uncertainty.
- A repeated geometric mark is not automatically a lifecycle assertion. Census
  geometry thresholds are carried in the profile and reused by the reader.
  The continuation gap is an explicit configurable bound, not a learned layout
  guarantee; an unfamiliar layout still needs review.
- No OCR is attempted. Textless pages remain visible in coverage and findings.
- A numeric prior is not inferred from an ambiguous thousands/decimal spelling.
  Tokens resolve locally first. A prior never removes grouping or uncertainty
  flags, and mixed notations never disable the document's quantity reader.
- Table observations must preserve cells and their source locations. Duplicate
  headers, ambiguous id columns and unreadable cells are surfaced, never repaired.
- Passing synthetic tests establishes those fixtures' behavior. A consumer still
  owns domain interpretation, reference resolution, review and its own acceptance.

## Slice 2 schema and policy migration

`document-profile/0.2` and `census/0.2` replace the released `0.1` records. The
original work-plan shorthand called the baseline v0, but reusing its already
published `0.1` identifier would conceal a schema change. Old serialized profiles
and censuses must be rebuilt; no silent conversion is performed. Serialized page
digests without `horizontal_rules` must be re-extracted: missing observations
must not be interpreted as proof that no horizontal rule exists. Token-first
quantity candidates use producer `quantitative/1.1`; the legacy harvest mode and
its producer remain unchanged.

The census now stores `windows` and one `continuation_policy`. The profile carries
that same policy at its top level; the old `reference_fields.value.max_gap_ratio`
location is retired. Each window stores its line views and character mapping,
value position, structural/guard stop, flags and `needs_review`. Readers assemble
that exact object; they do not rescan with independent geometry constants.

The inter-line gap histogram contains ratios to the preceding line's height,
rounded to three decimals. A derived valley needs two or more observations in
each mode and an empty interval wider than either mode's spread. Its midpoint is
the gap bound. Insufficient or overlapping modes use the fallback. The policy
records `source` (`derived`, `default`, `amended`) and the histogram. The default
gap ratio is 2.5 and the cap is eight following nonempty lines. An explicit
amendment bypasses derivation; changing policy requires a new census. A gap/cap
closure always sets `WINDOW_BOUND_HIT` and `needs_review`.

Physical locale evidence has an attached unit, comparator or percent sign, or
comes from a non-identifier table column. Structural numbering, date, revision,
identifier and page spans are excluded. Numbering does not claim an unmarked
physical decimal merely because it contains a dot. Only unambiguous token-local
readings vote for the document prior; repeated ambiguous tokens cannot vote
themselves into certainty. The field records separator counts, `n`, dominance
threshold, minimum count, prior strength, confidence and ambiguity diagnostics.

Default prior dominance is 90 percent, with at least two unambiguous
observations giving a weak prior (confidence at most 0.5); five give a dominant
prior. Unambiguous tokens retain their own normalization even when they contradict
the prior, and receive `SEPARATOR_AMBIGUOUS` individually. Ambiguous tokens without
a usable prior retain exact literals/offsets and withhold normalization. A
prior-dependent reading remains flagged (`LOCALE_PRIOR_USED`, with
`LOCALE_PRIOR_WEAK` where applicable) and reviewable. This replaces the old
whole-document UNKNOWN/zero-candidate behavior.

Use `CensusConfig(continuation_policy=ContinuationPolicy(...))` and
`ProfileConfig(locale_dominance_ratio=..., locale_minimum_n=...)` through the typed
stages or `explore(..., census_config=..., profile_config=...)`. Kernel manifests
accept `census_options` on `explorer_census` and `profile_options` on
`explorer_profile`; readers cannot override a sealed window. Full acceptance
runs `tests/` in a detached worktree with `PYTHONPATH` pinned to it, not merely the
Harvest/Saqqara subsets. Consumer confirmation remains a separate gate.

## Construct-failure boundary (E7.8 / X21)

Empty label observations are retained as counted EMPTY_LABEL findings and never
instantiated as value windows. A terminal label cut off by a rule is recorded as
INVALID_LABEL_WINDOW. Reading coverage counts these failures independently from
unknown-template fields. Census schema `0.3` adds the construct-finding ledger.

The library and each kernel catch expected construct errors (ValueError,
TypeError, KeyError) at document boundaries. They produce an explicit
`reading-failure/0.1` report, with source scope, stage, count and error type;
exception text is not copied into exports. Other documents are still attempted.
A unique known page is retained; an unknown location is not guessed. Programming
errors outside that bounded set still propagate. Failed inputs never become
successful evidence or silently repaired profiles.

## Occurrence-level reference decisions (E7.9)

`reference_fraction` is removed. A label is proposed as a reference field when
its identifier-bearing occurrences meet `minimum_occurrences` and form a strict
plurality among non-empty classes. A tied plurality remains UNKNOWN. Empty
windows never vote against a label. Per-label positives, non-empty negatives,
empty count, class counts, minimum support and observed confidence are recorded
in `reference_fields.diagnostics.reference_counts`. Confidence is positives divided
by positives plus non-empty negatives. This diagnostic confidence is retained
also for UNKNOWN reference classifications; it is not an acceptance verdict.

`unresolved_occurrences` lists every empty window by page, source span and window
id, with WINDOW_BOUND_HIT, EMPTY_CELL, RULE_STOP or NO_TEXT. Every occurrence
produces either a field or a field-level finding. Identifier-bearing occurrences
of an unselected label remain visible as UNDECIDABLE with REFERENCE_CLASS_UNKNOWN,
rather than disappearing. Non-identifier values receive their own finding.

A label contained by actual grid rules uses its cell, then the adjacent cell to
its right and the cell below. These cell boundaries determine membership; a row
rule crossed into an adjacent value cell does not behave like an unruled-text
stop. The safety cap remains explicit and reviewable. Default gap policies record
why derivation was unavailable: no_body_lines, too_few_gaps or unimodal. Histograms
use body lines, and the diagnostic is retained in profiles and reports.

The revised records use `census/0.4`, `document-profile/0.3` and
`value-window/0.2`; old stored windows/profiles must be rebuilt. These version
increments precede the independent privacy/table changes because the added
priority fixes can be delivered separately.
## Slice 3 privacy boundary (E10)

Privacy records advance census to `0.5` and profile to `0.4`, after the independently delivered priority fixes. `furniture.stamp_lines` contains exact
suspected identity sub-spans, source page/coordinates and evidence ids;
`stamp_count` includes an explicit measured zero rather than UNKNOWN. This is a
shape detector, not a name recognizer: it combines date/time shapes with
capitalized sequences, email or user-id shapes. Body lines require a date;
page-edge or recurring stamp lines can use a time. No name list ships in RAGIX.

`render_report`, `render_page_view` and `render_report_json` mask by default.
The page view is extracted text with retained coordinates, not a masked PDF
raster. Original PDF/image viewers remain the consumer's responsibility. The
CLI now writes masked report JSON, report HTML and extracted-page HTML by default.
`--observations` additionally writes an explicitly named `.observations.json`
backend artifact containing original evidence. It is not a presentation export.
The corresponding CLI/library parity test uses this explicit backend option.

To unmask a presentation, pass `MaskPolicy(unmask=True, reason="...")`. An empty
reason or a loose dictionary is refused. Policy digests, reason and masked-line
count appear in presentation provenance. Additional patterns/literals can be
supplied by the consumer; those private policy contents are not printed. Masking
is applied to a presentation copy and never changes census/profile observations.
The guard uses original census flags, so an amended profile cannot silently
unmask a source line.

Recurrence share is recorded once in census geometry policy. A profile can inherit
it, or `explore` can propagate an explicit profile configuration upstream before
census. A profile built against a differently configured census is refused.
A zero stamp count means zero detected stamps in the extracted text, not proof
that an image-only page contains no personal information.

## Slice 3 physical table recovery (E9)

Raw table observations may supply `cell_rows`, containing a header row and following
physical rows of `TableCell` records (text, box, cell id, source span ids and flags).
PDF intake now supplies these observations. Existing explicitly declared logical
header/row inputs without `cell_rows` retain their original reader path; they are
not treated as word-cell candidates with invented geometry.

Global recurring edge bands and rotated oversized observations are evaluated before
raw table candidacy. All remaining blocks are considered. Complete vertical rules
(including unions of contiguous collinear strokes) define columns when available; otherwise consistent following-row x bands do. The
x tolerance is a declared factor of median observed cell width, with an explicit
fallback and provenance. Every band requires `minimum_rows` supporting rows across
an unambiguous adjacent-page continuation group, after grouping. A single-row
fragment supplies provisional geometry and identifier-column position, not final
acceptance. Inconsistent counts, straddling cells and unsupported header-only bands
are refused. Positive gaps between rule strokes are never bridged.
The header must recur in another block/page and an identifier-like column must be
observed. Roles are generic id/free_text/short_code/empty/unknown proposals.

Accepted rows retain their original cell-member ids, bounding boxes and flags.
Repeated header fragments on adjacent pages are grouped only when normalized
headers, identifier-column positions and relative geometry agree, with exactly one
matching fragment on each neighboring page. Missing pages and ambiguous neighbors
cannot supply pooled support. Final roles are computed from the pooled cells. `continued_on` and `repetition_count`
are recorded; header repetitions never become additional data rows. This is an
explicit structural grouping rule, not a cross-document identity declaration.
A missing or ambiguous primary id does not erase the row. Duplicate header texts
use stable column ids in the presentation mapping so no cell overwrites another.
Lifecycle, unreadable and digit-join flags remain visible; no glyph is cleaned.

`CensusConfig(table_policy=TablePolicy(...))` controls x tolerance and row support.
The kernel accepts the same structure under `census_options.table_policy`.
`table_analysis` stores accepted tables, unresolved findings, excluded furniture
blocks and the actual candidate scope. Coverage distinguishes physical fragments,
logical continued tables, unresolved blocks and excluded furniture. A zero is
always relative to observed candidates; it does not certify an arbitrary layout.
Census/profile schemas now use `0.6`/`0.5`; rebuild earlier records. The separate
version increments preserve the independently deliverable priority/privacy fixes.
The short-fragment correction uses kernel version `0.5.1` without changing the
record schemas; recompute results cached by `0.5.0`.

The synthetic gates cover 3/5/8 columns, FR/EN, ruled/unruled layouts, wrapped and
empty cells, contamination, 40-row mappings and three-page continuation. Additional
controls cover one-row fragments across six pages, mixed-length fragments across
nine pages, segmented grids and native PDF cell-rectangle borders. Sensitivity
sweeps x factors 0.25/0.5/1.0, row support 2/3/4/5 and recurrence 0.4/0.5/0.6.
Digit-only document-family discovery and classifier calibration remain outside
this slice. Consumer confirmation against its sealed inventories remains required.


### Native cells, header collapse and refusal diagnostics

Native PDF cell rectangles use `TableCell.geometry_kind = "cell_box"`; word/glyph
boxes retain the default `text_box`. An extractor slot with no geometry is kept
in the raw matrix, but never becomes a physical rectangle covering the table.
A nonempty unlocated value is flagged and the table is refused. A genuine cell
with unreadable text keeps its geometry, null text and UNREADABLE_CELL flag.

For native cells, the order is explicit: **header collapse, continuation grouping,
then support checks**. Header and body rows define observed interval partitions.
Their shared boundaries collapse padding/subcells, with no cut through an
observed cell and no merger of distinct nonempty header labels. Gaps, overlapping
rectangles, conflicting labels and unsupported headers remain unresolved. No
column count is fixed. Raw matrices and cell identities remain unchanged; row
members record every physical subcell contributing to each logical column.

Native cell geometry survives overlapping lifecycle glyphs: those cells remain
flagged, never cleaned. Entire blocks whose content is classified as furniture
are excluded before candidacy. Text-box inputs retain their recurrence filter.

Every table refusal carries `diagnostics`: the exact stage and geometry route,
raw/retained/nonempty/unreadable cell counts per row, unlocated slot count,
full-height rule count, observed row-band counts, inferred bands and populated
band count, mapped support, pooled rows and pages spanned. Unsupported counts are
null rather than invented. These records are in `census.table_analysis.findings`
and the profile's `id_row_tables.unresolved` list, including the report profile.

These are additive records; old inputs still load, but must be re-extracted from
PDF to acquire native-cell geometry. Kernels use version **0.5.2**, invalidating
older cached outcomes. Re-running old intake JSON cannot recover geometry that
the previous adapter replaced with a table-sized placeholder.

## Slice 4 reference fields

Painted rules are read as edges within a declared tolerance, so per-cell borders no longer
hide a label's cell, and a border a hair inside the label's last glyph still separates it
from the next cell; an underline is emphasis and never an edge; overlapping line boxes are
not one row; a range stays one relation whether or not its numbers carry a marker. A label
must introduce a value: a phrase that holds an identifier is reference content, with or
without a colon after it, and is never a label; label words in prose are counted, never
read; a label's occurrence votes for it only when its value opens with reference content.
Role-word lines are listed on the window and never read. Labels, type words and role words
are policy data (`ReferencePolicy`, through `CensusConfig`): the shipped words are
language-generic, a consumer declares its own labels and acronyms. Gates, swept ranges and
replay identities are in `EXPLORER_SLICE4_FIELDS_VALIDATION.md`.

The revised records use `census/0.7`, `value-window/0.3` and `document-profile/0.6`; stored
records of earlier versions are refused and must be rebuilt. Kernel cache versions must move
with these records, or cached outcomes of the earlier reader are served.
