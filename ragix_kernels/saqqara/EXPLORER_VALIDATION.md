# Explorer v0 — synthetic validation, 2026-09-16

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

The deterministic Explorer chain runs through the typed library, CLI, four KOAS
adapters and an Orchestrator manifest. The synthetic replay digests agree on
x86_64 and aarch64. Optional model classification is schema-safe on the measured
packet but semantically unreliable; it remains disabled by default.

Implementation source: `dcc1399f84445cd76c62735048ee2035c62e038e`.
Final regression additions: `cddd455`. Branch: `feat/document-explorer`.
No merge or public push is part of this validation.

## Measured gates

| Boundary | Evidence | Result and limit |
|---|---|---|
| E0 prerequisites | `tests/harvest/test_explorer_prerequisites.py`, existing quantity/binding/field-view suites | Interval, unit, grouping, label-number, baseline-spread and nested numeric-smuggling controls pass. Scientific alphanumeric names remain admissible. |
| E1 census | `test_census_all_planted_category_multisets`, exact-span and textless-page controls | Every planted occurrence and category multiset matches on the generated fixture: no missed or extra planted patterns. French and English controls pass. This is fixture precision/recall, not a population estimate. |
| E2 profile | Mutation, evidence, schema and append-only-review tests | X1–X6 change the corresponding semantic profile values. Unsupported notation stays UNKNOWN. Evidence and template hashes may change. |
| E3 classification | Generated two-page packet, four candidate ids; results below | Model-off parity, id-only output, exhaustive accounting, refusal caching and schema checks pass. No semantic accuracy threshold was imposed. |
| E4 readers | Mutation, range, revision, forged/stale-source and nondefault-geometry tests | Planted fields retain both targets and all section members. Ranges remain unexpanded relations. Unknown required notation produces a counted finding. |
| E5 report | Negative-record schema, HTML injection and source-column regressions | Missing scope/rule/positive count is refused. Numbers belong to addressable records; source strings are escaped. Source columns named like run metadata remain intact. |
| E6 adapters/replay | CLI/library equality, kernel chain, Orchestrator/cache test with socket guard; cross-architecture gate tool | Same semantic report; no deterministic-chain network call. Eight generated cases, including a PDF round trip, match across architectures. |

Full command on each host:

```bash
python -m pytest tests/harvest tests/saqqara -q -p no:cacheprovider
```

- x86_64: **1,200 passed, 32 skipped**, 17.27 seconds.
- aarch64: **1,199 passed, 33 skipped**, 12.01 seconds.
- The additional ARM skip is the local pre-commit-hook installation check: the
  isolated clone has no installed hook. The guard's content, planted-violation
  and history tests still run. No hook was bypassed.
- History scan through `cddd455`: 1,180 blob revisions and six commit messages,
  zero findings. All commits retain sole human authorship.

These are observed single-run durations, not a comparative performance claim.

## Replay identity

`tools/explorer_gate.py --require-clean` checks the actual imported checkout and
platform lock. Both machines used the declared Python and deterministic PDF
extractor versions. SQLite versions differ as recorded in the platform locks;
SQLite does not participate in this chain's semantic digest.

| Record | SHA-256 |
|---|---|
| Imported package source, both architectures | `ca2731c3cb9fc5a1d1395bbfebd330cb6a010153d10216b0483c8ea1b44aa628` |
| Baseline synthetic report, both architectures | `23b2e112f62fb89868ee8d953c862650727c4874e03d1de808b61818e9473337` |
| Generated-PDF report, both architectures | `a3d772c74b7eb24207bf70204500931cf92ff31d9a0c5f5653da709d586be5de` |

The other equal cases mutate the label, marker, locale, column order, identifier
family and furniture. Hashing includes character mapping boxes at millipoint
precision. It excludes run telemetry only in metadata contexts; a source cell
called `timestamp` is evidence and remains hashed.

## Optional local-model baseline

One sealed, generated document supplies four unique census candidate ids
(identifier, label, page observation, body recurrence). Each model receives the
same packet and closed schema at temperature zero, with a bounded output budget.
The expected labels were planted before generation. There is one call per
configuration; latency includes loading and warm-state effects.

| Model | Thinking requested | Schema/accounting | Correct / candidate ids | Latency (s) |
|---|---|---|---:|---:|
| Qwen3 14B | off | valid | 1 / 4 | 17.04 |
| Qwen3 14B | on | valid | 1 / 4 | 13.83 |
| Qwen3 32B | off | valid | 3 / 4 | 50.86 |
| Qwen3 32B | on | valid | 3 / 4 | 30.37 |

The 14B model labels only the reference-field candidate correctly; identifier,
page and recurrence decisions are wrong. The 32B model additionally abstains
correctly on page and recurrence, but mislabels the identifier. Thus `accepted`
in the job manifest means **schema and accounting accepted**, never semantically
confirmed. No classification is applied to the deterministic profile. Thinking
shows no accuracy improvement on this packet; the timing sample cannot establish
a speed effect. Four labels are far too few for a general accuracy claim.

Model digests:

- Qwen3 14B: `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`.
- Qwen3 32B: `030ee887880fc378860c2dd35101da424377520441ae4bfe7be6deff8ade7840`.

An earlier measurement of Qwen3 30B-A3B returned HTTP 500 in both thinking modes.
It is a recorded server failure, not a semantic score, and was not repaired by
changing the kernel or silently substituting a response. Its digest was
`ad815644918f0eaab341c12b67837cc6dd4562342cdaf118f83d5d554cb37226`.

Reproduce with an explicitly selected local endpoint and a separate cache:

```bash
python tools/explorer_gate.py --require-clean --output /tmp/explorer-gate.json
python tools/explorer_gate.py --require-clean \
  --endpoint "$EXPLORER_LOCAL_ENDPOINT" --models qwen3:14b qwen3:32b \
  --cache /tmp/explorer-model-cache --output /tmp/explorer-model-gate.json
```

## Refutations retained as regressions

- Copying adjacent identifier context into the profile coupled label and marker
  mutations to the family field. The context stays in the census; profile values
  retain only the family shape and revision-marker style.
- Dotted section keys are not evidence for decimal locale. Locale derives from
  physical-number observations, and ambiguous notation remains unknown.
- A recurring header followed much later by an identifier is not a field label.
  The adjacency rule now requires compatible local geometry.
- A global volatile-key filter erased legitimate source columns. Exclusions now
  apply only in metadata contexts.
- Geometry defaults cannot be duplicated independently in census and reader.
  They are recorded in the profile and exercised by a nondefault configuration.

## Acceptance boundary

This record establishes the generated fixtures' behavior and a local-model smoke
baseline. It does not establish recall on arbitrary documents, correctness of
model interpretations, a complete OCR route, or a consumer's domain acceptance.
Continuation distance is an explicit configurable bound. Unknown connectors,
unsupported language/locale and ambiguous layouts still require review.
Consumer verification against its own inventories and its own policy remains a
separate acceptance step. See `EXPLORER.md` for the library and adapter entrypoints.

## Slice 2 — shared value windows and token-local notation, 2026-09-16

Implementation: `5ec3aa27e31add79f3e963a4a1177d15b984ad39`, branch
`feat/explorer-slice2`, based on merged `main` at `d320077`.
The committed implementation tree is
`4c1a9a31c2712b53a277bb04e33ffa9e56269732`.
This addendum preserves the earlier measurements; the changed schema and parser
produce new replay identities rather than pretending to replay old semantics.

### Measured synthetic gates

| Gate | Measurement | Result and boundary |
|---|---|---|
| E7.1 | Same line, next line, adjacent ruled cell, two line positions below, beyond bound; French and English labels | All ten controls pass. The first four positions retain the identifier-bearing value; the last stays empty and explicitly guard-limited. |
| E7.2 / X16 | Move the value onto the next line | Reference-field value and confidence, targets and section members are unchanged. Position/evidence and policy provenance remain distinct. |
| E7.3 | Inspect reader consumption and serialized replay | The reader consumes census window objects; every window shares the census policy. The reader has no independent gap/cap constants. |
| E7.4 | Role label, known table header/caption, numbered heading, horizontal rule, page and column stops; distant-header and cap controls | Structural stops remain outside the window. Guard closures retain WINDOW_BOUND_HIT and needs_review, including after serialization; a flagged partial field cannot become ready. |
| E8.1 / X17 | Dotted keys, dates, revisions, identifiers and page markers alongside physical comma/dot decimals | Every planted exclusion is absent from prior evidence; counts and prior strength match the physical observations. Comparator-only numbers and non-identifier cells are also exercised. |
| E8.2 | Mixed clear tokens, unresolved three-digit tails, weak prior and one prior contradiction | Detection continues. Clear tokens retain local normalization; unresolved tokens retain literals without normalized values. Prior-dependent or contradicting tokens carry review flags individually. |
| E8.3–E8.4 | Existing X3 plus exact offsets, direction and composite-member checks | Separator changes remain local to numeric profile values; literal spans survive; cell deduplication does not orphan composite members. |
| E7.7 / E8.7 | Full Cartesian sweep described below | All 108 configurations yield identical accepted field semantics and quantity literals, offsets, values and flags on the planted fixture. |
| E7.5 / E8.5 | Full committed-tree suite and replay on both architectures | Green, with measurements below. Existing public X1–X12 controls are retained. X13–X15 and X18–X20 are not introduced or claimed as implemented by this slice. |
| E7.6 / E8.6 | Consumer inventory confirmation | Not run here. Consumer acceptance still requires its sealed field/target/member and quantitative-mention comparisons, with every difference reported. |

### Threshold sensitivity and derivation

The sweep uses actual fallback policies, not hidden reader overrides:

- gap ratios: **1.5, 2.5, 4**;
- following-nonempty-line caps: **6, 8, 16**;
- prior dominance ratios: **0.8, 0.9, 1.0**;
- prior minimum observations: **2, 3, 4, 5**.

These are 108 combinations spanning the required ranges. The fixture has
structural label stops and locally interpretable physical numbers, so guard
variation cannot choose its layout or its numerical meaning. Identity hashes and
recorded configuration are expected to differ; recovered labels, target/member
semantics, literal offsets, normalized values and review flags are compared.
Separate adversarial controls deliberately exhaust bounds and require an explicit
review state, rather than treating a partial window as a successful full reading.

The bimodal-histogram control supplies gap ratios 0.6 (three observations) and
5.4 (two observations); the derived valley midpoint is 3. The profile records
`derived` and the histogram. A unimodal control records `default`; an explicit
amendment retains its supplied bound and records `amended`. Histogram interval
comparisons use the declared rounding precision, avoiding floating-point tie
artifacts. No constant or fixture was selected from a consumer corpus.

### Full-suite and architecture measurements

The staged implementation was applied to a detached worktree. Both staged tree
ids were checked equal before committing. `PYTHONPATH` was pinned to that
worktree, and both imported package roots were asserted to belong to it.
The test run checked the staged tree identity again on completion. The resulting
commit has exactly that tested tree. ARM validation used a separate detached
checkout of the same commit, with its own pinned `PYTHONPATH`.

| Platform | Full suite | Duration |
|---|---|---:|
| x86_64 | **2,088 passed, 38 skipped**, 139 warnings | 34.88 s |
| aarch64 | **2,088 passed, 38 skipped**, 140 warnings | 23.91 s |

These are committed-tree counts. They exclude ignored local-only tests from a
development checkout. The detached checkouts also skip the local hook-installation
check; content and history guards remain active. No ignored local tests or prior
runtime checkout was changed. Warnings are reported, not counted as failures;
the durations are single-run observations, not a performance comparison.

Example validation invocation from a detached worktree:

```bash
EXPLORER_CHECKOUT=$(git rev-parse --show-toplevel)
conda run --no-capture-output -n ragix-env env PYTHONPATH="$EXPLORER_CHECKOUT" \
  python -m pytest tests/ -q -p no:cacheprovider --disable-warnings
conda run --no-capture-output -n ragix-env env PYTHONPATH="$EXPLORER_CHECKOUT" \
  python tools/explorer_gate.py --require-clean --output /tmp/explorer-slice2.json
```

All **18** replay cases match between architectures: the original seven typed
mutations and PDF round trip, plus five value-window positions and five numeric
notation/prior cases. Platform locks are unchanged. There was no live model work.

| Record, identical on both architectures | SHA-256 |
|---|---|
| Imported package source | `a8f25668ccac6bdaa596c70e04e761e18195b9284ae501ce2ece508e15d6af56` |
| Baseline report | `17a673d2716d3a3b0252cd540863d451eb971d6f69d9d138cb8b27f2ca047e32` |
| Generated-PDF report | `f3850dcde1ef38d23875260f0069f4e5d741ddd72ef1c8e8bd016ca91753052b` |
| Next-line value report | `a8436c0849f92eaf0479ebd56a2601dfde66e4119b672f43f8130df4b87315aa` |
| Mixed local notation report | `5e66dc96c2e4879ec8721de95b33213939854fc2f37db2e6bf1b56e5b9a31ed6` |
| Individual prior-contradiction report | `364f65484f2267431f2a612f32802d785cf5cad4ef535098824471ea27748369` |

### Intentional contract changes and retained refutations

The old whole-document UNKNOWN assertion is intentionally superseded by the
lead's token-first rule. A field may record observed numeric evidence with no
usable document prior. Its reader still emits candidates; a genuinely ambiguous
literal cannot acquire a value merely by being repeated in the document.
`census/0.2`, `document-profile/0.2`, and opt-in `quantitative/1.1` distinguish this
behavior. Legacy harvest calls retain their original producer and behavior.
Old serialized digests without horizontal-rule observations are refused and must
be re-extracted; missing observations cannot establish the absence of a rule.

The negative tests additionally preserve these corrections: a known table header
must not become a plain label; dropping duplicate scalar children must not orphan
a table composite; a unitless comparator with unresolved direction stays
unresolved; a serialized guard stop cannot remove its review flag; horizontal
rule coordinates participate in geometry-normalized replay.

Synthetic success does not close consumer acceptance or establish recall on
unseen layouts. The adjacent-cell positive control has an observed ruled boundary;
unproven cross-column associations remain refused. The consumer confirmation is
a measurement step, not an opportunity to tune these defaults to its documents.

## Slice 3 — priority repairs, privacy, then tables (2026-09-16)

The priority order was revised before delivery: construct failure handling (E7.8),
occurrence-level references (E7.9), privacy (E10), then table recovery (E9).
Each implementation was committed separately after a full suite in a detached
worktree with pinned imports and a printed package path. No public push or merge
is asserted by this record.

| Commit | Scope | Detached master suite |
|---|---|---|
| `5630225` | Empty labels and document-scoped construct failures | 2,096 passed, 38 skipped |
| `3d310b6` | Plurality, per-occurrence findings, cell windows, fallback reasons | 2,107 passed, 38 skipped |
| `e3efbef` | Invalidate pre-plurality kernel cache entries | 2,108 passed, 38 skipped |
| `ea1edaf` | Default-masked presentations and immutable source observations | 2,141 passed, 38 skipped |
| `5bd77c6` | Recurrence-first physical table reconstruction | 2,200 passed, 38 skipped |

The staged tree and detached validation tree were checked equal before each
commit. Existing runtimes, ignored local tests and stored evidence were not
purged. Cache versions changed to invalidate old semantics without deleting
cache entries. Synthetic E9 gates were green before the declared 17:00 local
cutoff; E9 was not deferred or compressed to meet it.

### E7.8 and E7.9 adjudication

E7.8's zero-length/whitespace label and terminal-rule controls complete with a
counted finding. Empty labels never become window candidates. Injected construct
failures at census, profile, read and report stages produce an explicit failed
report while the following document is still processed. Exception text is not
copied into presentations, and an unknown page is not invented.

E7.9's requested synthetic fixture has 21 occurrences: **13 fields and 8 scoped
findings**, with 12 next-cell values and one same-line value. Empty windows are
not votes against the label. Its counts are positives 13, non-empty negatives 0,
empty 8; confidence is 1. The eight findings split equally among WINDOW_BOUND_HIT,
EMPTY_CELL, RULE_STOP and NO_TEXT. The reference field remains PROBED.
Minimum support 2, 3, 4 and 5 gives the same result. `reference_fraction` no longer
exists.

A competing free-text fixture (two identifier-bearing values, six free-text
values) stays UNKNOWN with confidence 0.25 and its counts. Its two candidates
remain visible as UNDECIDABLE, rather than disappearing. A tied plurality also
remains UNKNOWN. A plurality need not be a majority: the 3/2/2 class control is
PROBED with confidence 3/7.

Both right-hand and below-label grid value cells are recovered using their own
rules. Unruled text retains horizontal-rule stops. Default derivation records
no_body_lines, too_few_gaps or unimodal; the diagnostic appears in the profile and
report. The bimodal control retains its independently derived bound.

Priority replay at `3d310b6` matched in all five cases on x86_64 and aarch64;
ARM's full suite also measured 2,107 passed / 38 skipped. Imported package hash:
`cedfe266cb2d72ea35377ade96942658b7cadc15ce80ccbc24c5f583c814b7dc`.
The subsequent cache correction changes cache identity, not those reading rules.

### E10 adjudication

The source-shape suite covers six date notations, French/English name-like
sequences, time, email and user-id shapes, and decomposed Unicode. All planted
positives are flagged with exact sub-spans. All three declared negative controls
are unflagged: a date in an ordinary body sentence, a capitalized header without
a date, and a numbered heading. These fixture results are not a population-wide
PII precision/recall claim.

The six-page privacy fixture reports **six masked lines**. No planted name token
appears in default report HTML, presentation JSON or any extracted page view.
CLI default exports pass the same grep control. Unmasking requires a typed policy
and a nonempty reason; the reason and digest are recorded. A substituted mask
token cannot bypass this requirement. Extra private policy contents are not
printed. Profile amendments cannot erase upstream census flags.

Source/census/profile digests remain unchanged when masking is switched. The
explicit observation export preserves original Unicode and geometry. Original
PDF rasters are not produced by the extracted-text page view; a consumer rendering
them must enforce its separate masking gate before showing a report.

At `ea1edaf`, master and ARM both pass **2,141 tests with 38 skips**. Both agree on
source package hash
`1ef61c9fda8c8c5b39b93469b417a811b0f16fb505cd4a158e400140e7a5f406`
and masked presentation-set digest
`1dc7abb9261ff940ac6d567f1e689e8edc40f5ad0200fc8366e75055ec0cfc8f`.

### E9 adjudication and sensitivity

The 12 base layouts span French/English, ruled/unruled geometry and 3/5/8 logical
columns, each header supplied as three word-cells per column. Every layout
recovers the planted **40 rows**, including wrapped text and explicitly empty
cells. Every planted cell maps to its expected column. Recurring header blocks
are excluded before candidacy; body tables remain. Repeating a header on three
adjacent pages produces one continued table, `continued_on = 3`, with no additional
data row introduced by the repeat.

Straddling cells, inconsistent row-band counts and unsupported header-only bands
produce TABLE_UNRESOLVED findings. Rows remain in original observations. Header
and row contamination remain present and flagged, never cleaned. Duplicate header
literals cannot overwrite cells; such outputs use stable column ids. A column
changing from observed empty values to codes does not split an otherwise valid
continuation. Identifier-column positions and relative geometry still constrain
the grouping.

All **36 sensitivity combinations** pass:

- x-band tolerance factors: **0.25, 0.5, 1.0** times median observed cell width;
- supporting rows: **2, 3, 4, 5**;
- recurrence shares: **0.4, 0.5, 0.6**.

Tables and row mappings stay identical; the applied configuration and derived
width/tolerance remain recorded. Column reordering and furniture mutation controls
pass. A real generated-PDF intake control recovers a continued table across two
pages with 12 rows and two masked stamp lines. No column count, name list or
threshold was selected from consumer documents.

The full integration suite caught an edit error: an extra positional argument
had entered the existing geometry-census emission, causing affected documents to
produce failures. The call was corrected; no expected outcome or gate was
relaxed. This is why the full detached suite remains required after the focused
new-feature tests.

### Final source, replay and acceptance boundary

At implementation commit `5bd77c60a5d6e92d9a91cf29ddf040eb03bb1a96`:

| Platform | Full committed-tree suite | Observed duration |
|---|---|---:|
| x86_64 | **2,200 passed, 38 skipped**, 139 warnings | 40.66 s |
| aarch64 | **2,200 passed, 38 skipped**, 140 warnings | 29.27 s |

All **18 final replay cases** match across architectures: priority occurrence and
failure cases, the privacy fixture, and the twelve table layouts. The privacy
presentation hashes also match, with six masked source lines. Durations are
single-run observations, not comparative performance claims.

| Final record | SHA-256, identical on both architectures |
|---|---|
| Imported package source | `d6e92fcb438296cd82121c8f6b415ec978117a96854af4910d7d37f2df33b32f` |
| Occurrence-fixture report | `38b62db80268db7c9602ba79a55c9a07c1fc31f8148050718479b06127b2fdbd` |
| Privacy-fixture report | `3f754529bc29594ea28045121bc1d6f7c7eada633ff4854d3ae89d56c1d43119` |
| Ruled five-column English report | `775d06bc36494e193de12ef14ba8793107e7e4e2328721d26703d13c70f00d3a` |
| Unruled five-column French report | `fef9115ed0dbb95e66004bee7ae610dfd0813d74917907ef27a500e10148e38c` |
| Masked presentation set | `4c98d187a99e3dd0612566f6ddec0768468306cd34bfcedbd7dfed3687ba23c2` |

Reproduce from a clean detached checkout with its platform lock:

```bash
EXPLORER_CHECKOUT=$(git rev-parse --show-toplevel)
conda run --no-capture-output -n ragix-env env PYTHONPATH="$EXPLORER_CHECKOUT" \
  python -m pytest tests/ -q -p no:cacheprovider --disable-warnings
conda run --no-capture-output -n ragix-env env PYTHONPATH="$EXPLORER_CHECKOUT" \
  python tools/explorer_slice3_gate.py --output /tmp/explorer-slice3.json
```

Consumers must rebuild old records for `census/0.6` and `document-profile/0.5`.
Accepted physical reconstructions are in `census.table_analysis.tables` and the
flat `reading.tables`; profile metadata is in `id_row_tables.reconstructed`.
The legacy `id_row_tables.tables` list describes explicitly declared logical
inputs and is not the new reconstruction result. Unknown/error outputs are not
acceptance verdicts.

Private confirmation of field/target/member inventories, table rows, and actual
stamp-name masking remains **pending with the consumer's verification owner**.
It is required for acceptance. The consumer's original-PDF display gate remains
separate and mandatory. Digit-only document-family discovery, label ranking and
classifier calibration/replay are outside this slice as ruled. No live model
work, source purge, bridge declaration or domain conclusion was performed.


### Slice 3 correction: supporting rows after continuation

The initial 40-row fixture gave every page enough rows for independent acceptance.
It therefore did not falsify the premature per-fragment supporting-row check.
Consumer confirmation withheld acceptance after exposing a recall regression;
the synthetic gates above do not establish consumer acceptance.

The new one-row-per-page control failed with TOO_FEW_ROWS before correction.
A second control using contiguous segmented borders and touching physical cells
failed with INSUFFICIENT_BANDS. Both are independently synthetic. Continuation now
precedes the row-support decision, constrained by repeated normalized headers,
aligned identifier-column positions, relative geometry and unique neighbors on
consecutive pages. No acceptance threshold is reduced. Grid recognition unions
only observed collinear segments, using the existing three-decimal geometry rule;
it never fills positive gaps. Final roles use pooled observations. Every source
row and cell member remains traceable. Kernels advance to 0.5.1 to invalidate
cached results; census/profile schemas remain 0.6/0.5.

Regression controls cover one-row fragments on six pages and mixed-length
fragments on nine pages, native PDF cell-rectangle borders, missing continuation
pages, changed identifier columns, ambiguous same-page neighbors and gaps between
rule strokes. The one-row fixture passes all 108 combinations: three geometry
modes (unruled, full strokes, segmented strokes), x factors 0.25/0.5/1.0,
supporting rows 2/3/4/5 and recurrence shares 0.4/0.5/0.6. The replay driver includes
six additional continuation cases. Private confirmation remains a separate merge
gate; open field-reader findings are outside this bounded correction.

At correction commit `ac2be3dfea9a192c5e185719215c1a102c8e8ca1`, full suites
run in detached checkouts with `PYTHONPATH` pinned and both imported package paths
printed:

| Platform | Full suite | Observed duration |
|---|---|---:|
| x86_64 | **2,316 passed, 38 skipped**, 140 warnings | 47.35 s |
| aarch64 | **2,316 passed, 38 skipped**, 140 warnings | 32.45 s |

Both declared platform locks match. All **24 replay cases** and masked
presentation hashes match across architectures. All 18 pre-correction cases have
unchanged census, profile, reading and report digests (the failure-case ordinal
keys shift by six because the new cases precede them in the driver). Privacy
presentation output also remains unchanged.

| Record | SHA-256, identical on both architectures |
|---|---|
| Imported package source | `f7d37367f6f1790f31dc2eaca95d01cb786b3d93d9e0a1f4cb6286ce774ada22` |
| Six one-row fragments, unruled report | `ebcbcfdabb12f5c12bac42f51e3135e581f94aac5f3188303b224c76557a4892` |
| Six one-row fragments, segmented-grid report | `3e5747bd5d15ca88516f7df8c59618a17a6c3d5e75a0d9eff977d70a3b30dc1a` |
| Nine mixed-length fragments, segmented-grid report | `7bb1f775fd3648a33b1ba59c0f64099c9c6a932f3267d895d25b7a35fff1d36c` |

The unchanged reproduction commands above run the expanded gate. Source history
checks for `8181b35..ac2be3d` inspect 1,178 blob revisions and one commit message
with zero findings; the normal pre-commit hook ran. These synthetic results do
not close the consumer gate or the separate open field-reader findings. No
private corpus processing, private source changes, push or merge was performed.


### Slice 3 correction: native geometry and header-first collapse

The preceding correction was insufficient in consumer confirmation. Reason-count
changes alone do not prove continuation: INSUFFICIENT_BANDS is emitted before
that stage. Count-only instrumentation preserves the original refusal outcomes
and locates failures without exposing source literals.

Three independent synthetic falsifiers exposed the remaining mechanisms:

- Native extractor slots lacking geometry were expanded to table-sized boxes.
  Such placeholders connected columns in text-box clustering. A generated PDF
  with padded headers reproduces this for 3/5/8 logical columns.
- Native rectangles that touch are separate physical cells, unlike adjoining
  glyph boxes. Common header/body partition boundaries collapse header padding
  and varying subcell partitions before continuation and supporting-row checks.
  The varying-partition control has 12/18 physical body subcells and five logical
  headers. Merging distinct header labels remains a refusal.
- A rotated furniture span overlapping a native empty cell caused its geometry
  to be deleted. The regression fixture now retains that cell and flags the
  overlap. A separate native furniture-block fixture confirms whole-block
  exclusion still precedes candidacy.

E9.5b controls cover 15/23/30 consecutive pages, with either one row per page or
one through five rows cycling, on both ruled and unruled inputs. Native collapse
passes the existing 36 sensitivity combinations (x factor 0.25/0.5/1.0,
support 2/3/4/5, recurrence share 0.4/0.5/0.6). These are declared synthetic
coordinates and settings; none was chosen from consumer documents.

E9.12 refusals include the exact stage, route, raw and retained cell counts,
observed and populated bands, mapped support, pooled rows and pages. Null counts
mean not inferred at that stage. Roundtrip, malformed-count rejection and loading
old records without diagnostics are tested. The replay driver adds three native
padding layouts and one varying-subcell layout. Kernels advance to 0.5.2;
additive record defaults preserve loading, but fresh PDF intake is mandatory for
old documents lacking cell-geometry identity. Consumer comparisons and all their
numbers remain private; independent confirmation is required before merge.
