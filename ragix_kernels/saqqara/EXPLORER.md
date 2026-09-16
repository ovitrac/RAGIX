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
