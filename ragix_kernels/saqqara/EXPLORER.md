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

Focused development checks may use `tests/harvest tests/saqqara`. Acceptance
requires the full repository suite, including consumers of Saqqara's declared
surface:

```bash
python -m pytest tests/ -q -p no:cacheprovider
python tools/explorer_gate.py --require-clean --output /tmp/explorer-gate.json
```

When the declared kernel surface changes, inspect its consumers with
`rg -n 'DECLARED_KERNELS|FROZEN_COUNTS|pinned_lists' tests/` and update the
independent literal in
`tests/tender/test_t0_family.py::test_t0_7_saqqara_pinned_lists_are_untouched_by_this_family`
in the same change. Preserve the literal assertion and the frozen counts; do not
replace the cross-family guard with a comparison of a declaration to itself.
The independent guard caught the omitted update in PR #25; `f1f9954` corrected it.

The gate tool uses only generated fixtures. It checks synthetic replay and
imported-source cleanliness; it does not replace the full pytest suite. Its output contains per-category
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
- A numeric locale is not inferred from a single ambiguous thousands/decimal
  spelling. A locale declaration does not remove grouping review flags.
- Table observations must preserve cells and their source locations. Duplicate
  headers, ambiguous id columns and unreadable cells are surfaced, never repaired.
- Passing synthetic tests establishes those fixtures' behavior. A consumer still
  owns domain interpretation, reference resolution, review and its own acceptance.
