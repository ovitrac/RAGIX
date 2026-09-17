# PDF watermark analysis copies

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

`tools/pdf_clean.py` removes explicitly selected, isolated watermark text from a
separate PDF copy. The original remains authoritative. The JSON audit preserves
the removed label, page, source instruction range, input/output hashes and checks.
This is a standalone utility; it does not replace the document ingestion source.

## Installation and use

From a checkout, install the optional dependencies in the chosen environment:

```bash
python -m pip install -e '.[pdf-clean]'
python tools/pdf_clean.py inspect input.pdf --text SAMPLE --plan plan.json
python tools/pdf_clean.py apply input.pdf --plan plan.json --all-candidates \
  --output analysis.pdf --report audit.json
```

Inspect `plan.json` before applying. Use `--select ID [ID ...]` to remove individual
occurrences instead. The marker is case-sensitive, with Unicode NFC and whitespace
normalization. The whole isolated region must match it; substring matches do not
qualify. Rotation or translucency is required, so ordinary opaque headings are
not selected. Existing files are never overwritten. All processing is local.

The optional extra pins pikepdf and PyMuPDF. PyMuPDF remains an explicit opt-in
under its AGPL/commercial license; installing this extra does not change that.

## Why not rectangular redaction?

A watermark may overlap body text, grid rules and images. Rectangle redaction can
remove those objects as well. This tool deletes only a balanced text-only graphics
block or a text-only Form invocation. Unselected PDF content token bytes remain
identical. Shared Form resources are retained; selecting one invocation leaves
other invocations intact.

Before publishing a copy, the tool checks every page for:

- identical retained glyph identities, positions, sizes, orientations and opacity;
- identical page geometry, painted drawing observations and image observations;
- identical unselected content-token bytes after saving and reopening;
- zero changed RGB pixels outside the selected glyph support at the configured DPI;
- preserved encryption settings and permission flags.

Glyph support comes from a disposable full-opacity render of the selected text.
This avoids low-alpha antialias quantization losing edge pixels. It adds no bounding
rectangle or mask dilation. Source glyph comparison uses the original opacity.
Default rendering is 72 DPI; `inspect --dpi 144` selects a stricter render sampling
for that plan. These checks bound a transformation; they are not an all-resolution
proof of PDF equivalence or a document-validity verdict.

## Refusals and limits

Unsupported structures remain untouched. A mixed text block, raster watermark,
unknown operator, Type 3 font, text clipping, inline image, malformed content state,
unsupported Form graph or budget excess cannot become a removal candidate.
Inventory reports page-level structural refusals separately from candidates.
Absence of a candidate is not evidence that a page has no watermark.

Signed/certified PDFs and documents needing an opening password are refused.
A PDF that opens with an empty user password can be processed; its encryption and
permissions are preserved. Such PDFs use randomized encryption, so output byte
hashes need not repeat. The audit records the actual hash. The tool does not ask
for, guess or recover passwords.

Plans bind to the original hash and dependency versions. Apply re-derives every
candidate and refuses unknown ids, stale input, output aliases or collisions.
Any validation failure publishes neither a PDF nor an audit. Successful output
files are written completely before publication; the audit appears before the PDF.
A process crash can leave an audit without its derivative, never a partially
written published PDF. Temporary files may remain after an abrupt process kill.

Default resource budgets are 128 MiB input, 2,000 pages, 250,000 tokens and 500
candidate regions per page, and 16 million pixels per rendered page. Python callers
may pass an explicit `Limits` value; the plan records it. There is no OCR or model.

This is not a security redaction tool: unused resource objects, hidden layers,
metadata or attachments can retain text. Lifecycle labels remain evidence in the
original and audit. Always distribute those together with an analysis copy when
its transformation needs to be reviewed.

## Reproducible checks

```bash
python -m pytest tests/tools/test_pdf_clean.py -q
python -m pytest tests/ -q
python -m black --check ragix_core/pdf_watermark.py tools/pdf_clean.py tests/tools/
```

Synthetic cases exercise overlapped rules and body text; shared Forms; split content
streams; accents; page selection; optional content and images; low-alpha edges;
72/144 DPI; stale plans; collisions; encryption; malformed state; unsupported
structures; and actual injected collateral changes to fonts, tokens, rules and
images. They contain no consumer document fragments.

The suite launches the optional-renderer cases in a separate process, preserving
the default reader tests that assert no AGPL renderer has been loaded. To see each
case directly, run `python -m pytest tests/tools/pdf_clean_cases.py -q`.
