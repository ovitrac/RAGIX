# Legacy Office input

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Saqqara accepts `.doc` and `.xls` when `libreoffice` or `soffice` is on `PATH`,
with Writer and Calc installed. It converts a temporary copy to DOCX or XLSX,
then uses the existing reader and analyzer pipeline. Native DOCX and XLSX
remain independent of LibreOffice. No Python dependency is added.

If the executable is absent, `read_path` raises `ConversionUnavailable` with
“Install LibreOffice with Writer and Calc to read legacy .doc and .xls files.”
Corpus and kernel runs record a counted `converter-unavailable` refusal.
A failed conversion is an `unreadable-file` refusal; an empty or partial result
is never substituted. Password-protected or corrupt files receive no password
prompt and must produce a valid readable derivative within the timeout.

```python
from pathlib import Path
from ragix_kernels.saqqara.adapters import read_path
from ragix_kernels.saqqara.assets import AssetStore

observations = read_path(
    Path("document.doc"),
    conversion_store=AssetStore(Path("workspace/office-conversions")),
)
```

The same API accepts `workbook.xls`. `conversion_store` retains only converted
Office files. The existing `store` argument still opts into image extraction;
when supplied without a separate conversion store it retains both.

The `saqqara` kernel automatically retains derivatives in its workspace under
`assets/office-conversions`, addressed by SHA-256. Using `read_path` without an
asset store leaves the derivative temporary: its provenance explicitly says
`artifact_retained=false`.

## Provenance and limits

Original input bytes stay unchanged. Each converted observation carries a typed
`ConversionLocator` outside its fact vocabulary. Built trees carry this bridge
before their DOCX/XLSX locators, plus a copy in `tree.meta.conversion`. It records
original and derivative SHA-256, input and output formats, LibreOffice version,
export filter, retention status, and rule `office-conversion/1`.

`source_path` and `source_sha256` refer to the original. `source_format` names the
reader pipeline family (`docx` or `xlsx`); the bridge names the actual original
format. All following coordinates refer to the derivative, including any page
marks. Conversion does **not** establish layout or content equivalence with the
original. Formula text is read from the derivative; conversion may recalculate
cached values. Output bytes or layout may differ across LibreOffice versions,
fonts and platforms. Pin those inputs when comparing replay identities; retain
the actual derivative for source inspection.

Every conversion uses a fresh LibreOffice profile with no inherited trusted
locations, very-high macro security, and automatic Writer/Calc link updates
disabled. It runs headlessly without a shell. This is process isolation, not an
operating-system sandbox. Temporary files are removed on success and refusal.

`ConversionOptions` declares a 120-second process timeout, input and output
limits of 128 MiB, an expanded ZIP limit of 512 MiB, and 20,000 entries. The
low-level `converted_office` / `read_legacy` APIs accept alternate positive
limits. These are acceptance limits, not OS resource quotas during conversion.
A timeout terminates the conversion process group on POSIX. Package validation
checks expected main parts, ZIP paths and CRCs before invoking the native reader.

## Executable gates

The optional-conversion contract is specified in the Legacy Office addendum of
[SPEC.md](SPEC.md), exercised by `tests/saqqara/test_legacy_office.py`.
The failure tests use a controlled converter; real binary DOC and XLS tests run
when LibreOffice is installed and otherwise report an explicit skip. Fixtures
are constructed locally, with no downloaded documents. The DOC fixture is
exported from constructed RTF: direct DOCX-to-DOC fixture generation can itself
flatten a table in LibreOffice, independently of this import path.

LibreOffice documents the [command-line parameters and isolated profile option](https://help.libreoffice.org/latest/en-GB/text/shared/guide/start_parameters.html)
and [conversion filters](https://help.libreoffice.org/latest/en-US/text/shared/guide/convertfilters.html).
