"""Thin local explorer CLI: typed digest JSON or an explicitly selected PDF reader.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from ..census import digest_from_dict
from ..explorer import digest_pdf, explore, imported_provenance
from ...harvest.report import canonical_json, render_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-reader", choices=["pymupdf"])
    parser.add_argument("--extractor-version")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--provenance", action="store_true")
    args = parser.parse_args(argv)
    provenance = (
        imported_provenance(require_clean=args.gate) if args.provenance or args.gate else None
    )
    inputs = sorted(args.input.glob("*.pdf")) if args.input.is_dir() else [args.input]
    if not inputs:
        parser.error("no input documents")
    args.output.mkdir(parents=True, exist_ok=True)
    seen = set()
    for path in inputs:
        if path.suffix.lower() == ".pdf":
            if not args.pdf_reader:
                parser.error("PDF intake requires explicit --pdf-reader pymupdf")
            document = digest_pdf(path, expected_pymupdf=args.extractor_version)
        else:
            document = digest_from_dict(json.loads(path.read_text()))
        if document.source_id in seen:
            continue
        seen.add(document.source_id)
        result = explore(document, provenance=provenance)
        # Content identity, not a supplied path, is the output name.
        import hashlib

        name = hashlib.sha256(document.source_id.encode()).hexdigest()
        (args.output / (name + ".json")).write_text(canonical_json(asdict(result)) + "\n")
        (args.output / (name + ".html")).write_text(
            render_report(
                result.report,
                {
                    "field": "Declared field",
                    "quantity": "Literal quantity",
                    "table_row": "Observed row",
                },
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
