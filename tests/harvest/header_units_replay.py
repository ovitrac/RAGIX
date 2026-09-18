"""Clean synthetic cell-context, inheritance and relative-expression replay.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from ragix_kernels.harvest.report import replay_digest
from ragix_kernels.saqqara.explorer import explore, imported_provenance
from .test_header_units import context, read
from tests.saqqara.test_header_unit_context import table_fixture


EXPRESSIONS = (
    "23",
    "13 à 29",
    "23 ± 3",
    "23 V",
    "23 widgets",
    "23 €",
    "datum -13 to datum -31",
    "origin +7 au origin −11",
    "référence locale -17 à référence locale +23",
    "datum ±13 au datum ±31",
    "datum -13 V to datum -31 mm",
    "datum -13 to origin -31",
    "datum -13 to datum -31 approximately",
    "-13 to -31",
    "13 to 31",
    "45 - 13",
    "origin — 13",
    "RX-13",
    "datum -1.234",
    "datum -13,25",
    "2033-07-19",
)
HEADERS = ("Travel [mm]", "Temperature [° C]", "relative to datum [mm]", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    for text in EXPRESSIONS:
        for header in HEADERS:
            c = context(text=text, header=header)
            candidates = read(c)
            assert all(c.value.text[q.start : q.end] == q.raw for q in candidates)
            assert all(
                m.candidate_id in {q.candidate_id for q in candidates}
                for q in candidates
                for m in q.members
            )
            for q in candidates:
                if q.kind.startswith("relative_"):
                    assert (
                        q.number is None and q.lower is None and q.upper is None and q.needs_review
                    )
            cases[text + "|" + header] = replay_digest(candidates)
    for name, headers, rows in (
        ("columns", ("Channel", "Travel [mm]", "Supply [V]"), None),
        (
            "row-units",
            ("Channel", "Reading", "Note"),
            tuple((f"Axis {x} [mm]", str(n), "plain") for x, n in zip("ABCD", (23, 31, 47, 53))),
        ),
        (
            "relative",
            ("Channel", "Relative range", "Note"),
            tuple(
                (f"Axis {x} [mm]", f"datum -{n} au datum -{n+18}", "plain")
                for x, n in zip("ABCD", (13, 17, 23, 29))
            ),
        ),
        ("multiline", ("Channel", "Travel\n[mm]", "Supply\n[V]"), None),
    ):
        doc, _ = table_fixture(headers=headers, rows=rows)
        result = explore(doc)
        assert len(result.reading.cell_contexts) == 8
        cases["reader:" + name] = replay_digest([result.reading, result.report])
    output = {"provenance": imported_provenance(require_clean=True), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
