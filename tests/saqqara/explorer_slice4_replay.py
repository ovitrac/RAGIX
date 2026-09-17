"""Clean-checkout synthetic table replay for Slice 4; no private inputs or models.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import argparse
import json
from pathlib import Path
from .fixtures_explorer_slice4 import TableGeometry, realistic_table, ROW_COUNTS
from ragix_kernels.saqqara.explorer import explore, imported_provenance
from ragix_kernels.harvest.report import replay_digest, render_report_json, render_page_view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {"provenance": imported_provenance(require_clean=True), "cases": {}}
    cases = []
    for native in (12, 13, 14, 15):
        geometry = TableGeometry(native, 100, 32, 60, 45)
        cases.extend(
            (
                (f"T:{native}", realistic_table(geometry)),
                (f"control:{native}", realistic_table(geometry, all_controls=True)),
            )
        )
    for native in (12, 15):
        for width, height, points, angle in ((80, 24, 50, 30), (120, 40, 90, 60)):
            cases.append(
                (
                    f"range:{native}:{angle}",
                    realistic_table(TableGeometry(native, width, height, points, angle)),
                )
            )
    cases.append(("gap", realistic_table(TableGeometry(15, 100, 32, 60, 45), removed_page=17)))
    for name, fixture in cases:
        r = explore(fixture.document)
        assert getattr(r.report, "status", None) != "FAILED"
        rows = [row for table in r.census.table_analysis.tables for row in table.rows]
        if name == "gap":
            assert len(r.census.table_analysis.tables) == 2 and len(rows) == 108 - ROW_COUNTS[12]
        else:
            assert (
                len(r.census.table_analysis.tables) == 1
                and tuple(row.cells for row in rows) == fixture.expected_rows
            )
            flags = {
                ident
                for row in rows
                for ident, marks in row.cell_flags
                if "LIFECYCLE_GLYPH_SUSPECTED" in marks
            }
            assert flags == set(fixture.watermark_cells)
        presentations = (render_report_json(r.report), render_page_view(r.report, 1))
        # This fixture has no freeze line-stamps. Row masking is deferred.
        assert r.report.provenance["masked_lines"] == 0
        output["cases"][name] = {
            "census": replay_digest([r.census]),
            "profile": replay_digest([r.profile]),
            "reading": replay_digest([r.reading]),
            "report": r.report.replay_digest,
            "presentations": replay_digest(presentations),
            "rows": len(rows),
            "tables": len(r.census.table_analysis.tables),
            "masked_lines": r.report.provenance["masked_lines"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
