#!/usr/bin/env python3
"""Synthetic Explorer Slice 3 replay; no corpus and no model calls.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""
import argparse, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragix_kernels.saqqara.explorer import explore, imported_provenance
from ragix_kernels.harvest.report import (
    replay_digest,
    render_report,
    render_report_json,
    render_page_view,
)

root = ROOT
parser = argparse.ArgumentParser(description="Synthetic priority, privacy and table replay gate")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def load(name):
    spec = importlib.util.spec_from_file_location(name, root / "tests" / "saqqara" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


occ = load("test_explorer_occurrences")
privacy = load("test_explorer_privacy")
tables = load("test_explorer_tables")
failure = load("test_explorer_failures")
cases = {
    "occurrences": occ.occurrence_document(),
    "free_plurality": occ.class_document(
        ["ZX-R-001", "ZX-R-002", *("some words" for _ in range(6))]
    ),
    "privacy": privacy.fixture(),
}
for columns in (3, 5, 8):
    for ruled in (False, True):
        for language in ("en", "fr"):
            doc, expected = tables.fixture(columns, ruled, language)
            r = explore(doc)
            assert (
                tuple(row.cells for t in r.census.table_analysis.tables for row in t.rows)
                == expected
            )
            cases[f"table_{columns}_{ruled}_{language}"] = doc
for ruled, segmented in ((False, False), (True, False), (True, True)):
    for counts in ((1, 1, 1, 1, 1, 1), (1, 4, 2, 1, 3, 2, 4, 1, 2)):
        doc, expected = tables.short_fragments(ruled, segmented, counts)
        r = explore(doc)
        assert (
            tuple(row.cells for t in r.census.table_analysis.tables for row in t.rows) == expected
        )
        assert len(r.census.table_analysis.tables) == 1
        cases[f"short_{ruled}_{segmented}_{len(counts)}"] = doc
for columns in (3, 5, 8):
    doc, expected = tables.cell_box_fixture(columns)
    r = explore(doc)
    assert tuple(row.cells for t in r.census.table_analysis.tables for row in t.rows) == expected
    cases[f"native_padding_{columns}"] = doc
doc, expected = tables.varying_native_subcells()
r = explore(doc)
assert tuple(row.cells for t in r.census.table_analysis.tables for row in t.rows) == expected
cases["native_varying_subcells"] = doc
for value in (":", "   :", "Link:"):
    cases["failure_" + str(len(cases))] = failure.document(text=value, rule=True)
output = {"provenance": imported_provenance(require_clean=True), "cases": {}}
for name, doc in cases.items():
    r = explore(doc)
    assert getattr(r.report, "status", None) != "FAILED"
    output["cases"][name] = {
        "census": replay_digest([r.census]),
        "profile": replay_digest([r.profile]),
        "reading": replay_digest([r.reading]),
        "report": r.report.replay_digest,
    }
r = explore(privacy.fixture())
texts = [
    render_report(r.report, privacy.LABELS),
    render_report_json(r.report),
    *[render_page_view(r.report, p) for p in range(1, 7)],
]
assert all(token not in text for text in texts for name in privacy.NAMES for token in name.split())
output["privacy"] = {
    "masked_lines": r.report.provenance["masked_lines"],
    "presentations": replay_digest(texts),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
