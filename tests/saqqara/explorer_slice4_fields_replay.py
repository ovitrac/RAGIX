"""Clean-checkout synthetic reference-field replay for Slice 4; no private inputs or models.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Run from a clean checkout, the same command on each architecture:

    python -m tests.saqqara.explorer_slice4_fields_replay --output replay.json

The `cases` of two architectures must be equal; `provenance` records what ran.
"""

import argparse
import json
from pathlib import Path

from ragix_kernels.harvest.report import replay_digest
from ragix_kernels.saqqara.census import CensusConfig
from ragix_kernels.saqqara.explorer import explore, imported_provenance
from ragix_kernels.saqqara.value_windows import ReferencePolicy

from . import fixtures_explorer_slice4_fields as fx

LABELS = {fx.LABEL_EN, fx.LABEL_FR}
DECLARED = CensusConfig(
    reference_policy=ReferencePolicy(labels=(fx.LABEL_EN,), type_words=(fx.TYPE_WORD,))
)
# name -> (builder, configuration, reference fields expected per page)
CASES = {
    "a1": (fx.a1, None, 1),
    "a2:value": (lambda g: fx.a2(g, "value"), None, 1),
    "a2:label": (lambda g: fx.a2(g, "label"), None, 1),
    "a3": (fx.a3, None, 1),
    "a3:edge": (lambda g: fx.a3(g, edge=True), None, 0),
    "a5": (fx.a5, None, 1),
    "a6": (fx.a6, None, 1),
    "a7:declared": (lambda g: fx.a7(g, typed=True), DECLARED, 1),
    "a7:colon": (lambda g: fx.a7(g, colon_evidence=True), None, 2),
    "a8:right": (lambda g: fx.a8(g, "right", "per_cell", 0.3, True), None, 1),
    "a8:below": (lambda g: fx.a8(g, "below", "per_cell", 0.3, True), None, 1),
    "a9": (fx.a9, None, 1),
    "a9:role_only": (fx.role_only, None, 1),
    "a10:marked": (lambda g: fx.a10(g, "à", "§"), None, 1),
    "a10:bare": (lambda g: fx.a10(g, "to", ""), None, 1),
    "b1:grid": (fx.b1, None, 2),
    "b1:vertical": (lambda g: fx.b1(g, ruled="vertical_only"), None, 2),
    "b2:overhang": (lambda g: fx.b1(g, forms=("plain",), overhang=0.04), None, 1),
    "b3": (fx.b3, None, 0),
    "a11": (fx.a11, None, 1),
    "a12": (fx.a12, None, 1),
    "a12c": (lambda g: fx.a12(g, body_lines=0), None, 1),
    "a13": (fx.a13, None, 1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {"provenance": imported_provenance(require_clean=True), "cases": {}}
    for name, (build, config, expected) in CASES.items():
        for g in fx.GEOMETRIES:
            r = explore(build(g)) if config is None else explore(build(g), census_config=config)
            assert getattr(r.report, "status", None) != "FAILED"
            fields = [f for f in r.reading.fields if f.label in LABELS]
            assert len(fields) == expected * len(fx.PAGES), (name, g.name, len(fields))
            windows = [w for w in r.census.windows if w.label in LABELS]
            output["cases"][f"{name}@{g.name}"] = {
                "census": replay_digest([r.census]),
                "profile": replay_digest([r.profile]),
                "reading": replay_digest([r.reading]),
                "report": r.report.replay_digest,
                "fields": len(fields),
                "targets": sum(len(f.targets) for f in fields),
                "members": sum(len(t.sections) for f in fields for t in f.targets),
                "stops": sorted({w.stop_reason for w in windows}),
                "listed": sum(len(w.undecidable) for w in windows),
                "policy": r.census.continuation_policy.source,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
