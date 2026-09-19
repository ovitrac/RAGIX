"""Clean-checkout, synthetic region replay and boundary baseline measurement.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
from ragix_kernels.harvest.regions import RegionIndex, RegionWindow, FigureInput, image_payload
from ragix_kernels.harvest.region_types import json_data
from ragix_kernels.saqqara.explorer import explore, imported_provenance
from ragix_kernels.saqqara.renderable_regions import regions_from_explorer
from .test_region_boundaries import fixtures, line, score
from .test_renderable_regions import table, png, PAGES
from tests.saqqara.test_header_unit_context import table_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = {}
    for index, (lines, _) in enumerate(fixtures()):
        view = RegionIndex("synthetic", PAGES, lines)
        for member in lines:
            for before, after, same_page in ((0, 0, True), (1, 1, True), (2, 2, False)):
                result = view.get(member.member_id, window=RegionWindow(before, after, same_page))
                cases[f"prose:{index}:{member.member_id}:{before}:{same_page}"] = result.to_dict()
    cells = table()
    view = RegionIndex("synthetic", PAGES, (), tables=(cells,))
    for cell in cells:
        cases["cell:" + cell.member_id] = view.get(cell.member_id).to_dict()
    lines = (
        line("a", "• Outer", 20),
        line("b", "• Nested", 33, x=35),
        line("c", "wrapped", 46, x=38),
        line("d", "• Outer again", 59),
    )
    view = RegionIndex("synthetic", PAGES, lines)
    for m in lines:
        cases["list:" + m.member_id] = view.get(m.member_id).to_dict()
    figure = FigureInput(
        "f",
        "synthetic",
        1,
        (20, 20, 200, 100),
        image_payload(png(), "image/png"),
        caption_ids=("caption",),
    )
    view = RegionIndex(
        "synthetic",
        PAGES,
        (line("label", "Label", 30), line("caption", "Caption", 120)),
        figures=(figure,),
    )
    for ident in ("label", "caption"):
        cases["figure:" + ident] = view.get(ident).to_dict()
    doc, _ = table_fixture()
    view = regions_from_explorer(explore(doc))
    for region in view.regions:
        for m in region.members:
            cases["native:" + m.member_id] = view.get(m.member_id).to_dict()
    provenance = imported_provenance(require_clean=True)
    provenance["pillow"] = importlib.metadata.version("Pillow")
    output = {
        "provenance": provenance,
        "baseline_scores": {mode: score(mode) for mode in ("line", "section", "geometry")},
        "cases": cases,
        "payload_sha256": hashlib.sha256(json_data(cases).encode("utf-8")).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
