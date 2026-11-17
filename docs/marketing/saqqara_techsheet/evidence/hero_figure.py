"""
The cover figure: one page of the RAGIX tech sheet, rendered by Saqqara's default renderer, with the
boxes Saqqara records for that page drawn over it, one colour and one line style per node kind.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    cd <bench> && python hero_figure.py --page 3 --width 1600 --out <png> --boxes <json> --work <dir>

What is drawn, and from which record:
  paragraph       the text anchor the reader records (facts x, y in PDF user space), its width, and its
                  font size as the height. The reader records no text box; this rectangle is drawn from
                  those three facts. Read with the PyMuPDF text reader: with the default pypdf reader
                  most anchors of this file fall outside the page (EVIDENCE.md §4).
  figure          a picture placement, its box from the locator (the reader is given an asset store).
  vector_region   a drawn region (facts x, y, w, h) promoted by the library pass VectorRegionAnalyzer.
The page itself is rasterised by `saqqara.render.default_renderer()` (pypdfium2). Neither the figure
pass nor the region pass runs in a default kernel run (KOAS_SAQQARA.md §3.3).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ragix_kernels.saqqara.adapters import read_path, registered_adapters
from ragix_kernels.saqqara.analyzers.vector_regions import VectorRegionAnalyzer
from ragix_kernels.saqqara.assets import AssetStore
from ragix_kernels.saqqara.builder import build_tree
from ragix_kernels.saqqara.render import default_renderer

SOURCE = Path("inputs/public/Adservio-RAGIXTechSheet.pdf")

#: Categorical slots 1-3 of the reference palette, in fixed order; checked with the palette validator
#: (all checks pass on a light surface; aqua is under 3:1 contrast, hence the labelled legend). The
#: line style is a second encoding, so kind is never carried by colour alone.
KINDS = {
    "paragraph": {"rgb": (0x2a, 0x78, 0xd6), "label": "paragraph: text anchor × width × font size",
                  "width": 2, "dash": None, "fill": 40},
    "figure": {"rgb": (0xeb, 0x68, 0x34), "label": "figure: picture placement", "width": 5,
               "dash": None, "fill": 0},
    "vector_region": {"rgb": (0x1b, 0xaf, 0x7a), "label": "vector_region: drawn region", "width": 4,
                      "dash": (18, 10), "fill": 0},
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def walk(node, acc):
    acc.append(node)
    for child in node.get("children", []):
        walk(child, acc)
    return acc


def boxes_for(tree: dict, page_number: int) -> tuple[list[dict], dict]:
    """(kind, x0, y0, x1, y1) in PDF user space, bottom-up, for every drawn node of one page."""
    page = next(p for p in tree["root"]["children"]
                if p["kind"] == "page" and p["provenance"]["chain"][-1].get("page") == page_number)
    out = []
    for node in walk(page, [])[1:]:
        facts = node.get("facts") or {}
        locator = node["provenance"]["chain"][-1]
        if node["kind"] == "paragraph" and {"x", "y", "width", "font_size"} <= set(facts):
            out.append({"kind": "paragraph", "box": [facts["x"], facts["y"], facts["x"] + facts["width"],
                                                      facts["y"] + facts["font_size"]]})
        elif node["kind"] == "figure" and locator.get("bbox"):
            x0, ya, x1, yb = locator["bbox"]
            out.append({"kind": "figure", "box": [min(x0, x1), min(ya, yb), max(x0, x1), max(ya, yb)]})
        elif node["kind"] == "vector_region" and {"x", "y", "w", "h"} <= set(facts):
            out.append({"kind": "vector_region",
                        "box": [facts["x"], facts["y"], facts["x"] + facts["w"], facts["y"] + facts["h"]]})
    return out, page["facts"]


def dashed_rectangle(draw, xy, colour, width, dash):
    x0, y0, x1, y1 = xy
    on, off = dash
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        step = 0.0
        while step < length:
            end = min(step + on, length)
            p = (a[0] + (b[0] - a[0]) * step / length, a[1] + (b[1] - a[1]) * step / length)
            q = (a[0] + (b[0] - a[0]) * end / length, a[1] + (b[1] - a[1]) * end / length)
            draw.line([p, q], fill=colour, width=width)
            step += on + off


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()

    reader = registered_adapters()[".pdf"].configured({"text_reader": "pymupdf"})
    store = AssetStore(Path(args.work) / f"assets_p{args.page}")
    records = read_path(SOURCE, store=store, readers={"pdf": reader})
    tree = build_tree(records, str(SOURCE), reader.format, reader.format, reader.version).tree
    tree = VectorRegionAnalyzer(store=store, keep_raster=False).run(tree).tree.to_dict()
    boxes, page_facts = boxes_for(tree, args.page)
    width_pt, height_pt = page_facts["width"], page_facts["height"]

    renderer = default_renderer()
    dpi = args.width * 72.0 / width_pt
    payload, _media = renderer.render(SOURCE, args.page, (0.0, 0.0, width_pt, height_pt), dpi)
    page = Image.open(io.BytesIO(payload)).convert("RGBA")
    scale = page.width / width_pt

    overlay = Image.new("RGBA", page.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for kind in ("paragraph", "vector_region", "figure"):
        spec = KINDS[kind]
        for item in (b for b in boxes if b["kind"] == kind):
            x0, y0, x1, y1 = item["box"]
            xy = (x0 * scale, (height_pt - y1) * scale, x1 * scale, (height_pt - y0) * scale)
            item["pixels"] = [round(v, 1) for v in xy]
            if spec["fill"]:
                draw.rectangle(xy, fill=spec["rgb"] + (spec["fill"],))
            if spec["dash"]:
                dashed_rectangle(draw, xy, spec["rgb"] + (255,), spec["width"], spec["dash"])
            else:
                draw.rectangle(xy, outline=spec["rgb"] + (255,), width=spec["width"])
    composed = Image.alpha_composite(page, overlay)

    font = ImageFont.truetype(FONT, 26)
    small = ImageFont.truetype(FONT, 20)
    band = 64 + 44 * len(KINDS)
    canvas = Image.new("RGB", (composed.width, composed.height + band), (252, 252, 251))
    canvas.paste(composed.convert("RGB"), (0, 0))
    ink = ImageDraw.Draw(canvas)
    y = composed.height + 22
    counts = {kind: sum(1 for b in boxes if b["kind"] == kind) for kind in KINDS}
    for kind, spec in KINDS.items():
        swatch = (40, y + 4, 92, y + 30)
        if spec["dash"]:
            dashed_rectangle(ink, swatch, spec["rgb"], spec["width"], (10, 6))
        else:
            ink.rectangle(swatch, outline=spec["rgb"], width=spec["width"])
        ink.text((112, y), f"{spec['label']}  ({counts[kind]} on this page)", font=font, fill=(26, 26, 25))
        y += 44
    ink.text((40, y + 2), f"Page {args.page} of {SOURCE.name}, rendered by {renderer.name} {renderer.version}; "
                          f"boxes from Saqqara's tree (PyMuPDF text reader).", font=small, fill=(90, 90, 86))
    canvas.save(args.out)

    Path(args.boxes).write_text(json.dumps({
        "source": str(SOURCE), "page": args.page, "page_pt": [width_pt, height_pt], "dpi": round(dpi, 4),
        "renderer": f"{renderer.name} {renderer.version}", "reader": reader.version,
        "image_px": [canvas.width, canvas.height], "page_px": [composed.width, composed.height],
        "counts": counts, "boxes": boxes}, indent=1, sort_keys=True) + "\n")
    print(f"{args.out}: {canvas.width}×{canvas.height} px, {counts}")


if __name__ == "__main__":
    main()
