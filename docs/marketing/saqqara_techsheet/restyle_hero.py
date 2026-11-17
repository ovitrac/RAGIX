#!/usr/bin/env python3
"""Redraw the cover figure from its recorded boxes: outlines on a faded page, no fill.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

The measured figure (evidence/hero_p3.png) fills each text box with a light blue, which on a cover
reads as a browser text selection. This script keeps the measurement untouched: it reads the boxes
already recorded in evidence/hero_p3.json (pixel coordinates at the recorded resolution), renders the
same page of the same file with pypdfium2, fades it, and draws each box as an outline only.

Usage: python3 restyle_hero.py <source.pdf> <boxes.json> <out.png> [<out.png> ...]
"""
import hashlib
import json
import sys

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

EXPECTED_SHA256 = "2db618b40c1cfaf1753ca44092b62b1ea916d9eec909c22e36a17ac86be61e1e"  # inputs_manifest_1b.json

#: Outline only; the line style is a second encoding, so kind is never carried by colour alone.
STYLE = {
    "paragraph": {"rgb": (0x2a, 0x78, 0xd6), "width": 3, "dash": None},
    "vector_region": {"rgb": (0x1b, 0xaf, 0x7a), "width": 5, "dash": (18, 10)},
    "figure": {"rgb": (0xeb, 0x68, 0x34), "width": 6, "dash": None},
}
FADE = 0.82  # share of white blended into the greyscale page: the text recedes, the boxes carry the figure


def dashed(draw, xy, rgb, width, dash):
    x0, y0, x1, y1 = xy
    on, off = dash
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                               ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = abs(bx - ax) + abs(by - ay)
        t = 0.0
        while t < length:
            u = min(t + on, length)
            p = (ax + (bx - ax) * t / length, ay + (by - ay) * t / length)
            q = (ax + (bx - ax) * u / length, ay + (by - ay) * u / length)
            draw.line([p, q], fill=rgb, width=width)
            t += on + off


def main() -> None:
    source, boxes_path, *outs = sys.argv[1:]
    data = open(source, "rb").read()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("source PDF is not the measured one (sha256 mismatch)")
    facts = json.load(open(boxes_path, encoding="utf-8"))
    page_w, page_h = facts["page_px"]

    pdf = pdfium.PdfDocument(data)
    bitmap = pdf[facts["page"] - 1].render(scale=facts["dpi"] / 72.0)
    page = bitmap.to_pil().convert("L").convert("RGB")
    if page.size != (page_w, page_h):
        page = page.resize((page_w, page_h), Image.Resampling.LANCZOS)
    page = Image.blend(page, Image.new("RGB", page.size, (255, 255, 255)), FADE)

    draw = ImageDraw.Draw(page)
    for kind in ("paragraph", "vector_region", "figure"):
        spec = STYLE[kind]
        for item in (b for b in facts["boxes"] if b["kind"] == kind):
            xy = tuple(item["pixels"])
            if spec["dash"]:
                dashed(draw, xy, spec["rgb"], spec["width"], spec["dash"])
            else:
                draw.rectangle(xy, outline=spec["rgb"], width=spec["width"])
    for out in outs:
        page.save(out, optimize=True)
    print(f"{len(facts['boxes'])} boxes, {page.size[0]}x{page.size[1]} px -> {', '.join(outs)}")


if __name__ == "__main__":
    main()
