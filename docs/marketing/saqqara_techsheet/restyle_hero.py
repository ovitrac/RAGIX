#!/usr/bin/env python3
"""Draw the cover figure: one public page, readable, with the nodes Saqqara recorded outlined on it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

The measurement (evidence/hero_p3.png, evidence/hero_p3.json) fills each text box with light blue
over the page. On a cover that reads as selected text. This script keeps the page at full contrast,
so its text stays readable, and draws each recorded node as a thin outline with no fill.

A text node records an anchor on the baseline, a width and a font size, not a box; its rectangle runs
from the baseline up by the font size, so a line drawn on it crosses the descenders. The outline is
therefore drawn with a margin around that rectangle (MARGIN, in fractions of the font size), which
keeps every line off the glyphs. The boxes themselves are read unchanged from the evidence file, and
the source PDF is checked against the bytes that were measured; no reading is re-run.

Usage: python3 restyle_hero.py <source.pdf> <boxes.json> <WIDTHxHEIGHT[@RESERVE]> <out.png> [...]
       RESERVE keeps the last RESERVE pixels free of nodes (a caption band).
       dossier: 1600x1040 img/hero.png   slide 1: 1600x1560@240 saqqara-tech-sheet/images/hero.png
"""
import hashlib
import json
import sys

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

EXPECTED_SHA256 = "2db618b40c1cfaf1753ca44092b62b1ea916d9eec909c22e36a17ac86be61e1e"  # inputs_manifest_1b.json

#: Outline per kind, no fill; the line style is a second encoding, so kind is never carried by colour alone.
STYLE = {
    "paragraph": {"rgba": (0x2a, 0x78, 0xd6, 200), "width": 2, "dash": None},
    "vector_region": {"rgba": (0x1b, 0xaf, 0x7a, 255), "width": 4, "dash": (16, 9)},
    "figure": {"rgba": (0xeb, 0x68, 0x34, 255), "width": 5, "dash": None},
}
#: Margin around a text node's rectangle, as fractions of its height (the font size): left/right, top, bottom.
MARGIN = (0.25, 0.15, 0.35)
#: Fixed margin, in pixels, around picture and region boxes.
BOX_MARGIN = 6


def dashed(draw, xy, rgba, width, dash):
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
            draw.line([p, q], fill=rgba, width=width)
            t += on + off


def figure(page: Image.Image, facts: dict, width: int, height: int, reserve: int = 0) -> Image.Image:
    scale = width / page.width
    if scale != 1:
        page = page.resize((width, round(page.height * scale)), Image.Resampling.LANCZOS)
    top = max(0, round(min(b["pixels"][1] for b in facts["boxes"]) * scale - 0.05 * height))
    overlay = Image.new("RGBA", page.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    limit = top + height - reserve  # nodes stop above the last `reserve` pixels (a caption band)
    cut = top + height  # the crop ends above any node it would slice
    for kind in ("vector_region", "figure", "paragraph"):
        spec = STYLE[kind]
        for item in (b for b in facts["boxes"] if b["kind"] == kind):
            x0, y0, x1, y1 = (v * scale for v in item["pixels"])
            if kind == "paragraph":
                h = y1 - y0
                xy = (x0 - MARGIN[0] * h, y0 - MARGIN[1] * h, x1 + MARGIN[0] * h, y1 + MARGIN[2] * h)
            else:
                xy = (x0 - BOX_MARGIN, y0 - BOX_MARGIN, x1 + BOX_MARGIN, y1 + BOX_MARGIN)
            if xy[1] < limit < xy[3] + spec["width"]:
                cut = min(cut, xy[1] - 8)
            if spec["dash"]:
                dashed(draw, xy, spec["rgba"], spec["width"], spec["dash"])
            else:
                draw.rectangle(xy, outline=spec["rgba"], width=spec["width"])
    out = Image.alpha_composite(page.convert("RGBA"), overlay).convert("RGB").crop((0, top, width, top + height))
    if cut < top + height:  # blank what lies below the last whole node, keeping the size
        ImageDraw.Draw(out).rectangle((0, round(cut - top), width, height), fill=(255, 255, 255))
    return out


def main() -> None:
    source, boxes_path, *jobs = sys.argv[1:]
    if not jobs or len(jobs) % 2:
        raise SystemExit(__doc__)
    data = open(source, "rb").read()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("source PDF is not the measured one (sha256 mismatch)")
    facts = json.load(open(boxes_path, encoding="utf-8"))
    page_w, page_h = facts["page_px"]
    bitmap = pdfium.PdfDocument(data)[facts["page"] - 1].render(scale=facts["dpi"] / 72.0)
    page = bitmap.to_pil().convert("RGB")
    if page.size != (page_w, page_h):
        page = page.resize((page_w, page_h), Image.Resampling.LANCZOS)
    for size, out in zip(jobs[::2], jobs[1::2]):
        dims, _, reserve = size.lower().partition("@")
        width, height = (int(v) for v in dims.split("x"))
        figure(page, facts, width, height, int(reserve or 0)).save(out, optimize=True)
        print(f"{len(facts['boxes'])} boxes, {width}x{height} px -> {out}")


if __name__ == "__main__":
    main()
