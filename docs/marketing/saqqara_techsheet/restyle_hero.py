#!/usr/bin/env python3
"""Draw the cover figure: one public page, readable, with the nodes Saqqara recorded marked on it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

The measurement (evidence/hero_p3.png, evidence/hero_p3.json) fills each text box with light blue
over the page; on a cover that reads as selected text, and any frame around a line still looks like a
selection. Here the page stays at full contrast and no text is framed:

- a text node records an anchor (the start of a run, on its baseline), a width and a font size; a thin
  bar in the gutter marks each run that starts a line or stands apart on it, at its recorded anchor;
- a drawn region or a picture gets corner marks only.

The boxes are read unchanged from the evidence file, and the source PDF is checked against the bytes
that were measured; no reading is re-run.

Usage: python3 restyle_hero.py <source.pdf> <boxes.json> <WIDTHxHEIGHT[@RESERVE]> <out.png> [...]
       RESERVE keeps the last RESERVE pixels free of nodes (a caption band).
       dossier: 1600x1000 img/hero.png   slide 1: 1600x1560@240 saqqara-tech-sheet/images/hero.png
"""
import hashlib
import json
import sys

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

EXPECTED_SHA256 = "2db618b40c1cfaf1753ca44092b62b1ea916d9eec909c22e36a17ac86be61e1e"  # inputs_manifest_1b.json

#: Outline per kind, no fill; the line style is a second encoding, so kind is never carried by colour alone.
STYLE = {
    "paragraph": {"rgba": (0x2a, 0x78, 0xd6, 255), "width": 2, "dash": None},
    "vector_region": {"rgba": (0x1b, 0xaf, 0x7a, 255), "width": 4, "dash": (16, 9)},
    "figure": {"rgba": (0xeb, 0x68, 0x34, 255), "width": 5, "dash": None},
}
#: Text bar, in fractions of the font size: distance left of the anchor, width, extent above and below the baseline.
BAR = (0.45, 0.10, 0.80, 0.15)
#: A run is marked when it starts a line, or when it stands more than GAP font sizes from the previous run.
GAP = 1.5
#: Fixed margin, in pixels, around picture and region boxes, and the length of their corner marks.
BOX_MARGIN = 6
CORNER_ARM = 28


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


def corners(draw, xy, rgba, width, arm):
    """Corner marks only: a region is shown without framing what it holds."""
    x0, y0, x1, y1 = xy
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        draw.line([(cx, cy), (cx + dx * arm, cy)], fill=rgba, width=width)
        draw.line([(cx, cy), (cx, cy + dy * arm)], fill=rgba, width=width)


def figure(page: Image.Image, facts: dict, width: int, height: int, reserve: int = 0) -> Image.Image:
    scale = width / page.width
    if scale != 1:
        page = page.resize((width, round(page.height * scale)), Image.Resampling.LANCZOS)
    boxes = [dict(b, px=[v * scale for v in b["pixels"]]) for b in facts["boxes"]]
    top = max(0, round(min(b["px"][1] for b in boxes) - 0.05 * height))
    limit = top + height - reserve  # nodes stop above the last `reserve` pixels (a caption band)
    cut = top + height              # the crop ends above any node it would slice
    for b in boxes:
        x0, y0, x1, y1 = b["px"]
        if y0 < limit < y1 + max(BOX_MARGIN, 0.35 * (y1 - y0)):  # descenders hang below the baseline
            cut = min(cut, y0 - 0.4 * (y1 - y0) - 8)

    overlay = Image.new("RGBA", page.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for b in (b for b in boxes if b["kind"] in ("vector_region", "figure")):
        spec = STYLE[b["kind"]]
        x0, y0, x1, y1 = b["px"]
        corners(draw, (x0 - BOX_MARGIN, y0 - BOX_MARGIN, x1 + BOX_MARGIN, y1 + BOX_MARGIN),
                spec["rgba"], spec["width"], CORNER_ARM)
    # Text: a thin bar in the gutter of each run of text that starts a line or stands apart on it (a gap
    # wider than GAP font sizes from the previous run on the same baseline), at its recorded anchor.
    rows: dict[int, list] = {}
    for b in (b for b in boxes if b["kind"] == "paragraph"):
        rows.setdefault(round(b["px"][3] / 4), []).append(b)  # same baseline, within 4 px
    rgba = STYLE["paragraph"]["rgba"]
    for row in rows.values():
        row.sort(key=lambda b: b["px"][0])
        previous_end = None
        for b in row:
            x0, y0, x1, y1 = b["px"]
            h = y1 - y0  # the font size, in pixels
            if previous_end is None or x0 - previous_end > GAP * h:
                bx, w = x0 - BAR[0] * h, max(3.0, BAR[1] * h)
                draw.rectangle((bx - w, y1 - BAR[2] * h, bx, y1 + BAR[3] * h), fill=rgba)
            previous_end = x1 if previous_end is None else max(previous_end, x1)
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
