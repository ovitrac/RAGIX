#!/usr/bin/env python3
"""Draw the cover figure from the boxes Saqqara recorded on one public page.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

The measurement (evidence/hero_p3.png, evidence/hero_p3.json) draws Saqqara's boxes over the rendered
page. On a cover, a faded page of another document with a frame around every line reads as selected
text, not as a figure. This script keeps the measured boxes and drops the page: each recorded node is
drawn as an outline on the dark blue cover gradient of style.css, so the figure shows structure only.
The boxes are read unchanged from the evidence file; no reading is re-run.

Usage: python3 restyle_hero.py <boxes.json> <WIDTHxHEIGHT> <out.png> [<WIDTHxHEIGHT> <out.png> ...]
       e.g. restyle_hero.py evidence/hero_p3.json 1600x960 img/hero.png 1600x1560 saqqara-tech-sheet/images/hero.png
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

#: The cover gradient of style.css (.cover_hero--gradient), 135 degrees.
STOPS = [(0.00, (0x0a, 0x25, 0x40)), (0.40, (0x03, 0x69, 0xa1)), (0.70, (0x0e, 0xa5, 0xe9)), (1.00, (0x7d, 0xd3, 0xfc))]

#: Outline per kind; the line style is a second encoding, so kind is never carried by colour alone.
STYLE = {
    "paragraph": {"rgba": (255, 255, 255, 215), "fill": (255, 255, 255, 22), "width": 3, "dash": None},
    "vector_region": {"rgba": (0x7d, 0xd3, 0xfc, 255), "fill": None, "width": 5, "dash": (18, 10)},
    "figure": {"rgba": (0xfb, 0x92, 0x3c, 255), "fill": None, "width": 6, "dash": None},
}


def gradient(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width].astype(float)
    t = (x / max(width - 1, 1) + y / max(height - 1, 1)) / 2.0  # 135 degrees: top-left to bottom-right
    out = np.zeros((height, width, 3))
    for (t0, c0), (t1, c1) in zip(STOPS, STOPS[1:]):
        mask = (t >= t0) & (t <= t1)
        u = ((t - t0) / (t1 - t0))[mask][:, None]
        out[mask] = np.array(c0) * (1 - u) + np.array(c1) * u
    return Image.fromarray(out.round().astype(np.uint8), "RGB").convert("RGBA")


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


def blueprint(facts: dict, width: int, height: int) -> Image.Image:
    page_w = facts["page_px"][0]
    scale = width / page_w
    top = min(b["pixels"][1] for b in facts["boxes"]) * scale - 0.06 * height  # start just above the first node
    canvas = gradient(width, height)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for kind in ("paragraph", "vector_region", "figure"):
        spec = STYLE[kind]
        for item in (b for b in facts["boxes"] if b["kind"] == kind):
            x0, y0, x1, y1 = (v * scale for v in item["pixels"])
            xy = (x0, y0 - top, x1, y1 - top)
            if xy[1] > height:
                continue
            if spec["fill"]:
                draw.rectangle(xy, fill=spec["fill"])
            if spec["dash"]:
                dashed(draw, xy, spec["rgba"], spec["width"], spec["dash"])
            else:
                draw.rectangle(xy, outline=spec["rgba"], width=spec["width"])
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def main() -> None:
    boxes_path, *jobs = sys.argv[1:]
    if not jobs or len(jobs) % 2:
        raise SystemExit(__doc__)
    facts = json.load(open(boxes_path, encoding="utf-8"))
    for size, out in zip(jobs[::2], jobs[1::2]):
        width, height = (int(v) for v in size.lower().split("x"))
        blueprint(facts, width, height).save(out, optimize=True)
        print(f"{len(facts['boxes'])} boxes, {width}x{height} px -> {out}")


if __name__ == "__main__":
    main()
