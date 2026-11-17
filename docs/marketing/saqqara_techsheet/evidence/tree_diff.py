"""
Compare the trees of two saqqara run outputs node by node, with version strings normalised.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python tree_diff.py <a>/stage1/saqqara.json <b>/stage1/saqqara.json [--ignore-versions]

A tree records the reader that read it in its provenance (`kernel_version`, e.g. `0.8.0` or
`0.8.0+pymupdf-<version>`), so two reader versions always give two roots. By default the PyMuPDF version
inside that string is normalised; with `--ignore-versions` every `kernel_version`, `reader_version` and
`builder_version` is blanked. Then it lists every node whose text anchor (facts x, y, width) differs,
with its page, text, font and size, and counts the nodes that differ in anything else, anywhere in the
tree.
"""

from __future__ import annotations

import argparse
import json
import re

VERSION = re.compile(r"\+pymupdf-[0-9.]+")
VERSION_KEYS = ("kernel_version", "reader_version", "builder_version")
ANCHOR = ("x", "y", "width")


def load(path):
    data = json.load(open(path))
    return data.get("data", data)["documents"][0]


def norm(value, blank_versions):
    if isinstance(value, dict):
        return {k: ("" if blank_versions and k in VERSION_KEYS else norm(v, blank_versions))
                for k, v in value.items()}
    if isinstance(value, list):
        return [norm(v, blank_versions) for v in value]
    if isinstance(value, str):
        return VERSION.sub("+pymupdf-X", value)
    return value


def without_anchor(node):
    facts = {k: v for k, v in (node.get("facts") or {}).items() if k not in ANCHOR}
    return {k: v for k, v in node.items() if k not in ("children", "facts")} | {"facts": facts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--ignore-versions", action="store_true")
    args = parser.parse_args()
    a, b = (norm(load(p), args.ignore_versions) for p in (args.a, args.b))
    print("traces equal:", json.dumps(a["traces"], sort_keys=True) == json.dumps(b["traces"], sort_keys=True))
    if json.dumps(a["tree"], sort_keys=True) == json.dumps(b["tree"], sort_keys=True):
        print("trees equal once versions are normalised")
        return
    moved, other, shape = [], 0, 0

    def walk(na, nb, page):
        nonlocal other, shape
        if na.get("kind") == "page":
            page = na["provenance"]["chain"][-1].get("page")
        fa, fb = na.get("facts") or {}, nb.get("facts") or {}
        if any(fa.get(k) != fb.get(k) for k in ANCHOR):
            moved.append((page, (na.get("text") or "")[:24], fa.get("font"), fa.get("font_size"),
                          *(round(fb[k] - fa[k], 2) if isinstance(fa.get(k), (int, float))
                            and isinstance(fb.get(k), (int, float)) else (fa.get(k), fb.get(k)) for k in ANCHOR)))
        if json.dumps(without_anchor(na), sort_keys=True) != json.dumps(without_anchor(nb), sort_keys=True):
            other += 1
        ca, cb = na.get("children", []), nb.get("children", [])
        if len(ca) != len(cb):
            shape += 1
        for x, y in zip(ca, cb):
            walk(x, y, page)

    walk(a["tree"]["root"], b["tree"]["root"], None)
    print(f"anchors moved: {len(moved)}; nodes differing in anything else: {other}; "
          f"nodes with a different number of children: {shape}")
    print("page | text | font | size | dx | dy | dwidth")
    for row in moved[:40]:
        print(" | ".join(str(v) for v in row))
    if len(moved) > 40:
        print(f"… {len(moved) - 40} more")


if __name__ == "__main__":
    main()
