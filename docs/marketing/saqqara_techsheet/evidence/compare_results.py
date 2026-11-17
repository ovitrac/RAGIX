"""
Compare two bench results field by field, wall times aside, and list every difference.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python compare_results.py <a.json> <b.json> [--max 60]

Set aside before comparing: every key ending in `wall_s`, `wall_s_per_page`, `part_wall_s`, the renderer
label, and the host record (machine, CPU, versions, load). Compared: the parts, the inputs and the code
record. A difference in a work-directory path (`work_m2/…` against `work_m2b/…`) is listed like any other.
"""

from __future__ import annotations

import argparse
import collections
import json

SET_ASIDE = ("wall_s_per_page", "part_wall_s", "renderer")


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if not (k.endswith("wall_s") or k in SET_ASIDE)}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def diff(x, y, path, out):
    if type(x) != type(y):
        out.append((path, x, y))
    elif isinstance(x, dict):
        for key in sorted(set(x) | set(y)):
            if key not in x or key not in y:
                out.append((f"{path}/{key}", x.get(key), y.get(key)))
            else:
                diff(x[key], y[key], f"{path}/{key}", out)
    elif isinstance(x, list):
        if len(x) != len(y):
            out.append((f"{path} (length)", len(x), len(y)))
        for i, (p, q) in enumerate(zip(x, y)):
            diff(p, q, f"{path}[{i}]", out)
    elif x != y:
        out.append((path, x, y))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--max", type=int, default=60)
    args = parser.parse_args()
    a, b = (json.load(open(p)) for p in (args.a, args.b))
    out: list = []
    for key in ("parts", "inputs", "code"):
        diff(strip(a[key]), strip(b[key]), key, out)
    print(f"{len(out)} difference(s) outside wall times, the renderer label and the host record")
    leaf = collections.Counter(path.split("/")[-1].split("[")[0] for path, _, _ in out)
    print("by field:", dict(leaf.most_common()))
    for path, x, y in out[: args.max]:
        print(f"  {path}: {str(x)[:90]!s} | {str(y)[:90]!s}")


if __name__ == "__main__":
    main()
