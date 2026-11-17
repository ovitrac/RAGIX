"""
What differs between two of Docling's JSON exports of the same input.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python docling_json_diff.py <results.json> <work dir> <id> [<id> ...]

For each input: whether the counted measures (items, labels, tables and cells, pictures) agree across
every run of the results file; then, the first run against every other run, the number of differing
values in the exported JSON, every path where they occur (most frequent first), and for numbers the
largest absolute difference.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def diff(x, y, path, out):
    if type(x) != type(y):
        out.append((path, "type"))
    elif isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append((f"{path}/{k}", "missing"))
            else:
                diff(x[k], y[k], f"{path}/{k}", out)
    elif isinstance(x, list):
        if len(x) != len(y):
            out.append((path, f"length {len(x)} vs {len(y)}"))
        for p, q in zip(x, y):
            diff(p, q, f"{path}[]", out)
    elif x != y:
        out.append((path, abs(x - y) if isinstance(x, (int, float)) and not isinstance(x, bool) else "value"))


def main() -> None:
    results, work, ids = json.load(open(sys.argv[1])), Path(sys.argv[2]), sys.argv[3:]
    for ident in ids:
        runs = {f"T{x['threads']}_run{x['run']}": x for x in results["runs"] if x["id"] == ident}
        signature = {tag: (x["items"], json.dumps(x["labels"], sort_keys=True), len(x["tables"]),
                           sum(t["cells"] for t in x["tables"]), x["pictures"]) for tag, x in runs.items()}
        print(f"{ident}: counted measures equal across {len(runs)} runs: {len(set(signature.values())) == 1}")
        tags = sorted(runs)
        base = json.load(open(work / tags[0] / f"{ident}.json"))
        for tag in tags[1:]:
            other = json.load(open(work / tag / f"{ident}.json"))
            out: list = []
            diff(base, other, "", out)
            numbers = [v for _, v in out if isinstance(v, (int, float))]
            where = collections.Counter(p for p, _ in out).most_common()
            print(f"  {tags[0]} vs {tag}: {len(out)} differing values; max |delta| "
                  f"{max(numbers) if numbers else '–'}; every path: {where}")


if __name__ == "__main__":
    main()
