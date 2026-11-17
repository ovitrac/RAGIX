"""
Plan the downloads of the Docling environment (M4) without downloading anything.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python m4_plan_downloads.py <out.json> [--skip name,name] <requirement> [<requirement> ...]

`--skip` leaves packages out of the resolution (they come from another index, e.g. the PyTorch CPU wheels;
their own dependencies are then given as requirements). Every skipped request is listed.

Reads PyPI's JSON API (package metadata only; no file is written but <out.json>) and resolves the
closure of the requirements for CPython 3.12 on manylinux aarch64, following extras. For each package it
lists the version chosen (the newest release that satisfies every bound seen so far), the wheel file,
its size and its sha256. A package with no matching wheel is reported, not guessed. This is an
estimate of what pip will resolve, not a substitute for pip's own install report, which records what
was actually fetched.
"""

from __future__ import annotations

import json
import sys
import urllib.request

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

BASE_ENV = {"python_version": "3.12", "python_full_version": "3.12.13", "sys_platform": "linux",
            "platform_machine": "aarch64", "platform_system": "Linux", "os_name": "posix",
            "implementation_name": "cpython", "platform_python_implementation": "CPython"}
_cache: dict = {}


def meta(name, version=None):
    key = (name, version)
    if key not in _cache:
        url = f"https://pypi.org/pypi/{name}/json" if version is None else f"https://pypi.org/pypi/{name}/{version}/json"
        with urllib.request.urlopen(url, timeout=30) as response:
            _cache[key] = json.load(response)
    return _cache[key]


def wheel_ok(filename):
    try:
        _, _, _, tags = parse_wheel_filename(filename)
    except Exception:
        return False
    for t in tags:
        python_ok = t.interpreter in ("cp312", "py3", "py2") or (t.abi == "abi3" and t.interpreter.startswith("cp3"))
        abi_ok = t.abi in ("cp312", "none", "abi3")
        platform_ok = t.platform == "any" or ("aarch64" in t.platform and "manylinux" in t.platform)
        if python_ok and abi_ok and platform_ok:
            return True
    return False


def applies(marker, extras):
    if marker is None:
        return True
    return any(marker.evaluate(BASE_ENV | {"extra": e}) for e in (extras or {""}))


def main() -> None:
    out_path, roots = sys.argv[1], sys.argv[2:]
    skip = set()
    if roots[:1] == ["--skip"]:
        skip, roots = {canonicalize_name(n) for n in roots[1].split(",")}, roots[2:]
    skipped = []
    specs: dict = {}
    chosen: dict = {}
    queue = [(Requirement(r), "root") for r in roots]
    problems = []
    while queue:
        req, parent = queue.pop(0)
        name = canonicalize_name(req.name)
        if name in skip:
            skipped.append(f"{req} (from {parent})")
            continue
        specs.setdefault(name, []).append(str(req.specifier))
        if name in chosen:
            c = chosen[name]
            if not req.specifier.contains(c["version"], prereleases=True):
                problems.append(f"CONFLICT {name} {req.specifier} (from {parent}) vs chosen {c['version']}")
            new = set(req.extras) - set(c["extras"])
            if not new:
                continue
            c["extras"] = sorted(set(c["extras"]) | new)
            # only the dependencies that the newly requested extras bring
            for d in c["requires_dist"]:
                r = Requirement(d)
                if r.marker is not None and applies(r.marker, new) and not applies(r.marker, {""}):
                    queue.append((r, name))
            continue
        top = meta(name)
        versions = sorted((Version(v) for v, files in top["releases"].items()
                           if files and req.specifier.contains(v, prereleases=False)
                           and not Version(v).is_prerelease), reverse=True)
        pick = None
        for v in versions:
            wheels = [f for f in top["releases"][str(v)]
                      if f["packagetype"] == "bdist_wheel" and wheel_ok(f["filename"]) and not f.get("yanked")]
            if wheels:
                pick = (v, sorted(wheels, key=lambda f: f["size"])[0])
                break
        if pick is None:
            sdists = [f for v in versions[:1] for f in top["releases"][str(v)] if f["packagetype"] == "sdist"]
            problems.append(f"NO WHEEL {name} {req.specifier} (from {parent}); sdist: "
                            f"{[(f['filename'], f['size']) for f in sdists]}")
            continue
        v, f = pick
        info = meta(name, str(v))["info"]
        c = chosen[name] = {"version": str(v), "file": f["filename"], "bytes": f["size"],
                            "sha256": f["digests"]["sha256"], "url": f["url"],
                            "license": (info.get("license_expression") or info.get("license") or "")[:60],
                            "required_by": parent, "extras": sorted(req.extras),
                            "requires_dist": info.get("requires_dist") or []}
        # the base dependencies, plus those of the requested extras
        for d in c["requires_dist"]:
            r = Requirement(d)
            if applies(r.marker, {""} | set(req.extras)):
                queue.append((r, name))

    total = sum(c["bytes"] for c in chosen.values())
    for n, c in sorted(chosen.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"{n:32s} {c['version']:14s} {c['bytes']:>11d}  {c['file']}  [{c['license']}]")
    print(f"TOTAL {len(chosen)} wheels, {total} bytes")
    for p in problems:
        print(p)
    for k in skipped:
        print("SKIPPED", k)
    for c in chosen.values():
        c.pop("requires_dist", None)
    json.dump({"python": "3.12.13", "platform": "manylinux aarch64", "roots": roots, "packages": chosen,
               "total_bytes": total, "problems": problems, "skipped": skipped}, open(out_path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
