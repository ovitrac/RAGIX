"""
Plan the downloads of the pinned environment (M2b) without downloading anything.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python m2b_plan_downloads.py <out.json> [<requirement> ...]

Requirements given after <out.json> replace the default roots, e.g. exact pins for every package.

Reads PyPI's JSON API (package metadata only; no file is written but <out.json>), resolves the closure of
ragix[saqqara,explorer] at 20fdf7a's bounds for CPython 3.12 on manylinux aarch64, and lists each
package's chosen version, wheel file, size and sha256. A package with no matching wheel is reported, not
guessed. The result moves with PyPI: every bound except pymupdf==1.27.2.2 takes the latest release.
"""
import json, re, sys, urllib.request
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from packaging.tags import Tag
from packaging.utils import parse_wheel_filename, canonicalize_name

ROOTS = ["requests>=2.31.0", "pyyaml>=6.0", "python-docx>=1.0.0", "python-pptx>=0.6.21", "openpyxl>=3.1.0",
         "odfpy>=1.4.1", "pypdf>=4.0.0", "pypdfium2>=4.0.0", "numpy>=1.24.0", "pymupdf==1.27.2.2"]
ENV = {"python_version": "3.12", "python_full_version": "3.12.13", "sys_platform": "linux",
       "platform_machine": "aarch64", "platform_system": "Linux", "os_name": "posix",
       "implementation_name": "cpython", "platform_python_implementation": "CPython", "extra": ""}
def wheel_ok(fn):
    try:
        _, _, _, tags = parse_wheel_filename(fn)
    except Exception:
        return False
    for t in tags:
        if t.interpreter in ("cp312", "py3", "py2", "cp310", "cp311", "cp39", "cp38") and \
           (t.abi in ("cp312", "none", "abi3")) and \
           (t.platform == "any" or ("aarch64" in t.platform and "manylinux" in t.platform)):
            if t.abi == "abi3" or t.abi == "none" or t.interpreter == "cp312":
                return True
    return False

def meta(name, version=None):
    url = f"https://pypi.org/pypi/{name}/json" if version is None else f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)

chosen, queue = {}, [Requirement(r) for r in (sys.argv[2:] or ROOTS)]
while queue:
    req = queue.pop(0)
    name = canonicalize_name(req.name)
    if req.marker and not req.marker.evaluate(ENV):
        continue
    if name in chosen:
        if not req.specifier.contains(chosen[name]["version"], prereleases=False):
            print("CONFLICT", name, req.specifier, chosen[name]["version"], file=sys.stderr)
        continue
    top = meta(name)
    versions = sorted((Version(v) for v, files in top["releases"].items()
                       if files and not Version(v).is_prerelease and req.specifier.contains(v)), reverse=True)
    pick = None
    for v in versions:
        files = top["releases"][str(v)]
        wheels = [f for f in files if f["packagetype"] == "bdist_wheel" and wheel_ok(f["filename"]) and not f.get("yanked")]
        if wheels:
            pick = (v, sorted(wheels, key=lambda f: f["size"])[0]); break
    if pick is None:
        print("NO WHEEL", name, req.specifier, file=sys.stderr); continue
    v, f = pick
    info = meta(name, str(v))["info"]
    chosen[name] = {"version": str(v), "file": f["filename"], "bytes": f["size"], "sha256": f["digests"]["sha256"],
                    "url": f["url"], "license": (info.get("license_expression") or info.get("license") or "")[:40],
                    "required_by": str(req)}
    for d in info.get("requires_dist") or []:
        r = Requirement(d)
        if r.marker is None or r.marker.evaluate(ENV):
            queue.append(r)

total = sum(c["bytes"] for c in chosen.values())
for n, c in sorted(chosen.items()):
    print(f"{n:22s} {c['version']:12s} {c['bytes']:>10d}  {c['file']}  [{c['license']}]")
print(f"TOTAL {len(chosen)} wheels, {total} bytes ({total/1e6:.1f} MB)")
json.dump({"python": "3.12.13", "platform": "manylinux aarch64", "packages": chosen, "total_bytes": total},
          open(sys.argv[1], "w"), indent=1, sort_keys=True)
