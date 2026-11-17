"""
Copy a staged input set byte for byte and add tracked files to it as contrast inputs.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python extend_inputs.py --from <bench> --to <new bench> --repo <RAGIX checkout> --add <id>=<path> [...]

Every file of <bench>/inputs is copied and checked against its manifest by sha256 (a mismatch is refused),
so a new run reads exactly the bytes an earlier run read. Each `--add` copies a file that `git ls-files`
lists in <repo> to inputs/contrast/, labelled "contrast: not for the sheet" in the new manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", required=True)
    parser.add_argument("--to", dest="target", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--add", action="append", default=[])
    args = parser.parse_args()
    source, target, repo = Path(args.source), Path(args.target), Path(args.repo)
    if (target / "inputs").exists():
        raise SystemExit(f"refused: {target / 'inputs'} exists")
    manifest = json.loads((source / "inputs" / "MANIFEST.json").read_text())
    for entry in manifest["inputs"]:
        src, dst = source / entry["path"], target / entry["path"]
        if sha256(src) != entry["sha256"]:
            raise SystemExit(f"refused: {src} does not match its manifest")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    tracked = set(subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "ls-files"], check=True,
                                 capture_output=True, text=True).stdout.splitlines())
    for item in args.add:
        ident, rel = item.split("=", 1)
        if rel not in tracked:
            raise SystemExit(f"refused: {rel} is not tracked in {repo}")
        dst = target / "inputs" / "contrast" / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / rel, dst)
        manifest["inputs"].append({"id": ident, "path": str(dst.relative_to(target)), "repo_path": rel,
                                   "origin": "contrast: tracked, not for the sheet", "synthetic": False,
                                   "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    manifest["extended_from"] = sha256(source / "inputs" / "MANIFEST.json")
    (target / "inputs" / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    for entry in manifest["inputs"]:
        print(f"{entry['sha256'][:16]}  {entry['bytes']:>9}  {entry['path']}")


if __name__ == "__main__":
    main()
