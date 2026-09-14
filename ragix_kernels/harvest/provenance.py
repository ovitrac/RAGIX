#!/usr/bin/env python3
"""demoE2E step 00 — provenance of the demo corpus, reconstructed and pinned.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

What this step establishes, by content hash rather than by memory:

1. the DCE archive received from the client's contact (``HARVEST_DCE_ZIP``) and its unpacked copy
   (``HARVEST_DCE_DIR``) are the same files, one for one;
2. a per-file manifest of that corpus: path, size, sha256, date stored in the archive,
   duplicate groups (byte-identical files under different names);
3. the corpus-2 archive of record for the eleven R6 twins, since the extracted folders were
   cleaned by the lead on 2026-09-05.

The DCE is public procurement material (lead's ruling, 2026-09-05): file names and hashes are
committed. No file content is written anywhere by this step.

Run (from the lab root)::

    HARVEST_DCE_ZIP=... HARVEST_DCE_DIR=... HARVEST_RECORD_ZIP=... python -m ragix_kernels.harvest.provenance

Fails closed: any file present in the archive and absent or different on disk, or present on
disk and absent from the archive, exits non-zero and is listed. Re-running on an unchanged
corpus rewrites the same manifest (the journal gains one more run record, as it should).
"""

from __future__ import annotations

import collections
import datetime as _dt
import hashlib
import json
import os
import pathlib
import sys
import zipfile

from ragix_kernels.shared.journal import Journal, sha256_file

#: where the manifests are written, and the three inputs — all named by the environment, none here
HERE = pathlib.Path(os.environ.get("HARVEST_PROVENANCE_DIR", "."))
DCE_ZIP = pathlib.Path(os.environ.get("HARVEST_DCE_ZIP", "dce.zip")).expanduser()
DCE_DIR = pathlib.Path(os.environ.get("HARVEST_DCE_DIR", "dce")).expanduser()
C2_ZIP = pathlib.Path(os.environ.get("HARVEST_RECORD_ZIP", "record.zip")).expanduser()

OUT_DCE = HERE / "manifest_dce.json"
OUT_C2 = HERE / "manifest_record_zip.json"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def zip_manifest(zpath: pathlib.Path) -> tuple[list[dict], dict]:
    files = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            data = z.read(info)
            files.append({
                "path": info.filename,
                "size": info.file_size,
                "sha256": sha_bytes(data),
                "zip_date": _dt.datetime(*info.date_time).isoformat(timespec="minutes"),
            })
    by_hash = collections.defaultdict(list)
    for f in files:
        by_hash[f["sha256"]].append(f["path"])
    dup_groups = sorted((v for v in by_hash.values() if len(v) > 1), key=lambda v: v[0])
    return files, {"files": len(files), "distinct": len(by_hash), "duplicate_groups": len(dup_groups),
                   "bytes": sum(f["size"] for f in files), "duplicates": dup_groups}


def disk_manifest(root: pathlib.Path) -> dict[str, dict]:
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = pathlib.Path(dp) / fn
            rel = str(p.relative_to(root.parent))
            out[rel] = {"size": p.stat().st_size, "sha256": sha256_file(p, full=True)[7:]}
    return out


def main() -> int:
    j = Journal("00_provenance", actor="coord")
    j.event("demoE2E.provenance", phase="start",
            io={"dce_zip": sha256_file(DCE_ZIP, full=True), "record_zip": sha256_file(C2_ZIP, full=True)},
            note="reconstructing the provenance of the demo corpus by content hash")

    # --- 1. the DCE archive and its unpacked copy ------------------------------------------
    zfiles, ztot = zip_manifest(DCE_ZIP)
    disk = disk_manifest(DCE_DIR)
    zip_index = {f["path"]: f for f in zfiles}
    missing_on_disk = [p for p in zip_index if p not in disk]
    different = [p for p in zip_index if p in disk and disk[p]["sha256"] != zip_index[p]["sha256"]]
    extra_on_disk = [p for p in disk if p not in zip_index]
    identical = len(zip_index) - len(missing_on_disk) - len(different)

    ext = collections.Counter(pathlib.Path(f["path"]).suffix.lower().lstrip(".") for f in zfiles)
    top = collections.Counter("/".join(f["path"].split("/")[:2]) for f in zfiles)

    manifest = {
        "corpus": os.environ.get("HARVEST_CORPUS_LABEL", "the consultation's DCE"),
        "reuse_policy": "public procurement material — not confidential (lead's ruling 2026-09-05); "
                        "processed locally, rendered transparently",
        "source": {
            "archive": DCE_ZIP.name,
            "sha256": sha256_file(DCE_ZIP, full=True)[7:],
            "size": DCE_ZIP.stat().st_size,
            "downloaded": _dt.datetime.fromtimestamp(DCE_ZIP.stat().st_mtime).isoformat(timespec="minutes"),
            "received_from": os.environ.get("HARVEST_RECEIVED_FROM", "the client's contact"),
            "buyer_side_folder_date": max(f["zip_date"] for f in zfiles),
        },
        "unpacked": {
            "path": str(DCE_DIR),
            "files": len(disk),
            "identical_to_archive": identical,
            "missing_on_disk": missing_on_disk,
            "different_on_disk": different,
            "extra_on_disk": extra_on_disk,
        },
        "totals": {k: v for k, v in ztot.items() if k != "duplicates"},
        "by_extension": dict(ext.most_common()),
        "by_top_folder": dict(top.most_common()),
        "duplicate_groups": ztot["duplicates"],
        "observations": [
            "a diagnostic note counted 601 files; the archive holds 600 "
            "(548 xlsx, 32 pdf, 12 xls, 8 docx) — the note's figure is off by one.",
            "No digestion store or KOAS workspace of this corpus existed on any machine on "
            "2026-09-05: the 27 Aug measurements were made and not persisted.",
        ],
        "files": zfiles,
    }
    OUT_DCE.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # --- 2. the corpus-2 archive of record -------------------------------------------------
    c2files, c2tot = zip_manifest(C2_ZIP)
    c2 = {
        "corpus": "Corpus 2 — historical tender responses (D-0008: customer-confidential, outputs local only)",
        "source": {"archive": C2_ZIP.name, "sha256": sha256_file(C2_ZIP, full=True)[7:],
                   "size": C2_ZIP.stat().st_size,
                   "downloaded": _dt.datetime.fromtimestamp(C2_ZIP.stat().st_mtime).isoformat(timespec="minutes")},
        "layout": sorted({"/".join(f["path"].split("/")[:2]) for f in c2files}),
        "totals": {k: v for k, v in c2tot.items() if k != "duplicates"},
        "events": [
            {"date": "2026-09-05T15:47", "actor": "lead",
             "what": "the extracted folders and their '(2)' twin removed; the archive stays the "
                     "source of record for scripts/qa_twins_round0.json"},
        ],
        "files": [{"path": f["path"], "size": f["size"], "sha256": f["sha256"]} for f in c2files],
    }
    OUT_C2.write_text(json.dumps(c2, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    ok = not (missing_on_disk or different or extra_on_disk)
    j.event("demoE2E.provenance", phase="end",
            io={"dce_zip": sha256_file(DCE_ZIP, full=True), "manifest_dce": sha256_file(OUT_DCE),
                "record_zip": sha256_file(C2_ZIP, full=True), "manifest_record": sha256_file(OUT_C2)},
            metrics={"dce_files": len(zfiles), "dce_distinct": ztot["distinct"],
                     "dce_duplicate_groups": ztot["duplicate_groups"], "dce_bytes": ztot["bytes"],
                     "dce_identical_on_disk": identical, "dce_missing_on_disk": len(missing_on_disk),
                     "dce_different_on_disk": len(different), "dce_extra_on_disk": len(extra_on_disk),
                     "record_files": len(c2files)},
            decision={"success": ok, "gate": "archive ≡ unpacked folder, file by file"})

    print(f"DCE archive: {len(zfiles)} files, {ztot['distinct']} distinct, {ztot['duplicate_groups']} duplicate groups")
    print(f"unpacked copy: {len(disk)} files — identical {identical}, missing {len(missing_on_disk)}, "
          f"different {len(different)}, extra {len(extra_on_disk)}")
    print(f"corpus-2 archive of record: {len(c2files)} files under {len(c2['layout'])} projects")
    for p in missing_on_disk + different + extra_on_disk:
        print("  MISMATCH", p)
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
