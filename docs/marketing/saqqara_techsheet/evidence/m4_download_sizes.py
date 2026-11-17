"""
The files pip actually fetched, with their sizes: from pip's install reports, sizes from the indexes.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

    python m4_download_sizes.py <out.json> <pip report.json> [<pip report.json> ...]

For every item of every report: the package, its version, the file name, the URL and the sha256 that
pip checked. The size is read from PyPI's JSON listing for files.pythonhosted.org, and from an HTTP HEAD
(Content-Length) for any other host; no file is downloaded. A wheel pip built locally from a source
archive is listed with the archive's size.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request


def pypi_size(name: str, version: str, filename: str):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
        for f in json.load(response)["urls"]:
            if f["filename"] == filename:
                return f["size"]
    return None


def head_size(url: str):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers.get("Content-Length", 0)) or None


def main() -> None:
    out_path, reports = sys.argv[1], sys.argv[2:]
    rows = []
    for report in reports:
        for item in json.load(open(report))["install"]:
            url = item["download_info"]["url"]
            filename = urllib.parse.unquote(url.rsplit("/", 1)[1])
            name, version = item["metadata"]["name"], item["metadata"]["version"]
            host = urllib.parse.urlparse(url).netloc
            size = (pypi_size(name, version, filename) if host == "files.pythonhosted.org" else head_size(url))
            rows.append({"report": report.rsplit("/", 1)[-1], "name": name, "version": version, "file": filename,
                         "host": host, "bytes": size,
                         "sha256": (item["download_info"].get("archive_info") or {}).get("hashes", {}).get("sha256")})
    total = sum(r["bytes"] or 0 for r in rows)
    by_host = {}
    for r in rows:
        by_host[r["host"]] = by_host.get(r["host"], 0) + (r["bytes"] or 0)
    for r in sorted(rows, key=lambda r: -(r["bytes"] or 0)):
        print(f"{r['name']:28s} {r['version']:14s} {r['bytes'] or 0:>11d}  {r['host']}")
    print(f"TOTAL {len(rows)} files, {total} bytes; by host: {by_host}")
    json.dump({"files": rows, "total_bytes": total, "by_host": by_host}, open(out_path, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
