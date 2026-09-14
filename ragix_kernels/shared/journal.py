#!/usr/bin/env python3
"""ragix_kernels.shared.journal — the one recorder every step of a run appends to.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Records use the field names of RAGIX's ``koas.event/1.0`` envelope so that the
demo's own steps and the kernels' activity stream read as one grammar. A record
carries hashes, counts and decisions — never document content.

Beyond the envelope, two fields the demo needs and the stream lacks:

* ``versions {lab, ragix}`` — the two commits every record is pinned to;
* ``prev`` — the sha256 of the previous record of the same run, so a removed or
  edited line breaks the chain (``journal.py verify`` walks it). The first record
  of a run carries ``"genesis"``.

Preferred keys inside ``io`` are ``input_hash`` and ``output_hash`` (the stream's
names); other named references are allowed beside them. A step that runs a kernel
sets ``actor.type`` to ``system`` and fills ``kernel {name, version, stage}``; a
record that cites a node fills ``node_ref``.

Library use::

    from ragix_kernels.shared.journal import Journal
    j = Journal("00_provenance", actor="coord")
    j.event("demoE2E.provenance", phase="end",
            io={"input_hash": "sha256:…"}, metrics={"files": 600})

CLI use (from shell steps)::

    python -m ragix_kernels.shared.journal log --step 01_sync --actor coord \
        --scope demoE2E.sync --phase end --note "mirror written" --kv files=1234
    python -m ragix_kernels.shared.journal verify demoE2E/journal/01_sync.jsonl

The journal directory is `<KOAS_JOURNAL_ROOT>/journal` (default: `./journal`), and the two commits
pinned are those of the repository above that root and of this package's checkout.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import uuid

ROOT = pathlib.Path(os.environ.get("KOAS_JOURNAL_ROOT", ".")).resolve()
JOURNAL_DIR = ROOT / "journal"
LAB = ROOT.parent
RAGIX = pathlib.Path(__file__).resolve().parents[2]

SCHEMA = "koas.event/1.0"
GENESIS = "genesis"


def sha256_file(path: os.PathLike | str, full: bool = False) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    return "sha256:" + (d if full else d[:16])


def sha256_bytes(data: bytes, full: bool = False) -> str:
    d = hashlib.sha256(data).hexdigest()
    return "sha256:" + (d if full else d[:16])


def sha256_json(obj, full: bool = False) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"), full)


def record_hash(line: str) -> str:
    """The chain hashes the record exactly as written, one line, no newline."""
    return sha256_bytes(line.rstrip("\n").encode("utf-8"), full=True)


def git_sha(repo: pathlib.Path) -> str:
    """HEAD of ``repo``; suffixed ``-dirty`` when the working tree differs from it."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        # The journal and the runs folder are written BY the steps being pinned; they do
        # not make the code dirty. Everything else does.
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--",
                                ".", ":(exclude)demoE2E/journal", ":(exclude)demoE2E/runs"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 — a missing repo is a fact to record, not to hide
        return "unavailable"


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


class Journal:
    def __init__(self, step: str, actor: str = "coord", run_id: str | None = None,
                 actor_type: str = "operator"):
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        self.step = step
        self.actor = actor
        self.actor_type = actor_type
        self.run_id = run_id or f"{step}_{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.path = JOURNAL_DIR / f"{step}.jsonl"
        self.versions = {"lab": git_sha(LAB), "ragix": git_sha(RAGIX)}
        self.host = socket.gethostname()
        # A shell step calls the CLI once per record, so both the sequence and the chain
        # continue from what the file already holds for this run.
        self.seq = 0
        self.prev = GENESIS
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("run_id") == self.run_id:
                        self.seq = max(self.seq, int(rec.get("seq", 0)))
                        self.prev = record_hash(line)

    def event(self, scope: str, phase: str = "end", *, kernel: dict | None = None,
              node_ref: dict | None = None, **fields) -> dict:
        self.seq += 1
        rec = {
            "v": SCHEMA,
            "ts": now(),
            "event_id": str(uuid.uuid4()),
            "run_id": self.run_id,
            "seq": self.seq,
            "prev": self.prev,
            "actor": {"type": self.actor_type, "id": self.actor, "auth": "none"},
            "scope": scope,
            "phase": phase,
        }
        if kernel:
            rec["kernel"] = kernel
        if node_ref:
            rec["node_ref"] = node_ref
        for k, v in fields.items():
            if v is not None:
                rec[k] = v
        rec["sovereignty"] = {"local_only": True, "host": self.host}
        rec["versions"] = self.versions
        line = json.dumps(rec, ensure_ascii=False, sort_keys=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.prev = record_hash(line)
        return rec


def verify(path: pathlib.Path) -> tuple[int, list[str]]:
    """Walk every run in ``path``; return (records checked, problems).

    Records written before the chain existed carry no ``prev``; they are reported as
    unchained, not as broken — the record says what it is.
    """
    last: dict[str, str] = {}
    seqs: dict[str, int] = {}
    problems: list[str] = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {lineno}: not JSON")
                continue
            n += 1
            run = rec.get("run_id", "?")
            if "prev" not in rec:
                # Written before the chain existed (the first two steps of 2026-09-05, whose
                # first run also restarted seq per call). History is reported, never checked
                # against rules that did not exist when it was written, and never rewritten.
                problems.append(f"line {lineno}: run {run} seq {rec.get('seq')} unchained (written before the chain)")
                seqs[run] = int(rec.get("seq", 0))
                last[run] = record_hash(line)
                continue
            expected_seq = seqs.get(run, 0) + 1
            if rec.get("seq") != expected_seq:
                problems.append(f"line {lineno}: run {run} seq {rec.get('seq')} (expected {expected_seq})")
            seqs[run] = int(rec.get("seq", expected_seq))
            expected_prev = last.get(run, GENESIS)
            if rec["prev"] != expected_prev:
                problems.append(f"line {lineno}: run {run} seq {rec.get('seq')} chain broken")
            last[run] = record_hash(line)
    return n, problems


def _parse_kv(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except Exception:  # noqa: BLE001 — a bare string is a valid value
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    log = sub.add_parser("log", help="append one record")
    log.add_argument("--step", required=True)
    log.add_argument("--actor", default="coord")
    log.add_argument("--actor-type", default="operator", choices=["operator", "system", "auditor"])
    log.add_argument("--run-id", default=None)
    log.add_argument("--scope", required=True)
    log.add_argument("--phase", default="end")
    log.add_argument("--note", default=None)
    log.add_argument("--kv", nargs="*", default=[], help="metrics as key=value (JSON values accepted)")
    log.add_argument("--io", nargs="*", default=[], help="io refs as key=value; prefer input_hash / output_hash")
    log.add_argument("--decision", nargs="*", default=[], help="decision fields as key=value")
    log.add_argument("--kernel", nargs="*", default=[], help="kernel fields: name=… version=… stage=…")
    log.add_argument("--node-ref", nargs="*", default=[], help="node reference fields: level=… id=…")
    sha = sub.add_parser("sha", help="print sha256 of files")
    sha.add_argument("paths", nargs="+")
    sha.add_argument("--full", action="store_true")
    ver = sub.add_parser("verify", help="walk the chain of every run in a journal file")
    ver.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd == "sha":
        for p in args.paths:
            print(sha256_file(p, full=args.full), p)
        return 0

    if args.cmd == "verify":
        rc = 0
        for p in args.paths:
            n, problems = verify(pathlib.Path(p))
            broken = [x for x in problems if "unchained" not in x]
            unchained = len(problems) - len(broken)
            print(f"{p}: {n} records, {unchained} unchained, {len(broken)} problems")
            for x in broken:
                print("  ", x)
            rc = rc or (1 if broken else 0)
        return rc

    j = Journal(args.step, actor=args.actor, run_id=args.run_id, actor_type=args.actor_type)
    rec = j.event(args.scope, phase=args.phase,
                  kernel=_parse_kv(args.kernel) or None,
                  node_ref=_parse_kv(args.node_ref) or None,
                  metrics=_parse_kv(args.kv) or None,
                  io=_parse_kv(args.io) or None,
                  decision=_parse_kv(args.decision) or None,
                  note=args.note)
    print(rec["event_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
