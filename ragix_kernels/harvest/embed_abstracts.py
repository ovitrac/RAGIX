#!/usr/bin/env python3
"""demoE2E 26 — the E layer: every pass-1 abstract embedded once, on the executor's own embedder.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

RUNBOOK row 26 is the spec. The lead, 2026-09-11 19:2x: "I need a pyramid of knowledge and graph-RAG,
embeddings are required on node abstracts." One vector per abstracted node, with the store's own embedder
(`snowflake-arctic-embed2`, 1 024-d) as the executor serves it, through /api/embed on loopback — the same
embedder on the master would not replay, so this runs on the executor only. The vectors live in their own
space and never join the leaf lane.

Gates, each a refusal with its reason: the host is loopback; the embedder's tag is read from the server's
own /api/tags, never typed; no other model is resident (unless --allow-resident); every vector is exactly
the declared width; `truncate` is false, so an over-long abstract fails loudly rather than being cut.

Outputs, in input order: `node_embeddings.jsonl` (node_id, level, pass, embedder, digest, dim, made_on,
abstract_sha256, source_sha256, row) and `node_embeddings.npy` (float32, one row per line); with
--sidecar, the same vectors in the sidecar's `node_embeddings` table (D-0023). --resume embeds only what is
not there; --check N re-embeds the first N and compares bytes, which is the row's exit criterion; --smoke N
embeds N into out/smoke and touches nothing else. The model is released in a `finally`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

EMBEDDER = "snowflake-arctic-embed2"
DIM = 1024
LOOPBACK = re.compile(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$")


def refuse(message: str) -> None:
    sys.stderr.write(f"embed_abstracts: REFUSED — {message}\n")
    raise SystemExit(2)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get(host: str, path: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(host + path), timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post(host: str, path: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(host + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def load_abstracts(paths: list[str]) -> list[dict]:
    """Every node with an abstract, in file then line order. A node met twice with two texts is refused."""
    seen: dict[str, dict] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            text = (r.get("abstract") or "").strip()
            if not r.get("ok") or not text:
                continue
            rec = {"node_id": r["node_id"], "level": r.get("level"), "text": text,
                   "abstract_sha256": sha256_text(text), "source_sha256": r.get("source_sha256")}
            if r["node_id"] in seen and seen[r["node_id"]]["abstract_sha256"] != rec["abstract_sha256"]:
                refuse(f"{r['node_id']} carries two different abstracts across the inputs")
            seen.setdefault(r["node_id"], rec)
    return list(seen.values())


def embed(host: str, tag: str, texts: list[str], dim: int, timeout: float) -> np.ndarray:
    payload = post(host, "/api/embed", {"model": tag, "input": texts, "truncate": False}, timeout)
    vectors = payload.get("embeddings") or []
    if len(vectors) != len(texts):
        refuse(f"{len(texts)} texts sent, {len(vectors)} vectors returned")
    for v in vectors:
        if len(v) != dim:
            refuse(f"a vector of {len(v)} dimensions, {dim} declared")
    return np.asarray(vectors, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="row 26 — embed the pass-1 abstracts")
    ap.add_argument("--abstracts", nargs="+", required=True, help="pass-1 jsonl files (core, rest)")
    ap.add_argument("--out", default="demoE2E/verify/pass1/outputs_emb")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--embedder", default=EMBEDDER, help="the family; the exact tag is read from /api/tags")
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check", type=int, default=0, help="re-embed the first N and compare bytes")
    ap.add_argument("--smoke", type=int, default=0, help="embed N into out/smoke and stop")
    ap.add_argument("--sidecar", default="", help="also write the sidecar's node_embeddings table")
    ap.add_argument("--source-root", default="demoE2E/runs/02_collect/20260906T163718/run/saqqara.db")
    ap.add_argument("--source-sha256", default="53ff3f20f655ad663e1c55a41189e2e1f427b0ed26b221edec9dad2a6a5229f8")
    ap.add_argument("--allow-resident", action="store_true")
    a = ap.parse_args(argv)

    if not LOOPBACK.match(a.host):
        refuse(f"{a.host} is a remote host: this embeds with the executor's own server, on loopback")
    tags = {m.get("name"): m.get("digest", "") for m in get(a.host, "/api/tags").get("models", [])}
    tag = next((t for t in sorted(tags) if t == a.embedder or t.startswith(a.embedder + ":")), None)
    if tag is None:
        refuse(f"the embedder {a.embedder} is absent from this server's models: {sorted(tags)[:6]}")
    resident = [m.get("name") for m in get(a.host, "/api/ps").get("models", []) if m.get("name") != tag]
    if resident and not a.allow_resident:
        refuse(f"another model is resident ({resident}): this job is exclusive")

    records = load_abstracts(a.abstracts)
    out = Path(a.out) / ("smoke" if a.smoke else "")
    if a.smoke:
        records = records[:a.smoke]
    out.mkdir(parents=True, exist_ok=True)
    jsonl, npy = out / "node_embeddings.jsonl", out / "node_embeddings.npy"
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()] if a.resume and jsonl.exists() else []
    matrix = np.load(npy) if rows else np.zeros((0, a.dim), dtype=np.float32)
    done = {(r["node_id"], r["abstract_sha256"], r["embedder"]) for r in rows}
    todo = [r for r in records if (r["node_id"], r["abstract_sha256"], tag) not in done]
    made_on = socket.gethostname()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            vecs = embed(a.host, tag, [r["text"] for r in chunk], a.dim, a.timeout)
            for r in chunk:
                rows.append({"node_id": r["node_id"], "level": r["level"], "pass": 1, "embedder": tag,
                             "digest": tags[tag][:12], "dim": a.dim, "made_on": made_on,
                             "abstract_sha256": r["abstract_sha256"], "source_sha256": r["source_sha256"],
                             "row": len(rows)})
            matrix = np.vstack([matrix, vecs])
            jsonl.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows),
                             encoding="utf-8")                        # on disk as it goes: a job that dies keeps
            np.save(npy, matrix)                                      # what it already paid for
        check = None
        if a.check and rows:
            by_id = {r["node_id"]: r["text"] for r in records}
            probe = [r for r in rows[:a.check] if r["node_id"] in by_id]
            again = embed(a.host, tag, [by_id[r["node_id"]] for r in probe], a.dim, a.timeout)
            stored = matrix[[r["row"] for r in probe]]
            same = [again[k].tobytes() == stored[k].tobytes() for k in range(len(probe))]
            check = {"n": len(probe), "identical": sum(same),
                     "max_abs_diff": float(np.max(np.abs(again - stored))) if len(probe) else 0.0}
            (out / "check.json").write_text(json.dumps(check, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        try:                                                          # released even on SIGINT
            post(a.host, "/api/embed", {"model": tag, "keep_alive": 0}, 30.0)
        except (urllib.error.URLError, TimeoutError, OSError):
            pass

    if a.sidecar and not a.smoke:
        from .derived import DerivedStore
        store = DerivedStore(a.sidecar, a.source_root, a.source_sha256, run_id=f"26_embed_{now}")
        written = sum(store.write_embedding(node_id=r["node_id"], pass_=1, embedder=r["embedder"],
                                            digest=r["digest"], dim=r["dim"], made_on=r["made_on"],
                                            vector=matrix[r["row"]].tobytes(), abstract_sha256=r["abstract_sha256"],
                                            source_sha256=r["source_sha256"], created_at=now) for r in rows)
        store.commit()
        store.close()
    summary = {"embedder": tag, "digest": tags[tag][:12], "dim": a.dim, "made_on": made_on, "nodes": len(rows),
               "embedded_this_run": len(todo), "inputs": {p: sha256_text(Path(p).read_text(encoding="utf-8"))
                                                          for p in a.abstracts},
               "npy_sha256": hashlib.sha256(npy.read_bytes()).hexdigest() if npy.exists() else None,
               "check": check}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("embedder", "dim", "nodes", "embedded_this_run", "check")}))
    return 1 if check and check["identical"] != check["n"] else 0


if __name__ == "__main__":
    sys.exit(main())
