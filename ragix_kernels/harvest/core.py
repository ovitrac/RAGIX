#!/usr/bin/env python3
"""M10 — pass 1 over the contractual core: an abstract per node, bottom-up (RUNBOOK row 22, seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE RUN (rule T4), on the coordinating seat's word of 2026-09-11 carrying the lead's
mandate, as WP §8.2 describes pass 1. The rules R1 (budget by construction) and R2 (the digit and date
rule) are **imported** from `pass1.py`, never copied, and were shown to fail on the smoke's own
fabrication first (`check_rules.py`: `granite3.1-moe:3b` wrote « 11 septembre 2027 à 12h00 » over an
object whose only such date is 2026-09-11T12:00).

Q0 frame. The frozen store (53ff3f20f655ad66, empty WAL, mode=ro&immutable=1) and a **document set
   given by `--select`**, so the same code runs over another set on another machine (the coordinating
   seat's word, 2026-09-11, for the bench's run over the rest of the DCE):
     `core`             the 26 pieces `pieces.yaml` files under the roles CCTP, CCAP and RC —
                        **217 level-1 windows and 26 roll-ups, gated**: the numbers are declared and a
                        mismatch refuses, as they were when this script was written;
     `rest`             every document of the store that is **not** one of those 26 — the complement,
                        counted at run time and recorded, never gated against a number typed here;
     `roles:A,B`        the roles named, as `pieces.yaml` files them;
     `paths:FILE`       one source path per line, each matched against the store's `source_path` tail.
   A document with more than one level-1 roll-up — a workbook carries one per sheet — is **skipped and
   listed** in the summary's `skipped_multi_rollup`, because what a document node means for a workbook is
   the lead's question. The core has none. `--sheets-as-nodes` answers that question the way the store
   already answers it (the coordinating seat's word, 2026-09-11): **each sheet roll-up is a node** of
   level `sheet`, written from its windows' abstracts when it has windows and from its own text
   otherwise, and the workbook's document node — id `doc_<doc_id[:12]>`, which is synthetic because the
   store gives a workbook no single roll-up — is written from its sheets' abstracts. It is the same
   recurrence one level lower. Every sheet record carries `caveat`: an xlsx roll-up can repeat text,
   because the store's oversize fallback windows overlap by 200 characters at each seam (M2's substrate
   finding), so a sheet abstract is orientation about a text that may say a thing twice.
   `--expect windows,documents` gates any selection against a pair the caller declares; `core` sets that
   pair itself. `--host` must be loopback: this seat never calls a **remote** Ollama, which on the
   executor means the executor's own server and on the master the master's.
Q1 the three stages, bottom-up, 244 calls in all:
   (a) the 217 windows, each from **its own text**;
   (b) the 26 documents: the 24 that have windows from **their windows' abstracts in order**, the 2 that
       have none (the CCAP's annex 5, the DCE update notice) from their own text;
   (c) the DCE node from **the 26 document abstracts in piece order**.
   A stage reads only what the stage below produced. A node whose children all refused is skipped and
   recorded as such — never written from the text as a silent fallback.
Q2 the call. The master's Ollama only (loopback, gated), granite4.2:8b-chat, digest 0457c45172f4,
   `stream` false, `think` false, options `temperature` 0, `seed` 0, `num_ctx` 8192,
   `num_predict` = pass1.NUM_PREDICT (140, near the budget). One call per node, no retry; a timeout or a
   transport error is recorded as a refusal with its reason and the run continues. The two prompts are
   PROMPT_TEXT and PROMPT_ROLLUP below and their sha256 goes into every record.
Q3 the record, one JSON line per node written **as it completes** (`core_abstracts.jsonl`): `node_id`,
   `level` (window | document | dce), `piece`, `children` (ids), `model`, `prompt_sha256`, `abstract`,
   `words`, `over_budget`, `quantities_checked`, `dates_checked`, `sentences_dropped`, `dropped`,
   `sentences_trimmed`, `seconds`, `prompt_eval_count`, `eval_count`, `made_on`, `source_sha256`, `ok`,
   `refusal`. A re-run with --resume skips node ids already in the file, so an interruption costs the
   nodes in flight and nothing more.
Q4 exit criteria (the coordinating seat's, tonight): **at least 90 % of nodes with an in-budget abstract
   and no dropped sentence**, and the run **stops and records past 60 minutes** (STOP_AFTER), whatever
   remains. Both are reported, never adjusted.
Gates (refuse, exit 2): the store's prefix and an empty WAL; a non-loopback host; the model absent or
   carrying another digest;
   the window and roll-up counts against Q0; `ollama ps` not empty unless `--allow-resident` is given
   **on purpose** for a second process sharing the model (it is not implied by `--resume`, so a long
   resumed job keeps the gate); every node's source text re-read from the store.
AMENDED 2026-09-11 ~19:0x, on the lead's word ("150, 350, 800 - go") relayed by the coordinating seat
while this seat was away — the agent's edit, left for this seat's review, since no kernel is gated by the
author of its change. The ceiling is per level: LADDER window 150, sheet 150, document 350, dce 800,
the model choosing the length under it by the content; each prompt states its level's ceiling in words; `--budget-window`, `--budget-sheet`,
`--budget-document`, `--budget-dce` set it (defaults as given); every record carries `ceiling` and
`num_predict`, the summary the ladder and each level's count of `done_reason: length`; and the release of
the model now sits in a `finally`, because row 24's clean stop on SIGINT left it resident. The generation cap is
`num_predict_for(ceiling)`, about 2.2 tokens a French word plus a quarter, because two of row 24's
fourteen windows ended at the old 140 with `done_reason: length` — truncation must never be what binds.
The rule itself is unchanged: the longest prefix of whole sentences under the ceiling, a first sentence
over it kept and recorded `over_budget`. R2 is unchanged.

REVIEWED AND KEPT by the testing seat on the merge of 2026-09-11: the ladder, the per-level `--budget-*`
flags, the prompts stating their own ceiling, the records carrying `ceiling` and `num_predict`, the
per-level count of `done_reason: length` and the release in a `finally` are the agent's and are better
than what this seat had on its branch. Added here: **R1b** (a record whose `done_reason` is `length` is
refused by `pass1.abstract_of`, and still counted in the summary), **R3** substitution at window and sheet
grain only — a child's value id is not resolvable at a parent, so a roll-up gets no markers and a figure
it writes that is absent from its children's abstracts is dropped as before — and **R4**, the trap flags
of `traps.json` on every card whose byte range contains the span. `--traps` is a declared input and its
absence refuses.

**R1c, the input truncation gate**, added on the bench's finding of 2026-09-11: row 24's DCE node was
given 707 085 characters, the server cut the prompt to `num_ctx` — 198 853 tokens read as 8 192 — and the
record was stored **ok and clean**, a card written from four per cent of its input with nothing in the
record to say so. A record whose `prompt_eval_count` is at or above the call's `num_ctx` is therefore a
refusal, « input truncated by the server », never `ok`; `--num-ctx-<level>` sets the window per level so
the bench can raise it where the machine allows, and the refusal is what makes an unsized level visible
instead of plausible. Sizing the window from the input rather than refusing it is a design change and
belongs to the intermediate rung the lead is being asked to rule on, not to this gate.

Outputs: `core_abstracts.jsonl` (as it runs), `core_pass1.md` (for reading, abstracts verbatim with node
ids and pieces — the DCE is public), `core_summary.json` (the declaration, the counts, the verdict).
The journal record (`demoE2E/journal/22_pass1.jsonl`) is written by the caller through `journal.py`:
counts and hashes only, never a line of corpus text.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import hashlib
import json
import re
import sqlite3
import statistics
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NoReturn

from . import pass1 as P

#: the consultation's reference, as its file names and running headers print it. Read from the
#: environment because a reference names one consultation and this code names none.
CONSULTATION = os.environ.get("HARVEST_CONSULTATION", "REF-1")
STORE_SHA_PREFIX = "53ff3f20f655ad66"
MODEL = "granite4.2:8b-chat"
DIGEST = "0457c45172f4"
CORE_ROLES = ("CCTP", "CCAP", "RC")
CORE_WINDOWS = 217
CORE_ROLLUPS = 26
STOP_AFTER = 3600.0
TIMEOUT = 180.0
PASS_RATE = 0.90
PROMPT_TEXT = """Tu lis un extrait d'un document d'un marché public français (dossier de consultation).

Rédige en français un résumé en texte brut, {ceiling} mots au maximum — choisis toi-même sa longueur selon
ce que contient l'extrait, sans chercher à atteindre ce plafond : de quoi traite cet
extrait, sur quelle prestation ou quel équipement il porte, et ce qu'il impose au titulaire.

Contraintes :
- n'invente aucun chiffre et aucune date : ne cite un nombre ou une date que s'il figure dans l'extrait ;
- pas de liste, pas de titre, pas de guillemets, aucun commentaire sur ta réponse.

Extrait :
{text}
"""
PROMPT_ROLLUP = """Voici les résumés des parties d'un même document d'un marché public français, dans
l'ordre du document.

Rédige en français un résumé d'ensemble en texte brut, {ceiling} mots au maximum — choisis toi-même sa
longueur selon ce que contiennent les résumés, sans chercher à atteindre ce plafond : de quoi
traite ce document et ce qu'il impose au titulaire.

Contraintes :
- n'invente aucun chiffre et aucune date : ne cite un nombre ou une date que s'il figure dans les résumés ;
- pas de liste, pas de titre, pas de guillemets, aucun commentaire sur ta réponse.

Résumés des parties :
{text}
"""


def fail(msg: str) -> NoReturn:
    sys.stderr.write(f"core: REFUSED — {msg}\n")
    sys.exit(2)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


#: Every sheet record carries this, as the docstring above promises. It was assigned to the returned
#: dict AFTER `call` had already serialised the record, so the JSONL never carried it while the
#: in-memory dict did — found on the bench, 2026-09-12, on a running stage 2. One text, one place.
SHEET_CAVEAT = ("an xlsx roll-up can repeat text: the store's oversize fallback windows overlap by "
                "200 characters at each seam (M2)")


def post(host: str, path: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(host + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get(host: str, path: str, timeout: float = 20.0) -> dict:
    with urllib.request.urlopen(host + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def render_prompt(level: str, text: str, ladder: dict[str, int]) -> str:
    """The prompt a node of this level receives, stating its own ceiling in words."""
    return (PROMPT_TEXT if level == "window" else PROMPT_ROLLUP).format(text=text, ceiling=ladder[level])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M10 — pass 1 over the core (S4).")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--lab", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--traps", type=Path, default=None,
                    help="traps.json (default: <lab>/demoE2E/verify/family/outputs/traps.json) — R4")
    for level, default in (("window", 8192), ("sheet", 8192), ("document", 8192),
                           ("sub_family", 8192), ("family", 16384), ("dce", 16384)):
        ap.add_argument(f"--num-ctx-{level}", type=int, default=default, dest=f"num_ctx_{level}",
                        help=f"the {level} context window (default {default}) — R1c refuses an input "
                             f"the server had to cut")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--select", default="core",
                    help="core | rest | roles:A,B | paths:FILE (see Q0)")
    ap.add_argument("--expect", default=None, help="windows,documents — the caller's own gate")
    ap.add_argument("--stop-after", type=float, default=STOP_AFTER, dest="stop_after",
                    help="seconds; the run stops and records past it (default 3600)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--sheets-as-nodes", action="store_true", dest="sheets_as_nodes",
                    help="a workbook's sheet roll-ups become nodes; its document node comes from them")
    ap.add_argument("--allow-resident", action="store_true", dest="allow_resident",
                    help="proceed although a model is resident — for a second process sharing the same "
                         "model on purpose, never to get past the gate by accident")
    ap.add_argument("--limit", type=int, default=0, help="stop after n window calls (a dry check)")
    ap.add_argument("--families", type=Path, default=None,
                    help="families.json (default: <lab>/demoE2E/verify/family/outputs/families.json) — "
                         "the tree row 30 reads; with --select families this is a declared input")
    ap.add_argument("--cards", type=Path, nargs="*", default=[],
                    help="the document abstracts a family rung reads: the core_abstracts.jsonl of the "
                         "runs below it. A family's children are abstracts already on disk, never a "
                         "node re-summarised here")
    ap.add_argument("--expect-families", default="92,13,1", dest="expect_families",
                    help="sub_families,families,dce — the gate refuses on a mismatch, because a tree "
                         "that has changed shape since it was sized is a tree that must be re-sized")
    ap.add_argument("--workers", type=int, default=1,
                    help="calls in flight at once (default 1), at the window stage and at each "
                         "sub-family BAND — the two stages whose calls are independent of one another. "
                         "The levels still run in order, a roll-up never starting before its children "
                         "exist, and a family waits for its own sub-families. Above 1 the records are "
                         "written in COMPLETION order, so the "
                         "file's byte order is no longer the node order; the summary sorts, and "
                         "`--workers 4` must produce the same record SET as `--workers 1`, which "
                         "tests/test_core_workers.py asserts against a fake server. Note what this "
                         "does NOT buy: with OLLAMA_NUM_PARALLEL 1 the server queues the requests and "
                         "the wall time is unchanged — the flag is for a server that answers several "
                         "at once, and it is measured before it is believed.")
    for level, default in P.LADDER.items():
        ap.add_argument(f"--budget-{level}", type=int, default=default, dest=f"budget_{level}",
                        help=f"the {level} ceiling in words (default {default}, the lead's ladder)")
    a = ap.parse_args(argv)
    budgets = {level: getattr(a, f"budget_{level}") for level in P.LADDER}
    #: the family rungs summarise abstracts exactly as a document rung does, so they take the document
    #: ceiling; the DCE keeps its own. Declared here rather than in P.LADDER, which rows 22/24/29 read.
    budgets.setdefault("sub_family", budgets["document"])
    budgets.setdefault("family", budgets["document"])

    if not re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", a.host):
        fail(f"host {a.host} is not the master's loopback: this seat never calls the executor")
    store_sha = sha256_file(a.store)
    if not store_sha.startswith(STORE_SHA_PREFIX):
        fail(f"store sha256 {store_sha[:16]} is not {STORE_SHA_PREFIX}")
    wal = Path(str(a.store) + "-wal")
    if wal.exists() and wal.stat().st_size:
        fail("the store's WAL is not empty")
    try:
        tags = get(a.host, "/api/tags")
        resident = get(a.host, "/api/ps").get("models", [])
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        fail(f"the master's Ollama does not answer at {a.host}: {exc}")
    available = {m["name"]: m.get("digest", "") for m in tags.get("models", [])}
    model = a.model
    if model not in available:
        fail(f"{model} is not on the Ollama at {a.host}")
    if model == MODEL and not available[model].startswith(DIGEST):
        fail(f"{model} carries digest {available[model][:12]}, not the declared {DIGEST}")
    if resident and not a.allow_resident:
        fail("ollama ps is not empty: " + ", ".join(m["name"] for m in resident)
             + " — the GPU must be free before pass 1 starts, or --allow-resident given on purpose")

    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    from .pieces import load_pieces

    mapped = load_pieces(a.lab / "demoE2E/03_analyze/pieces.yaml")
    docs = db.execute("select doc_id, source_path from documents").fetchall()
    core_paths = [p for role in CORE_ROLES for p in sorted(mapped[role])]
    role_of = {p: role for role in CORE_ROLES for p in mapped[role]}
    if a.select == "core":
        wanted = list(core_paths)
    elif a.select == "rest":
        core_ids = {d[0] for d in docs if any(d[1].endswith("/" + p) for p in core_paths)}
        wanted = [d[1] for d in docs if d[0] not in core_ids]
    elif a.select.startswith("roles:"):
        wanted = [p for role in a.select.split(":", 1)[1].split(",") for p in sorted(mapped[role.strip()])]
    elif a.select == "families":
        wanted = []                                        # row 30 reads the tree, not the store
    elif a.select.startswith("paths:"):
        listing = Path(a.select.split(":", 1)[1])
        if not listing.exists():
            fail(f"{listing} missing")
        wanted = [line.strip() for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        fail(f"--select {a.select!r} is not core | rest | roles:A,B | paths:FILE")
    pieces: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in wanted:
            hits = [d for d in docs if d[1] == path or d[1].endswith("/" + path)]
            if len(hits) != 1:
                fail(f"{path}: {len(hits)} store documents")
            doc_id = hits[0][0]
            rolls = db.execute("select chunk_id, text from chunks where doc_id=? and level=1 "
                               "and parent_id is null order by seq", (doc_id,)).fetchall()
            if len(rolls) != 1 and not a.sheets_as_nodes:
                skipped.append({"path": path, "doc_id": doc_id, "rollups": len(rolls)})
                continue
            if not rolls:
                skipped.append({"path": path, "doc_id": doc_id, "rollups": 0})
                continue
            sheets = []
            for roll_id, roll_body in rolls:
                wins = db.execute("select chunk_id, text from chunks where parent_id=? and level=1 "
                                  "order by seq", (roll_id,)).fetchall()
                sheets.append({"roll_id": roll_id, "roll_text": roll_body, "windows": wins})
            pieces.append({"role": role_of.get(path, "other"), "path": path,
                           "piece": path.split("/")[-1], "doc_id": doc_id, "sheets": sheets,
                           "multi": len(sheets) > 1,
                           "node_id": sheets[0]["roll_id"] if len(sheets) == 1
                           else f"doc_{doc_id[:12]}",
                           "windows": [w for sh in sheets for w in sh["windows"]]})
    n_win = sum(len(p["windows"]) for p in pieces)
    expect: tuple[int, int] | None = None
    if a.select == "core":
        expect = (CORE_WINDOWS, CORE_ROLLUPS)
    if a.expect:
        try:
            w, d = (int(x) for x in a.expect.split(","))
        except ValueError:
            fail(f"--expect {a.expect!r} is not 'windows,documents'")
        expect = (w, d)
    if expect and (n_win, len(pieces)) != expect:
        fail(f"{len(pieces)} documents and {n_win} windows, {expect[1]} and {expect[0]} declared")
    print(json.dumps({"select": a.select, "documents": len(pieces), "windows": n_win,
                      "skipped_multi_rollup": len(skipped),
                      "gated_against": list(expect) if expect else None}))

    traps_path = a.traps or (a.lab / "demoE2E/verify/family/outputs/traps.json")
    if not traps_path.exists():
        fail(f"{traps_path} missing: the trap flags of R4 are a declared input")
    trap_spans: dict[str, list[dict[str, Any]]] = {}
    for trap in json.loads(traps_path.read_text(encoding="utf-8"))["traps"]:
        for span in trap.get("spans") or []:
            trap_spans.setdefault(span["chunk_id"], []).append(
                {"id": trap["id"], "question_fr": trap["question_fr"], "bytes": span["bytes"]})

    def traps_in(roll_id: str, roll_text: str, c0: int = 0, c1: int | None = None) -> list[dict[str, Any]]:
        """R4: the traps whose span lies inside this node's byte range of its roll-up."""
        mine = trap_spans.get(roll_id) or []
        if not mine:
            return []
        b0 = len(roll_text[:c0].encode("utf-8"))
        b1 = len(roll_text[:c1].encode("utf-8")) if c1 is not None else len(roll_text.encode("utf-8"))
        return [t for t in mine if t["bytes"][0] >= b0 and t["bytes"][1] <= b1]

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "core_abstracts.jsonl"
    done: dict[str, dict] = {}
    if a.resume and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["node_id"]] = row
        print(json.dumps({"resumed": len(done)}))
    handle = jsonl.open("a", encoding="utf-8")
    #: `call` is entered from several threads when --workers > 1: the duplicate check, the write and
    #: the `done` update are one critical section, or two workers can write the same node twice.
    ledger = threading.Lock()
    host_name = socket.gethostname()
    started = time.monotonic()
    stopped = False

    def call(node_id: str, level: str, piece: str, children: list[str], source: str,
             prompt: str, traps: list[dict[str, Any]] | None = None, recipe: str | None = None,
             used: list[tuple[str, str, str]] | None = None,
             caveat: str | None = None) -> dict[str, Any]:
        nonlocal stopped
        with ledger:
            if node_id in done:
                return done[node_id]
        if time.monotonic() - started > a.stop_after:
            stopped = True
            return {"node_id": node_id, "level": level, "ok": False,
                    "refusal": f"stopped at {a.stop_after / 60:.0f} minutes"}
        t0 = time.monotonic()
        record: dict[str, Any] = {"node_id": node_id, "level": level, "piece": piece,
                                  "children": children, "model": model,
                                  "prompt_sha256": sha256_text(prompt), "made_on": host_name,
                                  "source_sha256": sha256_text(source), "source_chars": len(source),
                                  # a parent's source is not in the store: what it was written from is
                                  # recorded here, or it can be audited against nothing (2026-09-12)
                                  **({"assembly": recipe,
                                      "children_used": [cid for cid, _, _ in used or []],
                                      "children_sha256": [sha256_text(x) for _, _, x in used or []]}
                                     if recipe else {}),
                                  "ceiling": budgets[level], "num_predict": P.num_predict_for(budgets[level]),
                                  **({"caveat": caveat} if caveat else {})}
        num_ctx = getattr(a, f"num_ctx_{level}", 8192)
        record["num_ctx"] = num_ctx
        body = {"model": model, "prompt": prompt.format(text=source, ceiling=budgets[level]), "stream": False,
                "think": False, "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx,
                                            "num_predict": P.num_predict_for(budgets[level])}}
        try:
            payload = post(a.host, "/api/generate", body, TIMEOUT)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            record.update(ok=False, refusal="timeout" if "timed out" in str(exc) else "transport",
                          detail=str(exc)[:120], seconds=round(time.monotonic() - t0, 2))
        else:
            read = payload.get("prompt_eval_count") or 0
            record.update(seconds=round(time.monotonic() - t0, 2), prompt_eval_count=read,
                          eval_count=payload.get("eval_count"), done_reason=payload.get("done_reason"))
            if read >= num_ctx:
                # R1c: the server cut the input; what the model read is not what this node is
                record.update(ok=False, refusal="input truncated by the server",
                              detail=f"{read} prompt tokens read, num_ctx {num_ctx}, "
                                     f"source {len(source)} characters")
            else:
                # R3 at window and sheet grain only; a roll-up's source is its children's abstracts
                values = P.offered_values(source) if level in ("window", "sheet") else []
                verdict = P.abstract_of((payload.get("response") or "").strip(), source,
                                        ceiling=budgets[level], done_reason=payload.get("done_reason"),
                                        values=values, traps=traps or [])
                record.update(**{k: verdict[k] for k in
                                 ("abstract", "words", "content_words", "over_budget", "thin",
                                  "values_used", "placeholders", "traps", "sentences_kept",
                                  "sentences_trimmed", "sentences_dropped", "dropped",
                                  "quantities_checked", "dates_checked", "ok", "refusal",
                                  "trimmed_after_length", "fragment_dropped")})
        with ledger:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            done[node_id] = record
        return record

    def window_jobs() -> list[tuple[int, str, str, str, list[dict[str, Any]]]]:
        """Every store read happens here, on this thread: the sqlite connection is never shared with a
        worker, and `traps_in` reads the store too. A worker receives only text it can no longer be
        made to fetch."""
        jobs: list[tuple[int, str, str, str, list[dict[str, Any]]]] = []
        for piece in pieces:
            for rank, (cid, text) in enumerate(piece["windows"], 1):
                if a.limit and len(jobs) >= a.limit:
                    return jobs
                here = next((sh for sh in piece["sheets"] if any(w[0] == cid for w in sh["windows"])),
                            piece["sheets"][0])
                meta = db.execute("select meta_json from chunks where chunk_id=?", (cid,)).fetchone()
                part = (json.loads(meta[0] or "{}").get("part") or {}).get("span") if meta else None
                traps = (traps_in(here["roll_id"], here["roll_text"], int(part[0]), int(part[1]))
                         if part else [])
                jobs.append((rank, cid, piece["piece"], text, traps))
        return jobs

    try:
        # (a) the windows — one at a time at `--workers 1`, a pool above it. Only this stage fans
        # out: a sheet or a document is written from its children's abstracts and cannot start before
        # they exist, so the LEVELS stay in order however many workers there are.
        jobs = window_jobs()

        def run_window(job: tuple[int, str, str, str, list[dict[str, Any]]]) -> dict[str, Any]:
            rank, cid, piece_name, text, traps = job
            r = call(cid, "window", piece_name, [], text, PROMPT_TEXT, traps)
            print(json.dumps({"level": "window", "piece": piece_name[:30], "part": rank,
                              "s": r.get("seconds"), "words": r.get("words"),
                              "dropped": r.get("sentences_dropped"), "ok": r.get("ok")},
                             ensure_ascii=False))
            return r

        if a.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
                for _ in pool.map(run_window, jobs):
                    if stopped:        # the calls already submitted still finish and record themselves
                        break
        else:
            for job in jobs:
                run_window(job)
                if stopped:
                    break

        # (b) the documents
        if not stopped and not a.limit:
            for piece in pieces:
                # (b1) a workbook's sheets, each a node of its own
                if piece["multi"]:
                    for sheet in piece["sheets"]:
                        kids = [cid for cid, _ in sheet["windows"]]
                        if kids:
                            used = [(k, piece["piece"], done[k]["abstract"]) for k in kids
                                    if done.get(k, {}).get("ok")]
                            parts = [x for _, _, x in used]
                            source = P.assemble(P.ROLLUP, [(n, x) for _, n, x in used])
                            prompt = PROMPT_ROLLUP
                        else:
                            used, parts = [], ["x"]
                            source, prompt = sheet["roll_text"], PROMPT_TEXT
                        if not parts:
                            continue
                        r = call(sheet["roll_id"], "sheet", piece["piece"], kids, source, prompt,
                                 traps_in(sheet["roll_id"], sheet["roll_text"]),
                                 recipe=P.ROLLUP if used else None, used=used or None,
                                 caveat=SHEET_CAVEAT)
                        if stopped:
                            break
                    if stopped:
                        break
                kids = ([sh["roll_id"] for sh in piece["sheets"]] if piece["multi"]
                        else [cid for cid, _ in piece["windows"]])
                used = [(k, piece["piece"], done[k]["abstract"]) for k in kids
                        if done.get(k, {}).get("ok")]
                parts = [x for _, _, x in used]
                if kids and not parts:
                    rec = {"node_id": piece["node_id"], "level": "document", "piece": piece["piece"],
                           "children": kids, "ok": False,
                           "refusal": "every child of this document refused"}
                    handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    done[piece["node_id"]] = rec
                    continue
                if kids:
                    source = P.assemble(P.ROLLUP, [(n, x) for _, n, x in used])
                    r = call(piece["node_id"], "document", piece["piece"], kids, source, PROMPT_ROLLUP,
                             traps_in(piece["sheets"][0]["roll_id"], piece["sheets"][0]["roll_text"]),
                             recipe=P.ROLLUP, used=used)
                else:
                    r = call(piece["node_id"], "document", piece["piece"], [],
                             piece["sheets"][0]["roll_text"], PROMPT_TEXT)
                print(json.dumps({"level": "document", "piece": piece["piece"][:30], "s": r.get("seconds"),
                                  "words": r.get("words"), "dropped": r.get("sentences_dropped"),
                                  "ok": r.get("ok")}, ensure_ascii=False))
                if stopped:
                    break

        # (c) the DCE node
        if not stopped and not a.limit:
            kids = [p["node_id"] for p in pieces]
            used = [(p["node_id"], p["piece"], done[p["node_id"]]["abstract"]) for p in pieces
                    if done.get(p["node_id"], {}).get("ok")]
            parts = [(n, x) for _, n, x in used]
            if parts:
                source = P.assemble(P.DCE, parts)
                r = call(f"n_dce_{a.select.replace(':', '_')}", "dce",
                         f"DCE {CONSULTATION} ({a.select})", kids, source, PROMPT_ROLLUP,
                         recipe=P.DCE, used=used)
                print(json.dumps({"level": "dce", "s": r.get("seconds"), "words": r.get("words"),
                                  "dropped": r.get("sentences_dropped"), "ok": r.get("ok")}))

        # (d) the family rung — row 30. The tree is READ, never re-derived: `families.py` computes it
        # from the store's paths under the sizing rule the lead ruled, and this stage only walks it.
        # Order is forced by the data: a node's source is its children's abstracts, so the deepest
        # sub-families go first, then each band above them, then the families, then the DCE. Only the
        # sub-family bands fan out — a family waits for its own sub-families by construction.
        if not stopped and a.select == "families":
            tree_path = a.families or (a.lab / "demoE2E/verify/family/outputs/families.json")
            if not tree_path.is_file():
                fail(f"{tree_path} missing: with --select families the tree is a declared input")
            tree = json.loads(tree_path.read_text(encoding="utf-8"))["tree"]
            cards: dict[str, dict[str, Any]] = {}
            for card_file in a.cards:
                if not card_file.is_file():
                    fail(f"{card_file} missing: a family's children are abstracts already on disk")
                for line in card_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("level") == "document" and row.get("ok") and row.get("abstract"):
                        cards[row["node_id"]] = row
            if not cards:
                fail("no document abstract was read: pass the runs below on --cards")

            bands: dict[int, list[tuple[dict[str, Any], str]]] = {}

            def walk(node: dict[str, Any], family_name: str, depth: int) -> None:
                for kid in node.get("children") or []:
                    walk(kid, family_name, depth + 1)
                bands.setdefault(depth, []).append((node, family_name))

            for fam in tree:
                for sub in fam.get("sub_families") or []:
                    walk(sub, fam["name"], 1)
            n_sub = sum(len(v) for v in bands.values())
            try:
                want = tuple(int(x) for x in a.expect_families.split(","))
            except ValueError:
                fail(f"--expect-families {a.expect_families!r} is not 'sub_families,families,dce'")
            if (n_sub, len(tree), 1) != want:
                fail(f"the tree holds {n_sub} sub-families and {len(tree)} families, "
                     f"{want[0]} and {want[1]} declared — re-size the row before running it")
            print(json.dumps({"select": "families", "sub_families": n_sub, "families": len(tree),
                              "bands": {str(k): len(v) for k, v in sorted(bands.items())},
                              "document_cards": len(cards), "gated_against": list(want)}))

            def parts_of(node: dict[str, Any], family_name: str) -> list[tuple[str, str, str]]:
                """(child id, display name, text) for the children this node is written from."""
                got: list[tuple[str, str, str]] = []
                for kid in node.get("children") or []:
                    rec = done.get(kid["node_id"])
                    if rec and rec.get("ok") and rec.get("abstract"):
                        got.append((kid["node_id"], kid.get("key_chain") or family_name,
                                    rec["abstract"]))
                for doc_id in node.get("node_ids") or []:
                    card = cards.get(doc_id)
                    if card:
                        got.append((doc_id, card.get("piece") or doc_id, card["abstract"]))
                return got

            def summarise(node: dict[str, Any], level: str, family_name: str,
                          piece: str) -> dict[str, Any]:
                used = parts_of(node, family_name)
                kids = [k["node_id"] for k in node.get("children") or []] + list(node.get("node_ids")
                                                                                or [])
                if not used:
                    rec = {"node_id": node["node_id"], "level": level, "piece": piece,
                           "children": kids, "ok": False,
                           "refusal": "every child of this node refused or has no abstract"}
                    with ledger:
                        handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        done[node["node_id"]] = rec
                    return rec
                source = P.assemble(P.ROLLUP, [(n, x) for _, n, x in used])
                return call(node["node_id"], level, piece, kids, source, PROMPT_ROLLUP, None,
                            recipe=P.ROLLUP, used=used)

            for depth in sorted(bands, reverse=True):        # the deepest band first
                band = bands[depth]

                def run_sub(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
                    node, family_name = item
                    r = summarise(node, "sub_family", family_name,
                                  f"{family_name} — {node.get('key_chain')}")
                    print(json.dumps({"level": "sub_family", "depth": depth,
                                      "key": node.get("key_chain"), "s": r.get("seconds"),
                                      "words": r.get("words"), "ok": r.get("ok")},
                                     ensure_ascii=False))
                    return r

                if a.workers > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
                        for _ in pool.map(run_sub, band):
                            if stopped:
                                break
                else:
                    for item in band:
                        run_sub(item)
                        if stopped:
                            break
                if stopped:
                    break

            if not stopped:
                for fam in tree:                              # then the families, never in parallel
                    node = {"node_id": fam["node_id"], "node_ids": fam.get("node_ids") or [],
                            "children": fam.get("sub_families") or []}
                    r = summarise(node, "family", fam["name"], fam["name"])
                    print(json.dumps({"level": "family", "family": fam["family_id"],
                                      "s": r.get("seconds"), "words": r.get("words"),
                                      "ok": r.get("ok")}, ensure_ascii=False))
                    if stopped:
                        break

            if not stopped:                                   # and the DCE, from the families alone
                used = [(f["node_id"], f["name"], done[f["node_id"]]["abstract"]) for f in tree
                        if done.get(f["node_id"], {}).get("ok")]
                if used:
                    source = P.assemble(P.DCE, [(n, x) for _, n, x in used])
                    r = call("n_dce_families", "dce", f"DCE {CONSULTATION} (families)",
                             [f["node_id"] for f in tree], source, PROMPT_ROLLUP, None,
                             recipe=P.DCE, used=used)
                    print(json.dumps({"level": "dce", "from": len(used), "s": r.get("seconds"),
                                      "words": r.get("words"), "ok": r.get("ok")}))
                else:
                    fail("every family refused: the DCE has nothing to read")

    finally:
        # the release runs on SIGINT too: row 24's clean stop left the model resident until released by hand
        handle.close()
        try:
            post(a.host, "/api/generate", {"model": model, "keep_alive": 0}, 30.0)
        except (urllib.error.URLError, TimeoutError, OSError):
            pass

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    #: at --workers > 1 the file is in completion order; the summary and the markdown must not be, or
    #: the same run would read differently for having been made faster.
    order = {"window": 0, "sheet": 1, "document": 2, "sub_family": 3, "family": 4, "dce": 5}
    rows.sort(key=lambda r: (order.get(r.get("level"), 9), r.get("piece") or "", r.get("node_id") or ""))
    by_level: dict[str, dict[str, Any]] = {}
    for level in ("window", "sheet", "document", "sub_family", "family", "dce"):
        mine = [r for r in rows if r.get("level") == level]
        ok = [r for r in mine if r.get("ok")]
        clean = [r for r in ok if not r.get("over_budget") and not r.get("sentences_dropped")]
        secs = [r["seconds"] for r in ok if r.get("seconds")]
        by_level[level] = {
            "nodes": len(mine), "ok": len(ok), "clean": len(clean),
            "refusals": [r["node_id"][:12] for r in mine if not r.get("ok")],
            "over_budget": sum(1 for r in ok if r.get("over_budget")),
            "done_length": sum(1 for r in mine if r.get("done_reason") == "length"),
            "sentences_dropped": sum(r.get("sentences_dropped", 0) for r in ok),
            "sentences_trimmed": sum(r.get("sentences_trimmed", 0) for r in ok),
            "quantities_checked": sum(r.get("quantities_checked", 0) for r in ok),
            "dates_checked": sum(r.get("dates_checked", 0) for r in ok),
            "words_median": round(statistics.median([r["words"] for r in ok]), 1) if ok else None,
            "median_s": round(statistics.median(secs), 2) if secs else None,
            "total_s": round(sum(secs), 1) if secs else None}
    total = len(rows)
    clean = sum(v["clean"] for v in by_level.values())
    verdict = {"nodes": total, "clean": clean,
               "clean_rate": round(clean / total, 4) if total else None,
               "threshold": PASS_RATE, "passes": bool(total) and clean / total >= PASS_RATE,
               "stopped_at_60_min": stopped, "wall_s": round(time.monotonic() - started, 1)}
    summary = {"declared": {"select": a.select, "documents": len(pieces), "windows": n_win,
                            "gated_against": list(expect) if expect else None,
                            "stop_after_s": a.stop_after,
                            "model": model, "digest": available[model][:12], "host": a.host,
                            "ladder": budgets,
                            "num_predict": {lv: P.num_predict_for(w) for lv, w in budgets.items()},
                            "prompt_text_sha256": sha256_text(PROMPT_TEXT),
                            "prompt_rollup_sha256": sha256_text(PROMPT_ROLLUP),
                            "rules": "pass1.py R1 and R2, imported",
                            "script_sha256": sha256_file(Path(__file__)),
                            "pass1_sha256": sha256_file(Path(__file__).resolve().parent / "pass1.py"),
                            "store_sha256": store_sha,
                            "stages": f"{n_win} windows, {len(pieces)} documents, 1 DCE",
                            "skipped_multi_rollup": skipped},
               "by_level": by_level, "verdict": verdict}
    (out / "core_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                      sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# Pass 1 over the contractual core — one abstract per node", "",
             f"Model `{model}` ({available[model][:12]}) · selection `{a.select}` · "
             f"store `{store_sha[:16]}` · rules "
             f"`pass1.py` R1 and R2 · ceilings " + " · ".join(f"{lv} {w}" for lv, w in budgets.items())
             + " words · `num_predict` " + " / ".join(str(P.num_predict_for(w)) for w in budgets.values()), "",
             "Pass-1 text is orientation, never evidence: no abstract below is a source and none may be "
             "cited in a brief (WP §8.1).", "",
             "| level | nodes | in budget and undropped | median | words (median) | dropped | trimmed |",
             "|---|---|---|---|---|---|---|"]
    for level, v in by_level.items():
        lines.append(f"| {level} | {v['nodes']} | {v['clean']} | {v['median_s']} s | {v['words_median']} | "
                     f"{v['sentences_dropped']} | {v['sentences_trimmed']} |")
    dce = [r for r in rows if r.get("level") == "dce" and r.get("ok")]
    if dce:
        lines += ["", "## The DCE node", "", f"> {dce[0]['abstract']}", ""]
    lines += ["", "## The documents", ""]
    for r in [x for x in rows if x.get("level") == "document"]:
        lines.append(f"**{r.get('piece')}** — `{r['node_id'][:16]}` · {len(r.get('children') or [])} parts"
                     + (f" · {r.get('words')} words" if r.get("ok") else f" · refused: {r.get('refusal')}"))
        if r.get("ok"):
            lines += [f"> {r['abstract']}", ""]
    lines += ["", "## The windows", ""]
    for r in [x for x in rows if x.get("level") == "window"]:
        if not r.get("ok"):
            lines.append(f"- `{r['node_id'][:16]}` {r.get('piece')} — refused: {r.get('refusal')}")
            continue
        flag = " · **over budget**" if r.get("over_budget") else ""
        lines.append(f"- `{r['node_id'][:16]}` {r.get('piece')} — {r['words']} words{flag}")
        lines.append(f"  > {r['abstract']}")
    (out / "core_pass1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verdict, sort_keys=True))
    db.close()
    return 0 if verdict["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
