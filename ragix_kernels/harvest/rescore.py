#!/usr/bin/env python3
"""Item 2 — the three recorded pass-1 runs re-scored under the rules as they now stand (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Rows 22, 24 and 29 were written under three different states of the rule: R2 alone, then R2.1 through
`grammars_fr` 1.2/1.3, then the refined R1b — and none of them under 1.4. This script asks one question
of all three at once: **which of the sentences the runs dropped were dropped by the rule and which by the
instrument?** It calls no model and writes into no register. Every rule it applies is imported from
`pass1.py`, never restated here, so a rule can never drift between the pipeline and its audit.

DECLARED BEFORE THE FIRST RUN (2026-09-12), in the order the checks fire:

X0 provenance. The three runs are named by their journal row and by the `sha256` that row recorded for
   `core_abstracts.jsonl` (`demoE2E/journal/{22_pass1,24_pass1_rest,29_pass1_core150}.jsonl`). The file is
   hashed before it is read and **refused on any mismatch**: a re-scoring of a file that is not the file
   the journal names is worthless, and silently re-scoring the wrong bytes is the failure mode this gate
   exists for.
X1 the rebuild gate (the coordinating seat's constraint). A record keeps its source's **hash**, not its
   text. Each node is rebuilt from the immutable store by `node_id` and **refused as `source drift` unless
   sha256(text) equals the record's `source_sha256`**. A drifted node is re-scored not at all — neither
   admitted nor dropped — because every verdict below is a statement about a specific source.
X2 R1c, retroactively: `pass1.input_truncation(record)`, which decides from the record alone and names
   what it assumed where `num_ctx` is absent (the three runs predate its recording). A refused node's
   sentences are **not** re-scored: an answer written on half its input is not evidence about a rule.
X3 R1b refined, retroactively: where `done_reason` is `length`, a final fragment without terminal
   punctuation is dropped before anything is scored, and only an empty remainder is a refusal.
X4 R2 and R2.1 under `grammars_fr` 1.4, per sentence, against the rebuilt source. Each sentence of the
   run is re-scored and lands in exactly one of four buckets:
     - **kept, still kept** — the rule agrees with the run;
     - **kept, now dropped** — a *regression of the new grammar or a catch the old rule missed*; listed
       in full, because this is the only bucket that can make a run worse than it was reported;
     - **dropped, re-admitted** — the run's drop was the instrument's, not the model's;
     - **dropped, still dropped** — the model's own offence, with the offence that convicts it.
   The scorable unit is the **sentence**: the raw answer is reconstructed as the record's kept text plus
   its recorded dropped sentences. **Their original order is not recorded and is not recovered** — it does
   not enter any verdict, since R2/R2.1 score each sentence against the source independently, but a rule
   that ever depends on sentence order cannot use this reconstruction.

THE FALSIFIER, declared before the run and named in the output: **the « 1 500 € » and the « NF EN » cases
of row 22 must stay dropped.** They are the two inventions the coordinating seat read by hand; a re-scorer
that re-admits either has widened the rule instead of correcting the instrument, and exits 1.

AMENDMENT 1 (2026-09-12, after the first run, which exited 1 on this falsifier — the declaration was
wrong, not the rule, and the first form is kept above so the correction is legible):
  * **« NF EN » stays the falsifier unchanged** — node `86421c9a8ea8`, offence span `13241`, still
    dropped under R2 in both runs that hold it.
  * **« 1 500 € » is withdrawn as an invention and inverted: it must be RE-ADMITTED.** The evidence is
    the sentence itself, node `a673de14003a`: the model wrote « une pénalité maximale de 1 500 € » and
    the source reads « pénalité de\n1\n500,00 € » → 1500.00 EUR. It was dropped because the `QUANTITY`
    of the day could not see a French thousands group and matched only `500 €`, which is absent from
    the source; today's `QUANTITY` reads `1 500 €` from the same sentence. The drop was the
    instrument's. Inverting it keeps the falsifier two-sided — one sentence that must stay out and one
    that must come back — so a rule that simply admitted everything would still fail.
AMENDMENT 2 (2026-09-12, same run): X1's gate applies **to the window rung only**, as the coordinating
  seat specified. Above it a node's source is assembled from its children's abstracts and is not in the
  store at all, so a mismatch there is not drift and must not be reported as one: `n_dce…` is in no
  chunks row, and 24 of 26 document nodes in rows 22/29 (68 of 431 in row 24) hash differently for that
  reason. They are counted as `source_not_in_store` and scored not at all — **and that is itself a
  finding: a parent's card cannot be audited against anything, because the assembled source it was
  written from is nowhere recorded.** Same family as the missing `num_ctx`.

Outputs: `outputs/rescore.json`, `outputs/rescore.md`. Public DCE spans may appear in them, never in a
message or a journal record.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
#: where the recorded runs live; their paths below are relative to it
RUNS_DIR = Path(os.environ.get("HARVEST_PASS1_DIR", "."))

from .pass1 import (GRAMMAR_VERSION, QUANTITY, QUANTITY_BOUNDED,  # noqa: E402
                   ParentDrift, canonical, check_sentence, rebuild_parent,
                   ends_whole, fold, input_truncation, read_dates, read_values, sentences)

#: the rules have no version number of their own, so the audit names them by the hash of the file it
#: imported them from — the same identifier the journal rows carry for the script that ran.
RULES_SHA = hashlib.sha256((HERE / "pass1.py").read_bytes()).hexdigest()[:16]

#: row -> the file it names, the sha256 its journal row recorded for it, and that row's run id. The
#: path may be relative to this directory or absolute: **stage 2 rewrote row 24's path with a
#: different run**, so the two are separate entries against separate hashes and their numbers never
#: fold together — which is X0's whole purpose.
RUNS = {
    "row22": {"path": "outputs/core_abstracts.jsonl",
              "sha256": "f628c558e479abf256826f8c7475ee727d430a7e7468fa293c74b23e00dfae85",
              "run_id": "22_pass1_20260911T181142_6ad586"},
    "row24": {"path": "outputs_rest150/core_abstracts.jsonl",
              "sha256": "59a853cd1a02ca14c0eea5f2834d74eb9f29249a9030fe8ff43aea00e35b18aa",
              "run_id": "24_pass1_rest_20260911T183615"},
    "row24_stage2": {"path": "outputs_rest150/core_abstracts.jsonl",
                     "sha256": "6d03ba736fb4825e118190999172fcf2e2f0dad3471bc3447074fe6d65f2529e",   # computed, never typed from a prefix
                     "run_id": "24_pass1_rest_stage2_20260912T092725"},
    #: row 30 is the FIRST run whose parents X1 can audit: its records carry the assembly, and the
    #: abstracts they were written from are the document cards named in `cards` plus the run's own
    #: sub-family records. Without those cards a leaf sub-family's children are simply absent and every
    #: parent would read as drifted — the map, not the rule, would be at fault.
    "row30": {"path": "outputs_families/core_abstracts.jsonl",
              "sha256": "53796ec5e343199ac981a08681965373292c04d56d6dda72e06f7ad436ec7e7c",
              "run_id": "30_pass1_families_20260912",
              "cards": ["outputs_core150/core_abstracts.jsonl",
                        "outputs_rest150/core_abstracts.jsonl"]},
    "row29": {"path": "outputs_core150/core_abstracts.jsonl",
              "sha256": "ecf7c859cb59619c46f135e107354ba8ba66f47b1f9030e94fe9f2c79c2f4c1a",
              "run_id": "29_pass1_core150_20260911T211123"},
}
#: the falsifier, two-sided after amendment 1: one sentence that must stay dropped and one that must be
#: re-admitted, both named by the node that holds them and by the offence the run recorded.
FALSIFIERS = (
    {"case": "NF EN 13241", "node": "86421c9a8ea8", "then_span": "13241", "must": "stay_dropped"},
    {"case": "1 500 € read as 500 €", "node": "a673de14003a", "then_span": "500",
     "must": "be_readmitted"},
)


def check_sentence_r21(sentence: str, source_folded: str, source_dates: set[str],
                       source_canonical: set[str]) -> list[dict[str, Any]]:
    """R2.1 exactly as it stood at `d50b580`, frozen here so the two rule states can be shown side by
    side. It lives in the audit and NOT in `pass1.py`, which carries one rule only: the pipeline must
    never be able to call a superseded rule by accident. It is never edited again — if it drifts from
    what the runs were scored under, the comparison below stops meaning anything.
    """
    bad: list[dict[str, Any]] = []
    for m in QUANTITY.finditer(sentence):
        span = m.group(0).strip()
        mine = {v.normalized for v in read_values(span) if v.normalized}
        if mine:
            if not (mine & source_canonical):
                bad.append({"kind": "quantity", "span": span, "rule": "R2.1"})
        elif fold(span) not in source_folded:
            bad.append({"kind": "quantity", "span": span, "rule": "R2"})
    for reading in read_dates(sentence):
        if reading.normalized and reading.normalized not in source_dates \
                and reading.normalized not in source_canonical:
            bad.append({"kind": "date", "span": reading.raw.strip(), "rule": "R2.1"})
    return bad


#: The variant the coordinating seat will put to the scientific lead. MEASURED HERE, APPLIED NOWHERE:
#: turning a drop into a flag relaxes fail-closed behaviour, which is RED by §8.2 of the contract and
#: the lead's alone to allow. Two of the kind names in the request do not exist in the grammar —
#: `KINDS` is (date, datetime, period, amount, percentage, duration, reference, quantity), with no
#: `count` and no `time`: « 2 visites » is a **quantity**, and a clock time is either declined or read
#: inside a `datetime`. The intent is mapped onto the real vocabulary, `period` counted as protected
#: because a date range is a temporal commitment.
PROTECTED_KINDS = {"amount", "percentage", "duration", "date", "datetime", "period"}
DIGITS = re.compile(r"\D")


def flag_variant(sentence: str, folded: str, canon: set[str], dates: set[str],
                 source_digits: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (dropped, flagged). A protected kind absent from the source is a drop, unchanged. A
    non-protected kind whose digits are absent is a drop — an invention. A non-protected kind whose
    digits are present but whose canonical form is absent is KEPT and flagged `unverified`, and such a
    figure is excluded from R3's substitution: flagged is not cited."""
    dropped: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    covered: list[tuple[int, int]] = []
    for v in read_values(sentence):
        covered.append((v.start, v.end))
        span = v.raw.strip()
        if v.kind in PROTECTED_KINDS:
            if v.normalized and v.normalized not in canon:
                dropped.append({"span": span, "kind": v.kind, "why": "protected value not in source"})
        else:
            dg = DIGITS.sub("", span)
            if dg and dg not in source_digits:
                dropped.append({"span": span, "kind": v.kind, "why": "figure absent from the source"})
            elif v.normalized and v.normalized not in canon:
                flagged.append({"span": span, "kind": v.kind, "state": "unverified"})
    for reading in read_dates(sentence):
        if reading.normalized and reading.normalized not in canon and reading.normalized not in dates:
            dropped.append({"span": reading.raw.strip(), "kind": "date", "why": "date not in source"})
    for m in QUANTITY_BOUNDED.finditer(sentence):
        if any(m.start() < end and start < m.end() for start, end in covered):
            continue
        span = m.group(0).strip()
        if any(c.isdigit() for c in span) and fold(span) not in folded:
            dropped.append({"span": span, "kind": "unread", "why": "the grammar cannot read it"})
    return dropped, flagged


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bare(h: str | None) -> str:
    return (h or "").split(":")[-1].strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Item 2 — re-score rows 22, 24 and 29 (S4).")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    runs: dict[str, Any] = {}
    still_dropped_all: list[dict[str, Any]] = []

    for row, spec in RUNS.items():
        folder, declared = spec["path"], spec["sha256"]
        path = Path(folder) if Path(folder).is_absolute() else RUNS_DIR / folder
        raw = path.read_bytes()
        got = sha256_of(raw)
        if got != declared:                                   # X0
            sys.stderr.write(f"rescore: REFUSED — {row}: {path.name} hashes {got[:16]}…, "
                             f"the journal records {declared[:16]}…\n")
            return 2
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]

        by_id = {r["node_id"]: r for r in records if r.get("node_id")}
        for card_file in spec.get("cards") or []:           # the abstracts a parent was written from
            cp = Path(card_file) if Path(card_file).is_absolute() else RUNS_DIR / card_file
            if not cp.is_file():
                sys.stderr.write(f"rescore: REFUSED — {row}: {cp} is declared as a card file and is "
                                 f"not there; a parent cannot be audited against a map with holes\n")
                return 2
            for line in cp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                card = json.loads(line)
                if card.get("node_id") and card.get("abstract"):
                    by_id.setdefault(card["node_id"], card)
        tally = {"records": len(records), "drifted": 0, "source_not_in_store": 0,
                 "parent_rebuilt": 0, "parent_drift": 0,
                 "refused_r1c": 0, "refused_r1b": 0,
                 "scored": 0, "kept_still_kept": 0, "kept_now_dropped": 0,
                 "dropped_readmitted": 0, "dropped_still_dropped": 0,
                 "kept_now_dropped_r21": 0, "dropped_readmitted_r21": 0,
                 "dropped_still_dropped_r21": 0}
        regressions: list[dict[str, Any]] = []
        readmitted: list[dict[str, Any]] = []
        still: list[dict[str, Any]] = []

        for rec in records:
            # AMENDMENT 3 (2026-09-12, third run): X2 BEFORE X1. R1c is a statement about the record —
            # tokens read against the context it was given — and holds whether or not the source can be
            # rebuilt. Fired after X1 it reported 0 refusals for all three runs, because the two nodes
            # it exists for (row 24's and row 29's DCE) are the very nodes whose source is not in the
            # store. A gate that never sees its own cases is not a gate.
            if input_truncation(rec)["refused"]:               # X2
                tally["refused_r1c"] += 1
                continue
            hit = db.execute("select text from chunks where chunk_id=?", (rec["node_id"],)).fetchone()
            if hit and sha256_of(hit[0].encode("utf-8")) == bare(rec.get("source_sha256")):
                source = hit[0]
            elif rec.get("level") == "window":
                # X1, amendment 2: at the window rung a mismatch is drift and a refusal.
                tally["drifted"] += 1
                continue
            else:
                # X1, AMENDMENT 4 (2026-09-12): above the window rung the source is not in the store,
                # so the parent is audited against the children it says it read — every child's
                # abstract must still hash to what the parent recorded, and the reassembled text to
                # the parent's own `source_sha256`. Where a run predates that provenance the parent
                # cannot be audited at all, which is a fact about the record and is counted as such.
                try:
                    rebuilt = rebuild_parent(rec, by_id)
                except ParentDrift:
                    tally["parent_drift"] += 1
                    continue
                if rebuilt is None:
                    tally["source_not_in_store"] += 1
                    continue
                source = rebuilt
                tally["parent_rebuilt"] += 1

            kept = sentences(rec.get("abstract") or "")
            if rec.get("done_reason") == "length" and kept and not ends_whole(kept[-1]):
                kept.pop()                                     # X3
            if rec.get("done_reason") == "length" and not kept:
                tally["refused_r1b"] += 1
                continue
            tally["scored"] += 1

            folded, dates, canon = fold(source), {r.normalized for r in read_dates(source)
                                                  if r.normalized}, canonical(source)
            for sentence in kept:                              # X4, the run's kept sentences
                bad = check_sentence(sentence, folded, dates, canon)
                if check_sentence_r21(sentence, folded, dates, canon):
                    tally["kept_now_dropped_r21"] += 1
                if bad:
                    tally["kept_now_dropped"] += 1
                    regressions.append({"node_id": rec["node_id"][:16], "level": rec.get("level"),
                                        "sentence": sentence, "offences": bad})
                else:
                    tally["kept_still_kept"] += 1
            for gone in rec.get("dropped") or []:              # X4, the run's dropped sentences
                sentence = gone.get("sentence") if isinstance(gone, dict) else str(gone)
                if not sentence:
                    continue
                bad = check_sentence(sentence, folded, dates, canon)
                if check_sentence_r21(sentence, folded, dates, canon):
                    tally["dropped_still_dropped_r21"] += 1
                else:
                    tally["dropped_readmitted_r21"] += 1
                entry = {"node_id": rec["node_id"][:16], "level": rec.get("level"),
                         "sentence": sentence,
                         "offences_then": (gone.get("offences") if isinstance(gone, dict) else None),
                         "offences_now": bad}
                if bad:
                    tally["dropped_still_dropped"] += 1
                    still.append(entry)
                    still_dropped_all.append({**entry, "run": row})
                else:
                    tally["dropped_readmitted"] += 1
                    readmitted.append(entry)

        runs[row] = {"folder": folder, "sha256": declared, "run_id": spec["run_id"], "tally": tally,
                     "kept_now_dropped": regressions, "dropped_readmitted": readmitted,
                     "dropped_still_dropped": still}

    # the declared falsifier, over every run's still-dropped and re-admitted lists
    readmitted_all = [{**e, "run": row} for row, v in runs.items() for e in v["dropped_readmitted"]]
    verdict_rows = []
    ok = True
    for case in FALSIFIERS:
        stays = [e for e in still_dropped_all if e["node_id"].startswith(case["node"])]
        back = [e for e in readmitted_all if e["node_id"].startswith(case["node"])]
        passes = bool(stays) and not back if case["must"] == "stay_dropped" else bool(back) and not stays
        verdict_rows.append({"case": case["case"], "node": case["node"], "must": case["must"],
                             "still_dropped": len(stays), "re_admitted": len(back), "passes": passes})
        ok = ok and passes

    verdict = {"rules_sha256": RULES_SHA, "grammar": GRAMMAR_VERSION,
               "runs": {k: v["tally"] for k, v in runs.items()},
               "falsifier": verdict_rows, "passes": ok}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rescore.json").write_text(json.dumps({"verdict": verdict, "runs": runs},
                                                   ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8")

    lines = [f"# Rows 22, 24 and 29 re-scored — `pass1.py` {RULES_SHA}, grammars {GRAMMAR_VERSION}", "",
             "Every rule imported from `pass1.py`; no model called; each node rebuilt from the immutable "
             "store and refused on a `source_sha256` mismatch.", "",
             "| run | records | drifted (window) | source not in store | refused R1c | refused R1b | "
             "scored | kept→**dropped** R2.1 → **R2.2** | dropped→**re-admitted** R2.1 → **R2.2** | "
             "dropped→dropped R2.1 → **R2.2** |", "|---|" + "---|" * 9]
    for row, v in runs.items():
        t = v["tally"]
        lines.append(f"| {row} (`{v['folder']}`) | {t['records']} | {t['drifted']} | "
                     f"{t['source_not_in_store']} | {t['refused_r1c']} "
                     f"| {t['refused_r1b']} | {t['scored']} | "
                     f"{t['kept_now_dropped_r21']} → **{t['kept_now_dropped']}** | "
                     f"{t['dropped_readmitted_r21']} → **{t['dropped_readmitted']}** | "
                     f"{t['dropped_still_dropped_r21']} → **{t['dropped_still_dropped']}** |")
    lines += ["", "## The declared falsifier", "",
              "| case | node | must | still dropped | re-admitted | passes |",
              "|---|---|---|---|---|---|"]
    for r in verdict_rows:
        lines.append(f"| {r['case']} | `{r['node']}` | {r['must'].replace('_', ' ')} | "
                     f"{r['still_dropped']} | {r['re_admitted']} | "
                     f"{'**yes**' if r['passes'] else 'NO'} |")
    for row, v in runs.items():
        if v["kept_now_dropped"]:
            lines += ["", f"## {row} — kept by the run, dropped by the rule as it now stands", ""]
            for e in v["kept_now_dropped"][:40]:
                lines.append(f"- `{e['node_id']}` ({e['level']}): "
                             + ", ".join(f"{o['kind']} « {o['span']} » [{o['rule']}]"
                                         for o in e["offences"]))
        if v["dropped_readmitted"]:
            lines += ["", f"## {row} — dropped by the run, re-admitted now", ""]
            for e in v["dropped_readmitted"][:40]:
                lines.append(f"- `{e['node_id']}` ({e['level']}): « {e['sentence'][:150]} »")
    (a.out / "rescore.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, sort_keys=True))
    db.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
