#!/usr/bin/env python3
"""M14 — the family rung: every document in exactly one family, and each family sized (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

WP §8.8.3 says the recurrence needs a rung between the DCE and its 570 documents, and gives the rule
that decides when a level splits: *a level may summarise its children only while their abstracts fit
that level's `num_ctx`, prompt included.* This script feeds the scientific lead's ruling on that rung
with numbers instead of an opinion: it assigns every document to one family from the store's own
`source_path`, then measures what each family's children would cost in one call.

No model is called; nothing is written into a register. The store is opened read-only and immutable.

DECLARED BEFORE THE FIRST RUN (2026-09-12), on the coordinating seat's relay:

R0 the frame. Every row of `documents` is assigned to **exactly one** family, and the script refuses
   unless the families partition the 570 documents — no document in two families, none in none. A
   partition that does not close is a defect of the rule, not a detail to patch afterwards.
R1 the families, in the order the rules fire. The path is taken relative to the DCE root (`/DCE v5/`):
     F1 `cctp_lots`      — everything under `CCTP/`: the 22 lot CCTP, one family with a child per lot,
                            the lot number read from the filename's leading digits;
     F2 `ccap`           — `CCAP <REF> signé.pdf` and its named annex FILES;
     F3 `rc_ae_bpu`      — the RC, its annex, the AE, and the 85 BPU files: the règlement and the price
                            schedule it governs, which the coordinating seat groups together;
     F4 `fiches_etablissements` — the folder of establishment forms (CCAP annexe 2's directory);
     F5 `quantification/<folder>` — the workbooks **by folder**: the second level under the
                            quantification annexe, which is a département number or the shared
                            `Catalogue des besoins`. One family per folder;
     F6 `dce_root`       — what remains at the DCE root: the CCTP annexes and the MAJ note.
   **Ambiguity declared rather than hidden:** the 53 establishment forms sit under a directory named
   for CCAP annexe 2, so « the CCAP with its annexes » could take them in. They are kept apart (F4)
   because 53 forms are a different kind of object from six contractual annexes, and because the
   sizing below is the point — a family's size is only informative if the family is one kind of thing.
   Both sizings are reported, so the lead can fold F4 into F2 with the number in front of him.
R2 the size of a family's call. A family's abstract is written from its members' **document-level**
   abstracts, so the cost is the sum of those abstracts' tokens plus the prompt. Characters are turned
   into tokens at **2.6 characters per token, the pilot's measured rate for this corpus** (FINDINGS
   §6), never at a guessed one; the rate is recorded in the output beside every number that uses it.
   A member with no abstract yet — a workbook whose stage 1 skipped it — is counted as `missing`; its
   own cost is never invented, and R3 says what is done about it.
R3 the verdict per family: `fits` at 8 192, at 16 384, or `splits`. The prompt's own tokens are added
   (PROMPT_TOKENS, a declared constant measured from the roll-up prompt), because a budget that
   ignores the instructions is not a budget. Where members are missing, the measured sum is a **floor**
   and a second figure is **projected** — the mean cost of that family's known abstracts times its
   document count — and the verdict is taken on the projection, with both numbers shown. A floor would
   say « fits » of a family whose members simply have not been written yet, which is the worst of the
   two errors available here: `rc_ae_bpu` has 3 abstracts of 88 and its floor fits 8 192 while its
   projection is thirty times that.
R4 the DCE above the families: the number of families and what their abstracts would cost at the
   ladder's document ceiling of 350 words, at the same rate — the question the rung exists to answer.

R5 THE TREE, added 2026-09-12 after the scientific lead ruled « family rung as recommended » and set
   the depth rule: *a level summarises its children only while their abstracts fit its `num_ctx`, else
   it splits by the next path level or by lot group*, with **16 384 for family and DCE calls and 8 192
   below**. So the tree is not declared, it is COMPUTED, and recomputing it after stage 2 can change
   it — which is the point of a rule that decides rather than a depth that is chosen:
     - a family whose documents fit `FAMILY_CTX` reads its documents directly, and has no sub-family;
     - otherwise it splits **by the next path level** (the département, then the establishment), each
       group becoming a sub-family that must fit `BELOW_CTX`; a group that still does not fit splits
       again, one level deeper, until the path runs out;
     - where the path runs out — the 85 BPU files and the 22 lot CCTP are flat — it splits **by lot
       group**: the lot number is read from the filename (`BPU (07)`, `07.CCTP_…`) and consecutive lots
       are banded into the largest groups that fit. A family with neither a deeper path nor a lot
       number would split by name order, and none does.
   A member whose abstract is missing is costed at the mean of its family's known abstracts, as in R3,
   and every node the tree carries says whether its size is measured or projected.
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
from typing import Any, NoReturn

#: the consultation's reference, as its file names and running headers print it. Read from the
#: environment because a reference names one consultation and this code names none.
CONSULTATION = os.environ.get("HARVEST_CONSULTATION", "REF-1")
DCE_ROOT = "/DCE v5/"
CHARS_PER_TOKEN = 2.6          # measured on this corpus, FINDINGS §6 (the pilot's 92 788 characters)
PROMPT_TOKENS = 220            # the roll-up prompt and its instructions, rounded up
CONTEXTS = (8192, 16384)
FAMILY_CTX = 16384             # the lead's ruling, 2026-09-12: family and DCE calls
BELOW_CTX = 8192               # and everything below them
#: the 8b's measured medians at the ladder, for the run estimate — row 29 for the core (28.25 s a
#: document call) and row 24 for the rest (13.55 s); a family or sub-family call summarises abstracts
#: exactly as a document call does, so the document median is the rate that applies to it.
SECONDS_PER_ROLLUP_CALL = 28.25
DOC_CEILING_WORDS = 350        # the ladder's document rung, which a family's abstract would use
CHARS_PER_WORD_FR = 6.1        # measured below over the runs' own document abstracts, kept as a check


def fail(msg: str) -> NoReturn:
    sys.stderr.write(f"families: REFUSED — {msg}\n")
    sys.exit(2)


def family_of(rel: str) -> tuple[str, str]:
    """(family id, family name) for one document path, by R1's rules in order."""
    parts = rel.split("/")
    top = parts[0]
    if top == "CCTP" and len(parts) > 1:
        return "cctp_lots", "Les 22 CCTP de lot"
    if top.startswith(f"CCTP {CONSULTATION} - Annexe 4") and len(parts) > 1:
        return f"quantification/{parts[1]}", f"Quantification — {parts[1]}"
    if top.startswith(f"CCAP {CONSULTATION} - Annexe 2 - Fiche") and len(parts) > 1:
        return "fiches_etablissements", "Fiches de renseignement des établissements"
    if top.startswith(f"AE {CONSULTATION} - Annexe 1 - BPU") and len(parts) > 1:
        return "rc_ae_bpu", "RC, AE et BPU"
    if len(parts) == 1:                                   # a file at the DCE root
        if top.startswith(f"CCAP {CONSULTATION}"):
            return "ccap", "CCAP et ses annexes"
        if top.startswith((f"RC {CONSULTATION}", f"AE {CONSULTATION}")):
            return "rc_ae_bpu", "RC, AE et BPU"
        return "dce_root", "Annexes CCTP et MAJ du DCE"
    return "dce_root", "Annexes CCTP et MAJ du DCE"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M14 — the family rung, sized (S4).")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--cards", required=True, type=Path, nargs="+",
                    help="core_abstracts.jsonl files whose document-level rows carry the abstracts")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    docs = db.execute("select doc_id, source_path from documents order by doc_id").fetchall()
    if not docs:
        fail("the store has no documents")

    # the abstracts, by the node id a document's card carries, and by doc_id through the store
    abstracts: dict[str, str] = {}
    for path in a.cards:
        if not path.is_file():
            fail(f"{path} is not a file")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("level") == "document" and row.get("ok") and row.get("abstract"):
                abstracts[row["node_id"]] = row["abstract"]
    #: a document node's id is its single roll-up's chunk id, or `doc_<doc_id[:12]>` for a workbook
    #: with several sheets — core.py's own rule, mirrored here rather than guessed
    node_for: dict[str, str] = {}
    for doc_id, _ in docs:
        rolls = [c for c, in db.execute("select chunk_id from chunks where doc_id=? and level=1 "
                                        "and parent_id is null order by seq", (doc_id,))]
        node_for[doc_id] = rolls[0] if len(rolls) == 1 else f"doc_{doc_id[:12]}"

    fams: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for doc_id, source in docs:
        i = source.find(DCE_ROOT)
        rel = source[i + len(DCE_ROOT):] if i >= 0 else source
        fid, fname = family_of(rel)
        if doc_id in seen:                                 # R0
            fail(f"{doc_id} assigned twice")
        seen.add(doc_id)
        fam = fams.setdefault(fid, {"family_id": fid, "name": fname, "members": [], "missing": 0,
                                    "abstract_chars": 0})
        node = node_for[doc_id]
        text = abstracts.get(node)
        lot = None
        if fid == "cctp_lots":
            m = re.match(r"(\d{1,2})\s*\.", rel.split("/")[-1])
            lot = m.group(1) if m else None
        fam["members"].append({"doc_id": doc_id, "path": rel, "node_id": node,
                               "abstract_chars": len(text) if text else None,
                               **({"lot": lot} if lot else {})})
        if text:
            fam["abstract_chars"] += len(text)
        else:
            fam["missing"] += 1

    if len(seen) != len(docs):                             # R0, the partition must close
        fail(f"{len(seen)} documents assigned of {len(docs)}")

    for fam in fams.values():
        fam["members"].sort(key=lambda m: m["path"])
        n = len(fam["members"])
        have = n - fam["missing"]
        floor = round(fam["abstract_chars"] / CHARS_PER_TOKEN) + PROMPT_TOKENS
        per_known = (fam["abstract_chars"] / have) if have else None
        projected = (round(per_known * n / CHARS_PER_TOKEN) + PROMPT_TOKENS) if per_known else None
        decide = projected if projected is not None else floor
        fam.update(documents=n, with_abstract=have,
                   tokens_floor=floor, tokens_projected=projected, tokens_decided_on=decide,
                   lower_bound=bool(fam["missing"]),
                   fits={str(c): decide <= c for c in CONTEXTS},
                   verdict=("fits 8192" if decide <= CONTEXTS[0] else
                            "fits 16384" if decide <= CONTEXTS[1] else "splits"))

    order = sorted(fams, key=lambda k: (-fams[k]["tokens_decided_on"], k))

    # R5 — the tree, computed by the sizing rule rather than chosen
    def lot_of(path: str) -> str | None:
        name = path.split("/")[-1]
        m = re.search(r"\((\d{1,2})\)", name) or re.match(r"(\d{1,2})\s*\.", name)
        return m.group(1) if m else None

    def cost(members: list[dict[str, Any]], mean: float) -> int:
        chars = sum(m["abstract_chars"] if m["abstract_chars"] else mean for m in members)
        return round(chars / CHARS_PER_TOKEN) + PROMPT_TOKENS

    def split(members: list[dict[str, Any]], depth: int, mean: float) -> list[dict[str, Any]]:
        """one level of the sizing rule: the groups a set of members becomes, each sized."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for m in members:
            parts = m["path"].split("/")
            key = parts[depth] if len(parts) > depth + 1 else None
            groups.setdefault(key if key is not None else "\u0000flat", []).append(m)
        if list(groups) == ["\u0000flat"]:                 # the path ran out: band by lot
            banded: dict[str, list[dict[str, Any]]] = {}
            band: list[dict[str, Any]] = []
            for m in sorted(members, key=lambda m: (lot_of(m["path"]) or "", m["path"])):
                if band and cost(band + [m], mean) > BELOW_CTX:
                    lots = [lot_of(x["path"]) for x in band]
                    banded[f"lots {lots[0]}–{lots[-1]}"] = band
                    band = []
                band.append(m)
            if band:
                lots = [lot_of(x["path"]) for x in band]
                banded[f"lots {lots[0]}–{lots[-1]}"] = band
            groups = banded
        out = []
        for key in sorted(groups):
            kids = sorted(groups[key], key=lambda m: m["path"])
            node = {"key": key, "documents": len(kids), "tokens": cost(kids, mean),
                    "projected": any(m["abstract_chars"] is None for m in kids)}
            node["fits_8192"] = node["tokens"] <= BELOW_CTX
            if not node["fits_8192"] and len(kids) > 1:
                node["children"] = split(kids, depth + 1, mean)
            else:
                #: the members' own node ids, so the runner reads a sub-family's children from the map
                #: instead of re-deriving them — an internal node's children are its sub-nodes and it
                #: carries no document of its own.
                node["node_ids"] = [m["node_id"] for m in kids]
            out.append(node)
        return out

    def flatten(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flat = []
        for n in nodes:
            flat.append(n)
            flat += flatten(n.get("children", []))
        return flat

    def name_chain(nodes: list[dict[str, Any]], family_id: str, chain: str = "") -> None:
        """A stable id per sub-family: the family and the chain of keys that reached it, hashed short
        for a node id and kept readable beside it. Deterministic, so a re-run names the same node."""
        for n in nodes:
            here = f"{chain}/{n['key']}" if chain else str(n["key"])
            n["key_chain"] = here
            n["node_id"] = "sf_" + hashlib.sha256(f"{family_id}\u0000{here}".encode()).hexdigest()[:12]
            name_chain(n.get("children", []), family_id, here)

    tree = []
    for k in order:
        fam = fams[k]
        have = fam["with_abstract"]
        mean = (fam["abstract_chars"] / have) if have else 0.0
        direct = fam["tokens_decided_on"] <= FAMILY_CTX
        node = {"family_id": k, "name": fam["name"], "documents": fam["documents"],
                "tokens": fam["tokens_decided_on"], "reads_documents_directly": direct,
                "node_id": "fam_" + hashlib.sha256(k.encode()).hexdigest()[:12],
                "node_ids": [m["node_id"] for m in fam["members"]] if direct else [],
                "sub_families": [] if direct else split(fam["members"], 1, mean)}
        name_chain(node["sub_families"], k)
        node["sub_family_calls"] = len(flatten(node["sub_families"]))
        tree.append(node)
    sub_calls = sum(n["sub_family_calls"] for n in tree)
    #: only a LEAF that does not fit is a problem: an internal node over the limit is precisely a node
    #: that split, and listing it beside a genuine leaf would make the plan look broken where it works.
    over = [n for n in flatten([c for t in tree for c in t["sub_families"]])
            if not n["fits_8192"] and not n.get("children")]
    # R4 — the DCE above the families
    measured_chars_per_word = None
    doc_cards = [t for t in abstracts.values()]
    if doc_cards:
        words = sum(len(t.split()) for t in doc_cards)
        measured_chars_per_word = round(sum(len(t) for t in doc_cards) / max(words, 1), 2)
    per_family_chars = DOC_CEILING_WORDS * (measured_chars_per_word or CHARS_PER_WORD_FR)
    dce_tokens = round(len(fams) * per_family_chars / CHARS_PER_TOKEN) + PROMPT_TOKENS

    out = {
        "declared": {"chars_per_token": CHARS_PER_TOKEN, "prompt_tokens": PROMPT_TOKENS,
                     "contexts": list(CONTEXTS), "document_ceiling_words": DOC_CEILING_WORDS,
                     "chars_per_word_measured": measured_chars_per_word,
                     "rules": "R0 partition · R1 the six family rules · R2 2.6 chars a token · "
                              "R3 fits/splits with the prompt counted · R4 the DCE from the families",
                     "cards": [str(p) for p in a.cards]},
        "counts": {"documents": len(docs), "families": len(fams),
                   "documents_with_abstract": sum(f["with_abstract"] for f in fams.values()),
                   "documents_missing_abstract": sum(f["missing"] for f in fams.values()),
                   "families_fitting_8192": sum(1 for f in fams.values() if f["fits"]["8192"]),
                   "families_fitting_16384": sum(1 for f in fams.values() if f["fits"]["16384"]),
                   "families_splitting": sum(1 for f in fams.values() if f["verdict"] == "splits")},
        "dce_from_families": {"families": len(fams), "words_each": DOC_CEILING_WORDS,
                              "tokens_estimated": dce_tokens,
                              "fits": {str(c): dce_tokens <= c for c in CONTEXTS}},
        "alternative_declared": {
            "note": "F4's 53 establishment forms folded into the CCAP family, as « the CCAP with its "
                    "annexes » could be read",
            "ccap_plus_fiches_documents": (len(fams.get("ccap", {}).get("members", []))
                                           + len(fams.get("fiches_etablissements", {}).get("members", []))),
            "ccap_plus_fiches_tokens": (round((fams.get("ccap", {}).get("abstract_chars", 0)
                                               + fams.get("fiches_etablissements", {}).get("abstract_chars", 0))
                                              / CHARS_PER_TOKEN) + PROMPT_TOKENS)},
        "families": [fams[k] for k in order],
        "tree": tree,
        "rebuild": {"declared": "R5 — the lead's ruling of 2026-09-12, the sizing rule deciding depth",
                    "family_ctx": FAMILY_CTX, "below_ctx": BELOW_CTX,
                    "calls": {"dce": 1, "families": len(tree), "sub_families": sub_calls,
                              "documents_existing": sum(f["with_abstract"] for f in fams.values()),
                              "documents_missing": sum(f["missing"] for f in fams.values())},
                    "new_calls_row30": 1 + len(tree) + sub_calls,
                    "seconds_per_call_measured": SECONDS_PER_ROLLUP_CALL,
                    "minutes_estimated": round((1 + len(tree) + sub_calls)
                                               * SECONDS_PER_ROLLUP_CALL / 60, 1),
                    "groups_still_over_8192": [{"key": n["key"], "documents": n["documents"],
                                                "tokens": n["tokens"]} for n in over]},
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "families.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True)
                                         + "\n", encoding="utf-8")
    lines = ["# Le rang « famille » — 570 documents, une famille chacun, et le coût d'un appel", "",
             f"Taux déclaré : **{CHARS_PER_TOKEN} caractères par token** (mesuré sur ce corpus), prompt "
             f"compté à {PROMPT_TOKENS} tokens. Aucun modèle appelé.", "",
             "| famille | documents | résumés | tokens mesurés (plancher) | tokens projetés | "
             "8 192 | 16 384 | verdict |", "|---|---|---|---|---|---|---|---|"]
    for k in order:
        f = fams[k]
        proj = f["tokens_projected"]
        lines.append(f"| `{f['family_id']}` — {f['name']} | {f['documents']} | {f['with_abstract']} | "
                     f"{f['tokens_floor']} | "
                     f"{proj if proj is not None else '—'}"
                     f"{' *(projeté sur ' + str(f['with_abstract']) + ' résumés)*' if f['lower_bound'] else ''} | "
                     f"{'oui' if f['fits']['8192'] else 'non'} | "
                     f"{'oui' if f['fits']['16384'] else 'non'} | **{f['verdict']}** |")
    lines += ["", f"**Le DCE au-dessus des familles** : {len(fams)} familles à {DOC_CEILING_WORDS} mots "
                  f"≈ **{dce_tokens} tokens** — "
                  + " · ".join(f"{c} : {'oui' if dce_tokens <= c else 'non'}" for c in CONTEXTS) + ".", ""]
    (a.out / "families.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"documents": len(docs), "families": len(fams),
                      "splitting": out["counts"]["families_splitting"],
                      "dce_tokens": dce_tokens}, ensure_ascii=False, sort_keys=True))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
