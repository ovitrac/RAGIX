#!/usr/bin/env python3
"""M2d — the traps register: one record per defect in the buyer's own CCTPs, for the human (seat S4).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

DECLARED BEFORE THE FIRST RUN (rule T4). No new detection: every record comes from one of three
sources, and nothing else is read.
  reading  — coord's eight defects, base/HANDOFF_REENTRY_20260911_COORD.md §7 (D1-D8);
  rule     — M2's declared orphans (outputs/orphans.json) and M2b's clusters and slots
             (outputs/template.json), cited by index or by cluster and piece;
  post hoc — demoE2E/verify/family/posthoc.py's findings, as FINDINGS §4 records them.
The sealed B-prime labels are not read; their TRAP statements join only after the seal lifts.
R1 spans. A span is a quote located in its piece's roll-up with every whitespace ignored (it must
   match exactly once, or name its match with `pick`), or a template.json fill; it is given as the
   roll-up chunk_id, UTF-8 byte offsets, first and last leaves, and its exact text, and is re-read
   from the store byte for byte. A defect outside the 22 CCTPs has no span and says so.
R2 evidence gate. A record citing an orphan or a template member must have a span in that piece
   overlapping the cited bytes; otherwise the run refuses.
R3 rank by consequence for a bidder, by class then by pieces involved (more first) then by id:
   A scope, duration or award (what the bidder commits to or is judged on: lot numbers, periods,
     renewals, weights); B a foreign obligation (another lot's clause that adds work, delay or cost);
   C a wrong cross-reference; D a wording or format variant.
Each record carries one sentence on what contradicts what (English) and the question a bidder would
put to the buyer (French, for the brief). Outputs: traps.json; manifest.json gains a 'traps' section.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, NoReturn

from . import family as F

CLASSES = {"A": "scope, duration or award", "B": "a foreign obligation", "C": "a wrong cross-reference",
           "D": "a wording or format variant"}


def S(piece: str, quote: str, pick: int | None = None) -> dict:
    return {"kind": "quote", "piece": piece, "quote": quote, "pick": pick}


def FILL(cluster: int, slot: int, piece: str) -> dict:
    return {"kind": "fill", "cluster": cluster, "slot": slot, "piece": piece}


REGISTER: list[dict[str, Any]] = [
    dict(id="T01", cls="A", coord=None, source=["rule"],
         evidence=[("orphans", 41), ("orphans", 234)],
         spans=[S("02", "dans le cadre des lots 21 à 25"), S("08", "dans le cadre des lots 16 à 20")],
         contradiction="The UPS CCTP names lots 21-25 and the lifts CCTP names lots 16-20, while the lot list (CCTP annexe 3) gives 21-25 to lifts and 16-20 to extinguishers.",
         question="Les CCTP 02 (onduleurs) et 08 (ascenseurs) visent respectivement les lots 21 à 25 et 16 à 20, que l'annexe 3 du CCTP attribue aux ascenseurs et aux extincteurs : quels lots chacun de ces CCTP couvre-t-il ?"),
    dict(id="T02", cls="A", coord="D4", source=["reading", "rule"],
         evidence=[("template", 1, "08"), ("template", 1, "13")],
         spans=[S("08", "Reconductible 2 fois 1 an jusqu’au 31 décembre"),
                S("13", "Reconductible 2 fois 1 an jusqu’au 31 décembre")],
         contradiction="The covers of the lifts and SOCAMEL-trolleys CCTPs stop the renewal at '31 décembre' with no year, where the other covers write 2030.",
         question="Les pages de garde des CCTP 08 et 13 arrêtent la reconduction « jusqu'au 31 décembre » sans année, contre 2030 pour les autres lots : la reconduction de ces lots court-elle jusqu'au 31 décembre 2030 ?"),
    dict(id="T03", cls="A", coord=None, source=["rule"],
         evidence=[("template", 1, "11"), ("template", 1, "00")],
         spans=[S("11", "Période du 1er janvier 2026 au 31 décembre 2028"),
                S("00", "Période du 1er janvier 2025 au 31 décembre 2026")],
         contradiction="The scrubbers CCTP starts the period on 1 January 2026 and the common CCTP gives 2025-2026, where the other covers give 1 January 2027 to 31 December 2028.",
         question="La page de garde du CCTP 11 fait commencer la période au 1er janvier 2026 et celle du CCTP 00 la fixe du 1er janvier 2025 au 31 décembre 2026, contre 2027-2028 ailleurs : quelle période d'exécution s'applique ?"),
    dict(id="T04", cls="B", coord="D2", source=["reading", "post hoc"], evidence=[],
         spans=[S("15", "Mise à disposition d’un équipement provisoires (en raison de la continuité de service)"),
                S("16", "Mise à disposition d’un équipement provisoires (en raison de la continuité de service)"),
                S("18", "Mise à disposition d’un équipement provisoires (en raison de la continuité de service)"),
                S("19", "Mise à disposition d’un équipement provisoires (en raison de la continuité de service)"),
                S("20", "Mise à disposition d’un équipement provisoires (en raison de la continuité de service)")],
         contradiction="The refrigeration lot's temporary-equipment clause (chilled water, mobile air conditioning, cold rooms within 12 hours) is repeated in the kitchen, heat-sealers, hoods and tanks CCTPs, whose equipment it does not concern.",
         question="Les CCTP cuisine, thermoscelleuses, hottes et bacs imposent la mise à disposition sous 12 heures d'eau glacée, de climatisation mobile et de chambres froides : cette obligation propre au lot froid s'applique-t-elle à ces lots ?"),
    dict(id="T05", cls="B", coord="D3", source=["reading", "post hoc"], evidence=[],
         spans=[S("02", "valable jusqu’à réparation complète de l’onduleur"),
                S("13", "valable jusqu’à réparation complète de l’onduleur"),
                S("14", "valable jusqu’à réparation complète de l’onduleur"),
                S("16", "valable jusqu’à réparation complète de l’onduleur")],
         contradiction="The free-loan clause 'until full repair of the UPS' belongs to the UPS lot but also closes the loan clauses of both meal-trolley CCTPs and the kitchen CCTP.",
         question="Les CCTP chariots repas et cuisine limitent le prêt gratuit « jusqu'à réparation complète de l'onduleur » : quelle est la durée du prêt pour ces équipements ?"),
    dict(id="T06", cls="B", coord="D1", source=["reading", "rule"],
         evidence=[("orphans", 40), ("orphans", 42), ("orphans", 45)],
         spans=[S("02", "portes, portails et barrières automatiques et leurs auxiliaires respectifs"),
                S("02", "La maintenance sur les portes , portails et autres automatismes dans le bâtiment est obligatoire"),
                S("02", "La norme NF EN 13 - 241 - 1 précise l’obligation de contrôle de sécurité tous les six mois sur les portes")],
         contradiction="The UPS CCTP states its object and its whole regulation section in terms of automatic doors, gates and barriers (NF EN 13241-1, six-monthly safety checks).",
         question="Le CCTP 02 (onduleurs) décrit l'objet et la réglementation des portes, portails et barrières automatiques : quel objet et quelle réglementation s'appliquent au lot onduleurs ?"),
    dict(id="T07", cls="B", coord="D5", source=["reading", "post hoc"], evidence=[],
         spans=[S("07", "des différents modèles et marques de groupes"), S("07", "électrogèn es.")],
         contradiction="The extinguishers CCTP asks for maintenance tables 'of the different models and makes of generators', text of the generators lot.",
         question="Le CCTP 07 demande des tableaux d'entretien « des différents modèles et marques de groupes électrogènes » : quels tableaux sont attendus pour les extincteurs, colonnes sèches, poteaux et RIA ?"),
    dict(id="T08", cls="B", coord="D6", source=["reading", "rule"], evidence=[("orphans", 331)],
         spans=[S("12", "Visite des 1 000 heures Opérations de maintenance des 50 heures complétées", 0),
                S("12", "Visite des 1 000 heures Opérations de maintenance des 50 heures complétées", 1),
                S("12", "Visite des 1 000 heures Opérations de maintenance des 500 heures complétées", 0),
                S("12", "Visite des 1 000 heures Opérations de maintenance des 500 heures complétées", 1)],
         contradiction="For two families of handling equipment the 1 000-hour visit is 'the 50-hour operations completed', where two other families build it on the 500-hour visit.",
         question="Au CCTP 12, la visite des 1 000 heures reprend « les opérations des 50 heures » pour deux familles d'engins et « des 500 heures » pour deux autres : sur quelle base cette visite est-elle chiffrée ?"),
    dict(id="T09", cls="B", coord=None, source=["rule"], evidence=[("orphans", 473), ("orphans", 475)],
         spans=[S("21", "Vidage par aspiration des bacs et cuves"),
                S("21", "Nettoyage des parois par jet d’eau claire haute pression")],
         contradiction="The access-control CCTP ends with tank-emptying and wall-cleaning operations that belong to the tanks and pipes CCTP, where the same sentences appear.",
         question="Le CCTP 21 (contrôle d'accès) se termine par des opérations de vidage de bacs et de nettoyage de parois propres au lot bacs et canalisations : font-elles partie du lot contrôle d'accès ?"),
    dict(id="T10", cls="C", coord=None, source=["post hoc"], evidence=[],
         spans=[S(p, "La liste des établissements bénéficiaires par lot figure à l’annexe 3 du CCAP") for p in ("02", "03", "08", "09")],
         contradiction="Four CCTPs place the list of beneficiary establishments per lot in CCAP annexe 3, which in the DCE is the list of treasuries; the establishments are CCAP annexe 1.",
         question="Les CCTP 02, 03, 08 et 09 renvoient à « l'annexe 3 du CCAP » pour la liste des établissements par lot, alors que cette annexe est la liste des trésoreries : quelle annexe fait foi ?"),
    dict(id="T11", cls="C", coord=None, source=["post hoc"], evidence=[],
         spans=[S("00", "Les devis des gaz frigorigènes et fluides concernant les lots 1 à 6")],
         contradiction="The common CCTP requires supplier invoices for refrigerant-gas quotations 'for lots 1 to 6', which are the generator lots in the lot list.",
         question="Le CCTP 00 exige les factures fournisseur pour les devis de gaz frigorigènes « des lots 1 à 6 », qui sont les lots groupes électrogènes : quels lots sont visés ?"),
    dict(id="T12", cls="D", coord=None, source=["rule"],
         evidence=[("template", 3, "13"), ("template", 3, "14"), ("template", 3, "21")],
         spans=[S(p, "Tout dépassement d’horaire ou de jour entraînera l’application des pénalités définies au CCAP") for p in ("13", "14", "21")],
         contradiction="Three CCTPs penalise 'any overrun of hour or day' where the eighteen others penalise 'any overrun of delays'.",
         question="Les CCTP 13, 14 et 21 pénalisent « tout dépassement d'horaire ou de jour » au lieu de « tout dépassement de délais » : les pénalités du CCAP s'y appliquent-elles dans les mêmes conditions ?"),
    dict(id="T13", cls="D", coord=None, source=["rule"], evidence=[],
         spans=[FILL(4, 1, p) for p in ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10", "11", "12", "13",
                                        "14", "15", "16", "17", "18", "19", "20", "21")],
         contradiction="The expertise price is placed in AE annexe 1.3 '(BPFU)' by thirteen CCTPs and '(BPU)' by eight.",
         question="L'annexe 1.3 de l'acte d'engagement est désignée « BPFU » par certains CCTP et « BPU » par d'autres : s'agit-il du même bordereau ?"),
    dict(id="T14", cls="D", coord=None, source=["rule"], evidence=[("template", 1, "05")],
         spans=[S("05", "Reconductible 2 fois 1 an jusqu’au 31 décembre 2030", 0),
                S("05", "Reconductible 2 fois 1 an jusqu’au 31 décembre 2030", 1)],
         contradiction="The nurse-call cover prints its renewal line twice.",
         question="La page de garde du CCTP 05 répète la ligne de reconduction : la durée est-elle bien de deux reconductions d'un an jusqu'au 31 décembre 2030 ?"),
    dict(id="T15", cls="C", coord="D7", source=["reading"], evidence=[], spans=[],
         contradiction="The RC numbers the CCTP annexes differently from the CCAP's list (maximum quantities, maximum amounts, quantification files); outside the 22 CCTPs.",
         question="Le RC et le CCAP numérotent différemment les annexes du CCTP (quantités maximales, montants maximums, fichiers de quantification) : quelle numérotation fait foi ?"),
    dict(id="T16", cls="A", coord="D8", source=["reading"], evidence=[], spans=[],
         contradiction="Group 4's award weights sum to 52; outside the 22 CCTPs.",
         question="Les pondérations des critères du groupe 4 totalisent 52 : quelle est la pondération exacte appliquée à l'analyse des offres ?"),
]


def fail(msg: str) -> NoReturn:
    sys.stderr.write(f"traps: REFUSED — {msg}\n")
    sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2d — the traps register (S4).")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--lab", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    store_sha = F.sha256_file(a.store)
    if not store_sha.startswith(F.STORE_SHA_PREFIX):
        fail(f"store sha256 {store_sha[:16]} is not {F.STORE_SHA_PREFIX}")
    wal = Path(str(a.store) + "-wal")
    if wal.exists() and wal.stat().st_size:
        fail("the store's WAL is not empty")
    for name in ("manifest.json", "orphans.json", "template.json"):
        if not (a.out / name).exists():
            fail(f"{name} missing: the earlier stages run first in the same directory")
    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    from ..pieces import load_pieces  # the run's own parser of the piece map

    cctp_paths = sorted(load_pieces(a.lab / "demoE2E/03_analyze/pieces.yaml")["CCTP"])
    docs = db.execute("select doc_id, source_path, doc_class from documents").fetchall()
    pieces: dict[str, F.Piece] = {}
    for p in cctp_paths:
        m = F.PIECE_RE.search(p)
        hits = [d for d in docs if d[2] == "pdf" and d[1].endswith("/" + p)]
        if not m or len(hits) != 1:
            fail(f"{p}: {len(hits)} store documents")
        no, doc_id = m.group(1), hits[0][0]
        leaves = db.execute("select chunk_id, text from chunks where doc_id=? and level=0 order by seq",
                            (doc_id,)).fetchall()
        rolls = db.execute("select chunk_id, text from chunks where doc_id=? and level=1 and parent_id is null",
                           (doc_id,)).fetchall()
        if len(rolls) != 1 or rolls[0][1] != "\n".join(t for _, t in leaves):
            fail(f"{p}: the roll-up is not its leaves joined by newlines")
        pieces[no] = F.Piece(no, doc_id, "CCTP/" + p.split("CCTP/", 1)[1], rolls[0][0], rolls[0][1], leaves)
    if len(pieces) != F.N_CCTP:
        fail(f"{len(pieces)} CCTPs in the frame")
    orphans = json.loads((a.out / "orphans.json").read_text(encoding="utf-8"))["orphans"]
    clusters = json.loads((a.out / "template.json").read_text(encoding="utf-8"))["clusters"]

    cache: dict = {}
    spans_checked = 0

    def span(no: str, c0: int, c1: int) -> dict:
        nonlocal spans_checked
        pc = pieces[no]
        loc = pc.locate(c0, c1)
        if not F.span_ok(db, cache, loc, pc.text[c0:c1]):
            fail(f"CCTP {no}: span at bytes {loc['bytes']} does not re-read from the store")
        spans_checked += 1
        return {"piece": no, "text": pc.text[c0:c1], **loc}

    def quote(no: str, q: str, pick: int | None, where: str) -> dict:
        text = pieces[no].text
        idx = [i for i, ch in enumerate(text) if not ch.isspace()]
        squeezed = "".join(text[i] for i in idx)
        needle = "".join(ch for ch in q if not ch.isspace())
        starts, p = [], squeezed.find(needle)
        while p != -1:
            starts.append(p)
            p = squeezed.find(needle, p + 1)
        if not starts or (pick is None and len(starts) > 1) or (pick is not None and pick >= len(starts)):
            fail(f"{where}: quote in CCTP {no} matches {len(starts)} times (pick {pick})")
        s = starts[pick or 0]
        return span(no, idx[s], idx[s + len(needle) - 1] + 1)

    records = []
    for r in REGISTER:
        spans = []
        for k, sp in enumerate(r["spans"]):
            where = f"{r['id']} span {k}"
            if sp["kind"] == "quote":
                spans.append(quote(sp["piece"], sp["quote"], sp["pick"], where))
            else:
                c = clusters[sp["cluster"]]
                fill = c["slots"][sp["slot"]]["fills"].get(sp["piece"])
                if fill is None:
                    fail(f"{where}: cluster {sp['cluster']} slot {sp['slot']} has no fill for CCTP {sp['piece']}")
                pc = pieces[sp["piece"]]
                c0 = pc.cum.index(fill["bytes"][0])
                c1 = pc.cum.index(fill["bytes"][1])
                spans.append({**span(sp["piece"], c0, c1), "cluster": sp["cluster"], "slot": sp["slot"]})
        for ev in r["evidence"]:  # R2
            if ev[0] == "orphans":
                o = orphans[ev[1]]
                if o["index"] != ev[1]:
                    fail(f"{r['id']}: orphans.json index {ev[1]} is misplaced")
                piece, (e0, e1) = o["piece"], o["bytes"]
            else:
                c = clusters[ev[1]]
                if ev[2] not in c["members"]:
                    fail(f"{r['id']}: CCTP {ev[2]} is not a member of cluster {ev[1]}")
                piece, (e0, e1) = ev[2], c["members"][ev[2]]["bytes"]
            if not any(s["piece"] == piece and s["bytes"][0] < e1 and e0 < s["bytes"][1] for s in spans):
                fail(f"{r['id']}: no span in CCTP {piece} overlaps the cited evidence {ev}")
        involved = sorted({s["piece"] for s in spans})
        records.append({
            "id": r["id"], "class": r["cls"], "class_label": CLASSES[r["cls"]], "coord_defect": r["coord"],
            "status": "located" if spans else "outside the 22 CCTPs", "pieces": involved,
            "source": r["source"],
            "evidence": [list(ev) for ev in r["evidence"]] + ([["card", "HANDOFF_REENTRY_20260911_COORD.md §7", r["coord"]]]
                                                             if r["coord"] else []),
            "contradiction": r["contradiction"], "question_fr": r["question"], "spans": spans})
    records.sort(key=lambda t: (t["class"], -len(t["pieces"]), t["id"]))
    for rank, t in enumerate(records, 1):
        t["rank"] = rank
    counts = {"traps": len(records), "located": sum(t["status"] == "located" for t in records),
              "outside": sum(t["status"] != "located" for t in records),
              "by_class": {c: sum(t["class"] == c for t in records) for c in CLASSES},
              "coord_defects": sorted(t["coord_defect"] for t in records if t["coord_defect"]),
              "spans": sum(len(t["spans"]) for t in records), "spans_reread": spans_checked}
    F.dump(a.out / "traps.json", {"declared": {"classes": CLASSES,
                                               "rank": "class A<B<C<D, then more pieces, then id",
                                               "sources": ["reading (card §7)", "rule (orphans.json, template.json)",
                                                           "post hoc (posthoc.py)"],
                                               "excluded": "the sealed B-prime labels are not read"},
                                  "counts": counts, "traps": records})
    man_path = a.out / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["traps"] = {"script_sha256": F.sha256_file(Path(__file__)), "counts": counts}
    F.dump(man_path, man)
    print(json.dumps(counts, sort_keys=True))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
