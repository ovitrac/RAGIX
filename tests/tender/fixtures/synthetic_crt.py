"""Synthetic CRT workbook — structural twin of the XLSX-lane evidence file.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Reproduces facts F1-F8 of ADJUDICATION_SUBSTRATE_XLSX_20260724 with **invented
content** — no client text, no personal data (adjudication §5). The evidence
workbook itself never enters the repository; this twin carries its structure:

  F1  declared dimensions inflated by style-only cells (1000x26 vs tiny content)
  F2  one sheet, several logical objects (merged title / table / merged text)
  F3  row-group label by omission (blank label cells under a populated group)
  F4  blank answer slot inside a bordered table (mandatory line left empty)
  F5  a genuine ListObject with named columns
  F6  a list data-validation sitting exactly on the fillable cells
  F7  hidden sheet with two disjoint 1-D vocabularies
  F8  borders delimit the table blocks; titles/instructions unbordered

``EXPECTED`` is the ground truth consumed by lane gates GX1-GX3
(WP_SUBSTRATE_XLSX_GENERIC_20260724 §3); ``V4_SCENARIOS`` records the three
falsification scenarios of adjudication V4, reserved for the domain-side
pairing WP. ``tests/test_fixture_synthetic_crt.py`` is the integrity gate that
keeps this ground truth honest.
"""

from __future__ import annotations

from pathlib import Path

SHEET_README = "Lisez moi"
SHEET_FORM = "Questionnaire technique"
SHEET_VOCAB = "Vocab"
SHEET_SURVEY = "Enquête"
# Added 2026-07-27 (L2): nested row titles by VERTICAL MERGE (RACI-style layout,
# fact F13) and a bordered table crossed by a fully blank spacer row (fact F14 —
# "inserted empty cells" must not split a box; S2 bridges what S1 separates).
SHEET_RACI = "Suivi RACI"
SHEET_GAPPED = "Formulaire espacé"

# ---------------------------------------------------------------- ground truth

# Block boundaries and types per sheet (gate GX1 / GX2). Ranges are the
# merged-projected populated extents; 9 blocks in total, as measured on the
# evidence workbook (fact F10).
EXPECTED = {
    "blocks": {
        SHEET_README: [
            ("text", "A1:C1"),    # merged title
            ("table", "A3:C8"),   # criteria table — F11 trap: B:C merged on every value row
            ("text", "A11:C13"),  # merged consignes block
        ],
        SHEET_FORM: [
            ("text", "B1:D2"),    # merged title + instructions
            ("table", "B4:D12"),  # bordered QA-shaped table (structure only)
        ],
        SHEET_VOCAB: [
            ("list", "E16:E18"),
            ("list", "B30:B31"),
        ],
        SHEET_SURVEY: [
            ("text", "B1:D2"),
            ("table", "B4:D9"),   # == ListObject ref, gate GX1 S0 check
        ],
        SHEET_RACI: [
            ("table", "A1:E10"),  # gapless; nested row titles via vertical merges (F13)
        ],
        SHEET_GAPPED: [
            ("table", "B2:D8"),   # blank row 5 inside the border box: S2 bridges it (F14)
        ],
    },
    # Anchor samples (gate GX3). row_chain is (locator, how) outermost-first;
    # "group-ffill" = inherited from the populated group label above (F3).
    "anchors": [
        {
            "sheet": SHEET_FORM, "cell": "D9",
            "col_header": "D4",
            "row_chain": [("B7", "group-ffill"), ("C9", "direct")],
        },
        {
            "sheet": SHEET_FORM, "cell": "D5",
            "col_header": "D4",
            "row_chain": [("B5", "direct"), ("C5", "direct")],
        },
        {
            "sheet": SHEET_VOCAB, "cell": "E17",   # 1-D list: leftmost label rule
            "col_header": None,
            "row_chain": [],
        },
        {
            "sheet": SHEET_RACI, "cell": "E6",     # F13: two-level chain via merges,
            "col_header": "E1",                    # deduped across the A:B-wide merge
            "row_chain": [("A5", "merged-proj"), ("B5", "merged-proj"),
                          ("C6", "direct")],
        },
        {
            "sheet": SHEET_RACI, "cell": "E2",     # A2:B4 merge spans two columns:
            "col_header": "E1",                    # its anchor A2 is a DIRECT cell
            "row_chain": [("A2", "direct"), ("C2", "direct"),   # and appears ONCE
                          ("D2", "direct")],       # (B2 projects to the same source)
        },
    ],
    "listobject": {"sheet": SHEET_SURVEY, "name": "Table_1", "ref": "B4:D9"},
    "validation": {"sheet": SHEET_SURVEY, "range": "C5:C9", "formula1": '"Oui,Non"'},
    "hidden_sheets": [SHEET_VOCAB],
    "blank_slot": {"sheet": SHEET_FORM, "cell": "D7"},          # F4
    "group_blank_labels": {"sheet": SHEET_FORM, "cells": ["B8", "B9", "B10", "B11"]},  # F3
}

# The three falsifications of adjudication V4, encoded as data for the future
# domain-side pairing gate (they are NOT exercised by the substrate lane).
V4_SCENARIOS = {
    "regex_multi_match_abstention": {
        "sheet": SHEET_SURVEY,
        "headers_matching_regex": ["C4", "D4"],   # both contain 'réponse'
        "deterministic_answer_column": "C",       # ListObject column + validation range
        "requirement": "structure-first identification must resolve where the "
                       "regex-only approach abstains",
    },
    "wrong_left_anchor": {
        "sheet": SHEET_FORM,
        "answer_cell": "D5",
        "naive_nearest_left": "C5",
        "required_chain": ["B5", "C5"],
        "requirement": "the pairing must keep the full row-label chain, not "
                       "only the nearest populated left cell",
    },
    "silent_blank_slot": {
        "sheet": SHEET_FORM,
        "cell": "D7",
        "requirement": "a blank slot in a mandatory row is surfaced, never "
                       "dropped from the view (contract §9.10)",
    },
}


# ------------------------------------------------------------------- builder

def build(path: Path) -> Path:
    """Write the synthetic workbook at ``path`` and return it."""
    from openpyxl import Workbook
    from openpyxl.styles import Border, Side
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.worksheet.table import Table

    thin = Side(style="thin")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)

    def _border_range(ws, ref: str) -> None:
        for row in ws[ref]:
            for cell in row:
                cell.border = box

    wb = Workbook()

    # -- Sheet 1: title / criteria table / consignes (F2, F11 trap) ----------
    ws = wb.active
    ws.title = SHEET_README
    ws["A1"] = "Consultation fictive — cadre de réponse (contenu inventé)"
    ws.merge_cells("A1:C1")
    ws["A3"] = "Critères d'évaluation"
    ws["B3"] = "Pondération"
    ws.merge_cells("B3:C3")
    criteria = [("CRITERE 1 - QUALITE DE LA DEMARCHE", 55),
                ("Organisation proposée", 20),
                ("Moyens mobilisés", 20),
                ("Exemples de réalisations", 15),
                ("CRITERE 2 - PRIX", 45)]
    for i, (label, weight) in enumerate(criteria, start=4):
        ws[f"A{i}"] = label
        ws[f"B{i}"] = weight
        ws.merge_cells(f"B{i}:C{i}")       # merged value cells on every row (F11)
    ws["A11"] = "CONSIGNES DE REPONSE"
    ws.merge_cells("A11:C11")
    ws["A12"] = ("Le présent classeur fictif sert de gabarit structurel : "
                 "chaque ligne appelle une réponse.")
    ws.merge_cells("A12:C12")
    ws["A13"] = "Toute ligne laissée vide sera signalée."
    ws.merge_cells("A13:C13")
    ws["Z1000"].border = box               # style-only cell inflates dims (F1)

    # -- Sheet 2: bordered table, group labels by omission, blank slot -------
    ws = wb.create_sheet(SHEET_FORM)
    ws["B1"] = "Questionnaire technique — lot fictif"
    ws.merge_cells("B1:D1")
    ws["B2"] = ("Dans la colonne « Réponse attendue du contributeur », "
                "répondre ligne à ligne (contenu inventé).")
    ws.merge_cells("B2:D2")
    ws["B4"], ws["C4"], ws["D4"] = ("Critère technique", "Détail critère",
                                    "Réponse attendue du contributeur")
    rows = {
        5:  ("Organisation de l'équipe", "Décrire l'organisation retenue.",
             "Réponse inventée : équipe en trinôme, rotation mensuelle."),
        6:  ("Suivi et indicateurs", "Décrire le suivi et trois indicateurs.",
             "Réponse inventée : trois indicateurs trimestriels."),
        7:  ("Profils disponibles par catégorie",
             "Lister les profils par catégorie et niveau.", None),   # F4: blank slot
        8:  (None, "Catégorie Alpha - junior", "Réponse inventée : 2 profils."),
        9:  (None, "Catégorie Alpha - senior", "Réponse inventée : 1 profil."),
        10: (None, "Catégorie Beta - junior", "Réponse inventée : 3 profils."),
        11: (None, "Catégorie Beta - senior", "Réponse inventée : 1 profil."),
        12: ("Gestion des variations de charge",
             "Décrire l'adaptation à la hausse et à la baisse.",
             "Réponse inventée : mutualisation interne."),
    }
    for r, (b, c, d) in rows.items():
        if b is not None:
            ws[f"B{r}"] = b                # B8:B11 stay blank on purpose (F3)
        ws[f"C{r}"] = c
        if d is not None:
            ws[f"D{r}"] = d
    _border_range(ws, "B4:D12")            # F8: box on the table only

    # -- Sheet 3: hidden vocabularies (F7) ------------------------------------
    ws = wb.create_sheet(SHEET_VOCAB)
    ws.sheet_state = "hidden"
    for cell, v in (("E16", "Oui"), ("E17", "Non, mais prévu"), ("E18", "Non"),
                    ("B30", "Prioritaire"), ("B31", "Secondaire")):
        ws[cell] = v

    # -- Sheet 4: ListObject + validation, two /réponse/ headers (F5, F6) ----
    ws = wb.create_sheet(SHEET_SURVEY)
    ws["B1"] = "Enquête fictive"
    ws.merge_cells("B1:D1")
    ws["B2"] = "Répondre par Oui ou Non dans la colonne « Réponse du contributeur »."
    ws.merge_cells("B2:D2")
    ws["B4"], ws["C4"], ws["D4"] = ("Sujet", "Réponse du contributeur",
                                    "Détail de la réponse attendue")
    survey = [
        ("Publiez-vous un rapport annuel fictif ?", "Oui",
         "Attendu inventé : préciser l'horizon."),
        ("Disposez-vous d'une charte interne fictive ?", "Non",
         "Attendu inventé : indiquer le calendrier."),
        ("Réalisez-vous un bilan périodique fictif ?", "Oui",
         "Attendu inventé : joindre la trame."),
        ("Formez-vous les équipes au gabarit fictif ?", "Oui",
         "Attendu inventé : indiquer la fréquence."),
        ("Suivez-vous un plan d'amélioration fictif ?", "Non",
         "Attendu inventé : décrire les jalons."),
    ]
    for i, (b, c, d) in enumerate(survey, start=5):
        ws[f"B{i}"], ws[f"C{i}"], ws[f"D{i}"] = b, c, d
    _border_range(ws, "B4:D9")
    dv = DataValidation(type="list", formula1='"Oui,Non"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add("C5:C9")                         # F6: validation on exactly the slots
    ws.add_table(Table(displayName="Table_1", ref="B4:D9"))   # F5

    # -- Sheet 5: nested row titles by vertical merge (F13, RACI layout) -----
    ws = wb.create_sheet(SHEET_RACI)
    ws["A1"] = "Tâches / Actions"
    ws.merge_cells("A1:B1")
    ws["C1"], ws["D1"], ws["E1"] = "Sous-tâche", "Échéance", "Responsable"
    ws["A2"] = "Phase préparation"
    ws.merge_cells("A2:B4")                 # group label spanning TWO columns
    for r, (task, due, who) in enumerate(
            [("Lecture du dossier fictif", "2026-01-05", "Personne A"),
             ("Cadrage inventé", "2026-01-08", "Personne B"),
             ("Plan de charge fictif", "2026-01-12", "Personne C")], start=2):
        ws[f"C{r}"], ws[f"D{r}"], ws[f"E{r}"] = task, due, who
    ws["A5"] = "Phase réalisation"
    ws.merge_cells("A5:A10")                # outer group, column A only
    ws["B5"] = "Volet documentaire"
    ws.merge_cells("B5:B7")                 # inner group
    ws["B8"] = "Volet données"
    ws.merge_cells("B8:B10")                # inner group
    for r, (task, who) in enumerate(
            [("Rédaction du gabarit fictif", "Personne A"),
             ("Relecture croisée fictive", "Personne B"),
             ("Mise en forme fictive", "Personne C"),
             ("Collecte fictive", "Personne B"),
             ("Contrôle qualité fictif", "Personne A"),
             ("Archivage fictif", "Personne C")], start=5):
        ws[f"C{r}"], ws[f"E{r}"] = task, who   # D left blank: data, not label
    _border_range(ws, "A1:E10")

    # -- Sheet 6: blank spacer row inside a bordered table (F14) -------------
    ws = wb.create_sheet(SHEET_GAPPED)
    ws["B2"], ws["C2"], ws["D2"] = "Rubrique", "Consigne", "Valeur"
    for r, (rub, cons, val) in enumerate(
            [("Identité fictive", "Renseigner le nom inventé.", "Entité X"),
             ("Périmètre fictif", "Renseigner le périmètre inventé.", "Lot Y")],
            start=3):
        ws[f"B{r}"], ws[f"C{r}"], ws[f"D{r}"] = rub, cons, val
    # row 5 stays entirely blank — the spacer the border box must bridge
    for r, (rub, cons, val) in enumerate(
            [("Calendrier fictif", "Renseigner la date inventée.", "2026-02-01"),
             ("Contact fictif", "Renseigner le contact inventé.", "Personne D"),
             ("Visa fictif", "Renseigner le visa inventé.", "OK")], start=6):
        ws[f"B{r}"], ws[f"C{r}"], ws[f"D{r}"] = rub, cons, val
    _border_range(ws, "B2:D8")              # borders cover the blank row too

    wb.save(path)
    return path
