#!/usr/bin/env python3
"""Act 3 « Interroger le dossier » — the answers file the player renders.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Six questions, one record each, every step deterministic and every step named in the record.

THE DESCENT is row 28's, imported and not copied: `descent.py` holds the ruled functions and
`map_test.Lane` the lexical scorer. Document grain and above, a node's vector is the mean of its
children's, renormalised (row 28 measured 0.2531 hit@3 against 0.1541 for the abstract's own vector on
the core frame). At window grain the text lane goes first, as the row's exit sentence reads it —
« document by embedding, then window by text » — and the abstract vector ranks the windows the text
lane cannot separate. Each descent step records the lane that produced it, because a number whose lane
is not written down cannot be compared with another run's.

THE QUERY LANE. Row 28 embeds every query twice, bare and `query: `-prefixed, and reports both; its
runbook paragraph declares neither of record. Coord's ruling of 2026-09-12 therefore takes the
prefixed lane, which is also the better one where the window is chosen (0.6416 against 0.6003). The
lane is written into every record.

SOURCED DIGITS ONLY (the lead's ruling, 2026-09-12). The model sees the K rows with their markers
unresolved — `{{claim:v3}}`, never « 4 heures » — and is told to copy a marker when it needs a figure.
The checker then judges the RESOLVED sentence: every run of digits it carries must stand in the
resolved text of a row that sentence cites, and a figure in no cited row is refused with the token
named. The first form of the rule refused any digit outside a marker, and 43 of the 108 accepted
summary sentences carry one — « 24/{{claim:v5}}, {{claim:v6}} sur 7 » keeps a literal 24 — so quoting
the evidence faithfully was impossible and every composition failed. Quoting is allowed; inventing is
not. The checker also refuses a marker absent from the cited rows, a sentence without a source, an
unknown row id, and an input the server had to cut (rule R1c). A refused composition is
dropped, the verdict becomes `needs_review`, and the K rows stand alone under it: the player renders
the evidence without the prose rather than prose without the evidence.

Markers are resolved only after the check passes, and `k_rows[*].text_resolved` never carries one —
the player refuses the build on `{{`, and it is right to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import descent as D
from .map_test import Lane
from .fr.grammars import read_values                               # noqa: E402
from .fr.dates import read as read_dates                           # noqa: E402
from .fr.cut import is_cut, joined_runs                               # noqa: E402

EMBEDDER = "snowflake-arctic-embed2"
QUERY_PREFIX = "query: "
MARKER = re.compile(r"\{\{claim:([A-Za-z0-9_.:-]+)\}\}")
DIGITS = re.compile(r"\d+")   # a figure is a run of digits; each must stand in a cited row
VERDICTS = ("supported", "supported_with_caveats", "needs_review",
            "abstain_no_evidence", "abstain_conflict")
#: the kinds whose question also reads the register. `date` has no `class` in either register —
#: the classes are amount, count, duration, frequency, penalty, percentage — so a row qualifies
#: for `date` when the kernel's own date grammar reads a normalised date in it. That is how the
#: CCAP row carrying « vendredi 11 septembre 2026 à 12 heures 00 » is reachable at all.
REGISTER_KINDS = ("date", "amount", "duration", "penalty")
RANK_CEILING = "needs_review"

SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"text": {"type": "string"},
                                     "sources": {"type": "array", "items": {"type": "string"}}},
                      "required": ["text", "sources"]}},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
    },
    "required": ["sentences", "verdict"],
}

PROMPT = """Tu réponds à une question sur un dossier de consultation, à partir des seules lignes de preuve ci-dessous.

QUESTION
{question}

LIGNES DE PREUVE (chacune a un identifiant ; les chiffres y sont remplacés par des marqueurs)
{rows}

RÈGLES
- N'écris aucun chiffre. Quand une valeur est nécessaire, recopie le marqueur tel quel, par exemple {{{{claim:v3}}}}.
- Chaque phrase cite au moins un identifiant de ligne, dans "sources".
- N'utilise rien qui ne soit pas dans les lignes ci-dessus.
- Réponds en français.
- verdict : supported si les lignes répondent pleinement ; supported_with_caveats si elles répondent en partie ;
  needs_review si elles sont ambiguës ; abstain_no_evidence si elles ne répondent pas ;
  abstain_conflict si elles se contredisent.

Réponds en JSON : {{"sentences": [{{"text": "...", "sources": ["..."]}}], "verdict": "..."}}
"""


def refuse(message: str) -> None:
    sys.stderr.write(f"ask: REFUSED — {message}\n")
    raise SystemExit(2)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def post(host: str, path: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(host + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get(host: str, path: str, timeout: float = 20.0) -> dict:
    with urllib.request.urlopen(host + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_tag(host: str, family: str) -> str:
    for model in get(host, "/api/tags").get("models") or []:
        if (model.get("name") or "").startswith(family):
            return model["name"]
    refuse(f"no model of the family {family!r} is on this server")


def embed(host: str, tag: str, text: str) -> np.ndarray:
    payload = post(host, "/api/embed", {"model": tag, "input": [text], "truncate": False}, 120.0)
    vectors = payload.get("embeddings") or []
    if len(vectors) != 1:
        refuse(f"{len(vectors)} vectors for one query")
    v = np.asarray(vectors[0], dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        refuse("a zero vector: the query embedded to nothing")
    return v / norm


def french_titles(path: Path) -> dict[str, str]:
    """« ### CCTP 12 — Engins de manutention » → {"CCTP 12": the whole heading}. The collection page's
    own French titles: a corpus file name must never reach the player (the map test's property)."""
    titles: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            heading = line[4:].strip()
            key = heading.split("—")[0].strip()
            titles[key] = heading
            parts = key.split()
            if len(parts) == 2:
                titles[f"{parts[0]}:{parts[1]}"] = heading
    return titles


def register_rows(commitments: Path, clauses: Path) -> list[dict]:
    """Every sentence of the two registers, with the classes its matches carry and the dates its text
    holds. `class` covers amount, count, duration, frequency, penalty, percentage; a date has no class,
    so the kernel's own grammar is asked — the same grammar the form uses, not a second reading."""
    rows: list[dict] = []
    for path, root, label in ((clauses, "documents", "clauses"), (commitments, "pieces", "commitments")):
        block = json.loads(path.read_text(encoding="utf-8")).get(root) or {}
        for key, entry in block.items():
            for sentence in entry.get("sentences") or []:
                text = sentence.get("text") or ""
                if not text.strip():
                    continue
                classes = {m.get("class") for m in sentence.get("matches") or [] if m.get("class")}
                dates = [r for r in read_dates(text) if r.normalized]
                if dates:
                    classes.add("date")
                rows.append({"register": label, "key": key, "role": entry.get("role") or key,
                             "chunk_id": sentence.get("chunk_id"), "text": text,
                             "pages": sentence.get("pages") or [],
                             "classes": sorted(classes),
                             "matches": sentence.get("matches") or [],
                             "dates": [{"raw": d.raw.strip(), "normalized": d.normalized} for d in dates],
                             "hash": sentence.get("hash") or "",
                             "declared_cut": bool(sentence.get("cut"))})
    return rows


def flagged_for_cut(text: str, values) -> list[str]:
    """The two opposite failures of the store's text layer, both from `tender.cut`.

    `is_cut` sees a value whose span begins on the far side of a digit run split by a newline; it needs
    the span's offset, not the text alone. `joined_runs` sees the opposite — a span that JOINS digits
    across a line break, « 202\n7 » read as 2027 — and it is the one the reviewer found on Q4. A row is
    flagged if either fires, and the join is not performed: the text stays as the store has it.
    """
    reasons = []
    for value in values:
        start = getattr(value, "start", None)
        raw = getattr(value, "raw", "") or ""
        if start is not None and is_cut(text, start):
            reasons.append(f"chiffres coupés par la lecture optique : « {raw.strip()[:40]} » commence après un nombre coupé par un saut de ligne")
        for run in joined_runs(raw):
            reasons.append(f"chiffres coupés par la lecture optique : « {run} » recolle des chiffres de part et d'autre d'un saut de ligne")
    return reasons


def piece_key(piece: str | None) -> str | None:
    """« 12.CCTP_Lot _Engins de manutention.pdf » → « CCTP 12 »; « CCAP <REF> signé.pdf » → « CCAP »."""
    if not piece:
        return None
    number = re.match(r"^(\d{2})[._ -]", piece)
    # not \b: « 00.CCTP_Communs… » follows CCTP with an underscore, which is a word character,
    # and the boundary would fail on exactly the file names this corpus uses.
    kind = re.search(r"(?<![A-Z])(CCTP|CCAP|RC|AE|BPU|DPGF)(?![A-Z])", piece.upper())
    if number and kind:
        return f"{kind.group(1)} {number.group(1)}"
    return kind.group(1) if kind else None


#: the English reasons the earlier passes stored, and their French of record. A reason produced before
#: this change cannot always be recomputed — the raw composition that caused it was dropped and never
#: stored — so `--reapply` translates what it cannot regenerate, and regenerates the rest.
TRANSLATIONS = (
    (re.compile(r"^sentence (\d+) carries the figure '([^']*)', which is in the resolved text of none "
                r"of the rows it cites$"),
     "la phrase {0} porte le chiffre « {1} » qu'aucune ligne citée ne contient"),
    (re.compile(r"^sentence (\d+) cites '([^']*)', which is not a row offered$"),
     "la phrase {0} cite « {1} », qui n'est pas une ligne proposée"),
    (re.compile(r"^sentence (\d+) cites no K row$"), "la phrase {0} ne cite aucune ligne de preuve"),
    (re.compile(r"^(\d+) sentence\(s\) dropped: every row they cite carries cut digits$"),
     "{0} phrase(s) écartée(s) : toutes les lignes qu'elles citent portent des chiffres coupés"),
    (re.compile(r"^the reviewer named (\d+) sentence\(s\) unfaithful: the composition is dropped and "
                r"the verdict is the reviewer's$"),
     "le relecteur a nommé {0} phrase(s) infidèle(s) : la composition est écartée et le verdict est "
     "celui du relecteur"),
    (re.compile(r"^(\d+) accepted window\(s\) in the first (\d+) ranks, two declared$"),
     "{0} fenêtre(s) acceptée(s) dans les {1} premiers rangs, deux déclarées"),
    (re.compile(r"^no K row under the accepted windows$"),
     "aucune ligne de preuve sous les fenêtres acceptées"),
    (re.compile(r"^no composition was produced$"), "aucune composition n'a été produite"),
    (re.compile(r"^no sentence$"), "aucune phrase"),
)


def in_french(reason: str) -> str:
    for pattern, template in TRANSLATIONS:
        hit = pattern.match(reason)
        if hit:
            return template.format(*hit.groups())
    return reason


def schema_objects(text: str) -> list:
    """Every top-level JSON object in `text` that has the shape the schema requires.

    For the « sans format imposé » condition: the model answers in free text, so the one object
    that matters has to be found in it. Balanced braces, strings and escapes respected. The shape
    tested here is only `sentences` (a list) and `verdict` (a string) — the vocabulary and the
    evidence remain the checker's business, so a bad verdict is still named as a bad verdict and
    not swallowed as an unreadable output.
    """
    found, i = [], 0
    while True:
        start = text.find("{", i)
        if start < 0:
            return found
        depth, j, in_string, escaped = 0, start, False, False
        while j < len(text):
            ch = text[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return found
        try:
            candidate = json.loads(text[start:j + 1])
        except json.JSONDecodeError:
            candidate = None
        if (isinstance(candidate, dict) and isinstance(candidate.get("sentences"), list)
                and isinstance(candidate.get("verdict"), str)):
            found.append(candidate)
            i = j + 1
        else:
            i = start + 1


def compose_and_check(host: str, model_tag: str, prompt: str, num_ctx: int, timeout: float,
                      k_rows: list[dict], values_of_row: dict, enforce_schema: bool = True) -> tuple:
    """One composition and the whole checker, shared by ask.py and 32_compare.

    Factored out for D-0026: the standard problem compares models, so the composer and the
    checker must be ONE implementation. A second copy would compare two checkers instead.
    Returns (composed, verdict, reasons, payload).
    """
    # `enforce_schema` False is the labelled variant « sans format imposé ». gpt-oss:120b returns
    # an EMPTY response under ollama's structured-output enforcement — done_reason stop, tokens
    # generated, nothing in `response` or `thinking` — so for that condition no `format` is sent and
    # the model thinks as it does. Nothing else changes: the checker below is the same checker.
    body_sent = {"model": model_tag, "prompt": prompt, "stream": False,
                 "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx}}
    if enforce_schema:
        body_sent["think"] = False
        body_sent["format"] = SCHEMA
    payload = post(host, "/api/generate", body_sent, timeout)
    read = payload.get("prompt_eval_count") or 0

    reasons, composed, verdict = [], None, "needs_review"
    if read >= num_ctx:
        reasons.append(f"R1c : le serveur a lu {read} jetons pour un num_ctx de {num_ctx} — "
                       "l\'entrée a été coupée et la réponse ne repose que sur une partie")
    else:
        raw_response = payload.get("response") or ""
        if enforce_schema:
            try:
                body = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                body = None
                reasons.append(f"la réponse n'est pas du JSON : {str(exc)[:90]}")
        else:
            # exactly one, or the output is not analysable: two candidate objects mean the composer
            # would be choosing, and choosing is not reading
            candidates = schema_objects(raw_response)
            body = candidates[0] if len(candidates) == 1 else None
            if body is None:
                reasons.append("sortie non analysable")
        if body is not None:
            said = body.get("verdict")
            if said not in VERDICTS:
                reasons.append(f"le verdict « {said} » est hors du vocabulaire")
            sentences = body.get("sentences") or []
            if not sentences:
                reasons.append("aucune phrase")

            def resolve(sentence: dict) -> str:
                """The sentence with every marker replaced by the value of a row it cites."""
                cited = [values_of_row.get(str(sid).strip().strip("[]"), {})
                         for sid in sentence.get("sources") or []]
                return MARKER.sub(lambda m: next((v[m.group(1)] for v in cited
                                                  if m.group(1) in v), m.group(0)),
                                  sentence.get("text") or "")
            known = {r["id"] for r in k_rows}
            resolved_text = {r["id"]: r["text_resolved"] for r in k_rows}
            for n, sentence in enumerate(sentences, 1):
                raw = sentence.get("text") or ""
                sources = [str(x).strip().strip("[]") for x in (sentence.get("sources") or [])]
                if not sources:
                    reasons.append(f"la phrase {n} ne cite aucune ligne de preuve")
                for sid in sources:
                    if sid not in known:
                        reasons.append(f"la phrase {n} cite « {sid} », qui n'est pas une ligne proposée")
                for claim in MARKER.findall(raw):
                    if not any(claim in values_of_row.get(sid, {}) for sid in sources):
                        reasons.append(f"la phrase {n} écrit {{{{claim:{claim}}}}}, qui ne figure "
                                       "dans aucune des lignes qu\'elle cite")
                # The lead's ruling: sourced digits only, judged on the RESOLVED text. A figure
                # the sentence carries must stand in the resolved text of a row it cites; a
                # figure in no cited row is refused and the reason names the token. Quoting the
                # evidence is therefore allowed; inventing a figure is not.
                cited_text = " ".join(resolved_text.get(sid, "") for sid in sources)
                for token in DIGITS.findall(resolve(sentence)):
                    if token not in cited_text:
                        reasons.append(f"la phrase {n} porte le chiffre « {token} » qu\'aucune ligne "
                                       "citée ne contient")
            flagged = {r["id"] for r in k_rows if r.get("needs_review")}
            kept, dropped_for_cut = [], []
            for sentence in sentences:
                cited = [str(x).strip().strip("[]") for x in (sentence.get("sources") or [])]
                if cited and all(sid in flagged for sid in cited):
                    dropped_for_cut.append(cited)
                else:
                    kept.append(sentence)
            if dropped_for_cut:
                reasons.append(f"{len(dropped_for_cut)} phrase(s) écartée(s) : toutes les lignes "
                               "qu\'elles citent portent des chiffres coupés")
            cites_flagged = any(str(x).strip().strip("[]") in flagged
                                for sentence in kept for x in (sentence.get("sources") or []))
            if not reasons:
                sentences = kept
                composed = {"sentences": [
                    {"text_resolved": resolve(sentence),
                     "sources": [str(x).strip().strip("[]")
                                 for x in (sentence.get("sources") or [])]}
                    for sentence in sentences]}
                verdict = said
                if cites_flagged and VERDICTS.index(said) < VERDICTS.index(RANK_CEILING):
                    verdict = RANK_CEILING   # a cited row carries cut digits: no higher claim
    if composed is None and not reasons:
        reasons.append("aucune composition n'a été produite")
    return composed, verdict, reasons, payload


def reapply(a) -> int:
    """Rewrite the review blocks and the checker's reasons of an existing answers file. No model is
    called: the checker runs again over the compositions and rows the file already holds, and every
    other field is left byte-identical. A record whose composition was dropped cannot have its original
    reason recomputed — the raw answer was never stored — so that reason is translated in place."""
    records = [json.loads(l) for l in a.out.read_text(encoding="utf-8").splitlines() if l.strip()]
    corrections = {}
    if a.corrections and a.corrections.is_file():
        for line in a.corrections.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                corrections[entry["question_id"]] = entry

    changed, out = [], []
    for record in records:
        before = json.dumps(record, ensure_ascii=False, sort_keys=True)
        fields = set()
        rows = record.get("k_rows") or []
        known = {r["id"] for r in rows}
        resolved = {r["id"]: r.get("text_resolved") or "" for r in rows}
        flagged = {r["id"] for r in rows if r.get("needs_review")}
        composed = record.get("composed")
        reasons = [in_french(r) for r in (record.get("checker") or {}).get("reasons") or []]

        if composed:                       # the checker, run again on what the file holds
            fresh = []
            for n, sentence in enumerate(composed.get("sentences") or [], 1):
                sources = [str(x) for x in sentence.get("sources") or []]
                if not sources:
                    fresh.append(f"la phrase {n} ne cite aucune ligne de preuve")
                for sid in sources:
                    if sid not in known:
                        fresh.append(f"la phrase {n} cite « {sid} », qui n'est pas une ligne proposée")
                if sources and all(sid in flagged for sid in sources):
                    fresh.append(f"la phrase {n} est écartée : toutes les lignes qu'elle cite portent "
                                 "des chiffres coupés")
                cited = " ".join(resolved.get(sid, "") for sid in sources)
                for token in DIGITS.findall(sentence.get("text_resolved") or ""):
                    if token not in cited:
                        fresh.append(f"la phrase {n} porte le chiffre « {token} » qu'aucune ligne "
                                     "citée ne contient")
            reasons = fresh

        correction = corrections.get(record.get("question_id"))
        if correction and composed:
            named = {(x.get("text_resolved") or "").strip()
                     for x in correction.get("unfaithful_sentences") or []}
            hit = [x for x in composed.get("sentences") or []
                   if (x.get("text_resolved") or "").strip() in named]
            if hit:
                composed = None
                record["verdict"] = correction.get("reviewer_verdict") or record.get("verdict")
                reasons.append(f"le relecteur a nommé {len(hit)} phrase(s) infidèle(s) : la composition "
                               "est écartée et le verdict est celui du relecteur")
        if correction and record.get("review"):
            # rule 13: the reviewer's words are carried whole, never cut to fit a field
            record["review"] = {"reviewer": correction.get("reviewer"),
                                "read_on": correction.get("read_on"),
                                "reason": "; ".join(x.get("reason") or ""
                                                    for x in correction.get("unfaithful_sentences") or []),
                                "note": correction.get("note"),
                                "matched_sentences": record["review"].get("matched_sentences")}
            fields.add("review")

        record["composed"] = composed
        record["checker"] = {"passed": composed is not None, "reasons": reasons}
        after = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if after != before:
            fields.add("checker.reasons")
            changed.append({"question_id": record.get("question_id"), "fields": sorted(fields)})
        out.append(record)

    a.out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in out),
                     encoding="utf-8")
    print(json.dumps({"reapplied": len(out), "changed": changed}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Act 3 — six questions over the dossier")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--records", required=True, type=Path, help="the form's records, run 2 merged with job 1")
    ap.add_argument("--nodes", required=True, type=Path, help="row 26's node_embeddings.jsonl")
    ap.add_argument("--node-vectors", required=True, type=Path, dest="node_vectors")
    ap.add_argument("--core", required=True, type=Path, help="the core documents' abstracts (the frame)")
    ap.add_argument("--families", required=True, type=Path)
    ap.add_argument("--pyramid", required=True, type=Path, help="the French family names for the screen")
    ap.add_argument("--commitments", required=True, type=Path, help="the register, per piece")
    ap.add_argument("--clauses", required=True, type=Path, help="the register, per CCAP/RC document")
    ap.add_argument("--corrections", type=Path, default=None,
                    help="the reviewer's reading: a composed sentence it names is dropped at write time")
    ap.add_argument("--register-top", type=int, default=5, dest="register_top")
    ap.add_argument("--collection", required=True, type=Path, help="collection_fr.md — the French titles")
    ap.add_argument("--questions", required=True, type=Path, help="the six, as json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="granite4.2:8b-chat")
    ap.add_argument("--embedder", default=EMBEDDER)
    ap.add_argument("--lane", default="prefixed", choices=("prefixed", "bare"))
    ap.add_argument("--documents", type=int, default=3, help="documents kept before the window step")
    ap.add_argument("--top", type=int, default=5, help="ACCEPTED windows kept")
    ap.add_argument("--depth", type=int, default=20, help="descent ranks walked to find them")
    ap.add_argument("--num-ctx", type=int, default=16384, dest="num_ctx")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--only", default="", help="comma-separated question ids, for the smoke")
    ap.add_argument("--allow-resident", action="store_true", dest="allow_resident")
    ap.add_argument("--reapply", action="store_true",
                    help="rewrite only the review blocks and the checker's reasons of an existing "
                         "answers file, from corrections.jsonl, with no model call")
    a = ap.parse_args(argv)

    if a.reapply:
        return reapply(a)

    if not re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", a.host):
        refuse(f"{a.host} is not loopback: this job runs on the executor's own server")
    resident = [m.get("name") for m in get(a.host, "/api/ps").get("models") or []]
    if resident and not a.allow_resident:
        refuse(f"a model is resident ({', '.join(resident)}): this job is exclusive")

    model_tag = a.model
    embed_tag = resolve_tag(a.host, a.embedder)

    nodes = D.read_jsonl(a.nodes)
    vectors = D.vectors_for(a.node_vectors, nodes, "row 26's node vectors")
    core_ids = {r["node_id"] for r in D.read_jsonl(a.core) if r.get("level") == "document"}
    index = {n["node_id"]: i for i, n in enumerate(nodes)}
    win_ids = {n["node_id"] for n in nodes if n.get("level") == "window"}

    db = sqlite3.connect(f"{a.store.resolve().as_uri()}?mode=ro&immutable=1", uri=True)

    # the document's vector is the mean of its windows', renormalised — row 28's ruled shape
    children_mean: dict[str, np.ndarray] = {}
    doc_of_window: dict[str, str] = {}
    doc_key: dict[str, str] = {}          # document node -> the documents table's doc_id
    for doc_id in core_ids:
        owner = db.execute("select doc_id from chunks where chunk_id=?", (doc_id,)).fetchone()
        if owner:
            doc_key[doc_id] = owner[0]
        kids = [cid for (cid,) in db.execute(
            "select chunk_id from chunks where parent_id=? and level=1 order by seq", (doc_id,))
            if cid in win_ids]
        for cid in kids:
            doc_of_window[cid] = doc_id
        if not kids:
            continue
        mean = vectors[[index[c] for c in kids]].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm:
            children_mean[doc_id] = (mean / norm).astype(np.float32)
    if not children_mean:
        refuse("no core document has embedded windows: the descent has no frame")

    records = [json.loads(l) for l in a.records.read_text(encoding="utf-8").splitlines() if l.strip()]
    accepted = {r["node_id"]: r for r in records if r.get("ok")}

    titles = french_titles(a.collection)
    families = json.loads(a.families.read_text(encoding="utf-8")).get("families") or []
    family_of_doc: dict[str, str] = {}
    for family in families:
        for member in family.get("members") or []:
            if member.get("doc_id"):
                family_of_doc[member["doc_id"]] = family.get("family_id") or ""

    pyramid_name: dict[str, str] = {}
    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("family_id") and node.get("name"):
                pyramid_name[node["family_id"]] = node["name"]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(json.loads(a.pyramid.read_text(encoding="utf-8")))

    register = register_rows(a.commitments, a.clauses)
    corrections = {}
    if a.corrections and a.corrections.is_file():
        for line in a.corrections.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                corrections[entry["question_id"]] = entry

    questions = json.loads(a.questions.read_text(encoding="utf-8"))
    wanted = {q.strip() for q in a.only.split(",") if q.strip()}
    if wanted:
        questions = [q for q in questions if q["id"] in wanted]
        if not questions:
            refuse(f"--only {a.only!r} names no question of the set")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    handle = a.out.open("a", encoding="utf-8", buffering=1)
    written = 0
    try:
        for q in questions:
            t0 = time.time()
            text = (QUERY_PREFIX if a.lane == "prefixed" else "") + q["question"]
            qv = embed(a.host, embed_tag, text)

            doc_ids = sorted(children_mean)
            scores = np.vstack([children_mean[d] for d in doc_ids]) @ qv
            order = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), doc_ids[i]))
            top_docs = [(doc_ids[i], float(scores[i])) for i in order[:a.documents]]

            # windows: the text lane first, the abstract vector where the text lane cannot separate
            candidates: list[dict] = []
            for doc_id, doc_score in top_docs:
                rows = db.execute("select chunk_id, text from chunks where parent_id=? and level=1 "
                                  "order by seq", (doc_id,)).fetchall()
                rows = [(cid, txt) for cid, txt in rows if cid in win_ids]
                if not rows:
                    continue
                # the map test's own lexical scorer, imported and not copied: it returns an ORDER,
                # so the text lane contributes a rank and the abstract vector breaks its ties. Text
                # first, as row 28's exit sentence reads the descent; both numbers are recorded.
                lane_obj = Lane({cid: txt for cid, txt in rows})
                seq_order = {cid: i for i, (cid, _) in enumerate(rows)}
                ranked = lane_obj.rank(q["question"], seq_order)
                text_rank = {cid: i for i, cid in enumerate(ranked)}
                for cid, _ in rows:
                    vector_score = float(vectors[index[cid]] @ qv) if cid in index else 0.0
                    candidates.append({"window": cid, "document": doc_id, "document_score": doc_score,
                                       "text_rank": text_rank.get(cid, len(ranked)),
                                       "vector_score": vector_score})
            if not candidates:
                refuse(f"{q['id']}: the top documents hold no embedded window")
            lane_used = "text.rank+abstract_vector.tiebreak"
            candidates.sort(key=lambda c: (c["text_rank"], -c["vector_score"], c["window"]))
            # The lead's ruling of 2026-09-12: the first `--top` ACCEPTED windows within the first
            # `--depth` ranks. A refused window is visited and recorded, never silently skipped — the
            # record must show what the descent walked past and why it was passed over.
            visited, windows = [], []
            for rank, c in enumerate(candidates[:a.depth], 1):
                is_accepted = c["window"] in accepted
                visited.append({"rank": rank, "node_id": c["window"], "accepted": is_accepted,
                                "text_rank": c["text_rank"], "score": round(c["vector_score"], 6)})
                if is_accepted and len(windows) < a.top:
                    windows.append(c)

            def document_of(chunk_id: str) -> tuple[str, str]:
                """(the store's doc_id, the collection page's French title) for a chunk."""
                owner = db.execute("select doc_id from chunks where chunk_id=?", (chunk_id,)).fetchone()
                if not owner:
                    return "", ""
                source = (db.execute("select source_path from documents where doc_id=?",
                                     (owner[0],)).fetchone() or [None])[0]
                key = piece_key(Path(source).name if source else None)
                return owner[0], titles.get(key or "", key or "")

            descent_path = []
            for doc_id, doc_score in top_docs:
                family = family_of_doc.get(doc_key.get(doc_id, ""), "")
                descent_path.append({"level": "family", "node_id": family,
                                     "title": pyramid_name.get(family, family),
                                     "score": round(doc_score, 6), "lane": f"children_mean.{a.lane}"})
                break
            for doc_id, doc_score in top_docs:
                owner, title = document_of(doc_id)
                # node_id is a chunk id here and anchors nothing on the drawing; doc_id does
                descent_path.append({"level": "document", "node_id": doc_id, "doc_id": owner,
                                     "title": title, "score": round(doc_score, 6),
                                     "lane": f"children_mean.{a.lane}"})
            for c in windows:
                owner, title = document_of(c["window"])
                descent_path.append({"level": "window", "node_id": c["window"], "doc_id": owner,
                                     "title": title,
                                     "score": round(c["vector_score"], 6),
                                     "text_rank": c["text_rank"], "lane": lane_used})

            # The K rows of those windows, among the accepted records. A K row is a SENTENCE of the
            # accepted summary with its values still as markers — a bare « {{claim:v1}} » under a label
            # gives the model nothing to compose from, and the record's own field is `text_resolved`,
            # which is a text. `summary_map` says which values each sentence carries.
            k_rows, values_of_row, unresolved = [], {}, {}
            for c in windows:
                record = accepted.get(c["window"])
                if not record:
                    continue
                row = db.execute("select text, doc_id, pages_json from chunks where chunk_id=?",
                                 (c["window"],)).fetchone()
                if not row:
                    continue
                window_text, doc_id, pages_json = row
                values = {f"v{i + 1}": v.raw.strip() for i, v in enumerate(read_values(window_text))}
                pages = json.loads(pages_json) if pages_json else []
                source = (db.execute("select source_path from documents where doc_id=?",
                                     (doc_id,)).fetchone() or [None])[0]
                key = piece_key(Path(source).name if source else None)
                by_value = {k.get("value_id"): k for k in record.get("k_rows") or []}
                sentences = record.get("summary") or []
                raw_values = list(read_values(window_text))
                owner_id, doc_title = document_of(c["window"])
                for entry in record.get("summary_map") or []:
                    position = int(entry.get("sentence") or 0)
                    if not 1 <= position <= len(sentences):
                        continue
                    text = sentences[position - 1]
                    children = [v for v in (entry.get("children") or []) if v in values]
                    # a colon-free id: the model truncated « <window>:s1 » at the colon and cited a
                    # window that was never offered as a row. The window is in the record's own field.
                    row_id = f"K{len(k_rows) + 1}"
                    values_of_row[row_id] = {v: values[v] for v in children}
                    carried = [raw_values[int(v[1:]) - 1] for v in children
                               if v[1:].isdigit() and int(v[1:]) - 1 < len(raw_values)]
                    cut_reasons = flagged_for_cut(window_text, carried)
                    unresolved[row_id] = text
                    k_rows.append({"id": row_id, "origin": "descent", "window_id": c["window"],
                                   "document": doc_title, "document_id": owner_id,
                                   "page": pages[0] if pages else None,
                                   "values": [{"value_id": v, "text": values[v],
                                               "act": (by_value.get(v) or {}).get("act"),
                                               "relevance": (by_value.get(v) or {}).get("relevance"),
                                               "canonical": (by_value.get(v) or {}).get("canonical")}
                                              for v in children],
                                   "needs_review": bool(cut_reasons),
                                   "cut_reasons": cut_reasons,
                                   "text_resolved": MARKER.sub(
                                       lambda m: values.get(m.group(1), m.group(0)), text)})

            # ---- the register lane: for a question whose kind names a value class, the descent's
            # windows are joined by the register's own rows of that kind, ranked lexically on the
            # question. The reviewer found Q3 naming no penalty though « 18.1 - pénalités de retard »
            # is in clauses.json, and Q5 answering from an abstract that misread the page.
            kind = q.get("kind") or ""
            register_used = []
            if kind in REGISTER_KINDS and register:
                pool = [r for r in register if kind in r["classes"]]
                if pool:
                    lane_reg = Lane({f"R{i}": r["text"] for i, r in enumerate(pool)})
                    ranked_reg = lane_reg.rank(q["question"], {f"R{i}": i for i in range(len(pool))})
                    for handle_id in ranked_reg[:a.register_top]:
                        r = pool[int(handle_id[1:])]
                        owner_id, doc_title = document_of(r["chunk_id"]) if r["chunk_id"] else ("", "")
                        vals = list(read_values(r["text"]))
                        cut_reasons = flagged_for_cut(r["text"], vals)
                        if r["declared_cut"]:
                            cut_reasons.append("chiffres coupés par la lecture optique : le registre signale cette phrase comme coupée")
                        row_id = f"K{len(k_rows) + 1}"
                        values_of_row[row_id] = {}
                        unresolved[row_id] = r["text"]
                        register_used.append(row_id)
                        k_rows.append({"id": row_id, "origin": "register",
                                       "register": r["register"], "role": r["role"],
                                       "window_id": r["chunk_id"], "document": doc_title or r["role"],
                                       "document_id": owner_id,
                                       "page": (r["pages"] or [None])[0],
                                       "values": [{"value_id": f"{r['hash'][:12]}:{i}",
                                                   "text": (m.get("text") or "").strip(),
                                                   "canonical": m.get("normalized"),
                                                   "act": m.get("class"), "relevance": "register"}
                                                  for i, m in enumerate(r["matches"])]
                                                 + [{"value_id": f"{r['hash'][:12]}:d{i}",
                                                     "text": d["raw"], "canonical": d["normalized"],
                                                     "act": "date", "relevance": "register"}
                                                    for i, d in enumerate(r["dates"])],
                                       "needs_review": bool(cut_reasons),
                                       "cut_reasons": cut_reasons,
                                       "text_resolved": r["text"]})

            made_on = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            base = {"question_id": q["id"], "question": q["question"], "lane": a.lane,
                    "window_lane": lane_used, "descent": descent_path, "visited": visited,
                    "accepted_windows": len(windows), "kind": q.get("kind"),
                    "register_rows": len(register_used), "k_rows": k_rows,
                    "model": model_tag, "embedder": embed_tag, "num_ctx": a.num_ctx, "made_on": made_on}

            if len(windows) < 2 or not k_rows:
                why = (f"{len(windows)} accepted window(s) in the first {a.depth} ranks, two declared"
                       if len(windows) < 2 else "no K row under the accepted windows")
                handle.write(json.dumps({**base, "composed": None,
                                         "checker": {"passed": False, "reasons": [why]},
                                         "verdict": "abstain_no_evidence", "prompt_sha256": None,
                                         "prompt_eval_count": None,
                                         "duration_s": round(time.time() - t0, 2)},
                                        ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
                print(json.dumps({"q": q["id"], "verdict": "abstain_no_evidence", "k_rows": 0},
                                 ensure_ascii=False), flush=True)
                continue

            listing = "\n".join(
                f"  {r['id']} ({r['document']}, p. {r['page']}) « {unresolved[r['id']]} »"
                for r in k_rows)
            prompt = PROMPT.format(question=q["question"], rows=listing)
            composed, verdict, reasons, payload = compose_and_check(
                a.host, model_tag, prompt, a.num_ctx, a.timeout, k_rows, values_of_row)
            read = payload.get("prompt_eval_count") or 0

            # ---- the reviewer's reading, applied at write time. A composed sentence the reviewer
            # named as unfaithful drops the WHOLE composition — the record then carries its K rows
            # under the reviewer's verdict, and says who read it and why. A correction that matches
            # nothing is left unmatched on purpose: the composition it judged is not this one.
            review = None
            correction = corrections.get(q["id"])
            if correction and composed:
                named = {(x.get("text_resolved") or "").strip()
                         for x in correction.get("unfaithful_sentences") or []}
                hit = [x for x in composed["sentences"] if (x.get("text_resolved") or "").strip() in named]
                if hit:
                    # rule 13: a reviewer's correction may not be lost, and half a sentence is lost.
                    # The reason and the note are carried whole.
                    reason = "; ".join(x.get("reason") or ""
                                       for x in correction.get("unfaithful_sentences") or [])
                    composed = None
                    verdict = correction.get("reviewer_verdict") or verdict
                    reasons.append(f"le relecteur a nommé {len(hit)} phrase(s) infidèle(s) : la "
                                   "composition est écartée et le verdict est celui du relecteur")
                    review = {"reviewer": correction.get("reviewer"),
                              "read_on": correction.get("read_on"), "reason": reason,
                              "note": correction.get("note"), "matched_sentences": len(hit)}
            record_out = {**base, "composed": composed,
                          "checker": {"passed": composed is not None, "reasons": reasons},
                          "verdict": verdict, "prompt_sha256": sha256_text(prompt),
                          "prompt_eval_count": read,
                          "duration_s": round(time.time() - t0, 2)}
            if review:
                record_out["review"] = review
            handle.write(json.dumps(record_out, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
            print(json.dumps({"q": q["id"], "verdict": verdict, "k_rows": len(k_rows),
                              "register": len(register_used),
                              "flagged": sum(1 for r in k_rows if r.get("needs_review")),
                              "review": bool(review), "passed": composed is not None,
                              "reasons": reasons[:2], "seconds": round(time.time() - t0, 2)},
                             ensure_ascii=False), flush=True)
    finally:
        handle.close()
        for tag in (model_tag, embed_tag):
            # the embedder is a model too: a job that leaves it resident is not exclusive any more
            try:
                post(a.host, "/api/generate", {"model": tag, "keep_alive": 0}, 30.0)
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
    print(json.dumps({"written": written, "out": str(a.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
