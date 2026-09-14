"""ragix_kernels.harvest.runner — the jobs that call a model and hand its answer to the form.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Three jobs, one contract. Each calls a local Ollama server, keeps the raw answer, and lets
`form.validate` judge it: a refusal is recorded with its reason and never repaired, a timeout or a
transport error is recorded as UNKNOWN because it teaches nothing about the model, and every record
is written the moment it is made so a job that dies keeps what it already paid for.

  bake-off   one call per object per candidate model; the model is given the object's text and the
             grammar's values as a numbered list and returns the form with ``{{claim:v<n>}}`` markers.
             A candidate refusing on most of its first objects is stopped early, for that role.
  nodes      K before T at document grain: the prompt carries the grammar's values and marks the
             piece's own (lot-specific) ones, and the summary is asked to be built around them. A
             pilot on a declared piece decides the writer; a document longer than the writer's
             context is harvested through its level-1 parts and flagged partial.
  window     pass 2 at window grain. The model is never asked to write a marker: it writes prose with
             the figures as the page carries them, `form.substitute` replaces each figure equal to an
             offered value with that value's marker, the provenance map is DERIVED from those
             substitutions, and `form.validate` judges the result at window grain. A pilot gates the
             core; the core runs in a bounded pool; `--only` restricts it to named windows; `--resume`
             skips what is already recorded; accepted windows go to the sidecar derived store.

The server is named by `--host` and defaults to localhost; nothing here sends a document anywhere
else. The store is opened read-only and immutable.

    python -m ragix_kernels.harvest.runner window --store saqqara.db --cards core_abstracts.jsonl \\
        --commitments commitments.json --out run/ --pilot-piece 07
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .form import (ENTITY_KINDS, FORM_VERSION, MAX_REFS_PER_SENTENCE, MIN_SENTENCE_WORDS, PLACEHOLDER,
                   RELEVANCE, TYPES, HarvestRefusal, grain_rules, marker, substitute, validate)
from .fr.cut import is_cut, joined_runs
from .fr.grammars import CHANNEL as GRAMMAR_CHANNEL
from .fr.grammars import read_values

#: the only server this module calls unless told otherwise
DEFAULT_HOST = "http://127.0.0.1:11434"


# ==================================================================================================
# the bake-off: one call per object per candidate, early stop on a disqualifying start
# ==================================================================================================

BUDGET_SENTENCES = 3
BAKEOFF_PROMPT_VERSION = "harvest-prompt/0.3"
#: a bad harvest disqualifies a tool rapidly: a candidate that
#: refuses on 5 of its first 8 objects is stopped there, for that role, and the minutes saved are recorded.
EARLY_STOP_AFTER, EARLY_STOP_OF = 5, 8

BAKEOFF_INSTRUCTIONS = """Tu analyses un extrait d'un dossier de consultation.

Rends UNIQUEMENT un objet JSON, sans texte autour, avec EXACTEMENT ces clés :
  node_id      : la valeur donnée ci-dessous, inchangée
  summary      : une liste d'au plus {budget} phrases. N'ÉCRIS AUCUN chiffre critique (date, heure,
                 montant, pourcentage, durée) : écris {{{{claim:<value_id>}}}} à sa place.
  summary_map  : [{{"sentence": 1, "children": ["v1", ...]}} ...] — une entrée par phrase, dans l'ordre.
                 "children" cite les identifiants de valeurs (v1, v2 ...) que la phrase utilise ; si la
                 phrase n'en cite aucun, laisse la liste vide : elle sera rattachée à l'objet lui-même.
  values       : [{{"value_id": "v1", "relevance": ..., "act": ..., "links": []}} ...] pour les valeurs
                 numérotées ci-dessous. Ne recopie JAMAIS la valeur elle-même.
  entities     : [{{"span": "<extrait du texte>", "kind": "person|organisation|place",
                 "relevance": ..., "act": ...}}] — le span doit être présent dans le texte ; les coupures
                 de ligne du PDF à l'intérieur d'un mot sont tolérées.
  interpreted  : {{"relevance": ..., "type": ..., "links": []}} pour l'objet entier.

relevance ∈ critical | important | informative | background
act (acte de langage, jamais une nature de valeur) ∈ informative | injunction | prohibition | condition | exception | definition | reference | penalty
"""


#: the closed vocabularies as a JSON schema, so the server's grammar enforces them and a model cannot
#: answer `duration` where an act is asked for, or misspell `injunction`. Form 0.5's three remaining
#: refusals on B′ were all of that shape, and no prompt wording prevents them the way a schema does.
def form_schema() -> dict:
    act = {"type": "string", "enum": list(TYPES)}
    relevance = {"type": "string", "enum": list(RELEVANCE)}
    return {
        "type": "object",
        "required": ["node_id", "summary", "summary_map", "values", "entities", "interpreted"],
        "properties": {
            "node_id": {"type": "string"},
            "summary": {"type": "array", "items": {"type": "string"}},
            "summary_map": {"type": "array", "items": {
                "type": "object", "required": ["sentence", "children"],
                "properties": {"sentence": {"type": "integer"},
                               "children": {"type": "array", "items": {"type": "string"}}}}},
            "values": {"type": "array", "items": {
                "type": "object", "required": ["value_id", "relevance", "act"],
                "properties": {"value_id": {"type": "string"}, "relevance": relevance, "act": act,
                               "links": {"type": "array", "items": {"type": "object"}}}}},
            "entities": {"type": "array", "items": {
                "type": "object", "required": ["span", "kind", "relevance", "act"],
                "properties": {"span": {"type": "string"},
                               "kind": {"type": "string", "enum": list(ENTITY_KINDS)},
                               "relevance": relevance, "act": act}}},
            "interpreted": {"type": "object", "required": ["relevance", "act"],
                            "properties": {"relevance": relevance, "act": act,
                                           "links": {"type": "array", "items": {"type": "object"}}}},
        },
    }


def call_ollama(model: str, prompt: str, timeout: float = 120.0,
                host: str = DEFAULT_HOST) -> tuple[str, float]:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": form_schema(),
                       "options": {"temperature": 0}}).encode("utf-8")
    request = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("response", ""), time.monotonic() - start


def prompt_for(node_id: str, text: str, values, children: list[str], min_words: int = 0) -> str:
    listed = "\n".join(f'  {v["value_id"]} : "{v["raw"]}" ({v["kind"]})' for v in values) or "  (aucune)"
    return (BAKEOFF_INSTRUCTIONS.format(budget=BUDGET_SENTENCES, min_words=min_words)
            + f"\nnode_id : {node_id}\nenfants : {json.dumps(children, ensure_ascii=False)}\n"
              f"valeurs numérotées :\n{listed}\n\ntexte :\n{text}\n")


def harvest_object(chunk_id: str, text: str, model: str, min_words: int = 0,
                   host: str = DEFAULT_HOST) -> dict:
    """One object, one call, one record. A refusal is recorded with its reason; nothing is repaired."""
    values = [{"value_id": f"v{i + 1}", "kind": v.kind, "normalized": v.normalized, "raw": v.raw,
               "start": v.start, "end": v.end} for i, v in enumerate(read_values(text))]
    # at window resolution the object is its own child, so a sentence maps to the values it draws on:
    # a model answered that way against form 0.2 and it was the better answer
    children = [f"{chunk_id}#leaf"]
    record: dict = {"chunk_id": chunk_id, "model": model, "grammar_values": values,
                    "grammar_channel": GRAMMAR_CHANNEL, "form_version": FORM_VERSION,
                    "prompt_version": BAKEOFF_PROMPT_VERSION, "raw_response": None, "latency_s": None}
    try:
        raw, seconds = call_ollama(model, prompt_for(chunk_id, text, values, children, min_words), host=host)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        record.update(ok=False, refusal="transport", detail=str(exc)[:200])
        return record
    record["raw_response"] = raw
    record["latency_s"] = round(seconds, 3)
    try:
        form = validate(raw, node_id=chunk_id, allowed_children=children,
                        allowed_claims=[v["value_id"] for v in values],
                        value_ids=[v["value_id"] for v in values], text=text, min_words=min_words)
    except HarvestRefusal as refusal:
        record.update(ok=False, refusal=refusal.reason, detail=refusal.detail)
        return record
    rendered = form.summary
    for value in values:                 # the placeholders take the grammar's value, never the model's
        if value["normalized"]:
            rendered = rendered.replace("{{claim:%s}}" % value["value_id"], value["normalized"])
    record.update(ok=True, refusal=None, rendered_summary=rendered,
                  form={"summary": form.summary, "summary_map": list(form.summary_map),
                        "interpreted": form.interpreted, "values": list(form.values),
                        "entities": list(form.entities)})
    return record


def run_bakeoff(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the harvest bake-off: one call per object per candidate")
    ap.add_argument("--store", required=True)
    ap.add_argument("--selection", required=True,
                    help="a JSON list of chunk ids, or {\"selected\": [...]}, in the order they are run")
    ap.add_argument("--models", required=True, help="comma-separated model tags, each a candidate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-words", type=int, default=0,
                    help="the node-prose role's floor; 0 for the structure role")
    ap.add_argument("--host", default=DEFAULT_HOST, help="the Ollama server (default: localhost)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    chunk_ids = selection["selected"] if isinstance(selection, dict) else selection
    con = sqlite3.connect(f"file:{args.store}?immutable=1", uri=True)
    texts: dict[str, str] = {}
    for chunk_id in chunk_ids:
        row = con.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            raise SystemExit(f"gate: chunk {chunk_id[:12]} absent from the store")
        texts[chunk_id] = row[0]
    con.close()
    (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        records, refused, stopped = [], 0, None
        for i, chunk_id in enumerate(chunk_ids):
            record = harvest_object(chunk_id, texts[chunk_id], model, args.min_words, host=args.host)
            records.append(record)
            refused += not record.get("ok")
            if i + 1 >= EARLY_STOP_OF and refused >= EARLY_STOP_AFTER and stopped is None:
                # disqualified for this role: the remaining calls are not spent
                stopped = {"after_objects": i + 1, "refusals": refused,
                           "rule": f"{EARLY_STOP_AFTER} refusals in the first {EARLY_STOP_OF} objects",
                           "calls_not_made": len(chunk_ids) - (i + 1)}
                break
        if stopped:
            seconds = sum(r["latency_s"] or 0 for r in records) / max(1, len(records))
            stopped["minutes_saved"] = round(stopped["calls_not_made"] * seconds / 60, 1)
            records.append({"disqualified": True, "model": model, **stopped})
        name = model.replace("/", "_").replace(":", "-")
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
            encoding="utf-8")
        refusals: dict[str, int] = {}
        for r in records:
            if not r.get("ok"):
                refusals[r.get("refusal", "unknown")] = refusals.get(r.get("refusal", "unknown"), 0) + 1
        print(json.dumps({"model": model, "objects": len(records), "stopped": stopped,
                          "ok": sum(1 for r in records if r.get("ok")), "refusals": refusals},
                         sort_keys=True))
    return 0

# ==================================================================================================
# the node roll-ups: K before T, a pilot, a writer and an optional challenger
# ==================================================================================================

WRITER = "mistral-small:24b"
#: a challenger worth running is a TEMPLATED tag: granite4.2 is published without a chat template, and
#: `models/granite42/Modelfile.30b-chat` builds this one. The CLI names no challenger unless told to.
CHALLENGER = "granite4.2:30b-chat"
FLOOR = grain_rules("rollup")["min_words"]     # 80, the roll-up floor
CHARS_PER_TOKEN = 2.73          # measured on a pilot, not assumed
OVERHEAD_TOKENS = 500           # instructions and the value list
ANSWER_TOKENS = 900             # room the answer needs inside num_ctx
COVERAGE_GATE = 5          # lot-specific values the pilot must cite for the harvest to be guided
#: what the smoke's object must be, after a first attempt drew a table dump. The objects this job
#: harvests are contractual prose; a pipe-delimited inventory is not one of them, and asking a model not
#: to write a critical value over a text whose grammar offers no value at all is an unsatisfiable
#: contract — it wrote the date it was echoing. These are the properties, declared as thresholds:
SMOKE_CHARS = (3000, 4200)
SMOKE_MIN_VALUES = 3       # something to cite: a {{claim:v1}} the model can use instead of the digits
SMOKE_LETTER_RATIO = 0.9   # letters over letters + digits + pipes: prose passes it, a table does not

NODE_PROMPT = """Tu analyses une pièce d'un dossier de consultation (marché public français).

Rends UNIQUEMENT un objet JSON. N'ÉCRIS AUCUN chiffre critique (date, heure, montant, pourcentage, durée) :
écris exactement {claim_marker}, en remplaçant v1 par l'identifiant de la valeur (liste ci-dessous).

Les valeurs marquées PROPRE sont les engagements chiffrés de cette pièce : ton résumé doit porter sur
elles. Les autres sont communes à tout le dossier ; ne construis pas le résumé autour d'elles.

  summary      : une liste de phrases en français, {floor} mots de contenu au minimum, centrée sur les
                 valeurs PROPRE, chacune citée par son {claim_marker}.
                 Au plus {max_refs} marqueurs par phrase, et au moins {min_words} mots de contenu par
                 phrase en dehors des marqueurs : une phrase qui n'est qu'une suite de marqueurs est
                 refusée.
  summary_map  : [{{"sentence": 1, "children": ["v1", ...]}} ...] — une entrée par phrase, dans l'ordre ;
                 liste vide si la phrase ne cite aucune valeur
  values       : [{{"value_id": "v1", "relevance": ..., "act": ...}} ...]
  entities     : [{{"span": "<extrait du texte>", "kind": "person|organisation|place",
                 "relevance": ..., "act": ...}}]
  interpreted  : {{"relevance": ..., "act": ...}}
  node_id      : la valeur donnée, inchangée

relevance ∈ critical | important | informative | background
act ∈ informative | injunction | prohibition | condition | exception | definition | reference | penalty

node_id : {node}
valeurs :
{values}

texte :
{text}
"""


def rendered_prompt(node: str, values: str, text: str, guided: bool = True) -> str:
    """The prompt as the model receives it. The markers come from the form, never from a typed brace."""
    template = NODE_PROMPT if guided else NODE_PROMPT.replace(
        "Les valeurs marquées PROPRE sont les engagements chiffrés de cette pièce : ton résumé doit "
        "porter sur\nelles. Les autres sont communes à tout le dossier ; ne construis pas le résumé "
        "autour d'elles.\n", "").replace(", centrée sur les\n                 valeurs PROPRE, chacune "
        "citée par son {claim_marker}", "")
    return template.format(floor=FLOOR, node=node, values=values, text=text,
                           claim_marker=marker("claim", "v1"), max_refs=MAX_REFS_PER_SENTENCE,
                           min_words=MIN_SENTENCE_WORDS)


def node_schema() -> dict:
    act = {"type": "string", "enum": list(TYPES)}
    relevance = {"type": "string", "enum": list(RELEVANCE)}
    return {"type": "object",
            "required": ["node_id", "summary", "summary_map", "values", "entities", "interpreted"],
            "properties": {
                "node_id": {"type": "string"},
                # minItems and minLength: without them the cheapest output a constrained decoder can
                # emit is [], which is what a model returned, and what was first misread as incapacity
                "summary": {"type": "array", "minItems": 1,
                            "items": {"type": "string", "minLength": 40}},
                "summary_map": {"type": "array", "minItems": 1, "items": {
                    "type": "object", "required": ["sentence", "children"],
                    "properties": {"sentence": {"type": "integer"},
                                   "children": {"type": "array", "items": {"type": "string"}}}}},
                "values": {"type": "array", "items": {
                    "type": "object", "required": ["value_id", "relevance", "act"],
                    "properties": {"value_id": {"type": "string"}, "relevance": relevance, "act": act}}},
                "entities": {"type": "array", "items": {
                    "type": "object", "required": ["span", "kind", "relevance", "act"],
                    "properties": {"span": {"type": "string"},
                                   "kind": {"type": "string", "enum": list(ENTITY_KINDS)},
                                   "relevance": relevance, "act": act}}},
                "interpreted": {"type": "object", "required": ["relevance", "act"],
                                "properties": {"relevance": relevance, "act": act}}}}


def smoke_object(con: sqlite3.Connection) -> tuple[tuple[str, str], dict]:
    """The smoke's object, by a declared rule rather than a hand pick.

    Among the level-1 parts of SMOKE_CHARS characters, **the first in sha256 order** whose grammar
    offers at least SMOKE_MIN_VALUES values and whose letters exceed SMOKE_LETTER_RATIO of its letters,
    digits and pipes. The first attempt took the smallest in the band and drew a pipe-delimited
    equipment inventory: the grammar offered nothing to cite and the model wrote the date it was
    echoing. One function, so the offline check and the run cannot select differently.
    """
    candidates = sorted(con.execute(
        "select chunk_id, text from chunks where level=1 and parent_id is not null "
        "and parent_id<>'' and length(text) between ? and ?", SMOKE_CHARS),
        key=lambda r: hashlib.sha256(r[0].encode("utf-8")).hexdigest())
    rejected = {"letter ratio": 0, "values": 0}
    for chunk_id, text in candidates:
        if letter_ratio(text) <= SMOKE_LETTER_RATIO:
            rejected["letter ratio"] += 1
            continue
        values = read_values(text)
        if len(values) < SMOKE_MIN_VALUES:
            rejected["values"] += 1
            continue
        return (chunk_id, text), {"node": chunk_id, "chars": len(text), "values": len(values),
                                  "letter_ratio": round(letter_ratio(text), 4),
                                  "first_eight_words": " ".join(text.split()[:8]),
                                  "candidates_in_band": len(candidates), "rejected": rejected,
                                  "examined": sum(rejected.values()) + 1}
    raise SystemExit(f"gate: no prose part with citable values among {len(candidates)} in the band; "
                     f"rejected {rejected}")


def letter_ratio(text: str) -> float:
    """Letters over letters, digits and pipes. Prose passes 0.9; a pipe-delimited inventory does not."""
    letters = sum(1 for c in text if c.isalpha())
    other = sum(1 for c in text if c.isdigit() or c == "|")
    return letters / (letters + other) if letters + other else 0.0


def fold(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


#: a pattern artefact of the register, not a commitment ("1 000 heures" splits into "000 heures")
ARTEFACTS = {"000heures"}


def own_values(piece: str, commitments: dict) -> set[str]:
    """The piece's own numeric commitments, folded. The denominator narrows further in `offered`."""
    out = set()
    for sentence in (commitments.get("pieces", {}).get(piece, {}).get("sentences") or []):
        for m in (sentence.get("matches") or []):
            if m.get("text") and fold(m["text"]) not in ARTEFACTS:
                out.add(fold(m["text"]))
    return out


def lot_specific(piece: str, commitments: dict) -> set[str]:
    """The piece's own commitments: those its register does not share with the rest of the family.

    Counting against every register match measures the cover and the template as well as the lot — "12
    heures", "1 an", "2 fois" belong to every piece. A value carried by at most two pieces is the lot's.
    It reproduced a reviewer's hand count for the pilot piece from the register alone, which is why it is
    the rule here rather than a list.
    """
    seen: dict[str, int] = {}
    for entry in (commitments.get("pieces") or {}).values():
        for value in {fold(m["text"]) for s in (entry.get("sentences") or [])
                      for m in (s.get("matches") or []) if m.get("text")}:
            seen[value] = seen.get(value, 0) + 1
    return {o for o in own_values(piece, commitments) if seen.get(o, 0) <= 2}


def offered(targets: set[str], values: list[dict]) -> set[str]:
    """Of those targets, the ones the grammar actually offers a value for — reported, never the denominator."""
    return {o for o in targets if any(o in fold(v["raw"]) or fold(v["raw"]) in o for v in values)}


#: what the server returns besides the content. A thinking model can put its reasoning here and leave the
#: constrained content empty — the hypothesis behind an empty `[]`, and the reason the whole
#: envelope is kept rather than the content string alone.
ENVELOPE = ("thinking", "done", "done_reason", "prompt_eval_count", "eval_count", "total_duration",
            "load_duration", "prompt_eval_duration", "eval_duration", "model", "created_at")


_CAPABILITIES: dict[str, list[str]] = {}


def capabilities(model: str, host: str = DEFAULT_HOST) -> list[str]:
    """What /api/show declares the model can do, asked once per model.

    Both halves were measured on a model published without a template: `think: true` is rejected 400 before the
    model is even loaded, `think: false` returns 200. So the field is not a call-breaker either way, and
    this gate is here for the contract rather than for the error — the field travels only where the server
    says it is understood, an unreadable /api/show means it is not sent, and the record says which.
    """
    if model not in _CAPABILITIES:
        body = json.dumps({"model": model}).encode("utf-8")
        request = urllib.request.Request(host.rstrip("/") + "/api/show", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            _CAPABILITIES[model] = [str(c) for c in (payload.get("capabilities") or [])]
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            _CAPABILITIES[model] = []
    return _CAPABILITIES[model]


def unload(model: str, host: str = DEFAULT_HOST) -> bool:
    """Free the model now instead of leaving it resident for the server's five minutes.

    NUM_PARALLEL is 1 and the memory is unified: the next model cannot load beside this one, and the
    exclusivity gate would be waiting on our own previous model. An empty prompt with keep_alive 0 is
    the server's own unload.
    """
    body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response.read()
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


#: where each record lands the moment it is made. The first run of this job held every record in
#: memory and wrote nodes.jsonl once at the end, so at the pilot's own report time the only thing on
#: disk was plan.json: if the job had died, forty minutes of calls would have died with it.
SINK: list[Path] = []


def emit(records: list[dict], record: dict) -> dict:
    """Keep the record and put it on disk at once, so a job that dies leaves what it already paid for."""
    records.append(record)
    if SINK:
        with SINK[0].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


_RESIDENT: list[str] = []


def switch(model: str, host: str = DEFAULT_HOST) -> None:
    """Unload the resident model when the next call needs a different one, and never otherwise."""
    if _RESIDENT and _RESIDENT[0] != model:
        unload(_RESIDENT.pop(), host)
    if not _RESIDENT:
        _RESIDENT.append(model)


def release(host: str = DEFAULT_HOST) -> None:
    """Nothing of ours stays resident when the job ends: the next run's exclusivity gate reads zero."""
    while _RESIDENT:
        unload(_RESIDENT.pop(), host)


def call_node(prompt: str, num_ctx: int, timeout: float, model: str = WRITER,
              think: bool = False, host: str = DEFAULT_HOST) -> tuple[dict, float]:
    """One call, think off where the server declares it understands the field, and the whole envelope."""
    body = {"model": model, "prompt": prompt, "stream": False, "format": node_schema(),
            "options": {"temperature": 0, "num_ctx": num_ctx}}
    if "thinking" in capabilities(model, host):
        body["think"] = think
    request = urllib.request.Request(host.rstrip("/") + "/api/generate", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8")), time.monotonic() - start


def envelope_of(payload: dict) -> dict:
    """Every field the server returned except the content and the token context, which are kept apart."""
    out = {k: payload[k] for k in ENVELOPE if k in payload}
    thinking = payload.get("thinking") or ""
    out["thinking_chars"] = len(thinking)
    out["content_chars"] = len(payload.get("response") or "")
    if "context" in payload:
        out["context_tokens"] = len(payload["context"])
    return out


def harvest_node(node: str, text: str, own: set[str], ceiling: int, timeout: float,
                 guided: bool = True, model: str = WRITER, host: str = DEFAULT_HOST) -> dict:
    # a value the store's text layer cut in two is never offered: its span is byte-exact, its hash
    # verifies and the number is wrong, so a model citing it would put a wrong critical value in the
    # prose through a placeholder that looks perfectly sound. It is recorded instead.
    read = read_values(text)
    cut = [v for v in read if is_cut(text, v.start)]
    # each offered value is kept whole in the record: an earlier run kept only a count, so its K rows named
    # value ids nobody could resolve without re-running the grammar version the job happened to carry
    values = [{"value_id": f"v{i + 1}", "kind": v.kind, "normalized": v.normalized, "raw": v.raw,
               "start": v.start, "end": v.end, "joined_runs": joined_runs(v.raw)}
              for i, v in enumerate(v for v in read if not is_cut(text, v.start))]
    listed = "\n".join(
        f'  {v["value_id"]} : "{v["raw"][:50]}" ({v["kind"]})'
        + ("  PROPRE" if guided and any(fold(v["raw"]) in o or o in fold(v["raw"]) for o in own) else "")
        for v in values) or "  (aucune)"
    needed = int(len(text) / CHARS_PER_TOKEN) + OVERHEAD_TOKENS + ANSWER_TOKENS
    record = {"node": node, "model": model, "chars": len(text), "values": len(values),
              "own_values_present": sum(1 for v in values
                                        if any(fold(v["raw"]) in o or o in fold(v["raw"]) for o in own)),
              "num_ctx": ceiling, "estimated_tokens": needed, "form": FORM_VERSION,
              "capabilities": capabilities(model, host), "think_field_sent": "thinking" in capabilities(model, host),
              "offered": values,
              "cut_values": [{"raw": v.raw, "kind": v.kind, "start": v.start,
                              "preceding": text[max(0, v.start - 24):v.start]} for v in cut]}
    if needed > ceiling:
        record.update(ok=False, refusal="does not fit", detail=f"{needed} tokens estimated, ceiling {ceiling}")
        return record
    switch(model, host)
    try:
        payload, seconds = call_node(rendered_prompt(node, listed, text, guided), ceiling, timeout, model,
                                     host=host)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # a client timeout says nothing about the model: it is unknown, and the record says so
        record.update(ok=False, refusal="timeout" if "timed out" in str(exc) else "transport",
                      outcome="unknown", detail=str(exc)[:120])
        return record
    raw = payload.get("response", "")
    record.update(latency_s=round(seconds, 1), raw_response=raw, envelope=envelope_of(payload),
                  thinking=(payload.get("thinking") or "")[:2000],
                  prompt_eval_count=payload.get("prompt_eval_count"))
    # the hypothesis: the reasoning goes to `thinking` and the constrained content comes back
    # empty. If that is what happened, it is re-called once with think off before any verdict is taken.
    if (not raw.strip() or raw.strip() in ("{}", "[]")) and (payload.get("thinking") or "").strip():
        record["think_leak"] = {"first_call": envelope_of(payload)}
        try:
            payload, seconds = call_node(rendered_prompt(node, listed, text, guided),
                                         ceiling, timeout, model, think=False, host=host)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            record.update(ok=False, refusal="timeout" if "timed out" in str(exc) else "transport",
                          outcome="unknown", detail=f"on the retry with think off: {str(exc)[:90]}")
            return record
        raw = payload.get("response", "")
        record.update(latency_s=round(seconds, 1), raw_response=raw, envelope=envelope_of(payload),
                      retried_with_think_off=True,
                      thinking=(payload.get("thinking") or "")[:2000])
    if (payload.get("prompt_eval_count") or 0) >= ceiling:
        record.update(ok=False, refusal="truncation",
                      detail=f"{payload.get('prompt_eval_count')} read, num_ctx {ceiling}")
        return record
    ids = [v["value_id"] for v in values]
    try:
        form = validate(raw, node_id=node, allowed_children=[f"{node}#leaf"], allowed_claims=ids,
                        value_ids=ids, text=text, **grain_rules("rollup"))
    except (HarvestRefusal, json.JSONDecodeError) as exc:
        record.update(ok=False, refusal=getattr(exc, "reason", "strict JSON"),
                      detail=str(getattr(exc, "detail", exc))[:120])
        return record
    cited = {v["raw"] for v in values if v["value_id"] in PLACEHOLDER.findall(form.summary)}
    covered = sorted(o for o in own if any(o in fold(c) or fold(c) in o for c in cited))
    record.update(ok=True, refusal=None, summary=form.summary,
                  summary_map=list(form.summary_map), k_values=list(form.values),
                  entities=list(form.entities), interpreted=form.interpreted,
                  coverage={"targets": len(own), "covered": len(covered),
                            "offered_by_the_grammar": len(offered(own, values)),
                            "uncitable": len(own) - len(offered(own, values))})
    return record


def run_nodes(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the node roll-ups: K before T, at document grain")
    for name in ("store", "commitments", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--pieces", required=True, type=Path,
                    help="a JSON object mapping a document id prefix to its piece name (a two-digit lot "
                         "piece, or a name such as RC or CCAP): the declared scope, never a path filter")
    ap.add_argument("--pilot-piece", required=True,
                    help="the piece the pilot reads whole before any other call, and whose lot-specific "
                         "values decide the writer")
    ap.add_argument("--whole-piece", default="CCAP",
                    help="the long piece a challenger that passed the pilot is asked to read whole")
    ap.add_argument("--ceiling", type=int, required=True, help="the writer's own context ceiling")
    ap.add_argument("--writer", default=WRITER, help="the writer's tag")
    ap.add_argument("--challenger", default="",
                    help="the challenger's tag as the server lists it; empty means none, and the pilot "
                         "runs the writer alone")
    ap.add_argument("--challenger-ceiling", type=int, default=131072)
    ap.add_argument("--coverage-gate", type=int, default=COVERAGE_GATE,
                    help="lot-specific values the pilot must cite for the harvest to be guided")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="a node-level call is slow; a client timeout is recorded as unknown, "
                         "never as a refusal by the model")
    ap.add_argument("--host", default=DEFAULT_HOST, help="the Ollama server (default: localhost)")
    ap.add_argument("--plan-only", action="store_true", help="print the plan and make no call")
    ap.add_argument("--smoke-object", action="store_true",
                    help="print the facts of the object the smoke would take, on CPU, and make no call")
    ap.add_argument("--smoke", default="", help="one call with this model over the smallest part, to prove "
                                                "the launcher end to end before the declared job")
    args = ap.parse_args(argv)
    # the progress lines of a long job must not sit in a pipe buffer for its whole run
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commitments = json.loads(Path(args.commitments).read_text(encoding="utf-8"))
    scope = json.loads(args.pieces.read_text(encoding="utf-8"))
    if not isinstance(scope, dict) or not scope:
        raise SystemExit("gate: --pieces declares no piece")

    con = sqlite3.connect(f"file:{args.store}?immutable=1", uri=True)
    core = [r[0] for r in con.execute(
        "select distinct doc_id from chunks where level=1 and (parent_id is null or parent_id='')")]
    plan, docs = [], {}
    for doc in sorted(core):
        roll = con.execute("select chunk_id, text from chunks where level=1 and doc_id=? "
                           "and (parent_id is null or parent_id='')", (doc,)).fetchone()
        if roll is None:
            continue
        name = next((v for k, v in scope.items() if doc.startswith(k)), None)
        if name is None:
            continue
        needed = int(len(roll[1]) / CHARS_PER_TOKEN) + OVERHEAD_TOKENS + ANSWER_TOKENS
        parts = con.execute("select chunk_id, text from chunks where level=1 and doc_id=? "
                            "and parent_id is not null and parent_id<>'' order by chunk_id", (doc,)).fetchall()
        docs[doc] = {"piece": name, "roll": roll, "parts": parts, "whole": needed <= args.ceiling}
        plan.append({"doc": doc[:8], "piece": name, "chars": len(roll[1]), "tokens": needed,
                     "whole": needed <= args.ceiling, "parts": len(parts)})
    con.close()
    calls = sum(1 if p["whole"] else p["parts"] + 1 for p in plan)
    print(json.dumps({"documents": len(plan), "whole": sum(p["whole"] for p in plan),
                      "through_parts": sum(not p["whole"] for p in plan), "calls": calls}, sort_keys=True))
    (out / "plan.json").write_text(json.dumps({"calls": calls, "documents": plan}, indent=1,
                                              sort_keys=True) + "\n", encoding="utf-8")
    if args.smoke_object:
        con = sqlite3.connect(f"file:{args.store}?immutable=1", uri=True)
        _, facts = smoke_object(con)
        con.close()
        print(json.dumps(facts, ensure_ascii=False, indent=1, sort_keys=True))
        return 0
    if args.plan_only:
        return 0

    if args.smoke:
        con = sqlite3.connect(f"file:{args.store}?immutable=1", uri=True)
        # a representative part: the smallest cannot reach the prose floor, and a table dump offers
        # nothing to cite — the smoke would fail on its own object rather than on the path
        row, facts = smoke_object(con)
        con.close()
        print(json.dumps({"stage": "smoke-object", **facts}, ensure_ascii=False, sort_keys=True))
        # 300 s: a cold load precedes the first token, and a client timeout would be recorded as a
        # broken path, which it would not be
        r = harvest_node(row[0], row[1], set(), 8192, 300.0, False, args.smoke, host=args.host)
        r["stage"] = "smoke"
        (out / "smoke.json").write_text(json.dumps(r, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
                                        encoding="utf-8")
        release(args.host)
        # the path completed if a record exists, whatever the answer was: the call was made, the JSON came
        # back and the validator ruled. Only a broken path returns non-zero here — a refused answer is the
        # launcher's gate to apply, after it has pulled the record back and kept it.
        path_broke = r.get("refusal") in ("transport", "timeout")
        print(json.dumps({"smoke": args.smoke, "node": row[0][:12], "chars": len(row[1]),
                          "values": r.get("values"), "ok": r.get("ok"), "refusal": r.get("refusal"),
                          "detail": r.get("detail"), "capabilities": r.get("capabilities"),
                          "path": "broken" if path_broke else "complete",
                          "thinking_chars": (r.get("envelope") or {}).get("thinking_chars")},
                         sort_keys=True))
        return 3 if path_broke else 0

    records, started = [], time.monotonic()
    SINK.clear()
    SINK.append(out / "nodes.jsonl")
    SINK[0].write_text("", encoding="utf-8")
    pilot_entry = next((v for v in docs.values() if v["piece"] == args.pilot_piece), None)
    if pilot_entry is None:
        raise SystemExit(f"gate: piece {args.pilot_piece} is not among the documents, and it is the "
                         "pilot's object")
    own_pilot = lot_specific(args.pilot_piece, commitments)
    base_writer = args.writer
    challenger = (args.challenger or "").strip()
    ceilings = {base_writer: args.ceiling}
    if challenger:
        ceilings[challenger] = args.challenger_ceiling
    else:
        print(json.dumps({"challenger": None,
                          "reason": "none declared: the pilot runs the writer alone and no node is "
                                    "harvested by a challenger"}, sort_keys=True))

    pilot = {}
    for model in ([base_writer, challenger] if challenger else [base_writer]):
        r = harvest_node(pilot_entry["roll"][0], pilot_entry["roll"][1], own_pilot, ceilings[model],
                         args.timeout, True, model, host=args.host)
        r["stage"] = "pilot"
        emit(records, r)
        pilot[model] = (r.get("coverage") or {}).get("covered", 0) if r.get("ok") else -1
        print(json.dumps({"stage": "pilot", "model": model, "ok": r.get("ok"), "refusal": r.get("refusal"),
                          "covered": pilot[model], "of": len(own_pilot)}, sort_keys=True))

    whole = next((v for v in docs.values() if v["piece"] == args.whole_piece), None)
    ccap_whole = None
    if challenger and pilot.get(challenger, -1) >= args.coverage_gate and whole is not None:
        r = harvest_node(whole["roll"][0], whole["roll"][1], set(), ceilings[challenger], args.timeout, True,
                         challenger, host=args.host)
        r["stage"] = "pilot-ccap"
        emit(records, r)
        ccap_whole = bool(r.get("ok"))
        print(json.dumps({"stage": "pilot-ccap", "model": challenger, "ok": r.get("ok"),
                          "refusal": r.get("refusal")}, sort_keys=True))

    # the rule, declared before the run: the challenger writes only if it covered more than the writer
    writer = challenger if challenger and pilot.get(challenger, -1) > pilot[base_writer] else base_writer
    guided = max(pilot.values()) >= args.coverage_gate
    print(json.dumps({"decision": {"writer": writer, "guided": guided, "challenger": challenger or None,
                                   "coverage": {k: v for k, v in pilot.items()},
                                   "of": len(own_pilot), "ccap_whole_by_challenger": ccap_whole}},
                     sort_keys=True))

    for doc, entry in docs.items():
        own = lot_specific(entry["piece"], commitments) if entry["piece"].isdigit() else set()
        ceiling = ceilings[writer]
        # the long pieces go whole to the challenger when it proved it can read them
        big_model = challenger if ccap_whole else writer
        big_ceiling = ceilings[big_model]
        if entry["whole"]:
            r = harvest_node(entry["roll"][0], entry["roll"][1], own, ceiling, args.timeout, guided, writer,
                             host=args.host)
            r.update(piece=entry["piece"], level="document")
            emit(records, r)
        elif ccap_whole:
            r = harvest_node(entry["roll"][0], entry["roll"][1], own, big_ceiling, args.timeout, guided,
                             big_model, host=args.host)
            r.update(piece=entry["piece"], level="document", whole_by=big_model)
            emit(records, r)
        else:
            chosen = [(cid, text) for cid, text in entry["parts"]
                      if not own or any(o in fold(text) for o in own)]
            if not chosen:                      # the registers point nowhere in this piece: read it all
                chosen = entry["parts"]
            skipped = [cid for cid, _ in entry["parts"] if cid not in {c for c, _ in chosen}]
            read, unread = [], list(skipped)
            for chunk_id, text in chosen:
                r = harvest_node(chunk_id, text, own, ceiling, args.timeout, guided, writer, host=args.host)
                r.update(piece=entry["piece"], level="part", parent=entry["roll"][0])
                emit(records, r)
                (read if r.get("ok") else unread).append(chunk_id)
            record_skipped = skipped
            joined = " ".join(r["summary"] for r in records
                              if r.get("level") == "part" and r.get("parent") == entry["roll"][0]
                              and r.get("ok"))
            roll = harvest_node(entry["roll"][0], joined or entry["roll"][1][:20000], own,
                                ceiling, args.timeout, guided, writer, host=args.host)
            roll.update(piece=entry["piece"], level="document", partial=True,
                        parts_read=read, parts_not_read=unread, parts_skipped=record_skipped,
                        parts_total=len(entry["parts"]))
            emit(records, roll)
        print(json.dumps({"piece": entry["piece"], "ok": records[-1].get("ok"),
                          "refusal": records[-1].get("refusal"),
                          "elapsed_min": round((time.monotonic() - started) / 60, 1)}, sort_keys=True))

    written = len([l for l in SINK[0].read_text(encoding="utf-8").splitlines() if l.strip()])
    if written != len(records):
        raise SystemExit(f"gate: {written} record(s) on disk against {len(records)} made")
    release(args.host)
    nodes = [r for r in records if r.get("level") == "document"]
    summary = {"guided": guided, "writer": writer, "challenger": challenger or None, "ccap_whole_by_challenger": ccap_whole,
               "pilot_coverage": pilot, "lot_specific_targets": len(own_pilot), "documents": len(nodes), "valid": sum(1 for r in nodes if r.get("ok")),
               "partial": [r["piece"] for r in nodes if r.get("partial")],
               "refusals": {}, "minutes": round((time.monotonic() - started) / 60, 1),
               "coverage_total": sum((r.get("coverage") or {}).get("covered", 0) for r in nodes)}
    for r in records:
        if not r.get("ok"):
            summary["refusals"][r.get("refusal")] = summary["refusals"].get(r.get("refusal"), 0) + 1
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0

# ==================================================================================================
# pass 2 at window grain: the model fills the form, the pipeline protects the values
# ==================================================================================================

WINDOW_PROMPT = """Tu lis un extrait d'un document de marché public français.

TEXTE :
---
{text}
---

VALEURS RELEVÉES DANS CE TEXTE par un outil déterministe (tu ne dois PAS les recopier autrement) :
{values}

Réponds UNIQUEMENT par un objet JSON, sans commentaire, avec exactement ces clés :

  "summary"   : 3 à 5 phrases de prose française qui résument ce que ce texte engage. Écris les
                chiffres EXACTEMENT comme le texte les écrit. N'invente aucun chiffre, aucune date,
                aucun montant. DANS CES PHRASES : de la prose seulement — aucune accolade, aucun
                marqueur, aucun code. Au moins trente mots de contenu.

  "values"    : une LISTE D'OBJETS, un par valeur ci-dessus. Chaque objet a exactement trois clés :
                  "value_id"  : l'identifiant ci-dessus, par exemple "v1"
                  "relevance" : CE QUE LA VALEUR PÈSE — uniquement l'un de : {relevance}
                  "act"       : CE QUE LA PHRASE FAIT — uniquement l'un de : {types}
                Ne mets jamais un mot de la liste "act" dans "relevance", ni l'inverse : les deux
                listes sont différentes et n'ont aucun mot en commun.

  "entities"  : liste (éventuellement vide) d'objets {{"kind": "...", "span": "..."}} recopiés
                littéralement du texte ; kind parmi person, organisation, place.

  "interpreted": un objet à deux clés pour le texte entier, avec les mêmes deux listes :
                {{"relevance": l'un de {relevance}, "act": l'un de {types}}}

EXEMPLE DE RÉPONSE BIEN FORMÉE (la forme, pas le contenu) :
{{"summary": ["Le titulaire assure la maintenance préventive des équipements et remet un compte rendu
après chaque intervention sur le site de l'établissement adhérent."],
  "values": [{{"value_id": "v1", "relevance": "critical", "act": "condition"}},
             {{"value_id": "v2", "relevance": "informative", "act": "informative"}}],
  "entities": [],
  "interpreted": {{"relevance": "critical", "act": "condition"}}}}"""


#: A timeout is NOT a refusal. A refusal is something
#: learnt — the model broke the contract, wrote a figure the grammar never offered, or the server cut
#: the input (R1c). A timeout or a transport error teaches nothing about the model, and counting it as
#: a refusal lets a slow or busy server look like a bad model. Three outcomes, and the exit gate reads
#: only the two that measure something.
#: the word the two vocabularies share — `informative` today. A K row carrying it in either field is
#: flagged, because the check cannot tell a correct one from a swapped one until the lists are disjoint.
AMBIGUOUS = set(RELEVANCE) & set(TYPES)
ACCEPTED, REFUSED, UNKNOWN = "accepted", "refused", "unknown"
UNKNOWN_REASONS = ("timeout", "transport")


def outcome_of(record: dict) -> str:
    if record.get("ok"):
        return ACCEPTED
    return UNKNOWN if record.get("refusal") in UNKNOWN_REASONS else REFUSED


#: Widened on 2026-09-12 by the run it was sized for. The first formula — slowest pilot call ×
#: workers + 60 s — gave 204.9 s, and the core's slowest call came in at 186.99 s: 91 % of it, one
#: slow window away from an "unknown" that would have said nothing about the model. Three pilot calls
#: cannot see a tail: the 214 core calls ran at a median of 40.75 s, p90 71.9, p95 79.0 and a maximum
#: of 187.0 — a tail 4.6× the median. So the queue term keeps its worst case and gains half again.
TIMEOUT_QUEUE_FACTOR = 1.5


def timeout_for(per_call_seconds: float, workers: int) -> float:
    """Sized from what the pilot measured, for the worst case — the
    parallel setting does not take, the workers queue, and the last request waits for all the others
    — plus half again for the tail the pilot is too small to show."""
    per_call = max(per_call_seconds, 1.0)
    return round(per_call * max(workers, 1) * TIMEOUT_QUEUE_FACTOR + max(60.0, per_call), 1)


#: the words that carry no content and cannot make a repeat meaningful on their own
_FUNCTION_WORDS = {
    "de", "du", "des", "la", "le", "les", "l", "un", "une", "et", "ou", "à", "au", "aux", "en", "dans",
    "sur", "sous", "par", "pour", "avec", "sans", "est", "sont", "être", "ont", "a", "qui", "que",
    "dont", "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur", "leurs", "il", "elle", "ils",
    "doit", "doivent", "peut", "peuvent", "plus", "tout", "tous", "toute", "toutes", "ne", "pas", "se",
}
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    """One summary ENTRY holds several sentences. Form 0.7 missed this and counted a phrase repeated
    across two sentences of one entry as a repeat inside one — 67 of a run's 82 refusals."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def rendered_text(sentence: str, values: list[dict]) -> str:
    """The line as a human sees it: every marker resolved to the value it stands for. Judging the
    marker-STRIPPED line invents phrases that were never written — « sous ouvrées maximum » is what
    closes up when the placeholders vanish, and six of that run's refusals were that phantom."""
    by_id = {v["value_id"]: (v.get("raw") or "").strip() for v in values}
    return re.sub(r"\{\{claim:([A-Za-z0-9_.:-]{1,64})\}\}",
                  lambda m: by_id.get(m.group(1), m.group(0)), sentence)


def fold_words(text: str) -> str:
    return " " + " ".join(re.split(r"[^\w'’-]+", (text or "").casefold())).strip() + " "


def repeated_span(sentence: str, length: int = 3) -> str | None:
    """A span of `length` words repeated in ONE sentence, carrying at least two content words —
    « le titulaire doit » is a subject across clauses, not padding, and it was 24 of that run's 82."""
    words = [w for w in re.split(r"[^\w'’-]+", sentence.casefold()) if w]
    seen: dict[tuple, int] = {}
    for start in range(max(0, len(words) - length + 1)):
        gram = tuple(words[start:start + length])
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] > 1 and sum(1 for w in gram if w not in _FUNCTION_WORDS and len(w) > 3) >= 2:
            return " ".join(gram)
    return None


def refuse(message: str) -> None:
    sys.stderr.write(f"runner: REFUSED — {message}\n")
    raise SystemExit(2)


def post(host: str, path: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(host.rstrip("/") + path,
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def num_ctx_for(text: str, ceiling: int) -> int:
    """Sized from the window, not from a habit: the prompt, the text at four characters a token, and
    room for the answer, rounded up to a power of two and capped at what the model declares."""
    needed = len(WINDOW_PROMPT) / 4 + len(text) / 4 + 900
    size = 1 << max(12, math.ceil(math.log2(max(needed, 1))))
    return int(min(size, ceiling))


#: Pass 1's cards carry `piece` as the SOURCE FILE NAME — « 12.CCTP_Lot _Engins de manutention.pdf »,
#: « CCAP REF-1 signé.pdf » — not as the two-digit piece the registers are keyed by. The pilot's
#: first launch refused on that ("piece 12 offers 0 windows") and it would otherwise have been worse
#: than a refusal: the same field feeds `lot_specific`, so every window would have been scored against
#: an empty target set and reported a silent coverage of 0/0. The number is derived here, once.
_PIECE_NUMBER = re.compile(r"^\s*(\d{2})\s*[.\-_]")


def piece_number(piece: str | None) -> str | None:
    """The two-digit piece a card belongs to, or None when its file carries no number (CCAP, RC, AE)."""
    match = _PIECE_NUMBER.match(str(piece or ""))
    return match.group(1) if match else None


def offered_values(text: str) -> list[dict]:
    return [{"value_id": f"v{i + 1}", "kind": v.kind, "raw": v.raw, "normalized": v.normalized}
            for i, v in enumerate(read_values(text))]


def render_window(text: str, values: list[dict]) -> str:
    listing = "\n".join(f"  {v['value_id']} : « {v['raw'].strip()} »"
                        + (f" ({v['normalized']})" if v["normalized"] else "")
                        for v in values) or "  (aucune)"
    return WINDOW_PROMPT.format(text=text, values=listing,
                         relevance=", ".join(RELEVANCE), types=", ".join(TYPES))


def judge(body: dict, text: str, values: list[dict], node_id: str) -> dict:
    """The pipeline's half: substitute, derive the map, and let the form judge at window grain."""
    sentences = body.get("summary")
    if isinstance(sentences, str):
        sentences = [sentences]
    if not isinstance(sentences, list) or not sentences:
        raise HarvestRefusal("no summary", "the answer carries no summary")
    # a model asked for prose can return objects; that is a refusal to be counted, not a TypeError
    # thrown three frames down inside the grammar
    if not all(isinstance(s, str) for s in sentences):
        raise HarvestRefusal("summary is not prose",
                             f"sentence types {sorted({type(s).__name__ for s in sentences})}")
    # and the same discipline one field over: 19 calls of the first run raised
    # « AttributeError: 'str' object has no attribute 'get' » because the model answered `values`
    # with a list of strings. A shape the form does not expect is a refusal with a name, never an
    # exception three frames down.
    rows = body.get("values")
    if rows is not None and not isinstance(rows, list):
        raise HarvestRefusal("values are not K rows", f"values is {type(rows).__name__}")
    if rows and not all(isinstance(r, dict) for r in rows):
        raise HarvestRefusal("values are not K rows",
                             f"row types {sorted({type(r).__name__ for r in rows})}")
    placed, made = substitute(sentences, values)
    own_leaf = f"{node_id}#leaf"
    per_sentence = []
    for index, sentence in enumerate(placed, start=1):
        cited = re.findall(r"\{\{claim:([A-Za-z0-9_.:-]{1,64})\}\}", sentence)
        # a sentence that cites no value still stands on this window: it maps to the node's own leaf,
        # which is the form's way of saying "supported by the text as a whole" rather than by a value
        per_sentence.append({"sentence": index, "children": cited or [own_leaf]})
    # A REPEATED PHRASE IS A FORM DEFECT — form 0.8, after a non-author read all 82 refusals of a run
    # and found the 0.7 rule catching three things it should not:
    #   67 repeated ACROSS sentences (« Le titulaire doit … . Le titulaire doit … »), which is correct
    #      French — the code treated one summary ENTRY as one sentence, and an entry holds several;
    #    6 were phantoms of the marker-stripped text (« sous ouvrées maximum » is what closes up when
    #      the placeholders are removed), so the judgement must be on the RENDERED line, not the
    #      stripped one;
    #    9 repeated inside one sentence as declared — and of those, a span that repeats in the WINDOW
    #      too is faithful quotation, not padding.
    # Beside it, the exact rule: the same value cited twice in one sentence, which is what the defect
    # was when it was first seen (« 7 jours sur 7 » twice) and needs no heuristic at all.
    source_repeats = []
    for entry_index, entry in enumerate(placed, start=1):
        rendered = rendered_text(entry, values)
        for sentence in split_sentences(rendered):
            twice = repeated_span(sentence)
            if twice:
                if fold_words(twice) in fold_words(text):
                    source_repeats.append(twice)      # the page says it twice; the summary is faithful
                    continue
                raise HarvestRefusal("repeated phrase",
                                     f"entry {entry_index} says « {twice} » twice in one sentence")
        for sentence in split_sentences(entry):
            cited = re.findall(r"\{\{claim:([A-Za-z0-9_.:-]{1,64})\}\}", sentence)
            doubled = [v for v in set(cited) if cited.count(v) > 1]
            if doubled:
                raise HarvestRefusal("repeated phrase",
                                     f"entry {entry_index} cites {doubled[0]} twice in one sentence")

    # ONE NORMALISED VALUE, ONE CLASSIFICATION. « 24h » and « 24 heures » are two spans of one value
    # (PT24H) and came back critical in one row and important in the other. The K layer is about
    # values, not spans: the first classification stands, the ids it covers are recorded, and a
    # disagreement is kept in the record rather than averaged away.
    ids = [v["value_id"] for v in values]
    canonical_of = {v["value_id"]: (v.get("normalized") or f"id:{v['value_id']}") for v in values}
    deduped: list[dict[str, Any]] = []
    by_canonical: dict[str, dict[str, Any]] = {}
    disagreements: list[dict[str, Any]] = []
    for row in [v for v in (body.get("values") or []) if v.get("value_id") in ids]:
        key = canonical_of[row["value_id"]]
        first = by_canonical.get(key)
        if first is None:
            # RELEVANCE and TYPES share `informative`, so a swapped one passes the vocabulary check
            # invisibly and the swaps counted on a first run are a floor. Until the two lists are made
            # disjoint, the row says when it carries the shared word, and a re-run can count what the
            # first run could not see.
            row = {**row, "covers": [row["value_id"]], "canonical": key,
                   "ambiguous_vocabulary": bool({row.get("relevance"), row.get("act")} & AMBIGUOUS)}
            by_canonical[key] = row
            deduped.append(row)
            continue
        first["covers"].append(row["value_id"])
        if (first.get("relevance"), first.get("act")) != (row.get("relevance"), row.get("act")):
            disagreements.append({"canonical": key, "kept": first["value_id"],
                                  "kept_as": [first.get("relevance"), first.get("act")],
                                  "dropped": row["value_id"],
                                  "dropped_as": [row.get("relevance"), row.get("act")]})
    payload = {"node_id": node_id, "summary": placed, "summary_map": per_sentence,
               "values": deduped,
               "entities": body.get("entities") or [],
               "interpreted": body.get("interpreted") or {}}
    form = validate(json.dumps(payload, ensure_ascii=False), node_id=node_id,
                    allowed_children=[*ids, own_leaf], allowed_claims=ids, value_ids=ids,
                    text=text, **grain_rules("window"))
    return {"form": form, "summary": placed, "map": per_sentence, "substitutions": made,
            "k_rows": payload["values"], "k_disagreements": disagreements}


def coverage_of(made: list[dict], targets: set[str]) -> tuple[int, int]:
    cited = {re.sub(r"\s+", "", m["raw"] or "").casefold() for m in made}
    return len({t for t in targets if t in cited}), len(targets)


#: coverage is a PIECE's number, not a window's mean.
#: A thin window offering one lot-specific value swings a mean by a third and says nothing about the
#: reading; summed over a piece, one missed value is one missed value. The threshold applies only
#: where there is enough to measure — three lot-specific values offered — and the rest are reported
#: with no threshold rather than judged against a number their size cannot support.
COVERAGE_FLOOR = 0.5
COVERAGE_MIN_OFFERED = 3


def coverage_by_piece(records: list[dict]) -> dict[str, dict]:
    pieces: dict[str, dict] = {}
    for record in records:
        if not record.get("ok"):
            continue
        # keyed by the TWO-DIGIT piece, which is what the registers are keyed by and what a reader
        # says out loud — not by « 12.CCTP_Lot _Engins de manutention.pdf »
        key = piece_number(record.get("piece")) or str(record.get("piece") or "?")
        bucket = pieces.setdefault(key, {"windows": 0, "cited": 0, "offered": 0,
                                         "file": str(record.get("piece") or "")})
        bucket["windows"] += 1
        bucket["cited"] += int(record.get("lot_specific_cited") or 0)
        bucket["offered"] += int(record.get("lot_specific_offered") or 0)
    for bucket in pieces.values():
        offered = bucket["offered"]
        bucket["coverage"] = f"{bucket['cited']}/{offered}"
        bucket["rate"] = round(bucket["cited"] / offered, 3) if offered else None
        bucket["gated"] = offered >= COVERAGE_MIN_OFFERED
        bucket["passes"] = (bucket["rate"] >= COVERAGE_FLOOR) if bucket["gated"] else None
    return dict(sorted(pieces.items()))


def run_window(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pass 2 at window grain: the model fills the form, the "
                                             "pipeline protects the values")
    ap.add_argument("--store", required=True)
    ap.add_argument("--cards", required=True, help="pass 1's core_abstracts.jsonl: the window set")
    ap.add_argument("--commitments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sidecar", default="")
    ap.add_argument("--model", default="granite4.2:8b-chat")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--ceiling", type=int, default=32768)
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="per request, seconds; 0 sizes it from the pilot's own measurement")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel requests for the CORE; the pilot always runs at 1, which is what "
                         "makes its per-call measurement an honest basis for the timeout")
    ap.add_argument("--stop-after", type=float, default=7200.0, help="seconds, the declared budget")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pilot-piece", default=None,
                    help="the two-digit piece whose first three windows are the pilot, when "
                         "--pilot-nodes names none")
    ap.add_argument("--pilot-nodes", default="",
                    help="the pilot's three windows, by id prefix, comma-separated; empty falls back "
                         "to the first three windows of --pilot-piece")
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--max-exceptions", type=int, default=5,
                    help="a call raising outside its caught classes is recorded as a refusal; more "
                         "than this many and the run stops, because a repeated crash is a defect and "
                         "not a measurement")
    ap.add_argument("--only", type=Path, default=None,
                    help="a file of node_ids, one per line: the CORE runs these windows and no "
                         "others. The pilot is unaffected — it is the launcher's own check and its "
                         "three windows are declared apart. An id naming no window is a refusal, never a "
                         "silent skip, because a subset that quietly shrinks is not the subset asked for.")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    import hashlib
    import sqlite3

    db = sqlite3.connect(f"file:{Path(a.store).resolve()}?mode=ro&immutable=1", uri=True)
    # the store's digest travels into every sidecar row: a row is stale when its source moves
    digest = hashlib.sha256()
    with open(a.store, "rb") as handle_store:
        for block in iter(lambda: handle_store.read(1 << 20), b""):
            digest.update(block)
    store_sha = digest.hexdigest()
    cards = [json.loads(l) for l in Path(a.cards).read_text(encoding="utf-8").splitlines() if l.strip()]
    windows = [c for c in cards if c.get("level") == "window" and c.get("node_id")]
    if not windows:
        refuse("the card set holds no window: there is nothing to fill a form over")
    commitments = json.loads(Path(a.commitments).read_text(encoding="utf-8"))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    done: dict[str, dict] = {}
    if a.resume and records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["node_id"]] = record

    host_name = socket.gethostname()
    started = time.time()
    handle = records_path.open("a", encoding="utf-8", buffering=1)

    # every window's text read ONCE, before any worker starts: a sqlite connection is not shared
    # across threads, and a job that reads the store from four workers is a job with a second defect
    # waiting for it
    texts: dict[str, tuple[str, str]] = {}
    for card in windows:
        row = db.execute("select text, doc_id from chunks where chunk_id=?",
                         (card["node_id"],)).fetchone()
        if row is not None:
            texts[card["node_id"]] = (row[0], row[1])

    def call(card: dict, timeout: float) -> dict:
        node_id = card["node_id"]
        if node_id not in texts:
            return {"node_id": node_id, "ok": False, "refusal": "absent from the store",
                    "outcome": REFUSED}
        text, doc_id = texts[node_id]
        values = offered_values(text)
        prompt = render_window(text, values)
        num_ctx = num_ctx_for(text, a.ceiling)
        record = {"node_id": node_id, "piece": card.get("piece"), "doc_id": doc_id,
                  "model": a.model, "form": FORM_VERSION, "grain": "window",
                  "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "num_ctx": num_ctx,
                  "offered": len(values), "made_on": host_name}
        t0 = time.time()
        try:
            payload = post(a.host, "/api/generate",
                           {"model": a.model, "prompt": prompt, "stream": False, "think": False,
                            "format": "json",
                            "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx}}, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # rule 64: nothing was learnt about the model here, so this is UNKNOWN and not a refusal
            timed_out = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
            return {**record, "ok": False, "refusal": "timeout" if timed_out else "transport",
                    "detail": str(exc)[:120], "timeout_s": timeout,
                    "seconds": round(time.time() - t0, 2)}
        read = payload.get("prompt_eval_count") or 0
        record.update(seconds=round(time.time() - t0, 2), prompt_eval_count=read,
                      eval_count=payload.get("eval_count"), done_reason=payload.get("done_reason"))
        if read >= num_ctx:                         # R1c, pass 1's gate: the server cut the input
            return {**record, "ok": False, "refusal": "input truncated by the server",
                    "detail": f"{read} prompt tokens read, num_ctx {num_ctx}"}
        try:
            body = json.loads(payload.get("response") or "")
        except json.JSONDecodeError as exc:
            return {**record, "ok": False, "refusal": "strict JSON", "detail": str(exc)[:120],
                    "raw_response": payload.get("response") or ""}
        try:
            verdict = judge(body, text, values, node_id)
        except HarvestRefusal as exc:
            # THE ANSWER TRAVELS WITH THE REFUSAL. The first run's 49 vocabulary refusals cannot be
            # re-scored offline, because only the offending value was kept and not what the model
            # wrote — so the size of a form defect had to be estimated from refusal details instead
            # of measured. A refused record now carries the answer it was refused for.
            return {**record, "ok": False, "refusal": exc.reason, "detail": exc.detail[:160],
                    "raw_response": payload.get("response") or ""}
        number = piece_number(card.get("piece"))
        own = lot_specific(number, commitments) if number else set()
        targets = offered(own, values) if own else set()
        cited, total = coverage_of(verdict["substitutions"], targets)
        return {**record, "ok": True, "refusal": None,
                "substitutions": len(verdict["substitutions"]),
                "summary": verdict["summary"], "summary_map": verdict["map"],
                "k_rows": verdict["k_rows"], "k_rows_count": len(verdict["k_rows"]),
                "k_disagreements": verdict["k_disagreements"],
                "lot_specific_offered": total, "lot_specific_cited": cited,
                "coverage": f"{cited}/{total}" if total else "0/0",
                "coverage_rate": round(cited / total, 3) if total else None}

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    writing = threading.Lock()
    raised: list[int] = []                 # exceptions per call to run(), judged after the record

    progress = (out / "progress.log").open("a", encoding="utf-8", buffering=1)

    def keep(record: dict) -> None:
        """The write path, and NOTHING in it may depend on the channel back to the caller. A first
        core run froze with nine records on disk while the server went on answering: whatever
        stopped the main thread, a progress line printed onto a remote shell's channel is a dependency the
        records must not have. Progress goes to a file beside them; stdout is best-effort."""
        record["outcome"] = outcome_of(record)
        with writing:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            done[record["node_id"]] = record
            line = json.dumps({"node": record["node_id"][:12], "outcome": record["outcome"],
                               "seconds": record.get("seconds"), "coverage": record.get("coverage"),
                               "refusal": record.get("refusal")}, ensure_ascii=False)
            progress.write(line + "\n")
            try:
                print(line, flush=True)
            except (BrokenPipeError, OSError):
                pass                      # the channel home is a convenience, never a dependency

    def run(cards_to_run: list[dict], label: str, timeout: float, workers: int = 1) -> list[dict]:
        todo = [c for c in cards_to_run if c["node_id"] not in done]
        results = [done[c["node_id"]] for c in cards_to_run if c["node_id"] in done]
        if workers <= 1:
            for card in todo:
                if time.time() - started > a.stop_after:
                    print(json.dumps({"stopped": label, "after_seconds": round(time.time() - started)}))
                    break
                record = call(card, timeout)
                keep(record)
                results.append(record)
            return results
        # The core, in parallel, with WORK IN FLIGHT BOUNDED at twice the workers. The first run
        # submitted all 214 cards at once: when the consuming loop stopped, the pool went on
        # executing every queued call to completion with nobody reading the results — 49 calls made,
        # 43 results thrown away, and the traceback trapped behind shutdown(wait=True) until a
        # SIGTERM killed it. Bounded, a failure of that kind costs at most the calls in flight and
        # surfaces in seconds; --stop-after can also stop something, which it could not before.
        pending = list(todo)
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            in_flight = {}
            while pending or in_flight:
                while pending and len(in_flight) < workers * 2:
                    if time.time() - started > a.stop_after:
                        pending = []
                        break
                    card = pending.pop(0)
                    in_flight[pool.submit(call, card, timeout)] = card
                if not in_flight:
                    break
                for future in as_completed(list(in_flight)):
                    card = in_flight.pop(future)
                    try:
                        record = future.result()
                    except BaseException as exc:          # noqa: BLE001 — nothing may pass silently
                        # fail closed and COUNTED: the node, the exception's type and its message,
                        # written like any other refusal so the next failure of this kind is in the
                        # record instead of in a traceback nobody caught
                        failures += 1
                        record = {"node_id": card["node_id"], "piece": card.get("piece"), "ok": False,
                                  "refusal": "exception", "detail": f"{type(exc).__name__}: {exc}"[:200]}
                    keep(record)
                    results.append(record)
                    break                                  # re-fill the pool, then wait again
        # the count is RETURNED, never raised here: the first run stopped inside this function and
        # so never wrote its sidecar rows or its summary, although all 217 records were on disk. The
        # ceiling is judged after the record is complete, not instead of completing it.
        raised.append(failures)
        return results

    def release() -> None:
        """The model, and the two open files. In a finally, because a clean stop on SIGINT once
        left a model resident and the next job refused on exclusivity — the release belongs to the
        job's exit, not to its happy path."""
        for stream in (handle, progress):
            try:
                stream.close()
            except OSError:
                pass
        try:
            post(a.host, "/api/generate", {"model": a.model, "keep_alive": 0}, 30)
        except (urllib.error.URLError, TimeoutError, OSError):
            pass

    # Registered rather than written as one `finally`, because the run this fixes did not end by
    # returning: it was SIGTERMed, and a finally does not run for a signal. atexit covers the normal
    # exits and SystemExit; the handler turns SIGTERM into SystemExit so it covers the kill too —
    # which is how a first core run left its model resident, to be released by hand.
    import atexit
    import signal
    atexit.register(release)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    # ---- 1. the pilot, inside the job
    # the pilot is its declared windows, named by their ids, so "the pilot's three windows" is literally
    # that and not "the first three of a piece in whatever order a file lists"
    wanted = [w.strip() for w in a.pilot_nodes.split(",") if w.strip()]
    pilot_cards = [c for w in wanted for c in windows if c["node_id"].startswith(w)] if wanted else \
        [c for c in windows if piece_number(c.get("piece")) == str(a.pilot_piece)][:3]
    if len(pilot_cards) < 3:
        named = ", ".join(wanted) if wanted else f"piece {a.pilot_piece}"
        refuse(f"{named} gives {len(pilot_cards)} windows, three declared for the pilot "
               f"(the cards' `piece` is a file name such as '07.CCTP_Lot _Pompes.pdf', "
               f"and the number is derived from it)")
    # the pilot always runs at one worker, whatever --workers says: its per-call time is what sizes
    # the core's timeout, and a time measured under contention would size it from itself
    pilot_timeout = a.timeout or 300.0
    pilot = run(pilot_cards, "pilot", pilot_timeout, workers=1)
    passing = [r for r in pilot if r.get("ok") and (r.get("coverage_rate") or 0) >= 0.5]
    measured = [r["seconds"] for r in pilot if r.get("seconds")]
    per_call = max(measured) if measured else 30.0
    core_timeout = a.timeout or timeout_for(per_call, a.workers)
    gate = {"declared": "contract passes and coverage reaches half, on two of the three",
            "windows": [{"node": r["node_id"][:12], "outcome": r.get("outcome"),
                         "coverage": r.get("coverage"), "seconds": r.get("seconds"),
                         "refusal": r.get("refusal")} for r in pilot],
            "passing": len(passing), "passes": len(passing) >= 2,
            "measured_per_call_seconds": round(per_call, 1), "workers": a.workers,
            "core_timeout_seconds": core_timeout,
            "timeout_rule": "the slowest pilot call × workers + a margin of at least 60 s — the worst "
                            "case being that the parallel setting does not take and the workers queue"}
    (out / "pilot.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False))
    if not gate["passes"]:
        handle.close()
        sys.stderr.write("runner: the pilot gate did not pass; the core is not run\n")
        return 3
    if a.pilot_only:
        handle.close()
        return 0

    # ---- 2. the core
    rest = [c for c in windows if c["node_id"] not in {p["node_id"] for p in pilot_cards}]
    if a.only:
        wanted_ids = {w.strip() for w in a.only.read_text(encoding="utf-8").splitlines() if w.strip()}
        rest = [c for c in rest if c["node_id"] in wanted_ids]
        found = {c["node_id"] for c in rest} | {p["node_id"] for p in pilot_cards}
        missing = sorted(wanted_ids - found)
        if missing:
            refuse(f"--only names {len(missing)} node_id(s) that are no window of the card set: "
                   f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}")
    if a.limit:
        rest = rest[:a.limit]
    run(rest, "core", core_timeout, workers=a.workers)
    handle.close()

    # ---- 3. the sidecar: accepted windows only, pass 2, with the model that wrote them
    written = 0
    if a.sidecar:
        from .derived import DerivedStore, KnowledgeRow     # the sidecar store, needed only here
        store = DerivedStore(a.sidecar, source_root=str(Path(a.store).name),
                             source_sha256=store_sha, run_id=out.name)
        made_on = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for record in done.values():
            if not record.get("ok"):
                continue                             # a refusal is recorded in the run, never as knowledge
            store.write_knowledge(KnowledgeRow(
                subject_id=record["node_id"], subject_kind="chunk",
                k={"values": record.get("k_rows") or [], "coverage": record.get("coverage"),
                   "substitutions": record.get("substitutions")},
                summary=" ".join(record.get("summary") or []),
                summary_map=tuple(record.get("summary_map") or []),
                model=record.get("model") or a.model, prompt_version=FORM_VERSION,
                pass_=2, prompt_sha256=record.get("prompt_sha256"), made_on=record.get("made_on"),
                children=tuple(c for entry in record.get("summary_map") or []
                               for c in entry.get("children") or []),
                selector="register"), made_on)
            written += 1
        store.conn.commit()
        store.conn.close()

    # the model is released by release(), registered at the top for every exit including a signal
    ok = [r for r in done.values() if r.get("ok")]
    pieces = coverage_by_piece(list(done.values()))
    # rule 64 in the arithmetic: an unknown is not in the denominator. A busy server must not be able
    # to make the model look worse, and a model must not be able to hide behind a busy server either.
    judged = [r for r in done.values() if outcome_of(r) in (ACCEPTED, REFUSED)]
    unknown = [r for r in done.values() if outcome_of(r) == UNKNOWN]
    contract_rate = round(len(ok) / len(judged), 3) if judged else None
    exit_gate = {"declared": "the contract passes on at least 80 % of the windows JUDGED (accepted or "
                             "refused); a timeout or a transport error is unknown and stays out of the "
                             "denominator (rule 64); coverage is reported per piece and gated at half "
                             "only where a piece offers at least three lot-specific values",
                 "judged": len(judged), "unknown": len(unknown),
                 "unknown_nodes": [r["node_id"][:12] for r in unknown][:12],
                 "contract_rate": contract_rate, "passes": bool(contract_rate and contract_rate >= 0.8),
                 "pieces_gated": [p for p, b in pieces.items() if b["gated"]],
                 "pieces_below_half": [p for p, b in pieces.items() if b["passes"] is False]}
    summary = {"windows": len(done), "accepted": len(ok), "refused": len(done) - len(ok),
               "exit": exit_gate, "coverage_by_piece": pieces,
               "refusals": {reason: sum(1 for r in done.values() if r.get("refusal") == reason)
                            for reason in sorted({r.get("refusal") for r in done.values()
                                                  if r.get("refusal")})},
               "median_seconds": sorted(r["seconds"] for r in done.values() if r.get("seconds"))
                                 [len([r for r in done.values() if r.get("seconds")]) // 2]
                                 if any(r.get("seconds") for r in done.values()) else None,
               "sidecar_rows": written, "pilot": gate, "model": a.model, "form": FORM_VERSION,
               "exceptions": sum(raised), "max_exceptions": a.max_exceptions,
               "seconds": round(time.time() - started, 1)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
                                      + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    # judged LAST, on a complete record: the run says what it found and then says it went wrong,
    # rather than going wrong and leaving the finding unwritten
    if sum(raised) > a.max_exceptions:
        sys.stderr.write(f"runner: {sum(raised)} calls raised, more than the "
                         f"{a.max_exceptions} declared — the record above is complete\n")
        return 4
    return 0


JOBS = {"bakeoff": "run_bakeoff", "nodes": "run_nodes", "window": "run_window"}


def main(argv: list[str] | None = None) -> int:
    """`python -m ragix_kernels.harvest.runner {bakeoff,nodes,window} ...`"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in JOBS:
        sys.stderr.write(f"usage: runner {{{','.join(JOBS)}}} [options]\n")
        return 2
    return globals()[JOBS[argv[0]]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
