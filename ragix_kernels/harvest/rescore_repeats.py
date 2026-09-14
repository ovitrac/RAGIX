#!/usr/bin/env python3
"""demoE2E 25 — re-score a run's refusals offline, on the answers it stored. No model call.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

PREPARED 2026-09-12, NOT APPLIED: the repeated-phrase rule is being read by the testing seat as the
non-author, and the rule it proposes decides what this scores. This file computes, it does not rule.

WHY IT EXISTS. Row 25's re-run refused 82 of 217 windows on `repeated phrase`, a rule this seat wrote
from the testing seat's observation (« 7 jours sur 7 » twice in one sentence) and implemented as "any
three-word span repeated inside a sentence". In French contract prose that fires on function-word
triples: the most common catch is « le titulaire doit », 24 times, which is a subject repeated across
clauses and not padding. The rule is mis-specified; what the defect is, exactly, is the question.

THE CANDIDATE RULE, as the coordinating seat framed it for form 0.8:
  B  the SAME CLAIM ID cited twice in one sentence — exact, not heuristic, and it is what the
     observed defect is: the same value stated twice, which after substitution is the same marker.
  C  a five-word verbatim span repeated in one sentence — rarer, and catches padding that carries no
     value at all.
Both are computed here, separately and together, over the stored answers. A record whose answer is
absent or unparseable is COUNTED AS SUCH and never guessed: the first run capped `raw_response` at
2 000 characters, so 19 of the 82 cannot be re-scored at all — which is why the cap is now gone.

    python3 demoE2E/25_form_8b/rescore_repeats.py --records ... --store ... --out ...
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from .fr.grammars import read_values                          # noqa: E402
from .form import PLACEHOLDER, HarvestRefusal, substitute  # noqa: E402

CLAIM = re.compile(r"\{\{claim:([A-Za-z0-9_.:-]{1,64})\}\}")


def words_of(sentence: str) -> list[str]:
    return [w for w in re.split(r"[^\w'’-]+", PLACEHOLDER.sub(" ", sentence).casefold()) if w]


def repeats_a_value(sentence: str) -> str | None:
    """Rule B: the same value cited twice in one sentence."""
    cited = CLAIM.findall(sentence)
    doubled = [v for v, n in Counter(cited).items() if n > 1]
    return doubled[0] if doubled else None


def repeats_a_span(sentence: str, length: int = 5) -> str | None:
    """Rule C: a verbatim span of `length` words repeated in one sentence."""
    words = words_of(sentence)
    grams = Counter(tuple(words[i:i + length]) for i in range(max(0, len(words) - length + 1)))
    for gram, n in grams.items():
        if n > 1:
            return " ".join(gram)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="row 25 — re-score refusals on their stored answers")
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--refusal", default="repeated phrase")
    ap.add_argument("--span", type=int, default=5)
    ap.add_argument("--full-judge", action="store_true",
                    help="re-judge each refused record END TO END under the form as it stands now, "
                         "not only the one check — which is what a corrected contract rate needs")
    a = ap.parse_args(argv)

    db = sqlite3.connect(f"file:{a.store.resolve()}?mode=ro&immutable=1", uri=True)
    records = [json.loads(l) for l in a.records.read_text(encoding="utf-8").splitlines() if l.strip()]
    chosen = [r for r in records if r.get("refusal") == a.refusal]

    if a.full_judge:
        # Every REFUSED record re-judged end to end under the form as it stands. Post hoc on the
        # answers the run stored; no model is called, and a record whose answer is missing or
        # truncated is counted as unreadable rather than guessed either way.
        from . import runner as job                   # pass 2's judge, offered values and form version
        accepted_before = [r for r in records if r.get("ok")]
        verdicts: Counter = Counter()
        for record in [r for r in records if not r.get("ok")]:
            answer = record.get("raw_response")
            if not answer:
                verdicts["unreadable: no answer stored"] += 1
                continue
            try:
                body = json.loads(answer)
            except json.JSONDecodeError:
                verdicts["unreadable: answer truncated or not JSON"] += 1
                continue
            text_row = db.execute("select text from chunks where chunk_id=?",
                                  (record["node_id"],)).fetchone()
            if text_row is None:
                verdicts["window absent from the store"] += 1
                continue
            values = job.offered_values(text_row[0])
            try:
                job.judge(body, text_row[0], values, record["node_id"])
                verdicts["would now be ACCEPTED"] += 1
            except HarvestRefusal as exc:
                verdicts[f"still refused: {exc.reason}"] += 1
        judged = sum(n for k, n in verdicts.items() if not k.startswith("unreadable"))
        now_accepted = verdicts["would now be ACCEPTED"]
        report = {"declared": {"records": str(a.records), "mode": "full re-judge, post hoc on the "
                               "answers the run stored; no model called",
                               "form": job.FORM_VERSION},
                  "accepted_as_run": len(accepted_before), "refused_as_run": len(records) - len(accepted_before),
                  "verdicts": dict(verdicts),
                  "corrected": {"accepted": len(accepted_before) + now_accepted,
                                "judged": len(accepted_before) + judged,
                                "rate": round((len(accepted_before) + now_accepted) /
                                              (len(accepted_before) + judged), 3) if judged else None,
                                "unreadable_excluded": sum(n for k, n in verdicts.items()
                                                           if k.startswith("unreadable"))}}
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return 0

    counts: Counter = Counter()
    per_record = []
    for record in chosen:
        row = {"node_id": record["node_id"], "piece": record.get("piece"),
               "was": record.get("detail")}
        answer = record.get("raw_response")
        if not answer:
            counts["no answer stored"] += 1
            per_record.append({**row, "verdict": "no answer stored"})
            continue
        try:
            body = json.loads(answer)
        except json.JSONDecodeError:
            # the first run capped the answer at 2 000 characters; a truncated answer is not a
            # malformed one, and must not be counted as either
            counts["answer truncated or not JSON"] += 1
            per_record.append({**row, "verdict": "answer truncated or not JSON"})
            continue
        text_row = db.execute("select text from chunks where chunk_id=?",
                              (record["node_id"],)).fetchone()
        if text_row is None:
            counts["window absent from the store"] += 1
            continue
        values = [{"value_id": f"v{i + 1}", "kind": v.kind, "raw": v.raw, "normalized": v.normalized}
                  for i, v in enumerate(read_values(text_row[0]))]
        sentences = body.get("summary")
        if isinstance(sentences, str):
            sentences = [sentences]
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            counts["would refuse earlier: summary is not prose"] += 1
            per_record.append({**row, "verdict": "would refuse earlier: summary is not prose"})
            continue
        try:
            placed, _ = substitute(sentences, values)
        except HarvestRefusal as exc:
            counts[f"would refuse earlier: {exc.reason}"] += 1
            per_record.append({**row, "verdict": f"would refuse earlier: {exc.reason}"})
            continue
        value_twice = next((v for s in placed if (v := repeats_a_value(s))), None)
        span_twice = next((s2 for s in placed if (s2 := repeats_a_span(s, a.span))), None)
        verdict = ("B: the same value twice" if value_twice else
                   f"C: a {a.span}-word span twice" if span_twice else
                   "would pass this check")
        counts[verdict] += 1
        per_record.append({**row, "verdict": verdict, "value": value_twice, "span": span_twice})

    report = {"declared": {"records": str(a.records), "refusal": a.refusal, "span_words": a.span,
                           "rule_B": "the same claim id cited twice in one sentence",
                           "rule_C": f"a {a.span}-word verbatim span repeated in one sentence",
                           "note": "post hoc on the answers the run stored; no model was called, and "
                                   "a record whose answer is missing or truncated is counted as such"},
              "refusals_examined": len(chosen), "counts": dict(counts), "records": per_record}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"refusals_examined": len(chosen), "counts": dict(counts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
