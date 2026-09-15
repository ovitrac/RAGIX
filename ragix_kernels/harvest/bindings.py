"""Strict candidate-id binding, exhaustive accounting and append-only replay records.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15

Semantic vocabularies and authority/applicability policy are supplied by the caller.
This library does not execute a model, declare source authority or decide compliance.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import re

from .quantitative import harvest

VERSION = "candidate-binding/1.0"


class BindingRefusal(ValueError):
    def __init__(self, rule, detail=""):
        self.rule = rule
        super().__init__(rule + (": " + detail if detail else ""))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BindingRefusal("duplicate_key", key)
        result[key] = value
    return result


@dataclass(frozen=True)
class BindingReport:
    output_sha256: str
    candidate_set_sha256: str
    policy_id: str
    claims: tuple[dict, ...]
    accounting: dict[str, tuple[str, ...]]
    abstentions: dict[str, str]
    review_reasons: dict[str, tuple[str, ...]]
    producer: str = VERSION
    outcome: str = "accepted"


@dataclass(frozen=True)
class BindingFailure:
    output_sha256: str
    candidate_set_sha256: str
    policy_id: str
    rule: str
    detail: str
    producer: str = VERSION
    outcome: str = "refused"


@dataclass(frozen=True)
class UnparsedObservation:
    source_id: str
    node_id: str
    start: int
    end: int
    raw: str
    suspected_kind: str
    reason: str


def validate_unparsed(observation, *, source_id: str, node_id: str, text: str):
    """A missed-expression report is exact source text, never a normalized quantity."""
    required = {"node_id", "start", "end", "exact_raw_span", "suspected_kind", "reason"}
    if not isinstance(observation, dict) or set(observation) != required:
        raise BindingRefusal("unparsed_schema")
    start, end = observation["start"], observation["end"]
    if (not source_id or observation["node_id"] != node_id or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(text) or text[start:end] != observation["exact_raw_span"]):
        raise BindingRefusal("unparsed_source_mismatch")
    if any(not isinstance(observation[k], str) or not observation[k].strip() for k in ("suspected_kind", "reason")):
        raise BindingRefusal("unparsed_schema")
    return UnparsedObservation(source_id, node_id, start, end, text[start:end],
                               observation["suspected_kind"], observation["reason"])


def validate_bindings(raw: str, candidates, *, node_texts: dict[str, str], policy_id: str,
                      semantic_validator) -> BindingReport:
    """Fail the entire packet on invalid output; never repair a model response.

    semantic_validator(claim, referenced_candidates) returns rejection rule strings.
    It is mandatory: a generic envelope check cannot establish domain semantics.
    A composite covers its immutable members for accounting, but only the parser
    can define that composite. Multiple claims may cover one candidate; abstention
    and accepted binding cannot both cover it.
    """
    if not policy_id or not callable(semantic_validator):
        raise ValueError("a versioned semantic policy is required")
    def invalid_constant(_):
        raise BindingRefusal("invalid_json_constant")
    try:
        packet = json.loads(raw, object_pairs_hook=_object, parse_constant=invalid_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise BindingRefusal("invalid_json") from exc
    if not isinstance(packet, dict) or set(packet) != {"claims", "abstentions"}:
        raise BindingRefusal("packet_schema")
    if not isinstance(packet["claims"], list) or not isinstance(packet["abstentions"], list):
        raise BindingRefusal("packet_schema")
    candidates = tuple(candidates)
    by_id = {c.candidate_id: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate supplied candidate")
    for c in candidates:
        text = node_texts.get(c.node_id)
        if text is None or not 0 <= c.start < c.end <= len(text) or text[c.start:c.end] != c.raw:
            raise BindingRefusal("candidate_source_mismatch")
        if any(m.candidate_id not in by_id for m in c.members):
            raise BindingRefusal("missing_composite_member")
    covered, abstained, reasons, claim_ids, accepted = defaultdict(list), {}, {}, set(), []
    required = {"claim_id", "node_id", "candidate_ids", "parameter", "constraint", "statement_role",
                "comparison_role", "source_authority", "quoted_source_authority", "applicability"}
    optional = {"modality", "relevance"}
    def descendants(cid, active=frozenset()):
        if cid in active:
            raise BindingRefusal("composite_cycle")
        result = {cid}
        for member in by_id[cid].members:
            result.update(descendants(member.candidate_id, active | {cid}))
        return result
    for claim in packet["claims"]:
        if not isinstance(claim, dict) or not required <= set(claim) or set(claim) - required - optional:
            raise BindingRefusal("claim_schema")
        ident, node, ids = claim["claim_id"], claim["node_id"], claim["candidate_ids"]
        if not isinstance(ident, str) or not ident or ident in claim_ids or not isinstance(node, str):
            raise BindingRefusal("claim_identity")
        claim_ids.add(ident)
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in by_id for i in ids):
            raise BindingRefusal("unknown_candidate")
        if (len(ids) != len(set(ids)) or any(by_id[i].node_id != node for i in ids)
                or len({by_id[i].source_id for i in ids}) != 1):
            raise BindingRefusal("candidate_scope")
        parameter = claim["parameter"]
        if not isinstance(parameter, dict) or set(parameter) != {"raw_name", "canonical_name", "canonical_name_status"}:
            raise BindingRefusal("parameter_schema")
        if (not isinstance(parameter["raw_name"], str) or not parameter["raw_name"].strip()
                or not isinstance(parameter["canonical_name"], (str, type(None)))
                or parameter["canonical_name_status"] != "proposed"):
            raise BindingRefusal("parameter_authority")
        for name in (parameter["raw_name"], parameter["canonical_name"] or ""):
            if (re.fullmatch(r"[\d\s+.,%<>=±\-\u2212]+", name)
                    or any(c.quantitative for c in harvest(name, source_id="parameter", node_id="parameter", classification="CONTENT"))):
                raise BindingRefusal("parameter_value_smuggling")
        constraint = claim["constraint"]
        if not isinstance(constraint, dict) or set(constraint) - {"kind", "composite_candidate_id"} or "kind" not in constraint:
            raise BindingRefusal("constraint_schema")
        kind = constraint["kind"]
        allowed = {"scalar", "range", "interval", "tolerance", "inequality", "duration", "cardinality", "symbolic", "rate", "other", "unknown"}
        if not isinstance(kind, str) or kind not in allowed:
            raise BindingRefusal("constraint_kind")
        if kind in {"range", "interval", "tolerance"}:
            composite = constraint.get("composite_candidate_id")
            expected = "interval" if kind == "range" else kind
            if composite not in ids or by_id[composite].kind != expected or not by_id[composite].members:
                raise BindingRefusal("model_assembled_composite")
        elif "composite_candidate_id" in constraint:
            raise BindingRefusal("unexpected_composite")
        failures = tuple(semantic_validator(claim, tuple(by_id[i] for i in ids)))
        if failures:
            raise BindingRefusal("semantic_policy", ",".join(failures))
        accounted = set().union(*(descendants(i) for i in ids))
        if (any(by_id[i].node_id != node for i in accounted)
                or len({by_id[i].source_id for i in accounted}) != 1):
            raise BindingRefusal("composite_scope")
        review = {flag for i in accounted for flag in by_id[i].flags}
        if any(by_id[i].normalization_status != "parsed" for i in accounted):
            review.add("NORMALIZATION_UNRESOLVED")
        if any(claim[key] in ("unknown", "UNKNOWN") for key in ("statement_role", "comparison_role", "source_authority", "quoted_source_authority")):
            review.add("SEMANTICS_UNKNOWN")
        reasons[ident] = tuple(sorted(review))
        for cid in accounted:
            covered[cid].append(ident)
        accepted.append(claim)
    for abstention in packet["abstentions"]:
        if not isinstance(abstention, dict) or set(abstention) != {"candidate_id", "reason"}:
            raise BindingRefusal("abstention_schema")
        cid, reason = abstention["candidate_id"], abstention["reason"]
        if (not isinstance(cid, str) or cid not in by_id or cid in abstained
                or not isinstance(reason, str) or not reason.strip()):
            raise BindingRefusal("abstention_identity")
        abstained[cid] = reason
    if set(covered) & set(abstained):
        raise BindingRefusal("binding_and_abstention")
    if set(by_id) != set(covered) | set(abstained):
        raise BindingRefusal("candidate_omitted")
    digest = hashlib.sha256(canonical(sorted((asdict(c) for c in candidates), key=lambda c: c["candidate_id"])).encode()).hexdigest()
    return BindingReport(hashlib.sha256(raw.encode()).hexdigest(), digest, policy_id, tuple(accepted),
                         {k: tuple(sorted(v)) for k, v in covered.items()}, abstained, reasons)


def record_batch(store, batch_id: str, report: BindingReport | BindingFailure, *, raw_output: str, created_at: str):
    """Append in DerivedStore's producer namespace, leaving commit to the caller.

    Retain exact output alongside the validation record. Identical replay is a no-op;
    changed output, candidates, policy or source under the same batch id is refused.
    """
    from .derived import KnowledgeRow

    if not batch_id:
        raise ValueError("batch_id required")
    if hashlib.sha256(raw_output.encode()).hexdigest() != report.output_sha256:
        raise BindingRefusal("raw_output_mismatch")
    subject = VERSION + ":" + batch_id
    payload = {"report": asdict(report), "raw_output": raw_output}
    previous = store.conn.execute(
        "SELECT knowledge_id, k_json, source_sha256 FROM knowledge WHERE subject_id=? AND harvest_schema=?",
        (subject, VERSION)).fetchall()
    if previous:
        if (len(previous) != 1 or canonical(json.loads(previous[0][1])) != canonical(payload)
                or previous[0][2] != store.source_sha256):
            raise BindingRefusal("batch_identity_conflict")
        return previous[0][0]
    return store.write_knowledge(KnowledgeRow(subject, "node", payload, harvest_schema=VERSION,
                                             prompt_version=report.policy_id), created_at)


def validate_and_record(store, batch_id: str, raw: str, candidates, *, node_texts,
                        policy_id: str, semantic_validator, created_at: str):
    """Persist either outcome, including exact refused output. Caller commits.

    Unlike validate_bindings, schema refusals return BindingFailure so the caller
    can commit their audit record without accidentally rolling back on an exception.
    Programming errors and storage conflicts still propagate.
    """
    candidates = tuple(candidates)
    try:
        report = validate_bindings(raw, candidates, node_texts=node_texts, policy_id=policy_id,
                                   semantic_validator=semantic_validator)
    except BindingRefusal as exc:
        digest = hashlib.sha256(canonical(sorted((asdict(c) for c in candidates), key=lambda c: c["candidate_id"])).encode()).hexdigest()
        report = BindingFailure(hashlib.sha256(raw.encode()).hexdigest(), digest, policy_id, exc.rule, str(exc))
    record_batch(store, batch_id, report, raw_output=raw, created_at=created_at)
    return report
