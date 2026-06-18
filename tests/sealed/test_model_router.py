"""
Tests for the Sprint 2bis model router/cascade (WP §8quater).

Policy-first routing, refusal cascade, unsafe-output blocking, post-check/leak gate, and
sanitized audit. Models are injected as fakes — no live model server.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from ragix_sealed.contracts import load_contracts
from ragix_sealed.model_router import (
    ModelResponse,
    ModelStatus,
    ModelRouter,
    RouterDecision,
)

PASS = lambda _content: True   # noqa: E731 (test helper)


def _model(status, content=None):
    return lambda task, text: ModelResponse(status=status, content=content)


def test_globally_forbidden_task_blocked():
    c = load_contracts()
    r = ModelRouter(c, models={}, post_check=PASS)
    res = r.run("reidentify_for_llm", "PLACEHOLDERIZED", "x")
    assert res.decision is RouterDecision.BLOCKED


def test_no_eligible_model():
    c = load_contracts()
    r = ModelRouter(c, models={}, post_check=PASS)
    res = r.run("definitely_not_a_task", "PLACEHOLDERIZED", "x")
    assert res.decision is RouterDecision.NO_ELIGIBLE_MODEL


def test_primary_answers_and_releases():
    c = load_contracts()
    models = {"primary": _model(ModelStatus.ANSWERED, "[PERSON_001] did X")}
    r = ModelRouter(c, models=models, post_check=PASS)
    res = r.run("timeline", "PLACEHOLDERIZED", "payload")
    assert res.decision is RouterDecision.RELEASED
    assert res.model_id == "primary"
    assert res.content == "[PERSON_001] did X"


def test_refusal_cascades_to_fallback():
    c = load_contracts()
    # 'ocr_cleanup' is allowed by fallback (SEMI_RAW_INTERNAL); primary doesn't allow it,
    # so only fallback is eligible — and it answers.
    models = {
        "primary": _model(ModelStatus.REFUSED),
        "fallback": _model(ModelStatus.ANSWERED, "cleaned [REFERENCE_001]"),
    }
    r = ModelRouter(c, models=models, post_check=PASS)
    res = r.run("ocr_cleanup", "SEMI_RAW_INTERNAL", "payload")
    assert res.decision is RouterDecision.RELEASED
    assert res.model_id == "fallback"


def test_post_check_failure_blocks():
    c = load_contracts()
    models = {"primary": _model(ModelStatus.ANSWERED, "leaks raw@example.com")}
    deny = lambda content: "@" not in content  # crude leak gate  # noqa: E731
    r = ModelRouter(c, models=models, post_check=deny)
    res = r.run("timeline", "PLACEHOLDERIZED", "payload")
    assert res.decision is RouterDecision.BLOCKED
    assert "post-check" in (res.reason or "")


def test_unsafe_output_blocks_no_fallback():
    c = load_contracts()
    models = {
        "primary": _model(ModelStatus.UNSAFE_OUTPUT),
        "fallback": _model(ModelStatus.ANSWERED, "should not be reached"),
    }
    r = ModelRouter(c, models=models, post_check=PASS)
    res = r.run("timeline", "PLACEHOLDERIZED", "payload")
    assert res.decision is RouterDecision.BLOCKED
    assert res.model_id == "primary"


def test_all_refuse_routes_to_human_review():
    c = load_contracts()
    models = {"primary": _model(ModelStatus.REFUSED), "fallback": _model(ModelStatus.ABSTAINED)}
    r = ModelRouter(c, models=models, post_check=PASS)
    res = r.run("timeline", "PLACEHOLDERIZED", "payload")
    # only 'primary' allows 'timeline' for PLACEHOLDERIZED; it refuses -> human review
    assert res.decision is RouterDecision.HUMAN_REVIEW


def test_audit_carries_no_raw_content():
    c = load_contracts()
    secret = "raw@example.com"
    models = {"primary": _model(ModelStatus.ANSWERED, secret)}
    r = ModelRouter(c, models=models, post_check=PASS)
    res = r.run("timeline", "PLACEHOLDERIZED", "payload-with-" + secret)
    import json
    assert secret not in json.dumps(res.audit)


def test_default_post_check_is_deny():
    """With no leak scanner wired, an ANSWERED output is NOT released by default."""
    c = load_contracts()
    models = {"primary": _model(ModelStatus.ANSWERED, "anything")}
    r = ModelRouter(c, models=models)  # no post_check
    res = r.run("timeline", "PLACEHOLDERIZED", "payload")
    assert res.decision is RouterDecision.BLOCKED
