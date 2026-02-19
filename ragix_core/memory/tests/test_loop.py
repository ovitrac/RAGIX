"""
Tests for ragix_core.memory.loop — protocol parsing, LLM invocation, and loop logic.

Uses mock LLM subprocess (no real LLM calls) and mock dispatcher (no real DB).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-19
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from ragix_core.memory.loop import (
    LoopConfig,
    LoopResult,
    ProtocolResponse,
    emit_trace,
    invoke_llm,
    parse_protocol,
    parse_protocol_json,
    parse_protocol_regex,
    run_loop,
)


# ---------------------------------------------------------------------------
# parse_protocol_json
# ---------------------------------------------------------------------------


class TestParseProtocolJson:
    def test_valid_stop(self):
        raw = '{"need_more": false, "query": null, "rationale": null, "stop": true}\n\nThe answer is 42.'
        resp = parse_protocol_json(raw)
        assert resp.parse_ok is True
        assert resp.need_more is False
        assert resp.stop is True
        assert resp.answer == "The answer is 42."
        assert resp.query is None

    def test_valid_continue(self):
        raw = '{"need_more": true, "query": "token refresh", "rationale": "missing refresh path", "stop": false}\n\nPartial answer here.'
        resp = parse_protocol_json(raw)
        assert resp.parse_ok is True
        assert resp.need_more is True
        assert resp.query == "token refresh"
        assert resp.rationale == "missing refresh path"
        assert resp.stop is False
        assert resp.answer == "Partial answer here."

    def test_invalid_json_fallback(self):
        raw = "This is just a plain answer with no JSON."
        resp = parse_protocol_json(raw)
        assert resp.parse_ok is False
        assert resp.need_more is False
        assert resp.stop is True
        assert resp.answer == raw.strip()

    def test_empty_query_forces_stop(self):
        raw = '{"need_more": true, "query": "", "rationale": null, "stop": false}\n\nAnswer.'
        resp = parse_protocol_json(raw)
        assert resp.need_more is False
        assert resp.stop is True

    def test_whitespace_query_forces_stop(self):
        raw = '{"need_more": true, "query": "   ", "rationale": null, "stop": false}\n\nAnswer.'
        resp = parse_protocol_json(raw)
        assert resp.need_more is False
        assert resp.stop is True

    def test_no_answer_body(self):
        raw = '{"need_more": false, "query": null, "rationale": null, "stop": true}'
        resp = parse_protocol_json(raw)
        assert resp.answer == ""

    def test_blank_line_skipped(self):
        raw = '{"need_more": false, "stop": true}\n\nLine 1\nLine 2'
        resp = parse_protocol_json(raw)
        assert "Line 1" in resp.answer
        assert "Line 2" in resp.answer

    def test_stop_inferred_from_need_more(self):
        raw = '{"need_more": false}\n\nDone.'
        resp = parse_protocol_json(raw)
        assert resp.stop is True  # inferred


# ---------------------------------------------------------------------------
# parse_protocol_regex
# ---------------------------------------------------------------------------


class TestParseProtocolRegex:
    def test_need_more_true(self):
        raw = "NEED_MORE: true\nQUERY: security policies\nSome answer."
        resp = parse_protocol_regex(raw)
        assert resp.need_more is True
        assert resp.query == "security policies"

    def test_need_more_false(self):
        raw = "NEED_MORE: false\nFinal answer."
        resp = parse_protocol_regex(raw)
        assert resp.need_more is False
        assert resp.stop is True

    def test_no_patterns(self):
        raw = "Just a plain answer."
        resp = parse_protocol_regex(raw)
        assert resp.need_more is False


# ---------------------------------------------------------------------------
# parse_protocol (dispatcher)
# ---------------------------------------------------------------------------


class TestParseProtocol:
    def test_json_mode(self):
        raw = '{"need_more": false, "stop": true}\n\nAnswer.'
        resp = parse_protocol(raw, "json")
        assert resp.stop is True

    def test_regex_mode(self):
        raw = "NEED_MORE: true\nQUERY: test\nAnswer."
        resp = parse_protocol(raw, "regex")
        assert resp.need_more is True

    def test_passive_mode(self):
        raw = "Some answer text."
        resp = parse_protocol(raw, "passive")
        assert resp.need_more is False
        assert resp.stop is True
        assert resp.answer == raw.strip()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            parse_protocol("text", "invalid_mode")


# ---------------------------------------------------------------------------
# emit_trace
# ---------------------------------------------------------------------------


class TestEmitTrace:
    def test_writes_jsonl(self):
        buf = io.StringIO()
        emit_trace({"iter": 1, "action": "continue"}, buf)
        line = buf.getvalue()
        assert line.endswith("\n")
        parsed = json.loads(line)
        assert parsed["iter"] == 1
        assert parsed["action"] == "continue"

    def test_non_ascii(self):
        buf = io.StringIO()
        emit_trace({"query": "requête française"}, buf)
        assert "requête" in buf.getvalue()


# ---------------------------------------------------------------------------
# invoke_llm
# ---------------------------------------------------------------------------


class TestInvokeLlm:
    @patch("ragix_core.memory.loop.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"need_more": false, "stop": true}\n\nAnswer text.',
            stderr="",
        )
        output = invoke_llm("echo test", "input context")
        assert "Answer text" in output

    @patch("ragix_core.memory.loop.subprocess.run")
    def test_nonzero_exit_raises(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="some error"
        )
        with pytest.raises(RuntimeError, match="failed"):
            invoke_llm("bad_cmd", "input")

    @patch("ragix_core.memory.loop.subprocess.run")
    def test_empty_output_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with pytest.raises(RuntimeError, match="empty output"):
            invoke_llm("cmd", "input")

    @patch("ragix_core.memory.loop.subprocess.run")
    def test_timeout_raises(self, mock_run):
        import subprocess as sp

        mock_run.side_effect = sp.TimeoutExpired("cmd", 300)
        with pytest.raises(RuntimeError, match="timed out"):
            invoke_llm("cmd", "input")


# ---------------------------------------------------------------------------
# run_loop — integration tests with mocked LLM
# ---------------------------------------------------------------------------


def _make_llm_response(need_more: bool, query: str = None, answer: str = ""):
    """Build a raw LLM output string following the JSON protocol."""
    header = {"need_more": need_more, "query": query, "stop": not need_more}
    return json.dumps(header) + "\n\n" + answer


class TestRunLoop:
    """Test the core loop with mocked LLM subprocess calls."""

    def _config(self, **overrides) -> LoopConfig:
        defaults = dict(
            llm_command="mock_llm",
            max_calls=3,
            threshold=0.92,
            query_threshold=0.90,
            stable_steps=2,
            similarity_mode="lexical",
            trace=False,
        )
        defaults.update(overrides)
        return LoopConfig(**defaults)

    # --- Single-shot: LLM stops immediately ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_single_shot_stop(self, mock_llm):
        mock_llm.return_value = _make_llm_response(
            need_more=False, answer="The auth flow uses JWT tokens."
        )
        result = run_loop("initial context", self._config())
        assert result.iterations == 1
        assert "JWT tokens" in result.answer
        assert result.exit_reason == "llm signaled stop"

    # --- Fixed-point convergence ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_fixed_point_convergence(self, mock_llm):
        # Iteration 1: need_more, iteration 2+3: same answer → stable
        mock_llm.side_effect = [
            _make_llm_response(True, "more details", "Answer v1 about auth."),
            _make_llm_response(True, "even more", "Answer v1 about auth."),
            _make_llm_response(True, "still more", "Answer v1 about auth."),
        ]
        config = self._config(stable_steps=2, stop_on_no_new=False)
        # Mock dispatcher that returns empty results (so we test sim only)
        dispatcher = MagicMock()
        dispatcher.dispatch.return_value = {"status": "ok", "items": []}

        result = run_loop("context", config, dispatcher=dispatcher)
        assert result.converged is True
        assert "stable" in result.exit_reason
        assert result.iterations == 3  # 1:continue, 2:stable(1), 3:stable(2)→stop

    # --- Max-calls reached ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_max_calls_reached(self, mock_llm):
        mock_llm.side_effect = [
            _make_llm_response(True, "q1", "Answer iteration 1"),
            _make_llm_response(True, "q2", "Answer iteration 2 — different"),
            _make_llm_response(True, "q3", "Answer iteration 3 — also different"),
        ]
        dispatcher = MagicMock()
        dispatcher.dispatch.return_value = {"status": "ok", "items": []}

        config = self._config(max_calls=3, stop_on_no_new=False)
        result = run_loop("context", config, dispatcher=dispatcher)
        assert result.iterations == 3
        assert "max_calls" in result.exit_reason
        assert result.converged is False

    # --- Query-cycle detection ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_query_cycle_exact(self, mock_llm):
        mock_llm.side_effect = [
            _make_llm_response(True, "auth flow", "Answer 1"),
            _make_llm_response(True, "auth flow", "Answer 2 — different"),
        ]
        dispatcher = MagicMock()
        dispatcher.dispatch.return_value = {"status": "ok", "items": []}

        config = self._config(stop_on_no_new=False)
        result = run_loop("context", config, dispatcher=dispatcher)
        assert "query cycle" in result.exit_reason.lower() or "cycle" in result.exit_reason.lower()

    # --- No new items stops loop ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_no_new_items_stops(self, mock_llm):
        mock_llm.side_effect = [
            _make_llm_response(True, "auth flow", "Partial answer"),
        ]
        dispatcher = MagicMock()
        dispatcher.dispatch.return_value = {"status": "ok", "items": []}

        config = self._config(stop_on_no_new=True)
        result = run_loop("context", config, dispatcher=dispatcher)
        assert result.iterations == 1
        assert "no new items" in result.exit_reason

    # --- LLM error handled gracefully ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_llm_error(self, mock_llm):
        mock_llm.side_effect = RuntimeError("LLM crashed")

        result = run_loop("context", self._config())
        assert "llm_error" in result.exit_reason
        assert result.iterations == 1

    # --- Trace emission ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_trace_emitted(self, mock_llm):
        mock_llm.return_value = _make_llm_response(
            need_more=False, answer="Final answer."
        )
        trace_buf = io.StringIO()
        config = self._config(trace=True)

        result = run_loop("context", config, trace_stream=trace_buf)
        trace_buf.seek(0)
        lines = trace_buf.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["iter"] == 1
        assert entry["action"] == "llm_stop"

    # --- Passive protocol: no JSON parsing, fixed-point only ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_passive_protocol(self, mock_llm):
        mock_llm.side_effect = [
            "First answer about authentication.",
            "First answer about authentication.",
            "First answer about authentication.",
        ]
        config = self._config(
            protocol="passive",
            stable_steps=2,
            max_calls=5,
        )
        result = run_loop("context", config)
        assert result.converged is True
        assert result.iterations == 3

    # --- No dispatcher: LLM gets same context every time ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_no_dispatcher(self, mock_llm):
        mock_llm.side_effect = [
            _make_llm_response(True, "more", "Partial."),
            _make_llm_response(False, None, "Complete answer."),
        ]
        # No dispatcher → no recall → same context repeated
        result = run_loop("static context", self._config(max_calls=3))
        assert result.iterations == 2
        assert result.answer == "Complete answer."

    # --- Invalid JSON fallback ---

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_invalid_json_treated_as_stop(self, mock_llm):
        mock_llm.return_value = "Not JSON at all. Just a plain answer."

        result = run_loop("context", self._config())
        assert result.iterations == 1
        assert result.answer == "Not JSON at all. Just a plain answer."
        assert "llm signaled stop" in result.exit_reason


# ---------------------------------------------------------------------------
# LoopConfig defaults
# ---------------------------------------------------------------------------


class TestLoopConfig:
    def test_defaults(self):
        c = LoopConfig()
        assert c.max_calls == 3
        assert c.threshold == 0.92
        assert c.stable_steps == 2
        assert c.similarity_mode == "auto"
        assert c.protocol == "json"

    def test_override(self):
        c = LoopConfig(max_calls=5, threshold=0.85)
        assert c.max_calls == 5
        assert c.threshold == 0.85
