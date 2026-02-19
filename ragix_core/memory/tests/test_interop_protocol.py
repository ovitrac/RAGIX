"""
Interop contract tests — shared JSON protocol between RAGIX and memctl.

Verifies that RAGIX's loop controller can correctly parse and produce
outputs compatible with memctl's loop convention (shared by design, not
by import). Tests use hardcoded protocol examples matching the contract
defined in both ROADMAP_LOOP.md files.

This file does NOT import from memctl. It tests RAGIX against the
shared protocol spec.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-19
"""

import io
import json
from unittest.mock import patch

import pytest

from ragix_core.memory.loop import (
    LOOP_PROTOCOL_PROMPT,
    LoopConfig,
    emit_trace,
    parse_protocol,
    parse_protocol_json,
    run_loop,
)
from ragix_core.memory.mcp.formatting import FORMAT_VERSION, format_injection_block
from ragix_core.memory.similarity import (
    lexical_similarity,
    normalize_text,
    token_jaccard,
)


# ---------------------------------------------------------------------------
# 1. Protocol prompt identity
# ---------------------------------------------------------------------------


class TestProtocolPromptContract:
    """The protocol prompt must match the shared spec exactly."""

    def test_prompt_contains_json_instruction(self):
        assert '"need_more"' in LOOP_PROTOCOL_PROMPT
        assert '"query"' in LOOP_PROTOCOL_PROMPT
        assert '"rationale"' in LOOP_PROTOCOL_PROMPT
        assert '"stop"' in LOOP_PROTOCOL_PROMPT

    def test_prompt_requires_first_line_json(self):
        assert "FIRST line" in LOOP_PROTOCOL_PROMPT
        assert "JSON object" in LOOP_PROTOCOL_PROMPT

    def test_prompt_forbids_markdown_wrapping(self):
        assert "Do NOT wrap it in markdown" in LOOP_PROTOCOL_PROMPT


# ---------------------------------------------------------------------------
# 2. JSON protocol parsing — memctl-compatible outputs
# ---------------------------------------------------------------------------


class TestJsonProtocolCompat:
    """RAGIX must parse all memctl-style LLM outputs correctly."""

    def test_memctl_stop_example(self):
        """Example from memctl ROADMAP §6: sufficient context."""
        raw = (
            '{"need_more": false, "query": null, "rationale": null, "stop": true}\n'
            "\n"
            "The authentication flow uses JWT tokens with a 15-minute expiry.\n"
            "Refresh tokens are stored server-side in Redis."
        )
        resp = parse_protocol_json(raw)
        assert resp.parse_ok is True
        assert resp.need_more is False
        assert resp.stop is True
        assert "JWT tokens" in resp.answer

    def test_memctl_continue_example(self):
        """Example from memctl ROADMAP §6: insufficient context."""
        raw = (
            '{"need_more": true, "query": "token refresh flow + error codes", '
            '"rationale": "missing refresh path details", "stop": false}\n'
            "\n"
            "The authentication flow uses JWT tokens with a 15-minute expiry..."
        )
        resp = parse_protocol_json(raw)
        assert resp.parse_ok is True
        assert resp.need_more is True
        assert resp.query == "token refresh flow + error codes"
        assert resp.rationale == "missing refresh path details"
        assert resp.stop is False

    def test_empty_query_coercion(self):
        """Both systems: empty query with need_more=true → stop."""
        raw = '{"need_more": true, "query": "", "stop": false}\n\nAnswer.'
        resp = parse_protocol_json(raw)
        assert resp.need_more is False
        assert resp.stop is True

    def test_missing_stop_field(self):
        """memctl infers stop from need_more if stop is absent."""
        raw = '{"need_more": false}\n\nDone.'
        resp = parse_protocol_json(raw)
        # stop inferred as !need_more
        assert resp.stop is True

    def test_unknown_fields_ignored(self):
        """Forward compatibility: unknown fields must not break parsing."""
        raw = (
            '{"need_more": false, "stop": true, "confidence": 0.95, '
            '"extra_field": "ignored"}\n\nAnswer.'
        )
        resp = parse_protocol_json(raw)
        assert resp.parse_ok is True
        assert resp.stop is True

    def test_all_three_protocol_modes(self):
        """json, regex, passive — all must be parseable without error."""
        raw_json = '{"need_more": false, "stop": true}\n\nAnswer.'
        raw_regex = "NEED_MORE: false\nFinal answer."
        raw_passive = "Plain text answer with no protocol."

        for raw, mode in [
            (raw_json, "json"),
            (raw_regex, "regex"),
            (raw_passive, "passive"),
        ]:
            resp = parse_protocol(raw, mode)
            assert resp.answer  # all produce a non-empty answer


# ---------------------------------------------------------------------------
# 3. Injection block format — FORMAT_VERSION=1 contract
# ---------------------------------------------------------------------------


class TestInjectionBlockCompat:
    """The injection block format must be parseable by both systems."""

    def _make_items(self):
        return [
            {
                "id": "item-001",
                "tier": "stm",
                "validation": "unverified",
                "type": "note",
                "title": "SQLite for storage",
                "content": "RAGIX uses SQLite FTS5 for full-text search.",
                "provenance": {"source_kind": "doc", "source_id": "arch.md"},
                "tags": ["architecture", "sqlite"],
                "confidence": 0.85,
            },
            {
                "id": "item-002",
                "tier": "stm",
                "validation": "verified",
                "type": "decision",
                "title": "Local-first design",
                "content": "All processing must be local. No cloud dependencies.",
                "provenance": {"source_kind": "chat", "source_id": "turn_3"},
                "tags": ["sovereignty", "constraint"],
                "confidence": 0.92,
            },
        ]

    def test_format_version_is_1(self):
        assert FORMAT_VERSION == 1

    def test_block_has_required_headers(self):
        block = format_injection_block(self._make_items(), budget_tokens=2000)
        assert "## Memory (Injected)" in block
        assert "format_version: 1" in block
        assert "budget_tokens:" in block
        assert "matched:" in block
        assert "used:" in block

    def test_block_has_end_marker(self):
        block = format_injection_block(self._make_items(), budget_tokens=2000)
        assert "--- End Memory" in block
        assert "format_version=1" in block

    def test_block_contains_items(self):
        block = format_injection_block(self._make_items(), budget_tokens=2000)
        assert "[1]" in block
        assert "SQLite for storage" in block
        assert "[2]" in block
        assert "Local-first design" in block

    def test_empty_items_returns_empty(self):
        block = format_injection_block([], budget_tokens=2000)
        assert block == ""

    def test_budget_trimming(self):
        """Budget must be enforced — fewer items if budget is tight."""
        items = self._make_items()
        small_block = format_injection_block(items, budget_tokens=50)
        large_block = format_injection_block(items, budget_tokens=5000)
        assert len(small_block) <= len(large_block)


# ---------------------------------------------------------------------------
# 4. Trace format — JSONL contract
# ---------------------------------------------------------------------------


class TestTraceFormatCompat:
    """Trace JSONL must have the fields expected by both systems."""

    REQUIRED_FIELDS = {"iter", "query", "new_items", "sim", "action"}

    def test_trace_has_required_fields(self):
        """Each trace entry must have the fields from ROADMAP §D5."""
        buf = io.StringIO()
        entry = {
            "iter": 1,
            "query": "auth flow",
            "new_items": 5,
            "sim": None,
            "sim_method": "lexical",
            "action": "continue",
        }
        emit_trace(entry, buf)
        parsed = json.loads(buf.getvalue())
        assert self.REQUIRED_FIELDS.issubset(parsed.keys())

    def test_trace_is_valid_jsonl(self):
        """Multiple trace entries must be valid JSONL (one per line)."""
        buf = io.StringIO()
        for i in range(3):
            emit_trace({"iter": i + 1, "action": "continue"}, buf)
        buf.seek(0)
        lines = buf.readlines()
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "iter" in parsed

    def test_ragix_extra_field_sim_method(self):
        """RAGIX traces include sim_method — must not break memctl parsing."""
        buf = io.StringIO()
        emit_trace(
            {"iter": 2, "sim": 0.87, "sim_method": "cosine", "action": "continue"},
            buf,
        )
        parsed = json.loads(buf.getvalue())
        assert parsed["sim_method"] == "cosine"
        # memctl would ignore unknown fields — this is forward-compatible

    def test_trace_example_from_roadmap(self):
        """Verify the exact example from ROADMAP §D5 is valid."""
        examples = [
            '{"iter":1,"query":"auth flow","new_items":5,"sim":null,"sim_method":"jaccard","action":"continue"}',
            '{"iter":2,"query":"token refresh","new_items":2,"sim":0.78,"sim_method":"cosine","action":"continue"}',
            '{"iter":3,"query":null,"new_items":0,"sim":0.94,"sim_method":"cosine","action":"fixed_point"}',
        ]
        for line in examples:
            parsed = json.loads(line)
            assert self.REQUIRED_FIELDS.issubset(parsed.keys())


# ---------------------------------------------------------------------------
# 5. Similarity contract — Tier A range compatibility
# ---------------------------------------------------------------------------


class TestSimilarityCompat:
    """Tier A similarity must produce compatible ranges with memctl."""

    def test_identical_text_is_one(self):
        assert lexical_similarity("hello world", "hello world") == 1.0

    def test_empty_text_is_one(self):
        assert lexical_similarity("", "") == 1.0

    def test_disjoint_text_low(self):
        """Disjoint token sets score low (SequenceMatcher adds residual)."""
        s = lexical_similarity("alpha beta gamma", "delta epsilon zeta")
        assert s < 0.3  # Jaccard=0, SequenceMatcher has char-level residual

    def test_range_zero_one(self):
        """Similarity must always be in [0, 1]."""
        pairs = [
            ("The quick brown fox", "The slow brown fox"),
            ("RAGIX memory subsystem", "memctl memory controller"),
            ("a", "z"),
            ("authentication token refresh", "auth token renew"),
        ]
        for a, b in pairs:
            s = lexical_similarity(a, b)
            assert 0.0 <= s <= 1.0, f"Out of range: {s} for ({a!r}, {b!r})"

    def test_normalize_compatible(self):
        """Both systems lowercase + strip punctuation + collapse whitespace."""
        assert normalize_text("Hello, World!") == "hello world"
        assert normalize_text("  foo   bar  ") == "foo bar"
        assert normalize_text("UPPER.CASE") == "uppercase"

    def test_jaccard_symmetry(self):
        """Jaccard must be symmetric."""
        a, b = "RAGIX memory", "memory RAGIX"
        assert token_jaccard(a, b) == token_jaccard(b, a)

    def test_memctl_default_weights_differ(self):
        """memctl uses 0.4/0.6, RAGIX uses 0.5/0.5 — both valid.

        This test documents the difference; it's intentional (RAGIX gives
        equal weight to set overlap and sequential ordering).
        """
        a = "RAGIX memory subsystem architecture"
        b = "memory architecture RAGIX subsystem"
        ragix_sim = lexical_similarity(a, b, w_jaccard=0.5, w_sequence=0.5)
        memctl_sim = lexical_similarity(a, b, w_jaccard=0.4, w_sequence=0.6)
        # Both valid, both in [0, 1]
        assert 0.0 <= ragix_sim <= 1.0
        assert 0.0 <= memctl_sim <= 1.0
        # Difference should be small for reasonable text
        assert abs(ragix_sim - memctl_sim) < 0.15


# ---------------------------------------------------------------------------
# 6. Exit code contract (D4)
# ---------------------------------------------------------------------------


class TestExitCodeContract:
    """Exit code semantics must match the shared spec."""

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_exit_0_on_normal_completion(self, mock_llm):
        """Code 0: LLM said stop."""
        mock_llm.return_value = (
            '{"need_more": false, "stop": true}\n\nFinal answer.'
        )
        config = LoopConfig(llm_command="mock", max_calls=3, similarity_mode="lexical")
        result = run_loop("context", config)
        assert result.answer == "Final answer."
        # No error condition
        assert "error" not in result.exit_reason.lower()

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_strict_mode_on_max_calls(self, mock_llm):
        """With --strict, max-calls without convergence is an error."""
        mock_llm.side_effect = [
            '{"need_more": true, "query": "q1", "stop": false}\n\nA1',
            '{"need_more": true, "query": "q2", "stop": false}\n\nA2 different',
        ]
        config = LoopConfig(
            llm_command="mock",
            max_calls=2,
            strict=True,
            similarity_mode="lexical",
            stop_on_no_new=False,
        )
        result = run_loop("context", config)
        assert "max_calls" in result.exit_reason
        assert result.converged is False


# ---------------------------------------------------------------------------
# 7. End-to-end mock: pipe | loop | pull pattern
# ---------------------------------------------------------------------------


class TestPipeLoopPullPattern:
    """Simulate the canonical 3-stage pipeline with mock LLM."""

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_injection_block_through_loop(self, mock_llm):
        """An injection block from pipe must flow through loop correctly."""
        # Simulate pipe output (injection block)
        items = [
            {
                "id": "chunk-1",
                "tier": "stm",
                "validation": "unverified",
                "type": "note",
                "title": "Auth overview",
                "content": "RAGIX uses JWT for authentication.",
                "provenance": {"source_kind": "doc", "source_id": "auth.md"},
                "tags": ["auth"],
                "confidence": 0.8,
            },
        ]
        injection_block = format_injection_block(items, budget_tokens=2000)

        # Mock LLM: stops after first call
        mock_llm.return_value = (
            '{"need_more": false, "stop": true}\n\n'
            "RAGIX uses JWT tokens for authentication, as documented in auth.md."
        )

        config = LoopConfig(
            llm_command="echo mock",
            max_calls=3,
            similarity_mode="lexical",
        )
        result = run_loop(injection_block, config)

        # The answer should be the LLM's response
        assert "JWT tokens" in result.answer
        assert result.iterations == 1

        # The answer is what would be piped to `pull`
        assert result.answer.strip()  # non-empty, suitable for pull

    @patch("ragix_core.memory.loop.invoke_llm")
    def test_loop_trace_recorded(self, mock_llm):
        """Trace must be recorded for each iteration."""
        mock_llm.side_effect = [
            '{"need_more": true, "query": "more details", "stop": false}\n\nPartial.',
            '{"need_more": false, "stop": true}\n\nComplete answer.',
        ]
        trace_buf = io.StringIO()
        config = LoopConfig(
            llm_command="mock",
            max_calls=3,
            similarity_mode="lexical",
            trace=True,
        )
        result = run_loop("context", config, trace_stream=trace_buf)

        trace_buf.seek(0)
        lines = trace_buf.readlines()
        assert len(lines) == 2

        t1 = json.loads(lines[0])
        assert t1["iter"] == 1
        assert t1["action"] == "continue"

        t2 = json.loads(lines[1])
        assert t2["iter"] == 2
        assert t2["action"] == "llm_stop"
