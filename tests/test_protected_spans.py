"""
Tests for the shared protected-span codec (ragix_kernels/shared/protected_spans).

Covers masking of every rule, round-trip fidelity, idempotency, cross-segment
token uniqueness, rule selection, and the restore diagnostics (hallucinated /
dropped placeholders).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import pytest

from ragix_kernels.shared.protected_spans import (
    protect,
    restore,
    SpanCounter,
    RULE_NAMES,
    TOKEN_RE,
)


# ---------------------------------------------------------------------------
# protect
# ---------------------------------------------------------------------------

class TestProtect:
    @pytest.mark.parametrize("span", [
        "`inline code`",
        "$x = 1$",
        "$$\\nabla \\cdot u = 0$$",
        "[label](https://example.com)",
        "![alt](img.png)",
        "https://example.com/a/b",
        "[Smith 2020]",
        "[Smith et al. 2020a]",
        "[12,34]",
        "<!-- a hidden comment -->",
    ])
    def test_each_span_is_masked(self, span):
        text = f"before {span} after"
        masked, mapping = protect(text)
        assert span not in masked            # the raw span is gone
        assert len(mapping) == 1
        tok = next(iter(mapping))
        assert TOKEN_RE.fullmatch(tok)
        assert mapping[tok] == span
        assert tok in masked

    def test_currency_is_not_masked(self):
        # "$5" / "$10" must not be mistaken for inline math
        text = "It costs $5 today and $10 tomorrow."
        masked, mapping = protect(text)
        assert masked == text
        assert mapping == {}

    def test_plain_text_untouched(self):
        text = "A perfectly ordinary sentence with no special spans."
        masked, mapping = protect(text)
        assert masked == text
        assert mapping == {}

    def test_tokens_are_sequential(self):
        text = "`a` then `b` then `c`"
        masked, mapping = protect(text)
        assert list(mapping) == ["⟦P0001⟧", "⟦P0002⟧", "⟦P0003⟧"]
        assert mapping["⟦P0001⟧"] == "`a`"
        assert mapping["⟦P0003⟧"] == "`c`"

    def test_idempotent_second_pass_is_noop(self):
        text = "math $x$ and code `y` and a [ref](http://z)"
        masked1, map1 = protect(text)
        masked2, map2 = protect(masked1)   # already-masked tokens match nothing
        assert masked2 == masked1
        assert map2 == {}


# ---------------------------------------------------------------------------
# SpanCounter — cross-segment uniqueness
# ---------------------------------------------------------------------------

class TestSpanCounter:
    def test_shared_counter_avoids_token_collision(self):
        counter = SpanCounter()
        m1, map1 = protect("first `a`", counter=counter)
        m2, map2 = protect("second `b`", counter=counter)
        # token namespaces must not overlap across segments
        assert set(map1) & set(map2) == set()
        assert list(map1) == ["⟦P0001⟧"]
        assert list(map2) == ["⟦P0002⟧"]

    def test_fresh_counter_restarts(self):
        _, map1 = protect("`a`")
        _, map2 = protect("`b`")
        assert list(map1) == list(map2) == ["⟦P0001⟧"]


# ---------------------------------------------------------------------------
# rule selection
# ---------------------------------------------------------------------------

class TestRuleSelection:
    def test_subset_applies_only_requested_rules(self):
        text = "math $x$ and code `y`"
        masked, mapping = protect(text, rules=["math_inline"])
        assert "`y`" in masked              # code rule not applied
        assert "$x$" not in masked          # math rule applied
        assert list(mapping.values()) == ["$x$"]

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="unknown protection rule"):
            protect("x", rules=["does_not_exist"])

    def test_rule_names_are_stable(self):
        # guards the public surface the kernels depend on
        assert "math_inline" in RULE_NAMES
        assert "code_fenced" in RULE_NAMES
        assert RULE_NAMES[0] == "comment_html"   # must run first


# ---------------------------------------------------------------------------
# round-trip + restore diagnostics
# ---------------------------------------------------------------------------

class TestRestore:
    def test_round_trip_is_lossless(self):
        text = (
            "See [Smith 2020] and the eq. $$E = mc^2$$ with inline $a_i$, "
            "code `f(x)`, image ![d](d.png), link [home](https://h.io), "
            "url https://raw.io/x and refs [1-3].\n"
            "<!-- note -->\n```\ncode block\n```"
        )
        masked, mapping = protect(text)
        restored, report = restore(masked, mapping)
        assert restored == text
        assert report.ok
        assert report.restored == len(mapping)
        assert report.hallucinated == []
        assert report.dropped == []

    def test_hallucinated_token_reported_and_left_verbatim(self):
        restored, report = restore("stray ⟦P0009⟧ here", mapping={})
        assert restored == "stray ⟦P0009⟧ here"
        assert report.hallucinated == ["⟦P0009⟧"]
        assert not report.ok
        assert report.restored == 0

    def test_dropped_token_reported(self):
        mapping = {"⟦P0001⟧": "`code`", "⟦P0002⟧": "$x$"}
        restored, report = restore("only ⟦P0001⟧ survived", mapping)
        assert restored == "only `code` survived"
        assert report.dropped == ["⟦P0002⟧"]
        assert report.hallucinated == []
        assert not report.ok
        assert report.restored == 1

    def test_restore_is_exact_for_special_chars(self):
        mapping = {"⟦P0001⟧": r"$\frac{\partial \rho}{\partial t}$"}
        restored, report = restore("flux ⟦P0001⟧ term", mapping)
        assert restored == r"flux $\frac{\partial \rho}{\partial t}$ term"
        assert report.ok


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
