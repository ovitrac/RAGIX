"""The prompt shows the model exactly the placeholder the validator accepts.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Twice in one day a prompt typed its own braces and passed them through `str.format`: `{{claim:v1}}`
in a template renders as `{claim:v1}`, the model wrote what it was shown, and the validator refused
it. The rule that replaces vigilance: nothing types braces, everything calls `form.marker()`, and
this test renders the real prompt and asserts that what the model is shown is what the validator
accepts — for claim, k and trap alike.
"""
from __future__ import annotations

import re

from ragix_kernels.harvest import form, runner

#: any brace-wrapped marker, whatever the brace count, so a single-brace rendering is caught rather
#: than silently not found
ANY_MARKER = re.compile(r"\{+\s*(claim|k|trap)\s*:[^{}]*\}+")


def _prompt() -> str:
    return runner.rendered_prompt(node="n1", values='  v1 : "48 heures" (duration)', text="Un texte.",
                                  guided=True)


def test_marker_builds_what_the_pattern_matches():
    for kind, ref in (("claim", "v1"), ("k", "n_dce:1"), ("trap", "T16")):
        written = form.marker(kind, ref)
        assert form.PATTERNS[kind].fullmatch(written), f"{kind}: {written!r} is not accepted"


def test_the_prompt_shows_only_markers_the_validator_accepts():
    shown = _prompt()
    found = [m.group(0) for m in ANY_MARKER.finditer(shown)]
    assert found, "the prompt shows no placeholder at all: the model is asked to cite nothing"
    for token in found:
        kind = token.strip("{} ").split(":", 1)[0]
        assert form.PATTERNS[kind].fullmatch(token), \
            f"the prompt shows {token!r}, which the validator does not accept"


def test_a_single_brace_rendering_is_caught():
    """The regression itself: the shape a run once shipped must fail this test."""
    assert not form.PATTERNS["claim"].fullmatch("{claim:v1}")
    assert form.PATTERNS["claim"].fullmatch("{{claim:v1}}")
