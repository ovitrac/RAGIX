"""
Tests for the KOAS-Translate backend seam JSON parser
(ragix_kernels/translate/backends.parse_json_lenient).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import pytest

from ragix_kernels.translate.backends import load_prompt, parse_json_lenient


def test_parses_clean_json():
    assert parse_json_lenient('{"status": "ok", "issues": []}') == {"status": "ok", "issues": []}


def test_strips_code_fence():
    raw = '```json\n{"status": "revise", "issues": []}\n```'
    assert parse_json_lenient(raw)["status"] == "revise"


def test_extracts_object_from_prose():
    raw = 'Voici le résultat : {"status": "ok", "issues": []} — fin.'
    assert parse_json_lenient(raw) == {"status": "ok", "issues": []}


def test_handles_nested_and_strings_with_braces():
    raw = 'prefix {"a": {"b": "}"}, "c": 1} suffix'
    assert parse_json_lenient(raw) == {"a": {"b": "}"}, "c": 1}


def test_non_object_json_returned_as_is():
    # whole-response parse succeeds → returns the bare value (caller handles non-dict)
    assert parse_json_lenient('"ok"') == "ok"


def test_unparseable_raises_valueerror():
    with pytest.raises(ValueError, match="did not return JSON"):
        parse_json_lenient("no json here at all")


# ---------------------------------------------------------------------------
# load_prompt — language-pair resolution
# ---------------------------------------------------------------------------

@pytest.fixture
def prompts(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "translate.txt").write_text("EN-FR default", encoding="utf-8")  # en-fr default
    (d / "translate.en-de.txt").write_text("EN-DE override", encoding="utf-8")
    return d / "translate.txt"


def test_default_pair_uses_bundled_default(prompts):
    assert load_prompt({}, prompts) == "EN-FR default"
    assert load_prompt({"lang_pair": "en-fr"}, prompts) == "EN-FR default"


def test_pair_override_is_used_when_present(prompts):
    assert load_prompt({"lang_pair": "en-de"}, prompts) == "EN-DE override"


def test_missing_pair_override_falls_back_to_default(prompts):
    assert load_prompt({"lang_pair": "en-es"}, prompts) == "EN-FR default"  # no es file


def test_explicit_overrides_win(prompts, tmp_path):
    assert load_prompt({"prompt_template": "INLINE"}, prompts) == "INLINE"
    explicit = tmp_path / "custom.txt"
    explicit.write_text("FROM PATH", encoding="utf-8")
    assert load_prompt({"prompt_path": str(explicit), "lang_pair": "en-de"}, prompts) == "FROM PATH"
