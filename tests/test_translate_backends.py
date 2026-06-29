"""
Tests for the KOAS-Translate backend seam JSON parser
(ragix_kernels/translate/backends.parse_json_lenient).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

import pytest

from ragix_kernels.translate.backends import parse_json_lenient


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
