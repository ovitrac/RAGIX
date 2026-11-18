"""`search` reads its configuration as `index` does, and refuses a bad one cleanly.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Two behaviours listed in docs/KOAS_SAQQARA_DEV.md §11 until 0.75: `search` built its
embedder from the provider and model only, so `embedder.base_url` and
`embedder.batch_size` reached `index` and not `search`; and `store_options` values
were cast with bool(), so the string "false" read as on. These tests hold the fix.
"""
from __future__ import annotations

import argparse

import pytest

from ragix_kernels.saqqara.store import embed as embed_module
from ragix_kernels.saqqara.store.config import load_config
from ragix_kernels.saqqara.store.embed import embedder_options
from ragix_kernels.saqqara.store.ports import build_store


# ── store_options: booleans only ─────────────────────────────────────────────

@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, "no"])
def test_a_store_option_that_is_not_a_boolean_is_refused(value):
    """Falsified by: load_config accepting any non-boolean store option."""
    with pytest.raises(ValueError, match=r"store_options\.table_cells must be true or false"):
        load_config({"store_options": {"table_cells": value}})


def test_a_store_option_given_as_an_override_is_refused_by_the_same_rule():
    with pytest.raises(ValueError, match=r"store_options\.rollup_without_window_overlap"):
        load_config(**{"store_options.rollup_without_window_overlap": "false"})


@pytest.mark.parametrize("value", [True, False])
def test_a_boolean_store_option_is_kept_as_given(value):
    assert load_config({"store_options": {"table_cells": value}}).section(
        "store_options")["table_cells"] is value


def test_a_file_that_is_not_yaml_is_refused_as_a_configuration_error(tmp_path):
    bad = tmp_path / "saqqara.yaml"
    bad.write_text("store: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a YAML configuration"):
        load_config(bad)


# ── embedder options: one rule for index and search ──────────────────────────

def test_embedder_options_pass_only_what_is_set():
    assert embedder_options({"base_url": "", "batch_size": 0}) == {}
    assert embedder_options({"base_url": "http://192.0.2.1:11434", "batch_size": 8}) == {
        "base_url": "http://192.0.2.1:11434", "batch_size": 8}
    assert embedder_options({"provider": "ollama", "model": "m"}) == {}


def _workspace_with_store(tmp_path, embedder: dict) -> argparse.Namespace:
    db = tmp_path / "saqqara.db"
    build_store({**load_config().section("store"), "path": str(db)})
    assert db.is_file()
    lines = ["store:", f"  path: {db}", "embedder:"]
    lines += [f"  {key}: {value}" for key, value in embedder.items()]
    config = tmp_path / "saqqara.yaml"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return argparse.Namespace(workspace=str(tmp_path), query="paragraphe",
                              config=str(config), top_k=5, json=True, verbose=False)


def test_search_builds_its_embedder_with_base_url_and_batch_size(tmp_path, monkeypatch):
    """Falsified by: search reaching another server, or batching differently, than index.

    The embedder is replaced by a recorder that returns None, so no server is
    contacted and the search runs on the lexical lane.
    """
    from ragix_kernels.saqqara.cli.saqqaractl import cmd_search

    calls = []

    def recorder(provider, model="", **options):
        calls.append((provider, model, options))
        return None

    monkeypatch.setattr(embed_module, "build_embedder", recorder)
    args = _workspace_with_store(tmp_path, {
        "provider": "ollama", "model": "some-model",
        "base_url": "http://192.0.2.1:11434", "batch_size": 7})
    assert cmd_search(args) == 0
    assert calls == [("ollama", "some-model",
                      {"base_url": "http://192.0.2.1:11434", "batch_size": 7})]


def test_search_without_those_keys_takes_the_defaults_as_index_does(tmp_path, monkeypatch):
    """An empty base_url is not passed; the defaults' batch_size is, as for index."""
    from ragix_kernels.saqqara.cli.saqqaractl import cmd_search

    calls = []
    monkeypatch.setattr(embed_module, "build_embedder",
                        lambda provider, model="", **options: calls.append(options))
    args = _workspace_with_store(tmp_path, {"provider": "ollama"})
    assert cmd_search(args) == 0
    defaults = load_config().section("embedder")
    assert defaults["base_url"] == ""
    assert calls == [{"batch_size": defaults["batch_size"]}]


# ── a refused configuration is an exit code and a reason, not a traceback ────

def test_search_refuses_a_bad_configuration_with_exit_code_2(tmp_path, capsys):
    from ragix_kernels.saqqara.cli.saqqaractl import cmd_search

    config = tmp_path / "saqqara.yaml"
    config.write_text("store_options:\n  table_cells: \"false\"\n", encoding="utf-8")
    args = argparse.Namespace(workspace=str(tmp_path), query="q", config=str(config),
                              top_k=5, json=True, verbose=False)
    assert cmd_search(args) == 2
    captured = capsys.readouterr()
    assert "configuration refused" in captured.err
    assert "store_options.table_cells" in captured.err
    assert captured.out == ""


def test_the_mcp_search_tool_keeps_the_reason_of_a_refusal(tmp_path):
    from ragix_kernels.saqqara.mcp.tools import register_saqqara_tools

    class Server:
        tools: dict = {}

        def tool(self):
            def register(fn):
                self.tools[fn.__name__] = fn
                return fn
            return register

    server = Server()
    register_saqqara_tools(server)
    config = tmp_path / "saqqara.yaml"
    config.write_text("embedder:\n  no_such_key: 1\n", encoding="utf-8")
    result = server.tools["koas_saqqara_search"](
        workspace=str(tmp_path), query="q", k=5, config=str(config))
    assert result["error"].startswith("search failed with code 2: configuration refused")
    assert "no_such_key" in result["error"]
    assert "\x1b[" not in result["error"]
