"""
Gate T0 — the family exists, is discovered, and reaches its store without touching it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

This family holds one kernel that computes almost nothing. That is deliberate: the
things being proved here are discovery, registration, the dependency declaration,
the two surfaces, the configuration refusals and the guard scope — and every one
of them is easier to get wrong than the arithmetic they will later carry. A
skeleton that cannot be wrong on substance is the only thing that isolates them.

The fixture is a real document store, built at test time from the generators the
saqqara gates use. Nothing is committed, and no corpus is read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "saqqara"))

import generators as G  # noqa: E402

from ragix_kernels.base import KernelInput  # noqa: E402
from ragix_kernels.saqqara.kernels.saqqara_index import SaqqaraIndexKernel  # noqa: E402
from ragix_kernels.saqqara.kernels.saqqara_run import SaqqaraKernel  # noqa: E402

from ragix_kernels.tender.kernels.tender_probe import TenderProbeKernel  # noqa: E402
from ragix_kernels.tender.models import ProbeResult  # noqa: E402

#: Every Kernel subclass this family defines, by the module that defines it.
#: Pinned, and amended deliberately at each addition — the same rule saqqara's
#: K0.3 applies, and for the same reason: a module that grows a Kernel becomes an
#: independently registered kernel with no other ceremony.
DECLARED_KERNELS = ["ragix_kernels.tender.kernels.tender_probe.TenderProbeKernel"]

#: The tools this family exposes, in registration order.
DECLARED_TOOLS = ["koas_tender_probe", "koas_tender_status"]


class _StubServer:
    def __init__(self):
        self.tools: dict = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


@pytest.fixture(scope="module")
def store_workspace(tmp_path_factory):
    """A real saqqara store, lexical-only, built from generated documents."""
    workspace = tmp_path_factory.mktemp("t0")
    source = workspace / "corpus"
    source.mkdir()
    for name, fixture in (("a.docx", "docx_two_tier"), ("b.md", "markdown_document")):
        G.FIXTURES[fixture](source / name)

    SaqqaraKernel().run(KernelInput(workspace=workspace,
                                    config={"source": {"path": str(source)}}))
    read = workspace / "stage1" / "saqqara.json"
    indexed = SaqqaraIndexKernel().run(KernelInput(
        workspace=workspace, config={}, dependencies={"document_tree": read}))
    assert indexed.success, indexed.errors
    return workspace, Path(indexed.data["status"]["path"])


@pytest.fixture
def config_file(tmp_path, store_workspace):
    _workspace, db = store_workspace
    path = tmp_path / "tender.yaml"
    path.write_text(f'store:\n  path: "{db}"\n', encoding="utf-8")
    return path


# ------------------------------------------------- T0.1 the family is discovered

def test_t0_1_the_registry_finds_exactly_the_declared_kernels():
    """A module that grows a Kernel registers itself; the list says which are meant.

    Falsified by: a kernel discovered under this family that no line declares.
    """
    from ragix_kernels.base import Kernel
    import importlib
    import pkgutil

    package = importlib.import_module("ragix_kernels.tender")
    found = []
    for _, module_name, _ in pkgutil.walk_packages(
        [str(Path(package.__file__).parent)], prefix="ragix_kernels.tender."
    ):
        module = importlib.import_module(module_name)
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, Kernel) and attr is not Kernel:
                if attr.__module__ == module_name:
                    found.append(f"{module_name}.{attr.__name__}")
    assert sorted(found) == sorted(DECLARED_KERNELS), found
    assert package.__doc__, "the family package must document itself"


# --------------------------------------- T0.2 the declaration is enforced, not decorative

def test_t0_2_the_kernel_declares_what_it_needs_and_gives():
    assert TenderProbeKernel.requires == ["document_store"]
    assert TenderProbeKernel.provides == ["tender_probe"]
    assert TenderProbeKernel.stage == 3


def test_t0_2_the_envelope_refuses_without_the_dependency(tmp_path):
    """Falsified by: a probe that proceeds with no store and reports success."""
    output = TenderProbeKernel().run(KernelInput(workspace=tmp_path, config={}))
    assert output.success is False
    assert any("document_store" in str(e) for e in output.errors), output.errors


def test_t0_2_a_dangling_dependency_path_is_refused(tmp_path):
    output = TenderProbeKernel().run(KernelInput(
        workspace=tmp_path, config={},
        dependencies={"document_store": tmp_path / "absent.db"}))
    assert output.success is False
    assert any("does not exist" in str(e) for e in output.errors), output.errors


def test_t0_2_the_registry_orders_this_family_after_the_store():
    from ragix_kernels.registry import KernelRegistry

    KernelRegistry.discover()
    probe = KernelRegistry.get("tender_probe")
    index = KernelRegistry.get("saqqara_index")
    assert probe is not None and index is not None
    assert probe.stage > index.stage


# ------------------------------------------------- T0.3 the probe reports the store

def test_t0_3_the_probe_reports_what_the_store_reports(store_workspace, config_file):
    """Counts equal to the store's own status, not a second count of its own."""
    workspace, db = store_workspace
    output = TenderProbeKernel().run(KernelInput(
        workspace=workspace, config={"config": str(config_file)},
        dependencies={"document_store": db}))
    assert output.success, output.errors

    result = ProbeResult.from_dict(output.data["probe"])
    from ragix_kernels.saqqara.store.ports import build_store
    truth = build_store({"provider": "sqlite", "path": str(db)}).status()

    assert result.documents == truth["documents"]
    assert result.chunks == truth["chunks"]
    assert result.store_path == str(db)


def test_t0_3_dense_disabled_is_stated_not_inferred(store_workspace, config_file):
    """The store was built with no embedder, so the probe says so in a field."""
    workspace, db = store_workspace
    output = TenderProbeKernel().run(KernelInput(
        workspace=workspace, config={"config": str(config_file)},
        dependencies={"document_store": db}))
    result = ProbeResult.from_dict(output.data["probe"])
    assert result.dense_enabled is False
    assert "disabled" in result.dense.lower()


def test_t0_3_the_record_round_trips():
    record = ProbeResult(store_path="/x.db", documents=2, chunks=7,
                         dense_enabled=False, dense="disabled (no embedder)")
    assert ProbeResult.from_dict(record.to_dict()).to_dict() == record.to_dict()
    keys = list(record.to_dict())
    assert keys == sorted(keys)


# --------------------------------------------------- T0.4 it reads and does not write

def test_t0_4_the_probe_does_not_touch_the_store(store_workspace, config_file):
    """Read-only, asserted on the bytes rather than on intent.

    Falsified by: any change to the database file across a probe — a journal, a
    vacuum, a timestamp, an accidental write.
    """
    workspace, db = store_workspace
    before = hashlib.sha256(db.read_bytes()).hexdigest()

    output = TenderProbeKernel().run(KernelInput(
        workspace=workspace, config={"config": str(config_file)},
        dependencies={"document_store": db}))
    assert output.success, output.errors

    after = hashlib.sha256(db.read_bytes()).hexdigest()
    assert after == before, "the probe modified the store it was asked to read"


# ------------------------------------------------------- T0.5 the two surfaces agree

def test_t0_5_the_mcp_surface_registers_exactly_the_declared_tools():
    from ragix_kernels.tender.mcp.tools import register_tender_tools

    server = _StubServer()
    register_tender_tools(server)
    assert list(server.tools) == DECLARED_TOOLS


def test_t0_5_cli_and_mcp_return_the_same_probe(store_workspace, config_file):
    """One shape, or two surfaces that will disagree about what the store holds."""
    import argparse
    import contextlib
    import io

    from ragix_kernels.tender.cli.tenderctl import cmd_probe
    from ragix_kernels.tender.mcp.tools import register_tender_tools

    args = argparse.Namespace(config=str(config_file), json=True, verbose=False)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert cmd_probe(args) == 0
    from_cli = json.loads(buffer.getvalue())

    server = _StubServer()
    register_tender_tools(server)
    from_mcp = server.tools["koas_tender_probe"](config=str(config_file))

    assert "error" not in from_mcp, from_mcp
    assert from_mcp["probe"] == from_cli["probe"]


def test_t0_5_the_server_registers_the_family_surface():
    """Checked on the source: importing the server pulls in the whole project."""
    server = ROOT / "MCP" / "ragix_mcp_server.py"
    if not server.is_file():
        pytest.skip("MCP server is not part of this checkout")
    text = server.read_text(encoding="utf-8")
    assert "register_tender_tools" in text, "the server does not register this family"


# --------------------------------------------------- T0.6 the configuration refuses

def test_t0_6_an_unknown_config_key_is_refused_with_its_path(tmp_path):
    from ragix_kernels.tender.config import load_config

    path = tmp_path / "bad.yaml"
    path.write_text("store:\n  pth: /x.db\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"store\.pth"):
        load_config(path)


def test_t0_6_a_missing_store_is_refused_and_never_created(tmp_path):
    """A probe that creates the store it was asked to inspect reports on itself."""
    from ragix_kernels.tender.config import load_config

    absent = tmp_path / "nowhere.db"
    path = tmp_path / "cfg.yaml"
    path.write_text(f'store:\n  path: "{absent}"\n', encoding="utf-8")
    config = load_config(path)

    output = TenderProbeKernel().run(KernelInput(
        workspace=tmp_path, config={"config": str(path)},
        dependencies={"document_store": tmp_path}))
    assert output.success is False
    assert not absent.exists(), "the probe created the store it was meant to read"
    assert config.get("store.path") == str(absent)


# ----------------------------------------- T0.7 the family leaves saqqara alone

def test_t0_7_saqqara_pinned_lists_are_untouched_by_this_family():
    """A new family must not move another's gates.

    Falsified by: saqqara's declared kernel list or frozen counts changing because
    something here was added.
    """
    sys.path.insert(0, str(ROOT / "tests" / "saqqara"))
    import test_k0_spec as saqqara_spec

    assert saqqara_spec.DECLARED_KERNELS == [
        "ragix_kernels.saqqara.kernels.saqqara_index.SaqqaraIndexKernel",
        "ragix_kernels.saqqara.kernels.saqqara_run.SaqqaraKernel",
    ]
    # The literal is the point: comparing the module against itself would pass
    # whatever it holds. It moves only when saqqara's own surface moves, in that
    # change's commit — never to make this test green.
    assert saqqara_spec.FROZEN_COUNTS == {
        "K1": 10, "K2": 25, "K3": 72, "K4": 3, "K6": 20, "K7": 22}
