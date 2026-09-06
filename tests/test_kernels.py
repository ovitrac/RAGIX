"""
Test KOAS Kernels

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

import tempfile
import json
from pathlib import Path
import pytest

from ragix_kernels.base import Kernel, KernelInput, KernelOutput
from ragix_kernels.registry import KernelRegistry


class TestKernelBase:
    """Test Kernel base class."""

    def test_kernel_input_creation(self):
        """Test KernelInput dataclass."""
        workspace = Path("/tmp/test")
        config = {"project": {"path": "/some/path"}}

        input = KernelInput(
            workspace=workspace,
            config=config,
            dependencies={}
        )

        assert input.workspace == workspace
        assert input.config == config
        assert input.dependencies == {}

    def test_kernel_input_string_path(self):
        """Test KernelInput converts string paths."""
        input = KernelInput(
            workspace="/tmp/test",  # String, should be converted
            config={},
        )

        assert isinstance(input.workspace, Path)


class TestKernelRegistry:
    """Test Kernel registry."""

    def setup_method(self):
        """Reset registry before each test."""
        KernelRegistry.reset()

    def test_discover_kernels(self):
        """Test kernel discovery."""
        count = KernelRegistry.discover()

        # Should discover at least ast_scan and metrics
        assert count >= 2

    def test_get_kernel(self):
        """Test getting a kernel by name."""
        KernelRegistry.discover()

        ast_kernel = KernelRegistry.get("ast_scan")
        assert ast_kernel.name == "ast_scan"
        assert ast_kernel.stage == 1

    def test_list_category(self):
        """Test listing kernels by category."""
        KernelRegistry.discover()

        audit_kernels = KernelRegistry.list_category("audit")
        assert "ast_scan" in audit_kernels
        assert "metrics" in audit_kernels

    def test_list_stage(self):
        """Test listing kernels by stage."""
        KernelRegistry.discover()

        stage1 = KernelRegistry.list_stage(1)
        assert "ast_scan" in stage1
        assert "metrics" in stage1

    def test_get_info(self):
        """Test getting kernel info."""
        KernelRegistry.discover()

        info = KernelRegistry.get_info("ast_scan")
        assert info["name"] == "ast_scan"
        assert info["version"] == "1.1.0"
        assert info["category"] == "audit"
        assert info["stage"] == 1
        assert info["requires"] == []
        assert "symbols" in info["provides"]

    def test_resolve_dependencies(self):
        """Test dependency resolution."""
        KernelRegistry.discover()

        # metrics depends on ast_scan
        ordered = KernelRegistry.resolve_dependencies(["metrics"])

        # ast_scan should come before metrics
        ast_idx = ordered.index("ast_scan")
        metrics_idx = ordered.index("metrics")
        assert ast_idx < metrics_idx


class TestASTScanKernel:
    """Test AST scan kernel."""

    def setup_method(self):
        """Reset registry and discover kernels."""
        KernelRegistry.reset()
        KernelRegistry.discover()

    def test_ast_scan_on_small_project(self):
        """Test AST scan on a small Python project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create a simple Python file
            test_file = workspace / "test_project" / "example.py"
            test_file.parent.mkdir(parents=True)
            test_file.write_text('''
class MyClass:
    """A simple class."""

    def my_method(self):
        return 42

    def another_method(self, x):
        if x > 0:
            return x * 2
        return 0

def standalone_function():
    """A standalone function."""
    return "hello"
''')

            # Create kernel input
            input = KernelInput(
                workspace=workspace,
                config={
                    "project": {
                        "path": str(workspace / "test_project"),
                        "language": "python",
                    }
                },
            )

            # Run kernel
            kernel = KernelRegistry.get_instance("ast_scan")
            output = kernel.run(input)

            # Verify output
            assert output.success
            assert output.kernel_name == "ast_scan"
            assert output.output_file.exists()

            # Check data
            data = output.data
            assert "symbols" in data
            assert "files" in data
            assert "statistics" in data

            # Check we found the class and methods
            symbols = data["symbols"]
            symbol_names = [s["name"] for s in symbols]
            assert "MyClass" in symbol_names or any("MyClass" in s["qualified_name"] for s in symbols)

    def test_ast_scan_summary(self):
        """Test that AST scan produces meaningful summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create a simple Python file
            test_file = workspace / "src" / "main.py"
            test_file.parent.mkdir(parents=True)
            test_file.write_text('def hello(): pass')

            input = KernelInput(
                workspace=workspace,
                config={"project": {"path": str(workspace / "src"), "language": "python"}},
            )

            kernel = KernelRegistry.get_instance("ast_scan")
            output = kernel.run(input)

            # Summary should mention key stats
            assert "AST scan complete" in output.summary
            assert len(output.summary) <= 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================ requires, resolved
#
# A `requires` entry is a kernel name, or a capability that exactly one kernel
# provides. Before this, `provides` was declared by every kernel and read by
# nothing: two kernels — `saqqara_index` and `tender_probe` — declared a capability
# and raised `KeyError: Kernel 'document_tree' not found` through the orchestrator
# while working perfectly when called directly.
#
# Making `provides` load-bearing across a registry that was never gated on it is
# the cost of that fix, so the invariant below is most of this change: it holds the
# whole registry, not the two kernels that prompted it, and it fails the day
# somebody adds a second provider for a capability something requires.

class TestRequiresResolution:
    """The resolver, and the invariant it puts on every registered kernel."""

    def test_every_requirement_resolves_to_exactly_one_kernel(self):
        """The invariant. Falsified by: an entry naming nothing, or two producers.

        Measured when this landed: 97 kernels, two requiring a capability, three
        capabilities with two providers each and none of them required by anyone.
        """
        KernelRegistry.discover()
        unresolved = []
        for name in sorted(KernelRegistry._kernels):
            for entry in KernelRegistry.get(name).requires:
                try:
                    KernelRegistry.resolve_requirement(entry)
                except (KeyError, ValueError) as exc:
                    unresolved.append(f"{name} requires {entry!r}: {exc}")
        assert not unresolved, "\n  ".join(["unresolvable requirements:"] + unresolved)

    def test_a_kernel_name_wins_over_a_capability_of_the_same_name(self):
        """Name first is not a preference: sixteen capabilities are also kernel
        names, so capability-first would silently re-point existing pipelines.

        Falsified by: a collision resolving to the provider rather than the kernel.
        """
        KernelRegistry.discover()
        both = [n for n in KernelRegistry._kernels
                if any(n in (KernelRegistry.get(o).provides or [])
                       for o in KernelRegistry._kernels if o != n)]
        assert both, "no name/capability collision left to prove the rule on"
        for name in both:
            assert KernelRegistry.resolve_requirement(name) == (name, "name")

    def test_a_capability_one_kernel_provides_resolves_to_it(self):
        KernelRegistry.discover()
        assert KernelRegistry.resolve_requirement("document_tree") == ("saqqara", "capability")
        assert KernelRegistry.resolution_of("saqqara_index") == [
            {"requires": "document_tree", "kernel": "saqqara", "how": "capability"}]

    def test_two_providers_is_an_error_that_names_them(self, monkeypatch):
        """Picking one would be this code choosing a producer nobody declared.

        Registered here rather than pointed at a real ambiguity, so the test does
        not go quiet the day somebody resolves that one.
        """
        class _A(Kernel):
            name, version, category, stage = "prov_a", "0.1.0", "test", 1
            description, requires, provides = "a", [], ["twice_provided"]
            def compute(self, input): return {}
            def summarize(self, data): return ""

        class _B(_A):
            name = "prov_b"

        KernelRegistry.discover()
        monkeypatch.setitem(KernelRegistry._kernels, "prov_a", _A)
        monkeypatch.setitem(KernelRegistry._kernels, "prov_b", _B)

        with pytest.raises(ValueError) as raised:
            KernelRegistry.resolve_requirement("twice_provided")
        assert "prov_a" in str(raised.value) and "prov_b" in str(raised.value)
        assert "Name the kernel" in str(raised.value)

    def test_an_unknown_requirement_says_what_exists(self):
        KernelRegistry.discover()
        with pytest.raises(KeyError) as raised:
            KernelRegistry.resolve_requirement("nothing_provides_this")
        assert "neither a kernel name nor a capability" in str(raised.value)

    def test_dependency_order_expands_through_a_capability(self):
        """The graph is over kernel names; an unresolved entry would sort as a
        kernel of its own and never find its producer.

        Falsified by: `saqqara_index` ordered without `saqqara` before it.
        """
        KernelRegistry.discover()
        order = KernelRegistry.resolve_dependencies(["saqqara_index"])
        assert order.index("saqqara") < order.index("saqqara_index")


class TestStageTwoThroughTheOrchestrator:
    """End to end: the indexer is reachable by the route the manifest uses.

    This is the case that failed. `saqqara_index` ran perfectly when a driver built
    its input by hand — which is what the demo had to do — and raised through
    `run_stage(2)`, so the bypass looked like a driver detail rather than a defect
    in the resolver.
    """

    def test_run_stage_two_reaches_the_indexer_and_records_what_it_resolved(self, tmp_path):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent / "saqqara"))
        import generators as G

        from ragix_kernels.orchestrator import Orchestrator

        source = tmp_path / "corpus"
        source.mkdir()
        G.FIXTURES["markdown_document"](source / "a.md")

        # `stage1:`/`stage2:` at the top level, which is what `from_yaml` reads.
        # Eight example manifests in this repository write `stages:` with numeric
        # keys instead, and the loader ignores the whole block: their kernels run
        # with no options at all. Recorded here because a test written from those
        # examples fails in a way that looks like the resolver.
        (tmp_path / "manifest.yaml").write_text(
            "audit:\n"
            "  name: resolver check\n"
            "stage1:\n"
            "  saqqara:\n"
            "    enabled: true\n"
            "    options:\n"
            "      source:\n"
            f"        path: {source}\n"
            "stage2:\n"
            "  saqqara_index:\n"
            "    enabled: true\n",
            encoding="utf-8",
        )

        orchestrator = Orchestrator(workspace=tmp_path)
        first = orchestrator.run_stage(1, kernels=["saqqara"])
        assert first and first[0].success, first

        second = orchestrator.run_stage(2, kernels=["saqqara_index"])
        assert second, "stage 2 ran no kernel — the indexer was not reached"
        output = second[0]
        assert output.success is True, output.errors
        assert output.data["status"]["chunks"] > 0

        assert output.dependencies_resolved == [
            {"requires": "document_tree", "kernel": "saqqara", "how": "capability"}], \
            "the run does not record which kernel served the requirement"


def test_the_activity_start_event_records_what_each_requirement_resolved_to():
    """The journal must say which kernel served a capability, not only that one did.

    Falsified by: a start event whose refs do not name the producer and how it was
    found — the day a second reader provides `document_tree`, a run with no such
    record cannot be told from a run served by the other one.
    """
    from ragix_kernels.activity import ActivityWriter

    with tempfile.TemporaryDirectory() as tmp:
        writer = ActivityWriter(workspace=Path(tmp), run_id="test-run")
        writer.emit_kernel_start(
            kernel_name="saqqara_index", kernel_version="0.1.0", stage=2,
            dependencies=[{"requires": "document_tree", "kernel": "saqqara",
                           "how": "capability"}],
        )
        events = [json.loads(line) for line in
                  (Path(tmp) / ".KOAS" / "activity" / "events.jsonl").read_text().splitlines()]

    (event,) = events
    assert event["refs"] == {"requires:document_tree": "saqqara (capability)"}
