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
        assert info["version"] == "1.0.0"
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
