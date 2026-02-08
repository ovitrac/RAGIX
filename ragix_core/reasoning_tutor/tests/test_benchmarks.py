#!/usr/bin/env python3
"""
Pytest tests for Interpreter-Tutor benchmarks.

These tests can be run with:
  pytest ragix_core/reasoning_tutor/tests/test_benchmarks.py -v

For CI, use markers:
  pytest -m "smoke" ...       # Quick smoke tests (no LLM)
  pytest -m "regression" ...  # Full regression tests (requires Ollama)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

import json
import pytest
import subprocess
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.regression_runner import (
    load_baseline, parse_existing_results, compare_results,
    FAST_MODEL, GOLD_MODEL, BASELINE_DIR
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture
def baseline_granite():
    """Load granite3.1-moe:3b baseline."""
    return load_baseline(FAST_MODEL)


@pytest.fixture
def baseline_deepseek():
    """Load deepseek-r1:14b baseline."""
    return load_baseline(GOLD_MODEL)


# ============================================================================
# SMOKE TESTS (no LLM required)
# ============================================================================

@pytest.mark.smoke
class TestSmoke:
    """Smoke tests that don't require Ollama."""

    def test_baseline_files_exist(self):
        """Verify baseline files are present."""
        baselines = list(BASELINE_DIR.glob("baseline_*.json"))
        assert len(baselines) >= 1, "No baseline files found"

    def test_baseline_granite_loads(self, baseline_granite):
        """Verify granite baseline loads correctly."""
        assert baseline_granite is not None
        assert baseline_granite.model == FAST_MODEL
        assert len(baseline_granite.benchmarks) == 6

    def test_baseline_granite_structure(self, baseline_granite):
        """Verify baseline structure is correct."""
        for bid, result in baseline_granite.benchmarks.items():
            assert result.benchmark_id == bid
            assert isinstance(result.success, bool)
            assert isinstance(result.final_score, int)
            assert result.total_turns >= 0

    def test_benchmark_yaml_files_exist(self, project_root):
        """Verify benchmark YAML files exist (B01-B10)."""
        benchmark_dir = project_root / "ragix_core" / "reasoning_tutor" / "benchmarks"
        # B01-B06 original + B07 stack trace + B08 diff + B09 cycle + B10 temporal
        for i in range(1, 11):
            pattern = f"0{i}_*.yaml" if i < 10 else f"{i}_*.yaml"
            matches = list(benchmark_dir.glob(pattern))
            assert len(matches) == 1, f"Missing benchmark {i:02d} YAML file"

    def test_scored_mode_imports(self):
        """Verify scored_mode.py imports without errors."""
        try:
            from ragix_core.reasoning_tutor.benchmarks import scored_mode
            assert hasattr(scored_mode, "run_scored_game")
            assert hasattr(scored_mode, "run_all_benchmarks")
        except ImportError as e:
            pytest.fail(f"Failed to import scored_mode: {e}")

    def test_tool_adapter_imports(self):
        """Verify tool_call_adapter.py imports without errors."""
        try:
            from ragix_core.reasoning_tutor import tool_call_adapter
            assert hasattr(tool_call_adapter, "get_adapter_for_model")
        except ImportError as e:
            pytest.fail(f"Failed to import tool_call_adapter: {e}")

    def test_synthesis_controller_imports(self):
        """Verify synthesis_controller.py imports without errors."""
        try:
            from ragix_core.reasoning_tutor import synthesis_controller
            assert hasattr(synthesis_controller, "SynthesisController")
        except ImportError as e:
            pytest.fail(f"Failed to import synthesis_controller: {e}")

    def test_b07_yaml_structure(self, project_root):
        """Verify B07 benchmark YAML has correct structure."""
        import yaml
        benchmark_path = project_root / "ragix_core/reasoning_tutor/benchmarks/07_stack_trace.yaml"
        assert benchmark_path.exists(), "B07 YAML file not found"

        with open(benchmark_path) as f:
            data = yaml.safe_load(f)

        # Check required fields
        assert data["meta"]["name"] == "Stack Trace Diagnosis"
        assert data["meta"]["category"] == "error_analysis"
        assert len(data["setup"]["files"]) == 5
        assert "divisor" in data["goal"]["text"].lower()

        # Check files include crash.log and config
        file_names = [f["name"] for f in data["setup"]["files"]]
        assert "logs/crash.log" in file_names
        assert "config/settings.yaml" in file_names
        assert "handler.py" in file_names

    def test_b07_synthesis_controller_goals(self):
        """Verify B07 is registered in synthesis controller."""
        from ragix_core.reasoning_tutor.synthesis_controller import BENCHMARK_GOALS
        assert "Stack Trace Diagnosis" in BENCHMARK_GOALS
        goals = BENCHMARK_GOALS["Stack Trace Diagnosis"]
        assert len(goals) >= 2  # At least crash_analyzed and config_found

    def test_b08_yaml_structure(self, project_root):
        """Verify B08 benchmark YAML has correct structure."""
        import yaml
        benchmark_path = project_root / "ragix_core/reasoning_tutor/benchmarks/08_diff_analysis.yaml"
        assert benchmark_path.exists(), "B08 YAML file not found"

        with open(benchmark_path) as f:
            data = yaml.safe_load(f)

        # Check required fields
        assert data["meta"]["name"] == "Diff Analysis"
        assert data["meta"]["category"] == "comparison"
        assert len(data["setup"]["files"]) == 5

        # Check files include both versions
        file_names = [f["name"] for f in data["setup"]["files"]]
        assert "src/calculator_v1.py" in file_names
        assert "src/calculator_v2.py" in file_names

    def test_b08_synthesis_controller_goals(self):
        """Verify B08 is registered in synthesis controller."""
        from ragix_core.reasoning_tutor.synthesis_controller import BENCHMARK_GOALS
        assert "Diff Analysis" in BENCHMARK_GOALS
        goals = BENCHMARK_GOALS["Diff Analysis"]
        assert len(goals) >= 2  # At least v1_read and v2_read

    def test_b09_yaml_structure(self, project_root):
        """Verify B09 benchmark YAML has correct structure."""
        import yaml
        benchmark_path = project_root / "ragix_core/reasoning_tutor/benchmarks/09_cycle_detection.yaml"
        assert benchmark_path.exists(), "B09 YAML file not found"

        with open(benchmark_path) as f:
            data = yaml.safe_load(f)

        # Check required fields
        assert data["meta"]["name"] == "Dependency Cycle Detection"
        assert data["meta"]["category"] == "graph_reasoning"
        assert len(data["setup"]["files"]) == 7

        # Check files include module files and error log
        file_names = [f["name"] for f in data["setup"]["files"]]
        assert "modules/auth.py" in file_names
        assert "modules/user.py" in file_names
        assert "modules/permissions.py" in file_names
        assert "logs/import_error.log" in file_names

    def test_b09_synthesis_controller_goals(self):
        """Verify B09 is registered in synthesis controller."""
        from ragix_core.reasoning_tutor.synthesis_controller import BENCHMARK_GOALS
        assert "Dependency Cycle Detection" in BENCHMARK_GOALS
        goals = BENCHMARK_GOALS["Dependency Cycle Detection"]
        assert len(goals) >= 3  # At least auth_import, user_import, permissions_import

    def test_b10_yaml_structure(self, project_root):
        """Verify B10 benchmark YAML has correct structure."""
        import yaml
        benchmark_path = project_root / "ragix_core/reasoning_tutor/benchmarks/10_temporal_correlation.yaml"
        assert benchmark_path.exists(), "B10 YAML file not found"

        with open(benchmark_path) as f:
            data = yaml.safe_load(f)

        # Check required fields
        assert data["meta"]["name"] == "Temporal Event Correlation"
        assert data["meta"]["category"] == "temporal_reasoning"
        assert len(data["setup"]["files"]) == 6

        # Check files include service logs and clock skew info
        file_names = [f["name"] for f in data["setup"]["files"]]
        assert "logs/service_a.log" in file_names
        assert "logs/service_b.log" in file_names
        assert "logs/service_c.log" in file_names
        assert "notes/clock_skew.txt" in file_names

    def test_b10_synthesis_controller_goals(self):
        """Verify B10 is registered in synthesis controller."""
        from ragix_core.reasoning_tutor.synthesis_controller import BENCHMARK_GOALS
        assert "Temporal Event Correlation" in BENCHMARK_GOALS
        goals = BENCHMARK_GOALS["Temporal Event Correlation"]
        assert len(goals) >= 4  # At least service logs + clock_skew


# ============================================================================
# HISTORICAL RESULT TESTS (validate stored results)
# ============================================================================

@pytest.mark.smoke
class TestHistoricalResults:
    """Test parsing and validation of historical result files."""

    def test_parse_round5_granite(self, project_root):
        """Parse Round 5 granite results."""
        path = project_root / "ragix_core/reasoning_tutor/results/round5/final/granite3_3b.jsonl"
        if not path.exists():
            pytest.skip("Round 5 results not found")

        results = parse_existing_results(path)
        assert len(results) == 6
        assert "01" in results
        assert results["01"].model == "granite3.1-moe:3b"

    def test_parse_round5_deepseek(self, project_root):
        """Parse Round 5 deepseek results."""
        path = project_root / "ragix_core/reasoning_tutor/results/round5/final/deepseek_14b.jsonl"
        if not path.exists():
            pytest.skip("Round 5 results not found")

        results = parse_existing_results(path)
        assert len(results) == 6
        # DeepSeek should have 6/6 wins
        wins = sum(1 for r in results.values() if r.success)
        assert wins == 6, f"Expected 6 wins, got {wins}"


# ============================================================================
# REGRESSION TESTS (require Ollama)
# ============================================================================

def ollama_available():
    """Check if Ollama is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.regression
@pytest.mark.skipif(not ollama_available(), reason="Ollama not available")
class TestRegression:
    """Regression tests requiring Ollama."""

    def test_regression_granite_b01(self, project_root, baseline_granite):
        """Run B01 with granite and compare to baseline."""
        if baseline_granite is None:
            pytest.skip("No baseline for granite")

        cmd = [
            sys.executable, "-m",
            "ragix_core.reasoning_tutor.benchmarks.scored_mode",
            "--models", FAST_MODEL,
            "--benchmark", "01",
            "--max-turns", "6",
            "--quiet"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=120
        )

        # Parse result
        for line in result.stdout.split("\n"):
            if '"type": "game_end"' in line:
                try:
                    record = json.loads(line)
                    actual_success = record.get("success", False)
                    baseline_success = baseline_granite.benchmarks["01"].success

                    # Win status should match
                    assert actual_success == baseline_success, \
                        f"B01 win status changed: {actual_success} vs baseline {baseline_success}"
                    return
                except json.JSONDecodeError:
                    continue

        pytest.fail("No game_end found in output")

    def test_regression_granite_b02(self, project_root, baseline_granite):
        """Run B02 with granite and compare to baseline."""
        if baseline_granite is None:
            pytest.skip("No baseline for granite")

        cmd = [
            sys.executable, "-m",
            "ragix_core.reasoning_tutor.benchmarks.scored_mode",
            "--models", FAST_MODEL,
            "--benchmark", "02",
            "--max-turns", "6",
            "--quiet"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=120
        )

        # Parse result
        for line in result.stdout.split("\n"):
            if '"type": "game_end"' in line:
                try:
                    record = json.loads(line)
                    actual_success = record.get("success", False)
                    baseline_success = baseline_granite.benchmarks["02"].success

                    assert actual_success == baseline_success, \
                        f"B02 win status changed: {actual_success} vs baseline {baseline_success}"
                    return
                except json.JSONDecodeError:
                    continue

        pytest.fail("No game_end found in output")


# ============================================================================
# COMPARISON LOGIC TESTS
# ============================================================================

@pytest.mark.smoke
class TestComparisonLogic:
    """Test the comparison logic itself."""

    def test_compare_identical_results(self, baseline_granite):
        """Comparing identical results should pass."""
        if baseline_granite is None:
            pytest.skip("No baseline")

        # Compare baseline against itself
        comparisons, passed = compare_results(
            baseline_granite, baseline_granite.benchmarks
        )
        assert passed, "Identical results should pass"
        assert all(c.passed for c in comparisons)

    def test_compare_detects_win_regression(self, baseline_granite):
        """Detect when a win becomes a loss."""
        if baseline_granite is None:
            pytest.skip("No baseline")

        # Modify a copy to simulate regression
        modified = dict(baseline_granite.benchmarks)
        from dataclasses import replace
        # Find a winning benchmark and flip it
        for bid, result in modified.items():
            if result.success:
                modified[bid] = replace(result, success=False)
                break

        comparisons, passed = compare_results(baseline_granite, modified)
        assert not passed, "Win regression should fail"

    def test_compare_tolerance(self, baseline_granite):
        """Score within tolerance should pass."""
        if baseline_granite is None:
            pytest.skip("No baseline")

        from dataclasses import replace
        modified = dict(baseline_granite.benchmarks)

        # Adjust scores by 5% (within default 10% tolerance)
        for bid, result in modified.items():
            new_score = int(result.final_score * 1.05)
            modified[bid] = replace(result, final_score=new_score)

        comparisons, passed = compare_results(
            baseline_granite, modified, score_tolerance_pct=10
        )
        # Should still pass (within tolerance)
        assert passed, "5% score change should be within 10% tolerance"
