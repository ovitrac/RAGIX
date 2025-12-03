#!/usr/bin/env python3
"""
Test script for MODERATE task execution flow.

This tests the end-to-end flow from classification to execution for MODERATE tasks
to verify step results are captured and displayed.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

import sys
import os
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragix_core import OllamaLLM, ShellSandbox
from ragix_unix.agent import UnixRAGAgent


def test_moderate_execution():
    """Test MODERATE task execution flow."""

    print("=" * 60)
    print("RAGIX MODERATE Task Execution Test")
    print("=" * 60)

    # Setup
    project_root = Path(__file__).parent.parent
    model = "qwen2.5:7b"

    print(f"\nProject root: {project_root}")
    print(f"Model: {model}")

    # Create LLM
    llm = OllamaLLM(model=model)

    # Create shell sandbox
    shell = ShellSandbox(
        root=str(project_root),
        profile="dev",
        dry_run=False,
    )

    # Create agent
    agent = UnixRAGAgent(
        llm=llm,
        shell=shell,
        use_reasoning_loop=True,
    )

    # Test query that should be classified as MODERATE
    # (doesn't match simple_patterns but needs tools)
    test_query = "Find all classes in ragix_core/reasoning.py"

    print(f"\n[Test] Query: {test_query}")

    # Classify
    complexity = agent._reasoning_loop.classify_task(test_query)
    print(f"  Complexity: {complexity.value}")

    # Execute
    print("  Executing...")
    cmd_result, response, traces = agent.step_with_reasoning(test_query)

    # Results
    print(f"\n  Traces ({len(traces)}):")
    for t in traces:
        print(f"    - {t.get('type')}: {str(t.get('content', ''))[:80]}...")

    if cmd_result:
        print(f"\n  Command result: returncode={cmd_result.returncode}")
        print(f"  stdout: {cmd_result.stdout[:500] if cmd_result.stdout else '(empty)'}")
    else:
        print("\n  Command result: None")

    if response:
        print(f"\n  Response:\n{'-'*40}\n{response[:1500]}\n{'-'*40}")
    else:
        print("\n  Response: None")

    print("\nTest complete!")


if __name__ == "__main__":
    test_moderate_execution()
