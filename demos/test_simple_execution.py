#!/usr/bin/env python3
"""
Test script for SIMPLE task execution flow.

This tests the end-to-end flow from classification to execution for SIMPLE tasks.

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


def test_simple_execution():
    """Test SIMPLE task execution flow."""

    print("=" * 60)
    print("RAGIX SIMPLE Task Execution Test")
    print("=" * 60)

    # Setup
    project_root = Path(__file__).parent.parent
    model = "qwen2.5:7b"  # Default model

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

    # Test queries
    test_cases = [
        {
            "query": "How many Python files are in the ragix_core directory?",
            "expected_complexity": "simple",
            "expect_result": True,
        },
        {
            "query": "List the files in ragix_core/",
            "expected_complexity": "simple",
            "expect_result": True,
        },
        {
            "query": "Count the lines in ragix_core/config.py",
            "expected_complexity": "simple",
            "expect_result": True,
        },
    ]

    print("\n" + "-" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Query: {test['query']}")

        # Classify
        complexity = agent._reasoning_loop.classify_task(test['query'])
        print(f"  Complexity: {complexity.value} (expected: {test['expected_complexity']})")

        # Execute
        print("  Executing...")
        cmd_result, response, traces = agent.step_with_reasoning(test['query'])

        # Results
        print(f"  Traces: {[t.get('type') for t in traces]}")

        if cmd_result:
            print(f"  Command result: returncode={cmd_result.returncode}")
            print(f"  stdout: {cmd_result.stdout[:200] if cmd_result.stdout else '(empty)'}")
        else:
            print("  Command result: None")

        if response:
            print(f"  Response: {response[:200]}...")
        else:
            print("  Response: None")

        # Check success
        if complexity.value == test['expected_complexity']:
            print("  ✅ Classification correct")
        else:
            print("  ❌ Classification incorrect")

        if test['expect_result'] and (cmd_result or response):
            print("  ✅ Got result")
        elif not test['expect_result'] and not cmd_result and not response:
            print("  ✅ No result as expected")
        else:
            print(f"  ⚠️ Unexpected result state (cmd_result={bool(cmd_result)}, response={bool(response)})")

        print("-" * 60)

    print("\nTest complete!")


if __name__ == "__main__":
    test_simple_execution()
