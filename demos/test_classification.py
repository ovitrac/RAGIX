#!/usr/bin/env python3
"""
Minimal test script for debugging task classification.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragix_core.reasoning import (
    TaskComplexity,
    GraphReasoningLoop,
    ReasoningLoop,
    create_reasoning_loop,
    get_reasoning_strategy,
)


def test_classification():
    """Test task classification for various queries."""

    print("=" * 60)
    print("RAGIX Task Classification Test")
    print("=" * 60)
    print(f"\nDefault reasoning strategy: {get_reasoning_strategy()}")
    print()

    # Test queries
    test_queries = [
        "How many Python files are in the ragix_core directory?",
        "Count the lines of code in ragix_core/config.py",
        "What is the SOLID principles?",
        "Find all classes in reasoning.py",
        "List all files in src/",
        "Analyze the structure of the reasoning module and list all classes",
        "Who are you?",
        "Hello",
        "Refactor the config module to use dataclasses",
    ]

    # Create a mock GraphReasoningLoop to test its classification
    # We can test the classify_task method directly

    print("Testing GraphReasoningLoop.classify_task():")
    print("-" * 60)

    # Create a minimal loop just for classification
    from ragix_core.reasoning import EpisodicMemory
    from ragix_core.agent_config import AgentConfig, AgentMode

    # Mock episodic memory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        episodic = EpisodicMemory(storage_path=Path(tmpdir))
        agent_config = AgentConfig(mode=AgentMode.MINIMAL)

        # Create GraphReasoningLoop
        loop = GraphReasoningLoop(
            llm_generate=lambda s, m: "",  # Mock LLM
            agent_config=agent_config,
            episodic_memory=episodic,
        )

        for query in test_queries:
            complexity = loop.classify_task(query)
            # Color coding
            color = {
                TaskComplexity.BYPASS: "\033[94m",   # Blue
                TaskComplexity.SIMPLE: "\033[92m",   # Green
                TaskComplexity.MODERATE: "\033[93m", # Yellow
                TaskComplexity.COMPLEX: "\033[91m",  # Red
            }.get(complexity, "")
            reset = "\033[0m"

            print(f"{color}[{complexity.value:^10}]{reset} {query[:50]}...")

    print("\n" + "=" * 60)
    print("Testing original ReasoningLoop.classify_task():")
    print("-" * 60)

    # Test original ReasoningLoop as well
    original_loop = ReasoningLoop(
        llm_generate=lambda s, m: "",
        agent_config=agent_config,
        episodic_memory=episodic,
    )

    for query in test_queries:
        complexity = original_loop.classify_task(query)
        color = {
            TaskComplexity.BYPASS: "\033[94m",
            TaskComplexity.SIMPLE: "\033[92m",
            TaskComplexity.MODERATE: "\033[93m",
            TaskComplexity.COMPLEX: "\033[91m",
        }.get(complexity, "")
        reset = "\033[0m"

        print(f"{color}[{complexity.value:^10}]{reset} {query[:50]}...")

    print()


def test_comparison():
    """Test that TaskComplexity comparison works correctly."""
    from ragix_core.reasoning import TaskComplexity as ReasoningTC
    import tempfile

    print("\n" + "=" * 60)
    print("TaskComplexity Comparison Test")
    print("=" * 60)

    # Create a mock GraphReasoningLoop
    from ragix_core.reasoning import EpisodicMemory, GraphReasoningLoop
    from ragix_core.agent_config import AgentConfig, AgentMode

    with tempfile.TemporaryDirectory() as tmpdir:
        episodic = EpisodicMemory(storage_path=Path(tmpdir))
        agent_config = AgentConfig(mode=AgentMode.MINIMAL)

        loop = GraphReasoningLoop(
            llm_generate=lambda s, m: "",
            agent_config=agent_config,
            episodic_memory=episodic,
        )

        # Test the comparison
        user_text = "How many Python files are in the ragix_core directory?"
        complexity = loop.classify_task(user_text)

        print(f"\nQuery: {user_text}")
        print(f"Returned complexity: {complexity}")
        print(f"Type: {type(complexity)}")
        print(f"Value: {complexity.value}")

        # Compare with agent.py's TaskComplexity
        print(f"\nComparison tests:")
        print(f"  complexity == ReasoningTC.SIMPLE: {complexity == ReasoningTC.SIMPLE}")
        print(f"  complexity == ReasoningTC.BYPASS: {complexity == ReasoningTC.BYPASS}")
        print(f"  complexity == ReasoningTC.MODERATE: {complexity == ReasoningTC.MODERATE}")

        # Check if they're the same class
        print(f"\n  Same class? {type(complexity) == ReasoningTC}")
        print(f"  Complexity class: {type(complexity).__module__}.{type(complexity).__name__}")
        print(f"  ReasoningTC class: {ReasoningTC.__module__}.{ReasoningTC.__name__}")


if __name__ == "__main__":
    test_classification()
    test_comparison()
