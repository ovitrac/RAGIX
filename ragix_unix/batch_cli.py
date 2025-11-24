#!/usr/bin/env python3
"""
RAGIX Batch CLI - Run workflows in CI/CD pipelines

Usage:
    ragix-batch <playbook.yaml> [options]

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from ragix_core import (
    BatchConfig,
    BatchExecutor,
    run_batch_sync,
)
from ragix_core.agents import BaseAgent, CodeAgent, DocAgent, GitAgent, TestAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def create_agent_factory(config: BatchConfig):
    """
    Create agent factory function for the batch executor.

    Args:
        config: Batch configuration

    Returns:
        Factory function that creates agents
    """

    def factory(workflow_id: str, node: Any) -> BaseAgent:
        """Create agent instance based on node configuration."""
        agent_type = node.agent_type.lower()

        if agent_type == "code_agent":
            return CodeAgent(workflow_id, node)
        elif agent_type == "doc_agent":
            return DocAgent(workflow_id, node)
        elif agent_type == "git_agent":
            return GitAgent(workflow_id, node)
        elif agent_type == "test_agent":
            return TestAgent(workflow_id, node)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    return factory


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAGIX Batch Executor - Run workflows in CI/CD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CI checks
  ragix-batch ci_checks.yaml

  # Run with custom output
  ragix-batch playbook.yaml --output-dir ./reports

  # Verbose output
  ragix-batch playbook.yaml --verbose

  # Save results as JSON
  ragix-batch playbook.yaml --json-report results.json

Example playbook.yaml:
  name: "CI Lint and Test"
  description: "Run linting and tests"
  fail_fast: true
  max_parallel: 2
  model: "mistral:instruct"
  profile: "safe-read-only"

  workflows:
    - name: "Lint Python Code"
      type: "linear"
      steps:
        - name: "run_ruff"
          agent: "code_agent"
          tools: ["bash"]
        - name: "run_black"
          agent: "code_agent"
          tools: ["bash"]

    - name: "Run Tests"
      type: "linear"
      steps:
        - name: "pytest"
          agent: "test_agent"
          tools: ["bash"]
""",
    )

    parser.add_argument("playbook", type=Path, help="Path to YAML playbook file")

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for reports (default: current directory)",
    )

    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Save JSON report to file",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first workflow failure (overrides config)",
    )

    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Continue on failures (overrides config)",
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Max parallel agents (overrides config)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output (errors only)"
    )

    args = parser.parse_args()

    # Set logging level
    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load configuration
    playbook_path = args.playbook.resolve()
    if not playbook_path.exists():
        logger.error(f"Playbook not found: {playbook_path}")
        sys.exit(10)

    try:
        config = BatchConfig.from_yaml(playbook_path)
        logger.info(f"Loaded playbook: {config.name}")
        if config.description:
            logger.info(f"Description: {config.description}")
    except Exception as e:
        logger.error(f"Failed to load playbook: {e}", exc_info=args.verbose)
        sys.exit(10)

    # Apply CLI overrides
    if args.fail_fast:
        config.fail_fast = True
    if args.no_fail_fast:
        config.fail_fast = False
    if args.max_parallel:
        config.max_parallel = args.max_parallel

    # Create agent factory
    agent_factory = create_agent_factory(config)

    # Execute batch
    try:
        logger.info("Starting batch execution...")
        result = run_batch_sync(config, agent_factory, verbose=args.verbose)

        # Print summary
        if not args.quiet:
            result.print_summary()

        # Save JSON report
        if args.json_report:
            result.save_json(args.json_report)
            logger.info(f"JSON report saved to: {args.json_report}")

        # Exit with appropriate code
        sys.exit(result.exit_code.value)

    except KeyboardInterrupt:
        logger.info("\nBatch execution cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Batch execution failed: {e}", exc_info=args.verbose)
        sys.exit(20)


if __name__ == "__main__":
    main()
