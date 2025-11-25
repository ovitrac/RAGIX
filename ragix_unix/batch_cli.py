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
    get_template_manager,
    list_builtin_templates,
)
from ragix_core.agents import BaseAgent, CodeAgent, DocAgent, GitAgent, TestAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def list_templates():
    """List available workflow templates."""
    templates = list_builtin_templates()
    print("\nAvailable workflow templates:\n")
    for name, description in templates.items():
        print(f"  {name}")
        print(f"    {description}\n")
    print("Use: ragix-batch --template <name> --params 'key=value,key2=value2'")


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

  # List available workflow templates
  ragix-batch --list-templates

  # Run a built-in template
  ragix-batch --template bug_fix --params "bug_description=TypeError in handler"

  # Run template with multiple params
  ragix-batch --template feature_addition --params "feature_name=caching,feature_spec=Add Redis cache"

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

    parser.add_argument(
        "playbook", type=Path, nargs="?", help="Path to YAML playbook file"
    )

    parser.add_argument(
        "--template",
        "-t",
        type=str,
        default=None,
        help="Use a built-in workflow template instead of playbook",
    )

    parser.add_argument(
        "--params",
        "-p",
        type=str,
        default=None,
        help="Template parameters as 'key=value,key2=value2'",
    )

    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available workflow templates",
    )

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

    # Handle --list-templates
    if args.list_templates:
        list_templates()
        sys.exit(0)

    # Handle --template mode
    if args.template:
        try:
            manager = get_template_manager()
            template = manager.get_template(args.template)
            if template is None:
                logger.error(f"Unknown template: {args.template}")
                logger.info("Use --list-templates to see available templates")
                sys.exit(10)

            # Parse parameters
            params = {}
            if args.params:
                for pair in args.params.split(","):
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        params[key.strip()] = value.strip()

            # Instantiate template as graph
            logger.info(f"Instantiating template: {args.template}")
            graph = template.instantiate(params)
            logger.info(f"Workflow: {graph.name}")
            logger.info(f"Steps: {len(graph.nodes)}")

            # Create a simple batch config from the template
            config = BatchConfig(
                name=graph.name,
                description=graph.description,
                fail_fast=True,
                max_parallel=1,
            )

            # Note: Full template execution would need GraphExecutor integration
            # For now, just show what would be executed
            print(f"\nWorkflow: {graph.name}")
            print(f"Description: {graph.description}")
            print(f"\nSteps:")
            for node in graph.nodes.values():
                print(f"  - {node.node_id} ({node.agent_type})")
                print(f"    Task: {node.task[:80]}...")
                print(f"    Tools: {', '.join(node.allowed_tools)}")
            print("\nNote: Full execution requires GraphExecutor integration (Task 3.4)")
            sys.exit(0)

        except Exception as e:
            logger.error(f"Failed to instantiate template: {e}", exc_info=args.verbose)
            sys.exit(10)

    # Load configuration from playbook
    if not args.playbook:
        logger.error("Either playbook or --template required")
        parser.print_help()
        sys.exit(10)

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
