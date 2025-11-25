#!/usr/bin/env python3
"""
RAGIX Workflow Example - Multi-Agent Bug Fix Pipeline

This example demonstrates how to use RAGIX v0.7 features:
- Workflow templates for common tasks
- Graph-based multi-agent execution
- Streaming events for real-time updates
- Caching and monitoring

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragix_core import (
    # Workflow Templates
    get_template_manager,
    list_builtin_templates,
    # Graph Execution
    GraphExecutor,
    ExecutionContext,
    ExecutionStatus,
    StreamEvent,
    # Agent Types
    AgentNode,
    # Monitoring
    get_agent_monitor,
    get_health_checker,
    # Caching
    create_llm_cache,
)


async def run_workflow_example():
    """Demonstrate workflow template execution with streaming."""
    print("=" * 60)
    print("RAGIX v0.7 - Workflow Example")
    print("=" * 60)

    # 1. List available templates
    print("\n1. Available Workflow Templates:")
    print("-" * 40)
    templates = list_builtin_templates()
    for name, description in templates.items():
        print(f"  - {name}: {description}")

    # 2. Create a workflow from template
    print("\n2. Creating Bug Fix Workflow:")
    print("-" * 40)

    manager = get_template_manager()
    graph = manager.instantiate(
        "bug_fix",
        {
            "bug_description": "TypeError: NoneType has no attribute 'process' in data_handler.py",
            "affected_files": "src/handlers/",
        }
    )

    print(f"  Workflow: {graph.name}")
    print(f"  Nodes: {len(graph.nodes)}")
    for node_id, node in graph.nodes.items():
        task = node.config.get("task", "") if node.config else ""
        task_preview = task[:50] + "..." if len(task) > 50 else task
        print(f"    - {node_id}: {task_preview}")

    # 3. Execute with streaming
    print("\n3. Executing Workflow (Simulation):")
    print("-" * 40)

    # Create a mock agent factory for demonstration
    async def mock_agent_run(context):
        await asyncio.sleep(0.5)  # Simulate work
        return "Completed successfully"

    class MockAgent:
        def __init__(self, node):
            self.node = node

        async def run(self, context):
            await asyncio.sleep(0.3)
            return f"Result for {self.node.id}"

    def agent_factory(workflow_id, node):
        return MockAgent(node)

    executor = GraphExecutor(graph)

    async for event in executor.execute_streaming(agent_factory):
        if isinstance(event, StreamEvent):
            if event.event_type == "workflow_started":
                print(f"  [START] Workflow {event.data.get('workflow_id')}")
            elif event.event_type == "node_started":
                print(f"  [RUNNING] {event.node_id}")
            elif event.event_type == "node_completed":
                print(f"  [DONE] {event.node_id}")
            elif event.event_type == "workflow_completed":
                print(f"  [FINISH] Status: {event.data.get('status')}")
        else:
            # Final ExecutionResult
            result = event
            print(f"\n  Final Status: {result.status.value}")
            print(f"  Completed Nodes: {len(result.completed_nodes)}")
            print(f"  Duration: {result.duration_seconds:.2f}s")

    print("\n" + "=" * 60)
    print("Workflow execution complete!")


async def run_monitoring_example():
    """Demonstrate monitoring capabilities."""
    print("\n" + "=" * 60)
    print("RAGIX v0.7 - Monitoring Example")
    print("=" * 60)

    # 1. Health checks
    print("\n1. System Health Checks:")
    print("-" * 40)

    checker = get_health_checker()
    report = checker.get_status_report()

    print(f"  Overall Status: {report['status']}")
    for name, check in report['checks'].items():
        status_icon = "✓" if check['status'] == "healthy" else "✗"
        print(f"  [{status_icon}] {name}: {check['message']}")

    # 2. Agent monitoring
    print("\n2. Agent Execution Monitoring:")
    print("-" * 40)

    monitor = get_agent_monitor()

    # Simulate some executions
    for i in range(3):
        exec_id = f"exec_{i+1:03d}"
        monitor.start_execution(exec_id, f"Task {i+1}", "code")
        monitor.record_tool_call("read_file", success=True, duration=0.1)
        monitor.record_tool_call("grep_search", success=True, duration=0.2)
        monitor.end_execution(exec_id, success=True, agent_type="code")

    # Add a failure
    monitor.start_execution("exec_fail", "Failing task", "code")
    monitor.end_execution("exec_fail", success=False, agent_type="code", error="Test error")

    stats = monitor.get_stats()
    print(f"  Active Executions: {stats['active_executions']}")
    print(f"  Recent Errors: {len(stats['recent_errors'])}")

    metrics = stats['metrics']
    if 'counters' in metrics:
        print(f"  Metrics collected: {len(metrics['counters'])} counters")


async def run_caching_example():
    """Demonstrate caching capabilities."""
    print("\n" + "=" * 60)
    print("RAGIX v0.7 - Caching Example")
    print("=" * 60)

    # 1. Create LLM cache
    print("\n1. LLM Response Caching:")
    print("-" * 40)

    cache = create_llm_cache(
        cache_type="memory",
        max_size=100,
        ttl=3600,
    )

    # Simulate caching queries
    queries = [
        "What is Python?",
        "Explain async/await",
        "What is Python?",  # Cache hit
        "How does RAGIX work?",
        "What is Python?",  # Cache hit
    ]

    for query in queries:
        cached = cache.get(query)
        if cached:
            print(f"  [HIT] {query[:30]}...")
        else:
            response = f"Response for: {query}"
            cache.set(query, response)
            print(f"  [MISS] {query[:30]}...")

    stats = cache.stats
    print(f"\n  Cache Statistics:")
    print(f"    Size: {stats['size']}")
    print(f"    Hits: {stats['hits']}")
    print(f"    Misses: {stats['misses']}")
    print(f"    Hit Rate: {stats['hit_rate']:.1%}")


async def main():
    """Run all examples."""
    await run_workflow_example()
    await run_monitoring_example()
    await run_caching_example()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
