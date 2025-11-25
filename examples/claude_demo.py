#!/usr/bin/env python3
"""
RAGIX Claude Demo - Complete Integration Demonstration

This example demonstrates RAGIX v0.7 working with Claude through MCP integration:
- Full RAGIX capabilities accessible to Claude
- Multi-agent workflow execution
- Hybrid search (BM25 + vector)
- Production monitoring and health checks
- Resilience patterns

This script can be run:
1. Standalone to test RAGIX features
2. As an MCP server tool callable by Claude Desktop/Claude Code

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragix_core import (
    # Core components
    OllamaLLM,
    ShellSandbox,
    Profile,
    compute_dry_run_from_profile,
    # Agent graph
    AgentGraph,
    AgentNode,
    AgentEdge,
    NodeStatus,
    GraphExecutor,
    ExecutionStatus,
    StreamEvent,
    # Workflow templates
    get_template_manager,
    list_builtin_templates,
    # Hybrid search
    BM25Index,
    BM25Document,
    Tokenizer,
    FusionStrategy,
    # Code chunking
    CodeChunk,
    ChunkType,
    # Caching
    create_llm_cache,
    LLMCache,
    # Monitoring
    get_health_checker,
    get_agent_monitor,
    get_metrics,
    check_ollama_health,
    check_disk_space,
    check_memory_usage,
    HealthStatus,
    # Resilience
    retry_async,
    RetryConfig,
    BackoffStrategy,
    CircuitBreaker,
    GracefulDegradation,
    FallbackChain,
    # Tool registry
    get_tool_registry,
    ToolCategory,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OLLAMA_MODEL = os.environ.get("UNIX_RAG_MODEL", "mistral")
SANDBOX_ROOT = os.environ.get("UNIX_RAG_SANDBOX", str(Path.cwd()))
AGENT_PROFILE = os.environ.get("UNIX_RAG_PROFILE", "dev")


# -----------------------------------------------------------------------------
# Demo Components
# -----------------------------------------------------------------------------

@dataclass
class DemoResult:
    """Result from a demonstration component."""
    component: str
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


class RAGIXClaudeDemo:
    """
    Comprehensive RAGIX v0.7 demonstration for Claude integration.

    This class encapsulates all RAGIX capabilities in a way that can be:
    1. Called directly from Python
    2. Exposed via MCP for Claude Desktop/Claude Code
    3. Used in automated testing
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        sandbox_root: str = SANDBOX_ROOT,
        profile: str = AGENT_PROFILE,
    ):
        self.model = model
        self.sandbox_root = Path(sandbox_root).resolve()
        self.profile = profile
        self.dry_run = compute_dry_run_from_profile(profile)

        # Initialize components lazily
        self._llm: Optional[OllamaLLM] = None
        self._shell: Optional[ShellSandbox] = None
        self._cache: Optional[LLMCache] = None
        self._bm25_index: Optional[BM25Index] = None

        # Initialize monitoring
        self.health_checker = get_health_checker()
        self.agent_monitor = get_agent_monitor()
        self.metrics = get_metrics()

        # Register health checks
        self._register_health_checks()

    def _register_health_checks(self) -> None:
        """Register health checks for RAGIX components."""
        self.health_checker.register("ollama", check_ollama_health)
        self.health_checker.register("disk", check_disk_space)
        self.health_checker.register("memory", check_memory_usage)

    @property
    def llm(self) -> OllamaLLM:
        """Lazy-loaded LLM instance."""
        if self._llm is None:
            self._llm = OllamaLLM(self.model)
        return self._llm

    @property
    def shell(self) -> ShellSandbox:
        """Lazy-loaded shell sandbox."""
        if self._shell is None:
            self._shell = ShellSandbox(
                root=str(self.sandbox_root),
                dry_run=self.dry_run,
                profile=self.profile,
            )
        return self._shell

    @property
    def cache(self) -> LLMCache:
        """Lazy-loaded LLM cache."""
        if self._cache is None:
            self._cache = create_llm_cache(
                cache_type="memory",
                max_size=100,
                ttl=3600,
            )
        return self._cache

    # -------------------------------------------------------------------------
    # Health and Status
    # -------------------------------------------------------------------------

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.

        Returns dict with:
        - overall_status: 'healthy' | 'degraded' | 'unhealthy'
        - components: individual component statuses
        - metrics: key system metrics
        """
        report = self.health_checker.get_status_report()

        return {
            "overall_status": report["status"],
            "components": report["checks"],
            "timestamp": datetime.now().isoformat(),
            "ragix_version": "0.7.0",
            "model": self.model,
            "sandbox": str(self.sandbox_root),
            "profile": self.profile,
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return self.metrics.export()

    # -------------------------------------------------------------------------
    # BM25 Search Demo
    # -------------------------------------------------------------------------

    def demo_bm25_search(self, query: str = "error handling") -> DemoResult:
        """
        Demonstrate BM25 keyword search.

        Args:
            query: Search query

        Returns:
            DemoResult with search results
        """
        start = datetime.now()

        # Create sample documents (simulating code chunks)
        sample_docs = [
            BM25Document(
                doc_id="main.py:1",
                file_path="main.py",
                start_line=1,
                end_line=20,
                chunk_type="function",
                name="main",
                tokens=["def", "main", "entry", "point", "application", "start"],
            ),
            BM25Document(
                doc_id="handler.py:10",
                file_path="handler.py",
                start_line=10,
                end_line=30,
                chunk_type="function",
                name="handle_error",
                tokens=["def", "handle", "error", "exception", "try", "catch", "log"],
            ),
            BM25Document(
                doc_id="utils.py:5",
                file_path="utils.py",
                start_line=5,
                end_line=15,
                chunk_type="function",
                name="validate_input",
                tokens=["def", "validate", "input", "check", "error", "raise"],
            ),
            BM25Document(
                doc_id="config.py:1",
                file_path="config.py",
                start_line=1,
                end_line=50,
                chunk_type="class",
                name="Config",
                tokens=["class", "config", "settings", "load", "yaml", "json"],
            ),
            BM25Document(
                doc_id="database.py:20",
                file_path="database.py",
                start_line=20,
                end_line=45,
                chunk_type="function",
                name="connect_db",
                tokens=["def", "connect", "database", "retry", "error", "handle", "connection"],
            ),
        ]

        # Build index
        index = BM25Index(k1=1.5, b=0.75)
        for doc in sample_docs:
            index.add_document(doc)

        # Search
        results = index.search(query, k=5)

        duration = (datetime.now() - start).total_seconds() * 1000

        formatted_results = []
        for r in results:
            formatted_results.append({
                "doc_id": r.doc_id,
                "file_path": r.file_path,
                "name": r.name,
                "score": round(r.score, 4),
                "matched_terms": r.matched_terms,
            })

        return DemoResult(
            component="BM25 Search",
            success=True,
            message=f"Found {len(results)} results for '{query}'",
            data={"query": query, "results": formatted_results},
            duration_ms=duration,
        )

    # -------------------------------------------------------------------------
    # Workflow Templates Demo
    # -------------------------------------------------------------------------

    def demo_workflow_templates(self) -> DemoResult:
        """
        Demonstrate workflow template system.

        Returns:
            DemoResult with available templates and sample instantiation
        """
        start = datetime.now()

        manager = get_template_manager()
        templates = list_builtin_templates()

        template_info = []
        for name in templates:
            template = manager.get_template(name)
            if template:
                template_info.append({
                    "name": name,
                    "description": template.description,
                    "parameters": [
                        {"name": p.name, "required": p.required, "description": p.description}
                        for p in template.parameters
                    ],
                    "steps": [s.name for s in template.steps],
                })

        # Demonstrate instantiation
        sample_graph = manager.instantiate("bug_fix", {
            "bug_description": "TypeError in handler.py line 42",
            "affected_files": "src/handlers/",
        })

        duration = (datetime.now() - start).total_seconds() * 1000

        return DemoResult(
            component="Workflow Templates",
            success=True,
            message=f"Loaded {len(templates)} workflow templates",
            data={
                "templates": template_info,
                "sample_instantiation": {
                    "template": "bug_fix",
                    "nodes": len(sample_graph.nodes),
                    "edges": len(sample_graph.edges),
                },
            },
            duration_ms=duration,
        )

    # -------------------------------------------------------------------------
    # Resilience Patterns Demo
    # -------------------------------------------------------------------------

    async def demo_resilience_patterns(self) -> DemoResult:
        """
        Demonstrate resilience patterns.

        Returns:
            DemoResult with resilience capabilities
        """
        start = datetime.now()
        results = {}

        # 1. Retry pattern
        retry_count = 0

        @retry_async(RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            strategy=BackoffStrategy.EXPONENTIAL,
        ))
        async def flaky_operation():
            nonlocal retry_count
            retry_count += 1
            if retry_count < 2:
                raise ConnectionError("Simulated failure")
            return "Success after retry"

        try:
            retry_result = await flaky_operation()
            results["retry"] = {
                "success": True,
                "attempts": retry_count,
                "result": retry_result,
            }
        except Exception as e:
            results["retry"] = {
                "success": False,
                "attempts": retry_count,
                "error": str(e),
            }

        # 2. Circuit breaker
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
        )

        results["circuit_breaker"] = {
            "state": breaker.state,
            "failure_count": breaker._failures,
            "is_allowed": breaker.is_allowed(),
        }

        # 3. Fallback chain
        def primary():
            raise ConnectionError("Primary down")

        def fallback():
            return "Fallback response"

        chain = FallbackChain(operations=[primary, fallback])
        fallback_result = chain.execute()

        results["fallback_chain"] = {
            "result": fallback_result,
            "used_fallback": True,
        }

        duration = (datetime.now() - start).total_seconds() * 1000

        return DemoResult(
            component="Resilience Patterns",
            success=True,
            message="Demonstrated retry, circuit breaker, and fallback patterns",
            data=results,
            duration_ms=duration,
        )

    # -------------------------------------------------------------------------
    # Caching Demo
    # -------------------------------------------------------------------------

    def demo_caching(self) -> DemoResult:
        """
        Demonstrate LLM response caching.

        Returns:
            DemoResult with caching capabilities
        """
        start = datetime.now()

        # Test cache operations
        test_query = "What is Python?"
        test_response = "Python is a high-level programming language."

        # Check miss
        cached = self.cache.get(test_query)
        miss = cached is None

        # Store
        self.cache.set(test_query, test_response)

        # Check hit
        cached = self.cache.get(test_query)
        hit = cached == test_response

        # Stats
        stats = self.cache.stats

        duration = (datetime.now() - start).total_seconds() * 1000

        return DemoResult(
            component="LLM Caching",
            success=True,
            message="Cache hit/miss working correctly",
            data={
                "initial_miss": miss,
                "subsequent_hit": hit,
                "stats": stats,
            },
            duration_ms=duration,
        )

    # -------------------------------------------------------------------------
    # Tool Registry Demo
    # -------------------------------------------------------------------------

    def demo_tool_registry(self) -> DemoResult:
        """
        Demonstrate tool registry.

        Returns:
            DemoResult with available tools
        """
        start = datetime.now()

        registry = get_tool_registry()

        # Get all tools by category
        tools_by_category: Dict[str, List[Dict[str, str]]] = {}
        for tool in registry.tools.values():
            cat_name = tool.category.value
            if cat_name not in tools_by_category:
                tools_by_category[cat_name] = []
            tools_by_category[cat_name].append({
                "name": tool.name,
                "description": tool.description[:100] + "..." if len(tool.description) > 100 else tool.description,
            })

        duration = (datetime.now() - start).total_seconds() * 1000

        return DemoResult(
            component="Tool Registry",
            success=True,
            message=f"Registered {len(registry.tools)} tools",
            data={
                "total_tools": len(registry.tools),
                "categories": list(tools_by_category.keys()),
                "by_category": tools_by_category,
            },
            duration_ms=duration,
        )

    # -------------------------------------------------------------------------
    # Agent Graph Demo
    # -------------------------------------------------------------------------

    async def demo_agent_graph(self) -> DemoResult:
        """
        Demonstrate agent graph execution with streaming.

        Returns:
            DemoResult with execution trace
        """
        start = datetime.now()

        # Create a simple test graph
        graph = AgentGraph(name="demo_workflow")

        # Add nodes
        graph.add_node(AgentNode(
            id="analyze",
            agent_type="code",
            name="Code Analysis",
            tools=["read_file", "grep_search"],
            config={"task": "Analyze the codebase structure"},
        ))
        graph.add_node(AgentNode(
            id="report",
            agent_type="doc",
            name="Documentation",
            tools=["read_file", "write_file"],
            config={"task": "Generate analysis report"},
        ))

        # Add edge
        graph.add_edge(AgentEdge(source_id="analyze", target_id="report"))

        # Validate
        is_valid = graph.validate()
        topo_order = graph.topological_sort() if is_valid else []

        duration = (datetime.now() - start).total_seconds() * 1000

        return DemoResult(
            component="Agent Graph",
            success=True,
            message=f"Created graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges",
            data={
                "name": graph.name,
                "nodes": [n.id for n in graph.nodes.values()],
                "edges": [(e.source_id, e.target_id) for e in graph.edges],
                "valid": is_valid,
                "execution_order": topo_order,
            },
            duration_ms=duration,
        )

    # -------------------------------------------------------------------------
    # Full Integration Demo
    # -------------------------------------------------------------------------

    async def run_full_demo(self) -> Dict[str, Any]:
        """
        Run complete RAGIX v0.7 demonstration.

        This method exercises all major v0.7 features and returns
        a comprehensive report suitable for Claude to analyze.

        Returns:
            Complete demonstration results
        """
        print("=" * 70)
        print("RAGIX v0.7 - Complete Claude Integration Demo")
        print("=" * 70)
        print()

        results = []

        # 1. System Health
        print("1. Checking system health...")
        health = self.get_system_health()
        results.append({
            "component": "System Health",
            "status": health["overall_status"],
            "data": health,
        })
        print(f"   Status: {health['overall_status']}")

        # 2. BM25 Search
        print("\n2. Running BM25 search demo...")
        bm25_result = self.demo_bm25_search()
        results.append({
            "component": bm25_result.component,
            "status": "success" if bm25_result.success else "failed",
            "data": bm25_result.data,
            "duration_ms": bm25_result.duration_ms,
        })
        print(f"   {bm25_result.message} ({bm25_result.duration_ms:.2f}ms)")

        # 3. Workflow Templates
        print("\n3. Loading workflow templates...")
        templates_result = self.demo_workflow_templates()
        results.append({
            "component": templates_result.component,
            "status": "success" if templates_result.success else "failed",
            "data": templates_result.data,
            "duration_ms": templates_result.duration_ms,
        })
        print(f"   {templates_result.message} ({templates_result.duration_ms:.2f}ms)")

        # 4. Resilience Patterns
        print("\n4. Testing resilience patterns...")
        resilience_result = await self.demo_resilience_patterns()
        results.append({
            "component": resilience_result.component,
            "status": "success" if resilience_result.success else "failed",
            "data": resilience_result.data,
            "duration_ms": resilience_result.duration_ms,
        })
        print(f"   {resilience_result.message} ({resilience_result.duration_ms:.2f}ms)")

        # 5. Caching
        print("\n5. Testing LLM caching...")
        cache_result = self.demo_caching()
        results.append({
            "component": cache_result.component,
            "status": "success" if cache_result.success else "failed",
            "data": cache_result.data,
            "duration_ms": cache_result.duration_ms,
        })
        print(f"   {cache_result.message} ({cache_result.duration_ms:.2f}ms)")

        # 6. Tool Registry
        print("\n6. Inspecting tool registry...")
        tools_result = self.demo_tool_registry()
        results.append({
            "component": tools_result.component,
            "status": "success" if tools_result.success else "failed",
            "data": tools_result.data,
            "duration_ms": tools_result.duration_ms,
        })
        print(f"   {tools_result.message} ({tools_result.duration_ms:.2f}ms)")

        # 7. Agent Graph
        print("\n7. Building agent graph...")
        graph_result = await self.demo_agent_graph()
        results.append({
            "component": graph_result.component,
            "status": "success" if graph_result.success else "failed",
            "data": graph_result.data,
            "duration_ms": graph_result.duration_ms,
        })
        print(f"   {graph_result.message} ({graph_result.duration_ms:.2f}ms)")

        # Summary
        print("\n" + "=" * 70)
        print("RAGIX v0.7 Demo Complete!")
        print("=" * 70)

        total_duration = sum(r.get("duration_ms", 0) for r in results)
        success_count = sum(1 for r in results if r["status"] == "success")

        summary = {
            "ragix_version": "0.7.0",
            "demo_timestamp": datetime.now().isoformat(),
            "model": self.model,
            "sandbox": str(self.sandbox_root),
            "profile": self.profile,
            "total_components": len(results),
            "successful": success_count,
            "total_duration_ms": total_duration,
            "results": results,
        }

        print(f"\nComponents tested: {success_count}/{len(results)}")
        print(f"Total duration: {total_duration:.2f}ms")

        return summary


# -----------------------------------------------------------------------------
# MCP-Compatible Functions (can be exposed via MCP server)
# -----------------------------------------------------------------------------

def ragix_demo_health() -> Dict[str, Any]:
    """
    MCP Tool: Get RAGIX system health.

    Returns comprehensive health status for Claude to analyze.
    """
    demo = RAGIXClaudeDemo()
    return demo.get_system_health()


def ragix_demo_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    MCP Tool: Search codebase using BM25.

    Args:
        query: Search query (keywords)
        top_k: Maximum results to return

    Returns:
        Search results with relevance scores
    """
    demo = RAGIXClaudeDemo()
    result = demo.demo_bm25_search(query)
    return result.data if result.data else {}


def ragix_demo_templates() -> Dict[str, Any]:
    """
    MCP Tool: List available workflow templates.

    Returns:
        All templates with parameters and steps
    """
    demo = RAGIXClaudeDemo()
    result = demo.demo_workflow_templates()
    return result.data if result.data else {}


def ragix_demo_tools() -> Dict[str, Any]:
    """
    MCP Tool: List available RAGIX tools.

    Returns:
        All registered tools by category
    """
    demo = RAGIXClaudeDemo()
    result = demo.demo_tool_registry()
    return result.data if result.data else {}


async def ragix_demo_full() -> Dict[str, Any]:
    """
    MCP Tool: Run complete RAGIX demonstration.

    This exercises all v0.7 features and returns a comprehensive
    report for Claude to analyze.

    Returns:
        Complete demo results
    """
    demo = RAGIXClaudeDemo()
    return await demo.run_full_demo()


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

async def main():
    """Run the Claude demo from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAGIX v0.7 Claude Integration Demo"
    )
    parser.add_argument(
        "--component",
        choices=["health", "search", "templates", "tools", "full"],
        default="full",
        help="Which component to demonstrate",
    )
    parser.add_argument(
        "--query",
        default="error handling",
        help="Search query (for search component)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    demo = RAGIXClaudeDemo()

    if args.component == "health":
        result = demo.get_system_health()
    elif args.component == "search":
        result = demo.demo_bm25_search(args.query)
        result = {"result": result.data, "message": result.message}
    elif args.component == "templates":
        result = demo.demo_workflow_templates()
        result = {"result": result.data, "message": result.message}
    elif args.component == "tools":
        result = demo.demo_tool_registry()
        result = {"result": result.data, "message": result.message}
    else:  # full
        result = await demo.run_full_demo()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif args.component != "full":
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
