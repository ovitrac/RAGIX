"""
RAGIX v0.30 Reflective Reasoning Graph

A graph-based reasoning system with:
- Task complexity classification (BYPASS/SIMPLE/MODERATE/COMPLEX)
- Structured planning with confidence tracking
- Reflection with read-only tool access
- Hybrid experience corpus for learning
- Graceful degradation on failures

Usage:
    from ragix_core.reasoning_v30 import (
        ReasoningGraph,
        ReasoningState,
        TaskComplexity,
        ClassifyNode,
        PlanNode,
        ExecuteNode,
        ReflectNode,
        VerifyNode,
        RespondNode,
        HybridExperienceCorpus,
        load_reasoning_config,
    )

    # Build graph
    graph = (GraphBuilder()
        .add_node(ClassifyNode(classify_fn))
        .add_node(DirectExecNode(llm_answer_fn))
        .add_node(PlanNode(generate_plan_fn, parse_plan_fn))
        .add_node(ExecuteNode(execute_step_fn))
        .add_node(ReflectNode(llm_reflect_fn, corpus, shell_fn))
        .add_node(VerifyNode(llm_verify_fn))
        .add_node(RespondNode())
        .build())

    # Run
    state = ReasoningState(goal="Find the largest Python file", session_id="test")
    final_state = graph.run(state)
    print(final_state.final_answer)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

__version__ = "0.30.0"
__author__ = "Olivier Vitrac, PhD, HDR"
__email__ = "olivier.vitrac@adservio.fr"

# Types
from .types import (
    TaskComplexity,
    StepStatus,
    ToolCall,
    ToolResult,
    PlanStep,
    Plan,
    ReflectionAttempt,
    ReasoningState,
    ReasoningEvent,
)

# Graph
from .graph import (
    BaseNode,
    ReasoningGraph,
    GraphBuilder,
    EndNode,
)

# Nodes
from .nodes import (
    ClassifyNode,
    DirectExecNode,
    PlanNode,
    ExecuteNode,
    ReflectNode,
    VerifyNode,
    RespondNode,
)

# Experience
from .experience import (
    ExperienceCorpus,
    HybridExperienceCorpus,
    SessionTraceWriter,
)

# Config
from .config import (
    ReasoningConfig,
    MaxReflectionsConfig,
    GraphConfig,
    ReflectConfig,
    ExperienceConfig,
    TracesConfig,
    AgentProfile,
    get_default_config,
    load_reasoning_config,
)

# Prompts
from .prompts import (
    CLASSIFY_PROMPT,
    PLAN_PROMPT,
    REFLECT_PROMPT,
    VERIFY_PROMPT,
    DIRECT_EXEC_PROMPT,
    render_prompt,
    render_classify_prompt,
    render_plan_prompt,
    render_reflect_prompt,
    render_verify_prompt,
    render_direct_exec_prompt,
    parse_complexity,
    extract_json_from_response,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__email__",
    # Types
    "TaskComplexity",
    "StepStatus",
    "ToolCall",
    "ToolResult",
    "PlanStep",
    "Plan",
    "ReflectionAttempt",
    "ReasoningState",
    "ReasoningEvent",
    # Graph
    "BaseNode",
    "ReasoningGraph",
    "GraphBuilder",
    "EndNode",
    # Nodes
    "ClassifyNode",
    "DirectExecNode",
    "PlanNode",
    "ExecuteNode",
    "ReflectNode",
    "VerifyNode",
    "RespondNode",
    # Experience
    "ExperienceCorpus",
    "HybridExperienceCorpus",
    "SessionTraceWriter",
    # Config
    "ReasoningConfig",
    "MaxReflectionsConfig",
    "GraphConfig",
    "ReflectConfig",
    "ExperienceConfig",
    "TracesConfig",
    "AgentProfile",
    "get_default_config",
    "load_reasoning_config",
    # Prompts
    "CLASSIFY_PROMPT",
    "PLAN_PROMPT",
    "REFLECT_PROMPT",
    "VERIFY_PROMPT",
    "DIRECT_EXEC_PROMPT",
    "render_prompt",
    "render_classify_prompt",
    "render_plan_prompt",
    "render_reflect_prompt",
    "render_verify_prompt",
    "render_direct_exec_prompt",
    "parse_complexity",
    "extract_json_from_response",
]
