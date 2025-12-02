"""
RAGIX Reasoning Graph - Graph-based Reasoning with Reflective Learning

This module implements the Reflective Reasoning Graph:
- ReasoningGraph: Orchestrates reasoning as a directed graph of nodes
- BaseNode: Abstract base class for graph nodes
- Node implementations: CLASSIFY, PLAN, EXECUTE, VERIFY, REFLECT, RESPOND

The graph enables self-correcting reasoning by:
1. Classifying task complexity
2. Planning multi-step execution
3. Executing with error capture
4. Reflecting on failures using past experience
5. Generating improved plans

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-02
"""

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

from .reasoning_types import (
    ReasoningState,
    ReasoningEvent,
    Plan,
    PlanStep,
    ReflectionAttempt,
    TaskComplexity,
    StepStatus,
    StopReason,
    get_max_reflections,
)
from .experience_corpus import HybridExperienceCorpus
from .agent_config import AgentRole, get_agent_persona

logger = logging.getLogger(__name__)


# =============================================================================
# Base Classes
# =============================================================================

class BaseNode(ABC):
    """
    Abstract base class for reasoning graph nodes.

    Each node transforms a ReasoningState and returns the next node to visit.
    """

    name: str = "BASE"

    @abstractmethod
    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """
        Execute node logic.

        Args:
            state: Current reasoning state

        Returns:
            Tuple of (updated_state, next_node_name)
        """
        pass


class ReasoningGraph:
    """
    Orchestrates reasoning as a directed graph of nodes.

    The graph flows from START through various nodes until reaching END.
    Each node can transition to any other node based on its logic.
    """

    def __init__(
        self,
        nodes: Dict[str, BaseNode],
        start: str = "CLASSIFY",
        end: str = "END",
        max_iterations: int = 50,
    ):
        """
        Initialize the reasoning graph.

        Args:
            nodes: Dictionary mapping node names to BaseNode instances
            start: Name of the starting node
            end: Name of the ending node
            max_iterations: Maximum iterations to prevent infinite loops
        """
        self.nodes = nodes
        self.start = start
        self.end = end
        self.max_iterations = max_iterations
        self.trace: List[Tuple[str, str]] = []  # (timestamp, node_name)

    def run(self, state: ReasoningState) -> ReasoningState:
        """
        Execute the graph until END or max_iterations.

        Args:
            state: Initial reasoning state

        Returns:
            Final reasoning state
        """
        current = self.start
        iterations = 0
        self.trace = []

        logger.info(f"Starting reasoning graph for goal: {state.goal[:50]}...")

        while current != self.end and iterations < self.max_iterations:
            timestamp = datetime.utcnow().isoformat()
            self.trace.append((timestamp, current))
            state.record_node_visit(current)

            if current not in self.nodes:
                logger.error(f"Unknown node: {current}")
                state.stop_reason = StopReason.ERROR
                state.last_error = f"Unknown node: {current}"
                break

            node = self.nodes[current]
            logger.debug(f"Executing node: {current}")

            try:
                state, next_node = node.run(state)
                current = next_node
            except Exception as e:
                logger.exception(f"Error in node {current}: {e}")
                state.stop_reason = StopReason.ERROR
                state.last_error = str(e)
                current = "RESPOND"  # Try to generate a response

            iterations += 1

        if iterations >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached")
            state.stop_reason = StopReason.MAX_ITERATIONS

        logger.info(
            f"Reasoning complete: {state.stop_reason.value if state.stop_reason else 'unknown'}"
        )
        return state

    def get_trace_summary(self) -> str:
        """Get a summary of the node execution trace."""
        if not self.trace:
            return "No trace available"

        nodes = [name for _, name in self.trace]
        return " -> ".join(nodes)


# =============================================================================
# Node Implementations
# =============================================================================

class ClassifyNode(BaseNode):
    """
    Classifies task complexity to determine reasoning strategy.

    Simple tasks → DIRECT_EXEC
    Moderate/Complex tasks → PLAN
    """

    name = "CLASSIFY"

    def __init__(self, classify_fn: Optional[Callable[[str], TaskComplexity]] = None):
        """
        Args:
            classify_fn: Optional custom classification function
        """
        self.classify_fn = classify_fn or self._default_classify

    def _default_classify(self, goal: str) -> TaskComplexity:
        """Default classification based on keywords."""
        goal_lower = goal.lower()

        # Conversational - always simple
        conversational = [
            "who are you", "what are you", "hello", "hi ", "help",
            "thanks", "thank you", "goodbye",
        ]
        if any(kw in goal_lower for kw in conversational):
            return TaskComplexity.SIMPLE

        # Complex indicators
        complex_keywords = [
            "and then", "after that", "multiple", "several",
            "refactor", "implement", "create", "fix bug",
            "largest", "smallest", "find and", "search and",
        ]
        if any(kw in goal_lower for kw in complex_keywords):
            return TaskComplexity.COMPLEX

        # Simple indicators
        simple_keywords = [
            "what is", "where is", "show me", "display",
            "read", "cat", "grep", "pwd", "ls",
        ]
        if any(kw in goal_lower for kw in simple_keywords):
            return TaskComplexity.SIMPLE

        return TaskComplexity.MODERATE

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        state.complexity = self.classify_fn(state.goal)
        state.max_reflections = get_max_reflections(state.complexity)

        logger.debug(f"Classified as {state.complexity.value}, max_reflections={state.max_reflections}")

        if state.complexity == TaskComplexity.SIMPLE:
            return state, "DIRECT_EXEC"
        return state, "PLAN"


class DirectExecNode(BaseNode):
    """
    Handles simple tasks without planning.

    Executes directly and transitions to RESPOND.
    """

    name = "DIRECT_EXEC"

    def __init__(self, execute_fn: Callable[[str], Tuple[Any, str]]):
        """
        Args:
            execute_fn: Function to execute a simple request
        """
        self.execute_fn = execute_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        try:
            result, message = self.execute_fn(state.goal)
            state.final_answer = message or str(result) or "Task completed"
            state.stop_reason = StopReason.SUCCESS
        except Exception as e:
            state.final_answer = f"Error executing task: {e}"
            state.stop_reason = StopReason.ERROR
            state.last_error = str(e)

        return state, "END"


class PlanNode(BaseNode):
    """
    Generates a structured plan for the task.

    Uses LLM to create a Plan with concrete steps.
    """

    name = "PLAN"

    def __init__(
        self,
        llm_generate: Callable[[str, List[Dict]], str],
        parse_plan_fn: Optional[Callable[[str, str], Plan]] = None,
    ):
        """
        Args:
            llm_generate: Function to call LLM: (system_prompt, messages) -> str
            parse_plan_fn: Optional custom plan parser
        """
        self.llm_generate = llm_generate
        self.parse_plan_fn = parse_plan_fn or self._default_parse_plan

    def _build_planning_prompt(self, state: ReasoningState) -> str:
        """Build the planning prompt with reflection context if available."""
        prompt_parts = [f"USER REQUEST:\n{state.goal}"]

        # Add reflection context if replanning
        if state.reflection_attempts:
            prompt_parts.append("\n\nPREVIOUS ATTEMPTS (learn from these):")
            for i, attempt in enumerate(state.reflection_attempts, 1):
                prompt_parts.append(
                    f"\nAttempt {i}:\n"
                    f"- Failed step: {attempt.failed_step_description}\n"
                    f"- Error: {attempt.error[:200]}...\n"
                    f"- Diagnosis: {attempt.diagnosis[:200]}..."
                )
            prompt_parts.append("\n\nGenerate an IMPROVED plan that avoids these failures.")

        prompt_parts.append("""

Generate a [PLAN] with concrete Unix commands. Each step should be specific.

CRITICAL RULES:
- ALWAYS use "*.md" with asterisk, NEVER ".md" without asterisk
- For "largest", use: sort -n | tail -1
- For "smallest", use: sort -n | head -1
- Use -exec for filenames with spaces

FORMAT:
[PLAN]
1. Objective: <what we're achieving>
2. Required: <data needed>
3. Steps:
   - 3.1 <command or action>
   - 3.2 <command or action>
4. Validation: <how to verify success>

Output ONLY the [PLAN] block.""")

        return "\n".join(prompt_parts)

    def _default_parse_plan(self, response: str, goal: str) -> Plan:
        """Parse LLM response into a Plan object."""
        objective = goal
        required_data = []
        steps = []
        validation = ""

        lines = response.split("\n")
        current_section = None
        step_num = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if "objective" in line.lower() or line.startswith("1."):
                current_section = "objective"
            elif "required" in line.lower() or line.startswith("2."):
                current_section = "required"
            elif "steps" in line.lower() or line.startswith("3."):
                current_section = "steps"
            elif "valid" in line.lower() or line.startswith("4."):
                current_section = "validation"
            elif line.startswith("-") or line.startswith("*"):
                content = line.lstrip("-* ").strip()
                if current_section == "objective":
                    objective = content or objective
                elif current_section == "required":
                    required_data.append(content)
                elif current_section == "steps":
                    step_match = re.match(r"^(\d+\.?\d*)\s*(.+)$", content)
                    if step_match:
                        content = step_match.group(2)
                    steps.append(PlanStep(num=step_num, description=content))
                    step_num += 1
                elif current_section == "validation":
                    validation = content
            elif current_section == "steps" and re.match(r"^\d+\.?\d*\s", line):
                step_match = re.match(r"^(\d+\.?\d*)\s*(.+)$", line)
                if step_match:
                    content = step_match.group(2).strip()
                    steps.append(PlanStep(num=step_num, description=content))
                    step_num += 1

        # Ensure at least one step
        if not steps:
            steps = [PlanStep(num=1, description=goal)]

        if not validation:
            validation = "Verify the result matches the request"

        return Plan(
            objective=objective,
            steps=steps,
            validation=validation,
            required_data=required_data or ["Current directory files"],
        )

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        planner_persona = get_agent_persona(AgentRole.PLANNER)
        planning_prompt = self._build_planning_prompt(state)

        messages = [{"role": "user", "content": planning_prompt}]
        response = self.llm_generate(planner_persona, messages)

        state.plan = self.parse_plan_fn(response, state.goal)
        state.current_step_index = 0

        logger.debug(f"Generated plan with {len(state.plan.steps)} steps")

        return state, "EXECUTE"


class ExecuteNode(BaseNode):
    """
    Executes plan steps one at a time.

    On success: continues to next step or VERIFY/RESPOND
    On failure: transitions to REFLECT if budget allows
    """

    name = "EXECUTE"

    def __init__(
        self,
        execute_step_fn: Callable[[PlanStep], PlanStep],
        emit_event_fn: Optional[Callable[[ReasoningEvent], None]] = None,
    ):
        """
        Args:
            execute_step_fn: Function to execute a single step
            emit_event_fn: Optional function to emit events to corpus
        """
        self.execute_step_fn = execute_step_fn
        self.emit_event_fn = emit_event_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        if state.plan is None:
            state.stop_reason = StopReason.NO_PLAN
            return state, "RESPOND"

        # Get current step
        step = state.plan.get_current_step(state.current_step_index)
        if step is None:
            # Plan complete
            if state.complexity == TaskComplexity.COMPLEX:
                return state, "VERIFY"
            return state, "RESPOND"

        # Execute the step
        logger.debug(f"Executing step {step.num}: {step.description[:50]}...")
        step = self.execute_step_fn(step)

        # Emit event
        if self.emit_event_fn:
            self.emit_event_fn(ReasoningEvent(
                timestamp=datetime.utcnow().isoformat(),
                session_id=state.session_id,
                event_type="execution",
                goal=state.goal,
                step_num=step.num,
                step_description=step.description,
                tool=step.tool,
                tool_input=str(step.args) if step.args else None,
                outcome_status="success" if step.status == StepStatus.SUCCESS else "failure",
                stdout=step.stdout,
                stderr=step.stderr,
                returncode=step.returncode,
                error=step.error,
            ))

        if step.status == StepStatus.FAILED:
            state.last_error = step.error or f"Step {step.num} failed"
            state.step_results.append(f"Step {step.num} FAILED: {step.error}")

            # Check if we can reflect
            if state.can_reflect():
                return state, "REFLECT"
            else:
                # Max reflections reached
                state.stop_reason = StopReason.MAX_REFLECTIONS
                return state, "RESPOND"

        # Success - record result and move on
        state.step_results.append(f"Step {step.num}: {step.result or 'completed'}")
        state.current_step_index += 1

        # Check if plan complete
        if state.plan.is_complete() or state.current_step_index >= len(state.plan.steps):
            if state.complexity == TaskComplexity.COMPLEX:
                return state, "VERIFY"
            return state, "RESPOND"

        # Continue execution
        return state, "EXECUTE"


class ReflectNode(BaseNode):
    """
    Reflects on failures and generates improved plans.

    Key capabilities:
    1. Use read-only tools to understand context
    2. Query experience corpus for similar past failures
    3. Generate diagnosis and improved plan
    """

    name = "REFLECT"

    # Read-only tools allowed during reflection
    ALLOWED_TOOLS = ["ls", "find", "grep", "head", "tail", "wc", "cat", "pwd", "tree"]

    def __init__(
        self,
        llm_generate: Callable[[str, List[Dict]], str],
        experience_corpus: HybridExperienceCorpus,
        shell_executor: Optional[Callable[[str], str]] = None,
        episodic_memory: Optional[Any] = None,
        emit_event_fn: Optional[Callable[[ReasoningEvent], None]] = None,
    ):
        """
        Args:
            llm_generate: LLM generation function
            experience_corpus: Hybrid experience corpus for RAG
            shell_executor: Optional shell execution for read-only commands
            episodic_memory: Optional episodic memory for session context
            emit_event_fn: Optional event emission function
        """
        self.llm_generate = llm_generate
        self.experience = experience_corpus
        self.shell = shell_executor
        self.episodic = episodic_memory
        self.emit_event_fn = emit_event_fn

    def _safe_shell(self, command: str) -> str:
        """Execute only read-only commands."""
        if not self.shell:
            return "[Shell not available]"

        cmd_parts = command.split()
        if not cmd_parts:
            return "[Empty command]"

        cmd_name = cmd_parts[0]
        if cmd_name not in self.ALLOWED_TOOLS:
            return f"[BLOCKED: {cmd_name} not in allowed tools: {self.ALLOWED_TOOLS}]"

        try:
            result = self.shell(command)
            # Truncate long outputs
            if len(result) > 1000:
                result = result[:1000] + "\n... (truncated)"
            return result
        except Exception as e:
            return f"[Error: {e}]"

    def _gather_context(self) -> str:
        """Gather file system context using read-only tools."""
        if not self.shell:
            return "[No shell available for context gathering]"

        context_commands = [
            ("Current directory", "pwd"),
            ("Directory listing", "ls -la"),
            ("Recent Python files", "find . -name '*.py' -type f | head -10"),
        ]

        parts = []
        for label, cmd in context_commands:
            result = self._safe_shell(cmd)
            parts.append(f"$ {cmd}\n{result}")

        return "\n\n".join(parts)

    def _extract_diagnosis(self, response: str) -> str:
        """Extract diagnosis from LLM response."""
        # Look for diagnosis section
        patterns = [
            r"(?:Diagnosis|What went wrong|Analysis)[:\s]*(.+?)(?=\n\n|\[PLAN\]|$)",
            r"(?:The error|The failure|The issue)[:\s]*(.+?)(?=\n\n|\[PLAN\]|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:500]

        # Fallback: first paragraph
        paragraphs = response.split("\n\n")
        if paragraphs:
            return paragraphs[0][:500]

        return "Unable to extract diagnosis"

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        failed_step = state.get_failed_step()
        if not failed_step:
            logger.warning("REFLECT called but no failed step found")
            return state, "RESPOND"

        logger.info(f"Reflecting on failure at step {failed_step.num}")

        # 1. Gather file system context
        file_context = self._gather_context()

        # 2. Query experience corpus
        query = f"goal: {state.goal}, error: {state.last_error}, step: {failed_step.description}"
        experience_context = self.experience.search(query, top_k=5)
        failure_lessons = self.experience.search_failures(query, top_k=3)

        # 3. Get episodic context if available
        episodic_context = ""
        if self.episodic and hasattr(self.episodic, "get_current_session_context"):
            episodic_context = self.episodic.get_current_session_context()

        # 4. Build reflection prompt
        reflection_prompt = f"""## Reflection Required

**Goal:** {state.goal}

**Failed Step {failed_step.num}:** {failed_step.description}

**Error:**
```
{state.last_error}
```

**Stderr:**
```
{failed_step.stderr or 'None'}
```

**Return code:** {failed_step.returncode}

---

**Current Directory Structure:**
```
{file_context}
```

---

**Similar Past Experiences:**
{experience_context}

**Lessons from Past Failures:**
{failure_lessons}

{f'**Session Context:**{chr(10)}{episodic_context}' if episodic_context else ''}

---

## Your Task

1. **Diagnose** what went wrong (be specific about the error cause)
2. **Explain** why the original approach failed
3. **Generate** a new [PLAN] that avoids this failure

Consider:
- Check if files/directories actually exist
- Verify command syntax for this OS
- Consider alternative approaches
- Learn from past similar failures

Output format:
1. Diagnosis paragraph
2. New [PLAN] block
"""

        # 5. Generate reflection response
        planner_persona = get_agent_persona(AgentRole.PLANNER)
        messages = [{"role": "user", "content": reflection_prompt}]
        response = self.llm_generate(planner_persona, messages)

        # 6. Extract diagnosis and new plan
        diagnosis = self._extract_diagnosis(response)

        # Parse new plan (reuse PlanNode's parser logic)
        new_plan = self._parse_plan(response, state.goal)

        # 7. Record reflection attempt
        attempt = ReflectionAttempt(
            timestamp=datetime.utcnow().isoformat(),
            failed_step_num=failed_step.num,
            failed_step_description=failed_step.description,
            error=state.last_error or "",
            diagnosis=diagnosis,
            new_plan_summary=f"{len(new_plan.steps)} steps: {new_plan.objective[:50]}...",
            context_used=experience_context[:500] if experience_context else None,
        )
        state.reflection_attempts.append(attempt)

        # 8. Update state
        state.plan = new_plan
        state.current_step_index = 0
        state.last_error = None
        state.reflection_count += 1

        # 9. Emit reflection event
        if self.emit_event_fn:
            self.emit_event_fn(ReasoningEvent(
                timestamp=datetime.utcnow().isoformat(),
                session_id=state.session_id,
                event_type="reflection",
                goal=state.goal,
                step_num=failed_step.num,
                step_description=failed_step.description,
                error=attempt.error,
                llm_critique=diagnosis,
                context_used=attempt.context_used,
                plan_steps_count=len(new_plan.steps),
            ))

        logger.info(f"Reflection {state.reflection_count}: generated new plan with {len(new_plan.steps)} steps")

        return state, "EXECUTE"

    def _parse_plan(self, response: str, goal: str) -> Plan:
        """Parse plan from response (simplified version)."""
        steps = []
        step_num = 1

        # Look for step patterns
        for line in response.split("\n"):
            line = line.strip()
            if re.match(r"^[-*]\s*\d+\.?\d*\s+", line) or re.match(r"^\d+\.?\d*\s+", line):
                # Extract step content
                content = re.sub(r"^[-*\s]*\d+\.?\d*\s*", "", line).strip()
                if content and len(content) > 5:
                    steps.append(PlanStep(num=step_num, description=content))
                    step_num += 1

        if not steps:
            steps = [PlanStep(num=1, description=goal)]

        return Plan(objective=goal, steps=steps)


class VerifyNode(BaseNode):
    """
    Verifies execution results against plan criteria.

    Only runs for COMPLEX tasks after all steps complete.
    """

    name = "VERIFY"

    def __init__(
        self,
        llm_generate: Callable[[str, List[Dict]], str],
    ):
        self.llm_generate = llm_generate

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        if not state.plan:
            return state, "RESPOND"

        # Build verification prompt
        results_text = "\n".join(state.step_results) if state.step_results else "No results recorded"

        verify_prompt = f"""Verify the execution results against the plan.

**Objective:** {state.plan.objective}

**Validation Criteria:** {state.plan.validation}

**Execution Results:**
{results_text}

Provide a brief [VERIFY] assessment:
- Status: PASS or FAIL
- Summary: One sentence explaining the outcome
- ANSWER: The direct answer to the user's question (if applicable)
"""

        verifier_persona = get_agent_persona(AgentRole.VERIFIER)
        messages = [{"role": "user", "content": verify_prompt}]
        response = self.llm_generate(verifier_persona, messages)

        # Parse verification
        is_pass = "PASS" in response.upper()

        # Add verification to results
        state.step_results.append(f"\n[VERIFY]\n{response}")

        if not is_pass and state.can_reflect():
            state.last_error = "Verification failed"
            return state, "REFLECT"

        return state, "RESPOND"


class RespondNode(BaseNode):
    """
    Generates the final response to the user.

    Handles both success and failure cases with appropriate formatting.
    """

    name = "RESPOND"

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        if state.stop_reason == StopReason.MAX_REFLECTIONS:
            state.final_answer = self._build_partial_response(state)
        elif state.stop_reason == StopReason.ERROR:
            state.final_answer = self._build_error_response(state)
        else:
            state.final_answer = self._build_success_response(state)
            state.stop_reason = StopReason.SUCCESS

        return state, "END"

    def _build_success_response(self, state: ReasoningState) -> str:
        """Build response for successful execution."""
        parts = []

        if state.plan:
            parts.append(f"📋 **{state.plan.objective}**\n")

            # Show step summary
            for step in state.plan.steps:
                icon = "✅" if step.status == StepStatus.SUCCESS else "❌"
                parts.append(f"  {icon} {step.description[:60]}...")

            parts.append("")

        # Add results
        if state.step_results:
            parts.append("**Results:**")
            for result in state.step_results:
                parts.append(result)

        return "\n".join(parts)

    def _build_partial_response(self, state: ReasoningState) -> str:
        """Build response when max reflections reached."""
        parts = [
            "## ⚠️ Partial Results",
            "",
            f"I attempted to complete your request but encountered persistent issues "
            f"after {state.reflection_count} recovery attempt(s).",
            "",
            "### What Was Tried",
        ]

        for i, attempt in enumerate(state.reflection_attempts, 1):
            parts.append(f"\n**Attempt {i}** (Step {attempt.failed_step_num})")
            parts.append(f"- Error: {attempt.error[:100]}...")
            parts.append(f"- Diagnosis: {attempt.diagnosis[:150]}...")

        parts.append("\n### Recommendation")
        parts.append(self._generate_recommendation(state))

        if state.step_results:
            parts.append("\n### Partial Output")
            for result in state.step_results[:5]:
                parts.append(result)

        return "\n".join(parts)

    def _build_error_response(self, state: ReasoningState) -> str:
        """Build response for error cases."""
        return f"## ❌ Error\n\nAn error occurred: {state.last_error}"

    def _generate_recommendation(self, state: ReasoningState) -> str:
        """Generate recommendation based on failure patterns."""
        if not state.reflection_attempts:
            return "Please try rephrasing your request or providing more details."

        errors = [a.error.lower() for a in state.reflection_attempts]

        if any("not found" in e or "no such" in e for e in errors):
            return "Some required files or directories may not exist. Please verify the project structure."

        if any("permission" in e for e in errors):
            return "There may be permission issues. Try running with appropriate access."

        if any("syntax" in e or "invalid" in e for e in errors):
            return "There may be command syntax issues. Consider checking the command format."

        return "Consider breaking this task into smaller steps or providing more specific guidance."


# =============================================================================
# Graph Factory
# =============================================================================

def create_reasoning_graph(
    llm_generate: Callable[[str, List[Dict]], str],
    execute_fn: Callable[[str], Tuple[Any, str]],
    execute_step_fn: Callable[[PlanStep], PlanStep],
    experience_corpus: HybridExperienceCorpus,
    shell_executor: Optional[Callable[[str], str]] = None,
    episodic_memory: Optional[Any] = None,
    emit_event_fn: Optional[Callable[[ReasoningEvent], None]] = None,
    classify_fn: Optional[Callable[[str], TaskComplexity]] = None,
) -> ReasoningGraph:
    """
    Factory function to create a fully configured ReasoningGraph.

    Args:
        llm_generate: LLM generation function
        execute_fn: Simple execution function for DIRECT_EXEC
        execute_step_fn: Step execution function for EXECUTE
        experience_corpus: Hybrid experience corpus
        shell_executor: Optional shell for REFLECT context gathering
        episodic_memory: Optional episodic memory
        emit_event_fn: Optional event emission function
        classify_fn: Optional custom classification function

    Returns:
        Configured ReasoningGraph instance
    """
    nodes: Dict[str, BaseNode] = {
        "CLASSIFY": ClassifyNode(classify_fn=classify_fn),
        "DIRECT_EXEC": DirectExecNode(execute_fn=execute_fn),
        "PLAN": PlanNode(llm_generate=llm_generate),
        "EXECUTE": ExecuteNode(
            execute_step_fn=execute_step_fn,
            emit_event_fn=emit_event_fn,
        ),
        "REFLECT": ReflectNode(
            llm_generate=llm_generate,
            experience_corpus=experience_corpus,
            shell_executor=shell_executor,
            episodic_memory=episodic_memory,
            emit_event_fn=emit_event_fn,
        ),
        "VERIFY": VerifyNode(llm_generate=llm_generate),
        "RESPOND": RespondNode(),
    }

    return ReasoningGraph(nodes=nodes)
