"""
RAGIX v0.30 Reasoning Nodes

Node implementations for the Reflective Reasoning Graph:
- ClassifyNode: Route by task complexity (BYPASS/SIMPLE/MODERATE/COMPLEX)
- DirectExecNode: Handle BYPASS and SIMPLE tasks
- PlanNode: Generate structured execution plans
- ExecuteNode: Execute plan steps with reflection routing
- ReflectNode: Diagnose failures and generate revised plans
- VerifyNode: Validate complex task results
- RespondNode: Format final response and emit events

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

from datetime import datetime
from typing import Callable, Dict, Tuple, Optional, List, Any
import logging
import re

from .types import (
    ReasoningState,
    ReasoningEvent,
    TaskComplexity,
    StepStatus,
    Plan,
    PlanStep,
    ToolCall,
    ToolResult,
    ReflectionAttempt,
)
from .graph import BaseNode
from .experience import HybridExperienceCorpus

logger = logging.getLogger(__name__)


# =============================================================================
# CLASSIFY Node
# =============================================================================

class ClassifyNode(BaseNode):
    """
    Classify task complexity to determine routing.

    Routes:
    - BYPASS -> DIRECT_EXEC (no tools, no plan)
    - SIMPLE -> DIRECT_EXEC (simple tool use without formal plan)
    - MODERATE/COMPLEX -> PLAN (structured planning required)
    """
    name = "CLASSIFY"

    def __init__(self, classify_fn: Callable[[str], TaskComplexity]):
        """
        Initialize classify node.

        Args:
            classify_fn: Function that takes goal string and returns TaskComplexity
        """
        self.classify_fn = classify_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """Classify goal and route to appropriate node."""
        logger.debug(f"Classifying goal: {state.goal[:100]}...")

        try:
            state.complexity = self.classify_fn(state.goal)
        except Exception as e:
            logger.warning(f"Classification failed, defaulting to MODERATE: {e}")
            state.complexity = TaskComplexity.MODERATE

        logger.info(f"Task classified as: {state.complexity.value}")

        # Route based on complexity
        if state.complexity == TaskComplexity.BYPASS:
            state.stop_reason = "bypass"
            return state, "DIRECT_EXEC"

        if state.complexity == TaskComplexity.SIMPLE:
            return state, "DIRECT_EXEC"

        # MODERATE and COMPLEX go to planning
        return state, "PLAN"


# =============================================================================
# DIRECT_EXEC Node
# =============================================================================

class DirectExecNode(BaseNode):
    """
    Handle BYPASS and SIMPLE tasks without formal planning.

    For BYPASS: Generate direct conversational answer
    For SIMPLE: Optionally execute a single tool, then respond
    """
    name = "DIRECT_EXEC"

    def __init__(
        self,
        llm_answer_fn: Callable[[str, TaskComplexity], Dict[str, Any]],
        simple_tool_fn: Optional[Callable[[str], Optional[ToolResult]]] = None
    ):
        """
        Initialize direct exec node.

        Args:
            llm_answer_fn: Function(goal, complexity) -> {"answer": str, "confidence": float}
            simple_tool_fn: Optional function for SIMPLE tasks to execute a single tool
        """
        self.llm_answer_fn = llm_answer_fn
        self.simple_tool_fn = simple_tool_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """Execute direct response generation."""
        logger.debug(f"Direct exec for {state.complexity.value} task")

        # For SIMPLE tasks, execute tool and use output directly
        if state.complexity == TaskComplexity.SIMPLE and self.simple_tool_fn:
            try:
                result = self.simple_tool_fn(state.goal)
                if result and result.success and result.stdout:
                    # For SIMPLE tasks: use the tool output as the answer
                    state.final_answer = f"**Command executed successfully**\n\n```\n{result.stdout.strip()}\n```"
                    state.confidence = 0.95
                    state.tool_result = result  # Store for reference
                    if not state.stop_reason:
                        state.stop_reason = "success"
                    return state, "RESPOND"
                elif result and not result.success:
                    # Tool failed - let LLM handle
                    logger.warning(f"Simple tool failed: {result.stderr or result.error}")
            except Exception as e:
                logger.warning(f"Simple tool execution failed: {e}")

        # For BYPASS tasks or when SIMPLE tool failed: generate LLM answer
        try:
            response = self.llm_answer_fn(state.goal, state.complexity)
            state.final_answer = response.get("answer", "")
            state.confidence = response.get("confidence")
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            state.final_answer = f"I encountered an error processing your request: {e}"
            state.confidence = 0.0

        if not state.stop_reason:
            state.stop_reason = "success"

        return state, "RESPOND"


# =============================================================================
# PLAN Node
# =============================================================================

class PlanNode(BaseNode):
    """
    Generate structured execution plan for MODERATE/COMPLEX tasks.

    Includes reflection context from previous attempts to avoid
    repeating the same mistakes.
    """
    name = "PLAN"

    def __init__(
        self,
        generate_plan_fn: Callable[[str, str], Dict[str, Any]],
        parse_plan_fn: Callable[[Dict[str, Any], str], Plan]
    ):
        """
        Initialize plan node.

        Args:
            generate_plan_fn: Function(goal, reflection_context) -> raw plan dict
            parse_plan_fn: Function(raw_dict, goal) -> Plan object
        """
        self.generate_plan_fn = generate_plan_fn
        self.parse_plan_fn = parse_plan_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """Generate execution plan."""
        logger.debug(f"Generating plan for: {state.goal[:100]}...")

        # Build reflection context from previous attempts
        reflection_context = ""
        if state.reflection_attempts:
            reflection_context = "\n\n## Previous attempts (avoid these mistakes):\n"
            for i, att in enumerate(state.reflection_attempts, 1):
                reflection_context += f"\n### Attempt {i}\n"
                reflection_context += f"- Failed step: {att.failed_step}\n"
                reflection_context += f"- Error: {att.error[:200]}\n"
                reflection_context += f"- Diagnosis: {att.diagnosis}\n"

        try:
            raw_plan = self.generate_plan_fn(state.goal, reflection_context)
            state.plan = self.parse_plan_fn(raw_plan, state.goal)
            state.current_step_index = 0

            # Transfer confidence from plan to state
            if state.plan and state.plan.confidence is not None:
                state.confidence = state.plan.confidence

            logger.info(f"Generated plan with {len(state.plan.steps) if state.plan else 0} steps")

        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            state.last_error = f"Failed to generate plan: {e}"
            state.stop_reason = "plan_generation_failed"
            return state, "RESPOND"

        return state, "EXECUTE"


# =============================================================================
# EXECUTE Node
# =============================================================================

class ExecuteNode(BaseNode):
    """
    Execute plan steps one at a time.

    On failure:
    - If reflection budget remains: route to REFLECT
    - Otherwise: route to RESPOND with graceful degradation

    On completion:
    - COMPLEX tasks: route to VERIFY
    - Others: route to RESPOND
    """
    name = "EXECUTE"

    # Default reflection budgets by complexity
    DEFAULT_MAX_REFLECTIONS = {
        TaskComplexity.BYPASS: 0,
        TaskComplexity.SIMPLE: 0,
        TaskComplexity.MODERATE: 1,
        TaskComplexity.COMPLEX: 3,
    }

    def __init__(
        self,
        execute_step_fn: Callable[[PlanStep, ReasoningState], PlanStep],
        max_reflections_by_complexity: Optional[Dict[TaskComplexity, int]] = None,
        emit_event_fn: Optional[Callable[[ReasoningEvent], None]] = None
    ):
        """
        Initialize execute node.

        Args:
            execute_step_fn: Function(step, state) -> updated step with result
            max_reflections_by_complexity: Override default reflection budgets
            emit_event_fn: Optional callback for experience corpus
        """
        self.execute_step_fn = execute_step_fn
        self.max_reflections = max_reflections_by_complexity or self.DEFAULT_MAX_REFLECTIONS
        self.emit_event_fn = emit_event_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """Execute current plan step."""
        # Check for valid plan
        if state.plan is None:
            logger.warning("No plan available for execution")
            state.stop_reason = "no_plan"
            return state, "RESPOND"

        # Get current step
        step = state.plan.get_current_step(state.current_step_index)
        if step is None:
            # Plan complete - check if verification needed
            if state.complexity == TaskComplexity.COMPLEX:
                return state, "VERIFY"
            state.stop_reason = "success"
            return state, "RESPOND"

        logger.debug(f"Executing step {step.num}: {step.description[:80]}...")

        # Execute the step
        try:
            step = self.execute_step_fn(step, state)
        except Exception as e:
            logger.error(f"Step execution raised exception: {e}")
            step.status = StepStatus.FAILED
            step.result = ToolResult(
                tool=step.tool_call.tool if step.tool_call else "unknown",
                returncode=-1,
                error=str(e)
            )
            state.last_error = f"Step {step.num} exception: {e}"

        # Emit execution event
        if self.emit_event_fn:
            self._emit_execution_event(step, state)

        # Handle failure
        if step.status == StepStatus.FAILED:
            state.last_error = state.last_error or f"Step {step.num} failed"

            max_refl = self.max_reflections.get(state.complexity, 0)
            if state.reflection_count < max_refl:
                logger.info(f"Step failed, routing to REFLECT ({state.reflection_count}/{max_refl})")
                return state, "REFLECT"
            else:
                logger.warning(f"Step failed, reflection budget exhausted")
                state.stop_reason = "max_reflections"
                return state, "RESPOND"

        # Step succeeded - advance to next
        state.current_step_index += 1
        logger.debug(f"Step {step.num} succeeded, advancing to step {state.current_step_index + 1}")

        # Check if plan complete
        if state.plan.is_complete():
            if state.complexity == TaskComplexity.COMPLEX:
                return state, "VERIFY"
            state.stop_reason = "success"
            return state, "RESPOND"

        # Continue execution
        return state, "EXECUTE"

    def _emit_execution_event(self, step: PlanStep, state: ReasoningState) -> None:
        """Emit execution event to experience corpus."""
        if not self.emit_event_fn:
            return

        event = ReasoningEvent.create_now(
            session_id=state.session_id,
            event_type="execution",
            goal=state.goal,
            step_num=step.num,
            step_description=step.description,
            tool=step.tool_call.tool if step.tool_call else None,
            tool_input=str(step.tool_call.args) if step.tool_call else None,
            outcome_status="success" if step.status == StepStatus.SUCCESS else "failure",
            stdout=step.result.stdout if step.result else None,
            stderr=step.result.stderr if step.result else None,
            returncode=step.result.returncode if step.result else None,
            error=state.last_error,
        )
        self.emit_event_fn(event)


# =============================================================================
# REFLECT Node
# =============================================================================

class ReflectNode(BaseNode):
    """
    Reflection node for diagnosing failures and generating revised plans.

    Key features:
    - Uses read-only tools to gather context
    - Queries experience corpus for similar past failures
    - Generates diagnosis (max 3 bullets for stability)
    - Produces new plan summary for PlanNode
    """
    name = "REFLECT"

    # Read-only tools allowed during reflection
    ALLOWED_TOOLS = ["ls", "find", "grep", "head", "tail", "wc", "cat", "pwd", "file"]

    def __init__(
        self,
        llm_reflect_fn: Callable[[Dict[str, Any]], Dict[str, str]],
        experience_corpus: HybridExperienceCorpus,
        shell_executor: Callable[[str], ToolResult],
        emit_event_fn: Optional[Callable[[ReasoningEvent], None]] = None
    ):
        """
        Initialize reflect node.

        Args:
            llm_reflect_fn: Function(payload) -> {"diagnosis": str, "new_plan_summary": str}
            experience_corpus: HybridExperienceCorpus for retrieving past experiences
            shell_executor: Function(command) -> ToolResult for read-only tools
            emit_event_fn: Optional callback for experience corpus
        """
        self.llm_reflect_fn = llm_reflect_fn
        self.experience = experience_corpus
        self.shell = shell_executor
        self.emit_event_fn = emit_event_fn

    def _safe_shell(self, command: str) -> ToolResult:
        """Execute only read-only commands."""
        parts = (command or "").split()
        cmd = parts[0] if parts else ""

        if cmd not in self.ALLOWED_TOOLS:
            return ToolResult(
                tool=cmd,
                returncode=1,
                stderr=f"[BLOCKED: {cmd} not in allowed read-only tools]"
            )

        try:
            return self.shell(command)
        except Exception as e:
            return ToolResult(tool=cmd, returncode=-1, error=str(e))

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """Diagnose failure and prepare for replanning."""
        logger.info(f"Reflecting on failure (attempt {state.reflection_count + 1})")

        if not state.plan:
            state.stop_reason = "no_plan"
            return state, "RESPOND"

        step = state.plan.get_current_step(state.current_step_index)

        # 1. Gather file context with read-only tools
        context_commands = [
            "pwd",
            "ls -la",
            "find . -maxdepth 2 -type f -name '*.py' | head -20",
        ]
        file_context_parts = []
        for cmd in context_commands:
            result = self._safe_shell(cmd)
            file_context_parts.append(f"$ {cmd}\n{result.stdout or result.stderr}")
        file_context = "\n\n".join(file_context_parts)

        # 2. Query experience corpus
        query = f"goal: {state.goal}\nerror: {state.last_error}"
        experience_context = self.experience.search(query, top_k=5)

        # 3. Build reflection payload
        payload = {
            "goal": state.goal,
            "failed_step_num": step.num if step else None,
            "failed_step_description": step.description if step else None,
            "error": state.last_error,
            "file_context": file_context[:2000],  # Truncate for context window
            "experience_context": experience_context,
            "previous_attempts": [
                {
                    "failed_step": a.failed_step,
                    "error": a.error[:200],
                    "diagnosis": a.diagnosis,
                    "new_plan_summary": a.new_plan_summary,
                }
                for a in state.reflection_attempts
            ],
        }

        # 4. Generate reflection
        try:
            response = self.llm_reflect_fn(payload)
            diagnosis = response.get("diagnosis", "Unknown failure")
            new_plan_summary = response.get("new_plan_summary", "Retry with adjustments")
        except Exception as e:
            logger.error(f"Reflection LLM call failed: {e}")
            diagnosis = f"Reflection failed: {e}"
            new_plan_summary = "Unable to generate revised plan"

        # 5. Record reflection attempt
        state.add_reflection(ReflectionAttempt(
            timestamp=datetime.utcnow().isoformat(),
            failed_step=step.num if step else -1,
            error=state.last_error or "",
            diagnosis=diagnosis,
            new_plan_summary=new_plan_summary,
        ))

        # 6. Emit reflection event
        if self.emit_event_fn:
            event = ReasoningEvent.create_now(
                session_id=state.session_id,
                event_type="reflection",
                goal=state.goal,
                step_num=step.num if step else None,
                step_description=step.description if step else None,
                error=state.last_error,
                llm_critique=diagnosis,
                context_used=experience_context[:500],
            )
            self.emit_event_fn(event)

        # Clear error and route to PLAN for replanning
        state.last_error = None
        logger.info(f"Reflection complete, routing to PLAN")

        return state, "PLAN"


# =============================================================================
# VERIFY Node
# =============================================================================

class VerifyNode(BaseNode):
    """
    Verification node for COMPLEX tasks.

    Reviews executed plan and results to:
    - Check if goal was achieved
    - Refine the final answer
    - Adjust confidence based on outcomes
    """
    name = "VERIFY"

    def __init__(
        self,
        llm_verify_fn: Callable[[ReasoningState], Dict[str, Any]],
        emit_event_fn: Optional[Callable[[ReasoningEvent], None]] = None
    ):
        """
        Initialize verify node.

        Args:
            llm_verify_fn: Function(state) -> {"answer": str, "confidence": float}
            emit_event_fn: Optional callback for experience corpus
        """
        self.llm_verify_fn = llm_verify_fn
        self.emit_event_fn = emit_event_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """Verify task completion and refine answer."""
        logger.debug("Verifying task completion")

        try:
            response = self.llm_verify_fn(state)
            state.final_answer = response.get("answer", state.final_answer)

            if response.get("confidence") is not None:
                state.confidence = response["confidence"]

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            # Keep existing answer but note the verification failure
            if state.final_answer:
                state.final_answer += f"\n\n(Note: Verification step failed: {e})"

        if not state.stop_reason:
            state.stop_reason = "success"

        # Emit verification event
        if self.emit_event_fn:
            event = ReasoningEvent.create_now(
                session_id=state.session_id,
                event_type="verification",
                goal=state.goal,
                outcome_status="success" if state.stop_reason == "success" else "failure",
                meta={"confidence": state.confidence},
            )
            self.emit_event_fn(event)

        return state, "RESPOND"


# =============================================================================
# RESPOND Node
# =============================================================================

class RespondNode(BaseNode):
    """
    Final response node.

    Formats the final answer based on stop_reason:
    - success: Normal response with results
    - max_reflections: Graceful degradation with attempt summary
    - bypass: Direct conversational response
    - error states: Error message with context
    """
    name = "RESPOND"

    def __init__(
        self,
        emit_event_fn: Optional[Callable[[ReasoningEvent], None]] = None
    ):
        """
        Initialize respond node.

        Args:
            emit_event_fn: Optional callback for experience corpus
        """
        self.emit_event_fn = emit_event_fn

    def run(self, state: ReasoningState) -> Tuple[ReasoningState, str]:
        """Format final response and emit closing event."""
        logger.debug(f"Generating response (stop_reason: {state.stop_reason})")

        # Handle graceful degradation for max_reflections
        if state.stop_reason == "max_reflections":
            state.final_answer = self._build_degraded_response(state)

        # Handle missing final answer
        if not state.final_answer:
            state.final_answer = self._build_fallback_response(state)

        # Emit final event
        if self.emit_event_fn:
            event = ReasoningEvent.create_now(
                session_id=state.session_id,
                event_type="respond",
                goal=state.goal,
                outcome_status=state.stop_reason,
                meta={
                    "confidence": state.confidence,
                    "reflection_count": state.reflection_count,
                    "steps_completed": len(state.plan.get_completed_steps()) if state.plan else 0,
                },
            )
            self.emit_event_fn(event)

        return state, "END"

    def _build_degraded_response(self, state: ReasoningState) -> str:
        """Build response when reflection budget exhausted."""
        lines = ["## Partial Results\n"]
        lines.append(f"I attempted to complete your request but encountered persistent issues "
                     f"after {state.reflection_count} recovery attempts.\n")

        # Summarize attempts
        if state.reflection_attempts:
            lines.append("### What Was Tried\n")
            for i, att in enumerate(state.reflection_attempts, 1):
                lines.append(f"**Attempt {i}** (Step {att.failed_step})")
                lines.append(f"- Error: {att.error[:100]}...")
                lines.append(f"- Diagnosis: {att.diagnosis[:200]}...")
                lines.append("")

        # Add recommendation
        lines.append("### Recommendation\n")
        lines.append(self._generate_recommendation(state))

        # Add partial results if any
        if state.plan:
            completed = state.plan.get_completed_steps()
            if completed:
                lines.append("\n### Partial Output\n")
                for step in completed:
                    lines.append(f"- Step {step.num}: {step.description[:80]}")
                    if step.result and step.result.stdout:
                        lines.append(f"  Output: {step.result.stdout[:200]}...")

        return "\n".join(lines)

    def _generate_recommendation(self, state: ReasoningState) -> str:
        """Generate recommendation based on failure patterns."""
        errors = [a.error.lower() for a in state.reflection_attempts]

        if any("not found" in e or "no such file" in e for e in errors):
            return "Some required files or commands may not exist. Please verify the project structure."

        if any("permission" in e for e in errors):
            return "There may be permission issues. Try running with appropriate access."

        if any("syntax" in e or "parse" in e for e in errors):
            return "There appear to be syntax issues. Please check the code for errors."

        return "Consider breaking this task into smaller steps or providing more specific guidance."

    def _build_fallback_response(self, state: ReasoningState) -> str:
        """Build fallback response when no answer was generated."""
        if state.stop_reason == "no_plan":
            return "I was unable to generate a plan for this task. Please try rephrasing your request."

        if state.stop_reason and "exception" in state.stop_reason:
            return f"An error occurred while processing your request. Please try again."

        if state.last_error:
            return f"I encountered an issue: {state.last_error}"

        # If we have completed steps with output, build answer from them
        if state.plan:
            completed = state.plan.get_completed_steps()
            if completed:
                parts = []
                for step in completed:
                    if step.result and step.result.stdout:
                        parts.append(f"## Step {step.num}: {step.description}\n\n```\n{step.result.stdout.strip()}\n```")
                if parts:
                    return "\n\n".join(parts)

        return "I was unable to complete this task. Please try again with more specific instructions."
