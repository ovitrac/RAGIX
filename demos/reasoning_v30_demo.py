#!/usr/bin/env python3
"""
RAGIX v0.30 Reflective Reasoning Graph — Comprehensive Demo

This demo showcases the full reasoning pipeline with real LLM calls:
1. Task classification (BYPASS/SIMPLE/MODERATE/COMPLEX)
2. Plan generation with confidence scoring
3. Step execution with Unix tools
4. Reflection on failures
5. Experience corpus learning

Requirements:
- Ollama running with mistral:7b-instruct or granite3.1-moe:3b
- RAGIX installed in development mode

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragix_core.reasoning_v30 import (
    # Types
    TaskComplexity,
    StepStatus,
    ToolCall,
    ToolResult,
    PlanStep,
    Plan,
    ReasoningState,
    ReasoningEvent,
    # Graph
    ReasoningGraph,
    GraphBuilder,
    EndNode,
    # Nodes
    ClassifyNode,
    DirectExecNode,
    PlanNode,
    ExecuteNode,
    ReflectNode,
    VerifyNode,
    RespondNode,
    # Experience
    HybridExperienceCorpus,
    SessionTraceWriter,
    # Config
    ReasoningConfig,
    AgentProfile,
    # Prompts
    render_classify_prompt,
    render_plan_prompt,
    render_reflect_prompt,
    render_verify_prompt,
    render_direct_exec_prompt,
    parse_complexity,
    extract_json_from_response,
)


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.ENDC}\n")


def print_section(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {text} ---{Colors.ENDC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_dim(text: str):
    print(f"{Colors.DIM}{text}{Colors.ENDC}")


# =============================================================================
# Ollama LLM Interface
# =============================================================================

class OllamaLLM:
    """Simple Ollama client for the demo."""

    def __init__(self, model: str = "mistral:7b-instruct", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self.api_url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate response from Ollama."""
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print_error(f"Ollama error: {e}")
            return ""

    def check_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return self.model in result.stdout
        except Exception:
            return False


# =============================================================================
# Shell Executor
# =============================================================================

class ShellExecutor:
    """Safe shell executor for the demo."""

    ALLOWED_COMMANDS = ["ls", "find", "grep", "head", "tail", "wc", "cat", "pwd", "file", "sort", "uniq"]
    MAX_OUTPUT = 5000

    def __init__(self, sandbox_root: Path):
        self.sandbox = sandbox_root

    def execute(self, command: str) -> ToolResult:
        """Execute a shell command safely."""
        start_time = time.time()

        # Parse command
        parts = command.split()
        if not parts:
            return ToolResult(tool="", returncode=1, stderr="Empty command")

        cmd = parts[0]

        # Security check
        if cmd not in self.ALLOWED_COMMANDS:
            return ToolResult(
                tool=cmd,
                returncode=1,
                stderr=f"Command '{cmd}' not allowed. Allowed: {self.ALLOWED_COMMANDS}"
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.sandbox)
            )

            stdout = result.stdout[:self.MAX_OUTPUT]
            stderr = result.stderr[:self.MAX_OUTPUT]

            duration_ms = (time.time() - start_time) * 1000

            return ToolResult(
                tool=cmd,
                returncode=result.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms
            )

        except subprocess.TimeoutExpired:
            return ToolResult(tool=cmd, returncode=-1, error="Command timed out")
        except Exception as e:
            return ToolResult(tool=cmd, returncode=-1, error=str(e))


# =============================================================================
# LLM Function Implementations
# =============================================================================

def create_classify_fn(llm: OllamaLLM):
    """Create classification function."""
    def classify(goal: str) -> TaskComplexity:
        prompt = render_classify_prompt(goal)
        response = llm.generate(prompt, max_tokens=50)

        complexity = parse_complexity(response)
        if complexity:
            return TaskComplexity(complexity)

        # Default fallback
        return TaskComplexity.MODERATE

    return classify


def create_direct_exec_fn(llm: OllamaLLM):
    """Create direct execution function for BYPASS/SIMPLE tasks."""
    def direct_exec(goal: str, complexity: TaskComplexity) -> Dict[str, Any]:
        prompt = render_direct_exec_prompt(goal)
        response = llm.generate(prompt, max_tokens=1500)

        result = extract_json_from_response(response)
        if result:
            return result

        # Fallback: use raw response
        return {"answer": response.strip(), "confidence": 0.6}

    return direct_exec


def create_simple_tool_fn(shell: ShellExecutor, llm: OllamaLLM):
    """Create function that infers and executes a single tool for SIMPLE tasks."""
    def simple_tool(goal: str) -> Optional[ToolResult]:
        # Ask LLM to generate a single shell command
        prompt = f"""You are a Unix shell assistant. The user wants to accomplish this task:

{goal}

Generate a single shell command that accomplishes this task.
Only use these commands: find, grep, ls, wc, head, tail, cat

IMPORTANT: Reply with ONLY the shell command, nothing else. No explanation.

Example:
Task: Count Python files in src/
Command: find src/ -name '*.py' | wc -l

Task: {goal}
Command:"""

        response = llm.generate(prompt, max_tokens=100)
        command = response.strip().split('\n')[0].strip()

        # Clean up common prefixes
        for prefix in ['Command:', '$', '> ', 'bash:', 'shell:']:
            if command.lower().startswith(prefix.lower()):
                command = command[len(prefix):].strip()

        if command:
            print_dim(f"  [SIMPLE] Executing: {command}")
            return shell.execute(command)

        return None

    return simple_tool


def create_plan_fn(llm: OllamaLLM):
    """Create plan generation function."""
    def generate_plan(goal: str, reflection_context: str) -> Dict[str, Any]:
        prompt = render_plan_prompt(goal, reflection_context)
        response = llm.generate(prompt, max_tokens=2000)

        result = extract_json_from_response(response)
        if result:
            return result

        # Fallback: create minimal plan
        return {
            "objective": goal,
            "steps": [{"num": 1, "description": "Execute task", "tool": "none", "args": {}}],
            "validation": "Task completed",
            "confidence": 0.5
        }

    return generate_plan


def create_parse_plan_fn():
    """Create plan parsing function."""
    def parse_plan(raw: Dict[str, Any], goal: str) -> Plan:
        steps = []
        for s in raw.get("steps", []):
            tool_call = None
            if s.get("tool") and s["tool"] != "none":
                tool_call = ToolCall(tool=s["tool"], args=s.get("args", {}))

            steps.append(PlanStep(
                num=s.get("num", len(steps) + 1),
                description=s.get("description", ""),
                tool_call=tool_call,
            ))

        return Plan(
            objective=raw.get("objective", goal),
            steps=steps,
            validation=raw.get("validation", ""),
            confidence=raw.get("confidence"),
        )

    return parse_plan


def create_execute_step_fn(shell: ShellExecutor):
    """Create step execution function."""
    def execute_step(step: PlanStep, state: ReasoningState) -> PlanStep:
        step.status = StepStatus.RUNNING

        if not step.tool_call:
            # No tool needed, mark as success
            step.status = StepStatus.SUCCESS
            step.result = ToolResult(tool="none", returncode=0, stdout="Step completed (no tool)")
            return step

        # Build command from tool_call
        tool = step.tool_call.tool
        args = step.tool_call.args

        # Construct command based on tool type
        if tool == "find":
            path = args.get("path", ".")
            pattern = args.get("pattern", "*")
            cmd = f"find {path} -name '{pattern}' -type f 2>/dev/null | head -20"
        elif tool == "grep":
            pattern = args.get("pattern", "")
            path = args.get("path", ".")
            cmd = f"grep -rn '{pattern}' {path} 2>/dev/null | head -20"
        elif tool == "wc":
            path = args.get("path", "")
            cmd = f"wc -l {path}"
        elif tool == "cat":
            path = args.get("path", "")
            cmd = f"cat {path} | head -50"
        elif tool == "head":
            path = args.get("path", "")
            n = args.get("n", 20)
            cmd = f"head -n {n} {path}"
        elif tool == "ls":
            path = args.get("path", ".")
            cmd = f"ls -la {path}"
        else:
            # Generic command
            cmd = f"{tool} {' '.join(str(v) for v in args.values())}"

        # Execute
        result = shell.execute(cmd)
        step.result = result

        if result.success:
            step.status = StepStatus.SUCCESS
        else:
            step.status = StepStatus.FAILED
            state.last_error = f"Command failed: {result.stderr or result.error}"

        return step

    return execute_step


def create_reflect_fn(llm: OllamaLLM):
    """Create reflection function."""
    def reflect(payload: Dict[str, Any]) -> Dict[str, str]:
        prompt = render_reflect_prompt(
            goal=payload.get("goal", ""),
            failed_step_num=payload.get("failed_step_num", 0),
            failed_step_description=payload.get("failed_step_description", ""),
            error=payload.get("error", ""),
            file_context=payload.get("file_context", ""),
            experience_context=payload.get("experience_context", ""),
            previous_attempts=str(payload.get("previous_attempts", [])),
        )
        response = llm.generate(prompt, max_tokens=1500)

        result = extract_json_from_response(response)
        if result:
            return result

        return {
            "diagnosis": "Unable to parse reflection response",
            "new_plan_summary": "Retry with adjusted approach"
        }

    return reflect


def create_verify_fn(llm: OllamaLLM):
    """Create verification function."""
    def verify(state: ReasoningState) -> Dict[str, Any]:
        # Build summary of executed steps
        plan_summary = state.plan.objective if state.plan else "No plan"
        step_results = ""
        if state.plan:
            for step in state.plan.steps:
                status = step.status.value
                output = ""
                if step.result:
                    output = step.result.stdout[:200] if step.result.stdout else step.result.stderr[:200]
                step_results += f"\nStep {step.num} [{status}]: {step.description[:50]}\n  Output: {output[:100]}"

        prompt = render_verify_prompt(
            goal=state.goal,
            plan_summary=plan_summary,
            step_results=step_results,
            current_answer=state.final_answer or "No answer yet"
        )
        response = llm.generate(prompt, max_tokens=1500)

        result = extract_json_from_response(response)
        if result:
            return result

        # Build answer from step outputs
        answer_parts = []
        if state.plan:
            for step in state.plan.get_completed_steps():
                if step.result and step.result.stdout:
                    answer_parts.append(step.result.stdout.strip())

        return {
            "answer": "\n".join(answer_parts) if answer_parts else response.strip(),
            "confidence": 0.7
        }

    return verify


# =============================================================================
# Demo Runner
# =============================================================================

class ReasoningDemo:
    """Demo orchestrator."""

    def __init__(self, model: str = "mistral:7b-instruct", sandbox: Optional[Path] = None):
        self.model = model
        self.sandbox = sandbox or Path.cwd()
        self.llm = OllamaLLM(model=model)
        self.shell = ShellExecutor(self.sandbox)
        self.events: list = []

        # Create experience corpus in temp location for demo
        self.corpus = HybridExperienceCorpus(
            global_root=Path("/tmp/ragix_demo/global"),
            project_root=Path("/tmp/ragix_demo/project"),
        )

    def emit_event(self, event: ReasoningEvent):
        """Collect events for analysis."""
        self.events.append(event)
        self.corpus.append(event)

    def build_graph(self) -> ReasoningGraph:
        """Build the reasoning graph with all nodes."""
        return (GraphBuilder()
            .add_node(ClassifyNode(create_classify_fn(self.llm)))
            .add_node(DirectExecNode(
                create_direct_exec_fn(self.llm),
                simple_tool_fn=create_simple_tool_fn(self.shell, self.llm)
            ))
            .add_node(PlanNode(create_plan_fn(self.llm), create_parse_plan_fn()))
            .add_node(ExecuteNode(
                create_execute_step_fn(self.shell),
                emit_event_fn=self.emit_event
            ))
            .add_node(ReflectNode(
                create_reflect_fn(self.llm),
                self.corpus,
                lambda cmd: self.shell.execute(cmd),
                emit_event_fn=self.emit_event
            ))
            .add_node(VerifyNode(create_verify_fn(self.llm), emit_event_fn=self.emit_event))
            .add_node(RespondNode(emit_event_fn=self.emit_event))
            .add_node(EndNode())
            .set_start("CLASSIFY")
            .set_end("END")
            .set_event_emitter(self.emit_event)
            .build())

    def run_task(self, goal: str, session_id: Optional[str] = None) -> ReasoningState:
        """Run a single reasoning task."""
        session_id = session_id or f"demo_{datetime.now().strftime('%H%M%S')}"
        self.events = []

        print_section(f"Task: {goal[:60]}...")
        print_info(f"Session: {session_id}")
        print_info(f"Model: {self.model}")

        # Create initial state
        state = ReasoningState(goal=goal, session_id=session_id)

        # Build and run graph
        graph = self.build_graph()

        print_section("Executing Reasoning Graph")
        start_time = time.time()

        final_state = graph.run(state, max_iterations=30)

        elapsed = time.time() - start_time

        # Print results
        print_section("Results")
        print(f"  Complexity: {Colors.CYAN}{final_state.complexity.value.upper()}{Colors.ENDC}")
        print(f"  Stop Reason: {Colors.GREEN if 'success' in str(final_state.stop_reason) else Colors.YELLOW}{final_state.stop_reason}{Colors.ENDC}")
        print(f"  Reflections: {final_state.reflection_count}")
        print(f"  Confidence: {final_state.confidence:.2f}" if final_state.confidence else "  Confidence: N/A")
        print(f"  Time: {elapsed:.2f}s")

        print_section("Graph Trace")
        print_dim(graph.get_trace_summary())

        if final_state.plan:
            print_section("Executed Plan")
            for step in final_state.plan.steps:
                status_color = Colors.GREEN if step.status == StepStatus.SUCCESS else Colors.RED
                print(f"  {status_color}[{step.status.value:^8}]{Colors.ENDC} Step {step.num}: {step.description[:50]}")

        print_section("Final Answer")
        if final_state.final_answer:
            # Truncate very long answers
            answer = final_state.final_answer
            if len(answer) > 1000:
                answer = answer[:1000] + "\n... (truncated)"
            print(answer)
        else:
            print_warning("No answer generated")

        return final_state


def main():
    """Run the comprehensive demo."""
    print_header("RAGIX v0.30 Reflective Reasoning Graph — Demo")

    # Check Ollama
    print_section("Environment Check")

    # Try mistral first, fallback to granite
    models_to_try = ["mistral:7b-instruct", "granite3.1-moe:3b"]
    selected_model = None

    for model in models_to_try:
        llm = OllamaLLM(model=model)
        if llm.check_available():
            selected_model = model
            print_success(f"Model available: {model}")
            break
        else:
            print_warning(f"Model not available: {model}")

    if not selected_model:
        print_error("No suitable model found. Please run: ollama pull mistral:7b-instruct")
        sys.exit(1)

    # Create demo runner
    sandbox = Path(__file__).parent.parent  # RAGIX root
    demo = ReasoningDemo(model=selected_model, sandbox=sandbox)

    print_info(f"Sandbox: {sandbox}")

    # ==========================================================================
    # Test Cases
    # ==========================================================================

    test_cases = [
        # BYPASS - Pure reasoning, no tools
        {
            "name": "BYPASS Test",
            "goal": "What is the difference between cyclomatic complexity and cognitive complexity in software metrics?",
            "expected_complexity": "bypass",
        },

        # SIMPLE - Single command
        {
            "name": "SIMPLE Test",
            "goal": "How many Python files are in the ragix_core directory?",
            "expected_complexity": "simple",
        },

        # MODERATE - Multi-step
        {
            "name": "MODERATE Test",
            "goal": "Find all Python files in ragix_core/reasoning_v30/ and count the total lines of code",
            "expected_complexity": "moderate",
        },

        # COMPLEX - Investigation
        {
            "name": "COMPLEX Test",
            "goal": "Analyze the structure of the reasoning_v30 module: list all classes defined, their methods, and identify the main entry points",
            "expected_complexity": "complex",
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print_header(f"Test {i}/{len(test_cases)}: {test['name']}")

        try:
            state = demo.run_task(test["goal"], session_id=f"test_{i}")

            # Record result
            results.append({
                "name": test["name"],
                "expected": test["expected_complexity"],
                "actual": state.complexity.value,
                "match": state.complexity.value == test["expected_complexity"],
                "success": state.stop_reason in ["success", "bypass"],
                "confidence": state.confidence,
                "reflections": state.reflection_count,
            })

        except Exception as e:
            print_error(f"Test failed with exception: {e}")
            results.append({
                "name": test["name"],
                "expected": test["expected_complexity"],
                "actual": "error",
                "match": False,
                "success": False,
                "error": str(e),
            })

        print("\n")
        time.sleep(1)  # Brief pause between tests

    # ==========================================================================
    # Summary
    # ==========================================================================

    print_header("Demo Summary")

    print_section("Test Results")
    for r in results:
        status = "✓" if r["success"] else "✗"
        match = "✓" if r.get("match") else "✗"
        color = Colors.GREEN if r["success"] else Colors.RED

        print(f"  {color}{status}{Colors.ENDC} {r['name']}")
        print(f"      Complexity: expected={r['expected']}, actual={r['actual']} [{match}]")
        if r.get("confidence"):
            print(f"      Confidence: {r['confidence']:.2f}")
        if r.get("reflections", 0) > 0:
            print(f"      Reflections: {r['reflections']}")

    # Statistics
    print_section("Statistics")
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    complexity_match = sum(1 for r in results if r.get("match"))

    print(f"  Tests Passed: {passed}/{total}")
    print(f"  Complexity Classification: {complexity_match}/{total} correct")
    print(f"  Events Logged: {len(demo.events)}")

    print_section("Experience Corpus")
    print(f"  Project events: {demo.corpus.project_corpus.get_event_count()}")
    print(f"  Global events: {demo.corpus.global_corpus.get_event_count()}")

    print("\n" + "="*70)
    print(f"{Colors.BOLD}Demo completed successfully!{Colors.ENDC}")
    print("="*70 + "\n")


def main_with_args():
    """Run demo with command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAGIX v0.30 Reflective Reasoning Graph Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reasoning_v30_demo.py --model mistral:7b-instruct
  python reasoning_v30_demo.py --model granite3.1-moe:3b --output results.json
  python reasoning_v30_demo.py --list-models
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Ollama model to use (e.g., mistral:7b-instruct, granite3.1-moe:3b)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for results (for benchmarking)"
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List available Ollama models and exit"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (for batch runs)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.3,
        help="LLM temperature (default: 0.3)"
    )

    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        print("Available Ollama models:")
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            print(result.stdout)
        except Exception as e:
            print(f"Error listing models: {e}")
        sys.exit(0)

    # Run main demo with specified model
    run_demo(
        model=args.model,
        output_file=args.output,
        quiet=args.quiet,
        temperature=args.temperature
    )


def run_demo(
    model: Optional[str] = None,
    output_file: Optional[str] = None,
    quiet: bool = False,
    temperature: float = 0.3
):
    """Run the comprehensive demo with configurable parameters."""
    import platform

    run_start = datetime.now()

    if not quiet:
        print_header("RAGIX v0.30 Reflective Reasoning Graph — Demo")
        print_section("Environment Check")

    # Model selection
    if model:
        llm = OllamaLLM(model=model, temperature=temperature)
        if not llm.check_available():
            print_error(f"Model not available: {model}")
            print_info(f"Try: ollama pull {model}")
            sys.exit(1)
        selected_model = model
        if not quiet:
            print_success(f"Model selected: {model}")
    else:
        # Auto-select from defaults
        models_to_try = ["mistral:7b-instruct", "granite3.1-moe:3b"]
        selected_model = None

        for m in models_to_try:
            llm = OllamaLLM(model=m, temperature=temperature)
            if llm.check_available():
                selected_model = m
                if not quiet:
                    print_success(f"Model available: {m}")
                break
            elif not quiet:
                print_warning(f"Model not available: {m}")

        if not selected_model:
            print_error("No suitable model found. Please run: ollama pull mistral:7b-instruct")
            sys.exit(1)

    # Create demo runner
    sandbox = Path(__file__).parent.parent  # RAGIX root
    demo = ReasoningDemo(model=selected_model, sandbox=sandbox)
    demo.llm.temperature = temperature

    if not quiet:
        print_info(f"Sandbox: {sandbox}")
        print_info(f"Temperature: {temperature}")

    # Test cases
    test_cases = [
        {
            "name": "BYPASS Test",
            "goal": "What is the difference between cyclomatic complexity and cognitive complexity in software metrics?",
            "expected_complexity": "bypass",
        },
        {
            "name": "SIMPLE Test",
            "goal": "How many Python files are in the ragix_core directory?",
            "expected_complexity": "simple",
        },
        {
            "name": "MODERATE Test",
            "goal": "Find all Python files in ragix_core/reasoning_v30/ and count the total lines of code",
            "expected_complexity": "moderate",
        },
        {
            "name": "COMPLEX Test",
            "goal": "Analyze the structure of the reasoning_v30 module: list all classes defined, their methods, and identify the main entry points",
            "expected_complexity": "complex",
        },
    ]

    results = []
    detailed_results = []

    for i, test in enumerate(test_cases, 1):
        if not quiet:
            print_header(f"Test {i}/{len(test_cases)}: {test['name']}")

        test_start = time.time()

        try:
            state = demo.run_task(test["goal"], session_id=f"test_{i}")
            test_elapsed = time.time() - test_start

            # Collect graph trace
            graph = demo.build_graph()
            # Re-run to get trace (already ran, but trace is on last graph)
            trace_summary = ""

            result = {
                "name": test["name"],
                "expected": test["expected_complexity"],
                "actual": state.complexity.value,
                "match": state.complexity.value == test["expected_complexity"],
                "success": state.stop_reason in ["success", "bypass"],
                "confidence": state.confidence,
                "reflections": state.reflection_count,
                "elapsed_seconds": round(test_elapsed, 2),
                "stop_reason": state.stop_reason,
            }
            results.append(result)

            # Detailed result for JSON export
            detailed = {
                **result,
                "goal": test["goal"],
                "final_answer": state.final_answer[:2000] if state.final_answer else None,
                "plan_steps": len(state.plan.steps) if state.plan else 0,
                "completed_steps": len(state.plan.get_completed_steps()) if state.plan else 0,
            }
            detailed_results.append(detailed)

        except Exception as e:
            test_elapsed = time.time() - test_start
            if not quiet:
                print_error(f"Test failed with exception: {e}")

            result = {
                "name": test["name"],
                "expected": test["expected_complexity"],
                "actual": "error",
                "match": False,
                "success": False,
                "error": str(e),
                "elapsed_seconds": round(test_elapsed, 2),
            }
            results.append(result)
            detailed_results.append({**result, "goal": test["goal"]})

        if not quiet:
            print("\n")
        time.sleep(1)

    run_elapsed = (datetime.now() - run_start).total_seconds()

    # Summary
    if not quiet:
        print_header("Demo Summary")
        print_section("Test Results")
        for r in results:
            status = "✓" if r["success"] else "✗"
            match = "✓" if r.get("match") else "✗"
            color = Colors.GREEN if r["success"] else Colors.RED

            print(f"  {color}{status}{Colors.ENDC} {r['name']}")
            print(f"      Complexity: expected={r['expected']}, actual={r['actual']} [{match}]")
            if r.get("confidence"):
                print(f"      Confidence: {r['confidence']:.2f}")
            if r.get("reflections", 0) > 0:
                print(f"      Reflections: {r['reflections']}")
            print(f"      Time: {r['elapsed_seconds']:.2f}s")

        print_section("Statistics")
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        complexity_match = sum(1 for r in results if r.get("match"))

        print(f"  Tests Passed: {passed}/{total}")
        print(f"  Complexity Classification: {complexity_match}/{total} correct")
        print(f"  Events Logged: {len(demo.events)}")
        print(f"  Total Time: {run_elapsed:.1f}s")

        print_section("Experience Corpus")
        print(f"  Project events: {demo.corpus.project_corpus.get_event_count()}")
        print(f"  Global events: {demo.corpus.global_corpus.get_event_count()}")

        print("\n" + "="*70)
        print(f"{Colors.BOLD}Demo completed successfully!{Colors.ENDC}")
        print("="*70 + "\n")

    # JSON output for benchmarking
    if output_file:
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        complexity_match = sum(1 for r in results if r.get("match"))
        avg_confidence = sum(r.get("confidence", 0) or 0 for r in results) / total
        total_reflections = sum(r.get("reflections", 0) for r in results)

        benchmark_data = {
            "meta": {
                "timestamp": run_start.isoformat(),
                "model": selected_model,
                "temperature": temperature,
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "ragix_version": "0.30.0",
            },
            "summary": {
                "tests_total": total,
                "tests_passed": passed,
                "pass_rate": round(passed / total, 2),
                "complexity_correct": complexity_match,
                "complexity_accuracy": round(complexity_match / total, 2),
                "avg_confidence": round(avg_confidence, 3),
                "total_reflections": total_reflections,
                "total_elapsed_seconds": round(run_elapsed, 2),
            },
            "tests": detailed_results,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(benchmark_data, f, indent=2, default=str)

        if not quiet:
            print_info(f"Results saved to: {output_file}")

    # Return summary for programmatic use
    return {
        "model": selected_model,
        "passed": passed,
        "total": total,
        "complexity_accuracy": complexity_match / total,
        "elapsed": run_elapsed,
    }


if __name__ == "__main__":
    main_with_args()
