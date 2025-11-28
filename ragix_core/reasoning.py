"""
RAGIX Reasoning System - Planner/Worker/Verifier Architecture

This module implements:
- EpisodicMemory: Session context persistence across turns
- ReasoningLoop: Planner/Worker/Verifier orchestration
- [PLAN] template generation and execution

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .agent_config import AgentConfig, AgentMode, AgentRole, get_agent_persona
from .knowledge_base import get_knowledge_base, KnowledgeBase


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"      # Single command, no planning needed
    MODERATE = "moderate"  # 2-3 steps, brief planning
    COMPLEX = "complex"    # Multi-step, full [PLAN] required


@dataclass
class EpisodeEntry:
    """Single episodic memory entry."""
    task_id: str
    timestamp: str
    user_goal: str
    plan_summary: str
    key_decisions: List[str]
    files_touched: List[str]
    commands_run: List[str]
    open_questions: List[str]
    result_summary: str

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "user_goal": self.user_goal,
            "plan_summary": self.plan_summary,
            "key_decisions": self.key_decisions,
            "files_touched": self.files_touched,
            "commands_run": self.commands_run,
            "open_questions": self.open_questions,
            "result_summary": self.result_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EpisodeEntry":
        return cls(**data)


@dataclass
class EpisodicMemory:
    """
    Episodic memory system for session context persistence.

    Stores what happened across turns to support longer reasoning and continuity.
    """
    storage_path: Path
    max_entries: int = 100
    entries: List[EpisodeEntry] = field(default_factory=list)
    current_session: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize storage and load existing entries."""
        self.storage_path = Path(self.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.storage_path / "episodic_log.jsonl"
        self._load_entries()
        self._init_current_session()

    def _init_current_session(self):
        """Initialize current session tracking."""
        self.current_session = {
            "task_id": f"{datetime.now().strftime('%Y-%m-%d')}-{os.getpid()}",
            "start_time": datetime.now().isoformat(),
            "user_goals": [],
            "key_decisions": [],
            "files_touched": set(),
            "commands_run": [],
            "open_questions": [],
        }

    def _load_entries(self):
        """Load existing episodic entries from storage."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self.entries.append(EpisodeEntry.from_dict(data))
            except Exception:
                self.entries = []

    def record_goal(self, goal: str):
        """Record a user goal in the current session."""
        self.current_session["user_goals"].append(goal)

    def record_decision(self, decision: str):
        """Record a key decision made during execution."""
        self.current_session["key_decisions"].append(decision)

    def record_file(self, filepath: str):
        """Record a file that was touched."""
        self.current_session["files_touched"].add(filepath)

    def record_command(self, command: str):
        """Record a command that was run."""
        self.current_session["commands_run"].append(command)

    def record_question(self, question: str):
        """Record an open question for follow-up."""
        self.current_session["open_questions"].append(question)

    def record_result(self, result: str):
        """Record a result from a command execution."""
        if "results" not in self.current_session:
            self.current_session["results"] = []
        self.current_session["results"].append(result)

    def load(self):
        """Explicitly reload entries from storage."""
        self._load_entries()

    def save_episode(self, plan_summary: str, result_summary: str) -> EpisodeEntry:
        """Save the current session as an episodic entry."""
        entry = EpisodeEntry(
            task_id=self.current_session["task_id"],
            timestamp=datetime.now().isoformat(),
            user_goal=" | ".join(self.current_session["user_goals"][-3:]),  # Last 3 goals
            plan_summary=plan_summary,
            key_decisions=self.current_session["key_decisions"][-5:],  # Last 5 decisions
            files_touched=list(self.current_session["files_touched"])[-10:],  # Last 10 files
            commands_run=self.current_session["commands_run"][-10:],  # Last 10 commands
            open_questions=self.current_session["open_questions"][-3:],  # Last 3 questions
            result_summary=result_summary,
        )

        # Append to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception:
            pass

        self.entries.append(entry)

        # Prune old entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        # Reset session for next episode
        self._init_current_session()

        return entry

    def get_context(self, keywords: Optional[List[str]] = None, limit: int = 3) -> str:
        """
        Get relevant episodic context for injection into prompt.

        Args:
            keywords: Optional keywords to filter relevant episodes
            limit: Maximum number of episodes to return
        """
        if not self.entries:
            return ""

        relevant = self.entries

        # Filter by keywords if provided
        if keywords:
            def score_entry(entry: EpisodeEntry) -> int:
                text = f"{entry.user_goal} {entry.plan_summary} {entry.result_summary}"
                text = text.lower()
                return sum(1 for kw in keywords if kw.lower() in text)

            relevant = sorted(relevant, key=score_entry, reverse=True)

        # Take most recent/relevant
        relevant = relevant[-limit:]

        if not relevant:
            return ""

        context_lines = ["[EPISODIC CONTEXT]"]
        for entry in relevant:
            context_lines.append(f"- Task: {entry.user_goal}")
            context_lines.append(f"  Result: {entry.result_summary}")
            if entry.files_touched:
                context_lines.append(f"  Files: {', '.join(entry.files_touched[:3])}")

        return "\n".join(context_lines)

    def get_current_session_context(self) -> str:
        """Get context from the current ongoing session."""
        if not self.current_session["commands_run"] and not self.current_session["key_decisions"]:
            return ""

        lines = ["[CURRENT SESSION]"]

        if self.current_session["user_goals"]:
            lines.append(f"- Previous goals: {' -> '.join(self.current_session['user_goals'][-3:])}")

        if self.current_session["commands_run"]:
            recent_cmds = self.current_session["commands_run"][-5:]
            lines.append(f"- Recent commands: {len(self.current_session['commands_run'])} total")
            for cmd in recent_cmds:
                lines.append(f"    $ {cmd[:80]}...")

        if self.current_session["files_touched"]:
            lines.append(f"- Files touched: {', '.join(list(self.current_session['files_touched'])[:5])}")

        if self.current_session["key_decisions"]:
            lines.append(f"- Key decisions: {'; '.join(self.current_session['key_decisions'][-3:])}")

        return "\n".join(lines)


@dataclass
class PlanStep:
    """Single step in a plan."""
    number: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: str = ""


@dataclass
class Plan:
    """Structured plan for a complex task."""
    objective: str
    required_data: List[str]
    steps: List[PlanStep]
    validation_criteria: List[str]

    def to_text(self) -> str:
        """Convert plan to [PLAN] text format."""
        lines = ["[PLAN]"]
        lines.append(f"1. Objective: {self.objective}")
        lines.append(f"2. Required data: {', '.join(self.required_data)}")
        lines.append("3. Steps:")
        for step in self.steps:
            status_icon = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "failed": "[!]"}
            lines.append(f"   {step.number}. {status_icon.get(step.status, '[ ]')} {step.description}")
        lines.append(f"4. Validation: {', '.join(self.validation_criteria)}")
        return "\n".join(lines)

    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step being executed."""
        for step in self.steps:
            if step.status in ("pending", "in_progress"):
                return step
        return None

    def mark_step_complete(self, step_number: int, result: str = ""):
        """Mark a step as completed."""
        for step in self.steps:
            if step.number == step_number:
                step.status = "completed"
                step.result = result
                break

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.status == "completed" for step in self.steps)


@dataclass
class ReasoningLoop:
    """
    Planner/Worker/Verifier reasoning loop.

    Orchestrates multi-step tasks using:
    - Planner: Creates structured [PLAN] for complex tasks
    - Worker: Executes steps atomically
    - Verifier: Validates results against criteria
    - KnowledgeBase: Injects patterns and rules for 7B model improvement
    """
    llm_generate: callable  # Function to call LLM: (system_prompt, messages) -> str
    agent_config: AgentConfig
    episodic_memory: EpisodicMemory
    reasoning_traces: List[Dict[str, Any]] = field(default_factory=list)
    current_plan: Optional[Plan] = None
    knowledge_base: KnowledgeBase = field(default_factory=get_knowledge_base)

    def _add_trace(self, trace_type: str, content: str, metadata: Optional[Dict] = None):
        """Add a reasoning trace for visualization (internal)."""
        trace = {
            "type": trace_type,
            "agent": trace_type.split("_")[0] if "_" in trace_type else trace_type,
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "metadata": metadata or {},
        }
        self.reasoning_traces.append(trace)
        return trace

    def add_trace(self, agent: str, content: str, metadata: Optional[Dict] = None):
        """Add a reasoning trace for visualization (public API)."""
        trace = {
            "type": agent,
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "metadata": metadata or {},
        }
        self.reasoning_traces.append(trace)
        return trace

    def classify_task(self, user_input: str) -> TaskComplexity:
        """
        Classify task complexity to determine if planning is needed.

        Simple tasks (no planning):
        - Single file operations
        - Direct questions
        - Single command requests

        Complex tasks (full planning):
        - Multiple files
        - Multi-step operations
        - Editing + verification
        """
        input_lower = user_input.lower().strip()

        # Conversational / identity questions - always SIMPLE (no planning needed)
        conversational_patterns = [
            "who are you", "what are you", "hello", "hi ", "hey ",
            "help", "how are you", "what can you do", "your name",
            "introduce yourself", "tell me about yourself",
            "thanks", "thank you", "goodbye", "bye",
            "good morning", "good afternoon", "good evening",
            # Handle common typos
            "wo are you", "wh are you", "whoare you",
        ]
        if any(kw in input_lower for kw in conversational_patterns):
            return TaskComplexity.SIMPLE

        # Questions starting with "who", "what are", etc. without file/code context are simple
        if input_lower.startswith(("who ", "wo ")) and not any(ext in input_lower for ext in ['.py', '.md', 'file', 'code']):
            return TaskComplexity.SIMPLE

        # Very short inputs (< 4 words) without file/code keywords are likely simple
        words = input_lower.split()
        if len(words) <= 3 and not any(ext in input_lower for ext in ['.py', '.md', '.java', '.js', 'file', 'code', 'find', 'grep']):
            return TaskComplexity.SIMPLE

        # Keywords indicating complexity (multi-step operations)
        complex_keywords = [
            "and then", "after that", "multiple", "several",
            "refactor", "implement", "create", "add feature", "fix bug",
            "update", "modify", "change", "edit",
            "find and", "search and", "list and",
            " and find", " and search", " and list",
            "largest", "smallest", "biggest", "most", "least",
        ]

        # Check for explicit multi-step indicators
        if any(kw in input_lower for kw in complex_keywords):
            return TaskComplexity.COMPLEX

        # Simple single-command patterns
        simple_patterns = [
            "what is", "where is", "show me", "display", "print",
            "read", "cat", "grep", "pwd", "ls", "cd",
            "explain", "describe", "how does", "why does",
        ]

        # Check for simple single-step indicators
        if any(kw in input_lower for kw in simple_patterns):
            return TaskComplexity.SIMPLE

        # "list" alone is simple, "list ... and" is complex
        if "list" in input_lower:
            if " and " in input_lower:
                return TaskComplexity.COMPLEX
            return TaskComplexity.SIMPLE

        # "find" alone is simple
        if "find" in input_lower:
            if " and " in input_lower:
                return TaskComplexity.COMPLEX
            return TaskComplexity.SIMPLE

        # "search" alone is moderate (might need follow-up)
        if "search" in input_lower:
            if " and " in input_lower:
                return TaskComplexity.COMPLEX
            return TaskComplexity.MODERATE

        # Default to moderate for ambiguous cases
        return TaskComplexity.MODERATE

    def generate_plan(self, user_input: str, context: str = "") -> Plan:
        """
        Use Planner agent to generate a structured [PLAN].
        """
        # Get planner persona
        planner_persona = get_agent_persona(AgentRole.PLANNER)

        # Get knowledge base injection for this query
        kb_injection = self.knowledge_base.get_prompt_injection(user_input)

        # Build planning prompt with Unix-focused examples
        planning_prompt = f"""{planner_persona}

USER REQUEST:
{user_input}

{context}

{kb_injection}

Generate a [PLAN] with concrete Unix commands. Each step should be a specific bash command.

CRITICAL RULES for commands:
- ALWAYS use "*.md" with asterisk, NEVER ".md" without asterisk!
  CORRECT: find . -name "*.md" -type f
  WRONG:   find . -name ".md" -type f
- Use -exec for filenames with spaces
- For "largest", use: sort -n | tail -1 (NOT head!)
- For "smallest", use: sort -n | head -1

EXAMPLE for "find the largest markdown file":
[PLAN]
1. Define the objective.
   - Find the largest markdown file by line count
2. Identify required data.
   - Line counts for all .md files
3. List steps.
   - 3.1 find . -name "*.md" -type f -exec wc -l {{}} + | grep -v " total$" | sort -n | tail -1
4. Validation criteria.
   - Report filename and line count

NOTE: For "find the largest X file", you need ONE command that does everything:
find . -name "*.EXT" -type f -exec wc -l {{}} + | grep -v " total$" | sort -n | tail -1

EXAMPLE for "search for error handling code":
[PLAN]
1. Define the objective.
   - Find code that handles errors or exceptions
2. Identify required data.
   - Source files containing error/exception patterns
3. List steps.
   - 3.1 grep -rn "except" --include="*.py" .
4. Validation criteria.
   - Show file locations with error handling patterns

EXAMPLE for "list files with more than N lines" (e.g., "markdown files with more than 1000 lines"):
[PLAN]
1. Define the objective.
   - Find markdown files exceeding 1000 lines
2. Identify required data.
   - Line counts for all .md files
3. List steps.
   - 3.1 find . -name "*.md" -type f -exec wc -l {{}} + | grep -v " total$" | awk '$1 > 1000'
4. Validation criteria.
   - List files with line count > 1000

EXAMPLE for "list all markdown files":
[PLAN]
1. Define the objective.
   - List all markdown files in the project
2. Identify required data.
   - File paths matching *.md
3. List steps.
   - 3.1 find . -name "*.md" -type f
4. Validation criteria.
   - Show list of .md file paths

CRITICAL COMMAND PATTERNS:
- List files: find . -name "*.EXT" -type f
- Count lines: find . -name "*.EXT" -type f -exec wc -l {{}} +
- Files > N lines: find . -name "*.EXT" -type f -exec wc -l {{}} + | grep -v " total$" | awk '$1 > N'
- Largest file: find . -name "*.EXT" -type f -exec wc -l {{}} + | grep -v " total$" | sort -n | tail -1
- Search content: grep -rn "pattern" --include="*.EXT" .

Now generate a [PLAN] for the user request. Use actual Unix commands (find, grep, wc, ls, cat, head, tail, sort, awk) in the steps.
Remember: Use "*.md" not ".md" for patterns!

Output ONLY the [PLAN] block, nothing else.
"""

        # Add trace
        self._add_trace("planning", f"Generating plan for: {user_input[:100]}...")

        # Call LLM
        messages = [{"role": "user", "content": planning_prompt}]
        response = self.llm_generate(planner_persona, messages)

        # Parse the plan
        plan = self._parse_plan(response, user_input)

        self._add_trace("plan_generated", plan.to_text())
        self.current_plan = plan

        return plan

    def _parse_plan(self, response: str, original_request: str) -> Plan:
        """Parse LLM response into a Plan object."""
        # Extract sections from response
        objective = original_request
        required_data = []
        steps = []
        validation = []

        lines = response.split('\n')
        current_section = None
        step_num = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if "objective" in line.lower() or line.startswith("1."):
                current_section = "objective"
            elif "required" in line.lower() or "identify" in line.lower() or line.startswith("2."):
                current_section = "required"
            elif "steps" in line.lower() or "list steps" in line.lower() or line.startswith("3."):
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
                    # Parse step number if present
                    step_match = re.match(r'^(\d+\.?\d*)\s*(.+)$', content)
                    if step_match:
                        content = step_match.group(2)
                    steps.append(PlanStep(number=step_num, description=content))
                    step_num += 1
                elif current_section == "validation":
                    validation.append(content)
            elif current_section == "steps" and re.match(r'^\d+\.?\d*\s', line):
                # Step with number prefix
                step_match = re.match(r'^(\d+\.?\d*)\s*(.+)$', line)
                if step_match:
                    content = step_match.group(2).strip()
                    steps.append(PlanStep(number=step_num, description=content))
                    step_num += 1

        # Ensure we have at least one step
        if not steps:
            steps = [PlanStep(number=1, description=original_request)]

        if not validation:
            validation = ["Verify the result matches the request"]

        return Plan(
            objective=objective,
            required_data=required_data or ["Files in current directory"],
            steps=steps,
            validation_criteria=validation,
        )

    def execute_step(self, step: PlanStep, execute_fn: callable) -> Tuple[bool, str]:
        """
        Use Worker agent to execute a single plan step.

        The worker translates step descriptions into concrete bash commands
        before executing them.

        Args:
            step: The plan step to execute
            execute_fn: Function to execute actions (shell.run or agent.step)

        Returns:
            (success, result_message)
        """
        worker_persona = get_agent_persona(AgentRole.WORKER)

        step.status = "in_progress"

        # Add trace
        self._add_trace("worker", f"Step {step.number}: {step.description}", {"step": step.number})

        # Record in episodic memory
        self.episodic_memory.record_decision(f"Executing step {step.number}: {step.description}")

        # Get knowledge base rules for worker
        kb_rules = self.knowledge_base.get_rules_text(priority_threshold=80)

        # Build a worker prompt that asks for a concrete bash command
        # This is crucial - we need to translate descriptive steps into executable commands
        worker_prompt = f"""Execute this step by running a COMPLETE bash command.

STEP TO EXECUTE: {step.description}

{kb_rules}

CRITICAL: You MUST output a COMPLETE, WORKING bash command. No partial or incomplete commands!

RESPONSE FORMAT (choose ONE):
{{"action": "bash", "command": "YOUR_COMPLETE_COMMAND_HERE"}}
{{"action": "bash_and_respond", "command": "COMMAND", "message": "explanation"}}
{{"action": "respond", "message": "response if no command needed"}}

COMMAND TEMPLATES (copy and adapt):
1. List all files of type:
   {{"action": "bash", "command": "find . -name \\"*.md\\" -type f"}}

2. Files with more than N lines:
   {{"action": "bash", "command": "find . -name \\"*.md\\" -type f -exec wc -l {{}} + | grep -v \\" total$\\" | awk \\"\\$1 > 1000\\""}}

3. Find LARGEST file by lines:
   {{"action": "bash", "command": "find . -name \\"*.md\\" -type f -exec wc -l {{}} + | grep -v \\" total$\\" | sort -n | tail -1"}}

4. Search for pattern:
   {{"action": "bash", "command": "grep -rn \\"pattern\\" --include=\\"*.py\\" ."}}

5. Count lines in all files:
   {{"action": "bash", "command": "find . -name \\"*.md\\" -type f -exec wc -l {{}} +"}}

CRITICAL RULES:
- ALWAYS use "*.md" with asterisk, NEVER ".md" alone
- For "largest/most" use: sort -n | tail -1
- For "smallest/least" use: sort -n | head -1
- For "> N lines" use: awk '$1 > N'
- Use -exec for filenames with spaces

Output ONLY valid JSON, nothing else."""

        # Execute the step with the worker prompt
        try:
            result, message = execute_fn(worker_prompt)

            if result is not None:
                # Record command
                if hasattr(result, 'command'):
                    self.episodic_memory.record_command(result.command)

                step.status = "completed"
                step.result = message or str(result)

                self._add_trace("step_completed", f"Step {step.number} completed", {"result": step.result[:200] if step.result else "OK"})

                return True, step.result
            elif message:
                step.status = "completed"
                step.result = message
                return True, message
            else:
                step.status = "failed"
                return False, "No result from step execution"

        except Exception as e:
            step.status = "failed"
            step.result = str(e)
            self._add_trace("step_failed", f"Step {step.number} failed: {e}")
            return False, str(e)

    def verify_result(self, plan: Plan, final_result: str) -> Tuple[bool, str]:
        """
        Use Verifier agent to check results against plan criteria.

        Returns:
            (is_valid, verification_report)
        """
        verifier_persona = get_agent_persona(AgentRole.VERIFIER)

        # Build verification prompt
        verify_prompt = f"""{verifier_persona}

ORIGINAL PLAN:
{plan.to_text()}

EXECUTION RESULTS:
{final_result}

Verify the result and provide a clear answer. Output a [VERIFY] block:

[VERIFY]
- ANSWER: <State the direct answer to the user's question in one line. E.g., "The largest file is V07_PROGRESS.md with 1184 lines">
- Status: <PASS or FAIL>
- Notes: <any issues or additional observations>

IMPORTANT: The ANSWER line must directly answer the user's original question with specific values from the results.

Output ONLY the [VERIFY] block.
"""

        self._add_trace("verification", "Verifying results against plan")

        messages = [{"role": "user", "content": verify_prompt}]
        response = self.llm_generate(verifier_persona, messages)

        # Parse verification
        is_pass = "PASS" in response.upper() or "pass" in response.lower()

        self._add_trace("verify_complete", response, {"passed": is_pass})

        return is_pass, response

    def run(self, user_input: str, execute_fn: callable) -> Tuple[str, List[Dict]]:
        """
        Run the full reasoning loop for a user request.

        Args:
            user_input: The user's request
            execute_fn: Function to execute steps (agent.step)

        Returns:
            (final_response, reasoning_traces)
        """
        # Record goal
        self.episodic_memory.record_goal(user_input)

        # Classify task complexity
        complexity = self.classify_task(user_input)
        self._add_trace("classification", f"Task complexity: {complexity.value}")

        # Get episodic context
        session_context = self.episodic_memory.get_current_session_context()
        historical_context = self.episodic_memory.get_context(
            keywords=user_input.split()[:5]
        )

        context = f"{session_context}\n{historical_context}".strip()

        if complexity == TaskComplexity.SIMPLE:
            # Check for conversational queries that don't need agent execution
            input_lower = user_input.lower().strip()
            greeting_patterns = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
            identity_patterns = ["who are you", "what are you", "your name", "introduce yourself"]
            help_patterns = ["help", "what can you do", "how do you work"]

            # Direct responses for greetings
            if any(g in input_lower for g in greeting_patterns) and len(input_lower.split()) <= 3:
                self._add_trace("direct_execution", "Greeting - responding directly")
                return "Hello! I am RAGIX, a Unix-RAG development assistant. How can I help you today?", self.reasoning_traces

            # Direct responses for identity questions
            if any(p in input_lower for p in identity_patterns):
                self._add_trace("direct_execution", "Identity query - responding directly")
                return "I am RAGIX, a Unix-RAG development assistant. I help you explore and work with code using Unix tools like find, grep, wc, and more. I run locally with full auditability.", self.reasoning_traces

            # Direct responses for help
            if any(p in input_lower for p in help_patterns) and "file" not in input_lower:
                self._add_trace("direct_execution", "Help query - responding directly")
                return "I can help you:\n- Find and list files (find, ls)\n- Search code (grep)\n- Analyze file sizes and line counts (wc)\n- Read and edit files\n- Run shell commands safely\n\nJust ask in natural language!", self.reasoning_traces

            # Direct execution without planning for other simple tasks
            self._add_trace("direct_execution", "Simple task - executing directly")

            # Add context to the request if available
            if context:
                enhanced_input = f"{context}\n\nCurrent request: {user_input}"
            else:
                enhanced_input = user_input

            result, message = execute_fn(enhanced_input)

            response = message or (str(result) if result else "Task executed")
            return response, self.reasoning_traces

        # For moderate/complex tasks, generate a plan
        plan = self.generate_plan(user_input, context)

        # Execute each step
        results = []
        for step in plan.steps:
            success, result = self.execute_step(step, execute_fn)
            results.append(f"Step {step.number}: {result}")

            if not success and complexity == TaskComplexity.COMPLEX:
                # For complex tasks, we might want to stop on failure
                self._add_trace("execution_halted", f"Stopping due to failure at step {step.number}")
                break

        final_result = "\n".join(results)

        # Verify if complex task
        if complexity == TaskComplexity.COMPLEX and plan.is_complete():
            is_valid, verify_report = self.verify_result(plan, final_result)
            final_result = f"{final_result}\n\n{verify_report}"

        # Save episode
        self.episodic_memory.save_episode(
            plan_summary=plan.objective,
            result_summary=final_result[:500]
        )

        # Extract ANSWER from verification if present
        answer_line = ""
        verification_status = ""
        if "[VERIFY]" in final_result:
            for line in final_result.split("\n"):
                if "ANSWER:" in line:
                    answer_line = line.split("ANSWER:")[-1].strip()
                if "Status:" in line:
                    if "PASS" in line.upper():
                        verification_status = "PASS"
                    elif "FAIL" in line.upper():
                        verification_status = "FAIL"

        # Build concise response with emojis
        status_emoji = {"completed": "\u2705", "failed": "\u274C", "pending": "\u23F3"}  # ✅ ❌ ⏳
        verify_emoji = "\u2705" if verification_status == "PASS" else "\u274C" if verification_status == "FAIL" else ""

        response_lines = []

        # Show answer prominently if available
        if answer_line:
            response_lines.append(f"{verify_emoji} **{answer_line}**")
            response_lines.append("")

        # Concise execution summary
        response_lines.append(f"\U0001F4CB Plan: {plan.objective[:80]}{'...' if len(plan.objective) > 80 else ''}")
        for step in plan.steps:
            emoji = status_emoji.get(step.status, "\u2753")
            desc = step.description[:60] + "..." if len(step.description) > 60 else step.description
            response_lines.append(f"  {emoji} {desc}")

        # Add formatted results as collapsed detail
        response_lines.append("")
        response_lines.append("<!-- DETAILS_START -->")

        for step in plan.steps:
            if step.result:
                response_lines.append(f"--- Step {step.number} ---")
                # Parse CommandResult-style output for cleaner display
                result_text = step.result
                if "CommandResult(" in result_text or "command=" in result_text:
                    # Extract key parts from CommandResult string
                    formatted = self._format_command_result(result_text)
                    response_lines.append(formatted)
                else:
                    # Plain text result
                    response_lines.append(result_text[:500] if len(result_text) > 500 else result_text)
                response_lines.append("")

        response_lines.append("<!-- DETAILS_END -->")

        # Add raw JSON for copy option (hidden, UI will handle)
        response_lines.append("<!-- JSON_START -->")
        import json as json_module
        raw_data = {
            "plan": plan.objective,
            "steps": [
                {"num": s.number, "desc": s.description, "status": s.status, "result": s.result[:1000] if s.result else ""}
                for s in plan.steps
            ]
        }
        try:
            response_lines.append(json_module.dumps(raw_data, indent=2, ensure_ascii=False))
        except:
            response_lines.append("{}")
        response_lines.append("<!-- JSON_END -->")

        return "\n".join(response_lines), self.reasoning_traces

    def _format_command_result(self, result_text: str) -> str:
        """Parse CommandResult string and format it cleanly."""
        lines = []

        def extract_field(text: str, field: str) -> str:
            """Extract a field value from CommandResult repr string."""
            # Match field='value' or field="value", handling content with quotes
            pattern = rf"{field}=(['\"])"
            match = re.search(pattern, text)
            if not match:
                return ""
            quote = match.group(1)
            start = match.end()
            # Find closing quote - look for unescaped quote followed by , or )
            pos = start
            while pos < len(text):
                if text[pos] == quote:
                    # Check if it's escaped
                    if pos > 0 and text[pos-1] == '\\':
                        pos += 1
                        continue
                    return text[start:pos]
                pos += 1
            return text[start:min(start+500, len(text))]

        # Extract command
        cmd = extract_field(result_text, "command")
        if cmd:
            lines.append(f"$ {cmd}")

        # Extract cwd
        cwd = extract_field(result_text, "cwd")
        if cwd:
            lines.append(f"(cwd: {cwd})")

        # Extract stdout - handle escaped newlines
        stdout = extract_field(result_text, "stdout")
        if stdout:
            # Unescape newlines and other escapes
            stdout = stdout.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"').strip()
            if stdout:
                # Limit output lines
                stdout_lines = stdout.split("\n")
                if len(stdout_lines) > 20:
                    stdout = "\n".join(stdout_lines[:20]) + f"\n... ({len(stdout_lines) - 20} more lines)"
                lines.append("")
                lines.append(stdout)

        # Extract stderr
        stderr = extract_field(result_text, "stderr")
        if stderr:
            stderr = stderr.replace("\\n", "\n").replace("\\t", "\t").strip()
            if stderr:
                lines.append(f"\nSTDERR: {stderr}")

        # Extract return code
        rc_match = re.search(r"returncode=(\d+)", result_text)
        if rc_match:
            rc = rc_match.group(1)
            status = "\u2705" if rc == "0" else "\u274C"
            lines.append(f"\n{status} Exit code: {rc}")

        return "\n".join(lines) if lines else result_text[:300]

    def get_traces(self) -> List[Dict]:
        """Get all reasoning traces for visualization."""
        return self.reasoning_traces

    def clear_traces(self):
        """Clear reasoning traces."""
        self.reasoning_traces = []
