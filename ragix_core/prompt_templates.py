"""
Prompt Templates - Specialized prompts for different agent tasks

Provides templates for various scenarios including few-shot examples,
chain-of-thought reasoning, and task-specific guidance.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class TaskType(str, Enum):
    """Types of tasks agents can perform."""

    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    EXPLORATION = "exploration"


@dataclass
class FewShotExample:
    """A few-shot example for in-context learning."""

    task: str
    thinking: str
    tool_calls: List[Dict[str, str]]
    result: str


@dataclass
class PromptTemplate:
    """A prompt template with optional few-shot examples."""

    name: str
    task_type: TaskType
    system_prompt: str
    task_prefix: str = ""
    examples: List[FewShotExample] = field(default_factory=list)
    guidelines: List[str] = field(default_factory=list)

    def format(
        self,
        task: str,
        context: Optional[str] = None,
        tools_section: Optional[str] = None,
    ) -> str:
        """
        Format the complete prompt.

        Args:
            task: The specific task to accomplish
            context: Additional context
            tools_section: Available tools description

        Returns:
            Formatted prompt string
        """
        parts = [self.system_prompt]

        if tools_section:
            parts.append(f"\n## Available Tools\n{tools_section}")

        if self.guidelines:
            parts.append("\n## Guidelines")
            for i, guideline in enumerate(self.guidelines, 1):
                parts.append(f"{i}. {guideline}")

        if self.examples:
            parts.append("\n## Examples")
            for ex in self.examples:
                parts.append(f"\n**Task:** {ex.task}")
                parts.append(f"**Thinking:** {ex.thinking}")
                parts.append("**Actions:**")
                for tc in ex.tool_calls:
                    parts.append(f"  - {tc}")
                parts.append(f"**Result:** {ex.result}")

        if context:
            parts.append(f"\n## Context\n{context}")

        if self.task_prefix:
            parts.append(f"\n{self.task_prefix}")

        parts.append(f"\n## Your Task\n{task}")

        return "\n".join(parts)


# === BUILT-IN TEMPLATES ===

BUG_FIX_TEMPLATE = PromptTemplate(
    name="bug_fix",
    task_type=TaskType.BUG_FIX,
    system_prompt="""You are a RAGIX agent specialized in finding and fixing bugs.

Your approach should be methodical:
1. Understand the bug report/symptoms
2. Locate the relevant code
3. Identify the root cause
4. Implement a fix
5. Verify the fix works""",
    task_prefix="Let's systematically diagnose and fix this bug.",
    guidelines=[
        "Search for error messages or stack traces in the codebase",
        "Read the relevant code to understand the logic",
        "Look for similar patterns that might indicate the same bug",
        "Make minimal changes to fix the issue",
        "Verify with tests or manual inspection",
    ],
    examples=[
        FewShotExample(
            task="Fix the TypeError when user input is None",
            thinking="First, I need to find where user input is handled and where the TypeError might occur.",
            tool_calls=[
                {"action": "grep_search", "pattern": "TypeError", "path": "."},
                {"action": "read_file", "path": "src/handlers/user_input.py"},
                {"action": "edit_file", "path": "src/handlers/user_input.py",
                 "old_text": "def process(data):", "new_text": "def process(data):\n    if data is None:\n        return None"},
            ],
            result="Added null check to prevent TypeError when user input is None.",
        ),
    ],
)


FEATURE_TEMPLATE = PromptTemplate(
    name="feature",
    task_type=TaskType.FEATURE,
    system_prompt="""You are a RAGIX agent specialized in implementing new features.

Your approach should follow these steps:
1. Understand the feature requirements
2. Identify where to add the new code
3. Check existing patterns in the codebase
4. Implement the feature following project conventions
5. Add necessary tests or documentation""",
    task_prefix="Let's implement this feature step by step.",
    guidelines=[
        "Study existing code patterns before writing new code",
        "Follow the project's coding conventions",
        "Keep changes focused and minimal",
        "Consider edge cases and error handling",
        "Update related documentation if needed",
    ],
)


REFACTOR_TEMPLATE = PromptTemplate(
    name="refactor",
    task_type=TaskType.REFACTOR,
    system_prompt="""You are a RAGIX agent specialized in code refactoring.

Your approach should be:
1. Understand the current code structure
2. Identify the refactoring goals
3. Plan the changes to minimize risk
4. Make incremental changes
5. Verify behavior is preserved""",
    task_prefix="Let's refactor this code carefully.",
    guidelines=[
        "Run tests before and after refactoring",
        "Make small, incremental changes",
        "Preserve external behavior",
        "Improve readability and maintainability",
        "Document significant structural changes",
    ],
)


CODE_REVIEW_TEMPLATE = PromptTemplate(
    name="code_review",
    task_type=TaskType.CODE_REVIEW,
    system_prompt="""You are a RAGIX agent specialized in code review.

Your review should check for:
1. Correctness - Does the code work as intended?
2. Security - Are there potential vulnerabilities?
3. Performance - Are there inefficiencies?
4. Readability - Is the code clear and well-documented?
5. Best practices - Does it follow project conventions?""",
    task_prefix="Let me review this code thoroughly.",
    guidelines=[
        "Be constructive and specific in feedback",
        "Prioritize critical issues over style preferences",
        "Suggest concrete improvements",
        "Acknowledge good practices",
        "Consider the broader system impact",
    ],
)


DOCUMENTATION_TEMPLATE = PromptTemplate(
    name="documentation",
    task_type=TaskType.DOCUMENTATION,
    system_prompt="""You are a RAGIX agent specialized in documentation.

Your approach should:
1. Understand the code's purpose and usage
2. Identify what documentation is needed
3. Write clear, accurate documentation
4. Include examples where helpful
5. Ensure consistency with existing docs""",
    task_prefix="Let me document this properly.",
    guidelines=[
        "Be clear and concise",
        "Include code examples",
        "Document parameters, return values, and exceptions",
        "Update related documentation",
        "Follow existing documentation style",
    ],
)


TESTING_TEMPLATE = PromptTemplate(
    name="testing",
    task_type=TaskType.TESTING,
    system_prompt="""You are a RAGIX agent specialized in testing.

Your approach should:
1. Understand what needs to be tested
2. Identify test scenarios (happy path, edge cases, errors)
3. Check existing test patterns
4. Write comprehensive tests
5. Verify tests pass""",
    task_prefix="Let me create thorough tests.",
    guidelines=[
        "Cover happy path and edge cases",
        "Test error conditions",
        "Follow existing test patterns",
        "Keep tests focused and readable",
        "Use descriptive test names",
    ],
)


EXPLORATION_TEMPLATE = PromptTemplate(
    name="exploration",
    task_type=TaskType.EXPLORATION,
    system_prompt="""You are a RAGIX agent specialized in codebase exploration.

Your goal is to understand the codebase and answer questions about it.
Be thorough in your exploration and provide accurate information.""",
    task_prefix="Let me explore the codebase to answer this.",
    guidelines=[
        "Start with a high-level overview",
        "Dive deeper into relevant areas",
        "Cross-reference multiple files",
        "Note patterns and conventions",
        "Provide concrete examples",
    ],
)


# === CHAIN OF THOUGHT TEMPLATES ===

CHAIN_OF_THOUGHT_PREFIX = """Think through this step by step:

<thinking>
1. First, I need to understand...
2. Then, I should check...
3. Next, I'll...
4. Finally, I'll verify...
</thinking>

Now let me execute this plan:
"""


ERROR_RECOVERY_PROMPT = """The previous action resulted in an error:

{error}

Let me try a different approach:

<thinking>
1. What went wrong: {error_analysis}
2. Alternative approach: {alternative}
</thinking>

Attempting alternative:
"""


# === TEMPLATE REGISTRY ===

TEMPLATES: Dict[TaskType, PromptTemplate] = {
    TaskType.BUG_FIX: BUG_FIX_TEMPLATE,
    TaskType.FEATURE: FEATURE_TEMPLATE,
    TaskType.REFACTOR: REFACTOR_TEMPLATE,
    TaskType.CODE_REVIEW: CODE_REVIEW_TEMPLATE,
    TaskType.DOCUMENTATION: DOCUMENTATION_TEMPLATE,
    TaskType.TESTING: TESTING_TEMPLATE,
    TaskType.EXPLORATION: EXPLORATION_TEMPLATE,
}


def get_template(task_type: TaskType) -> PromptTemplate:
    """
    Get a prompt template by task type.

    Args:
        task_type: Type of task

    Returns:
        PromptTemplate for the task type
    """
    return TEMPLATES.get(task_type, EXPLORATION_TEMPLATE)


def detect_task_type(task: str) -> TaskType:
    """
    Attempt to detect the task type from the task description.

    Args:
        task: Task description

    Returns:
        Detected TaskType
    """
    task_lower = task.lower()

    if any(word in task_lower for word in ["bug", "fix", "error", "crash", "broken"]):
        return TaskType.BUG_FIX

    if any(word in task_lower for word in ["add", "implement", "create", "new feature"]):
        return TaskType.FEATURE

    if any(word in task_lower for word in ["refactor", "clean", "restructure", "reorganize"]):
        return TaskType.REFACTOR

    if any(word in task_lower for word in ["review", "check", "audit", "inspect"]):
        return TaskType.CODE_REVIEW

    if any(word in task_lower for word in ["document", "docs", "readme", "comment"]):
        return TaskType.DOCUMENTATION

    if any(word in task_lower for word in ["test", "unittest", "coverage"]):
        return TaskType.TESTING

    return TaskType.EXPLORATION


def build_prompt(
    task: str,
    task_type: Optional[TaskType] = None,
    tools_section: Optional[str] = None,
    context: Optional[str] = None,
    use_chain_of_thought: bool = True,
) -> str:
    """
    Build a complete prompt for a task.

    Args:
        task: Task description
        task_type: Type of task (auto-detected if not provided)
        tools_section: Available tools description
        context: Additional context
        use_chain_of_thought: Include chain-of-thought prefix

    Returns:
        Complete formatted prompt
    """
    if task_type is None:
        task_type = detect_task_type(task)

    template = get_template(task_type)
    prompt = template.format(task, context, tools_section)

    if use_chain_of_thought:
        prompt += f"\n\n{CHAIN_OF_THOUGHT_PREFIX}"

    return prompt
