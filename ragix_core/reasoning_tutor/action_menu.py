# =============================================================================
# Action Menu - Semantic Abstraction for Slim LLMs
# =============================================================================
#
# Key insight: Slim LLMs speak natural language, not bash.
# Solution: Provide a MENU of semantic actions they can pick from.
#
# Like MCP tools, like card games - players pick from available moves.
# Even kids learn fast when given the right interface!
#
# Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
# Version: 0.1.0 (2025-12-22)
#
# =============================================================================

"""
Action Menu System for Slim LLM Assistance.

Instead of asking LLMs to write bash:
    "grep -r 'EUREKA' ."  (bash syntax - error prone)

Offer semantic actions:
    SEARCH_CONTENT("EUREKA")  (intent-based - any LLM can pick this)

The Tutor translates actions to safe shell commands.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
import shlex


class ActionCategory(Enum):
    """Categories of actions."""
    SEARCH = "search"           # Find things
    READ = "read"               # View content
    COUNT = "count"             # Enumerate
    CHECK = "check"             # Verify existence/properties
    LIST = "list"               # List items
    COMPARE = "compare"         # Diff/compare


@dataclass
class Action:
    """
    A semantic action the LLM can pick from the menu.

    Example:
        Action(
            name="SEARCH_CONTENT",
            description="Find files containing a text pattern",
            parameters=["pattern", "path?"],
            template="grep -r {pattern} {path}",
            category=ActionCategory.SEARCH,
        )
    """
    name: str
    description: str
    parameters: list[str]       # Required params, optional end with ?
    template: str               # Shell command template
    category: ActionCategory
    examples: list[str] = field(default_factory=list)

    def format_for_llm(self) -> str:
        """Format action as menu item for LLM."""
        params = ", ".join(self.parameters)
        examples_str = ""
        if self.examples:
            examples_str = f"\n    Examples: {', '.join(self.examples)}"
        return f"  - {self.name}({params}): {self.description}{examples_str}"

    def to_command(self, **kwargs) -> str:
        """Convert action invocation to shell command."""
        # Handle optional parameters
        for param in self.parameters:
            if param.endswith("?"):
                key = param[:-1]
                if key not in kwargs:
                    kwargs[key] = "."  # Default for paths
            else:
                if param not in kwargs:
                    raise ValueError(f"Missing required parameter: {param}")

        # Quote parameters for safety
        safe_kwargs = {k: shlex.quote(str(v)) for k, v in kwargs.items()}
        return self.template.format(**safe_kwargs)


# =============================================================================
# Standard Action Library
# =============================================================================

STANDARD_ACTIONS = [
    # Search actions
    Action(
        name="SEARCH_CONTENT",
        description="Find files containing a text pattern",
        parameters=["pattern", "path?"],
        template="grep -r -l {pattern} {path}",
        category=ActionCategory.SEARCH,
        examples=["SEARCH_CONTENT('TODO')", "SEARCH_CONTENT('error', 'logs/')"],
    ),
    Action(
        name="SEARCH_CONTENT_SHOW",
        description="Find and show lines containing a pattern",
        parameters=["pattern", "path?"],
        template="grep -r -n {pattern} {path}",
        category=ActionCategory.SEARCH,
        examples=["SEARCH_CONTENT_SHOW('EUREKA')"],
    ),
    Action(
        name="SEARCH_FILENAME",
        description="Find files by name pattern",
        parameters=["pattern", "path?"],
        template="find {path} -name {pattern}",
        category=ActionCategory.SEARCH,
        examples=["SEARCH_FILENAME('*.py')", "SEARCH_FILENAME('config*', 'etc/')"],
    ),

    # Read actions
    Action(
        name="READ_FILE",
        description="Read entire file content",
        parameters=["path"],
        template="cat {path}",
        category=ActionCategory.READ,
        examples=["READ_FILE('config.yaml')"],
    ),
    Action(
        name="READ_HEAD",
        description="Read first N lines of a file",
        parameters=["path", "lines?"],
        template="head -n {lines} {path}",
        category=ActionCategory.READ,
        examples=["READ_HEAD('log.txt', 20)"],
    ),
    Action(
        name="READ_TAIL",
        description="Read last N lines of a file",
        parameters=["path", "lines?"],
        template="tail -n {lines} {path}",
        category=ActionCategory.READ,
        examples=["READ_TAIL('error.log', 50)"],
    ),

    # Count actions
    Action(
        name="COUNT_LINES",
        description="Count lines in file(s)",
        parameters=["path"],
        template="wc -l {path}",
        category=ActionCategory.COUNT,
        examples=["COUNT_LINES('*.py')", "COUNT_LINES('src/')"],
    ),
    Action(
        name="COUNT_FILES",
        description="Count files matching pattern",
        parameters=["pattern", "path?"],
        template="find {path} -name {pattern} | wc -l",
        category=ActionCategory.COUNT,
        examples=["COUNT_FILES('*.txt')"],
    ),
    Action(
        name="COUNT_MATCHES",
        description="Count occurrences of pattern in files",
        parameters=["pattern", "path?"],
        template="grep -r -c {pattern} {path} | awk -F: '{{sum+=$2}} END {{print sum}}'",
        category=ActionCategory.COUNT,
        examples=["COUNT_MATCHES('TODO')"],
    ),

    # Check actions
    Action(
        name="CHECK_EXISTS",
        description="Check if file or directory exists",
        parameters=["path"],
        template="test -e {path} && echo 'EXISTS' || echo 'NOT_FOUND'",
        category=ActionCategory.CHECK,
        examples=["CHECK_EXISTS('config.yaml')"],
    ),
    Action(
        name="CHECK_FILE",
        description="Check if path is a file",
        parameters=["path"],
        template="test -f {path} && echo 'IS_FILE' || echo 'NOT_A_FILE'",
        category=ActionCategory.CHECK,
        examples=["CHECK_FILE('script.py')"],
    ),
    Action(
        name="CHECK_DIR",
        description="Check if path is a directory",
        parameters=["path"],
        template="test -d {path} && echo 'IS_DIR' || echo 'NOT_A_DIR'",
        category=ActionCategory.CHECK,
        examples=["CHECK_DIR('src/')"],
    ),
    Action(
        name="CHECK_EMPTY",
        description="Check if file is empty",
        parameters=["path"],
        template="test -s {path} && echo 'HAS_CONTENT' || echo 'EMPTY'",
        category=ActionCategory.CHECK,
        examples=["CHECK_EMPTY('output.log')"],
    ),

    # List actions
    Action(
        name="LIST_FILES",
        description="List files in directory",
        parameters=["path?"],
        template="ls -la {path}",
        category=ActionCategory.LIST,
        examples=["LIST_FILES()", "LIST_FILES('src/')"],
    ),
    Action(
        name="LIST_TREE",
        description="Show directory tree structure",
        parameters=["path?", "depth?"],
        template="find {path} -maxdepth {depth} -print | head -50",
        category=ActionCategory.LIST,
        examples=["LIST_TREE('.', 2)"],
    ),
    Action(
        name="LIST_RECENT",
        description="List recently modified files",
        parameters=["path?", "count?"],
        template="find {path} -type f -mtime -1 | head -{count}",
        category=ActionCategory.LIST,
        examples=["LIST_RECENT('.', 10)"],
    ),

    # Compare actions
    Action(
        name="COMPARE_FILES",
        description="Compare two files",
        parameters=["file1", "file2"],
        template="diff {file1} {file2}",
        category=ActionCategory.COMPARE,
        examples=["COMPARE_FILES('old.txt', 'new.txt')"],
    ),
]


# =============================================================================
# Action Menu Generator
# =============================================================================

class ActionMenu:
    """
    Generates action menus for LLMs.

    Usage:
        menu = ActionMenu()
        prompt = menu.generate_prompt(goal="Find files with TODO")
        # LLM picks: SEARCH_CONTENT("TODO")
        command = menu.parse_action("SEARCH_CONTENT('TODO')")
        # Returns: grep -r -l 'TODO' .
    """

    def __init__(self, actions: list[Action] = None):
        self.actions = actions or STANDARD_ACTIONS
        self._action_map = {a.name: a for a in self.actions}

    def generate_menu(self, categories: list[ActionCategory] = None) -> str:
        """Generate the action menu as text."""
        lines = ["Available actions (pick one):"]

        if categories:
            filtered = [a for a in self.actions if a.category in categories]
        else:
            filtered = self.actions

        # Group by category
        by_category = {}
        for action in filtered:
            cat = action.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(action)

        for cat_name, actions in sorted(by_category.items()):
            lines.append(f"\n[{cat_name.upper()}]")
            for action in actions:
                lines.append(action.format_for_llm())

        return "\n".join(lines)

    def generate_prompt(self, goal: str, context: str = "") -> str:
        """Generate a full prompt with action menu."""
        menu = self.generate_menu()

        prompt = f"""Your goal: {goal}

{menu}

Pick ONE action and respond with ONLY the action call.
Example response: SEARCH_CONTENT("error")

{context}
Your action:"""
        return prompt

    def parse_action(self, response: str) -> Optional[str]:
        """
        Parse LLM response and convert to shell command.

        Input:  "SEARCH_CONTENT('TODO')"
        Output: "grep -r -l 'TODO' ."
        """
        import re

        # Clean response
        response = response.strip()

        # Pattern: ACTION_NAME(arg1, arg2, ...)
        match = re.match(r"(\w+)\s*\((.*)\)", response)
        if not match:
            return None

        action_name = match.group(1)
        args_str = match.group(2)

        if action_name not in self._action_map:
            return None

        action = self._action_map[action_name]

        # Parse arguments
        # Handle quoted strings and simple values
        args = []
        if args_str.strip():
            # Simple parser for quoted/unquoted args, comma-separated
            parts = [p.strip() for p in args_str.split(",")]
            for part in parts:
                # Remove quotes
                part = part.strip().strip("'\"")
                if part:
                    args.append(part)

        # Map to parameters
        kwargs = {}
        param_names = [p.rstrip("?") for p in action.parameters]
        for i, arg in enumerate(args):
            if i < len(param_names):
                kwargs[param_names[i]] = arg

        # Set defaults for optional params
        for param in action.parameters:
            if param.endswith("?"):
                key = param[:-1]
                if key not in kwargs:
                    if key == "path":
                        kwargs[key] = "."
                    elif key == "lines":
                        kwargs[key] = "20"
                    elif key == "depth":
                        kwargs[key] = "3"
                    elif key == "count":
                        kwargs[key] = "20"

        try:
            return action.to_command(**kwargs)
        except Exception:
            return None

    def get_action(self, name: str) -> Optional[Action]:
        """Get action by name."""
        return self._action_map.get(name)


# =============================================================================
# Integration with Tutor
# =============================================================================

def generate_action_menu_prompt(goal: str, truths: list[str] = None) -> str:
    """
    Generate a prompt for slim LLMs using the action menu.

    This replaces the free-form "write bash" approach with
    structured action selection.
    """
    menu = ActionMenu()

    context = ""
    if truths:
        context = f"Known truths: {truths}"

    return menu.generate_prompt(goal, context)


def translate_action_to_command(action_response: str) -> Optional[str]:
    """
    Translate LLM action choice to shell command.

    Input:  "SEARCH_CONTENT('EUREKA')"
    Output: "grep -r -l 'EUREKA' ."
    """
    menu = ActionMenu()
    return menu.parse_action(action_response)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    menu = ActionMenu()

    print("=" * 60)
    print("ACTION MENU FOR SLIM LLMs")
    print("=" * 60)
    print()
    print(menu.generate_menu())
    print()
    print("=" * 60)
    print("EXAMPLE TRANSLATIONS")
    print("=" * 60)

    examples = [
        "SEARCH_CONTENT('EUREKA')",
        "SEARCH_CONTENT('TODO', 'src/')",
        "COUNT_LINES('*.py')",
        "CHECK_EXISTS('config.yaml')",
        "READ_HEAD('README.md', 10)",
        "LIST_FILES()",
    ]

    for example in examples:
        cmd = menu.parse_action(example)
        print(f"\n  {example}")
        print(f"  → {cmd}")
