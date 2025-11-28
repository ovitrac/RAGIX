"""
RAGIX Knowledge Base - Pattern Storage for 7B Model Reasoning Improvement

This module provides:
- KnowledgeBase: Stores command patterns, rules, and examples
- PatternMatcher: Matches user queries to relevant knowledge
- Prompt injection for planner/worker

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class CommandPattern:
    """A Unix command pattern with examples."""
    name: str
    description: str
    pattern: str  # The correct command pattern with placeholders
    examples: List[str]  # Concrete examples
    triggers: List[str]  # Keywords that trigger this pattern
    common_mistakes: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ReasoningRule:
    """A rule for the reasoning system."""
    name: str
    condition: str  # When this rule applies
    action: str  # What to do
    priority: int = 0  # Higher = more important
    examples: List[str] = field(default_factory=list)


@dataclass
class KnowledgeBase:
    """
    Knowledge base for improving 7B model reasoning.

    Stores:
    - Command patterns: Correct Unix command templates
    - Reasoning rules: When to apply specific strategies
    - Common mistakes: What to avoid
    - Examples: Working examples for few-shot learning
    """

    patterns: Dict[str, CommandPattern] = field(default_factory=dict)
    rules: List[ReasoningRule] = field(default_factory=list)
    common_mistakes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize with default patterns and rules."""
        self._load_default_patterns()
        self._load_default_rules()
        self._load_default_mistakes()

    def _load_default_patterns(self):
        """Load built-in Unix command patterns."""

        # File listing patterns
        self.patterns["list_files"] = CommandPattern(
            name="list_files",
            description="List files of a specific type",
            pattern='find . -name "*.{EXT}" -type f',
            examples=[
                'find . -name "*.md" -type f',
                'find . -name "*.py" -type f',
                'find . -name "*.java" -type f',
            ],
            triggers=["list", "show", "find files", "all files", "files of type"],
            common_mistakes=[
                'find . -name ".md" -type f  # WRONG: missing asterisk',
                'ls *.md  # May miss files in subdirectories',
            ],
            notes="ALWAYS use asterisk: *.ext, not .ext"
        )

        # Line count patterns
        self.patterns["count_lines"] = CommandPattern(
            name="count_lines",
            description="Count lines in files",
            pattern='find . -name "*.{EXT}" -type f -exec wc -l {} +',
            examples=[
                'find . -name "*.md" -type f -exec wc -l {} +',
                'find . -name "*.py" -type f -exec wc -l {} +',
            ],
            triggers=["count lines", "line count", "how many lines", "total lines"],
            common_mistakes=[
                'wc -l *.md  # May fail with spaces in filenames',
            ],
            notes="Use -exec wc -l {} + for proper handling of special characters"
        )

        # Largest file pattern
        self.patterns["largest_file"] = CommandPattern(
            name="largest_file",
            description="Find the largest file by line count",
            pattern='find . -name "*.{EXT}" -type f -exec wc -l {} + | grep -v " total$" | sort -n | tail -1',
            examples=[
                'find . -name "*.md" -type f -exec wc -l {} + | grep -v " total$" | sort -n | tail -1',
                'find . -name "*.py" -type f -exec wc -l {} + | grep -v " total$" | sort -n | tail -1',
            ],
            triggers=["largest", "biggest", "most lines", "maximum"],
            common_mistakes=[
                'sort -n | head -1  # WRONG: gives SMALLEST, not largest',
                'sort -rn | tail -1  # WRONG: reverse sort then tail gives smallest',
            ],
            notes="CRITICAL: For LARGEST use 'sort -n | tail -1', for SMALLEST use 'sort -n | head -1'"
        )

        # Smallest file pattern
        self.patterns["smallest_file"] = CommandPattern(
            name="smallest_file",
            description="Find the smallest file by line count",
            pattern='find . -name "*.{EXT}" -type f -exec wc -l {} + | grep -v " total$" | sort -n | head -1',
            examples=[
                'find . -name "*.md" -type f -exec wc -l {} + | grep -v " total$" | sort -n | head -1',
            ],
            triggers=["smallest", "shortest", "fewest lines", "minimum", "least"],
            common_mistakes=[
                'sort -n | tail -1  # WRONG: gives LARGEST, not smallest',
            ],
            notes="CRITICAL: For SMALLEST use 'sort -n | head -1'"
        )

        # Files over N lines
        self.patterns["files_over_n_lines"] = CommandPattern(
            name="files_over_n_lines",
            description="Find files with more than N lines",
            pattern='find . -name "*.{EXT}" -type f -exec wc -l {} + | grep -v " total$" | awk \'$1 > {N}\'',
            examples=[
                'find . -name "*.md" -type f -exec wc -l {} + | grep -v " total$" | awk \'$1 > 100\'',
                'find . -name "*.py" -type f -exec wc -l {} + | grep -v " total$" | awk \'$1 > 500\'',
                'find . -name "*.md" -type f -exec wc -l {} + | grep -v " total$" | awk \'$1 > 1000\'',
            ],
            triggers=["more than", "greater than", "over", "exceeds", "above", "> lines"],
            common_mistakes=[
                'grep -r  # Does not count lines',
                'wc -l | grep  # Needs awk for numeric comparison',
            ],
            notes="Use awk for numeric comparison: awk '$1 > N'"
        )

        # Files under N lines
        self.patterns["files_under_n_lines"] = CommandPattern(
            name="files_under_n_lines",
            description="Find files with fewer than N lines",
            pattern='find . -name "*.{EXT}" -type f -exec wc -l {} + | grep -v " total$" | awk \'$1 < {N}\'',
            examples=[
                'find . -name "*.md" -type f -exec wc -l {} + | grep -v " total$" | awk \'$1 < 50\'',
            ],
            triggers=["less than", "fewer than", "under", "below", "< lines"],
            notes="Use awk for numeric comparison: awk '$1 < N'"
        )

        # Search content
        self.patterns["search_content"] = CommandPattern(
            name="search_content",
            description="Search for a pattern in files",
            pattern='grep -rn "{PATTERN}" --include="*.{EXT}" .',
            examples=[
                'grep -rn "def " --include="*.py" .',
                'grep -rn "class " --include="*.java" .',
                'grep -rn "TODO" --include="*.md" .',
            ],
            triggers=["search", "find text", "grep", "look for", "containing"],
            common_mistakes=[
                'grep -r "pattern"  # Missing -n for line numbers',
                'grep "pattern" *.py  # May miss subdirectories',
            ],
            notes="Always use -n for line numbers and --include for file filtering"
        )

        # Read file content
        self.patterns["read_file"] = CommandPattern(
            name="read_file",
            description="Read file content",
            pattern="cat {FILE}",
            examples=[
                "cat README.md",
                "head -50 large_file.py",
                "tail -20 logfile.log",
            ],
            triggers=["read", "show content", "display", "cat", "view"],
            notes="Use head/tail for large files to avoid output overload"
        )

        # File size (bytes)
        self.patterns["file_size_bytes"] = CommandPattern(
            name="file_size_bytes",
            description="Find files by byte size",
            pattern='find . -name "*.{EXT}" -type f -exec ls -l {} + | sort -k5 -n',
            examples=[
                'find . -name "*.md" -type f -exec ls -l {} + | sort -k5 -n | tail -1',
            ],
            triggers=["file size", "bytes", "kb", "mb", "disk space"],
            notes="Use -k5 to sort by size column in ls -l output"
        )

        # Directory listing
        self.patterns["directory_tree"] = CommandPattern(
            name="directory_tree",
            description="Show directory structure",
            pattern="find . -type d | head -50",
            examples=[
                "find . -type d -name 'src*'",
                "ls -la",
                "tree -L 2",  # If tree is installed
            ],
            triggers=["directories", "folders", "structure", "tree"],
            notes="Use 'tree' if available, otherwise 'find -type d'"
        )

    def _load_default_rules(self):
        """Load built-in reasoning rules."""

        self.rules = [
            ReasoningRule(
                name="asterisk_in_glob",
                condition="File extension pattern",
                action='ALWAYS use asterisk: "*.ext" not ".ext"',
                priority=100,
                examples=[
                    'CORRECT: find . -name "*.md"',
                    'WRONG: find . -name ".md"',
                ]
            ),
            ReasoningRule(
                name="largest_vs_smallest",
                condition="Finding largest or smallest",
                action='For LARGEST: sort -n | tail -1. For SMALLEST: sort -n | head -1',
                priority=100,
                examples=[
                    'LARGEST: sort -n | tail -1',
                    'SMALLEST: sort -n | head -1',
                ]
            ),
            ReasoningRule(
                name="numeric_comparison",
                condition="Comparing line counts or numbers",
                action='Use awk for numeric comparison: awk \'$1 > N\' or awk \'$1 < N\'',
                priority=90,
                examples=[
                    'More than 100 lines: awk \'$1 > 100\'',
                    'Less than 50 lines: awk \'$1 < 50\'',
                ]
            ),
            ReasoningRule(
                name="exclude_total",
                condition="Using wc -l with multiple files",
                action='Add | grep -v " total$" to exclude the total line',
                priority=80,
                examples=[
                    'find . -name "*.md" -exec wc -l {} + | grep -v " total$"',
                ]
            ),
            ReasoningRule(
                name="json_output",
                condition="Generating JSON action output",
                action='Output ONLY valid JSON, escape quotes properly',
                priority=100,
                examples=[
                    '{"action": "bash", "command": "find . -name \\"*.md\\""}',
                ]
            ),
            ReasoningRule(
                name="complete_commands",
                condition="Generating bash commands",
                action='Generate COMPLETE commands, never partial or truncated',
                priority=100,
                examples=[
                    'COMPLETE: find . -name "*.md" -type f -exec wc -l {} +',
                    'INCOMPLETE: find . -name "*.md" -type f -exec',
                ]
            ),
            ReasoningRule(
                name="single_json_object",
                condition="Response format",
                action='Output exactly ONE JSON object, not multiple',
                priority=95,
                examples=[
                    'CORRECT: {"action": "bash", "command": "ls"}',
                    'WRONG: {"action": "bash"} {"command": "ls"}',
                ]
            ),
        ]

    def _load_default_mistakes(self):
        """Load common mistakes to avoid."""
        self.common_mistakes = {
            "missing_asterisk": 'Using ".md" instead of "*.md" in find patterns',
            "wrong_sort_direction": 'Using sort -n | head -1 for largest (should be tail)',
            "incomplete_command": 'Truncating commands or missing pipe stages',
            "malformed_json": 'Invalid JSON syntax in action output',
            "missing_grep_v": 'Not excluding "total" line from wc output',
            "recursive_search": 'Using grep without -r for subdirectories',
            "line_numbers": 'Forgetting -n in grep for line numbers',
        }

    def load_from_file(self, filepath: Path) -> bool:
        """Load additional patterns from a YAML file."""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            if "patterns" in data:
                for name, pdata in data["patterns"].items():
                    self.patterns[name] = CommandPattern(
                        name=name,
                        description=pdata.get("description", ""),
                        pattern=pdata.get("pattern", ""),
                        examples=pdata.get("examples", []),
                        triggers=pdata.get("triggers", []),
                        common_mistakes=pdata.get("common_mistakes", []),
                        notes=pdata.get("notes", ""),
                    )

            if "rules" in data:
                for rdata in data["rules"]:
                    self.rules.append(ReasoningRule(
                        name=rdata.get("name", ""),
                        condition=rdata.get("condition", ""),
                        action=rdata.get("action", ""),
                        priority=rdata.get("priority", 0),
                        examples=rdata.get("examples", []),
                    ))

            if "common_mistakes" in data:
                self.common_mistakes.update(data["common_mistakes"])

            return True
        except Exception as e:
            print(f"Warning: Could not load knowledge base from {filepath}: {e}")
            return False

    def save_to_file(self, filepath: Path):
        """Save knowledge base to a YAML file."""
        data = {
            "patterns": {},
            "rules": [],
            "common_mistakes": self.common_mistakes,
        }

        for name, pattern in self.patterns.items():
            data["patterns"][name] = {
                "description": pattern.description,
                "pattern": pattern.pattern,
                "examples": pattern.examples,
                "triggers": pattern.triggers,
                "common_mistakes": pattern.common_mistakes,
                "notes": pattern.notes,
            }

        for rule in self.rules:
            data["rules"].append({
                "name": rule.name,
                "condition": rule.condition,
                "action": rule.action,
                "priority": rule.priority,
                "examples": rule.examples,
            })

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def match_patterns(self, user_input: str, limit: int = 3) -> List[CommandPattern]:
        """Find relevant command patterns for a user query."""
        input_lower = user_input.lower()
        matches = []

        for pattern in self.patterns.values():
            score = 0
            for trigger in pattern.triggers:
                if trigger.lower() in input_lower:
                    score += 1
            if score > 0:
                matches.append((score, pattern))

        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]

    def get_rules_text(self, priority_threshold: int = 50) -> str:
        """Get high-priority rules as text for prompt injection."""
        high_priority = [r for r in self.rules if r.priority >= priority_threshold]
        high_priority.sort(key=lambda r: r.priority, reverse=True)

        lines = ["CRITICAL RULES:"]
        for rule in high_priority:
            lines.append(f"- {rule.action}")
            if rule.examples:
                for ex in rule.examples[:2]:
                    lines.append(f"  Example: {ex}")

        return "\n".join(lines)

    def get_patterns_text(self, user_input: str) -> str:
        """Get relevant patterns as text for prompt injection."""
        patterns = self.match_patterns(user_input)

        if not patterns:
            return ""

        lines = ["RELEVANT COMMAND PATTERNS:"]
        for pattern in patterns:
            lines.append(f"\n## {pattern.name}: {pattern.description}")
            lines.append(f"Template: {pattern.pattern}")
            if pattern.examples:
                lines.append("Examples:")
                for ex in pattern.examples[:2]:
                    lines.append(f"  {ex}")
            if pattern.notes:
                lines.append(f"NOTE: {pattern.notes}")

        return "\n".join(lines)

    def get_mistakes_text(self) -> str:
        """Get common mistakes as text for prompt injection."""
        lines = ["COMMON MISTAKES TO AVOID:"]
        for name, mistake in self.common_mistakes.items():
            lines.append(f"- {mistake}")
        return "\n".join(lines)

    def get_prompt_injection(self, user_input: str) -> str:
        """
        Get complete knowledge base injection for prompts.

        This is the main method to call when building prompts.
        """
        sections = [
            self.get_rules_text(),
            "",
            self.get_patterns_text(user_input),
            "",
            self.get_mistakes_text(),
        ]
        return "\n".join(sections)


# Global instance for easy access
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get the global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()

        # Try to load custom rules if available
        custom_path = Path(__file__).parent / "knowledge_rules.yaml"
        if custom_path.exists():
            _knowledge_base.load_from_file(custom_path)

    return _knowledge_base


def reset_knowledge_base():
    """Reset the global knowledge base (useful for testing)."""
    global _knowledge_base
    _knowledge_base = None
