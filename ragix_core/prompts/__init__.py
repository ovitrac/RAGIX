"""
RAGIX Prompt Database

A structured collection of demo prompts organized by complexity level
for use with the reasoning_v30 graph and ragix-web UI.

Usage:
    from ragix_core.prompts import PromptDatabase, get_prompts_by_complexity

    db = PromptDatabase()

    # Get all prompts for a complexity level
    simple_prompts = db.get_by_complexity("simple")

    # Get quick actions for UI
    quick_actions = db.get_quick_actions()

    # Search prompts
    results = db.search("python files")

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Prompt:
    """A single prompt entry."""
    id: str
    name: str
    prompt: str
    complexity: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    expected_tool: Optional[str] = None
    expected_steps: Optional[int] = None
    may_reflect: bool = False
    icon: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "complexity": self.complexity,
            "category": self.category,
            "tags": self.tags,
            "expected_tool": self.expected_tool,
            "expected_steps": self.expected_steps,
            "may_reflect": self.may_reflect,
            "icon": self.icon,
        }


class PromptDatabase:
    """
    Manager for the prompt database.

    Loads prompts from YAML files and provides search/filter capabilities.
    """

    COMPLEXITY_ORDER = ["bypass", "simple", "moderate", "complex"]

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize the prompt database.

        Args:
            prompts_dir: Directory containing prompt YAML files.
                        Defaults to the module directory.
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent

        self.prompts_dir = Path(prompts_dir)
        self._prompts: Dict[str, Prompt] = {}
        self._by_complexity: Dict[str, List[Prompt]] = {
            "bypass": [],
            "simple": [],
            "moderate": [],
            "complex": [],
        }
        self._quick_actions: List[Prompt] = []
        self._categories: Dict[str, List[Prompt]] = {}

        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all prompts from YAML files."""
        yaml_files = list(self.prompts_dir.glob("*.yaml")) + list(self.prompts_dir.glob("*.yml"))

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)

                if data:
                    self._parse_yaml_data(data)
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

    def _parse_yaml_data(self, data: Dict[str, Any]) -> None:
        """Parse loaded YAML data into Prompt objects."""
        # Parse complexity-sorted prompts
        for complexity in self.COMPLEXITY_ORDER:
            if complexity in data:
                for item in data[complexity]:
                    prompt = Prompt(
                        id=item.get("id", ""),
                        name=item.get("name", ""),
                        prompt=item.get("prompt", ""),
                        complexity=complexity,
                        category=item.get("category", "general"),
                        tags=item.get("tags", []),
                        expected_tool=item.get("expected_tool"),
                        expected_steps=item.get("expected_steps"),
                        may_reflect=item.get("may_reflect", False),
                        icon=item.get("icon"),
                    )
                    self._add_prompt(prompt)

        # Parse domain-specific prompts
        if "domain_specific" in data:
            for domain, prompts in data["domain_specific"].items():
                for item in prompts:
                    complexity = item.get("complexity", "moderate")
                    prompt = Prompt(
                        id=item.get("id", ""),
                        name=item.get("name", ""),
                        prompt=item.get("prompt", ""),
                        complexity=complexity,
                        category=item.get("category", domain),
                        tags=item.get("tags", [domain]),
                    )
                    self._add_prompt(prompt)

        # Parse quick actions
        if "quick_actions" in data:
            for item in data["quick_actions"]:
                prompt = Prompt(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    prompt=item.get("prompt", ""),
                    complexity=item.get("complexity", "simple"),
                    category="quick_action",
                    icon=item.get("icon"),
                )
                self._quick_actions.append(prompt)
                self._add_prompt(prompt)

    def _add_prompt(self, prompt: Prompt) -> None:
        """Add a prompt to all indexes."""
        self._prompts[prompt.id] = prompt

        if prompt.complexity in self._by_complexity:
            self._by_complexity[prompt.complexity].append(prompt)

        if prompt.category not in self._categories:
            self._categories[prompt.category] = []
        self._categories[prompt.category].append(prompt)

    def get_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """Get a prompt by its ID."""
        return self._prompts.get(prompt_id)

    def get_by_complexity(self, complexity: str) -> List[Prompt]:
        """Get all prompts for a given complexity level."""
        return self._by_complexity.get(complexity.lower(), [])

    def get_by_category(self, category: str) -> List[Prompt]:
        """Get all prompts in a category."""
        return self._categories.get(category, [])

    def get_quick_actions(self) -> List[Prompt]:
        """Get quick action prompts for UI buttons."""
        return self._quick_actions

    def get_all(self) -> List[Prompt]:
        """Get all prompts."""
        return list(self._prompts.values())

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self._categories.keys())

    def search(self, query: str, complexity: Optional[str] = None) -> List[Prompt]:
        """
        Search prompts by keyword.

        Args:
            query: Search query (matches name, prompt text, tags)
            complexity: Optional complexity filter

        Returns:
            List of matching prompts
        """
        query_lower = query.lower()
        results = []

        for prompt in self._prompts.values():
            # Filter by complexity if specified
            if complexity and prompt.complexity != complexity.lower():
                continue

            # Search in name, prompt text, and tags
            if (query_lower in prompt.name.lower() or
                query_lower in prompt.prompt.lower() or
                any(query_lower in tag.lower() for tag in prompt.tags)):
                results.append(prompt)

        return results

    def get_random(self, complexity: Optional[str] = None) -> Optional[Prompt]:
        """Get a random prompt, optionally filtered by complexity."""
        import random

        if complexity:
            prompts = self.get_by_complexity(complexity)
        else:
            prompts = self.get_all()

        return random.choice(prompts) if prompts else None

    def to_dict(self) -> Dict[str, Any]:
        """Export database as dictionary for API responses."""
        return {
            "version": "1.0",
            "total_prompts": len(self._prompts),
            "by_complexity": {
                k: [p.to_dict() for p in v]
                for k, v in self._by_complexity.items()
            },
            "quick_actions": [p.to_dict() for p in self._quick_actions],
            "categories": list(self._categories.keys()),
        }

    def get_summary(self) -> Dict[str, int]:
        """Get count summary by complexity."""
        return {
            complexity: len(prompts)
            for complexity, prompts in self._by_complexity.items()
        }


# Module-level convenience functions
_default_db: Optional[PromptDatabase] = None


def get_prompt_database() -> PromptDatabase:
    """Get the default prompt database instance."""
    global _default_db
    if _default_db is None:
        _default_db = PromptDatabase()
    return _default_db


def get_prompts_by_complexity(complexity: str) -> List[Prompt]:
    """Convenience function to get prompts by complexity."""
    return get_prompt_database().get_by_complexity(complexity)


def get_quick_actions() -> List[Prompt]:
    """Convenience function to get quick action prompts."""
    return get_prompt_database().get_quick_actions()


def search_prompts(query: str, complexity: Optional[str] = None) -> List[Prompt]:
    """Convenience function to search prompts."""
    return get_prompt_database().search(query, complexity)


__all__ = [
    "Prompt",
    "PromptDatabase",
    "get_prompt_database",
    "get_prompts_by_complexity",
    "get_quick_actions",
    "search_prompts",
]
