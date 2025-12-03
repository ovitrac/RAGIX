"""
RAGIX v0.30 Reasoning Configuration

Configuration loader for reasoning graph settings from ragix.yaml:
- ReasoningConfig: Main configuration dataclass
- load_reasoning_config: Load from YAML file
- get_default_config: Get default configuration

Example ragix.yaml section:
    reasoning:
      strategy: "graph_v30"
      max_reflections:
        bypass: 0
        simple: 0
        moderate: 1
        complex: 3
      graph:
        max_iterations: 50
      reflect:
        allowed_tools: ["ls", "find", "grep", "head", "tail", "wc", "cat", "pwd"]
        max_context_chars: 2000
      experience:
        global_root: "~/.ragix"
        project_root: ".ragix"
        global_max_age_days: 90
        project_max_age_days: 30
        top_k: 5
      traces:
        enabled: true
        path: ".ragix/reasoning_traces"
        max_per_session: 1000

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-03
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import yaml

from .types import TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class MaxReflectionsConfig:
    """Configuration for reflection budgets by complexity."""
    bypass: int = 0
    simple: int = 0
    moderate: int = 1
    complex: int = 3

    def to_dict(self) -> Dict[TaskComplexity, int]:
        """Convert to TaskComplexity-keyed dict for node use."""
        return {
            TaskComplexity.BYPASS: self.bypass,
            TaskComplexity.SIMPLE: self.simple,
            TaskComplexity.MODERATE: self.moderate,
            TaskComplexity.COMPLEX: self.complex,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaxReflectionsConfig":
        """Create from dictionary."""
        return cls(
            bypass=data.get("bypass", 0),
            simple=data.get("simple", 0),
            moderate=data.get("moderate", 1),
            complex=data.get("complex", 3),
        )


@dataclass
class GraphConfig:
    """Configuration for graph execution."""
    max_iterations: int = 50

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphConfig":
        """Create from dictionary."""
        return cls(max_iterations=data.get("max_iterations", 50))


@dataclass
class ReflectConfig:
    """Configuration for reflection node."""
    allowed_tools: List[str] = field(default_factory=lambda: [
        "ls", "find", "grep", "head", "tail", "wc", "cat", "pwd", "file"
    ])
    max_context_chars: int = 2000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectConfig":
        """Create from dictionary."""
        return cls(
            allowed_tools=data.get("allowed_tools", [
                "ls", "find", "grep", "head", "tail", "wc", "cat", "pwd", "file"
            ]),
            max_context_chars=data.get("max_context_chars", 2000),
        )


@dataclass
class ExperienceConfig:
    """Configuration for experience corpus."""
    global_root: str = "~/.ragix"
    project_root: str = ".ragix"
    global_max_age_days: int = 90
    project_max_age_days: int = 30
    top_k: int = 5

    @property
    def global_path(self) -> Path:
        """Get expanded global root path."""
        return Path(self.global_root).expanduser()

    @property
    def project_path(self) -> Path:
        """Get project root path."""
        return Path(self.project_root)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceConfig":
        """Create from dictionary."""
        return cls(
            global_root=data.get("global_root", "~/.ragix"),
            project_root=data.get("project_root", ".ragix"),
            global_max_age_days=data.get("global_max_age_days", 90),
            project_max_age_days=data.get("project_max_age_days", 30),
            top_k=data.get("top_k", 5),
        )


@dataclass
class TracesConfig:
    """Configuration for session traces."""
    enabled: bool = True
    path: str = ".ragix/reasoning_traces"
    max_per_session: int = 1000

    @property
    def traces_path(self) -> Path:
        """Get traces directory path."""
        return Path(self.path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TracesConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            path=data.get("path", ".ragix/reasoning_traces"),
            max_per_session=data.get("max_per_session", 1000),
        )


@dataclass
class ReasoningConfig:
    """
    Main reasoning configuration.

    Contains all settings for the reasoning graph, including:
    - strategy: Which reasoning strategy to use (loop_v1, graph_v2, graph_v30)
    - max_reflections: Reflection budgets by complexity
    - graph: Graph execution settings
    - reflect: Reflection node settings
    - experience: Experience corpus settings
    - traces: Session trace settings
    """
    strategy: str = "graph_v30"
    max_reflections: MaxReflectionsConfig = field(default_factory=MaxReflectionsConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    reflect: ReflectConfig = field(default_factory=ReflectConfig)
    experience: ExperienceConfig = field(default_factory=ExperienceConfig)
    traces: TracesConfig = field(default_factory=TracesConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningConfig":
        """Create from dictionary (typically from YAML)."""
        return cls(
            strategy=data.get("strategy", "graph_v30"),
            max_reflections=MaxReflectionsConfig.from_dict(
                data.get("max_reflections", {})
            ),
            graph=GraphConfig.from_dict(data.get("graph", {})),
            reflect=ReflectConfig.from_dict(data.get("reflect", {})),
            experience=ExperienceConfig.from_dict(data.get("experience", {})),
            traces=TracesConfig.from_dict(data.get("traces", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy,
            "max_reflections": {
                "bypass": self.max_reflections.bypass,
                "simple": self.max_reflections.simple,
                "moderate": self.max_reflections.moderate,
                "complex": self.max_reflections.complex,
            },
            "graph": {"max_iterations": self.graph.max_iterations},
            "reflect": {
                "allowed_tools": self.reflect.allowed_tools,
                "max_context_chars": self.reflect.max_context_chars,
            },
            "experience": {
                "global_root": self.experience.global_root,
                "project_root": self.experience.project_root,
                "global_max_age_days": self.experience.global_max_age_days,
                "project_max_age_days": self.experience.project_max_age_days,
                "top_k": self.experience.top_k,
            },
            "traces": {
                "enabled": self.traces.enabled,
                "path": self.traces.path,
                "max_per_session": self.traces.max_per_session,
            },
        }


def get_default_config() -> ReasoningConfig:
    """Get default reasoning configuration."""
    return ReasoningConfig()


def load_reasoning_config(
    config_path: Optional[Path] = None,
    config_dict: Optional[Dict[str, Any]] = None
) -> ReasoningConfig:
    """
    Load reasoning configuration.

    Args:
        config_path: Path to ragix.yaml file
        config_dict: Pre-loaded config dictionary (takes precedence over file)

    Returns:
        ReasoningConfig instance

    Loading order:
    1. If config_dict provided, use it directly
    2. If config_path provided, load from file
    3. Try default locations: ./ragix.yaml, ~/.ragix/ragix.yaml
    4. Fall back to defaults
    """
    # Use provided dict
    if config_dict is not None:
        reasoning_section = config_dict.get("reasoning", {})
        return ReasoningConfig.from_dict(reasoning_section)

    # Try provided path
    if config_path:
        config_path = Path(config_path).expanduser()
        if config_path.exists():
            return _load_from_file(config_path)
        else:
            logger.warning(f"Config file not found: {config_path}")

    # Try default locations
    default_paths = [
        Path("ragix.yaml"),
        Path("ragix.yml"),
        Path.home() / ".ragix" / "ragix.yaml",
        Path.home() / ".ragix" / "ragix.yml",
    ]

    for path in default_paths:
        if path.exists():
            logger.debug(f"Loading config from: {path}")
            return _load_from_file(path)

    # Fall back to defaults
    logger.debug("Using default reasoning configuration")
    return get_default_config()


def _load_from_file(path: Path) -> ReasoningConfig:
    """Load configuration from YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        reasoning_section = data.get("reasoning", {})
        return ReasoningConfig.from_dict(reasoning_section)

    except Exception as e:
        logger.warning(f"Failed to load config from {path}: {e}")
        return get_default_config()


# =============================================================================
# Agent Profile Configuration
# =============================================================================

@dataclass
class AgentProfile:
    """
    Agent profile defining capabilities and constraints.

    Profiles:
    - safe: No tools, cloud models only, simple reflection, no memory
    - dev: Read + write tools, cloud/local models, full reflection, memory on
    - sovereign: Full tool access, local models only, full reflection, memory on
    """
    name: str
    tools_allowed: List[str] = field(default_factory=list)
    models_allowed: List[str] = field(default_factory=list)
    reflection_enabled: bool = True
    memory_enabled: bool = True
    local_only: bool = False

    @classmethod
    def safe(cls) -> "AgentProfile":
        """Safe profile - minimal capabilities."""
        return cls(
            name="safe",
            tools_allowed=[],
            models_allowed=["claude", "gpt-4", "gpt-3.5"],
            reflection_enabled=False,
            memory_enabled=False,
            local_only=False,
        )

    @classmethod
    def dev(cls) -> "AgentProfile":
        """Dev profile - standard development capabilities."""
        return cls(
            name="dev",
            tools_allowed=[
                "ls", "find", "grep", "head", "tail", "wc", "cat", "pwd", "file",
                "edit_file", "rt-grep", "rt-find", "ragix-ast",
            ],
            models_allowed=["mistral", "qwen", "granite", "claude", "gpt-4"],
            reflection_enabled=True,
            memory_enabled=True,
            local_only=False,
        )

    @classmethod
    def sovereign(cls) -> "AgentProfile":
        """Sovereign profile - full local capabilities."""
        return cls(
            name="sovereign",
            tools_allowed=[
                "ls", "find", "grep", "head", "tail", "wc", "cat", "pwd", "file",
                "edit_file", "rt-grep", "rt-find", "ragix-ast",
                "python", "bash", "make",
            ],
            models_allowed=["mistral", "qwen", "granite", "deepseek", "llama"],
            reflection_enabled=True,
            memory_enabled=True,
            local_only=True,
        )

    @classmethod
    def from_name(cls, name: str) -> "AgentProfile":
        """Get profile by name."""
        profiles = {
            "safe": cls.safe,
            "dev": cls.dev,
            "sovereign": cls.sovereign,
        }
        factory = profiles.get(name.lower())
        if factory:
            return factory()
        raise ValueError(f"Unknown profile: {name}")
