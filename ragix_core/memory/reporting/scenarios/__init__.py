"""
Scenario base class and registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

if False:  # TYPE_CHECKING
    from ragix_core.memory.reporting.engine import ReportEngine


class ScenarioBase(ABC):
    """Base class for all reporting scenarios."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique scenario identifier."""
        ...

    @property
    @abstractmethod
    def default_config_name(self) -> str:
        """Name of the builtin YAML config (without path)."""
        ...

    @abstractmethod
    def run(self, engine: "ReportEngine", config: dict) -> str:
        """Execute the scenario, return Markdown string."""
        ...


# Registry — populated by imports
SCENARIO_REGISTRY: Dict[str, Type[ScenarioBase]] = {}


def register_scenario(cls: Type[ScenarioBase]) -> Type[ScenarioBase]:
    """Decorator to register a scenario class."""
    instance = cls()
    SCENARIO_REGISTRY[instance.id] = cls
    return cls


def get_scenario(name: str) -> ScenarioBase:
    """Instantiate a scenario by name."""
    if name not in SCENARIO_REGISTRY:
        available = ", ".join(sorted(SCENARIO_REGISTRY.keys()))
        raise KeyError(
            f"Unknown scenario '{name}'. Available: {available}"
        )
    return SCENARIO_REGISTRY[name]()


def list_scenario_ids() -> list:
    """Return sorted list of registered scenario IDs."""
    return sorted(SCENARIO_REGISTRY.keys())


# Import scenarios to trigger registration
from ragix_core.memory.reporting.scenarios.summarize_content import SummarizeContentScenario  # noqa: E402,F401
from ragix_core.memory.reporting.scenarios.benchmarks import BenchmarkScenario  # noqa: E402,F401
from ragix_core.memory.reporting.scenarios.regression_min import RegressionMinScenario  # noqa: E402,F401
