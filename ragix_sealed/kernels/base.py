"""
RAGIX-Sealed — inventory kernel base (WP §14-§16, Sprint 4).

Level-1 (Inventory) kernels describe the cooled corpus *without interpretation*: counts,
distributions, quality, review burden. They consume the sanitized per-document records
(``IngestStatus``) and emit ORCHESTRATOR_METRICS-safe results — opaque ids and numbers
only, never content.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..ingest.pipeline import IngestStatus

# States in which a document is considered cooled/indexable (default; overridden from
# the contract when available).
_DEFAULT_COOLED = ("COOLED_INDEXABLE", "REASONABLE")
_BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class CorpusView:
    """Read model the inventory kernels operate on. Carries only sanitized records."""

    documents: List[IngestStatus]
    cooled_states: Sequence[str] = _DEFAULT_COOLED
    chunk_count: Optional[int] = None

    @classmethod
    def from_statuses(
        cls,
        statuses: Sequence[IngestStatus],
        contracts: Any = None,
        chunk_count: Optional[int] = None,
    ) -> "CorpusView":
        cooled = _DEFAULT_COOLED
        if contracts is not None:
            cooled = tuple(
                contracts.state_machine.get("document_flow", {}).get("cooled_states", _DEFAULT_COOLED)
            )
        return cls(documents=list(statuses), cooled_states=cooled, chunk_count=chunk_count)

    def is_cooled(self, status: IngestStatus) -> bool:
        return status.state in self.cooled_states

    def is_blocked(self, status: IngestStatus) -> bool:
        return status.state == _BLOCKED


@dataclass(frozen=True)
class KernelResult:
    """An inventory kernel result — metrics only, ORCHESTRATOR_METRICS-safe."""

    kernel: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        return {"kernel": self.kernel, "metrics": self.metrics}


class InventoryKernel(ABC):
    """A Level-1 inventory kernel. Produces metrics, never interpretive output."""

    name: str = "inventory"

    @abstractmethod
    def run(self, view: CorpusView) -> KernelResult:
        ...
