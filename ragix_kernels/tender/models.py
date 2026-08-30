"""
Records this family produces.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

One record so far, and it is a real one rather than a placeholder: an empty
dataclass proves nothing about serialisation, and the first thing to get right in
a family that will carry evidence is that its records round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ProbeResult"]


@dataclass
class ProbeResult:
    """What a store holds, as the store itself reports it.

    `dense_enabled` is a field rather than something a reader infers from a count:
    a store with no vectors because nothing was embedded and a store with no
    vectors because embedding failed are different situations, and only the store
    can tell them apart. `dense` carries its own sentence.
    """

    store_path: str
    documents: int
    chunks: int
    dense_enabled: bool
    dense: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "dense": self.dense,
            "dense_enabled": self.dense_enabled,
            "documents": self.documents,
            "store_path": self.store_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbeResult":
        return cls(
            store_path=data["store_path"], documents=data["documents"],
            chunks=data["chunks"], dense_enabled=bool(data["dense_enabled"]),
            dense=data.get("dense", ""),
        )
