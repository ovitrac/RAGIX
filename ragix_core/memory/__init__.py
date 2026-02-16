"""
RAGIX Memory Subsystem

Persistent, taggable, auditable, policy-governed memory for LLM-assisted workflows.
Implements Q*-style QA search, hybrid retrieval, and memory palace browsing.

Architecture:
    types.py     - Data model (MemoryItem, MemoryProvenance, etc.)
    config.py    - Configuration dataclasses
    store.py     - SQLite persistent store with revision history
    embedder.py  - Embedding backends (Ollama, mock)
    policy.py    - Write governance (secret/injection blocks)
    proposer.py  - LLM output parsing for memory proposals
    recall.py    - Hybrid retrieval engine (tags + embeddings + provenance)
    qsearch.py   - Q*-style agenda search controller
    palace.py    - Memory palace location index + browse API
    consolidate.py - STM -> MTM -> LTM promotion pipeline
    tools.py     - JSON tool API (memory.*) dispatcher
    middleware.py - Chat pipeline integration hooks
    cli.py       - CLI utilities for dev/debug

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from ragix_core.memory.types import (
    MemoryTier,
    MemoryType,
    ValidationState,
    MemoryProvenance,
    MemoryItem,
    MemoryProposal,
    MemoryEvent,
    MemoryLink,
)

__all__ = [
    "MemoryTier",
    "MemoryType",
    "ValidationState",
    "MemoryProvenance",
    "MemoryItem",
    "MemoryProposal",
    "MemoryEvent",
    "MemoryLink",
]
