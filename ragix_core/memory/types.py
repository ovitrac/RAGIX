"""
Memory Data Model — First-Class Objects

Defines the canonical memory item schema, proposals, events, and links.
All memory items are immutable once written; updates create revisions.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

# ---------------------------------------------------------------------------
# Type aliases (Literal unions for validation)
# ---------------------------------------------------------------------------

MemoryTier = Literal["stm", "mtm", "ltm"]
MemoryType = Literal[
    "fact", "decision", "definition", "constraint",
    "pattern", "todo", "pointer", "note",
]
ValidationState = Literal["unverified", "verified", "contested", "retracted"]
SourceKind = Literal["chat", "doc", "tool", "mixed"]

# Valid values for runtime checks
VALID_TIERS: set = {"stm", "mtm", "ltm"}
VALID_TYPES: set = {
    "fact", "decision", "definition", "constraint",
    "pattern", "todo", "pointer", "note",
}
VALID_VALIDATION_STATES: set = {"unverified", "verified", "contested", "retracted"}
VALID_SOURCE_KINDS: set = {"chat", "doc", "tool", "mixed"}


def _now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _generate_id(prefix: str = "MEM") -> str:
    """Generate a unique memory ID with prefix."""
    short = uuid.uuid4().hex[:12]
    return f"{prefix}-{short}"


def content_hash(text: str) -> str:
    """SHA-256 content hash with prefix."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class MemoryProvenance:
    """Tracks the origin of a memory item."""

    source_kind: SourceKind = "chat"
    source_id: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    content_hashes: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize provenance to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryProvenance:
        """Deserialize provenance from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Memory Item (canonical)
# ---------------------------------------------------------------------------

@dataclass
class MemoryItem:
    """
    Canonical memory item — first-class object with full provenance.

    Rules:
    - content must be concise; long evidence uses type="pointer" with chunk refs.
    - provenance is mandatory for MTM/LTM; STM allows minimal provenance.
    - Updates create revisions (never overwrite without history).
    """

    id: str = field(default_factory=lambda: _generate_id("MEM"))
    tier: MemoryTier = "stm"
    type: MemoryType = "note"
    title: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)  # {rel, to}
    provenance: MemoryProvenance = field(default_factory=MemoryProvenance)
    confidence: float = 0.5
    validation: ValidationState = "unverified"
    scope: str = "project"
    expires_at: Optional[str] = None
    usage_count: int = 0
    last_used_at: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    # Domain-specific identifiers
    rule_id: Optional[str] = None  # e.g. "RIE-RHEL-042" for RIE documents
    # V3.0: Corpus identity for cross-corpus operations
    corpus_id: Optional[str] = None  # e.g. "corp_energy-rie-2026Q1"
    # Supersession tracking
    superseded_by: Optional[str] = None
    archived: bool = False
    # V3.3: Injectable flag — prevents quarantined items from entering recall/inject
    injectable: bool = True

    def __post_init__(self):
        """Validate tier, type, and validation state; coerce dict provenance."""
        if self.tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier: {self.tier!r}")
        if self.type not in VALID_TYPES:
            # LLM may produce unknown types — map to closest or default to "note"
            _TYPE_MAP = {"process": "pattern", "rule": "constraint", "requirement": "constraint"}
            self.type = _TYPE_MAP.get(self.type, "note")
        if self.validation not in VALID_VALIDATION_STATES:
            raise ValueError(f"Invalid validation state: {self.validation!r}")
        if isinstance(self.provenance, dict):
            self.provenance = MemoryProvenance.from_dict(self.provenance)

    @property
    def content_hash(self) -> str:
        """Hash of the canonical content."""
        return content_hash(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (JSON-safe)."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryItem:
        """Deserialize from dict."""
        data = dict(d)
        if "provenance" in data and isinstance(data["provenance"], dict):
            data["provenance"] = MemoryProvenance.from_dict(data["provenance"])
        # Filter to known fields
        known = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_json(self) -> str:
        """Serialize to indented JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def touch(self) -> None:
        """Update usage tracking."""
        self.usage_count += 1
        self.last_used_at = _now_iso()
        self.updated_at = _now_iso()

    def format_inject(self) -> str:
        """Format for context injection into LLM prompt."""
        tag_str = ", ".join(self.tags) if self.tags else "none"
        prov = f"{self.provenance.source_kind}:{self.provenance.source_id}"
        return (
            f"[MEMORY: {self.id} | {self.type} | {self.tier} | "
            f"tags={tag_str} | provenance={prov}]\n"
            f"{self.title}\n"
            f"{self.content}\n"
            f"[/MEMORY]"
        )

    def format_catalog_entry(self) -> Dict[str, Any]:
        """Format for memory catalog (frontier)."""
        return {
            "id": self.id,
            "title": self.title,
            "tags": self.tags,
            "tier": self.tier,
            "type": self.type,
            "confidence": self.confidence,
            "validation": self.validation,
        }


# ---------------------------------------------------------------------------
# Memory Proposal (what the LLM emits, pre-governance)
# ---------------------------------------------------------------------------

@dataclass
class MemoryProposal:
    """
    A memory candidate proposed by the LLM.

    Not stored directly — must pass through policy.py first.
    """

    type: MemoryType = "note"
    title: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    why_store: str = ""
    provenance_hint: Dict[str, str] = field(default_factory=dict)
    scope: str = "project"
    rule_id: Optional[str] = None  # e.g. "RIE-RHEL-042"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize proposal to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryProposal:
        """Deserialize proposal from a dictionary, filtering to known fields."""
        known = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_memory_item(
        self,
        tier: MemoryTier = "stm",
        scope: Optional[str] = None,
        confidence: float = 0.5,
    ) -> MemoryItem:
        """Convert accepted proposal to a MemoryItem."""
        prov_hint = self.provenance_hint or {}
        provenance = MemoryProvenance(
            source_kind=prov_hint.get("source_kind", "chat"),
            source_id=prov_hint.get("source_id", ""),
            chunk_ids=prov_hint.get("chunk_ids", []),
            content_hashes=prov_hint.get("content_hashes", []),
        )
        return MemoryItem(
            tier=tier,
            type=self.type,
            title=self.title,
            content=self.content,
            tags=list(self.tags),
            provenance=provenance,
            confidence=confidence,
            scope=scope or self.scope,
            rule_id=self.rule_id,
        )


# ---------------------------------------------------------------------------
# Memory Event (audit log entry)
# ---------------------------------------------------------------------------

@dataclass
class MemoryEvent:
    """Audit log entry for any memory operation."""

    id: str = field(default_factory=lambda: _generate_id("EVT"))
    action: str = ""  # e.g. "write", "read", "update", "search", "consolidate"
    item_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Memory Link (inter-item relationships)
# ---------------------------------------------------------------------------

@dataclass
class MemoryLink:
    """Typed link between two memory items."""

    src_id: str = ""
    dst_id: str = ""
    rel: str = ""  # e.g. "supports", "contradicts", "refines", "supersedes"
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize link to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryLink:
        """Deserialize link from a dictionary."""
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# V3.0: Corpus Metadata
# ---------------------------------------------------------------------------

@dataclass
class CorpusMetadata:
    """
    Metadata for a corpus version — enables cross-corpus drift detection.

    A corpus is a versioned collection of documents (e.g., "corp_energy-rie-2026Q1").
    Parent corpus enables lineage tracking (e.g., Q1 evolved from Q4).
    """

    corpus_id: str = ""             # e.g. "corp_energy-rie-2026Q1"
    corpus_label: str = ""          # human-readable, e.g. "CORP-ENERGY RIE — Q1 2026"
    parent_corpus_id: Optional[str] = None  # e.g. "corp_energy-rie-2025Q4"
    doc_count: int = 0
    item_count: int = 0
    scope: str = "project"
    ingested_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize corpus metadata to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CorpusMetadata:
        """Deserialize corpus metadata from a dictionary."""
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})
