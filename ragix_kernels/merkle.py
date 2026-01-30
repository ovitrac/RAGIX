"""
Merkle tree computation for pyramidal provenance.

Implements MUST M3: inputs_merkle_root
Implements MUST M5: Canonical ordering for reproducibility

From docs/SOVEREIGN_LLM_OPERATIONS.md specification.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-01-30
"""

import hashlib
import json
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def sha256(data: str) -> str:
    """Compute SHA256 hash of string data."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_inputs_merkle_root(children: List[Dict[str, Any]]) -> str:
    """
    Compute Merkle root of child inputs for cache key.

    MUST M5: Children ordered by (file_path, chunk_index) for determinism.

    Args:
        children: List of child dicts with keys:
            - file_path: Path to source file
            - chunk_index: Index within file (default 0)
            - content: Text content (optional if content_hash provided)
            - content_hash: Pre-computed hash (optional)

    Returns:
        SHA256 Merkle root as hex string
    """
    if not children:
        return sha256("")

    # Step 1: Sort children deterministically (MUST M5)
    sorted_children = sorted(
        children,
        key=lambda c: (
            c.get("file_path", ""),
            c.get("chunk_index", 0)
        )
    )

    # Step 2: Hash each child's content
    child_hashes = []
    for c in sorted_children:
        # Use pre-computed hash if available, otherwise compute
        content_hash = c.get("content_hash")
        if not content_hash:
            content = c.get("content", "")
            content_hash = sha256(content)
        child_hashes.append(content_hash)

    # Step 3: Build Merkle tree (binary tree of hashes)
    while len(child_hashes) > 1:
        # Pad with duplicate of last if odd number
        if len(child_hashes) % 2 == 1:
            child_hashes.append(child_hashes[-1])

        # Combine pairs
        child_hashes = [
            sha256(child_hashes[i] + child_hashes[i + 1])
            for i in range(0, len(child_hashes), 2)
        ]

    return child_hashes[0]


@dataclass
class NodeRef:
    """
    Reference for synthesis provenance.

    Used for tracking what an LLM call is summarizing in the pyramid.
    """
    level: str  # chunk | doc | group | domain | corpus
    node_id: str
    parents: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    children_count: int = 0
    inputs_merkle_root: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level,
            "node_id": self.node_id,
            "parents": self.parents,
            "children_ids": self.children_ids,
            "children_count": self.children_count,
            "inputs_merkle_root": self.inputs_merkle_root,
        }


def build_node_ref(
    level: str,
    node_id: str,
    parents: List[str],
    children: List[Dict[str, Any]]
) -> NodeRef:
    """
    Build a NodeRef for synthesis provenance.

    Args:
        level: Hierarchy level (chunk, doc, group, domain, corpus)
        node_id: Unique identifier for this node
        parents: List of parent node IDs
        children: List of child dicts with content/hash

    Returns:
        NodeRef with computed Merkle root
    """
    children_ids = [
        c.get("node_id", c.get("file_path", f"child_{i}"))
        for i, c in enumerate(children)
    ]

    return NodeRef(
        level=level,
        node_id=node_id,
        parents=parents,
        children_ids=children_ids,
        children_count=len(children),
        inputs_merkle_root=compute_inputs_merkle_root(children),
    )


def canonicalize_llm_request(request: Dict[str, Any]) -> str:
    """
    Produce stable JSON for cache key computation.

    Implements MUST M3: Forced caching with call_hash.

    Args:
        request: LLM request dict with model, messages, etc.

    Returns:
        Canonical JSON string (sorted keys, minimal whitespace)
    """
    canonical = {
        "model": request.get("model", ""),
        "temperature": request.get("temperature", 0.0),
        "template_id": request.get("template_id"),
        "template_version": request.get("template_version"),
        "messages": _canonicalize_messages(request.get("messages", [])),
    }
    # Sorted keys, no whitespace variance
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _canonicalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Normalize message content for stable hashing.

    Removes volatile fields like timestamps and run IDs.
    """
    result = []
    for msg in messages:
        content = msg.get("content", "")
        # Normalize whitespace (collapse multiple spaces/newlines)
        content = re.sub(r"\s+", " ", content).strip()
        # Remove volatile fields (timestamps, run IDs)
        content = re.sub(r"run_\d{8}_\d{6}_[a-f0-9]+", "RUN_ID", content)
        # Remove ISO timestamps
        content = re.sub(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            "TIMESTAMP",
            content
        )
        result.append({
            "role": msg.get("role", "user"),
            "content": content
        })
    return result


def compute_call_hash(request: Dict[str, Any]) -> str:
    """
    Compute SHA256 hash of canonical request.

    Args:
        request: LLM request dict

    Returns:
        SHA256 hex digest
    """
    canonical = canonicalize_llm_request(request)
    return sha256(canonical)


def compute_response_hash(response: str) -> str:
    """
    Compute hash of LLM response for verification.

    Args:
        response: Raw response text

    Returns:
        SHA256 hex digest
    """
    # Normalize whitespace but preserve structure
    normalized = re.sub(r"[ \t]+", " ", response)
    return sha256(normalized)


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for an LLM call.

    Combines request hash, response hash, and pyramidal context.
    """
    call_hash: str
    response_hash: str
    node_ref: Optional[NodeRef] = None
    template_id: str = ""
    template_version: str = ""
    model: str = ""
    temperature: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for cache storage."""
        return {
            "call_hash": self.call_hash,
            "response_hash": self.response_hash,
            "node_ref": self.node_ref.to_dict() if self.node_ref else None,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "model": self.model,
            "temperature": self.temperature,
        }


def build_provenance_record(
    request: Dict[str, Any],
    response: str,
    node_ref: Optional[NodeRef] = None
) -> ProvenanceRecord:
    """
    Build complete provenance record for an LLM call.

    Args:
        request: Original LLM request
        response: LLM response text
        node_ref: Optional pyramidal context

    Returns:
        ProvenanceRecord with all hashes computed
    """
    return ProvenanceRecord(
        call_hash=compute_call_hash(request),
        response_hash=compute_response_hash(response),
        node_ref=node_ref,
        template_id=request.get("template_id", ""),
        template_version=request.get("template_version", ""),
        model=request.get("model", ""),
        temperature=request.get("temperature", 0.0),
    )
