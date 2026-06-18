"""
RAGIX-Sealed — warm ingestion pipeline (WP §8, Sprint 2 MVP).

Drives a document through the contract's ``document_flow`` state machine from RECEIVED to
COOLED_INDEXABLE (or BLOCKED), reusing Sprint 0 crypto+vault and Sprint 1 contracts:

  RECEIVED → QUARANTINED → FINGERPRINTED → ORIGINAL_ENCRYPTED → NORMALIZED_INTERNAL
  → METADATA_SCRUBBED → ENTITY_DETECTED → PLACEHOLDERIZED → LEAK_SCANNED
  → [HUMAN_REVIEWED_IF_REQUIRED] → COOLED_INDEXABLE        (FAIL ⇒ BLOCKED)

Every transition is validated against the loaded contract — an illegal step raises
``IngestError`` rather than silently proceeding. The original bytes are sealed immediately
(AES-256-GCM + AAD). The public result (``IngestStatus``) and the cooled document carry no
raw content: the cooled text is placeholderized and leak-scanned (SANITIZED_LLM_SAFE);
status is ORCHESTRATOR_METRICS-safe.

This MVP handles TXT/MD (dependency-free). PDF/DOCX/OCR/multimodal are deferred.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ragix_core.crypto.sealed_aead import SealedBlob, SealingAAD, open_bytes, seal_bytes

from ..vault.backend import PlaceholderPolicy
from .detect import detect_entities
from .extract import extract_text
from .ids import CaseContext, raw_sha256
from .leak_scan import scan as leak_scan


class IngestError(Exception):
    """Raised on an illegal state transition or an ingestion failure."""


@dataclass(frozen=True)
class IngestStatus:
    """Public, ORCHESTRATOR_METRICS-safe ingestion result. No raw content."""

    doc_id: str
    state: str
    source_kind: str
    entity_counts: Dict[str, int]
    leak_verdict: str
    human_review_required: bool
    pages: int = 1

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "state": self.state,
            "source_kind": self.source_kind,
            "entity_counts": dict(self.entity_counts),
            "leak_verdict": self.leak_verdict,
            "human_review_required": self.human_review_required,
            "pages": self.pages,
        }


@dataclass(frozen=True)
class CooledDocument:
    """A cooled, indexable document. ``text`` is placeholderized + leak-scanned."""

    doc_id: str
    text: str  # SANITIZED_LLM_SAFE placeholderized text
    sanitized_metadata: Dict[str, Any]
    entity_counts: Dict[str, int]


class SealedIngestor:
    """Stateful ingestion engine for one or more cases (in-memory MVP store)."""

    def __init__(self, contracts: Any, vault: Any) -> None:
        self.contracts = contracts
        self.vault = vault
        self._flow = contracts.state_machine["document_flow"]
        self._states = self._flow["states"]
        self._entity_classes = contracts.placeholder_schema["entity_classes"]
        # in-memory stores (persistence is Sprint 2+ ops work)
        self._sealed_originals: Dict[str, SealedBlob] = {}
        self._original_aad: Dict[str, SealingAAD] = {}
        self._cooled: Dict[str, CooledDocument] = {}

    # -- state machine helper ----------------------------------------------------------

    def _advance(self, current: str, target: str) -> str:
        allowed = (self._states.get(current) or {}).get("to", [])
        if target not in allowed:
            raise IngestError(f"illegal transition {current} -> {target}")
        return target

    def _policy_for(self, etype: str) -> PlaceholderPolicy:
        spec = self._entity_classes.get(etype, {})
        return PlaceholderPolicy(
            entity_type=etype,
            fmt=spec.get("format", "[{etype}_{n:03d}]"),
            role_visible=bool(spec.get("role_visible", False)),
            schema_version=self.contracts.placeholder_schema.get(
                "schema_version", "placeholder-schema-0.1"
            ),
        )

    # -- main entry point --------------------------------------------------------------

    def ingest(
        self,
        ctx: CaseContext,
        raw_bytes: bytes,
        source_kind: str,
        ingestion_counter: int,
    ) -> IngestStatus:
        """Ingest one document end to end. Returns a sanitized status."""
        state = self._flow["initial"]  # RECEIVED

        # Fingerprint + opaque id.
        rawh = raw_sha256(raw_bytes)
        doc_id = ctx.doc_id(rawh, ingestion_counter)
        state = self._advance(state, "QUARANTINED")
        state = self._advance(state, "FINGERPRINTED")

        # Seal the original immediately (retained original), AAD-bound to this state.
        aad = SealingAAD(
            case_id=ctx.case_id,
            doc_id=doc_id,
            raw_sha256=rawh,
            policy_version=ctx.policy_version,
            placeholder_schema_version=ctx.placeholder_schema_version,
            ingestion_state="ORIGINAL_ENCRYPTED",
        )
        self._sealed_originals[doc_id] = seal_bytes(raw_bytes, aad, ctx.key)
        self._original_aad[doc_id] = aad
        state = self._advance(state, "ORIGINAL_ENCRYPTED")

        # Normalize (extract text). Raw text is INTERNAL (never returned).
        text = extract_text(raw_bytes, source_kind)
        state = self._advance(state, "NORMALIZED_INTERNAL")

        # Metadata scrub (MVP: minimal sanitized metadata; no raw filename/path).
        sanitized_meta = {
            "document_id": doc_id,
            "document_kind": source_kind.lower().lstrip("."),
            "pages": 1,
            "ocr_required": False,
            "language": None,
        }
        state = self._advance(state, "METADATA_SCRUBBED")

        # Detect high-confidence entities.
        detections = detect_entities(text, self.contracts.placeholder_schema)
        state = self._advance(state, "ENTITY_DETECTED")

        # Placeholderize via the vault (raw values sealed in the entity vault).
        ptext, refs, raw_values = self._placeholderize(ctx.case_id, text, detections)
        entity_counts = dict(Counter(r.entity_type for r in refs))
        state = self._advance(state, "PLACEHOLDERIZED")

        # Leak scan (egress filter). FAIL ⇒ BLOCKED.
        verdict = leak_scan(ptext, raw_values, self.contracts.placeholder_schema)
        if verdict.verdict == "FAIL":
            state = self._advance(state, "BLOCKED")
            return IngestStatus(
                doc_id=doc_id, state=state, source_kind=source_kind,
                entity_counts=entity_counts, leak_verdict=verdict.verdict,
                human_review_required=False,
            )
        state = self._advance(state, "LEAK_SCANNED")

        # Human review gate (v0: required only when the scan is UNCERTAIN).
        human_required = verdict.verdict == "UNCERTAIN"
        if human_required:
            state = self._advance(state, "HUMAN_REVIEWED_IF_REQUIRED")
        state = self._advance(state, "COOLED_INDEXABLE")

        self._cooled[doc_id] = CooledDocument(
            doc_id=doc_id, text=ptext, sanitized_metadata=sanitized_meta,
            entity_counts=entity_counts,
        )
        return IngestStatus(
            doc_id=doc_id, state=state, source_kind=source_kind,
            entity_counts=entity_counts, leak_verdict=verdict.verdict,
            human_review_required=human_required,
        )

    # -- placeholderization ------------------------------------------------------------

    def _placeholderize(self, case_id: str, text: str, detections: List[Any]):
        """Rebuild text with vault placeholders; return (text, refs, raw_values)."""
        parts: List[str] = []
        refs = []
        raw_values: List[str] = []
        idx = 0
        for d in sorted(detections, key=lambda x: x.start):
            parts.append(text[idx:d.start])
            ref = self.vault.create_placeholder(
                case_id, d.entity_type, d.value, role=None, policy=self._policy_for(d.entity_type)
            )
            parts.append(ref.placeholder)
            refs.append(ref)
            raw_values.append(d.value)
            idx = d.end
        parts.append(text[idx:])
        return "".join(parts), refs, raw_values

    # -- accessors ---------------------------------------------------------------------

    def get_cooled(self, doc_id: str) -> Optional[CooledDocument]:
        """Return the cooled (placeholderized, leak-scanned) document, or None."""
        return self._cooled.get(doc_id)

    def open_original(self, ctx: CaseContext, doc_id: str) -> bytes:
        """Re-open the sealed original — INTERNAL/human path only (requires the case key)."""
        blob = self._sealed_originals.get(doc_id)
        if blob is None:
            raise IngestError(f"unknown doc_id: {doc_id}")
        return open_bytes(blob, self._original_aad[doc_id], ctx.key)
