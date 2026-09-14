"""tender.claims — ClaimRecord 1.0, scoped to the deadline slice (WP-3D 3.2a-D).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Implemented as base/WP_TENDER_LAYERS_20260910.md §I.7 states it, signed by the lead on 2026-09-10
(D-0021), and no more:

  claim_id       sha256 over (field, value.normalized, provenance.sources)
  field          offer_deadline | questions_deadline | visit_window_start | visit_window_end
  value          type: date | datetime · raw: the exact text · normalized: ISO 8601, local time as written
  provenance     origin: observed | derived
                 sources: one span for an observed value, one per operand for a derived one
                 derivation: none, or the expression over the operands
                 channel: the grammar and its version
  applicability  an Applicability 1.1 (tender.records)

Two readings of the block are made here and stated. ``value.raw`` is the exact text of one of the
sources: the observed span itself, or, for a derived value, the span that states the field (the
clause for the questions deadline, the start as written for the visit window); its sha256 is
checked against the sources. The sources of a derived claim are listed in the order of its operands.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

from .records import Applicability

#: 1.1 as signed in WP §2.2 (D-0022), in code as D-0024. 1.0 is a subset: every 1.0 claim keeps its id.
CLAIM_SCHEMA_VERSION = "1.1"
FIELDS = ("offer_deadline", "questions_deadline", "visit_window_start", "visit_window_end")
OBSERVED_TYPES = ("date", "datetime")
#: the statements an interpreted claim may make — about children, never a new critical value. These
#: four names render §2.2 item 3 (a conflict, an authority rank, a coverage judgement, a relevance); they
#: are the agent's naming and fall to the lead's falsification with the rest of the dimension enums.
INTERPRETED_FIELDS = ("conflict", "authority_rank", "coverage", "relevance")
INTERPRETED_TYPES = ("flag", "rank", "judgement", "label")
VALUE_TYPES = OBSERVED_TYPES + INTERPRETED_TYPES
#: the prohibition of the signed block: no interpreted claim carries a value of these types
CRITICAL_TYPES = ("date", "datetime", "amount", "percentage", "duration", "quantity")
ORIGINS = ("observed", "derived", "aggregated", "interpreted")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_ISO = {"date": re.compile(r"\d{4}-\d{2}-\d{2}"), "datetime": re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceSpan:
    """One exact span of the store: {doc_id, chunk_id, node_ids, char_start, char_end, span_sha256}."""

    doc_id: str
    chunk_id: str
    node_ids: tuple[str, ...]
    char_start: int
    char_end: int
    span_sha256: str

    def __post_init__(self) -> None:
        if not self.doc_id or not self.chunk_id:
            raise ValueError("source span without its document or chunk (rule 2)")
        if not self.node_ids:
            raise ValueError("source span without node ids")
        if not 0 <= self.char_start < self.char_end:
            raise ValueError(f"offsets {self.char_start}..{self.char_end} are not a span")
        if not _HEX64.fullmatch(self.span_sha256):
            raise ValueError("span_sha256 is not a sha256")


@dataclass(frozen=True)
class ClaimValue:
    type: str
    raw: str
    normalized: str

    def __post_init__(self) -> None:
        if self.type not in VALUE_TYPES:
            raise ValueError(f"value type {self.type!r} outside vocabulary")
        if not self.raw:
            raise ValueError("value without its raw text")
        if self.type in _ISO:                               # a calendar value is ISO and real; a label is not
            if not _ISO[self.type].fullmatch(self.normalized or ""):
                raise ValueError(f"normalized {self.normalized!r} is not ISO 8601 for a {self.type}")
            (dt.date if self.type == "date" else dt.datetime).fromisoformat(self.normalized)


@dataclass(frozen=True)
class Provenance:
    origin: str
    sources: tuple  # SourceSpan for observed and derived; child claim ids for aggregated and interpreted
    derivation: Optional[str]
    channel: str

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f"origin {self.origin!r} outside vocabulary")
        if not self.channel:
            raise ValueError("provenance without its channel")
        if self.origin == "observed":
            if len(self.sources) != 1:
                raise ValueError("an observed value keeps exactly one span")
            if self.derivation is not None:
                raise ValueError("an observed value has no derivation")
        elif self.origin == "derived":
            if not self.derivation:
                raise ValueError("a derived value keeps its expression")
            if not self.sources:
                raise ValueError("a derived value keeps one span per operand")
        else:
            # 1.1: aggregated and interpreted rest on claims, not spans — they reference and copy nothing
            if not self.sources or not all(isinstance(x, str) and _HEX64.fullmatch(x) for x in self.sources):
                raise ValueError(f"an {self.origin} claim references its children's claim ids, and only them")
            if self.origin == "aggregated" and self.derivation is not None:
                raise ValueError("an aggregated claim carries its children's value; it has no derivation")
            if self.origin == "interpreted" and not self.derivation:
                raise ValueError("an interpreted claim names the rule of its judgement")
        if self.origin in ("observed", "derived") and not all(isinstance(x, SourceSpan) for x in self.sources):
            raise ValueError(f"an {self.origin} value keeps spans of the store")


@dataclass
class ClaimRecord:
    field: str
    value: ClaimValue
    provenance: Provenance
    applicability: Applicability
    schema_version: str = CLAIM_SCHEMA_VERSION
    claim_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version not in ("1.0", "1.1"):
            raise ValueError(f"schema_version {self.schema_version!r} is neither 1.0 nor 1.1")
        origin = self.provenance.origin
        if self.schema_version == "1.0" and origin not in ("observed", "derived"):
            raise ValueError(f"a 1.0 claim cannot be {origin}: that origin arrived with 1.1")
        if origin == "interpreted":
            if self.value.type in CRITICAL_TYPES:
                raise ValueError(f"an interpreted claim may not carry a value of type {self.value.type} "
                                 "(ClaimRecord 1.1, D-0022): a statement about children never becomes the "
                                 "source of a critical value")
            if self.field not in INTERPRETED_FIELDS:
                raise ValueError(f"field {self.field!r} outside the interpreted vocabulary")
        elif self.field not in FIELDS:
            raise ValueError(f"field {self.field!r} outside the slice's vocabulary")
        if origin in ("observed", "derived") and self.value.type not in OBSERVED_TYPES:
            raise ValueError(f"an {origin} value is a calendar value, not a {self.value.type}")
        if not isinstance(self.applicability, Applicability):
            raise TypeError("applicability must be an Applicability")
        if origin in ("observed", "derived") and \
                sha256_text(self.value.raw) not in {s.span_sha256 for s in self.provenance.sources}:
            raise ValueError("value.raw is not the exact text of a source span")
        identity = self.identity()
        if self.claim_id and self.claim_id != identity:
            raise ValueError("claim_id does not match the claim")
        self.claim_id = identity

    def identity(self) -> str:
        # a SourceSpan serialises exactly as in 1.0, so no 1.0 id moves; a claim id stands as itself
        payload = [self.field, self.value.normalized,
                   [x if isinstance(x, str) else asdict(x) for x in self.provenance.sources]]
        return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
