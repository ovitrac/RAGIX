"""tender.records — QuestionRecord / AnswerRecord / Applicability (v1 DRAFT).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Frozen-contract candidates (CLAUDE.md §5.1 — freeze order: objects BEFORE
retrieval). Status: **1.0-draft**, RED — the freeze is the lead's signature,
not this file. Fields are the minimum demanded by the restart doctrine
(NOTE_DOCTRINE_RESTART_20260820): machine-annotation channel + trace on
every record, precedence keys as ORDERED NAMED FIELDS (never a blended
score), applicability separated from content.

Non-negotiables encoded here:
  - rule 1: an AnswerRecord cannot exist without its source_question_id
    (constructor raises);
  - rule 2: provenance is mandatory on both records;
  - rule 14: ``outcome`` is a PRECEDENCE key (reuse priority), never a
    per-answer quality label;
  - temporal doctrine: absence of a date is ``undated``, not timeless.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "1.0-draft"

#: machine-annotation channels (closed vocabulary; extending it is a decision)
CHANNELS = (
    "twin-diff-chains",     # xlsx blank/filled diff + projection lattice
    "anchors-only-chains",  # xlsx filled-only anchors (incl. label-content)
    "docx-chains",          # docx data-form chains (grid_headers adapter)
    "pdf-colon-split",      # 'Label : Valeur' split (S2, declared filters)
    "llm-schema",           # harnessed LLM lane (quote-verified)
    "human",                # audit / gold countersignature
)

#: tender outcome — precedence only (rule 14)
OUTCOMES = ("won", "lost", "pending", "unknown")

#: Applicability version: 1.0 plus the two fields of §I.7 (signed 2026-09-10, D-0021)
APPLICABILITY_VERSION = "1.1"
AUTHORITY_SPAN_KEYS = ("doc_id", "chunk_id", "char_start", "char_end", "span_sha256")


@dataclass
class Applicability:
    """Where and when a record may be reused. Separate from content."""

    project: str                                   # e.g. "PROJECT-A_LOT-6"
    entity: Optional[str] = None                   # Adservio / a partner / ...
    valid_from: Optional[str] = None               # ISO date or None = undated
    valid_until: Optional[str] = None
    reuse_policy: Optional[str] = None             # None = unrestricted (declared)
    outcome: str = "unknown"                       # precedence key, rule 14
    #: 1.1, the deadline slice's two additions (WP_TENDER_LAYERS §I.7, D-0021); None = not stated
    clause_scope: Optional[dict[str, Any]] = None  # {"pieces": [doc_id, ...] | "consultation", "lots": "all" | [...]}
    authority: Optional[dict[str, Any]] = None     # {"rank": int, 1 governs, "established_by": [span, ...]}

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome {self.outcome!r} outside vocabulary")
        if self.clause_scope is not None:
            if set(self.clause_scope) != {"pieces", "lots"}:
                raise ValueError("clause_scope holds pieces and lots, nothing else")
            pieces, lots = self.clause_scope["pieces"], self.clause_scope["lots"]
            if not (pieces == "consultation" or (isinstance(pieces, list) and pieces
                                                 and all(isinstance(x, str) and x for x in pieces))):
                raise ValueError("clause_scope.pieces: a list of doc ids, or the whole consultation")
            if not (lots == "all" or (isinstance(lots, list) and lots)):
                raise ValueError("clause_scope.lots: all, or a list")
        if self.authority is not None:
            if set(self.authority) != {"rank", "established_by"}:
                raise ValueError("authority holds rank and established_by, nothing else")
            rank = self.authority["rank"]
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
                raise ValueError("authority.rank: a positive integer, 1 governs")
            spans = self.authority["established_by"]
            if not spans or any(set(s) != set(AUTHORITY_SPAN_KEYS) for s in spans):
                raise ValueError(f"authority.established_by: spans of {AUTHORITY_SPAN_KEYS}")


@dataclass
class QuestionRecord:
    """One question extracted from the corpus, with full provenance.

    ``text`` is the composed question (display convention: deepest column
    header | row labels outer->inner | detail rungs — F-M3.3, pending);
    ``components`` keeps the exact source refs so the composition is
    re-derivable and re-anchorable.
    """

    text: str
    channel: str
    source_path: str                               # doc path relative to corpus
    applicability: Applicability
    components: list[dict[str, Any]] = field(default_factory=list)
    #: grid coordinates (lead, 2026-08-20): axis -> {value, channel, trace}.
    #: The axis taxonomy is a frozen contract of its own (mined from the
    #: corpus, then lead-frozen — the glossary lesson); "unknown" is a valid
    #: coordinate on every axis. Populated by the classification lane, empty
    #: at extraction.
    criteria: dict[str, Any] = field(default_factory=dict)
    locator: dict[str, Any] = field(default_factory=dict)  # sheet/page/cells...
    source_format: Optional[str] = None
    kernel_versions: dict[str, str] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    qid: str = ""                                  # derived if empty

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(f"channel {self.channel!r} outside vocabulary")
        if not self.text or not self.text.strip():
            raise ValueError("QuestionRecord without text")
        if not self.source_path:
            raise ValueError("QuestionRecord without provenance (rule 2)")
        if not self.qid:
            key = f"{self.source_path}|{self.locator}|{self.text}"
            self.qid = "q_" + hashlib.sha256(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerRecord:
    """One answer, forever bound to its originating question (rule 1)."""

    text: str
    source_question_id: str                        # MANDATORY — rule 1
    channel: str
    source_path: str
    applicability: Applicability
    locator: dict[str, Any] = field(default_factory=dict)
    answer_type: str = "elaborated"                # "needle" | "elaborated"
    date: Optional[str] = None                     # None = undated, NOT timeless
    kernel_versions: dict[str, str] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    aid: str = ""

    def __post_init__(self) -> None:
        if not self.source_question_id:
            raise ValueError("AnswerRecord without source question (rule 1)")
        if self.channel not in CHANNELS:
            raise ValueError(f"channel {self.channel!r} outside vocabulary")
        if not self.source_path:
            raise ValueError("AnswerRecord without provenance (rule 2)")
        if self.answer_type not in ("needle", "elaborated"):
            raise ValueError(f"answer_type {self.answer_type!r} outside vocabulary")
        if not self.aid:
            key = f"{self.source_path}|{self.locator}|{self.source_question_id}"
            self.aid = "a_" + hashlib.sha256(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------- DocumentFacts

#: Where the policy facts live inside the substrate's own document record.
#: One key, so a reader knows where to look and the kernel's `meta` stays
#: otherwise free.
FACTS_KEY = "tender_facts"

#: The routing classes, closed. A class outside this list is a defect, not a
#: new kind of document.
ROUTING_CLASSES = ("P1", "P2", "P3", "D1", "D2", "X1", "M1", "E1")


@dataclass
class DocumentFacts:
    """The three document facts the retrieval policy needs and the substrate
    does not carry (§5.1 sign-off, 2026-08-31).

    **`routing_class`, never `doc_class`.** The substrate's `doc_class` is the
    FILE FORMAT; this is the structural class that says how a document must be
    read — `P2` is "a pdf that declares no outline", not "a pdf". Two records
    with one field name and two referents is the trap this rename exists to
    prevent, and it would sit exactly on the boundary this lane draws.

    **`document_date` is the document's own date**, and absence is `undated`
    rather than timeless (temporal doctrine). `digested_at` is never a
    substitute: it records when digestion ran, which is a fact about the run.

    Persisted inside the substrate record's `meta` under `FACTS_KEY` rather
    than in a second record: the store's `DocumentRecord` is the storage
    contract and this side does not fork it.
    """

    routing_class: str
    document_date: Optional[str] = None
    quality: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.routing_class not in ROUTING_CLASSES:
            raise ValueError(
                f"routing_class {self.routing_class!r} outside {ROUTING_CLASSES}")

    @property
    def dated(self) -> bool:
        """Undated is a state, not a missing value."""
        return bool(self.document_date)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def of(cls, document: Any) -> Optional["DocumentFacts"]:
        """Read the facts from a substrate document record, or None.

        Fails closed on a document that carries none: a policy that silently
        assumed a class would filter on a guess.
        """
        meta = getattr(document, "meta", None) or {}
        raw = meta.get(FACTS_KEY)
        if not isinstance(raw, dict) or "routing_class" not in raw:
            return None
        return cls(routing_class=raw["routing_class"],
                   document_date=raw.get("document_date"),
                   quality=dict(raw.get("quality") or {}))

    def into(self, meta: dict[str, Any]) -> dict[str, Any]:
        """The substrate `meta` with these facts written into it."""
        return {**meta, FACTS_KEY: self.to_dict()}
