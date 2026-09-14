"""tender.requirements — ``RequirementRecord`` (v0-draft, RED).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

The derived object of the requirement lane (``DESIGN_REQUIREMENT_LANE_20260821``
§3). Sibling of ``tender.records`` on purpose: ``QuestionRecord`` /
``AnswerRecord`` are 1.0-draft freeze candidates, this one is 0-draft — mixing
two freeze states in one file would blur which contract is under signature.

**Nothing extracts at scale before the lead signs** (gate GR3-0b). Building the
object, its constructors and its unit tests is GREEN work.

Two scales, one object (DESIGN §4 — the lead's answer to the granularity
problem):

  - ``coarse``: a theme CHOSEN in the closed mined vocabulary
    (``tender.vocabulary``) or ``unknown``. No predicate, no parent: at this
    scale splitting is impossible by construction.
  - ``fine``: a free normalized predicate under three hard constraints —
    (i) one continuous verbatim span, (ii) exactly one modality, (iii) a
    MANDATORY ``parent_rid`` pointing at a coarse record. Split/merge then
    show up as observable 1↔n alignments instead of silent drift.

Non-negotiables encoded here:
  - rule 2: no derived object without provenance (``source_kind``,
    ``source_id``, ``source_path`` are mandatory);
  - rule 9 by construction: ``from_extraction`` refuses a span that is not a
    whitespace-normalized substring of the payload the extractor was shown —
    the caller counts the drop and abstains, it never repairs;
  - rule 12: the vocabulary version is stamped in ``kernel_versions`` so a
    re-mined vocabulary can never masquerade as the frozen one;
  - closed vocabularies everywhere, ``unknown`` always allowed.

Divergence from DESIGN §3 proposed here and flagged for the lead: ``implicit``
is a first-class boolean rather than a trace key (DESIGN §7.3 — a BPU cell
implies "price per profile" without stating it; the span is then structural,
inherited from the M3 anchor chain). A gate must be able to filter on it
without parsing a free-form trace.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from . import vocabulary
from .records import Applicability

SCHEMA_VERSION = "0-draft"

SCALES = ("coarse", "fine")

#: closed modality vocabulary (DESIGN §3; 7 values + ``unknown`` is one of them)
MODALITIES = ("must", "should", "describe", "prove", "price", "commit",
              "unknown")

#: what the requirement was read from
SOURCE_KINDS = ("question", "answer", "source_doc")

#: annotation channels of this lane (baseline / LLM / human countersignature)
REQUIREMENT_CHANNELS = ("vocab-tagger", "llm-schema", "human")

#: vocabulary file used for theme validation; None = the shipped v0-draft
_VOCAB_PATH: Optional[str] = None


def use_vocabulary(path: Optional[str]) -> None:
    """Point theme validation at another vocabulary version (tests, re-mining)."""
    global _VOCAB_PATH
    _VOCAB_PATH = path


class SpanNotVerbatim(ValueError):
    """The evidence span is not a substring of the payload (rule 9 guard)."""


def norm_span(s: str) -> str:
    """Whitespace-normalized form used for verbatim verification.

    Identical convention to ``scripts/triage_llm._norm`` (the proven harness):
    collapse whitespace, strip, lowercase. Nothing else is normalized — an
    extractor that rewrites the text fails the guard, which is the point.
    """
    return re.sub(r"\s+", " ", s).strip().lower()


def span_verbatim(span: str, payload: str) -> bool:
    n = norm_span(span)
    return bool(n) and n in norm_span(payload)


@dataclass
class RequirementRecord:
    """One requirement predicate, at one scale, with its verbatim span."""

    scale: str
    evidence_span: str                              # VERBATIM (rule 9)
    source_kind: str
    source_id: str                                  # qid / aid / node-id
    source_path: str
    applicability: Applicability
    channel: str
    theme: str = ""                                 # coarse coordinate
    predicate: str = ""                             # fine scale only
    modality: str = "unknown"
    parent_rid: Optional[str] = None                # mandatory when fine
    implicit: bool = False                          # structural span (§7.3)
    locator: dict[str, Any] = field(default_factory=dict)
    kernel_versions: dict[str, str] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    rid: str = ""                                   # derived if empty

    def __post_init__(self) -> None:
        if self.scale not in SCALES:
            raise ValueError(f"scale {self.scale!r} outside vocabulary")
        if self.modality not in MODALITIES:
            raise ValueError(f"modality {self.modality!r} outside vocabulary")
        if self.channel not in REQUIREMENT_CHANNELS:
            raise ValueError(f"channel {self.channel!r} outside vocabulary")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"source_kind {self.source_kind!r} outside vocabulary")

        # rule 2 — provenance is not optional on a derived object
        if not self.source_id:
            raise ValueError("RequirementRecord without source_id (rule 2)")
        if not self.source_path:
            raise ValueError("RequirementRecord without source_path (rule 2)")
        if not isinstance(self.applicability, Applicability):
            raise ValueError("RequirementRecord without Applicability (rule 2)")

        if not self.evidence_span or not self.evidence_span.strip():
            raise ValueError("RequirementRecord without evidence span")

        if self.scale == "coarse":
            if not self.theme:
                raise ValueError("coarse requirement without theme "
                                 "(use 'unknown' to abstain)")
            if self.predicate:
                raise ValueError("coarse requirement carries a predicate — "
                                 "the coarse scale cannot be split (DESIGN §4)")
            if self.parent_rid is not None:
                raise ValueError("coarse requirement carries a parent_rid")
        else:                                        # fine
            if not self.predicate or not self.predicate.strip():
                raise ValueError("fine requirement without predicate")
            if not self.parent_rid:
                raise ValueError("fine requirement without parent_rid "
                                 "(mandatory, DESIGN §4)")

        if self.theme and not vocabulary.is_valid_theme(self.theme, _VOCAB_PATH):
            raise ValueError(f"theme {self.theme!r} outside the closed "
                             f"vocabulary (v{vocabulary.version(_VOCAB_PATH)})")

        # rule 12 — which vocabulary produced this coordinate
        self.kernel_versions.setdefault("axes_coarse",
                                        vocabulary.version(_VOCAB_PATH))
        self.kernel_versions.setdefault("requirement_schema", SCHEMA_VERSION)

        if not self.rid:
            key = (f"{self.source_kind}:{self.source_id}|{self.locator}|"
                   f"{norm_span(self.evidence_span)}|{self.scale}|"
                   f"{self.theme}|{self.predicate}")
            self.rid = "r_" + hashlib.sha256(key.encode()).hexdigest()[:16]

    # ── construction under the verbatim guard ────────────────────────────────

    @classmethod
    def from_extraction(cls, *, payload: str, **kw: Any) -> "RequirementRecord":
        """Build a record from an extractor's output, span-guarded.

        Raises :class:`SpanNotVerbatim` when the span is not a
        whitespace-normalized substring of the exact payload the extractor was
        shown. The caller counts the drop and abstains — it never repairs the
        span, never truncates, never re-quotes (CLAUDE.md §12).
        """
        span = kw.get("evidence_span", "")
        if not span_verbatim(span, payload):
            raise SpanNotVerbatim(
                f"span {span[:60]!r} not found in payload "
                f"({len(payload)} chars)")
        rec = cls(**kw)
        rec.trace.setdefault("span_guard", "verified")
        rec.trace.setdefault("payload_sha256",
                             hashlib.sha256(payload.encode()).hexdigest()[:16])
        return rec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
