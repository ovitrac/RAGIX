"""tender.contract — what the composing model may see, and what it may cite.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Two of the fourteen non-negotiable rules (CLAUDE.md §9) are not policy that a
lane can be trusted to follow — they are a **shape**, and this module is that
shape:

    8. No LLM may see unrestricted corpus content during final composition;
       it receives only selected evidence.
    9. No cited claim may refer to evidence that was not present in the
       producer payload.

Rule 8 is made structural rather than promised. `ProducerPayload.for_model()`
returns exactly the dictionary handed to the model, built from nothing but the
selected `EvidenceItem`s; the payload holds no store, no tree, no retriever and
no corpus path, so there is nothing for a later edit to reach through. A lane
that wants to show the model more has to add it to the payload, where it is
visible, rather than pass a handle that happens to be in scope.

Rule 9 is a **verification**, and it is deliberately not a repair. `verify()`
raises `ContractViolation`. An answer citing evidence that was never in its own
payload is a malformed model output, and §12 forbids silently repairing one —
dropping the citation would leave a claim standing with nothing behind it, which
is the exact failure the rule names. What the caller does with the exception is
policy: route to human review, abstain, retry. The contract's job is to make the
violation impossible to miss, not to choose the remedy.

**Written fresh (R4, lead 2026-08-31), not ported.** The orchestrator contract it
replaces stays with the quarantined package, along with the bilingual gatekeeper
shape that belongs to that lineage. Nothing here carries a language enum: the
lab's answer language is not a settled contract, and inventing one to fill the
same slot would be a schema decision taken by resemblance. When it is needed it
is the lead's to set.

**The verdict vocabulary is transcribed, not invented.** CLAUDE.md §5.2 fixes
both the internal verdicts and their one-to-one human rendering, and §11 requires
that mapping to stay one-to-one — so the mapping lives here, next to the values,
where a test can hold it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

CONTRACT_VERSION = "1.0-draft"

#: Answer-level verdicts (CLAUDE.md §5.2). Closed: a sixth is a decision.
VERDICTS = (
    "supported",
    "supported_with_caveats",
    "needs_review",
    "abstain_no_evidence",
    "abstain_conflict",
)

#: The human-facing rendering Adservio committed to in a tender response.
#: One-to-one with VERDICTS, and §11 forbids a third taxonomy.
HUMAN_RENDERING = {
    "supported": "PRÊT",
    "supported_with_caveats": "PRÊT AVEC RÉSERVES",
    "needs_review": "À VÉRIFIER",
    "abstain_no_evidence": "PREUVE MANQUANTE",
    "abstain_conflict": "CONTRADICTOIRE",
}

#: The verdicts that assert something. They require evidence; the others are
#: states of the evidence and must not carry any.
ASSERTING = ("supported", "supported_with_caveats")
ABSTAINING = ("abstain_no_evidence", "abstain_conflict")

_WHITESPACE = re.compile(r"\s+")


class ContractViolation(ValueError):
    """Rule 8 or rule 9 broken by a payload/answer pair. Never repaired here."""


def _normalised(text: str) -> str:
    """Whitespace-collapsed text, for comparing a quote to its source.

    A declared comparison rule, not a repair: a model reflows lines, and holding
    a quote to the source's line breaks would fail honest citations while
    catching nothing. Everything else — words, order, accents, punctuation — must
    match, so a paraphrase is still a violation.
    """
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class EvidenceItem:
    """One selected piece of evidence, with the provenance that makes it citable.

    `evidence_id` is derived from the chunk it came from, so the same evidence
    carries the same id in every payload and a citation stays meaningful when a
    trace is read back later.
    """

    text: str
    doc_id: str
    chunk_id: str
    node_ids: tuple[str, ...]
    section_path: tuple[str, ...] = ()
    pages: tuple[int, ...] = ()
    routing_class: Optional[str] = None
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("evidence without text")
        if not self.doc_id or not self.chunk_id:
            raise ValueError("evidence without provenance (rule 2)")
        if not self.node_ids:
            raise ValueError(
                "evidence naming no node: a citation to it could not be traced "
                "back to the document")
        if not self.evidence_id:
            key = f"{self.doc_id}|{self.chunk_id}"
            object.__setattr__(
                self, "evidence_id",
                "e_" + hashlib.sha256(key.encode()).hexdigest()[:16])

    @classmethod
    def of_chunk(cls, chunk: Any, *, routing_class: Optional[str] = None) -> "EvidenceItem":
        """From a retrieved `ChunkRecord`. The store already refuses a chunk that
        names no node, so the provenance this depends on exists by construction."""
        return cls(
            text=chunk.text, doc_id=chunk.doc_id, chunk_id=chunk.chunk_id,
            node_ids=tuple(chunk.node_ids),
            section_path=tuple(chunk.section_path or ()),
            pages=tuple(chunk.pages or ()),
            routing_class=routing_class,
        )

    def for_model(self) -> dict[str, Any]:
        """This item as the model sees it: the text, and the handle to cite it by."""
        return {"evidence_id": self.evidence_id, "text": self.text,
                "section_path": list(self.section_path)}


@dataclass(frozen=True)
class ProducerPayload:
    """Everything the composing model may see. Rule 8: there is nothing else.

    The payload is a value. It holds no store, no tree and no corpus path, so
    "the model saw only this" is a property of the object rather than a habit of
    the lane that built it.
    """

    question: str
    evidence: tuple[EvidenceItem, ...] = ()

    def __post_init__(self) -> None:
        if not self.question or not self.question.strip():
            raise ValueError("payload without a question")
        seen = [item.evidence_id for item in self.evidence]
        if len(seen) != len(set(seen)):
            raise ValueError("two evidence items with the same id in one payload")

    def for_model(self) -> dict[str, Any]:
        """Exactly the dictionary handed to the model — nothing derived later."""
        return {"contract_version": CONTRACT_VERSION,
                "question": self.question,
                "evidence": [item.for_model() for item in self.evidence]}

    def find(self, evidence_id: str) -> Optional[EvidenceItem]:
        for item in self.evidence:
            if item.evidence_id == evidence_id:
                return item
        return None


@dataclass(frozen=True)
class Citation:
    """A claim's pointer into the payload, with the span it rests on."""

    evidence_id: str
    quote: str

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("citation naming no evidence")
        if not self.quote or not self.quote.strip():
            raise ValueError(
                "citation without a quote: a pointer with no span cannot be "
                "verified, and an unverifiable citation is not one")


@dataclass(frozen=True)
class ComposedAnswer:
    """What the composing lane produced, before anything is shown to a human."""

    text: str
    verdict: str
    citations: tuple[Citation, ...] = ()

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict {self.verdict!r} outside {VERDICTS}")
        if self.verdict in ABSTAINING and self.citations:
            raise ValueError(
                f"{self.verdict} carries citations: an abstention is a state of "
                "the evidence, not an answer with sources")

    @property
    def rendering(self) -> str:
        """The human-facing label (§5.2), never a third taxonomy."""
        return HUMAN_RENDERING[self.verdict]


def verify(answer: ComposedAnswer, payload: ProducerPayload) -> None:
    """Rule 9, and the evidential floor under rule 11. Raises, never repairs.

    Three ways to fail, and each is a different defect:

      - a citation names evidence **absent from this payload** — the model cited
        something it was not given, which is rule 9 exactly;
      - a citation's quote **is not in** the evidence it names — the pointer
        resolves but the span does not, so the claim rests on text nobody
        supplied. Compared on whitespace-collapsed text (see `_normalised`);
      - an **asserting verdict with no citation at all** — an answer that claims
        support while naming none is the unsupported answer rule 11 forbids.

    Raises:
        ContractViolation: with every violation found, not just the first. A
            caller routing to human review needs the whole account.
    """
    violations: list[str] = []

    for index, citation in enumerate(answer.citations):
        item = payload.find(citation.evidence_id)
        if item is None:
            violations.append(
                f"citation {index}: evidence {citation.evidence_id!r} was not in "
                "the producer payload (rule 9)")
            continue
        if _normalised(citation.quote) not in _normalised(item.text):
            violations.append(
                f"citation {index}: the quote is not in evidence "
                f"{citation.evidence_id!r} — the pointer resolves, the span does not")

    if answer.verdict in ASSERTING and not answer.citations:
        violations.append(
            f"verdict {answer.verdict!r} with no citation: an answer claiming "
            "support must name what supports it (rule 11)")

    if violations:
        raise ContractViolation("; ".join(violations))


def payload_from_retrieval(question: str, result: Any, *,
                           routing_classes: Optional[dict[str, str]] = None,
                           limit: Optional[int] = None) -> ProducerPayload:
    """The selected chunks of a `RetrievalResult`, and only those.

    `parents`, `related` and `objects` are deliberately NOT included: they are
    context the retrieval lane assembled for a human reading the trace, and
    quietly adding them to the payload would widen what the model sees behind a
    convenience function — the very move rule 8 exists to prevent. A lane that
    wants them in the payload passes them as evidence, explicitly.
    """
    classes = routing_classes or {}
    selected: Sequence[Any] = result.selected
    if limit is not None:
        selected = selected[:limit]
    return ProducerPayload(
        question=question,
        evidence=tuple(
            EvidenceItem.of_chunk(hit.chunk,
                                  routing_class=classes.get(hit.chunk.doc_id))
            for hit in selected),
    )
