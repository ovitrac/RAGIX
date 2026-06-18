"""
RAGIX-Sealed — Level-1 inventory kernels (WP §16, Sprint 4).

Each kernel answers "what is in the corpus?" with metrics only — no interpretation, no
content, opaque ids. Safe for the public orchestrator (`sealed.run_inventory_kernel`).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from collections import Counter

from .base import CorpusView, InventoryKernel, KernelResult


class CorpusMetricsKernel(InventoryKernel):
    """Top-level counts and distributions."""

    name = "corpus_metrics"

    def run(self, view: CorpusView) -> KernelResult:
        docs = view.documents
        metrics = {
            "documents_total": len(docs),
            "documents_indexable": sum(1 for d in docs if view.is_cooled(d)),
            "documents_blocked": sum(1 for d in docs if view.is_blocked(d)),
            "pages_total": sum(d.pages for d in docs),
            "source_kinds": dict(Counter(d.source_kind for d in docs)),
            "human_review_required": sum(1 for d in docs if d.human_review_required),
        }
        if view.chunk_count is not None:
            metrics["chunks_total"] = view.chunk_count
        return KernelResult(self.name, metrics)


class TypologyKernel(InventoryKernel):
    """Distribution of documents by kind."""

    name = "typology"

    def run(self, view: CorpusView) -> KernelResult:
        return KernelResult(self.name, {"document_kinds": dict(Counter(d.source_kind for d in view.documents))})


class EntityInventoryKernel(InventoryKernel):
    """Aggregate placeholder-class counts across the corpus."""

    name = "entity_inventory"

    def run(self, view: CorpusView) -> KernelResult:
        totals: Counter = Counter()
        for d in view.documents:
            totals.update(d.entity_counts or {})
        return KernelResult(self.name, {"entity_counts": dict(totals)})


class QualityRiskKernel(InventoryKernel):
    """Leak-scan outcomes, blocked count, and review burden."""

    name = "quality_risk"

    def run(self, view: CorpusView) -> KernelResult:
        docs = view.documents
        verdicts = Counter(d.leak_verdict for d in docs)
        metrics = {
            "leak_verdicts": dict(verdicts),
            "blocked": sum(1 for d in docs if view.is_blocked(d)),
            "review_required": sum(1 for d in docs if d.human_review_required),
            "uncertain": verdicts.get("UNCERTAIN", 0),
        }
        return KernelResult(self.name, metrics)


class ReviewQueueKernel(InventoryKernel):
    """Opaque ids of documents needing review or that were blocked. No content."""

    name = "review_queue"

    def run(self, view: CorpusView) -> KernelResult:
        metrics = {
            "review_required_docs": [d.doc_id for d in view.documents if d.human_review_required],
            "blocked_docs": [d.doc_id for d in view.documents if view.is_blocked(d)],
        }
        return KernelResult(self.name, metrics)


# Registration order = display order.
ALL_INVENTORY_KERNELS = [
    CorpusMetricsKernel(),
    TypologyKernel(),
    EntityInventoryKernel(),
    QualityRiskKernel(),
    ReviewQueueKernel(),
]
