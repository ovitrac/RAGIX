"""
The tender domain library: records, claims, the producer contract, and the lanes over a store.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-14

Ported as they are from the lab's tender-response work: logic, rules, thresholds and schema
versions unchanged (ClaimRecord 1.1, Applicability 1.1, QuestionRecord and AnswerRecord
1.0-draft, RequirementRecord 0-draft). A library, not a kernel family: no Kernel subclass
lives here, so the family's pinned kernel list in `tests/tender/` does not move.

    records         QuestionRecord, AnswerRecord, Applicability 1.1, DocumentFacts
    claims          ClaimRecord 1.1: value, provenance, source spans, identity
    contract        what the composing model may see (rule 8) and cite (rule 9); the verdicts
    authority       the deadline's authority records, the discovery and blocking gates, the resolution
    routing         the routing class of an analyzed saqqara tree
    requirements    RequirementRecord, coarse and fine, under the verbatim span guard
    retrieval       lane weights, boosts, filters, expansion and budget over the saqqara retriever
    pipeline        the corpus lifecycle over a saqqara store: index, update, sync, rechunk, trash
    tagger          the deterministic coarse-axis tagger, the baseline a model lane must beat
    vocabulary      the closed coarse-axis vocabulary (the packaged v0-draft, under data/)
    deadline_slice  the deadline claims read from chunk text (uses ragix_kernels.harvest.fr.dates)
    pyramid         aggregation, the deterministic summary, zoom and rollup over claims
    agreement       Cohen's kappa and the disagreement table
    substrate       one way to obtain an analyzed saqqara tree

Nothing is imported at package level. The family's package walk imports every module, so a
module that needs another family (deadline_slice, the harvest French readers) imports it where
it is used.
"""
