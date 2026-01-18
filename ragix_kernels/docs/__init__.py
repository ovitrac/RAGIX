"""
RAGIX Document Summarization Kernels — Hierarchical Document Analysis

Stage 1: Document Collection
- doc_metadata: Extract document metadata and statistics from RAG index
- doc_concepts: Extract/aggregate concepts from RAG knowledge graph
- doc_structure: Detect document structure (headings, sections)

Stage 2: Document Analysis
- doc_cluster: Group documents by topic similarity (hierarchical clustering)
- doc_cluster_leiden: Community detection using Leiden algorithm (parallel to hierarchical)
- doc_cluster_reconcile: Reconcile hierarchical and Leiden clustering results
- doc_extract: Extract key sentences per concept (with quality scoring v1.1)
- doc_coverage: Analyze concept coverage across documents
- doc_func_extract: Extract structured functionalities from SPD documents

Stage 3: Synthesis
- doc_pyramid: Build hierarchical summary structure (document → group → domain → corpus)
- doc_summarize: Generate LLM-based per-document summaries
- doc_summarize_tutored: Two-stage summarization with tutor verification
- doc_compare: Detect inter-document discrepancies and inconsistencies
- doc_visualize: Generate publication-quality visualizations (7 figure types including word cloud: SVG/PNG/PDF)
- doc_report_assemble: Assemble comprehensive Markdown report
- doc_final_report: Consolidated final report with appendices and sovereignty attestation

Architecture follows KOAS patterns:
- LLM used only in synthesis stage for summarization (not for extraction)
- Dual clustering: Hierarchical + Leiden in parallel for robustness
- Tutor-Worker pattern: Worker LLM (Granite 3B) + Tutor LLM (Mistral 7B)
- LLM response caching with sovereignty tracking
- Structured output + summary for LLM consumption
- Full traceability and reproducibility
- Uses existing RAG graph (File → Chunk → Concept)

Enhanced in v1.2:
- Dual clustering (doc_cluster + doc_cluster_leiden)
- Cluster reconciliation with dual-view fallback (doc_cluster_reconcile)
- Tutor-verified summaries (doc_summarize_tutored)
- Final report at run root (doc_final_report)
- LLM caching with sovereignty attestation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

# Kernels are registered automatically by the registry
# via package discovery. No explicit imports needed here.

__all__ = []  # Auto-populated by registry
