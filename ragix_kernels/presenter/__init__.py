"""
KOAS Presenter Kernels — Folder-to-Slides via MARP

A kernel family for transforming a corpus of Markdown/text documents
into structured MARP slide decks, with semantic extraction of equations,
illustrations, tables, and code blocks.

Stage 1 — Collection (deterministic):
    pres_folder_scan:        Recursive folder walk, file classification, hashing
    pres_content_extract:    Markdown/text → SemanticUnit[] (mistune AST)
    pres_asset_catalog:      Unified inventory of images, equations, tables, diagrams

Stage 2 — Structuring (deterministic + optional LLM):
    pres_semantic_normalize: Topic clustering, dedup, roles, narrative (LLM boundary)
    pres_slide_plan:         NormalizedCorpus → SlideDeck JSON (deterministic rules)
    pres_layout_assign:      Slide type → MARP layout directives

Stage 3 — Rendering (deterministic):
    pres_marp_render:        SlideDeck JSON → MARP Markdown
    pres_marp_export:        marp-cli → PDF / HTML / PPTX

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

__version__ = "0.1.0"

# Kernels are registered automatically by the KernelRegistry
# via package discovery (pkgutil.walk_packages).
# No explicit imports needed here.

__all__ = []
