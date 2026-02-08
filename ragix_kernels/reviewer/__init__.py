"""
KOAS Reviewer Kernels — Traceable, Reversible Markdown Review

A kernel family for reviewing and editing Markdown documents larger than
context windows, with full traceability (change IDs, ledger, patches)
and selective revert capability.

Stage 1 — Collection (deterministic):
    md_inventory:          File stats, SHA-256, front-matter detection
    md_structure:          Heading tree, section anchors, numbering patterns
    md_protected_regions:  Code fences, inline code, YAML, tables, math
    md_chunk:              Structure-aligned chunk plan with hash-stable IDs

Stage 2 — Analysis (deterministic + LLM edge):
    md_consistency_scan:   AI leftovers, register shifts, duplicates, broken refs
    md_numbering_control:  Heading/figure/table numbering validation
    md_pyramid:            Bottom-up hierarchical summaries (single-document)
    md_edit_plan:          Constrained LLM produces structured edit ops per chunk

Stage 3 — Reporting (deterministic):
    md_apply_ops:              Validate + apply edit ops + forward/inverse patches
    md_inline_notes_inject:    GitHub alert blocks (REVIEWER: prefix)
    md_review_report_assemble: Generate REVIEW_doc.md from ledger
    md_revert:                 Selective inverse-patch application by change ID

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

__version__ = "0.1.0"

# Kernels are registered automatically by the KernelRegistry
# via package discovery (pkgutil.walk_packages).
# No explicit imports needed here.

__all__ = []
