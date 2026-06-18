"""
RAGIX-Sealed reporting kernels (WP §19-§20, Sprint 6).

Builds sanitized reviewer-facing reports from inventory + analysis, then applies a §10.1
export mode. Re-identification is a controlled, human-authorized export — never toward an
LLM.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from .export import (
    ExportMode,
    ReportAuthorizationError,
    ReportError,
    WATERMARK,
    render,
)
from .report import (
    ReportSection,
    SealedReport,
    build_audit_attestation,
    build_commitment_matrix,
    build_sanitized_memo,
)

__all__ = [
    "ReportSection",
    "SealedReport",
    "build_sanitized_memo",
    "build_commitment_matrix",
    "build_audit_attestation",
    "ExportMode",
    "render",
    "WATERMARK",
    "ReportError",
    "ReportAuthorizationError",
]
