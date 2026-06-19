"""
RAGIX-Sealed — report export modes (WP §10.1/§20, Sprint 6).

Applies one of the four export modes to a sanitized report:

- SANITIZED_LLM_SAFE   default; placeholders only; leak-gated before release.
- HUMAN_AUTHORIZED     re-identified for the authorized human; requires an
                       AuthorizationToken + a resolver; watermarked; never for an LLM.
- AUDIT_ONLY           attestation dict; provenance/metrics/versions; no content.
- ORCHESTRATOR_METRICS counts only.

Re-identification is a controlled human export, NOT a reasoning kernel (K5): there is no
path that returns a raw value toward an LLM.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Optional

from ..ingest.leak_scan import scan as leak_scan
from ..kernels.analysis import PLACEHOLDER_RE
from ..vault.backend import (
    AuthorizationToken,
    ReidentificationPurpose,
    VaultAuthorizationError,
    VaultError,
)
from .report import SealedReport

WATERMARK = (
    "> [HUMAN-AUTHORIZED EXPORT — contains re-identified values. "
    "Do not share with any LLM or external system.]\n\n"
)

# placeholder token -> raw value (human-authorized resolution).
Resolver = Callable[[str], str]


class ExportMode(Enum):
    SANITIZED_LLM_SAFE = "SANITIZED_LLM_SAFE"
    HUMAN_AUTHORIZED = "HUMAN_AUTHORIZED"
    AUDIT_ONLY = "AUDIT_ONLY"
    ORCHESTRATOR_METRICS = "ORCHESTRATOR_METRICS"


class ReportError(Exception):
    """Raised when a report cannot be safely rendered in the requested mode."""


class ReportAuthorizationError(ReportError):
    """Raised when a human-authorized export lacks valid authorization."""


def _reidentify(text: str, resolver: Resolver) -> str:
    """Replace every placeholder token in ``text`` with its resolved raw value."""
    return PLACEHOLDER_RE.sub(lambda m: resolver(m.group(0)), text)


def render(
    report: SealedReport,
    mode: ExportMode,
    *,
    schema: Optional[Dict[str, Any]] = None,
    authorization: Optional[AuthorizationToken] = None,
    purpose: ReidentificationPurpose = ReidentificationPurpose.REPORT_EXPORT_HUMAN,
    resolver: Optional[Resolver] = None,
    attestation: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Any:
    """Render ``report`` under ``mode``. Returns markdown (text modes) or a dict."""
    if mode is ExportMode.SANITIZED_LLM_SAFE:
        text = report.to_markdown()
        if schema is not None and leak_scan(text, [], schema).verdict != "PASS":
            raise ReportError("sanitized report failed leak scan; not released")
        return text

    if mode is ExportMode.HUMAN_AUTHORIZED:
        if authorization is None or not authorization.authorizes(purpose):
            raise ReportAuthorizationError("human-authorized export requires valid authorization")
        if resolver is None:
            raise ReportError("human-authorized export requires a resolver")
        return WATERMARK + _reidentify(report.to_markdown(), resolver)

    if mode is ExportMode.AUDIT_ONLY:
        return dict(attestation or {})

    if mode is ExportMode.ORCHESTRATOR_METRICS:
        return dict(metrics or {})

    raise ReportError(f"unknown export mode: {mode!r}")  # pragma: no cover


def _canonical(token: str) -> str:
    """Strip any visible role: '[PERSON_001 | role]' -> '[PERSON_001]'."""
    m = PLACEHOLDER_RE.search(token)
    return f"[{m.group(1)}_{m.group(2)}]" if m else token


def vault_resolver(
    vault: Any,
    case_id: str,
    authorization: AuthorizationToken,
    purpose: ReidentificationPurpose = ReidentificationPurpose.REPORT_EXPORT_HUMAN,
) -> Resolver:
    """Build a placeholder->raw resolver backed by the sealed vault.

    Tries the token as written, then its canonical (role-stripped) form, since the vault
    stores canonical placeholders. An unknown placeholder is left untouched (safe). An
    authorization failure propagates — it is never silently swallowed.
    """
    def _resolve(token: str) -> str:
        for candidate in (token, _canonical(token)):
            try:
                return vault.resolve_placeholder(case_id, candidate, purpose, authorization)
            except VaultAuthorizationError:
                raise
            except VaultError:
                continue
        return token  # unknown placeholder: leave it placeholderized (deny-by-default)
    return _resolve


def render_reidentified(
    report: SealedReport,
    vault: Any,
    case_id: str,
    authorization: AuthorizationToken,
    purpose: ReidentificationPurpose = ReidentificationPurpose.REPORT_EXPORT_HUMAN,
) -> str:
    """Convenience HUMAN_AUTHORIZED render that re-identifies via the sealed vault."""
    return render(
        report, ExportMode.HUMAN_AUTHORIZED,
        authorization=authorization, purpose=purpose,
        resolver=vault_resolver(vault, case_id, authorization, purpose),
    )
