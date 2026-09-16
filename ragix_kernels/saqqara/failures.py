"""Counted construct failures, distinct from document evidence and success.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, dataclass
from .field_views import stable_id

CONSTRUCT_ERRORS = (ValueError, TypeError, KeyError)


@dataclass(frozen=True)
class ConstructFinding:
    record_id: str
    source_id: str | None
    stage: str
    reason: str
    page: int | None = None
    span_id: str | None = None
    inspected_count: int = 1
    error_type: str | None = None

    def __post_init__(self):
        if not self.record_id or not self.stage or not self.reason or self.inspected_count < 1:
            raise ValueError("construct finding requires scope and positive count")


def construct_finding(source_id, stage, reason, *, page=None, span_id=None, error=None):
    kind = type(error).__name__ if error is not None else None
    return ConstructFinding(
        stable_id("construct-finding/0.1", source_id, stage, reason, page, span_id, kind),
        source_id,
        stage,
        reason,
        page,
        span_id,
        error_type=kind,
    )


@dataclass(frozen=True)
class FailureReport:
    source_id: str | None
    findings: tuple[ConstructFinding, ...]
    replay_digest: str
    status: str = "FAILED"
    version: str = "reading-failure/0.1"


def failure_report(finding):
    if isinstance(finding, dict):
        finding = ConstructFinding(**finding)
    return FailureReport(
        finding.source_id, (finding,), stable_id("reading-failure/0.1", asdict(finding))
    )


def stage_failure(document, stage, error):
    source = (
        document.get("source_id")
        if isinstance(document, dict)
        else getattr(document, "source_id", None)
    )
    pages = (
        document.get("pages", ()) if isinstance(document, dict) else getattr(document, "pages", ())
    )
    page = getattr(error, "page", None)
    if page is None and isinstance(pages, (tuple, list)) and len(pages) == 1:
        candidate = (
            pages[0].get("page") if isinstance(pages[0], dict) else getattr(pages[0], "page", None)
        )
        page = candidate if type(candidate) is int and candidate >= 1 else None
    # A generic exception does not identify a page: never invent the first page
    # as the location of a failure. Construct-aware errors may provide it.
    return construct_finding(
        source if isinstance(source, str) else None,
        stage,
        "INVALID_CONSTRUCT",
        page=page,
        span_id=getattr(error, "span_id", None),
        error=error,
    )
