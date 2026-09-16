"""X21: invalid label observations and isolated stage failures stay visible.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import asdict, replace
import pytest
from ragix_kernels.saqqara.census import DocumentDigest, PageDigest
from ragix_kernels.saqqara.field_views import TextSpan, HorizontalRule
from ragix_kernels.saqqara.explorer import explore
from ragix_kernels.harvest.report import render_report
from ragix_kernels.saqqara.failures import FailureReport


def document(source="synthetic-error", text="Link: AB-K-731", rule=False):
    span = TextSpan(source, "s", 1, text, (20, 100, 250, 110), origin=(20, 110), font_size=10)
    return DocumentDigest(
        source,
        (
            PageDigest(
                1,
                600,
                800,
                (span,),
                horizontal_rules=(HorizontalRule(120, 0, 500),) if rule else (),
            ),
        ),
        "synthetic",
        "1",
    )


@pytest.mark.parametrize(
    "text,rule,reason",
    [
        (":", True, "EMPTY_LABEL"),
        ("   :", True, "EMPTY_LABEL"),
        ("Link:", True, "INVALID_LABEL_WINDOW"),
    ],
)
def test_e7_8_empty_label_and_terminal_rule_are_counted(text, rule, reason):
    result = explore(document(text=text, rule=rule))
    assert not isinstance(result.report, FailureReport)
    assert len(result.census.construct_findings) == 1
    assert result.census.construct_findings[0].reason == reason
    assert result.report.coverage.construct_failures == 1
    assert not any(not w.label.strip() for w in result.census.windows)
    assert (
        "reading"
        in render_report(
            result.report, {"field": "field", "quantity": "quantity", "table_row": "row"}
        ).lower()
    )


@pytest.mark.parametrize(
    "stage,function",
    [
        ("census", "census"),
        ("profile", "derive_profile"),
        ("read", "read_document"),
        ("report", "build_report"),
    ],
)
def test_e7_8_kernel_stage_failure_does_not_drop_other_documents(
    tmp_path, monkeypatch, stage, function
):
    import ragix_kernels.saqqara.kernels.explorer as kernels
    from ragix_kernels.base import KernelInput

    original = getattr(kernels, function)

    def injected(first, *args, **kwargs):
        if first.source_id == "bad":
            raise ValueError("malformed synthetic construct")
        return original(first, *args, **kwargs)

    monkeypatch.setattr(kernels, function, injected)
    previous = None
    for cls in (
        kernels.CensusKernel,
        kernels.ProfileKernel,
        kernels.ReadKernel,
        kernels.ReportKernel,
    ):
        config = (
            {"documents": [asdict(document("bad")), asdict(document("good"))]}
            if previous is None
            else {}
        )
        dependencies = {} if previous is None else {cls.requires[0]: previous.output_file}
        previous = cls().run(KernelInput(tmp_path, config, dependencies))
        assert previous.success, previous.errors
    bad, good = previous.data["reports"]
    assert bad["status"] == "FAILED" and bad["findings"][0]["stage"] == stage
    assert bad["findings"][0]["inspected_count"] == 1 and good["source_id"] == "good"


def test_library_failure_report_is_renderable_and_does_not_quote_input(monkeypatch):
    import ragix_kernels.saqqara.explorer as module

    def injected(*a, **k):
        raise ValueError("do not copy private exception text")

    monkeypatch.setattr(module, "derive_profile", injected)
    result = module.explore(document())
    assert result.report.status == "FAILED"
    rendered = render_report(result.report, {})
    assert "private exception" not in rendered and "INVALID_CONSTRUCT" in rendered


def test_changed_reading_rules_cannot_reuse_pre_plurality_kernel_cache(tmp_path):
    from ragix_kernels.base import KernelInput
    from ragix_kernels.saqqara.kernels.explorer import (
        CensusKernel,
        ProfileKernel,
        ReadKernel,
        ReportKernel,
    )

    for cls in (CensusKernel, ProfileKernel, ReadKernel, ReportKernel):
        current = cls()
        old = cls()
        old.version = "0.2.0"
        data = KernelInput(tmp_path, {"documents": [asdict(document())]})
        assert current._hash_input(data) != old._hash_input(data)
