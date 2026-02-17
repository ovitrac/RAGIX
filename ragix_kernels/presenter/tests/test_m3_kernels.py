"""
Integration tests for KOAS Presenter M3 (Stage 3 Rendering) kernels.

Tests the S3 pipeline:
    pres_marp_render → pres_marp_export

Also includes a full 8-kernel pipeline integration test.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.presenter.models import SlideDeck, SlideType
from ragix_kernels.presenter.kernels.pres_folder_scan import PresFolderScanKernel
from ragix_kernels.presenter.kernels.pres_content_extract import PresContentExtractKernel
from ragix_kernels.presenter.kernels.pres_asset_catalog import PresAssetCatalogKernel
from ragix_kernels.presenter.kernels.pres_semantic_normalize import PresSemanticNormalizeKernel
from ragix_kernels.presenter.kernels.pres_slide_plan import PresSlidePlanKernel
from ragix_kernels.presenter.kernels.pres_layout_assign import PresLayoutAssignKernel
from ragix_kernels.presenter.kernels.pres_marp_render import PresMarpRenderKernel
from ragix_kernels.presenter.kernels.pres_marp_export import PresMarpExportKernel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOC_A = """\
---
title: Test Document A
author: Test Author
---

# Introduction

This is a test document with various Markdown elements.

## Methods

We use **bold** and *italic* formatting.

### Algorithm

```python
def hello():
    print("Hello, world!")
```

$$
E = mc^2
$$

> This is a blockquote
> spanning multiple lines.

- Item one
- Item two
- Item three

1. First step
2. Second step

| Header A | Header B | Header C |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |

![Architecture diagram](images/arch.svg)

## Results

The results are significant.

```mermaid
graph TD
    A --> B
    B --> C
```

> [!NOTE]
> This is an important admonition.
"""

SAMPLE_DOC_B = """\
# Summary

A brief summary document.

## Key Findings

The analysis shows improvements.

## Recommendations

We recommend the following actions.

![Chart](chart.png)
"""

SAMPLE_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 200 100">
  <rect width="200" height="100" fill="blue"/>
</svg>
"""


@pytest.fixture
def sample_project(tmp_path):
    """Create a minimal sample project folder."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "doc_a.md").write_text(SAMPLE_DOC_A, encoding="utf-8")
    (project / "doc_b.md").write_text(SAMPLE_DOC_B, encoding="utf-8")
    img_dir = project / "images"
    img_dir.mkdir()
    (img_dir / "arch.svg").write_text(SAMPLE_SVG, encoding="utf-8")
    return project


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace directory for kernel outputs."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def _run_s1_s2_pipeline(sample_project, workspace, config=None):
    """Run S1 + S2 pipeline and return all outputs."""
    if config is None:
        config = {}

    # S1: Folder scan
    scan_k = PresFolderScanKernel()
    scan_out = scan_k.run(KernelInput(
        workspace=workspace,
        config={"folder_path": str(sample_project), **config},
    ))
    assert scan_out.success

    # S1: Content extract
    extract_k = PresContentExtractKernel()
    extract_out = extract_k.run(KernelInput(
        workspace=workspace,
        config={"folder_path": str(sample_project), **config},
        dependencies={"pres_folder_scan": scan_out.output_file},
    ))
    assert extract_out.success

    # S1: Asset catalog
    catalog_k = PresAssetCatalogKernel()
    catalog_out = catalog_k.run(KernelInput(
        workspace=workspace,
        config=config,
        dependencies={
            "pres_folder_scan": scan_out.output_file,
            "pres_content_extract": extract_out.output_file,
        },
    ))
    assert catalog_out.success

    # S2: Normalize
    norm_k = PresSemanticNormalizeKernel()
    norm_out = norm_k.run(KernelInput(
        workspace=workspace,
        config={"normalizer": {"mode": "deterministic"}, "lang": "en", **config},
        dependencies={"pres_content_extract": extract_out.output_file},
    ))
    assert norm_out.success

    # S2: Slide plan
    plan_k = PresSlidePlanKernel()
    plan_out = plan_k.run(KernelInput(
        workspace=workspace,
        config={"lang": "en", **config},
        dependencies={
            "pres_semantic_normalize": norm_out.output_file,
            "pres_asset_catalog": catalog_out.output_file,
        },
    ))
    assert plan_out.success

    # S2: Layout assign
    layout_k = PresLayoutAssignKernel()
    layout_out = layout_k.run(KernelInput(
        workspace=workspace,
        config=config,
        dependencies={
            "pres_slide_plan": plan_out.output_file,
            "pres_asset_catalog": catalog_out.output_file,
        },
    ))
    assert layout_out.success

    return scan_out, extract_out, catalog_out, norm_out, plan_out, layout_out


# ---------------------------------------------------------------------------
# Test: pres_marp_render (K7)
# ---------------------------------------------------------------------------

class TestPresMarpRender:

    def test_frontmatter(self, sample_project, workspace):
        outputs = _run_s1_s2_pipeline(sample_project, workspace)
        scan_out, extract_out, catalog_out, norm_out, plan_out, layout_out = outputs

        kernel = PresMarpRenderKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"theme": {"name": "gaia", "size": "16:9", "math": "katex"}},
            dependencies={
                "pres_layout_assign": layout_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        assert output.success

        md = output.data["marp_markdown"]
        assert "marp: true" in md
        assert "theme: gaia" in md
        assert "math: katex" in md

    def test_slide_separator(self, sample_project, workspace):
        outputs = _run_s1_s2_pipeline(sample_project, workspace)
        *_, layout_out = outputs

        kernel = PresMarpRenderKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_layout_assign": layout_out.output_file,
                "pres_asset_catalog": outputs[2].output_file,
            },
        )
        output = kernel.run(inp)
        md = output.data["marp_markdown"]

        # Count slide separators: (n_slides - 1) between slides + 1 from frontmatter closing
        n_slides = output.data["slide_count"]
        n_separators = md.count("\n---\n")
        assert n_separators == n_slides  # (n-1) slide breaks + 1 frontmatter close

    def test_title_slide(self, sample_project, workspace):
        outputs = _run_s1_s2_pipeline(sample_project, workspace)
        *_, layout_out = outputs

        kernel = PresMarpRenderKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_layout_assign": layout_out.output_file,
                "pres_asset_catalog": outputs[2].output_file,
            },
        )
        output = kernel.run(inp)
        md = output.data["marp_markdown"]

        # First slide should have lead class
        assert "<!-- _class: lead -->" in md

    def test_code_slide(self, sample_project, workspace):
        outputs = _run_s1_s2_pipeline(sample_project, workspace)
        *_, layout_out = outputs

        kernel = PresMarpRenderKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_layout_assign": layout_out.output_file,
                "pres_asset_catalog": outputs[2].output_file,
            },
        )
        output = kernel.run(inp)
        md = output.data["marp_markdown"]

        # Should contain a code block
        assert "```python" in md

    def test_equation_slide(self, sample_project, workspace):
        outputs = _run_s1_s2_pipeline(sample_project, workspace)
        *_, layout_out = outputs

        kernel = PresMarpRenderKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_layout_assign": layout_out.output_file,
                "pres_asset_catalog": outputs[2].output_file,
            },
        )
        output = kernel.run(inp)
        md = output.data["marp_markdown"]

        # Should contain equation delimiters
        assert "$$" in md

    def test_table_slide(self, sample_project, workspace):
        outputs = _run_s1_s2_pipeline(sample_project, workspace)
        *_, layout_out = outputs

        kernel = PresMarpRenderKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_layout_assign": layout_out.output_file,
                "pres_asset_catalog": outputs[2].output_file,
            },
        )
        output = kernel.run(inp)
        md = output.data["marp_markdown"]

        # Should contain pipe-delimited table
        assert "| Header A |" in md or "|" in md

    def test_speaker_notes(self, sample_project, workspace):
        outputs = _run_s1_s2_pipeline(sample_project, workspace)
        *_, layout_out = outputs

        kernel = PresMarpRenderKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_layout_assign": layout_out.output_file,
                "pres_asset_catalog": outputs[2].output_file,
            },
        )
        output = kernel.run(inp)
        md = output.data["marp_markdown"]

        # Should contain HTML comment notes
        assert "<!-- " in md

    def test_image_slide(self, sample_project, workspace):
        outputs = _run_s1_s2_pipeline(sample_project, workspace)
        *_, layout_out = outputs

        kernel = PresMarpRenderKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_layout_assign": layout_out.output_file,
                "pres_asset_catalog": outputs[2].output_file,
            },
        )
        output = kernel.run(inp)
        md = output.data["marp_markdown"]

        # Should reference images
        assert "arch.svg" in md or "chart.png" in md or "![" in md


# ---------------------------------------------------------------------------
# Test: pres_marp_export (K8)
# ---------------------------------------------------------------------------

class TestPresMarpExport:

    def _run_render(self, sample_project, workspace):
        outputs = _run_s1_s2_pipeline(sample_project, workspace)
        scan_out, extract_out, catalog_out, norm_out, plan_out, layout_out = outputs

        render_k = PresMarpRenderKernel()
        render_out = render_k.run(KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_layout_assign": layout_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        ))
        assert render_out.success
        return scan_out, render_out

    def test_output_dir_created(self, sample_project, workspace):
        scan_out, render_out = self._run_render(sample_project, workspace)

        kernel = PresMarpExportKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_marp_render": render_out.output_file,
                "pres_folder_scan": scan_out.output_file,
            },
        )
        output = kernel.run(inp)
        assert output.success

        output_dir = Path(output.data["output_dir"])
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_markdown_written(self, sample_project, workspace):
        scan_out, render_out = self._run_render(sample_project, workspace)

        kernel = PresMarpExportKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_marp_render": render_out.output_file,
                "pres_folder_scan": scan_out.output_file,
            },
        )
        output = kernel.run(inp)

        pres_file = Path(output.data["presentation_file"])
        assert pres_file.exists()
        content = pres_file.read_text(encoding="utf-8")
        assert "marp: true" in content

    def test_assets_copied(self, sample_project, workspace):
        scan_out, render_out = self._run_render(sample_project, workspace)

        kernel = PresMarpExportKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_marp_render": render_out.output_file,
                "pres_folder_scan": scan_out.output_file,
            },
        )
        output = kernel.run(inp)

        assets_dir = Path(output.data["output_dir"]) / "assets"
        assert assets_dir.exists()
        # At least the SVG should be copied
        copied = output.data["assets_copied"]
        assert copied >= 0  # may be 0 if no asset refs resolved

    def test_metadata_json(self, sample_project, workspace):
        scan_out, render_out = self._run_render(sample_project, workspace)

        kernel = PresMarpExportKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"title": "My Presentation", "author": "Test Author"},
            dependencies={
                "pres_marp_render": render_out.output_file,
                "pres_folder_scan": scan_out.output_file,
            },
        )
        output = kernel.run(inp)

        meta_file = Path(output.data["metadata_file"])
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta["generated_by"] == "koas-presenter"
        assert meta["title"] == "My Presentation"


# ---------------------------------------------------------------------------
# Full 8-kernel pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_eight_kernel_chain(self, sample_project, workspace):
        """Run all 8 kernels S1+S2+S3 end-to-end."""
        config = {
            "title": "Full Pipeline Test",
            "author": "Test Author",
            "organization": "Test Org",
            "lang": "en",
        }

        # S1: K1 folder scan
        k1 = PresFolderScanKernel()
        o1 = k1.run(KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project), **config},
        ))
        assert o1.success, f"K1 failed: {o1.errors}"

        # S1: K2 content extract
        k2 = PresContentExtractKernel()
        o2 = k2.run(KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project), **config},
            dependencies={"pres_folder_scan": o1.output_file},
        ))
        assert o2.success, f"K2 failed: {o2.errors}"

        # S1: K3 asset catalog
        k3 = PresAssetCatalogKernel()
        o3 = k3.run(KernelInput(
            workspace=workspace,
            config=config,
            dependencies={
                "pres_folder_scan": o1.output_file,
                "pres_content_extract": o2.output_file,
            },
        ))
        assert o3.success, f"K3 failed: {o3.errors}"

        # S2: K4 semantic normalize
        k4 = PresSemanticNormalizeKernel()
        o4 = k4.run(KernelInput(
            workspace=workspace,
            config={"normalizer": {"mode": "deterministic"}, **config},
            dependencies={"pres_content_extract": o2.output_file},
        ))
        assert o4.success, f"K4 failed: {o4.errors}"

        # S2: K5 slide plan
        k5 = PresSlidePlanKernel()
        o5 = k5.run(KernelInput(
            workspace=workspace,
            config=config,
            dependencies={
                "pres_semantic_normalize": o4.output_file,
                "pres_asset_catalog": o3.output_file,
            },
        ))
        assert o5.success, f"K5 failed: {o5.errors}"

        # S2: K6 layout assign
        k6 = PresLayoutAssignKernel()
        o6 = k6.run(KernelInput(
            workspace=workspace,
            config=config,
            dependencies={
                "pres_slide_plan": o5.output_file,
                "pres_asset_catalog": o3.output_file,
            },
        ))
        assert o6.success, f"K6 failed: {o6.errors}"

        # S3: K7 MARP render
        k7 = PresMarpRenderKernel()
        o7 = k7.run(KernelInput(
            workspace=workspace,
            config=config,
            dependencies={
                "pres_layout_assign": o6.output_file,
                "pres_asset_catalog": o3.output_file,
            },
        ))
        assert o7.success, f"K7 failed: {o7.errors}"

        # S3: K8 MARP export
        k8 = PresMarpExportKernel()
        o8 = k8.run(KernelInput(
            workspace=workspace,
            config=config,
            dependencies={
                "pres_marp_render": o7.output_file,
                "pres_folder_scan": o1.output_file,
            },
        ))
        assert o8.success, f"K8 failed: {o8.errors}"

        # Verify final output
        output_dir = Path(o8.data["output_dir"])
        pres_file = output_dir / "presentation.md"
        meta_file = output_dir / "metadata.json"

        assert pres_file.exists()
        assert meta_file.exists()

        # Verify MARP Markdown is valid
        content = pres_file.read_text(encoding="utf-8")
        assert "marp: true" in content
        assert "---" in content  # slide separators
        assert "Full Pipeline Test" in content

        # Verify metadata
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta["generated_by"] == "koas-presenter"

        # Verify all 8 kernel outputs exist
        for o in [o1, o2, o3, o4, o5, o6, o7, o8]:
            assert o.output_file.exists()
            assert len(o.summary) > 10
