"""
Integration tests for KOAS Presenter M1 (Stage 1 Collection) kernels.

Tests the 3-kernel pipeline:
    pres_folder_scan → pres_content_extract → pres_asset_catalog

Uses a minimal sample project fixture with Markdown files exercising
all 13 UnitType values.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.presenter.models import (
    AssetCatalog,
    AssetType,
    ContentCorpus,
    FileEntry,
    FileType,
    SemanticUnit,
    UnitType,
)
from ragix_kernels.presenter.kernels.pres_folder_scan import PresFolderScanKernel
from ragix_kernels.presenter.kernels.pres_content_extract import (
    PresContentExtractKernel,
    parse_document,
)
from ragix_kernels.presenter.kernels.pres_asset_catalog import PresAssetCatalogKernel


# ---------------------------------------------------------------------------
# Fixtures: sample project
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

The results are $\\alpha = 0.05$ significant.

```mermaid
graph TD
    A --> B
    B --> C
```

> [!NOTE]
> This is an important admonition
> with multiple lines.
"""

SAMPLE_DOC_B = """\
# Summary

A brief summary document.

## Key Findings

The analysis shows improvements.

![Chart](chart.png)
"""

SAMPLE_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 200 100">
  <rect width="200" height="100" fill="blue"/>
</svg>
"""


@pytest.fixture
def sample_project(tmp_path):
    """Create a minimal sample project folder (separate from workspace)."""
    project = tmp_path / "project"
    project.mkdir()

    # Documents
    (project / "doc_a.md").write_text(SAMPLE_DOC_A, encoding="utf-8")
    (project / "doc_b.md").write_text(SAMPLE_DOC_B, encoding="utf-8")

    # Assets
    img_dir = project / "images"
    img_dir.mkdir()
    (img_dir / "arch.svg").write_text(SAMPLE_SVG, encoding="utf-8")

    # Plain text
    (project / "notes.txt").write_text("Some plain text notes.\nLine 2.\n", encoding="utf-8")

    # Data file
    (project / "data.json").write_text('{"key": "value"}', encoding="utf-8")

    # Should be excluded
    git_dir = project / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("gitconfig", encoding="utf-8")

    return project


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace directory for kernel outputs (separate from project)."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


# ---------------------------------------------------------------------------
# Kernel 1: pres_folder_scan
# ---------------------------------------------------------------------------

class TestPresFolderScan:

    def test_basic_scan(self, sample_project, workspace):
        kernel = PresFolderScanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
        )
        output = kernel.run(inp)
        assert output.success
        data = output.data

        files = [FileEntry.from_dict(f) for f in data["files"]]
        stats = data["statistics"]

        # Should find: doc_a.md, doc_b.md, notes.txt, images/arch.svg, data.json
        assert stats["total_files"] >= 5
        assert stats["documents"] >= 3  # 2 md + 1 txt
        assert stats["assets"] >= 1     # arch.svg

        # .git should be excluded
        paths = {f.path for f in files}
        assert not any(".git" in p for p in paths)

    def test_file_classification(self, sample_project, workspace):
        kernel = PresFolderScanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
        )
        output = kernel.run(inp)
        files = [FileEntry.from_dict(f) for f in output.data["files"]]
        types = {f.path: f.file_type for f in files}

        assert types.get("doc_a.md") == FileType.DOCUMENT
        assert types.get("doc_b.md") == FileType.DOCUMENT
        assert types.get("notes.txt") == FileType.DOCUMENT
        # images/arch.svg path uses OS separator
        svg_files = [f for f in files if f.path.endswith("arch.svg")]
        assert len(svg_files) == 1
        assert svg_files[0].file_type == FileType.ASSET
        json_files = [f for f in files if f.path.endswith("data.json")]
        assert len(json_files) == 1
        assert json_files[0].file_type == FileType.DATA

    def test_hash_format(self, sample_project, workspace):
        kernel = PresFolderScanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
        )
        output = kernel.run(inp)
        files = [FileEntry.from_dict(f) for f in output.data["files"]]
        for f in files:
            assert f.file_hash.startswith("sha256:"), f"Bad hash format: {f.file_hash}"
            assert len(f.file_hash) == len("sha256:") + 64

    def test_front_matter_detection(self, sample_project, workspace):
        kernel = PresFolderScanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
        )
        output = kernel.run(inp)
        files = [FileEntry.from_dict(f) for f in output.data["files"]]

        doc_a = [f for f in files if f.path == "doc_a.md"]
        assert len(doc_a) == 1
        assert doc_a[0].front_matter is not None
        assert doc_a[0].front_matter.get("title") == "Test Document A"

        doc_b = [f for f in files if f.path == "doc_b.md"]
        assert len(doc_b) == 1
        assert doc_b[0].front_matter is None

    def test_line_count(self, sample_project, workspace):
        kernel = PresFolderScanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
        )
        output = kernel.run(inp)
        files = [FileEntry.from_dict(f) for f in output.data["files"]]

        doc_a = [f for f in files if f.path == "doc_a.md"][0]
        assert doc_a.line_count > 0
        # SVG should have 0 lines (not a DOCUMENT type)
        svg_files = [f for f in files if f.path.endswith("arch.svg")]
        assert svg_files[0].line_count == 0

    def test_validation_missing_folder(self, workspace):
        kernel = PresFolderScanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": "/nonexistent/path"},
        )
        output = kernel.run(inp)
        assert not output.success

    def test_output_persisted(self, sample_project, workspace):
        kernel = PresFolderScanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
        )
        output = kernel.run(inp)
        assert output.output_file.exists()
        persisted = json.loads(output.output_file.read_text())
        assert persisted["_meta"]["kernel_name"] == "pres_folder_scan"
        assert persisted["_meta"]["success"] is True


# ---------------------------------------------------------------------------
# Kernel 2: pres_content_extract (parser unit tests)
# ---------------------------------------------------------------------------

class TestParseDocument:
    """Unit tests for the parse_document function."""

    def test_headings(self):
        text = "# H1 Title\n\nSome text.\n\n## H2 Sub\n\nMore text."
        units = parse_document(text, "test.md", "test")
        headings = [u for u in units if u.type == UnitType.HEADING]
        assert len(headings) == 2
        assert headings[0].content == "H1 Title"
        assert headings[0].metadata["level"] == 1
        assert headings[1].content == "H2 Sub"
        assert headings[1].metadata["level"] == 2

    def test_heading_path(self):
        text = "# Top\n\n## Sub\n\nParagraph here."
        units = parse_document(text, "test.md", "test")
        paragraphs = [u for u in units if u.type == UnitType.PARAGRAPH]
        assert len(paragraphs) >= 1
        assert paragraphs[0].heading_path == ["Top", "Sub"]

    def test_code_block(self):
        text = "# Title\n\n```python\ndef foo():\n    pass\n```\n"
        units = parse_document(text, "test.md", "test")
        code = [u for u in units if u.type == UnitType.CODE_BLOCK]
        assert len(code) == 1
        assert code[0].metadata["language"] == "python"
        assert "def foo():" in code[0].content

    def test_mermaid(self):
        text = "```mermaid\ngraph TD\n    A --> B\n```\n"
        units = parse_document(text, "test.md", "test")
        mermaid = [u for u in units if u.type == UnitType.MERMAID]
        assert len(mermaid) == 1
        assert "graph" in mermaid[0].metadata.get("diagram_type", "")

    def test_math_block(self):
        text = "$$\nE = mc^2\n$$\n"
        units = parse_document(text, "test.md", "test")
        math = [u for u in units if u.type == UnitType.EQUATION_BLOCK]
        assert len(math) == 1
        assert math[0].metadata["latex"] == "E = mc^2"

    def test_table(self):
        text = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
        units = parse_document(text, "test.md", "test")
        tables = [u for u in units if u.type == UnitType.TABLE]
        assert len(tables) == 1
        assert tables[0].metadata["rows"] >= 2
        assert tables[0].metadata["cols"] == 2
        assert tables[0].metadata["has_header"] is True

    def test_blockquote(self):
        text = "> Line one\n> Line two\n"
        units = parse_document(text, "test.md", "test")
        bqs = [u for u in units if u.type == UnitType.BLOCKQUOTE]
        assert len(bqs) == 1

    def test_admonition(self):
        text = "> [!WARNING]\n> Be careful here\n"
        units = parse_document(text, "test.md", "test")
        adms = [u for u in units if u.type == UnitType.ADMONITION]
        assert len(adms) == 1
        assert adms[0].metadata.get("admonition_type") == "WARNING"

    def test_bullet_list(self):
        text = "- alpha\n- beta\n- gamma\n"
        units = parse_document(text, "test.md", "test")
        bl = [u for u in units if u.type == UnitType.BULLET_LIST]
        assert len(bl) == 1

    def test_numbered_list(self):
        text = "1. First\n2. Second\n3. Third\n"
        units = parse_document(text, "test.md", "test")
        nl = [u for u in units if u.type == UnitType.NUMBERED_LIST]
        assert len(nl) == 1

    def test_image_ref(self):
        text = "![Alt text](image.png)\n"
        units = parse_document(text, "test.md", "test")
        imgs = [u for u in units if u.type == UnitType.IMAGE_REF]
        assert len(imgs) == 1
        assert imgs[0].metadata["alt"] == "Alt text"
        assert imgs[0].metadata["path"] == "image.png"

    def test_front_matter(self):
        text = "---\ntitle: Test\n---\n\n# Heading\n"
        units = parse_document(text, "test.md", "test")
        fm = [u for u in units if u.type == UnitType.FRONT_MATTER]
        assert len(fm) == 1
        assert "title" in fm[0].metadata.get("keys", [])

    def test_paragraph(self):
        text = "This is a paragraph.\nContinued on next line.\n"
        units = parse_document(text, "test.md", "test")
        paras = [u for u in units if u.type == UnitType.PARAGRAPH]
        assert len(paras) >= 1

    def test_unit_id_format(self):
        text = "# Title\n\nSome text.\n"
        units = parse_document(text, "test.md", "test")
        for u in units:
            assert u.id.startswith("test:L")
            parts = u.id.split(":L")[1].split("-L")
            assert len(parts) == 2
            assert all(p.isdigit() for p in parts)

    def test_token_estimation(self):
        text = "A paragraph with some words for token estimation testing.\n"
        units = parse_document(text, "test.md", "test")
        for u in units:
            assert u.tokens > 0

    def test_all_13_types_in_sample(self):
        """Verify all 13 UnitType values can be extracted from SAMPLE_DOC_A."""
        units = parse_document(SAMPLE_DOC_A, "doc_a.md", "doc_a")
        types_found = {u.type for u in units}
        expected = {
            UnitType.FRONT_MATTER,
            UnitType.HEADING,
            UnitType.PARAGRAPH,
            UnitType.CODE_BLOCK,
            UnitType.EQUATION_BLOCK,
            UnitType.BLOCKQUOTE,
            UnitType.BULLET_LIST,
            UnitType.NUMBERED_LIST,
            UnitType.TABLE,
            UnitType.IMAGE_REF,
            UnitType.MERMAID,
            UnitType.ADMONITION,
        }
        missing = expected - types_found
        assert not missing, f"Missing UnitTypes: {missing}"


# ---------------------------------------------------------------------------
# Kernel 2: pres_content_extract (integration)
# ---------------------------------------------------------------------------

class TestPresContentExtract:

    def _run_scan(self, sample_project, workspace):
        kernel = PresFolderScanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
        )
        return kernel.run(inp)

    def test_extract_from_scan(self, sample_project, workspace):
        scan_out = self._run_scan(sample_project, workspace)
        assert scan_out.success

        kernel = PresContentExtractKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
            dependencies={"pres_folder_scan": scan_out.output_file},
        )
        output = kernel.run(inp)
        assert output.success

        data = output.data
        assert data["total_documents"] >= 2
        assert len(data["units"]) > 0
        assert data["total_tokens"] > 0

    def test_outline_tree(self, sample_project, workspace):
        scan_out = self._run_scan(sample_project, workspace)
        kernel = PresContentExtractKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
            dependencies={"pres_folder_scan": scan_out.output_file},
        )
        output = kernel.run(inp)
        assert output.success

        outline = output.data.get("outline", [])
        assert len(outline) > 0

    def test_corpus_roundtrip(self, sample_project, workspace):
        scan_out = self._run_scan(sample_project, workspace)
        kernel = PresContentExtractKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
            dependencies={"pres_folder_scan": scan_out.output_file},
        )
        output = kernel.run(inp)
        assert output.success

        corpus = ContentCorpus.from_dict(output.data)
        roundtrip = ContentCorpus.from_dict(corpus.to_dict())
        assert len(roundtrip.units) == len(corpus.units)
        assert roundtrip.total_tokens == corpus.total_tokens


# ---------------------------------------------------------------------------
# Kernel 3: pres_asset_catalog
# ---------------------------------------------------------------------------

class TestPresAssetCatalog:

    def _run_pipeline(self, sample_project, workspace):
        scan_k = PresFolderScanKernel()
        scan_out = scan_k.run(KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
        ))
        assert scan_out.success

        extract_k = PresContentExtractKernel()
        extract_out = extract_k.run(KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
            dependencies={"pres_folder_scan": scan_out.output_file},
        ))
        assert extract_out.success

        return scan_out, extract_out

    def test_catalog_from_pipeline(self, sample_project, workspace):
        scan_out, extract_out = self._run_pipeline(sample_project, workspace)

        kernel = PresAssetCatalogKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_folder_scan": scan_out.output_file,
                "pres_content_extract": extract_out.output_file,
            },
        )
        output = kernel.run(inp)
        assert output.success

        stats = output.data.get("statistics", {})
        assert stats["total_assets"] > 0
        assert stats["file_system_assets"] >= 1  # arch.svg

    def test_by_type_index(self, sample_project, workspace):
        scan_out, extract_out = self._run_pipeline(sample_project, workspace)

        kernel = PresAssetCatalogKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_folder_scan": scan_out.output_file,
                "pres_content_extract": extract_out.output_file,
            },
        )
        output = kernel.run(inp)
        by_type = output.data.get("by_type", {})

        # We expect at least images and equations from SAMPLE_DOC_A
        assert "image" in by_type
        assert "equation" in by_type

    def test_by_file_index(self, sample_project, workspace):
        scan_out, extract_out = self._run_pipeline(sample_project, workspace)

        kernel = PresAssetCatalogKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_folder_scan": scan_out.output_file,
                "pres_content_extract": extract_out.output_file,
            },
        )
        output = kernel.run(inp)
        by_file = output.data.get("by_file", {})

        # doc_a.md should have assets (code, math, table, mermaid, image)
        assert any("doc_a" in k for k in by_file.keys())

    def test_svg_dimensions(self, sample_project, workspace):
        scan_out, extract_out = self._run_pipeline(sample_project, workspace)

        kernel = PresAssetCatalogKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_folder_scan": scan_out.output_file,
                "pres_content_extract": extract_out.output_file,
            },
        )
        output = kernel.run(inp)
        assets = output.data.get("assets", [])

        svg_assets = [a for a in assets if a.get("format") == "svg"]
        assert len(svg_assets) >= 1
        dims = svg_assets[0].get("dimensions")
        assert dims is not None
        assert dims == [200, 100]

    def test_catalog_roundtrip(self, sample_project, workspace):
        scan_out, extract_out = self._run_pipeline(sample_project, workspace)

        kernel = PresAssetCatalogKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_folder_scan": scan_out.output_file,
                "pres_content_extract": extract_out.output_file,
            },
        )
        output = kernel.run(inp)
        # Extract just the catalog fields (without statistics)
        catalog_data = {
            "assets": output.data["assets"],
            "by_type": output.data["by_type"],
            "by_file": output.data["by_file"],
        }
        catalog = AssetCatalog.from_dict(catalog_data)
        roundtrip = AssetCatalog.from_dict(catalog.to_dict())
        assert len(roundtrip.assets) == len(catalog.assets)


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_three_kernel_chain(self, sample_project, workspace):
        """Run the complete 3-kernel S1 pipeline end-to-end."""
        # K1: folder scan
        scan_k = PresFolderScanKernel()
        scan_out = scan_k.run(KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
        ))
        assert scan_out.success, f"Scan failed: {scan_out.errors}"

        # K2: content extract
        extract_k = PresContentExtractKernel()
        extract_out = extract_k.run(KernelInput(
            workspace=workspace,
            config={"folder_path": str(sample_project)},
            dependencies={"pres_folder_scan": scan_out.output_file},
        ))
        assert extract_out.success, f"Extract failed: {extract_out.errors}"

        # K3: asset catalog
        catalog_k = PresAssetCatalogKernel()
        catalog_out = catalog_k.run(KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_folder_scan": scan_out.output_file,
                "pres_content_extract": extract_out.output_file,
            },
        ))
        assert catalog_out.success, f"Catalog failed: {catalog_out.errors}"

        # All outputs persisted
        assert scan_out.output_file.exists()
        assert extract_out.output_file.exists()
        assert catalog_out.output_file.exists()

        # Summaries are reasonable
        assert len(scan_out.summary) > 10
        assert len(extract_out.summary) > 10
        assert len(catalog_out.summary) > 10
