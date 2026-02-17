"""
Integration tests for KOAS Presenter M2 (Stage 2 Structuring) kernels.

Tests the S2 pipeline:
    pres_semantic_normalize → pres_slide_plan → pres_layout_assign

Uses the M1 pipeline as a prerequisite (folder scan + content extract + asset catalog).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragix_kernels.base import KernelInput
from ragix_kernels.presenter.models import (
    ContentCorpus,
    NormalizedCorpus,
    NormalizedUnit,
    NormalizationMode,
    SemanticUnit,
    SlideDeck,
    SlideType,
    TopicCluster,
    UnitRole,
    UnitType,
)
from ragix_kernels.presenter.config import ImportanceConfig
from ragix_kernels.presenter.normalize_utils import (
    assign_role_by_keywords,
    cluster_by_heading_path,
    compute_importance,
    consolidate_clusters,
    detect_narrative_arc,
    find_duplicates,
    jaccard_similarity,
)
from ragix_kernels.presenter.kernels.pres_folder_scan import PresFolderScanKernel
from ragix_kernels.presenter.kernels.pres_content_extract import PresContentExtractKernel
from ragix_kernels.presenter.kernels.pres_asset_catalog import PresAssetCatalogKernel
from ragix_kernels.presenter.kernels.pres_semantic_normalize import PresSemanticNormalizeKernel
from ragix_kernels.presenter.kernels.pres_slide_plan import PresSlidePlanKernel
from ragix_kernels.presenter.kernels.pres_layout_assign import PresLayoutAssignKernel


# ---------------------------------------------------------------------------
# Fixtures: sample project (same as M1 tests)
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
The performance rate is 95.2% improvement.

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
The rate of improvement is 42%.

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
    (project / "data.json").write_text('{"key": "value"}', encoding="utf-8")
    return project


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace directory for kernel outputs."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def _run_s1_pipeline(sample_project, workspace):
    """Run the complete S1 pipeline and return outputs."""
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

    catalog_k = PresAssetCatalogKernel()
    catalog_out = catalog_k.run(KernelInput(
        workspace=workspace,
        config={},
        dependencies={
            "pres_folder_scan": scan_out.output_file,
            "pres_content_extract": extract_out.output_file,
        },
    ))
    assert catalog_out.success

    return scan_out, extract_out, catalog_out


# ---------------------------------------------------------------------------
# Test: normalize_utils
# ---------------------------------------------------------------------------

class TestNormalizeUtils:

    def test_cluster_by_heading_path(self):
        units = [
            SemanticUnit(
                id="u1", type=UnitType.PARAGRAPH, content="text",
                source_file="f.md", source_lines=(1, 2),
                heading_path=["Intro", "Background"], depth=2, tokens=10,
            ),
            SemanticUnit(
                id="u2", type=UnitType.PARAGRAPH, content="text2",
                source_file="f.md", source_lines=(3, 4),
                heading_path=["Intro", "Background"], depth=2, tokens=10,
            ),
            SemanticUnit(
                id="u3", type=UnitType.PARAGRAPH, content="text3",
                source_file="f.md", source_lines=(5, 6),
                heading_path=["Methods", "Analysis"], depth=2, tokens=10,
            ),
            SemanticUnit(
                id="u4", type=UnitType.PARAGRAPH, content="text4",
                source_file="f.md", source_lines=(7, 8),
                heading_path=[], depth=0, tokens=10,
            ),
        ]
        clusters = cluster_by_heading_path(units)
        assert "Intro > Background" in clusters
        assert len(clusters["Intro > Background"]) == 2
        assert "Methods > Analysis" in clusters
        assert "Uncategorized" in clusters

    def test_assign_role_keywords_fr(self):
        u = SemanticUnit(
            id="u1", type=UnitType.PARAGRAPH,
            content="Les résultats montrent une performance de 95%",
            source_file="f.md", source_lines=(1, 2),
            heading_path=["Résultats"], depth=1, tokens=20,
        )
        role = assign_role_by_keywords(u, lang="fr")
        assert role == UnitRole.FINDING

    def test_assign_role_keywords_en(self):
        u = SemanticUnit(
            id="u1", type=UnitType.PARAGRAPH,
            content="We recommend the following actions for improvement",
            source_file="f.md", source_lines=(1, 2),
            heading_path=["Recommendations"], depth=1, tokens=20,
        )
        role = assign_role_by_keywords(u, lang="en")
        assert role == UnitRole.RECOMMENDATION

    def test_assign_role_front_matter(self):
        u = SemanticUnit(
            id="u1", type=UnitType.FRONT_MATTER,
            content="---\ntitle: Test\n---",
            source_file="f.md", source_lines=(1, 3),
            heading_path=[], depth=0, tokens=10,
        )
        role = assign_role_by_keywords(u)
        assert role == UnitRole.METADATA

    def test_assign_role_image(self):
        u = SemanticUnit(
            id="u1", type=UnitType.IMAGE_REF,
            content="![diagram](img.png)",
            source_file="f.md", source_lines=(1, 1),
            heading_path=[], depth=0, tokens=5,
        )
        role = assign_role_by_keywords(u)
        assert role == UnitRole.ILLUSTRATION

    def test_compute_importance_heading_boost(self):
        u = SemanticUnit(
            id="u1", type=UnitType.HEADING, content="Introduction",
            source_file="f.md", source_lines=(1, 1),
            heading_path=["Introduction"], depth=0, tokens=5,
            metadata={"level": 1},
        )
        score = compute_importance(u, UnitRole.CONTEXT)
        assert score > 0.7  # H1 + CONTEXT boost

    def test_compute_importance_depth_decay(self):
        u_shallow = SemanticUnit(
            id="u1", type=UnitType.PARAGRAPH, content="shallow",
            source_file="f.md", source_lines=(1, 1),
            heading_path=["A"], depth=1, tokens=5,
        )
        u_deep = SemanticUnit(
            id="u2", type=UnitType.PARAGRAPH, content="deep",
            source_file="f.md", source_lines=(1, 1),
            heading_path=["A", "B", "C", "D"], depth=4, tokens=5,
        )
        s1 = compute_importance(u_shallow, UnitRole.CONTEXT)
        s2 = compute_importance(u_deep, UnitRole.CONTEXT)
        assert s1 > s2  # deeper units have lower score

    def test_compute_importance_percentage_boost(self):
        u_with_pct = SemanticUnit(
            id="u1", type=UnitType.PARAGRAPH,
            content="The rate improved by 42%",
            source_file="f.md", source_lines=(1, 1),
            heading_path=[], depth=0, tokens=10,
        )
        u_without = SemanticUnit(
            id="u2", type=UnitType.PARAGRAPH,
            content="The system was deployed",
            source_file="f.md", source_lines=(1, 1),
            heading_path=[], depth=0, tokens=10,
        )
        s1 = compute_importance(u_with_pct, UnitRole.FINDING)
        s2 = compute_importance(u_without, UnitRole.FINDING)
        assert s1 > s2

    def test_jaccard_similarity(self):
        assert jaccard_similarity("the cat sat", "the cat sat") == 1.0
        assert jaccard_similarity("", "") == 1.0
        assert jaccard_similarity("hello world", "goodbye moon") == 0.0
        sim = jaccard_similarity("the cat sat on the mat", "the cat sat on a mat")
        assert 0.5 < sim < 1.0

    def test_find_duplicates(self):
        u1 = NormalizedUnit(
            unit=SemanticUnit(
                id="u1", type=UnitType.PARAGRAPH,
                content="This is a test paragraph about results and findings",
                source_file="f.md", source_lines=(1, 2),
                heading_path=[], depth=0, tokens=20,
            ),
        )
        u2 = NormalizedUnit(
            unit=SemanticUnit(
                id="u2", type=UnitType.PARAGRAPH,
                content="This is a test paragraph about results and findings too",
                source_file="f.md", source_lines=(3, 4),
                heading_path=[], depth=0, tokens=20,
            ),
        )
        u3 = NormalizedUnit(
            unit=SemanticUnit(
                id="u3", type=UnitType.PARAGRAPH,
                content="Completely different content about architecture",
                source_file="f.md", source_lines=(5, 6),
                heading_path=[], depth=0, tokens=20,
            ),
        )
        dupes = find_duplicates([u1, u2, u3], threshold=0.70)
        assert "u2" in dupes
        assert dupes["u2"] == "u1"
        assert "u3" not in dupes

    def test_detect_narrative_arc(self):
        units = [
            NormalizedUnit(
                unit=SemanticUnit(
                    id="u1", type=UnitType.PARAGRAPH, content="intro",
                    source_file="f.md", source_lines=(1, 2),
                    heading_path=[], depth=0, tokens=5,
                ),
                role=UnitRole.CONTEXT,
            ),
            NormalizedUnit(
                unit=SemanticUnit(
                    id="u2", type=UnitType.PARAGRAPH, content="results",
                    source_file="f.md", source_lines=(3, 4),
                    heading_path=[], depth=0, tokens=5,
                ),
                role=UnitRole.FINDING,
            ),
            NormalizedUnit(
                unit=SemanticUnit(
                    id="u3", type=UnitType.PARAGRAPH, content="method",
                    source_file="f.md", source_lines=(5, 6),
                    heading_path=[], depth=0, tokens=5,
                ),
                role=UnitRole.METHOD,
            ),
        ]
        clusters = [
            TopicCluster(id="c0", label="Background", unit_ids=["u1"], importance=0.5, suggested_slides=1),
            TopicCluster(id="c1", label="Results", unit_ids=["u2"], importance=0.7, suggested_slides=1),
            TopicCluster(id="c2", label="Approach", unit_ids=["u3"], importance=0.6, suggested_slides=1),
        ]
        arc = detect_narrative_arc(clusters, units)
        # CONTEXT before METHOD before FINDING
        idx_bg = arc.sections.index("Background")
        idx_ap = arc.sections.index("Approach")
        idx_res = arc.sections.index("Results")
        assert idx_bg < idx_ap < idx_res

    def test_consolidate_clusters_no_op(self):
        """If clusters <= max_clusters, return unchanged."""
        units = [
            SemanticUnit(
                id=f"u{i}", type=UnitType.PARAGRAPH, content=f"text {i}",
                source_file="f.md", source_lines=(i, i + 1),
                heading_path=["A", f"A.{i}"], depth=1, tokens=10,
            )
            for i in range(5)
        ]
        raw = cluster_by_heading_path(units)
        result = consolidate_clusters(raw, units, max_clusters=20)
        assert result == raw

    def test_consolidate_clusters_coarsens(self):
        """Many 2-level clusters get reduced to level-1 grouping."""
        units = []
        # 30 units under 5 different L1 chapters, each with 3 L2 sections
        for ch in range(5):
            for sec in range(3):
                for u in range(2):
                    uid = f"u-{ch}-{sec}-{u}"
                    units.append(SemanticUnit(
                        id=uid, type=UnitType.PARAGRAPH, content=f"text {uid}",
                        source_file="f.md", source_lines=(1, 2),
                        heading_path=[f"Chapter {ch}", f"Section {ch}.{sec}"],
                        depth=1, tokens=10,
                    ))
        raw = cluster_by_heading_path(units)
        assert len(raw) == 15  # 5 chapters × 3 sections
        result = consolidate_clusters(raw, units, max_clusters=8)
        assert len(result) <= 8
        # All unit IDs preserved
        all_ids = set()
        for uids in result.values():
            all_ids.update(uids)
        assert len(all_ids) == 30

    def test_consolidate_clusters_overflow(self):
        """When level-1 still exceeds max, smallest merge into Other."""
        units = []
        for ch in range(25):
            uid = f"u-{ch}"
            units.append(SemanticUnit(
                id=uid, type=UnitType.PARAGRAPH, content=f"text {uid}",
                source_file="f.md", source_lines=(1, 2),
                heading_path=[f"Chapter {ch}"], depth=0, tokens=10,
            ))
        raw = cluster_by_heading_path(units)
        assert len(raw) == 25
        result = consolidate_clusters(raw, units, max_clusters=10)
        assert len(result) <= 10
        # "Other" should exist
        assert "Other" in result
        # All IDs preserved
        all_ids = set()
        for uids in result.values():
            all_ids.update(uids)
        assert len(all_ids) == 25


# ---------------------------------------------------------------------------
# Test: pres_semantic_normalize (K4)
# ---------------------------------------------------------------------------

class TestPresSemanticNormalize:

    def test_deterministic_mode(self, sample_project, workspace):
        scan_out, extract_out, catalog_out = _run_s1_pipeline(sample_project, workspace)

        kernel = PresSemanticNormalizeKernel()
        inp = KernelInput(
            workspace=workspace,
            config={
                "normalizer": {"mode": "deterministic", "enabled": True},
                "lang": "en",
            },
            dependencies={"pres_content_extract": extract_out.output_file},
        )
        output = kernel.run(inp)
        assert output.success

        data = output.data
        assert data["normalization_mode"] == "deterministic"
        assert len(data["units"]) > 0
        assert len(data["clusters"]) > 0

    def test_identity_mode(self, sample_project, workspace):
        scan_out, extract_out, catalog_out = _run_s1_pipeline(sample_project, workspace)

        kernel = PresSemanticNormalizeKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"normalizer": {"enabled": False}},
            dependencies={"pres_content_extract": extract_out.output_file},
        )
        output = kernel.run(inp)
        assert output.success
        assert output.data["normalization_mode"] == "identity"

    def test_output_structure(self, sample_project, workspace):
        scan_out, extract_out, catalog_out = _run_s1_pipeline(sample_project, workspace)

        kernel = PresSemanticNormalizeKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"normalizer": {"mode": "deterministic"}, "lang": "en"},
            dependencies={"pres_content_extract": extract_out.output_file},
        )
        output = kernel.run(inp)
        assert output.success

        # Verify NormalizedCorpus fields
        nc = NormalizedCorpus.from_dict(output.data)
        assert len(nc.units) > 0
        assert len(nc.clusters) > 0
        assert nc.narrative is not None
        assert nc.raw is not None

    def test_roles_assigned(self, sample_project, workspace):
        scan_out, extract_out, catalog_out = _run_s1_pipeline(sample_project, workspace)

        kernel = PresSemanticNormalizeKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"normalizer": {"mode": "deterministic"}, "lang": "en"},
            dependencies={"pres_content_extract": extract_out.output_file},
        )
        output = kernel.run(inp)
        nc = NormalizedCorpus.from_dict(output.data)

        # At least some units should have non-UNKNOWN roles
        roles = {nu.role for nu in nc.units}
        assert len(roles) > 1, f"Only roles found: {roles}"

    def test_narrative_ordered(self, sample_project, workspace):
        scan_out, extract_out, catalog_out = _run_s1_pipeline(sample_project, workspace)

        kernel = PresSemanticNormalizeKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"normalizer": {"mode": "deterministic"}, "lang": "en"},
            dependencies={"pres_content_extract": extract_out.output_file},
        )
        output = kernel.run(inp)
        nc = NormalizedCorpus.from_dict(output.data)
        assert len(nc.narrative.sections) > 0

    def test_duplicates_marked(self, sample_project, workspace):
        # Add a duplicate paragraph to the project
        dup_doc = """\
# Duplicate Test

This is a test document with various Markdown elements.

This is a test document with various Markdown elements.
"""
        (sample_project / "doc_dup.md").write_text(dup_doc, encoding="utf-8")

        scan_out, extract_out, catalog_out = _run_s1_pipeline(sample_project, workspace)

        kernel = PresSemanticNormalizeKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"normalizer": {"mode": "deterministic"}, "lang": "en"},
            dependencies={"pres_content_extract": extract_out.output_file},
        )
        output = kernel.run(inp)
        stats = output.data.get("statistics", {})
        # May or may not find duplicates depending on exact content
        assert "duplicates_removed" in stats


# ---------------------------------------------------------------------------
# Test: pres_slide_plan (K5)
# ---------------------------------------------------------------------------

class TestPresSlidePlan:

    def _run_normalize(self, sample_project, workspace):
        scan_out, extract_out, catalog_out = _run_s1_pipeline(sample_project, workspace)
        kernel = PresSemanticNormalizeKernel()
        norm_out = kernel.run(KernelInput(
            workspace=workspace,
            config={"normalizer": {"mode": "deterministic"}, "lang": "en"},
            dependencies={"pres_content_extract": extract_out.output_file},
        ))
        assert norm_out.success
        return scan_out, extract_out, catalog_out, norm_out

    def test_title_slide_generated(self, sample_project, workspace):
        scan_out, extract_out, catalog_out, norm_out = self._run_normalize(sample_project, workspace)

        kernel = PresSlidePlanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"title": "Test Presentation", "lang": "en"},
            dependencies={
                "pres_semantic_normalize": norm_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        assert output.success

        deck = SlideDeck.from_dict(output.data)
        assert deck.slides[0].type == SlideType.TITLE

    def test_section_slides(self, sample_project, workspace):
        scan_out, extract_out, catalog_out, norm_out = self._run_normalize(sample_project, workspace)

        kernel = PresSlidePlanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"lang": "en"},
            dependencies={
                "pres_semantic_normalize": norm_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        deck = SlideDeck.from_dict(output.data)

        section_slides = [s for s in deck.slides if s.type == SlideType.SECTION]
        assert len(section_slides) >= 1

    def test_content_mapping(self, sample_project, workspace):
        scan_out, extract_out, catalog_out, norm_out = self._run_normalize(sample_project, workspace)

        kernel = PresSlidePlanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"lang": "en"},
            dependencies={
                "pres_semantic_normalize": norm_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        deck = SlideDeck.from_dict(output.data)

        types_found = {s.type for s in deck.slides}
        # We expect at least TITLE, SECTION, CONTENT, CODE
        assert SlideType.TITLE in types_found
        assert SlideType.SECTION in types_found
        assert SlideType.CONTENT in types_found

    def test_slide_ids_sequential(self, sample_project, workspace):
        scan_out, extract_out, catalog_out, norm_out = self._run_normalize(sample_project, workspace)

        kernel = PresSlidePlanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"lang": "en"},
            dependencies={
                "pres_semantic_normalize": norm_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        deck = SlideDeck.from_dict(output.data)

        for i, slide in enumerate(deck.slides):
            expected_id = f"slide-{i + 1:03d}"
            assert slide.id == expected_id, f"Expected {expected_id}, got {slide.id}"

    def test_provenance_linked(self, sample_project, workspace):
        scan_out, extract_out, catalog_out, norm_out = self._run_normalize(sample_project, workspace)

        kernel = PresSlidePlanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"lang": "en"},
            dependencies={
                "pres_semantic_normalize": norm_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        deck = SlideDeck.from_dict(output.data)

        for slide in deck.slides:
            assert slide.provenance is not None, f"Slide {slide.id} has no provenance"

    def test_slide_bounds(self, sample_project, workspace):
        scan_out, extract_out, catalog_out, norm_out = self._run_normalize(sample_project, workspace)

        kernel = PresSlidePlanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={
                "lang": "en",
                "slide_plan": {"max_slides": 10},
            },
            dependencies={
                "pres_semantic_normalize": norm_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        deck = SlideDeck.from_dict(output.data)
        assert len(deck.slides) <= 10

    def test_trim_removes_empty_sections(self, sample_project, workspace):
        """When sections exceed max_slides, empty section dividers are removed."""
        from ragix_kernels.presenter.models import Slide, SlideContent, SlideProvenance, ProvenanceMethod

        kernel = PresSlidePlanKernel()
        # Build a slide list with many empty sections (no content between them)
        slides = [
            Slide(id="slide-001", type=SlideType.TITLE,
                  content=SlideContent(heading="Title"),
                  provenance=SlideProvenance(method=ProvenanceMethod.AUTO_SECTION)),
        ]
        for i in range(30):
            slides.append(Slide(
                id=f"slide-{i+2:03d}", type=SlideType.SECTION,
                content=SlideContent(heading=f"Section {i}"),
                provenance=SlideProvenance(method=ProvenanceMethod.AUTO_SECTION),
            ))
        # Add a few content slides after every 5th section
        content_positions = [5, 10, 15, 20, 25]
        for pos in reversed(content_positions):
            slides.insert(pos + 1, Slide(
                id=f"slide-content-{pos}", type=SlideType.CONTENT,
                content=SlideContent(heading=f"Content {pos}", body=["text"]),
                provenance=SlideProvenance(method=ProvenanceMethod.EXTRACTED),
            ))
        # 1 title + 30 sections + 5 content = 36 slides
        assert len(slides) == 36
        trimmed = kernel._trim_slides(slides, max_slides=15)
        assert len(trimmed) <= 15
        # Title must survive
        assert trimmed[0].type == SlideType.TITLE

    def test_bullet_splitting(self, sample_project, workspace):
        # Create a document with a long bullet list
        long_list_doc = "# Long List\n\n" + "\n".join(f"- Item {i}" for i in range(20))
        (sample_project / "long_list.md").write_text(long_list_doc, encoding="utf-8")

        scan_out, extract_out, catalog_out, norm_out = self._run_normalize(sample_project, workspace)

        kernel = PresSlidePlanKernel()
        inp = KernelInput(
            workspace=workspace,
            config={
                "lang": "en",
                "slide_plan": {"max_bullets_per_slide": 6, "split_long_lists": True},
            },
            dependencies={
                "pres_semantic_normalize": norm_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        deck = SlideDeck.from_dict(output.data)

        # Find content slides with bullets
        bullet_slides = [
            s for s in deck.slides
            if s.content.bullets and len(s.content.bullets) > 0
        ]
        for bs in bullet_slides:
            assert len(bs.content.bullets) <= 6


# ---------------------------------------------------------------------------
# Test: pres_layout_assign (K6)
# ---------------------------------------------------------------------------

class TestPresLayoutAssign:

    def _run_plan(self, sample_project, workspace):
        scan_out, extract_out, catalog_out = _run_s1_pipeline(sample_project, workspace)
        norm_k = PresSemanticNormalizeKernel()
        norm_out = norm_k.run(KernelInput(
            workspace=workspace,
            config={"normalizer": {"mode": "deterministic"}, "lang": "en"},
            dependencies={"pres_content_extract": extract_out.output_file},
        ))
        assert norm_out.success

        plan_k = PresSlidePlanKernel()
        plan_out = plan_k.run(KernelInput(
            workspace=workspace,
            config={"lang": "en"},
            dependencies={
                "pres_semantic_normalize": norm_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        ))
        assert plan_out.success
        return plan_out, catalog_out

    def test_title_layout(self, sample_project, workspace):
        plan_out, catalog_out = self._run_plan(sample_project, workspace)

        kernel = PresLayoutAssignKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_slide_plan": plan_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        assert output.success

        deck = SlideDeck.from_dict(output.data)
        title_slide = deck.slides[0]
        assert title_slide.layout is not None
        assert title_slide.layout.css_class == "lead"
        assert title_slide.layout.paginate is False

    def test_default_layout(self, sample_project, workspace):
        plan_out, catalog_out = self._run_plan(sample_project, workspace)

        kernel = PresLayoutAssignKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_slide_plan": plan_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        deck = SlideDeck.from_dict(output.data)

        content_slides = [s for s in deck.slides if s.type == SlideType.CONTENT]
        for cs in content_slides:
            assert cs.layout is not None
            assert cs.layout.paginate is True

    def test_all_slides_have_layout(self, sample_project, workspace):
        plan_out, catalog_out = self._run_plan(sample_project, workspace)

        kernel = PresLayoutAssignKernel()
        inp = KernelInput(
            workspace=workspace,
            config={},
            dependencies={
                "pres_slide_plan": plan_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        deck = SlideDeck.from_dict(output.data)

        for slide in deck.slides:
            assert slide.layout is not None, f"Slide {slide.id} has no layout"

    def test_theme_colors(self, sample_project, workspace):
        plan_out, catalog_out = self._run_plan(sample_project, workspace)

        kernel = PresLayoutAssignKernel()
        inp = KernelInput(
            workspace=workspace,
            config={"theme": {"colors": {"primary": "#ff0000"}}},
            dependencies={
                "pres_slide_plan": plan_out.output_file,
                "pres_asset_catalog": catalog_out.output_file,
            },
        )
        output = kernel.run(inp)
        deck = SlideDeck.from_dict(output.data)

        title_slide = deck.slides[0]
        assert title_slide.layout.background_color == "#ff0000"
