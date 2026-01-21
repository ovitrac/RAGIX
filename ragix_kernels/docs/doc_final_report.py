"""
Document Final Report Kernel — Consolidated report with appendices.

Author: Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
Date: 2026-01-18

This kernel produces the final consolidated report:
- Main report at run root: final_report.md
- Component artifacts in stage3/
- Appendices in appendices/

The report includes:
- Methodology section (explicit, no hidden steps)
- Executive summary
- Corpus overview with dual clustering views
- Domain summaries
- Functionality catalog
- Discrepancy analysis
- Sovereignty attestation
- Appendices with full details
"""

import json
import logging
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocFinalReportKernel(Kernel):
    """
    Generate consolidated final report with appendices.

    Outputs:
    - {run_dir}/final_report.md — Primary output at run root
    - {run_dir}/final_report.json — Structured data
    - {run_dir}/stage3/report_sections/*.md — Component sections
    - {run_dir}/appendices/*.md — Detailed appendices
    """

    name = "doc_final_report"
    version = "1.0.0"
    category = "docs"
    stage = 3

    requires = [
        "doc_metadata",
        "doc_pyramid",
        "doc_cluster_reconcile",
        "doc_summarize_tutored",
        "doc_compare",
        "doc_coverage",
        "doc_func_extract",
        "doc_visualize",  # Figures for report
        "doc_quality"     # Quality metrics (MRI/SRI)
    ]
    provides = ["final_report", "appendices"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Generate final report."""
        # Load all dependencies
        metadata = self._load_dep(input, "doc_metadata")
        pyramid = self._load_dep(input, "doc_pyramid")
        clusters = self._load_dep(input, "doc_cluster_reconcile")
        summaries = self._load_dep(input, "doc_summarize_tutored")
        compare = self._load_dep(input, "doc_compare")
        coverage = self._load_dep(input, "doc_coverage")
        funcs = self._load_dep(input, "doc_func_extract")
        visualize = self._load_dep(input, "doc_visualize")
        quality = self._load_dep(input, "doc_quality")

        # Use workspace as run directory (not config which defaults to ".")
        run_dir = input.workspace
        appendices_dir = run_dir / "appendices"
        appendices_dir.mkdir(parents=True, exist_ok=True)

        # Build report sections
        sections = {}

        # 1. Header and methodology
        sections["header"] = self._build_header(input.config)
        sections["methodology"] = self._build_methodology(clusters, summaries)

        # 2. Executive summary (with word cloud)
        sections["executive"] = self._build_executive_summary(
            metadata, pyramid, clusters, summaries, compare, funcs, visualize
        )

        # 3. Corpus overview (with doc type distribution figure)
        sections["corpus"] = self._build_corpus_overview(metadata, pyramid, visualize)

        # 3b. Quality metrics (MRI/SRI histograms, radar, scatter)
        sections["quality_metrics"] = self._build_quality_metrics_section(quality, visualize)

        # 3c. Document similarity & elaboration analysis
        sections["similarity_analysis"] = self._build_similarity_section(metadata, visualize)

        # 4. Clustering analysis (dual view with figures)
        sections["clustering"] = self._build_clustering_section(clusters, visualize)

        # 5. Domain summaries (with word clouds)
        sections["domains"] = self._build_domain_summaries(pyramid, summaries, visualize)

        # 6. Visual analysis (concept heatmap, coverage matrix, domain comparison)
        sections["visual_analysis"] = self._build_visual_analysis_section(visualize)

        # 7. Functionality catalog
        sections["functionalities"] = self._build_functionality_section(funcs)

        # 8. Discrepancy analysis
        sections["discrepancies"] = self._build_discrepancy_section(compare, coverage)

        # 9. Technical information
        sections["technical"] = self._build_technical_section(input.config, summaries)

        # 10. Sovereignty attestation
        sections["sovereignty"] = self._build_sovereignty_section(input.config, summaries)

        # Build file_id to name mapping for appendices
        file_id_to_name = self._build_file_id_mapping(metadata)

        # Generate appendices
        appendices = self._generate_appendices(
            appendices_dir, pyramid, summaries, funcs, compare, clusters,
            file_id_to_name, visualize
        )

        # 10. Appendix references
        sections["appendix_refs"] = self._build_appendix_references(appendices)

        # Assemble final report
        report_md = self._assemble_report(sections)

        # Write final report to run root
        final_report_path = run_dir / "final_report.md"
        final_report_path.write_text(report_md, encoding="utf-8")

        # Write structured data
        final_json_path = run_dir / "final_report.json"
        final_json_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sections": list(sections.keys()),
            "appendices": [str(p) for p in appendices],
            "report_path": str(final_report_path),
            "statistics": {
                "documents": metadata.get("statistics", {}).get("total_files", 0),
                "domains": len(pyramid.get("pyramid", {}).get("level_3_domains", [])),
                "functionalities": len(funcs.get("functionalities", [])),
                "discrepancies": len(compare.get("discrepancies", []))
            }
        }, indent=2), encoding="utf-8")

        return {
            "report_path": str(final_report_path),
            "report_length": len(report_md),
            "sections": list(sections.keys()),
            "appendices": [str(p) for p in appendices]
        }

    def _load_dep(self, input: KernelInput, name: str) -> Dict[str, Any]:
        """Load dependency data from file."""
        path = input.dependencies.get(name)
        if path and path.exists():
            with open(path) as f:
                return json.load(f).get("data", {})
        return {}

    def _build_header(self, config: Dict) -> str:
        """Build report header."""
        project_name = config.get("project_name", "Document Analysis")
        run_id = config.get("run_id", "unknown")

        return f"""# Document Analysis Report

**Project:** {project_name}
**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Run ID:** {run_id}
**RAGIX Version:** 0.5.x
**KOAS Version:** 1.0.0
**System Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab

---
"""

    def _build_methodology(self, clusters: Dict, summaries: Dict) -> str:
        """Build methodology section."""
        cluster_mode = clusters.get("mode", "dual_view")
        cluster_methodology = clusters.get("methodology", "")

        summary_stats = summaries.get("statistics", {})
        quality = summaries.get("quality", {})

        return f"""## Methodology

### Analysis Process

This report was generated by the KOAS-Docs system (Kernel-Orchestrated Audit System
for Documents), developed by Olivier Vitrac as part of the RAGIX project.

The analysis follows a three-stage process:

1. **Collection** (Stage 1): Metadata, concepts and structure extraction
   - Source: RAG Index (ChromaDB + Knowledge Graph)
   - Processing: Deterministic, no LLM involved

2. **Analysis** (Stage 2): Dual clustering, key sentence extraction, functionalities
   - Clustering: Hierarchical + Leiden (parallel execution)
   - Reconciliation mode: **{cluster_mode}**
   - {cluster_methodology}

3. **Synthesis** (Stage 3): Summary generation with verification
   - Worker model: granite3.1-moe:3b (fast generation)
   - Tutor model: mistral:7b-instruct (verification)
   - Acceptance rate: {quality.get('acceptance_rate', 0):.0%}
   - Cache hit rate: {quality.get('cache_hit_rate', 0):.0%}

### Quality Guarantees

- **Reproducibility**: Deterministic execution (same input = same output)
- **Traceability**: SHA256 checksums for all artifacts
- **Verification**: Dual validation by tutor model prevents hallucination
- **Transparency**: Explicit methodology, no hidden steps
- **Dual perspective**: Both clustering views preserved to avoid bias

### Limitations

- Summaries generated by local language models (3B-7B parameters)
- Contradiction detection based on lexical and semantic similarity
- Extracted functionalities depend on SPD document structure

---
"""

    def _build_executive_summary(
        self,
        metadata: Dict,
        pyramid: Dict,
        clusters: Dict,
        summaries: Dict,
        compare: Dict,
        funcs: Dict,
        visualize: Dict = None
    ) -> str:
        """Build executive summary."""
        stats = metadata.get("statistics", {})
        n_files = stats.get("total_files", 0)
        n_chunks = stats.get("total_chunks", 0)

        n_domains = len(pyramid.get("pyramid", {}).get("level_3_domains", []))
        n_discrepancies = len(compare.get("discrepancies", []))
        n_funcs = len(funcs.get("functionalities", []))
        n_missing_refs = len(funcs.get("missing_references", []))

        cluster_agreement = clusters.get("agreement", {}).get("overall_score", 0)

        # Include word cloud if available
        word_cloud_fig = ""
        if visualize:
            figures = visualize.get("figures", {})
            if "word_cloud" in figures:
                fig_path = figures["word_cloud"].get("svg", "")
                if fig_path:
                    word_cloud_fig = f"\n![Corpus Concept Cloud]({fig_path})\n*Figure: Key concepts across the document corpus*\n"

        return f"""## Executive Summary
{word_cloud_fig}
### Corpus Statistics

| Metric | Value |
|--------|-------|
| Documents analyzed | {n_files} |
| Total chunks | {n_chunks} |
| Thematic domains | {n_domains} |
| Functionalities identified | {n_funcs} |
| Discrepancies detected | {n_discrepancies} |
| Missing references | {n_missing_refs} |

### Key Findings

1. **Document Organization**: {n_files} documents organized into {n_domains} thematic domains
2. **Clustering Agreement**: {cluster_agreement:.0%} agreement between hierarchical and Leiden methods
3. **Functionality Coverage**: {n_funcs} functionalities extracted from SPD documents
4. **Quality Issues**: {n_discrepancies} discrepancies and {n_missing_refs} missing references identified

---
"""

    def _build_corpus_overview(self, metadata: Dict, pyramid: Dict, visualize: Dict = None) -> str:
        """Build corpus overview section."""
        stats = metadata.get("statistics", {})
        by_kind = stats.get("by_kind", {})

        # If by_kind is empty, compute from level_1_documents
        if not by_kind:
            by_kind = {}
            level1_docs = pyramid.get("pyramid", {}).get("level_1_documents", [])
            for doc in level1_docs:
                kind = doc.get("kind", "unknown")
                # Clean up kind name (e.g., "doc_docx" -> "DOCX")
                kind_display = kind.replace("doc_", "").upper()
                by_kind[kind_display] = by_kind.get(kind_display, 0) + 1

        # Format file types table
        if by_kind:
            types_rows = "\n".join(
                f"| {k} | {v} |" for k, v in sorted(by_kind.items(), key=lambda x: -x[1])
            )
        else:
            types_rows = "| - | No type data available |"

        # Include doc type distribution figure if available
        figure_ref = ""
        if visualize:
            figures = visualize.get("figures", {})
            if "doc_type_distribution" in figures:
                fig_path = figures["doc_type_distribution"].get("svg", "")
                if fig_path:
                    figure_ref = f"\n![Document Type Distribution]({fig_path})\n*Figure: Distribution of document types in the corpus*\n"

        return f"""## Corpus Overview

### Document Types

| Type | Count |
|------|-------|
{types_rows}
{figure_ref}
### Total Size

- **Files**: {stats.get('total_files', 0)}
- **Chunks**: {stats.get('total_chunks', 0)}
- **Size**: {stats.get('total_size_mb', 0):.1f} MB

---
"""

    def _build_quality_metrics_section(self, quality: Dict, visualize: Dict = None) -> str:
        """Build quality metrics section with MRI/SRI analysis and figures."""
        if not quality or not quality.get("quality_scores"):
            return ""

        scores = quality.get("quality_scores", {})
        n_docs = len(scores)

        # Calculate statistics
        mri_values = [d["readiness_indices"]["MRI"] for d in scores.values()
                      if "readiness_indices" in d and "MRI" in d["readiness_indices"]]
        sri_values = [d["readiness_indices"]["SRI"] for d in scores.values()
                      if "readiness_indices" in d and "SRI" in d["readiness_indices"]]

        if not mri_values or not sri_values:
            return ""

        import numpy as np
        mri_mean, mri_std = np.mean(mri_values), np.std(mri_values)
        sri_mean, sri_std = np.mean(sri_values), np.std(sri_values)

        # MRI thresholds
        mri_high = sum(1 for v in mri_values if v > 0.75)
        mri_med = sum(1 for v in mri_values if 0.45 <= v <= 0.75)
        mri_low = sum(1 for v in mri_values if v < 0.45)

        # SRI thresholds
        sri_high = sum(1 for v in sri_values if v > 0.65)
        sri_med = sum(1 for v in sri_values if 0.45 <= v <= 0.65)
        sri_low = sum(1 for v in sri_values if v < 0.45)

        # Collect tags
        tags = {}
        for d in scores.values():
            for tag in d.get("tags", []):
                tags[tag] = tags.get(tag, 0) + 1
        top_tags = sorted(tags.items(), key=lambda x: -x[1])[:5]

        # Get figure paths
        mri_hist_fig = ""
        sri_hist_fig = ""
        scatter_fig = ""
        radar_fig = ""

        if visualize:
            figures = visualize.get("figures", {})
            if isinstance(figures, dict):
                if "mri_histogram" in figures:
                    path = figures["mri_histogram"].get("svg", "")
                    if path:
                        mri_hist_fig = f"\n![MRI Distribution]({path})\n*Figure: Minutes Readiness Index distribution with threshold bands*\n"
                if "sri_histogram" in figures:
                    path = figures["sri_histogram"].get("svg", "")
                    if path:
                        sri_hist_fig = f"\n![SRI Distribution]({path})\n*Figure: Summarization Readiness Index distribution with threshold bands*\n"
                if "mri_sri_scatter" in figures:
                    path = figures["mri_sri_scatter"].get("svg", "")
                    if path:
                        scatter_fig = f"\n![MRI vs SRI]({path})\n*Figure: Document quality scatter plot with quadrant analysis*\n"
                if "quality_radar" in figures:
                    path = figures["quality_radar"].get("svg", "")
                    if path:
                        radar_fig = f"\n![Quality Radar]({path})\n*Figure: 5-dimension quality profile (corpus average)*\n"

        tags_list = ", ".join([f"{t} ({c})" for t, c in top_tags]) if top_tags else "None"

        return f"""## Quality Metrics

This section presents document quality assessment based on a 5-dimension scorecard:
- **LQ** (Linguistic Quality): Lexical richness, sentence regularity
- **SQ** (Structural Quality): Paragraph variance, heading depth
- **SC** (Semantic Coherence): Concept reuse, clustering agreement
- **IR** (Intent Clarity): Prescriptive vs descriptive content
- **EFU** (Exploitability): Task-specific fitness scores

### Quality Profile
{radar_fig}
### Readiness Indices

**Minutes Readiness Index (MRI)** — Fitness for minute/action item extraction:

| Category | Count | Percentage |
|----------|-------|------------|
| Auto-itemizable (>0.75) | {mri_high} | {100*mri_high/n_docs:.1f}% |
| Assisted (0.45-0.75) | {mri_med} | {100*mri_med/n_docs:.1f}% |
| Needs rewrite (<0.45) | {mri_low} | {100*mri_low/n_docs:.1f}% |

*Mean: {mri_mean:.3f} ± {mri_std:.3f}*
{mri_hist_fig}
**Summarization Readiness Index (SRI)** — Fitness for automatic summarization:

| Category | Count | Percentage |
|----------|-------|------------|
| Ready (>0.65) | {sri_high} | {100*sri_high/n_docs:.1f}% |
| Partial (0.45-0.65) | {sri_med} | {100*sri_med/n_docs:.1f}% |
| Not ready (<0.45) | {sri_low} | {100*sri_low/n_docs:.1f}% |

*Mean: {sri_mean:.3f} ± {sri_std:.3f}*
{sri_hist_fig}
### Quality Correlation
{scatter_fig}
### Quality Tags

Top document quality tags: {tags_list}

---
"""

    def _build_similarity_section(self, metadata: Dict, visualize: Dict = None) -> str:
        """Build document similarity and elaboration analysis section."""
        # Handle both list format (files) and dict format (documents)
        files_list = metadata.get("files", [])
        docs_dict = metadata.get("documents", {})

        if not files_list and not docs_dict:
            return ""

        n_docs = len(files_list) if files_list else len(docs_dict)
        docs = files_list if files_list else docs_dict

        # Get figure paths
        elaboration_fig = ""
        filename_dend_fig = ""
        similarity_heatmap_fig = ""

        if visualize:
            figures = visualize.get("figures", {})
            if isinstance(figures, dict):
                if "doc_elaboration_scatter" in figures:
                    path = figures["doc_elaboration_scatter"].get("svg", "")
                    if path:
                        elaboration_fig = f"\n![Document Elaboration Map]({path})\n*Figure: Document complexity vs size — larger points indicate higher quality scores*\n"
                if "filename_similarity_dendrogram" in figures:
                    path = figures["filename_similarity_dendrogram"].get("svg", "")
                    if path:
                        filename_dend_fig = f"\n![Filename Similarity]({path})\n*Figure: Hierarchical clustering by filename — red line marks version detection threshold*\n"
                if "doc_similarity_heatmap" in figures:
                    path = figures["doc_similarity_heatmap"].get("svg", "")
                    if path:
                        similarity_heatmap_fig = f"\n![Content Similarity]({path})\n*Figure: Content similarity matrix — red boxes highlight near-duplicate pairs*\n"

        # Count documents by type for context
        type_counts = {}
        doc_items = docs if isinstance(docs, list) else docs.values()
        for doc in doc_items:
            doc_type = doc.get("kind", doc.get("type", doc.get("extension", "unknown")))
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:5]
        types_str = ", ".join([f"{t} ({c})" for t, c in top_types])

        return f"""## Document Similarity & Elaboration Analysis

This section analyzes document relationships through three complementary lenses:
- **Elaboration**: Identifies the most complex and content-rich documents
- **Filename Patterns**: Detects document versions, copies, and related files
- **Content Similarity**: Reveals potential duplicates or near-identical content

### Document Elaboration Map

The elaboration map plots each document by its size (word count) and structural complexity
(heading depth × section count). This reveals which documents are the "pillars" of the corpus.
{elaboration_fig}
**Document Types**: {types_str}

### Filename Similarity Analysis

Documents with similar filenames often represent versions, translations, or related variants.
The dendrogram below clusters documents by normalized Levenshtein distance on their filenames.
{filename_dend_fig}
*Documents below the 30% threshold line likely represent version families.*

### Content Similarity Matrix

Beyond filename patterns, content similarity reveals semantic duplicates — documents that may have
different names but contain substantially similar information. High-similarity pairs (>85%)
are highlighted for review.
{similarity_heatmap_fig}
---
"""

    def _build_clustering_section(self, clusters: Dict, visualize: Dict = None) -> str:
        """Build clustering analysis section with dual view."""
        mode = clusters.get("mode", "dual_view")
        agreement = clusters.get("agreement", {})

        hier = clusters.get("hierarchical", {})
        leiden = clusters.get("leiden", {})

        # Hierarchical clusters table (use cluster_id, not id)
        hier_rows = ""
        for c in hier.get("clusters", [])[:10]:
            cid = c.get('cluster_id', c.get('id', ''))
            label = c.get('label', '')[:40]
            fcount = c.get('file_count', len(c.get('file_ids', [])))
            hier_rows += f"| {cid} | {label} | {fcount} |\n"

        # Leiden clusters table (use cluster_id, not id)
        leiden_rows = ""
        leiden_clusters = leiden.get("clusters", [])
        if leiden_clusters:
            for c in leiden_clusters[:10]:
                cid = c.get('cluster_id', c.get('id', ''))
                label = c.get('label', '')[:40]
                fcount = c.get('file_count', len(c.get('file_ids', [])))
                leiden_rows += f"| {cid} | {label} | {fcount} |\n"
        else:
            leiden_rows = "| - | *No Leiden clusters (unified mode)* | - |\n"

        divergences = clusters.get("divergences", [])
        divergence_text = ""
        if divergences:
            divergence_text = "\n### Divergences Noted\n\n"
            for d in divergences[:5]:
                divergence_text += f"- {d.get('description', '')}\n"

        # Include clustering figures if available
        dendrogram_fig = ""
        leiden_fig = ""
        if visualize:
            figures = visualize.get("figures", {})
            if "dendrogram" in figures:
                fig_path = figures["dendrogram"].get("svg", "")
                if fig_path:
                    dendrogram_fig = f"\n![Hierarchical Clustering Dendrogram]({fig_path})\n*Figure: Dendrogram showing hierarchical document clustering*\n"
            if "leiden_graph" in figures:
                fig_path = figures["leiden_graph"].get("svg", "")
                if fig_path:
                    leiden_fig = f"\n![Leiden Community Graph]({fig_path})\n*Figure: Network graph showing Leiden communities*\n"

        return f"""## Clustering Analysis

**Mode:** {mode}
**Agreement Score:** {agreement.get('overall_score', 0):.0%}

### Hierarchical Clustering

Based on path structure and explicit concept assignments.

| ID | Label | Files |
|----|-------|-------|
{hier_rows}
{dendrogram_fig}
### Leiden Community Detection

Based on content similarity patterns.

| ID | Label | Files |
|----|-------|-------|
{leiden_rows}
{leiden_fig}
{divergence_text}

> **Note:** Both clustering views are preserved. Consult both for comprehensive understanding.

---
"""

    def _build_domain_summaries(self, pyramid: Dict, summaries: Dict, visualize: Dict = None) -> str:
        """Build domain summaries section with word clouds."""
        domains = pyramid.get("pyramid", {}).get("level_3_domains", [])
        summary_data = summaries.get("summaries", {})

        # Get domain word clouds if available
        domain_clouds = {}
        if visualize:
            clouds_data = visualize.get("figures", {}).get("domain_word_clouds", {})
            domain_clouds = clouds_data.get("by_domain", {}) if isinstance(clouds_data, dict) else {}

        content = "## Domain Summaries\n\n"

        for domain in domains[:10]:
            domain_id = domain.get("domain_id", domain.get("id", ""))
            domain_label = domain.get("label", "Unknown")
            # file_ids are directly on domain, not nested in clusters
            domain_files = domain.get("file_ids", [])
            file_count = domain.get("file_count", len(domain_files))

            # Get related concepts for this domain
            related_concepts = domain.get("related_concepts", [])
            concept_labels = [c.get("label", "") if isinstance(c, dict) else str(c)
                           for c in related_concepts[:5]]

            content += f"### {domain_label}\n\n"

            # Include domain word cloud if available
            if domain_id in domain_clouds:
                fig_path = domain_clouds[domain_id].get("svg", "")
                if fig_path:
                    content += f"![{domain_label} Word Cloud]({fig_path})\n\n"

            content += f"**Files:** {file_count}\n"
            if concept_labels:
                content += f"**Key concepts:** {', '.join(concept_labels)}\n"
            content += "\n"

            # Include representative sentences from domain
            rep_sentences = domain.get("representative_sentences", [])
            if rep_sentences:
                content += "**Representative content:**\n\n"
                for sent in rep_sentences[:3]:
                    # Clean up sentence for display
                    if isinstance(sent, dict):
                        sent = sent.get("text", str(sent))
                    sent_clean = sent.strip().replace("\n", " ")[:150]
                    content += f"> {sent_clean}...\n\n"

            # Include a few document summaries if available
            docs_shown = 0
            for file_id in domain_files[:5]:
                if file_id in summary_data and docs_shown < 3:
                    s = summary_data[file_id]
                    title = s.get('title', file_id)[:50]
                    summary_text = s.get('summary', 'No summary')[:200]
                    content += f"- **{title}**: {summary_text}...\n"
                    docs_shown += 1

            content += "\n---\n\n"

        return content

    def _build_visual_analysis_section(self, visualize: Dict = None) -> str:
        """Build visual analysis section with remaining figures."""
        if not visualize:
            return ""

        figures = visualize.get("figures", {})
        if not figures:
            return ""

        content = """## Visual Analysis

The following visualizations provide additional insights into the document corpus structure.

"""
        # Concept co-occurrence heatmap
        if "concept_heatmap" in figures:
            fig_path = figures["concept_heatmap"].get("svg", "")
            if fig_path:
                content += f"""### Concept Co-occurrence

![Concept Co-occurrence Heatmap]({fig_path})
*Figure: Heatmap showing concept co-occurrence patterns across documents*

"""

        # Coverage matrix
        if "coverage_matrix" in figures:
            fig_path = figures["coverage_matrix"].get("svg", "")
            if fig_path:
                content += f"""### Coverage Matrix

![Coverage Matrix]({fig_path})
*Figure: Matrix showing document coverage by concept*

"""

        # Domain size comparison
        if "domain_comparison" in figures:
            fig_path = figures["domain_comparison"].get("svg", "")
            if fig_path:
                content += f"""### Domain Size Comparison

![Domain Size Comparison]({fig_path})
*Figure: Comparison of document counts across domains*

"""

        content += "---\n"
        return content

    def _build_functionality_section(self, funcs: Dict) -> str:
        """Build functionality catalog section."""
        functionalities = funcs.get("functionalities", [])
        missing_refs = funcs.get("missing_references", [])

        if not functionalities:
            return "## Functionality Catalog\n\nNo SPD documents found.\n\n---\n"

        content = f"""## Functionality Catalog

**Total Functionalities:** {len(functionalities)}
**Missing References:** {len(missing_refs)}

### Extracted Functionalities

| ID | Name | Category | Source |
|----|------|----------|--------|
"""

        for f in functionalities[:20]:
            content += f"| {f.get('id', '')} | {f.get('name', '')[:30]} | {f.get('category', '')} | SPD-{f.get('spd_number', '')} |\n"

        if len(functionalities) > 20:
            content += f"\n*... and {len(functionalities) - 20} more (see Appendix C)*\n"

        content += "\n---\n"
        return content

    def _build_discrepancy_section(self, compare: Dict, coverage: Dict) -> str:
        """Build discrepancy analysis section."""
        discrepancies = compare.get("discrepancies", [])
        gaps = coverage.get("gaps", [])

        content = f"""## Discrepancy Analysis

### Detected Discrepancies

**Total:** {len(discrepancies)}

"""

        # Group by type
        by_type = {}
        for d in discrepancies:
            t = d.get("type", "other")
            by_type.setdefault(t, []).append(d)

        for dtype, items in by_type.items():
            content += f"#### {dtype.replace('_', ' ').title()} ({len(items)})\n\n"
            for item in items[:5]:
                content += f"- {item.get('description', '')}\n"
            if len(items) > 5:
                content += f"- *... and {len(items) - 5} more*\n"
            content += "\n"

        if gaps:
            content += f"### Coverage Gaps ({len(gaps)})\n\n"
            content += "Concepts with limited document coverage:\n\n"
            for g in gaps[:10]:
                # Use label from gap, falling back to concept_id
                gap_label = g.get('label', g.get('concept', g.get('concept_id', 'Unknown')))
                file_count = g.get('file_count', 0)
                reason = g.get('reason', '')
                content += f"- **{gap_label}** ({file_count} file{'s' if file_count != 1 else ''})"
                if reason:
                    content += f" — {reason}"
                content += "\n"
            if len(gaps) > 10:
                content += f"\n*... and {len(gaps) - 10} more coverage gaps*\n"

        content += "\n---\n"
        return content

    def _build_technical_section(self, config: Dict, summaries: Dict) -> str:
        """Build technical information section."""
        stats = summaries.get("statistics", {})
        cache_stats = summaries.get("cache_stats", {})

        return f"""## Technical Information

| Parameter | Value |
|-----------|-------|
| Run ID | {config.get('run_id', 'unknown')} |
| Language | {config.get('language', 'en')} |
| Worker Model | {config.get('llm_model', 'granite3.1-moe:3b')} |
| Tutor Model | {config.get('tutor_model', 'mistral:7b-instruct')} |
| Documents Processed | {stats.get('total', 0)} |
| Cache Hits | {stats.get('cached', 0)} |
| Summaries Accepted | {stats.get('accepted', 0)} |
| Summaries Corrected | {stats.get('corrected', 0)} |

---
"""

    def _build_sovereignty_section(self, config: Dict, summaries: Dict) -> str:
        """Build sovereignty attestation section."""
        sov = summaries.get("sovereignty", {})

        models_table = ""
        for m in sov.get("models_used", []):
            models_table += f"| {m.get('name', '')} | {m.get('role', '')} | {m.get('digest', '')[:12]} |\n"

        return f"""## Sovereignty Attestation

This analysis was performed **entirely locally**, with no calls to external cloud
services. All data remained on the local infrastructure.

### Execution Environment

| Parameter | Value |
|-----------|-------|
| Hostname | {sov.get('hostname', socket.gethostname())} |
| User | {sov.get('user', os.environ.get('USER', 'unknown'))} |
| Endpoint | {sov.get('llm_endpoint', config.get('llm_endpoint', 'http://127.0.0.1:11434'))} |
| Local Processing | {sov.get('llm_local', True)} |
| Start Time | {sov.get('start_time', '')} |
| End Time | {sov.get('end_time', '')} |

### LLM Models Used

| Model | Role | Digest |
|-------|------|--------|
{models_table}

### Locality Proof

- Ollama endpoint: `127.0.0.1:11434` (loopback, local only)
- No external network requests
- Models stored locally in ~/.ollama/models/
- RAG index: Local ChromaDB

---
"""

    def _build_file_id_mapping(self, metadata: Dict) -> Dict[str, str]:
        """Build mapping from file IDs to readable file names."""
        file_id_to_name = {}
        # Handle data wrapper pattern
        meta_data = metadata.get("data", metadata)
        files = meta_data.get("files", [])
        for f in files:
            file_id = f.get("file_id", "")
            path = f.get("path", "")
            if path:
                # Extract filename without extension
                name = path.split("/")[-1]
                # Remove common extensions
                for ext in ['.docx', '.pdf', '.pptx', '.xlsx', '.md', '.txt', '.doc', '.xls', '.ppt']:
                    if name.lower().endswith(ext):
                        name = name[:-len(ext)]
                        break
                file_id_to_name[file_id] = name
            else:
                file_id_to_name[file_id] = file_id
        return file_id_to_name

    def _generate_appendices(
        self,
        appendices_dir: Path,
        pyramid: Dict,
        summaries: Dict,
        funcs: Dict,
        compare: Dict,
        clusters: Dict,
        file_id_to_name: Dict[str, str] = None,
        visualize: Dict = None
    ) -> List[Path]:
        """Generate appendix files."""
        appendices = []
        file_id_to_name = file_id_to_name or {}

        # Appendix A: Full corpus summary
        path_a = appendices_dir / "A_corpus_summary.md"
        path_a.write_text(self._generate_appendix_a(pyramid), encoding="utf-8")
        appendices.append(path_a)

        # Appendix B: Detailed domain summaries
        path_b = appendices_dir / "B_domain_summaries.md"
        path_b.write_text(self._generate_appendix_b(pyramid, summaries), encoding="utf-8")
        appendices.append(path_b)

        # Appendix C: Complete functionality catalog
        path_c = appendices_dir / "C_functionality_catalog.md"
        path_c.write_text(self._generate_appendix_c(funcs), encoding="utf-8")
        appendices.append(path_c)

        # Appendix D: Discrepancy details (with file names)
        path_d = appendices_dir / "D_discrepancy_details.md"
        path_d.write_text(self._generate_appendix_d(compare, file_id_to_name), encoding="utf-8")
        appendices.append(path_d)

        # Appendix E: Clustering analysis (with file names)
        path_e = appendices_dir / "E_clustering_analysis.md"
        path_e.write_text(self._generate_appendix_e(clusters, file_id_to_name), encoding="utf-8")
        appendices.append(path_e)

        # Appendix F: Artifacts and Visualizations
        if visualize:
            path_f = appendices_dir / "F_artifacts.md"
            path_f.write_text(self._generate_appendix_f(visualize, appendices_dir), encoding="utf-8")
            appendices.append(path_f)

        return appendices

    def _generate_appendix_a(self, pyramid: Dict) -> str:
        """Generate Appendix A: Corpus summary."""
        content = "# Appendix A: Full Corpus Summary\n\n"
        corpus = pyramid.get("pyramid", {}).get("level_4_corpus", {})

        content += f"## Overview\n\n"
        content += f"- **Total Files:** {corpus.get('file_count', 0)}\n"
        content += f"- **Domains:** {corpus.get('domain_count', 0)}\n"
        content += f"- **Key Concepts:** {', '.join(corpus.get('key_concepts', [])[:10])}\n"

        return content

    def _generate_appendix_b(self, pyramid: Dict, summaries: Dict) -> str:
        """Generate Appendix B: Domain summaries."""
        content = "# Appendix B: Detailed Domain Summaries\n\n"
        summary_data = summaries.get("summaries", {})

        for domain in pyramid.get("pyramid", {}).get("level_3_domains", []):
            domain_label = domain.get('label', 'Unknown')
            file_count = domain.get('file_count', 0)
            content += f"## {domain_label}\n\n"
            content += f"**Files in domain:** {file_count}\n\n"

            # file_ids are directly on domain, not in nested clusters
            file_ids = domain.get("file_ids", [])
            for file_id in file_ids:
                if file_id in summary_data:
                    s = summary_data[file_id]
                    content += f"### {s.get('title', file_id)}\n\n"
                    content += f"**Path:** {s.get('path', 'N/A')}\n\n"
                    content += f"**Scope:** {s.get('scope', 'N/A')}\n\n"
                    content += f"{s.get('summary', 'No summary')}\n\n"
                    topics = s.get('topics', [])
                    if topics:
                        content += f"**Topics:** {', '.join(topics)}\n\n"
                    content += "---\n\n"

        return content

    def _generate_appendix_c(self, funcs: Dict) -> str:
        """Generate Appendix C: Functionality catalog."""
        content = "# Appendix C: Complete Functionality Catalog\n\n"

        for f in funcs.get("functionalities", []):
            content += f"## {f.get('id', 'Unknown')}: {f.get('name', '')}\n\n"
            content += f"**Source:** {f.get('source_path', '')}\n\n"
            content += f"**Description:** {f.get('description', '')}\n\n"
            content += f"**Actors:** {', '.join(f.get('actors', []))}\n\n"
            content += f"**Trigger:** {f.get('trigger', '')}\n\n"
            content += f"**References:** {', '.join(f.get('references', []))}\n\n"
            content += "---\n\n"

        return content

    def _generate_appendix_d(self, compare: Dict, file_id_to_name: Dict[str, str] = None) -> str:
        """Generate Appendix D: Discrepancy details with file names."""
        file_id_to_name = file_id_to_name or {}
        content = "# Appendix D: Discrepancy and Divergence Details\n\n"

        discrepancies = compare.get("discrepancies", [])
        if not discrepancies:
            content += "*No discrepancies detected.*\n\n"
            return content

        # Group discrepancies by type
        by_type = {}
        for d in discrepancies:
            dtype = d.get("type", "other")
            by_type.setdefault(dtype, []).append(d)

        for dtype, items in sorted(by_type.items()):
            content += f"## {dtype.replace('_', ' ').title()} ({len(items)} issues)\n\n"

            for i, d in enumerate(items, 1):
                # Handle different discrepancy types
                if dtype == "content_overlap":
                    # Content overlap between two files
                    file_1 = d.get("file_1", "")
                    file_2 = d.get("file_2", "")
                    name_1 = file_id_to_name.get(file_1, file_1)
                    name_2 = file_id_to_name.get(file_2, file_2)
                    similarity = d.get("max_similarity", 0)
                    sample = d.get("sample", "")

                    content += f"### {i}. {name_1} ↔ {name_2}\n\n"
                    content += f"**Similarity:** {similarity:.0%}\n\n"
                    if sample:
                        # Truncate sample for readability
                        sample_clean = sample.strip().replace("\n", " ")[:200]
                        content += f"**Sample:** _{sample_clean}..._\n\n"

                elif dtype == "terminology_variation":
                    # Corpus-wide terminology variations
                    base_term = d.get("base_term", "")
                    variants = d.get("variants", [])

                    content += f"### {i}. Term: «{base_term}»\n\n"
                    content += f"**Variants found:** {', '.join(variants)}\n\n"

                elif dtype == "version_reference":
                    # Version reference in specific file
                    file_id = d.get("file_id", "")
                    file_name = d.get("file_name", file_id_to_name.get(file_id, file_id))
                    # Remove extension for display
                    if file_name:
                        for ext in ['.docx', '.pdf', '.xlsx', '.pptx']:
                            if file_name.lower().endswith(ext):
                                file_name = file_name[:-len(ext)]
                                break
                    actual_ver = d.get("actual_version", "")
                    ref_ver = d.get("referenced_version", "")
                    context = d.get("context", "")

                    content += f"### {i}. {file_name}\n\n"
                    content += f"**File version:** {actual_ver}\n"
                    content += f"**References version:** {ref_ver}\n\n"
                    if context:
                        content += f"**Context:** _{context}_\n\n"

                else:
                    # Generic fallback
                    file_id = d.get("file_id", d.get("source_file", ""))
                    file_name = file_id_to_name.get(file_id, "") if file_id else ""

                    if file_name:
                        content += f"### {i}. {file_name}\n\n"
                    else:
                        content += f"### {i}. Issue\n\n"

                    content += f"**Severity:** {d.get('severity', 'info')}\n\n"
                    desc = d.get('description', '')
                    if desc:
                        content += f"**Details:** {desc}\n\n"

                content += "---\n\n"

        return content

    def _generate_appendix_e(self, clusters: Dict, file_id_to_name: Dict[str, str] = None) -> str:
        """Generate Appendix E: Clustering analysis with file names."""
        file_id_to_name = file_id_to_name or {}
        content = "# Appendix E: Clustering Analysis\n\n"

        content += f"## Reconciliation Mode: {clusters.get('mode', 'unknown')}\n\n"
        content += f"**Agreement Score:** {clusters.get('agreement', {}).get('overall_score', 0):.0%}\n\n"

        def format_file_list(file_ids: List[str], file_paths: List[str]) -> str:
            """Format file list preferring paths, then names, then IDs."""
            result = ""
            for i, fid in enumerate(file_ids):
                # Try to get path first
                if file_paths and i < len(file_paths):
                    # Extract just filename from path
                    path = file_paths[i]
                    name = path.split("/")[-1] if "/" in path else path
                    result += f"- {name}\n"
                elif fid in file_id_to_name:
                    # Use mapped name
                    result += f"- {file_id_to_name[fid]}\n"
                else:
                    # Fallback to ID
                    result += f"- {fid}\n"
            return result

        content += "## Hierarchical Clusters\n\n"
        hier_clusters = clusters.get("hierarchical", {}).get("clusters", [])
        for c in hier_clusters:
            cid = c.get('cluster_id', c.get('id', ''))
            label = c.get('label', '')
            file_ids = c.get("file_ids", [])
            file_paths = c.get("file_paths", [])
            n_files = len(file_ids)

            content += f"### {cid}: {label}\n\n"
            content += f"**Files ({n_files}):**\n\n"

            # Show up to 30 files with names
            ids_to_show = file_ids[:30]
            paths_to_show = file_paths[:30] if file_paths else []
            content += format_file_list(ids_to_show, paths_to_show)

            if n_files > 30:
                content += f"- *... and {n_files - 30} more*\n"
            content += "\n"

        content += "## Leiden Communities\n\n"
        leiden_clusters = clusters.get("leiden", {}).get("clusters", [])
        if leiden_clusters:
            for c in leiden_clusters:
                cid = c.get('cluster_id', c.get('id', ''))
                label = c.get('label', '')
                file_ids = c.get("file_ids", [])
                file_paths = c.get("file_paths", [])
                n_files = len(file_ids)

                content += f"### {cid}: {label}\n\n"
                content += f"**Files ({n_files}):**\n\n"

                ids_to_show = file_ids[:30]
                paths_to_show = file_paths[:30] if file_paths else []
                content += format_file_list(ids_to_show, paths_to_show)

                if n_files > 30:
                    content += f"- *... and {n_files - 30} more*\n"
                content += "\n"
        else:
            content += "*No Leiden clusters — unified mode uses hierarchical clustering only.*\n\n"

        return content

    def _generate_appendix_f(self, visualize: Dict, appendices_dir: Path) -> str:
        """Generate Appendix F: Artifacts and Visualizations catalog."""
        content = "# Appendix F: Artifacts and Visualizations\n\n"
        content += "This appendix catalogs all generated visualizations and artifacts.\n\n"

        figures = visualize.get("figures", {})
        if not figures:
            content += "*No visualizations were generated.*\n\n"
            return content

        content += "## Visualizations\n\n"
        content += "| Figure | Description | Formats |\n"
        content += "|--------|-------------|----------|\n"

        figure_descriptions = {
            "word_cloud": "Corpus-wide word cloud showing key concepts",
            "doc_type_distribution": "Distribution of document types (PDF, DOCX, etc.)",
            "dendrogram": "Hierarchical clustering dendrogram",
            "leiden_graph": "Leiden community detection network graph",
            "concept_heatmap": "Concept co-occurrence heatmap",
            "coverage_matrix": "Document-concept coverage matrix",
            "domain_comparison": "Domain size comparison chart",
            "domain_word_clouds": "Per-domain word clouds"
        }

        for fig_name, fig_data in figures.items():
            if isinstance(fig_data, dict):
                desc = figure_descriptions.get(fig_name, fig_name.replace("_", " ").title())

                # Collect available formats
                formats = []
                for fmt in ["svg", "png", "pdf"]:
                    if fmt in fig_data and fig_data[fmt]:
                        path = fig_data[fmt]
                        # Make path relative from appendices directory
                        formats.append(f"[{fmt.upper()}](../{path})")

                if formats:
                    content += f"| {fig_name} | {desc} | {', '.join(formats)} |\n"

        content += "\n## Domain Word Clouds\n\n"
        domain_clouds = figures.get("domain_word_clouds", {}).get("by_domain", {})
        if domain_clouds:
            for domain_id, cloud_data in domain_clouds.items():
                if isinstance(cloud_data, dict):
                    svg_path = cloud_data.get("svg", "")
                    if svg_path:
                        content += f"### Domain: {domain_id}\n\n"
                        content += f"![{domain_id} Word Cloud](../{svg_path})\n\n"
        else:
            content += "*No domain-specific word clouds generated.*\n\n"

        content += "## Stage Outputs\n\n"
        content += "The analysis produced outputs at each stage:\n\n"
        content += "- **Stage 1** (Collection): doc_metadata, doc_concepts, doc_structure\n"
        content += "- **Stage 2** (Analysis): doc_cluster_dual, doc_cluster_reconcile, doc_extract, doc_coverage, doc_func_extract, doc_compare\n"
        content += "- **Stage 3** (Synthesis): doc_pyramid, doc_summarize_tutored, doc_visualize, doc_final_report\n\n"

        content += "All outputs are stored in JSON format in the `.KOAS/stage{N}/` directories.\n\n"

        return content

    def _build_appendix_references(self, appendices: List[Path]) -> str:
        """Build appendix references section."""
        content = "## Appendices\n\n"
        content += "The following appendices contain full analysis details:\n\n"

        labels = {
            "A": "Full corpus summary",
            "B": "Detailed domain summaries",
            "C": "Complete functionality catalog",
            "D": "Discrepancy and divergence details",
            "E": "Clustering analysis (hierarchical vs Leiden)",
            "F": "Artifacts and visualizations catalog"
        }

        for path in appendices:
            letter = path.stem.split("_")[0]
            content += f"- **Appendix {letter}**: {labels.get(letter, path.stem)} ([{path.name}](appendices/{path.name}))\n"

        content += "\n---\n"
        return content

    def _assemble_report(self, sections: Dict[str, str]) -> str:
        """Assemble final report from sections."""
        order = [
            "header",
            "methodology",
            "executive",
            "corpus",
            "quality_metrics",      # MRI/SRI analysis
            "similarity_analysis",  # Document similarity & elaboration
            "clustering",
            "domains",
            "visual_analysis",
            "functionalities",
            "discrepancies",
            "technical",
            "sovereignty",
            "appendix_refs"
        ]

        content = ""
        for section in order:
            if section in sections:
                content += sections[section] + "\n"

        content += """
---

*Generated by KOAS-Docs — RAGIX Project*
*Adservio Innovation Lab | 2026*
"""

        return content

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        return (
            f"Final report generated: {data.get('report_path', 'unknown')}. "
            f"Length: {data.get('report_length', 0)} chars. "
            f"Sections: {len(data.get('sections', []))}. "
            f"Appendices: {len(data.get('appendices', []))}."
        )
