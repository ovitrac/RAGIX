from __future__ import annotations

"""
Document Visualization Kernel — Generate figures for document analysis reports.

This kernel generates publication-quality visualizations from KOAS document analysis:
1. Clustering dendrogram (hierarchical clustering tree)
2. Leiden community graph (network visualization)
3. Document type distribution (pie/bar chart)
4. Concept co-occurrence heatmap
5. Coverage matrix (documents × concepts)
6. Domain size comparison (bar chart)
7. Word cloud (for abstract/executive summary)

Output formats: SVG (primary for reports), PNG and PDF (for other uses)
Location: assets/ folder alongside the final report

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# Find a font that supports Unicode/French characters
def _find_unicode_font() -> Optional[str]:
    """Find a font with good Unicode support for word clouds."""
    font_candidates = [
        # DejaVu fonts (good Unicode support)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        # Liberation fonts
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Noto fonts (excellent Unicode coverage)
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        # Ubuntu fonts
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        # macOS fonts
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Windows fonts
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return font_path
    return None

UNICODE_FONT = _find_unicode_font()

# Stopwords for French and English (used for content-based word clouds)
FRENCH_STOPWORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'à', 'au', 'aux',
    'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'que', 'qui', 'quoi',
    'dont', 'où', 'ce', 'cette', 'ces', 'cet', 'son', 'sa', 'ses', 'leur',
    'leurs', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'notre', 'nos', 'votre',
    'vos', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'je', 'tu',
    'se', 'lui', 'y', 'en', 'pour', 'par', 'avec', 'sans', 'sous', 'sur',
    'dans', 'entre', 'vers', 'chez', 'depuis', 'pendant', 'avant', 'après',
    'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir',
    'vouloir', 'devoir', 'falloir', 'prendre', 'mettre', 'donner', 'tenir',
    'est', 'sont', 'était', 'été', 'sera', 'serait', 'ont', 'avait', 'aura',
    'fait', 'dit', 'peut', 'doit', 'faut', 'peut-être', 'aussi', 'ainsi',
    'bien', 'très', 'plus', 'moins', 'tout', 'tous', 'toute', 'toutes',
    'autre', 'autres', 'même', 'mêmes', 'tel', 'tels', 'telle', 'telles',
    'quel', 'quelle', 'quels', 'quelles', 'chaque', 'plusieurs', 'quelque',
    'certains', 'certaines', 'aucun', 'aucune', 'nul', 'nulle', 'comme',
    'quand', 'si', 'alors', 'donc', 'encore', 'toujours', 'jamais', 'déjà',
    'soit', 'soient', 'afin', 'cependant', 'néanmoins', 'toutefois',
    'notamment', 'également', 'particulièrement', 'principalement',
    'concernant', 'relatif', 'relative', 'conformément', 'suivant',
}
ENGLISH_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'at', 'from', 'by', 'on', 'off', 'for', 'in', 'out', 'over', 'to',
    'into', 'with', 'about', 'against', 'between', 'through', 'during',
    'before', 'after', 'above', 'below', 'up', 'down', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'would', 'could', 'should', 'may', 'might',
    'must', 'shall', 'will', 'can', 'of', 'that', 'this', 'these', 'those',
    'which', 'who', 'whom', 'what', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'also', 'now', 'here', 'there', 'any', 'our', 'your', 'their',
    'its', 'his', 'her', 'my', 'it', 'we', 'they', 'he', 'she', 'i', 'you',
}
ALL_STOPWORDS = FRENCH_STOPWORDS | ENGLISH_STOPWORDS

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)

# Color palette for consistent styling
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#3B3B3B',      # Dark gray
    'background': '#FFFFFF',   # White
    'grid': '#E5E5E5',         # Light gray
}

# Extended color palette for multiple categories
PALETTE = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4E71',
    '#53A548', '#E84855', '#2D3047', '#419D78', '#E0A458',
    '#7768AE', '#1B998B', '#ED6A5A', '#5C5C5C', '#9BC53D'
]

# Document type colors (handle both uppercase and doc_ prefix variations)
DOC_TYPE_COLORS = {
    'DOCX': '#2B579A',   # Word blue
    'DOC_DOCX': '#2B579A',
    'PDF': '#E53935',    # PDF red
    'DOC_PDF': '#E53935',
    'PPTX': '#D24726',   # PowerPoint orange
    'DOC_PPTX': '#D24726',
    'XLSX': '#217346',   # Excel green
    'DOC_XLSX': '#217346',
    'MARKDOWN': '#7B1FA2',  # Purple
    'DOC_MARKDOWN': '#7B1FA2',
    'TXT': '#00838F',    # Teal
    'DOC_TXT': '#00838F',
    'OTHER': '#757575',  # Gray
}


class DocVisualizeKernel(Kernel):
    """
    Generate visualizations for document analysis reports.

    Produces 7 figure types in SVG, PNG, and PDF formats:
    1. dendrogram - Hierarchical clustering tree
    2. leiden_graph - Network visualization of Leiden communities
    3. doc_type_distribution - Pie/bar chart of document types
    4. concept_heatmap - Heatmap of concept relationships
    5. coverage_matrix - Documents × concepts coverage
    6. domain_comparison - Bar chart of domain sizes
    7. word_cloud - Concept cloud for abstract/executive summary
    """

    name = "doc_visualize"
    stage = 3
    version = "1.2.2"  # Fixed concepts format handling for content similarity heatmap
    requires = ["doc_metadata", "doc_concepts", "doc_cluster", "doc_cluster_leiden",
                "doc_coverage", "doc_pyramid", "doc_extract", "doc_quality"]  # Added doc_quality for MRI/SRI figures
    provides = ["visualizations", "figure_manifest"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Generate all visualizations."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available, skipping visualizations")
            return {"error": "matplotlib not installed", "figures": {}}

        # Setup output directory in workspace (run directory) for portability
        # This ensures figures are alongside the final report
        assets_dir = input.workspace / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Load dependencies
        metadata = self._load_dependency(input, "doc_metadata")
        concepts = self._load_dependency(input, "doc_concepts")
        clusters = self._load_dependency(input, "doc_cluster")
        leiden = self._load_dependency(input, "doc_cluster_leiden")
        coverage = self._load_dependency(input, "doc_coverage")
        pyramid = self._load_dependency(input, "doc_pyramid")
        extract = self._load_dependency(input, "doc_extract")  # Raw sentences
        quality = self._load_dependency(input, "doc_quality")  # Quality scores (optional)

        figures = {}  # Dict keyed by figure name for easy lookup

        # Set global matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'figure.dpi': 150,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })

        # 1. Clustering dendrogram
        try:
            fig_info = self._generate_dendrogram(clusters, metadata, assets_dir)
            if fig_info:
                figures["dendrogram"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate dendrogram: {e}")

        # 2. Leiden community graph
        try:
            fig_info = self._generate_leiden_graph(leiden, metadata, assets_dir)
            if fig_info:
                figures["leiden_graph"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate Leiden graph: {e}")

        # 3. Document type distribution
        try:
            fig_info = self._generate_doc_type_distribution(metadata, assets_dir)
            if fig_info:
                figures["doc_type_distribution"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate doc type distribution: {e}")

        # 4. Concept co-occurrence heatmap
        try:
            fig_info = self._generate_concept_heatmap(concepts, assets_dir)
            if fig_info:
                figures["concept_heatmap"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate concept heatmap: {e}")

        # 5. Coverage matrix
        try:
            fig_info = self._generate_coverage_matrix(coverage, concepts, metadata, assets_dir)
            if fig_info:
                figures["coverage_matrix"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate coverage matrix: {e}")

        # 6. Domain size comparison
        try:
            fig_info = self._generate_domain_comparison(pyramid, assets_dir)
            if fig_info:
                figures["domain_comparison"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate domain comparison: {e}")

        # 7. Word cloud (for executive summary) - using raw content from doc_extract
        try:
            fig_info = self._generate_word_cloud(concepts, pyramid, extract, assets_dir)
            if fig_info:
                figures["word_cloud"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate word cloud: {e}")

        # 8. Per-domain word clouds (for domain summaries) - using raw content
        try:
            domain_clouds = self._generate_domain_word_clouds(concepts, pyramid, extract, assets_dir)
            if domain_clouds:
                figures["domain_word_clouds"] = domain_clouds
        except Exception as e:
            logger.warning(f"Failed to generate domain word clouds: {e}")

        # 9-12. Quality visualizations (if doc_quality data available)
        if quality and quality.get("quality_scores"):
            # 9. MRI histogram
            try:
                fig_info = self._generate_mri_histogram(quality, assets_dir)
                if fig_info:
                    figures["mri_histogram"] = fig_info
            except Exception as e:
                logger.warning(f"Failed to generate MRI histogram: {e}")

            # 10. SRI histogram
            try:
                fig_info = self._generate_sri_histogram(quality, assets_dir)
                if fig_info:
                    figures["sri_histogram"] = fig_info
            except Exception as e:
                logger.warning(f"Failed to generate SRI histogram: {e}")

            # 11. MRI vs SRI scatter plot
            try:
                fig_info = self._generate_mri_sri_scatter(quality, assets_dir)
                if fig_info:
                    figures["mri_sri_scatter"] = fig_info
            except Exception as e:
                logger.warning(f"Failed to generate MRI/SRI scatter: {e}")

            # 12. Quality radar (5-dimension average)
            try:
                fig_info = self._generate_quality_radar(quality, assets_dir)
                if fig_info:
                    figures["quality_radar"] = fig_info
            except Exception as e:
                logger.warning(f"Failed to generate quality radar: {e}")

        # 13-15. Document similarity and elaboration visualizations
        # 13. Document elaboration scatter (complexity vs size)
        try:
            fig_info = self._generate_elaboration_scatter(metadata, quality, assets_dir)
            if fig_info:
                figures["doc_elaboration_scatter"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate elaboration scatter: {e}")

        # 14. Filename similarity dendrogram
        try:
            fig_info = self._generate_filename_dendrogram(metadata, assets_dir)
            if fig_info:
                figures["filename_similarity_dendrogram"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate filename dendrogram: {e}")

        # 15. Content similarity heatmap
        try:
            fig_info = self._generate_content_similarity_heatmap(concepts, metadata, assets_dir)
            if fig_info:
                figures["doc_similarity_heatmap"] = fig_info
        except Exception as e:
            logger.warning(f"Failed to generate content similarity heatmap: {e}")

        # Generate figure manifest (also as list for compatibility)
        figures_list = list(figures.values())
        manifest = {
            "assets_dir": str(assets_dir),
            "figures": figures_list,
            "formats": ["svg", "png", "pdf"],
            "count": len(figures),
        }

        # Save manifest
        manifest_path = assets_dir / "figure_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info(f"Generated {len(figures)} visualizations in {assets_dir}")

        return {
            "figures": figures,  # Dict for easy lookup by name
            "figures_list": figures_list,  # List for iteration
            "assets_dir": str(assets_dir),
            "manifest": manifest,
        }

    def _load_dependency(self, input: KernelInput, name: str) -> Dict[str, Any]:
        """Load dependency data from file."""
        path = input.dependencies.get(name)
        if path and path.exists():
            with open(path) as f:
                return json.load(f).get("data", {})
        return {}

    def _save_figure(self, fig: plt.Figure, name: str, assets_dir: Path,
                     title: str, description: str) -> Dict[str, Any]:
        """Save figure in SVG, PNG, and PDF formats.

        Returns relative paths (relative to workspace root) for markdown compatibility.
        """
        paths = {}
        rel_paths = {}

        # Compute relative path from workspace to assets
        # assets_dir is typically workspace/assets or workspace/assets/domain_clouds
        # We want paths like "assets/figure.svg" for the report
        workspace = assets_dir
        while workspace.name != "assets" and workspace.parent != workspace:
            workspace = workspace.parent
        workspace = workspace.parent  # Go up from "assets"

        for fmt in ['svg', 'png', 'pdf']:
            path = assets_dir / f"{name}.{fmt}"
            fig.savefig(path, format=fmt, facecolor='white', edgecolor='none')
            paths[fmt] = str(path)
            # Compute relative path from workspace root
            try:
                rel_paths[fmt] = str(path.relative_to(workspace))
            except ValueError:
                rel_paths[fmt] = str(path)

        plt.close(fig)

        return {
            "name": name,
            "title": title,
            "description": description,
            "paths": paths,  # Absolute paths for programmatic access
            "rel_paths": rel_paths,  # Relative paths for markdown
            "svg": rel_paths['svg'],  # Primary format for reports (now relative)
        }

    def _generate_dendrogram(self, clusters: Dict, metadata: Dict,
                             assets_dir: Path) -> Optional[Dict]:
        """Generate hierarchical clustering dendrogram."""
        if not HAS_SCIPY:
            logger.warning("scipy not available, skipping dendrogram")
            return None

        # Get cluster data
        cluster_data = clusters.get("data", clusters)
        cluster_list = cluster_data.get("clusters", [])

        if not cluster_list:
            logger.warning("No clusters found for dendrogram")
            return None

        # Build distance matrix from cluster assignments
        files = metadata.get("data", metadata).get("files", [])
        n_files = len(files)

        if n_files < 3:
            logger.warning("Not enough files for dendrogram")
            return None

        # Create labels (truncated filenames)
        labels = []
        for f in files[:50]:  # Limit to 50 for readability
            name = Path(f.get("path", f.get("file_id", "unknown"))).stem
            labels.append(name[:20] + "..." if len(name) > 20 else name)

        # Create synthetic distance matrix based on cluster membership
        n = len(labels)
        dist_matrix = np.zeros((n, n))

        # Assign files to clusters
        file_to_cluster = {}
        for i, cluster in enumerate(cluster_list):
            for fid in cluster.get("file_ids", []):
                file_to_cluster[fid] = i

        # Build distance matrix (same cluster = 0.2, different = 0.8)
        file_ids = [f.get("file_id") for f in files[:50]]
        for i in range(n):
            for j in range(i+1, n):
                ci = file_to_cluster.get(file_ids[i], -1)
                cj = file_to_cluster.get(file_ids[j], -1)
                dist = 0.2 if ci == cj and ci >= 0 else 0.8
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Generate dendrogram
        fig, ax = plt.subplots(figsize=(14, 8))

        # Compute linkage
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method='ward')

        # Plot dendrogram
        dendrogram(
            Z,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax,
            color_threshold=0.5,
        )

        ax.set_title("Document Clustering Dendrogram", fontsize=14, fontweight='bold')
        ax.set_xlabel("Documents", fontsize=11)
        ax.set_ylabel("Distance", fontsize=11)

        # Add cluster count annotation
        ax.annotate(
            f"{len(cluster_list)} clusters identified",
            xy=(0.02, 0.98), xycoords='axes fraction',
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        return self._save_figure(
            fig, "clustering_dendrogram", assets_dir,
            "Document Clustering Dendrogram",
            f"Hierarchical clustering of {n} documents into {len(cluster_list)} clusters"
        )

    def _generate_leiden_graph(self, leiden: Dict, metadata: Dict,
                               assets_dir: Path) -> Optional[Dict]:
        """Generate Leiden community network graph with document labels."""
        if not HAS_NETWORKX:
            logger.warning("networkx not available, skipping Leiden graph")
            return None

        # Leiden data uses 'optimal_clusters' not 'clusters'
        clusters = leiden.get("optimal_clusters", leiden.get("clusters", []))

        if not clusters:
            logger.warning("No Leiden clusters found")
            return None

        # Build file ID to name mapping from metadata (handle data wrapper)
        file_id_to_name = {}
        meta_data = metadata.get("data", metadata)
        meta_files = meta_data.get("files", [])
        for f in meta_files:
            fid = f.get("file_id")
            path = f.get("path", fid)
            # Extract just the filename from path (without extension for cleaner display)
            if path:
                name = path.split("/")[-1]
                # Remove common extensions
                for ext in ['.docx', '.pdf', '.pptx', '.xlsx', '.md', '.txt']:
                    if name.lower().endswith(ext):
                        name = name[:-len(ext)]
                        break
            else:
                name = fid
            # Truncate long names
            file_id_to_name[fid] = name[:25] + "…" if len(name) > 25 else name

        # Build graph
        G = nx.Graph()

        # Add nodes with cluster assignment and name
        file_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for fid in cluster.get("file_ids", []):
                file_to_cluster[fid] = i
                name = file_id_to_name.get(fid, fid)
                G.add_node(fid, cluster=i, name=name)

        # Add edges within clusters (for visualization)
        for cluster in clusters:
            fids = cluster.get("file_ids", [])
            for i, fid1 in enumerate(fids):
                for fid2 in fids[i+1:min(i+5, len(fids))]:  # Limit edges
                    G.add_edge(fid1, fid2)

        if len(G.nodes()) == 0:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))

        # Layout - use kamada_kawai for better spreading
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # Draw nodes by cluster
        for i, cluster in enumerate(clusters):
            fids = [fid for fid in cluster.get("file_ids", []) if fid in G.nodes()]
            if fids:
                color = PALETTE[i % len(PALETTE)]
                nx.draw_networkx_nodes(
                    G, pos, nodelist=fids,
                    node_color=color, node_size=150,
                    alpha=0.8, ax=ax
                )

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax)

        # Draw labels for ALL nodes (with small font to fit)
        # Create labels dict for all nodes
        all_labels = {fid: G.nodes[fid].get('name', fid) for fid in G.nodes()}

        # Draw all labels with adjusted positions to reduce overlap
        # Offset labels slightly above nodes
        label_pos = {node: (x, y + 0.02) for node, (x, y) in pos.items()}

        nx.draw_networkx_labels(
            G, label_pos, labels=all_labels,
            font_size=6, font_color='#222222', ax=ax,
            verticalalignment='bottom'
        )

        # Legend
        legend_patches = []
        for i, cluster in enumerate(clusters[:10]):  # Max 10 in legend
            color = PALETTE[i % len(PALETTE)]
            label = cluster.get("label", f"Cluster {i+1}")[:25]
            legend_patches.append(mpatches.Patch(color=color, label=label))

        ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

        ax.set_title("Leiden Community Detection", fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add stats annotation
        ax.annotate(
            f"{len(clusters)} communities\n{len(G.nodes())} documents",
            xy=(0.98, 0.02), xycoords='axes fraction',
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        return self._save_figure(
            fig, "leiden_community_graph", assets_dir,
            "Leiden Community Detection",
            f"Network visualization of {len(clusters)} communities with {len(G.nodes())} documents"
        )

    def _generate_doc_type_distribution(self, metadata: Dict,
                                        assets_dir: Path) -> Optional[Dict]:
        """Generate document type distribution chart."""
        meta_data = metadata.get("data", metadata)
        stats = meta_data.get("statistics", {})
        by_kind = stats.get("by_kind", {})

        if not by_kind:
            # Try to compute from files
            files = meta_data.get("files", [])
            by_kind = {}
            for f in files:
                kind = f.get("kind", "OTHER").upper()
                by_kind[kind] = by_kind.get(kind, 0) + 1

        if not by_kind:
            logger.warning("No document type data found")
            return None

        # Prepare data
        types = list(by_kind.keys())
        counts = list(by_kind.values())
        colors = [DOC_TYPE_COLORS.get(t, '#999999') for t in types]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        wedges, texts, autotexts = ax1.pie(
            counts, labels=types, autopct='%1.1f%%',
            colors=colors, startangle=90,
            explode=[0.02] * len(types)
        )
        ax1.set_title("Document Type Distribution", fontsize=12, fontweight='bold')

        # Bar chart
        bars = ax2.barh(types, counts, color=colors)
        ax2.set_xlabel("Number of Documents", fontsize=11)
        ax2.set_title("Document Count by Type", fontsize=12, fontweight='bold')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', fontsize=10)

        # Total annotation
        total = sum(counts)
        fig.suptitle(f"Total: {total} Documents", fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        return self._save_figure(
            fig, "document_type_distribution", assets_dir,
            "Document Type Distribution",
            f"Distribution of {total} documents across {len(types)} types"
        )

    def _generate_concept_heatmap(self, concepts: Dict,
                                  assets_dir: Path) -> Optional[Dict]:
        """Generate concept co-occurrence heatmap."""
        # Data keys: concepts, cooccurrence (not cooccurrence_matrix)
        cooccurrence = concepts.get("cooccurrence", {})
        concept_list = concepts.get("concepts", [])

        if not cooccurrence or not concept_list:
            logger.warning("No co-occurrence data found")
            return None

        # Get top concepts (limit for readability)
        top_concepts = concept_list[:25]
        concept_names = [c.get("label", c.get("name", c.get("concept_id", "?")))[:15] for c in top_concepts]
        concept_ids = [c.get("concept_id") for c in top_concepts]

        # Build matrix
        # cooccurrence structure: {concept_id: {other_concept_id: count, ...}, ...}
        n = len(concept_ids)
        matrix = np.zeros((n, n))

        for i, cid1 in enumerate(concept_ids):
            cid1_cooc = cooccurrence.get(cid1, {})
            for j, cid2 in enumerate(concept_ids):
                # Get count from cid1's cooccurrence dict
                val = cid1_cooc.get(cid2, 0)
                if val == 0:
                    # Try reverse lookup
                    cid2_cooc = cooccurrence.get(cid2, {})
                    val = cid2_cooc.get(cid1, 0)
                matrix[i, j] = val

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'custom', ['#FFFFFF', '#E3F2FD', '#90CAF9', '#2E86AB', '#1A5276']
        )

        # Use log scale for better visualization of varying magnitudes
        # Add small epsilon to avoid log(0)
        from matplotlib.colors import LogNorm, Normalize

        matrix_min = matrix[matrix > 0].min() if (matrix > 0).any() else 1
        matrix_max = matrix.max() if matrix.max() > 0 else 1

        if matrix_max > matrix_min * 10:  # Use log scale if range is large
            # Log normalization (add epsilon to handle zeros)
            matrix_display = matrix + 0.1  # Small offset for zeros
            norm = LogNorm(vmin=matrix_display.min(), vmax=matrix_display.max())
            im = ax.imshow(matrix_display, cmap=cmap, aspect='auto', norm=norm)
            scale_label = 'Co-occurrence (log scale)'
        else:
            # Linear normalization for small ranges
            im = ax.imshow(matrix, cmap=cmap, aspect='auto')
            scale_label = 'Co-occurrence Count'

        # Labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(concept_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(concept_names, fontsize=8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(scale_label, fontsize=10)

        ax.set_title("Concept Co-occurrence Heatmap", fontsize=14, fontweight='bold')

        plt.tight_layout()

        return self._save_figure(
            fig, "concept_cooccurrence_heatmap", assets_dir,
            "Concept Co-occurrence Heatmap",
            f"Co-occurrence relationships between top {n} concepts"
        )

    def _generate_coverage_matrix(self, coverage: Dict, concepts: Dict,
                                  metadata: Dict, assets_dir: Path) -> Optional[Dict]:
        """Generate coverage matrix (documents × concepts)."""
        # Coverage data uses coverage_matrix: file_id -> [list of concept_ids]
        coverage_matrix = coverage.get("coverage_matrix", {})
        concept_list = concepts.get("concepts", [])[:20]  # Top 20 concepts

        if not coverage_matrix or not concept_list:
            logger.warning("No coverage data found")
            return None

        # Build file ID to name mapping from metadata (handle data wrapper)
        file_id_to_name = {}
        meta_data = metadata.get("data", metadata)
        meta_files = meta_data.get("files", [])
        for f in meta_files:
            fid = f.get("file_id")
            path = f.get("path", fid)
            # Extract just the filename from path
            if path:
                name = path.split("/")[-1]
                # Remove common extensions
                for ext in ['.docx', '.pdf', '.pptx', '.xlsx', '.md', '.txt']:
                    if name.lower().endswith(ext):
                        name = name[:-len(ext)]
                        break
            else:
                name = fid
            # Truncate long names
            file_id_to_name[fid] = name[:20] + "…" if len(name) > 20 else name

        # Build file → concepts mapping from coverage_matrix
        file_concepts = {}
        for fid, concepts_list in coverage_matrix.items():
            if isinstance(concepts_list, list):
                file_concepts[fid] = set(concepts_list)
            elif isinstance(concepts_list, dict):
                file_concepts[fid] = set(concepts_list.keys())

        # Get unique files and concepts
        files = list(file_concepts.keys())[:30]  # Limit for readability
        concept_ids = [c.get("concept_id") for c in concept_list]
        # Use label as primary, name as fallback
        concept_names = [c.get("label", c.get("name", "?"))[:15] for c in concept_list]
        # Use actual file names instead of IDs
        file_names = [file_id_to_name.get(f, f) for f in files]

        # Build matrix
        matrix = np.zeros((len(files), len(concept_ids)))
        for i, fid in enumerate(files):
            for j, cid in enumerate(concept_ids):
                if cid in file_concepts.get(fid, set()):
                    matrix[i, j] = 1

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Heatmap
        cmap = LinearSegmentedColormap.from_list('coverage', ['#FFFFFF', '#2E86AB'])
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')

        # Labels
        ax.set_xticks(range(len(concept_ids)))
        ax.set_yticks(range(len(files)))
        ax.set_xticklabels(concept_names, rotation=45, ha='right', fontsize=8)

        # Use actual file names for y-axis labels
        ax.set_yticklabels(file_names, fontsize=7)

        ax.set_xlabel("Concepts", fontsize=11)
        ax.set_ylabel("Documents", fontsize=11)
        ax.set_title("Document-Concept Coverage Matrix", fontsize=14, fontweight='bold')

        # Coverage stats
        coverage_pct = (matrix.sum() / matrix.size) * 100
        ax.annotate(
            f"Coverage: {coverage_pct:.1f}%",
            xy=(0.98, 0.02), xycoords='axes fraction',
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        return self._save_figure(
            fig, "coverage_matrix", assets_dir,
            "Document-Concept Coverage Matrix",
            f"Coverage of {len(concept_ids)} concepts across {len(files)} documents ({coverage_pct:.1f}% coverage)"
        )

    def _generate_domain_comparison(self, pyramid: Dict,
                                    assets_dir: Path) -> Optional[Dict]:
        """Generate domain document count comparison chart."""
        # Navigate: data -> pyramid -> level_3_domains
        pyramid_data = pyramid.get("pyramid", pyramid)
        domains = pyramid_data.get("level_3_domains", [])

        if not domains:
            logger.warning("No domain data found")
            return None

        # Prepare data - documents only
        domain_names = []
        doc_counts = []

        for domain in domains:
            name = domain.get("label", domain.get("domain_id", "?"))
            # Truncate long names
            name = name[:35] + "…" if len(name) > 35 else name
            domain_names.append(name)
            doc_counts.append(len(domain.get("file_ids", [])))

        # Sort by document count (descending)
        sorted_idx = np.argsort(doc_counts)[::-1]
        domain_names = [domain_names[i] for i in sorted_idx]
        doc_counts = [doc_counts[i] for i in sorted_idx]

        # Create horizontal bar chart for better label readability
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(domain_names))

        # Horizontal bars with color gradient based on size
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(domain_names))]
        bars = ax.barh(y_pos, doc_counts, color=colors, alpha=0.85)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(domain_names, fontsize=10)
        ax.set_xlabel("Number of Documents", fontsize=11)
        ax.set_title("Documents per Domain", fontsize=14, fontweight='bold')

        # Add value labels at end of each bar
        for bar, count in zip(bars, doc_counts):
            width = bar.get_width()
            ax.annotate(f'{int(count)}',
                       xy=(width, bar.get_y() + bar.get_height()/2),
                       xytext=(5, 0), textcoords="offset points",
                       ha='left', va='center', fontsize=10, fontweight='bold')

        # Grid
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Invert y-axis so largest is at top
        ax.invert_yaxis()

        plt.tight_layout()

        return self._save_figure(
            fig, "domain_size_comparison", assets_dir,
            "Documents per Domain",
            f"Document distribution across {len(domains)} domains"
        )

    def _extract_word_frequencies(self, texts: List[str], min_word_length: int = 3,
                                    min_frequency: int = 2) -> Dict[str, int]:
        """Extract word frequencies from text, removing stopwords.

        Args:
            texts: List of text strings to process
            min_word_length: Minimum word length to include
            min_frequency: Minimum frequency to include a word

        Returns:
            Dictionary of word -> frequency
        """
        import re
        import unicodedata

        word_counts = {}
        # Pattern to match words including French accented characters
        word_pattern = re.compile(r"[a-zA-ZÀ-ÿ]+(?:'[a-zA-ZÀ-ÿ]+)?")

        # Common encoding error patterns to fix (UTF-8 mojibake from Windows-1252)
        # These patterns occur when UTF-8 text is decoded as Latin-1/Windows-1252
        encoding_fixes = {
            # Simple apostrophe-based errors
            "e'": "é", "e`": "è", "a`": "à", "a^": "â",
            "o^": "ô", "u`": "ù", "i^": "î", "e^": "ê",
            "c,": "ç",
            # Reversed apostrophe
            "'e": "é", "`e": "è", "`a": "à",
            # Common mojibake patterns (UTF-8 decoded as Latin-1)
            "\xc3\xa9": "é",  # é
            "\xc3\xa8": "è",  # è
            "\xc3\xa0": "à",  # à
            "\xc3\xb4": "ô",  # ô
            "\xc3\xa2": "â",  # â
            "\xc3\xae": "î",  # î
            "\xc3\xbb": "û",  # û
            "\xc3\xa7": "ç",  # ç
            "\xc3\x89": "É",  # É
            "\xc3\x80": "À",  # À
            "\xc3\xaa": "ê",  # ê
            "\xc3\xb9": "ù",  # ù
        }

        for text in texts:
            if not text:
                continue

            # Fix common encoding errors
            for wrong, right in encoding_fixes.items():
                text = text.replace(wrong, right)

            # Normalize Unicode (NFC form)
            text = unicodedata.normalize('NFC', text)
            # Tokenize
            words = word_pattern.findall(text.lower())
            for word in words:
                # Skip short words, stopwords, and numbers
                if len(word) < min_word_length:
                    continue
                if word in ALL_STOPWORDS:
                    continue
                word_counts[word] = word_counts.get(word, 0) + 1

        # Filter by minimum frequency
        return {w: c for w, c in word_counts.items() if c >= min_frequency}

    def _generate_word_cloud(self, concepts: Dict, pyramid: Dict, extract: Dict,
                             assets_dir: Path) -> Optional[Dict]:
        """Generate word cloud from raw content (for abstract/executive summary).

        Uses actual document content from doc_extract, not just concept names.
        """
        if not HAS_WORDCLOUD:
            logger.warning("wordcloud not available, skipping word cloud")
            return None

        # Try to get raw content from doc_extract first (preferred)
        texts = []

        # Get sentences from extract kernel (by_concept -> sentences -> text)
        by_concept = extract.get("by_concept", {})
        for concept_id, concept_data in by_concept.items():
            sentences = concept_data.get("sentences", [])
            for sent in sentences:
                if isinstance(sent, dict):
                    texts.append(sent.get("text", ""))
                elif isinstance(sent, str):
                    texts.append(sent)

        # Also try by_file sentences
        by_file = extract.get("by_file", {})
        for file_id, file_data in by_file.items():
            sentences = file_data.get("sentences", [])
            for sent in sentences:
                if isinstance(sent, dict):
                    texts.append(sent.get("text", ""))
                elif isinstance(sent, str):
                    texts.append(sent)

        # Extract word frequencies from content
        if texts:
            word_freq = self._extract_word_frequencies(texts, min_word_length=3, min_frequency=2)
            source = "content"
        else:
            # Fallback to concepts if no extract data
            logger.info("No extract data, falling back to concepts for word cloud")
            concept_data = concepts.get("data", concepts)
            concept_list = concept_data.get("concepts", [])

            if not concept_list:
                logger.warning("No concepts found for word cloud")
                return None

            word_freq = {}
            for c in concept_list:
                name = c.get("name", c.get("label", c.get("concept_id", "")))
                name = name.replace("_", " ").strip()
                if len(name) < 2:
                    continue
                weight = c.get("file_count", c.get("frequency", c.get("weight", 1)))
                word_freq[name] = weight
            source = "concepts"

        if not word_freq:
            return None

        logger.info(f"Word cloud using {len(word_freq)} words from {source}")

        # Create word cloud with Unicode font support
        wc_params = {
            'width': 1200,
            'height': 600,
            'background_color': 'white',
            'colormap': 'viridis',
            'max_words': 100,
            'min_font_size': 10,
            'max_font_size': 80,
            'prefer_horizontal': 0.7,
            'random_state': 42,
            'regexp': r"[\w'À-ÿ]+",  # Include French accented characters
            'normalize_plurals': False,  # Preserve exact words
        }
        if UNICODE_FONT:
            wc_params['font_path'] = UNICODE_FONT
            logger.debug(f"Using font: {UNICODE_FONT}")
        wc = WordCloud(**wc_params)
        wc.generate_from_frequencies(word_freq)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Corpus Concept Word Cloud", fontsize=14, fontweight='bold', pad=20)

        # Add stats annotation
        ax.annotate(
            f"{len(word_freq)} concepts",
            xy=(0.98, 0.02), xycoords='axes fraction',
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        return self._save_figure(
            fig, "word_cloud", assets_dir,
            "Corpus Concept Word Cloud",
            f"Word cloud visualization of {len(word_freq)} key concepts (for abstract)"
        )

    def _generate_domain_word_clouds(self, concepts: Dict, pyramid: Dict, extract: Dict,
                                     assets_dir: Path) -> Optional[Dict]:
        """Generate word clouds for each domain (for domain summaries).

        Uses raw content from files in each domain when available.
        """
        if not HAS_WORDCLOUD:
            logger.warning("wordcloud not available, skipping domain word clouds")
            return None

        # Navigate: data -> pyramid -> level_3_domains
        pyramid_data = pyramid.get("pyramid", pyramid)
        domains = pyramid_data.get("level_3_domains", [])

        if not domains:
            logger.warning("No domains found for domain word clouds")
            return None

        # Get sentences by file from extract (for content-based clouds)
        by_file = extract.get("by_file", {})

        concept_data = concepts.get("data", concepts)
        concept_list = concept_data.get("concepts", [])

        # Build concept_id to name/weight mapping
        concept_info = {}
        for c in concept_list:
            cid = c.get("concept_id")
            if cid:
                concept_info[cid] = {
                    "name": c.get("name", c.get("label", cid)),
                    "weight": c.get("file_count", c.get("frequency", 1))
                }

        domain_figures = {}
        domains_dir = assets_dir / "domain_clouds"
        domains_dir.mkdir(parents=True, exist_ok=True)

        # Build file_id -> key_sentences from level_1_documents (pyramid data)
        level1_docs = pyramid_data.get("level_1_documents", [])
        file_key_sentences = {}
        for doc in level1_docs:
            fid = doc.get("file_id")
            if fid:
                sentences = doc.get("key_sentences", [])
                file_key_sentences[fid] = sentences

        for domain in domains[:10]:  # Limit to 10 domains
            domain_id = domain.get("domain_id", domain.get("id", "unknown"))
            domain_label = domain.get("label", domain_id)
            domain_files = domain.get("file_ids", [])

            # Collect text from multiple sources for comprehensive coverage
            texts = []

            # Source 1: Domain representative sentences (from pyramid)
            rep_sentences = domain.get("representative_sentences", [])
            for sent in rep_sentences:
                if isinstance(sent, dict):
                    texts.append(sent.get("text", ""))
                elif isinstance(sent, str):
                    texts.append(sent)

            # Source 2: Key sentences from level_1_documents (pyramid)
            for file_id in domain_files:
                if file_id in file_key_sentences:
                    for sent in file_key_sentences[file_id]:
                        if isinstance(sent, dict):
                            texts.append(sent.get("text", ""))
                        elif isinstance(sent, str):
                            texts.append(sent)

            # Source 3: Sentences from doc_extract by_file
            for file_id in domain_files:
                if file_id in by_file:
                    sentences = by_file[file_id].get("sentences", [])
                    for sent in sentences:
                        if isinstance(sent, dict):
                            texts.append(sent.get("text", ""))
                        elif isinstance(sent, str):
                            texts.append(sent)

            if texts:
                # Use content-based word frequencies
                word_freq = self._extract_word_frequencies(texts, min_word_length=3, min_frequency=1)
                logger.debug(f"Domain {domain_id}: {len(texts)} texts -> {len(word_freq)} words")
            else:
                # Fallback to concepts
                domain_concepts = domain.get("concepts", domain.get("related_concepts", []))
                if not domain_concepts:
                    continue

                word_freq = {}
                for concept in domain_concepts:
                    if isinstance(concept, dict):
                        name = concept.get("name", concept.get("label", ""))
                        weight = concept.get("weight", concept.get("score", 1))
                    else:
                        info = concept_info.get(concept, {})
                        name = info.get("name", str(concept))
                        weight = info.get("weight", 1)

                    name = name.replace("_", " ").strip()
                    if len(name) >= 2:
                        word_freq[name] = weight
                logger.debug(f"Domain {domain_id}: using {len(word_freq)} concepts (fallback)")

            if len(word_freq) < 3:
                continue

            # Generate word cloud with Unicode font support
            try:
                wc_params = {
                    'width': 800,
                    'height': 400,
                    'background_color': 'white',
                    'colormap': 'plasma',
                    'max_words': 50,
                    'min_font_size': 8,
                    'max_font_size': 60,
                    'prefer_horizontal': 0.8,
                    'random_state': 42,
                    'regexp': r"[\w'À-ÿ]+",  # Include French accented characters
                    'normalize_plurals': False,
                }
                if UNICODE_FONT:
                    wc_params['font_path'] = UNICODE_FONT
                wc = WordCloud(**wc_params)
                wc.generate_from_frequencies(word_freq)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')

                # Truncate title if too long
                title = domain_label[:40] + "..." if len(domain_label) > 40 else domain_label
                ax.set_title(f"{title}", fontsize=12, fontweight='bold', pad=10)

                plt.tight_layout()

                # Save with domain-specific name
                safe_name = "".join(c if c.isalnum() else "_" for c in domain_id)
                fig_info = self._save_figure(
                    fig, f"domain_cloud_{safe_name}", domains_dir,
                    f"{domain_label} Word Cloud",
                    f"Key concepts for domain {domain_label}"
                )
                domain_figures[domain_id] = fig_info

            except Exception as e:
                logger.warning(f"Failed to generate word cloud for domain {domain_id}: {e}")

        if not domain_figures:
            return None

        return {
            "domains_dir": str(domains_dir),
            "count": len(domain_figures),
            "by_domain": domain_figures,
        }

    # =========================================================================
    # Quality Visualizations (from doc_quality kernel)
    # =========================================================================

    def _generate_mri_histogram(self, quality: Dict, assets_dir: Path) -> Optional[Dict]:
        """Generate MRI (Minutes Readiness Index) histogram with threshold bands."""
        scores = quality.get("quality_scores", {})
        if not scores:
            return None

        mri_values = [doc["readiness_indices"]["MRI"] for doc in scores.values()
                      if "readiness_indices" in doc and "MRI" in doc["readiness_indices"]]

        if not mri_values:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        n, bins, patches = ax.hist(mri_values, bins=20, range=(0, 1),
                                   color='steelblue', edgecolor='white', alpha=0.8)

        # Color by threshold bands
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge >= 0.75:
                patch.set_facecolor('#2ecc71')  # Green - auto-itemizable
            elif left_edge >= 0.45:
                patch.set_facecolor('#f39c12')  # Orange - assisted
            else:
                patch.set_facecolor('#e74c3c')  # Red - needs rewrite

        # Threshold lines
        ax.axvline(x=0.45, color='#f39c12', linestyle='--', linewidth=2, label='Assisted (0.45)')
        ax.axvline(x=0.75, color='#2ecc71', linestyle='--', linewidth=2, label='Auto (0.75)')

        # Labels
        ax.set_xlabel('MRI Score', fontsize=12)
        ax.set_ylabel('Number of Documents', fontsize=12)
        ax.set_title('Minutes Readiness Index (MRI) Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        # Stats annotation
        stats_text = f"n={len(mri_values)}, μ={np.mean(mri_values):.2f}, σ={np.std(mri_values):.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        return self._save_figure(fig, "mri_histogram", assets_dir,
                                 "MRI Distribution",
                                 "Minutes Readiness Index histogram with threshold bands")

    def _generate_sri_histogram(self, quality: Dict, assets_dir: Path) -> Optional[Dict]:
        """Generate SRI (Summarization Readiness Index) histogram with threshold bands."""
        scores = quality.get("quality_scores", {})
        if not scores:
            return None

        sri_values = [doc["readiness_indices"]["SRI"] for doc in scores.values()
                      if "readiness_indices" in doc and "SRI" in doc["readiness_indices"]]

        if not sri_values:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        n, bins, patches = ax.hist(sri_values, bins=20, range=(0, 1),
                                   color='steelblue', edgecolor='white', alpha=0.8)

        # Color by threshold bands
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge >= 0.65:
                patch.set_facecolor('#2ecc71')  # Green - ready
            elif left_edge >= 0.45:
                patch.set_facecolor('#f39c12')  # Orange - partial
            else:
                patch.set_facecolor('#e74c3c')  # Red - not ready

        # Threshold lines
        ax.axvline(x=0.45, color='#f39c12', linestyle='--', linewidth=2, label='Partial (0.45)')
        ax.axvline(x=0.65, color='#2ecc71', linestyle='--', linewidth=2, label='Ready (0.65)')

        # Labels
        ax.set_xlabel('SRI Score', fontsize=12)
        ax.set_ylabel('Number of Documents', fontsize=12)
        ax.set_title('Summarization Readiness Index (SRI) Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')

        # Stats annotation (bottom-left to avoid legend overlap)
        stats_text = f"n={len(sri_values)}, μ={np.mean(sri_values):.2f}, σ={np.std(sri_values):.2f}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        return self._save_figure(fig, "sri_histogram", assets_dir,
                                 "SRI Distribution",
                                 "Summarization Readiness Index histogram with threshold bands")

    def _generate_mri_sri_scatter(self, quality: Dict, assets_dir: Path) -> Optional[Dict]:
        """Generate MRI vs SRI scatter plot with quadrant analysis."""
        scores = quality.get("quality_scores", {})
        if not scores:
            return None

        mri_values = []
        sri_values = []
        labels = []

        for doc_id, doc in scores.items():
            if "readiness_indices" in doc:
                ri = doc["readiness_indices"]
                if "MRI" in ri and "SRI" in ri:
                    mri_values.append(ri["MRI"])
                    sri_values.append(ri["SRI"])
                    labels.append(doc.get("path", doc_id)[-30:])  # Truncate long paths

        if len(mri_values) < 2:
            return None

        fig, ax = plt.subplots(figsize=(10, 10))

        # Scatter plot with color by combined quality
        combined = [(m + s) / 2 for m, s in zip(mri_values, sri_values)]
        scatter = ax.scatter(mri_values, sri_values, c=combined, cmap='RdYlGn',
                             s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

        # Quadrant lines
        ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.45, color='gray', linestyle='--', alpha=0.5)

        # Quadrant labels
        ax.text(0.22, 0.85, 'Low MRI\nHigh SRI', ha='center', fontsize=9, alpha=0.7)
        ax.text(0.75, 0.85, 'High MRI\nHigh SRI', ha='center', fontsize=9, alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
        ax.text(0.22, 0.25, 'Low MRI\nLow SRI', ha='center', fontsize=9, alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
        ax.text(0.75, 0.25, 'High MRI\nLow SRI', ha='center', fontsize=9, alpha=0.7)

        # Labels
        ax.set_xlabel('MRI (Minutes Readiness)', fontsize=12)
        ax.set_ylabel('SRI (Summarization Readiness)', fontsize=12)
        ax.set_title('Document Quality: MRI vs SRI', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Combined Quality', fontsize=10)

        # Correlation annotation
        corr = np.corrcoef(mri_values, sri_values)[0, 1]
        ax.text(0.98, 0.02, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10,
                ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        return self._save_figure(fig, "mri_sri_scatter", assets_dir,
                                 "MRI vs SRI Scatter",
                                 f"Document quality scatter plot (n={len(mri_values)}, r={corr:.2f})")

    def _generate_quality_radar(self, quality: Dict, assets_dir: Path) -> Optional[Dict]:
        """Generate 5-dimension quality radar chart (corpus average)."""
        scores = quality.get("quality_scores", {})
        if not scores:
            return None

        # Collect dimension scores
        dimensions = ["LQ", "SQ", "SC", "IR", "EFU"]
        dim_labels = [
            "Linguistic\nQuality",
            "Structural\nQuality",
            "Semantic\nCoherence",
            "Intent\nClarity",
            "Exploitability"
        ]

        # Calculate averages
        dim_values = {d: [] for d in dimensions}
        for doc in scores.values():
            doc_scores = doc.get("scores", {})
            for d in dimensions:
                if d == "EFU":
                    # EFU is nested, take average of sub-scores
                    efu = doc_scores.get("EFU", {})
                    if efu:
                        dim_values[d].append(np.mean(list(efu.values())))
                elif d == "IR":
                    # Use 1 - entropy for clarity (higher = clearer)
                    entropy = doc_scores.get("IR_entropy", 0.5)
                    dim_values[d].append(1 - entropy)
                elif d in doc_scores:
                    dim_values[d].append(doc_scores[d])

        # Calculate means
        means = [np.mean(dim_values[d]) if dim_values[d] else 0 for d in dimensions]

        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        means += means[:1]  # Close the polygon
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Plot
        ax.fill(angles, means, color='steelblue', alpha=0.25)
        ax.plot(angles, means, color='steelblue', linewidth=2, marker='o', markersize=8)

        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)

        # Title
        ax.set_title('Document Quality Profile\n(Corpus Average)', fontsize=14, fontweight='bold', y=1.08)

        # Add value annotations
        for angle, mean, label in zip(angles[:-1], means[:-1], dimensions):
            ax.annotate(f'{mean:.2f}', xy=(angle, mean), xytext=(angle, mean + 0.1),
                        ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()

        return self._save_figure(fig, "quality_radar", assets_dir,
                                 "Quality Radar",
                                 "5-dimension quality profile (corpus average)")

    # =========================================================================
    # Document Similarity & Elaboration Visualizations
    # =========================================================================

    def _generate_elaboration_scatter(self, metadata: Dict, quality: Dict,
                                       assets_dir: Path) -> Optional[Dict]:
        """Generate document elaboration scatter plot (complexity vs size).

        X-axis: Word count (document size)
        Y-axis: Structural complexity (heading depth × section count)
        Color: Document type
        Size: Quality score (MRI)
        Labels: Top 10 most elaborate documents
        """
        # Handle both list format (files) and dict format (documents)
        files_list = metadata.get("files", [])
        docs_dict = metadata.get("documents", {})

        if not files_list and not docs_dict:
            return None

        # Extract data for each document
        data_points = []

        # Process list format (preferred)
        if files_list:
            for doc in files_list:
                doc_id = doc.get("file_id", doc.get("path", ""))
                # Estimate word count from size_bytes (roughly 5 chars per word)
                size_bytes = doc.get("size_bytes", 0) or 0
                word_count = max(size_bytes // 5, 100) if size_bytes else 100

                # Use chunk_count as proxy for structure complexity
                chunk_count = doc.get("chunk_count", 1) or 1
                structure_score = chunk_count

                # Get quality score if available
                mri = 0.5  # Default
                if quality and quality.get("quality_scores"):
                    q_data = quality["quality_scores"].get(doc_id, {})
                    if "readiness_indices" in q_data:
                        mri = q_data["readiness_indices"].get("MRI", 0.5)

                # Get document type from kind or extension
                doc_type = doc.get("kind", doc.get("extension", "unknown"))
                filename = doc.get("path", doc_id).split("/")[-1]

                data_points.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "word_count": word_count,
                    "structure": structure_score,
                    "mri": mri,
                    "doc_type": doc_type,
                    "elaboration": word_count * structure_score
                })
        else:
            # Process dict format (fallback)
            for doc_id, doc in docs_dict.items():
                word_count = doc.get("word_count", 0) or doc.get("size_words", 0)
                if word_count == 0:
                    content = doc.get("content", "")
                    word_count = len(content.split()) if content else 100

                heading_depth = doc.get("heading_depth", 1) or 1
                section_count = doc.get("section_count", 1) or doc.get("num_sections", 1) or 1
                structure_score = heading_depth * section_count

                mri = 0.5
                if quality and quality.get("quality_scores"):
                    q_data = quality["quality_scores"].get(doc_id, {})
                    if "readiness_indices" in q_data:
                        mri = q_data["readiness_indices"].get("MRI", 0.5)

                doc_type = doc.get("type", "unknown") or doc.get("doc_type", "unknown")
                filename = doc.get("filename", doc.get("name", doc_id))

                data_points.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "word_count": word_count,
                    "structure": structure_score,
                    "mri": mri,
                    "doc_type": doc_type,
                    "elaboration": word_count * structure_score
                })

        if len(data_points) < 3:
            return None

        # Sort by elaboration for top 10 labels
        data_points.sort(key=lambda x: x["elaboration"], reverse=True)
        top_10_ids = {d["doc_id"] for d in data_points[:10]}

        # Prepare plot data
        import numpy as np

        word_counts = np.array([d["word_count"] for d in data_points])
        structures = np.array([d["structure"] for d in data_points])
        mris = np.array([d["mri"] for d in data_points])

        # Get unique types for coloring
        types = [d["doc_type"] for d in data_points]
        unique_types = list(set(types))
        type_colors = {t: plt.cm.Set2(i / max(len(unique_types), 1))
                       for i, t in enumerate(unique_types)}
        colors = [type_colors[t] for t in types]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Scale point sizes by MRI (50-300 range)
        sizes = 50 + 250 * mris

        # Scatter plot
        scatter = ax.scatter(word_counts, structures, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Add labels for top 10
        for i, d in enumerate(data_points):
            if d["doc_id"] in top_10_ids:
                # Truncate filename for display
                label = d["filename"][:25] + "..." if len(d["filename"]) > 28 else d["filename"]
                ax.annotate(label, (word_counts[i], structures[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3))

        # Labels and title
        ax.set_xlabel('Word Count (Document Size)', fontsize=12)
        ax.set_ylabel('Structural Complexity (Heading Depth × Sections)', fontsize=12)
        ax.set_title('Document Elaboration Map\n(Size indicates quality score)', fontsize=14, fontweight='bold')

        # Legend for document types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=type_colors[t], markersize=10, label=t)
                          for t in unique_types[:8]]  # Limit to 8 types
        ax.legend(handles=legend_elements, loc='upper right', title='Document Type', fontsize=8)

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Log scale for word count if range is large
        if word_counts.max() / max(word_counts.min(), 1) > 100:
            ax.set_xscale('log')

        plt.tight_layout()

        return self._save_figure(fig, "doc_elaboration_scatter", assets_dir,
                                 "Document Elaboration Map",
                                 "Scatter plot showing document complexity vs size with top elaborated docs labeled")

    def _generate_filename_dendrogram(self, metadata: Dict, assets_dir: Path) -> Optional[Dict]:
        """Generate filename similarity dendrogram using Levenshtein distance.

        Detects document versions, copies, and related files by name patterns.
        """
        # Handle both list format (files) and dict format (documents)
        files_list = metadata.get("files", [])
        docs_dict = metadata.get("documents", {})

        if not files_list and not docs_dict:
            return None

        # Get filenames
        filenames = []
        doc_ids = []

        if files_list:
            for doc in files_list:
                doc_id = doc.get("file_id", doc.get("path", ""))
                fname = doc.get("path", doc_id)
                # Normalize: remove extension and path
                fname_clean = fname.rsplit('.', 1)[0] if '.' in fname else fname
                fname_clean = fname_clean.rsplit('/', 1)[-1] if '/' in fname_clean else fname_clean
                filenames.append(fname_clean)
                doc_ids.append(doc_id)
        else:
            for doc_id, doc in docs_dict.items():
                fname = doc.get("filename", doc.get("name", doc_id))
                fname_clean = fname.rsplit('.', 1)[0] if '.' in fname else fname
                fname_clean = fname_clean.rsplit('/', 1)[-1] if '/' in fname_clean else fname_clean
                filenames.append(fname_clean)
                doc_ids.append(doc_id)

        if len(filenames) < 3:
            return None

        # Compute Levenshtein distance matrix
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Compute Levenshtein edit distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)

            prev_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                curr_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev_row[j + 1] + 1
                    deletions = curr_row[j] + 1
                    substitutions = prev_row[j] + (c1 != c2)
                    curr_row.append(min(insertions, deletions, substitutions))
                prev_row = curr_row
            return prev_row[-1]

        def normalized_distance(s1: str, s2: str) -> float:
            """Normalized Levenshtein distance (0-1 scale)."""
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                return 0.0
            return levenshtein_distance(s1, s2) / max_len

        import numpy as np
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform

        n = len(filenames)

        # Build distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = normalized_distance(filenames[i].lower(), filenames[j].lower())
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Convert to condensed form for linkage
        condensed = squareform(dist_matrix)

        # Hierarchical clustering
        Z = linkage(condensed, method='average')

        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, n * 0.15)))

        # Generate dendrogram
        # Truncate labels for display
        labels = [f[:30] + "..." if len(f) > 33 else f for f in filenames]

        dend = dendrogram(Z, labels=labels, orientation='left', ax=ax,
                          color_threshold=0.3,  # Threshold for version detection
                          above_threshold_color='gray')

        # Add threshold line
        ax.axvline(x=0.3, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(0.32, 0.02, 'Version threshold (30%)', transform=ax.transAxes,
                fontsize=9, color='red', alpha=0.8)

        ax.set_xlabel('Normalized Levenshtein Distance', fontsize=12)
        ax.set_title('Filename Similarity Dendrogram\n(Detects versions, copies, related documents)',
                    fontsize=14, fontweight='bold')

        # Add stats
        n_clusters = len(set(dend['color_list'])) - 1  # Exclude gray
        stats_text = f"n = {n} documents\n~{n_clusters} version groups"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        return self._save_figure(fig, "filename_similarity_dendrogram", assets_dir,
                                 "Filename Similarity Dendrogram",
                                 "Hierarchical clustering of documents by filename similarity")

    def _generate_content_similarity_heatmap(self, concepts: Dict, metadata: Dict,
                                              assets_dir: Path) -> Optional[Dict]:
        """Generate content similarity heatmap using concept vectors.

        Detects plagiarism, copies, or near-duplicate content.
        """
        # Try different concept data formats
        doc_concepts = concepts.get("file_concepts", {}) or concepts.get("document_concepts", {})
        if not doc_concepts or len(doc_concepts) < 3:
            return None

        import numpy as np
        from scipy.cluster.hierarchy import linkage, dendrogram

        # Build concept vocabulary from the concept definitions
        concept_defs = concepts.get("concepts", [])
        all_concepts = set()

        # Extract concept labels from definitions
        for c in concept_defs:
            if isinstance(c, dict):
                label = c.get("label", c.get("concept_id", "")).lower()
                if label:
                    all_concepts.add(label)
            elif isinstance(c, str):
                all_concepts.add(c.lower())

        # Also extract from document concepts if definitions are sparse
        for doc_data in doc_concepts.values():
            if isinstance(doc_data, list):
                for c in doc_data:
                    if isinstance(c, str):
                        all_concepts.add(c.lower())
                    elif isinstance(c, dict):
                        label = c.get("label", c.get("concept", c.get("concept_id", ""))).lower()
                        if label:
                            all_concepts.add(label)

        if len(all_concepts) < 5:
            return None

        concept_list = sorted(all_concepts)
        concept_idx = {c: i for i, c in enumerate(concept_list)}

        # Build document vectors
        doc_ids = list(doc_concepts.keys())
        n_docs = len(doc_ids)
        n_concepts = len(concept_list)

        vectors = np.zeros((n_docs, n_concepts))
        for i, doc_id in enumerate(doc_ids):
            doc_data = doc_concepts[doc_id]
            if isinstance(doc_data, dict):
                concepts_list = doc_data.get("concepts", [])
            else:
                concepts_list = doc_data if isinstance(doc_data, list) else []

            for c in concepts_list:
                if isinstance(c, str):
                    c_name = c.lower()
                elif isinstance(c, dict):
                    # Handle multiple possible key names
                    c_name = c.get("label", c.get("concept", c.get("concept_id", c.get("name", "")))).lower()
                else:
                    continue
                if c_name and c_name in concept_idx:
                    vectors[i, concept_idx[c_name]] = 1.0

        # Compute cosine similarity matrix
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors_norm = vectors / norms

        similarity_matrix = vectors_norm @ vectors_norm.T

        # Get filenames for labels - handle both list and dict formats
        files_list = metadata.get("files", [])
        docs_dict = metadata.get("documents", {})

        # Build a lookup for file_id -> filename
        file_lookup = {}
        if files_list:
            for doc in files_list:
                fid = doc.get("file_id", doc.get("path", ""))
                fpath = doc.get("path", fid)
                file_lookup[fid] = fpath.split("/")[-1] if "/" in fpath else fpath
        elif docs_dict:
            for doc_id, doc in docs_dict.items():
                file_lookup[doc_id] = doc.get("filename", doc.get("name", doc_id))

        labels = []
        for doc_id in doc_ids:
            fname = file_lookup.get(doc_id, doc_id)
            # Truncate for display
            label = fname[:20] + "..." if len(fname) > 23 else fname
            labels.append(label)

        # Reorder by hierarchical clustering
        if n_docs > 2:
            # Convert similarity to distance
            dist_matrix = 1 - similarity_matrix
            np.fill_diagonal(dist_matrix, 0)

            condensed = []
            for i in range(n_docs):
                for j in range(i + 1, n_docs):
                    condensed.append(dist_matrix[i, j])
            condensed = np.array(condensed)

            Z = linkage(condensed, method='average')
            dend = dendrogram(Z, no_plot=True)
            order = dend['leaves']

            # Reorder matrix and labels
            similarity_matrix = similarity_matrix[order][:, order]
            labels = [labels[i] for i in order]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Heatmap
        im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Content Similarity (Cosine)', fontsize=11)

        # Labels
        ax.set_xticks(range(n_docs))
        ax.set_yticks(range(n_docs))
        ax.set_xticklabels(labels, rotation=90, fontsize=7, ha='center')
        ax.set_yticklabels(labels, fontsize=7)

        ax.set_title('Document Content Similarity Heatmap\n(Based on concept overlap)',
                    fontsize=14, fontweight='bold')

        # Highlight high similarity pairs (>0.85)
        high_sim_count = 0
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                if similarity_matrix[i, j] > 0.85:
                    high_sim_count += 1
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                fill=False, edgecolor='red', linewidth=2))

        # Stats
        stats_text = f"n = {n_docs} documents\nHigh similarity pairs (>0.85): {high_sim_count}"
        ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        return self._save_figure(fig, "doc_similarity_heatmap", assets_dir,
                                 "Content Similarity Heatmap",
                                 "Document similarity matrix based on concept overlap")

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate summary of visualizations."""
        figures = data.get("figures", {})
        assets_dir = data.get("assets_dir", "assets")

        if not figures:
            return "No visualizations generated."

        # Handle both dict and list formats
        if isinstance(figures, dict):
            fig_names = list(figures.keys())
            count = len(fig_names)
        else:
            fig_names = [f.get('name', '?') for f in figures]
            count = len(figures)

        return (
            f"Generated {count} visualizations in {assets_dir}. "
            f"Figures: {', '.join(fig_names)}. "
            f"Formats: SVG, PNG, PDF."
        )
