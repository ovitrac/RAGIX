# Plan: Generic Hierarchical Document Summarization System

**Project:** RAGIX
**Context:** VDP Audit (159 documents, 5,515 chunks indexed)
**Goal:** Design a generic KOAS-based pyramidal summarization system
**Date:** 2025-01-18

---

## 1. Problem Statement

The user needs to:
1. Summarize 200+ documents (specifications, not code)
2. Produce a **hierarchical/pyramidal view** (corpus → domain → document → section)
3. Keep the solution **generic** (reusable across projects)
4. Maintain **sovereignty** (no external API calls for raw data)
5. Use **slim LLMs** (Granite) for processing

### Current RAG State (VDP)
- **159 files indexed** (69 DOCX, 51 PDF, 9 PPTX, 7 XLSX)
- **5,515 chunks** in ChromaDB
- **Knowledge Graph**: File → Chunk → Concept hierarchy
- **Profile**: `docs_only` (1024 chars/chunk)

### KOAS Gap
- No existing summarization kernel
- All current kernels are code-focused (AST, metrics, complexity)
- Need new kernels for **document analysis**

---

## 2. Proposed Architecture: 3-Stage Document Summarization

Following the KOAS pattern (pure computation → structured output → LLM consumption):

### Stage 1: Document Collection (New Kernels)

| Kernel | Purpose | Output |
|--------|---------|--------|
| `doc_metadata` | Extract document metadata and statistics | File metadata, chunk counts, section structure |
| `doc_concepts` | Extract/aggregate concepts from RAG graph | Concept hierarchy, co-occurrence matrix |
| `doc_structure` | Detect document structure (headings, sections) | Document outline per file |

### Stage 2: Document Analysis (New Kernels)

| Kernel | Purpose | Output |
|--------|---------|--------|
| `doc_cluster` | Group documents by topic similarity | Clusters with files, centroid concepts |
| `doc_extract` | Extract key sentences per concept | Representative sentences per domain |
| `doc_coverage` | Analyze concept coverage across documents | Coverage matrix, gap analysis |

### Stage 3: Hierarchical Synthesis (New Kernel)

| Kernel | Purpose | Output |
|--------|---------|--------|
| `doc_pyramid` | Build pyramidal summary structure | Multi-level summary JSON + Markdown |

---

## 3. Core Design: `doc_pyramid` Kernel

The **key innovation** is using the RAG graph structure for hierarchical organization:

```
Level 4: CORPUS SUMMARY (1 summary)
    └── Level 3: DOMAIN SUMMARIES (N summaries, e.g., "Authentication", "Data Model")
        └── Level 2: DOCUMENT GROUP SUMMARIES (per cluster)
            └── Level 1: DOCUMENT SUMMARIES (per file)
```

### Pyramid Construction Algorithm

```python
class DocPyramidKernel(Kernel):
    """
    Build hierarchical document summary from RAG graph.

    Uses graph traversal to aggregate:
    1. Concepts → Group related chunks
    2. Chunks → Identify representative content
    3. Files → Cluster by concept overlap
    4. Clusters → Form domains
    """

    name = "doc_pyramid"
    stage = 3
    requires = ["doc_metadata", "doc_concepts", "doc_cluster", "doc_extract"]
    provides = ["hierarchical_summary", "pyramid_markdown"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load dependencies
        concepts = self._load_dependency("doc_concepts")
        clusters = self._load_dependency("doc_cluster")
        extracts = self._load_dependency("doc_extract")

        # Build pyramid bottom-up
        pyramid = {
            "level_1_documents": [],      # Per-document summaries
            "level_2_groups": [],          # Per-cluster summaries
            "level_3_domains": [],         # Per-concept-cluster summaries
            "level_4_corpus": None,        # Overall corpus summary
        }

        # Level 1: Aggregate per file
        for file_id, file_data in extracts["by_file"].items():
            pyramid["level_1_documents"].append({
                "file_id": file_id,
                "path": file_data["path"],
                "key_sentences": file_data["sentences"][:5],
                "concepts": file_data["concepts"],
                "chunk_count": file_data["chunk_count"],
            })

        # Level 2: Aggregate per cluster
        for cluster in clusters["clusters"]:
            pyramid["level_2_groups"].append({
                "cluster_id": cluster["id"],
                "label": cluster["label"],
                "files": cluster["file_ids"],
                "centroid_concepts": cluster["centroid_concepts"],
                "representative_sentences": self._merge_sentences(
                    [extracts["by_file"][fid] for fid in cluster["file_ids"]]
                ),
            })

        # Level 3: Domains (top-level concept groups)
        pyramid["level_3_domains"] = self._build_domains(concepts, clusters)

        # Level 4: Corpus summary (structured data for LLM)
        pyramid["level_4_corpus"] = self._build_corpus_summary(pyramid)

        return pyramid
```

---

## 4. Implementation Plan

### Phase A: Stage 1 Kernels (Collection)

**Files to create:**
- `ragix_kernels/docs/doc_metadata.py`
- `ragix_kernels/docs/doc_concepts.py`
- `ragix_kernels/docs/doc_structure.py`
- `ragix_kernels/docs/__init__.py`

**Key Implementation:**

```python
# ragix_kernels/docs/doc_metadata.py
class DocMetadataKernel(Kernel):
    name = "doc_metadata"
    stage = 1
    requires = []  # Uses RAG directly
    provides = ["doc_metadata", "doc_statistics"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        from ragix_core.rag_project import RAGProject, MetadataStore

        project_path = Path(input.config["project"]["path"])
        rag = RAGProject(project_path)
        metadata = MetadataStore(project_path)

        files = []
        for file_meta in metadata.list_files():
            files.append({
                "file_id": file_meta.file_id,
                "path": file_meta.path,
                "kind": file_meta.kind.value,
                "chunk_count": file_meta.chunk_count,
                "size_bytes": file_meta.size_bytes,
                "last_modified": file_meta.last_modified.isoformat(),
            })

        # Compute statistics
        stats = {
            "total_files": len(files),
            "total_chunks": sum(f["chunk_count"] for f in files),
            "by_kind": self._count_by_kind(files),
            "by_extension": self._count_by_extension(files),
        }

        return {"files": files, "statistics": stats}
```

### Phase B: Stage 2 Kernels (Analysis)

**Files to create:**
- `ragix_kernels/docs/doc_cluster.py`
- `ragix_kernels/docs/doc_extract.py`
- `ragix_kernels/docs/doc_coverage.py`

**Key Implementation:**

```python
# ragix_kernels/docs/doc_cluster.py
class DocClusterKernel(Kernel):
    name = "doc_cluster"
    stage = 2
    requires = ["doc_metadata", "doc_concepts"]
    provides = ["doc_clusters", "cluster_hierarchy"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        from ragix_core.rag_project import KnowledgeGraph, NodeType, EdgeType

        # Load RAG graph
        project_path = Path(input.config["project"]["path"])
        graph = KnowledgeGraph(project_path)
        graph.load()

        # Build file-concept matrix
        files = graph.get_nodes_by_type(NodeType.FILE)
        concepts = graph.get_nodes_by_type(NodeType.CONCEPT)

        # Compute file vectors based on concept coverage
        file_vectors = {}
        for file_node in files:
            chunks = graph.get_chunks_for_file(file_node.id)
            concept_weights = {}
            for chunk in chunks:
                for edge in graph.get_edges_from(chunk.id):
                    if edge.type == EdgeType.MENTIONS.value:
                        concept_id = edge.target
                        score = edge.data.get("score", 1.0)
                        concept_weights[concept_id] = concept_weights.get(concept_id, 0) + score
            file_vectors[file_node.id] = concept_weights

        # Hierarchical clustering (no external API needed)
        clusters = self._hierarchical_cluster(file_vectors, concepts)

        return {"clusters": clusters, "file_vectors": file_vectors}
```

### Phase C: Stage 3 Kernel (Synthesis)

**Files to create:**
- `ragix_kernels/docs/doc_pyramid.py`

**Output Format:**

```json
{
  "pyramid": {
    "level_4_corpus": {
      "title": "VDP Technical Specifications",
      "file_count": 159,
      "chunk_count": 5515,
      "domain_count": 6,
      "key_concepts": ["authentication", "data_model", "api", "security", "performance"],
      "coverage_gaps": ["error_handling", "logging"]
    },
    "level_3_domains": [
      {
        "domain_id": "D01",
        "label": "Authentication & Access Control",
        "concepts": ["oauth", "jwt", "rbac", "session"],
        "file_count": 23,
        "representative_sentences": [...]
      }
    ],
    "level_2_groups": [...],
    "level_1_documents": [...]
  },
  "markdown": "# Corpus Summary\n...",
  "statistics": {
    "processing_time_ms": 1234,
    "levels_generated": 4
  }
}
```

---

## 5. LLM Integration (Post-Kernel)

The kernels produce **structured data**. The LLM generates **natural language summaries** from this data.

### LLM Prompt Template

```markdown
You are a technical writer synthesizing document specifications.

## Input Data (from doc_pyramid kernel)
{pyramid_json}

## Task
Generate a {level} summary for the {target} domain.

## Constraints
- Maximum {max_words} words
- Preserve technical accuracy
- Highlight key concepts: {concepts}
- Note coverage gaps: {gaps}
```

### Integration with KOAS Orchestrator

The orchestrator can optionally call a local LLM (Ollama/Granite) after Stage 3 to generate prose summaries from kernel outputs:

```python
# In orchestrator.py (enhancement)
def generate_llm_summaries(self, pyramid_output: Dict) -> Dict[str, str]:
    """Generate prose summaries from pyramid data using local LLM."""
    from ragix_core.llm import OllamaClient

    client = OllamaClient(model="granite")
    summaries = {}

    for domain in pyramid_output["level_3_domains"]:
        prompt = self._build_summary_prompt(domain)
        summaries[domain["domain_id"]] = client.generate(prompt)

    return summaries
```

---

## 6. File Structure

```
ragix_kernels/
├── base.py                    # Existing kernel base class
├── registry.py                # Existing kernel registry
├── orchestrator.py            # Enhanced with LLM summary option
├── audit/                     # Existing code audit kernels
│   ├── ast_scan.py
│   ├── metrics.py
│   └── ...
└── docs/                      # NEW: Document summarization kernels
    ├── __init__.py
    ├── doc_metadata.py        # Stage 1: Document metadata
    ├── doc_concepts.py        # Stage 1: Concept extraction
    ├── doc_structure.py       # Stage 1: Document structure
    ├── doc_cluster.py         # Stage 2: Document clustering
    ├── doc_extract.py         # Stage 2: Key sentence extraction
    ├── doc_coverage.py        # Stage 2: Coverage analysis
    └── doc_pyramid.py         # Stage 3: Hierarchical synthesis
```

---

## 7. Configuration (manifest.yaml extension)

```yaml
# Stage 1: Document Collection
stage1:
  doc_metadata:
    enabled: true
    options:
      include_structure: true

  doc_concepts:
    enabled: true
    options:
      min_concept_frequency: 3
      max_concepts: 100

  doc_structure:
    enabled: true
    options:
      detect_headings: true
      extract_toc: true

# Stage 2: Document Analysis
stage2:
  doc_cluster:
    enabled: true
    options:
      method: "hierarchical"  # or "kmeans"
      n_clusters: auto        # or integer
      min_cluster_size: 3

  doc_extract:
    enabled: true
    options:
      sentences_per_concept: 3
      sentences_per_file: 5

  doc_coverage:
    enabled: true
    options:
      reference_concepts: []  # optional expected concepts

# Stage 3: Pyramid Generation
stage3:
  doc_pyramid:
    enabled: true
    options:
      levels: 4               # 1-4
      include_markdown: true
      language: "fr"          # or "en"
```

---

## 8. Verification Plan

### Unit Tests
1. Test `doc_metadata` on VDP project → expect 159 files, correct statistics
2. Test `doc_concepts` → expect concepts extracted from graph
3. Test `doc_cluster` → expect coherent groupings

### Integration Test
```bash
# Initialize workspace for VDP
python -m ragix_kernels.orchestrator init \
  --workspace ./audit/VDP/workspace \
  --project /home/olivi/Documents/Adservio/audit/VDP/src \
  --language docs

# Run all stages
python -m ragix_kernels.orchestrator run \
  --workspace ./audit/VDP/workspace \
  --all

# Check pyramid output
cat ./audit/VDP/workspace/stage3/doc_pyramid.json | jq '.pyramid.level_4_corpus'
```

### Expected Output
- Stage 1: `doc_metadata.json`, `doc_concepts.json`, `doc_structure.json`
- Stage 2: `doc_cluster.json`, `doc_extract.json`, `doc_coverage.json`
- Stage 3: `doc_pyramid.json` with multi-level hierarchy + `doc_pyramid.md` Markdown

---

## 9. Dependencies

### Existing (No Changes)
- `ragix_core.rag_project.RAGProject` - Access RAG data
- `ragix_core.rag_project.KnowledgeGraph` - Graph traversal
- `ragix_core.rag_project.MetadataStore` - File metadata
- `ragix_kernels.base.Kernel` - Kernel base class

### New (Optional)
- `scikit-learn` - For clustering (if not using pure graph-based clustering)
- Already available in most environments

---

## 10. Summary

This plan creates a **generic document summarization system** that:

1. **Follows KOAS patterns** - Pure computation, no LLM inside kernels
2. **Uses existing RAG graph** - Leverages File→Chunk→Concept structure
3. **Builds pyramid bottom-up** - Document → Group → Domain → Corpus
4. **Is project-agnostic** - Works on any indexed project
5. **Integrates with slim LLMs** - Granite/Mistral generate prose from structured data
6. **Maintains sovereignty** - All processing local, no external API calls

### Critical Files to Create
1. `ragix_kernels/docs/__init__.py`
2. `ragix_kernels/docs/doc_metadata.py` (Stage 1)
3. `ragix_kernels/docs/doc_concepts.py` (Stage 1)
4. `ragix_kernels/docs/doc_cluster.py` (Stage 2)
5. `ragix_kernels/docs/doc_extract.py` (Stage 2)
6. `ragix_kernels/docs/doc_pyramid.py` (Stage 3)

### Implementation Order
1. Phase A: Stage 1 kernels (foundation)
2. Phase B: Stage 2 kernels (analysis)
3. Phase C: Stage 3 kernel (synthesis)
4. Phase D: Integration tests on VDP

---

## 11. Microsoft GraphRAG: Opportunity Analysis

### What is GraphRAG?

[Microsoft GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/) is a structured, hierarchical approach to Retrieval Augmented Generation released in 2024. The foundational paper ["From Local to Global: A Graph RAG Approach to Query-Focused Summarization"](https://arxiv.org/abs/2404.16130) (Edge et al., April 2024) introduces a technique that combines:

1. **Entity extraction** → Build a knowledge graph from source documents
2. **Community detection** → Use the Leiden algorithm for hierarchical clustering
3. **Pre-generated summaries** → LLM summarizes each community at multiple levels
4. **Map-reduce answering** → Parallel partial answers consolidated into final response

### Key Technical Innovations

| Component | GraphRAG Approach | RAGIX Equivalent |
|-----------|------------------|------------------|
| **Graph Structure** | Entity-relationship graph extracted by LLM | File→Chunk→Concept graph (already indexed) |
| **Hierarchical Clustering** | Leiden algorithm on entity graph | Hierarchical clustering on file-concept vectors |
| **Summary Generation** | Bottom-up LLM summaries per community | Structured data extraction, LLM summary post-kernel |
| **Query Processing** | Map-reduce over community summaries | Pyramid levels as context for LLM |

### GraphRAG Performance Results

According to Microsoft Research:
- **70-80% win rate** over naive RAG on comprehensiveness and diversity
- **~2-3% token use** per query compared to full source text summarization
- Effective on datasets in the **1 million token range**

### Why GraphRAG is Relevant to Our Design

GraphRAG validates our architectural choices:

1. **Hierarchical communities = Our pyramid levels** — Both approaches recognize that global understanding requires hierarchical aggregation, not flat retrieval.

2. **Pre-generated summaries = Our kernel outputs** — GraphRAG pre-computes community summaries at indexing time. Our kernels pre-compute structured data (concepts, clusters, extracts) for later LLM consumption.

3. **Map-reduce pattern = Our multi-level synthesis** — GraphRAG processes community summaries in parallel then consolidates. Our `doc_pyramid` kernel aggregates from documents → groups → domains → corpus.

### Differences and Trade-offs

| Aspect | GraphRAG | RAGIX/KOAS Approach |
|--------|----------|---------------------|
| **LLM in indexing** | Heavy (entity extraction, summarization) | None (pure computation) |
| **Sovereignty** | Requires LLM API during indexing | Fully local until final summarization |
| **Token cost** | High at index time, low at query time | Minimal at index time, moderate at synthesis |
| **Customization** | Limited (LLM-driven) | Full control (kernel parameters) |
| **Reproducibility** | Depends on LLM consistency | Deterministic (same input = same output) |

### Integration Opportunity: Hybrid Approach

We could adopt GraphRAG's Leiden community detection for the `doc_cluster` kernel while maintaining KOAS sovereignty:

```python
# Enhanced doc_cluster kernel with Leiden algorithm
class DocClusterKernel(Kernel):
    def compute(self, input: KernelInput) -> Dict[str, Any]:
        method = input.config.get("method", "hierarchical")

        if method == "leiden":
            # Use Leiden algorithm (from python-igraph or leidenalg)
            import igraph as ig
            import leidenalg

            # Build graph from file-concept similarities
            g = self._build_similarity_graph(file_vectors)

            # Multi-resolution Leiden clustering
            partitions = []
            for resolution in [0.1, 0.5, 1.0, 2.0]:
                partition = leidenalg.find_partition(
                    g,
                    leidenalg.RBConfigurationVertexPartition,
                    resolution_parameter=resolution
                )
                partitions.append({
                    "resolution": resolution,
                    "communities": self._extract_communities(partition)
                })

            return {"clusters": partitions, "method": "leiden"}
        else:
            # Existing hierarchical clustering
            return self._hierarchical_cluster(file_vectors, concepts)
```

### Recommendation

**Adopt Leiden community detection as an optional clustering method** in the `doc_cluster` kernel:

1. **Default**: Use existing hierarchical clustering (no new dependencies)
2. **Optional**: Enable Leiden with `method: "leiden"` for GraphRAG-style community detection
3. **New dependency**: `leidenalg` + `python-igraph` (optional, only if Leiden is enabled)

This provides:
- **Compatibility** with GraphRAG research findings
- **Sovereignty** maintained (no LLM during indexing)
- **Flexibility** to choose clustering approach per project

### References

- [GraphRAG: New tool for complex data discovery now on GitHub](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization (arXiv)](https://arxiv.org/abs/2404.16130)
- [GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [GraphRAG: Improving global search via dynamic community selection](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/)
