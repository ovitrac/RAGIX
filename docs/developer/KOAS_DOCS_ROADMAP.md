# KOAS-Docs Improvement Roadmap

**Document Summarization System — Evolution Plan**

**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Version:** 1.0
**Date:** 2026-01-18
**Status:** Active Development

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State (v1.0)](#2-current-state-v10)
3. [Roadmap Overview](#3-roadmap-overview)
4. [Phase 1: Dual Clustering Architecture](#4-phase-1-dual-clustering-architecture)
5. [Phase 2: Tutor Model Verification](#5-phase-2-tutor-model-verification)
6. [Phase 3: LLM Response Caching](#6-phase-3-llm-response-caching)
7. [Phase 4: Semantic Discrepancy Detection](#7-phase-4-semantic-discrepancy-detection)
8. [Phase 5: Final Report Architecture](#8-phase-5-final-report-architecture)
9. [Phase 6: Incremental Updates](#9-phase-6-incremental-updates)
10. [Implementation Priority](#10-implementation-priority)
11. [Success Metrics](#11-success-metrics)

---

## 1. Executive Summary

This roadmap defines the evolution of KOAS-Docs from v1.0 to v2.0, focusing on:

1. **Robustness**: Dual clustering with reconciliation to avoid single-method bias
2. **Quality assurance**: Tutor model verification to prevent hallucination
3. **Efficiency**: LLM response caching to reduce redundant computation
4. **Clarity**: Final report with explicit methodology and appendices
5. **Maintainability**: Incremental updates based on document changes

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tutor model | `mistral:7b-instruct` | French-capable, 7B for review of 3B outputs |
| Dual clustering | Hierarchical + Leiden | Parallel execution, reconciliation or dual-view |
| Caching | Hash-based with model versioning | Deterministic cache invalidation |
| Final report location | `{run}/final_report.md` | Top-level, clear naming |

---

## 2. Current State (v1.0)

### Capabilities

- 11 kernels across 3 stages
- Per-document LLM summaries (Granite 3B)
- Hierarchical clustering only
- Terminology-based discrepancy detection
- Stage-based output structure

### Limitations

| Limitation | Impact | Priority |
|------------|--------|----------|
| Single clustering method | Potential bias | **High** |
| No summary verification | Hallucination risk | **High** |
| No LLM caching | Redundant computation | **Medium** |
| Lexical discrepancy only | Misses semantic conflicts | **Medium** |
| Fragmented outputs | Navigation complexity | **Medium** |
| Full recomputation | Inefficient updates | **Low** |

---

## 3. Roadmap Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KOAS-Docs ROADMAP                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  v1.0 (Current)                                                     │
│    │                                                                │
│    ├──▶ Phase 1: Dual Clustering (Hierarchical + Leiden)            │
│    │      └── Parallel execution, reconciliation logic              │
│    │                                                                │
│    ├──▶ Phase 2: Tutor Model Verification                           │
│    │      └── Mistral 7B reviews Granite 3B summaries               │
│    │                                                                │
│    ├──▶ Phase 3: LLM Response Caching                               │
│    │      └── Hash-based cache, model versioning                    │
│    │                                                                │
│    ├──▶ Phase 4: Semantic Discrepancy Detection                     │
│    │      └── Embedding-based contradiction detection               │
│    │                                                                │
│    ├──▶ Phase 5: Final Report Architecture                          │
│    │      └── Consolidated report with appendices                   │
│    │                                                                │
│    └──▶ Phase 6: Incremental Updates                                │
│           └── Change detection, partial recomputation               │
│                                                                     │
│  v2.0 (Target)                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1: Dual Clustering Architecture

### Objective

Run **hierarchical clustering** and **Leiden community detection** in parallel, then reconcile results or present both perspectives when they diverge.

### Design

```
                    doc_concepts
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    doc_cluster_hier           doc_cluster_leiden
    (hierarchical)             (community detection)
            │                         │
            └────────────┬────────────┘
                         ▼
              doc_cluster_reconcile
                         │
                         ▼
              ┌──────────┴──────────┐
              │                     │
        (if similar)          (if divergent)
              │                     │
              ▼                     ▼
       unified_clusters      dual_view_clusters
```

### New Kernels

#### `doc_cluster_leiden`

```python
class DocClusterLeidenKernel(Kernel):
    """
    Leiden community detection on document similarity graph.

    Uses python-igraph + leidenalg for multi-resolution clustering.
    Produces communities at multiple resolution levels for comparison.
    """
    name = "doc_cluster_leiden"
    version = "1.0.0"
    stage = 2
    requires = ["doc_metadata", "doc_concepts"]
    provides = ["leiden_communities", "resolution_levels"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        import igraph as ig
        import leidenalg

        # Build similarity graph from concept overlap
        graph = self._build_similarity_graph(concepts, files)

        # Multi-resolution Leiden
        resolutions = [0.1, 0.5, 1.0, 2.0, 5.0]
        communities = {}

        for res in resolutions:
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=res
            )
            communities[res] = self._extract_communities(partition)

        # Select optimal resolution (modularity-based)
        optimal = self._select_optimal_resolution(communities)

        return {
            "communities": communities,
            "optimal_resolution": optimal,
            "optimal_clusters": communities[optimal]
        }
```

#### `doc_cluster_reconcile`

```python
class DocClusterReconcileKernel(Kernel):
    """
    Reconcile hierarchical and Leiden clustering results.

    Decision logic:
    1. If cluster overlap > 80%: merge into unified view
    2. If overlap 50-80%: attempt reconciliation with warnings
    3. If overlap < 50%: keep dual view, document divergence
    """
    name = "doc_cluster_reconcile"
    version = "1.0.0"
    stage = 2
    requires = ["doc_cluster", "doc_cluster_leiden"]
    provides = ["reconciled_clusters", "clustering_report"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        hier = self._load_dependency("doc_cluster")
        leiden = self._load_dependency("doc_cluster_leiden")

        # Compute cluster overlap (Jaccard on file sets)
        overlap = self._compute_overlap(hier, leiden)

        if overlap > 0.8:
            # High agreement: merge
            return {
                "mode": "unified",
                "clusters": self._merge_clusters(hier, leiden),
                "agreement_score": overlap,
                "methodology": "Both methods produced similar groupings (Jaccard > 0.8). "
                              "Unified view uses hierarchical labels with Leiden validation."
            }
        elif overlap > 0.5:
            # Moderate agreement: reconcile with warnings
            return {
                "mode": "reconciled",
                "clusters": self._reconcile_clusters(hier, leiden),
                "agreement_score": overlap,
                "divergences": self._identify_divergences(hier, leiden),
                "methodology": "Methods showed moderate agreement (0.5 < Jaccard < 0.8). "
                              "Reconciled view prioritizes hierarchical structure with "
                              "Leiden-detected communities highlighted."
            }
        else:
            # Low agreement: dual view
            return {
                "mode": "dual_view",
                "hierarchical_clusters": hier["clusters"],
                "leiden_clusters": leiden["optimal_clusters"],
                "agreement_score": overlap,
                "methodology": "Clustering methods produced divergent results (Jaccard < 0.5). "
                              "Both views are preserved. Hierarchical clustering groups by "
                              "path structure and explicit concepts. Leiden clustering groups "
                              "by content similarity patterns. Users should consult both "
                              "perspectives for comprehensive understanding."
            }
```

### Dependencies

```bash
pip install python-igraph leidenalg
```

### Output Structure

```json
{
  "mode": "dual_view",
  "agreement_score": 0.42,
  "methodology": "Clustering methods produced divergent results...",
  "hierarchical_clusters": [...],
  "leiden_clusters": [...],
  "divergence_analysis": {
    "documents_in_different_clusters": ["F001", "F015", "F089"],
    "possible_reasons": [
      "Hierarchical sensitive to path structure",
      "Leiden sensitive to content similarity"
    ]
  }
}
```

---

## 5. Phase 2: Tutor Model Verification

### Objective

Use a **larger tutor model** (`mistral:7b-instruct`) to verify summaries generated by the **worker model** (`granite3.1-moe:3b`), preventing hallucination and ensuring accuracy.

### Model Selection

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| Worker | `granite3.1-moe:3b` | 3B | Fast generation, bulk processing |
| Tutor | `mistral:7b-instruct` | 7B | Quality verification, French fluency |

### Design

```
┌────────────────────────────────────────────────────────────────────────┐
│                    TUTOR VERIFICATION PATTERN                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Document Chunks ──▶ Worker (Granite 3B) ──▶ Draft Summary             │
│                                                     │                  │
│                                                     ▼                  │
│                                            ┌───────────────┐           │
│                                            │ Tutor Check   │           │
│                                            │ (Mistral 7B)  │           │
│                                            └───────┬───────┘           │
│                                                    │                   │
│                              ┌─────────────────────┼────────────┐      │
│                              ▼                     ▼            ▼      │
│                         [ACCEPT]              [REFINE]      [REJECT]   │
│                              │                     │            │      │
│                              ▼                     ▼            ▼      │
│                        Use draft          Tutor corrects   Re-generate │
│                                            summary         with tutor  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Tutor Prompt

```markdown
Tu es un vérificateur de résumés documentaires. Ta tâche est de valider
la qualité et l'exactitude d'un résumé généré automatiquement.

## Document original (extraits clés)
{key_sentences}

## Résumé à vérifier
{draft_summary}

## Instructions
1. Vérifie que le résumé ne contient PAS d'informations absentes des extraits
2. Vérifie que les thèmes identifiés sont cohérents avec le contenu
3. Vérifie que le périmètre est correctement identifié

## Format de réponse
VERDICT: [ACCEPTÉ | CORRIGÉ | REJETÉ]
RAISON: [Explication courte]
CORRECTION: [Si CORRIGÉ, fournir le résumé corrigé]
```

### Implementation

```python
class DocSummarizeTutoredKernel(Kernel):
    """
    Per-document summaries with tutor verification.

    Uses two-stage generation:
    1. Worker (Granite 3B): Generate draft summary
    2. Tutor (Mistral 7B): Verify and correct if needed
    """
    name = "doc_summarize_tutored"
    version = "1.0.0"
    stage = 3

    WORKER_MODEL = "granite3.1-moe:3b"
    TUTOR_MODEL = "mistral:7b-instruct"

    def _generate_with_verification(self, doc_context: Dict) -> Dict:
        # Stage 1: Worker generates draft
        draft = self._call_llm(
            self.WORKER_MODEL,
            self._build_worker_prompt(doc_context)
        )

        # Stage 2: Tutor verifies
        verification = self._call_llm(
            self.TUTOR_MODEL,
            self._build_tutor_prompt(doc_context, draft)
        )

        verdict = self._parse_verdict(verification)

        if verdict["status"] == "ACCEPTÉ":
            return {"summary": draft, "verified": True, "tutor_action": "accepted"}
        elif verdict["status"] == "CORRIGÉ":
            return {"summary": verdict["correction"], "verified": True, "tutor_action": "corrected"}
        else:  # REJETÉ
            # Re-generate with tutor
            regenerated = self._call_llm(
                self.TUTOR_MODEL,
                self._build_regeneration_prompt(doc_context)
            )
            return {"summary": regenerated, "verified": True, "tutor_action": "regenerated"}
```

### Quality Metrics

Track tutor actions to monitor worker model quality:

```json
{
  "tutor_statistics": {
    "total_summaries": 137,
    "accepted": 112,
    "corrected": 20,
    "regenerated": 5,
    "acceptance_rate": 0.82,
    "correction_rate": 0.15,
    "rejection_rate": 0.03
  }
}
```

---

## 6. Phase 3: LLM Response Caching

### Objective

Cache LLM responses to avoid redundant computation when:
- Same document content is processed again
- Same prompt is sent to the same model
- Pipeline is re-run without source changes

### Cache Key Design

```python
def compute_cache_key(
    model: str,
    prompt: str,
    temperature: float = 0.3
) -> str:
    """
    Generate deterministic cache key.

    Key components:
    - Model name and version
    - Prompt hash (SHA256)
    - Temperature (affects output)
    """
    import hashlib

    key_data = f"{model}:{temperature}:{prompt}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]
```

### Cache Structure

```
.KOAS/cache/
├── llm_responses/
│   ├── granite3.1-moe_3b/
│   │   ├── a1b2c3d4e5f6g7h8.json  # Cached response
│   │   ├── b2c3d4e5f6g7h8i9.json
│   │   └── ...
│   └── mistral_7b-instruct/
│       ├── c3d4e5f6g7h8i9j0.json
│       └── ...
├── cache_index.json  # Index for fast lookup
└── cache_stats.json  # Hit/miss statistics
```

### Cache Entry Format

```json
{
  "cache_key": "a1b2c3d4e5f6g7h8",
  "model": "granite3.1-moe:3b",
  "model_digest": "sha256:b43d80d7fca7...",
  "prompt_hash": "sha256:...",
  "temperature": 0.3,
  "created_at": "2026-01-18T14:30:00.123456Z",
  "response": "...",
  "response_hash": "sha256:...",
  "usage": {
    "prompt_tokens": 512,
    "completion_tokens": 128
  },
  "sovereignty": {
    "endpoint": "http://127.0.0.1:11434",
    "hostname": "LX-Olivier2023",
    "user": "olivi",
    "local": true
  }
}
```

### Implementation

```python
class LLMCache:
    """
    Persistent cache for LLM responses.

    Features:
    - Hash-based key derivation
    - Model version awareness
    - TTL support (optional)
    - Statistics tracking
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"hits": 0, "misses": 0}

    def get(self, model: str, prompt: str, temperature: float = 0.3) -> Optional[str]:
        key = self._compute_key(model, prompt, temperature)
        cache_file = self._get_cache_path(model, key)

        if cache_file.exists():
            self.stats["hits"] += 1
            data = json.loads(cache_file.read_text())
            return data["response"]

        self.stats["misses"] += 1
        return None

    def put(self, model: str, prompt: str, response: str, temperature: float = 0.3):
        key = self._compute_key(model, prompt, temperature)
        cache_file = self._get_cache_path(model, key)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "cache_key": key,
            "model": model,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "temperature": temperature,
            "created_at": datetime.utcnow().isoformat(),
            "response": response
        }))
```

### Integration with Kernels

```python
class DocSummarizeKernel(Kernel):
    def __init__(self):
        self.cache = LLMCache(self.workspace / ".KOAS/cache")

    def _call_llm_cached(self, model: str, prompt: str) -> str:
        # Check cache first
        cached = self.cache.get(model, prompt)
        if cached:
            return cached

        # Generate and cache
        response = self._call_llm(model, prompt)
        self.cache.put(model, prompt, response)
        return response
```

---

## 7. Phase 4: Semantic Discrepancy Detection

### Objective

Detect **semantic contradictions** between documents, not just terminology variations.

### Approach

Use sentence embeddings to find documents making **similar claims** about the same concepts, then verify consistency.

```
Document A: "L'authentification utilise OAuth 2.0"
Document B: "L'authentification repose sur SAML"

→ Semantic similarity: HIGH (both about authentication)
→ Content contradiction: POTENTIAL (different protocols)
→ Flag for review
```

### Implementation

```python
class DocCompareSemanticKernel(Kernel):
    """
    Semantic discrepancy detection using embeddings.
    """
    name = "doc_compare_semantic"
    version = "1.0.0"
    stage = 3
    requires = ["doc_extract", "doc_concepts"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Group sentences by concept
        sentences_by_concept = self._group_by_concept(extracts)

        contradictions = []

        for concept, sentences in sentences_by_concept.items():
            # Compute embeddings
            embeddings = model.encode([s["text"] for s in sentences])

            # Find high-similarity pairs
            for i, j in self._find_similar_pairs(embeddings, threshold=0.8):
                sent_i, sent_j = sentences[i], sentences[j]

                # Check if from different documents
                if sent_i["file_id"] != sent_j["file_id"]:
                    # Verify semantic consistency (simple negation check)
                    if self._may_contradict(sent_i["text"], sent_j["text"]):
                        contradictions.append({
                            "concept": concept,
                            "sentence_a": sent_i,
                            "sentence_b": sent_j,
                            "similarity": float(cosine_similarity(...)),
                            "potential_contradiction": True
                        })

        return {"semantic_contradictions": contradictions}
```

---

## 8. Phase 5: Final Report Architecture

### Objective

Produce a **consolidated final report** that:
1. Lives at the run root (not buried in stage3/)
2. Includes explicit methodology
3. Contains intermediate reports as appendices
4. Clearly states what was analyzed and how

### Output Structure

```
.KOAS/runs/run_YYYYMMDD_HHMMSS_XXXXXX/
├── final_report.md          ← PRIMARY OUTPUT
├── final_report.json        ← Structured data
├── audit_trail.json
├── stage1/
├── stage2/
├── stage3/
├── summaries/
└── appendices/
    ├── A_corpus_summary.md
    ├── B_domain_summaries.md
    ├── C_functionality_catalog.md
    ├── D_discrepancy_details.md
    └── E_clustering_analysis.md
```

### Final Report Structure

```markdown
# Document Analysis Report

**Project:** DOCSET Technical Specifications
**Date:** 2026-01-18
**Run ID:** run_20260118_134851_7bd0c4
**RAGIX Version:** 0.5.x
**KOAS Version:** 1.0.0
**System Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab

---

## Methodology

### Analysis Process

This report was generated by the KOAS-Docs system (Kernel-Orchestrated Audit System
for Documents), developed by Olivier Vitrac as part of the RAGIX project.

The analysis follows a three-stage process:

1. **Collection** (Stage 1): Metadata, concepts and structure extraction
   - Source: RAG Index (ChromaDB + Knowledge Graph)
   - 137 documents indexed, 5,481 chunks

2. **Analysis** (Stage 2): Clustering, key sentence extraction, functionalities
   - Clustering: Hierarchical + Leiden (parallel execution)
   - Reconciliation mode: [unified|reconciled|dual_view]
   - Method agreement score: X.XX

3. **Synthesis** (Stage 3): Summary generation with verification
   - Generation model: granite3.1-moe:3b
   - Verification model: mistral:7b-instruct
   - Acceptance rate: XX%

### Quality Guarantees

- **Reproducibility**: Deterministic execution (same input = same output)
- **Traceability**: SHA256 checksums for all artifacts
- **Verification**: Dual validation by tutor model
- **Transparency**: Explicit methodology, no hidden steps

### Limitations

- Summaries are generated by local language models (3B-7B parameters)
- Contradiction detection is based on lexical and semantic similarity
- Extracted functionalities depend on SPD document structure

---

## Executive Summary

[Generated content]

---

## Corpus Overview

### Statistics
[Tables and metrics]

### Thematic Organization
[Cluster/domain overview]

---

## Domain Summaries

### Domain 1: [Label]
[Domain summary with document list]

### Domain 2: [Label]
[...]

---

## Functionality Catalog

[If SPD documents present]

---

## Discrepancy Analysis

### Detected Discrepancies
[Discrepancy summary]

### Recommendations
[Action items]

---

## Appendices

The following appendices contain the full analysis details:

- **Appendix A**: Full corpus summary
- **Appendix B**: Detailed domain summaries
- **Appendix C**: Complete functionality catalog
- **Appendix D**: Discrepancy and divergence details
- **Appendix E**: Clustering analysis (hierarchical vs Leiden)

---

## Technical Information

| Parameter | Value |
|-----------|-------|
| Execution time | XXX seconds |
| LLM requests | XXX (YYY from cache) |
| Kernels executed | 11 |
| Audit trail checksum | sha256:... |

---

## Sovereignty Attestation

This analysis was performed **entirely locally**, with no calls to external cloud
services. All data remained on the client's infrastructure.

### Execution Environment

| Parameter | Value |
|-----------|-------|
| **Hostname** | LX-Olivier2023 |
| **User** | olivi |
| **Platform** | Linux 6.8.0-90-generic |
| **Python** | 3.12.12 (conda-forge) |
| **Start time** | 2026-01-18T13:48:51.195335Z |
| **End time** | 2026-01-18T14:05:10.442935Z |
| **Timezone** | Europe/Paris (UTC+1) |

### LLM Models Used

| Model | Role | Endpoint | Digest |
|-------|------|----------|--------|
| granite3.1-moe:3b | Worker (generation) | http://127.0.0.1:11434 | b43d80d7fca7 |
| mistral:7b-instruct | Tutor (verification) | http://127.0.0.1:11434 | 6577803aa9a0 |

### Locality Proof

- **Ollama Endpoint**: `127.0.0.1:11434` (loopback, local only)
- **No external requests**: Verifiable via network audit
- **Downloaded models**: Stored locally in ~/.ollama/models/
- **RAG Index**: Local ChromaDB in .RAG/

### Execution Signature

```
Run ID:        run_20260118_134851_7bd0c4
Audit Trail:   sha256:XXXX...
Configuration: sha256:XXXX...
```

---

*Generated by KOAS-Docs — RAGIX Project*
*Adservio Innovation Lab | 2026*
```

### New Kernel: `doc_final_report`

```python
class DocFinalReportKernel(Kernel):
    """
    Generate consolidated final report with appendices.

    Outputs:
    - final_report.md at run root
    - final_report.json (structured data)
    - appendices/*.md (detailed sections)
    """
    name = "doc_final_report"
    version = "1.0.0"
    stage = 3
    requires = [
        "doc_pyramid", "doc_summarize", "doc_compare",
        "doc_coverage", "doc_func_extract", "doc_cluster_reconcile"
    ]
    provides = ["final_report"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load all dependencies
        pyramid = self._load_dependency("doc_pyramid")
        summaries = self._load_dependency("doc_summarize")
        compare = self._load_dependency("doc_compare")
        coverage = self._load_dependency("doc_coverage")
        funcs = self._load_dependency("doc_func_extract")
        clusters = self._load_dependency("doc_cluster_reconcile")

        # Build methodology section
        methodology = self._build_methodology(clusters, summaries)

        # Generate appendices
        appendices = self._generate_appendices(...)

        # Assemble final report
        report = self._assemble_report(
            methodology=methodology,
            pyramid=pyramid,
            summaries=summaries,
            compare=compare,
            funcs=funcs,
            appendices=appendices
        )

        # Write to run root
        final_path = self.run_dir / "final_report.md"
        final_path.write_text(report)

        return {"report_path": str(final_path)}
```

---

## 9. Phase 6: Incremental Updates

### Objective

Avoid full recomputation when only some documents have changed.

### Change Detection

```python
def detect_changes(
    previous_run: Path,
    current_rag: RAGProject
) -> Dict[str, List[str]]:
    """
    Compare current RAG state with previous run.

    Returns:
        {
            "added": [file_ids],
            "modified": [file_ids],
            "deleted": [file_ids],
            "unchanged": [file_ids]
        }
    """
    previous_metadata = load_json(previous_run / "stage1/doc_metadata.json")
    current_metadata = current_rag.metadata_store.list_files()

    # Compare by file_id and content hash
    ...
```

### Selective Recomputation

```
Document changes detected:
  - 3 added, 2 modified, 1 deleted

Recomputation plan:
  ✓ doc_metadata: FULL (index changed)
  ✓ doc_concepts: PARTIAL (only affected files)
  ✓ doc_cluster: FULL (clustering depends on all files)
  ✓ doc_extract: PARTIAL (only changed files)
  ✓ doc_summarize: PARTIAL (only changed files, use cache for unchanged)
```

---

## 10. Implementation Priority

### Phase Prioritization

| Phase | Priority | Effort | Impact | Dependencies |
|-------|----------|--------|--------|--------------|
| **Phase 3: Caching** | 1 | Medium | High | None |
| **Phase 2: Tutor Model** | 2 | Medium | High | Phase 3 |
| **Phase 5: Final Report** | 3 | Low | High | None |
| **Phase 1: Dual Clustering** | 4 | High | High | leidenalg |
| **Phase 4: Semantic Detection** | 5 | Medium | Medium | sentence-transformers |
| **Phase 6: Incremental** | 6 | High | Medium | All phases |

### Recommended Implementation Order

```
Week 1-2: Phase 3 (Caching)
  └── Foundation for efficient iteration

Week 3-4: Phase 2 (Tutor Model)
  └── Quality assurance, French fluency

Week 5: Phase 5 (Final Report)
  └── Consolidate outputs, methodology

Week 6-8: Phase 1 (Dual Clustering)
  └── Robustness, reconciliation logic

Week 9-10: Phase 4 (Semantic Detection)
  └── Enhanced discrepancy detection

Week 11-12: Phase 6 (Incremental)
  └── Operational efficiency
```

---

## 11. Success Metrics

### Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tutor acceptance rate | >85% | Summaries accepted without correction |
| Cache hit rate | >60% | Re-runs with unchanged documents |
| Clustering agreement | >0.6 | Jaccard overlap between methods |
| Semantic contradiction precision | >80% | True contradictions / flagged |

### Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Full pipeline time | <10 min | ~16 min |
| Incremental update time | <2 min | N/A |
| LLM calls per document | <3 | ~2 |
| Cache storage | <100 MB | N/A |

### Operational Metrics

| Metric | Target |
|--------|--------|
| Report completeness | All sections populated |
| Methodology clarity | No unexplained steps |
| Reproducibility | 100% deterministic |
| Auditability | Full checksum chain |

---

## Appendix: New Dependencies

```
# requirements-koas-docs.txt

# Core (existing)
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Phase 1: Dual Clustering
python-igraph>=0.10.0
leidenalg>=0.10.0

# Phase 4: Semantic Detection
# (uses existing sentence-transformers)

# Optional: Visualization
matplotlib>=3.7.0
networkx>=3.0
```

---

*KOAS-Docs Roadmap — RAGIX Project*
*Adservio Innovation Lab | 2026*
