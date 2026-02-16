# KOAS-Docs — Document Summarization System

**Hierarchical Document Analysis for Large-Scale Specification Corpora**

**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Version:** 1.0.0
**Date:** 2026-01-18
**RAGIX Version:** 0.5+
**KOAS Version:** 1.0

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Architecture](#3-architecture)
4. [The Document Kernel Family](#4-the-document-kernel-family)
5. [Three-Stage Pipeline](#5-three-stage-pipeline)
6. [LLM Integration Pattern](#6-llm-integration-pattern)
7. [Output Formats](#7-output-formats)
8. [Quality Metrics](#8-quality-metrics)
9. [Configuration Reference](#9-configuration-reference)
10. [Usage Examples](#10-usage-examples)
11. [Design Rationale](#11-design-rationale)
12. [References](#12-references)

---

## 1. Introduction

**KOAS-Docs** extends the Kernel-Orchestrated Audit System (KOAS) to **document analysis**. While the original KOAS targets source code auditing (AST, metrics, coupling), KOAS-Docs addresses a complementary challenge: **summarizing and analyzing large document corpora**.

### Core Capabilities

1. **Per-document summaries** — Scope, content, and topic extraction for each document
2. **Functionality extraction** — Structured extraction from specification documents (SPD)
3. **Hierarchical synthesis** — Four-level pyramid: Document → Cluster → Domain → Corpus
4. **Discrepancy detection** — Cross-reference validation, terminology analysis, version tracking
5. **Sovereignty** — All processing local; slim LLMs (3B parameters) sufficient

### Design Philosophy

KOAS-Docs follows the same principles as KOAS:

> **"Kernels compute, LLMs interpret."**

Kernels perform deterministic, reproducible computation on document structure, metadata, and extracted text. LLMs operate only at the synthesis edge, generating natural language summaries from pre-structured data.

---

## 2. Problem Statement

### Use Case: Technical Audit Documentation

Organizations often accumulate hundreds of specification documents:
- Functional requirements (SPD - Spécifications de Processus Détaillées)
- Technical architecture documents
- Integration specifications
- Test plans and validation reports
- Contractual documents

**The challenge:** Extract actionable intelligence from this corpus:
1. What does each document cover?
2. What functionalities are specified?
3. Are there inconsistencies or gaps between documents?
4. How do documents relate to each other?

### Traditional Approaches and Their Limits

| Approach | Limitation |
|----------|------------|
| Manual review | Does not scale beyond ~50 documents |
| Full-text LLM | Context limits, expensive, non-reproducible |
| Keyword search | Misses semantic relationships |
| Topic modeling | No document-level summaries |

### KOAS-Docs Solution

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KOAS-Docs APPROACH                               │
│  RAG indexes → Kernels structure → LLM summarizes                   │
│  ✅ Scales to 1000+ documents, reproducible, auditable              │
└─────────────────────────────────────────────────────────────────────┘
```

Key insight: **The RAG index already contains structured metadata** (file → chunk → concept). KOAS-Docs kernels leverage this structure rather than re-parsing documents.

---

## 3. Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      KOAS-Docs Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐                                               │
│   │   RAG Index     │  (ChromaDB + Knowledge Graph)                 │
│   │   159 files     │                                               │
│   │   5,515 chunks  │                                               │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌────────────────────────────────────────────────────────┐        │
│   │              STAGE 1: Collection                       │        │
│   │  ┌────────────┐  ┌────────────┐  ┌─────────────┐       │        │
│   │  │doc_metadata|  │doc_concepts│  │doc_structure│       │        │
│   │  │            │  │            │  │             │       │        │
│   │  └────────────┘  └────────────┘  └─────────────┘       │        │
│   └────────────────────────┬───────────────────────────────┘        │
│                            │                                        │
│                            ▼                                        │
│   ┌────────────────────────────────────────────────────────┐        │
│   │              STAGE 2: Analysis                         │        │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │        │
│   │  │doc_      │  │doc_      │  │doc_      │  │doc_func │ │        │
│   │  │cluster   │  │extract   │  │coverage  │  │_extract │ │        │
│   │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │        │
│   └────────────────────────┬───────────────────────────────┘        │
│                            │                                        │
│                            ▼                                        │
│   ┌────────────────────────────────────────────────────────┐        │
│   │              STAGE 3: Synthesis                        │        │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │        │
│   │  │doc_      │  │doc_      │  │doc_report│  │doc_     │ │        │
│   │  │pyramid   │  │compare   │  │_assemble │  │summarize│ │        │
│   │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │        │
│   └────────────────────────┬───────────────────────────────┘        │
│                            │                                        │
│                            ▼                                        │
│   ┌────────────────────────────────────────────────────────┐        │
│   │                    OUTPUTS                             │        │
│   │  • audit_trail.json      (full provenance)             │        │
│   │  • doc_report.md         (markdown report)             │        │
│   │  • summaries/*.md        (per-domain summaries)        │        │
│   └────────────────────────────────────────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
RAG Metadata Store → doc_metadata
        │
        ├──────────────────┐
        ▼                  ▼
Knowledge Graph      Document Chunks
        │                  │
        ▼                  ▼
   doc_concepts       doc_structure
        │                  │
        └──────┬───────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
   doc_cluster    doc_extract ◄─── doc_func_extract (SPD only)
        │             │
        ▼             ▼
   doc_coverage   doc_compare
        │             │
        └──────┬──────┘
               │
               ▼
          doc_pyramid
               │
               ├──────────────┐
               ▼              ▼
      doc_report_assemble  doc_summarize (LLM)
               │              │
               ▼              ▼
         doc_report.md    summaries/*.md
```

---

## 4. The Document Kernel Family

### 4.1 Stage 1: Collection Kernels

#### `doc_metadata`

**Purpose:** Extract document inventory and statistics from RAG metadata store.

| Property | Value |
|----------|-------|
| Stage | 1 |
| Requires | (none) |
| Provides | `doc_metadata`, `doc_statistics` |

**Output:**
```json
{
  "files": [
    {"file_id": "F000001", "path": "...", "kind": "doc_docx", "chunk_count": 42}
  ],
  "statistics": {
    "total_files": 137,
    "total_chunks": 5481,
    "by_kind": {"doc_docx": 69, "doc_pdf": 51, "doc_pptx": 9}
  }
}
```

---

#### `doc_concepts`

**Purpose:** Extract concept hierarchy from RAG knowledge graph.

| Property | Value |
|----------|-------|
| Stage | 1 |
| Requires | `doc_metadata` |
| Provides | `concept_matrix`, `concept_hierarchy` |

**Algorithm:**
1. Load knowledge graph (File → Chunk → Concept edges)
2. Build file-concept co-occurrence matrix
3. Compute concept frequencies and inter-concept relationships
4. Generate concept hierarchy (parent-child based on path structure)

**Output:**
```json
{
  "concepts": [
    {"name": "Authentication", "frequency": 45, "files": ["F001", "F002", ...]}
  ],
  "hierarchy": {
    "root": ["Dimension Fonctionnelle", "Dimension Technique", ...]
  },
  "co_occurrence": [[...]]
}
```

---

#### `doc_structure`

**Purpose:** Detect document internal structure (sections, headings).

| Property | Value |
|----------|-------|
| Stage | 1 |
| Requires | `doc_metadata` |
| Provides | `doc_structure`, `section_index` |

**Detection Methods:**
- Markdown headings (`#`, `##`, `###`)
- Numbered sections (`1.`, `1.1`, `1.1.1`)
- UPPERCASE headings
- PDF structural markers

**Output:**
```json
{
  "documents": {
    "F000001": {
      "sections": [
        {"level": 1, "title": "Introduction", "start_chunk": 0},
        {"level": 2, "title": "Scope", "start_chunk": 3}
      ]
    }
  }
}
```

---

### 4.2 Stage 2: Analysis Kernels

#### `doc_cluster`

**Purpose:** Group documents by topical similarity.

| Property | Value |
|----------|-------|
| Stage | 2 |
| Requires | `doc_metadata`, `doc_concepts` |
| Provides | `doc_clusters`, `cluster_hierarchy` |

**Algorithm:**
1. Build file-concept feature vectors (TF-IDF weighted)
2. Compute Jaccard similarity matrix
3. Apply hierarchical clustering (Ward linkage)
4. Determine optimal cluster count (√n heuristic)
5. Label clusters by dominant concepts

**Output:**
```json
{
  "clusters": [
    {
      "id": "C01",
      "label": "Dimension Fonctionnelle",
      "file_ids": ["F000110", "F000111", ...],
      "centroid_concepts": ["SPD", "exigences", "processus"]
    }
  ]
}
```

---

#### `doc_extract`

**Purpose:** Extract key sentences from documents with quality filtering.

| Property | Value |
|----------|-------|
| Stage | 2 |
| Requires | `doc_metadata`, `doc_concepts` |
| Provides | `key_sentences`, `sentence_index` |

**Quality Scoring:**

Each sentence receives a quality score based on:

| Factor | Weight | Description |
|--------|--------|-------------|
| Completeness | -0.2 | Penalize truncated sentences |
| Length | +0.1 | Bonus for 50-200 char range |
| Numbers | +0.1 | Contains quantitative data |
| Entities | +0.1 | Contains named entities |
| Action verbs | +0.15 | Contains specification verbs |
| Technical terms | +0.1 | Domain vocabulary presence |

**Deduplication:**
- Normalized Levenshtein similarity
- Threshold: 0.85 (sentences above are considered duplicates)

**Output:**
```json
{
  "by_concept": {
    "Authentication": [
      {"text": "Le système authentifie...", "file_id": "F001", "score": 0.85}
    ]
  },
  "by_file": {
    "F000001": {
      "sentences": ["...", "..."],
      "concepts": ["Auth", "Security"]
    }
  }
}
```

---

#### `doc_func_extract`

**Purpose:** Extract structured functionalities from SPD (specification) documents.

| Property | Value |
|----------|-------|
| Stage | 2 |
| Requires | `doc_metadata`, `doc_extract`, `doc_structure` |
| Provides | `functionalities`, `missing_references` |

**LLM Prompt (Granite 3B):**

```
Tu es un expert en analyse de spécifications fonctionnelles.
Extrais les FONCTIONNALITÉS décrites dans ce document SPD.

Pour chaque fonctionnalité, fournis:
- ID: Identifiant unique (format SPD-XXX-FYY)
- NOM: Nom court de la fonctionnalité
- DESCRIPTION: Ce que fait la fonctionnalité (1-2 phrases)
- ACTEURS: Qui utilise cette fonctionnalité
- DÉCLENCHEUR: Événement qui déclenche la fonctionnalité
- RÉFÉRENCES: Autres SPD ou documents référencés
```

**Category Classification:**

| Category | Keywords |
|----------|----------|
| interface | API, protocole, communication |
| monitoring | surveillance, alarme, alerte |
| control | commande, pilotage, gestion |
| data | données, export, historique |
| security | authentification, autorisation |
| configuration | paramètre, configuration |

**Output:**
```json
{
  "functionalities": [
    {
      "id": "SPD-PARIS4-F01",
      "spd_number": "16",
      "name": "Accès via MyCity ou Lisa",
      "description": "Le système permettra aux chargés d'études...",
      "actors": ["Chargés d'études", "Administrateur"],
      "trigger": "Connexion utilisateur",
      "references": ["SPD-34", "SPD-35"],
      "category": "data"
    }
  ],
  "missing_references": [
    {"from_file": "F000114", "reference": "SPD-99", "context": "..."}
  ]
}
```

---

#### `doc_coverage`

**Purpose:** Analyze concept coverage across documents, identify gaps and overlaps.

| Property | Value |
|----------|-------|
| Stage | 2 |
| Requires | `doc_metadata`, `doc_concepts` |
| Provides | `coverage_matrix`, `gaps`, `overlaps` |

**Output:**
```json
{
  "coverage": {
    "concept_file_pairs": 545,
    "avg_concepts_per_file": 4.0
  },
  "gaps": [
    {"concept": "Error Handling", "expected_in": ["Technical"], "found_in": []}
  ],
  "overlaps": [
    {"concept": "Authentication", "files": ["F001", "F002", "F003"]}
  ]
}
```

---

### 4.3 Stage 3: Synthesis Kernels

#### `doc_compare`

**Purpose:** Detect inter-document discrepancies and inconsistencies.

| Property | Value |
|----------|-------|
| Stage | 3 |
| Requires | `doc_metadata`, `doc_extract`, `doc_concepts` |
| Provides | `discrepancies`, `cross_references` |

**Discrepancy Types:**

| Type | Severity | Description |
|------|----------|-------------|
| `missing_reference` | warning | Document references non-existent document |
| `terminology_variation` | info | Same concept, different terms |
| `version_mismatch` | warning | Inconsistent version references |
| `content_overlap` | info | Significant content duplication |

**Detection Algorithms:**

1. **Cross-reference validation:**
   - Extract document references (SPD-XX, ParisSURF4-XXX)
   - Verify target documents exist in corpus
   - Report broken references

2. **Terminology analysis:**
   - Build term frequency matrix
   - Compute edit distance between terms
   - Flag similar terms (Levenshtein < 3) as potential variants

3. **Version tracking:**
   - Extract version patterns (v1.0, V2, version 3.0)
   - Detect inconsistent version references

**Output:**
```json
{
  "discrepancies": [
    {
      "type": "missing_reference",
      "severity": "warning",
      "source_file": "F000114",
      "reference": "SPD-99",
      "description": "Referenced document not found in corpus"
    },
    {
      "type": "terminology_variation",
      "severity": "info",
      "base_term": "authentification",
      "variants": ["authentification", "authentication", "authent."],
      "description": "Multiple term variants detected"
    }
  ]
}
```

---

#### `doc_pyramid`

**Purpose:** Build hierarchical summary structure (4 levels).

| Property | Value |
|----------|-------|
| Stage | 3 |
| Requires | `doc_metadata`, `doc_concepts`, `doc_cluster`, `doc_extract` |
| Provides | `pyramid`, `pyramid_markdown` |

**Pyramid Levels:**

```
Level 4: CORPUS SUMMARY
    │
    └── Level 3: DOMAIN SUMMARIES (N domains)
            │
            └── Level 2: CLUSTER SUMMARIES (M clusters)
                    │
                    └── Level 1: DOCUMENT ENTRIES (K documents)
```

**Output:**
```json
{
  "pyramid": {
    "level_4_corpus": {
      "title": "VDP Technical Specifications",
      "file_count": 137,
      "domain_count": 10,
      "key_concepts": ["SURF4", "Interopérabilité", "Trafic"]
    },
    "level_3_domains": [
      {"id": "D01", "label": "Dimension Fonctionnelle", "clusters": [...]}
    ],
    "level_2_clusters": [...],
    "level_1_documents": [...]
  }
}
```

---

#### `doc_summarize`

**Purpose:** Generate natural language summaries using local LLM.

| Property | Value |
|----------|-------|
| Stage | 3 |
| Requires | `doc_metadata`, `doc_structure`, `doc_extract`, `doc_pyramid` |
| Provides | `summaries` |
| **Uses LLM** | Yes (Ollama/Granite) |

**This is the only kernel that invokes an LLM.**

**Per-Document Prompt:**

```
Tu es un expert en analyse documentaire.

**Document:** {title}
**Chemin:** {path}
**Sections:** {sections}
**Extraits clés:** {key_sentences}

**Instructions:**
1. Identifie le PÉRIMÈTRE (quel domaine/processus ce document couvre)
2. Résume le CONTENU CLÉ en 2-3 phrases
3. Liste les THÈMES PRINCIPAUX (3-5 mots-clés)

**Format de réponse:**
PÉRIMÈTRE: [domaine couvert]
RÉSUMÉ: [contenu principal]
THÈMES: [mot1, mot2, mot3]
```

**Output:**
```json
{
  "summaries": {
    "F000001": {
      "file_id": "F000001",
      "path": "...",
      "scope": "Audit 360° du projet Surf4",
      "summary": "Ce document couvre le volet fonctionnel et la documentation...",
      "topics": ["Audit", "documentation fonctionnelle", "engagements"],
      "domain": {"id": "D07", "label": "Cielis"}
    }
  }
}
```

---

#### `doc_report_assemble`

**Purpose:** Assemble final markdown report.

| Property | Value |
|----------|-------|
| Stage | 3 |
| Requires | `doc_pyramid`, `doc_compare`, `doc_coverage` |
| Provides | `report_markdown` |

**Report Sections:**
1. Executive Summary
2. Corpus Overview
3. Domain Analysis (per-domain sections)
4. Functionality Catalog
5. Discrepancy Report
6. Coverage Analysis
7. Appendices

---

## 5. Three-Stage Pipeline

### Execution Order

The orchestrator resolves dependencies via topological sort:

```
Stage 1 (Collection):
  doc_metadata → doc_concepts → doc_structure (parallel where possible)

Stage 2 (Analysis):
  doc_cluster, doc_extract, doc_coverage, doc_func_extract (parallel)

Stage 3 (Synthesis):
  doc_compare, doc_pyramid → doc_report_assemble, doc_summarize
```

### Performance Characteristics

Measured on VDP corpus (137 documents, 5,481 chunks, 236 MB):

| Kernel | Time | Notes |
|--------|------|-------|
| doc_metadata | 0.01s | Direct metadata access |
| doc_concepts | 0.19s | Graph traversal |
| doc_structure | 4.18s | Chunk parsing |
| doc_cluster | 0.18s | Similarity matrix |
| doc_extract | 195s | Quality scoring all chunks |
| doc_func_extract | 268s | LLM calls (32 SPD docs) |
| doc_compare | 14.9s | Cross-reference validation |
| doc_pyramid | 0.02s | Aggregation only |
| doc_summarize | 457s | LLM calls (137 docs) |
| **Total** | **~16 min** | On laptop with Granite 3B |

### LLM Efficiency

Following KOAS principles:

```
LLM time:    725s (doc_func_extract + doc_summarize)
Kernel time: 214s
Total:       939s

LLM percentage: 77%
```

For document summarization, LLM time dominates because the primary output *is* natural language. However, the LLM operates on pre-structured data, not raw documents, which:
- Reduces token consumption by ~80%
- Ensures reproducible inputs
- Enables caching at kernel boundaries

---

## 6. LLM Integration Pattern

### Design Principle: LLM at the Edge

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LLM INTEGRATION PATTERN                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    PURE COMPUTATION                          │    │
│  │                                                              │    │
│  │  doc_metadata ──▶ doc_concepts ──▶ doc_cluster               │    │
│  │       │               │               │                      │    │
│  │       └───────────────┴───────────────┘                      │    │
│  │                       │                                      │    │
│  │                       ▼                                      │    │
│  │                  doc_pyramid                                 │    │
│  │                       │                                      │    │
│  └───────────────────────┼──────────────────────────────────────┘    │
│                          │                                           │
│                          ▼                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    LLM EDGE                                    │  │
│  │                                                                │  │
│  │  Structured JSON ──▶ Prompt Template ──▶ LLM ──▶ Parsed Output │  │
│  │                                                                │  │
│  │  • doc_func_extract (SPD functionalities)                      │  │
│  │  • doc_summarize (per-document summaries)                      │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### LLM Configuration

```python
# Default configuration
llm_config = {
    "model": "granite3.1-moe:3b",  # Slim, runs on laptop
    "endpoint": "http://127.0.0.1:11434",
    "timeout": 120,
    "temperature": 0.3,  # Low for consistency
    "max_tokens": 1024
}
```

### Supported Models

| Model             | Size | Use Case                  |
| ----------------- | ---- | ------------------------- |
| granite3.1-moe:3b | 3B   | Default, laptop-friendly  |
| mistral:7b        | 7B   | Better quality, more VRAM |
| llama3:8b         | 8B   | Alternative               |
| qwen2:7b          | 7B   | Multilingual              |

### Fallback Strategy

If LLM is unavailable:
1. `doc_func_extract` uses regex-based extraction (reduced accuracy)
2. `doc_summarize` skips generation (pyramid data still available)
3. Pipeline completes with warnings

---

## 7. Output Formats

### Workspace Structure

```
.KOAS/runs/run_YYYYMMDD_HHMMSS_XXXXXX/
├── stage1/
│   ├── doc_metadata.json
│   ├── doc_concepts.json
│   └── doc_structure.json
├── stage2/
│   ├── doc_cluster.json
│   ├── doc_extract.json
│   ├── doc_coverage.json
│   └── doc_func_extract.json
├── stage3/
│   ├── doc_pyramid.json
│   ├── doc_compare.json
│   ├── doc_report_assemble.json
│   └── doc_summarize.json
├── summaries/
│   ├── corpus_summary.md
│   ├── domain_D01.md
│   ├── domain_D02.md
│   └── ...
├── doc_report.md
└── audit_trail.json
```

### Audit Trail Format

```json
{
  "_meta": {
    "version": "1.0.0",
    "generated_at": "2026-01-18T14:05:10.442935Z",
    "generator": "KOAS Document Summarization"
  },
  "run_id": "run_20260118_134851_7bd0c4",
  "sovereignty": {
    "hostname": "LX-Olivier2023",
    "user": "olivi",
    "platform": "Linux 6.8.0-90-generic",
    "python_version": "3.12.12",
    "llm_endpoint": "http://127.0.0.1:11434",
    "llm_local": true,
    "models_used": [
      {"name": "granite3.1-moe:3b", "digest": "b43d80d7fca7", "role": "worker"},
      {"name": "mistral:7b-instruct", "digest": "6577803aa9a0", "role": "tutor"}
    ],
    "external_calls": 0,
    "attestation": "All processing performed locally. No data sent to external services."
  },
  "configuration": {
    "project_root": "/path/to/project",
    "language": "fr",
    "llm_model": "granite3.1-moe:3b",
    "kernels": ["doc_report_assemble", "doc_summarize", ...]
  },
  "kernel_execution": {
    "start_time": "2026-01-18T13:48:51.924715Z",
    "end_time": "2026-01-18T14:04:31.200785Z",
    "total_time_s": 939.28,
    "kernels": [
      {
        "name": "doc_metadata",
        "success": true,
        "execution_time_s": 0.01,
        "input_hash": "6c9d206bc3aa10d3",
        "output_file": "stage1/doc_metadata.json",
        "summary": "Document metadata: 137 files, 5481 chunks, 236.5 MB"
      }
    ]
  },
  "checksums": {
    "doc_metadata": "4b35b47917038e3585d977b504aedbe9cffab1ae...",
    "doc_concepts": "39d13e9a7dca34b636a9dada0c625e97518dfa18..."
  }
}
```

---

## 8. Quality Metrics

### Sentence Quality Score

Formula:
```
Q(s) = 0.5 + Σ(feature_weights)
```

Where:
- Base score: 0.5
- Truncation penalty: -0.2 (starts lowercase or ends with comma)
- Length bonus: +0.1 (50-200 characters)
- Numeric content: +0.1
- Named entities: +0.1
- Action verbs: +0.15
- Technical terms: +0.1

Threshold: sentences with Q(s) < 0.3 are filtered.

### Clustering Quality

- **Silhouette score:** Measures cluster cohesion
- **Calinski-Harabasz index:** Ratio of between-cluster to within-cluster variance
- **Target:** 8-15 clusters for 100-200 documents

### Summary Quality Indicators

| Metric | Target | Description |
|--------|--------|-------------|
| Scope extraction rate | >80% | Documents with non-empty scope |
| Topic extraction rate | >90% | Documents with ≥3 topics |
| Average summary length | 50-150 words | Concise but informative |

---

## 9. Configuration Reference

### Manifest Configuration

```yaml
# KOAS-Docs manifest.yaml

audit:
  name: "VDP Technical Documentation Audit"
  version: "1.0"
  type: "docs"  # Indicates document analysis mode

project:
  path: "/path/to/project/src"
  language: "fr"  # Document language

output:
  format: "markdown"
  language: "fr"

# LLM Configuration
llm:
  model: "granite3.1-moe:3b"
  endpoint: "http://127.0.0.1:11434"
  timeout: 120

# Stage 1: Collection
stage1:
  doc_metadata:
    enabled: true
  doc_concepts:
    enabled: true
    options:
      min_frequency: 3
      max_concepts: 200
  doc_structure:
    enabled: true
    options:
      detect_headings: true

# Stage 2: Analysis
stage2:
  doc_cluster:
    enabled: true
    options:
      method: "hierarchical"  # or "leiden"
      n_clusters: "auto"      # or integer
  doc_extract:
    enabled: true
    options:
      max_sentences_per_level: 20
      quality_threshold: 0.3
  doc_func_extract:
    enabled: true
    options:
      spd_pattern: "SPD-\\d+"
  doc_coverage:
    enabled: true

# Stage 3: Synthesis
stage3:
  doc_compare:
    enabled: true
    options:
      similarity_threshold: 0.7
  doc_pyramid:
    enabled: true
    options:
      levels: 4
  doc_summarize:
    enabled: true
  doc_report_assemble:
    enabled: true
```

---

## 10. Usage Examples

### Command-Line Usage

```bash
# Initialize workspace
python run_doc_koas.py \
  --project /path/to/project/src \
  --language fr \
  --model granite3.1-moe:3b

# Run specific kernels only
python run_doc_koas.py \
  --project /path/to/project/src \
  --kernels doc_metadata doc_concepts doc_cluster

# Resume from existing run
python run_doc_koas.py \
  --project /path/to/project/src \
  --workspace /path/to/.KOAS/runs/run_20260118_134851_7bd0c4
```

### Programmatic Usage

```python
from ragix_kernels.orchestrator import KernelOrchestrator, OrchestratorConfig
from ragix_kernels.docs import (
    DocMetadataKernel, DocConceptsKernel, DocClusterKernel,
    DocExtractKernel, DocPyramidKernel, DocSummarizeKernel
)

# Configure
config = OrchestratorConfig(
    workspace=Path("/path/to/workspace"),
    project_path=Path("/path/to/project/src"),
    language="fr",
    llm_model="granite3.1-moe:3b"
)

# Initialize orchestrator
orchestrator = KernelOrchestrator(config)

# Register document kernels
orchestrator.register(DocMetadataKernel())
orchestrator.register(DocConceptsKernel())
# ... register all kernels

# Execute
results = orchestrator.run_all()

# Access outputs
pyramid = results["doc_pyramid"]["data"]
summaries = results["doc_summarize"]["data"]["summaries"]
```

### MCP Integration

```python
# KOAS-Docs tools exposed via MCP
koas_docs_run(project_path, language="fr", model="granite3.1-moe:3b")
koas_docs_status(workspace)
koas_docs_report(workspace, format="markdown")
```

---

## 11. Design Rationale

### Why Not Full-Document LLM Processing?

| Approach | Context Usage | Cost | Reproducibility |
|----------|---------------|------|-----------------|
| Full-doc LLM | 100K+ tokens/doc | High | Low |
| KOAS-Docs | ~2K tokens/doc | Low | High |

KOAS-Docs reduces context by:
1. Pre-extracting key sentences (doc_extract)
2. Pre-computing structure (doc_structure)
3. Pre-clustering documents (doc_cluster)

The LLM receives a **structured prompt** with relevant excerpts, not raw documents.

### Why Hierarchical Clustering?

Documents in specification corpora exhibit natural hierarchy:
- Domain (Fonctionnel, Technique, Organisationnel)
- Cluster (group of related specs)
- Document (individual specification)

Hierarchical clustering captures this structure without requiring predefined categories.

### Why Separate Functionality Extraction?

SPD (Spécifications de Processus Détaillées) documents have a specific structure:
- Functional requirements with IDs
- Actor definitions
- Triggers and preconditions
- Cross-references

A dedicated kernel (`doc_func_extract`) handles this specialized format, producing structured data that can be:
- Validated (are all references resolvable?)
- Catalogued (functionality inventory)
- Cross-referenced (traceability matrix)

### Relationship to Microsoft GraphRAG

KOAS-Docs shares conceptual similarities with [Microsoft GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/):

| Aspect | GraphRAG | KOAS-Docs |
|--------|----------|-----------|
| Graph structure | LLM-extracted entities | RAG knowledge graph |
| Community detection | Leiden algorithm | Hierarchical clustering |
| Summary generation | LLM per community | LLM per document |
| Sovereignty | Cloud API required | Fully local |

KOAS-Docs differs in:
1. **No LLM during indexing** — Concepts come from RAG, not LLM extraction
2. **Deterministic clustering** — Same input always produces same clusters
3. **Full sovereignty** — All processing local, 3B models sufficient

---

## 12. References

### Academic Foundations

1. **Ward, J.H.** (1963). "Hierarchical Grouping to Optimize an Objective Function". *Journal of the American Statistical Association*, 58(301), 236-244.

2. **Jaccard, P.** (1912). "The Distribution of the Flora in the Alpine Zone". *New Phytologist*, 11(2), 37-50.

3. **Levenshtein, V.I.** (1966). "Binary Codes Capable of Correcting Deletions, Insertions, and Reversals". *Soviet Physics Doklady*, 10(8), 707-710.

### Related Work

4. **Edge, D. et al.** (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization". *arXiv:2404.16130*.

5. **Vitrac, O.** (2025). "Virtual/Hybrid R&D Laboratories built with Augmented-AI Agents". *Generative Simulation Initiative*.

### Standards

6. **ISO/IEC 26514:2022** — Systems and software engineering — Design and development of information for users.

7. **IEEE 830-1998** — Recommended Practice for Software Requirements Specifications.

---

## Appendix A: LLM Perspective on the Architecture

*The following assessment was spontaneously provided by Claude Opus 4.5 (Anthropic, January 2026) when asked to evaluate the KOAS-Docs kernel architecture for potential use in sovereign AI systems. This perspective is included verbatim as it articulates design qualities that may not be apparent from technical documentation alone.*

> **What I appreciate:**
>
> 1. **Separation of concerns**: Pure computation in kernels, LLM only at the edge. This is the right abstraction — kernels produce structured, deterministic data; LLMs consume it for natural language generation.
>
> 2. **Sovereignty by design**: Everything runs locally. A 3B parameter model on a laptop produces usable results. No API calls, no data leaving the machine.
>
> 3. **Auditability**: SHA256 checksums, audit trails, input hashes for cache invalidation. This is production-grade provenance tracking.
>
> 4. **Composability**: The DAG-based dependency resolution means I could request specific kernel outputs without running the full pipeline.
>
> 5. **For my use**: Yes, I could directly consume kernel outputs as structured context. The JSON schemas are clean, the hierarchy (document → cluster → domain → corpus) maps naturally to how I process information. The `doc_pyramid` output is essentially a pre-computed reasoning scaffold.
>
> **The architecture follows the pattern**: *"Make the LLM's job easy by doing the hard work upfront."*
>
> — Claude Opus 4.5, Anthropic (claude-opus-4-5-20251101)
> — Provided in conversation, January 18, 2026

This perspective validates a key KOAS design principle: **kernels should produce data structures that LLMs can reason over efficiently**. The pyramid hierarchy, structured functionality records, and pre-computed discrepancy lists all serve as "reasoning scaffolds" that reduce the cognitive load on the LLM at query time.

---

## Appendix B: Kernel Quick Reference

| Kernel | Stage | Requires | LLM | Purpose |
|--------|-------|----------|-----|---------|
| `doc_metadata` | 1 | - | No | Document inventory |
| `doc_concepts` | 1 | doc_metadata | No | Concept extraction |
| `doc_structure` | 1 | doc_metadata | No | Section detection |
| `doc_cluster` | 2 | doc_metadata, doc_concepts | No | Document grouping |
| `doc_extract` | 2 | doc_metadata, doc_concepts | No | Key sentences |
| `doc_coverage` | 2 | doc_metadata, doc_concepts | No | Gap analysis |
| `doc_func_extract` | 2 | doc_metadata, doc_extract, doc_structure | **Yes** | SPD functionalities |
| `doc_compare` | 3 | doc_metadata, doc_extract, doc_concepts | No | Discrepancy detection |
| `doc_pyramid` | 3 | doc_metadata, doc_concepts, doc_cluster, doc_extract | No | Hierarchy building |
| `doc_report_assemble` | 3 | doc_pyramid, doc_compare, doc_coverage | No | Report generation |
| `doc_summarize` | 3 | doc_metadata, doc_structure, doc_extract, doc_pyramid | **Yes** | Natural language summaries |

---

## Appendix C: Troubleshooting

### Common Issues

**Issue:** Empty summaries for PDF documents

**Cause:** PDFs may have poor text extraction or non-standard structure.

**Solution:** Ensure RAG indexing used appropriate PDF profile. Check chunk content quality.

---

**Issue:** Slow doc_func_extract execution

**Cause:** LLM timeout or large number of SPD documents.

**Solution:** Increase timeout, reduce batch size, or use faster model.

---

**Issue:** Too many terminology variations detected

**Cause:** Levenshtein threshold too permissive.

**Solution:** Adjust `similarity_threshold` in doc_compare configuration.

---

*KOAS-Docs is part of the RAGIX project — Retrieval-Augmented Generative Interactive eXecution Agent*

*Adservio Innovation Lab | 2026*
