# KOAS-Docs Roadmap — v1.3

**Author:** Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
**Date:** 2026-01-18
**Status:** Active Development

---

## Executive Summary

KOAS-Docs v1.2 successfully implemented dual clustering (Hierarchical + Leiden) with tutor-verified LLM summaries. This roadmap addresses identified improvements from the VDP audit run to enhance report quality, visualization, and performance.

---

## 1. Report Quality Improvements

### 1.1 Enhanced Chunk Context (Priority: HIGH)

**Current Issue:** Chunks and labels are too short, lacking context for understanding.

**Solution:** Implement `grep -A -B` style context extraction with:
- Configurable context lines (default: 2-3 lines before/after)
- Ellipsis (`...`) for truncated content
- **Highlighting** of the central chunk in the middle
- Smart truncation that preserves semantic boundaries

**Implementation:**
```python
# In doc_extract.py
def extract_with_context(chunk: str, context_before: int = 3, context_after: int = 3) -> str:
    """Extract chunk with surrounding context, highlighted."""
    lines = chunk.split('\n')
    # Add ellipsis and highlight markers
    # ... (preceding context)
    # >>> CHUNK CONTENT <<<
    # ... (following context)
```

**Files to modify:**
- `ragix_kernels/docs/doc_extract.py`
- `ragix_kernels/docs/doc_final_report.py`
- `ragix_core/rag_project.py` (chunk retrieval with context)

**Note:** Default RAG indexing parameters for pure text may need adjustment in `ragix-web` for better chunk boundaries.

---

### 1.2 Hierarchical Leiden Clustering (Priority: HIGH)

**Current Issue:** Leiden produces flat clustering (4 communities at single resolution). Expected: hierarchical tree structure.

**Solution:** Multi-resolution Leiden with tree construction:
1. Run Leiden at multiple resolutions: [0.1, 0.3, 0.5, 0.7, 1.0]
2. Build hierarchical tree from resolution levels
3. Detect natural community nesting (parent-child relationships)
4. Output tree structure compatible with visualization

**Implementation:**
```python
# In doc_cluster_leiden.py
def build_hierarchical_tree(communities: Dict[float, List]) -> Dict:
    """Build tree from multi-resolution Leiden communities."""
    tree = {"root": {"children": []}}
    # Map communities across resolutions
    # Identify containment relationships
    # Build parent-child tree
    return tree
```

**Output format:**
```json
{
  "tree": {
    "level_0": {"L0.1_C00": {"children": ["L0.5_C00", "L0.5_C01"]}},
    "level_1": {"L0.5_C00": {"children": ["L1.0_C00", "L1.0_C01"]}},
    ...
  }
}
```

**Files to modify:**
- `ragix_kernels/docs/doc_cluster_leiden.py`
- `ragix_kernels/docs/doc_final_report.py` (tree rendering)

---

### 1.3 Semantic Filtering for Terminology Detection (Priority: HIGH) ✅ DONE

**Previous Issue:** False positives in terminology variation detection due to:
- Edit distance matching without semantic understanding
- Cross-language confusion (French/English)
- Word fragments and noise
- Different words with similar spelling (outils ≠ utiles)

**Solution Implemented:**
1. **French lemmatizer** - Rule-based lemmatization (verb conjugations, plurals, gender)
2. **Semantic exclusion list** - Explicit pairs that should NOT be grouped
3. **Fragment filtering** - Filter out noise and incomplete words
4. **Dual validation** - Both spelling similarity AND semantic relatedness required

**False Positives Now Blocked:**
```
outils ≠ utiles (tools ≠ useful)
rapports ≠ apportés (reports ≠ brought)
export ≠ report (different words)
échelle ≠ elle (scale ≠ she)
auteur ≠ acteur (author ≠ actor)
notion ≠ option (different meanings)
interne ≠ internet (internal ≠ internet)
```

**Files modified:**
- `ragix_kernels/docs/doc_compare.py` (added semantic filtering functions)

---

### 1.4 Source Citations in Discrepancy Analysis (Priority: MEDIUM)

**Current Issue:** Discrepancies list issues without citing source documents.

**Solution:** Include explicit source references:
```markdown
#### Terminology Variation (54)
- Multiple term variants: contents, contient, contenus
  - Source: `F000012` (ParisSURF4-SPD-01.docx:L45), `F000034` (Plan_Formation.docx:L120)
```

**Files to modify:**
- `ragix_kernels/docs/doc_compare.py`
- `ragix_kernels/docs/doc_final_report.py`

---

## 2. Visualization & Graphs (Priority: HIGH)

### 2.1 Graph Generation Framework

**Rationale:** KOAS is deterministic → reproducible graphs are essential for analysis.

**Supported formats:**
1. **Mermaid** (inline in Markdown, lightweight)
2. **Python/Matplotlib** (PNG export, publication quality)
3. **Interactive HTML** (optional, for web viewing)

**Graphs to implement:**

| Graph | Type | Location |
|-------|------|----------|
| Document type distribution | Pie/Bar | Corpus Overview |
| Clustering dendrogram | Tree | Clustering Analysis |
| Leiden community hierarchy | Nested circles | Clustering Analysis |
| Concept co-occurrence | Network | Domain Summaries |
| Processing timeline | Gantt | Technical Information |
| LLM call distribution | Bar | Technical Information |

**Implementation:**
```python
# New file: ragix_kernels/docs/doc_visualize.py
class DocVisualizeKernel(Kernel):
    """Generate visualizations for KOAS reports."""

    def generate_mermaid_tree(self, clusters: Dict) -> str:
        """Generate Mermaid tree diagram."""

    def generate_pyplot_charts(self, data: Dict) -> List[Path]:
        """Generate matplotlib charts, save as PNG."""
```

---

### 2.2 Word Clouds for Appendices (Priority: MEDIUM)

**Purpose:** Visual summary of key terms per domain/appendix.

**Implementation:**
- Use `wordcloud` library (pure Python, no external deps)
- Generate per-domain word clouds
- Include in appendices B, C, D

```python
from wordcloud import WordCloud

def generate_word_cloud(concepts: List[str], output_path: Path) -> Path:
    """Generate word cloud from concept list."""
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate(' '.join(concepts))
    wc.to_file(output_path)
    return output_path
```

**Files to modify:**
- `ragix_kernels/docs/doc_final_report.py`
- New: `ragix_kernels/docs/doc_visualize.py`

---

## 3. LLM Performance Optimization (Priority: HIGH)

### 3.1 Batch LLM Calls by Model

**Current Issue:** Alternating Granite↔Mistral calls cause GPU unload/reload overhead (~30s per switch).

**Solution:** Reorganize calls to batch by model:
1. Collect all Worker (Granite) prompts
2. Execute all Granite calls in batch
3. Collect all Tutor (Mistral) prompts
4. Execute all Mistral calls in batch

**Expected improvement:** ~40-50% reduction in stage 3 time.

**Implementation:**
```python
# In doc_summarize_tutored.py
def compute(self, input: KernelInput) -> Dict:
    # Phase 1: Batch all worker calls
    worker_prompts = self._collect_worker_prompts(documents)
    worker_responses = self._batch_llm_call(worker_prompts, model="granite")

    # Phase 2: Batch all tutor calls
    tutor_prompts = self._collect_tutor_prompts(worker_responses)
    tutor_responses = self._batch_llm_call(tutor_prompts, model="mistral")

    # Phase 3: Process results
    return self._merge_results(worker_responses, tutor_responses)
```

**Files to modify:**
- `ragix_kernels/docs/doc_summarize_tutored.py`
- `ragix_kernels/docs/doc_func_extract.py`
- `ragix_kernels/cache.py` (batch cache lookup)

---

### 3.2 Tutor Role Clarification

**Current Issue:** Tutor repeats crude analysis instead of synthesizing findings.

**Solution:** Redefine tutor prompts to focus on:
- **Validation**: Is the worker summary accurate?
- **Synthesis**: Consolidate multiple worker outputs
- **Correction**: Fix factual errors, not regenerate

**Updated prompt template:**
```
You are a TUTOR verifying a summary generated by a worker model.

WORKER SUMMARY:
{worker_summary}

SOURCE CHUNKS:
{source_chunks}

TASK:
1. Verify accuracy (does summary match source?)
2. If accurate: respond "APPROVED"
3. If inaccurate: provide ONLY the corrected parts, not a full rewrite
```

---

### 3.3 Enhanced Caching with Tolerance

**Current Issue:** Cache hit rate 0% — identical requests not cached.

**Root causes:**
1. Cache path mismatch (`review/cache/` vs `src/.KOAS/cache/`)
2. No fuzzy matching for near-identical prompts

**Solution:**
1. **Unified cache path**: Always use `.KOAS/cache/` relative to workspace
2. **Prompt hashing with normalization**: Normalize whitespace, case before hashing
3. **Tolerance matching**: Accept cache hit if prompt similarity > 95%

```python
# In ragix_kernels/cache.py
def get_cache_key(prompt: str, model: str) -> str:
    """Generate cache key with normalization."""
    normalized = ' '.join(prompt.lower().split())  # Normalize whitespace
    return hashlib.sha256(f"{model}:{normalized}".encode()).hexdigest()[:16]

def fuzzy_cache_lookup(prompt: str, model: str, tolerance: float = 0.95) -> Optional[str]:
    """Lookup with fuzzy matching for near-identical prompts."""
```

**Files to modify:**
- `ragix_kernels/cache.py`
- `ragix_kernels/run_doc_koas.py`

---

## 4. Artifact Management (Priority: HIGH)

### 4.1 Fix Output Location

**Current Issue:** Artifacts written to `src/` (source folder), polluting RAG index.

**Solution:** Always write to `.KOAS/runs/<run_id>/`:
```
src/
├── Bearing Point/     # Source documents (preserved)
├── Cielis/            # Source documents (preserved)
├── .KOAS/
│   ├── cache/         # LLM response cache
│   └── runs/
│       └── run_YYYYMMDD_HHMMSS_xxxxxx/
│           ├── manifest.yaml
│           ├── final_report.md
│           ├── stage1/
│           ├── stage2/
│           ├── stage3/
│           ├── appendices/
│           ├── assets/        # Generated graphs
│           └── logs/
└── .RAG/              # RAG index (preserved)
```

**Files to modify:**
- `ragix_kernels/run_doc_koas.py` (workspace handling)
- `ragix_kernels/orchestrator.py` (output paths)

---

### 4.2 Symlink Latest Run

**Feature:** Create/update `latest` symlink for easy access:
```bash
src/.KOAS/latest -> runs/run_20260118_192611_success/
```

**Files to modify:**
- `ragix_kernels/run_doc_koas.py`

---

## 5. Report Metrics Enhancement (Priority: MEDIUM)

### 5.1 Detailed LLM Statistics

**Current:** Basic stats (cache hits, summaries accepted).

**Enhanced:**
```markdown
### LLM Processing Statistics

| Model | Role | Requests | Tokens In | Tokens Out | Time (s) | Cache Hits |
|-------|------|----------|-----------|------------|----------|------------|
| granite3.1-moe:3b | worker | 274 | 125,430 | 34,210 | 892.3 | 0 |
| mistral:7b-instruct | tutor | 137 | 89,120 | 12,340 | 1,376.8 | 0 |
| **Total** | | **411** | **214,550** | **46,550** | **2,269.1** | **0** |

### Processing Timeline

| Stage | Kernels | Duration | % Total |
|-------|---------|----------|---------|
| Stage 1 | 15 | 13.1s | 0.5% |
| Stage 2 | 19 | 396.6s | 14.6% |
| Stage 3 | 12 | 2,269.2s | 83.7% |
| **Total** | **46** | **2,711.1s** | **100%** |
```

**Files to modify:**
- `ragix_kernels/docs/doc_summarize_tutored.py` (track stats)
- `ragix_kernels/docs/doc_final_report.py` (render stats)
- `ragix_kernels/orchestrator.py` (aggregate timing)

---

### 5.2 Run ID Tracking

**Current Issue:** Run ID shows "unknown".

**Solution:** Generate and propagate run ID:
```python
run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
```

**Files to modify:**
- `ragix_kernels/run_doc_koas.py`
- `ragix_kernels/orchestrator.py`

---

## 6. Implementation Priority

### Phase 1 (Immediate — v1.3.0)
1. ✅ Fix Leiden field name mismatch (DONE)
2. ✅ Remove resolution 2.0 (DONE)
3. ✅ Fix terminology variation false positives with semantic filtering (DONE)
4. [ ] Fix artifact output location
5. [ ] Fix caching (unified path)
6. [ ] Add Run ID tracking
7. [ ] Add detailed LLM statistics

### Phase 2 (Short-term — v1.4.0)
7. [ ] Batch LLM calls by model
8. [ ] Hierarchical Leiden tree
9. [ ] Source citations in discrepancies
10. [ ] Enhanced chunk context (grep -A -B style)

### Phase 3 (Medium-term — v1.5.0)
11. [ ] Graph generation framework (Mermaid + Matplotlib)
12. [ ] Word clouds for appendices
13. [ ] Tutor role refinement
14. [ ] Fuzzy cache matching

---

## 7. Dependencies

### New Dependencies (Optional)
```
# requirements-koas-docs-viz.txt
wordcloud>=1.9.0      # Word cloud generation
matplotlib>=3.8.0     # Chart generation (likely already present)
```

### Existing Dependencies
```
# requirements-koas-docs.txt (already created)
python-igraph>=1.0.0
leidenalg>=0.10.0
```

---

## 8. Testing Checklist

- [ ] Leiden hierarchical tree on VDP corpus
- [ ] Batch LLM calls performance benchmark
- [ ] Cache hit rate > 80% on re-run
- [ ] Artifact isolation (no pollution of src/)
- [ ] Graph generation (Mermaid + PNG)
- [ ] Word cloud generation
- [ ] Full pipeline re-run < 20 minutes (with cache)

---

## 9. Notes

### What Works Well
- Dual clustering (Hierarchical + Leiden) provides complementary views
- Tutor verification catches hallucinations
- Appendices provide detailed drill-down
- Sovereignty attestation ensures local processing
- Deterministic execution enables reproducibility

### Known Limitations
- Current: ~57 minutes for 137 docs (stage 3 dominated by LLM)
- Expected after optimization: ~15-20 minutes with caching + batching
- Leiden at resolution > 1.0 causes exponential computation on dense graphs

---

*Roadmap maintained as part of RAGIX project — Adservio Innovation Lab*
