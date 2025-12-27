# TODO: Interpreter-Tutor Development Roadmap

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Created:** 2025-12-22
**Status:** Active Development

---

## 1. Multi-LLM Benchmark Campaign

### 1.1 Target Models (Ollama Installed)

| Model | Parameters | Size | Priority |
|-------|------------|------|----------|
| granite3.1-moe:3b | 3B (MoE) | 2.0 GB | High (baseline) |
| llama3.2:3b | 3B | 2.0 GB | High |
| phi3:latest | 3.8B | 2.2 GB | High |
| dolphin-mistral:7b-v2.6-dpo-laser | 7B | 4.1 GB | Medium |
| mistral:7b-instruct | 7B | 4.4 GB | High |
| mistral:latest | 7B | 4.4 GB | Medium (compare with instruct) |
| qwen2.5:7b | 7B | 4.7 GB | Medium |
| qwen2.5-coder:7b | 7B | 4.7 GB | High (code-specialized) |
| llama3:latest | 8B | 4.7 GB | Medium |
| deepseek-r1:14b | 14B | 9.0 GB | High (reasoning-focused) |
| qwen2.5-coder:14b | 14B | 9.0 GB | High (large code model) |

### 1.2 Benchmark Execution Plan

```bash
# Run full benchmark suite across all models
python -m ragix_core.reasoning_tutor.benchmarks.scored_mode \
    --models "granite3.1-moe:3b,llama3.2:3b,phi3:latest,mistral:7b-instruct,qwen2.5-coder:7b,deepseek-r1:14b,qwen2.5-coder:14b" \
    --max-turns 8 \
    --output results/benchmark_$(date +%Y%m%d).json
```

### 1.3 Tasks

- [x] **Structured logging system** for benchmark runs ✓ IMPLEMENTED
  - JSON-Lines format for each turn (`BenchmarkLogger` class)
  - Fields: model, benchmark, turn, action, result, score, latency
  - Aggregate statistics per model/benchmark
  - Usage: `python -m ... --output results/benchmark.jsonl`

- [x] **Results persistence** ✓ IMPLEMENTED
  - Save to `results/` directory with timestamps
  - Include model metadata (parameters, context size, quantization)
  - Export to CSV for external analysis (`extract_features.py`)

- [x] **Batch runner script** ✓ IMPLEMENTED
  - `run_benchmark_campaign.sh` — run all models sequentially
  - Priority mode (7 models) or all mode (11 models)
  - Progress reporting with summary at end

---

## 2. Visualization & Analysis

### 2.1 PCA/PCoA Analysis

**Goal:** Understand what differentiates LLM performance beyond raw parameters.

**Feature Matrix (per model × benchmark):**

| Feature | Description |
|---------|-------------|
| `score` | Total points earned |
| `success_rate` | % of benchmarks passed |
| `avg_turns` | Average turns to solution |
| `syntax_errors` | Count of shell syntax errors |
| `repeated_actions` | Count of repeated commands |
| `card_usage` | How often card deck was used |
| `memory_efficiency` | Re-read ratio in memory benchmark |
| `latency_mean` | Average response time |
| `latency_std` | Response time variance |

**Tasks:**

- [x] **Feature extraction script** (`analysis/extract_features.py`) ✓ IMPLEMENTED
  - Parse JSON-Lines logs
  - Build feature matrix (models × features)
  - Export to CSV with model metadata
  - Usage: `python analysis/extract_features.py results/benchmark.jsonl`

- [x] **PCA visualization** (`analysis/visualize.py`) ✓ IMPLEMENTED
  - 2D scatter with model labels
  - Loadings plot showing feature contributions
  - Explained variance ratio

- [x] **Dendrogram clustering** (`analysis/visualize.py`) ✓ IMPLEMENTED
  - Hierarchical clustering (Ward method)
  - Visual dendrogram output
  - Identify model "families" by behavior

- [x] **Score heatmap** (`analysis/visualize.py`) ✓ IMPLEMENTED
  - Model × Benchmark heatmap
  - Color-coded by score

- [ ] **PCoA on distance matrix** (TODO)
  - Bray-Curtis or Euclidean distance
  - Compare with PCA results

### 2.2 Visualization Outputs

```
results/
├── benchmark_20251222.jsonl       # Raw logs
├── feature_matrix.csv              # Extracted features
├── pca_2d.png                      # 2D PCA plot
├── pca_3d.html                     # Interactive 3D (Plotly)
├── dendrogram.png                  # Hierarchical clustering
├── heatmap_scores.png              # Model × Benchmark heatmap
└── report.md                       # Auto-generated summary
```

### 2.3 Research Questions

1. **Does model size predict performance?**
   - Correlate parameters with score
   - Identify outliers (small but effective, large but poor)

2. **Does specialization matter?**
   - Compare `qwen2.5-coder` vs `qwen2.5` (same base, different training)
   - Compare `mistral:7b-instruct` vs `mistral:latest`

3. **What behaviors cluster together?**
   - Do all 3B models fail similarly?
   - Do code-specialized models excel at different benchmarks?

4. **Is context size the bottleneck?**
   - Correlate context window with memory benchmark performance
   - Test with artificially truncated context

---

## 3. Card Deck Extension System

### 3.1 Current Deck Structure

The card deck acts as a **procedural RAG** — generic solutions that slim LLMs can invoke when they lack specific bash syntax knowledge.

Current cards in `action_menu.py`:
- LIST_FILES(path)
- SEARCH_CONTENT(pattern)
- READ_FILE(path)
- COUNT_LINES(path)
- SEARCH_FILENAME(pattern)
- COUNT_ALL_LINES(path, pattern)

### 3.2 New Domain Decks Needed

| Domain | Cards | Use Case |
|--------|-------|----------|
| **Git Operations** | GIT_STATUS, GIT_DIFF, GIT_LOG, GIT_BLAME | Code audit, version tracking |
| **Python Analysis** | FIND_IMPORTS, FIND_CLASSES, FIND_FUNCTIONS, CHECK_SYNTAX | Code understanding |
| **Data Processing** | READ_CSV, COUNT_ROWS, FILTER_ROWS, AGGREGATE | Data pipeline tasks |
| **Security Audit** | FIND_SECRETS, CHECK_PERMISSIONS, SCAN_PORTS | Security assessment |
| **Documentation** | FIND_DOCSTRINGS, EXTRACT_README, LIST_ENDPOINTS | API documentation |

### 3.3 Deck Format Specification

```yaml
# decks/git_operations.yaml
deck:
  id: git_operations
  name: "Git Operations Deck"
  domain: version_control
  author: auto-generated
  validated: false

cards:
  - id: GIT_STATUS
    name: "Check Git Status"
    description: "Show working tree status"
    template: "git status --porcelain"
    output_parser: "lines"

  - id: GIT_DIFF
    name: "Show Changes"
    description: "Show unstaged changes"
    template: "git diff --stat"
    parameters:
      - name: path
        optional: true
        default: "."
    output_parser: "diff_stats"

  - id: GIT_LOG
    name: "Recent Commits"
    description: "Show recent commit history"
    template: "git log --oneline -n {count}"
    parameters:
      - name: count
        type: int
        default: 10
    output_parser: "lines"
```

### 3.4 Tasks

- [ ] **Deck schema definition** (`schema/deck.schema.json`)
- [ ] **Deck loader** — read YAML, validate, register cards
- [ ] **Deck selector** — choose appropriate deck based on goal/context
- [ ] **5 domain decks** — git, python, data, security, docs

---

## 4. Deck Generation by Proficient LLMs

### 4.1 Architecture: Fat LLM as Deck Generator

```
┌─────────────────────────────────────────────────────────────┐
│                    DECK GENERATION PIPELINE                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐     │
│  │ SLIM LLM │ FAILS   │ FAT LLM  │ PROPOSES│ VALIDATOR│     │
│  │ (Player) │────────▶│(Teacher) │────────▶│  (Test)  │     │
│  └──────────┘         └──────────┘         └──────────┘     │
│       │                    │                    │           │
│       │                    ▼                    ▼           │
│       │              ┌──────────┐         ┌──────────┐      │
│       │              │NEW CARDS │         │ RESULTS  │      │
│       │              │ (Draft)  │────────▶│(Pass/Fail)│     │
│       │              └──────────┘         └────┬─────┘      │
│       │                                        │            │
│       │                                        ▼            │
│       │              ┌──────────┐         ┌──────────┐      │
│       └──────────────│ PROMOTED │◀────────│  REVIEW  │      │
│         USE          │  CARDS   │ APPROVE │ (Human)  │      │
│                      └──────────┘         └──────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Generation Workflow

1. **Failure Detection**
   - Slim LLM fails benchmark or produces repeated syntax errors
   - Capture: goal, attempted commands, error messages

2. **Card Proposal**
   - Send failure context to fat LLM (Claude, GPT-4, or local large model)
   - Request: "Propose a card that would help solve this"
   - Output: YAML card definition

3. **Automated Validation**
   - Parse proposed card against schema
   - Execute template in sandbox with test inputs
   - Verify output parser works

4. **Integration Testing**
   - Re-run failed benchmark with new card available
   - Measure improvement in score/success

5. **Human Review (Optional)**
   - Flag cards for security review
   - Approve for promotion to permanent deck

### 4.3 Tasks

- [ ] **Failure context collector** — capture what went wrong
- [ ] **Card proposal prompt** — template for fat LLM
- [ ] **Card validator** — schema + sandbox execution
- [ ] **Integration test runner** — before/after comparison
- [ ] **Promotion workflow** — draft → validated → permanent

---

## 5. Validation & Testing Infrastructure

### 5.1 Card Testing Framework

```python
# tests/test_deck.py
def test_card_execution():
    """Each card must execute without error in sandbox."""
    for card in deck.cards:
        result = sandbox.execute(card.template.format(**test_params))
        assert result.rc == 0 or card.allows_failure

def test_card_output_parser():
    """Output parser must handle card output."""
    for card in deck.cards:
        output = sandbox.execute(card.template)
        parsed = card.parse_output(output.stdout)
        assert parsed is not None

def test_card_helps_benchmark():
    """Card should improve benchmark score."""
    score_without = run_benchmark(deck=None)
    score_with = run_benchmark(deck=deck)
    assert score_with >= score_without
```

### 5.2 Continuous Validation

- [ ] **Pre-commit hook** — validate deck YAML syntax
- [ ] **CI pipeline** — run deck tests on PR
- [ ] **Nightly benchmark** — detect regressions
- [ ] **Model update detection** — re-run when Ollama models update

### 5.3 Quality Metrics for Cards

| Metric | Definition | Target |
|--------|------------|--------|
| `usage_rate` | How often card is chosen | > 10% |
| `success_rate` | Card leads to progress | > 70% |
| `error_rate` | Card causes errors | < 5% |
| `redundancy` | Overlaps with other cards | Low |
| `coverage` | Benchmarks helped by card | Broad |

---

## 6. Logging System Design

### 6.1 Log Format (JSON-Lines)

```jsonl
{"ts": "2025-12-22T14:30:00Z", "type": "game_start", "game_id": "g001", "model": "granite3.1-moe:3b", "benchmark": "01_find_needle"}
{"ts": "2025-12-22T14:30:01Z", "type": "turn", "game_id": "g001", "turn": 1, "action": "SEARCH_CONTENT", "args": {"pattern": "EUREKA"}, "latency_ms": 450}
{"ts": "2025-12-22T14:30:02Z", "type": "result", "game_id": "g001", "turn": 1, "success": true, "output_preview": "data/file_delta.txt", "score_delta": 50}
{"ts": "2025-12-22T14:30:05Z", "type": "game_end", "game_id": "g001", "success": true, "total_turns": 2, "total_score": 350, "duration_s": 5.2}
```

### 6.2 Log Aggregation

```python
# analysis/aggregate_logs.py
def aggregate_by_model(log_file: Path) -> pd.DataFrame:
    """Aggregate game logs into model-level statistics."""
    games = parse_jsonl(log_file)
    return pd.DataFrame([
        {
            "model": g["model"],
            "benchmark": g["benchmark"],
            "success": g["success"],
            "turns": g["total_turns"],
            "score": g["total_score"],
            "duration": g["duration_s"],
        }
        for g in games if g["type"] == "game_end"
    ])
```

### 6.3 Tasks

- [ ] **Logging module** (`logging_utils.py`)
  - Context manager for game logging
  - Automatic timestamps and game IDs
  - Rotation and compression

- [ ] **Log viewer CLI** (`view_logs.py`)
  - Filter by model, benchmark, date
  - Summary statistics
  - Export to CSV/Excel

- [ ] **Dashboard** (optional, Streamlit)
  - Real-time benchmark progress
  - Historical comparisons
  - Interactive filtering

---

## 7. Priority Matrix

### Critical Remediations (from Academic Review - 2025-12-23)

| Task | Gap | Impact | Effort | Priority | Status |
|------|-----|--------|--------|----------|--------|
| **Meta-Cards (Tactical)** | #1 Complexity | Critical | Medium | **P0** | TODO |
| **Guided Dare Mechanism** | #1 Complexity | Critical | Low | **P0** | TODO |
| **Justification Scoring** | #4 Metrics | Critical | Medium | **P0** | TODO |
| **Failure Detection Triggers** | #2 Card Debt | Critical | Medium | **P0** | ✓ DONE |
| Macro Learning | #2 Card Debt | High | High | P1 | TODO |
| FOCUS(NodeSubset) | #3 Rigidity | Medium | Medium | P1 | TODO |
| Multi-Score System | #4 Metrics | High | Medium | P1 | TODO |
| Planning Phase | #1 Complexity | High | High | P2 | TODO |
| Card Graduation | #2 Card Debt | Medium | Medium | P2 | TODO |
| Hypothesis Branches | #3 Rigidity | Medium | High | P2 | TODO |

**Note (Follow-up Review - VALIDATED):** Gap #2 response marked "solid". Refinements added:

1. **Progress Stall via PCG** (not empty outputs):
   > *"Empty outputs is too weak. A model can babble without progress."*
   - **Implementation**: `len(pcg.nodes)` at turn T vs T-N
   - No new Truth/Observation/Entity nodes = no progress

2. **Repetition vs Circular require different responses**:
   | Failure | Diagnosis | Fat LLM Response |
   |---------|-----------|------------------|
   | Repetition (`cat A` ×3) | Syntactic stupidity | Alternative syntax card |
   | Circular (`ls→cat→ls`) | Strategic disorientation | Strategic/tactical card |

3. **Meta-cognitive transformation**:
   > *"This transforms from execution loop (bash codes) to meta-cognitive system (game dynamics)"*

See `CRITIQUE_RESPONSE.md` R2.0 for full `FailureDetector` + `get_fat_llm_instruction()` implementation.

### Original Tasks

| Task | Impact | Effort | Priority | Status |
|------|--------|--------|----------|--------|
| Structured logging | High | Medium | P0 | ✓ DONE |
| Multi-model benchmark run | High | Low | P0 | ✓ DONE |
| Feature extraction | High | Medium | P1 | ✓ DONE |
| PCA/dendrogram | Medium | Medium | P1 | ✓ DONE |
| Deck schema | Medium | Low | P1 | TODO |
| Git operations deck | Medium | Low | P2 | DEPRIORITIZED |
| ~~Fat LLM deck generation~~ | ~~High~~ | ~~High~~ | ~~P2~~ | **CANCELLED** (Gap #2) |
| Card validation framework | Medium | Medium | P2 | TODO |
| Dashboard | Low | High | P3 | TODO |

---

## 8. Next Steps (Immediate)

### Completed (2025-12-23)
- [x] JSON-Lines logging in `scored_mode.py` (`BenchmarkLogger` class)
- [x] Batch runner script (`run_benchmark_campaign.sh`)
- [x] Feature extraction (`analysis/extract_features.py`)
- [x] Visualization suite (`analysis/visualize.py`)
- [x] Benchmark campaign on 7 models (results in `results/`)
- [x] Fixed impossible benchmarks (B04, B06 success criteria)
- [x] Academic critique response (`CRITIQUE_RESPONSE.md`)
- [x] **FailureDetector implementation** (`failure_detector.py`)
  - Detects: EXPLICIT_ERROR, REPETITION_LOOP, CIRCULAR_PATTERN, PROGRESS_STALL, EXHAUSTION
  - Integrated with `scored_mode.py` for real-time detection
  - `get_fat_llm_instruction()` generates differentiated prompts per failure type
  - Logs failure events to JSON-Lines output

### Critical P0 Tasks (from Academic Review)

**Gap #1 - Complexity Wall:**
1. **Implement Meta-Cards** - Tactical strategies beyond atomic actions
   ```yaml
   # Examples:
   - EXPLORE_AND_REPORT(dir)      # Multi-step exploration
   - VERIFY_IMPORTS(entry_file)   # Import chain verification
   - SYNTHESIZE_FINDINGS()        # Compile evidence into conclusion
   ```

2. **Guided Dare Mechanism** - When CHECK returns undecidable, suggest specific actions
   ```python
   if verdict == "undecidable":
       suggestions = tutor.generate_dare_suggestions(context)
       # Don't leave LLM stuck - offer concrete options
   ```

**Gap #4 - Metric Bias:**
3. **Justification Scoring** - Require reasoning, not just correct answers
   ```python
   class Move:
       action: str
       justification: str  # NEW: Why this action?
       linked_evidence: List[str]  # NEW: Supporting observations

   # Penalties:
   unjustified_action: int = -10
   conclusion_without_evidence: int = -30
   ```

### Deprioritized/Cancelled
- ~~Fat LLM deck generation~~ → **CANCELLED** (creates dependency, not autonomy)
- Git operations deck → **DEPRIORITIZED** (solve Gap #2 with Macro Learning instead)

### Reference
See `CRITIQUE_RESPONSE.md` for full remediation plan addressing all 4 gaps.

---

## References

- TRIZ: Theory of Inventive Problem Solving (Altshuller)
- PCA: Principal Component Analysis for dimensionality reduction
- PCoA: Principal Coordinates Analysis for distance-based ordination
- Hierarchical Clustering: Ward's method for model grouping
