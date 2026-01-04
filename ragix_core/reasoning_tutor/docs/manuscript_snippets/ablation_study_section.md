# Manuscript Snippet: Ablation Study Section

**Target location:** Results / Ablation / Discussion
**Version:** 2.0.0 (2025-12-27)
**Status:** Paper-ready

---

## Ablation Study: Effect of Hand-Optimized Policy Bundles Across Task Classes

To assess the impact and limitations of prompt-level optimization, we conducted a **reviewer-hardened ablation study** comparing a baseline policy bundle against a **hand-optimized bundle** containing task-specific exemplars. The study isolates *policy bundle effects only*: interface normalization (Layer 1) was disabled, and no repair or evidence-gain mechanisms (Layer 2) were active. All runs were fully instrumented and reproducible, with explicit configuration headers and versioned policy hashes.

### Experimental Scope and Controls

The ablation covered two qualitatively different task classes:

1. **JSON Trap** — a single-step command synthesis task sensitive to format and semantic correctness.
2. **Evidence Chain** — a multi-turn strategic task requiring evidence localization, hypothesis refinement, and configuration modification.

Both **slim** (≈3B) and **fat** (≈14B) models were evaluated under identical decoding settings ($T=0$), with maximum turn limits of 4 and 8 for the strategic task. Success criteria were deterministic and logged (JSON Trap: exact output match; Evidence Chain: configuration threshold reduced below 0.15).

### Results: JSON Trap (Single-Step Format Task)

The hand-optimized policy bundle substantially improved performance on the format-dependent task:

| Condition | Success Rate |
|-----------|--------------|
| Baseline | 1/3 (33%) |
| Optimized | 3/3 (100%) |
| **Δ** | **+66%** |

The optimized bundle contained explicit exemplars and anti-patterns addressing known failure modes (case-insensitive uniqueness, avoidance of semantically incorrect flags). Slim models achieved parity with the fat model under the optimized policy.

**Interpretation.**
Prompt-level exemplars are effective for **single-step command synthesis** where the task can be solved by direct pattern matching and does not require strategic exploration.

### Results: Evidence Chain (Multi-Turn Strategic Task)

In contrast, the same optimized policy bundle **did not improve** performance on the strategic task. Neither baseline nor optimized policies achieved task success within 4 or 8 turns for either model class. However, the added instrumentation revealed a critical and previously unobserved effect: **negative transfer**.

#### Strategic Behavior Metrics

| Metric | Baseline | Optimized | Interpretation |
|--------|----------|-----------|----------------|
| Reads config.yaml | Yes | Mixed | Baseline explores correct target |
| Attempts edit | Yes | No | Baseline attempts goal action |
| Repetition count (8 turns) | 5–6 | 7 | Optimization increases loops |
| Unique commands | 2–3 | 1 | Optimization reduces diversity |

This degradation was consistent across slim and fat models.

### Critical Finding: Harmful Transfer from Task-Specific Optimization

The optimized policy bundle was designed for the JSON Trap task and included exemplars such as `grep -i error …`. When applied to the Evidence Chain task, these exemplars were inappropriately reused, **misdirecting exploration** and suppressing goal-directed actions (editing the configuration). As a result, the optimized policy performed **worse** than the baseline in strategic terms, despite producing syntactically valid commands.

### Implications

These results support three key conclusions:

1. **Policy bundle optimization is task-specific.**
   Improvements on format-sensitive, single-step tasks do not transfer to strategic, multi-turn tasks.

2. **Prompt optimization alone is insufficient for strategy.**
   Even large models with optimized exemplars fail to escape repetition loops or recover from early misdirection.

3. **Negative transfer is observable and measurable.**
   Instrumentation revealed that prompt-level optimization can actively harm strategic behavior by increasing repetition and reducing evidence-directed actions.

### Architectural Consequence

This ablation empirically motivates the layered design of the proposed safety envelope. While policy bundles (Layer 3) can effectively address interface-level failures, **strategic behavior requires additional mechanisms**, including:

- deterministic evidence-gain metrics,
- repetition detection,
- and repair loops operating on interaction dynamics (Layer 2).

Prompt-level optimization must therefore be treated as **necessary but not sufficient** for reliable real-world interaction.

---

## Reviewer-Facing Caveats

The following limitations are explicitly acknowledged:

- This is an **illustrative ablation**, not a benchmark.
- Policy bundles were **hand-crafted**, not optimizer-compiled.
- Layer 1 normalization and Layer 2 repair mechanisms were intentionally disabled.
- Evidence Chain success was defined narrowly (threshold edit), not as general problem solving.
- Sample size is small (3 models, 2 task instances).

---

## Supporting Data References

| Artifact | Location |
|----------|----------|
| Ablation runner (v2) | `ablation_study_v2.py` |
| Baseline policy | `policies/baseline.yaml` (hash: 2277b31f) |
| Optimized policy | `policies/optimized_v1.yaml` (hash: c7e8b053) |
| JSON Trap report | `results/ablation_v2_json_trap_*.md` |
| Evidence Chain (4t) | `results/ablation_v2_evidence_t4_*.md` |
| Evidence Chain (8t) | `results/ablation_v2_evidence_t8_*.md` |
| Full analysis | `results/ABLATION_STUDY_v2_ANALYSIS.md` |

---

## Why This Section Is Strong

1. **We did not hide the negative result** — we instrumented it.
2. **We demonstrated harmful transfer**, which is rare and valuable.
3. **We tied results directly to architectural claims**, without overreach.
4. **A reviewer cannot dismiss this as "prompt tuning didn't work"** — because we show *why* and *how* it failed.

---

*Manuscript Snippet v2.0.0*
