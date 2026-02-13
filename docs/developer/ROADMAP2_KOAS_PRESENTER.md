# I. Critical Review of the Current Presentation

## 1️⃣ Slide Volume: 120 Slides Is Structurally Correct — But Strategically Wrong

The planner is doing what it is designed to do:

* Proportional budget allocation
* Chapter-by-chapter coverage
* No synthesis
* No cross-cluster compression

This creates a **document-mirroring deck**.

That is appropriate for:

* Technical archive
* Traceable review
* Internal technical committee

It is not appropriate for:

* Executive board
* Steering committee
* Client summary

### Structural Problem

You are optimizing:

> coverage fidelity

But not optimizing:

> narrative compression

The deterministic planner allocates slides by section weight.
It does not:

* Merge minor sections
* Collapse repeated patterns
* Aggregate related findings across chapters
* Produce a "10-slide decision narrative"

This is why the result feels long.

The system is correct.
The strategy is incomplete.

---

## 2️⃣ Figures: Why They Appear Truncated

The truncation issue is structural, not cosmetic.

Current MARP behavior:

* `image_text` uses `![bg left:40%](img.svg)`
* SVGs are inserted as background
* Scaling depends on viewport and CSS
* MARP background image cropping is not semantic-aware

Common causes of truncation:

* SVG has fixed viewBox not matching slide ratio
* CSS `background-size: cover`
* No aspect-ratio constraint
* Using background image instead of inline image
* No bounding-box adaptation

This is a rendering policy issue.

The planner only chooses slide type.
The renderer assumes MARP background layout will behave.

It doesn’t — especially for complex diagrams.

---

## 3️⃣ Table Density and Readability

Even though overflow handling exists (`max_rows`, `max_cols`),
what is missing:

* Font scaling per table density
* Dynamic split threshold by slide area
* Visual priority between table and caption

Right now:
Tables fit technically.
They don’t fit visually.

---

## 4️⃣ Structural Redundancy

Even with heading deduplication:

* Some sections feel repetitive
* Technical metrics appear in multiple chapters
* Annexes inflate deck volume

This is because:
The planner does not compute cross-chapter similarity.

Deterministic dedup is intra-cluster.
Not inter-cluster.

---

# II. Roadmap for Improvement (v1 → v2 → v3)

I’ll structure this pragmatically.

---

# 🟦 PHASE 1 — Structural Compression (No LLM Required)

### Goal: Reduce slide count by 30–50% without losing information

## 1️⃣ Introduce “Compression Mode”

Add:

```yaml
slide_plan:
  mode: full | compressed | executive
```

### Compressed Mode Rules:

* Merge clusters under N units
* Cap slides per section (max 5)
* Collapse consecutive content slides
* Remove annexes unless referenced
* Aggregate tables under same heading

### Executive Mode:

* Only include roles:

  * problem
  * finding
  * recommendation
  * risk
* Ignore context/method unless high importance
* Max 20 slides
* Cross-chapter aggregation enabled

This can be done deterministically.

No LLM required.

---

## 2️⃣ Add Cross-Cluster Deduplication

Currently:
Dedup is local.

Add:

```python
dedupe_global(corpus, threshold=0.75)
```

Applied before slide planning.

This alone can cut 10–20 slides.

---

## 3️⃣ Add “Low-Value Section Filter”

Annexes:
Currently fully expanded.

Add rule:

```yaml
exclude_sections_if_label_contains:
  - "Annexe"
  - "Appendix"
```

Unless:

```yaml
force_include: true
```

---

# 🟨 PHASE 2 — Layout Intelligence (Renderer-Level Fixes)

This is where figure truncation must be solved.

## 1️⃣ Stop Using Background Images for Complex Figures

Instead of:

```
![bg left:40%](fig.svg)
```

Use inline:

```
<div class="figure">
<img src="fig.svg" />
</div>
```

And control with CSS:

```
.figure img {
   max-height: 80vh;
   max-width: 45vw;
   object-fit: contain;
}
```

This avoids cropping.

---

## 2️⃣ Auto Aspect-Ratio Detection

At S1:
Extract SVG dimensions from viewBox.

At S2:
Choose layout:

* Tall image → full-height centered
* Wide image → full-width
* Square → split layout

SlideType selection should depend on aspect ratio.

Right now it does not.

---

## 3️⃣ Table Scaling Strategy

Implement:

```
if table_density > threshold:
    reduce_font_size(class="table-small")
```

Add CSS classes:

* `.table-normal`
* `.table-small`
* `.table-tiny`

Based on:

* rows × cols
* estimated character width

---

## 4️⃣ Introduce “Figure-First Slide Type”

Current:
`image_text`

Add:
`image_full_centered`

Used when:
Image importance > text importance.

This avoids 40% split when 100% would be better.

---

# 🟥 PHASE 3 — Narrative Intelligence (Optional LLM Layer)

Only after structural + layout are fixed.

### 1️⃣ Cross-Chapter Synthesis

Use normalizer LLM to:

* Aggregate all findings into one “Global Findings” cluster
* Aggregate risks
* Aggregate recommendations

Produce:
Executive layer above chapter layer.

This solves:
“Too linear” feeling.

---

### 2️⃣ Slide Density Scoring

Compute:

```
visual_density = word_count + (table_weight * 3) + (equation_weight * 4)
```

If density > threshold:
Split automatically.

Currently split is budget-based, not visual-based.

---

# III. Strategic Recommendation

Before touching LLM:

Fix rendering policy.

Because:
Layout issues are more visible than semantic issues.

---

# IV. Concrete Roadmap Proposal

### v1.1 — Rendering Fix Release (Fast Win)

* Inline image rendering (no background for SVG)
* Aspect-ratio-aware slide type
* Table size classes
* Figure-full slide type

→ Solve truncation.

---

### v1.2 — Compression Mode

* compressed mode
* executive mode
* global dedupe
* annex exclusion

→ Reduce slide count to 40–60 range.

---

### v2.0 — Narrative Layer

* cross-chapter synthesis
* LLM-assisted clustering
* role-priority selection
* speaker narrative generation


