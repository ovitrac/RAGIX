# KOAS Presenter — Slide Deck Generation from Documents

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 1.2.0
**Date:** 2026-02-12
**Status:** Production (deterministic pipeline) | LLM normalizer: Phase 1 (deterministic heuristics) | Compression: v1.2 | Layout Intelligence: v1.2

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Data Models](#data-models)
4. [Configuration](#configuration)
5. [CLI: `presenterctl`](#cli-presenterctl)
6. [MCP Integration](#mcp-integration)
7. [Theme System](#theme-system)
8. [Content Extraction Strategy](#content-extraction-strategy)
9. [Slide Planning: Budget, Deduplication, and Filtering](#slide-planning-budget-deduplication-and-filtering)
10. [Compression Modes (v1.2)](#compression-modes-v12)
11. [Layout Intelligence (v1.2)](#layout-intelligence-v12)
12. [Design Decisions](#design-decisions)
13. [Production Benchmark: IOWIZME Audit](#production-benchmark-iowizme-audit)
14. [Requirements](#requirements)
15. [Implementation Status](#implementation-status)
16. [Future Directions](#future-directions)
17. [Related Documentation](#related-documentation)

---

## 1. Overview

The KOAS Presenter is a kernel family (`ragix_kernels/presenter/`) that transforms
a folder of Markdown/text documents — with their equations, illustrations, tables, and
code blocks — into structured [MARP](https://marp.app/) slide decks, then exports to
PDF/HTML.

```bash
# Full archive (120 slides)
python -m ragix_kernels.presenter.cli.presenterctl render ./report/ \
  --format both --theme koas-professional --max-slides 120

# Executive summary (25 slides, findings/recommendations only)
python -m ragix_kernels.presenter.cli.presenterctl render ./report/ \
  --format both --theme koas-professional --compression executive
```

**Codebase:** 23 files, ~9,100 lines (8 kernels, 26 dataclasses, 9 enums, 3 CLI subcommands, 3 MCP tools, 1 custom theme, 3 test suites). v1.2 adds compression modes and layout intelligence.

### Core Architectural Principles

> **P1. Separate content generation from layout rendering.**

> **P2. Separate raw extraction from semantic normalization.**

These create three clean boundaries:

```
   RAW EXTRACTION          SEMANTIC NORMALIZATION       LAYOUT RENDERING
 +------------------+    +-----------------------+    +----------------------+
 |  S1 Folder scan  |    |  S2a Normalize        |    |  S3a MARP renderer   |
 |  S1 Content ext. |--->|  S2b Slide planner    |--->|  S3b MARP export     |
 |  S1 Asset catalog|    |  S2c Layout mapper    |    |       (marp-cli)     |
 +------------------+    +-----------------------+    +----------------------+
     ^ source files       ContentCorpus -> JSON         v .pdf / .html
                         NormalizedCorpus  schema
```

Two contracts:
- **`NormalizedCorpus`** — between raw extraction (S1) and structuring (S2)
- **`SlideDeck` JSON schema** — between structuring (S2) and rendering (S3)

Content generation knows *what* to present; layout rendering knows *how* to render it.
Each boundary can be replaced or improved independently.

### Why MARP?

| Property | Benefit |
|----------|---------|
| Markdown-native | Zero impedance mismatch with KOAS document corpus |
| KaTeX/MathJax built-in | Block equations pass through unmodified |
| CLI export | `marp-cli` -> PDF/HTML/PPTX, no GUI dependency |
| Theme system | CSS-based, supports custom corporate themes |
| Speaker notes | HTML comments -> presenter mode / PDF annotations |
| Image syntax | `![bg left:40%](img.svg)` — split layouts from Markdown |
| Auto-scaling | Long code blocks and equations scale to fit |
| Sovereign | Local binary, no cloud dependency |

---

## 2. Pipeline Architecture

### Stage Map

```
folder/
  +-- 00_INDEX.md
  +-- 01_SYNTHESE.md
  +-- ...
  +-- 12_ANNEXES.md
  +-- assets/
  |   +-- fig01.svg
  |   +-- fig02.png
  +-- data/
      +-- metrics.json

         +----------------------- S1: COLLECTION ------------------------------+
         |                                                                     |
         |   pres_folder_scan --> pres_content_extract --> pres_asset_catalog  |
         |   (file tree, types)   (semantic units)         (image/eq/table     |
         |                                                  inventory)         |
         +---------------------------------+-----------------------------------+
                                           | ContentCorpus
         +----------------------- S2: STRUCTURING -----------------------------+
         |                                                                     |
         |   pres_semantic_normalize --> pres_slide_plan --> pres_layout_assign|
         |   (cluster, dedupe,           (content -> slides) (slide -> MARP    |
         |    role, importance)                                directives)     |
         +---------------------------------+-----------------------------------+
                                           | SlideDeck JSON (THE CONTRACT)
         +----------------------- S3: RENDERING -------------------------------+
         |                                                                     |
         |   pres_marp_render --> pres_marp_export                             |
         |   (JSON -> .md MARP)   (asset bundle + marp-cli -> PDF/HTML)        |
         +---------------------------------------------------------------------+
```

### Kernel Inventory (8 kernels)

| # | Kernel | Stage | LLM? | Input | Output |
|---|--------|-------|------|-------|--------|
| 1 | `pres_folder_scan` | S1 | No | folder path, glob patterns | `FileTree` — paths, types, sizes, hashes |
| 2 | `pres_content_extract` | S1 | No | FileTree + file contents | `ContentCorpus` — semantic units per file |
| 3 | `pres_asset_catalog` | S1 | No | FileTree + ContentCorpus | `AssetCatalog` — images, equations, tables, code |
| 4 | **`pres_semantic_normalize`** | **S2** | **Optional** | ContentCorpus | **`NormalizedCorpus`** — clustered, deduplicated, ordered |
| 5 | `pres_slide_plan` | S2 | No | NormalizedCorpus + AssetCatalog | `SlideDeck` (JSON schema) |
| 6 | `pres_layout_assign` | S2 | No | SlideDeck | `SlideDeck` (enriched with MARP layout directives) |
| 7 | `pres_marp_render` | S3 | No | SlideDeck + AssetCatalog | `.md` file (MARP syntax) |
| 8 | `pres_marp_export` | S3 | No | `.md` file + theme CSS | `.pdf` / `.html` via `marp-cli` |

### Dependency Graph

```
pres_folder_scan
    +---> pres_content_extract
    |        +---> pres_asset_catalog
    |        |        |
    |        v        |
    |    pres_semantic_normalize  (optional LLM --- identity fallback)
    |        |        |
    |        v        v
    |    pres_slide_plan
    |        |
    |        v
    |    pres_layout_assign
    |        |
    v        v
    pres_marp_render
         |
         v
    pres_marp_export
```

**Bypass path:** When `normalizer.mode = deterministic` (default), the normalizer uses
heading-path clustering, Jaccard deduplication, keyword-based role assignment, and
heuristic importance scoring — no LLM calls. This preserves a fully deterministic,
reproducible pipeline.

---

## 3. Data Models

All models are defined in `ragix_kernels/presenter/models.py` (884 lines, 9 enums, 26 dataclasses).
Every dataclass implements `to_dict()` and `from_dict()` for JSON round-trip serialization.

### S1: Content Extraction

Documents are decomposed into **semantic units** — the atomic building blocks of slides.

**13 unit types (`UnitType`):**

| UnitType | Description |
|----------|-------------|
| `heading` | Section title (H1-H6) |
| `paragraph` | Prose text block |
| `bullet_list` | Unordered list |
| `numbered_list` | Ordered list |
| `table` | Markdown table |
| `code_block` | Fenced code (` ```lang ... ``` `) |
| `equation_block` | Display math (`$$...$$`) |
| `equation_inline` | Inline math (`$...$`) |
| `blockquote` | `>` quoted text |
| `image_ref` | `![alt](path)` reference |
| `mermaid` | ` ```mermaid ... ``` ` diagram |
| `front_matter` | YAML front matter |
| `admonition` | `> [!NOTE]` / `> [!WARNING]` blocks |

**5 file types (`FileType`):** `DOCUMENT`, `ASSET`, `DATA`, `CONFIG`, `UNKNOWN`.

**5 asset types (`AssetType`):** `IMAGE`, `EQUATION`, `TABLE`, `CODE`, `DIAGRAM`.

Key models: `FileEntry`, `SemanticUnit`, `ContentCorpus`, `OutlineNode`, `Asset`, `AssetCatalog`.

### S2: Semantic Normalization

The normalizer transforms raw `ContentCorpus` into `NormalizedCorpus` by adding:

| Operation | LLM mode | Deterministic fallback (default) |
|-----------|----------|----------------------------------|
| **Topic clustering** | Embedding + HDBSCAN, optional LLM refinement | `cluster_by_heading_path(max_levels=2)` |
| **Deduplication** | Cosine similarity > 0.85 | `jaccard_similarity() > 0.70` (intra-cluster), `> 0.80` (global, v1.2) |
| **Role assignment** | LLM classification -> 10 roles | `assign_role_by_keywords()` (bilingual FR/EN lexicon) |
| **Importance scoring** | LLM relevance (0-1) | `compute_importance()` (type + role + signals + depth) |
| **Narrative ordering** | LLM arc generation | `detect_narrative_arc()` (role-based ordering) |
| **Cluster consolidation** | N/A | `consolidate_clusters(max_clusters=20)` (merge smallest) |

**10 semantic roles (`UnitRole`):** `context`, `problem`, `method`, `finding`, `recommendation`,
`conclusion`, `reference`, `illustration`, `metadata`, `unknown`.

**3 normalization modes (`NormalizationMode`):** `IDENTITY`, `DETERMINISTIC`, `LLM`.

Key models: `NormalizedUnit`, `TopicCluster`, `NarrativeArc`, `NormalizedCorpus`.

### Importance Scoring (deterministic)

Importance is a composite score in `[0.0, 1.0]` computed by `compute_importance()` in `normalize_utils.py`:

| Factor | Values |
|--------|--------|
| **Type boost** | `HEADING` H1/H2/H3: +0.3/+0.2/+0.1; `IMAGE_REF`: +0.2 |
| **Role boost** | `FINDING`/`RECOMMENDATION`: +0.2; `PROBLEM`: +0.15; `ILLUSTRATION`: +0.15; `METHOD`/`CONCLUSION`: +0.1; `CONTEXT`: +0.05 |
| **Content signals** | Percentage patterns: +0.1; Large numbers: +0.05; List types: +0.05 |
| **Depth decay** | -0.1 per heading nesting level |

### S2-S3 Contract: SlideDeck JSON Schema

The central data structure separating content generation from layout rendering.

**12 slide types (`SlideType`):** `title`, `section`, `content`, `two_column`, `image_text`,
`image_full`, `equation`, `table`, `code`, `quote`, `summary`, `blank`.

**4 provenance methods:** `extracted`, `synthesized`, `user_outline`, `auto_section`.

Structure:
```json
{
  "metadata": { "title": "...", "author": "...", "lang": "fr", ... },
  "theme": { "name": "koas-professional", "size": "16:9", "math": "katex" },
  "slides": [
    {
      "id": "slide-001",
      "type": "content",
      "content": { "heading": "...", "bullets": ["..."] },
      "notes": "Speaker notes...",
      "provenance": { "source_file": "...", "source_lines": [42, 58], ... },
      "layout": { "class": "...", "paginate": true, "inline_image": false, "table_class": "", ... }
    }
  ]
}
```

---

## 4. Configuration

All settings are in `ragix_kernels/presenter/config.py` as 16 nested dataclasses
with `to_dict()`/`from_dict()` round-trip serialization.

### Key Configuration Sections

| Section | Key Settings |
|---------|-------------|
| `folder_scan` | include/exclude patterns, max depth, symlinks |
| `normalizer` | enabled, mode (`auto`/`deterministic`/`llm`), model, max_clusters (20), clustering, budget |
| `slide_plan` | max bullets/words per slide, equation standalone, min/max slides (8/60), **compression mode**, per-section cap, annex exclusion, executive filter |
| `table_overflow` | max rows (12), max cols (8), strategy (`split`/`image`/`truncate`) |
| `theme` | name (`default`/`gaia`/`koas-professional`), custom CSS path, size, math engine, colors, logo |
| `notes` | depth (`none`/`section_only`/`file_line`/`full`) |
| `export` | format (`md`/`pdf`/`html`/`pptx`/`png`/`both`), pdf_notes, pdf_outlines |
| `llm` | backend (`ollama`), endpoint, model, temperature (0.1), timeout (120s), strict_sovereign |

### Example Configuration

```yaml
presenter:
  folder_scan:
    include_patterns: ["**/*.md", "**/*.txt"]
    exclude_patterns: ["**/node_modules/**", "**/.git/**"]
  normalizer:
    enabled: true
    mode: "deterministic"
    max_clusters: 20
    deduplication:
      threshold: 0.70
      global_threshold: 0.80       # v1.2 — cross-cluster dedupe (compressed/executive only)
  slide_plan:
    max_bullets_per_slide: 6
    max_words_per_slide: 80
    max_slides: 120
    section_order: "document"
    compression: "full"              # v1.2 — full | compressed | executive
    max_slides_per_section: 0        # v1.2 — 0 = unlimited; >0 = hard cap per section
    annex_exclude_patterns:          # v1.2 — cluster labels to exclude in compressed/executive
      - "annexe"
      - "annex"
      - "appendix"
      - "appendice"
    executive_min_importance: 0.5    # v1.2 — minimum importance for IMAGE_REF/TABLE in executive mode
  theme:
    name: "koas-professional"
    size: "16:9"
    math: "katex"
  export:
    format: "both"
```

---

## 5. CLI: `presenterctl`

**Entry point:** `python -m ragix_kernels.presenter.cli.presenterctl`
**Source:** `ragix_kernels/presenter/cli/presenterctl.py` (~520 lines)

```
presenterctl — KOAS Presentation Generator

SUBCOMMANDS:
  render     Full S1->S2->S3 pipeline: folder -> presentation
  export     Export existing workspace to PDF/HTML
  show       Display workspace info and stage completion

USAGE:
  presenterctl render <folder> [options]
  presenterctl export <workspace> [options]
  presenterctl show <workspace> [options]
```

### `render` — Full Pipeline

```
presenterctl render <folder> [options]

OPTIONS:
  -w, --workspace <path>       KOAS workspace for artifacts
  --mode <mode>                deterministic | llm | auto (default: deterministic)
  --model <name>               LLM model for normalizer (e.g., mistral-small:24b)
  --max-slides <n>             Maximum slide count (default: 60, or 25 for executive)
  --compression <mode>         full | compressed | executive (default: full) [v1.2]
  --title <text>               Presentation title (default: from front matter)
  --author <text>              Author name
  --organization <text>        Organization name
  --date <YYYY-MM-DD>          Presentation date
  --section-order <order>      document | narrative (default: document)
  -f, --format <fmt>           md | pdf | html | both (default: md)
  --theme <name>               Theme: default | gaia | koas-professional (default: koas-professional)
  -v, --verbose                Verbose logging
```

### `export` — Re-export Workspace

```
presenterctl export <workspace> [options]

OPTIONS:
  -f, --format <fmt>           pdf | html | both (default: pdf)
  --theme <name>               Override theme for export
  -v, --verbose                Verbose logging
```

### `show` — Workspace Info

```
presenterctl show <workspace> [-v]
```

Displays stage completion status, file sizes, slide count, metadata, and theme information.

### Example Workflows

```bash
# Quick: folder -> PDF with professional theme
presenterctl render ./report/ -f pdf --theme koas-professional

# Full pipeline with explicit options (archive mode, 120 slides)
presenterctl render ./report/ \
  -w ./workspace/ \
  --title "Audit IOWIZME" --author "Olivier Vitrac" \
  --organization "Adservio" --max-slides 120 \
  --format both --theme koas-professional -v

# Compressed mode: annex exclusion + per-section cap + global dedupe
presenterctl render ./report/ \
  --compression compressed --format both -v

# Executive mode: 25 slides, findings/recommendations only
presenterctl render ./report/ \
  --compression executive --format both -v

# Re-export workspace with different theme
presenterctl export ./workspace/ -f html --theme gaia

# Inspect workspace
presenterctl show ./workspace/ -v
```

### Implementation Patterns

The CLI follows the same patterns as `reviewctl`:

- **`_KERNEL_MAP`**: lazy kernel imports (8 entries)
- **`_get_presenter_kernel(name)`**: direct imports bypassing global registry (avoids optional deps)
- **`_run_kernel(name, workspace, config, verbose)`**: single kernel runner with ANSI color output
- **`_discover_dependencies(requires, workspace)`**: scans `stage{1,2,3}/` for `{kernel}.json` outputs
- **`_resolve_workspace(folder, workspace)`**: default `.presenter/<stem>_<hash12>/` inside folder

---

## 6. MCP Integration

3 tools via `register_presenter_tools(mcp_server)`:

**Source:** `ragix_kernels/presenter/mcp/tools.py` (~400 lines)

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `presenter_render` | Full pipeline: folder -> presentation | `folder_path`, `mode`, `max_slides`, `export_format`, `theme` |
| `presenter_export` | Re-export existing workspace | `workspace`, `export_format`, `theme` |
| `presenter_status` | Query workspace status | `workspace` |

### Registration

```python
from ragix_kernels.presenter.mcp import register_presenter_tools

register_presenter_tools(mcp_server)
```

### Response Format

All tools return standardized dicts:

```python
{
    "status": "success",  # or "error"
    "summary": "Rendered 120 slides (56 content, 24 image_text, ...)",
    "workspace": "/path/to/workspace",
    "output_dir": "/path/to/output",
    "slide_count": 120,
    "presentation_file": "/path/to/presentation.md",
    "pdf_file": "/path/to/presentation.pdf",
    "html_file": "/path/to/presentation.html",
}
```

---

## 7. Theme System

### Built-in MARP Themes

| Theme | Style | Best For |
|-------|-------|----------|
| `default` | Clean, minimal | Technical presentations |
| `gaia` | Modern, warm tones | Design-forward talks |
| `uncover` | Bold, high-contrast | Executive summaries |
| **`koas-professional`** | **Corporate blue, data-heavy styling** | **Audit reports, technical decks** |

### `koas-professional` Theme

**Source:** `ragix_kernels/presenter/themes/koas-professional.css` (~300 lines)

Extends `@import 'default'` with a professional corporate palette:

**CSS Variables:**

| Variable | Value | Purpose |
|----------|-------|---------|
| `--koas-primary` | `#0066cc` | Headers, table headers, lead background |
| `--koas-accent` | `#e17055` | Blockquote borders, hover highlights |
| `--koas-text` | `#2d3436` | Body text |
| `--koas-bg-alt` | `#f8f9fa` | Code blocks, alternating backgrounds |
| `--koas-table-header` | `#0066cc` | Table header background |
| `--koas-table-stripe` | `#f0f4f8` | Table striped rows |

**Slide Classes:**

| Class | Description |
|-------|-------------|
| `section.lead` | Title slide: centered white text on blue background |
| `section.section` | Section divider: gradient blue (135deg, `#0066cc` -> `#004a99`) |
| `.columns` / `.column` | Two-column flexbox layout (40px gap) |

**Layout Intelligence Classes (v1.1):**

| Class | Description |
|-------|-------------|
| `.figure` | Inline image container (45% width, float left, max-height 65vh) |
| `.figure-full` | Full-width inline image (100% width, max-height 80vh) |
| `.table-small` | Dense table scaling (font-size 0.65em, compact padding) |
| `.table-tiny` | Very dense table scaling (font-size 0.50em, minimal padding) |

**Typography:** Inter/Segoe UI (body), JetBrains Mono/Fira Code (code).

### Custom Theme Development

Themes are resolved by `_resolve_theme_css()` in `pres_marp_export.py`:

1. `theme.custom_css_path` (absolute or relative) — highest priority
2. Bundled theme by name: `ragix_kernels/presenter/themes/{name}.css`
3. `None` — marp-cli built-in default

Themes must style all 12 slide types. The MARP renderer emits `<!-- _class: X -->` directives.

```
ragix_kernels/presenter/themes/
+-- koas-professional.css   # Corporate identity (implemented)
```

---

## 8. Content Extraction Strategy

### Equation Handling

| Pattern | Detection | Slide Treatment |
|---------|-----------|----------------|
| `$...$` inline | Regex + AST | Preserved in parent paragraph/bullet |
| `$$...$$` block | Regex + AST | Standalone `equation` slide |
| `\[...\]` display | Regex | Convert to `$$...$$` |
| `\(...\)` inline | Regex | Convert to `$...$` |
| `\begin{align}...` | Regex | Wrap in `$$...$$` |

MARP uses KaTeX natively — block equations pass through unmodified.

### Illustration Handling

- **File-system images** (SVG/PNG/JPG) -> cataloged with dimensions, format, hash
- **Document-referenced images** -> linked to `AssetCatalog`, caption from alt text
- **Mermaid diagrams** -> pre-render to SVG if mermaid-cli available, else code block
- **Large tables** (>= 5 rows/cols) -> cataloged as assets for potential image rendering
- **Image slides** -> `![bg left:40%](assets/fig.svg)` split layout with caption text

### Multi-Document Ordering

| Strategy | Logic | Use Case |
|----------|-------|----------|
| `document` (default) | First-appearance order of cluster labels | Hierarchical corpus |
| `narrative` | Role-based arc: CONTEXT -> PROBLEM -> METHOD -> FINDING -> RECOMMENDATION -> CONCLUSION | Cross-document synthesis |

---

## 9. Slide Planning: Budget, Deduplication, and Filtering

### Budget Allocation

The slide planner (`pres_slide_plan.py`) distributes slides proportionally:

1. **Total budget:** `max_slides` (default 60, configurable up to 120+)
2. **Reserved:** 1 title slide + N section dividers
3. **Content budget:** `max_slides - 1 - N_sections`
4. **Per-section allocation:** proportional to unit count in each cluster
5. **Rounding adjustment:** redistribute residual to largest sections

Within each section, units are sorted by importance (highest first) and mapped to slides
until the section budget is exhausted.

### Meta-Cluster Filtering (v5)

Boilerplate clusters are automatically excluded via substring matching:

```
"table des matieres", "table des figures", "table des tableaux",
"historique du document", "uncategorized",
"table of contents", "list of figures", "list of tables"
```

Any cluster whose label *contains* one of these patterns is skipped. This prevents
table-of-contents, figure lists, and document-history sections from wasting slide budget.

### Heading Deduplication (v5)

When multiple units in a section share the same heading (e.g., several paragraphs under
"1.3 Constats majeurs"), the slide planner appends a numeric suffix to avoid confusion:

- First occurrence: `"1.3 Constats majeurs"`
- Second occurrence: `"1.3 Constats majeurs (2)"`
- Third occurrence: `"1.3 Constats majeurs (3)"`

### IMAGE_REF Importance Boost (v5)

Image references receive a +0.2 type boost and the ILLUSTRATION role boost was increased
from +0.05 to +0.15. This ensures figures compete effectively with text content for
slide budget, preventing the image-exclusion problem observed in early versions where
all 26 SVGs were dropped in favor of paragraphs and tables.

---

## 10. Compression Modes (v1.2)

The `--compression` flag controls how aggressively the pipeline compresses content into fewer slides.
Three modes progressively trade completeness for conciseness — all purely deterministic, no LLM.

### Mode Summary

| Setting | `full` | `compressed` | `executive` |
|---------|--------|--------------|-------------|
| **Max slides** | 60 (user-configurable) | 60 (user-configurable) | 25 (auto-cap) |
| **Per-section cap** | Unlimited | 8 (configurable) | Unlimited |
| **Annex exclusion** | Off | On | On |
| **Global cross-cluster dedupe** | Off | On (threshold 0.80) | On (threshold 0.80) |
| **Role filter** | Off | Off | FINDING / RECOMMENDATION / PROBLEM |

### `full` (default)

Archive mode. Every non-duplicate, non-boilerplate unit competes for slide budget.
Identical behavior to v1.0 — no compression is applied. The proportional budget
allocation distributes slides across all sections.

### `compressed`

Structural compression without content loss of important material:

1. **Annex exclusion**: clusters whose label contains `annexe`, `annex`, `appendix`, or `appendice` are
   excluded entirely from the slide budget.
2. **Global cross-cluster deduplication**: after intra-cluster Jaccard dedup (threshold 0.70),
   a second pass compares all active units *across different clusters* at threshold 0.80 (stricter
   to avoid cross-topic false positives). Same-type constraint, headings skipped. Canonical unit
   is the one with higher importance.
3. **Per-section cap**: each section is limited to `max_slides_per_section` slides (default 8 in
   compressed mode). Budget is `min(proportional_budget, cap)`.

### `executive`

Decision-focused mode for executive audiences:

1. All `compressed` filters apply (annex exclusion + global dedupe).
2. **Role filter**: only `FINDING`, `RECOMMENDATION`, and `PROBLEM` roles pass through.
   `IMAGE_REF` and `TABLE` types are allowed if their importance score >= `executive_min_importance`
   (default 0.5), regardless of role.
3. **Auto-cap**: `max_slides` defaults to 25 when no explicit `--max-slides` is provided.
   Per-section cap is unlimited (proportional allocation within the 25-slide budget).

### Configuration Keys

```yaml
slide_plan:
  compression: "full"                  # full | compressed | executive
  max_slides_per_section: 0            # 0 = unlimited; >0 = hard cap
  annex_exclude_patterns:              # cluster labels to exclude
    - "annexe"
    - "annex"
    - "appendix"
    - "appendice"
  executive_min_importance: 0.5        # IMAGE_REF/TABLE threshold in executive mode

normalizer:
  deduplication:
    global_threshold: 0.80             # cross-cluster dedupe (compressed/executive)
```

---

## 11. Layout Intelligence (v1.2)

The layout assignment kernel (`pres_layout_assign.py`) uses asset metadata and table
structure to make layout decisions, avoiding visual problems like cropped images and
unreadable dense tables.

### All-Inline Image Rendering (v1.2)

**All** images now use inline HTML rendering instead of MARP background syntax. This
eliminates the `![bg left:40%]` cropping problem where SVGs were truncated or shrunk
to fill the background area. The kernel looks up image dimensions from the `AssetCatalog`
(by `asset_id` first, then by path basename as fallback) and assigns a CSS class:

| Condition | CSS Class | Rendering |
|-----------|-----------|-----------|
| `IMAGE_FULL` slide | `figure-full` | `<div class="figure-full">` — 100% width, 75vh max-height |
| Portrait/tall (w/h < 0.9) | `figure` | `<div class="figure">` — 50% width, floated left |
| Landscape (w/h >= 0.9) | `figure-landscape` | `<div class="figure-landscape">` — 100% width top, 50vh max-height |

The `SlideLayout` model carries an `image_class` field (`""`, `"figure"`, `"figure-landscape"`,
or `"figure-full"`) that the MARP renderer uses directly for the wrapping `<div>`.

**Path-based asset lookup:** `AssetCatalog.get_by_path()` performs exact match on
`source_file`/`path`, then falls back to basename matching — necessary because
`pres_slide_plan` sets `path` but not `asset_id` on `SlideImage`.

### Table Density Scaling

For `TABLE` slides, the kernel computes `rows * cols` cell count:

| Condition | Class | Font Size | Padding |
|-----------|-------|-----------|---------|
| cells > 80 or rows > 12 | `table-tiny` | 0.50em | 3px 6px |
| cells > 40 or rows > 8 | `table-small` | 0.65em | 4px 10px |
| Otherwise | *(default)* | Normal | Normal |

The MARP renderer wraps the table in `<div class="{table_class}">` when a density class
is assigned, and the `koas-professional` theme applies the font/padding rules.

### Dependencies

`pres_layout_assign` requires both `pres_slide_plan` and `pres_asset_catalog` (previously
only `pres_slide_plan`). The catalog provides image dimensions for aspect-ratio computation.

---

## 12. Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | JSON-first schema between S2 and S3 | Schema validation, tooling, debug visibility |
| D2 | MARP over Reveal.js | Markdown-native, CLI export, sovereign |
| D3 | Deterministic planner by default | Reproducible, debuggable, LLM-free default |
| D4 | Provenance on every slide | KOAS traceability requirement |
| D5 | Asset deduplication | Prevent redundant file copies |
| D6 | LLM isolation in normalizer only | LLM never touches layout, assets, or MARP syntax |
| D7 | Proportional budget allocation | Larger chapters get more slides automatically |
| D8 | Substring meta-filter | Robust against label variations ("Table des tableaux principaux") |
| D9 | Heading dedup within sections | Prevents confusing duplicate slide headings |
| D10 | Three compression modes (full/compressed/executive) | Same corpus, three audience levels, zero LLM cost |
| D11 | All-inline image rendering | Eliminates MARP bg cropping; CSS classes for portrait/landscape/full |
| D12 | Table density CSS classes | Large tables remain readable without manual sizing |

---

## 13. Production Benchmark: IOWIZME Audit

Full pipeline run on a 14-document French technical audit report (2,831 units, 149K tokens, 27 SVG figures).

### Input Corpus

| Metric | Value |
|--------|-------|
| Source folder | `/home/olivi/Documents/Adservio/audit/IOWIZME/report/` |
| Documents | 14 Markdown files |
| Total files | 41 (14 docs, 27 assets) |
| Semantic units | 2,831 |
| Tokens estimated | 149,474 |
| Assets cataloged | 517 (28 image, 475 table, 10 code, 4 equation) |
| Language | French |

### Pipeline Results — Three Compression Modes (v6, deterministic)

| Metric | `full` | `compressed` | `executive` |
|--------|--------|--------------|-------------|
| **Slides** | **120** | **60** | **25** |
| Content slides | 56 | 18 | 2 |
| Image slides | 24 | 24 | 10 |
| Table slides | 24 | 3 | 0 |
| Section dividers | 15 | 14 | 12 |
| Title slides | 1 | 1 | 1 |
| Annex units excluded | 0 | 58 | 58 |
| Executive-filtered units | 0 | 0 | 416 |
| Active units (planner input) | 868 | 810 | 394 |
| MARP Markdown | 50 KB | 17 KB | 6 KB |
| Assets bundled | 23 | 23 | 9 |
| PDF size | 977 KB | 711 KB | 326 KB |
| HTML size | 323 KB | 242 KB | 128 KB |
| **Reduction vs full** | — | **50%** | **79%** |

### Layout Intelligence Results (v1.2)

| Metric | Value |
|--------|-------|
| Tables auto-scaled | 9 (7 `table-small`, 2 `table-tiny`) |
| Inline images (landscape) | 22 (`figure-landscape` class) |
| Inline images (portrait) | 1 (`figure` class — `fig06_reviewer_pipeline.svg`, aspect 0.33) |
| Legacy bg images | 1 (missing JPG, no dimensions in catalog) |
| MARP bg images | 0 (all images with known dimensions use inline rendering) |

### Quality Metrics

| Metric | Value |
|--------|-------|
| Chapter coverage (full) | 13/13 (100%) |
| SVG inclusion (full) | 24/26 (92.3%) |
| SVG inclusion (executive) | 10/26 (38.5%) — findings-related only |
| Duplicate headings | 0 |
| Meta-sections (boilerplate) | 0 |
| Broken image references | 0 |
| Total execution time | ~6s (S1+S2) + ~3s (marp-cli PDF+HTML) per mode |

### Slide Distribution by Chapter (full mode, 120 slides)

| Section | Content | Table | Image | Total |
|---------|---------|-------|-------|-------|
| Ch.0 Index | 0 | 1 | 0 | 1 |
| Ch.1 Synthese executive | 3 | 0 | 1 | 4 |
| Ch.2 Contexte et perimetre | 5 | 0 | 2 | 7 |
| Ch.3 Methodologie | 11 | 2 | 2 | 15 |
| Ch.4 Architecture | 5 | 3 | 4 | 12 |
| Ch.5 Dette technique | 4 | 4 | 3 | 11 |
| Ch.6 Analyse ECH-SIAS | 6 | 3 | 2 | 11 |
| Ch.7 Revue documentaire | 4 | 3 | 3 | 10 |
| Ch.8 Conformite RIE | 3 | 1 | 1 | 5 |
| Ch.9 Estimation MCO | 5 | 3 | 3 | 11 |
| Ch.10 Recommandations | 3 | 2 | 2 | 7 |
| Ch.11 Ecarts | 2 | 2 | 1 | 5 |
| Ch.12 Annexes | 5 | 2 | 0 | 7 |
| **Total** | **56** | **26** | **24** | **106** |

---

## 14. Requirements

- Python 3.10+
- `marp-cli` for PDF/HTML export (`npx @marp-team/marp-cli` or `npm install -g @marp-team/marp-cli`)
- Chromium/Chrome for PDF export (Puppeteer, used by marp-cli)
- Optional: `sentence-transformers` + `hdbscan` for embedding-based clustering (LLM mode)
- Optional: Ollama for LLM-assisted normalization

**Fallback:** Without `marp-cli`, the pipeline produces valid MARP Markdown (`.md`)
that can be opened in VS Code with the Marp extension.

---

## 15. Implementation Status

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| **M0** | Foundation: package, config (16 dataclasses), models (26 dataclasses, 9 enums), JSON schema | **Done** |
| **M1** | S1 kernels: folder scan, content extract (13 unit types), asset catalog | **Done** |
| **M2** | S2 planner: slide plan (budget allocation, meta-filter, heading dedup) + layout assign | **Done** |
| **M3** | S3 render: MARP Markdown generation (12 slide types) | **Done** |
| **M4** | CLI (`presenterctl`) + marp-cli export (`run_marp_cli()`) | **Done** |
| **M5** | S2 normalizer: deterministic heuristics (clustering, dedup, roles, importance, narrative arc) | **Done** (Phase 1) |
| **M6** | Custom themes: `koas-professional` CSS (300 lines, corporate palette) | **Done** (1 theme) |
| **M7** | MCP integration: 3 tools (`presenter_render`, `presenter_export`, `presenter_status`) | **Done** |
| **M9** | **v1.2 Layout Intelligence**: all-inline image rendering, table density CSS classes, path-based lookup | **Done** |
| **M10** | **v1.2 Structural Compression**: 3 modes (`full`/`compressed`/`executive`), annex exclusion, global dedupe, per-section cap, executive role filter | **Done** |
| M8 | Editable PPTX via `python-pptx` (S3-alt renderer) | Planned |

### v1.2.1 — All-Inline Image Rendering (2026-02-13)

- **All images now use inline HTML rendering** — eliminates MARP `![bg left:40%]` cropping
- New `figure-landscape` CSS class for landscape images (100% width top, 50vh max-height)
- `SlideLayout.image_class` field: `""` | `"figure"` | `"figure-landscape"` | `"figure-full"`
- Path-based asset lookup: `AssetCatalog.get_by_path()` with basename fallback
- Tuned CSS: `.figure` width 50%, `.figure-full` max-height 75vh
- IOWIZME: 22 landscape + 1 portrait inline, 0 MARP bg (figures now readable in PDF)
- New shared tool: `ragix_kernels/shared/md_renumber.py` — section/figure/table/cross-ref auto-renumbering

### v1.2 — Structural Compression (2026-02-12)

- `--compression full|compressed|executive` CLI flag
- Annex exclusion: cluster labels matching configurable patterns removed from budget
- Global cross-cluster deduplication (Jaccard threshold 0.80, stricter than intra-cluster 0.70)
- Per-section budget cap (default 8 in compressed mode)
- Executive role filter: FINDING/RECOMMENDATION/PROBLEM + high-importance IMAGE_REF/TABLE
- Executive auto-cap: 25 slides when no explicit `--max-slides`
- IOWIZME validation: full 120 -> compressed 60 (50%) -> executive 25 (79%)

### v1.1 — Layout Intelligence (2026-02-12)

- Aspect-ratio-aware image layout: portrait (w/h < 0.9) -> inline `<div class="figure">`, landscape -> MARP bg
- Table density scaling: `table-small` (cells > 40) and `table-tiny` (cells > 80) CSS classes
- `pres_layout_assign` now depends on `pres_asset_catalog` for image dimensions
- New CSS: `.figure`, `.figure-full`, `.table-small`, `.table-tiny` in `koas-professional.css`
- IOWIZME: 9 tables auto-scaled (7 small, 2 tiny), 0 inline images (all SVGs landscape)
- Superseded by v1.2.1 for image rendering (landscape images still used MARP bg in v1.1)

### v5 Quality Fixes (2026-02-12)

- Meta-cluster filter: exact match -> substring match (fixes "Table des tableaux principaux" leak)
- Heading dedup: "(N)" suffix for repeated headings within a section (0 duplicates)
- IMAGE_REF importance boost: +0.2 type boost + ILLUSTRATION role 0.05 -> 0.15 (24/26 SVGs included)

### Test Suites

| Suite | Scope | File |
|-------|-------|------|
| `test_m1_kernels.py` | S1: folder scan, content extract, asset catalog | 642 lines |
| `test_m2_kernels.py` | S2: normalization, slide plan, layout assign | 839 lines |
| `test_m3_kernels.py` | S3: MARP render, export | 585 lines |

---

## 16. Future Directions

### M5 Phase 2 — LLM-Assisted Normalization

The normalizer currently operates in deterministic mode only. Phase 2 would add:

- **Embedding-based clustering:** `sentence-transformers` (`all-MiniLM-L6-v2`) + HDBSCAN, replacing heading-path prefix grouping for messy/mixed-file corpora
- **LLM cluster refinement:** lightweight operations on cluster *summaries* (merge/split/relabel), bounded by `max_llm_calls` budget
- **LLM role classification:** replace keyword matching with contextual role assignment
- **LLM narrative arc:** generate speaker notes and presentation narrative
- **Tiered budget escalation:** T0 (identity) -> T1 (top-K refinement) -> T2 (all clusters + roles) -> T3 (full polish + speaker notes)

### M8 — Editable PPTX

A second S3 renderer consuming the same `SlideDeck` JSON contract:

- `python-pptx` for native PowerPoint generation
- Editable text, tables, and images (vs MARP's image-based PDF)
- Corporate template support (`.potx` master slides)
- Independent of marp-cli — no Node.js dependency

### Additional Themes

Planned CSS themes to complement `koas-professional`:

| Theme | Style | Use Case |
|-------|-------|----------|
| `koas-academic` | Serif fonts, formal layout | Conference talks, academic presentations |
| `koas-audit` | High-density tables, red/amber/green | Formal audit reports |
| `koas-workshop` | Informal, large fonts | Workshop recaps, training materials |

### Other Potential Extensions

- **`presenterctl preview`**: `marp-cli` server mode with live reload (`--watch` on `SlideDeck` JSON)
- **Outline-driven mode (Mode B)**: user-provided YAML outline -> deterministic slide selection via "outline projection"
- **Incremental regeneration**: cache with stable unit/cluster IDs, invalidate per-file on change
- **Mermaid pre-rendering**: `mmdc` integration for SVG generation from Mermaid blocks
- **Multi-language support**: localized section labels, auto-detected language from content

---

## 17. Related Documentation

| Document | Description |
|----------|-------------|
| [ROADMAP_KOAS_PRESENTER.md](developer/ROADMAP_KOAS_PRESENTER.md) | Full design roadmap (1,672 lines) |
| [ROADMAP2_KOAS_PRESENTER.md](developer/ROADMAP2_KOAS_PRESENTER.md) | v1.1-v1.2 roadmap: compression + layout intelligence |
| [KOAS.md](KOAS.md) | KOAS philosophy and code audit |
| [KOAS_DOCS.md](KOAS_DOCS.md) | Document summarization system |
| [KOAS_REVIEW.md](KOAS_REVIEW.md) | Document reviewer pipeline (v0.5.0) |
