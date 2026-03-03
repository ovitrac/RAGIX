# KOAS Presenter — Slide Deck Generation from Documents

**Author:** Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
**Version:** 2.1.0
**Date:** 2026-03-03
**Status:** Production (deterministic pipeline) | LLM normalizer: Phase 1 (deterministic heuristics) | Compression: v1.2 | Layout Intelligence: v1.2 | Typography & HTML Post-Processing: v2.0 | Hand-crafted presentations: v2.1

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
12. [MARP Post-Processing & HTML Export (v2.0)](#marp-post-processing--html-export-v20)
13. [Design Decisions](#design-decisions)
14. [Production Benchmarks](#production-benchmarks)
15. [Requirements](#requirements)
16. [Implementation Status](#implementation-status)
17. [Future Directions](#future-directions)
18. [Related Documentation](#related-documentation)

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

**Codebase:** 25 files, ~12,300 lines (8 kernels + shared post-processor, 26 dataclasses, 9 enums, 3 CLI subcommands, 3 MCP tools, 1 custom theme + typography config, 3 test suites). v2.0 adds 24-transform MARP post-processing, layout directives, and HTML export enhancements. v2.1 adds hand-crafted presentation workflow with companion handout generation and review integration.

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
  --toc / --no-toc             Enable/disable TOC slide generation (default: on) [v1.2.2]
  --postprocess / --no-postprocess  Enable/disable post-processing pipeline (default: on) [v1.3]
  --logos-dir <path>           Directory containing logo images (PNG/JPG) [v1.3]
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
  --title "Audit ACME-ERP" --author "Olivier Vitrac" \
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

**Source:** `ragix_kernels/presenter/themes/koas-professional.css` (~410 lines)

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

**Typography (v2.0):** Inter/Segoe UI (body), JetBrains Mono/Fira Code (code). Base font: 22px (was 26px). h2: 1.5em (33px). Blockquote: 0.88em (19.4px). Page numbers: 12px. Tables: 0.82em base. All font-size rules use `!important` (MARP `@import 'default'` re-emits rules with identical specificity; only `!important` reliably wins).

### Custom Theme Development

Themes are resolved by `_resolve_theme_css()` in `pres_marp_export.py`:

1. `theme.custom_css_path` (absolute or relative) — highest priority
2. Bundled theme by name: `ragix_kernels/presenter/themes/{name}.css`
3. `None` — marp-cli built-in default

Themes must style all 12 slide types. The MARP renderer emits `<!-- _class: X -->` directives.

```
ragix_kernels/presenter/themes/
+-- koas-professional.css   # Corporate identity (implemented, ~410 lines)
+-- koas-typography.yaml    # Typography hierarchy & layout parameter reference
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

## 12. MARP Post-Processing & HTML Export (v2.0)

The v2.0 release introduces a deterministic post-processing pipeline and HTML export
enhancements that solve three classes of MARP rendering issues: image centering, layout
table collapse, and asset portability.

**Source:** `ragix_kernels/shared/marp_postprocess.py` (~3,150 lines, 24 transforms)

### Problem: MARP CSS Specificity

MARP prefixes all theme CSS selectors with `div#\:\$p > svg > foreignobject > section`,
giving them specificity **(1, 0, 5)**. This defeats:

- Global `<style>` blocks (specificity ~(0, 0, 2))
- Scoped `<style scoped>` blocks
- CSS classes defined in the theme

**Only inline `style` attributes** reliably override the theme at render time. The
post-processing pipeline injects these inline styles at the Markdown level (before
marp-cli), ensuring fixes apply to both HTML and PDF output.

### Post-Processing Transforms (24 steps)

| # | Transform | Version | Purpose |
|---|-----------|---------|---------|
| 0 | `strip_postprocess_artifacts` | v1.5 | Idempotency guard (removes prior injections) |
| 1 | `rewrite_title_slide` | v1.0 | Styled title with metadata |
| 2 | `strip_heading_numbers` | v1.0 | Remove "1.2.3" section prefixes |
| 3 | `strip_heading_pipes` | v1.5 | Remove `## \| Title` pipe artifacts |
| 4 | `normalize_section_dividers` | v1.0 | "Chapitre N — Title" format |
| 5 | `renumber_chapters` | v1.4 | Sequential `Chapitre N` numbering after "# Other" removal |
| 6 | `strip_source_comments` | v1.0 | Remove provenance `<!-- source: ... -->` comments |
| 7 | `strip_navigation_rapide` | v1.4 | Remove `> **Navigation rapide** : ...` blockquotes |
| 8 | `fix_singleton_numbered_lists` | v1.0 | "1." → "-" (single-item lists) |
| 9 | `remove_trailing_sommaire` | v1.0 | Drop trailing TOC artifacts |
| 10 | `remove_garbled_sommaire` | v1.5 | Remove lead+content garbled Sommaire blocks |
| 11 | `remove_empty_chapter_dividers` | v1.5 | Drop consecutive section dividers (empty chapters) |
| 12 | `clean_toc_slide` | v1.0 | Format table of contents (v1.5: pipe stripping, Sommaire detection) |
| 13 | `layout_preprocess` | v2.0 | Image dimension probing, shape classification, auto-layout |
| 14 | `expand_layout_directives` | v2.0 | `[I,T]`/`[T,I]`/`[I;T]`/`[I,I;t,t]` → inline HTML tables |
| 15 | `compact_layout_slides` | v2.0 | Reduce vertical padding on layout-directive slides |
| 16 | `auto_classify_tables` | v1.0 | `table-small`/`table-tiny` CSS classes |
| 17 | `auto_constrain_figure_slides` | v1.3 | Scoped `max-height` for figure+text slides |
| 18 | `auto_shrink_dense_slides` | v1.3 | Font-size reduction for text-heavy slides (TOC exempt, cascading cap) |
| 19 | `inject_progress_bar` | v1.0 | Orange progress bar (CSS counter) |
| 20 | `inject_chapter_nav` | v1.0 | Chapter navigation overlay |
| 21 | `inject_chapter_footer` | v1.0 | Footer with chapter name |
| 22 | `inject_traceability_slide` | v1.0 | Provenance metadata slide |
| 23 | `inject_logos` | v1.0 | Company logos on lead slides |

### Layout Directives

Authors can use shorthand HTML comments instead of hand-crafting inline HTML tables.
The `expand_layout_directives` transform (step 8) expands these into full inline-style
HTML with `display:table !important` to survive MARP's CSS sanitization.

**Syntax:**

```markdown
<!-- layout: [I,T] -->
<!-- I: assets/diagram.svg | alt: Architecture | h: 400px -->

**Text content** goes here — tables, lists, paragraphs.

<!-- /layout -->
```

**Supported layouts:**

| Directive | Pattern | Description |
|-----------|---------|-------------|
| `[I,T]` | Side-by-side | Image left (38%), text right (62%) |
| `[T,I]` | Side-by-side reversed | Text left, image right |
| `[I;T]` | Vertical stack | Image on top (constrained height), text below |
| `[I,I;t,t]` | 2×2 grid | Two images top row, two text blocks bottom |

**Image parameters:** `<!-- I: path | alt: text | h: NNNpx | w: NN% -->`

### Layout Pre-Processing (Image Dimension Probing)

The `layout_preprocess()` function (called before `postprocess_marp()`) probes image
dimensions to auto-detect optimal layout:

- Portrait images (h/w > 1.2) → `[I,T]` side-by-side
- Landscape images (w/h > 1.5) → `[I;T]` vertical stack
- Multiple images per slide → `[I,I;t,t]` grid

This allows the pipeline to start from a plain Markdown file with `![](image.png)`
references and auto-generate appropriate layout directives.

### HTML Post-Processing (after marp-cli)

Three functions fix issues that can only be addressed in the final HTML output:

| Step | Function | Purpose |
|------|----------|---------|
| 1 | `center_images_in_html()` | MARP strips `display:block`/`margin:auto` — re-injects centering |
| 2 | `fix_layout_tables_in_html()` | MARP sets `display:block` on `<table>` — restores `display:table` |
| 3 | `embed_images_in_html()` | Base64 data URIs for self-contained HTML (no external assets) |

**Image embedding details:**
- Raster images downscaled to max 2,000px (≈200 DPI on 16:9 slides)
- Transparent PNGs stay PNG; opaque images convert to JPEG (quality 85)
- SVGs embedded as `data:image/svg+xml;base64,...`
- Requires Pillow (`PIL`) for raster processing

### `_TABLE_BASE` — Layout Table Inline Styles

All layout tables use a shared constant for inline styles:

```
display:table !important;border:none;border-collapse:collapse;
width:100%;margin:0;background:transparent
```

The `display:table !important` is critical: without it, MARP's theme CSS sets
`display:block` on `<table>` elements, causing `<tr>` to shrink to content width
instead of spanning the full slide. This fix works in both HTML and PDF because the
inline style is present in the Markdown source, surviving Puppeteer rendering.

### Integration in `pres_marp_export.py` (v2.0.0)

The export kernel now calls HTML post-processing automatically after marp-cli:

```python
# ExportConfig fields (v2.0)
center_images: bool = True           # Fix image centering
fix_layout_tables: bool = True       # Fix display:block → display:table
embed_images: bool = False           # Base64 embedding (large file size)
embed_max_dim: int = 2000            # Max pixel dimension for raster images
embed_jpeg_quality: int = 85         # JPEG quality for opaque images
```

The `--html` flag is now passed to marp-cli for PDF export as well, ensuring HTML tags
in Markdown (layout tables, styled divs) are rendered correctly in PDF output.

### Build Pipeline (Standalone Mode)

For hand-crafted presentations (outside the `presenterctl` pipeline), a `build.sh`
script orchestrates the full build:

```
presentation.md
  │ (auto-refresh .bak when source changes)
  ▼
presentation.md.bak
  │ (layout_preprocess — image dimension probing)
  ▼
presentation_layout.md        ◄── optional manual review
  │ (postprocess_marp — 16 deterministic transforms)
  ▼
presentation_pp.md
  │ (npx @marp-team/marp-cli)
  ▼
presentation_postprocessed.html
  │ (center_images + fix_layout_tables + embed_images)
  ▼
Self-contained HTML + PDF
```

Key build features:
- Auto-refresh `.bak` from `presentation.md` when source is newer
- Stale layout file invalidation when source changes
- Default pipeline: source → layout → postprocess → HTML + PDF (embedded images)
- Two-step review: `--layout-only` generates layout for manual tuning, `--from-layout` rebuilds

---

## 13. Design Decisions

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
| D13 | `display:table !important` in Markdown source | Survives Puppeteer PDF rendering; fixes HTML and PDF in one place |
| D14 | HTML post-processing after marp-cli | Image centering and layout table fixes cannot be done at Markdown level (MARP strips them) |
| D15 | Base64 image embedding as opt-in | Self-contained HTML for sharing; large file size trade-off (~5-10x) |

---

## 14. Production Benchmarks

### Production Benchmark: ACME-ERP Audit (Auto-Generated)

Full pipeline run on a 14-document French technical audit report (2,831 units, 149K tokens, 27 SVG figures).

### Input Corpus

| Metric | Value |
|--------|-------|
| Source folder | `<audit_workspace>/ACME-ERP/report/` |
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
| Ch.6 Analyse ECH-MSG-HUB | 6 | 3 | 2 | 11 |
| Ch.7 Revue documentaire | 4 | 3 | 3 | 10 |
| Ch.8 Conformite RIE | 3 | 1 | 1 | 5 |
| Ch.9 Estimation MCO | 5 | 3 | 3 | 11 |
| Ch.10 Recommandations | 3 | 2 | 2 | 7 |
| Ch.11 Ecarts | 2 | 2 | 1 | 5 |
| Ch.12 Annexes | 5 | 2 | 0 | 7 |
| **Total** | **56** | **26** | **24** | **106** |

### Production Benchmark: SAT-AUDIT Restitution (Hand-Crafted)

The SAT-AUDIT project demonstrates the **hand-crafted presentation workflow** — where the
pipeline acts as a post-processing and export layer for manually authored MARP Markdown,
rather than generating slides from a document corpus.

**Context:** 30-slide restitution for a Java application separation audit, derived from
a 4,359-line technical report and a 1,430-line architecture appendix.

| Metric | Value |
|--------|-------|
| Source | `presentation.md` — 770 lines, hand-crafted MARP Markdown |
| Report | Technical audit report — 4,359 lines, 13 sections + 8 annexes |
| Assets | 31 files (22 SVGs + 6 PNGs + 3 logos) |
| Slides | 35 (30 content + TOC + section dividers + traceability) |
| Layout directives | 12 (5 manual `[I,T]` + 7 auto-detected) |
| Images centered (HTML post-processing) | 18 |
| HTML size | ~200 KB |
| Build time | ~1s HTML-only, ~3s HTML+PDF |

**Companion deliverables:**

| Deliverable | File | Description |
|-------------|------|-------------|
| Handout/Glossary | `handout_glossaire.md` | 179 lines, ~80 acronyms in 8 sections, printable 3-4 A4 pages |
| Changelog | `CHANGELOG.md` | Version history (v2, v1 enrichie, v1, handout) |

**Hand-crafted workflow features:**

- **Two-step build pipeline**: `.bak` → `layout_preprocess()` → `postprocess_marp()` → marp-cli → HTML post-processing
- **Review integration**: diff reviewer's file, cherry-pick fixes only, preserve enrichments
- **Handout generation**: standalone glossary document extracted from source report (~80 curated acronyms from 200+)
- **Layout directives**: authors write `<!-- layout: [I,T] -->` shorthand instead of 15-attribute inline HTML
- **Build flags**: `--layout-only` (review), `--from-layout` (rebuild), `--no-auto-layout`, `--embed`

---

## 15. Requirements

- Python 3.10+
- `marp-cli` for PDF/HTML export (`npx @marp-team/marp-cli` or `npm install -g @marp-team/marp-cli`)
- Chromium/Chrome for PDF export (Puppeteer, used by marp-cli)
- Optional: `sentence-transformers` + `hdbscan` for embedding-based clustering (LLM mode)
- Optional: Ollama for LLM-assisted normalization
- Optional: Pillow (`PIL`) for base64 image embedding in HTML export (v2.0)

**Fallback:** Without `marp-cli`, the pipeline produces valid MARP Markdown (`.md`)
that can be opened in VS Code with the Marp extension.

---

## 16. Implementation Status

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
| **M11** | **v2.0 MARP Post-Processing & HTML Export**: 24-transform pipeline, layout directives, HTML post-processing (image centering, layout table fix, base64 embedding), `pres_marp_export` v2.0.0, standalone build pipeline | **Done** |
| **M12** | **v2.1 Hand-Crafted Presentation Workflow**: companion handout generation, review integration (diff + cherry-pick), layout directives for manual authoring, two-step build pipeline | **Done** |
| M8 | Editable PPTX via `python-pptx` (S3-alt renderer) | Planned |

### v2.1.0 — Hand-Crafted Presentation Workflow (2026-03-03)

- **Hand-crafted presentation support**: the pipeline now works as a post-processing and export layer for manually authored MARP Markdown, not just auto-generated slide decks
- **Companion handout generation**: standalone glossary/aide-mémoire documents extracted from the source report, with curated acronym lists organized by domain (e.g., ~80 acronyms from a 200+ extraction)
- **Review integration workflow**: unified diff of reviewer's changes, selective cherry-picking of fixes (terminology, accents, factual corrections) while preserving layout and enrichments
- **Changelog tracking**: structured version history with per-slide change references
- **Production validation**: SAT-AUDIT restitution — 35 slides, 12 layout directives (5 manual + 7 auto), 31 assets, ~200K HTML
- **Typography refinement**: base font 26→22px, h2 ratio 1.3→1.5:1 over body, blockquote 0.88em, page numbers 12px
- **`koas-typography.yaml`**: configuration reference file documenting the full typography hierarchy and layout parameters

### v2.0.0 — MARP Post-Processing & HTML Export (2026-03-03)

- **24-transform deterministic pipeline** in `marp_postprocess.py` (~3,150 lines, zero LLM)
- **Layout directives**: `[I,T]`, `[T,I]`, `[I;T]`, `[I,I;t,t]` → inline HTML tables with `display:table !important`
- **Layout pre-processing**: image dimension probing, shape classification, auto-layout assignment
- **HTML post-processing** (3 functions integrated into `pres_marp_export.py` v2.0.0):
  - `center_images_in_html()` — fix MARP stripping `display:block`/`margin:auto`
  - `fix_layout_tables_in_html()` — restore `display:table` (MARP sets `display:block`)
  - `embed_images_in_html()` — base64 data URIs for self-contained HTML
- **`_TABLE_BASE` with `display:table !important`** — fixes layout tables in both HTML and PDF
- **`ExportConfig` v2.0**: 5 new fields (`center_images`, `fix_layout_tables`, `embed_images`, `embed_max_dim`, `embed_jpeg_quality`)
- **`run_marp_cli()` fix**: passes `--html` flag for PDF export (enables HTML tags in MARP Markdown)
- **Standalone build pipeline**: auto-refresh `.bak`, stale layout invalidation, default embedded HTML
- **SAT-AUDIT restitution**: 35 slides, self-contained HTML, embedded images

### v1.5.0 — Idempotency & Content Cleanup (2026-03-02)

- **`strip_postprocess_artifacts()`** — idempotency guard at pipeline start; strips previously injected progress bars, chapter nav, footer, CSS, scoped styles
- **`strip_heading_pipes()`** — removes `## | Title` pipe artifacts
- **`remove_garbled_sommaire()`** — removes lead+content garbled Sommaire blocks with self-refs
- **`remove_empty_chapter_dividers()`** — drops consecutive section dividers (empty chapters), triggers `renumber_chapters()`
- **`clean_toc_slide()`** v1.5: `## Sommaire` detection, pipe artifact + bold marker stripping from TOC entries
- **Pipeline robustness**: `_parse_marp()`/`_join_marp()` replaces `str.split("---")` for artifact-resilient slide parsing

### v1.4.0 — Content Cleanup Transforms (2026-03-02)

- **`strip_navigation_rapide()`** — removes `> **Navigation rapide** : ...` blockquotes (document cross-refs); drops heading-only slides emptied after removal
- **`renumber_chapters()`** — 2 bug fixes: "# Other" detection checks ANY line in slide; single-pass `re.sub` with callback prevents double-rename on overlapping chapter numbers
- **`normalize_section_dividers()`** — strips bold markers (`**`) from titles
- **`_BARE_NUM_HEADING_RE`** regex extended to match `# N | Title` pipe separator format

### v1.3.0 — Layout Fixes (2026-03-01)

- **`auto_constrain_figure_slides()`** — scoped `max-height` (38vh/42vh/55vh) for figure+text slides preventing overflow
- **`auto_shrink_dense_slides()`** — TOC exemption + cascading shrink cap (0.88em for table slides)
- **Table readability**: `_TABLE_SCALE` harmonized with `max()` floors; `table-small` 0.65em, `table-tiny` 0.50em
- **`_PROGRESS_CSS`** baselines: landscape 44vh, portrait 48vh, full 62vh
- **`koas-professional.css`**: bottom padding 40→60px, `.fig-caption` class
- **Pipeline integration**: `postprocess_marp()` called automatically in `pres_marp_export.py`; `--postprocess/--no-postprocess` CLI flag

### v1.2.1 — All-Inline Image Rendering (2026-02-13)

- **All images now use inline HTML rendering** — eliminates MARP `![bg left:40%]` cropping
- New `figure-landscape` CSS class for landscape images (100% width top, 50vh max-height)
- `SlideLayout.image_class` field: `""` | `"figure"` | `"figure-landscape"` | `"figure-full"`
- Path-based asset lookup: `AssetCatalog.get_by_path()` with basename fallback
- Tuned CSS: `.figure` width 50%, `.figure-full` max-height 75vh
- ACME-ERP: 22 landscape + 1 portrait inline, 0 MARP bg (figures now readable in PDF)
- New shared tool: `ragix_kernels/shared/md_renumber.py` — section/figure/table/cross-ref auto-renumbering

### v1.2 — Structural Compression (2026-02-12)

- `--compression full|compressed|executive` CLI flag
- Annex exclusion: cluster labels matching configurable patterns removed from budget
- Global cross-cluster deduplication (Jaccard threshold 0.80, stricter than intra-cluster 0.70)
- Per-section budget cap (default 8 in compressed mode)
- Executive role filter: FINDING/RECOMMENDATION/PROBLEM + high-importance IMAGE_REF/TABLE
- Executive auto-cap: 25 slides when no explicit `--max-slides`
- ACME-ERP validation: full 120 -> compressed 60 (50%) -> executive 25 (79%)

### v1.1 — Layout Intelligence (2026-02-12)

- Aspect-ratio-aware image layout: portrait (w/h < 0.9) -> inline `<div class="figure">`, landscape -> MARP bg
- Table density scaling: `table-small` (cells > 40) and `table-tiny` (cells > 80) CSS classes
- `pres_layout_assign` now depends on `pres_asset_catalog` for image dimensions
- New CSS: `.figure`, `.figure-full`, `.table-small`, `.table-tiny` in `koas-professional.css`
- ACME-ERP: 9 tables auto-scaled (7 small, 2 tiny), 0 inline images (all SVGs landscape)
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

## 17. Future Directions

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

## 18. Related Documentation

| Document | Description |
|----------|-------------|
| [ROADMAP_KOAS_PRESENTER.md](developer/ROADMAP_KOAS_PRESENTER.md) | Full design roadmap (1,672 lines) |
| [ROADMAP2_KOAS_PRESENTER.md](developer/ROADMAP2_KOAS_PRESENTER.md) | v1.1-v1.2 roadmap: compression + layout intelligence |
| [KOAS.md](KOAS.md) | KOAS philosophy and code audit |
| [KOAS_DOCS.md](KOAS_DOCS.md) | Document summarization system |
| [KOAS_REVIEW.md](KOAS_REVIEW.md) | Document reviewer pipeline (v0.5.0) |
