# Renderable regions: source-exact JSON

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

A line or exact `CellSpan` can request its enclosing `PROSE`, `TABLE`, `LIST` or
`FIGURE` region. This is an explicit library envelope over existing observations.
It does not change extraction, select an answer, or generate HTML or CSS.

```python
from ragix_kernels.saqqara.renderable_regions import regions_from_explorer
from ragix_kernels.harvest.regions import RegionWindow

index = regions_from_explorer(explorer_result)
region = index.get(line_id, window=RegionWindow(before=1, after=1, same_page=True))
payload = region.to_dict()       # JSON-compatible structured data
encoded = region.to_json()      # deterministic UTF-8 JSON; no text normalization
```

The same `get` accepts a `CellSpan` from stored table context. Source, table, page,
box and character slice are checked; a stale or foreign span is refused.

## Contract

The JSON root carries `region_id`, `source_id`, `page`, `bbox`, `kind`, `rule`,
`anchor_member_id`, `anchor`, ordered `members`, `neighbourhood`, `flags` and the
complete `boundary_policy`. Every member retains its original text, box, page and
source span ids. No whitespace, Unicode, hyphen or lifecycle glyph is cleaned.
Unchanged source inputs, policy and request produce byte-identical JSON.

The anchor is the caller's requested member/slice, not a relevance or evidence
judgment. Region identity does not depend on which member was requested or on the
neighbourhood window. Neighbours have `usage: CONTEXT_ONLY` and no selected anchor.
Their ids are for navigation and provenance; a consumer must not silently promote
them into the caller's cited evidence.

Coordinates are page points with a top-left origin. A cross-page region has
`CROSSES_PAGE`, member-level pages, `pages` and separate `page_boxes`. Its root
`page` and `bbox` describe the first page only; coordinates from different pages
are never flattened into one fictitious page.

A table includes all observed cells, their row/column indices, spans, header and
positional-stub flags. Table source lines are retained as `LINE` members flagged
`SOURCE_LINE_ALIAS`, so a line-based request keeps its original scored id. Render
the table structure from `CELL` members and use the alias geometry for highlighting;
the alias is not an additional cell. A lattice without observed cells cannot
become a table. Unreadable cells and lifecycle contamination stay explicit.

Lists retain the original marker text. `list_item` and `list_depth` describe the
derived item/indentation structure, including wrapped lines. Explicit headings
are structural boundaries; an isolated numbered heading is not automatically a
list. Overlapping figure/table ownership refuses an ambiguous line anchor.

## Boundary policy and measured baselines

`BoundaryPolicy` declares geometry thresholds, page-edge bounds and cross-page
linking. Known heading line ids, section/column assignments and explicit
continuation pairs can be supplied to `regions_from_explorer`. A paragraph is not
inferred merely because its lines share a section. Headings, tables and figures
stop an ordinary prose join. Uncertain page continuations are flagged; explicit
continuation links can resolve them without changing the source text.

The independently planted fixture inventory has five layouts: two paragraphs
inside a section, parallel columns, an indented opener, a heading boundary and a
paragraph crossing a page break. Exact grouping scores:

| Policy | Exact layouts |
|---|---:|
| One region per visual line | 0/5 |
| Section-only grouping | 2/5 |
| Declared geometry with structural boundaries | 5/5 |

The geometry result stays invariant over the declared sweep: gap ratios
0.5/1.0/1.5, indentation ratios 1.0/1.5/2.0 and page-edge fractions 0.05/0.12/0.2
(135 layout evaluations). These are synthetic falsifiers, not a measured accuracy
claim for arbitrary documents. Ambiguous geometry needs explicit structural hints.
Reading order is the declared page/top/left order; complex layouts should supply
column/section context rather than expect semantic reading-order inference.

## Neighbourhood and refusals

The requested before/after counts and verbatim generated `window_rule` are included
in the result. With `same_page=True`, siblings must touch the anchor's page. A
short window at a page/document edge always carries `TRUNCATED_AT_WINDOW`; zero
requested neighbours do not. Missing source, ambiguous ownership, invalid topology
and resource-budget excess raise `RegionRefused` with a stable reason code.

Explorer table assembly isolates a refusal to the affected table. Other tables,
prose, lists and figures remain available. Omitted tables appear in
`index.refusals`, with source id, table id, pages, observed member ids, stable
`code`, and rule `table-region/1`. `index.refusal_report()` returns these records
and their count as ordinary JSON-compatible data. Check this report when assessing
coverage; a returned index does not assert that every table was rendered.

Native band geometry can refuse with `STRADDLING_OR_OUTSIDE_BANDS` (including a
small gap or overlap), `BAND_COUNT_VARIES` (including a spanning cell that prevents
unambiguous header mapping), or `TABLE_COLUMN_LAYOUT_UNAVAILABLE` (including an
empty physical row). Missing source members, header geometry, header columns or
cell geometry use `TABLE_SOURCE_MEMBER_MISSING`, `TABLE_HEADER_GEOMETRY_UNAVAILABLE`,
`TABLE_HEADER_COLUMN_UNAVAILABLE`, and `TABLE_CELL_GEOMETRY_UNAVAILABLE`.
These are typed `RegionRefused` failures. Geometry tolerances are unchanged;
unsupported topology is recorded rather than guessed. Direct `table_members`
calls stay strict unless the caller supplies a `refusals` list to receive omissions.
Document-level identity, stale presentation and invalid context failures still
refuse the call.

The data-only gate is strict: literal tags, entities or CSS/style tokens in any
selected member or neighbour refuse that response. They are never escaped into
changed source text or silently omitted. JSON quoting itself is ordinary JSON
serialization, not HTML escaping. The object is a source envelope, not a sanitizer.

## Figures and base64

Raster support uses the optional `regions` extra:

```bash
python -m pip install '.[regions]'
```

`image_payload(bytes, media_type)` and `image_from_store(asset_store, asset_id)`
produce an image object with `encoding: base64`, `media_type`, `data`, checksum,
byte count, dimensions and original asset identity. PNG and JPEG are validated and
fully decoded within declared byte/pixel limits using the [Pillow image API](https://pillow.readthedocs.io/en/stable/reference/Image.html).
SVG, animated images, mismatched hashes and malformed rasters are refused. The
base64 decodes to the supplied raster bytes exactly.

`figures_from_tree` reuses existing `figure` and `vector_region` nodes and the
asset store. An explicit `to_region_box(page, pdf_box)` converts the reader's
coordinates; there is no implicit PDF-user-space/top-left conversion. Captions can
be linked by node address and original line id. A vector lattice with an existing
raster remains `FIGURE`, flagged `LATTICE_AS_IMAGE`, never a reconstructed table.
An upstream inferred figure retains `FIGURE_STRUCTURE_INFERRED`; this envelope
does not promote its origin to a direct observation.
A figure's `figure_bbox` positions the raster separately from a caption that may
extend the enclosing region's `bbox`.

For vector/raw assets, one uniquely associated stored raster is reused. Multiple
rasters require an explicit `raster_choices` selection. Missing or unsupported
images carry `image_reason` and `FIGURE_IMAGE_UNAVAILABLE`; missing/corrupt source
assets remain failures, not apparent absence.

An optional existing renderer port can be supplied with `source_path` and `dpi`.
Only the unavailable-raster case uses it. The source file must hash to the declared
source id before and after rendering, and the image records the renderer, version
and DPI. No renderer is loaded or invoked by default. No model or OCR is called.
Neither source paths nor provider endpoints appear in the returned envelope.

Default limits: 100,000 members, 16 MiB JSON output, 8 MiB per raster, 64 MiB per
source asset and 16 million pixels per raster. They are configurable through
`RegionLimits`. Budget failures never return an unflagged partial region.

The data contains original source text and image bytes, including any source
metadata. Privacy masking, image redaction, display rendering and citation policy
remain consumer responsibilities. This API does not grant publication clearance.

## Reproduce

```bash
python -m pytest tests/harvest/test_region_boundaries.py \
  tests/harvest/test_renderable_regions.py tests/saqqara/test_renderable_region_adapters.py -q
python -m tests.harvest.renderable_regions_replay --output replay.json
python -m pytest tests/ -q
```

Replay compares structured case payloads across architectures. Its separate
provenance records source and library versions. Existing Explorer output and kernel
cache versions are unchanged because this envelope runs only when explicitly called.
