"""Envelope adapters over Explorer observations and existing figure assets.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import replace
import hashlib
from pathlib import Path
from ..harvest.regions import (
    RegionIndex,
    RegionMember,
    PageGeometry,
    BoundaryPolicy,
    RegionLimits,
    FigureInput,
    RegionRefused,
    TableRegionRefusal,
    image_from_store,
    image_payload,
    identity,
)
from ..harvest.table_context import context_from_dict
from .census import page_lines
from .table_views import _native_header_bands, _map


def _member(
    source, table_id, page, row, column, cell, *, header=False, label=False, column_span=1, flags=()
):
    fs = set(cell.flags) | set(flags)
    if cell.text is None:
        fs.add("UNREADABLE_CELL")
    return RegionMember(
        cell.cell_id,
        source,
        page,
        "CELL",
        cell.text,
        cell.bbox,
        cell.source_spans,
        tuple(sorted(fs)),
        table_id,
        row,
        column,
        1,
        column_span,
        header,
        label,
    )


def _recovered_members(document, table, physical, labels, observations, known):
    members = {}
    for row_index, row in enumerate(table.rows, 1):
        flags = dict(row.cell_flags)
        for column, ids in enumerate(row.members):
            for ident in ids:
                if ident not in physical:
                    raise RegionRefused("TABLE_SOURCE_MEMBER_MISSING")
                c = physical[ident]
                members[ident] = _member(
                    document.source_id,
                    table.table_id,
                    row.page,
                    row_index,
                    column,
                    c,
                    label=ident in labels,
                    flags=flags.get(ident, ()),
                )
    for fragment in table.fragments:
        origin = table.policy.get("inherited_headers", {}).get(fragment, fragment)
        observed = observations.get(origin)
        if observed is None or not observed.cell_rows or not observed.cell_rows[0]:
            raise RegionRefused("TABLE_HEADER_GEOMETRY_UNAVAILABLE")
        row = observed.cell_rows[0]
        left, right = min(c.bbox[0] for c in row), max(c.bbox[2] for c in row)
        if not table.bands or table.bands[-1][1] <= table.bands[0][0]:
            raise RegionRefused("TABLE_COLUMN_LAYOUT_UNAVAILABLE")
        a, b = table.bands[0][0], table.bands[-1][1]
        bands = tuple(
            (
                left + (x - a) / (b - a) * (right - left),
                left + (y - a) / (b - a) * (right - left),
            )
            for x, y in table.bands
        )
        for c in row:
            if c.cell_id in members:
                continue
            if c.cell_id in known:
                column = known[c.cell_id].column
                span = known[c.cell_id].column_span
            else:
                columns = [
                    i for i, (x, y) in enumerate(bands) if min(c.bbox[2], y) > max(c.bbox[0], x)
                ]
                if not columns:
                    raise RegionRefused("TABLE_HEADER_COLUMN_UNAVAILABLE")
                column, span = columns[0], columns[-1] - columns[0] + 1
            members[c.cell_id] = _member(
                document.source_id,
                table.table_id,
                observed.page,
                0,
                column,
                c,
                header=True,
                column_span=span,
                flags=table.flags,
            )
    return tuple(members.values())


def _observed_members(document, table, known, headers, labels):
    if any(
        c.geometry_kind != "cell_box" or "MISSING_CELL_GEOMETRY" in c.flags
        for row in table.cell_rows
        for c in row
    ):
        raise RegionRefused("TABLE_CELL_GEOMETRY_UNAVAILABLE")
    rows = table.cell_rows
    bands = _native_header_bands(rows[0], rows[1:])
    if not bands:
        raise RegionRefused("TABLE_COLUMN_LAYOUT_UNAVAILABLE")
    members = []
    for r, row in enumerate(rows):
        try:
            mapped = _map(row, bands, 0)
        except ValueError as error:
            raise RegionRefused("TABLE_COLUMN_LAYOUT_UNAVAILABLE") from error
        for column, group in enumerate(mapped):
            for c in group:
                k = known.get(c.cell_id)
                members.append(
                    _member(
                        document.source_id,
                        table.table_id,
                        table.page,
                        r,
                        column,
                        c,
                        header=c.cell_id in headers or (r == 0 and bool(any(table.headers))),
                        label=c.cell_id in labels,
                        column_span=k.column_span if k else 1,
                    )
                )
    return tuple(members)


def _table_refusal(error, sink, document, table_id, fragments, pages, observations):
    if sink is None:
        raise error
    member_ids = tuple(
        sorted(
            {
                cell.cell_id
                for fragment in fragments
                if fragment in observations
                for row in observations[fragment].cell_rows
                for cell in row
            }
        )
    )
    sink.append(
        TableRegionRefusal(document.source_id, table_id, tuple(pages), error.code, member_ids)
    )


def table_members(result, *, refusals=None):
    """Keep strict direct reads; a supplied sink enables counted per-table refusal."""
    document = result.document
    observations = {t.table_id: t for p in document.pages for t in p.tables}
    physical = {c.cell_id: c for t in observations.values() for row in t.cell_rows for c in row}
    contexts = [context_from_dict(c) for c in result.reading.cell_contexts]
    known = {
        c.cell_id: c for ctx in contexts for c in (ctx.value, *ctx.column_headers, *ctx.row_labels)
    }
    headers = {c.cell_id for ctx in contexts for c in ctx.column_headers}
    labels = {c.cell_id for ctx in contexts for c in ctx.row_labels}
    tables, used = [], set()
    for table in result.census.table_analysis.tables:
        used.update(table.fragments)
        try:
            tables.append(
                _recovered_members(document, table, physical, labels, observations, known)
            )
        except RegionRefused as error:
            _table_refusal(
                error,
                refusals,
                document,
                table.table_id,
                table.fragments,
                table.pages,
                observations,
            )
    for table in observations.values():
        if table.table_id in used or table.table_id in result.census.table_analysis.excluded:
            continue
        if not table.cell_rows:
            continue  # Strokes or flattened strings are not observed cells.
        try:
            tables.append(_observed_members(document, table, known, headers, labels))
        except RegionRefused as error:
            _table_refusal(
                error,
                refusals,
                document,
                table.table_id,
                (table.table_id,),
                (table.page,),
                observations,
            )
    return tuple(tables)


def regions_from_explorer(
    result,
    *,
    figures=(),
    headings=(),
    sections=None,
    columns=None,
    continuations=(),
    policy=BoundaryPolicy(),
    limits=RegionLimits(),
):
    """Reuse actual source lines, cells and furniture decisions; perform no extraction."""
    if result.reading is None or getattr(result.report, "status", None) == "FAILED":
        raise RegionRefused("EXPLORER_RESULT_UNREADABLE")
    sections = sections or {}
    columns = columns or {}
    headings = set(headings)
    excluded = set(result.reading.furniture)
    presentation = {p["line_id"]: p for p in result.report.presentation_lines}
    lines = []
    for page in result.document.pages:
        for view in page_lines(page):
            if view.view_id in excluded:
                continue
            stored = presentation.get(view.view_id)
            if stored is None or stored["text"] != view.text:
                raise RegionRefused("STALE_PRESENTATION_LINE")
            lines.append(
                RegionMember(
                    view.view_id,
                    view.source_id,
                    view.page,
                    "HEADING" if view.view_id in headings else "LINE",
                    view.text,
                    view.bbox,
                    tuple(dict.fromkeys(r.span_id for r in view.mapping if r is not None)),
                    view.flags,
                    section_id=sections.get(view.view_id),
                    column_id=columns.get(view.view_id),
                )
            )
    if not headings <= {m.member_id for m in lines}:
        raise RegionRefused("UNKNOWN_HEADING_MEMBER")
    refusals = []
    tables = table_members(result, refusals=refusals)
    return RegionIndex(
        result.document.source_id,
        tuple(PageGeometry(p.page, p.width, p.height) for p in result.document.pages),
        lines,
        tables=tables,
        refusals=refusals,
        figures=figures,
        policy=policy,
        limits=limits,
        continuations=continuations,
    )


def figures_from_tree(
    tree,
    store,
    *,
    source_id,
    to_region_box,
    caption_links=None,
    raster_choices=None,
    limits=RegionLimits(),
    renderer=None,
    source_path=None,
    dpi=150,
):
    """Reuse figure observations and assets; rendering is an explicit optional port.

    `to_region_box(page, pdf_box)` is an explicit coordinate transform supplied by
    the caller; PDF user-space coordinates must never be silently mixed with the
    top-left coordinates of Explorer. Missing geometry refuses the adapter; missing
    raster bytes are represented by a figure's explicit image_reason.
    """
    if tree.root.provenance.source_sha256 not in (None, source_id):
        raise RegionRefused("FIGURE_TREE_SOURCE_MISMATCH")
    caption_links = caption_links or {}
    raster_choices = raster_choices or {}
    figures = []
    source = Path(source_path) if source_path is not None else None
    if renderer is not None:
        if source is None or type(dpi) is not int or dpi <= 0:
            raise RegionRefused("RENDER_SOURCE_AND_DPI_REQUIRED")
        with source.open("rb") as stream:
            if _file_hash(stream) != source_id:
                raise RegionRefused("RENDER_SOURCE_HASH_MISMATCH")

    def walk(node, address, page=None):
        nonlocal figures
        for locator in node.provenance.chain:
            value = getattr(locator, "page", None)
            if value:
                page = value
        if node.kind in {"figure", "vector_region"}:
            f = node.facts
            if page is None or not all(k in f for k in ("x", "y", "w", "h", "asset")):
                raise RegionRefused("FIGURE_GEOMETRY_UNAVAILABLE")
            raw = (f["x"], f["y"], f["x"] + f["w"], f["y"] + f["h"])
            box = tuple(to_region_box(page, raw))
            ident = identity("figure-observation/1", source_id, address, f["asset"])
            flags = ("LATTICE_AS_IMAGE",) if f.get("rule") == "table-as-image" else ()
            if node.origin == "inferred":
                flags += ("FIGURE_STRUCTURE_INFERRED",)
            try:
                image = image_from_store(
                    store, f["asset"], raster_asset=raster_choices.get(f["asset"]), limits=limits
                )
                reason = None
            except RegionRefused as error:
                image = None
                reason = error.code
                if reason == "FIGURE_RASTER_UNAVAILABLE" and renderer is not None:
                    from .render import RenderFailed

                    try:
                        store.read(f["asset"])
                        data, media = renderer.render(source, page, raw, dpi)
                        image = image_payload(
                            data,
                            media,
                            source_asset=f["asset"],
                            conversion_rule="bounded-render/1",
                            limits=limits,
                            renderer=renderer.name,
                            renderer_version=renderer.version,
                            dpi=dpi,
                        )
                        reason = None
                    except (RenderFailed, RegionRefused, KeyError):
                        reason = "FIGURE_RENDER_FAILED"
            figures.append(
                FigureInput(
                    ident,
                    source_id,
                    page,
                    box,
                    image,
                    caption_ids=tuple(caption_links.get(address, ())),
                    flags=flags,
                    image_reason=reason,
                )
            )
        for i, child in enumerate(node.children):
            walk(child, f"{address}.{i}" if address else str(i), page)

    walk(tree.root, "")
    if renderer is not None:
        with source.open("rb") as stream:
            if _file_hash(stream) != source_id:
                raise RegionRefused("RENDER_SOURCE_CHANGED")
    return tuple(figures)


def _file_hash(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()
