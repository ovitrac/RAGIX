"""Closed data contracts for source-exact renderable regions.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import re

RULE = "renderable-region/1"
KINDS = frozenset({"PROSE", "TABLE", "LIST", "FIGURE"})
MEMBER_KINDS = frozenset({"LINE", "CELL", "CAPTION", "HEADING"})


class RegionRefused(ValueError):
    """A region cannot satisfy the declared source or output contract."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def json_data(value):
    """No Unicode normalization: member text must keep its original code points."""
    return json.dumps(
        asdict(value) if is_dataclass(value) else value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def identity(*parts):
    return hashlib.sha256(json_data(parts).encode("utf-8")).hexdigest()


def check_box(box):
    if (
        not isinstance(box, tuple)
        or len(box) != 4
        or any(type(n) not in (int, float) or not math.isfinite(n) for n in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise RegionRefused("INVALID_GEOMETRY")


def box_union(boxes):
    boxes = tuple(boxes)
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


_MARKUP = re.compile(
    r"<(?:/?[A-Za-z][^>]*|!--[\s\S]*?--|!DOCTYPE[^>]*|\?xml[^>]*\?)>"
    r"|&(?:\#[0-9]+|\#x[0-9a-f]+|[a-z][a-z0-9]+);"
    r"|\bstyle\s*="
    r"|(?:[.#][\w-]+|[a-z][\w-]*)\s*\{[^}]*:[^}]*\}"
    r"|\b(?:color|background(?:-color)?|font(?:-size|-family)?|display|position|margin|padding|border)\s*:[^;\n{}]+[;}]"
    r"|\bdisplay\s*:\s*(?:none|block|inline|grid|flex|inline-block)\s*$"
    r"|\bposition\s*:\s*(?:absolute|relative|fixed|static|sticky)\s*$",
    re.I,
)


def reject_markup(value):
    if isinstance(value, str) and _MARKUP.search(value):
        raise RegionRefused("MARKUP_OR_STYLE_IN_DATA")
    if isinstance(value, dict):
        for key, item in value.items():
            reject_markup(key)
            reject_markup(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_markup(item)


@dataclass(frozen=True)
class RegionMember:
    member_id: str
    source_id: str
    page: int
    kind: str
    text: str | None
    bbox: tuple[float, float, float, float]
    source_spans: tuple[str, ...]
    flags: tuple[str, ...] = ()
    table_id: str | None = None
    row: int | None = None
    column: int | None = None
    row_span: int | None = None
    column_span: int | None = None
    is_header: bool = False
    is_row_label: bool = False
    section_id: str | None = None
    column_id: str | None = None
    list_item: int | None = None
    list_depth: int | None = None

    def __post_init__(self):
        if (
            not isinstance(self.member_id, str)
            or not self.member_id
            or not isinstance(self.source_id, str)
            or not self.source_id
            or type(self.page) is not int
            or self.page < 1
            or self.kind not in MEMBER_KINDS
            or (not isinstance(self.text, str) and not (self.kind == "CELL" and self.text is None))
            or not isinstance(self.source_spans, tuple)
            or not all(isinstance(s, str) and s for s in self.source_spans)
        ):
            raise RegionRefused("INVALID_REGION_MEMBER")
        check_box(self.bbox)
        if self.kind == "CELL":
            if (
                not self.table_id
                or any(
                    type(n) is not int
                    for n in (self.row, self.column, self.row_span, self.column_span)
                )
                or min(self.row, self.column) < 0
                or min(self.row_span, self.column_span) < 1
            ):
                raise RegionRefused("INVALID_CELL_TOPOLOGY")
        elif any(n is not None for n in (self.row, self.column, self.row_span, self.column_span)):
            raise RegionRefused("CELL_TOPOLOGY_ON_NON_CELL")


@dataclass(frozen=True)
class PageGeometry:
    page: int
    width: float
    height: float

    def __post_init__(self):
        if (
            type(self.page) is not int
            or self.page < 1
            or not all(math.isfinite(x) and x > 0 for x in (self.width, self.height))
        ):
            raise RegionRefused("INVALID_PAGE_GEOMETRY")


@dataclass(frozen=True)
class BoundaryPolicy:
    mode: str = "geometry"
    max_gap_ratio: float = 1.0
    max_indent_ratio: float = 1.5
    page_edge_fraction: float = 0.12
    cross_page: bool = True
    rule: str = "region-boundary/1"

    def __post_init__(self):
        if (
            self.mode not in {"line", "section", "geometry"}
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                for x in (self.max_gap_ratio, self.max_indent_ratio)
            )
            or not 0 < self.page_edge_fraction < 0.5
            or type(self.cross_page) is not bool
            or self.rule != "region-boundary/1"
        ):
            raise RegionRefused("INVALID_BOUNDARY_POLICY")


@dataclass(frozen=True)
class RegionWindow:
    before: int = 1
    after: int = 1
    same_page: bool = True

    def __post_init__(self):
        if (
            any(type(n) is not int or n < 0 for n in (self.before, self.after))
            or type(self.same_page) is not bool
        ):
            raise RegionRefused("INVALID_NEIGHBOURHOOD_WINDOW")

    @property
    def rule(self):
        return f"{self.before} region before, {self.after} after, " + (
            "same page" if self.same_page else "document order"
        )


@dataclass(frozen=True)
class RegionLimits:
    max_members: int = 100000
    max_output_bytes: int = 16 * 1024 * 1024
    max_image_bytes: int = 8 * 1024 * 1024
    max_source_asset_bytes: int = 64 * 1024 * 1024
    max_image_pixels: int = 16000000

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in asdict(self).values()):
            raise RegionRefused("INVALID_REGION_LIMIT")


@dataclass(frozen=True)
class FigureImage:
    encoding: str
    media_type: str
    data: str
    sha256: str
    byte_count: int
    width: int
    height: int
    source_asset: str
    conversion_rule: str = "source-raster/1"
    renderer: str | None = None
    renderer_version: str | None = None
    dpi: int | None = None

    def __post_init__(self):
        if (
            self.encoding != "base64"
            or self.media_type not in {"image/png", "image/jpeg"}
            or not isinstance(self.data, str)
            or not self.data.isascii()
            or any(type(n) is not int or n <= 0 for n in (self.byte_count, self.width, self.height))
            or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", self.source_asset)
        ):
            raise RegionRefused("INVALID_FIGURE_PAYLOAD")
        if self.dpi is not None and (type(self.dpi) is not int or self.dpi <= 0):
            raise RegionRefused("INVALID_FIGURE_DPI")


@dataclass(frozen=True)
class FigureInput:
    figure_id: str
    source_id: str
    page: int
    bbox: tuple[float, float, float, float]
    image: FigureImage | None = None
    member_ids: tuple[str, ...] = ()
    caption_ids: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    image_reason: str | None = None

    def __post_init__(self):
        if not self.figure_id or not self.source_id or type(self.page) is not int or self.page < 1:
            raise RegionRefused("INVALID_FIGURE")
        check_box(self.bbox)
        if self.image is not None and not isinstance(self.image, FigureImage):
            raise RegionRefused("INVALID_FIGURE_PAYLOAD")
        if self.image is None and not self.image_reason:
            raise RegionRefused("MISSING_FIGURE_IMAGE_REASON")


@dataclass(frozen=True)
class Region:
    region_id: str
    source_id: str
    page: int
    bbox: tuple[float, float, float, float]
    kind: str
    rule: str
    members: tuple[RegionMember, ...]
    pages: tuple[int, ...]
    page_boxes: tuple[tuple[int, tuple[float, float, float, float]], ...]
    flags: tuple[str, ...] = ()
    image: FigureImage | None = None
    image_reason: str | None = None
    figure_id: str | None = None
    figure_bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if (
            self.kind not in KINDS
            or self.rule != RULE
            or not self.region_id
            or not self.source_id
            or not self.pages
            or self.pages != tuple(sorted(set(self.pages)))
            or self.page != self.pages[0]
            or tuple(p for p, _ in self.page_boxes) != self.pages
            or self.bbox != self.page_boxes[0][1]
            or any(m.source_id != self.source_id or m.page not in self.pages for m in self.members)
            or len({m.member_id for m in self.members}) != len(self.members)
        ):
            raise RegionRefused("INVALID_REGION")
        for _, box in self.page_boxes:
            check_box(box)
        if self.kind == "FIGURE":
            if not self.figure_id or self.figure_bbox is None:
                raise RegionRefused("FIGURE_GEOMETRY_REQUIRED")
            check_box(self.figure_bbox)
        if self.kind != "FIGURE" and (self.image is not None or self.image_reason is not None):
            raise RegionRefused("IMAGE_ON_NON_FIGURE")


@dataclass(frozen=True)
class Neighbourhood:
    before: tuple[Region, ...]
    after: tuple[Region, ...]
    requested_before: int
    requested_after: int
    scope_page: int | None
    window_rule: str
    usage: str = "CONTEXT_ONLY"


@dataclass(frozen=True)
class Anchor:
    member_id: str
    page: int
    start: int
    end: int
    text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderableRegion:
    region: Region
    anchor: Anchor
    neighbourhood: Neighbourhood
    boundary_policy: BoundaryPolicy
    flags: tuple[str, ...]

    @property
    def region_id(self):
        return self.region.region_id

    @property
    def anchor_member_id(self):
        return self.anchor.member_id

    def to_dict(self):
        data = {
            **asdict(self.region),
            "anchor_member_id": self.anchor_member_id,
            "anchor": asdict(self.anchor),
            "neighbourhood": asdict(self.neighbourhood),
            "boundary_policy": asdict(self.boundary_policy),
            "flags": list(self.flags),
        }
        reject_markup(data)
        return data

    def to_json(self):
        return json_data(self.to_dict())
