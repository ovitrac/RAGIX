"""Derived, boundary-aware text views; observations and character provenance survive.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15

Coordinates are unrotated page points with a top-left origin. This module has no
PDF dependency, classifier policy, document vocabulary, model call or KOAS dependency.
"""

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Callable

VERSION = "field-view/1.0"
STATES = frozenset({"CONTENT", "FURNITURE", "UNKNOWN"})
BBox = tuple[float, float, float, float]


def stable_id(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class TextSpan:
    source_id: str
    span_id: str
    page: int
    text: str
    bbox: BBox
    glyph_boxes: tuple[BBox, ...] = ()
    origin: tuple[float, float] = (0, 0)
    direction: tuple[float, float] = (1, 0)
    font_size: float = 0
    state: str = "UNKNOWN"
    flags: tuple[str, ...] = ()
    source_offset: int = 0

    def __post_init__(self):
        if self.state not in STATES or self.page < 1 or not self.source_id or not self.span_id:
            raise ValueError("invalid span identity, page or classification")
        if self.glyph_boxes and len(self.glyph_boxes) != len(self.text):
            raise ValueError("one glyph box per character required")


@dataclass(frozen=True)
class VerticalRule:
    x: float
    top: float
    bottom: float

    def crosses(self, box: BBox) -> bool:
        return box[0] < self.x < box[2] and self.top < box[3] and self.bottom > box[1]


@dataclass(frozen=True)
class CharacterRef:
    span_id: str
    offset: int
    bbox: BBox


@dataclass(frozen=True)
class TextView:
    view_id: str
    source_id: str
    page: int
    text: str
    mapping: tuple[CharacterRef | None, ...]
    state: str
    flags: tuple[str, ...]
    bbox: BBox
    producer: str = VERSION

    def source_refs(self, start: int, end: int) -> tuple[CharacterRef, ...]:
        if not 0 <= start < end <= len(self.text):
            raise ValueError("span outside view")
        refs = tuple(ref for ref in self.mapping[start:end] if ref is not None)
        if not refs:
            raise ValueError("span contains only inserted separators")
        return refs


def union_box(boxes) -> BBox:
    boxes = tuple(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def classify(spans, classifier: Callable[[TextSpan], str]) -> list[TextSpan]:
    """Apply an explicit caller-owned policy, without changing text or identity."""
    return [replace(span, state=classifier(span)) for span in spans]


def split_at_rules(span: TextSpan, rules) -> list[TextSpan]:
    """Partition glyph runs at rules; ambiguous glyph intersections stay flagged."""
    rules = sorted((r for r in rules if r.crosses(span.bbox)), key=lambda r: r.x)
    if not rules:
        return [span]
    if not span.glyph_boxes or abs(span.direction[1]) > 1e-6 or span.direction[0] <= 0:
        return [replace(span, flags=tuple(sorted(set(span.flags) | {"CELL_BOUNDARY_UNRESOLVED"})))]
    if any(not char.isspace() and rule.crosses(box)
           for char, box in zip(span.text, span.glyph_boxes) for rule in rules):
        return [replace(span, flags=tuple(sorted(set(span.flags) | {"RULE_INTERSECTS_GLYPH"})))]
    groups = []
    start, previous = 0, None
    for i, box in enumerate(span.glyph_boxes):
        cell = sum((box[0] + box[2]) / 2 > rule.x for rule in rules)
        if previous is not None and cell != previous:
            groups.append((start, i))
            start = i
        previous = cell
    groups.append((start, len(span.text)))
    return [replace(span, text=span.text[a:b], glyph_boxes=span.glyph_boxes[a:b],
                    bbox=union_box(span.glyph_boxes[a:b]), source_offset=span.source_offset + a)
            for a, b in groups if a < b]


def assemble(spans, separators=None) -> TextView:
    """Materialize an explicit association; inserted separators have no source glyph."""
    spans = tuple(spans)
    if not spans or len({(s.source_id, s.page) for s in spans}) != 1:
        raise ValueError("a view needs spans from exactly one copy and page")
    if any(s.state == "FURNITURE" for s in spans):
        raise ValueError("furniture cannot enter a content view")
    separators = tuple(separators) if separators is not None else (" ",) * (len(spans) - 1)
    if len(separators) != len(spans) - 1 or any(s not in ("", " ", "\n") for s in separators):
        raise ValueError("explicit whitespace separator per boundary required")
    text, mapping, flags = "", [], set()
    for i, span in enumerate(spans):
        if i:
            sep = separators[i - 1]
            if text[-1:].isdigit() and span.text[:1].isdigit():
                flags.add("DIGIT_JOIN")
            text += sep
            mapping.extend([None] * len(sep))
        text += span.text
        flags.update(span.flags)
        if not span.glyph_boxes:
            flags.add("MISSING_GLYPH_GEOMETRY")
        mapping.extend(CharacterRef(span.span_id, span.source_offset + j,
                                    span.glyph_boxes[j] if span.glyph_boxes else span.bbox)
                       for j in range(len(span.text)))
    state = "CONTENT" if all(s.state == "CONTENT" for s in spans) else "UNKNOWN"
    if state == "UNKNOWN":
        flags.add("CLASSIFICATION_UNKNOWN")
    identity = [(s.span_id, s.source_offset, len(s.text), s.state, s.flags) for s in spans]
    return TextView(stable_id(VERSION, spans[0].source_id, identity, separators), spans[0].source_id,
                    spans[0].page, text, tuple(mapping), state, tuple(sorted(flags)),
                    union_box(s.bbox for s in spans))


def line_views(spans, rules=()) -> list[TextView]:
    """Join adjacent upright fragments only, after native-span boundary checks.

    Rules and spans belong to one page. Cross-cell field associations are a caller
    decision and must use assemble explicitly, never relax this line join.
    """
    spans, rules = tuple(spans), tuple(rules)
    if len({(s.source_id, s.page) for s in spans}) > 1:
        raise ValueError("line_views accepts one copy/page at a time")
    pieces = [p for s in spans if s.state != "FURNITURE" for p in split_at_rules(s, rules)
              if p.text.strip()]
    # Cluster against the first baseline, never the previous fragment: pairwise
    # proximity would transitively drift into the next row. Then order by x.
    rows = []
    for piece in sorted(pieces, key=lambda s: (s.origin[1], s.bbox[0], s.span_id, s.source_offset)):
        if (rows and piece.font_size > 0 and rows[-1][0].font_size > 0
                and piece.origin[1] - rows[-1][0].origin[1]
                <= min(piece.font_size, rows[-1][0].font_size) * 0.1):
            rows[-1].append(piece)
        else:
            rows.append([piece])
    pieces = [s for row in rows for s in sorted(row, key=lambda s: (s.bbox[0], s.span_id, s.source_offset))]
    groups, separators = [], []
    for span in pieces:
        sep = None
        if groups:
            prev = groups[-1][-1]
            size = max(prev.font_size, span.font_size)
            gap = span.bbox[0] - prev.bbox[2]
            upright = all(abs(s.direction[1]) <= 1e-6 and s.direction[0] > 0 for s in (prev, span))
            box = union_box((prev.bbox, span.bbox))
            if (size > 0 and upright and not prev.flags and not span.flags
                    and abs(prev.origin[1] - span.origin[1]) <= size * 0.1
                    and max(s.origin[1] for s in (*groups[-1], span))
                        - min(s.origin[1] for s in (*groups[-1], span))
                        <= min(s.font_size for s in (*groups[-1], span)) * 0.1
                    and -size * 0.1 <= gap <= size
                    and not any(rule.crosses(box) for rule in rules)):
                sep = " " if (gap > size * 0.1 or prev.text[-1:].isspace()
                              or span.text[:1].isspace()) else ""
        if sep is None:
            groups.append([span])
            separators.append([])
        else:
            groups[-1].append(span)
            separators[-1].append(sep)
    return [assemble(group, seps) for group, seps in zip(groups, separators)]
