"""Synthetic reference-field geometry for the slice-4 field gates (A-series).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio

Every label, identifier, type word and title here is synthetic. Geometry enters as
ranges: the corners below are range ends plus one interior point, and no gate may
depend on a single value inside them. Ranges (points): line height 12-18;
label-to-value offset -0.5..+0.5; underline of 1-2 rules, 0.8-1.6 apart, 0-2.5
above the label's bottom edge, label width +/- 3; gaps inside a field -2.5..+7;
gap to the next block 0.2-0.6 line heights; block gap 2.2 line heights.
"""

from dataclasses import dataclass

from ragix_kernels.saqqara.census import DocumentDigest, PageDigest
from ragix_kernels.saqqara.field_views import HorizontalRule, TextSpan, VerticalRule

SOURCE = "synthetic-slice4-fields"
LABEL_EN = "Tested reference"
LABEL_FR = "Référence vérifiée"
TYPE_WORD = "ZRS"  # a consumer's acronym: never a shipped default
PAGES = (1, 2, 3)
BODY = "ordinary body words of a paragraph"


@dataclass(frozen=True)
class Geometry:
    name: str
    line_height: float
    value_offset: float
    underline: tuple[float, ...]  # distances above the label's bottom edge
    underline_slack: float  # added to the label width on each side
    inner_gaps: tuple[float, float]  # between lines 1-2 and 2-3 of one field
    stop_gap_ratio: float


LOW = Geometry("low", 12.0, -0.5, (0.0,), -3.0, (-2.5, 0.0), 0.2)
LOW2 = Geometry("low2", 12.0, 0.5, (0.0, 0.8), 3.0, (0.0, -2.5), 0.6)
MID = Geometry("mid", 15.0, 0.0, (1.0, 2.1), 0.0, (-1.0, 3.0), 0.4)
HIGH = Geometry("high", 18.0, 0.5, (0.9, 2.5), 3.0, (7.0, 7.0), 0.6)
GEOMETRIES = (LOW, LOW2, MID, HIGH)


def span(value, page, ident, x, top, g, right=None):
    """One upright span whose glyph boxes tile [x, right]."""
    size = 0.75 * g.line_height
    width = (right - x) / len(value) if right is not None else 0.5 * size
    boxes = tuple(
        (x + width * i, top, x + width * (i + 1), top + g.line_height) for i in range(len(value))
    )
    return TextSpan(
        SOURCE,
        f"s:{page}:{ident}",
        page,
        value,
        (x, top, boxes[-1][2], top + g.line_height),
        boxes,
        (x, top + 0.8 * g.line_height),
        font_size=size,
    )


def underline(label, g):
    left, _, right, bottom = label.bbox
    return tuple(
        HorizontalRule(bottom - lift, left - g.underline_slack, right + g.underline_slack)
        for lift in g.underline
    )


def document(pages):
    return DocumentDigest(SOURCE, tuple(pages), "synthetic", "1")


def identifier(page, family="CDE"):
    return f"AB-{family}-90000{page}"


def labelled(page, g, label=LABEL_EN, value=None, top=100.0, x=60.0, colon=" :"):
    """Label span and adjacent value span on one baseline, label underlined."""
    left = span(label + colon, page, "l", x, top, g, right=x + 120)
    spans = [left]
    if value:
        spans.append(span(value, page, "v", left.bbox[2] + g.value_offset, top, g))
    return spans, underline(left, g)


def next_block(page, g, top, x=60.0):
    return span("Bloc suivant :", page, "n", x, top, g)


def stacked(page, g, lines, top, gaps=(0.0,), x=60.0, prefix="b"):
    """Lines sharing one left edge; returns the spans and the next free top."""
    spans = []
    for i, line in enumerate(lines):
        spans.append(span(line, page, f"{prefix}{i}", x, top, g))
        top += g.line_height + gaps[i % len(gaps)]
    return spans, top


def a1(g):
    pages = []
    for p in PAGES:
        spans, rules = labelled(p, g, value=f"{TYPE_WORD} {identifier(p)} V 1.0 §4.1 §4.2")
        below = 100.0 + g.line_height * (1 + g.stop_gap_ratio)
        pages.append(
            PageDigest(p, 600, 800, (*spans, next_block(p, g, below)), (), horizontal_rules=rules)
        )
    return document(pages)


def a2(g, wrap="value"):
    """The value wraps onto two further lines; an open range is closed by the last one."""
    pages = []
    for p in PAGES:
        spans, rules = labelled(p, g, value=f"{TYPE_WORD} {identifier(p)} V 1.0 §4.1 §4.2")
        x = spans[1].bbox[0] if wrap == "value" else spans[0].bbox[0]
        top = 100.0 + g.line_height + g.inner_gaps[0]
        second = span("§4.3 §4.4 §5.1 à", p, "v2", x, top, g)
        top += g.line_height + g.inner_gaps[1]
        third = span("§5.4 AB-FORM-900002 « titre de formulaire »", p, "v3", x, top, g)
        top += g.line_height * (1 + g.stop_gap_ratio)
        pages.append(
            PageDigest(
                p,
                600,
                800,
                (*spans, second, third, next_block(p, g, top)),
                (),
                horizontal_rules=rules,
            )
        )
    return document(pages)


def a3(g, edge=False):
    """Nothing after the label; the value starts the next line.

    With `edge`, a rule as wide as a table lies on the label's bottom edge instead
    of the underline: that is the top edge of something, and it closes the window.
    """
    pages = []
    for p in PAGES:
        spans, rules = labelled(p, g, label=LABEL_FR)
        if edge:
            rules = (HorizontalRule(spans[0].bbox[3], 40.0, 540.0),)
        top = 100.0 + g.line_height + max(g.inner_gaps)
        value = span(f"{TYPE_WORD} {identifier(p)} V 1.0 §4.1", p, "v", 60.0, top, g)
        pages.append(PageDigest(p, 600, 800, (*spans, value), (), horizontal_rules=rules))
    return document(pages)


def prose(g, lines):
    """True labelled fields, and lower on each page a paragraph naming the label."""
    pages = []
    for p in PAGES:
        spans, rules = labelled(p, g, label=LABEL_FR, value=f"{identifier(p)} V 1.0 §4.1")
        below = 100.0 + g.line_height * (1 + g.stop_gap_ratio)
        paragraph, _ = stacked(p, g, lines, 300.0, (max(g.inner_gaps),), prefix="p")
        pages.append(
            PageDigest(
                p,
                600,
                800,
                (*spans, next_block(p, g, below), *paragraph),
                (),
                horizontal_rules=rules,
            )
        )
    return document(pages)


def a5(g):
    """The label words start a line inside a sentence."""
    return prose(
        g, ("… les résultats portent la", "référence vérifiée par le laboratoire du site.")
    )


def a6(g):
    """The label words start a paragraph, with no colon and no reference content."""
    return prose(
        g,
        (
            "Fin du paragraphe précédent.",
            "Référence vérifiée ensuite lors de la revue annuelle.",
        ),
    )


def a7(g, typed=False, colon_evidence=False):
    """A colon-less label that does introduce a value."""
    pages = []
    for p in PAGES:
        head = f"{TYPE_WORD} " if typed else ""
        # Without a colon the label span ends with its space, as extracted text does.
        spans, rules = labelled(p, g, value=f"{head}{identifier(p)} V 1.0 §3.1", colon=" ")
        if colon_evidence:  # the same label, with its colon, elsewhere on the page
            more, more_rules = labelled(p, g, value=f"{identifier(p, 'FGH')} V 2.0 §3.2", top=300.0)
            more = [
                span(s.text, p, f"c{i}", s.bbox[0], 300.0, g, right=s.bbox[2])
                for i, s in enumerate(more)
            ]
            spans, rules = [*spans, *more], (*rules, *more_rules)
        pages.append(PageDigest(p, 600, 800, tuple(spans), (), horizontal_rules=rules))
    return document(pages)


def a8(g, position, borders="shared", jitter=0.0, underlined=False):
    """Label in a ruled cell; value in the cell to the right or in the cell below.

    `shared` paints each rule once across the grid; `per_cell` paints every cell's
    own borders, which then coincide only within `jitter`. A rule 200 points below
    the grid must change nothing.
    """
    pages = []
    h = g.line_height
    for p in PAGES:
        label = span(LABEL_EN + " :", p, "l", 64.0, 100.0, g, right=184.0)
        # The value keeps inside its own cell at every line height.
        value = span(
            f"{identifier(p)} V 1.0 §4.1",
            p,
            "v",
            224.0 if position == "right" else 64.0,
            100.0 if position == "right" else 100.0 + h + 14.0,
            g,
            right=474.0 if position == "right" else 214.0,
        )
        xs = (60.0, 220.0, 480.0)
        ys = (96.0, 100.0 + h + 6.0, 100.0 + 2 * h + 22.0)
        if borders == "shared":
            vertical = tuple(VerticalRule(x, ys[0], ys[-1]) for x in xs)
            horizontal = tuple(HorizontalRule(y, xs[0], xs[-1]) for y in ys)
        else:
            vertical, horizontal = [], []
            for row, (top, bottom) in enumerate(zip(ys, ys[1:])):
                for col, (left, right) in enumerate(zip(xs, xs[1:])):
                    d = jitter * ((row + col) % 2)
                    vertical += [
                        VerticalRule(left + d, top + d, bottom + d),
                        VerticalRule(right + d, top + d, bottom + d),
                    ]
                    horizontal += [
                        HorizontalRule(top + d, left + d, right + d),
                        HorizontalRule(bottom + d, left + d, right + d),
                    ]
            vertical, horizontal = tuple(vertical), tuple(horizontal)
        horizontal += (HorizontalRule(ys[-1] + 200.0, xs[0], xs[-1]),)
        if underlined:
            horizontal += underline(label, g)
        pages.append(PageDigest(p, 600, 800, (label, value), vertical, horizontal_rules=horizontal))
    return document(pages)


ROLE_IDENTIFIERS = ("AB-SOP-900005", "AB-OPE-900006", "AB-WIN-900007")
ROLE_LINES = (
    "Procédure n° AB-SOP-900005 « Titre d'une procédure qui continue sur la",
    "ligne suivante »",
    "Rapport : AB-OPE-900006 « Titre d'un rapport »",
    "Instruction : AB-WIN-900007 « Titre »",
)


def a9(g):
    """A one-line field, then role-word statements (one wrapped), then a caption."""
    pages = []
    for p in PAGES:
        spans, rules = labelled(
            p, g, label=LABEL_FR, value=f"{TYPE_WORD} {identifier(p)} V 1.0 §4.1"
        )
        gaps = tuple(max(gap, 0.0) for gap in g.inner_gaps)
        more, _ = stacked(
            p, g, (*ROLE_LINES, "Tableau 3"), 100.0 + g.line_height + gaps[0], gaps, prefix="r"
        )
        pages.append(PageDigest(p, 600, 800, (*spans, *more), (), horizontal_rules=rules))
    return document(pages)


def a10(g, connector="à", marker="§"):
    """A range is the whole member list."""
    pages = []
    for p in PAGES:
        spans, rules = labelled(
            p,
            g,
            label=LABEL_FR,
            value=f"{identifier(p)} V 1.0 {marker}4.1 {connector} {marker}4.12",
        )
        below = 100.0 + g.line_height * (1 + g.stop_gap_ratio)
        pages.append(
            PageDigest(p, 600, 800, (*spans, next_block(p, g, below)), (), horizontal_rules=rules)
        )
    return document(pages)


def a11(g):
    """The field is the last body line of a page; the next page opens with section tokens."""
    pages = []
    for p in PAGES:
        spans = [span("§4.5 §4.6", p, "top", 60.0, 100.0, g)]
        field, rules = labelled(
            p, g, label=LABEL_FR, value=f"{identifier(p)} V 1.0 §4.1", top=700.0
        )
        pages.append(PageDigest(p, 600, 800, (*spans, *field), (), horizontal_rules=rules))
    return document(pages)


def a12(g, body_lines=3):
    """A body paragraph, a block gap, the field line, a block gap, a section token.

    The paragraph gives the document its own within-block gaps, so the bound that
    separates blocks is derived and never rests on a default. `body_lines=0` is
    the control in which nothing can be derived.
    """
    pages = []
    block = 2.2 * g.line_height
    for p in PAGES:
        body, top = stacked(p, g, (BODY,) * body_lines, 120.0, (0.0, 7.0))
        if body_lines:
            top += block - (0.0, 7.0)[(body_lines - 1) % 2]
        field = span(f"{LABEL_EN} : {identifier(p)} V 1.0 §4.1", p, "l", 60.0, top, g)
        far = span("§9.1", p, "far", 60.0, top + g.line_height + block, g)
        pages.append(PageDigest(p, 600, 800, (*body, field, far), ()))
    return document(pages)


def a13(g):
    """A numbered heading follows the field."""
    pages = []
    for p in PAGES:
        spans, rules = labelled(p, g, label=LABEL_FR, value=f"{identifier(p)} V 1.0 §4.1")
        top = 100.0 + g.line_height + max(g.inner_gaps)
        heading = span("5.2 Essais de fonctionnement", p, "h", 60.0, top, g)
        pages.append(PageDigest(p, 600, 800, (*spans, heading), (), horizontal_rules=rules))
    return document(pages)


def role_only(g):
    """A label with nothing after it but a role-word line."""
    pages = []
    for p in PAGES:
        spans, rules = labelled(p, g, label=LABEL_FR)
        top = 100.0 + g.line_height + max(g.inner_gaps)
        role = span(ROLE_LINES[2], p, "r", 60.0, top, g)
        more, more_rules = labelled(p, g, label=LABEL_FR, value=f"{identifier(p)} V 1.0", top=400.0)
        more = [
            span(s.text, p, f"m{i}", s.bbox[0], 400.0, g, right=s.bbox[2])
            for i, s in enumerate(more)
        ]
        pages.append(
            PageDigest(
                p, 600, 800, (*spans, role, *more), (), horizontal_rules=(*rules, *more_rules)
            )
        )
    return document(pages)


REFERENCE = "AB-CDE-900001"  # one document cited on every page, as a protocol cites its source
VALUE_FORMS = {
    # revision glued to the first section sign
    "bare": (f"{TYPE_WORD} {REFERENCE} V 1.0§4.1 §4.2", [("single", "4.1"), ("single", "4.2")]),
    # the same phrase before a colon
    "colon": (f"{TYPE_WORD} {REFERENCE} V 1.0: §4.1 à §4.3", [("range", "4.1", "4.3")]),
    "plain": (f"{REFERENCE} V 1.0 §4.1", [("single", "4.1")]),
}


def b1(g, forms=("bare", "colon"), overhang=0.0, ruled="grid", top_offset=0.3):
    """A label ends its cell; its value opens the next cell on the same row.

    The value opens with an acronym, an identifier and a revision. With `colon` pages in
    the same document, that phrase also stands before a colon: it is reference content
    all the same, never a label. `overhang` is how far the label's last glyph runs past
    the border between the two cells; the value starts at the border.
    """
    pages = []
    h = g.line_height
    number = 0
    for form in forms:
        for _ in PAGES:
            number += 1
            label = span(LABEL_EN + " :", number, "l", 80.0, 100.0, g, right=182.0)
            border = label.bbox[2] - overhang
            value = span(
                VALUE_FORMS[form][0],
                number,
                "v",
                border,
                100.0 + top_offset,
                g,
                right=border + 300.0,
            )
            vertical = (VerticalRule(border, 96.0, 104.0 + h),)
            horizontal = ()
            if ruled == "grid":
                vertical = tuple(
                    VerticalRule(x, 96.0, 104.0 + h) for x in (76.0, border, border + 320.0)
                )
                horizontal = tuple(
                    HorizontalRule(y, 76.0, border + 320.0) for y in (96.0, 104.0 + h)
                )
            below = next_block(number, g, 104.0 + h + g.line_height * g.stop_gap_ratio + 8.0)
            pages.append(
                PageDigest(
                    number, 600, 800, (label, value, below), vertical, horizontal_rules=horizontal
                )
            )
    return document(pages)


PROSE_LABEL = "Remarque"


def b3(g):
    """A one-word label with its colon, followed by prose that happens to quote identifiers."""
    pages = []
    for p in PAGES:
        first = span(
            f"{PROSE_LABEL} : les essais de la pompe (voir le plan) n° {identifier(p)}",
            p,
            "w1",
            60.0,
            100.0,
            g,
        )
        top = 100.0 + 3 * g.line_height
        lines = (
            f"{PROSE_LABEL} : la vérification suit l'ordre des essais prévus pour l'ensemble",
            f"du lot n° {identifier(p, 'FGH')}",
            f"puis celui du lot {identifier(p, 'JKL')}",
            "fin.",
        )
        more, _ = stacked(p, g, lines, top, g.inner_gaps, prefix="w2")
        pages.append(PageDigest(p, 600, 800, (first, *more), ()))
    return document(pages)
