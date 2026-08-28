"""
Fixture generators for the saqqara gates.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-27

Every fixture is built by code at test time. No document is committed to this
repository, which is what lets fixture content be proved rather than screened:
the origin of every character is the function below that wrote it.

Each generator takes a destination path and returns it. Shapes are described in
words here because the shape IS the ground truth — a generator whose intent
lives only in its own code cannot be reviewed against the specification.

FIXTURES is the registry the K0 gate checks against SPEC.md, in both
directions: a fixture named in the specification must exist here, and a
generator here must be named by some proposition. Neither drifts silently.

Status: P0 stubs. Each raises until its phase implements it; a stub that
returned an empty file would let a gate pass on nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

from ragix_kernels.saqqara.model import (
    CANONICAL_JSON, DocumentLocator, DocxLocator, Locator, MdLocator, Node,
    PdfLocator, PptxLocator, Provenance, Tree, XlsxLocator,
)


def _pending(phase: str, shape: str) -> Callable[[Path], Path]:
    """A generator not yet written. Refuses rather than producing an empty file."""

    def build(path: Path) -> Path:
        raise NotImplementedError(f"{phase}: {shape}")

    build.__doc__ = f"[{phase}] {shape}"
    build.pending = True          # read by the K0 gate: a stub must refuse, not return nothing
    return build


# --------------------------------------------------------------- spreadsheets



# ------------------------------------------------------------ word processing


# ------------------------------------------------------------------ documents


# --------------------------------------------------------------------- trees

# ---------------------------------------------------------------------- trees

#: Version stamped on every node these generators build, so a fixture change is
#: visible in any signature computed from it.
FIXTURE_KERNEL = "saqqara.fixtures"
FIXTURE_VERSION = "0.1.0"


def _prov(source: str, fmt: str, *chain: Locator, sha: str | None = None) -> Provenance:
    return Provenance(
        source_path=source,
        source_format=fmt,
        chain=chain,
        kernel=FIXTURE_KERNEL,
        kernel_version=FIXTURE_VERSION,
        source_sha256=sha,
    )


def build_trees() -> Dict[str, Tree]:
    """One small tree per format, each pinning that format's locator convention.

    Two of them are shaped to oppose each other on structure: the presentation
    tree puts all of its text under headings, the markdown tree puts none of it
    under any. Without that contrast a claim about separating structured mass
    from flat mass has nothing to separate.
    """
    trees: Dict[str, Tree] = {}

    # pdf — pages and boxes, one heading covering one paragraph
    src = "fixture.pdf"
    trees["pdf"] = Tree(
        meta={"title": "Paged fixture", "pages": 2},
        root=Node(
            kind="document",
            provenance=_prov(src, "pdf", DocumentLocator(), sha="0" * 64),
            children=[
                Node(
                    kind="heading",
                    level=1,
                    text="First part",
                    provenance=_prov(src, "pdf", PdfLocator(page=1, bbox=(72.0, 700.0, 520.0, 720.0))),
                    children=[
                        Node(
                            kind="paragraph",
                            text="Body text on the first page.",
                            provenance=_prov(src, "pdf", PdfLocator(page=1, bbox=(72.0, 600.0, 520.0, 690.0))),
                        )
                    ],
                ),
                Node(
                    kind="paragraph",
                    text="An orphan paragraph on the second page.",
                    provenance=_prov(src, "pdf", PdfLocator(page=2, bbox=(72.0, 640.0, 520.0, 700.0))),
                ),
            ],
        ),
    )

    # docx — flows, paragraphs and runs; the heading is inferred, not read
    src = "fixture.docx"
    trees["docx"] = Tree(
        meta={"title": "Flowed fixture"},
        root=Node(
            kind="document",
            provenance=_prov(src, "docx", DocumentLocator()),
            children=[
                Node(
                    kind="heading",
                    level=1,
                    text="Inferred title",
                    origin="inferred",
                    confidence=0.6,
                    provenance=_prov(src, "docx", DocxLocator(flow="body", paragraph=0)),
                    children=[
                        Node(
                            kind="paragraph",
                            text="A paragraph the reader took verbatim.",
                            provenance=_prov(src, "docx", DocxLocator(flow="body", paragraph=1, run=0)),
                        )
                    ],
                ),
                Node(
                    kind="table",
                    provenance=_prov(src, "docx", DocxLocator(flow="header:first", table_index=0)),
                    facts={"n_rows": 1, "n_cols": 2},
                    children=[
                        Node(
                            kind="paragraph",
                            text="Header cell",
                            provenance=_prov(
                                src, "docx", DocxLocator(flow="header:first", table_index=0, row=0, col=0)
                            ),
                            facts={"bold": True},
                        )
                    ],
                ),
            ],
        ),
    )

    # xlsx — sheets, cells, merged extents
    src = "fixture.xlsx"
    trees["xlsx"] = Tree(
        meta={"sheets": ["Data"]},
        root=Node(
            kind="document",
            provenance=_prov(src, "xlsx", DocumentLocator()),
            children=[
                Node(
                    kind="section",
                    text="Data",
                    provenance=_prov(src, "xlsx", XlsxLocator(sheet="Data", sheet_index=0)),
                    children=[
                        Node(
                            kind="table",
                            span="A1:B2",
                            provenance=_prov(src, "xlsx", XlsxLocator(sheet="Data", sheet_index=0)),
                            children=[
                                Node(
                                    kind="paragraph",
                                    text="Quantity",
                                    provenance=_prov(
                                        src,
                                        "xlsx",
                                        XlsxLocator(
                                            sheet="Data", sheet_index=0, cell="A1", row=1, col=1,
                                            merged_range="A1:B1",
                                        ),
                                    ),
                                    facts={"dtype": "s", "bold": True, "number_format": "General",
                                           "locked": True},
                                )
                            ],
                        )
                    ],
                )
            ],
        ),
    )

    # pptx — one-based slides and shapes; everything sits under a slide heading
    src = "fixture.pptx"
    trees["pptx"] = Tree(
        meta={"slides": 1},
        root=Node(
            kind="document",
            provenance=_prov(src, "pptx", DocumentLocator()),
            children=[
                Node(
                    kind="heading",
                    level=1,
                    text="Slide title",
                    provenance=_prov(src, "pptx", PptxLocator(slide=1, shape=0)),
                    children=[
                        Node(
                            kind="list",
                            provenance=_prov(src, "pptx", PptxLocator(slide=1, shape=1)),
                            children=[
                                Node(
                                    kind="list_item",
                                    text="One bullet, entirely under its title.",
                                    provenance=_prov(src, "pptx", PptxLocator(slide=1, shape=1)),
                                )
                            ],
                        )
                    ],
                )
            ],
        ),
    )

    # md — lines; deliberately flat, no heading anywhere
    src = "fixture.md"
    trees["md"] = Tree(
        meta={},
        root=Node(
            kind="document",
            provenance=_prov(src, "md", DocumentLocator()),
            children=[
                Node(
                    kind="paragraph",
                    text="Text with no heading above it, on purpose.",
                    provenance=_prov(src, "md", MdLocator(line=1)),
                ),
                Node(
                    kind="paragraph",
                    text="A second unstructured paragraph.",
                    provenance=_prov(src, "md", MdLocator(line=3)),
                ),
            ],
        ),
    )

    return trees


def trees_per_format(path: Path) -> Path:
    """Write the per-format trees as one canonical JSON document."""
    payload = {fmt: tree.to_dict() for fmt, tree in sorted(build_trees().items())}
    path.write_text(json.dumps(payload, **CANONICAL_JSON), encoding="utf-8")
    return path

# ------------------------------------------------------------------- sections


# ----------------------------------------------------------------------- pdf

# These two fixtures are written byte by byte rather than through a library.
# The reason is the rule that governs every fixture here: it must be generated
# by code, with no external provenance. The library that reads pdf in this
# package does not author them, and pulling in a second one to build two small
# files would add a dependency to answer a question the format itself answers
# in about a hundred lines. Writing the bytes also makes the ground truth
# explicit — the outline below IS the expected outline, not a by-product of
# somebody else's writer.

#: The outline the `pdf_outline` fixture declares. Title, page (1-based), depth.
PDF_OUTLINE_ENTRIES = [
    ("Contexte", 1, 1),
    ("Moyens", 2, 1),
    ("Moyens humains", 2, 2),
    ("Calendrier", 3, 1),
]

#: The body line each page of `pdf_outline` carries, after its heading.
PDF_PAGE_BODY = [
    "Le contexte de l'operation, en une ligne.",
    "Les moyens engages, en une ligne.",
    "Le calendrier previsionnel, en une ligne.",
]


def _pdf_escape(text: str) -> bytes:
    out = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return out.encode("ascii", errors="replace")


def _pdf_assemble(objects: list[bytes]) -> bytes:
    """Lay out numbered objects, then the cross-reference table and trailer.

    Objects arrive already rendered, in order, object 1 first. Offsets are
    measured as the bytes are laid down, which is the only way to get an xref
    table that agrees with the file it describes.
    """
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def _pdf_stream(payload: bytes, extra: str = "") -> bytes:
    return (
        f"<< /Length {len(payload)}{extra} >>\nstream\n".encode("ascii")
        + payload
        + b"\nendstream"
    )


def pdf_outline(path: Path) -> Path:
    """Three pages of text under a declared outline, one entry of which nests.

    The nested entry matters: an outline is a tree, and a reader that flattens
    it reports the right number of headings at the wrong depths.
    """
    page_count = len(PDF_PAGE_BODY)
    first_page_obj = 4                                   # 1 catalog, 2 pages, 3 font
    content_obj = first_page_obj + page_count
    outline_root = content_obj + page_count
    first_item = outline_root + 1

    objects: list[bytes] = []

    kids = " ".join(f"{first_page_obj + i} 0 R" for i in range(page_count))
    objects.append(f"<< /Type /Catalog /Pages 2 0 R /Outlines {outline_root} 0 R >>".encode())
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index in range(page_count):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_obj + index} 0 R >>"
            ).encode()
        )

    headings = [title for title, _, depth in PDF_OUTLINE_ENTRIES if depth == 1]
    for index, body in enumerate(PDF_PAGE_BODY):
        payload = (
            b"BT /F1 18 Tf 72 760 Td (" + _pdf_escape(headings[index]) + b") Tj ET\n"
            b"BT /F1 11 Tf 72 720 Td (" + _pdf_escape(body) + b") Tj ET\n"
        )
        objects.append(_pdf_stream(payload))

    # outline tree: three top-level items, the second holding one child
    top = [i for i, (_, _, depth) in enumerate(PDF_OUTLINE_ENTRIES) if depth == 1]
    child_index = next(i for i, (_, _, depth) in enumerate(PDF_OUTLINE_ENTRIES) if depth == 2)
    obj_of = {entry: first_item + entry for entry in range(len(PDF_OUTLINE_ENTRIES))}

    objects.append(
        (
            f"<< /Type /Outlines /First {obj_of[top[0]]} 0 R "
            f"/Last {obj_of[top[-1]]} 0 R /Count {len(PDF_OUTLINE_ENTRIES)} >>"
        ).encode()
    )

    for entry in range(len(PDF_OUTLINE_ENTRIES)):
        title, page, depth = PDF_OUTLINE_ENTRIES[entry]
        parts = [
            b"<< /Title (" + _pdf_escape(title) + b")",
            f" /Dest [{first_page_obj + page - 1} 0 R /Fit]".encode(),
        ]
        if depth == 1:
            parts.append(f" /Parent {outline_root} 0 R".encode())
            place = top.index(entry)
            if place > 0:
                parts.append(f" /Prev {obj_of[top[place - 1]]} 0 R".encode())
            if place < len(top) - 1:
                parts.append(f" /Next {obj_of[top[place + 1]]} 0 R".encode())
        else:
            parts.append(f" /Parent {obj_of[entry - 1]} 0 R".encode())
        parts.append(b" >>")
        objects.append(b"".join(parts))

    # attach the child to its parent, now that both object numbers are known
    parent = child_index - 1
    body = objects[obj_of[parent] - 1]
    objects[obj_of[parent] - 1] = body[:-3] + (
        f" /First {obj_of[child_index]} 0 R /Last {obj_of[child_index]} 0 R /Count 1 >>"
    ).encode()

    path.write_bytes(_pdf_assemble(objects))
    return path


def pdf_no_text_layer(path: Path) -> Path:
    """One page carrying an image and not a single text operator.

    This is the scanned page: a reader that reports it as empty has told a
    truth about the bytes and a lie about the document.
    """
    import zlib

    width = height = 8
    raw = bytes(bytearray((row * 32 + col * 4) % 256 for row in range(height) for col in range(width)))
    image = zlib.compress(raw)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /XObject << /Im1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        _pdf_stream(b"q 400 0 0 400 " + b"97 300 cm /Im1 Do Q\n"),
        _pdf_stream(
            image,
            extra=(
                f" /Type /XObject /Subtype /Image /Width {width} /Height {height}"
                " /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode"
            ),
        ),
    ]

    path.write_bytes(_pdf_assemble(objects))
    return path


# ------------------------------------------------------- sections and outline

#: The eight channels, closed. A record from anywhere else is a defect.
SECTION_CHANNELS = (
    "pdf-numbered-heading",
    "pdf-native-toc",
    "docx-heading-style",
    "docx-numbering-property",
    "xlsx-sheet-name",
    "xlsx-section-row",
    "xlsx-block-title",
    "pptx-slide-title",
)

#: What each channel must find in `sections_multi_channel`, and what must be
#: refused. Written here so a test compares against an intention rather than
#: against whatever the collector happened to produce.
EXPECTED_SECTIONS = {
    "pdf-native-toc": ["Contexte", "Moyens"],
    "pdf-numbered-heading": ["Contexte", "Moyens engages"],
    "docx-heading-style": ["Titre principal", "Sous-titre"],
    "docx-numbering-property": ["Portee du document"],
    "xlsx-sheet-name": ["Chapitre A", "Chapitre B"],
    "xlsx-block-title": ["TABLEAU DE SUIVI"],
    "xlsx-section-row": ["SECTION - PHASE DEUX"],
    "pptx-slide-title": ["Ouverture", "Fermeture"],
    # the printed contents page: title, dot leaders, page number
    "refused_printed_contents": 2,
    # a bare number with no line beneath it to name it
    "refused_orphan_number": 1,
}

#: The running-header fixture: one numbered line repeats on every page.
RUNNING_HEADER = "Rapport technique"
RUNNING_HEADER_PAGES = 4
RUNNING_HEADER_SECTIONS = ["Contexte", "Moyens", "Delais", "Suivi"]


def _pdf_pages(path: Path, pages: list[list[str]], outline: list | None = None) -> Path:
    """Write a laid-out document with the given lines per page, one line per row."""
    page_count = len(pages)
    first_page_obj = 4
    content_obj = first_page_obj + page_count
    outline_root = content_obj + page_count

    objects: list[bytes] = []
    kids = " ".join(f"{first_page_obj + i} 0 R" for i in range(page_count))
    catalog = f"<< /Type /Catalog /Pages 2 0 R"
    if outline:
        catalog += f" /Outlines {outline_root} 0 R"
    objects.append((catalog + " >>").encode())
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index in range(page_count):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_obj + index} 0 R >>"
            ).encode()
        )

    for lines in pages:
        payload = b""
        top = 780
        for offset, line in enumerate(lines):
            y = top - offset * 24
            payload += (
                b"BT /F1 12 Tf 72 " + str(y).encode() + b" Td ("
                + _pdf_escape(line) + b") Tj ET\n"
            )
        objects.append(_pdf_stream(payload))

    if outline:
        first_item = outline_root + 1
        objects.append(
            (
                f"<< /Type /Outlines /First {first_item} 0 R "
                f"/Last {first_item + len(outline) - 1} 0 R /Count {len(outline)} >>"
            ).encode()
        )
        for index, (title, page) in enumerate(outline):
            parts = [
                b"<< /Title (" + _pdf_escape(title) + b")",
                f" /Dest [{first_page_obj + page - 1} 0 R /Fit]".encode(),
                f" /Parent {outline_root} 0 R".encode(),
            ]
            if index > 0:
                parts.append(f" /Prev {first_item + index - 1} 0 R".encode())
            if index < len(outline) - 1:
                parts.append(f" /Next {first_item + index + 1} 0 R".encode())
            parts.append(b" >>")
            objects.append(b"".join(parts))

    path.write_bytes(_pdf_assemble(objects))
    return path


def _numbered_paragraph(doc, text: str, level: int = 0):
    """A paragraph numbered by property rather than by typed text.

    This is the case that defeats a text-only reader: the file stores the words
    and nothing else, and the number the reader sees on screen is generated.
    """
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    paragraph = doc.add_paragraph(text)
    properties = paragraph._p.get_or_add_pPr()
    numbering = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), str(level))
    num_id = OxmlElement("w:numId")
    num_id.set(qn("w:val"), "1")
    numbering.append(ilvl)
    numbering.append(num_id)
    properties.append(numbering)
    return paragraph


def sections_multi_channel(path: Path) -> Path:
    """One document per channel, plus an index naming them.

    Written as four files beside a manifest rather than as one, because the eight
    channels do not coexist in any single format: a sheet name and a slide title
    cannot both be facts about the same file. The manifest is what `path` returns,
    so the fixture is still one addressable thing.
    """
    import json as _json

    stem = path.stem
    written = {}

    # --- laid out: a printed contents page, a heading on one line, a split heading
    pdf_path = path.with_name(f"{stem}.pdf")
    _pdf_pages(
        pdf_path,
        pages=[
            [
                "Sommaire",
                "1. Contexte ........................... 2",
                "2. Moyens ............................. 3",
            ],
            ["1. Contexte", "Le contexte de l'operation, en une ligne."],
            # the number alone, its title on the next line: one heading, two lines
            ["2.", "Moyens engages", "Les moyens, en une ligne."],
            # a bare number with nothing beneath it: a page number, not a label
            ["Le dernier paragraphe du document.", "9."],
        ],
        outline=[("Contexte", 2), ("Moyens", 3)],
    )
    written["pdf"] = pdf_path.name

    # --- flowed: styled headings, and one numbered by property
    from docx import Document

    doc = Document()
    doc.add_paragraph("Titre principal", style="Heading 1")
    doc.add_paragraph("Un paragraphe d'introduction.")
    doc.add_paragraph("Sous-titre", style="Heading 2")
    doc.add_paragraph("Un paragraphe de contenu.")
    _numbered_paragraph(doc, "Portee du document")
    doc.add_paragraph("Le paragraphe qui suit la clause numerotee.")
    docx_path = path.with_name(f"{stem}.docx")
    doc.save(docx_path)
    written["docx"] = docx_path.name

    # --- gridded: sheet names, a block title, a section row inside a table
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Chapitre A"
    ws["B2"] = "TABLEAU DE SUIVI"
    ws.merge_cells("B2:E2")
    _bold(ws, "B2")
    for col, head in zip("BCDE", ("Poste", "Prevu", "Reel", "Ecart")):
        ws[f"{col}3"] = head
        _bold(ws, f"{col}3")
    for row, poste in enumerate(("Etudes", "Travaux"), start=4):
        ws[f"B{row}"] = poste
        for col, value in zip("CDE", (10 * row, 9 * row, row)):
            ws[f"{col}{row}"] = value
    ws["B6"] = "SECTION - PHASE DEUX"
    ws.merge_cells("B6:E6")
    _bold(ws, "B6")
    for row, poste in enumerate(("Reception", "Garantie"), start=7):
        ws[f"B{row}"] = poste
        for col, value in zip("CDE", (10 * row, 9 * row, row)):
            ws[f"{col}{row}"] = value
    wb.create_sheet("Chapitre B")["A1"] = "Une note isolee."
    xlsx_path = path.with_name(f"{stem}.xlsx")
    wb.save(xlsx_path)
    written["xlsx"] = xlsx_path.name

    # --- projected: slide titles
    from pptx import Presentation

    deck = Presentation()
    layout = deck.slide_layouts[1]
    for title, body in (("Ouverture", "Le premier point"), ("Fermeture", "Le dernier point")):
        slide = deck.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    pptx_path = path.with_name(f"{stem}.pptx")
    deck.save(pptx_path)
    written["pptx"] = pptx_path.name

    path.write_text(_json.dumps({"files": written}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def running_headers(path: Path) -> Path:
    """A numbered line that repeats on every page, and four real headings.

    The repeated line is the trap. It looks exactly like a section — numbered,
    short, at the top of a page — and it is furniture. Read as a section it
    squats at the head of every ancestry in the document.
    """
    pages = [
        [f"1. {RUNNING_HEADER}", f"{index + 2}. {section}", f"Le corps de la partie {index + 1}."]
        for index, section in enumerate(RUNNING_HEADER_SECTIONS)
    ]
    return _pdf_pages(path, pages=pages)


#: The typed-label cases, as built trees. Each key is one case of the grammar.
OUTLINE_CASES = {
    "legal_walk": ["1. Contexte", "1.1. Perimetre", "1.2. Moyens", "2. Calendrier",
                   "2.1. Jalons", "3. Suivi"],
    # The trap: a numbered list is numbered and is not an outline. It never
    # descends, and nothing but its numbers suggests it is structural.
    "enumeration": ["1. premier article", "2. deuxieme article", "3. troisieme article",
                    "4. quatrieme article", "5. cinquieme article"],
    # The same shape, corroborated by style. The pair is the point: one rule,
    # tested from both sides, rather than one case written twice.
    "flat_corroborated": ["1. Alpha", "2. Beta", "3. Gamma", "4. Delta"],
    "short_chain": ["1. Un", "2. Deux"],
    "annex_restart": ["1. Contexte", "1.1. Perimetre", "2. Moyens",
                      "A. Annexe technique", "A.1. Contenu", "A.2. Format"],
    "stray_restart": ["1. Contexte", "1.1. Perimetre", "7. Egare", "1.2. Moyens", "2. Calendrier"],
}

#: Cases whose labels are also styled, which is what corroborates a flat chain.
OUTLINE_BOLD = {"flat_corroborated"}

#: What the outline analyzer must promote in each case, and why the rest is not.
EXPECTED_OUTLINE = {
    "legal_walk": {"promoted": 6, "walks": 1},
    "enumeration": {"promoted": 0, "reason": "flat-uncorroborated"},
    "flat_corroborated": {"promoted": 4, "walks": 1},
    "short_chain": {"promoted": 0, "reason": "chain-too-short"},
    "annex_restart": {"promoted": 6, "walks": 2},
    "stray_restart": {"promoted": 4, "reason": "illegal-step"},
}


def build_outline_trees() -> Dict[str, Tree]:
    """One tree per case, paragraphs only: the label sequence is the whole input."""
    trees: Dict[str, Tree] = {}
    for case, lines in OUTLINE_CASES.items():
        source = f"{case}.docx"
        root = Node(
            kind="document",
            provenance=_prov(source, "docx", DocumentLocator()),
            children=[
                Node(
                    kind="paragraph",
                    text=line,
                    provenance=_prov(source, "docx", DocxLocator(flow="body", paragraph=index)),
                    facts={
                        "style": "Normal",
                        "numbered": False,
                        "bold_frac": 1.0 if case in OUTLINE_BOLD else 0.0,
                    },
                )
                for index, line in enumerate(lines)
            ],
        )
        trees[case] = Tree(root=root, meta={"case": case})
    return trees


def numbered_outline_trees(path: Path) -> Path:
    """Write the typed-label trees as one canonical JSON document."""
    payload = {case: tree.to_dict() for case, tree in sorted(build_outline_trees().items())}
    path.write_text(json.dumps(payload, **CANONICAL_JSON), encoding="utf-8")
    return path



def empty_string_cells(path: Path) -> Path:
    """A form whose unfilled answers hold an empty string rather than nothing.

    Real spreadsheets are full of these: a cell that was typed into and cleared,
    or written by a tool that stores "" for an unfilled field. On screen it is
    blank. Read as content it counts as a value, and the shape of the table
    changes with nothing visible to explain it.

    Three shapes here: an empty string in a ruled answer column, a whitespace-only
    string in the same column, and a genuinely absent value beside them, so the
    three can be told apart by what the reader emits.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Border, Side

    thin = Side(style="thin")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)

    wb = Workbook()
    ws = wb.active
    ws.title = "Formulaire"
    for col, head in zip("ABC", ("Question", "Reponse", "Visa")):
        ws[f"{col}1"] = head
        _bold(ws, f"{col}1")
    for row, question in enumerate(("Organisation", "Moyens", "Delais"), start=2):
        ws[f"A{row}"] = question
        ws[f"A{row}"].border = box
        ws[f"B{row}"].border = box
        ws[f"C{row}"].border = box
    ws["B2"] = ""                    # typed into, then cleared
    ws["B3"] = "   "                 # whitespace only
    # B4 is left genuinely absent, for the contrast
    ws["C2"] = "ok"
    wb.save(path)
    return path


#: What the reader must say about each of the three shapes above. The data types
#: are the file's own words, read rather than guessed: a cell written as an empty
#: string and one written as whitespace are stored differently, and a cell that
#: was never written reports the type a spreadsheet gives an empty cell. All three
#: are blank; only the last one never held a string.
EXPECTED_EMPTY_STRINGS = {
    "B2": {"text": None, "dtype": "inlineStr", "held_a_string": True},
    "B3": {"text": None, "dtype": "s", "held_a_string": True},
    "B4": {"text": None, "dtype": "n", "held_a_string": False},
}




# ------------------------------------------------------------------ refusals




# ------------------------------------------------- grids for the analyzers

# One fixture per case of the merge-orientation taxonomy. Invented content
# throughout. EXPECTED_GRID below is the ground truth the K3 gates compare
# against — written here, next to the builder that produces it, so that a
# fixture cannot drift away from what it is supposed to demonstrate.

#: Marks a rung whose label tile exists but was left empty. A missing sub-label
#: is a fact about the document; a chain that silently skipped it would report a
#: shorter ancestry than the document actually has.
BLANK_RUNG = "(blank-label)"

EXPECTED_GRID = {
    "two_tier_header": {
        "core": "B4:C5", "header_rows": [2, 3], "label_cols": ["A"],
        "title": None, "section_rows": [], "uncertain": False, "islands": 1,
        "samples": [
            {"cell": "B4", "col_chain": ["Moyens engages", "Humains"], "row_chain": ["Lot A"]},
            {"cell": "C5", "col_chain": ["Moyens engages", "Techniques"], "row_chain": ["Lot B"]},
        ],
        "guarded": ["A2", "B2", "B3", "A4"],
    },
    "label_tiling": {
        "core": "C2:E12", "header_rows": [1], "label_cols": ["A", "B"],
        "title": None, "section_rows": [], "uncertain": False, "islands": 1,
        "band_header": ("A1:B1", "Actions"),
        "tiling": [("A2:B4", "Phase amont"), ("A5:A11", "Production"),
                   ("B5:B7", "Redaction"), ("B8:B9", "Relecture"),
                   ("B10:B11", None), ("A12:B12", "Cloture")],
        "samples": [
            {"cell": "C3", "col_chain": ["Qui"], "row_chain": ["Phase amont"]},
            {"cell": "C6", "col_chain": ["Qui"], "row_chain": ["Production", "Redaction"]},
            # the empty tile surfaces as an addressable rung, not as a hole
            {"cell": "D10", "col_chain": ["Quand"], "row_chain": ["Production", BLANK_RUNG]},
            {"cell": "E12", "col_chain": ["Statut"], "row_chain": ["Cloture"]},
        ],
        "guarded": ["A1", "C1", "A5", "B10"],
    },
    "full_width_title": {
        "core": "C4:F8", "header_rows": [3], "label_cols": ["B"],
        "title": ("B2:F2", "SUIVI BUDGETAIRE"), "section_rows": [],
        "uncertain": False, "islands": 1,
    },
    "section_row": {
        "core": "C3:D9", "header_rows": [2], "label_cols": ["A", "B"],
        "title": None, "section_rows": [6], "uncertain": False, "islands": 1,
    },
    # Three text columns under one header row: nothing structurally marks column A
    # as labels rather than content. The correct outcome is to claim none — this
    # fixture is the negative test that keeps the label rule from inventing one.
    "merged_answer_area": {
        "core": "A3:C6", "header_rows": [2], "label_cols": [],
        "title": None, "section_rows": [], "uncertain": False, "islands": 1,
    },
    "numeric_bold_header": {
        "core": "B2:E3", "header_rows": [1], "label_cols": ["A"],
        "title": None, "section_rows": [], "uncertain": False, "islands": 1,
    },
    # No header anywhere, but a real code/value contrast between the columns:
    # the absence of a header band does not make the label column disappear.
    "headerless_list": {
        "core": "B1:B4", "header_rows": [], "label_cols": ["A"],
        "title": None, "section_rows": [], "uncertain": False, "islands": 1,
    },
    "undecidable_block": {"uncertain": True, "reason": "uniform-block"},
    "overlapping_merges": {"uncertain": True, "reason": "non-laminar-band-merges"},
    "totals_row_and_column": {
        "core": "B2:D5", "header_rows": [1], "label_cols": ["A"],
        "title": None, "section_rows": [], "uncertain": False, "islands": 1,
    },
    "two_islands": {"islands": 2, "uncertain": False},
}

#: Block boundaries per sheet for the multi-object workbook, as (type, range).
#: The segmentation gate compares against this, not against its own output.
EXPECTED_BLOCKS = {
    "Lisez moi": [("text", "A1:C1"), ("table", "A3:C5"), ("text", "A7:C8")],
    "Questionnaire": [("text", "B1:D2"), ("table", "B4:D8"), ("table", "F1:F3")],
    "Vocab": [("list", "E16:E18"), ("list", "B30:B31")],
}


def _grid_sheet(title: str):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = title
    return wb, ws


def label_tiling(path: Path) -> Path:
    """A label column tiled by merges, including one tile left EMPTY.

    The empty tile is the whole point. A missing sub-label is a fact about the
    document — somebody left it blank — and a reader that silently skips it
    produces a chain with a hole in it that nothing downstream can see.
    """
    wb, ws = _grid_sheet("Suivi")
    ws["A1"] = "Actions"
    ws.merge_cells("A1:B1")
    _bold(ws, "A1")
    for col, head in zip("CDE", ("Qui", "Quand", "Statut")):
        ws[f"{col}1"] = head
        _bold(ws, f"{col}1")

    ws["A2"] = "Phase amont"
    ws.merge_cells("A2:B4")
    ws["A5"] = "Production"
    ws.merge_cells("A5:A11")
    ws["B5"] = "Redaction"
    ws.merge_cells("B5:B7")
    ws["B8"] = "Relecture"
    ws.merge_cells("B8:B9")
    ws.merge_cells("B10:B11")                 # merged, and deliberately empty
    ws["A12"] = "Cloture"
    ws.merge_cells("A12:B12")

    for row in range(2, 13):
        ws[f"C{row}"] = f"R{row}"
        ws[f"D{row}"] = f"S{row}"
        ws[f"E{row}"] = "ouvert" if row % 2 else "clos"
    wb.save(path)
    return path


def full_width_title(path: Path) -> Path:
    """A merged title spanning the whole table, above the header row."""
    wb, ws = _grid_sheet("Budget")
    ws["B2"] = "SUIVI BUDGETAIRE"
    ws.merge_cells("B2:F2")
    _bold(ws, "B2")
    for col, head in zip("BCDEF", ("Poste", "Prevu", "Engage", "Solde", "Ecart")):
        ws[f"{col}3"] = head
        _bold(ws, f"{col}3")
    for row, poste in enumerate(("Achats", "Recette", "Transport", "Divers", "Reserve"), start=4):
        ws[f"B{row}"] = poste
        for col, value in zip("CDEF", (10 * row, 8 * row, 2 * row, row)):
            ws[f"{col}{row}"] = value
    wb.save(path)
    return path


def section_row(path: Path) -> Path:
    """A merged section row cutting the data in two, headers above both halves."""
    wb, ws = _grid_sheet("Bordereau")
    for col, head in zip("ABCD", ("Lot", "Poste", "Quantite", "Unite")):
        ws[f"{col}2"] = head
        _bold(ws, f"{col}2")
    rows = [("L1", "Terrassement", 120, "m3"), ("L1", "Remblai", 80, "m3"),
            ("L1", "Evacuation", 40, "m3")]
    for index, (lot, poste, quantite, unite) in enumerate(rows, start=3):
        ws[f"A{index}"], ws[f"B{index}"] = lot, poste
        ws[f"C{index}"], ws[f"D{index}"] = quantite, unite
    ws["A6"] = "SECTION - SECOND OEUVRE"
    ws.merge_cells("A6:D6")
    _bold(ws, "A6")
    rows = [("L2", "Cloisons", 60, "m2"), ("L2", "Peinture", 200, "m2"),
            ("L2", "Sols", 150, "m2")]
    for index, (lot, poste, quantite, unite) in enumerate(rows, start=7):
        ws[f"A{index}"], ws[f"B{index}"] = lot, poste
        ws[f"C{index}"], ws[f"D{index}"] = quantite, unite
    wb.save(path)
    return path


def merged_answer_area(path: Path) -> Path:
    """A merged region inside the core: one answer spanning several cells."""
    wb, ws = _grid_sheet("Reponses")
    # Every column here is text, so no column is structurally a label column.
    # That is the point: the analyzer must claim none rather than guess.
    for col, head in zip("ABC", ("Question", "Reponse", "Commentaire")):
        ws[f"{col}2"] = head
        _bold(ws, f"{col}2")
    for row, question in enumerate(("Organisation", "Moyens", "Delais", "Suivi"), start=3):
        ws[f"A{row}"] = question
    ws["B3"] = "Une reponse qui couvre deux lignes"
    ws.merge_cells("B3:B4")
    ws["B5"] = "Une autre"
    ws["B6"] = "Une troisieme"
    for row in range(3, 7):
        ws[f"C{row}"] = f"note {row}"
    wb.save(path)
    return path


def headerless_list(path: Path) -> Path:
    """A run of values with no header at all. Abstaining on the band is correct."""
    wb, ws = _grid_sheet("Liste")
    for row, (code, value) in enumerate(
        (("AA", 12), ("AB", 14), ("AC", 9), ("AD", 21)), start=1
    ):
        ws[f"A{row}"], ws[f"B{row}"] = code, value
    wb.save(path)
    return path


def undecidable_block(path: Path) -> Path:
    """A uniform block: same type everywhere, no style, no border, no evidence.

    There is nothing here to decide with, so the only correct outcome is to say
    so. A fixture that could be resolved by a clever rule would not test
    abstention; it would test cleverness.
    """
    wb, ws = _grid_sheet("Ambigu")
    for row in range(1, 5):
        for col in "ABC":
            ws[f"{col}{row}"] = f"{col}{row}"
    wb.save(path)
    return path


def totals_row_and_column(path: Path) -> Path:
    """A bottom total row and a right total column: both are data, not structure."""
    wb, ws = _grid_sheet("Totaux")
    for col, head in zip("ABCD", ("Poste", "T1", "T2", "Total")):
        ws[f"{col}1"] = head
        _bold(ws, f"{col}1")
    for row, poste in enumerate(("Achats", "Ventes", "Stock"), start=2):
        ws[f"A{row}"] = poste
        ws[f"B{row}"], ws[f"C{row}"] = 10 * row, 5 * row
        ws[f"D{row}"] = 15 * row
    ws["A5"] = "TOTAL"
    _bold(ws, "A5")
    ws["B5"], ws["C5"], ws["D5"] = 90, 45, 135
    wb.save(path)
    return path


def two_islands(path: Path) -> Path:
    """Two disconnected value regions inside one bordered box.

    The box says "one block"; the content says "two". The analyzer must report
    the disagreement and must NOT resolve it by re-segmenting on its own
    authority — the border is evidence somebody drew deliberately.
    """
    from openpyxl.styles import Border, Side

    wb, ws = _grid_sheet("Volets")
    thin = Side(style="thin")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws["A1"], ws["B1"] = "Volet gauche", "Valeur"
    _bold(ws, "A1")
    _bold(ws, "B1")
    for row, (label, value) in enumerate((("Alpha", 3), ("Beta", 5)), start=2):
        ws[f"A{row}"], ws[f"B{row}"] = label, value

    ws["E1"], ws["F1"] = "Volet droit", "Valeur"
    _bold(ws, "E1")
    _bold(ws, "F1")
    for row, (label, value) in enumerate((("Gamma", 7), ("Delta", 9)), start=2):
        ws[f"E{row}"], ws[f"F{row}"] = label, value

    for row in range(1, 4):                      # one border around both islands
        for col in "ABCDEF":
            ws[f"{col}{row}"].border = box
    wb.save(path)
    return path


def overlapping_merges(path: Path) -> Path:
    """Two header merges whose column ranges overlap without containment.

    A header band is a nesting of spans: each tier refines the one above it.
    These two refine nothing — [A,B] and [B,C] cross. There is no tree here, so
    there is no chain to read, and guessing one would produce ancestry that the
    document does not support.
    """
    wb, ws = _grid_sheet("Chevauchement")
    ws["A1"] = "Groupe un"
    ws.merge_cells("A1:B1")                      # columns 1-2
    _bold(ws, "A1")
    ws["B2"] = "Groupe deux"
    ws.merge_cells("B2:C2")                      # columns 2-3 — crosses, never nests
    _bold(ws, "B2")
    for row in range(3, 6):
        for col_index, col in enumerate("ABC", start=1):
            ws[f"{col}{row}"] = row * 10 + col_index
    wb.save(path)
    return path


def ambiguous_layout(path: Path) -> Path:
    """One header-ish row over a single column: neither clearly a table nor a list."""
    wb, ws = _grid_sheet("Indecis")
    ws["A1"] = "Rubrique"
    _bold(ws, "A1")
    for row, value in enumerate(("Premiere", "Deuxieme", "Troisieme"), start=2):
        ws[f"A{row}"] = value
    wb.save(path)
    return path



# --------------------------------------------------------- cross-format twins

# The same logical table, written once as a spreadsheet and once as a
# word-processing document. Same labels, same tiers, same values, same shape.
# Everything that differs between the two files is format: merged ranges against
# column spans and vertical merges, recorded data types against text that happens
# to look like a number. If one grid core really consumes both, these two must
# analyse identically — and that is an equality a test can assert rather than a
# resemblance a reader has to judge.

TWIN_HEADER_TIER1 = "Moyens"
TWIN_SUBHEADS = ("Humains", "Techniques")
TWIN_LABEL = "Lot"
TWIN_ROWS = (("Lot A", "4", "2"), ("Lot B", "7", "3"))

#: What both twins must produce, in the neutral one-based A1 vocabulary.
EXPECTED_TWIN = {
    "core": "B3:C4",
    "header_rows": [1, 2],
    "label_cols": ["A"],
    "samples": [
        {"cell": "B3", "col_chain": [TWIN_HEADER_TIER1, "Humains"], "row_chain": ["Lot A"]},
        {"cell": "C4", "col_chain": [TWIN_HEADER_TIER1, "Techniques"], "row_chain": ["Lot B"]},
    ],
}


def twin_grid_xlsx(path: Path) -> Path:
    """The twin, as a spreadsheet: merged ranges and recorded data types."""
    wb, ws = _grid_sheet("Twin")
    ws["A1"] = TWIN_LABEL
    ws.merge_cells("A1:A2")                       # the label spans both tiers
    _bold(ws, "A1")
    ws["B1"] = TWIN_HEADER_TIER1
    ws.merge_cells("B1:C1")                       # one super-header over two columns
    _bold(ws, "B1")
    for col, head in zip("BC", TWIN_SUBHEADS):
        ws[f"{col}2"] = head
        _bold(ws, f"{col}2")
    for index, (label, left, right) in enumerate(TWIN_ROWS, start=3):
        ws[f"A{index}"] = label
        ws[f"B{index}"] = int(left)
        ws[f"C{index}"] = int(right)
    wb.save(path)
    return path


def twin_grid_pptx(path: Path) -> Path:
    """The same twin again, on a slide.

    A third format for the same logical table. If one grid core really reads them
    all, this must analyse identically to the other two — and a slide is the least
    grid-like place any of them lives, which is what makes it worth asking.
    """
    from pptx import Presentation
    from pptx.util import Inches

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])       # blank
    shape = slide.shapes.add_table(4, 3, Inches(1), Inches(1), Inches(6), Inches(3))
    table = shape.table

    table.cell(0, 0).text = TWIN_LABEL
    table.cell(0, 0).merge(table.cell(1, 0))                   # the label spans both tiers
    table.cell(0, 1).merge(table.cell(0, 2))                   # one super-header, two columns
    table.cell(0, 1).text = TWIN_HEADER_TIER1
    for col, head in enumerate(TWIN_SUBHEADS, start=1):
        table.cell(1, col).text = head
    for index, (label, left, right) in enumerate(TWIN_ROWS, start=2):
        table.cell(index, 0).text = label
        table.cell(index, 1).text = left
        table.cell(index, 2).text = right

    for cell in (table.cell(0, 0), table.cell(0, 1), table.cell(1, 1), table.cell(1, 2)):
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True

    deck.save(path)
    return path


def twin_grid_docx(path: Path) -> Path:
    """The twin, as a document: a column span, a vertical merge, numbers as text."""
    from docx import Document

    doc = Document()
    table = _table(doc, 4, 3)
    table.cell(0, 0).text = TWIN_LABEL
    table.cell(0, 0).merge(table.cell(1, 0))      # the label spans both tiers
    span = table.cell(0, 1).merge(table.cell(0, 2))
    span.text = TWIN_HEADER_TIER1                 # one super-header over two columns
    for col, head in enumerate(TWIN_SUBHEADS, start=1):
        table.cell(1, col).text = head
    for index, (label, left, right) in enumerate(TWIN_ROWS, start=2):
        table.cell(index, 0).text = label
        table.cell(index, 1).text = left
        table.cell(index, 2).text = right
    for cell in (table.cell(0, 0), span, table.cell(1, 1), table.cell(1, 2)):
        for run in cell.paragraphs[0].runs:
            run.bold = True
    doc.save(path)
    return path


# ------------------------------------------------------------- raw xml helpers

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _qn(tag: str) -> str:
    prefix, name = tag.split(":", 1)
    return f"{{{_W}}}{name}" if prefix == "w" else tag


def _shade(cell, fill: str) -> None:
    """Paint a cell. Shading is a style fact a reader must be able to observe."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def _form_text(paragraph) -> None:
    """A legacy form field: a slot that carries a marker rather than a border."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.text = " FORMTEXT "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


def _form_checkbox(paragraph) -> None:
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.text = " FORMCHECKBOX "
    run._r.append(begin)
    run._r.append(instr)


def _content_control(cell) -> None:
    """A structured document tag: the modern spelling of the same intent."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    sdt = OxmlElement("w:sdt")
    props = OxmlElement("w:sdtPr")
    alias = OxmlElement("w:alias")
    alias.set(qn("w:val"), "answer")
    props.append(alias)
    content = OxmlElement("w:sdtContent")
    sdt.append(props)
    sdt.append(content)
    cell._tc.append(sdt)


# ------------------------------------------------------------------ workbooks

SHADE_FILL = "D9D9D9"


def _bold(ws, ref: str) -> None:
    from openpyxl.styles import Font

    ws[ref].font = Font(bold=True)


def numeric_bold_header(path: Path) -> Path:
    """Years as headers: bold, and numeric. Neither fact alone decides anything.

    This is the fixture that keeps a reader honest. A reader that discards
    numeric cells as "not header material", or that rewrites a year as a label,
    has interpreted where it was asked to observe.
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Trajectoire"
    for col, year in enumerate((2024, 2025, 2026), start=2):
        ws.cell(row=1, column=col, value=year)
        _bold(ws, ws.cell(row=1, column=col).coordinate)
    ws["A1"] = "Poste"
    _bold(ws, "A1")
    for row, (label, values) in enumerate(
        (("Effectif", (12, 14, 15)), ("Budget", (100.5, 110.0, 118.25))), start=2
    ):
        ws.cell(row=row, column=1, value=label)
        for col, value in enumerate(values, start=2):
            cell = ws.cell(row=row, column=col, value=value)
            if label == "Budget":
                cell.number_format = "#,##0.00"
    ws["E1"] = "Total"                       # a total column, headed like the rest
    _bold(ws, "E1")
    ws["E2"] = "=SUM(B2:D2)"                 # exactly one formula in this fixture
    ws["E3"] = 328.75
    ws.protection.sheet = False
    wb.save(path)
    return path


def two_tier_header(path: Path) -> Path:
    """Two header tiers, with one label column merged vertically across both.

    The vertical merge is the point: its facts belong to the anchor, and the
    continuation beneath it must not acquire a copy of them.
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Charges"
    ws["A2"] = "Lot"
    _bold(ws, "A2")
    ws.merge_cells("A2:A3")                     # one label, two tiers deep
    ws["B2"] = "Moyens engages"
    _bold(ws, "B2")
    ws.merge_cells("B2:C2")                     # one super-header over two columns
    ws["B3"], ws["C3"] = "Humains", "Techniques"
    _bold(ws, "B3")
    _bold(ws, "C3")
    for row, (lot, humains, techniques) in enumerate(
        (("Lot A", 4, 2), ("Lot B", 7, 3)), start=4
    ):
        ws.cell(row=row, column=1, value=lot)
        ws.cell(row=row, column=2, value=humains)
        ws.cell(row=row, column=3, value=techniques)
    wb.save(path)
    return path


def mixed_workbook(path: Path) -> Path:
    """Several logical objects per sheet, plus a sheet that is only vocabulary.

    Sheets: an instruction sheet whose title and guidance are merged text blocks
    around a bordered table; a form sheet with a blank answer slot inside its
    border and a blank spacer row crossing it; a hidden vocabulary sheet holding
    two disjoint one-dimensional lists.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Border, Side
    from openpyxl.worksheet.table import Table, TableStyleInfo

    thin = Side(style="thin")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)

    wb = Workbook()

    ws = wb.active
    ws.title = "Lisez moi"
    ws["A1"] = "Notice fictive"
    ws.merge_cells("A1:C1")
    _bold(ws, "A1")
    for row, (crit, poids) in enumerate(
        (("Critere", "Poids"), ("Delai", 30), ("Qualite", 70)), start=3
    ):
        ws.cell(row=row, column=1, value=crit).border = box
        ws.cell(row=row, column=2, value=poids).border = box
        ws.cell(row=row, column=3, value=None).border = box
    _bold(ws, "A3")
    _bold(ws, "B3")
    ws["A7"] = "Consignes : renseigner chaque ligne."
    ws.merge_cells("A7:C8")

    ws = wb.create_sheet("Questionnaire")
    ws["B1"] = "Formulaire"
    ws.merge_cells("B1:D2")
    _bold(ws, "B1")
    for row, question in enumerate(("Q1 Organisation", "Q2 Moyens", "Q3 Delais"), start=4):
        ws.cell(row=row, column=2, value=question).border = box
        ws.cell(row=row, column=3, value=None).border = box     # the answer slot
        ws.cell(row=row, column=4, value=None).border = box
    for col in (2, 3, 4):                                       # a blank spacer row
        ws.cell(row=7, column=col, value=None).border = box
    for row, question in enumerate(("Q4 Suivi",), start=8):
        ws.cell(row=row, column=2, value=question).border = box
        ws.cell(row=row, column=3, value=None).border = box
        ws.cell(row=row, column=4, value=None).border = box
    ws["F1"], ws["F2"], ws["F3"] = "Reference", "R-1", "R-2"
    table = Table(displayName="References", ref="F1:F3")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showRowStripes=True)
    ws.add_table(table)

    ws = wb.create_sheet("Vocab")
    for row, value in enumerate(("Oui", "Non", "Sans objet"), start=16):
        ws.cell(row=row, column=5, value=value)
    for row, value in enumerate(("Interne", "Externe"), start=30):
        ws.cell(row=row, column=2, value=value)
    ws.sheet_state = "hidden"

    wb.save(path)
    return path


# ------------------------------------------------------- word-processing files

#: What each word-processing fixture must contain, for the K2 gate to check
#: against something written down rather than against the reader's own output.
EXPECTED_DOCX = {
    "docx_two_tier": {
        "flow": "body",
        "table_index": 0,
        "n_rows": 4,
        "n_grid_cols": 4,
        "grid_spans": [(0, 1, 3)],          # (row, grid column, span)
        "vmerge_restart": [(0, 0)],
        "vmerge_continue": [(1, 0)],
        "shaded": [(0, 0), (0, 1)],
        "n_empty": 0,
    },
    "docx_label_tiling": {
        "flow": "body",
        "table_index": 0,
        "n_rows": 4,
        "n_grid_cols": 3,
        "vmerge_restart": [(0, 0), (2, 0)],
        "vmerge_continue": [(1, 0), (3, 0)],
        "slot_col": 2,
        "n_empty": 4,
    },
    "docx_markers": {"flow": "body", "table_index": 0, "in_cell_markers": 2, "plain_slots": 1},
    "docx_header_stream": {"flow": "header", "table_index": 0, "n_rows": 1, "n_grid_cols": 2},
    "docx_nested": {"parent_flow": "body", "nested_rows": 2, "nested_cols": 2},
    "docx_twin_pair": {"answers": 3},
}


def _table(doc, rows: int, cols: int):
    table = doc.add_table(rows=rows, cols=cols)
    table.style = "Table Grid"
    return table


def docx_two_tier(path: Path) -> Path:
    """A two-tier column header: one span across three columns, one merge down."""
    from docx import Document

    doc = Document()
    table = _table(doc, 4, 4)
    table.cell(0, 0).text = "Lot"
    table.cell(0, 0).merge(table.cell(1, 0))                    # down through both tiers
    span = table.cell(0, 1).merge(table.cell(0, 3))             # across three columns
    span.text = "Moyens mobilises"
    for col, name in enumerate(("Humains", "Techniques", "Total"), start=1):
        table.cell(1, col).text = name
    for row, (lot, a, b, total) in enumerate(
        (("Lot A", "4", "2", "6"), ("Lot B", "7", "3", "10")), start=2
    ):
        for col, value in enumerate((lot, a, b, total)):
            table.cell(row, col).text = value
    for cell in (table.cell(0, 0), span):
        _shade(cell, SHADE_FILL)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for col in range(1, 4):
        for run in table.cell(1, col).paragraphs[0].runs:
            run.bold = True
    doc.save(path)
    return path


def docx_label_tiling(path: Path) -> Path:
    """A label column tiled by vertical merges, a detail column, an empty slot column."""
    from docx import Document

    doc = Document()
    table = _table(doc, 4, 3)
    table.cell(0, 0).text = "Phase amont"
    table.cell(0, 0).merge(table.cell(1, 0))
    table.cell(2, 0).text = "Phase aval"
    table.cell(2, 0).merge(table.cell(3, 0))
    for row, detail in enumerate(("cadrer", "planifier", "livrer", "clore")):
        table.cell(row, 1).text = detail
        table.cell(row, 2).text = ""                            # the slot, left blank
    doc.save(path)
    return path


def docx_layout_prose(path: Path) -> Path:
    """Two tables that are not tables: prose in a grid, no style, no borders."""
    from docx import Document

    doc = Document()
    one = doc.add_table(rows=1, cols=1)
    one.cell(0, 0).text = "Un paragraphe de presentation, place dans une cellule unique."
    doc.add_paragraph("")
    two = doc.add_table(rows=1, cols=2)
    two.cell(0, 0).text = "Colonne de gauche, du texte suivi."
    two.cell(0, 1).text = "Colonne de droite, du texte suivi egalement."
    doc.save(path)
    return path


def docx_markers(path: Path) -> Path:
    """Markers inside cells, a plain empty slot, and one marker outside any table."""
    from docx import Document

    doc = Document()
    table = _table(doc, 3, 2)
    table.cell(0, 0).text = "Question a"
    _form_text(table.cell(0, 1).paragraphs[0])
    table.cell(1, 0).text = "Question b"
    _content_control(table.cell(1, 1))
    table.cell(2, 0).text = "Question c"
    table.cell(2, 1).text = ""                                  # a slot with no marker
    _form_checkbox(doc.add_paragraph("Je certifie l'exactitude : "))
    doc.save(path)
    return path


def docx_header_stream(path: Path) -> Path:
    """A table living in the page header, invisible to a reader that walks the body."""
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    header = doc.sections[0].header
    table = header.add_table(rows=1, cols=2, width=Inches(6))
    table.cell(0, 0).text = "Reference"
    table.cell(0, 1).text = "DOC-001"
    doc.add_paragraph("Le corps du document, avec sa propre table.")
    body = _table(doc, 1, 1)
    body.cell(0, 0).text = "Table du corps"
    doc.save(path)
    return path


def docx_nested(path: Path) -> Path:
    """A table inside a cell of another table."""
    from docx import Document

    doc = Document()
    outer = _table(doc, 1, 2)
    outer.cell(0, 0).text = "Cellule porteuse"
    inner = outer.cell(0, 1).add_table(rows=2, cols=2)
    inner.style = "Table Grid"
    for row in range(2):
        for col in range(2):
            inner.cell(row, col).text = f"n{row}{col}"
    doc.save(path)
    return path


def docx_twin_pair(path: Path) -> Path:
    """A blank form and its filled twin.

    Writes the blank form at `path` and the filled one beside it, suffixed
    `_filled`. Returns the blank one: the pair is addressed from it.
    """
    from docx import Document

    filled_path = path.with_name(f"{path.stem}_filled{path.suffix}")
    answers = ("Trois equipes", "Douze mois", "Deux sites")

    for target, values in ((path, ("", "", "")), (filled_path, answers)):
        doc = Document()
        table = _table(doc, 4, 2)
        table.cell(0, 0).text = "Question"
        table.cell(0, 1).text = "Reponse"
        table.cell(1, 0).text = "Moyens"
        table.cell(2, 0).text = "Duree"
        table.cell(3, 0).text = "Sites"
        for row, value in enumerate(values, start=1):
            table.cell(row, 1).text = value
        doc.save(target)

    return path


# ---------------------------------------------------- presentations, markdown

def slide_deck(path: Path) -> Path:
    """Two slides, each with a title, a body, and notes that are not on the slide."""
    from pptx import Presentation

    deck = Presentation()
    layout = deck.slide_layouts[1]
    for title, body, note in (
        ("Premiere diapositive", "Un point, puis un autre", "Ce que dit l'orateur, pas la diapo."),
        ("Seconde diapositive", "Une conclusion", "Une remarque reservee a l'orateur."),
    ):
        slide = deck.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        slide.notes_slide.notes_text_frame.text = note
    deck.save(path)
    return path


def markdown_document(path: Path) -> Path:
    """Front matter, then headings and paragraphs.

    The third front-matter line is deliberately not a `key: value` pair: it is
    what sends the reader down its reserved `_unparsed` key, which is both the
    one open-vocabulary reserved name in the kernel and one of the two facts
    whose value is a list rather than a scalar (K2.4, K2.21).
    """
    path.write_text(
        "---\n"
        "title: Document de controle\n"
        "language: fr\n"
        "une ligne sans deux-points\n"
        "---\n"
        "\n"
        "# Titre principal\n"
        "\n"
        "Un premier paragraphe, sur une ligne connue.\n"
        "\n"
        "## Sous-titre\n"
        "\n"
        "Un second paragraphe.\n",
        encoding="utf-8",
    )
    return path



# ------------------------------------------------- headings shown, not declared

def _pdf_typeset(path: Path, pages: list[list[list[tuple[str, float]]]]) -> Path:
    """Write a laid-out document line by line, each line a list of (text, size).

    A line holding more than one segment is written as several text-showing
    operations at the same vertical position, which is what a real document does
    whenever a line changes font mid-way — and what the line assembly of K3.59
    has to put back together.
    """
    page_count = len(pages)
    first_page_obj = 4
    content_obj = first_page_obj + page_count

    objects: list[bytes] = []
    kids = " ".join(f"{first_page_obj + i} 0 R" for i in range(page_count))
    objects.append(f"<< /Type /Catalog /Pages 2 0 R >>".encode())
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index in range(page_count):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_obj + index} 0 R >>"
            ).encode()
        )

    for lines in pages:
        payload = b""
        y = 800
        for line in lines:
            x = 72
            for text, size in line:
                payload += (
                    b"BT /F1 " + f"{size:g}".encode() + b" Tf "
                    + f"{x:g} {y:g}".encode() + b" Td ("
                    + _pdf_escape(text) + b") Tj ET\n"
                )
                x += 6 * len(text)
            y -= 26
        objects.append(_pdf_stream(payload))

    path.write_bytes(_pdf_assemble(objects))
    return path


#: A body line: long, ends in a full stop, set at the body size.
_BODY = (
    "Cette phrase de corps de texte occupe une ligne entiere et se termine "
    "par un point final."
)


def format_headings_pdf(path: Path) -> Path:
    """Four sizes: a title, a heading tier split across two nearby sizes, a body,
    and one decorative line nothing supports.

    Every trap the gauntlet has to refuse is present at a promotable size, so
    that refusing them cannot be confused with never having seen them: a line
    that ends in a full stop, and a line too long to be a heading. The heading
    tier is written at 15 and 15.5 so that a rule which treats them as two tiers
    reports two levels where a reader sees one. The 18-point line is heading
    shaped and alone: below the title factor, unsupported as a ladder tier, and
    therefore droppable only for want of support.
    """
    body = [[(_BODY, 12.0)]] * 4
    return _pdf_typeset(path, [
        [
            [("Rapport annuel de conformite", 24.0)],
            *body,
            [("Perimetre", 15.0)],
            *body,
            [("Gouvernance", 15.5)],
            *body,
            [("Mesures ", 15.0), ("techniques", 15.0)],       # one line, two segments
            *body,
            [("Une ligne decorative isolee", 18.0)],
            *body,
            [("Ceci ressemble a un titre mais se termine par un point.", 15.0)],
            [(
                "Une ligne beaucoup trop longue pour etre un titre car elle "
                "enchaine les mots bien au dela de ce qu un lecteur accepterait "
                "de lire comme un intitule de section quelconque", 15.0,
            )],
            *body,
        ],
    ])


def _weighted(doc, parts, size=None):
    """One paragraph from (text, bold) parts, so the bold fraction is exact."""
    from docx.shared import Pt

    paragraph = doc.add_paragraph()
    for text, bold in parts:
        run = paragraph.add_run(text)
        run.bold = bold
        if size is not None:
            run.font.size = Pt(size)
    return paragraph


#: Twenty words, one of them bold. The population this fixture exists for: a
#: boolean `bold` reports True here and True on a fully bold heading, so a rule
#: reading the boolean promotes both and a rule reading the fraction promotes
#: neither this nor anything like it.
_ONE_IN_TWENTY = (
    [("Le prestataire applique la procedure ", False), ("integralement", True)]
    + [(" et la documente selon les regles internes en vigueur cette annee la.", False)]
)


def format_headings_docx(path: Path) -> Path:
    """A document that styles nothing: its headings are carried by weight alone.

    Every trap is present, and each is a different way of being bold without being
    a heading: an emphasis inside a sentence, a fully bold line that ends in a full
    stop, and a bold line far too long to be an intitule. The one-bold-word-in-
    twenty paragraph is the case a boolean cannot tell from a heading at all.
    """
    from docx import Document

    doc = Document()
    _weighted(doc, [("Perimetre de la prestation", True)], size=11)
    _weighted(doc, [("Le corps du document se lit normalement et se termine par un "
                     "point final.", False)], size=11)
    _weighted(doc, [("Gouvernance et pilotage", True)], size=11)
    _weighted(doc, [("Une seconde phrase de corps, sans aucune emphase particuliere.",
                     False)], size=11)
    _weighted(doc, [("Moyens techniques", True)], size=11)
    _weighted(doc, _ONE_IN_TWENTY, size=11)                       # the negative case
    _weighted(doc, [("Cette ligne est en gras mais se termine par un point.", True)],
              size=11)
    _weighted(doc, [("Une ligne entierement en gras beaucoup trop longue pour etre un "
                     "intitule car elle enchaine bien plus de mots qu un lecteur "
                     "accepterait", True)], size=11)
    doc.save(path)
    return path


def no_weight_contrast(path: Path) -> Path:
    """Every run identical: the negative for the weight branch.

    Nothing here is bolder than anything else, so a rule that reads contrast finds
    none and a rule that reads its own defaults finds a document full of headings.
    """
    from docx import Document

    doc = Document()
    for text in ("Introduction", "Le corps du document se lit normalement.",
                 "Perimetre", "Une seconde phrase de corps sans emphase.",
                 "Gouvernance"):
        _weighted(doc, [(text, False)], size=11)
    doc.save(path)
    return path


def _pdf_matrix_pages(path: Path, pages) -> Path:
    """Write pages whose type is sized through explicit matrices.

    Each entry is (text, tf, tm, cm): the size operand, the text matrix and the
    page transformation to set around it. This is the only way to write the same
    visual type four different ways, which is what the fixture below needs.
    """
    page_count = len(pages)
    first_page_obj = 4
    content_obj = first_page_obj + page_count

    objects: list[bytes] = []
    kids = " ".join(f"{first_page_obj + i} 0 R" for i in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index in range(page_count):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_obj + index} 0 R >>"
            ).encode()
        )

    for entries in pages:
        payload = b""
        for text, tf, tm, cm in entries:
            matrix = " ".join(f"{v:g}" for v in tm)
            payload += b"q\n"
            if cm is not None:
                payload += (" ".join(f"{v:g}" for v in cm) + " cm\n").encode("ascii")
            payload += (
                b"BT /F1 " + f"{tf:g}".encode() + b" Tf " + matrix.encode() + b" Tm ("
                + _pdf_escape(text) + b") Tj ET\nQ\n"
            )
        objects.append(_pdf_stream(payload))

    path.write_bytes(_pdf_assemble(objects))
    return path


#: The two effective sizes the four pages below must all be read at. Chosen so
#: every route reaches them without rounding: 0.5 and 18 and 10 are exact in
#: binary, so a difference in the reading is a difference in the method.
TYPE_SCALE_SIZES = (18.0, 10.0)


def pdf_type_scales(path: Path) -> Path:
    """One document, four pages, the same type sized four different ways.

    A reader is told a size by an operand and by two matrices, and only their
    product is the size on the page. These pages set the SAME visual sizes and
    disagree about which of the three carries them:

      page 1  the operand carries it        Tf 18            identity matrices
      page 2  the text matrix carries it    Tf 1             Tm scaled by 18
      page 3  the page transform carries it Tf 36            cm scaled by 0.5
      page 4  as page 2, rotated a quarter turn — the scale is the same, and a
              reader that takes a single matrix cell rather than the length of a
              column will read zero here

    Every page must read as {18.0, 10.0}. A reader reporting the operand alone
    passes page 1 and fails the other three.
    """
    big, small = TYPE_SCALE_SIZES
    ident = (1, 0, 0, 1, 72, 700)

    page_operand = [("Titre du document", big, ident, None),
                    ("Corps du texte", small, (1, 0, 0, 1, 72, 650), None)]
    page_text_matrix = [("Titre du document", 1, (big, 0, 0, big, 72, 700), None),
                        ("Corps du texte", 1, (small, 0, 0, small, 72, 650), None)]
    page_page_transform = [("Titre du document", big * 2, ident, (0.5, 0, 0, 0.5, 0, 0)),
                           ("Corps du texte", small * 2, (1, 0, 0, 1, 72, 650),
                            (0.5, 0, 0, 0.5, 0, 0))]
    page_rotated = [("Titre du document", 1, (0, big, -big, 0, 300, 400), None),
                    ("Corps du texte", 1, (0, small, -small, 0, 350, 400), None)]

    return _pdf_matrix_pages(path, [page_operand, page_text_matrix,
                                    page_page_transform, page_rotated])


def pdf_declared_outline(path: Path) -> Path:
    """A document that declares its own outline AND shows size contrast.

    Both signals are present and they are not equals: the outline is what the
    document says about itself, the sizes are what a reader infers from how it
    looks. Set beside each other, the inference has nothing to add and every
    chance to disagree — so the analyzer must decline, and say how much it
    declined to promote.

    Written with the same title text in the outline and on the page, so that a
    reader which mistook one for the other would be caught by the count rather
    than by the words.
    """
    titles = ["Perimetre", "Gouvernance", "Moyens techniques"]
    body = ("Cette phrase de corps de texte occupe une ligne entiere et se "
            "termine par un point final.")

    objects: list[bytes] = []
    outline_root, first_item = 6, 7
    objects.append(f"<< /Type /Catalog /Pages 2 0 R /Outlines {outline_root} 0 R >>".encode())
    objects.append(b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                   b"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>")

    payload = b""
    y = 800
    for title in titles:                       # the same headings, set larger
        payload += (b"BT /F1 16 Tf 72 " + f"{y:g}".encode() + b" Td ("
                    + _pdf_escape(title) + b") Tj ET\n")
        y -= 26
        for _ in range(3):
            payload += (b"BT /F1 10 Tf 72 " + f"{y:g}".encode() + b" Td ("
                        + _pdf_escape(body) + b") Tj ET\n")
            y -= 26
    objects.append(_pdf_stream(payload))

    objects.append(
        (f"<< /Type /Outlines /First {first_item} 0 R "
         f"/Last {first_item + len(titles) - 1} 0 R /Count {len(titles)} >>").encode())
    for index, title in enumerate(titles):
        parts = [b"<< /Title (" + _pdf_escape(title) + b")",
                 b" /Dest [4 0 R /Fit]",
                 f" /Parent {outline_root} 0 R".encode()]
        if index > 0:
            parts.append(f" /Prev {first_item + index - 1} 0 R".encode())
        if index < len(titles) - 1:
            parts.append(f" /Next {first_item + index + 1} 0 R".encode())
        parts.append(b" >>")
        objects.append(b"".join(parts))

    path.write_bytes(_pdf_assemble(objects))
    return path


# ------------------------------------------------------- objects: images a file holds

#: Four by four, eight bits, grey. Generated here, so the origin of every byte in
#: every fixture picture is this function and nothing else.
def _grey_pixels(width: int = 4, height: int = 4) -> bytes:
    step = max(1, 255 // max(1, width * height - 1))
    return bytes(min(255, i * step) for i in range(width * height))


def _pdf_with_images(path: Path, placements, inline: bool = False,
                     text: str | None = None) -> Path:
    """One page carrying one image XObject, drawn once per placement.

    `placements` are (a, b, c, d, e, f) matrices: the image occupies the unit
    square under each, which is what the reader has to recover. Passing two
    placements draws the SAME object twice, which is the whole point of the
    fixture that uses it.
    """
    pixels = _grey_pixels()
    resources = "/XObject << /Im0 6 0 R >>"
    if text is not None:
        resources = "/Font << /F1 3 0 R >> " + resources

    payload = b""
    for matrix in placements:
        payload += (b"q " + " ".join(f"{v:g}" for v in matrix).encode("ascii")
                    + b" cm /Im0 Do Q\n")
    if inline:
        # BI/ID/EI: the bytes live in the content stream, not in an object.
        payload += (b"q 20 0 0 20 72 500 cm\n"
                    b"BI /W 2 /H 2 /CS /G /BPC 8 ID " + bytes([0, 85, 170, 255])
                    + b" EI Q\n")
    if text is not None:
        payload += (b"BT /F1 10 Tf 72 300 Td (" + _pdf_escape(text) + b") Tj ET\n")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << "
         + resources + " >> /Contents 5 0 R >>").encode("ascii"),
        _pdf_stream(payload),
        (b"<< /Type /XObject /Subtype /Image /Width 4 /Height 4 "
         b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length "
         + str(len(pixels)).encode("ascii") + b" >>\nstream\n" + pixels
         + b"\nendstream"),
    ]
    path.write_bytes(_pdf_assemble(objects))
    return path


def _form(payload: bytes, resources: str, matrix: str = "") -> bytes:
    """A Form XObject: a content stream invoked by name, with its own resources."""
    extra = (" /Type /XObject /Subtype /Form /BBox [0 0 595 842] "
             + matrix + " /Resources << " + resources + " >>")
    return _pdf_stream(payload, extra=extra)


def pdf_image_in_form(path: Path) -> Path:
    """One image drawn twice, never at page level: once inside a form, once two deep.

    A page description may put a picture on the page without ever naming it in
    the page's own resources. Both placements below are of the SAME image, and
    neither is reachable by walking the page content stream alone.

    The geometry is chosen so that a reader which descends but does not compose
    the transformation is caught as surely as one which does not descend:

        page      q 2 0 0 2 10 20 cm /Fm0 Do Q
        Fm0       q 50 0 0 25 0 0 cm /Im0 Do Q       -> (10, 20) .. (110, 70)
                  q 1 0 0 1 100 200 cm /Fm1 Do Q
        Fm1       /Matrix [1 0 0 1 5 5]
                  q 30 0 0 15 0 0 cm /Im0 Do Q       -> (220, 430) .. (280, 460)

    The inner form carries a non-identity `/Matrix`, which maps form space into
    the space that invoked it and composes like any other transformation. A
    reader that ignores it is out by exactly the ten points it contributes.
    """
    pixels = _grey_pixels()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
         b"/Resources << /XObject << /Fm0 5 0 R >> >> /Contents 4 0 R >>"),
        _pdf_stream(b"q 2 0 0 2 10 20 cm /Fm0 Do Q\n"),
        _form(b"q 50 0 0 25 0 0 cm /Im0 Do Q\nq 1 0 0 1 100 200 cm /Fm1 Do Q\n",
              "/XObject << /Im0 7 0 R /Fm1 6 0 R >>"),
        _form(b"q 30 0 0 15 0 0 cm /Im0 Do Q\n",
              "/XObject << /Im0 7 0 R >>", matrix="/Matrix [1 0 0 1 5 5]"),
        (b"<< /Type /XObject /Subtype /Image /Width 4 /Height 4 "
         b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length "
         + str(len(pixels)).encode("ascii") + b" >>\nstream\n" + pixels
         + b"\nendstream"),
    ]
    path.write_bytes(_pdf_assemble(objects))
    return path


#: How deep the chain in `pdf_form_pathologies` goes. One more than the reader's
#: declared limit, so the limit is exercised rather than described.
FORM_CHAIN = 8


def pdf_form_pathologies(path: Path) -> Path:
    """Two ways a descent can fail to terminate, in one file.

    A form that invokes itself, and a chain of forms deeper than any reader
    should follow with an image at the bottom of it. Neither is exotic: a
    generator emitting a page as nested groups produces the second routinely,
    and the first is what a damaged file looks like. A descent needs a limit and
    a cycle guard, and both must COUNT when they bite -- a bound that stops
    silently is the same lost placement in a new place.
    """
    pixels = _grey_pixels()
    cycle_obj = 5
    chain_first = 6
    image_obj = chain_first + FORM_CHAIN

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
         f"/Resources << /XObject << /Fc {cycle_obj} 0 R /Fd0 {chain_first} 0 R >> >> "
         f"/Contents 4 0 R >>").encode("ascii"),
        _pdf_stream(b"q /Fc Do Q\nq /Fd0 Do Q\n"),
        _form(b"q /Fc Do Q\n", f"/XObject << /Fc {cycle_obj} 0 R >>"),
    ]
    for index in range(FORM_CHAIN):
        last = index == FORM_CHAIN - 1
        if last:
            objects.append(_form(b"q 10 0 0 10 0 0 cm /Im0 Do Q\n",
                                 f"/XObject << /Im0 {image_obj} 0 R >>"))
        else:
            objects.append(_form(f"q /Fd{index + 1} Do Q\n".encode("ascii"),
                                 f"/XObject << /Fd{index + 1} {chain_first + index + 1} 0 R >>"))
    objects.append(
        b"<< /Type /XObject /Subtype /Image /Width 4 /Height 4 "
        b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length "
        + str(len(pixels)).encode("ascii") + b" >>\nstream\n" + pixels
        + b"\nendstream")
    path.write_bytes(_pdf_assemble(objects))
    return path


def pdf_image_xobject(path: Path) -> Path:
    """One image, drawn once, at a known place and a known size.

    The matrix scales the unit square to 100 by 50 points and puts its corner at
    (72, 700). The image itself is 4 by 4 pixels — deliberately unlike its
    placement, so a reader reporting pixel counts as page geometry is caught.
    """
    return _pdf_with_images(path, [(100, 0, 0, 50, 72, 700)])


def pdf_image_twice(path: Path) -> Path:
    """ONE image object, drawn TWICE, at two positions and two scales.

    Two placements, two nodes, one asset. A reader that emits one node per stored
    object reports this document as holding one picture, which is true of its
    storage and false of its pages.
    """
    return _pdf_with_images(path, [(100, 0, 0, 50, 72, 700),
                                   (60, 0, 0, 30, 300, 400)])


def pdf_inline_image(path: Path) -> Path:
    """An image carried inline in the content stream, beside one held as an object.

    The inline one is a named, counted skip in this phase; the object one must
    still be read. A fixture holding only the inline image could not tell a skip
    from a reader that found nothing.
    """
    return _pdf_with_images(path, [(100, 0, 0, 50, 72, 700)], inline=True)


def pdf_image_unreadable(path: Path) -> Path:
    """Three placements: one readable, one that decodes to nothing, one absent.

    Written from measured behaviour rather than assumed: a stream declaring a
    compression its bytes do not honour does NOT raise — it decodes to zero
    bytes, which a reader will happily store as a picture of no length under a
    perfectly valid hash. A resource pointing at an object that does not exist
    raises instead. Both must be counted, and neither must become an asset.
    """
    pixels = _grey_pixels()
    payload = (b"q 100 0 0 50 72 700 cm /Im0 Do Q\n"
               b"q 100 0 0 50 72 600 cm /Im1 Do Q\n"
               b"q 100 0 0 50 72 500 cm /Im2 Do Q\n")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources "
         b"<< /XObject << /Im0 6 0 R /Im1 7 0 R /Im2 99 0 R >> >> "
         b"/Contents 5 0 R >>"),
        _pdf_stream(payload),
        (b"<< /Type /XObject /Subtype /Image /Width 4 /Height 4 "
         b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length "
         + str(len(pixels)).encode("ascii") + b" >>\nstream\n" + pixels
         + b"\nendstream"),
        (b"<< /Type /XObject /Subtype /Image /Width 4 /Height 4 "
         b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
         b"/Length 16 >>\nstream\n" + b"not compressed!!" + b"\nendstream"),
    ]
    path.write_bytes(_pdf_assemble(objects))
    return path


def _png_bytes(width: int = 4, height: int = 4) -> bytes:
    """A greyscale PNG, assembled here from its chunks.

    Written by hand rather than by an imaging library so that the bytes of every
    fixture picture are produced by this repository's own code and by nothing
    else — the same reason no document is ever committed here.
    """
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes(_grey_pixels(width, 1)) for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def docx_embedded_image(path: Path) -> Path:
    """A word-processing document holding one picture as a part."""
    from io import BytesIO

    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_paragraph("Un paragraphe avant la figure.")
    doc.add_picture(BytesIO(_png_bytes()), width=Pt(72))
    doc.add_paragraph("Un paragraphe apres la figure.")
    doc.save(path)
    return path


def docx_image_unreadable(path: Path) -> Path:
    """A word-processing document holding two pictures the reader cannot take whole.

    One header will not parse; one part is present but empty. Built by writing a
    valid document and then damaging it, because the library refuses to *write* a
    picture it cannot read -- which is precisely why a reader meets these shapes
    only in documents it did not produce.
    """
    import zipfile
    from io import BytesIO

    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_paragraph("Un paragraphe avant les figures.")
    for size in (4, 5):                    # distinct bytes, or the library stores one part
        doc.add_picture(BytesIO(_png_bytes(size, size)), width=Pt(72))
    doc.add_paragraph("Un paragraphe apres les figures.")
    intact = path.with_name(path.stem + "_intact.docx")
    doc.save(intact)

    # A part removed outright is NOT one of the shapes: measured, the library
    # then refuses to open the package at all, which is a document-level failure
    # and not a picture the reader could have counted.
    with zipfile.ZipFile(intact) as src:
        media = sorted(i.filename for i in src.infolist()
                       if i.filename.startswith("word/media/"))
        assert len(media) == 2, media
        unparseable, empty = media
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
            for item in src.infolist():
                payload = src.read(item.filename)
                if item.filename == unparseable:
                    payload = b"NOT-AN-IMAGE-BUT-CERTAINLY-BYTES"
                elif item.filename == empty:
                    payload = b""
                out.writestr(item, payload)
    intact.unlink()
    return path


def pptx_image_unreadable(path: Path) -> Path:
    """A presentation holding two pictures the reader cannot take whole.

    One picture's header will not parse -- its bytes are still reachable, so it
    survives with less known about it. The other's part is gone from the package
    entirely, and unlike a word-processing document, a presentation still opens:
    the bytes are simply unreachable, which is the one shape that is a counted
    skip rather than a picture with unknowns.
    """
    import zipfile
    from io import BytesIO

    from pptx import Presentation
    from pptx.util import Pt

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    for offset, size in ((0, 4), (200, 5)):   # distinct bytes, or one part is stored
        slide.shapes.add_picture(BytesIO(_png_bytes(size, size)),
                                 Pt(72), Pt(144 + offset), width=Pt(96), height=Pt(48))
    intact = path.with_name(path.stem + "_intact.pptx")
    deck.save(intact)

    with zipfile.ZipFile(intact) as src:
        media = sorted(i.filename for i in src.infolist()
                       if i.filename.startswith("ppt/media/"))
        assert len(media) == 2, media
        unparseable, absent = media
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
            for item in src.infolist():
                if item.filename == absent:
                    continue              # the shape survives its bytes
                payload = src.read(item.filename)
                if item.filename == unparseable:
                    payload = b"NOT-AN-IMAGE-BUT-CERTAINLY-BYTES"
                out.writestr(item, payload)
    intact.unlink()
    return path


def pptx_embedded_image(path: Path) -> Path:
    """A presentation holding one picture, at a position the format states."""
    from io import BytesIO

    from pptx import Presentation
    from pptx.util import Emu, Pt

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Une diapositive avec une figure"
    slide.shapes.add_picture(BytesIO(_png_bytes()), Pt(72), Pt(144),
                             width=Pt(96), height=Pt(48))
    deck.save(path)
    return path


def xlsx_embedded_image(path: Path) -> Path:
    """A workbook holding one picture, anchored to a cell."""
    from io import BytesIO

    from openpyxl import Workbook
    from openpyxl.drawing.image import Image

    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Reference"
    ws["A2"] = "R-1"
    image = Image(BytesIO(_png_bytes()))
    ws.add_image(image, "C3")
    wb.save(path)
    return path


def no_format_contrast(path: Path) -> Path:
    """Every line at one size: the negative. Nothing here is larger than anything.

    A method that elects a heading tier regardless will elect one here, so this
    fixture is what separates a rule that reads contrast from a rule that reads
    its own defaults.
    """
    return _pdf_typeset(path, [[
        [("Introduction", 12.0)],
        [(_BODY, 12.0)],
        [("Perimetre", 12.0)],
        [(_BODY, 12.0)],
        [("Gouvernance", 12.0)],
        [(_BODY, 12.0)],
    ]])


# ------------------------------------------------------------------ refusals

def unsupported_format(path: Path) -> Path:
    """A file no adapter claims. Being unreadable is its whole purpose."""
    target = path.with_suffix(".unknown")
    target.write_bytes(b"\x89SAQ\x00 not a document\n")
    return target


def duplicate_pair(path: Path) -> Path:
    """Two byte-identical copies of one document, to be counted once.

    Writes the original at `path` and its copy beside it, suffixed `_copy`.
    """
    original = markdown_document(path)
    copy = path.with_name(f"{path.stem}_copy{path.suffix}")
    copy.write_bytes(original.read_bytes())
    return original


#: Fixture name -> generator. Checked against SPEC.md by the K0 gate.
FIXTURES: Dict[str, Callable[[Path], Path]] = {
    "mixed_workbook": mixed_workbook,
    "two_tier_header": two_tier_header,
    "label_tiling": label_tiling,
    "full_width_title": full_width_title,
    "section_row": section_row,
    "merged_answer_area": merged_answer_area,
    "numeric_bold_header": numeric_bold_header,
    "headerless_list": headerless_list,
    "undecidable_block": undecidable_block,
    "totals_row_and_column": totals_row_and_column,
    "two_islands": two_islands,
    "overlapping_merges": overlapping_merges,
    "ambiguous_layout": ambiguous_layout,
    "docx_two_tier": docx_two_tier,
    "docx_label_tiling": docx_label_tiling,
    "docx_layout_prose": docx_layout_prose,
    "docx_markers": docx_markers,
    "docx_header_stream": docx_header_stream,
    "docx_nested": docx_nested,
    "docx_twin_pair": docx_twin_pair,
    "pdf_outline": pdf_outline,
    "pdf_no_text_layer": pdf_no_text_layer,
    "trees_per_format": trees_per_format,
    "sections_multi_channel": sections_multi_channel,
    "running_headers": running_headers,
    "slide_deck": slide_deck,
    "markdown_document": markdown_document,
    "numbered_outline_trees": numbered_outline_trees,
    "twin_grid_xlsx": twin_grid_xlsx,
    "twin_grid_docx": twin_grid_docx,
    "twin_grid_pptx": twin_grid_pptx,
    "empty_string_cells": empty_string_cells,
    "pdf_type_scales": pdf_type_scales,
    "format_headings_pdf": format_headings_pdf,
    "pdf_image_xobject": pdf_image_xobject,
    "pdf_image_in_form": pdf_image_in_form,
    "pdf_form_pathologies": pdf_form_pathologies,
    "pdf_image_twice": pdf_image_twice,
    "pdf_image_unreadable": pdf_image_unreadable,
    "docx_embedded_image": docx_embedded_image,
    "docx_image_unreadable": docx_image_unreadable,
    "pptx_image_unreadable": pptx_image_unreadable,
    "pptx_embedded_image": pptx_embedded_image,
    "xlsx_embedded_image": xlsx_embedded_image,
    "pdf_inline_image": pdf_inline_image,
    "pdf_declared_outline": pdf_declared_outline,
    "format_headings_docx": format_headings_docx,
    "no_weight_contrast": no_weight_contrast,
    "no_format_contrast": no_format_contrast,
    "unsupported_format": unsupported_format,
    "duplicate_pair": duplicate_pair,
}


#: The suffix each fixture's bytes require, declared beside the fixture itself.
#:
#: A fixture writes one format. Which one is a property of the fixture, and it used
#: to be recorded in a table inside a test module — so a new fixture was handed to
#: the word-processing reader until somebody remembered to edit a file it has
#: nothing to do with. That failed closed and loudly, four times, with a message
#: about a zip archive that said nothing about the real mistake. The declaration
#: belongs here, and `test_k0_1_every_fixture_declares_its_suffix` refuses a
#: registry where the two halves disagree.
FIXTURE_SUFFIX: Dict[str, str] = {
    "ambiguous_layout": ".xlsx",
    "docx_header_stream": ".docx",
    "docx_label_tiling": ".docx",
    "docx_layout_prose": ".docx",
    "docx_markers": ".docx",
    "docx_nested": ".docx",
    "docx_twin_pair": ".docx",
    "docx_two_tier": ".docx",
    "duplicate_pair": ".md",
    "empty_string_cells": ".xlsx",
    "format_headings_docx": ".docx",
    "format_headings_pdf": ".pdf",
    "full_width_title": ".xlsx",
    "headerless_list": ".xlsx",
    "label_tiling": ".xlsx",
    "markdown_document": ".md",
    "merged_answer_area": ".xlsx",
    "mixed_workbook": ".xlsx",
    "no_format_contrast": ".pdf",
    "no_weight_contrast": ".docx",
    "numbered_outline_trees": ".json",
    "numeric_bold_header": ".xlsx",
    "overlapping_merges": ".xlsx",
    "pdf_declared_outline": ".pdf",
    "pdf_image_twice": ".pdf",
    "pdf_image_unreadable": ".pdf",
    "docx_embedded_image": ".docx",
    "docx_image_unreadable": ".docx",
    "pptx_image_unreadable": ".pptx",
    "pptx_embedded_image": ".pptx",
    "xlsx_embedded_image": ".xlsx",
    "pdf_image_xobject": ".pdf",
    "pdf_image_in_form": ".pdf",
    "pdf_form_pathologies": ".pdf",
    "pdf_inline_image": ".pdf",
    "pdf_no_text_layer": ".pdf",
    "pdf_outline": ".pdf",
    "pdf_type_scales": ".pdf",
    "running_headers": ".pdf",
    "section_row": ".xlsx",
    "sections_multi_channel": ".json",
    "slide_deck": ".pptx",
    "totals_row_and_column": ".xlsx",
    "trees_per_format": ".json",
    "twin_grid_docx": ".docx",
    "twin_grid_pptx": ".pptx",
    "twin_grid_xlsx": ".xlsx",
    "two_islands": ".xlsx",
    "two_tier_header": ".xlsx",
    "undecidable_block": ".xlsx",
    "unsupported_format": ".tmp",
}


def fixture_path(name: str, root: Path) -> Path:
    """Where a fixture should be written: its own name, its own declared suffix."""
    return root / f"{name}{FIXTURE_SUFFIX[name]}"
