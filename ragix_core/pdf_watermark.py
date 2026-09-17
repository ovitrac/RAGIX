"""Conservative, token-preserving watermark removal for analysis derivatives.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unicodedata

VERSION = "pdf-watermark/1"
TEXT_SHOW = {"Tj", "TJ", "'", '"'}
PATH_PAINT = {"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"}
TEXT_STATE = {
    "q",
    "Q",
    "cm",
    "gs",
    "BT",
    "ET",
    "Tf",
    "Tr",
    "Ts",
    "Tz",
    "Tc",
    "Tw",
    "TL",
    "Tm",
    "Td",
    "TD",
    "T*",
    "g",
    "G",
    "rg",
    "RG",
    "k",
    "K",
    "cs",
    "CS",
    "sc",
    "SC",
    "scn",
    "SCN",
    "w",
    "J",
    "j",
    "M",
    "d",
    "ri",
    "i",
}


class Refused(ValueError):
    """No derivative may be published when a safety invariant is unproved."""


def _libraries():
    try:
        import pikepdf
        import pymupdf
        import numpy
    except ImportError as error:
        raise Refused("MISSING_DEPENDENCY: install ragix[pdf-clean]") from error
    return pikepdf, pymupdf, numpy


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value):
    return _hash(_json(value).encode())


def _versions():
    pikepdf, mupdf, numpy = _libraries()
    return {
        "pikepdf": pikepdf.__version__,
        "pymupdf": mupdf.__version__,
        "numpy": numpy.__version__,
    }


def _normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 128 * 1024 * 1024
    max_pages: int = 2000
    max_tokens_per_page: int = 250000
    max_candidates_per_page: int = 500
    max_pixels_per_page: int = 16000000
    dpi: int = 72

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise Refused("INVALID_LIMIT")


@dataclass(frozen=True)
class Token:
    kind: str
    raw: bytes


@dataclass(frozen=True)
class Operation:
    start: int
    end: int
    name: str


@dataclass(frozen=True)
class Region:
    start: int
    end: int
    kind: str


def _tokens(page, limits):
    pikepdf, _, _ = _libraries()

    class Collector(pikepdf.TokenFilter):
        def __init__(self):
            super().__init__()
            self.tokens = []
            self.problem = None

        def handle_token(self, token):
            kind = token.type_.name
            if kind == "bad":
                self.problem = "INVALID_PDF_TOKEN"
            if len(self.tokens) >= limits.max_tokens_per_page:
                self.problem = "TOKEN_LIMIT"
                raise ValueError(self.problem)
            if kind != "eof":
                self.tokens.append(Token(kind, token.raw_value))
            return token

    collector = Collector()
    try:
        raw = page.get_filtered_contents(collector)
    except Exception as error:
        raise Refused(collector.problem or "TOKEN_PARSE_FAILED") from error
    if collector.problem:
        raise Refused(collector.problem)
    tokens = tuple(collector.tokens)
    if b"".join(t.raw for t in tokens) != raw:
        raise Refused("TOKEN_ROUNDTRIP_MISMATCH")
    if any(t.kind == "inline_image" for t in tokens):
        raise Refused("INLINE_IMAGE_UNSUPPORTED")
    return tokens


def _operations(tokens):
    nesting = []
    start = None
    output = []
    pairs = {"array_close": "array_open", "dict_close": "dict_open"}
    for i, token in enumerate(tokens):
        if token.kind in {"space", "comment"}:
            continue
        if start is None:
            start = i
        if token.kind in {"array_open", "dict_open"}:
            nesting.append(token.kind)
        elif token.kind in pairs:
            if not nesting or nesting.pop() != pairs[token.kind]:
                raise Refused("UNBALANCED_OPERAND")
        elif token.kind == "word" and not nesting:
            try:
                name = token.raw.decode("ascii")
            except UnicodeError as error:
                raise Refused("INVALID_OPERATOR") from error
            output.append(Operation(start, i + 1, name))
            start = None
    if nesting or start is not None:
        raise Refused("INCOMPLETE_CONTENT_STREAM")
    return tuple(output)


def _name_operand(tokens, op):
    pikepdf, _, _ = _libraries()
    operands = [t for t in tokens[op.start : op.end - 1] if t.kind not in {"space", "comment"}]
    if len(operands) != 1 or operands[0].kind != "name_":
        raise Refused("INVALID_RESOURCE_OPERAND")
    return str(pikepdf.Object.parse(operands[0].raw))


def _text_form(form, resources, seen=()):
    """Only text/state Form graphs qualify; caller graphics state is implicit-local."""
    pikepdf, _, _ = _libraries()
    if str(form.get("/Subtype", "")) != "/Form" or form.objgen in seen or len(seen) >= 32:
        return False
    if form.get("/Group") is not None:
        return False
    resources = form.get("/Resources", resources)
    if _type3(resources):
        return False
    depth = 0
    text = False
    shows = False
    try:
        for instruction in pikepdf.parse_content_stream(form):
            name = str(instruction.operator)
            if name == "q":
                depth += 1
            elif name == "Q":
                depth -= 1
                if depth < 0:
                    return False
            elif name == "BT":
                if text:
                    return False
                text = True
            elif name == "ET":
                if not text:
                    return False
                text = False
            if name == "Tr" and (
                len(instruction.operands) != 1 or instruction.operands[0] not in (0, 1, 2, 3)
            ):
                return False
            if name in TEXT_SHOW:
                if not text:
                    return False
                shows = True
            elif name == "Do":
                child = resources.get("/XObject", {}).get(str(instruction.operands[0]))
                if child is None or not _text_form(child, resources, (*seen, form.objgen)):
                    return False
                shows = True
            elif name not in TEXT_STATE:
                return False
    except (ValueError, TypeError, IndexError, AttributeError, pikepdf.PdfError):
        return False
    return depth == 0 and not text and shows


def _type3(resources):
    return any(
        str(font.get("/Subtype", "")) == "/Type3" for font in resources.get("/Font", {}).values()
    )


def _regions(page, tokens, operations, limits):
    stack = []
    text = False
    marked = 0
    regions = []
    resources = page.Resources
    if _type3(resources):
        raise Refused("TYPE3_FONT_UNSUPPORTED")
    form_ok = {}
    for index, op in enumerate(operations):
        if op.name == "Tr":
            values = [
                t.raw for t in tokens[op.start : op.end - 1] if t.kind not in {"space", "comment"}
            ]
            if values not in ([b"0"], [b"1"], [b"2"], [b"3"]):
                raise Refused("TEXT_CLIPPING_UNSUPPORTED")
        if op.name == "BT":
            if text:
                raise Refused("NESTED_TEXT_OBJECT")
            text = True
        elif op.name == "ET":
            if not text:
                raise Refused("UNBALANCED_TEXT_OBJECT")
            text = False
        elif op.name in {"BMC", "BDC"}:
            marked += 1
        elif op.name == "EMC":
            marked -= 1
            if marked < 0:
                raise Refused("UNBALANCED_MARKED_CONTENT")
        elif op.name == "q":
            stack.append((index, text, marked))
        elif op.name == "Q":
            if not stack:
                raise Refused("UNBALANCED_GRAPHICS_STATE")
            beginning, in_text, mark_start = stack.pop()
            block = operations[beginning : index + 1]
            if (
                not in_text
                and not text
                and mark_start == marked
                and any(i.name in TEXT_SHOW for i in block)
                and all(i.name in TEXT_STATE | TEXT_SHOW for i in block)
            ):
                regions.append(Region(block[0].start, op.end, "graphics_block"))
        elif op.name == "Do" and not text:
            key = _name_operand(tokens, op)
            if key not in form_ok:
                form = resources.get("/XObject", {}).get(key)
                form_ok[key] = form is not None and _text_form(form, resources)
            if form_ok[key]:
                regions.append(Region(op.start, op.end, "form_invocation"))
        if op.name in TEXT_SHOW and not text:
            raise Refused("TEXT_OUTSIDE_TEXT_OBJECT")
    if stack or text or marked:
        raise Refused("UNBALANCED_CONTENT_STATE")
    if len(regions) > limits.max_candidates_per_page:
        raise Refused("CANDIDATE_LIMIT")
    return tuple(regions)


def _filtered(tokens, operations, regions, *, isolate=False):
    chosen = set(i for region in regions for i in range(region.start, region.end))
    if not isolate:
        return b"".join(t.raw for i, t in enumerate(tokens) if i not in chosen)
    replace = {}
    for op in operations:
        if any(i in chosen for i in range(op.start, op.end)):
            continue
        if op.name in TEXT_SHOW:
            # Preserve quote spacing and TJ displacements; suppress only strings.
            for i in range(op.start, op.end):
                if tokens[i].kind == "string":
                    replace[i] = b"()"
        elif op.name in PATH_PAINT:
            for i in range(op.start, op.end):
                replace[i] = b""
            replace[op.end - 1] = b"n"
        elif op.name in {"Do", "sh"}:
            for i in range(op.start, op.end):
                replace[i] = b""
    return b"".join(replace.get(i, t.raw) for i, t in enumerate(tokens))


def _opaque_resources(resources, seen=None, depth=0):
    """Rendering-only copy: measure full glyph support before alpha quantization."""
    seen = set() if seen is None else seen
    if depth > 32:
        raise Refused("RESOURCE_DEPTH_LIMIT")
    for state in resources.get("/ExtGState", {}).values():
        state["/ca"] = 1
        state["/CA"] = 1
    for form in resources.get("/XObject", {}).values():
        if str(form.get("/Subtype", "")) != "/Form" or form.objgen in seen:
            continue
        seen.add(form.objgen)
        _opaque_resources(form.get("/Resources", {}), seen, depth + 1)


def _single_page(pdf, number, content, *, ink_support=False):
    pikepdf, _, _ = _libraries()
    with pikepdf.Pdf.new() as copy:
        copy.pages.append(pdf.pages[number])
        if "/OCProperties" in pdf.Root:
            source = pdf.Root.OCProperties
            if not source.is_indirect:
                source = pdf.make_indirect(source)
            copy.Root.OCProperties = copy.copy_foreign(source)
        page = copy.pages[0]
        if "/Annots" in page:
            del page["/Annots"]
        page.Contents = copy.make_stream(content)
        if ink_support:
            _opaque_resources(page.Resources)
        target = io.BytesIO()
        copy.save(target, deterministic_id=True)
        return target.getvalue()


def _glyphs(page):
    records = []
    text = []
    appearances = []
    for span in page.get_texttrace():
        appearances.append((tuple(span["dir"]), span["opacity"]))
        for code, gid, origin, box in span["chars"]:
            text.append(chr(code) if 0 <= code <= 0x10FFFF else "\ufffd")
            records.append(
                (
                    code,
                    gid,
                    span["font"],
                    span["size"],
                    tuple(span["dir"]),
                    span["type"],
                    span["opacity"],
                    tuple(origin),
                    tuple(box),
                )
            )
    return Counter(records), "".join(text), appearances


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if k not in {"seqno", "number", "xref"}}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "__iter__") and not isinstance(value, str):
        return list(value)
    return value


def _nontext(page):
    return {
        "drawings": _plain(page.get_drawings()),
        "images": _plain(page.get_image_info(hashes=True)),
    }


def _pixels(page, limits, alpha=False):
    _, _, np = _libraries()
    scale = limits.dpi / 72
    if (
        math.ceil(page.rect.width * scale) * math.ceil(page.rect.height * scale)
        > limits.max_pixels_per_page
    ):
        raise Refused("PIXEL_LIMIT")
    pix = page.get_pixmap(dpi=limits.dpi, alpha=alpha)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def _security(pdf):
    result = {"encrypted": pdf.is_encrypted, "permissions": dict(pdf.allow._asdict())}
    if pdf.is_encrypted:
        info = pdf.encryption
        result.update(
            revision=info.R,
            version=info.V,
            bits=info.bits,
            stream_method=str(info.stream_method),
            string_method=str(info.string_method),
        )
    return result


def _open(data):
    pikepdf, _, _ = _libraries()
    try:
        return pikepdf.open(io.BytesIO(data), attempt_recovery=False)
    except pikepdf.PasswordError as error:
        raise Refused("OPEN_PASSWORD_REQUIRED") from error
    except pikepdf.PdfError as error:
        raise Refused("INVALID_PDF") from error


def _check_document(pdf, limits):
    if not 0 < len(pdf.pages) <= limits.max_pages:
        raise Refused("PAGE_LIMIT")
    if pdf.Root.get("/Perms") is not None:
        raise Refused("CERTIFIED_PDF_UNSUPPORTED")
    fields = list(pdf.Root.get("/AcroForm", {}).get("/Fields", []))
    seen = set()
    while fields:
        field = fields.pop()
        if field.is_indirect:
            if field.objgen in seen:
                continue
            seen.add(field.objgen)
        if str(field.get("/FT", "")) == "/Sig":
            raise Refused("SIGNATURE_FIELD_UNSUPPORTED")
        fields.extend(field.get("/Kids", []))
    if pdf.get_warnings():
        raise Refused("PDF_PARSER_WARNINGS")


def _source(path, limits):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > limits.max_file_bytes:
        raise Refused("INPUT_FILE_LIMIT")
    return path.read_bytes()


def _inventory(data, marker, limits):
    pikepdf, mupdf, _ = _libraries()
    if not _normalized(marker):
        raise Refused("EMPTY_MARKER")
    source = _hash(data)
    candidates = []
    refusals = []
    with _open(data) as pdf:
        _check_document(pdf, limits)
        for number, page in enumerate(pdf.pages):
            try:
                tokens = _tokens(page, limits)
                ops = _operations(tokens)
                regions = _regions(page, tokens, ops, limits)
            except Refused as error:
                refusals.append({"page": number + 1, "reason": str(error)})
                continue
            matches = []
            for region in regions:
                isolated = _single_page(
                    pdf, number, _filtered(tokens, ops, (region,), isolate=True)
                )
                with mupdf.open(stream=isolated, filetype="pdf") as view:
                    glyphs, text, appearance = _glyphs(view[0])
                    if _normalized(text) != _normalized(marker):
                        continue
                    if not glyphs or "\ufffd" in text:
                        continue
                    if not any(
                        direction != (1.0, 0.0) or opacity < 1 for direction, opacity in appearance
                    ):
                        continue
                    boxes = [record[-1] for record in glyphs]
                    box = [
                        min(b[0] for b in boxes),
                        min(b[1] for b in boxes),
                        max(b[2] for b in boxes),
                        max(b[3] for b in boxes),
                    ]
                raw = b"".join(t.raw for t in tokens[region.start : region.end])
                matches.append(
                    {
                        "candidate_id": _digest(
                            [source, number, region.start, region.end, _hash(raw)]
                        ),
                        "page": number + 1,
                        "kind": region.kind,
                        "start_token": region.start,
                        "end_token": region.end,
                        "stream_sha256": _hash(b"".join(t.raw for t in tokens)),
                        "region_sha256": _hash(raw),
                        "text": text,
                        "bbox": box,
                        "glyph_count": sum(glyphs.values()),
                    }
                )
            # Prefer a containing safe graphics block; never expose overlapping removals.
            for match in sorted(matches, key=lambda r: (r["start_token"], -r["end_token"])):
                if any(
                    old["page"] == match["page"]
                    and old["start_token"] <= match["start_token"]
                    and old["end_token"] >= match["end_token"]
                    for old in candidates
                ):
                    continue
                candidates.append(match)
        if pdf.get_warnings():
            raise Refused("PDF_PARSER_WARNINGS")
        pages = len(pdf.pages)
        security = _security(pdf)
    return {
        "schema": VERSION,
        "input_sha256": source,
        "page_count": pages,
        "source_security": security,
        "marker": marker,
        "limits": vars(limits),
        "dependencies": _versions(),
        "candidates": candidates,
        "refusals": refusals,
        "status": "CANDIDATES" if candidates else "NO_SUPPORTED_CANDIDATES",
    }


def inspect_pdf(path, marker, limits=Limits()):
    return _inventory(_source(path, limits), marker, limits)


def _validate(original, edited, source_pdf, selected, limits):
    pikepdf, mupdf, np = _libraries()
    # Save/reload must preserve every unselected content token byte, on all pages.
    with _open(edited) as written:
        _check_document(written, limits)
        if _security(written) != _security(source_pdf):
            raise Refused("ENCRYPTION_OR_PERMISSIONS_CHANGED")
        if len(written.pages) != len(source_pdf.pages):
            raise Refused("PAGE_COUNT_CHANGED")
        for number, page in enumerate(source_pdf.pages):
            tokens = _tokens(page, limits)
            expected = _filtered(tokens, (), selected.get(number, ()))
            actual = b"".join(t.raw for t in _tokens(written.pages[number], limits))
            if actual != expected:
                raise Refused("RETAINED_TOKENS_CHANGED")
    checks = []
    with (
        mupdf.open(stream=original, filetype="pdf") as before,
        mupdf.open(stream=edited, filetype="pdf") as after,
    ):
        if len(before) != len(after):
            raise Refused("PAGE_COUNT_CHANGED")
        for number, (a, b) in enumerate(zip(before, after)):
            if (a.rect, a.mediabox, a.cropbox, a.rotation) != (
                b.rect,
                b.mediabox,
                b.cropbox,
                b.rotation,
            ):
                raise Refused("PAGE_GEOMETRY_CHANGED")
            removed = Counter()
            selected_page = selected.get(number, ())
            allowed = None
            if selected_page:
                page = source_pdf.pages[number]
                tokens = _tokens(page, limits)
                ops = _operations(tokens)
                isolated = _single_page(
                    source_pdf, number, _filtered(tokens, ops, selected_page, isolate=True)
                )
                with mupdf.open(stream=isolated, filetype="pdf") as subset:
                    removed, _, _ = _glyphs(subset[0])
                # Measure glyph support at full opacity in a disposable render copy.
                # Low alpha can round antialiased edge coverage to zero on isolation
                # while changing a source background pixel. Geometry is unchanged;
                # no bounding-box allowance or mask dilation is used.
                ink = _single_page(
                    source_pdf,
                    number,
                    _filtered(tokens, ops, selected_page, isolate=True),
                    ink_support=True,
                )
                with mupdf.open(stream=ink, filetype="pdf") as subset:
                    allowed = (_pixels(subset[0], limits, alpha=True)[:, :, -1] > 0) | np.any(
                        _pixels(subset[0], limits) != 255, axis=2
                    )
            original_glyphs, _, _ = _glyphs(a)
            retained, _, _ = _glyphs(b)
            if any(n > original_glyphs[k] for k, n in removed.items()):
                raise Refused("REMOVAL_NOT_IN_SOURCE")
            expected = original_glyphs - removed
            if expected != retained:
                raise Refused("RETAINED_GLYPHS_CHANGED")
            if _nontext(a) != _nontext(b):
                raise Refused("NON_TEXT_CONTENT_CHANGED")
            pa, pb = _pixels(a, limits), _pixels(b, limits)
            if pa.shape != pb.shape:
                raise Refused("RENDER_GEOMETRY_CHANGED")
            delta = np.any(pa != pb, axis=2)
            if allowed is None:
                allowed = np.zeros(delta.shape, dtype=bool)
            if allowed.shape != delta.shape or np.any(delta & ~allowed):
                raise Refused("PIXELS_CHANGED_OUTSIDE_MARKER")
            checks.append(
                {
                    "page": number + 1,
                    "removed_glyphs": sum(removed.values()),
                    "retained_glyphs": sum(retained.values()),
                    "changed_pixels": int(delta.sum()),
                    "outside_marker_changed_pixels": 0,
                    "retained_glyph_digest": _digest(sorted((k, n) for k, n in retained.items())),
                }
            )
    return checks


def apply_plan(source, plan, candidate_ids, output, report):
    pikepdf, _, _ = _libraries()
    if plan.get("schema") != VERSION or plan.get("dependencies") != _versions():
        raise Refused("PLAN_VERSION_MISMATCH")
    limits = Limits(**plan["limits"])
    data = _source(source, limits)
    if _hash(data) != plan.get("input_sha256"):
        raise Refused("STALE_INPUT")
    current = _inventory(data, plan["marker"], limits)
    ids = tuple(candidate_ids)
    if not ids or len(set(ids)) != len(ids):
        raise Refused("EMPTY_OR_DUPLICATE_SELECTION")
    lookup = {c["candidate_id"]: c for c in current["candidates"]}
    if any(ident not in lookup for ident in ids):
        raise Refused("UNKNOWN_CANDIDATE")
    output, report = Path(output), Path(report)
    if (
        output.resolve() == Path(source).resolve()
        or report.resolve() == Path(source).resolve()
        or output.resolve() == report.resolve()
    ):
        raise Refused("OUTPUT_ALIASES_INPUT")
    if output.exists() or report.exists():
        raise Refused("OUTPUT_EXISTS")
    if not output.parent.is_dir() or not report.parent.is_dir():
        raise Refused("OUTPUT_PARENT_MISSING")
    selected = {}
    for ident in ids:
        c = lookup[ident]
        selected.setdefault(c["page"] - 1, []).append(
            Region(c["start_token"], c["end_token"], c["kind"])
        )
    with _open(data) as pdf:
        _check_document(pdf, limits)
        for number, regions in selected.items():
            page = pdf.pages[number]
            tokens = _tokens(page, limits)
            ops = _operations(tokens)
            page.Contents = pdf.make_stream(_filtered(tokens, ops, regions))
        buffer = io.BytesIO()
        pdf.save(
            buffer,
            encryption=True if pdf.is_encrypted else None,
            deterministic_id=not pdf.is_encrypted,
        )
        if pdf.get_warnings():
            raise Refused("PDF_PARSER_WARNINGS")
        edited = buffer.getvalue()
    with _open(data) as original:
        checks = _validate(data, edited, original, selected, limits)
    audit = {
        "schema": VERSION,
        "status": "VALIDATED_DERIVATIVE",
        "authoritative_source": "original",
        "source_security": current["source_security"],
        "encryption_and_permissions": "preserved",
        "input_sha256": _hash(data),
        "output_sha256": _hash(edited),
        "marker": plan["marker"],
        "selected": [lookup[ident] for ident in ids],
        "dependencies": _versions(),
        "limits": vars(limits),
        "checks": checks,
        "warning": "Analysis derivative; removed lifecycle text remains evidence in this audit.",
    }
    if _source(source, limits) != data:
        raise Refused("SOURCE_CHANGED_DURING_VALIDATION")
    # Publish complete files without overwriting racing writers. The PDF appears
    # only after its audit is durable. Temporary files live on each target's FS.
    created, temporary = [], []
    try:
        for target, payload in (
            (report, (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode()),
            (output, edited),
        ):
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".pdf-clean-", delete=False
            ) as stream:
                temporary.append(Path(stream.name))
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(stream.name, target)
            created.append(target)
    except Exception:
        for path in created:
            path.unlink()
        raise
    finally:
        for path in temporary:
            path.unlink()
    return audit
