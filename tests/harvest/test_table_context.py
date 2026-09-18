"""Independent structural-context falsifiers; no consumer text or semantics.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from dataclasses import replace
import pytest
from ragix_kernels.harvest.table_context import Cell, CellContext, associate


def cell(ident, row, column, text, **kwargs):
    return Cell(
        "synthetic",
        "table-a",
        ident,
        1,
        row,
        column,
        text,
        (column * 60, row * 20, column * 60 + 55, row * 20 + 15),
        (ident + "-text",),
        **kwargs,
    )


def test_two_header_lines_and_row_label_remain_separate():
    h1 = cell("h1", 0, 1, "Displacement")
    h2 = cell("h2", 1, 1, "[mm]")
    label = cell("label", 2, 0, "Axis A")
    value = cell("value", 2, 1, "23")
    c = associate(value, (h1, h2, label, value), header_rows=(0, 1), row_label_columns=(0,))
    assert c.column_headers == (h1, h2) and c.row_labels == (label,)
    assert h2.span(1, 3).raw == "mm" and h2.span(1, 3).cell_id == "h2"


def test_no_header_never_borrows_first_row():
    a = cell("a", 0, 1, "[V]")
    b = cell("b", 1, 1, "17")
    c = associate(b, (a, b))
    assert not c.column_headers and not c.row_labels and c.flags == ("NO_COLUMN_HEADER",)


def test_column_and_row_isolation():
    a = cell("a", 0, 1, "[mm]")
    b = cell("b", 0, 2, "[V]")
    x = cell("x", 1, 1, "23")
    y = cell("y", 2, 0, "Channel")
    c = associate(x, (a, b, x, y), header_rows=(0,), row_label_columns=(0,))
    assert c.column_headers == (a,) and not c.row_labels


def test_spanning_headers_must_cover_whole_value():
    h = cell("h", 0, 1, "[mm]", column_span=2)
    v = cell("v", 1, 1, "23", column_span=2)
    assert associate(v, (h, v), header_rows=(0,)).column_headers == (h,)
    assert not associate(v, (replace(h, column_span=1), v), header_rows=(0,)).column_headers


def test_cross_copy_and_wrong_column_evidence_is_rejected():
    v = cell("v", 1, 1, "23")
    h = cell("h", 0, 1, "[mm]")
    with pytest.raises(ValueError):
        CellContext(v, (replace(h, source_id="other"),))
    with pytest.raises(ValueError):
        CellContext(v, (replace(h, column=2),))
    with pytest.raises(ValueError):
        v.span(0, 5)


def test_context_json_roundtrip_revalidates_source_and_preserves_spans():
    from dataclasses import asdict
    import json
    from ragix_kernels.harvest.table_context import context_from_dict

    h = cell("h", 0, 1, "[mm]")
    v = cell("v", 1, 1, "23")
    original = associate(v, (h, v), header_rows=(0,))
    data = json.loads(json.dumps(asdict(original)))
    assert context_from_dict({"record_id": "external-record", **data}) == original
    data["column_headers"][0]["source_id"] = "other"
    with pytest.raises(ValueError):
        context_from_dict(data)
