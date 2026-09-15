"""Optional-reader integration using newly generated, non-documentary fixtures.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-09-15
"""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest

from ragix_kernels.saqqara.adapters.pdf_mupdf import MuPdfTextReader
from ragix_kernels.saqqara.field_views import classify, line_views


@pytest.mark.parametrize("ruled", [False, True])
@pytest.mark.parametrize("split_fonts", [False, True])
@pytest.mark.parametrize("rotation", [0, 90])
def test_native_coalescence_and_join_boundaries(tmp_path, ruled, split_fonts, rotation):
    if importlib.util.find_spec("pymupdf") is None:
        pytest.skip("optional reader not installed")
    # Optional imports stay in a subprocess; default-route import guards remain meaningful.
    subprocess.run([sys.executable, "-c",
                    "import runpy,sys; from pathlib import Path; "
                    "runpy.run_path(sys.argv[1])['_exercise'](Path(sys.argv[2]), "
                    "bool(int(sys.argv[3])), bool(int(sys.argv[4])), int(sys.argv[5]))",
                    str(Path(__file__).resolve()), str(tmp_path), str(int(ruled)),
                    str(int(split_fonts)), str(rotation)], check=True, capture_output=True, text=True)


def _exercise(tmp_path, ruled, split_fonts, rotation):
    import pymupdf as mupdf
    path = tmp_path / "synthetic.pdf"
    with mupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        left = "R-17"
        end = 40 + mupdf.get_text_length(left, fontsize=10)
        page.insert_text((40, 100), left, fontsize=10)
        page.insert_text((end + 3, 100), "Descriptor", fontsize=10, fontname="hebo" if split_fonts else "helv")
        if ruled:
            page.draw_line((end + 1.5, 80), (end + 1.5, 110), width=0.5)
        page.set_rotation(rotation)
        doc.save(path)
    source = hashlib.sha256(path.read_bytes()).hexdigest()
    port = MuPdfTextReader()
    port.open(path)
    try:
        geometry = port.page_geometry(1, source_id=source)
        assert (geometry["width"], geometry["height"]) == (300, 200)
        raw = geometry["spans"]
        views = line_views(classify(raw, lambda s: "CONTENT"), geometry["vertical_rules"])
        both = [v for v in views if "R-17" in v.text and "Descriptor" in v.text]
        assert bool(both) is not ruled
        observed = {s.span_id: s for s in raw}
        for view in views:
            assert len(view.text) == len(view.mapping)
            for char, ref in zip(view.text, view.mapping):
                if ref is not None:
                    assert observed[ref.span_id].text[ref.offset] == char
        with pytest.raises(ValueError):
            port.page_geometry(0, source_id=source)
    finally:
        port.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source
