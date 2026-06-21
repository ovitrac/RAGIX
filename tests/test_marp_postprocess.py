"""
RAGIX Presenter — marp_postprocess v2.1.1 transform tests

Covers the three layout-tuning transforms that were missing dedicated tests:

- ``expand_accent_directives``  (v2.1)   — ``<!-- accent: COLOR -->`` → scoped CSS
- ``center_standalone_images``  (v2.1.1) — flex-wrap standalone ``<img>`` (HTML + PDF)
- ``inject_lightbox_in_html``   (v2.1.1) — click-to-zoom overlay on the final HTML

These transforms are sensitive to small layout details (block detection, indent
preservation, idempotency), so each behaviour is asserted explicitly.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-18
"""

import pytest

from ragix_kernels.shared.marp_postprocess import (
    expand_accent_directives,
    center_standalone_images,
    inject_lightbox_in_html,
    _ACCENT_PALETTE,
)


# ---------------------------------------------------------------------------
# expand_accent_directives
# ---------------------------------------------------------------------------

class TestExpandAccentDirectives:
    def test_table_accent_emits_scoped_style_and_consumes_comment(self):
        content = "<!-- accent: coral -->\n| A | B |\n| - | - |\n| 1 | 2 |\n"
        out = expand_accent_directives(content)
        assert "<style scoped>" in out
        # coral border colour drives the table border-left + th background
        assert _ACCENT_PALETTE["coral"]["border"] in out
        assert "table {" in out
        # the directive comment is consumed (replaced), the table survives
        assert "<!-- accent:" not in out
        assert "| A | B |" in out

    def test_list_accent_targets_li(self):
        content = "<!-- accent: sky -->\n- first\n- second\n"
        out = expand_accent_directives(content)
        assert "li {" in out
        assert _ACCENT_PALETTE["sky"]["border"] in out

    def test_ordered_list_accent_targets_li(self):
        content = "<!-- accent: mint -->\n1. first\n2. second\n"
        out = expand_accent_directives(content)
        assert "li {" in out
        assert _ACCENT_PALETTE["mint"]["border"] in out

    def test_blockquote_accent_targets_blockquote(self):
        content = "<!-- accent: lavender -->\n> quoted line\n"
        out = expand_accent_directives(content)
        assert "blockquote {" in out
        assert _ACCENT_PALETTE["lavender"]["border"] in out

    def test_unknown_color_is_left_untouched(self):
        content = "<!-- accent: rainbow -->\n| A | B |\n"
        out = expand_accent_directives(content)
        # unknown palette → directive preserved verbatim, no style injected
        assert "<!-- accent: rainbow -->" in out
        assert "<style scoped>" not in out

    def test_directive_without_recognized_block_is_untouched(self):
        content = "<!-- accent: coral -->\njust a paragraph of text\n"
        out = expand_accent_directives(content)
        assert "<!-- accent: coral -->" in out
        assert "<style scoped>" not in out

    def test_idempotent(self):
        content = "<!-- accent: sage -->\n| A | B |\n"
        once = expand_accent_directives(content)
        twice = expand_accent_directives(once)
        # comment already consumed → second pass is a no-op
        assert once == twice


# ---------------------------------------------------------------------------
# center_standalone_images
# ---------------------------------------------------------------------------

class TestCenterStandaloneImages:
    def test_standalone_image_is_flex_wrapped(self):
        content = '<img src="fig.png" style="max-height:460px;object-fit:contain" />'
        out = center_standalone_images(content)
        assert "display:flex;justify-content:center" in out
        assert 'src="fig.png"' in out

    def test_indentation_is_preserved(self):
        content = '    <img src="fig.png" style="object-fit:contain" />'
        out = center_standalone_images(content)
        assert out.startswith("    <div style=\"display:flex")

    def test_image_inside_figure_container_is_skipped(self):
        content = (
            '<div class="figure-landscape">\n'
            '<img src="fig.png" style="object-fit:contain" />\n'
            "</div>"
        )
        out = center_standalone_images(content)
        # container already centers — no double-wrap
        assert "display:flex;justify-content:center" not in out

    def test_image_without_object_fit_is_skipped(self):
        content = '<img src="logo.png" style="height:40px" />'
        out = center_standalone_images(content)
        assert "display:flex" not in out
        assert out == content

    def test_inline_image_with_trailing_text_is_skipped(self):
        # not standalone on its own line → must not match
        content = '<img src="x.png" style="object-fit:contain" /> caption text'
        out = center_standalone_images(content)
        assert "display:flex" not in out


# ---------------------------------------------------------------------------
# inject_lightbox_in_html
# ---------------------------------------------------------------------------

def _write_html(tmp_path, body):
    p = tmp_path / "deck.html"
    p.write_text(f"<html><body>\n{body}\n</body></html>")
    return p


class TestInjectLightboxInHtml:
    def test_marks_object_fit_image_and_injects_overlay(self, tmp_path):
        p = _write_html(tmp_path, '<img src="a.png" style="object-fit:contain" />')
        n = inject_lightbox_in_html(str(p))
        assert n == 1
        html = p.read_text()
        assert "marp-zoomable" in html
        assert "marp-lightbox-overlay" in html

    def test_marks_max_height_image(self, tmp_path):
        p = _write_html(tmp_path, '<img src="a.png" style="max-height:300px" />')
        n = inject_lightbox_in_html(str(p))
        assert n == 1

    def test_idempotent_second_pass_returns_zero(self, tmp_path):
        p = _write_html(tmp_path, '<img src="a.png" style="object-fit:contain" />')
        first = inject_lightbox_in_html(str(p))
        second = inject_lightbox_in_html(str(p))
        assert first == 1
        assert second == 0

    def test_emoji_image_is_excluded(self, tmp_path):
        p = _write_html(
            tmp_path,
            '<img class="emoji" src="e.png" style="max-height:20px" />',
        )
        n = inject_lightbox_in_html(str(p))
        assert n == 0
        assert "marp-lightbox-overlay" not in p.read_text()

    def test_image_without_style_is_excluded(self, tmp_path):
        # matches object-fit text but has no style= attribute → icon, skipped
        p = _write_html(tmp_path, "<img src=\"x.png\" data-css=\"object-fit:contain\">")
        n = inject_lightbox_in_html(str(p))
        assert n == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
