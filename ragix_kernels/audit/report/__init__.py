"""
KOAS Report Generation — Stage 3 Kernels

Provides internationalized report generation with:
- Configurable language (fr, en)
- Custom YAML frontmatter (header/footer)
- Section-based report assembly
- Synthesis from Stage 1 & 2 kernel outputs

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from ragix_kernels.audit.report.i18n import I18N, get_translator
from ragix_kernels.audit.report.templates import ReportTemplate, FrontMatter
from ragix_kernels.audit.report.renderer import MarkdownRenderer

__all__ = [
    "I18N",
    "get_translator",
    "ReportTemplate",
    "FrontMatter",
    "MarkdownRenderer",
]
