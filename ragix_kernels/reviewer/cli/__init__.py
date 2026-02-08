"""
KOAS Reviewer CLI — reviewctl command-line interface.

Usage:
    python -m ragix_kernels.reviewer.cli.reviewctl review doc.md
    python -m ragix_kernels.reviewer.cli.reviewctl report doc.md
    python -m ragix_kernels.reviewer.cli.reviewctl revert doc.md RVW-0001
    python -m ragix_kernels.reviewer.cli.reviewctl show doc.md RVW-0001
    python -m ragix_kernels.reviewer.cli.reviewctl grep doc.md "typo"

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

from ragix_kernels.reviewer.cli.reviewctl import main
