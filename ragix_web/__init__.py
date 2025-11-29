"""
RAGIX Web UI - Local-first web interface

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

# Import version from centralized source
try:
    from ragix_core.version import __version__
except ImportError:
    __version__ = "0.21.0"
