"""
RAGIX Version Management - Centralized version for all components

This module provides a single source of truth for the RAGIX version.
All components should import from here to ensure consistency.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-28
"""

# =============================================================================
# RAGIX Version - Single Source of Truth
# =============================================================================

__version__ = "0.31.1"

# Semantic versioning components
VERSION_MAJOR = 0
VERSION_MINOR = 31
VERSION_PATCH = 1
VERSION_SUFFIX = ""  # e.g., "alpha", "beta", "rc1", ""

# Build metadata
BUILD_DATE = "2025-12-04"
BUILD_AUTHOR = "Olivier Vitrac, PhD, HDR"
BUILD_EMAIL = "olivier.vitrac@adservio.fr"
BUILD_ORG = "Adservio"

# Full version string with optional suffix
VERSION_FULL = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
if VERSION_SUFFIX:
    VERSION_FULL = f"{VERSION_FULL}-{VERSION_SUFFIX}"


def get_version() -> str:
    """Get the current RAGIX version string."""
    return __version__


def get_version_info() -> dict:
    """Get detailed version information."""
    return {
        "version": __version__,
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR,
        "patch": VERSION_PATCH,
        "suffix": VERSION_SUFFIX,
        "full": VERSION_FULL,
        "build_date": BUILD_DATE,
        "author": BUILD_AUTHOR,
        "email": BUILD_EMAIL,
        "organization": BUILD_ORG,
    }


def get_banner() -> str:
    """Get a formatted version banner for CLI tools."""
    return f"""
╔══════════════════════════════════════════════════════════════╗
║  RAGIX v{__version__:<10}                                        ║
║  Retrieval-Augmented Generative Interactive eXecution        ║
║  {BUILD_AUTHOR} | {BUILD_ORG}                      ║
║  Build: {BUILD_DATE}                                         ║
╚══════════════════════════════════════════════════════════════╝
"""


def get_short_banner() -> str:
    """Get a compact version banner."""
    return f"RAGIX v{__version__} | {BUILD_ORG} | {BUILD_DATE}"


# For module-level access
VERSION = __version__
