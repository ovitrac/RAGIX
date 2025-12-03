"""
Mock repository utilities module for testing.
"""

import os
from pathlib import Path
from typing import List, Optional


def find_files(directory: str, pattern: str = "*") -> List[str]:
    """Find files matching pattern in directory."""
    path = Path(directory)
    return [str(f) for f in path.glob(pattern)]


def read_file(filepath: str) -> Optional[str]:
    """Read file contents."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None


def count_lines(filepath: str) -> int:
    """Count lines in a file."""
    content = read_file(filepath)
    if content is None:
        return 0
    return len(content.splitlines())


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(filepath)
    except FileNotFoundError:
        return 0


# TODO: Add more utility functions
# TODO: Implement caching
# FIXME: Handle encoding issues
