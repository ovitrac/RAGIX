"""
File Stats Plugin - File and codebase statistics tools

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def file_stats(path: str) -> Dict[str, Any]:
    """
    Get statistics for a file.

    Args:
        path: Path to file

    Returns:
        File statistics
    """
    file_path = Path(path)

    if not file_path.exists():
        return {
            "success": False,
            "error": f"File not found: {path}",
        }

    if not file_path.is_file():
        return {
            "success": False,
            "error": f"Not a file: {path}",
        }

    stats = file_path.stat()

    result = {
        "success": True,
        "path": str(file_path.absolute()),
        "name": file_path.name,
        "extension": file_path.suffix,
        "size_bytes": stats.st_size,
        "size_human": _human_size(stats.st_size),
        "modified": stats.st_mtime,
        "created": stats.st_ctime,
    }

    # Try to read as text
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.splitlines()

        result["encoding"] = "utf-8"
        result["line_count"] = len(lines)
        result["char_count"] = len(content)
        result["word_count"] = len(content.split())
        result["is_text"] = True

        # Line length stats
        if lines:
            line_lengths = [len(line) for line in lines]
            result["avg_line_length"] = sum(line_lengths) / len(line_lengths)
            result["max_line_length"] = max(line_lengths)

        # Empty line count
        result["blank_lines"] = sum(1 for line in lines if not line.strip())

    except UnicodeDecodeError:
        result["is_text"] = False
        result["encoding"] = "binary"

    except Exception as e:
        result["read_error"] = str(e)

    return result


def directory_stats(path: str, recursive: bool = True) -> Dict[str, Any]:
    """
    Get statistics for a directory.

    Args:
        path: Path to directory
        recursive: Include subdirectories

    Returns:
        Directory statistics
    """
    dir_path = Path(path)

    if not dir_path.exists():
        return {
            "success": False,
            "error": f"Directory not found: {path}",
        }

    if not dir_path.is_dir():
        return {
            "success": False,
            "error": f"Not a directory: {path}",
        }

    # Collect stats
    file_count = 0
    dir_count = 0
    total_size = 0
    extensions: Dict[str, int] = defaultdict(int)
    extension_sizes: Dict[str, int] = defaultdict(int)

    if recursive:
        items = dir_path.rglob("*")
    else:
        items = dir_path.glob("*")

    for item in items:
        if item.is_file():
            file_count += 1
            size = item.stat().st_size
            total_size += size

            ext = item.suffix.lower() or "(no ext)"
            extensions[ext] += 1
            extension_sizes[ext] += size

        elif item.is_dir():
            dir_count += 1

    # Top extensions by count
    top_extensions = sorted(
        extensions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Top extensions by size
    top_by_size = sorted(
        extension_sizes.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return {
        "success": True,
        "path": str(dir_path.absolute()),
        "recursive": recursive,
        "file_count": file_count,
        "directory_count": dir_count,
        "total_size_bytes": total_size,
        "total_size_human": _human_size(total_size),
        "extensions": dict(extensions),
        "top_extensions_by_count": top_extensions,
        "top_extensions_by_size": [
            (ext, _human_size(size)) for ext, size in top_by_size
        ],
    }


def code_stats(
    path: str,
    extensions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get code statistics (lines of code, comments, blanks).

    Args:
        path: Path to file or directory
        extensions: File extensions to include (default: common code files)

    Returns:
        Code statistics
    """
    target = Path(path)

    if not target.exists():
        return {
            "success": False,
            "error": f"Path not found: {path}",
        }

    # Default extensions
    if extensions is None:
        extensions = [
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".cpp", ".c", ".h", ".hpp",
            ".go", ".rs", ".rb", ".php",
            ".sh", ".bash", ".zsh",
        ]

    # Normalize extensions
    extensions = [e if e.startswith(".") else f".{e}" for e in extensions]

    # Collect files
    files = []
    if target.is_file():
        if target.suffix.lower() in extensions:
            files.append(target)
    else:
        for ext in extensions:
            files.extend(target.rglob(f"*{ext}"))

    # Analyze files
    total_lines = 0
    total_code = 0
    total_comments = 0
    total_blanks = 0
    file_stats_list = []

    for file_path in files:
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.splitlines()

            # Analyze lines
            code = 0
            comments = 0
            blanks = 0

            in_block_comment = False
            comment_patterns = _get_comment_patterns(file_path.suffix)

            for line in lines:
                stripped = line.strip()

                if not stripped:
                    blanks += 1
                    continue

                # Check for block comments
                if comment_patterns["block_start"] and comment_patterns["block_end"]:
                    if comment_patterns["block_start"] in stripped:
                        in_block_comment = True
                    if in_block_comment:
                        comments += 1
                        if comment_patterns["block_end"] in stripped:
                            in_block_comment = False
                        continue

                # Check for line comments
                if comment_patterns["line"]:
                    is_comment = False
                    for pattern in comment_patterns["line"]:
                        if stripped.startswith(pattern):
                            is_comment = True
                            break
                    if is_comment:
                        comments += 1
                        continue

                code += 1

            file_stats_list.append({
                "file": str(file_path),
                "lines": len(lines),
                "code": code,
                "comments": comments,
                "blanks": blanks,
            })

            total_lines += len(lines)
            total_code += code
            total_comments += comments
            total_blanks += blanks

        except Exception as e:
            file_stats_list.append({
                "file": str(file_path),
                "error": str(e),
            })

    # Language breakdown
    by_language: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "files": 0, "lines": 0, "code": 0
    })

    for fs in file_stats_list:
        if "error" in fs:
            continue
        ext = Path(fs["file"]).suffix.lower()
        by_language[ext]["files"] += 1
        by_language[ext]["lines"] += fs["lines"]
        by_language[ext]["code"] += fs["code"]

    return {
        "success": True,
        "path": str(target.absolute()),
        "file_count": len(files),
        "total_lines": total_lines,
        "code_lines": total_code,
        "comment_lines": total_comments,
        "blank_lines": total_blanks,
        "code_ratio": round(total_code / total_lines * 100, 1) if total_lines else 0,
        "comment_ratio": round(total_comments / total_lines * 100, 1) if total_lines else 0,
        "by_language": dict(by_language),
        "files": file_stats_list[:20] if len(file_stats_list) > 20 else file_stats_list,
        "truncated": len(file_stats_list) > 20,
    }


def _human_size(size: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _get_comment_patterns(extension: str) -> Dict[str, Any]:
    """Get comment patterns for a file extension."""
    patterns = {
        ".py": {"line": ["#"], "block_start": '"""', "block_end": '"""'},
        ".js": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".ts": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".jsx": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".tsx": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".java": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".c": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".cpp": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".h": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".hpp": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".go": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".rs": {"line": ["//"], "block_start": "/*", "block_end": "*/"},
        ".rb": {"line": ["#"], "block_start": "=begin", "block_end": "=end"},
        ".php": {"line": ["//", "#"], "block_start": "/*", "block_end": "*/"},
        ".sh": {"line": ["#"], "block_start": None, "block_end": None},
        ".bash": {"line": ["#"], "block_start": None, "block_end": None},
    }

    return patterns.get(extension.lower(), {"line": ["#"], "block_start": None, "block_end": None})
