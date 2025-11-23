#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RIAGIX Unix Toolbox (v0.4)
=========================

Small, robust CLI helpers for sovereign/local LLMs.

Commands:
    - find     : recursive file listing with filters
    - stats    : global / per-extension stats
    - top      : top-N files by size/mtime/lines
    - lines    : count lines per file
    - grep     : search text in files (OR on patterns, AND via piping)
    - replace  : safe bulk text replacement with backups
    - doc2md   : convert docx/odt/pdf to Markdown via pandoc

Design goals:
    - Simple, predictable CLIs.
    - Safe defaults (no destructive behavior by surprise).
    - Clear error messages + basic install hints when tools are missing.



Author: Olivier Vitrac | Adservio Innovation Lab | olivier.vitrac@adservio.fr
Contact: olivier.vitrac@adservio.fr
"""

import argparse
import json
import os
import sys
import subprocess
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Tuple
import re


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def is_binary_file(path: Path, blocksize: int = 1024) -> bool:
    """Heuristic to detect binary files by presence of NUL bytes."""
    try:
        with path.open("rb") as f:
            chunk = f.read(blocksize)
        return b"\0" in chunk
    except Exception:
        return True  # conservative


def iter_files(root: Path, max_depth: Optional[int] = None) -> Iterable[Path]:
    """Yield files below root, with an optional max_depth (0 = only root)."""
    root = root.resolve()
    root_depth = len(root.parts)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if max_depth is not None:
            depth = len(p.resolve().parts) - root_depth
            if depth > max_depth:
                continue
        yield p


def file_info(path: Path) -> Dict[str, Any]:
    """Return basic file info as dict."""
    st = path.stat()
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
        "ext": path.suffix.lower(),
    }


def count_lines(path: Path) -> int:
    """Count text lines in a file, ignoring errors and treating as text."""
    n = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                n += 1
    except Exception:
        return -1
    return n


def print_json(obj: Any):
    json.dump(obj, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def print_tsv(rows: Iterable[Iterable[Any]]):
    for row in rows:
        sys.stdout.write("\t".join(str(x) for x in row) + "\n")


# ---------------------------------------------------------------------------
# Shared path collectors
# ---------------------------------------------------------------------------

def collect_paths_for_replace(args: argparse.Namespace) -> List[Path]:
    """Determine which files to process (from root/ext/glob or stdin)."""
    paths: List[Path] = []

    if getattr(args, "from_stdin", False):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            p = Path(line).expanduser()
            if p.is_file():
                paths.append(p)
        return paths

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        return paths

    exts = {e.lower() if e.startswith(".") else "." + e.lower()
            for e in (args.ext or [])}

    for p in iter_files(root, getattr(args, "max_depth", None)):
        if exts and p.suffix.lower() not in exts:
            continue
        paths.append(p)

    return paths


def collect_paths_for_grep(args: argparse.Namespace) -> List[Path]:
    """Determine which files to scan (from root/ext or stdin)."""
    return collect_paths_for_replace(args)


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------

def cmd_find(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.stderr.write(f"Root not found: {root}\n")
        return 1

    exts = {e.lower() if e.startswith(".") else "." + e.lower()
            for e in (args.ext or [])}
    name_substr = args.name or None

    results: List[Dict[str, Any]] = []
    for p in iter_files(root, args.max_depth):
        if exts and p.suffix.lower() not in exts:
            continue
        if name_substr and name_substr not in p.name:
            continue

        info = file_info(p)

        # size filter
        if args.min_size is not None and info["size"] < args.min_size:
            continue
        if args.max_size is not None and info["size"] > args.max_size:
            continue

        # date filter
        if args.since or args.until:
            mtime = datetime.fromisoformat(info["mtime"])
            if args.since:
                since_dt = datetime.fromisoformat(args.since)
                if mtime < since_dt:
                    continue
            if args.until:
                until_dt = datetime.fromisoformat(args.until)
                if mtime > until_dt:
                    continue

        results.append(info)

    if args.json:
        print_json(results)
    else:
        rows = ((r["path"], r["size"], r["mtime"], r["ext"]) for r in results)
        print_tsv(rows)

    return 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def cmd_stats(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.stderr.write(f"Root not found: {root}\n")
        return 1

    total_files = 0
    total_size = 0
    total_lines = 0
    per_ext: Dict[str, Dict[str, Any]] = {}

    for p in iter_files(root, args.max_depth):
        info = file_info(p)
        ext = info["ext"]
        total_files += 1
        total_size += info["size"]

        if args.with_lines and not is_binary_file(p):
            nlines = count_lines(p)
            if nlines >= 0:
                total_lines += nlines
        else:
            nlines = None

        if args.by_ext:
            d = per_ext.setdefault(ext, {"files": 0, "size": 0, "lines": 0})
            d["files"] += 1
            d["size"] += info["size"]
            if nlines is not None:
                d["lines"] += nlines

    result: Dict[str, Any] = {
        "root": str(root),
        "files": total_files,
        "size": total_size,
    }
    if args.with_lines:
        result["lines"] = total_lines
    if args.by_ext:
        result["by_ext"] = per_ext

    if args.json:
        print_json(result)
    else:
        sys.stdout.write(f"root\t{result['root']}\n")
        sys.stdout.write(f"files\t{total_files}\n")
        sys.stdout.write(f"size_bytes\t{total_size}\n")
        if args.with_lines:
            sys.stdout.write(f"lines\t{total_lines}\n")
        if args.by_ext:
            sys.stdout.write("by_ext\n")
            for ext, d in sorted(per_ext.items(), key=lambda kv: kv[0]):
                row = [ext or "<noext>", d["files"], d["size"]]
                if args.with_lines:
                    row.append(d["lines"])
                sys.stdout.write("\t".join(str(x) for x in row) + "\n")

    return 0


# ---------------------------------------------------------------------------
# top
# ---------------------------------------------------------------------------

def cmd_top(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.stderr.write(f"Root not found: {root}\n")
        return 1

    # Collect info + optional lines
    items: List[Dict[str, Any]] = []
    for p in iter_files(root, args.max_depth):
        info = file_info(p)
        if args.by == "lines" and not is_binary_file(p):
            info["lines"] = count_lines(p)
        items.append(info)

    key = args.by
    if key == "mtime":
        def sort_key(x):  # parse ISO once
            return datetime.fromisoformat(x["mtime"])
    else:
        def sort_key(x):  # size or lines
            return x.get(key, 0)

    items.sort(key=sort_key, reverse=not args.ascending)
    items = items[: args.n]

    if args.json:
        print_json(items)
    else:
        rows = []
        for it in items:
            if key == "lines":
                rows.append([it["path"], it.get("lines", -1)])
            elif key == "size":
                rows.append([it["path"], it["size"]])
            else:  # mtime
                rows.append([it["path"], it["mtime"]])
        print_tsv(rows)

    return 0


# ---------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------

def cmd_lines(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.stderr.write(f"Root not found: {root}\n")
        return 1

    exts = {e.lower() if e.startswith(".") else "." + e.lower()
            for e in (args.ext or [])}

    results: List[Dict[str, Any]] = []
    for p in iter_files(root, args.max_depth):
        if exts and p.suffix.lower() not in exts:
            continue
        if args.skip_binary and is_binary_file(p):
            continue
        n = count_lines(p)
        results.append({"path": str(p), "lines": n})

    if args.json:
        print_json(results)
    else:
        rows = ((r["path"], r["lines"]) for r in results)
        print_tsv(rows)

    return 0


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------

def cmd_grep(args: argparse.Namespace) -> int:
    """Search text in files (OR on patterns, AND via piping)."""
    if not args.pattern:
        sys.stderr.write("At least one --pattern (-e) is required.\n")
        return 1

    paths = collect_paths_for_grep(args)
    if not paths:
        sys.stderr.write("No files selected for grep.\n")
        return 1

    # prepare patterns
    if args.regex:
        patterns = [re.compile(p, re.IGNORECASE if args.ignore_case else 0)
                    for p in args.pattern]
    else:
        if args.ignore_case:
            patterns = [p.lower() for p in args.pattern]
        else:
            patterns = list(args.pattern)

    results_files: Dict[str, int] = {}   # path -> match_count
    results_lines: List[Dict[str, Any]] = []

    for p in paths:
        if args.skip_binary and is_binary_file(p):
            continue

        try:
            f = p.open("r", encoding="utf-8", errors="ignore")
        except Exception as e:
            sys.stderr.write(f"Skipping {p}: cannot read ({e})\n")
            continue

        match_count_for_file = 0
        with f:
            for lineno, line in enumerate(f, start=1):
                text = line.rstrip("\n")

                if args.regex:
                    ok = any(r.search(text) for r in patterns)
                else:
                    hay = text.lower() if args.ignore_case else text
                    ok = any(pat in hay for pat in patterns)

                if not ok:
                    continue

                match_count_for_file += 1

                if args.files_with_matches:
                    # we only care that there is at least one match
                    results_files[str(p)] = results_files.get(str(p), 0) + 1
                    break  # go to next file

                # line-level output
                rec = {
                    "path": str(p),
                    "line": lineno,
                    "text": text,
                }
                results_lines.append(rec)

                if args.max_matches_per_file is not None and \
                   match_count_for_file >= args.max_matches_per_file:
                    break

        if match_count_for_file > 0 and args.files_with_matches:
            continue

    # Output
    if args.json:
        if args.files_with_matches:
            out = [
                {"path": path, "matches": count}
                for path, count in sorted(results_files.items())
            ]
            print_json(out)
        else:
            print_json(results_lines)
    else:
        if args.files_with_matches:
            for path in sorted(results_files.keys()):
                sys.stdout.write(path + "\n")
        else:
            # classic grep-like: path:line:text
            for rec in results_lines:
                sys.stdout.write(
                    f"{rec['path']}:{rec['line']}:{rec['text']}\n"
                )

    return 0


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------

def cmd_replace(args: argparse.Namespace) -> int:
    if not args.old:
        sys.stderr.write("Missing --old pattern\n")
        return 1

    paths = collect_paths_for_replace(args)
    if not paths:
        sys.stderr.write("No files selected for replacement.\n")
        return 1

    backup_ext = None if args.no_backup else (args.backup_ext or ".bak")
    total_files = 0
    total_repl = 0

    if args.regex:
        pattern = re.compile(args.old)

    for p in paths:
        if is_binary_file(p):
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            sys.stderr.write(f"Skipping {p}: cannot read ({e})\n")
            continue

        if args.regex:
            new_text, n = pattern.subn(args.new, text)
        else:
            n = text.count(args.old)
            new_text = text.replace(args.old, args.new)

        if n == 0:
            continue

        total_files += 1
        total_repl += n

        if args.dry_run:
            sys.stdout.write(f"[DRY] {p}: {n} replacement(s)\n")
            continue

        # backup
        if backup_ext:
            backup_path = p.with_suffix(p.suffix + backup_ext)
            try:
                shutil.copy2(p, backup_path)
            except Exception as e:
                sys.stderr.write(f"Warning: cannot backup {p} ({e})\n")

        try:
            p.write_text(new_text, encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"Error writing {p}: {e}\n")

    if not args.dry_run:
        sys.stdout.write(
            f"Replacements: {total_repl} in {total_files} file(s)\n"
        )

    return 0


# ---------------------------------------------------------------------------
# doc2md
# ---------------------------------------------------------------------------

def ensure_pandoc() -> bool:
    """Return True if pandoc is available, otherwise print install hints."""
    if shutil.which("pandoc"):
        return True

    sys.stderr.write(
        "Error: 'pandoc' is not installed or not in PATH.\n"
        "Install suggestions:\n"
        "  - Debian/Ubuntu :  sudo apt-get install pandoc\n"
        "  - macOS (brew)  :  brew install pandoc\n"
        "  - Other         :  see https://pandoc.org/installing.html\n"
    )
    return False


def convert_one_doc(in_path: Path, out_dir: Path, overwrite: bool) -> Tuple[Path, int]:
    """Convert a single docx/odt/pdf to Markdown using pandoc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (in_path.stem + ".md")

    if out_path.exists() and not overwrite:
        return out_path, 0

    cmd = [
        "pandoc",
        str(in_path),
        "-o",
        str(out_path),
        "--to",
        "gfm",  # GitHub-flavored Markdown
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        ensure_pandoc()
        return out_path, 1

    if cp.returncode != 0:
        sys.stderr.write(
            f"pandoc failed for {in_path} (rc={cp.returncode}):\n{cp.stderr}\n"
        )
        return out_path, cp.returncode

    return out_path, 0


def cmd_doc2md(args: argparse.Namespace) -> int:
    if not ensure_pandoc():
        return 1

    in_paths: List[Path] = []
    for p_str in args.inputs:
        p = Path(p_str).expanduser().resolve()
        if p.is_dir():
            for ext in (".docx", ".odt", ".pdf"):
                in_paths.extend(p.rglob("*" + ext))
        elif p.is_file():
            in_paths.append(p)
        else:
            sys.stderr.write(f"Warning: input not found: {p}\n")

    if not in_paths:
        sys.stderr.write("No input documents to convert.\n")
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve()
    nb_ok = 0
    nb_fail = 0
    converted: List[str] = []

    for p in in_paths:
        out_path, rc = convert_one_doc(p, out_dir, args.overwrite)
        if rc == 0:
            nb_ok += 1
            converted.append(str(out_path))
        else:
            nb_fail += 1

    if args.json:
        result = {"ok": nb_ok, "failed": nb_fail, "outputs": converted}
        print_json(result)
    else:
        sys.stdout.write(f"Converted: {nb_ok}, Failed: {nb_fail}\n")
        for c in converted:
            sys.stdout.write(c + "\n")

    return 0


# ---------------------------------------------------------------------------
# SWE Navigation (v0.4)
# ---------------------------------------------------------------------------

@dataclass
class ViewState:
    """Track last viewed window per file for scroll operations."""
    file_windows: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Structure: {path: {"start_line": N, "chunk_size": 100, "overlap": 2}}

    @classmethod
    def load(cls, state_file: Path) -> "ViewState":
        """Load from JSON, return empty if missing/invalid."""
        if not state_file.exists():
            return cls()
        try:
            with state_file.open("r") as f:
                data = json.load(f)
            return cls(file_windows=data.get("file_windows", {}))
        except Exception:
            # Corrupted or invalid - return empty state
            return cls()

    def save(self, state_file: Path):
        """Save to JSON atomically."""
        data = {"file_windows": self.file_windows}
        temp_file = state_file.with_suffix(".tmp")
        try:
            with temp_file.open("w") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(state_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e

    def get_window(self, path: str) -> Optional[Dict[str, int]]:
        """Get last window for path."""
        return self.file_windows.get(path)

    def set_window(self, path: str, start_line: int, chunk_size: int = 100, overlap: int = 2):
        """Update window for path."""
        self.file_windows[path] = {
            "start_line": start_line,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }


def open_window(
    path: Path,
    start: Optional[int] = None,
    center_line: Optional[int] = None,
    line_range: Optional[Tuple[int, int]] = None,
    chunk_size: int = 100,
    max_window: int = 200
) -> str:
    """
    Open a file window with line numbers.

    Modes:
    - open_window(path) → lines 1-100
    - open_window(path, center_line=50) → centered 100-line window around line 50
    - open_window(path, line_range=(10, 50)) → explicit range (enforces max_window)
    - open_window(path, start=100) → lines 100-199

    Returns formatted output:
    --- path (lines start-end) ---
    start: first line
    start+1: second line
    ...
    """
    if not path.exists() or not path.is_file():
        return f"Error: file not found or not accessible: {path}"

    # Read all lines
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading {path}: {e}"

    total_lines = len(lines)
    if total_lines == 0:
        return f"--- {path} (empty file) ---"

    # Determine window boundaries
    if line_range is not None:
        start_line, end_line = line_range
        # Enforce max_window
        if end_line - start_line + 1 > max_window:
            end_line = start_line + max_window - 1
    elif center_line is not None:
        half = chunk_size // 2
        start_line = max(1, center_line - half)
        end_line = min(total_lines, start_line + chunk_size - 1)
    elif start is not None:
        start_line = start
        end_line = min(total_lines, start_line + chunk_size - 1)
    else:
        # Default: from beginning
        start_line = 1
        end_line = min(total_lines, chunk_size)

    # Clamp boundaries
    start_line = max(1, min(start_line, total_lines))
    end_line = max(start_line, min(end_line, total_lines))

    # Format output
    output_lines = [f"--- {path} (lines {start_line}-{end_line} of {total_lines}) ---"]
    for i in range(start_line - 1, end_line):
        output_lines.append(f"{i+1:6d}: {lines[i].rstrip()}")

    return "\n".join(output_lines)


def scroll_window(
    path: Path,
    direction: str,
    state_file: Path,
    chunk_size: int = 100,
    overlap: int = 2
) -> str:
    """
    Scroll view window up/down with overlap.

    Uses ViewState to track last position.
    If no prior view, behaves like open_window(path).
    """
    if not path.exists() or not path.is_file():
        return f"Error: file not found: {path}"

    # Load state
    state = ViewState.load(state_file)
    window = state.get_window(str(path))

    if window is None:
        # No prior view - start from beginning
        result = open_window(path, start=1, chunk_size=chunk_size)
        state.set_window(str(path), 1, chunk_size, overlap)
        state.save(state_file)
        return result

    # Calculate new start_line
    prev_start = window["start_line"]
    prev_chunk = window.get("chunk_size", chunk_size)
    prev_overlap = window.get("overlap", overlap)

    if direction == "+":
        new_start = prev_start + prev_chunk - prev_overlap
    else:  # "-"
        new_start = max(1, prev_start - prev_chunk + prev_overlap)

    # Open window at new position
    result = open_window(path, start=new_start, chunk_size=chunk_size)

    # Update state
    state.set_window(str(path), new_start, chunk_size, overlap)
    state.save(state_file)

    return result


# ---------------------------------------------------------------------------
# SWE Search (single file)
# ---------------------------------------------------------------------------

def grep_file(
    path: Path,
    pattern: str,
    regex: bool = False,
    ignore_case: bool = False,
    max_matches: Optional[int] = None
) -> str:
    """
    Search within a single file.

    Returns:
    line_no: matching text
    line_no: matching text
    ...
    """
    if not path.exists() or not path.is_file():
        return f"Error: file not found: {path}"

    if is_binary_file(path):
        return f"Error: binary file (skipped): {path}"

    # Prepare pattern
    if regex:
        try:
            pat = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            return f"Error: invalid regex: {e}"
    else:
        if ignore_case:
            pattern_cmp = pattern.lower()
        else:
            pattern_cmp = pattern

    # Search
    results = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, start=1):
                text = line.rstrip("\n")

                if regex:
                    match = pat.search(text)
                else:
                    hay = text.lower() if ignore_case else text
                    match = pattern_cmp in hay

                if match:
                    results.append(f"{lineno:6d}: {text}")
                    if max_matches is not None and len(results) >= max_matches:
                        break
    except Exception as e:
        return f"Error reading {path}: {e}"

    if not results:
        return f"No matches found in {path}"

    return "\n".join(results)


# ---------------------------------------------------------------------------
# SWE Edit (line-based)
# ---------------------------------------------------------------------------

def check_swe_edit_allowed(profile: str) -> Tuple[bool, str]:
    """
    Check if SWE edit operations are allowed in current profile.

    Returns (allowed: bool, message: str)
    """
    ragix_enable_swe = os.environ.get("RAGIX_ENABLE_SWE", "1") == "1"

    if not ragix_enable_swe:
        return False, "SWE tools disabled (RAGIX_ENABLE_SWE=0)"

    if profile == "safe-read-only":
        return False, "Edit operations blocked in safe-read-only profile"

    return True, ""


def edit_range(
    path: Path,
    start_line: int,
    end_line: int,
    new_text: str,
    show_diff: bool = False,
    profile: str = "dev",
    backup_ext: str = ".bak"
) -> str:
    """
    Replace lines [start_line, end_line] (1-based, inclusive) with new_text.

    Safety:
    - Respects profile: blocks in "safe-read-only" or shows dry-run
    - Creates .bak backup
    - Atomic write via temp file

    Returns:
    ✓ Edited path:start-end
    [edited region preview]
    [optional: git diff if show_diff=True]
    """
    if not path.exists() or not path.is_file():
        return f"Error: file not found: {path}"

    if is_binary_file(path):
        return f"Error: binary file (cannot edit): {path}"

    # Profile check
    allowed, msg = check_swe_edit_allowed(profile)
    if not allowed:
        return f"Error: {msg}"

    # Read lines
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading {path}: {e}"

    total_lines = len(lines)

    # Validate range
    if start_line < 1 or end_line < start_line or start_line > total_lines:
        return f"Error: invalid line range {start_line}-{end_line} (file has {total_lines} lines)"

    # Clamp end_line
    end_line = min(end_line, total_lines)

    # Backup
    backup_path = path.with_suffix(path.suffix + backup_ext)
    try:
        shutil.copy2(path, backup_path)
    except Exception as e:
        return f"Error creating backup: {e}"

    # Prepare new content
    new_lines = new_text.splitlines(keepends=True)
    # Ensure last line has newline if new_text was non-empty
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    # Replace range
    updated = lines[: start_line - 1] + new_lines + lines[end_line:]

    # Write atomically
    temp_path = path.with_suffix(".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            f.writelines(updated)
        temp_path.replace(path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return f"Error writing {path}: {e}"

    # Format output
    output_lines = [f"✓ Edited {path}:{start_line}-{end_line}"]

    # Show edited region (±2 lines context)
    preview_start = max(0, start_line - 3)
    preview_end = min(len(updated), start_line + len(new_lines) + 2)
    output_lines.append("\n--- Edited region ---")
    for i in range(preview_start, preview_end):
        output_lines.append(f"{i+1:6d}: {updated[i].rstrip()}")

    # Optional diff
    if show_diff or os.environ.get("RAGIX_AUTO_DIFF", "0") == "1":
        try:
            result = subprocess.run(
                ["git", "diff", "--", str(path)],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                output_lines.append("\n--- git diff ---")
                output_lines.append(result.stdout)
        except Exception:
            pass  # Git not available or timeout

    return "\n".join(output_lines)


def insert_at(
    path: Path,
    line: int,
    new_text: str,
    show_diff: bool = False,
    profile: str = "dev",
    backup_ext: str = ".bak"
) -> str:
    """
    Insert new_text before line (1-based).

    line=N+1 appends to end.
    Returns formatted confirmation + preview.
    """
    if not path.exists() or not path.is_file():
        return f"Error: file not found: {path}"

    if is_binary_file(path):
        return f"Error: binary file (cannot edit): {path}"

    # Profile check
    allowed, msg = check_swe_edit_allowed(profile)
    if not allowed:
        return f"Error: {msg}"

    # Read lines
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return f"Error reading {path}: {e}"

    total_lines = len(lines)

    # Validate line
    if line < 1 or line > total_lines + 1:
        return f"Error: invalid line {line} (file has {total_lines} lines, max insert at {total_lines + 1})"

    # Backup
    backup_path = path.with_suffix(path.suffix + backup_ext)
    try:
        shutil.copy2(path, backup_path)
    except Exception as e:
        return f"Error creating backup: {e}"

    # Prepare new content
    new_lines = new_text.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    # Insert
    updated = lines[: line - 1] + new_lines + lines[line - 1 :]

    # Write atomically
    temp_path = path.with_suffix(".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            f.writelines(updated)
        temp_path.replace(path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return f"Error writing {path}: {e}"

    # Format output
    output_lines = [f"✓ Inserted at {path}:{line}"]

    # Show inserted region (±2 lines context)
    preview_start = max(0, line - 3)
    preview_end = min(len(updated), line + len(new_lines) + 2)
    output_lines.append("\n--- Inserted region ---")
    for i in range(preview_start, preview_end):
        output_lines.append(f"{i+1:6d}: {updated[i].rstrip()}")

    # Optional diff
    if show_diff or os.environ.get("RAGIX_AUTO_DIFF", "0") == "1":
        try:
            result = subprocess.run(
                ["git", "diff", "--", str(path)],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                output_lines.append("\n--- git diff ---")
                output_lines.append(result.stdout)
        except Exception:
            pass

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# SWE CLI Commands
# ---------------------------------------------------------------------------

def cmd_open(args: argparse.Namespace) -> int:
    """Handle 'open' command: rt open path[:line|start-end]"""
    path_spec = args.path_spec
    chunk_size = args.chunk_size

    # Parse path_spec: path, path:line, or path:start-end
    if ":" in path_spec:
        path_str, spec = path_spec.rsplit(":", 1)
        path = Path(path_str).expanduser().resolve()

        if "-" in spec:
            # Range: path:start-end
            try:
                start, end = spec.split("-")
                start_line = int(start)
                end_line = int(end)
                result = open_window(path, line_range=(start_line, end_line), chunk_size=chunk_size)
            except ValueError:
                sys.stderr.write(f"Invalid range spec: {spec}\n")
                return 1
        else:
            # Center line: path:line
            try:
                center = int(spec)
                result = open_window(path, center_line=center, chunk_size=chunk_size)
            except ValueError:
                sys.stderr.write(f"Invalid line number: {spec}\n")
                return 1
    else:
        # Simple path: from beginning
        path = Path(path_spec).expanduser().resolve()
        result = open_window(path, chunk_size=chunk_size)

    sys.stdout.write(result + "\n")
    return 0


def cmd_scroll(args: argparse.Namespace) -> int:
    """Handle 'scroll' command: rt scroll path +/-"""
    path = Path(args.path).expanduser().resolve()
    direction = args.direction
    chunk_size = args.chunk_size
    overlap = args.overlap

    # State file in current directory
    state_file = Path(".ragix_view_state.json")

    result = scroll_window(path, direction, state_file, chunk_size, overlap)
    sys.stdout.write(result + "\n")
    return 0


def cmd_grep_file(args: argparse.Namespace) -> int:
    """Handle 'grep-file' command: rt grep-file pattern path"""
    path = Path(args.path).expanduser().resolve()
    pattern = args.pattern
    regex = args.regex
    ignore_case = args.ignore_case
    max_matches = getattr(args, "max_matches", None)

    result = grep_file(path, pattern, regex, ignore_case, max_matches)
    sys.stdout.write(result + "\n")
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    """Handle 'edit' command: rt edit path start_line end_line < new_text"""
    path = Path(args.path).expanduser().resolve()
    start_line = args.start_line
    end_line = args.end_line
    show_diff = args.show_diff
    profile = os.environ.get("UNIX_RAG_PROFILE", "dev").lower()

    # Read new text from stdin
    new_text = sys.stdin.read()

    result = edit_range(path, start_line, end_line, new_text, show_diff, profile)
    sys.stdout.write(result + "\n")
    return 0


def cmd_insert(args: argparse.Namespace) -> int:
    """Handle 'insert' command: rt insert path line < new_text"""
    path = Path(args.path).expanduser().resolve()
    line = args.line
    show_diff = args.show_diff
    profile = os.environ.get("UNIX_RAG_PROFILE", "dev").lower()

    # Read new text from stdin
    new_text = sys.stdin.read()

    result = insert_at(path, line, new_text, show_diff, profile)
    sys.stdout.write(result + "\n")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ragix Unix Toolbox – robust helpers for sovereign LLMs"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # find
    pf = sub.add_parser("find", help="recursive file listing with filters")
    pf.add_argument("root", help="root directory")
    pf.add_argument("--ext", action="append", help="extension filter (repeatable, e.g. --ext py)")
    pf.add_argument("--name", help="substring filter on file name")
    pf.add_argument("--max-depth", type=int, help="maximum depth (0 = only root)")
    pf.add_argument("--min-size", type=int, help="min size in bytes")
    pf.add_argument("--max-size", type=int, help="max size in bytes")
    pf.add_argument("--since", help="filter mtime >= ISO date (YYYY-MM-DD or full ISO)")
    pf.add_argument("--until", help="filter mtime <= ISO date")
    pf.add_argument("--json", action="store_true", help="output JSON")
    pf.set_defaults(func=cmd_find)

    # stats
    ps = sub.add_parser("stats", help="global / per-extension statistics")
    ps.add_argument("root", help="root directory")
    ps.add_argument("--max-depth", type=int, help="maximum depth")
    ps.add_argument("--with-lines", action="store_true", help="count lines (slow on large trees)")
    ps.add_argument("--by-ext", action="store_true", help="group statistics by extension")
    ps.add_argument("--json", action="store_true", help="output JSON")
    ps.set_defaults(func=cmd_stats)

    # top
    pt = sub.add_parser("top", help="top-N files by size/mtime/lines")
    pt.add_argument("root", help="root directory")
    pt.add_argument("-n", type=int, default=20, help="number of files to show")
    pt.add_argument("--max-depth", type=int, help="maximum depth")
    pt.add_argument("--by", choices=["size", "mtime", "lines"], default="size", help="metric")
    pt.add_argument("--ascending", action="store_true", help="sort ascending (default: descending)")
    pt.add_argument("--json", action="store_true", help="output JSON")
    pt.set_defaults(func=cmd_top)

    # lines
    pl = sub.add_parser("lines", help="count lines per file")
    pl.add_argument("root", help="root directory")
    pl.add_argument("--max-depth", type=int, help="maximum depth")
    pl.add_argument("--ext", action="append", help="extension filter")
    pl.add_argument("--skip-binary", action="store_true", help="skip binary files")
    pl.add_argument("--json", action="store_true", help="output JSON")
    pl.set_defaults(func=cmd_lines)

    # grep
    pg = sub.add_parser("grep", help="search text in files (OR on patterns, AND via piping)")
    pg.add_argument("--root", default=".", help="root directory (ignored if --from-stdin is used)")
    pg.add_argument("--max-depth", type=int, help="maximum depth for recursive search")
    pg.add_argument("--ext", action="append", help="extension filter (repeatable, e.g. --ext py)")
    pg.add_argument("--from-stdin", action="store_true",
                    help="read file paths from stdin (one per line) instead of scanning --root")
    pg.add_argument("-e", "--pattern", action="append",
                    help="pattern to search (repeatable, OR logic)", required=True)
    pg.add_argument("--regex", action="store_true", help="treat patterns as regular expressions")
    pg.add_argument("-i", "--ignore-case", action="store_true", help="case-insensitive search")
    pg.add_argument("--skip-binary", action="store_true", help="skip binary files")
    pg.add_argument("-l", "--files-with-matches", action="store_true",
                    help="list only file paths that contain at least one match")
    pg.add_argument("--max-matches-per-file", type=int, help="stop after N matches per file")
    pg.add_argument("--json", action="store_true", help="output JSON")
    pg.set_defaults(func=cmd_grep)

    # replace
    pr = sub.add_parser("replace", help="safe bulk text replacement")
    pr.add_argument("--root", default=".", help="root directory (if not using --from-stdin)")
    pr.add_argument("--max-depth", type=int, help="maximum depth")
    pr.add_argument("--ext", action="append", help="extension filter (repeatable)")
    pr.add_argument("--from-stdin", action="store_true",
                    help="read explicit file paths from stdin (one per line)")
    pr.add_argument("--old", required=True, help="text or regex to replace")
    pr.add_argument("--new", default="", help="replacement text")
    pr.add_argument("--regex", action="store_true", help="treat --old as regex pattern")
    pr.add_argument("--dry-run", action="store_true", help="do not modify files, just report")
    pr.add_argument("--no-backup", action="store_true", help="do not create backups")
    pr.add_argument("--backup-ext", help="backup extension (default: .bak)")
    pr.set_defaults(func=cmd_replace)

    # doc2md
    pd = sub.add_parser("doc2md", help="convert docx/odt/pdf to Markdown via pandoc")
    pd.add_argument("inputs", nargs="+",
                    help="input files or directories (directories are scanned recursively)")
    pd.add_argument("--out-dir", default="md_out",
                    help="output directory for .md files (default: md_out)")
    pd.add_argument("--overwrite", action="store_true",
                    help="overwrite existing .md files")
    pd.add_argument("--json", action="store_true", help="output JSON summary")
    pd.set_defaults(func=cmd_doc2md)

    # SWE tools: open
    po = sub.add_parser("open", help="Open file window with line numbers (SWE)")
    po.add_argument("path_spec", help="path, path:line, or path:start-end")
    po.add_argument("--chunk-size", type=int, default=100, help="window size (default 100)")
    po.set_defaults(func=cmd_open)

    # SWE tools: scroll
    psc = sub.add_parser("scroll", help="Scroll file view up/down (SWE)")
    psc.add_argument("path", help="file path")
    psc.add_argument("direction", choices=["+", "-"], help="scroll direction")
    psc.add_argument("--chunk-size", type=int, default=100, help="window size")
    psc.add_argument("--overlap", type=int, default=2, help="line overlap between windows")
    psc.set_defaults(func=cmd_scroll)

    # SWE tools: grep-file
    pgf = sub.add_parser("grep-file", help="Search within single file (SWE)")
    pgf.add_argument("pattern", help="search pattern")
    pgf.add_argument("path", help="file path")
    pgf.add_argument("--regex", action="store_true", help="treat pattern as regex")
    pgf.add_argument("-i", "--ignore-case", action="store_true", help="case-insensitive search")
    pgf.add_argument("--max-matches", type=int, help="stop after N matches")
    pgf.set_defaults(func=cmd_grep_file)

    # SWE tools: edit
    pe = sub.add_parser("edit", help="Replace line range (SWE, reads from stdin)")
    pe.add_argument("path", help="file path")
    pe.add_argument("start_line", type=int, help="start line (1-based)")
    pe.add_argument("end_line", type=int, help="end line (inclusive)")
    pe.add_argument("--show-diff", action="store_true", help="show git diff after edit")
    pe.set_defaults(func=cmd_edit)

    # SWE tools: insert
    pi = sub.add_parser("insert", help="Insert before line (SWE, reads from stdin)")
    pi.add_argument("path", help="file path")
    pi.add_argument("line", type=int, help="line number (1-based)")
    pi.add_argument("--show-diff", action="store_true", help="show git diff after insert")
    pi.set_defaults(func=cmd_insert)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
