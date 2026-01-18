#!/usr/bin/env python3
"""
Markdown Fixer — Clean and normalize markdown documents.

Author: Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
Date: 2026-01-18

Fixes common markdown issues:
- Section renumbering (1., 1.1, 1.1.1, etc.)
- Heading level normalization
- List renumbering
- Stray formatting artifacts (**, *, etc.)
- Trailing whitespace
- Inconsistent blank lines

Usage:
    python fix_markdown.py input.md [output.md] [--options]
    python fix_markdown.py input.md --dry-run
    python fix_markdown.py input.md --renumber-sections
    python fix_markdown.py input.md --fix-artifacts
    python fix_markdown.py input.md --all
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FixerConfig:
    """Configuration for markdown fixer."""
    renumber_sections: bool = False
    renumber_lists: bool = False
    fix_artifacts: bool = False
    normalize_headings: bool = False
    fix_whitespace: bool = False
    fix_blank_lines: bool = False
    all_fixes: bool = False
    dry_run: bool = False


@dataclass
class FixResult:
    """Result of a fix operation."""
    original_lines: int = 0
    fixed_lines: int = 0
    changes: List[Tuple[int, str, str]] = field(default_factory=list)


class MarkdownFixer:
    """Fix common markdown issues."""

    # Patterns
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')
    NUMBERED_SECTION_PATTERN = re.compile(r'^(#{1,6})\s+(\d+(?:\.\d+)*\.?)\s+(.+)$')
    NUMBERED_LIST_PATTERN = re.compile(r'^(\s*)(\d+)\.\s+(.+)$')
    STRAY_BOLD_PATTERN = re.compile(r'^\*\*\s*$|^\*\*$')
    STRAY_ITALIC_PATTERN = re.compile(r'^\*\s*$|^\*$')
    TRAILING_WHITESPACE_PATTERN = re.compile(r'\s+$')
    MULTIPLE_BLANK_LINES_PATTERN = re.compile(r'\n{3,}')

    def __init__(self, config: FixerConfig):
        self.config = config

    def fix(self, content: str) -> Tuple[str, FixResult]:
        """Apply all configured fixes to content."""
        result = FixResult()
        lines = content.split('\n')
        result.original_lines = len(lines)

        if self.config.all_fixes or self.config.fix_whitespace:
            lines = self._fix_trailing_whitespace(lines, result)

        if self.config.all_fixes or self.config.fix_artifacts:
            lines = self._fix_stray_artifacts(lines, result)

        if self.config.all_fixes or self.config.renumber_sections:
            lines = self._renumber_sections(lines, result)

        if self.config.all_fixes or self.config.renumber_lists:
            lines = self._renumber_lists(lines, result)

        if self.config.all_fixes or self.config.normalize_headings:
            lines = self._normalize_headings(lines, result)

        content = '\n'.join(lines)

        if self.config.all_fixes or self.config.fix_blank_lines:
            content = self._fix_blank_lines(content, result)

        result.fixed_lines = len(content.split('\n'))
        return content, result

    def _fix_trailing_whitespace(self, lines: List[str], result: FixResult) -> List[str]:
        """Remove trailing whitespace from lines."""
        fixed = []
        for i, line in enumerate(lines):
            new_line = self.TRAILING_WHITESPACE_PATTERN.sub('', line)
            if new_line != line:
                result.changes.append((i + 1, 'whitespace', f'Removed trailing whitespace'))
            fixed.append(new_line)
        return fixed

    def _fix_stray_artifacts(self, lines: List[str], result: FixResult) -> List[str]:
        """Remove stray formatting artifacts like isolated ** or *."""
        fixed = []
        for i, line in enumerate(lines):
            # Remove isolated ** or * on their own line
            if self.STRAY_BOLD_PATTERN.match(line.strip()) or self.STRAY_ITALIC_PATTERN.match(line.strip()):
                result.changes.append((i + 1, 'artifact', f'Removed stray formatting: "{line.strip()}"'))
                continue

            # Fix ** at start/end of content blocks
            new_line = line
            if new_line.startswith('** ') and not new_line.rstrip().endswith('**'):
                new_line = new_line[3:]
                result.changes.append((i + 1, 'artifact', f'Removed leading **'))
            if new_line.rstrip().endswith(' **') and not new_line.lstrip().startswith('**'):
                new_line = new_line.rstrip()[:-3]
                result.changes.append((i + 1, 'artifact', f'Removed trailing **'))

            fixed.append(new_line)
        return fixed

    def _renumber_sections(self, lines: List[str], result: FixResult) -> List[str]:
        """Renumber section headings sequentially."""
        fixed = []
        counters = [0] * 6  # For h1-h6

        for i, line in enumerate(lines):
            match = self.NUMBERED_SECTION_PATTERN.match(line)
            if match:
                hashes = match.group(1)
                old_number = match.group(2)
                title = match.group(3)
                level = len(hashes) - 1  # 0-indexed

                # Increment counter for this level
                counters[level] += 1
                # Reset all deeper counters
                for j in range(level + 1, 6):
                    counters[j] = 0

                # Build new number
                new_number = '.'.join(str(counters[k]) for k in range(level + 1) if counters[k] > 0)

                if old_number.rstrip('.') != new_number:
                    result.changes.append((i + 1, 'renumber', f'Section {old_number} → {new_number}'))

                new_line = f"{hashes} {new_number}. {title}"
                fixed.append(new_line)
            else:
                # Check for unnumbered headings at level 1
                heading_match = self.HEADING_PATTERN.match(line)
                if heading_match:
                    hashes = heading_match.group(1)
                    level = len(hashes) - 1
                    # Reset deeper counters when hitting an unnumbered heading
                    for j in range(level + 1, 6):
                        counters[j] = 0
                fixed.append(line)

        return fixed

    def _renumber_lists(self, lines: List[str], result: FixResult) -> List[str]:
        """Renumber ordered lists sequentially."""
        fixed = []
        list_counters = {}  # indent_level -> counter

        for i, line in enumerate(lines):
            match = self.NUMBERED_LIST_PATTERN.match(line)
            if match:
                indent = match.group(1)
                old_num = match.group(2)
                content = match.group(3)
                indent_level = len(indent)

                # Get or initialize counter for this indent level
                if indent_level not in list_counters:
                    list_counters[indent_level] = 0
                list_counters[indent_level] += 1

                # Reset deeper levels
                for level in list(list_counters.keys()):
                    if level > indent_level:
                        del list_counters[level]

                new_num = list_counters[indent_level]
                if int(old_num) != new_num:
                    result.changes.append((i + 1, 'list', f'List item {old_num} → {new_num}'))

                new_line = f"{indent}{new_num}. {content}"
                fixed.append(new_line)
            else:
                # Reset list counters on non-list content
                if line.strip() and not line.strip().startswith('-'):
                    list_counters.clear()
                fixed.append(line)

        return fixed

    def _normalize_headings(self, lines: List[str], result: FixResult) -> List[str]:
        """Ensure consistent heading format."""
        fixed = []
        for i, line in enumerate(lines):
            match = self.HEADING_PATTERN.match(line)
            if match:
                hashes = match.group(1)
                title = match.group(2).strip()
                # Ensure single space after hashes
                new_line = f"{hashes} {title}"
                if new_line != line:
                    result.changes.append((i + 1, 'heading', f'Normalized heading format'))
                fixed.append(new_line)
            else:
                fixed.append(line)
        return fixed

    def _fix_blank_lines(self, content: str, result: FixResult) -> str:
        """Normalize multiple blank lines to maximum of two."""
        original = content
        content = self.MULTIPLE_BLANK_LINES_PATTERN.sub('\n\n', content)
        if content != original:
            result.changes.append((0, 'blank_lines', 'Normalized blank lines'))
        return content


def main():
    parser = argparse.ArgumentParser(
        description='Fix and normalize markdown documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fix_markdown.py input.md --all
    python fix_markdown.py input.md output.md --renumber-sections
    python fix_markdown.py input.md --fix-artifacts --dry-run
        """
    )

    parser.add_argument('input', type=Path, help='Input markdown file')
    parser.add_argument('output', type=Path, nargs='?', help='Output file (default: overwrite input)')

    parser.add_argument('--renumber-sections', action='store_true',
                       help='Renumber section headings sequentially')
    parser.add_argument('--renumber-lists', action='store_true',
                       help='Renumber ordered lists sequentially')
    parser.add_argument('--fix-artifacts', action='store_true',
                       help='Remove stray formatting artifacts (**, *, etc.)')
    parser.add_argument('--normalize-headings', action='store_true',
                       help='Normalize heading format')
    parser.add_argument('--fix-whitespace', action='store_true',
                       help='Remove trailing whitespace')
    parser.add_argument('--fix-blank-lines', action='store_true',
                       help='Normalize multiple blank lines')
    parser.add_argument('--all', action='store_true',
                       help='Apply all fixes')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show changes without writing')

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Build config
    config = FixerConfig(
        renumber_sections=args.renumber_sections,
        renumber_lists=args.renumber_lists,
        fix_artifacts=args.fix_artifacts,
        normalize_headings=args.normalize_headings,
        fix_whitespace=args.fix_whitespace,
        fix_blank_lines=args.fix_blank_lines,
        all_fixes=args.all,
        dry_run=args.dry_run
    )

    # Check if any fix is enabled
    any_fix = any([
        config.renumber_sections, config.renumber_lists, config.fix_artifacts,
        config.normalize_headings, config.fix_whitespace, config.fix_blank_lines,
        config.all_fixes
    ])

    if not any_fix:
        print("Warning: No fixes enabled. Use --all or specific --fix-* options.", file=sys.stderr)
        print("Run with --help for usage information.", file=sys.stderr)
        sys.exit(1)

    # Read input
    content = args.input.read_text(encoding='utf-8')

    # Apply fixes
    fixer = MarkdownFixer(config)
    fixed_content, result = fixer.fix(content)

    # Report changes
    print(f"Markdown Fixer Report")
    print(f"=" * 50)
    print(f"Input:  {args.input}")
    print(f"Lines:  {result.original_lines} → {result.fixed_lines}")
    print(f"Changes: {len(result.changes)}")
    print()

    if result.changes:
        print("Changes:")
        for line_num, change_type, description in result.changes[:50]:  # Limit output
            if line_num > 0:
                print(f"  Line {line_num:4d} [{change_type}]: {description}")
            else:
                print(f"  [global] [{change_type}]: {description}")

        if len(result.changes) > 50:
            print(f"  ... and {len(result.changes) - 50} more changes")
    else:
        print("No changes needed.")

    # Write output
    if not config.dry_run and result.changes:
        output_path = args.output or args.input
        output_path.write_text(fixed_content, encoding='utf-8')
        print(f"\nWritten to: {output_path}")
    elif config.dry_run:
        print(f"\n[DRY RUN] No files modified.")


if __name__ == '__main__':
    main()
