"""
WASP Markdown Parser - Markdown parsing and manipulation tools

Deterministic Markdown processing tools.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class NodeType(str, Enum):
    """Markdown AST node types."""
    DOCUMENT = "document"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    CODE_INLINE = "code_inline"
    BLOCKQUOTE = "blockquote"
    LIST = "list"
    LIST_ITEM = "list_item"
    LINK = "link"
    IMAGE = "image"
    EMPHASIS = "emphasis"
    STRONG = "strong"
    HORIZONTAL_RULE = "horizontal_rule"
    TABLE = "table"
    FRONTMATTER = "frontmatter"
    TEXT = "text"


@dataclass
class ASTNode:
    """Markdown AST node."""
    type: NodeType
    content: str = ""
    children: List["ASTNode"] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "type": self.type.value,
            "line": self.line,
        }
        if self.content:
            result["content"] = self.content
        if self.attrs:
            result["attrs"] = self.attrs
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


def parse_markdown(content: str) -> Dict[str, Any]:
    """
    Parse Markdown content to structured AST.

    Args:
        content: Markdown string to parse

    Returns:
        dict with keys:
            - success: bool
            - ast: AST structure as dict
            - stats: document statistics
    """
    result = {"success": True}

    lines = content.splitlines()
    root = ASTNode(type=NodeType.DOCUMENT)

    i = 0
    while i < len(lines):
        line = lines[i]
        line_num = i + 1

        # Frontmatter (YAML between ---)
        if i == 0 and line.strip() == "---":
            end_idx = _find_frontmatter_end(lines, i + 1)
            if end_idx > 0:
                fm_content = "\n".join(lines[i + 1:end_idx])
                node = ASTNode(
                    type=NodeType.FRONTMATTER,
                    content=fm_content,
                    line=line_num,
                )
                root.children.append(node)
                i = end_idx + 1
                continue

        # Heading
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            node = ASTNode(
                type=NodeType.HEADING,
                content=text,
                attrs={"level": level},
                line=line_num,
            )
            root.children.append(node)
            i += 1
            continue

        # Code block (fenced)
        code_match = re.match(r'^```(\w*)$', line)
        if code_match:
            lang = code_match.group(1) or None
            end_idx, code_content = _find_code_block_end(lines, i + 1)
            node = ASTNode(
                type=NodeType.CODE_BLOCK,
                content=code_content,
                attrs={"language": lang} if lang else {},
                line=line_num,
            )
            root.children.append(node)
            i = end_idx + 1
            continue

        # Blockquote
        if line.startswith(">"):
            quote_lines, end_idx = _collect_blockquote(lines, i)
            node = ASTNode(
                type=NodeType.BLOCKQUOTE,
                content="\n".join(quote_lines),
                line=line_num,
            )
            root.children.append(node)
            i = end_idx
            continue

        # Horizontal rule
        if re.match(r'^[-*_]{3,}\s*$', line):
            node = ASTNode(type=NodeType.HORIZONTAL_RULE, line=line_num)
            root.children.append(node)
            i += 1
            continue

        # List (unordered or ordered)
        list_match = re.match(r'^(\s*)([*\-+]|\d+\.)\s+(.*)$', line)
        if list_match:
            list_node, end_idx = _parse_list(lines, i)
            root.children.append(list_node)
            i = end_idx
            continue

        # Table
        if "|" in line and i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i + 1]):
            table_node, end_idx = _parse_table(lines, i)
            if table_node:
                root.children.append(table_node)
                i = end_idx
                continue

        # Paragraph (default)
        if line.strip():
            para_lines, end_idx = _collect_paragraph(lines, i)
            node = ASTNode(
                type=NodeType.PARAGRAPH,
                content="\n".join(para_lines),
                line=line_num,
            )
            root.children.append(node)
            i = end_idx
            continue

        i += 1

    # Compute stats
    result["ast"] = root.to_dict()
    result["stats"] = _compute_stats(root)

    return result


def extract_headers(content: str) -> Dict[str, Any]:
    """
    Extract all headers from Markdown content.

    Args:
        content: Markdown string

    Returns:
        dict with keys:
            - success: bool
            - headers: list of header objects with level, text, line
            - count: total header count
    """
    result = {"success": True, "headers": []}

    for i, line in enumerate(content.splitlines(), 1):
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            result["headers"].append({
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
                "line": i,
            })

    result["count"] = len(result["headers"])
    return result


def extract_code_blocks(content: str) -> Dict[str, Any]:
    """
    Extract all code blocks from Markdown content.

    Args:
        content: Markdown string

    Returns:
        dict with keys:
            - success: bool
            - blocks: list of code block objects with language, content, line
            - count: total code block count
    """
    result = {"success": True, "blocks": []}

    lines = content.splitlines()
    i = 0

    while i < len(lines):
        match = re.match(r'^```(\w*)$', lines[i])
        if match:
            lang = match.group(1) or None
            start_line = i + 1
            end_idx, code_content = _find_code_block_end(lines, i + 1)

            result["blocks"].append({
                "language": lang,
                "content": code_content,
                "line": start_line,
                "line_count": end_idx - i - 1,
            })

            i = end_idx + 1
        else:
            i += 1

    result["count"] = len(result["blocks"])
    return result


def extract_links(content: str) -> Dict[str, Any]:
    """
    Extract all links from Markdown content.

    Args:
        content: Markdown string

    Returns:
        dict with keys:
            - success: bool
            - links: list of link objects with text, url, title, line, type
            - count: total link count
    """
    result = {"success": True, "links": []}

    # Inline links: [text](url "title")
    inline_pattern = r'\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)'

    # Reference links: [text][ref] and [ref]: url "title"
    ref_def_pattern = r'^\[([^\]]+)\]:\s+(\S+)(?:\s+"([^"]*)")?$'

    for i, line in enumerate(content.splitlines(), 1):
        # Inline links
        for match in re.finditer(inline_pattern, line):
            result["links"].append({
                "text": match.group(1),
                "url": match.group(2),
                "title": match.group(3),
                "line": i,
                "type": "inline",
            })

        # Reference definitions
        ref_match = re.match(ref_def_pattern, line)
        if ref_match:
            result["links"].append({
                "text": ref_match.group(1),
                "url": ref_match.group(2),
                "title": ref_match.group(3),
                "line": i,
                "type": "reference",
            })

    # Image links: ![alt](url)
    img_pattern = r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)'
    for i, line in enumerate(content.splitlines(), 1):
        for match in re.finditer(img_pattern, line):
            result["links"].append({
                "text": match.group(1),
                "url": match.group(2),
                "title": match.group(3),
                "line": i,
                "type": "image",
            })

    result["count"] = len(result["links"])
    return result


def extract_frontmatter(content: str) -> Dict[str, Any]:
    """
    Extract YAML frontmatter from Markdown content.

    Args:
        content: Markdown string

    Returns:
        dict with keys:
            - success: bool
            - has_frontmatter: bool
            - frontmatter: parsed frontmatter dict (if valid YAML)
            - raw: raw frontmatter string
    """
    result = {"success": True, "has_frontmatter": False}

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return result

    end_idx = _find_frontmatter_end(lines, 1)
    if end_idx < 0:
        return result

    result["has_frontmatter"] = True
    result["raw"] = "\n".join(lines[1:end_idx])

    # Try to parse as YAML
    try:
        import yaml
        result["frontmatter"] = yaml.safe_load(result["raw"])
    except Exception:
        result["frontmatter"] = None
        result["parse_error"] = "Failed to parse YAML"

    return result


def renumber_sections(
    content: str,
    start_level: int = 1,
    number_format: str = "{num}. ",
) -> Dict[str, Any]:
    """
    Renumber Markdown section headers.

    Args:
        content: Markdown string
        start_level: Minimum header level to number (default: 1)
        number_format: Format string for numbers (default: "{num}. ")

    Returns:
        dict with keys:
            - success: bool
            - content: renumbered content
            - changes: number of headers modified
    """
    result = {"success": True, "changes": 0}

    lines = content.splitlines()
    counters = [0] * 7  # Counters for levels 1-6

    output_lines = []
    for line in lines:
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()

            # Remove existing numbering
            text = re.sub(r'^\d+(\.\d+)*\.?\s*', '', text)

            if level >= start_level:
                # Increment counter for this level
                counters[level] += 1

                # Reset lower level counters
                for i in range(level + 1, 7):
                    counters[i] = 0

                # Build section number
                num_parts = [str(counters[i]) for i in range(start_level, level + 1)]
                section_num = ".".join(num_parts)

                # Format new header
                prefix = "#" * level
                new_text = number_format.format(num=section_num) + text
                output_lines.append(f"{prefix} {new_text}")
                result["changes"] += 1
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)

    result["content"] = "\n".join(output_lines)
    return result


def generate_toc(
    content: str,
    max_level: int = 3,
    bullet: str = "-",
) -> Dict[str, Any]:
    """
    Generate table of contents from Markdown headers.

    Args:
        content: Markdown string
        max_level: Maximum header level to include (default: 3)
        bullet: Bullet character for list items

    Returns:
        dict with keys:
            - success: bool
            - toc: generated TOC as Markdown string
            - entries: list of TOC entries
    """
    result = {"success": True, "entries": []}

    headers = extract_headers(content)["headers"]

    toc_lines = []
    for header in headers:
        if header["level"] > max_level:
            continue

        # Create anchor (simplified slug)
        anchor = _slugify(header["text"])
        indent = "  " * (header["level"] - 1)
        toc_lines.append(f"{indent}{bullet} [{header['text']}](#{anchor})")

        result["entries"].append({
            "text": header["text"],
            "level": header["level"],
            "anchor": anchor,
        })

    result["toc"] = "\n".join(toc_lines)
    return result


# Helper functions

def _find_frontmatter_end(lines: List[str], start: int) -> int:
    """Find the closing --- of frontmatter."""
    for i in range(start, len(lines)):
        if lines[i].strip() == "---":
            return i
    return -1


def _find_code_block_end(lines: List[str], start: int) -> Tuple[int, str]:
    """Find the closing ``` of a code block."""
    code_lines = []
    for i in range(start, len(lines)):
        if lines[i].startswith("```"):
            return i, "\n".join(code_lines)
        code_lines.append(lines[i])
    return len(lines), "\n".join(code_lines)


def _collect_blockquote(lines: List[str], start: int) -> Tuple[List[str], int]:
    """Collect consecutive blockquote lines."""
    quote_lines = []
    i = start
    while i < len(lines) and lines[i].startswith(">"):
        # Remove > prefix
        quote_lines.append(re.sub(r'^>\s?', '', lines[i]))
        i += 1
    return quote_lines, i


def _collect_paragraph(lines: List[str], start: int) -> Tuple[List[str], int]:
    """Collect consecutive non-empty paragraph lines."""
    para_lines = []
    i = start
    while i < len(lines):
        line = lines[i]
        # Stop at empty line or block element
        if not line.strip():
            break
        if re.match(r'^(#{1,6}\s|```|>|\s*[-*+]\s|\s*\d+\.\s|[-*_]{3,})', line):
            break
        para_lines.append(line)
        i += 1
    return para_lines, i


def _parse_list(lines: List[str], start: int) -> Tuple[ASTNode, int]:
    """Parse a list block."""
    items = []
    i = start
    line_num = start + 1

    # Detect list type
    first_match = re.match(r'^(\s*)([*\-+]|\d+\.)\s+(.*)$', lines[start])
    is_ordered = bool(re.match(r'\d+\.', first_match.group(2)))

    while i < len(lines):
        match = re.match(r'^(\s*)([*\-+]|\d+\.)\s+(.*)$', lines[i])
        if not match:
            # Check for continuation or nested content
            if lines[i].startswith("  ") or lines[i].startswith("\t"):
                if items:
                    items[-1]["content"] += "\n" + lines[i].strip()
                i += 1
                continue
            break

        items.append({
            "content": match.group(3),
            "indent": len(match.group(1)),
            "line": i + 1,
        })
        i += 1

    list_node = ASTNode(
        type=NodeType.LIST,
        attrs={"ordered": is_ordered},
        line=line_num,
    )

    for item in items:
        item_node = ASTNode(
            type=NodeType.LIST_ITEM,
            content=item["content"],
            attrs={"indent": item["indent"]},
            line=item["line"],
        )
        list_node.children.append(item_node)

    return list_node, i


def _parse_table(lines: List[str], start: int) -> Tuple[Optional[ASTNode], int]:
    """Parse a table block."""
    if start + 1 >= len(lines):
        return None, start + 1

    # Parse header row
    header_line = lines[start]
    separator_line = lines[start + 1]

    if not re.match(r'^[\s|:-]+$', separator_line):
        return None, start + 1

    # Extract columns
    headers = [c.strip() for c in header_line.strip('|').split('|')]

    # Parse data rows
    rows = []
    i = start + 2
    while i < len(lines) and '|' in lines[i]:
        cells = [c.strip() for c in lines[i].strip('|').split('|')]
        rows.append(cells)
        i += 1

    table_node = ASTNode(
        type=NodeType.TABLE,
        attrs={
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers),
        },
        line=start + 1,
    )

    return table_node, i


def _compute_stats(root: ASTNode) -> Dict[str, Any]:
    """Compute document statistics from AST."""
    stats = {
        "heading_count": 0,
        "paragraph_count": 0,
        "code_block_count": 0,
        "list_count": 0,
        "link_count": 0,
        "table_count": 0,
        "has_frontmatter": False,
    }

    def visit(node: ASTNode):
        if node.type == NodeType.HEADING:
            stats["heading_count"] += 1
        elif node.type == NodeType.PARAGRAPH:
            stats["paragraph_count"] += 1
        elif node.type == NodeType.CODE_BLOCK:
            stats["code_block_count"] += 1
        elif node.type == NodeType.LIST:
            stats["list_count"] += 1
        elif node.type == NodeType.TABLE:
            stats["table_count"] += 1
        elif node.type == NodeType.FRONTMATTER:
            stats["has_frontmatter"] = True

        for child in node.children:
            visit(child)

    visit(root)
    return stats


def _slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    # Lowercase
    slug = text.lower()
    # Remove special characters
    slug = re.sub(r'[^\w\s-]', '', slug)
    # Replace spaces with hyphens
    slug = re.sub(r'[\s_]+', '-', slug)
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    return slug
