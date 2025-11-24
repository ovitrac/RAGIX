"""
Code Chunking for Semantic Indexing

Parses source code into semantic chunks (functions, classes, docstrings).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ChunkType(str, Enum):
    """Type of code chunk."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    DOCSTRING = "docstring"
    COMMENT = "comment"
    MODULE = "module"
    MARKDOWN_SECTION = "markdown_section"
    UNKNOWN = "unknown"


@dataclass
class CodeChunk:
    """
    A semantic chunk of code.

    Attributes:
        file_path: Path to source file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        chunk_type: Type of chunk
        name: Name (function/class name, or generated)
        content: Text content of chunk
        language: Programming language
        metadata: Additional metadata
    """
    file_path: str
    start_line: int
    end_line: int
    chunk_type: ChunkType
    name: str
    content: str
    language: str
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type.value,
            "name": self.name,
            "content": self.content,
            "language": self.language,
            "metadata": self.metadata
        }


class PythonChunker:
    """Chunk Python code using AST parsing."""

    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Chunk a Python file into semantic units.

        Args:
            file_path: Path to Python file

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))
        except Exception as e:
            # If parsing fails, return file as single chunk
            return [self._create_fallback_chunk(file_path, str(e))]

        # Extract chunks from AST
        for node in ast.walk(tree):
            chunk = self._extract_chunk(node, file_path, content)
            if chunk:
                chunks.append(chunk)

        # Add module-level docstring if present
        docstring = ast.get_docstring(tree)
        if docstring:
            chunks.insert(0, CodeChunk(
                file_path=str(file_path),
                start_line=1,
                end_line=docstring.count('\n') + 3,  # Approximate
                chunk_type=ChunkType.DOCSTRING,
                name=f"{file_path.stem}_module_docstring",
                content=docstring,
                language="python",
                metadata={"scope": "module"}
            ))

        return chunks

    def _extract_chunk(
        self,
        node: ast.AST,
        file_path: Path,
        content: str
    ) -> Optional[CodeChunk]:
        """Extract chunk from AST node."""

        if isinstance(node, ast.FunctionDef):
            return self._extract_function(node, file_path, content)
        elif isinstance(node, ast.ClassDef):
            return self._extract_class(node, file_path, content)

        return None

    def _extract_function(
        self,
        node: ast.FunctionDef,
        file_path: Path,
        content: str
    ) -> CodeChunk:
        """Extract function chunk."""
        lines = content.split('\n')
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get function text
        func_lines = lines[start_line - 1:end_line]
        func_text = '\n'.join(func_lines)

        # Determine if it's a method (inside a class)
        chunk_type = ChunkType.METHOD if self._is_method(node) else ChunkType.FUNCTION

        # Extract docstring
        docstring = ast.get_docstring(node)

        return CodeChunk(
            file_path=str(file_path),
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            name=node.name,
            content=func_text,
            language="python",
            metadata={
                "docstring": docstring,
                "args": [arg.arg for arg in node.args.args],
                "decorator_list": [self._get_decorator_name(d) for d in node.decorator_list]
            }
        )

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        content: str
    ) -> CodeChunk:
        """Extract class chunk."""
        lines = content.split('\n')
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get class text
        class_lines = lines[start_line - 1:end_line]
        class_text = '\n'.join(class_lines)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Get base classes
        bases = [self._get_name(base) for base in node.bases]

        return CodeChunk(
            file_path=str(file_path),
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType.CLASS,
            name=node.name,
            content=class_text,
            language="python",
            metadata={
                "docstring": docstring,
                "bases": bases,
                "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            }
        )

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if function is a method (has 'self' or 'cls' as first arg)."""
        if not node.args.args:
            return False
        first_arg = node.args.args[0].arg
        return first_arg in ('self', 'cls')

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        return str(decorator)

    def _get_name(self, node: ast.expr) -> str:
        """Get name from expression node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)

    def _create_fallback_chunk(self, file_path: Path, error: str) -> CodeChunk:
        """Create fallback chunk when parsing fails."""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.count('\n') + 1
        except Exception:
            content = ""
            lines = 0

        return CodeChunk(
            file_path=str(file_path),
            start_line=1,
            end_line=lines,
            chunk_type=ChunkType.UNKNOWN,
            name=file_path.name,
            content=content,
            language="python",
            metadata={"parse_error": error}
        )


class MarkdownChunker:
    """Chunk Markdown files by sections."""

    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Chunk Markdown file by headers.

        Args:
            file_path: Path to Markdown file

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception:
            return []

        lines = content.split('\n')
        current_section = []
        current_header = None
        start_line = 1

        for i, line in enumerate(lines, 1):
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save previous section
                if current_section:
                    chunks.append(self._create_section_chunk(
                        file_path,
                        start_line,
                        i - 1,
                        current_header or "Untitled",
                        '\n'.join(current_section)
                    ))

                # Start new section
                current_header = header_match.group(2)
                current_section = [line]
                start_line = i
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            chunks.append(self._create_section_chunk(
                file_path,
                start_line,
                len(lines),
                current_header or "Untitled",
                '\n'.join(current_section)
            ))

        return chunks

    def _create_section_chunk(
        self,
        file_path: Path,
        start_line: int,
        end_line: int,
        header: str,
        content: str
    ) -> CodeChunk:
        """Create chunk for Markdown section."""
        return CodeChunk(
            file_path=str(file_path),
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType.MARKDOWN_SECTION,
            name=header,
            content=content,
            language="markdown",
            metadata={"header": header}
        )


class GenericChunker:
    """Generic chunker for unsupported file types."""

    def __init__(self, max_chunk_lines: int = 100):
        self.max_chunk_lines = max_chunk_lines

    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Chunk file into fixed-size chunks.

        Args:
            file_path: Path to file

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception:
            return []

        lines = content.split('\n')
        total_lines = len(lines)

        for start in range(0, total_lines, self.max_chunk_lines):
            end = min(start + self.max_chunk_lines, total_lines)
            chunk_lines = lines[start:end]
            chunk_content = '\n'.join(chunk_lines)

            chunks.append(CodeChunk(
                file_path=str(file_path),
                start_line=start + 1,
                end_line=end,
                chunk_type=ChunkType.UNKNOWN,
                name=f"{file_path.name}_chunk_{start // self.max_chunk_lines}",
                content=chunk_content,
                language=file_path.suffix.lstrip('.') or "text",
                metadata={"chunk_index": start // self.max_chunk_lines}
            ))

        return chunks


def chunk_file(file_path: Path, language: Optional[str] = None) -> List[CodeChunk]:
    """
    Chunk a file based on its language.

    Args:
        file_path: Path to file
        language: Optional language override

    Returns:
        List of CodeChunk objects
    """
    if language is None:
        language = file_path.suffix.lstrip('.')

    if language == 'py':
        chunker = PythonChunker()
    elif language == 'md':
        chunker = MarkdownChunker()
    else:
        chunker = GenericChunker()

    return chunker.chunk_file(file_path)
