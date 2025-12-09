"""
RAGIX Project RAG - Profile-based Chunking

Provides chunking strategies based on indexing profiles:
    - docs_only: Sentence/paragraph-based for long-form text
    - mixed_docs_code: Hybrid approach for tech docs + code
    - code_only: Line-based chunking preserving code structure

Key features:
    - Preserves line numbers for code → doc pairing
    - Configurable chunk size and overlap
    - Language-aware code chunking

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from .config import IndexingProfile, PROFILE_MIXED

logger = logging.getLogger(__name__)

# =============================================================================
# Chunk Data Structure
# =============================================================================

@dataclass
class Chunk:
    """
    A text chunk with position metadata.

    Preserves line numbers for citations and future AST integration.
    """
    content: str
    chunk_index: int
    offset_start: int           # Character offset in original text
    offset_end: int
    line_start: int             # 1-indexed line number
    line_end: int
    text_preview: str = ""      # First ~200 chars for UI

    def __post_init__(self):
        if not self.text_preview:
            self.text_preview = self.content[:200].strip()
            if len(self.content) > 200:
                self.text_preview += "..."


# =============================================================================
# Chunker Class
# =============================================================================

class Chunker:
    """
    Profile-based text chunker.

    Supports different strategies based on content type:
    - sentence+paragraph: For documentation
    - line+paragraph: For mixed content
    - line-based: For source code
    """

    def __init__(self, profile: Optional[IndexingProfile] = None):
        """
        Initialize chunker with a profile.

        Args:
            profile: Indexing profile (defaults to mixed_docs_code)
        """
        self.profile = profile or PROFILE_MIXED
        self.chunk_size = self.profile.chunk_size
        self.chunk_overlap = self.profile.chunk_overlap
        self.level = self.profile.level
        self.code_mode = self.profile.code_mode

    def chunk(self, text: str, is_code: bool = False) -> List[Chunk]:
        """
        Split text into chunks based on profile settings.

        Args:
            text: Text content to chunk
            is_code: Whether this is source code

        Returns:
            List of Chunk objects with position metadata
        """
        if not text or not text.strip():
            return []

        # Choose strategy based on profile and content
        if is_code or (self.code_mode and self._looks_like_code(text)):
            return self._chunk_code(text)
        elif self.level == "sentence+paragraph":
            return self._chunk_sentences(text)
        else:
            return self._chunk_paragraphs(text)

    def _chunk_code(self, text: str) -> List[Chunk]:
        """
        Chunk source code preserving line boundaries.

        Strategy: Split by logical blocks (functions, classes) when possible,
        fall back to fixed line counts.
        """
        lines = text.split('\n')
        chunks = []
        current_lines = []
        current_start_line = 1
        current_start_offset = 0
        current_offset = 0
        chunk_index = 0

        # Target lines per chunk (approximate based on avg chars per line)
        avg_line_length = len(text) / max(len(lines), 1)
        target_lines = max(10, int(self.chunk_size / max(avg_line_length, 1)))

        for i, line in enumerate(lines):
            line_num = i + 1
            current_lines.append(line)
            current_offset += len(line) + 1  # +1 for newline

            # Check if we should create a chunk
            should_split = False

            if len(current_lines) >= target_lines:
                # Look for natural break points
                if self._is_block_boundary(line, lines[i + 1] if i + 1 < len(lines) else ""):
                    should_split = True
                elif len(current_lines) >= target_lines * 1.5:
                    # Force split if too long
                    should_split = True

            if should_split and current_lines:
                content = '\n'.join(current_lines)
                chunks.append(Chunk(
                    content=content,
                    chunk_index=chunk_index,
                    offset_start=current_start_offset,
                    offset_end=current_offset,
                    line_start=current_start_line,
                    line_end=line_num,
                ))
                chunk_index += 1

                # Handle overlap (keep last N lines)
                overlap_lines = max(2, int(self.chunk_overlap / max(avg_line_length, 1)))
                if overlap_lines < len(current_lines):
                    current_lines = current_lines[-overlap_lines:]
                    current_start_line = line_num - overlap_lines + 1
                    current_start_offset = current_offset - sum(len(l) + 1 for l in current_lines)
                else:
                    current_lines = []
                    current_start_line = line_num + 1
                    current_start_offset = current_offset

        # Final chunk
        if current_lines:
            content = '\n'.join(current_lines)
            chunks.append(Chunk(
                content=content,
                chunk_index=chunk_index,
                offset_start=current_start_offset,
                offset_end=len(text),
                line_start=current_start_line,
                line_end=len(lines),
            ))

        return chunks

    def _chunk_paragraphs(self, text: str) -> List[Chunk]:
        """
        Chunk text by paragraphs with overlap.

        Used for mixed docs + code content.
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_content = ""
        current_start_offset = 0
        current_start_line = 1
        chunk_index = 0

        # Track line numbers
        lines_before = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_lines = para.count('\n') + 1

            # Check if adding this paragraph exceeds chunk size
            if current_content and len(current_content) + len(para) > self.chunk_size:
                # Save current chunk
                content_lines = current_content.count('\n') + 1
                chunks.append(Chunk(
                    content=current_content.strip(),
                    chunk_index=chunk_index,
                    offset_start=current_start_offset,
                    offset_end=current_start_offset + len(current_content),
                    line_start=current_start_line,
                    line_end=current_start_line + content_lines - 1,
                ))
                chunk_index += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_content) > self.chunk_overlap:
                    overlap_text = current_content[-self.chunk_overlap:]
                    current_content = overlap_text + "\n\n" + para
                    current_start_offset = current_start_offset + len(current_content) - len(overlap_text) - len(para) - 2
                else:
                    current_content = para
                    current_start_offset += len(current_content) + 2

                current_start_line = lines_before + 1
            else:
                if current_content:
                    current_content += "\n\n" + para
                else:
                    current_content = para

            lines_before += para_lines + 1  # +1 for paragraph separator

        # Final chunk
        if current_content.strip():
            content_lines = current_content.count('\n') + 1
            chunks.append(Chunk(
                content=current_content.strip(),
                chunk_index=chunk_index,
                offset_start=current_start_offset,
                offset_end=len(text),
                line_start=current_start_line,
                line_end=current_start_line + content_lines - 1,
            ))

        return chunks

    def _chunk_sentences(self, text: str) -> List[Chunk]:
        """
        Chunk text by sentences, respecting paragraph boundaries.

        Used for documentation with long sentences.
        """
        # Split into sentences (basic approach)
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)

        chunks = []
        current_content = ""
        current_start_offset = 0
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_content and len(current_content) + len(sentence) > self.chunk_size:
                # Save current chunk
                chunks.append(Chunk(
                    content=current_content.strip(),
                    chunk_index=chunk_index,
                    offset_start=current_start_offset,
                    offset_end=current_start_offset + len(current_content),
                    line_start=self._count_lines_before(text, current_start_offset) + 1,
                    line_end=self._count_lines_before(text, current_start_offset + len(current_content)) + 1,
                ))
                chunk_index += 1

                # Overlap
                if self.chunk_overlap > 0:
                    overlap_text = current_content[-self.chunk_overlap:]
                    current_content = overlap_text + " " + sentence
                else:
                    current_content = sentence
                    current_start_offset += len(current_content) + 1
            else:
                if current_content:
                    current_content += " " + sentence
                else:
                    current_content = sentence

        # Final chunk
        if current_content.strip():
            chunks.append(Chunk(
                content=current_content.strip(),
                chunk_index=chunk_index,
                offset_start=current_start_offset,
                offset_end=len(text),
                line_start=self._count_lines_before(text, current_start_offset) + 1,
                line_end=text.count('\n') + 1,
            ))

        return chunks

    def _is_block_boundary(self, current_line: str, next_line: str) -> bool:
        """Check if we're at a natural code block boundary."""
        # Empty line often indicates block boundary
        if not current_line.strip():
            return True

        # End of function/class/block patterns
        block_end_patterns = [
            r'^\s*}\s*$',           # Closing brace
            r'^\s*end\s*$',         # Ruby/Lua end
            r'^\s*pass\s*$',        # Python pass
            r'^\s*return\b',        # Return statement
        ]

        for pattern in block_end_patterns:
            if re.match(pattern, current_line, re.IGNORECASE):
                return True

        # Start of new block in next line
        block_start_patterns = [
            r'^\s*(public|private|protected|static|def|class|interface|enum|func|function|async)\b',
            r'^\s*@\w+',            # Decorator/annotation
            r'^#.*',                # Comment/preprocessor at start
        ]

        for pattern in block_start_patterns:
            if re.match(pattern, next_line):
                return True

        return False

    def _looks_like_code(self, text: str) -> bool:
        """Heuristic to detect if text is source code."""
        # Check for common code patterns
        code_indicators = [
            r'^\s*(import|from|package|using|include)\b',
            r'^\s*(def|func|function|class|interface|struct|enum)\b',
            r'^\s*(public|private|protected|static)\b',
            r'[{}\[\]();]',
            r'^\s*#include\b',
            r'^\s*@\w+\s*$',  # Decorators/annotations
        ]

        lines = text.split('\n')[:50]  # Check first 50 lines
        code_line_count = 0

        for line in lines:
            for pattern in code_indicators:
                if re.search(pattern, line):
                    code_line_count += 1
                    break

        # If >30% of lines look like code, treat as code
        return code_line_count / max(len(lines), 1) > 0.3

    def _count_lines_before(self, text: str, offset: int) -> int:
        """Count number of newlines before a given offset."""
        return text[:offset].count('\n')


# =============================================================================
# Convenience Functions
# =============================================================================

def chunk_text(
    text: str,
    profile: Optional[IndexingProfile] = None,
    is_code: bool = False,
) -> List[Chunk]:
    """
    Convenience function to chunk text.

    Args:
        text: Text to chunk
        profile: Indexing profile (defaults to mixed)
        is_code: Whether this is source code

    Returns:
        List of Chunk objects
    """
    chunker = Chunker(profile)
    return chunker.chunk(text, is_code=is_code)


def chunk_code(
    code: str,
    profile: Optional[IndexingProfile] = None,
) -> List[Chunk]:
    """
    Convenience function to chunk source code.

    Args:
        code: Source code to chunk
        profile: Indexing profile

    Returns:
        List of Chunk objects with line numbers
    """
    chunker = Chunker(profile)
    return chunker.chunk(code, is_code=True)


def estimate_chunk_count(text: str, chunk_size: int = 512) -> int:
    """
    Estimate number of chunks for a text.

    Useful for progress estimation before actual chunking.
    """
    if not text:
        return 0
    return max(1, len(text) // chunk_size)
