"""
Retrieval Abstraction for RAGIX

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from typing import Protocol, List, Dict, Any
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """
    Single retrieval result (file, line, match).

    Attributes:
        file_path: Path to the file containing the match
        line_number: Line number (1-indexed)
        line_content: Content of the matching line
        score: Optional relevance score (for ranking)
    """
    file_path: str
    line_number: int
    line_content: str
    score: float = 1.0


class Retriever(Protocol):
    """
    Protocol for retrieval implementations.

    RAGIX uses Unix-RAG pattern: grep-based retrieval with minimal overhead.
    This protocol allows plugging different retrieval backends while maintaining
    the same interface.
    """

    def search(self, query: str, max_results: int = 50) -> List[RetrievalResult]:
        """
        Search for query in the project.

        Args:
            query: Search query (pattern/keyword)
            max_results: Maximum number of results to return

        Returns:
            List of RetrievalResult objects
        """
        ...


class GrepRetriever:
    """
    Grep-based retriever using Unix tools via ShellSandbox.

    This is the default RAGIX retrieval method:
    - Uses `grep -R -n` for recursive search
    - Returns file:line:content format
    - Fast and transparent
    """

    def __init__(self, shell_sandbox):
        """
        Initialize grep retriever.

        Args:
            shell_sandbox: ShellSandbox instance for command execution
        """
        self.shell = shell_sandbox

    def search(self, query: str, max_results: int = 50) -> List[RetrievalResult]:
        """
        Search using grep -R -n.

        Args:
            query: Pattern to search for
            max_results: Maximum results to return

        Returns:
            List of RetrievalResult objects
        """
        # Escape quotes in query
        escaped_query = query.replace('"', '\\"')

        # Use grep to search recursively
        cmd = f'grep -R -n "{escaped_query}" . 2>/dev/null | head -n {max_results}'
        result = self.shell.run(cmd)

        if result.returncode != 0 or not result.stdout.strip():
            return []

        # Parse grep output: file:line:content
        results = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(':', 2)
            if len(parts) >= 3:
                file_path = parts[0]
                try:
                    line_number = int(parts[1])
                except ValueError:
                    continue
                line_content = parts[2] if len(parts) > 2 else ""

                results.append(RetrievalResult(
                    file_path=file_path,
                    line_number=line_number,
                    line_content=line_content,
                    score=1.0
                ))

        return results


def format_retrieval_results(results: List[RetrievalResult], max_display: int = 20) -> str:
    """
    Format retrieval results for display to the agent.

    Args:
        results: List of RetrievalResult objects
        max_display: Maximum results to format

    Returns:
        Formatted string suitable for feeding to LLM
    """
    if not results:
        return "No results found."

    lines = [f"Found {len(results)} results:\n"]
    for i, res in enumerate(results[:max_display], 1):
        lines.append(f"{i}. {res.file_path}:{res.line_number}")
        lines.append(f"   {res.line_content.strip()}")
        lines.append("")

    if len(results) > max_display:
        lines.append(f"... and {len(results) - max_display} more results")

    return "\n".join(lines)
