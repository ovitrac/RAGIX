"""
Kernel: AST Scan
Stage: 1 (Data Collection)
Wraps: ragix_core.dependencies.DependencyGraph

Extracts Abstract Syntax Tree data from source code:
- Parses all source files in a project directory
- Extracts symbols (classes, methods, functions)
- Identifies dependencies between symbols
- Computes basic statistics

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class ASTScanKernel(Kernel):
    """
    Extract AST data from source code.

    This kernel scans a project directory, parses source files,
    and extracts structural information (symbols, dependencies).

    Configuration options:
        project.path: Path to project directory (required)
        project.language: Language to scan (default: "java")
        project.patterns: Custom glob patterns (optional)
        include_tests: Include test directories (default: false)
        max_depth: Maximum directory depth (default: 10)
        parse_javadoc: Parse Javadoc comments (default: true)

    Output:
        files: List of parsed files with metadata
        symbols: List of extracted symbols
        dependencies: List of dependencies
        statistics: Computed statistics
    """

    name = "ast_scan"
    version = "1.1.0"
    category = "audit"
    stage = 1
    description = "Extract AST data from source code"

    requires = []  # No dependencies — first kernel
    provides = ["files", "symbols", "dependencies"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Run AST scan and extract structural data."""

        # Import here to avoid circular imports
        from ragix_core.dependencies import DependencyGraph

        # Extract configuration
        project_config = input.config.get("project", {})
        project_path = Path(project_config.get("path", "."))
        language = project_config.get("language", "java")
        custom_patterns = project_config.get("patterns", None)

        # Scan options
        include_tests = input.config.get("include_tests", False)
        recursive = input.config.get("recursive", True)

        # Determine file patterns based on language
        if custom_patterns:
            patterns = custom_patterns
        else:
            lang_patterns = {
                "java": ["*.java"],
                "python": ["*.py"],
                "py": ["*.py"],
                "javascript": ["*.js"],
                "js": ["*.js"],
                "typescript": ["*.ts"],
                "ts": ["*.ts"],
            }
            patterns = lang_patterns.get(language.lower(), ["*.java", "*.py"])

        if not project_path.exists():
            raise RuntimeError(f"Project path does not exist: {project_path}")

        logger.info(f"[ast_scan] Scanning {project_path} for {patterns}")

        # Build dependency graph
        graph = DependencyGraph()
        stats = graph.add_directory(project_path, patterns, recursive=recursive)

        # Get all symbols
        symbols = graph.get_symbols()
        dependencies = graph.get_all_dependencies()

        # Filter test files if requested
        if not include_tests:
            # Only filter if path contains /test/, /tests/, etc. as directory names
            test_patterns = ["/test/", "/tests/", "/Test/", "/Tests/", "/__test__/", "/__tests__/"]
            symbols = [
                s for s in symbols
                if not any(p in str(s.location.file) for p in test_patterns)
            ]

        # Build file list with basic info
        files_data = self._build_files_data(symbols)

        # Build symbols data
        symbols_data = self._build_symbols_data(symbols)

        # Enrich symbols with annotations and CC from ASTNode metadata
        self._enrich_symbols_from_ast(graph, symbols_data)

        # Build dependencies data
        deps_data = self._build_dependencies_data(dependencies, symbols_data)

        # Compute statistics
        statistics = {
            "total_files": len(files_data),
            "total_symbols": len(symbols_data),
            "total_dependencies": len(deps_data),
            "parse_success": stats.get("parsed", 0),
            "parse_failed": stats.get("failed", 0),
            "by_type": self._count_by_type(symbols_data),
            "by_visibility": self._count_by_visibility(symbols_data),
        }

        return {
            "project": {
                "path": str(project_path.resolve()),
                "language": language,
                "patterns": patterns,
            },
            "files": files_data,
            "symbols": symbols_data,
            "dependencies": deps_data,
            "statistics": statistics,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate LLM-consumable summary."""
        stats = data.get("statistics", {})
        project = data.get("project", {})

        by_type = stats.get("by_type", {})
        classes = by_type.get("class", 0)
        methods = by_type.get("method", 0)
        functions = by_type.get("function", 0)

        return (
            f"AST scan complete for {project.get('language', 'unknown')} project. "
            f"Analyzed {stats.get('total_files', 0)} files: "
            f"{classes} classes, {methods} methods, {functions} functions. "
            f"Total symbols: {stats.get('total_symbols', 0)}, "
            f"Dependencies: {stats.get('total_dependencies', 0)}. "
            f"Parse errors: {stats.get('parse_failed', 0)}."
        )

    def _build_files_data(self, symbols: List) -> List[Dict[str, Any]]:
        """Build file list with metadata from symbols."""
        files: Dict[str, Dict[str, Any]] = {}

        for sym in symbols:
            if not sym.location or not sym.location.file:
                continue

            file_path = str(sym.location.file)
            if file_path not in files:
                files[file_path] = {
                    "path": file_path,
                    "symbols": 0,
                    "classes": 0,
                    "methods": 0,
                    "functions": 0,
                }

            files[file_path]["symbols"] += 1
            if sym.node_type.value == "class":
                files[file_path]["classes"] += 1
            elif sym.node_type.value == "method":
                files[file_path]["methods"] += 1
            elif sym.node_type.value == "function":
                files[file_path]["functions"] += 1

        return list(files.values())

    def _build_symbols_data(self, symbols: List) -> List[Dict[str, Any]]:
        """Convert symbols to JSON-serializable format."""
        result = []
        for sym in symbols:
            result.append({
                "name": sym.name,
                "qualified_name": sym.qualified_name,
                "type": sym.node_type.value,
                "visibility": sym.visibility.value if hasattr(sym, 'visibility') else "unknown",
                "file": str(sym.location.file) if sym.location and sym.location.file else None,
                "line": sym.location.line if sym.location else 0,
                "end_line": sym.location.end_line if sym.location else 0,
            })
        return result

    def _build_dependencies_data(
        self,
        dependencies: List,
        symbols_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert dependencies to JSON-serializable format."""
        # Build set of valid symbol names for filtering
        valid_symbols = {s["qualified_name"] for s in symbols_data}

        result = []
        for dep in dependencies:
            # Only include dependencies between known symbols
            if dep.source in valid_symbols or dep.target in valid_symbols:
                result.append({
                    "source": dep.source,
                    "target": dep.target,
                    "type": dep.dep_type.value,
                })
        return result

    def _enrich_symbols_from_ast(
        self,
        graph,
        symbols_data: List[Dict[str, Any]]
    ) -> None:
        """
        Enrich symbols with annotations and CC from ASTNode tree.

        Walks graph._files to extract decorators (annotations) and
        cyclomatic_complexity stored in ASTNode.metadata by the Java
        AST backend. Adds new OPTIONAL fields to each symbol dict:
          - annotations: List[str]  (class/method/field annotations)
          - cyclomatic_complexity: int  (method/constructor CC)

        These are additive — existing consumers use .get() and won't break.
        """
        # Build lookup: qualified_name → ASTNode
        from ragix_core.ast_base import NodeType
        node_map: Dict[str, Any] = {}

        def _collect(node):
            qname = node.get_qualified_name() if hasattr(node, 'get_qualified_name') else None
            if qname:
                node_map[qname] = node
            for child in getattr(node, 'children', []):
                _collect(child)

        for _path, ast_node in getattr(graph, '_files', {}).items():
            _collect(ast_node)

        # Enrich each symbol
        for sym in symbols_data:
            qname = sym.get("qualified_name", "")
            ast_node = node_map.get(qname)
            if ast_node is None:
                continue
            # Annotations (decorators in ASTNode)
            decorators = getattr(ast_node, 'decorators', [])
            if decorators:
                sym["annotations"] = list(decorators)
            # Cyclomatic complexity from javalang walker
            cc = ast_node.metadata.get("cyclomatic_complexity") if hasattr(ast_node, 'metadata') else None
            if cc is not None:
                sym["cyclomatic_complexity"] = cc

    def _count_by_type(self, symbols: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count symbols by type."""
        counts: Dict[str, int] = {}
        for sym in symbols:
            t = sym.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _count_by_visibility(self, symbols: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count symbols by visibility."""
        counts: Dict[str, int] = {}
        for sym in symbols:
            v = sym.get("visibility", "unknown")
            counts[v] = counts.get(v, 0) + 1
        return counts
