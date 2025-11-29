"""
Dependency Analysis - Build and analyze dependency graphs

Tracks dependencies at module, class, and method levels.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .ast_base import (
    ASTNode,
    ASTBackend,
    Language,
    NodeType,
    SourceLocation,
    Symbol,
    get_ast_registry,
)


class DependencyType(str, Enum):
    """Types of dependencies between symbols."""
    IMPORT = "import"               # Module/package imports
    INHERITANCE = "inheritance"     # Class extends
    IMPLEMENTATION = "implementation"  # Class implements interface
    COMPOSITION = "composition"     # Field type reference
    CALL = "call"                   # Method/function call
    ANNOTATION = "annotation"       # Decorator/annotation usage
    TYPE_REFERENCE = "type_reference"  # Type hint/parameter type
    INSTANTIATION = "instantiation"  # new/constructor call


@dataclass
class Dependency:
    """A dependency between two symbols."""
    source: str          # Qualified name of source symbol
    target: str          # Qualified name of target symbol
    dep_type: DependencyType
    location: SourceLocation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.dep_type))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dependency):
            return (
                self.source == other.source
                and self.target == other.target
                and self.dep_type == other.dep_type
            )
        return False

    def __repr__(self) -> str:
        return f"{self.source} --[{self.dep_type.value}]--> {self.target}"


@dataclass
class DependencyStats:
    """Statistics about dependencies."""
    total_dependencies: int = 0
    by_type: Dict[DependencyType, int] = field(default_factory=dict)
    afferent_coupling: Dict[str, int] = field(default_factory=dict)  # Incoming deps
    efferent_coupling: Dict[str, int] = field(default_factory=dict)  # Outgoing deps
    cycles: List[List[str]] = field(default_factory=list)


class DependencyGraph:
    """
    Build and query dependency graphs.

    Supports:
    - Multi-file analysis
    - Multiple dependency types
    - Cycle detection
    - Coupling metrics
    - Export to various formats
    """

    def __init__(self):
        self._dependencies: Set[Dependency] = set()
        self._symbols: Dict[str, Symbol] = {}
        self._files: Dict[Path, ASTNode] = {}
        self._outgoing: Dict[str, Set[Dependency]] = defaultdict(set)
        self._incoming: Dict[str, Set[Dependency]] = defaultdict(set)

    @classmethod
    def from_cached_data(
        cls,
        symbols: List[Dict[str, Any]],
        dependencies: List[Dict[str, Any]]
    ) -> 'DependencyGraph':
        """
        Reconstruct a DependencyGraph from cached data.

        This creates a "light" graph without full AST nodes, suitable for
        visualizations and most analysis operations. Much faster than re-parsing.

        Args:
            symbols: List of symbol dicts with 'name' and 'type' keys
            dependencies: List of dependency dicts with 'source', 'target', 'type' keys

        Returns:
            Reconstructed DependencyGraph
        """
        graph = cls()

        # Reconstruct symbols with location data from cache
        for sym_data in symbols:
            name = sym_data.get('name', '')
            type_str = sym_data.get('type', 'UNKNOWN')

            # Convert type string to NodeType enum
            try:
                node_type = NodeType(type_str)
            except ValueError:
                node_type = NodeType.UNKNOWN

            # Reconstruct location from cached data
            file_str = sym_data.get('file', '')
            location = SourceLocation(
                file=Path(file_str) if file_str else Path("."),
                line=sym_data.get('line', 0),
                column=0,
                end_line=sym_data.get('end_line', 0),
                end_column=0
            )

            symbol = Symbol(
                name=name.split('.')[-1] if '.' in name else name,
                qualified_name=name,
                node_type=node_type,
                location=location,
                language=Language.UNKNOWN
            )
            graph._symbols[name] = symbol

        # Reconstruct dependencies (location not cached - use placeholder)
        dep_location = SourceLocation(file=Path("."), line=0, column=0)
        for dep_data in dependencies:
            source = dep_data.get('source', '')
            target = dep_data.get('target', '')
            type_str = dep_data.get('type', 'import')

            # Convert type string to DependencyType enum
            try:
                dep_type = DependencyType(type_str)
            except ValueError:
                dep_type = DependencyType.IMPORT

            dep = Dependency(
                source=source,
                target=target,
                dep_type=dep_type,
                location=dep_location
            )

            graph._dependencies.add(dep)
            graph._outgoing[source].add(dep)
            graph._incoming[target].add(dep)

        return graph

    def add_file(self, path: Path) -> Optional[ASTNode]:
        """
        Add a file to the dependency graph.

        Args:
            path: Path to source file

        Returns:
            ASTNode if parsed successfully, None otherwise
        """
        registry = get_ast_registry()
        backend = registry.get_backend_for_file(path)

        if not backend:
            return None

        try:
            ast = backend.parse_file(path)
            if ast.metadata.get("parse_error"):
                return None

            self._files[path] = ast

            # Extract symbols
            symbols = backend.get_symbols(ast)
            for symbol in symbols:
                self._symbols[symbol.qualified_name] = symbol

            # Extract dependencies
            self._extract_dependencies(ast, backend)

            return ast

        except Exception as e:
            return None

    def add_directory(
        self,
        path: Path,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> Dict[str, int]:
        """
        Add all matching files from a directory.

        Args:
            path: Directory path
            patterns: File patterns (e.g., ["*.py", "*.java"])
            recursive: Whether to search recursively

        Returns:
            Dict with counts: {"parsed": N, "failed": M}
        """
        if patterns is None:
            patterns = ["*.py", "*.java", "*.js", "*.ts"]

        stats = {"parsed": 0, "failed": 0}

        for pattern in patterns:
            if recursive:
                files = path.rglob(pattern)
            else:
                files = path.glob(pattern)

            for file_path in files:
                if file_path.is_file():
                    result = self.add_file(file_path)
                    if result and not result.metadata.get("parse_error"):
                        stats["parsed"] += 1
                    else:
                        stats["failed"] += 1

        return stats

    def _extract_dependencies(self, ast: ASTNode, backend: ASTBackend) -> None:
        """Extract dependencies from an AST."""
        self._extract_from_node(ast, [], backend.language)

    def _extract_from_node(
        self,
        node: ASTNode,
        scope: List[str],
        language: Language,
    ) -> None:
        """Recursively extract dependencies from AST nodes."""

        # For import nodes, don't include the import path in scope
        if node.node_type in (NodeType.IMPORT, NodeType.IMPORT_FROM):
            # Import source is the parent scope (the class/module doing the import)
            source_name = ".".join(scope) if scope else node.location.file.stem
            for imp in node.imports:
                dep = Dependency(
                    source=source_name,
                    target=imp,
                    dep_type=DependencyType.IMPORT,
                    location=node.location,
                )
                self._add_dependency(dep)
            return  # Don't recurse into import nodes

        current_scope = scope + [node.name] if node.name else scope
        source_name = ".".join(current_scope)

        # Inheritance
        if node.extends:
            dep = Dependency(
                source=source_name,
                target=node.extends,
                dep_type=DependencyType.INHERITANCE,
                location=node.location,
            )
            self._add_dependency(dep)

        # Implementation
        for impl in node.implements:
            dep = Dependency(
                source=source_name,
                target=impl,
                dep_type=DependencyType.IMPLEMENTATION,
                location=node.location,
            )
            self._add_dependency(dep)

        # Decorators/Annotations
        for dec in node.decorators:
            dep = Dependency(
                source=source_name,
                target=dec,
                dep_type=DependencyType.ANNOTATION,
                location=node.location,
            )
            self._add_dependency(dep)

        # Type references (return type, field type)
        if node.type_info:
            self._extract_type_deps(source_name, node.type_info, node.location)

        if node.return_type:
            self._extract_type_deps(source_name, node.return_type, node.location)

        # Method calls from metadata
        if "calls" in node.metadata:
            for call in node.metadata["calls"]:
                dep = Dependency(
                    source=source_name,
                    target=call,
                    dep_type=DependencyType.CALL,
                    location=node.location,
                )
                self._add_dependency(dep)

        # Parameter types
        for param in node.parameters:
            if param.type_info:
                self._extract_type_deps(source_name, param.type_info, param.location)

        # Recurse into children
        for child in node.children:
            self._extract_from_node(child, current_scope, language)

    def _extract_type_deps(
        self,
        source: str,
        type_info,
        location: SourceLocation,
    ) -> None:
        """Extract dependencies from type information."""
        # Skip primitive types
        primitives = {
            "int", "str", "bool", "float", "None", "Any",
            "void", "boolean", "char", "byte", "short", "long", "double",
            "String", "Integer", "Boolean", "Object",
        }

        if type_info.name not in primitives:
            dep = Dependency(
                source=source,
                target=type_info.name,
                dep_type=DependencyType.TYPE_REFERENCE,
                location=location,
            )
            self._add_dependency(dep)

        # Recurse into generic type arguments
        for arg in type_info.type_arguments:
            self._extract_type_deps(source, arg, location)

    def _add_dependency(self, dep: Dependency) -> None:
        """Add a dependency to the graph."""
        self._dependencies.add(dep)
        self._outgoing[dep.source].add(dep)
        self._incoming[dep.target].add(dep)

    # Query methods

    def get_dependencies(
        self,
        symbol: str,
        dep_type: Optional[DependencyType] = None,
    ) -> List[Dependency]:
        """Get all dependencies from a symbol (outgoing)."""
        deps = list(self._outgoing.get(symbol, set()))
        if dep_type:
            deps = [d for d in deps if d.dep_type == dep_type]
        return deps

    def get_dependents(
        self,
        symbol: str,
        dep_type: Optional[DependencyType] = None,
    ) -> List[Dependency]:
        """Get all symbols that depend on this symbol (incoming)."""
        deps = list(self._incoming.get(symbol, set()))
        if dep_type:
            deps = [d for d in deps if d.dep_type == dep_type]
        return deps

    def get_all_dependencies(
        self,
        dep_type: Optional[DependencyType] = None,
    ) -> List[Dependency]:
        """Get all dependencies in the graph."""
        deps = list(self._dependencies)
        if dep_type:
            deps = [d for d in deps if d.dep_type == dep_type]
        return deps

    def get_symbols(self) -> List[Symbol]:
        """Get all symbols in the graph."""
        return list(self._symbols.values())

    def get_symbol(self, name: str) -> Optional[Symbol]:
        """Get a symbol by name."""
        return self._symbols.get(name)

    def find_symbols(self, pattern: str) -> List[Symbol]:
        """Find symbols matching a pattern (supports * wildcard)."""
        import fnmatch
        return [
            s for s in self._symbols.values()
            if fnmatch.fnmatch(s.qualified_name, pattern) or fnmatch.fnmatch(s.name, pattern)
        ]

    # Analysis methods

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect circular dependencies.

        Returns:
            List of cycles, where each cycle is a list of symbol names
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in self._outgoing.get(node, set()):
                target = dep.target
                if target not in visited:
                    dfs(target, path)
                elif target in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(target)
                    cycle = path[cycle_start:] + [target]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for symbol in self._symbols:
            if symbol not in visited:
                dfs(symbol, [])

        return cycles

    def get_stats(self) -> DependencyStats:
        """Calculate dependency statistics."""
        stats = DependencyStats(total_dependencies=len(self._dependencies))

        # Count by type
        for dep in self._dependencies:
            stats.by_type[dep.dep_type] = stats.by_type.get(dep.dep_type, 0) + 1

        # Coupling metrics
        for symbol in self._symbols:
            stats.afferent_coupling[symbol] = len(self._incoming.get(symbol, set()))
            stats.efferent_coupling[symbol] = len(self._outgoing.get(symbol, set()))

        # Detect cycles
        stats.cycles = self.detect_cycles()

        return stats

    def get_unused_imports(self) -> List[Dependency]:
        """Find imports that are not used."""
        unused = []

        for dep in self._dependencies:
            if dep.dep_type == DependencyType.IMPORT:
                # Check if the imported symbol is used elsewhere
                target = dep.target
                # Simplified check: is target referenced as a dependency target?
                used = any(
                    d.dep_type != DependencyType.IMPORT and target in d.target
                    for d in self._dependencies
                )
                if not used:
                    unused.append(dep)

        return unused

    def get_orphan_symbols(self) -> List[Symbol]:
        """Find symbols with no incoming or outgoing dependencies."""
        orphans = []

        for name, symbol in self._symbols.items():
            if symbol.node_type in (NodeType.IMPORT, NodeType.IMPORT_FROM):
                continue
            if not self._incoming.get(name) and not self._outgoing.get(name):
                orphans.append(symbol)

        return orphans

    # Export methods

    def export_dot(
        self,
        dep_types: Optional[List[DependencyType]] = None,
        cluster_by_file: bool = False,
    ) -> str:
        """
        Export graph to DOT format for Graphviz.

        Args:
            dep_types: Filter to specific dependency types
            cluster_by_file: Group nodes by source file

        Returns:
            DOT format string
        """
        lines = ["digraph DependencyGraph {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")
        lines.append("")

        # Edge styles by type
        edge_styles = {
            DependencyType.IMPORT: 'style=dashed, color=gray',
            DependencyType.INHERITANCE: 'color=blue, penwidth=2',
            DependencyType.IMPLEMENTATION: 'color=green, style=dashed',
            DependencyType.CALL: 'color=black',
            DependencyType.COMPOSITION: 'color=purple',
            DependencyType.ANNOTATION: 'color=orange, style=dotted',
            DependencyType.TYPE_REFERENCE: 'color=gray, style=dotted',
        }

        # Add edges
        for dep in self._dependencies:
            if dep_types and dep.dep_type not in dep_types:
                continue

            src = dep.source.replace(".", "_").replace("-", "_")
            tgt = dep.target.replace(".", "_").replace("-", "_")
            style = edge_styles.get(dep.dep_type, "")

            lines.append(f'  "{src}" -> "{tgt}" [{style}];')

        lines.append("}")
        return "\n".join(lines)

    def export_mermaid(
        self,
        dep_types: Optional[List[DependencyType]] = None,
    ) -> str:
        """Export graph to Mermaid format."""
        lines = ["graph LR"]

        # Edge styles
        edge_markers = {
            DependencyType.INHERITANCE: "-->|extends|",
            DependencyType.IMPLEMENTATION: "-.->|implements|",
            DependencyType.CALL: "-->",
            DependencyType.IMPORT: "-.->|import|",
        }

        for dep in self._dependencies:
            if dep_types and dep.dep_type not in dep_types:
                continue

            marker = edge_markers.get(dep.dep_type, "-->")
            src = dep.source.replace(".", "_")
            tgt = dep.target.replace(".", "_")
            lines.append(f"    {src}{marker}{tgt}")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export graph to JSON format."""
        return {
            "symbols": [
                {
                    "name": s.name,
                    "qualified_name": s.qualified_name,
                    "type": s.node_type.value,
                    "file": str(s.location.file),
                    "line": s.location.line,
                }
                for s in self._symbols.values()
            ],
            "dependencies": [
                {
                    "source": d.source,
                    "target": d.target,
                    "type": d.dep_type.value,
                    "file": str(d.location.file),
                    "line": d.location.line,
                }
                for d in self._dependencies
            ],
            "stats": {
                "symbol_count": len(self._symbols),
                "dependency_count": len(self._dependencies),
            },
        }

    def __len__(self) -> int:
        return len(self._dependencies)

    def __repr__(self) -> str:
        return f"DependencyGraph({len(self._symbols)} symbols, {len(self._dependencies)} deps)"


def build_dependency_graph(
    paths: List[Path],
    patterns: Optional[List[str]] = None,
    use_cache: bool = True,
) -> DependencyGraph:
    """
    Build a dependency graph from multiple paths.

    Uses caching by default - if the project hasn't changed (based on file fingerprint),
    returns a reconstructed graph from cache without re-parsing files.

    Args:
        paths: List of files or directories
        patterns: File patterns for directories
        use_cache: Whether to use cached results (default True)

    Returns:
        DependencyGraph instance
    """
    # Import cache here to avoid circular imports
    try:
        from .analysis_cache import get_cache
        cache_available = True
    except ImportError:
        cache_available = False
        use_cache = False

    # For single directory path, try cache first
    if use_cache and cache_available and len(paths) == 1 and paths[0].is_dir():
        cache = get_cache()
        target_path = paths[0].resolve()
        fingerprint, file_count, total_size = cache.get_fingerprint(target_path)

        if cache.is_cached(fingerprint):
            cached = cache.load(fingerprint)
            if cached and cached.symbols and cached.dependencies:
                # Reconstruct from cache - FAST PATH (no file parsing!)
                return DependencyGraph.from_cached_data(
                    symbols=cached.symbols,
                    dependencies=cached.dependencies
                )

    # Build fresh graph
    graph = DependencyGraph()

    for path in paths:
        if path.is_file():
            graph.add_file(path)
        elif path.is_dir():
            graph.add_directory(path, patterns)

    # Cache results for single directory
    if use_cache and cache_available and len(paths) == 1 and paths[0].is_dir():
        target_path = paths[0].resolve()
        fingerprint, file_count, total_size = cache.get_fingerprint(target_path)

        # Only cache if not already cached (fingerprint changed)
        if not cache.is_cached(fingerprint):
            # Filter out imports and modules - only cache actual code entities
            excluded_types = {NodeType.IMPORT, NodeType.IMPORT_FROM, NodeType.MODULE}
            symbols = [
                {
                    'name': s.qualified_name,
                    'type': s.node_type.value,
                    'file': str(s.location.file) if s.location and s.location.file else '',
                    'line': s.location.line if s.location else 0,
                    'end_line': s.location.end_line if s.location and s.location.end_line else 0
                }
                for s in graph.get_symbols()
                if s.node_type not in excluded_types
            ]
            dependencies = [
                {'source': d.source, 'target': d.target, 'type': d.dep_type.value}
                for d in graph._dependencies
            ]
            cycles = graph.detect_cycles()

            # Calculate basic metrics
            metrics_dict = {
                'files': len(graph._files),
                'symbols': len(graph._symbols),
                'dependencies': len(graph._dependencies),
            }

            packages = {}
            for sym in graph.get_symbols():
                if sym.node_type not in excluded_types:
                    pkg = sym.qualified_name.rsplit('.', 1)[0] if '.' in sym.qualified_name else '(default)'
                    packages[pkg] = packages.get(pkg, 0) + 1

            cache.save(
                fingerprint=fingerprint,
                project_path=target_path,
                symbols=symbols,
                dependencies=dependencies,
                metrics=metrics_dict,
                cycles=cycles,
                packages=packages,
                file_count=file_count,
                total_size=total_size
            )

    return graph
