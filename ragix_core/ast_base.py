"""
AST Base Types - Unified Abstract Syntax Tree representation

Multi-language AST parsing with unified node types for Python, Java, JavaScript.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Set, Tuple, Union


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


class NodeType(str, Enum):
    """Unified AST node types across languages."""
    # Structure
    MODULE = "module"
    PACKAGE = "package"
    NAMESPACE = "namespace"

    # Types
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    ANNOTATION_TYPE = "annotation_type"

    # Members
    METHOD = "method"
    FUNCTION = "function"
    CONSTRUCTOR = "constructor"
    FIELD = "field"
    PROPERTY = "property"
    CONSTANT = "constant"

    # Imports/Exports
    IMPORT = "import"
    EXPORT = "export"
    IMPORT_FROM = "import_from"

    # Decorators/Annotations
    DECORATOR = "decorator"
    ANNOTATION = "annotation"

    # Parameters
    PARAMETER = "parameter"
    TYPE_PARAMETER = "type_parameter"

    # Statements
    ASSIGNMENT = "assignment"
    CALL = "call"
    RETURN = "return"

    # Other
    COMMENT = "comment"
    DOCSTRING = "docstring"
    UNKNOWN = "unknown"


class Visibility(str, Enum):
    """Member visibility modifiers."""
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    PACKAGE = "package"  # Java default
    INTERNAL = "internal"  # TypeScript
    UNKNOWN = "unknown"


@dataclass
class SourceLocation:
    """Location in source code."""
    file: Path
    line: int
    column: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def __str__(self) -> str:
        if self.end_line and self.end_line != self.line:
            return f"{self.file}:{self.line}-{self.end_line}"
        return f"{self.file}:{self.line}"


@dataclass
class TypeInfo:
    """Type information for typed languages."""
    name: str
    is_array: bool = False
    is_generic: bool = False
    type_arguments: List["TypeInfo"] = field(default_factory=list)
    is_optional: bool = False

    def __str__(self) -> str:
        base = self.name
        if self.type_arguments:
            args = ", ".join(str(t) for t in self.type_arguments)
            base = f"{base}<{args}>"
        if self.is_array:
            base = f"{base}[]"
        if self.is_optional:
            base = f"{base}?"
        return base


@dataclass
class Symbol:
    """A named symbol in the codebase."""
    name: str
    qualified_name: str  # Full path: module.class.method
    node_type: NodeType
    location: SourceLocation
    language: Language
    visibility: Visibility = Visibility.UNKNOWN
    type_info: Optional[TypeInfo] = None
    docstring: Optional[str] = None
    is_static: bool = False
    is_abstract: bool = False
    is_final: bool = False
    is_async: bool = False

    def __hash__(self) -> int:
        return hash(self.qualified_name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Symbol):
            return self.qualified_name == other.qualified_name
        return False


@dataclass
class ASTNode:
    """
    Unified AST node representation.

    Provides a language-agnostic view of code structure.
    """
    node_type: NodeType
    name: str
    location: SourceLocation
    children: List["ASTNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields
    visibility: Visibility = Visibility.UNKNOWN
    type_info: Optional[TypeInfo] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    parameters: List["ASTNode"] = field(default_factory=list)
    return_type: Optional[TypeInfo] = None
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    def get_children_by_type(self, node_type: NodeType) -> List["ASTNode"]:
        """Get all direct children of a specific type."""
        return [c for c in self.children if c.node_type == node_type]

    def find_all(self, node_type: NodeType) -> Iterator["ASTNode"]:
        """Recursively find all nodes of a specific type."""
        if self.node_type == node_type:
            yield self
        for child in self.children:
            yield from child.find_all(node_type)

    def find_by_name(self, name: str) -> Optional["ASTNode"]:
        """Find a node by name (recursive)."""
        if self.name == name:
            return self
        for child in self.children:
            found = child.find_by_name(name)
            if found:
                return found
        return None

    def get_qualified_name(self, separator: str = ".") -> str:
        """Build qualified name from metadata."""
        parts = self.metadata.get("qualified_parts", [self.name])
        return separator.join(parts)

    def to_symbol(self, language: Language) -> Symbol:
        """Convert to a Symbol."""
        return Symbol(
            name=self.name,
            qualified_name=self.get_qualified_name(),
            node_type=self.node_type,
            location=self.location,
            language=language,
            visibility=self.visibility,
            type_info=self.type_info,
            docstring=self.docstring,
            is_static="static" in self.modifiers,
            is_abstract="abstract" in self.modifiers,
            is_final="final" in self.modifiers,
            is_async="async" in self.modifiers,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "type": self.node_type.value,
            "name": self.name,
            "location": str(self.location),
        }
        if self.visibility != Visibility.UNKNOWN:
            result["visibility"] = self.visibility.value
        if self.type_info:
            result["type_info"] = str(self.type_info)
        if self.docstring:
            result["docstring"] = self.docstring[:100] + "..." if len(self.docstring) > 100 else self.docstring
        if self.decorators:
            result["decorators"] = self.decorators
        if self.modifiers:
            result["modifiers"] = self.modifiers
        if self.extends:
            result["extends"] = self.extends
        if self.implements:
            result["implements"] = self.implements
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result

    def __repr__(self) -> str:
        return f"ASTNode({self.node_type.value}, {self.name}, {self.location})"


class ASTBackend(ABC):
    """
    Abstract base class for language-specific AST parsers.

    Each language implements its own backend.
    """

    @property
    @abstractmethod
    def language(self) -> Language:
        """The language this backend handles."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """File extensions this backend can parse."""
        pass

    @abstractmethod
    def parse_file(self, path: Path) -> ASTNode:
        """
        Parse a file and return the AST.

        Args:
            path: Path to the source file

        Returns:
            Root ASTNode representing the file
        """
        pass

    @abstractmethod
    def parse_string(self, content: str, filename: str = "<string>") -> ASTNode:
        """
        Parse source code string and return the AST.

        Args:
            content: Source code content
            filename: Virtual filename for error reporting

        Returns:
            Root ASTNode representing the code
        """
        pass

    def can_parse(self, path: Path) -> bool:
        """Check if this backend can parse the given file."""
        return path.suffix.lower() in self.file_extensions

    def get_symbols(self, node: ASTNode) -> List[Symbol]:
        """
        Extract all symbols from an AST.

        Args:
            node: Root AST node

        Returns:
            List of symbols found in the AST
        """
        symbols = []
        self._collect_symbols(node, symbols, [])
        return symbols

    def _collect_symbols(
        self,
        node: ASTNode,
        symbols: List[Symbol],
        scope: List[str],
    ) -> None:
        """Recursively collect symbols."""
        # Add current node if it's a named symbol
        if node.name and node.node_type not in (NodeType.MODULE, NodeType.UNKNOWN):
            current_scope = scope + [node.name]
            node.metadata["qualified_parts"] = current_scope
            symbols.append(node.to_symbol(self.language))
        else:
            current_scope = scope

        # Recurse into children
        for child in node.children:
            self._collect_symbols(child, symbols, current_scope)


class ASTRegistry:
    """
    Registry of AST backends for different languages.
    """

    def __init__(self):
        self._backends: Dict[Language, ASTBackend] = {}
        self._extension_map: Dict[str, Language] = {}

    def register(self, backend: ASTBackend) -> None:
        """Register an AST backend."""
        self._backends[backend.language] = backend
        for ext in backend.file_extensions:
            self._extension_map[ext.lower()] = backend.language

    def get_backend(self, language: Language) -> Optional[ASTBackend]:
        """Get backend for a language."""
        return self._backends.get(language)

    def get_backend_for_file(self, path: Path) -> Optional[ASTBackend]:
        """Get backend for a file based on extension."""
        ext = path.suffix.lower()
        if ext in self._extension_map:
            language = self._extension_map[ext]
            return self._backends.get(language)
        return None

    def detect_language(self, path: Path) -> Language:
        """Detect language from file extension."""
        ext = path.suffix.lower()
        return self._extension_map.get(ext, Language.UNKNOWN)

    def list_languages(self) -> List[Language]:
        """List registered languages."""
        return list(self._backends.keys())

    def parse_file(self, path: Path) -> Optional[ASTNode]:
        """Parse a file using the appropriate backend."""
        backend = self.get_backend_for_file(path)
        if backend:
            return backend.parse_file(path)
        return None


# Global registry instance
_registry: Optional[ASTRegistry] = None


def get_ast_registry() -> ASTRegistry:
    """Get the global AST registry."""
    global _registry
    if _registry is None:
        _registry = ASTRegistry()
    return _registry


def register_backend(backend: ASTBackend) -> None:
    """Register a backend in the global registry."""
    get_ast_registry().register(backend)


def parse_file(path: Path) -> Optional[ASTNode]:
    """Parse a file using the global registry."""
    return get_ast_registry().parse_file(path)


def detect_language(path: Path) -> Language:
    """Detect language from file path."""
    return get_ast_registry().detect_language(path)


# Utility functions

def format_ast_tree(node: ASTNode, indent: int = 0, max_depth: int = 10) -> str:
    """Format an AST as a tree string."""
    if indent >= max_depth * 2:
        return " " * indent + "...\n"

    lines = []
    prefix = " " * indent
    type_str = node.node_type.value
    name_str = node.name if node.name else "(anonymous)"

    # Add modifiers
    mods = []
    if node.visibility != Visibility.UNKNOWN:
        mods.append(node.visibility.value)
    mods.extend(node.modifiers)

    mod_str = " ".join(mods)
    if mod_str:
        lines.append(f"{prefix}{type_str} {mod_str} {name_str}")
    else:
        lines.append(f"{prefix}{type_str} {name_str}")

    # Add type info
    if node.type_info:
        lines[-1] += f": {node.type_info}"

    # Add extends/implements
    if node.extends:
        lines.append(f"{prefix}  extends {node.extends}")
    if node.implements:
        lines.append(f"{prefix}  implements {', '.join(node.implements)}")

    # Recurse
    for child in node.children:
        lines.append(format_ast_tree(child, indent + 2, max_depth))

    return "\n".join(lines)


def count_nodes(node: ASTNode) -> Dict[NodeType, int]:
    """Count nodes by type."""
    counts: Dict[NodeType, int] = {}

    def visit(n: ASTNode):
        counts[n.node_type] = counts.get(n.node_type, 0) + 1
        for child in n.children:
            visit(child)

    visit(node)
    return counts
