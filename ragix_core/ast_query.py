"""
AST Query Language - Pattern-based search for AST nodes

Query syntax:
  type:pattern     - Match by node type (class:*Service, method:get*)
  name:pattern     - Match by name
  extends:pattern  - Match classes extending pattern
  implements:pattern - Match classes implementing pattern
  calls:pattern    - Match methods calling pattern
  file:pattern     - Match by file path
  @annotation      - Match by decorator/annotation

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from .ast_base import ASTNode, NodeType, Symbol
from .dependencies import DependencyGraph, DependencyType


class QueryType(str, Enum):
    """Types of query predicates."""
    TYPE = "type"           # class:*, method:*, function:*
    NAME = "name"           # name:*Pattern*
    EXTENDS = "extends"     # extends:BaseClass
    IMPLEMENTS = "implements"  # implements:Interface
    CALLS = "calls"         # calls:methodName
    ANNOTATION = "annotation"  # @Deprecated
    FILE = "file"           # file:*.java
    VISIBILITY = "visibility"  # visibility:public
    MODIFIER = "modifier"   # modifier:static
    RETURNS = "returns"     # returns:String


@dataclass
class QueryPredicate:
    """A single query predicate."""
    query_type: QueryType
    pattern: str
    negated: bool = False

    def matches_symbol(self, symbol: Symbol) -> bool:
        """Check if symbol matches this predicate."""
        result = self._match(symbol)
        return not result if self.negated else result

    def _match(self, symbol: Symbol) -> bool:
        """Internal matching logic."""
        if self.query_type == QueryType.TYPE:
            type_name = symbol.node_type.value
            return fnmatch.fnmatch(type_name, self.pattern)

        elif self.query_type == QueryType.NAME:
            return (
                fnmatch.fnmatch(symbol.name, self.pattern) or
                fnmatch.fnmatch(symbol.qualified_name, self.pattern)
            )

        elif self.query_type == QueryType.VISIBILITY:
            return symbol.visibility.value == self.pattern.lower()

        elif self.query_type == QueryType.FILE:
            return fnmatch.fnmatch(str(symbol.location.file), self.pattern)

        # These require AST node, not just symbol
        return True

    def matches_node(self, node: ASTNode) -> bool:
        """Check if AST node matches this predicate."""
        result = self._match_node(node)
        return not result if self.negated else result

    def _match_node(self, node: ASTNode) -> bool:
        """Internal node matching."""
        if self.query_type == QueryType.TYPE:
            return fnmatch.fnmatch(node.node_type.value, self.pattern)

        elif self.query_type == QueryType.NAME:
            if not node.name:
                return False
            qualified = node.get_qualified_name()
            return (
                fnmatch.fnmatch(node.name, self.pattern) or
                fnmatch.fnmatch(qualified, self.pattern)
            )

        elif self.query_type == QueryType.EXTENDS:
            return node.extends and fnmatch.fnmatch(node.extends, self.pattern)

        elif self.query_type == QueryType.IMPLEMENTS:
            return any(fnmatch.fnmatch(i, self.pattern) for i in node.implements)

        elif self.query_type == QueryType.ANNOTATION:
            return any(fnmatch.fnmatch(d, self.pattern) for d in node.decorators)

        elif self.query_type == QueryType.CALLS:
            calls = node.metadata.get("calls", [])
            return any(fnmatch.fnmatch(c, self.pattern) for c in calls)

        elif self.query_type == QueryType.VISIBILITY:
            return node.visibility.value == self.pattern.lower()

        elif self.query_type == QueryType.MODIFIER:
            return self.pattern in node.modifiers

        elif self.query_type == QueryType.RETURNS:
            if node.return_type:
                return fnmatch.fnmatch(str(node.return_type), self.pattern)
            return False

        elif self.query_type == QueryType.FILE:
            return fnmatch.fnmatch(str(node.location.file), self.pattern)

        return False


@dataclass
class ASTQuery:
    """A complete AST query with multiple predicates."""
    predicates: List[QueryPredicate] = field(default_factory=list)
    raw_query: str = ""

    def matches_symbol(self, symbol: Symbol) -> bool:
        """Check if symbol matches all predicates."""
        return all(p.matches_symbol(symbol) for p in self.predicates)

    def matches_node(self, node: ASTNode) -> bool:
        """Check if AST node matches all predicates."""
        return all(p.matches_node(node) for p in self.predicates)

    def __str__(self) -> str:
        return self.raw_query


@dataclass
class QueryMatch:
    """A query match result."""
    symbol: Optional[Symbol]
    node: Optional[ASTNode]
    score: float = 1.0  # For ranked results

    @property
    def name(self) -> str:
        if self.symbol:
            return self.symbol.qualified_name
        if self.node:
            return self.node.get_qualified_name()
        return ""

    @property
    def location(self) -> str:
        if self.symbol:
            return str(self.symbol.location)
        if self.node:
            return str(self.node.location)
        return ""


def parse_query(query_str: str) -> ASTQuery:
    """
    Parse a query string into an ASTQuery.

    Query syntax:
        type:pattern        - Match node type (class, method, function, etc.)
        name:pattern        - Match by name
        extends:pattern     - Match classes extending
        implements:pattern  - Match classes implementing
        calls:pattern       - Match methods that call
        @annotation         - Match by decorator/annotation
        file:pattern        - Match by file path
        visibility:public   - Match by visibility
        modifier:static     - Match by modifier
        returns:type        - Match by return type
        !predicate          - Negate a predicate

    Patterns support * and ? wildcards.

    Examples:
        "class:*Service"
        "method:get* returns:String"
        "class:* extends:Base*"
        "@Override method:*"
        "!visibility:private class:*"
    """
    query = ASTQuery(raw_query=query_str)
    tokens = _tokenize(query_str)

    for token in tokens:
        predicate = _parse_predicate(token)
        if predicate:
            query.predicates.append(predicate)

    return query


def _tokenize(query_str: str) -> List[str]:
    """Split query into tokens, respecting quoted strings."""
    tokens = []
    current = ""
    in_quotes = False

    for char in query_str:
        if char == '"' or char == "'":
            in_quotes = not in_quotes
        elif char == " " and not in_quotes:
            if current:
                tokens.append(current)
                current = ""
        else:
            current += char

    if current:
        tokens.append(current)

    return tokens


def _parse_predicate(token: str) -> Optional[QueryPredicate]:
    """Parse a single predicate token."""
    # Check for negation
    negated = False
    if token.startswith("!"):
        negated = True
        token = token[1:]

    # Annotation shorthand: @Name -> annotation:Name
    if token.startswith("@"):
        return QueryPredicate(
            query_type=QueryType.ANNOTATION,
            pattern=token[1:],
            negated=negated,
        )

    # Standard predicate: type:pattern
    if ":" in token:
        parts = token.split(":", 1)
        query_type_str = parts[0].lower()
        pattern = parts[1]

        # Map string to QueryType
        type_map = {
            "type": QueryType.TYPE,
            "class": QueryType.TYPE,
            "method": QueryType.TYPE,
            "function": QueryType.TYPE,
            "name": QueryType.NAME,
            "extends": QueryType.EXTENDS,
            "implements": QueryType.IMPLEMENTS,
            "calls": QueryType.CALLS,
            "file": QueryType.FILE,
            "visibility": QueryType.VISIBILITY,
            "vis": QueryType.VISIBILITY,
            "modifier": QueryType.MODIFIER,
            "mod": QueryType.MODIFIER,
            "returns": QueryType.RETURNS,
            "return": QueryType.RETURNS,
            "annotation": QueryType.ANNOTATION,
            "decorator": QueryType.ANNOTATION,
        }

        if query_type_str in type_map:
            query_type = type_map[query_type_str]

            # Special handling for type shortcuts
            if query_type_str in ("class", "method", "function"):
                # class:*Service -> type:class AND name:*Service
                pattern = query_type_str  # The type itself

            return QueryPredicate(
                query_type=query_type,
                pattern=pattern,
                negated=negated,
            )

    # Bare pattern -> name search
    return QueryPredicate(
        query_type=QueryType.NAME,
        pattern=token,
        negated=negated,
    )


def execute_query(
    query: ASTQuery,
    graph: DependencyGraph,
) -> List[QueryMatch]:
    """
    Execute a query against a dependency graph.

    Args:
        query: Parsed ASTQuery
        graph: DependencyGraph to search

    Returns:
        List of matching results
    """
    matches = []

    for symbol in graph.get_symbols():
        if query.matches_symbol(symbol):
            matches.append(QueryMatch(symbol=symbol, node=None))

    return matches


def execute_query_on_ast(
    query: ASTQuery,
    root: ASTNode,
) -> List[QueryMatch]:
    """
    Execute a query against an AST tree.

    Args:
        query: Parsed ASTQuery
        root: Root ASTNode to search

    Returns:
        List of matching results
    """
    matches = []

    def visit(node: ASTNode):
        if query.matches_node(node):
            matches.append(QueryMatch(symbol=None, node=node))
        for child in node.children:
            visit(child)

    visit(root)
    return matches


# Convenience functions

def find_classes(graph: DependencyGraph, pattern: str = "*") -> List[Symbol]:
    """Find all classes matching pattern."""
    query = parse_query(f"type:class name:{pattern}")
    matches = execute_query(query, graph)
    return [m.symbol for m in matches if m.symbol]


def find_methods(graph: DependencyGraph, pattern: str = "*") -> List[Symbol]:
    """Find all methods matching pattern."""
    query = parse_query(f"type:method name:{pattern}")
    matches = execute_query(query, graph)
    return [m.symbol for m in matches if m.symbol]


def find_by_annotation(graph: DependencyGraph, annotation: str) -> List[Symbol]:
    """Find all symbols with a specific annotation."""
    query = parse_query(f"@{annotation}")
    matches = execute_query(query, graph)
    return [m.symbol for m in matches if m.symbol]


def find_subclasses(graph: DependencyGraph, base_class: str) -> List[Symbol]:
    """Find all classes extending a base class."""
    query = parse_query(f"extends:{base_class}")
    matches = execute_query(query, graph)
    return [m.symbol for m in matches if m.symbol]
