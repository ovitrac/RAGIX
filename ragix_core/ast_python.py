"""
Python AST Backend - Parse Python source code to unified AST

Uses Python's stdlib ast module for parsing.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ast_base import (
    ASTBackend,
    ASTNode,
    Language,
    NodeType,
    SourceLocation,
    TypeInfo,
    Visibility,
    register_backend,
)


class PythonASTBackend(ASTBackend):
    """
    Python AST backend using stdlib ast module.

    Extracts:
    - Classes with inheritance
    - Functions and methods
    - Imports (import and from...import)
    - Decorators
    - Type annotations
    - Docstrings
    - Assignments (module-level constants)
    """

    @property
    def language(self) -> Language:
        return Language.PYTHON

    @property
    def file_extensions(self) -> List[str]:
        return [".py", ".pyi"]

    def parse_file(self, path: Path) -> ASTNode:
        """Parse a Python file."""
        content = path.read_text(encoding="utf-8")
        return self.parse_string(content, str(path))

    def parse_string(self, content: str, filename: str = "<string>") -> ASTNode:
        """Parse Python source code."""
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError as e:
            # Return error node
            return ASTNode(
                node_type=NodeType.MODULE,
                name=Path(filename).stem,
                location=SourceLocation(Path(filename), e.lineno or 1),
                metadata={"error": str(e), "parse_error": True},
            )

        # Convert to unified AST
        return self._convert_module(tree, Path(filename))

    def _convert_module(self, tree: ast.Module, path: Path) -> ASTNode:
        """Convert ast.Module to ASTNode."""
        module_name = path.stem

        # Get module docstring
        docstring = ast.get_docstring(tree)

        module_node = ASTNode(
            node_type=NodeType.MODULE,
            name=module_name,
            location=SourceLocation(path, 1),
            docstring=docstring,
            metadata={"path": str(path)},
        )

        # Process top-level statements
        for stmt in tree.body:
            child = self._convert_statement(stmt, path, [module_name])
            if child:
                module_node.children.append(child)

        return module_node

    def _convert_statement(
        self,
        stmt: ast.stmt,
        path: Path,
        scope: List[str],
    ) -> Optional[ASTNode]:
        """Convert a statement to ASTNode."""

        if isinstance(stmt, ast.ClassDef):
            return self._convert_class(stmt, path, scope)

        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._convert_function(stmt, path, scope)

        elif isinstance(stmt, ast.Import):
            return self._convert_import(stmt, path)

        elif isinstance(stmt, ast.ImportFrom):
            return self._convert_import_from(stmt, path)

        elif isinstance(stmt, ast.Assign):
            return self._convert_assign(stmt, path, scope)

        elif isinstance(stmt, ast.AnnAssign):
            return self._convert_ann_assign(stmt, path, scope)

        return None

    def _convert_class(
        self,
        node: ast.ClassDef,
        path: Path,
        scope: List[str],
    ) -> ASTNode:
        """Convert a class definition."""
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Get base classes
        bases = [self._get_name(b) for b in node.bases]
        extends = bases[0] if bases else None
        implements = bases[1:] if len(bases) > 1 else []

        # Determine visibility
        visibility = self._get_visibility(node.name)

        class_node = ASTNode(
            node_type=NodeType.CLASS,
            name=node.name,
            location=SourceLocation(path, node.lineno, node.col_offset),
            visibility=visibility,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            extends=extends,
            implements=implements,
            metadata={"qualified_parts": scope + [node.name]},
        )

        # Process class body
        new_scope = scope + [node.name]
        for stmt in node.body:
            child = self._convert_statement(stmt, path, new_scope)
            if child:
                class_node.children.append(child)

        return class_node

    def _convert_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        path: Path,
        scope: List[str],
    ) -> ASTNode:
        """Convert a function/method definition."""
        # Determine if it's a method (inside a class)
        is_method = len(scope) > 1 and any(
            s[0].isupper() for s in scope[1:]
        )

        node_type = NodeType.METHOD if is_method else NodeType.FUNCTION

        # Check for special methods
        if node.name == "__init__":
            node_type = NodeType.CONSTRUCTOR

        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Get modifiers
        modifiers = []
        if isinstance(node, ast.AsyncFunctionDef):
            modifiers.append("async")
        if "staticmethod" in decorators:
            modifiers.append("static")
        if "classmethod" in decorators:
            modifiers.append("classmethod")
        if "abstractmethod" in decorators:
            modifiers.append("abstract")
        if "property" in decorators:
            node_type = NodeType.PROPERTY

        # Get return type
        return_type = None
        if node.returns:
            return_type = self._get_type_info(node.returns)

        # Determine visibility
        visibility = self._get_visibility(node.name)

        func_node = ASTNode(
            node_type=node_type,
            name=node.name,
            location=SourceLocation(path, node.lineno, node.col_offset),
            visibility=visibility,
            docstring=ast.get_docstring(node),
            decorators=decorators,
            modifiers=modifiers,
            return_type=return_type,
            metadata={"qualified_parts": scope + [node.name]},
        )

        # Add parameters
        for arg in node.args.args:
            param = self._convert_arg(arg, path)
            func_node.parameters.append(param)

        # Check for calls inside the function (for dependency analysis)
        calls = self._extract_calls(node)
        if calls:
            func_node.metadata["calls"] = calls

        return func_node

    def _convert_arg(self, arg: ast.arg, path: Path) -> ASTNode:
        """Convert a function argument."""
        type_info = None
        if arg.annotation:
            type_info = self._get_type_info(arg.annotation)

        return ASTNode(
            node_type=NodeType.PARAMETER,
            name=arg.arg,
            location=SourceLocation(path, arg.lineno, arg.col_offset),
            type_info=type_info,
        )

    def _convert_import(self, node: ast.Import, path: Path) -> ASTNode:
        """Convert an import statement."""
        names = []
        for alias in node.names:
            if alias.asname:
                names.append(f"{alias.name} as {alias.asname}")
            else:
                names.append(alias.name)

        return ASTNode(
            node_type=NodeType.IMPORT,
            name=", ".join(names),
            location=SourceLocation(path, node.lineno, node.col_offset),
            imports=[alias.name for alias in node.names],
            metadata={"aliases": {a.asname: a.name for a in node.names if a.asname}},
        )

    def _convert_import_from(self, node: ast.ImportFrom, path: Path) -> ASTNode:
        """Convert a from...import statement."""
        module = node.module or ""
        names = []
        for alias in node.names:
            if alias.asname:
                names.append(f"{alias.name} as {alias.asname}")
            else:
                names.append(alias.name)

        import_str = f"from {module} import {', '.join(names)}"

        # Build full import paths
        imports = []
        for alias in node.names:
            if alias.name == "*":
                imports.append(f"{module}.*")
            else:
                imports.append(f"{module}.{alias.name}")

        return ASTNode(
            node_type=NodeType.IMPORT_FROM,
            name=import_str,
            location=SourceLocation(path, node.lineno, node.col_offset),
            imports=imports,
            metadata={
                "module": module,
                "level": node.level,  # Relative import level
                "names": [a.name for a in node.names],
            },
        )

    def _convert_assign(
        self,
        node: ast.Assign,
        path: Path,
        scope: List[str],
    ) -> Optional[ASTNode]:
        """Convert an assignment (module-level constants)."""
        # Only handle simple name assignments at module level
        if len(scope) > 1:
            return None

        for target in node.targets:
            if isinstance(target, ast.Name):
                # Check if it looks like a constant (ALL_CAPS)
                if target.id.isupper():
                    node_type = NodeType.CONSTANT
                else:
                    node_type = NodeType.FIELD

                return ASTNode(
                    node_type=node_type,
                    name=target.id,
                    location=SourceLocation(path, node.lineno, node.col_offset),
                    visibility=self._get_visibility(target.id),
                    metadata={
                        "qualified_parts": scope + [target.id],
                        "value_type": type(node.value).__name__,
                    },
                )

        return None

    def _convert_ann_assign(
        self,
        node: ast.AnnAssign,
        path: Path,
        scope: List[str],
    ) -> Optional[ASTNode]:
        """Convert an annotated assignment."""
        if not isinstance(node.target, ast.Name):
            return None

        type_info = self._get_type_info(node.annotation)
        name = node.target.id

        # Determine node type
        if name.isupper():
            node_type = NodeType.CONSTANT
        else:
            node_type = NodeType.FIELD

        return ASTNode(
            node_type=node_type,
            name=name,
            location=SourceLocation(path, node.lineno, node.col_offset),
            visibility=self._get_visibility(name),
            type_info=type_info,
            metadata={"qualified_parts": scope + [name]},
        )

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Extract decorator name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_name(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return str(node)

    def _get_name(self, node: ast.expr) -> str:
        """Get the name from an expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self._get_name(node.value)
            return f"{value}[...]"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ""

    def _get_type_info(self, node: ast.expr) -> TypeInfo:
        """Extract type information from annotation."""
        if isinstance(node, ast.Name):
            return TypeInfo(name=node.id)

        elif isinstance(node, ast.Attribute):
            return TypeInfo(name=self._get_name(node))

        elif isinstance(node, ast.Subscript):
            base_name = self._get_name(node.value)

            # Handle generic types
            if isinstance(node.slice, ast.Tuple):
                type_args = [self._get_type_info(elt) for elt in node.slice.elts]
            else:
                type_args = [self._get_type_info(node.slice)]

            # Special handling for Optional, List, etc.
            if base_name == "Optional":
                inner = type_args[0] if type_args else TypeInfo(name="Any")
                inner.is_optional = True
                return inner
            elif base_name == "List":
                inner = type_args[0] if type_args else TypeInfo(name="Any")
                inner.is_array = True
                return inner

            return TypeInfo(
                name=base_name,
                is_generic=True,
                type_arguments=type_args,
            )

        elif isinstance(node, ast.Constant):
            if node.value is None:
                return TypeInfo(name="None")
            return TypeInfo(name=str(type(node.value).__name__))

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type: X | Y
            left = self._get_type_info(node.left)
            right = self._get_type_info(node.right)
            return TypeInfo(
                name="Union",
                is_generic=True,
                type_arguments=[left, right],
            )

        return TypeInfo(name="Any")

    def _get_visibility(self, name: str) -> Visibility:
        """Determine visibility from Python naming conventions."""
        if name.startswith("__") and not name.endswith("__"):
            return Visibility.PRIVATE
        elif name.startswith("_"):
            return Visibility.PROTECTED
        return Visibility.PUBLIC

    def _extract_calls(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Extract function/method calls from a function body."""
        calls = []

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, call_node: ast.Call):
                if isinstance(call_node.func, ast.Name):
                    calls.append(call_node.func.id)
                elif isinstance(call_node.func, ast.Attribute):
                    # Get the full call path
                    parts = []
                    current = call_node.func
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                    parts.reverse()
                    calls.append(".".join(parts))
                self.generic_visit(call_node)

        CallVisitor().visit(node)
        return calls


# Register the backend
def register_python_backend() -> PythonASTBackend:
    """Create and register the Python AST backend."""
    backend = PythonASTBackend()
    register_backend(backend)
    return backend


# Auto-register on import
_python_backend = register_python_backend()


def get_python_backend() -> PythonASTBackend:
    """Get the Python AST backend."""
    return _python_backend
