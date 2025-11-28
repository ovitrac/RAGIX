"""
Java AST Backend - Parse Java source code to unified AST

Uses javalang library for parsing.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import javalang
    from javalang.tree import (
        ClassDeclaration,
        InterfaceDeclaration,
        EnumDeclaration,
        AnnotationDeclaration,
        MethodDeclaration,
        ConstructorDeclaration,
        FieldDeclaration,
        FormalParameter,
        Import,
        PackageDeclaration,
        Annotation,
        CompilationUnit,
        Node as JavaNode,
    )
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    javalang = None

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


class JavaASTBackend(ASTBackend):
    """
    Java AST backend using javalang library.

    Extracts:
    - Classes, interfaces, enums, annotations
    - Methods and constructors
    - Fields
    - Imports (regular and static)
    - Package declarations
    - Annotations/decorators
    - Modifiers (public, private, static, final, abstract)
    - Type information with generics
    """

    def __init__(self):
        if not JAVALANG_AVAILABLE:
            raise ImportError(
                "javalang is required for Java AST parsing. "
                "Install with: pip install javalang"
            )

    @property
    def language(self) -> Language:
        return Language.JAVA

    @property
    def file_extensions(self) -> List[str]:
        return [".java"]

    def parse_file(self, path: Path) -> ASTNode:
        """Parse a Java file."""
        content = path.read_text(encoding="utf-8")
        return self.parse_string(content, str(path))

    def parse_string(self, content: str, filename: str = "<string>") -> ASTNode:
        """Parse Java source code."""
        try:
            tree = javalang.parse.parse(content)
        except javalang.parser.JavaSyntaxError as e:
            return ASTNode(
                node_type=NodeType.MODULE,
                name=Path(filename).stem,
                location=SourceLocation(Path(filename), 1),
                metadata={"error": str(e), "parse_error": True},
            )
        except Exception as e:
            return ASTNode(
                node_type=NodeType.MODULE,
                name=Path(filename).stem,
                location=SourceLocation(Path(filename), 1),
                metadata={"error": str(e), "parse_error": True},
            )

        return self._convert_compilation_unit(tree, Path(filename))

    def _convert_compilation_unit(
        self,
        tree: CompilationUnit,
        path: Path,
    ) -> ASTNode:
        """Convert CompilationUnit to ASTNode."""
        # Get package name
        package_name = ""
        if tree.package:
            package_name = tree.package.name

        module_name = path.stem
        full_name = f"{package_name}.{module_name}" if package_name else module_name

        module_node = ASTNode(
            node_type=NodeType.MODULE,
            name=module_name,
            location=SourceLocation(path, 1),
            metadata={
                "path": str(path),
                "package": package_name,
                "qualified_parts": [full_name],
            },
        )

        # Add package declaration
        if tree.package:
            package_node = ASTNode(
                node_type=NodeType.PACKAGE,
                name=package_name,
                location=SourceLocation(path, tree.package.position.line if tree.package.position else 1),
            )
            module_node.children.append(package_node)

        # Add imports
        for imp in tree.imports or []:
            import_node = self._convert_import(imp, path)
            module_node.children.append(import_node)
            module_node.imports.append(import_node.name)

        # Add type declarations
        scope = [package_name] if package_name else []
        for type_decl in tree.types or []:
            if type_decl:
                type_node = self._convert_type_declaration(type_decl, path, scope)
                if type_node:
                    module_node.children.append(type_node)

        return module_node

    def _convert_import(self, imp: Import, path: Path) -> ASTNode:
        """Convert an import declaration."""
        import_path = imp.path
        is_static = imp.static
        is_wildcard = imp.wildcard

        if is_wildcard:
            import_path += ".*"

        name = f"static {import_path}" if is_static else import_path

        return ASTNode(
            node_type=NodeType.IMPORT,
            name=name,
            location=SourceLocation(path, imp.position.line if imp.position else 1),
            imports=[import_path],
            metadata={
                "static": is_static,
                "wildcard": is_wildcard,
            },
        )

    def _convert_type_declaration(
        self,
        node: Union[ClassDeclaration, InterfaceDeclaration, EnumDeclaration, AnnotationDeclaration],
        path: Path,
        scope: List[str],
    ) -> Optional[ASTNode]:
        """Convert a type declaration (class, interface, enum, annotation)."""

        if isinstance(node, ClassDeclaration):
            node_type = NodeType.CLASS
        elif isinstance(node, InterfaceDeclaration):
            node_type = NodeType.INTERFACE
        elif isinstance(node, EnumDeclaration):
            node_type = NodeType.ENUM
        elif isinstance(node, AnnotationDeclaration):
            node_type = NodeType.ANNOTATION_TYPE
        else:
            return None

        # Get position
        line = node.position.line if node.position else 1

        # Get modifiers
        modifiers = list(node.modifiers) if node.modifiers else []
        visibility = self._get_visibility(modifiers)

        # Get annotations
        annotations = self._get_annotations(node.annotations)

        # Get extends
        extends = None
        if hasattr(node, "extends") and node.extends:
            if isinstance(node.extends, list):
                extends = node.extends[0].name if node.extends else None
            else:
                extends = node.extends.name if node.extends else None

        # Get implements
        implements = []
        if hasattr(node, "implements") and node.implements:
            implements = [i.name for i in node.implements]

        # Get type parameters
        type_params = []
        if hasattr(node, "type_parameters") and node.type_parameters:
            type_params = [tp.name for tp in node.type_parameters]

        new_scope = scope + [node.name]

        type_node = ASTNode(
            node_type=node_type,
            name=node.name,
            location=SourceLocation(path, line),
            visibility=visibility,
            decorators=annotations,
            modifiers=[m for m in modifiers if m not in ("public", "private", "protected")],
            extends=extends,
            implements=implements,
            docstring=self._get_javadoc(node),
            metadata={
                "qualified_parts": new_scope,
                "type_parameters": type_params,
            },
        )

        # Process body
        if hasattr(node, "body") and node.body:
            for member in node.body:
                member_node = self._convert_member(member, path, new_scope)
                if member_node:
                    type_node.children.append(member_node)

        return type_node

    def _convert_member(
        self,
        node: JavaNode,
        path: Path,
        scope: List[str],
    ) -> Optional[ASTNode]:
        """Convert a class member."""

        if isinstance(node, MethodDeclaration):
            return self._convert_method(node, path, scope)

        elif isinstance(node, ConstructorDeclaration):
            return self._convert_constructor(node, path, scope)

        elif isinstance(node, FieldDeclaration):
            return self._convert_field(node, path, scope)

        elif isinstance(node, (ClassDeclaration, InterfaceDeclaration, EnumDeclaration)):
            # Inner class
            return self._convert_type_declaration(node, path, scope)

        return None

    def _convert_method(
        self,
        node: MethodDeclaration,
        path: Path,
        scope: List[str],
    ) -> ASTNode:
        """Convert a method declaration."""
        line = node.position.line if node.position else 1

        modifiers = list(node.modifiers) if node.modifiers else []
        visibility = self._get_visibility(modifiers)
        annotations = self._get_annotations(node.annotations)

        # Get return type
        return_type = None
        if node.return_type:
            return_type = self._get_type_info(node.return_type)

        new_scope = scope + [node.name]

        method_node = ASTNode(
            node_type=NodeType.METHOD,
            name=node.name,
            location=SourceLocation(path, line),
            visibility=visibility,
            decorators=annotations,
            modifiers=[m for m in modifiers if m not in ("public", "private", "protected")],
            return_type=return_type,
            docstring=self._get_javadoc(node),
            metadata={
                "qualified_parts": new_scope,
                "throws": [t.name for t in node.throws] if node.throws else [],
            },
        )

        # Add parameters
        if node.parameters:
            for param in node.parameters:
                param_node = self._convert_parameter(param, path)
                method_node.parameters.append(param_node)

        # Extract method calls
        calls = self._extract_calls(node)
        if calls:
            method_node.metadata["calls"] = calls

        return method_node

    def _convert_constructor(
        self,
        node: ConstructorDeclaration,
        path: Path,
        scope: List[str],
    ) -> ASTNode:
        """Convert a constructor declaration."""
        line = node.position.line if node.position else 1

        modifiers = list(node.modifiers) if node.modifiers else []
        visibility = self._get_visibility(modifiers)
        annotations = self._get_annotations(node.annotations)

        new_scope = scope + [node.name]

        ctor_node = ASTNode(
            node_type=NodeType.CONSTRUCTOR,
            name=node.name,
            location=SourceLocation(path, line),
            visibility=visibility,
            decorators=annotations,
            modifiers=[m for m in modifiers if m not in ("public", "private", "protected")],
            docstring=self._get_javadoc(node),
            metadata={
                "qualified_parts": new_scope,
                "throws": [t.name for t in node.throws] if node.throws else [],
            },
        )

        # Add parameters
        if node.parameters:
            for param in node.parameters:
                param_node = self._convert_parameter(param, path)
                ctor_node.parameters.append(param_node)

        return ctor_node

    def _convert_field(
        self,
        node: FieldDeclaration,
        path: Path,
        scope: List[str],
    ) -> ASTNode:
        """Convert a field declaration."""
        line = node.position.line if node.position else 1

        modifiers = list(node.modifiers) if node.modifiers else []
        visibility = self._get_visibility(modifiers)
        annotations = self._get_annotations(node.annotations)

        # Get type
        type_info = self._get_type_info(node.type) if node.type else None

        # Get field name(s) - there can be multiple declarators
        names = [decl.name for decl in node.declarators]
        name = names[0] if names else "unknown"

        # Determine if constant (static final, typically UPPER_CASE)
        is_constant = "static" in modifiers and "final" in modifiers
        node_type = NodeType.CONSTANT if is_constant else NodeType.FIELD

        new_scope = scope + [name]

        return ASTNode(
            node_type=node_type,
            name=name,
            location=SourceLocation(path, line),
            visibility=visibility,
            type_info=type_info,
            decorators=annotations,
            modifiers=[m for m in modifiers if m not in ("public", "private", "protected")],
            docstring=self._get_javadoc(node),
            metadata={
                "qualified_parts": new_scope,
                "all_names": names if len(names) > 1 else None,
            },
        )

    def _convert_parameter(
        self,
        param: FormalParameter,
        path: Path,
    ) -> ASTNode:
        """Convert a method parameter."""
        type_info = self._get_type_info(param.type) if param.type else None

        # Check for varargs
        if param.varargs:
            if type_info:
                type_info.name += "..."

        return ASTNode(
            node_type=NodeType.PARAMETER,
            name=param.name,
            location=SourceLocation(path, param.position.line if param.position else 1),
            type_info=type_info,
            modifiers=list(param.modifiers) if param.modifiers else [],
        )

    def _get_type_info(self, type_node) -> TypeInfo:
        """Extract type information from a type node."""
        if type_node is None:
            return TypeInfo(name="void")

        # Basic type name
        name = type_node.name if hasattr(type_node, "name") else str(type_node)

        # Check for array dimensions
        is_array = False
        if hasattr(type_node, "dimensions") and type_node.dimensions:
            is_array = True

        # Check for type arguments (generics)
        type_arguments = []
        is_generic = False
        if hasattr(type_node, "arguments") and type_node.arguments:
            is_generic = True
            for arg in type_node.arguments:
                if hasattr(arg, "type") and arg.type:
                    type_arguments.append(self._get_type_info(arg.type))
                elif hasattr(arg, "name"):
                    type_arguments.append(TypeInfo(name=arg.name))

        return TypeInfo(
            name=name,
            is_array=is_array,
            is_generic=is_generic,
            type_arguments=type_arguments,
        )

    def _get_visibility(self, modifiers: List[str]) -> Visibility:
        """Get visibility from modifiers."""
        if "public" in modifiers:
            return Visibility.PUBLIC
        elif "private" in modifiers:
            return Visibility.PRIVATE
        elif "protected" in modifiers:
            return Visibility.PROTECTED
        return Visibility.PACKAGE  # Java default

    def _get_annotations(self, annotations) -> List[str]:
        """Extract annotation names."""
        if not annotations:
            return []
        return [a.name for a in annotations]

    def _get_javadoc(self, node) -> Optional[str]:
        """
        Extract Javadoc comment if available and substantive.

        Filters out auto-generated placeholder Javadocs like:
        - "The Class Foo."
        - "The Interface Bar."
        - Empty @author tags
        """
        if not hasattr(node, "documentation") or not node.documentation:
            return None

        doc = node.documentation

        # Strip comment markers and get content
        content = doc.replace('/**', '').replace('*/', '').strip()
        lines = [l.strip().lstrip('*').strip() for l in content.split('\n')]
        lines = [l for l in lines if l]  # Remove empty lines

        if not lines:
            return None

        # Check for substantive content
        import re

        # Placeholder patterns (auto-generated, not real docs)
        placeholder_patterns = [
            r"^The (Class|Interface|Enum|Type)\s+\w+\.?$",  # "The Class Foo."
            r"^@author\s*$",  # Empty @author
            r"^TODO\b",  # TODO placeholders
            r"^Auto-generated",  # Auto-generated
            r"^\w+\.java$",  # Just filename
        ]

        has_real_content = False

        for line in lines:
            # Skip lines that are just @author with a name (common but not documentation)
            if re.match(r'^@author\s+\S', line):
                continue

            # Check for @param, @return, @throws, @see with content = real doc
            if re.match(r'@(param|return|throws|exception|see|since|version)\s+\S', line):
                has_real_content = True
                break

            # Check for substantive description (>40 chars and not a placeholder pattern)
            if len(line) > 40:
                is_placeholder = any(re.match(p, line, re.IGNORECASE) for p in placeholder_patterns)
                if not is_placeholder:
                    has_real_content = True
                    break

            # Short but not matching placeholder patterns might be real
            if len(line) > 15 and not any(re.match(p, line, re.IGNORECASE) for p in placeholder_patterns):
                # Check it's not just "The Class X" variants
                if not re.match(r'^(The\s+)?\w+\s+(class|interface|enum|implementation|test)\.?$', line, re.IGNORECASE):
                    has_real_content = True
                    break

        return doc if has_real_content else None

    def _extract_calls(self, node: MethodDeclaration) -> List[str]:
        """Extract method calls from a method body."""
        calls = []

        def visit(n):
            if n is None:
                return

            # Check for method invocation
            node_type = type(n).__name__
            if node_type == "MethodInvocation":
                qualifier = getattr(n, "qualifier", None)
                member = getattr(n, "member", None)
                if qualifier and member:
                    calls.append(f"{qualifier}.{member}")
                elif member:
                    calls.append(member)

            # Recurse into children
            if hasattr(n, "children"):
                for child in n.children:
                    if isinstance(child, list):
                        for item in child:
                            visit(item)
                    else:
                        visit(child)

        if hasattr(node, "body") and node.body:
            for stmt in node.body:
                visit(stmt)

        return list(set(calls))  # Unique calls


# Registration

def register_java_backend() -> Optional[JavaASTBackend]:
    """Create and register the Java AST backend."""
    if not JAVALANG_AVAILABLE:
        return None
    backend = JavaASTBackend()
    register_backend(backend)
    return backend


# Auto-register on import if javalang is available
_java_backend: Optional[JavaASTBackend] = None
if JAVALANG_AVAILABLE:
    _java_backend = register_java_backend()


def get_java_backend() -> Optional[JavaASTBackend]:
    """Get the Java AST backend."""
    return _java_backend


def is_java_available() -> bool:
    """Check if Java AST parsing is available."""
    return JAVALANG_AVAILABLE
