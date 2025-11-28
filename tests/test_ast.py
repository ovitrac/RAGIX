"""
Tests for AST analysis modules

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

import tempfile
from pathlib import Path

import pytest

from ragix_core.ast_base import (
    ASTNode,
    Language,
    NodeType,
    SourceLocation,
    Symbol,
    TypeInfo,
    Visibility,
    format_ast_tree,
    get_ast_registry,
)
from ragix_core.ast_python import PythonASTBackend, get_python_backend
from ragix_core.ast_query import (
    ASTQuery,
    QueryPredicate,
    QueryType,
    execute_query,
    execute_query_on_ast,
    parse_query,
)
from ragix_core.dependencies import (
    Dependency,
    DependencyGraph,
    DependencyType,
    build_dependency_graph,
)

try:
    from ragix_core.ast_java import JavaASTBackend, get_java_backend, is_java_available
    JAVA_AVAILABLE = is_java_available()
except ImportError:
    JAVA_AVAILABLE = False


# ========== AST Base Tests ==========

class TestSourceLocation:
    def test_creation(self):
        loc = SourceLocation(Path("test.py"), 10, 5)
        assert loc.file == Path("test.py")
        assert loc.line == 10
        assert loc.column == 5

    def test_str_representation(self):
        loc = SourceLocation(Path("test.py"), 10, 5)
        assert str(loc) == "test.py:10:5"

    def test_str_without_column(self):
        loc = SourceLocation(Path("test.py"), 10)
        assert str(loc) == "test.py:10"


class TestTypeInfo:
    def test_simple_type(self):
        t = TypeInfo(name="str")
        assert str(t) == "str"

    def test_array_type(self):
        t = TypeInfo(name="int", is_array=True)
        assert t.is_array

    def test_generic_type(self):
        inner = TypeInfo(name="str")
        t = TypeInfo(name="List", is_generic=True, type_arguments=[inner])
        assert t.is_generic
        assert len(t.type_arguments) == 1
        assert "List" in str(t)


class TestASTNode:
    def test_creation(self):
        node = ASTNode(
            node_type=NodeType.CLASS,
            name="TestClass",
            location=SourceLocation(Path("test.py"), 1),
        )
        assert node.node_type == NodeType.CLASS
        assert node.name == "TestClass"

    def test_children(self):
        parent = ASTNode(
            node_type=NodeType.CLASS,
            name="Parent",
            location=SourceLocation(Path("test.py"), 1),
        )
        child = ASTNode(
            node_type=NodeType.METHOD,
            name="method",
            location=SourceLocation(Path("test.py"), 5),
        )
        parent.children.append(child)
        assert len(parent.children) == 1

    def test_qualified_name(self):
        node = ASTNode(
            node_type=NodeType.METHOD,
            name="test_method",
            location=SourceLocation(Path("test.py"), 1),
            metadata={"qualified_parts": ["module", "TestClass", "test_method"]},
        )
        assert node.get_qualified_name() == "module.TestClass.test_method"

    def test_to_symbol(self):
        node = ASTNode(
            node_type=NodeType.CLASS,
            name="MyClass",
            location=SourceLocation(Path("test.py"), 1),
            visibility=Visibility.PUBLIC,
        )
        symbol = node.to_symbol(Language.PYTHON)
        assert symbol.name == "MyClass"
        assert symbol.node_type == NodeType.CLASS


class TestASTRegistry:
    def test_get_registry(self):
        registry = get_ast_registry()
        assert registry is not None

    def test_python_backend_registered(self):
        registry = get_ast_registry()
        backend = registry.get_backend(Language.PYTHON)
        assert backend is not None
        assert isinstance(backend, PythonASTBackend)

    def test_list_languages(self):
        registry = get_ast_registry()
        langs = registry.list_languages()
        assert Language.PYTHON in langs


# ========== Python AST Backend Tests ==========

class TestPythonASTBackend:
    @pytest.fixture
    def backend(self):
        return get_python_backend()

    def test_language(self, backend):
        assert backend.language == Language.PYTHON

    def test_file_extensions(self, backend):
        assert ".py" in backend.file_extensions

    def test_parse_simple_function(self, backend):
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        ast = backend.parse_string(code, "test.py")
        assert ast.node_type == NodeType.MODULE

        # Find the function
        func = None
        for child in ast.children:
            if child.node_type == NodeType.FUNCTION and child.name == "hello":
                func = child
                break

        assert func is not None
        assert func.docstring == "Say hello."
        assert func.return_type.name == "str"
        assert len(func.parameters) == 1

    def test_parse_class(self, backend):
        code = '''
class MyClass:
    """A test class."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        return self.value
'''
        ast = backend.parse_string(code, "test.py")

        # Find the class
        cls = None
        for child in ast.children:
            if child.node_type == NodeType.CLASS:
                cls = child
                break

        assert cls is not None
        assert cls.name == "MyClass"
        assert cls.docstring == "A test class."

        # Check methods
        method_names = [c.name for c in cls.children if c.node_type in (NodeType.METHOD, NodeType.CONSTRUCTOR)]
        assert "__init__" in method_names
        assert "get_value" in method_names

    def test_parse_imports(self, backend):
        code = '''
import os
from pathlib import Path
from typing import List, Optional
'''
        ast = backend.parse_string(code, "test.py")

        imports = [c for c in ast.children if c.node_type in (NodeType.IMPORT, NodeType.IMPORT_FROM)]
        assert len(imports) == 3

    def test_parse_decorators(self, backend):
        code = '''
class MyClass:
    @property
    def value(self):
        return self._value

    @staticmethod
    def create():
        return MyClass()
'''
        ast = backend.parse_string(code, "test.py")

        cls = ast.children[0]
        methods = [c for c in cls.children if c.node_type in (NodeType.METHOD, NodeType.PROPERTY)]

        prop = next((m for m in methods if m.name == "value"), None)
        assert prop is not None
        assert prop.node_type == NodeType.PROPERTY

        static = next((m for m in methods if m.name == "create"), None)
        assert static is not None
        assert "static" in static.modifiers

    def test_parse_inheritance(self, backend):
        code = '''
class Child(Parent):
    pass
'''
        ast = backend.parse_string(code, "test.py")
        cls = ast.children[0]
        assert cls.extends == "Parent"

    def test_get_symbols(self, backend):
        code = '''
class TestClass:
    def method(self):
        pass

def standalone_function():
    pass
'''
        ast = backend.parse_string(code, "test.py")
        symbols = backend.get_symbols(ast)

        names = [s.name for s in symbols]
        assert "TestClass" in names
        assert "standalone_function" in names


# ========== Java AST Backend Tests ==========

@pytest.mark.skipif(not JAVA_AVAILABLE, reason="javalang not available")
class TestJavaASTBackend:
    @pytest.fixture
    def backend(self):
        return get_java_backend()

    def test_language(self, backend):
        assert backend.language == Language.JAVA

    def test_file_extensions(self, backend):
        assert ".java" in backend.file_extensions

    def test_parse_simple_class(self, backend):
        code = '''
package com.example;

public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
'''
        ast = backend.parse_string(code, "HelloWorld.java")
        assert ast.node_type == NodeType.MODULE

        # Find the class
        cls = None
        for child in ast.children:
            if child.node_type == NodeType.CLASS:
                cls = child
                break

        assert cls is not None
        assert cls.name == "HelloWorld"
        assert cls.visibility == Visibility.PUBLIC

    def test_parse_interface(self, backend):
        code = '''
package com.example;

public interface Service {
    void execute();
    String getName();
}
'''
        ast = backend.parse_string(code, "Service.java")

        iface = None
        for child in ast.children:
            if child.node_type == NodeType.INTERFACE:
                iface = child
                break

        assert iface is not None
        assert iface.name == "Service"

    def test_parse_annotations(self, backend):
        code = '''
package com.example;

import org.springframework.stereotype.Service;

@Service
public class MyService {
    @Override
    public String toString() {
        return "MyService";
    }
}
'''
        ast = backend.parse_string(code, "MyService.java")

        cls = None
        for child in ast.children:
            if child.node_type == NodeType.CLASS:
                cls = child
                break

        assert cls is not None
        assert "Service" in cls.decorators

    def test_parse_extends_implements(self, backend):
        code = '''
package com.example;

public class MyClass extends BaseClass implements Interface1, Interface2 {
}
'''
        ast = backend.parse_string(code, "MyClass.java")

        cls = None
        for child in ast.children:
            if child.node_type == NodeType.CLASS:
                cls = child
                break

        assert cls is not None
        assert cls.extends == "BaseClass"
        assert "Interface1" in cls.implements
        assert "Interface2" in cls.implements


# ========== Dependency Graph Tests ==========

class TestDependencyGraph:
    def test_empty_graph(self):
        graph = DependencyGraph()
        assert len(graph) == 0
        assert len(graph.get_symbols()) == 0

    def test_add_python_file(self):
        code = '''
class MyClass:
    def method(self):
        pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = DependencyGraph()
            ast = graph.add_file(Path(f.name))

            assert ast is not None
            assert len(graph.get_symbols()) > 0

    def test_find_symbols(self):
        code = '''
class MyService:
    pass

class OtherClass:
    pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = DependencyGraph()
            graph.add_file(Path(f.name))

            matches = graph.find_symbols("*Service")
            assert any("MyService" in s.name for s in matches)

    def test_dependency_extraction(self):
        code = '''
from typing import List

class Parent:
    pass

class Child(Parent):
    items: List[str]
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = DependencyGraph()
            graph.add_file(Path(f.name))

            deps = graph.get_all_dependencies()

            # Should have inheritance dependency
            inheritance_deps = [d for d in deps if d.dep_type == DependencyType.INHERITANCE]
            assert len(inheritance_deps) > 0


class TestDependencyStats:
    def test_stats_calculation(self):
        code = '''
class A:
    pass

class B(A):
    pass

class C(A):
    pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = DependencyGraph()
            graph.add_file(Path(f.name))

            stats = graph.get_stats()
            assert stats.total_dependencies >= 0


# ========== Query Language Tests ==========

class TestQueryParser:
    def test_parse_type_query(self):
        query = parse_query("type:class")
        assert len(query.predicates) == 1
        assert query.predicates[0].query_type == QueryType.TYPE

    def test_parse_name_query(self):
        query = parse_query("name:*Service")
        assert query.predicates[0].query_type == QueryType.NAME
        assert query.predicates[0].pattern == "*Service"

    def test_parse_annotation_query(self):
        query = parse_query("@Override")
        assert query.predicates[0].query_type == QueryType.ANNOTATION
        assert query.predicates[0].pattern == "Override"

    def test_parse_negated_query(self):
        query = parse_query("!visibility:private")
        assert query.predicates[0].negated

    def test_parse_compound_query(self):
        query = parse_query("type:class name:*Service")
        assert len(query.predicates) == 2

    def test_parse_extends_query(self):
        query = parse_query("extends:Base*")
        assert query.predicates[0].query_type == QueryType.EXTENDS


class TestQueryExecution:
    def test_execute_type_query(self):
        code = '''
class MyClass:
    def method(self):
        pass

def my_function():
    pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = DependencyGraph()
            graph.add_file(Path(f.name))

            query = parse_query("type:class")
            matches = execute_query(query, graph)

            assert any(m.symbol.node_type == NodeType.CLASS for m in matches if m.symbol)

    def test_execute_name_query(self):
        code = '''
class UserService:
    pass

class ProductService:
    pass

class OtherClass:
    pass
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = DependencyGraph()
            graph.add_file(Path(f.name))

            query = parse_query("name:*Service")
            matches = execute_query(query, graph)

            names = [m.symbol.name for m in matches if m.symbol]
            assert "UserService" in names
            assert "ProductService" in names
            assert "OtherClass" not in names


# ========== AST Visualization Tests ==========

class TestVisualization:
    def test_format_ast_tree(self):
        node = ASTNode(
            node_type=NodeType.CLASS,
            name="TestClass",
            location=SourceLocation(Path("test.py"), 1),
        )
        child = ASTNode(
            node_type=NodeType.METHOD,
            name="method",
            location=SourceLocation(Path("test.py"), 5),
        )
        node.children.append(child)

        tree_str = format_ast_tree(node)
        assert "TestClass" in tree_str
        assert "method" in tree_str


# ========== Maven Integration Tests ==========

class TestMavenParser:
    def test_parse_simple_pom(self):
        from ragix_core.maven import parse_pom, MavenParser

        pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-app</artifactId>
    <version>1.0.0</version>

    <dependencies>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(pom_content)
            f.flush()

            project = parse_pom(Path(f.name))

            assert project.coordinate.group_id == "com.example"
            assert project.coordinate.artifact_id == "test-app"
            assert len(project.dependencies) == 1
            assert project.dependencies[0].is_test

    def test_property_resolution(self):
        from ragix_core.maven import MavenParser

        pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-app</artifactId>
    <version>1.0.0</version>

    <properties>
        <spring.version>5.3.0</spring.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-core</artifactId>
            <version>${spring.version}</version>
        </dependency>
    </dependencies>
</project>'''

        parser = MavenParser()
        project = parser.parse_string(pom_content)

        dep = project.dependencies[0]
        assert dep.coordinate.version == "5.3.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
