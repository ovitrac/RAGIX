"""
Code Metrics - Professional code quality metrics calculation

Calculates:
- Cyclomatic complexity
- Cognitive complexity
- Lines of code (LOC, SLOC, comment lines)
- Method count per class
- Depth of inheritance
- Technical debt estimation
- Maintainability index

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ast_base import ASTNode, NodeType, Language
from .dependencies import DependencyGraph


class ComplexityLevel(str, Enum):
    """Complexity rating levels."""
    LOW = "low"           # 1-10
    MODERATE = "moderate" # 11-20
    HIGH = "high"         # 21-50
    VERY_HIGH = "very_high"  # 50+


@dataclass
class MethodMetrics:
    """Metrics for a single method/function."""
    name: str
    qualified_name: str
    file: str
    line: int
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    lines_of_code: int = 0
    parameter_count: int = 0
    return_statements: int = 0
    nested_depth: int = 0

    @property
    def complexity_level(self) -> ComplexityLevel:
        """Get complexity rating."""
        cc = self.cyclomatic_complexity
        if cc <= 10:
            return ComplexityLevel.LOW
        elif cc <= 20:
            return ComplexityLevel.MODERATE
        elif cc <= 50:
            return ComplexityLevel.HIGH
        return ComplexityLevel.VERY_HIGH

    @property
    def is_complex(self) -> bool:
        """Check if method exceeds complexity threshold."""
        return self.cyclomatic_complexity > 10

    @property
    def estimated_debt_minutes(self) -> int:
        """Estimate technical debt in minutes to fix."""
        debt = 0
        # High cyclomatic complexity
        if self.cyclomatic_complexity > 10:
            debt += (self.cyclomatic_complexity - 10) * 5
        # High cognitive complexity
        if self.cognitive_complexity > 15:
            debt += (self.cognitive_complexity - 15) * 3
        # Too many parameters
        if self.parameter_count > 5:
            debt += (self.parameter_count - 5) * 10
        # Deep nesting
        if self.nested_depth > 4:
            debt += (self.nested_depth - 4) * 15
        return debt


@dataclass
class ClassMetrics:
    """Metrics for a class."""
    name: str
    qualified_name: str
    file: str
    line: int
    lines_of_code: int = 0
    method_count: int = 0
    field_count: int = 0
    public_methods: int = 0
    private_methods: int = 0
    inheritance_depth: int = 0
    method_metrics: List[MethodMetrics] = field(default_factory=list)

    @property
    def total_complexity(self) -> int:
        """Sum of method complexities."""
        return sum(m.cyclomatic_complexity for m in self.method_metrics)

    @property
    def avg_method_complexity(self) -> float:
        """Average method complexity."""
        if not self.method_metrics:
            return 0.0
        return self.total_complexity / len(self.method_metrics)

    @property
    def max_method_complexity(self) -> int:
        """Maximum method complexity."""
        if not self.method_metrics:
            return 0
        return max(m.cyclomatic_complexity for m in self.method_metrics)

    @property
    def complex_methods(self) -> List[MethodMetrics]:
        """Methods with complexity > 10."""
        return [m for m in self.method_metrics if m.is_complex]

    @property
    def estimated_debt_minutes(self) -> int:
        """Total estimated technical debt."""
        debt = sum(m.estimated_debt_minutes for m in self.method_metrics)
        # Large class penalty
        if self.lines_of_code > 500:
            debt += (self.lines_of_code - 500) // 10
        # Too many methods
        if self.method_count > 20:
            debt += (self.method_count - 20) * 5
        return debt


@dataclass
class FileMetrics:
    """Metrics for a source file."""
    path: str
    language: Language
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    class_count: int = 0
    function_count: int = 0
    import_count: int = 0
    class_metrics: List[ClassMetrics] = field(default_factory=list)
    function_metrics: List[MethodMetrics] = field(default_factory=list)

    @property
    def comment_ratio(self) -> float:
        """Ratio of comment lines to code lines."""
        if self.code_lines == 0:
            return 0.0
        return self.comment_lines / self.code_lines

    @property
    def total_complexity(self) -> int:
        """Total cyclomatic complexity."""
        cc = sum(c.total_complexity for c in self.class_metrics)
        cc += sum(f.cyclomatic_complexity for f in self.function_metrics)
        return cc

    @property
    def estimated_debt_minutes(self) -> int:
        """Total technical debt estimate."""
        debt = sum(c.estimated_debt_minutes for c in self.class_metrics)
        debt += sum(f.estimated_debt_minutes for f in self.function_metrics)
        return debt


@dataclass
class ProjectMetrics:
    """Aggregated metrics for a project."""
    name: str
    root_path: str
    file_metrics: List[FileMetrics] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.file_metrics)

    @property
    def total_lines(self) -> int:
        return sum(f.total_lines for f in self.file_metrics)

    @property
    def total_code_lines(self) -> int:
        return sum(f.code_lines for f in self.file_metrics)

    @property
    def total_comment_lines(self) -> int:
        return sum(f.comment_lines for f in self.file_metrics)

    @property
    def total_classes(self) -> int:
        return sum(f.class_count for f in self.file_metrics)

    @property
    def total_functions(self) -> int:
        return sum(f.function_count for f in self.file_metrics)

    @property
    def total_complexity(self) -> int:
        return sum(f.total_complexity for f in self.file_metrics)

    @property
    def avg_complexity_per_method(self) -> float:
        methods = []
        for fm in self.file_metrics:
            for cm in fm.class_metrics:
                methods.extend(cm.method_metrics)
            methods.extend(fm.function_metrics)
        if not methods:
            return 0.0
        return sum(m.cyclomatic_complexity for m in methods) / len(methods)

    @property
    def estimated_debt_hours(self) -> float:
        """Total technical debt in hours."""
        minutes = sum(f.estimated_debt_minutes for f in self.file_metrics)
        return minutes / 60.0

    @property
    def estimated_debt_days(self) -> float:
        """Total technical debt in person-days (8h)."""
        return self.estimated_debt_hours / 8.0

    @property
    def maintainability_index(self) -> float:
        """
        Calculate Maintainability Index (0-100 scale, higher is better).
        Based on Visual Studio formula.
        """
        if self.total_code_lines == 0:
            return 100.0

        loc = self.total_code_lines
        cc = self.total_complexity

        # Simplified MI calculation
        # MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC)
        # We use a simplified version
        import math
        halstead_volume = loc * math.log2(max(loc, 1))  # Simplified
        mi = 171 - 5.2 * math.log(max(halstead_volume, 1)) - 0.23 * cc - 16.2 * math.log(max(loc, 1))

        # Normalize to 0-100
        mi = max(0, min(100, mi * 100 / 171))
        return round(mi, 1)

    def get_hotspots(self, limit: int = 20) -> List[Tuple[str, int]]:
        """Get most complex methods/functions."""
        all_methods = []
        for fm in self.file_metrics:
            for cm in fm.class_metrics:
                for mm in cm.method_metrics:
                    all_methods.append((f"{mm.qualified_name} ({fm.path}:{mm.line})", mm.cyclomatic_complexity))
            for ff in fm.function_metrics:
                all_methods.append((f"{ff.qualified_name} ({fm.path}:{ff.line})", ff.cyclomatic_complexity))

        all_methods.sort(key=lambda x: -x[1])
        return all_methods[:limit]

    def summary(self) -> Dict[str, Any]:
        """Get summary metrics."""
        return {
            "files": self.total_files,
            "lines": {
                "total": self.total_lines,
                "code": self.total_code_lines,
                "comments": self.total_comment_lines,
            },
            "classes": self.total_classes,
            "functions": self.total_functions,
            "complexity": {
                "total": self.total_complexity,
                "avg_per_method": round(self.avg_complexity_per_method, 2),
            },
            "technical_debt": {
                "hours": round(self.estimated_debt_hours, 1),
                "days": round(self.estimated_debt_days, 1),
            },
            "maintainability_index": self.maintainability_index,
        }


class PythonMetricsCalculator:
    """Calculate metrics for Python code."""

    def calculate_file_metrics(self, path: Path) -> FileMetrics:
        """Calculate metrics for a Python file."""
        content = path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")

        metrics = FileMetrics(
            path=str(path),
            language=Language.PYTHON,
            total_lines=len(lines),
        )

        # Count line types
        in_multiline_string = False
        for line in lines:
            stripped = line.strip()

            # Track multiline strings (used as comments)
            if '"""' in stripped or "'''" in stripped:
                in_multiline_string = not in_multiline_string
                metrics.comment_lines += 1
                continue

            if in_multiline_string:
                metrics.comment_lines += 1
            elif not stripped:
                metrics.blank_lines += 1
            elif stripped.startswith("#"):
                metrics.comment_lines += 1
            else:
                metrics.code_lines += 1

        # Parse AST for detailed metrics
        try:
            tree = ast.parse(content)
            self._analyze_ast(tree, metrics, path)
        except SyntaxError:
            pass

        return metrics

    def _analyze_ast(self, tree: ast.Module, metrics: FileMetrics, path: Path) -> None:
        """Analyze AST for metrics."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                metrics.import_count += len(node.names)
            elif isinstance(node, ast.ImportFrom):
                metrics.import_count += len(node.names)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                metrics.class_count += 1
                class_metrics = self._analyze_class(node, path)
                metrics.class_metrics.append(class_metrics)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics.function_count += 1
                func_metrics = self._analyze_function(node, path, "")
                metrics.function_metrics.append(func_metrics)

    def _analyze_class(self, node: ast.ClassDef, path: Path) -> ClassMetrics:
        """Analyze a class definition."""
        metrics = ClassMetrics(
            name=node.name,
            qualified_name=node.name,
            file=str(path),
            line=node.lineno,
        )

        # Count lines (approximate)
        if hasattr(node, "end_lineno") and node.end_lineno:
            metrics.lines_of_code = node.end_lineno - node.lineno + 1

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics.method_count += 1
                method_metrics = self._analyze_function(item, path, node.name)
                metrics.method_metrics.append(method_metrics)

                # Count visibility
                if item.name.startswith("_"):
                    metrics.private_methods += 1
                else:
                    metrics.public_methods += 1

            elif isinstance(item, ast.Assign):
                metrics.field_count += len(item.targets)
            elif isinstance(item, ast.AnnAssign):
                metrics.field_count += 1

        return metrics

    def _analyze_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        path: Path,
        class_name: str,
    ) -> MethodMetrics:
        """Analyze a function/method."""
        qualified = f"{class_name}.{node.name}" if class_name else node.name

        metrics = MethodMetrics(
            name=node.name,
            qualified_name=qualified,
            file=str(path),
            line=node.lineno,
            parameter_count=len(node.args.args),
        )

        # Calculate lines
        if hasattr(node, "end_lineno") and node.end_lineno:
            metrics.lines_of_code = node.end_lineno - node.lineno + 1

        # Calculate cyclomatic complexity
        metrics.cyclomatic_complexity = self._calculate_cyclomatic(node)

        # Calculate cognitive complexity
        metrics.cognitive_complexity = self._calculate_cognitive(node)

        # Count returns
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                metrics.return_statements += 1

        # Calculate max nesting depth
        metrics.nested_depth = self._calculate_nesting_depth(node)

        return metrics

    def _calculate_cyclomatic(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                if child.ifs:
                    complexity += len(child.ifs)
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.Match):  # Python 3.10+
                complexity += 1
            elif isinstance(child, ast.match_case):
                complexity += 1

        return complexity

    def _calculate_cognitive(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate cognitive complexity (simplified)."""
        complexity = 0

        for child in ast.iter_child_nodes(node):
            # Structural complexity (increases with nesting)
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + depth
                complexity += self._calculate_cognitive(child, depth + 1)
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1 + depth
                complexity += self._calculate_cognitive(child, depth + 1)
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.Lambda):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
            else:
                complexity += self._calculate_cognitive(child, depth)

        return complexity

    def _calculate_nesting_depth(self, node: ast.AST, current: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = current

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child, current + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_nesting_depth(child, current)
                max_depth = max(max_depth, child_depth)

        return max_depth


class JavaMetricsCalculator:
    """Calculate metrics for Java code using AST nodes."""

    def calculate_from_ast(self, ast_node: ASTNode) -> FileMetrics:
        """Calculate metrics from an ASTNode."""
        metrics = FileMetrics(
            path=str(ast_node.location.file) if ast_node.location.file else "unknown",
            language=Language.JAVA,
        )

        # Count imports
        for child in ast_node.children:
            if child.node_type in (NodeType.IMPORT, NodeType.IMPORT_FROM):
                metrics.import_count += 1

        # Analyze classes
        for child in ast_node.children:
            if child.node_type in (NodeType.CLASS, NodeType.INTERFACE, NodeType.ENUM):
                metrics.class_count += 1
                class_metrics = self._analyze_class_node(child)
                metrics.class_metrics.append(class_metrics)

        return metrics

    def _analyze_class_node(self, node: ASTNode) -> ClassMetrics:
        """Analyze a class ASTNode."""
        metrics = ClassMetrics(
            name=node.name,
            qualified_name=node.get_qualified_name(),
            file=str(node.location.file) if node.location.file else "unknown",
            line=node.location.line,
        )

        # Calculate lines
        if node.location.end_line:
            metrics.lines_of_code = node.location.end_line - node.location.line + 1

        for child in node.children:
            if child.node_type in (NodeType.METHOD, NodeType.CONSTRUCTOR):
                metrics.method_count += 1
                method_metrics = self._analyze_method_node(child)
                metrics.method_metrics.append(method_metrics)

                if child.visibility.value == "public":
                    metrics.public_methods += 1
                elif child.visibility.value in ("private", "protected"):
                    metrics.private_methods += 1

            elif child.node_type in (NodeType.FIELD, NodeType.CONSTANT):
                metrics.field_count += 1

        return metrics

    def _analyze_method_node(self, node: ASTNode) -> MethodMetrics:
        """Analyze a method ASTNode."""
        metrics = MethodMetrics(
            name=node.name,
            qualified_name=node.get_qualified_name(),
            file=str(node.location.file) if node.location.file else "unknown",
            line=node.location.line,
            parameter_count=len(node.parameters),
        )

        # Calculate lines
        if node.location.end_line:
            metrics.lines_of_code = node.location.end_line - node.location.line + 1

        # Estimate complexity from metadata
        calls = node.metadata.get("calls", [])
        metrics.cyclomatic_complexity = 1 + len(calls) // 5  # Rough estimate

        return metrics


def calculate_project_metrics(
    path: Path,
    language: Optional[Language] = None,
) -> ProjectMetrics:
    """
    Calculate metrics for an entire project.

    Args:
        path: Project root path
        language: Specific language to analyze (None for all)

    Returns:
        ProjectMetrics with aggregated data
    """
    metrics = ProjectMetrics(
        name=path.name,
        root_path=str(path),
    )

    python_calc = PythonMetricsCalculator()

    # Find and analyze files
    patterns = []
    if language is None or language == Language.PYTHON:
        patterns.append("**/*.py")
    if language is None or language == Language.JAVA:
        patterns.append("**/*.java")

    for pattern in patterns:
        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    if file_path.suffix == ".py":
                        file_metrics = python_calc.calculate_file_metrics(file_path)
                        metrics.file_metrics.append(file_metrics)
                except Exception:
                    continue

    return metrics


def calculate_metrics_from_graph(graph: DependencyGraph) -> ProjectMetrics:
    """Calculate metrics from an existing dependency graph."""
    metrics = ProjectMetrics(
        name="Project",
        root_path="",
    )

    java_calc = JavaMetricsCalculator()

    # Group by file
    files_processed = set()
    for file_path, ast_node in graph._files.items():
        if str(file_path) in files_processed:
            continue
        files_processed.add(str(file_path))

        if file_path.suffix == ".java":
            file_metrics = java_calc.calculate_from_ast(ast_node)

            # Add line counts from file
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                file_metrics.total_lines = len(lines)

                in_multiline = False
                for line in lines:
                    stripped = line.strip()
                    if "/*" in stripped:
                        in_multiline = True
                    if "*/" in stripped:
                        in_multiline = False
                        file_metrics.comment_lines += 1
                        continue

                    if in_multiline or stripped.startswith("//") or stripped.startswith("*"):
                        file_metrics.comment_lines += 1
                    elif not stripped:
                        file_metrics.blank_lines += 1
                    else:
                        file_metrics.code_lines += 1
            except Exception:
                pass

            metrics.file_metrics.append(file_metrics)

    return metrics
