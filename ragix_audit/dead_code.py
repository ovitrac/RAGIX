"""
Dead Code Detection Module — Identify Unreachable and Orphan Code

Detects potentially dead code through:
- Entry point discovery (main methods, controllers, event handlers)
- BFS reachability analysis from entry points
- Orphan package detection (no incoming dependencies)
- Unused class/method identification

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-11
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import re

logger = logging.getLogger(__name__)


class EntryPointType(Enum):
    """Types of application entry points."""
    MAIN_METHOD = "main_method"
    CONTROLLER = "controller"
    REST_ENDPOINT = "rest_endpoint"
    EVENT_HANDLER = "event_handler"
    SCHEDULED_TASK = "scheduled_task"
    MESSAGE_LISTENER = "message_listener"
    TEST_CLASS = "test_class"
    SPRING_BOOT = "spring_boot"


@dataclass
class EntryPoint:
    """An application entry point."""
    name: str
    file_path: str
    entry_type: EntryPointType
    line_number: int = 0
    confidence: float = 1.0  # 0-1, how confident we are this is an entry point

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file_path,
            "type": self.entry_type.value,
            "line": self.line_number,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class DeadCodeCandidate:
    """A potential dead code element."""
    name: str
    file_path: str
    element_type: str  # "class", "method", "package"
    line_number: int = 0
    reason: str = ""
    confidence: float = 0.5  # Higher = more likely dead

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file_path,
            "type": self.element_type,
            "line": self.line_number,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class DeadCodeAnalysis:
    """Complete dead code analysis results."""
    # Entry points discovered
    entry_points: List[EntryPoint] = field(default_factory=list)
    entry_points_by_type: Dict[str, int] = field(default_factory=dict)

    # Reachability
    reachable_classes: Set[str] = field(default_factory=set)
    reachable_packages: Set[str] = field(default_factory=set)
    total_classes: int = 0
    total_packages: int = 0

    # Dead code candidates
    dead_candidates: List[DeadCodeCandidate] = field(default_factory=list)
    orphan_packages: List[str] = field(default_factory=list)

    # Summary
    reachability_ratio: float = 0.0  # % of classes reachable from entry points
    estimated_dead_loc: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_points": {
                "total": len(self.entry_points),
                "by_type": self.entry_points_by_type,
                "list": [ep.to_dict() for ep in self.entry_points[:20]],
            },
            "reachability": {
                "reachable_classes": len(self.reachable_classes),
                "total_classes": self.total_classes,
                "ratio": round(self.reachability_ratio, 2),
                "reachable_packages": len(self.reachable_packages),
                "total_packages": self.total_packages,
            },
            "dead_code": {
                "candidates": len(self.dead_candidates),
                "orphan_packages": len(self.orphan_packages),
                "estimated_dead_loc": self.estimated_dead_loc,
                "candidates_list": [dc.to_dict() for dc in self.dead_candidates[:30]],
                "orphan_packages_list": self.orphan_packages[:20],
            },
        }


class EntryPointDetector:
    """Detects application entry points from code patterns."""

    # Patterns for entry point detection
    PATTERNS = {
        EntryPointType.MAIN_METHOD: [
            r'public\s+static\s+void\s+main\s*\(',
            r'def\s+main\s*\(',  # Python
        ],
        EntryPointType.CONTROLLER: [
            r'@Controller',
            r'@RestController',
            r'@RequestMapping',
            r'@WebServlet',
        ],
        EntryPointType.REST_ENDPOINT: [
            r'@GetMapping',
            r'@PostMapping',
            r'@PutMapping',
            r'@DeleteMapping',
            r'@RequestMapping.*method\s*=',
        ],
        EntryPointType.EVENT_HANDLER: [
            r'@EventListener',
            r'@KafkaListener',
            r'@RabbitListener',
            r'@JmsListener',
            r'implements\s+.*Listener',
        ],
        EntryPointType.SCHEDULED_TASK: [
            r'@Scheduled',
            r'@EnableScheduling',
            r'implements\s+.*Job',
        ],
        EntryPointType.MESSAGE_LISTENER: [
            r'@MessageMapping',
            r'@SubscribeMapping',
        ],
        EntryPointType.TEST_CLASS: [
            r'@Test',
            r'@RunWith',
            r'@SpringBootTest',
            r'extends\s+.*TestCase',
        ],
        EntryPointType.SPRING_BOOT: [
            r'@SpringBootApplication',
            r'@EnableAutoConfiguration',
        ],
    }

    def __init__(self):
        self.compiled_patterns: Dict[EntryPointType, List[re.Pattern]] = {}
        for ep_type, patterns in self.PATTERNS.items():
            self.compiled_patterns[ep_type] = [re.compile(p) for p in patterns]

    def detect_in_file(self, file_path: str, content: str) -> List[EntryPoint]:
        """Detect entry points in a source file."""
        entry_points = []
        lines = content.split('\n')

        for ep_type, patterns in self.compiled_patterns.items():
            for i, line in enumerate(lines, 1):
                for pattern in patterns:
                    if pattern.search(line):
                        # Extract name from context
                        name = self._extract_name(lines, i - 1, ep_type)
                        entry_points.append(EntryPoint(
                            name=name,
                            file_path=file_path,
                            entry_type=ep_type,
                            line_number=i,
                            confidence=self._compute_confidence(ep_type, line),
                        ))
                        break  # Found match for this type

        return entry_points

    def _extract_name(self, lines: List[str], line_idx: int, ep_type: EntryPointType) -> str:
        """Extract the name of the entry point."""
        # Look for class or method name near the annotation
        for offset in range(0, min(5, len(lines) - line_idx)):
            line = lines[line_idx + offset]
            # Class pattern
            class_match = re.search(r'class\s+(\w+)', line)
            if class_match:
                return class_match.group(1)
            # Method pattern
            method_match = re.search(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\(', line)
            if method_match:
                return method_match.group(1)

        return f"unknown_{ep_type.value}"

    def _compute_confidence(self, ep_type: EntryPointType, line: str) -> float:
        """Compute confidence score for entry point detection."""
        if ep_type == EntryPointType.MAIN_METHOD:
            return 1.0  # main() is always an entry point
        elif ep_type == EntryPointType.SPRING_BOOT:
            return 1.0
        elif ep_type in (EntryPointType.CONTROLLER, EntryPointType.REST_ENDPOINT):
            return 0.95
        elif ep_type == EntryPointType.TEST_CLASS:
            return 0.9
        else:
            return 0.8


class DeadCodeDetector:
    """Detects dead code through reachability analysis."""

    def __init__(self):
        self.entry_detector = EntryPointDetector()
        self.analysis: Optional[DeadCodeAnalysis] = None

    def analyze(
        self,
        file_contents: Dict[str, str],  # {file_path: content}
        dependencies: Dict[str, Set[str]],  # {class_name: {referenced_classes}}
        class_to_file: Optional[Dict[str, str]] = None,  # {class_name: file_path}
    ) -> DeadCodeAnalysis:
        """
        Perform complete dead code analysis.

        Args:
            file_contents: Map of file paths to their content
            dependencies: Class dependency graph
            class_to_file: Map of class names to file paths

        Returns:
            DeadCodeAnalysis with all results
        """
        analysis = DeadCodeAnalysis()

        # Step 1: Detect entry points
        for file_path, content in file_contents.items():
            entry_points = self.entry_detector.detect_in_file(file_path, content)
            analysis.entry_points.extend(entry_points)

        # Count by type
        for ep in analysis.entry_points:
            type_name = ep.entry_type.value
            analysis.entry_points_by_type[type_name] = analysis.entry_points_by_type.get(type_name, 0) + 1

        # Step 2: Build class set and package set
        all_classes = set(dependencies.keys())
        for targets in dependencies.values():
            all_classes.update(targets)
        analysis.total_classes = len(all_classes)

        all_packages = set()
        for cls in all_classes:
            pkg = self._extract_package(cls)
            if pkg:
                all_packages.add(pkg)
        analysis.total_packages = len(all_packages)

        # Step 3: BFS from entry points
        entry_classes = self._get_entry_point_classes(analysis.entry_points, class_to_file)
        analysis.reachable_classes = self._bfs_reachability(entry_classes, dependencies)

        # Also compute reachable packages
        for cls in analysis.reachable_classes:
            pkg = self._extract_package(cls)
            if pkg:
                analysis.reachable_packages.add(pkg)

        # Step 4: Compute reachability ratio
        if analysis.total_classes > 0:
            analysis.reachability_ratio = len(analysis.reachable_classes) / analysis.total_classes

        # Step 5: Identify dead code candidates
        unreachable = all_classes - analysis.reachable_classes
        for cls in unreachable:
            # Skip test classes
            if 'Test' in cls or 'test' in cls.lower():
                continue

            file_path = class_to_file.get(cls, "") if class_to_file else ""
            analysis.dead_candidates.append(DeadCodeCandidate(
                name=cls,
                file_path=file_path,
                element_type="class",
                reason="Not reachable from any entry point",
                confidence=0.7 if file_path else 0.5,
            ))

        # Step 6: Find orphan packages (no incoming dependencies)
        package_deps = self._build_package_dependencies(dependencies)
        for pkg in all_packages:
            if pkg not in package_deps or not package_deps[pkg]:
                # Check if it's not an entry point package
                is_entry_pkg = any(
                    self._extract_package(cls) == pkg for cls in entry_classes
                )
                if not is_entry_pkg:
                    analysis.orphan_packages.append(pkg)

        # Sort by confidence
        analysis.dead_candidates.sort(key=lambda x: x.confidence, reverse=True)

        self.analysis = analysis
        return analysis

    def _get_entry_point_classes(
        self,
        entry_points: List[EntryPoint],
        class_to_file: Optional[Dict[str, str]]
    ) -> Set[str]:
        """Get class names for entry points."""
        classes = set()

        # Create reverse mapping
        file_to_classes: Dict[str, List[str]] = {}
        if class_to_file:
            for cls, file_path in class_to_file.items():
                if file_path not in file_to_classes:
                    file_to_classes[file_path] = []
                file_to_classes[file_path].append(cls)

        for ep in entry_points:
            # If we have the mapping, use it
            if ep.file_path in file_to_classes:
                classes.update(file_to_classes[ep.file_path])
            else:
                # Use the entry point name as class name
                classes.add(ep.name)

        return classes

    def _bfs_reachability(
        self,
        start_nodes: Set[str],
        dependencies: Dict[str, Set[str]]
    ) -> Set[str]:
        """BFS to find all reachable classes from start nodes."""
        reachable = set()
        queue = deque(start_nodes)

        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            reachable.add(current)

            # Add all classes this one depends on
            if current in dependencies:
                for dep in dependencies[current]:
                    if dep not in reachable:
                        queue.append(dep)

        return reachable

    def _extract_package(self, class_name: str) -> Optional[str]:
        """Extract package name from fully qualified class name."""
        if '.' in class_name:
            parts = class_name.rsplit('.', 1)
            return parts[0] if len(parts) > 1 else None
        return None

    def _build_package_dependencies(
        self,
        dependencies: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Build package-level dependency graph (incoming deps)."""
        pkg_deps: Dict[str, Set[str]] = {}

        for source, targets in dependencies.items():
            source_pkg = self._extract_package(source)

            for target in targets:
                target_pkg = self._extract_package(target)
                if target_pkg and source_pkg != target_pkg:
                    if target_pkg not in pkg_deps:
                        pkg_deps[target_pkg] = set()
                    pkg_deps[target_pkg].add(source_pkg)

        return pkg_deps


def estimate_dead_code_from_file_metrics(
    file_metrics: List[Any],
    dead_classes: Set[str],
    class_to_file: Optional[Dict[str, str]] = None
) -> Tuple[int, int]:
    """
    Estimate LOC of dead code.

    Returns:
        Tuple of (estimated_dead_loc, total_loc)
    """
    dead_files = set()
    if class_to_file:
        for cls in dead_classes:
            if cls in class_to_file:
                dead_files.add(class_to_file[cls])

    dead_loc = 0
    total_loc = 0

    for fm in file_metrics:
        loc = getattr(fm, 'code_lines', 0)
        total_loc += loc

        if fm.path in dead_files:
            dead_loc += loc

    return dead_loc, total_loc
