"""
Maven Integration - Parse POM files and extract project structure

Extracts:
- Project metadata (groupId, artifactId, version)
- Dependencies with scopes
- Modules for multi-module projects
- Parent POMs
- Plugin configuration
- Property resolution

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-27
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class DependencyScope(str, Enum):
    """Maven dependency scopes."""
    COMPILE = "compile"
    PROVIDED = "provided"
    RUNTIME = "runtime"
    TEST = "test"
    SYSTEM = "system"
    IMPORT = "import"


@dataclass
class MavenCoordinate:
    """Maven artifact coordinates (GAV)."""
    group_id: str
    artifact_id: str
    version: Optional[str] = None
    classifier: Optional[str] = None
    packaging: str = "jar"

    @property
    def gav(self) -> str:
        """Get groupId:artifactId:version string."""
        v = self.version or "?"
        return f"{self.group_id}:{self.artifact_id}:{v}"

    def __str__(self) -> str:
        return self.gav

    def __hash__(self) -> int:
        return hash((self.group_id, self.artifact_id, self.version))


@dataclass
class MavenDependency:
    """A Maven dependency declaration."""
    coordinate: MavenCoordinate
    scope: DependencyScope = DependencyScope.COMPILE
    optional: bool = False
    exclusions: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def is_test(self) -> bool:
        return self.scope == DependencyScope.TEST

    @property
    def is_provided(self) -> bool:
        return self.scope == DependencyScope.PROVIDED


@dataclass
class MavenPlugin:
    """A Maven plugin declaration."""
    coordinate: MavenCoordinate
    executions: List[Dict[str, Any]] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MavenProject:
    """Represents a Maven project (parsed POM)."""
    coordinate: MavenCoordinate
    parent: Optional[MavenCoordinate] = None
    name: Optional[str] = None
    description: Optional[str] = None
    packaging: str = "jar"
    properties: Dict[str, str] = field(default_factory=dict)
    dependencies: List[MavenDependency] = field(default_factory=list)
    dependency_management: List[MavenDependency] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)
    plugins: List[MavenPlugin] = field(default_factory=list)
    pom_path: Optional[Path] = None
    source_directories: List[Path] = field(default_factory=list)
    test_directories: List[Path] = field(default_factory=list)

    @property
    def is_multi_module(self) -> bool:
        """Check if this is a multi-module (parent) project."""
        return len(self.modules) > 0 or self.packaging == "pom"

    def get_dependency(self, group_id: str, artifact_id: str) -> Optional[MavenDependency]:
        """Find a dependency by GAV."""
        for dep in self.dependencies:
            if dep.coordinate.group_id == group_id and dep.coordinate.artifact_id == artifact_id:
                return dep
        return None

    def get_test_dependencies(self) -> List[MavenDependency]:
        """Get all test-scope dependencies."""
        return [d for d in self.dependencies if d.is_test]

    def get_compile_dependencies(self) -> List[MavenDependency]:
        """Get all compile-scope dependencies."""
        return [d for d in self.dependencies if d.scope == DependencyScope.COMPILE]


class MavenParser:
    """Parse Maven POM files."""

    MAVEN_NS = "{http://maven.apache.org/POM/4.0.0}"

    def __init__(self):
        self._property_pattern = re.compile(r"\$\{([^}]+)\}")

    def parse_file(self, path: Path) -> MavenProject:
        """Parse a POM file."""
        tree = ET.parse(path)
        root = tree.getroot()
        return self._parse_project(root, path)

    def parse_string(self, content: str, path: Optional[Path] = None) -> MavenProject:
        """Parse POM content from string."""
        root = ET.fromstring(content)
        return self._parse_project(root, path)

    def _parse_project(self, root: ET.Element, pom_path: Optional[Path]) -> MavenProject:
        """Parse a project element."""
        ns = self.MAVEN_NS

        # Extract properties first for variable resolution
        properties = self._parse_properties(root, ns)

        # Parse parent if exists
        parent = None
        parent_elem = root.find(f"{ns}parent")
        if parent_elem is not None:
            parent = MavenCoordinate(
                group_id=self._get_text(parent_elem, f"{ns}groupId", ""),
                artifact_id=self._get_text(parent_elem, f"{ns}artifactId", ""),
                version=self._get_text(parent_elem, f"{ns}version"),
            )

        # Parse coordinate (inherit from parent if not specified)
        group_id = self._get_text(root, f"{ns}groupId")
        if not group_id and parent:
            group_id = parent.group_id

        artifact_id = self._get_text(root, f"{ns}artifactId", "unknown")

        version = self._get_text(root, f"{ns}version")
        if not version and parent:
            version = parent.version

        # Resolve property references
        version = self._resolve_properties(version, properties)

        coordinate = MavenCoordinate(
            group_id=group_id or "",
            artifact_id=artifact_id,
            version=version,
        )

        # Parse packaging
        packaging = self._get_text(root, f"{ns}packaging", "jar")

        # Parse name and description
        name = self._get_text(root, f"{ns}name")
        description = self._get_text(root, f"{ns}description")

        # Parse modules
        modules = []
        modules_elem = root.find(f"{ns}modules")
        if modules_elem is not None:
            for module in modules_elem.findall(f"{ns}module"):
                if module.text:
                    modules.append(module.text.strip())

        # Parse dependencies
        dependencies = self._parse_dependencies(root, ns, properties)

        # Parse dependency management
        dep_mgmt = []
        dep_mgmt_elem = root.find(f"{ns}dependencyManagement")
        if dep_mgmt_elem is not None:
            dep_mgmt = self._parse_dependencies(dep_mgmt_elem, ns, properties)

        # Parse plugins
        plugins = self._parse_plugins(root, ns, properties)

        # Determine source directories
        source_dirs = []
        test_dirs = []
        if pom_path:
            project_dir = pom_path.parent
            # Default Maven conventions
            src_main = project_dir / "src" / "main" / "java"
            src_test = project_dir / "src" / "test" / "java"
            if src_main.exists():
                source_dirs.append(src_main)
            if src_test.exists():
                test_dirs.append(src_test)

        return MavenProject(
            coordinate=coordinate,
            parent=parent,
            name=name,
            description=description,
            packaging=packaging,
            properties=properties,
            dependencies=dependencies,
            dependency_management=dep_mgmt,
            modules=modules,
            plugins=plugins,
            pom_path=pom_path,
            source_directories=source_dirs,
            test_directories=test_dirs,
        )

    def _parse_properties(self, root: ET.Element, ns: str) -> Dict[str, str]:
        """Parse properties section."""
        props = {}
        props_elem = root.find(f"{ns}properties")
        if props_elem is not None:
            for child in props_elem:
                # Remove namespace prefix from tag
                tag = child.tag.replace(ns, "")
                if child.text:
                    props[tag] = child.text.strip()
        return props

    def _parse_dependencies(
        self,
        parent: ET.Element,
        ns: str,
        properties: Dict[str, str],
    ) -> List[MavenDependency]:
        """Parse dependencies section."""
        deps = []
        deps_elem = parent.find(f"{ns}dependencies")
        if deps_elem is None:
            return deps

        for dep_elem in deps_elem.findall(f"{ns}dependency"):
            group_id = self._get_text(dep_elem, f"{ns}groupId", "")
            artifact_id = self._get_text(dep_elem, f"{ns}artifactId", "")
            version = self._get_text(dep_elem, f"{ns}version")
            scope_str = self._get_text(dep_elem, f"{ns}scope", "compile")
            optional_str = self._get_text(dep_elem, f"{ns}optional", "false")
            classifier = self._get_text(dep_elem, f"{ns}classifier")

            # Resolve properties
            group_id = self._resolve_properties(group_id, properties)
            artifact_id = self._resolve_properties(artifact_id, properties)
            version = self._resolve_properties(version, properties)

            # Parse scope
            try:
                scope = DependencyScope(scope_str.lower())
            except ValueError:
                scope = DependencyScope.COMPILE

            # Parse exclusions
            exclusions = []
            exclusions_elem = dep_elem.find(f"{ns}exclusions")
            if exclusions_elem is not None:
                for exc in exclusions_elem.findall(f"{ns}exclusion"):
                    exc_group = self._get_text(exc, f"{ns}groupId", "*")
                    exc_artifact = self._get_text(exc, f"{ns}artifactId", "*")
                    exclusions.append((exc_group, exc_artifact))

            coord = MavenCoordinate(
                group_id=group_id,
                artifact_id=artifact_id,
                version=version,
                classifier=classifier,
            )

            deps.append(MavenDependency(
                coordinate=coord,
                scope=scope,
                optional=optional_str.lower() == "true",
                exclusions=exclusions,
            ))

        return deps

    def _parse_plugins(
        self,
        root: ET.Element,
        ns: str,
        properties: Dict[str, str],
    ) -> List[MavenPlugin]:
        """Parse build plugins."""
        plugins = []
        build_elem = root.find(f"{ns}build")
        if build_elem is None:
            return plugins

        plugins_elem = build_elem.find(f"{ns}plugins")
        if plugins_elem is None:
            return plugins

        for plugin_elem in plugins_elem.findall(f"{ns}plugin"):
            group_id = self._get_text(plugin_elem, f"{ns}groupId", "org.apache.maven.plugins")
            artifact_id = self._get_text(plugin_elem, f"{ns}artifactId", "")
            version = self._get_text(plugin_elem, f"{ns}version")

            # Resolve properties
            version = self._resolve_properties(version, properties)

            coord = MavenCoordinate(
                group_id=group_id,
                artifact_id=artifact_id,
                version=version,
            )

            # Parse configuration (simplified - just get direct children)
            config = {}
            config_elem = plugin_elem.find(f"{ns}configuration")
            if config_elem is not None:
                for child in config_elem:
                    tag = child.tag.replace(ns, "")
                    config[tag] = child.text if child.text else ""

            plugins.append(MavenPlugin(
                coordinate=coord,
                configuration=config,
            ))

        return plugins

    def _get_text(
        self,
        elem: ET.Element,
        path: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Get text content of a child element."""
        child = elem.find(path)
        if child is not None and child.text:
            return child.text.strip()
        return default

    def _resolve_properties(
        self,
        value: Optional[str],
        properties: Dict[str, str],
    ) -> Optional[str]:
        """Resolve ${property} references in a value."""
        if not value:
            return value

        def replacer(match):
            prop_name = match.group(1)
            # Check for built-in properties
            if prop_name == "project.version":
                return properties.get("version", match.group(0))
            if prop_name == "project.groupId":
                return properties.get("groupId", match.group(0))
            return properties.get(prop_name, match.group(0))

        return self._property_pattern.sub(replacer, value)


class MavenProjectScanner:
    """Scan for Maven projects in a directory tree."""

    def __init__(self):
        self.parser = MavenParser()

    def scan(self, root_path: Path, recursive: bool = True) -> List[MavenProject]:
        """Scan for Maven projects."""
        projects = []

        if recursive:
            pom_files = list(root_path.rglob("pom.xml"))
        else:
            pom_files = list(root_path.glob("pom.xml"))

        for pom_file in pom_files:
            try:
                project = self.parser.parse_file(pom_file)
                projects.append(project)
            except Exception as e:
                # Log error but continue scanning
                print(f"Warning: Failed to parse {pom_file}: {e}")

        return projects

    def build_dependency_tree(
        self,
        projects: List[MavenProject],
    ) -> Dict[str, Set[str]]:
        """Build a dependency tree from scanned projects."""
        tree: Dict[str, Set[str]] = {}

        # Index projects by GAV
        project_index = {p.coordinate.gav: p for p in projects}

        for project in projects:
            gav = project.coordinate.gav
            tree[gav] = set()

            for dep in project.dependencies:
                dep_gav = dep.coordinate.gav
                tree[gav].add(dep_gav)

        return tree


# Convenience functions

def parse_pom(path: Path) -> MavenProject:
    """Parse a POM file."""
    parser = MavenParser()
    return parser.parse_file(path)


def scan_maven_projects(root_path: Path) -> List[MavenProject]:
    """Scan for Maven projects in a directory."""
    scanner = MavenProjectScanner()
    return scanner.scan(root_path)


def get_project_dependencies(project: MavenProject) -> List[str]:
    """Get list of dependency GAVs for a project."""
    return [dep.coordinate.gav for dep in project.dependencies]


def find_dependency_conflicts(projects: List[MavenProject]) -> List[Dict[str, Any]]:
    """Find dependency version conflicts across projects."""
    conflicts = []

    # Index all dependencies by groupId:artifactId
    dep_versions: Dict[str, Dict[str, Set[str]]] = {}

    for project in projects:
        for dep in project.dependencies:
            key = f"{dep.coordinate.group_id}:{dep.coordinate.artifact_id}"
            version = dep.coordinate.version or "unspecified"

            if key not in dep_versions:
                dep_versions[key] = {}
            if version not in dep_versions[key]:
                dep_versions[key][version] = set()

            dep_versions[key][version].add(project.coordinate.gav)

    # Find conflicts (same artifact with different versions)
    for artifact, versions in dep_versions.items():
        if len(versions) > 1:
            conflicts.append({
                "artifact": artifact,
                "versions": {
                    v: list(projects_using)
                    for v, projects_using in versions.items()
                },
            })

    return conflicts
