"""
Service Auto-Detector — Multi-source service/component detection

Combines multiple data sources to auto-detect services and generate audit config:
1. File system scan (paths, filenames)
2. RAG index (concepts, chunks) - if available
3. AST analysis (classes, packages) - if available
4. Content patterns (annotations, comments)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-10
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

from ragix_audit.component_mapper import ComponentMapper, ComponentType, Component

logger = logging.getLogger(__name__)


class DetectionSource(Enum):
    """Source of component detection."""
    FILESYSTEM = "filesystem"       # Path/filename patterns
    RAG_CONCEPT = "rag_concept"     # RAG knowledge graph concepts
    RAG_CHUNK = "rag_chunk"         # RAG chunk content
    AST_CLASS = "ast_class"         # AST class names
    AST_PACKAGE = "ast_package"     # AST package structure
    CONTENT_ANNOTATION = "annotation"  # @Service, @Component annotations
    CONTENT_JAVADOC = "javadoc"     # Javadoc @since, @version tags
    POM_ARTIFACT = "pom"            # Maven artifact IDs


@dataclass
class DetectedService:
    """A detected service/component with evidence from multiple sources."""
    id: str                                  # SK02, SC04, SG01
    type: ComponentType
    name: Optional[str] = None               # Human-readable name if found
    description: Optional[str] = None        # Description if found

    # Detection evidence
    sources: List[DetectionSource] = field(default_factory=list)
    confidence: float = 0.0                  # 0-1 confidence score

    # File information
    files: List[str] = field(default_factory=list)
    packages: Set[str] = field(default_factory=set)
    main_package: Optional[str] = None

    # Metadata from various sources
    rag_concepts: List[str] = field(default_factory=list)
    ast_classes: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)

    # Timeline hints
    first_seen: Optional[datetime] = None
    doc_version: Optional[str] = None

    def add_source(self, source: DetectionSource):
        """Add a detection source if not already present."""
        if source not in self.sources:
            self.sources.append(source)
            self._update_confidence()

    def _update_confidence(self):
        """Update confidence based on number and type of sources."""
        # Base confidence per source type
        source_weights = {
            DetectionSource.FILESYSTEM: 0.3,
            DetectionSource.RAG_CONCEPT: 0.2,
            DetectionSource.RAG_CHUNK: 0.15,
            DetectionSource.AST_CLASS: 0.25,
            DetectionSource.AST_PACKAGE: 0.2,
            DetectionSource.CONTENT_ANNOTATION: 0.3,
            DetectionSource.CONTENT_JAVADOC: 0.15,
            DetectionSource.POM_ARTIFACT: 0.2,
        }

        total = sum(source_weights.get(s, 0.1) for s in self.sources)
        self.confidence = min(1.0, total)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "service_type": self.type.value,  # Alias for compatibility with audit router
            "name": self.name,
            "description": self.description,
            "sources": [s.value for s in self.sources],
            "confidence": round(self.confidence, 3),
            "file_count": len(self.files),
            "packages": list(self.packages),
            "main_package": self.main_package,
            "rag_concepts": self.rag_concepts[:5],  # Limit
            "ast_classes": self.ast_classes[:10],   # Limit
            "annotations": self.annotations[:5],    # Limit
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "doc_version": self.doc_version,
        }


@dataclass
class AuditConfig:
    """Generated audit configuration for a project."""
    project_path: str
    project_name: str
    detected_at: datetime = field(default_factory=datetime.now)

    # Detected services
    services: Dict[str, DetectedService] = field(default_factory=dict)

    # Detection sources used
    sources_available: List[str] = field(default_factory=list)

    # Project metadata
    total_files: int = 0
    total_packages: int = 0
    languages: List[str] = field(default_factory=list)

    # Thresholds (can be tuned)
    thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "drift_days": 90,
        "new_component_days": 180,
        "legacy_years": 3,
    })

    # Risk weights (can be tuned)
    risk_weights: Dict[str, float] = field(default_factory=lambda: {
        "volatility": 0.20,
        "impact": 0.25,
        "complexity": 0.20,
        "maturity": 0.25,
        "doc_gap": 0.10,
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_path": self.project_path,
            "project_name": self.project_name,
            "detected_at": self.detected_at.isoformat(),
            "services": {k: v.to_dict() for k, v in self.services.items()},
            "sources_available": self.sources_available,
            "summary": {
                "total_services": len(self.services),
                "by_type": self._count_by_type(),
                "avg_confidence": self._avg_confidence(),
            },
            "total_files": self.total_files,
            "total_packages": self.total_packages,
            "languages": self.languages,
            "thresholds": self.thresholds,
            "risk_weights": self.risk_weights,
        }

    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for svc in self.services.values():
            t = svc.type.value
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _avg_confidence(self) -> float:
        if not self.services:
            return 0.0
        return sum(s.confidence for s in self.services.values()) / len(self.services)


class ServiceDetector:
    """
    Multi-source service detector.

    Combines:
    - ComponentMapper (filesystem scan)
    - RAG index (if available)
    - AST analysis (if available)

    Generates an AuditConfig with detected services.
    """

    # Annotation patterns for service detection
    ANNOTATION_PATTERNS = [
        # IOWIZME: SK/SC/SG patterns
        re.compile(r'@Service\s*\(\s*["\']?(SK|SC|SG)\d{2}["\']?\s*\)', re.IGNORECASE),
        re.compile(r'@Component\s*\(\s*["\']?(SK|SC|SG)\d{2}["\']?\s*\)', re.IGNORECASE),
        re.compile(r'@Named\s*\(\s*["\']?(SK|SC|SG)\d{2}["\']?\s*\)', re.IGNORECASE),
        # SIAS: spre## patterns (e.g., @Service("spre28ws"), @Component(value = "spre13"))
        re.compile(r'@Service\s*\(\s*["\']?(spre\d{2}(?:ws)?|sprebpm(?:ws)?|spremail)["\']?\s*\)', re.IGNORECASE),
        re.compile(r'@Component\s*\(\s*(?:value\s*=\s*)?["\']?(spre\d{2}(?:ws)?|sprebpm(?:ws)?|spremail)["\']?\s*\)', re.IGNORECASE),
        # SIAS: s[ActionName] patterns for task operations
        re.compile(r'@Component\s*\(\s*(?:value\s*=\s*)?["\'](s[A-Z][a-zA-Z]+)["\']\s*\)', re.IGNORECASE),
    ]

    # Service name patterns in comments/docs
    NAME_PATTERNS = [
        re.compile(r'Service\s+(?:Key\s+)?(SK\d{2})\s*[-:]\s*(.+?)(?:\n|$)', re.IGNORECASE),
        re.compile(r'Screen\s+(?:Code\s+)?(SC\d{2})\s*[-:]\s*(.+?)(?:\n|$)', re.IGNORECASE),
        re.compile(r'(S[KCG]\d{2})\s*[-:=]\s*["\'"]?([^"\'"\n]{5,50})["\'"]?', re.IGNORECASE),
        # SIAS: spre## service patterns
        re.compile(r'(spre\d{2}(?:ws)?)\s*[-:=]\s*["\'"]?([^"\'"\n]{5,50})["\'"]?', re.IGNORECASE),
    ]

    # SIAS-specific component patterns for broader detection
    SIAS_COMPONENT_PATTERNS = [
        re.compile(r'\b(spre\d{2}(?:ws)?)\b', re.IGNORECASE),
        re.compile(r'\b(sprebpm(?:ws)?)\b', re.IGNORECASE),
        re.compile(r'\b(spremail)\b', re.IGNORECASE),
    ]

    def __init__(
        self,
        project_path: str,
        rag_project: Optional[Any] = None,
        ast_graph: Optional[Any] = None,
    ):
        """
        Initialize service detector.

        Args:
            project_path: Root path of the project to analyze
            rag_project: Optional RAGProject instance (from ragix_core.rag_project)
            ast_graph: Optional dependency graph (from ragix_core.dependencies)
        """
        self.project_path = Path(project_path)
        self.rag_project = rag_project
        self.ast_graph = ast_graph

        self.mapper = ComponentMapper()
        self.services: Dict[str, DetectedService] = {}

    def detect(self) -> AuditConfig:
        """
        Run full detection and generate AuditConfig.

        Returns:
            AuditConfig with detected services and suggested configuration
        """
        sources_available = []

        # 1. Filesystem scan (always available)
        logger.info("Scanning filesystem...")
        self._detect_from_filesystem()
        sources_available.append("filesystem")

        # 2. RAG index (if available)
        if self.rag_project is not None:
            logger.info("Enriching from RAG index...")
            self._detect_from_rag()
            sources_available.append("rag")

        # 3. AST graph (if available)
        if self.ast_graph is not None:
            logger.info("Enriching from AST analysis...")
            self._detect_from_ast()
            sources_available.append("ast")

        # 4. Content analysis (annotations, javadoc)
        logger.info("Analyzing file content...")
        self._detect_from_content()
        sources_available.append("content")

        # 5. Detect languages
        languages = self._detect_languages()

        # Build config
        config = AuditConfig(
            project_path=str(self.project_path),
            project_name=self.project_path.name,
            services=self.services,
            sources_available=sources_available,
            total_files=sum(len(s.files) for s in self.services.values()),
            total_packages=len(set(p for s in self.services.values() for p in s.packages)),
            languages=languages,
        )

        logger.info(f"Detected {len(self.services)} services from {sources_available}")
        return config

    def _detect_from_filesystem(self):
        """Detect services from filesystem patterns."""
        # Use ComponentMapper for initial detection
        # Handle multi-module Maven projects
        src_path = self.project_path / "src"
        if src_path.exists():
            java_in_src = list(src_path.rglob("*.java"))[:1]
            if not java_in_src:
                src_path = self.project_path
        else:
            src_path = self.project_path

        logger.info(f"Filesystem scan from: {src_path}")
        components = self.mapper.scan_directory(src_path)

        for comp_id, comp in components.items():
            svc = self._get_or_create_service(comp_id, comp.type)
            svc.files.extend(comp.files)
            svc.packages.update(comp.packages)
            svc.add_source(DetectionSource.FILESYSTEM)

            # Set main package (most common)
            if comp.packages:
                pkg_counts = {}
                for pkg in comp.packages:
                    pkg_counts[pkg] = pkg_counts.get(pkg, 0) + 1
                svc.main_package = max(pkg_counts, key=pkg_counts.get)

    def _detect_from_rag(self):
        """Enrich detection from RAG knowledge graph."""
        if not self.rag_project:
            return

        try:
            # Access the knowledge graph
            graph = getattr(self.rag_project, '_graph', None)
            if graph is None:
                return

            # Import NodeType from rag_project
            from ragix_core.rag_project.graph import NodeType

            # Get concepts that match SK/SC/SG patterns
            concepts = graph.get_nodes_by_type(NodeType.CONCEPT)
            comp_pattern = re.compile(r'\b(SK|SC|SG)\d{2}\b', re.IGNORECASE)

            for concept in concepts:
                match = comp_pattern.search(concept.label)
                if match:
                    comp_id = match.group(0).upper()
                    comp_type = self._id_to_type(comp_id)

                    svc = self._get_or_create_service(comp_id, comp_type)
                    svc.rag_concepts.append(concept.label)
                    svc.add_source(DetectionSource.RAG_CONCEPT)

                    # Try to extract name from concept label
                    if not svc.name and len(concept.label) > 4:
                        potential_name = concept.label.replace(comp_id, "").strip(" -:")
                        if len(potential_name) > 3:
                            svc.name = potential_name

            # Get chunks and look for component references
            chunks = graph.get_nodes_by_type(NodeType.CHUNK)
            for chunk in chunks[:500]:  # Limit to avoid long scans
                metadata = chunk.metadata or {}
                content = metadata.get("content", "")

                for match in comp_pattern.finditer(content):
                    comp_id = match.group(0).upper()
                    comp_type = self._id_to_type(comp_id)

                    svc = self._get_or_create_service(comp_id, comp_type)
                    svc.add_source(DetectionSource.RAG_CHUNK)

        except Exception as e:
            logger.warning(f"RAG enrichment failed: {e}")

    def _detect_from_ast(self):
        """Enrich detection from AST dependency graph."""
        if not self.ast_graph:
            return

        try:
            # The ast_graph is a DependencyGraph with nodes
            nodes = getattr(self.ast_graph, 'nodes', {})

            comp_pattern = re.compile(r'\b(SK|SC|SG)\d{2}\b', re.IGNORECASE)

            for node_id, node in nodes.items():
                # Check node name
                match = comp_pattern.search(node.name)
                if match:
                    comp_id = match.group(0).upper()
                    comp_type = self._id_to_type(comp_id)

                    svc = self._get_or_create_service(comp_id, comp_type)
                    svc.ast_classes.append(node.name)
                    svc.add_source(DetectionSource.AST_CLASS)

                # Check package
                package = getattr(node, 'package', '') or ''
                if package:
                    match = comp_pattern.search(package)
                    if match:
                        comp_id = match.group(0).upper()
                        comp_type = self._id_to_type(comp_id)

                        svc = self._get_or_create_service(comp_id, comp_type)
                        svc.packages.add(package)
                        svc.add_source(DetectionSource.AST_PACKAGE)

        except Exception as e:
            logger.warning(f"AST enrichment failed: {e}")

    def _detect_from_content(self):
        """Analyze file content for annotations and documentation."""
        # Determine source path - handle multi-module Maven projects
        src_path = self.project_path / "src"
        if src_path.exists():
            # Check if src has Java files directly
            java_in_src = list(src_path.rglob("*.java"))[:1]
            if not java_in_src:
                # Multi-module project: scan from root
                src_path = self.project_path
        else:
            src_path = self.project_path

        logger.info(f"Scanning Java files from: {src_path}")

        for java_file in src_path.rglob("*.java"):
            try:
                content = java_file.read_text(encoding='utf-8', errors='ignore')
                self._analyze_java_content(content, java_file)
            except Exception:
                continue

        # Scan documentation files for service names
        for doc_file in self.project_path.rglob("*.md"):
            try:
                content = doc_file.read_text(encoding='utf-8', errors='ignore')
                self._analyze_doc_content(content)
            except Exception:
                continue

    def _analyze_java_content(self, content: str, file_path: Path):
        """Analyze Java file content for service indicators."""
        # Check for @Service, @Component annotations
        for pattern in self.ANNOTATION_PATTERNS:
            for match in pattern.finditer(content):
                # Extract the component ID from the captured group
                comp_id = match.group(1)
                # _get_or_create_service handles normalization

                comp_type = self._id_to_type(comp_id)

                svc = self._get_or_create_service(comp_id, comp_type)
                svc.annotations.append(match.group(0))
                svc.add_source(DetectionSource.CONTENT_ANNOTATION)

        # Also check for SIAS patterns in file content (directory structure patterns)
        for pattern in self.SIAS_COMPONENT_PATTERNS:
            for match in pattern.finditer(content):
                comp_id = match.group(1)  # Will be uppercased in _get_or_create_service
                comp_type = self._id_to_type(comp_id)

                svc = self._get_or_create_service(comp_id, comp_type)
                svc.add_source(DetectionSource.CONTENT_ANNOTATION)

        # Check for Javadoc @since tags
        since_pattern = re.compile(r'@since\s+[Vv]?(\d+(?:\.\d+)*)', re.IGNORECASE)
        for match in since_pattern.finditer(content):
            version = match.group(1)
            # Associate with any service in this file - SK/SC/SG patterns
            comp_pattern = re.compile(r'\b(SK|SC|SG)\d{2}\b', re.IGNORECASE)
            for comp_match in comp_pattern.finditer(content):
                comp_id = comp_match.group(0).upper()
                comp_type = self._id_to_type(comp_id)

                svc = self._get_or_create_service(comp_id, comp_type)
                if not svc.doc_version:
                    svc.doc_version = f"V{version}"
                svc.add_source(DetectionSource.CONTENT_JAVADOC)

            # Also check SIAS patterns
            for sias_pattern in self.SIAS_COMPONENT_PATTERNS:
                for comp_match in sias_pattern.finditer(content):
                    comp_id = comp_match.group(1)  # Will be uppercased
                    comp_type = self._id_to_type(comp_id)

                    svc = self._get_or_create_service(comp_id, comp_type)
                    if not svc.doc_version:
                        svc.doc_version = f"V{version}"
                    svc.add_source(DetectionSource.CONTENT_JAVADOC)

    def _analyze_doc_content(self, content: str):
        """Analyze documentation for service names and descriptions."""
        for pattern in self.NAME_PATTERNS:
            for match in pattern.finditer(content):
                comp_id = match.group(1).upper()
                name = match.group(2).strip() if len(match.groups()) > 1 else None
                comp_type = self._id_to_type(comp_id)

                svc = self._get_or_create_service(comp_id, comp_type)
                if name and not svc.name:
                    svc.name = name

    def _get_or_create_service(self, comp_id: str, comp_type: ComponentType) -> DetectedService:
        """Get existing service or create new one."""
        # Normalize all IDs to uppercase for consistency with ComponentMapper/TimelineScanner
        comp_id = comp_id.upper()
        if comp_id not in self.services:
            self.services[comp_id] = DetectedService(id=comp_id, type=comp_type)
        return self.services[comp_id]

    def _id_to_type(self, comp_id: str) -> ComponentType:
        """Convert component ID prefix to type."""
        comp_lower = comp_id.lower()

        # SIAS patterns
        if comp_lower.startswith('spre'):
            if 'ws' in comp_lower:
                return ComponentType.SERVICE  # Web service
            return ComponentType.SERVICE  # JMS or general service
        if comp_lower.startswith('s') and len(comp_id) > 4 and comp_id[1].isupper():
            # SIAS task operations like sAffecterTache
            return ComponentType.SERVICE

        # IOWIZME patterns
        prefix = comp_id[:2].upper()
        if prefix == "SK":
            return ComponentType.SERVICE
        elif prefix == "SC":
            return ComponentType.SCREEN
        elif prefix == "SG":
            return ComponentType.GENERAL
        return ComponentType.GENERAL

    def _detect_languages(self) -> List[str]:
        """Detect programming languages in the project."""
        extensions = {
            ".java": "Java",
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".go": "Go",
            ".rs": "Rust",
        }

        found = set()
        for ext, lang in extensions.items():
            if list(self.project_path.rglob(f"*{ext}"))[:1]:  # Quick check
                found.add(lang)

        return sorted(found)


def detect_services(
    project_path: str,
    rag_project: Optional[Any] = None,
    ast_graph: Optional[Any] = None,
) -> AuditConfig:
    """
    Convenience function to detect services in a project.

    Args:
        project_path: Root path of the project
        rag_project: Optional RAGProject instance for RAG enrichment
        ast_graph: Optional DependencyGraph for AST enrichment

    Returns:
        AuditConfig with detected services
    """
    detector = ServiceDetector(project_path, rag_project, ast_graph)
    return detector.detect()


def load_rag_project(project_path: str) -> Optional[Any]:
    """
    Try to load existing RAG project from .RAG directory.

    Returns RAGProject if available, None otherwise.
    """
    try:
        from ragix_core.rag_project import RAGProject

        rag_dir = Path(project_path) / ".RAG"
        if rag_dir.exists():
            project = RAGProject(project_path)
            if project.is_indexed():
                logger.info(f"Loaded existing RAG index from {rag_dir}")
                return project
    except Exception as e:
        logger.debug(f"Could not load RAG project: {e}")

    return None


def load_ast_graph(project_path: str) -> Optional[Any]:
    """
    Try to load or build AST dependency graph.

    Returns DependencyGraph if available, None otherwise.
    """
    try:
        from ragix_core import build_dependency_graph
        from ragix_core.analysis_cache import get_cache, get_or_analyze

        # Check cache first
        cache = get_cache()
        cached = cache.get(project_path)
        if cached and cached.graph:
            logger.info("Loaded AST graph from cache")
            return cached.graph

        # Build fresh (can be slow for large projects)
        src_path = Path(project_path) / "src" if (Path(project_path) / "src").exists() else Path(project_path)

        # Only build if not too large
        java_files = list(src_path.rglob("*.java"))
        if len(java_files) > 2000:
            logger.warning(f"Project too large ({len(java_files)} files), skipping AST build")
            return None

        logger.info(f"Building AST graph for {len(java_files)} files...")
        graph = build_dependency_graph(src_path)
        return graph

    except Exception as e:
        logger.debug(f"Could not load AST graph: {e}")

    return None


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "/home/olivi/Documents/Adservio/audit/IOWIZME"

    print(f"Detecting services in {path}...")

    # Try to load RAG and AST
    rag = load_rag_project(path)
    ast = load_ast_graph(path)

    # Run detection
    config = detect_services(path, rag, ast)

    print("\n=== Audit Config ===\n")
    print(json.dumps(config.to_dict(), indent=2, default=str))
