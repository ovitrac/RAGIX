"""
Component Mapper — Detect SK/SC/SG components from file paths and content

Maps files to business components (Service Keys, Screen Codes, General Services)
based on path patterns, package names, and content analysis.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Type of business component."""
    SERVICE = "service"      # SK01-SK14: Service Keys
    SCREEN = "screen"        # SC01-SC08: Screen Codes
    GENERAL = "general"      # SG01+: General Services
    UNKNOWN = "unknown"      # Unclassified


@dataclass
class Component:
    """A business component (SK/SC/SG)."""
    id: str                          # e.g., "SK02", "SC04"
    type: ComponentType
    files: List[str] = field(default_factory=list)
    packages: Set[str] = field(default_factory=set)
    description: Optional[str] = None

    @property
    def file_count(self) -> int:
        return len(self.files)

    def add_file(self, file_path: str, package: Optional[str] = None):
        if file_path not in self.files:
            self.files.append(file_path)
        if package:
            self.packages.add(package)


class ComponentMapper:
    """
    Maps files and code to business components (SK/SC/SG).

    Detection strategies:
    1. Path-based: /sk02/, /SC04/, etc.
    2. Package-based: fr.iowizmi.iok.sk02
    3. Content-based: @Service("SK02"), class SK02Service
    4. Filename-based: SK02Handler.java
    """

    # Component ID patterns
    COMPONENT_PATTERNS = {
        # Service Keys (SK01-SK14)
        r'\bSK[0-9]{2}\b': ComponentType.SERVICE,
        r'\bsk[0-9]{2}\b': ComponentType.SERVICE,

        # Screen Codes (SC01-SC08)
        r'\bSC[0-9]{2}\b': ComponentType.SCREEN,
        r'\bsc[0-9]{2}\b': ComponentType.SCREEN,

        # General Services (SG01+)
        r'\bSG[0-9]{2}\b': ComponentType.GENERAL,
        r'\bsg[0-9]{2}\b': ComponentType.GENERAL,
    }

    # Package patterns for component detection
    PACKAGE_PATTERNS = {
        r'fr\.iowizmi\.iok\.sk': ComponentType.SERVICE,
        r'fr\.iowizmi\.iok\.sc': ComponentType.SCREEN,
        r'fr\.iowizmi\.iok\.sg': ComponentType.GENERAL,
        r'fr\.iowizmi\.ui\.': ComponentType.SCREEN,
        r'fr\.iowizmi\.service\.': ComponentType.SERVICE,
    }

    def __init__(self):
        self.components: Dict[str, Component] = {}
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), ctype)
            for pattern, ctype in self.COMPONENT_PATTERNS.items()
        ]
        self._compiled_package_patterns = [
            (re.compile(pattern), ctype)
            for pattern, ctype in self.PACKAGE_PATTERNS.items()
        ]

    def detect_from_path(self, file_path: Path) -> List[Tuple[str, ComponentType]]:
        """
        Detect component IDs from file path.

        Returns list of (component_id, component_type) tuples.
        """
        path_str = str(file_path)
        results = []

        for pattern, ctype in self._compiled_patterns:
            matches = pattern.findall(path_str)
            for match in matches:
                component_id = match.upper()
                results.append((component_id, ctype))

        return results

    def detect_from_package(self, package_name: str) -> Optional[ComponentType]:
        """Detect component type from Java package name."""
        for pattern, ctype in self._compiled_package_patterns:
            if pattern.search(package_name):
                return ctype
        return None

    def detect_from_content(self, content: str) -> List[Tuple[str, ComponentType]]:
        """
        Detect component IDs from file content.

        Looks for:
        - @Service("SK02")
        - class SK02Handler
        - SK02.process()
        - Comments mentioning SK/SC/SG
        """
        results = []
        seen = set()

        for pattern, ctype in self._compiled_patterns:
            matches = pattern.findall(content)
            for match in matches:
                component_id = match.upper()
                if component_id not in seen:
                    seen.add(component_id)
                    results.append((component_id, ctype))

        return results

    def map_file(
        self,
        file_path: Path,
        content: Optional[str] = None,
        package: Optional[str] = None
    ) -> List[str]:
        """
        Map a file to its component(s).

        Returns list of component IDs this file belongs to.
        """
        component_ids = []

        # 1. Detect from path
        for comp_id, comp_type in self.detect_from_path(file_path):
            self._ensure_component(comp_id, comp_type)
            self.components[comp_id].add_file(str(file_path), package)
            component_ids.append(comp_id)

        # 2. Detect from content
        if content:
            for comp_id, comp_type in self.detect_from_content(content):
                self._ensure_component(comp_id, comp_type)
                if str(file_path) not in self.components[comp_id].files:
                    self.components[comp_id].add_file(str(file_path), package)
                if comp_id not in component_ids:
                    component_ids.append(comp_id)

        return component_ids

    def _ensure_component(self, comp_id: str, comp_type: ComponentType):
        """Ensure component exists in registry."""
        if comp_id not in self.components:
            self.components[comp_id] = Component(
                id=comp_id,
                type=comp_type
            )

    def scan_directory(
        self,
        root_path: Path,
        extensions: Optional[List[str]] = None,
        read_content: bool = True
    ) -> Dict[str, Component]:
        """
        Scan a directory tree and map all files to components.

        Args:
            root_path: Root directory to scan
            extensions: File extensions to include (default: .java)
            read_content: Whether to read file content for detection

        Returns:
            Dictionary of component_id -> Component
        """
        if extensions is None:
            extensions = ['.java']

        root_path = Path(root_path)
        file_count = 0

        for ext in extensions:
            for file_path in root_path.rglob(f'*{ext}'):
                if file_path.is_file():
                    file_count += 1
                    content = None
                    package = None

                    if read_content:
                        try:
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                            # Extract package from Java file
                            package_match = re.search(r'^package\s+([\w.]+);', content, re.MULTILINE)
                            if package_match:
                                package = package_match.group(1)
                        except Exception as e:
                            logger.warning(f"Failed to read {file_path}: {e}")

                    self.map_file(file_path, content, package)

        logger.info(f"Scanned {file_count} files, found {len(self.components)} components")
        return self.components

    def get_component(self, comp_id: str) -> Optional[Component]:
        """Get a component by ID."""
        return self.components.get(comp_id.upper())

    def get_components_by_type(self, comp_type: ComponentType) -> List[Component]:
        """Get all components of a given type."""
        return [c for c in self.components.values() if c.type == comp_type]

    def get_summary(self) -> Dict:
        """Get a summary of all detected components."""
        by_type = {}
        for comp_type in ComponentType:
            components = self.get_components_by_type(comp_type)
            if components:
                by_type[comp_type.value] = {
                    "count": len(components),
                    "components": [c.id for c in sorted(components, key=lambda x: x.id)],
                    "total_files": sum(c.file_count for c in components)
                }

        return {
            "total_components": len(self.components),
            "by_type": by_type,
            "components": {
                c.id: {
                    "type": c.type.value,
                    "file_count": c.file_count,
                    "packages": list(c.packages)
                }
                for c in sorted(self.components.values(), key=lambda x: x.id)
            }
        }


def detect_component_id(text: str) -> Optional[str]:
    """
    Quick utility to extract a single component ID from text.

    Returns the first SK/SC/SG pattern found, or None.
    """
    pattern = re.compile(r'\b(SK|SC|SG)[0-9]{2}\b', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(0).upper()
    return None


if __name__ == "__main__":
    # Test with IOWIZME
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/home/olivi/Documents/Adservio/audit/IOWIZME/src"

    mapper = ComponentMapper()
    components = mapper.scan_directory(Path(path))

    print("\n=== Component Mapping Summary ===\n")
    summary = mapper.get_summary()

    for comp_type, info in summary["by_type"].items():
        print(f"{comp_type.upper()}: {info['count']} components, {info['total_files']} files")
        for comp_id in info["components"]:
            comp = mapper.get_component(comp_id)
            print(f"  - {comp_id}: {comp.file_count} files")

    print(f"\nTotal: {summary['total_components']} components")
