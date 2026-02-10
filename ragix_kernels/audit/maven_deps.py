"""
Kernel: Maven Dependencies Extraction
Stage: 1 (Data Collection)

Parses all pom.xml files in a project directory to extract:
- Module declarations (groupId, artifactId, version, packaging)
- Dependencies with versions and scopes
- Parent POM relationships
- Property definitions and substitution

Uses stdlib xml.etree.ElementTree only (no external deps).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-09
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import re

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)

# Maven POM namespace
MVN_NS = "{http://maven.apache.org/POM/4.0.0}"


class MavenDepsKernel(Kernel):
    """
    Extract Maven dependencies from pom.xml files.

    Scans the project directory for all pom.xml files, extracts module
    declarations, dependency lists, parent relationships, and properties.
    Resolves ${property} references where possible.

    Configuration:
        project.path: Path to project directory (required)
    """

    name = "maven_deps"
    version = "1.0.0"
    category = "audit"
    stage = 1
    description = "Extract Maven dependencies from pom.xml files"
    requires = []
    provides = ["maven_dependencies", "maven_versions"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        project_config = input.config.get("project", {})
        project_path = Path(project_config.get("path", "."))

        if not project_path.exists():
            raise RuntimeError(f"Project path does not exist: {project_path}")

        # Find all pom.xml files
        pom_files = sorted(project_path.rglob("pom.xml"))
        logger.info(f"[maven_deps] Found {len(pom_files)} pom.xml files")

        modules = []
        all_deps: Dict[str, Dict[str, Any]] = {}  # key: groupId:artifactId
        parse_errors = 0

        for pom_path in pom_files:
            try:
                module = self._parse_pom(pom_path, project_path)
                if module:
                    modules.append(module)
                    # Aggregate dependencies
                    for dep in module.get("dependencies", []):
                        key = f"{dep['groupId']}:{dep['artifactId']}"
                        if key not in all_deps:
                            all_deps[key] = {
                                "groupId": dep["groupId"],
                                "artifactId": dep["artifactId"],
                                "version": dep.get("version", "inherited"),
                                "scope": dep.get("scope", "compile"),
                                "used_by": [],
                            }
                        all_deps[key]["used_by"].append(module["artifactId"])
                        # Keep most specific version
                        if dep.get("version") and all_deps[key]["version"] == "inherited":
                            all_deps[key]["version"] = dep["version"]
            except Exception as e:
                logger.warning(f"[maven_deps] Error parsing {pom_path}: {e}")
                parse_errors += 1

        return {
            "modules": modules,
            "all_dependencies": list(all_deps.values()),
            "statistics": {
                "pom_files_found": len(pom_files),
                "pom_files_parsed": len(modules),
                "modules_found": len(modules),
                "distinct_dependencies": len(all_deps),
                "parse_errors": parse_errors,
            },
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        stats = data.get("statistics", {})
        return (
            f"Maven analysis: {stats.get('pom_files_parsed', 0)} pom.xml parsed, "
            f"{stats.get('modules_found', 0)} modules, "
            f"{stats.get('distinct_dependencies', 0)} distinct dependencies. "
            f"Parse errors: {stats.get('parse_errors', 0)}."
        )

    def _parse_pom(self, pom_path: Path, project_root: Path) -> Optional[Dict[str, Any]]:
        """Parse a single pom.xml file."""
        tree = ET.parse(pom_path)
        root = tree.getroot()

        # Extract properties for ${variable} resolution
        properties = self._extract_properties(root)

        # Module identity
        group_id = self._get_text(root, "groupId", properties)
        artifact_id = self._get_text(root, "artifactId", properties)
        version = self._get_text(root, "version", properties)
        packaging = self._get_text(root, "packaging", properties) or "jar"

        if not artifact_id:
            return None

        # Parent POM
        parent = None
        parent_el = root.find(f"{MVN_NS}parent")
        if parent_el is not None:
            parent = {
                "groupId": self._get_text(parent_el, "groupId", properties) or "",
                "artifactId": self._get_text(parent_el, "artifactId", properties) or "",
                "version": self._get_text(parent_el, "version", properties) or "",
            }
            # Inherit groupId/version from parent if missing
            if not group_id and parent:
                group_id = parent["groupId"]
            if not version and parent:
                version = parent["version"]

        # Dependencies
        dependencies = []
        deps_el = root.find(f"{MVN_NS}dependencies")
        if deps_el is not None:
            for dep_el in deps_el.findall(f"{MVN_NS}dependency"):
                dep = self._parse_dependency(dep_el, properties)
                if dep:
                    dependencies.append(dep)

        # Dependency management (BOMs, version constraints)
        dep_mgmt = []
        mgmt_el = root.find(f"{MVN_NS}dependencyManagement")
        if mgmt_el is not None:
            mgmt_deps = mgmt_el.find(f"{MVN_NS}dependencies")
            if mgmt_deps is not None:
                for dep_el in mgmt_deps.findall(f"{MVN_NS}dependency"):
                    dep = self._parse_dependency(dep_el, properties)
                    if dep:
                        dep_mgmt.append(dep)

        # Sub-modules declared
        sub_modules = []
        modules_el = root.find(f"{MVN_NS}modules")
        if modules_el is not None:
            for mod_el in modules_el.findall(f"{MVN_NS}module"):
                if mod_el.text:
                    sub_modules.append(mod_el.text.strip())

        # Relative path from project root
        try:
            rel_path = str(pom_path.parent.relative_to(project_root))
        except ValueError:
            rel_path = str(pom_path.parent)

        return {
            "module_path": rel_path,
            "groupId": group_id or "",
            "artifactId": artifact_id,
            "version": version or "",
            "packaging": packaging,
            "parent": parent,
            "properties": properties,
            "dependencies": dependencies,
            "dependency_management": dep_mgmt,
            "sub_modules": sub_modules,
        }

    def _parse_dependency(self, dep_el, properties: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Parse a single <dependency> element."""
        group_id = self._get_text(dep_el, "groupId", properties)
        artifact_id = self._get_text(dep_el, "artifactId", properties)
        if not group_id or not artifact_id:
            return None

        version = self._get_text(dep_el, "version", properties)
        scope = self._get_text(dep_el, "scope", properties) or "compile"
        dep_type = self._get_text(dep_el, "type", properties) or "jar"
        optional = self._get_text(dep_el, "optional", properties) == "true"

        # Exclusions
        exclusions = []
        excl_el = dep_el.find(f"{MVN_NS}exclusions")
        if excl_el is not None:
            for ex in excl_el.findall(f"{MVN_NS}exclusion"):
                ex_g = self._get_text(ex, "groupId", properties)
                ex_a = self._get_text(ex, "artifactId", properties)
                if ex_g and ex_a:
                    exclusions.append(f"{ex_g}:{ex_a}")

        return {
            "groupId": group_id,
            "artifactId": artifact_id,
            "version": version or "inherited",
            "scope": scope,
            "type": dep_type,
            "optional": optional,
            "exclusions": exclusions,
        }

    def _extract_properties(self, root) -> Dict[str, str]:
        """Extract <properties> as a flat dict."""
        props = {}
        props_el = root.find(f"{MVN_NS}properties")
        if props_el is not None:
            for child in props_el:
                tag = child.tag.replace(MVN_NS, "")
                if child.text:
                    props[tag] = child.text.strip()

        # Also add standard project properties
        for key in ("groupId", "artifactId", "version"):
            el = root.find(f"{MVN_NS}{key}")
            if el is not None and el.text:
                props[f"project.{key}"] = el.text.strip()

        return props

    def _get_text(self, element, tag: str, properties: Dict[str, str]) -> Optional[str]:
        """Get text of a child element, resolving ${property} references."""
        el = element.find(f"{MVN_NS}{tag}")
        if el is None or not el.text:
            return None
        text = el.text.strip()
        return self._resolve_properties(text, properties)

    def _resolve_properties(self, text: str, properties: Dict[str, str]) -> str:
        """Resolve ${property.name} references in a string."""
        def _replacer(match):
            prop_name = match.group(1)
            return properties.get(prop_name, match.group(0))

        return re.sub(r'\$\{([^}]+)\}', _replacer, text)
