"""
Kernel Registry — Automatic discovery and dependency resolution.

The registry provides:
1. Auto-discovery of kernels in the ragix_kernels package
2. Registration of external kernels
3. Dependency resolution via topological sort
4. Query by name, category, or stage

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-14
"""

import importlib
import pkgutil
import logging
from pathlib import Path
from typing import Dict, List, Type, Optional, Set
from collections import defaultdict

from ragix_kernels.base import Kernel, KernelClass

logger = logging.getLogger(__name__)


class KernelRegistry:
    """
    Discovers and manages available kernels.

    The registry automatically scans the ragix_kernels package for
    kernel implementations and makes them available by name.

    Example:
        # Discover all kernels
        KernelRegistry.discover()

        # Get a specific kernel
        ast_kernel = KernelRegistry.get("ast_scan")

        # List kernels by category
        audit_kernels = KernelRegistry.list_category("audit")

        # Resolve dependencies
        ordered = KernelRegistry.resolve_dependencies(["hotspots", "ast_scan"])
    """

    _kernels: Dict[str, KernelClass] = {}
    _categories: Dict[str, List[str]] = defaultdict(list)
    _stages: Dict[int, List[str]] = defaultdict(list)
    _discovered: bool = False

    @classmethod
    def discover(cls, package_path: str = "ragix_kernels") -> int:
        """
        Auto-discover all kernels in the package.

        Scans all modules in ragix_kernels and its subpackages,
        finding classes that inherit from Kernel.

        Args:
            package_path: Python package path to scan

        Returns:
            Number of kernels discovered
        """
        if cls._discovered:
            logger.debug("Kernels already discovered, skipping")
            return len(cls._kernels)

        count = 0
        try:
            package = importlib.import_module(package_path)
        except ImportError as e:
            logger.error(f"Failed to import {package_path}: {e}")
            return 0

        # Walk through all modules in the package
        package_dir = Path(package.__file__).parent

        for finder, module_name, is_pkg in pkgutil.walk_packages(
            [str(package_dir)],
            prefix=f"{package_path}."
        ):
            # Skip __init__ and base modules
            if module_name.endswith(("__init__", "base", "registry")):
                continue

            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                logger.warning(f"Failed to import {module_name}: {e}")
                continue

            # Find all Kernel subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Kernel)
                    and attr is not Kernel
                    and hasattr(attr, 'name')
                    and attr.name != "base"  # Skip abstract base
                ):
                    cls.register(attr)
                    count += 1

        cls._discovered = True
        logger.info(f"Discovered {count} kernels in {package_path}")
        return count

    @classmethod
    def register(cls, kernel_class: KernelClass) -> None:
        """
        Register a kernel class.

        Args:
            kernel_class: Kernel class to register

        Raises:
            ValueError: If kernel with same name already registered
        """
        name = kernel_class.name
        category = kernel_class.category
        stage = kernel_class.stage

        if name in cls._kernels:
            existing = cls._kernels[name]
            if existing is not kernel_class:
                logger.warning(
                    f"Kernel '{name}' already registered "
                    f"(existing: {existing.__module__}, new: {kernel_class.__module__})"
                )
            return

        cls._kernels[name] = kernel_class
        cls._categories[category].append(name)
        cls._stages[stage].append(name)

        logger.debug(f"Registered kernel: {name} (category={category}, stage={stage})")

    @classmethod
    def get(cls, name: str) -> KernelClass:
        """
        Get kernel class by name.

        Args:
            name: Kernel name

        Returns:
            Kernel class

        Raises:
            KeyError: If kernel not found
        """
        cls._ensure_discovered()

        if name not in cls._kernels:
            available = ", ".join(sorted(cls._kernels.keys()))
            raise KeyError(f"Kernel '{name}' not found. Available: {available}")
        return cls._kernels[name]

    @classmethod
    def get_instance(cls, name: str) -> Kernel:
        """
        Get an instance of a kernel by name.

        Args:
            name: Kernel name

        Returns:
            Kernel instance
        """
        kernel_class = cls.get(name)
        return kernel_class()

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered kernel names."""
        cls._ensure_discovered()
        return sorted(cls._kernels.keys())

    @classmethod
    def list_category(cls, category: str) -> List[str]:
        """
        List all kernels in a category.

        Args:
            category: Category name (audit, transform, test, docs)

        Returns:
            List of kernel names
        """
        cls._ensure_discovered()
        return sorted(cls._categories.get(category, []))

    @classmethod
    def list_stage(cls, stage: int) -> List[str]:
        """
        List all kernels in a stage.

        Args:
            stage: Stage number (1, 2, or 3)

        Returns:
            List of kernel names
        """
        cls._ensure_discovered()
        return sorted(cls._stages.get(stage, []))

    @classmethod
    def get_info(cls, name: str) -> Dict:
        """
        Get detailed info about a kernel.

        Args:
            name: Kernel name

        Returns:
            Dictionary with kernel metadata
        """
        kernel_class = cls.get(name)
        return {
            "name": kernel_class.name,
            "version": kernel_class.version,
            "category": kernel_class.category,
            "stage": kernel_class.stage,
            "description": kernel_class.description,
            "requires": kernel_class.requires,
            "provides": kernel_class.provides,
            "module": kernel_class.__module__,
        }

    @classmethod
    def resolve_dependencies(cls, kernel_names: List[str]) -> List[str]:
        """
        Topologically sort kernels by dependencies.

        Ensures kernels are executed in the correct order,
        with dependencies before dependents.

        Args:
            kernel_names: List of kernel names to sort

        Returns:
            Sorted list of kernel names

        Raises:
            ValueError: If circular dependency detected
        """
        cls._ensure_discovered()

        # Expand with all required dependencies
        all_kernels = set(kernel_names)
        to_process = list(kernel_names)

        while to_process:
            name = to_process.pop()
            kernel_class = cls.get(name)
            for dep in kernel_class.requires:
                if dep not in all_kernels:
                    all_kernels.add(dep)
                    to_process.append(dep)

        # Build dependency graph
        graph: Dict[str, Set[str]] = {}
        for name in all_kernels:
            kernel_class = cls.get(name)
            graph[name] = set(kernel_class.requires)

        # Topological sort (Kahn's algorithm)
        in_degree = {name: len(deps) for name, deps in graph.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_kernels = []

        while queue:
            # Sort for determinism
            queue.sort()
            name = queue.pop(0)
            sorted_kernels.append(name)

            # Update in-degrees
            for other_name, deps in graph.items():
                if name in deps:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0:
                        queue.append(other_name)

        if len(sorted_kernels) != len(all_kernels):
            # Find cycle
            remaining = set(all_kernels) - set(sorted_kernels)
            raise ValueError(f"Circular dependency detected among: {remaining}")

        return sorted_kernels

    @classmethod
    def _ensure_discovered(cls):
        """Ensure kernels have been discovered."""
        if not cls._discovered:
            cls.discover()

    @classmethod
    def reset(cls):
        """Reset the registry (mainly for testing)."""
        cls._kernels.clear()
        cls._categories.clear()
        cls._stages.clear()
        cls._discovered = False


# Convenience functions for module-level access
def discover_kernels() -> int:
    """Discover all available kernels."""
    return KernelRegistry.discover()


def get_kernel(name: str) -> KernelClass:
    """Get a kernel class by name."""
    return KernelRegistry.get(name)


def list_kernels(category: Optional[str] = None, stage: Optional[int] = None) -> List[str]:
    """
    List available kernels.

    Args:
        category: Filter by category (optional)
        stage: Filter by stage (optional)

    Returns:
        List of kernel names
    """
    if category:
        return KernelRegistry.list_category(category)
    elif stage is not None:
        return KernelRegistry.list_stage(stage)
    else:
        return KernelRegistry.list_all()
