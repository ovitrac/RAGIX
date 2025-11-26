"""
Plugin System - Extensible tool and workflow plugin architecture for RAGIX

Provides a plugin system with trust levels, capability restrictions, and safe loading.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import hashlib
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import yaml

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Types of plugins supported by RAGIX."""
    TOOL = "tool"
    WORKFLOW = "workflow"
    # Future: AGENT = "agent", BACKEND = "backend", SEARCH = "search"


class TrustLevel(str, Enum):
    """Trust levels for plugins."""
    BUILTIN = "builtin"      # Core RAGIX plugins, full access
    TRUSTED = "trusted"      # Verified plugins, most capabilities
    UNTRUSTED = "untrusted"  # Third-party, restricted capabilities


class PluginCapability(str, Enum):
    """Capabilities that can be granted to plugins."""
    SHELL_READ = "shell:read"         # Read-only shell commands
    SHELL_WRITE = "shell:write"       # Write shell commands
    SHELL_EXECUTE = "shell:execute"   # Execute arbitrary shell
    FILE_READ = "file:read"           # Read files
    FILE_WRITE = "file:write"         # Write files
    NETWORK = "network"               # Network access
    LLM_CALL = "llm:call"             # Call LLM backend
    GIT_READ = "git:read"             # Read git operations
    GIT_WRITE = "git:write"           # Write git operations
    SEARCH = "search"                 # Search operations
    WORKFLOW_EXECUTE = "workflow:execute"  # Execute workflows


# Default capabilities by trust level
DEFAULT_CAPABILITIES: Dict[TrustLevel, Set[PluginCapability]] = {
    TrustLevel.BUILTIN: set(PluginCapability),  # All capabilities
    TrustLevel.TRUSTED: {
        PluginCapability.SHELL_READ,
        PluginCapability.FILE_READ,
        PluginCapability.FILE_WRITE,
        PluginCapability.GIT_READ,
        PluginCapability.GIT_WRITE,
        PluginCapability.SEARCH,
        PluginCapability.LLM_CALL,
        PluginCapability.WORKFLOW_EXECUTE,
    },
    TrustLevel.UNTRUSTED: {
        PluginCapability.SHELL_READ,
        PluginCapability.FILE_READ,
        PluginCapability.SEARCH,
    },
}


@dataclass
class PluginTool:
    """Definition of a tool provided by a plugin."""
    name: str
    description: str
    entry_point: str  # "module:function" format
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    required_capabilities: Set[PluginCapability] = field(default_factory=set)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    handler: Optional[Callable] = None  # Loaded at runtime


@dataclass
class PluginWorkflow:
    """Definition of a workflow template provided by a plugin."""
    name: str
    description: str
    entry_point: str  # Path to workflow YAML or "module:function"
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    required_capabilities: Set[PluginCapability] = field(default_factory=set)


@dataclass
class PluginManifest:
    """
    Plugin manifest defining metadata and capabilities.

    Loaded from plugin.yaml in each plugin directory.
    """
    name: str
    version: str
    description: str
    plugin_type: PluginType
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    author: str = ""
    homepage: str = ""
    license: str = ""
    capabilities: Set[PluginCapability] = field(default_factory=set)
    tools: List[PluginTool] = field(default_factory=list)
    workflows: List[PluginWorkflow] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    python_requires: str = ">=3.10"
    checksum: Optional[str] = None  # SHA256 of plugin contents

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PluginManifest":
        """
        Load manifest from YAML file.

        Args:
            yaml_path: Path to plugin.yaml

        Returns:
            PluginManifest instance
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse plugin type
        plugin_type = PluginType(data.get("type", "tool"))

        # Parse trust level
        trust_level = TrustLevel(data.get("trust", "untrusted"))

        # Parse capabilities
        capabilities = set()
        for cap_str in data.get("capabilities", []):
            try:
                capabilities.add(PluginCapability(cap_str))
            except ValueError:
                logger.warning(f"Unknown capability: {cap_str}")

        # Parse tools
        tools = []
        for tool_data in data.get("tools", []):
            tool_caps = set()
            for cap_str in tool_data.get("capabilities", []):
                try:
                    tool_caps.add(PluginCapability(cap_str))
                except ValueError:
                    pass

            tools.append(PluginTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                entry_point=tool_data["entry"],
                parameters=tool_data.get("parameters", []),
                required_capabilities=tool_caps,
                examples=tool_data.get("examples", []),
            ))

        # Parse workflows
        workflows = []
        for wf_data in data.get("workflows", []):
            wf_caps = set()
            for cap_str in wf_data.get("capabilities", []):
                try:
                    wf_caps.add(PluginCapability(cap_str))
                except ValueError:
                    pass

            workflows.append(PluginWorkflow(
                name=wf_data["name"],
                description=wf_data.get("description", ""),
                entry_point=wf_data["entry"],
                parameters=wf_data.get("parameters", []),
                required_capabilities=wf_caps,
            ))

        return cls(
            name=data["name"],
            version=data.get("version", "0.0.1"),
            description=data.get("description", ""),
            plugin_type=plugin_type,
            trust_level=trust_level,
            author=data.get("author", ""),
            homepage=data.get("homepage", ""),
            license=data.get("license", ""),
            capabilities=capabilities,
            tools=tools,
            workflows=workflows,
            dependencies=data.get("dependencies", []),
            python_requires=data.get("python_requires", ">=3.10"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "type": self.plugin_type.value,
            "trust": self.trust_level.value,
            "author": self.author,
            "homepage": self.homepage,
            "license": self.license,
            "capabilities": [c.value for c in self.capabilities],
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "entry": t.entry_point,
                    "parameters": t.parameters,
                    "capabilities": [c.value for c in t.required_capabilities],
                }
                for t in self.tools
            ],
            "workflows": [
                {
                    "name": w.name,
                    "description": w.description,
                    "entry": w.entry_point,
                    "parameters": w.parameters,
                    "capabilities": [c.value for c in w.required_capabilities],
                }
                for w in self.workflows
            ],
            "dependencies": self.dependencies,
            "python_requires": self.python_requires,
        }


@dataclass
class LoadedPlugin:
    """A fully loaded and validated plugin."""
    manifest: PluginManifest
    path: Path
    loaded_tools: Dict[str, Callable] = field(default_factory=dict)
    loaded_workflows: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class PluginManager:
    """
    Manages plugin discovery, loading, and execution.

    Handles:
    - Plugin discovery from filesystem
    - Trust level enforcement
    - Capability checking
    - Safe plugin loading
    """

    # Explicit allowlist for plugin discovery
    ALLOWED_PLUGIN_DIRS = [
        "plugins",           # Project plugins
        ".ragix/plugins",    # User plugins (project)
    ]

    def __init__(
        self,
        plugin_dirs: Optional[List[Path]] = None,
        trust_overrides: Optional[Dict[str, TrustLevel]] = None,
    ):
        """
        Initialize plugin manager.

        Args:
            plugin_dirs: Additional plugin directories to scan
            trust_overrides: Override trust levels for specific plugins
        """
        self.plugins: Dict[str, LoadedPlugin] = {}
        self.plugin_dirs: List[Path] = []
        self.trust_overrides = trust_overrides or {}
        self._tool_index: Dict[str, str] = {}  # tool_name -> plugin_name
        self._workflow_index: Dict[str, str] = {}  # workflow_name -> plugin_name

        # Setup plugin directories
        self._setup_plugin_dirs(plugin_dirs)

    def _setup_plugin_dirs(self, additional_dirs: Optional[List[Path]] = None):
        """Setup plugin discovery directories."""
        # Project directories
        cwd = Path.cwd()
        for dir_name in self.ALLOWED_PLUGIN_DIRS:
            dir_path = cwd / dir_name
            if dir_path.exists() and dir_path.is_dir():
                self.plugin_dirs.append(dir_path)

        # User global directory
        user_plugins = Path.home() / ".ragix" / "plugins"
        if user_plugins.exists():
            self.plugin_dirs.append(user_plugins)

        # Additional directories
        if additional_dirs:
            for d in additional_dirs:
                if d.exists() and d.is_dir():
                    self.plugin_dirs.append(d)

        logger.debug(f"Plugin directories: {self.plugin_dirs}")

    def discover(self) -> Dict[str, PluginManifest]:
        """
        Discover all available plugins.

        Returns:
            Dictionary of plugin_name -> PluginManifest
        """
        discovered = {}

        for plugin_dir in self.plugin_dirs:
            for item in plugin_dir.iterdir():
                if not item.is_dir():
                    continue

                manifest_path = item / "plugin.yaml"
                if not manifest_path.exists():
                    # Also check plugin.yml
                    manifest_path = item / "plugin.yml"
                    if not manifest_path.exists():
                        continue

                try:
                    manifest = PluginManifest.from_yaml(manifest_path)

                    # Apply trust override if specified
                    if manifest.name in self.trust_overrides:
                        manifest.trust_level = self.trust_overrides[manifest.name]

                    discovered[manifest.name] = manifest
                    logger.debug(f"Discovered plugin: {manifest.name} v{manifest.version}")

                except Exception as e:
                    logger.warning(f"Failed to load manifest from {manifest_path}: {e}")

        return discovered

    def load_plugin(self, name: str) -> Optional[LoadedPlugin]:
        """
        Load a plugin by name.

        Args:
            name: Plugin name

        Returns:
            LoadedPlugin if successful, None otherwise
        """
        if name in self.plugins:
            return self.plugins[name]

        # Find plugin directory
        plugin_path = None
        for plugin_dir in self.plugin_dirs:
            candidate = plugin_dir / name
            if candidate.exists() and (candidate / "plugin.yaml").exists():
                plugin_path = candidate
                break
            if candidate.exists() and (candidate / "plugin.yml").exists():
                plugin_path = candidate
                break

        if not plugin_path:
            logger.error(f"Plugin not found: {name}")
            return None

        # Load manifest
        manifest_path = plugin_path / "plugin.yaml"
        if not manifest_path.exists():
            manifest_path = plugin_path / "plugin.yml"

        try:
            manifest = PluginManifest.from_yaml(manifest_path)

            # Apply trust override
            if manifest.name in self.trust_overrides:
                manifest.trust_level = self.trust_overrides[manifest.name]

            # Validate capabilities against trust level
            allowed_caps = DEFAULT_CAPABILITIES.get(manifest.trust_level, set())

            # Check all requested capabilities are allowed
            for cap in manifest.capabilities:
                if cap not in allowed_caps:
                    logger.warning(
                        f"Plugin {name} requests capability {cap.value} "
                        f"not allowed for trust level {manifest.trust_level.value}"
                    )

            # Create loaded plugin
            loaded = LoadedPlugin(
                manifest=manifest,
                path=plugin_path,
            )

            # Load tools
            for tool in manifest.tools:
                handler = self._load_entry_point(plugin_path, tool.entry_point)
                if handler:
                    tool.handler = handler
                    loaded.loaded_tools[tool.name] = handler
                    self._tool_index[tool.name] = name

            # Load workflows
            for workflow in manifest.workflows:
                wf_data = self._load_workflow(plugin_path, workflow.entry_point)
                if wf_data:
                    loaded.loaded_workflows[workflow.name] = wf_data
                    self._workflow_index[workflow.name] = name

            self.plugins[name] = loaded
            logger.info(f"Loaded plugin: {name} v{manifest.version}")
            return loaded

        except Exception as e:
            logger.error(f"Failed to load plugin {name}: {e}")
            return None

    def _load_entry_point(self, plugin_path: Path, entry_point: str) -> Optional[Callable]:
        """
        Load a function from an entry point specification.

        Args:
            plugin_path: Base path of the plugin
            entry_point: "module:function" format

        Returns:
            Callable if successful
        """
        if ":" not in entry_point:
            logger.error(f"Invalid entry point format: {entry_point}")
            return None

        module_name, func_name = entry_point.rsplit(":", 1)

        # Find module file
        module_file = plugin_path / f"{module_name}.py"
        if not module_file.exists():
            # Try as package
            module_file = plugin_path / module_name / "__init__.py"
            if not module_file.exists():
                logger.error(f"Module not found: {module_name}")
                return None

        try:
            spec = importlib.util.spec_from_file_location(
                f"ragix_plugin_{module_name}",
                module_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)

                if hasattr(module, func_name):
                    return getattr(module, func_name)
                else:
                    logger.error(f"Function {func_name} not found in {module_name}")

        except Exception as e:
            logger.error(f"Failed to load entry point {entry_point}: {e}")

        return None

    def _load_workflow(self, plugin_path: Path, entry_point: str) -> Optional[Dict]:
        """
        Load a workflow from an entry point.

        Args:
            plugin_path: Base path of the plugin
            entry_point: YAML path or "module:function"

        Returns:
            Workflow data if successful
        """
        if entry_point.endswith((".yaml", ".yml")):
            # Load YAML workflow
            wf_path = plugin_path / entry_point
            if wf_path.exists():
                with open(wf_path, 'r') as f:
                    return yaml.safe_load(f)
        else:
            # Load as function
            handler = self._load_entry_point(plugin_path, entry_point)
            if handler:
                return {"handler": handler}

        return None

    def load_all(self) -> int:
        """
        Load all discovered plugins.

        Returns:
            Number of plugins loaded
        """
        discovered = self.discover()
        count = 0

        for name in discovered:
            if self.load_plugin(name):
                count += 1

        return count

    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool handler function if found
        """
        if name not in self._tool_index:
            return None

        plugin_name = self._tool_index[name]
        plugin = self.plugins.get(plugin_name)

        if not plugin or not plugin.enabled:
            return None

        return plugin.loaded_tools.get(name)

    def get_workflow(self, name: str) -> Optional[Dict]:
        """
        Get a workflow by name.

        Args:
            name: Workflow name

        Returns:
            Workflow data if found
        """
        if name not in self._workflow_index:
            return None

        plugin_name = self._workflow_index[name]
        plugin = self.plugins.get(plugin_name)

        if not plugin or not plugin.enabled:
            return None

        return plugin.loaded_workflows.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools from plugins.

        Returns:
            List of tool information dictionaries
        """
        tools = []

        for plugin_name, plugin in self.plugins.items():
            if not plugin.enabled:
                continue

            for tool in plugin.manifest.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "plugin": plugin_name,
                    "version": plugin.manifest.version,
                    "trust": plugin.manifest.trust_level.value,
                    "loaded": tool.name in plugin.loaded_tools,
                })

        return tools

    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all available workflows from plugins.

        Returns:
            List of workflow information dictionaries
        """
        workflows = []

        for plugin_name, plugin in self.plugins.items():
            if not plugin.enabled:
                continue

            for wf in plugin.manifest.workflows:
                workflows.append({
                    "name": wf.name,
                    "description": wf.description,
                    "plugin": plugin_name,
                    "version": plugin.manifest.version,
                    "trust": plugin.manifest.trust_level.value,
                    "loaded": wf.name in plugin.loaded_workflows,
                })

        return workflows

    def check_capability(
        self,
        plugin_name: str,
        capability: PluginCapability
    ) -> bool:
        """
        Check if a plugin has a specific capability.

        Args:
            plugin_name: Name of the plugin
            capability: Capability to check

        Returns:
            True if capability is granted
        """
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return False

        allowed = DEFAULT_CAPABILITIES.get(plugin.manifest.trust_level, set())
        return capability in allowed and capability in plugin.manifest.capabilities

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin."""
        if name in self.plugins:
            self.plugins[name].enabled = True
            return True
        return False

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin."""
        if name in self.plugins:
            self.plugins[name].enabled = False
            return True
        return False

    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if unloaded
        """
        if name not in self.plugins:
            return False

        plugin = self.plugins[name]

        # Remove from indexes
        for tool_name in list(self._tool_index.keys()):
            if self._tool_index[tool_name] == name:
                del self._tool_index[tool_name]

        for wf_name in list(self._workflow_index.keys()):
            if self._workflow_index[wf_name] == name:
                del self._workflow_index[wf_name]

        del self.plugins[name]
        logger.info(f"Unloaded plugin: {name}")
        return True

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a plugin.

        Args:
            name: Plugin name

        Returns:
            Plugin information dictionary
        """
        plugin = self.plugins.get(name)
        if not plugin:
            # Try to discover it
            discovered = self.discover()
            if name in discovered:
                manifest = discovered[name]
                return {
                    "name": manifest.name,
                    "version": manifest.version,
                    "description": manifest.description,
                    "type": manifest.plugin_type.value,
                    "trust": manifest.trust_level.value,
                    "author": manifest.author,
                    "homepage": manifest.homepage,
                    "license": manifest.license,
                    "loaded": False,
                    "enabled": False,
                    "tools": [t.name for t in manifest.tools],
                    "workflows": [w.name for w in manifest.workflows],
                }
            return None

        return {
            "name": plugin.manifest.name,
            "version": plugin.manifest.version,
            "description": plugin.manifest.description,
            "type": plugin.manifest.plugin_type.value,
            "trust": plugin.manifest.trust_level.value,
            "author": plugin.manifest.author,
            "homepage": plugin.manifest.homepage,
            "license": plugin.manifest.license,
            "path": str(plugin.path),
            "loaded": True,
            "enabled": plugin.enabled,
            "tools": list(plugin.loaded_tools.keys()),
            "workflows": list(plugin.loaded_workflows.keys()),
            "capabilities": [c.value for c in plugin.manifest.capabilities],
        }


def compute_plugin_checksum(plugin_path: Path) -> str:
    """
    Compute SHA256 checksum of plugin contents.

    Args:
        plugin_path: Path to plugin directory

    Returns:
        Hex-encoded SHA256 hash
    """
    hasher = hashlib.sha256()

    for file_path in sorted(plugin_path.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith("."):
            # Add relative path to hash
            rel_path = file_path.relative_to(plugin_path)
            hasher.update(str(rel_path).encode())

            # Add file contents
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)

    return hasher.hexdigest()


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
