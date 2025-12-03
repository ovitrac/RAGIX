"""
RAGIX Unified Configuration System
===================================

Loads and manages configuration from ragix.yaml with environment variable overrides.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-26
"""

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from ragix_core.agent_config import AgentConfig, AgentMode

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class LLMConfig:
    """LLM backend configuration."""
    backend: str = "ollama"
    model: str = "qwen2.5:7b"  # Best performing model in v0.30 benchmarks
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None  # For cloud backends
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120


@dataclass
class MCPConfig:
    """MCP server configuration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 5173
    tools: List[str] = field(default_factory=lambda: [
        "ragix_chat", "ragix_scan_repo", "ragix_read_file",
        "ragix_search", "ragix_workflow", "ragix_health", "ragix_templates"
    ])


@dataclass
class SafetyConfig:
    """Safety and security configuration."""
    profile: str = "dev"
    allow_git_destructive: bool = False
    additional_denylist: List[str] = field(default_factory=list)
    air_gapped: bool = False  # If True, only allow local LLM backends
    log_hashing: bool = True  # Enable SHA256 log signatures
    hash_algorithm: str = "sha256"


@dataclass
class SearchConfig:
    """Search and retrieval configuration."""
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    fusion_strategy: str = "rrf"
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    top_k: int = 10
    index_path: str = ".ragix/index"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = ".agent_logs"
    commands_log: str = "commands.log"
    hash_log: str = "commands.log.sha256"
    max_log_size_mb: int = 100
    backup_count: int = 5


@dataclass
class WebUIConfig:
    """Web UI configuration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 8501
    theme: str = "dark"


@dataclass
class HardwareConfig:
    """Hardware profile configuration."""
    vram_gb: float = 8.0
    prefer_cpu: bool = False
    max_concurrent_models: int = 1


@dataclass
class RAGIXConfig:
    """Root configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    webui: WebUIConfig = field(default_factory=WebUIConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    sandbox_root: str = "."
    version: str = "0.7.1"


# =============================================================================
# Configuration Loader
# =============================================================================

def find_config_file(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find ragix.yaml by searching upward from start_path.

    Search order:
    1. start_path / ragix.yaml
    2. start_path / .ragix / ragix.yaml
    3. Parent directories (recursive)
    4. ~/.config/ragix/ragix.yaml
    5. /etc/ragix/ragix.yaml

    Args:
        start_path: Starting directory (defaults to cwd)

    Returns:
        Path to config file or None if not found
    """
    if start_path is None:
        start_path = Path.cwd()

    start_path = Path(start_path).resolve()

    # Search upward
    current = start_path
    for _ in range(10):  # Max 10 levels up
        candidates = [
            current / "ragix.yaml",
            current / ".ragix" / "ragix.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        parent = current.parent
        if parent == current:
            break
        current = parent

    # Check user config
    user_config = Path.home() / ".config" / "ragix" / "ragix.yaml"
    if user_config.exists():
        return user_config

    # Check system config
    system_config = Path("/etc/ragix/ragix.yaml")
    if system_config.exists():
        return system_config

    return None


def load_config(config_path: Optional[Path] = None) -> RAGIXConfig:
    """
    Load configuration from YAML file with environment variable overrides.

    Environment variables override config file values:
    - RAGIX_LLM_BACKEND -> llm.backend
    - RAGIX_LLM_MODEL -> llm.model
    - RAGIX_PROFILE -> safety.profile
    - RAGIX_AIR_GAPPED -> safety.air_gapped
    - RAGIX_MCP_ENABLED -> mcp.enabled
    - RAGIX_MCP_PORT -> mcp.port
    - RAGIX_LOG_HASHING -> safety.log_hashing
    - UNIX_RAG_MODEL -> llm.model (legacy)
    - UNIX_RAG_SANDBOX -> sandbox_root (legacy)
    - UNIX_RAG_PROFILE -> safety.profile (legacy)

    Args:
        config_path: Path to config file (auto-detected if None)

    Returns:
        RAGIXConfig instance
    """
    config = RAGIXConfig()

    # Find and load config file
    if config_path is None:
        config_path = find_config_file()

    if config_path and config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            config = _parse_config_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    else:
        logger.info("No config file found, using defaults")

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Validate configuration
    _validate_config(config)

    return config


def _parse_config_dict(data: Dict[str, Any]) -> RAGIXConfig:
    """Parse configuration dictionary into RAGIXConfig."""
    config = RAGIXConfig()

    # LLM config
    if "llm" in data:
        llm = data["llm"]
        config.llm = LLMConfig(
            backend=llm.get("backend", config.llm.backend),
            model=llm.get("model", config.llm.model),
            base_url=llm.get("base_url", config.llm.base_url),
            api_key=llm.get("api_key"),
            temperature=llm.get("temperature", config.llm.temperature),
            max_tokens=llm.get("max_tokens", config.llm.max_tokens),
            timeout=llm.get("timeout", config.llm.timeout),
        )

    # MCP config
    if "mcp" in data:
        mcp = data["mcp"]
        config.mcp = MCPConfig(
            enabled=mcp.get("enabled", config.mcp.enabled),
            host=mcp.get("host", config.mcp.host),
            port=mcp.get("port", config.mcp.port),
            tools=mcp.get("tools", config.mcp.tools),
        )

    # Safety config
    if "safety" in data:
        safety = data["safety"]
        config.safety = SafetyConfig(
            profile=safety.get("profile", config.safety.profile),
            allow_git_destructive=safety.get("allow_git_destructive", config.safety.allow_git_destructive),
            additional_denylist=safety.get("additional_denylist", config.safety.additional_denylist),
            air_gapped=safety.get("air_gapped", config.safety.air_gapped),
            log_hashing=safety.get("log_hashing", config.safety.log_hashing),
            hash_algorithm=safety.get("hash_algorithm", config.safety.hash_algorithm),
        )

    # Search config
    if "search" in data:
        search = data["search"]
        config.search = SearchConfig(
            enabled=search.get("enabled", config.search.enabled),
            embedding_model=search.get("embedding_model", config.search.embedding_model),
            fusion_strategy=search.get("fusion_strategy", config.search.fusion_strategy),
            bm25_weight=search.get("bm25_weight", config.search.bm25_weight),
            vector_weight=search.get("vector_weight", config.search.vector_weight),
            top_k=search.get("top_k", config.search.top_k),
            index_path=search.get("index_path", config.search.index_path),
        )

    # Logging config
    if "logging" in data:
        log = data["logging"]
        config.logging = LoggingConfig(
            level=log.get("level", config.logging.level),
            log_dir=log.get("log_dir", config.logging.log_dir),
            commands_log=log.get("commands_log", config.logging.commands_log),
            hash_log=log.get("hash_log", config.logging.hash_log),
            max_log_size_mb=log.get("max_log_size_mb", config.logging.max_log_size_mb),
            backup_count=log.get("backup_count", config.logging.backup_count),
        )

    # Web UI config
    if "webui" in data:
        webui = data["webui"]
        config.webui = WebUIConfig(
            enabled=webui.get("enabled", config.webui.enabled),
            host=webui.get("host", config.webui.host),
            port=webui.get("port", config.webui.port),
            theme=webui.get("theme", config.webui.theme),
        )

    # Agents config (Planner-Worker-Verifier)
    if "agents" in data:
        config.agents = AgentConfig.from_dict(data["agents"])

    # Hardware config
    if "hardware" in data:
        hw = data["hardware"]
        config.hardware = HardwareConfig(
            vram_gb=hw.get("vram_gb", config.hardware.vram_gb),
            prefer_cpu=hw.get("prefer_cpu", config.hardware.prefer_cpu),
            max_concurrent_models=hw.get("max_concurrent_models", config.hardware.max_concurrent_models),
        )

    # Root level
    config.sandbox_root = data.get("sandbox_root", config.sandbox_root)
    config.version = data.get("version", config.version)

    return config


def _apply_env_overrides(config: RAGIXConfig) -> RAGIXConfig:
    """Apply environment variable overrides to config."""

    # New RAGIX_ prefixed variables
    if os.environ.get("RAGIX_LLM_BACKEND"):
        config.llm.backend = os.environ["RAGIX_LLM_BACKEND"]

    if os.environ.get("RAGIX_LLM_MODEL"):
        config.llm.model = os.environ["RAGIX_LLM_MODEL"]

    if os.environ.get("RAGIX_PROFILE"):
        config.safety.profile = os.environ["RAGIX_PROFILE"]

    if os.environ.get("RAGIX_AIR_GAPPED"):
        config.safety.air_gapped = os.environ["RAGIX_AIR_GAPPED"].lower() in ("true", "1", "yes")

    if os.environ.get("RAGIX_MCP_ENABLED"):
        config.mcp.enabled = os.environ["RAGIX_MCP_ENABLED"].lower() in ("true", "1", "yes")

    if os.environ.get("RAGIX_MCP_PORT"):
        config.mcp.port = int(os.environ["RAGIX_MCP_PORT"])

    if os.environ.get("RAGIX_LOG_HASHING"):
        config.safety.log_hashing = os.environ["RAGIX_LOG_HASHING"].lower() in ("true", "1", "yes")

    if os.environ.get("RAGIX_SANDBOX_ROOT"):
        config.sandbox_root = os.environ["RAGIX_SANDBOX_ROOT"]

    # Legacy UNIX_RAG_ variables (backward compatibility)
    if os.environ.get("UNIX_RAG_MODEL"):
        config.llm.model = os.environ["UNIX_RAG_MODEL"]

    if os.environ.get("UNIX_RAG_SANDBOX"):
        config.sandbox_root = os.environ["UNIX_RAG_SANDBOX"]

    if os.environ.get("UNIX_RAG_PROFILE"):
        config.safety.profile = os.environ["UNIX_RAG_PROFILE"]

    if os.environ.get("UNIX_RAG_ALLOW_GIT_DESTRUCTIVE"):
        config.safety.allow_git_destructive = os.environ["UNIX_RAG_ALLOW_GIT_DESTRUCTIVE"].lower() in ("true", "1", "yes")

    # Agent configuration overrides
    if os.environ.get("RAGIX_AGENT_MODE"):
        mode_str = os.environ["RAGIX_AGENT_MODE"].lower()
        try:
            config.agents.mode = AgentMode(mode_str)
        except ValueError:
            logger.warning(f"Unknown agent mode '{mode_str}', keeping {config.agents.mode.value}")

    if os.environ.get("RAGIX_PLANNER_MODEL"):
        config.agents.planner_model = os.environ["RAGIX_PLANNER_MODEL"]

    if os.environ.get("RAGIX_WORKER_MODEL"):
        config.agents.worker_model = os.environ["RAGIX_WORKER_MODEL"]

    if os.environ.get("RAGIX_VERIFIER_MODEL"):
        config.agents.verifier_model = os.environ["RAGIX_VERIFIER_MODEL"]

    return config


def _validate_config(config: RAGIXConfig) -> None:
    """Validate configuration and log warnings."""

    # Check air-gapped mode vs cloud backends
    if config.safety.air_gapped and config.llm.backend in ("claude", "openai"):
        logger.warning(
            f"Air-gapped mode enabled but LLM backend is '{config.llm.backend}' (cloud). "
            "This violates sovereignty. Forcing backend to 'ollama'."
        )
        config.llm.backend = "ollama"

    # Check profile validity
    valid_profiles = ("strict", "safe-read-only", "dev", "unsafe")
    if config.safety.profile not in valid_profiles:
        logger.warning(f"Unknown profile '{config.safety.profile}', defaulting to 'dev'")
        config.safety.profile = "dev"

    # Check search fusion strategy
    valid_strategies = ("rrf", "weighted", "interleave", "bm25_only", "vector_only")
    if config.search.fusion_strategy.lower() not in valid_strategies:
        logger.warning(f"Unknown fusion strategy '{config.search.fusion_strategy}', defaulting to 'rrf'")
        config.search.fusion_strategy = "rrf"


def save_config(config: RAGIXConfig, path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: RAGIXConfig instance
        path: Output path
    """
    data = {
        "version": config.version,
        "sandbox_root": config.sandbox_root,
        "llm": {
            "backend": config.llm.backend,
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
            "timeout": config.llm.timeout,
        },
        "mcp": {
            "enabled": config.mcp.enabled,
            "host": config.mcp.host,
            "port": config.mcp.port,
            "tools": config.mcp.tools,
        },
        "safety": {
            "profile": config.safety.profile,
            "allow_git_destructive": config.safety.allow_git_destructive,
            "additional_denylist": config.safety.additional_denylist,
            "air_gapped": config.safety.air_gapped,
            "log_hashing": config.safety.log_hashing,
            "hash_algorithm": config.safety.hash_algorithm,
        },
        "search": {
            "enabled": config.search.enabled,
            "embedding_model": config.search.embedding_model,
            "fusion_strategy": config.search.fusion_strategy,
            "bm25_weight": config.search.bm25_weight,
            "vector_weight": config.search.vector_weight,
            "top_k": config.search.top_k,
            "index_path": config.search.index_path,
        },
        "logging": {
            "level": config.logging.level,
            "log_dir": config.logging.log_dir,
            "commands_log": config.logging.commands_log,
            "hash_log": config.logging.hash_log,
            "max_log_size_mb": config.logging.max_log_size_mb,
            "backup_count": config.logging.backup_count,
        },
        "webui": {
            "enabled": config.webui.enabled,
            "host": config.webui.host,
            "port": config.webui.port,
            "theme": config.webui.theme,
        },
        "agents": config.agents.to_dict(),
        "hardware": {
            "vram_gb": config.hardware.vram_gb,
            "prefer_cpu": config.hardware.prefer_cpu,
            "max_concurrent_models": config.hardware.max_concurrent_models,
        },
    }

    # Don't save API keys
    if config.llm.api_key:
        data["llm"]["api_key"] = "*** SET VIA ENVIRONMENT VARIABLE ***"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to: {path}")


# =============================================================================
# Global Config Instance
# =============================================================================

_global_config: Optional[RAGIXConfig] = None


def get_config() -> RAGIXConfig:
    """Get the global configuration instance (lazy-loaded)."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def reload_config(config_path: Optional[Path] = None) -> RAGIXConfig:
    """Reload configuration from file."""
    global _global_config
    _global_config = load_config(config_path)
    return _global_config
