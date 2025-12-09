"""
RAGIX Project RAG - Configuration Management

Handles .RAG/config.yaml loading, saving, and profile management.

Profiles:
    - docs_only: Scientific/functional docs, long sentences, reasoning
    - mixed_docs_code: Tech documents mixed with code (typical audit)
    - code_only: Source-code-centric projects

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-09
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

RAG_DIR_NAME = ".RAG"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_VERSION = 1


class ProfileType(str, Enum):
    """Indexing profile types."""
    DOCS_ONLY = "docs_only"
    MIXED_DOCS_CODE = "mixed_docs_code"
    CODE_ONLY = "code_only"


# =============================================================================
# Profile Definitions
# =============================================================================

@dataclass
class IndexingProfile:
    """
    Indexing profile configuration.

    Defines how files are chunked and processed based on content type.
    """
    name: str
    description: str
    chunk_size: int = 512          # tokens or ~characters
    chunk_overlap: int = 64        # overlap between chunks
    level: str = "line+paragraph"  # chunking strategy
    multilingual: bool = True      # FR+EN support
    code_mode: bool = True         # treat code specially

    # Embedding configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexingProfile":
        return cls(**data)


# Predefined profiles (from REVIEW_RAG.md)
PROFILE_DOCS_ONLY = IndexingProfile(
    name="docs_only",
    description="Scientific / functional docs, long sentences, reasoning",
    chunk_size=1024,
    chunk_overlap=128,
    level="sentence+paragraph",
    multilingual=True,
    code_mode=False,
)

PROFILE_MIXED = IndexingProfile(
    name="mixed_docs_code",
    description="Tech documents mixed with code (typical audit project)",
    chunk_size=512,
    chunk_overlap=64,
    level="line+paragraph",
    multilingual=True,
    code_mode=True,
)

PROFILE_CODE_ONLY = IndexingProfile(
    name="code_only",
    description="Source-code-centric project",
    chunk_size=256,
    chunk_overlap=32,
    level="line-based",
    multilingual=False,
    code_mode=True,
)

DEFAULT_PROFILES = {
    ProfileType.DOCS_ONLY: PROFILE_DOCS_ONLY,
    ProfileType.MIXED_DOCS_CODE: PROFILE_MIXED,
    ProfileType.CODE_ONLY: PROFILE_CODE_ONLY,
}


# =============================================================================
# Vector Store Configuration
# =============================================================================

@dataclass
class VectorStoreConfig:
    """ChromaDB vector store configuration."""
    backend: str = "chroma"
    path: str = "chroma"           # relative to .RAG/
    use_gpu: str = "auto"          # "auto", "always", "never"
    collection_prefix: str = "rag" # prefix for collection names

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorStoreConfig":
        return cls(**data)


# =============================================================================
# Indexing Filters
# =============================================================================

@dataclass
class IndexingFilters:
    """File inclusion/exclusion filters."""
    include_globs: List[str] = field(default_factory=lambda: [
        # Code
        "**/*.java", "**/*.py", "**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx",
        "**/*.c", "**/*.cpp", "**/*.h", "**/*.hpp", "**/*.go", "**/*.rs",
        "**/*.rb", "**/*.php", "**/*.swift", "**/*.kt", "**/*.scala",
        # Config
        "**/*.xml", "**/*.json", "**/*.yaml", "**/*.yml", "**/*.toml",
        "**/*.properties", "**/*.ini", "**/*.cfg",
        # Docs
        "**/*.md", "**/*.txt", "**/*.rst",
        # Office (require conversion)
        "**/*.docx", "**/*.pptx", "**/*.xlsx", "**/*.pdf",
        "**/*.odt", "**/*.odp", "**/*.ods",
        # Build
        "**/pom.xml", "**/build.gradle", "**/Makefile", "**/CMakeLists.txt",
    ])

    exclude_globs: List[str] = field(default_factory=lambda: [
        ".git/**",
        ".RAG/**",
        ".ragix/**",
        "target/**",
        "build/**",
        "dist/**",
        "node_modules/**",
        "__pycache__/**",
        "*.pyc",
        ".venv/**",
        "venv/**",
        ".idea/**",
        ".vscode/**",
    ])

    # Max file size for indexing (10 MB default)
    max_file_size_bytes: int = 10 * 1024 * 1024

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexingFilters":
        return cls(**data)


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class RAGConfig:
    """
    Complete RAG configuration for a project.

    Stored in .RAG/config.yaml
    """
    version: int = CONFIG_VERSION
    active_profile: str = ProfileType.MIXED_DOCS_CODE.value

    # Profiles (can be customized per project)
    profiles: Dict[str, IndexingProfile] = field(default_factory=lambda: {
        p.name: p for p in DEFAULT_PROFILES.values()
    })

    # Vector store settings
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    # File filters
    indexing: IndexingFilters = field(default_factory=IndexingFilters)

    # Project metadata
    project_name: Optional[str] = None
    project_description: Optional[str] = None

    def get_active_profile(self) -> IndexingProfile:
        """Get the currently active indexing profile."""
        if self.active_profile in self.profiles:
            return self.profiles[self.active_profile]
        return PROFILE_MIXED  # fallback

    def set_active_profile(self, profile_name: str) -> None:
        """Set the active profile by name."""
        if profile_name in self.profiles:
            self.active_profile = profile_name
        else:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(self.profiles.keys())}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "version": self.version,
            "active_profile": self.active_profile,
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "vector_store": self.vector_store.to_dict(),
            "indexing": self.indexing.to_dict(),
            "project_name": self.project_name,
            "project_description": self.project_description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """Create from dictionary (YAML deserialization)."""
        profiles = {}
        if "profiles" in data:
            for name, profile_data in data["profiles"].items():
                profiles[name] = IndexingProfile.from_dict(profile_data)

        vector_store = VectorStoreConfig()
        if "vector_store" in data:
            vector_store = VectorStoreConfig.from_dict(data["vector_store"])

        indexing = IndexingFilters()
        if "indexing" in data:
            indexing = IndexingFilters.from_dict(data["indexing"])

        return cls(
            version=data.get("version", CONFIG_VERSION),
            active_profile=data.get("active_profile", ProfileType.MIXED_DOCS_CODE.value),
            profiles=profiles if profiles else {p.name: p for p in DEFAULT_PROFILES.values()},
            vector_store=vector_store,
            indexing=indexing,
            project_name=data.get("project_name"),
            project_description=data.get("project_description"),
        )


# =============================================================================
# Load / Save Functions
# =============================================================================

def get_rag_dir(project_root: Path) -> Path:
    """Get the .RAG directory path for a project."""
    return project_root / RAG_DIR_NAME


def get_config_path(project_root: Path) -> Path:
    """Get the config.yaml path for a project."""
    return get_rag_dir(project_root) / CONFIG_FILE_NAME


def load_config(project_root: Path) -> Optional[RAGConfig]:
    """
    Load RAG configuration from .RAG/config.yaml.

    Args:
        project_root: Project root directory

    Returns:
        RAGConfig if config exists, None otherwise
    """
    config_path = get_config_path(project_root)

    if not config_path.exists():
        logger.debug(f"No RAG config found at {config_path}")
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return get_default_config()

        return RAGConfig.from_dict(data)

    except Exception as e:
        logger.error(f"Failed to load RAG config from {config_path}: {e}")
        return None


def save_config(project_root: Path, config: RAGConfig) -> bool:
    """
    Save RAG configuration to .RAG/config.yaml.

    Args:
        project_root: Project root directory
        config: Configuration to save

    Returns:
        True if saved successfully
    """
    rag_dir = get_rag_dir(project_root)
    config_path = get_config_path(project_root)

    try:
        # Create .RAG directory if needed
        rag_dir.mkdir(parents=True, exist_ok=True)

        # Write config
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(f"Saved RAG config to {config_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save RAG config to {config_path}: {e}")
        return False


def get_default_config(
    project_name: Optional[str] = None,
    profile: ProfileType = ProfileType.MIXED_DOCS_CODE,
) -> RAGConfig:
    """
    Get default RAG configuration.

    Args:
        project_name: Optional project name
        profile: Initial profile to use

    Returns:
        Default RAGConfig
    """
    return RAGConfig(
        active_profile=profile.value,
        project_name=project_name,
    )


def ensure_rag_initialized(
    project_root: Path,
    profile: ProfileType = ProfileType.MIXED_DOCS_CODE,
    project_name: Optional[str] = None,
) -> RAGConfig:
    """
    Ensure .RAG/ is initialized with config.yaml.

    Creates default config if not exists, loads existing otherwise.

    Args:
        project_root: Project root directory
        profile: Profile to use for new config
        project_name: Optional project name

    Returns:
        RAGConfig (existing or newly created)
    """
    existing = load_config(project_root)
    if existing:
        return existing

    # Create new config
    config = get_default_config(
        project_name=project_name or project_root.name,
        profile=profile,
    )

    save_config(project_root, config)
    return config


def has_rag_index(project_root: Path) -> bool:
    """Check if a project has an existing RAG index."""
    rag_dir = get_rag_dir(project_root)
    config_path = get_config_path(project_root)
    chroma_path = rag_dir / "chroma"

    return config_path.exists() and chroma_path.exists()
