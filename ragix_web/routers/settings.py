"""
RAGIX Settings Router - RAG and Quality configuration endpoints

Provides API for managing RAG parameters, quality settings, and configuration profiles.
These settings are session-scoped by default but can be persisted to ragix.yaml.

Supports three indexing modes:
- Pure Code: Optimized for code repositories
- Code + Docs: Hybrid code and documentation
- Pure Docs: Document-only with specialized profiles

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-01-20
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


# =============================================================================
# Indexing Mode Constants
# =============================================================================

INDEXING_MODES = {
    "pure_code": {
        "name": "Pure Code",
        "description": "Optimized for code repositories without documentation",
        "icon": "code",
    },
    "code_docs": {
        "name": "Code + Docs",
        "description": "Hybrid mode for code with documentation",
        "icon": "file-code",
    },
    "pure_docs": {
        "name": "Pure Docs",
        "description": "Document-only indexing with specialized profiles",
        "icon": "file-text",
    },
}


# =============================================================================
# Session State for Settings
# =============================================================================

_settings_state: Dict[str, Dict[str, Any]] = {}


def _load_docs_profiles() -> Dict[str, Any]:
    """Load docs profiles from YAML registry."""
    try:
        profiles_path = Path(__file__).parent.parent.parent / "ragix_kernels" / "docs" / "schemas" / "docs_profiles.yaml"
        if profiles_path.exists():
            with open(profiles_path, 'r') as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load docs profiles: {e}")
    return {}


def _get_settings_state(session_id: str) -> Dict[str, Any]:
    """Get or initialize settings state for a session."""
    if session_id not in _settings_state:
        # Load defaults from config
        try:
            from ragix_core.config import get_config
            config = get_config()
            _settings_state[session_id] = {
                "indexing_mode": "pure_docs",  # Default mode
                "active_profile": None,        # Active docs profile
                "profile_overrides": {},       # User overrides to profile
                "rag": {
                    "embedding_model": config.search.embedding_model,
                    "fusion_strategy": config.search.fusion_strategy,
                    "bm25_weight": config.search.bm25_weight,
                    "vector_weight": config.search.vector_weight,
                    "top_k": config.search.top_k,
                    "rrf_k": config.search.rrf_k,
                    "chunk_size": config.search.chunk_size,
                    "chunk_overlap": config.search.chunk_overlap,
                    "max_file_size_mb": config.search.max_file_size_mb,
                },
                "quality": {
                    "enable": True,
                    "task_focus": "mixed",
                    "quality_threshold": 0.4,
                    "base_score": 0.5,
                    "enable_llm_intent": False,
                    "llm_intent_model": "granite3.1-moe:3b",
                    "intent_labels": ["descriptive", "prescriptive", "decisional", "informational"],
                    "mri_auto_threshold": 0.75,
                    "mri_assisted_threshold": 0.45,
                    "sri_auto_threshold": 0.70,
                    "sri_assisted_threshold": 0.50,
                },
                "chunking": {
                    "strategy": "semantic",
                    "chunk_size": 1000,      # chars (≈250 tokens)
                    "overlap": 200,          # chars
                    # Legacy token-based (for profile compatibility)
                    "target_tokens": 250,
                    "min_tokens": 100,
                    "max_tokens": 500,
                    "overlap_tokens": 50,
                    "boundary_policy": "prefer_paragraphs",
                    "preserve_bullets": True,
                    "preserve_tables": True,
                },
                "structure": {
                    "detect_headings": True,
                    "heading_regex_profile": "md",
                    "keep_bullets_together": True,
                    "keep_tables_as_blocks": True,
                    "split_on_pagebreaks": False,
                },
                "retrieval": {
                    "top_k_default": 10,
                    "rerank": "none",
                    "rerank_top_n": 20,
                },
            }
        except Exception as e:
            logger.warning(f"Failed to load config defaults: {e}")
            _settings_state[session_id] = _get_default_settings()

    return _settings_state[session_id]


def _get_default_settings() -> Dict[str, Any]:
    """Get default settings when config is not available."""
    return {
        "indexing_mode": "pure_docs",
        "active_profile": None,
        "profile_overrides": {},
        "rag": {
            "embedding_model": "all-MiniLM-L6-v2",
            "fusion_strategy": "rrf",
            "bm25_weight": 0.5,
            "vector_weight": 0.5,
            "top_k": 10,
            "rrf_k": 60,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_file_size_mb": 10,
        },
        "quality": {
            "enable": True,
            "task_focus": "mixed",
            "quality_threshold": 0.4,
            "base_score": 0.5,
            "enable_llm_intent": False,
            "llm_intent_model": "granite3.1-moe:3b",
            "intent_labels": ["descriptive", "prescriptive", "decisional", "informational"],
            "mri_auto_threshold": 0.75,
            "mri_assisted_threshold": 0.45,
            "sri_auto_threshold": 0.70,
            "sri_assisted_threshold": 0.50,
        },
        "chunking": {
            "strategy": "semantic",
            "chunk_size": 1000,      # chars (≈250 tokens)
            "overlap": 200,          # chars
            # Legacy token-based (for profile compatibility)
            "target_tokens": 250,
            "min_tokens": 100,
            "max_tokens": 500,
            "overlap_tokens": 50,
            "boundary_policy": "prefer_paragraphs",
            "preserve_bullets": True,
            "preserve_tables": True,
        },
        "structure": {
            "detect_headings": True,
            "heading_regex_profile": "md",
            "keep_bullets_together": True,
            "keep_tables_as_blocks": True,
            "split_on_pagebreaks": False,
        },
        "retrieval": {
            "top_k_default": 10,
            "rerank": "none",
            "rerank_top_n": 20,
        },
    }


# =============================================================================
# Request/Response Models
# =============================================================================

class IndexingModeRequest(BaseModel):
    """Request to set indexing mode."""
    mode: str = Field(..., description="Indexing mode: pure_code, code_docs, pure_docs")


class RAGSettingsRequest(BaseModel):
    """Request to update RAG settings."""
    embedding_model: Optional[str] = None
    fusion_strategy: Optional[str] = None
    bm25_weight: Optional[float] = Field(None, ge=0, le=1)
    vector_weight: Optional[float] = Field(None, ge=0, le=1)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    rrf_k: Optional[int] = Field(None, ge=1, le=200)
    chunk_size: Optional[int] = Field(None, ge=100, le=10000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=1000)
    max_file_size_mb: Optional[int] = Field(None, ge=1, le=100)


class QualitySettingsRequest(BaseModel):
    """Request to update quality settings."""
    quality_threshold: Optional[float] = Field(None, ge=0, le=1)
    base_score: Optional[float] = Field(None, ge=0, le=1)
    enable_llm_intent: Optional[bool] = None
    llm_intent_model: Optional[str] = None
    mri_auto_threshold: Optional[float] = Field(None, ge=0, le=1)
    mri_assisted_threshold: Optional[float] = Field(None, ge=0, le=1)
    sri_auto_threshold: Optional[float] = Field(None, ge=0, le=1)
    sri_assisted_threshold: Optional[float] = Field(None, ge=0, le=1)


class ChunkingSettingsRequest(BaseModel):
    """Request to update chunking settings (all values in chars)."""
    strategy: Optional[str] = None
    chunk_size: Optional[int] = Field(None, ge=300, le=5000, description="Chunk size in chars")
    overlap: Optional[int] = Field(None, ge=0, le=800, description="Overlap in chars")
    # Legacy token-based fields (for backwards compatibility)
    target_tokens: Optional[int] = Field(None, ge=50, le=2000)
    min_tokens: Optional[int] = Field(None, ge=20, le=1000)
    max_tokens: Optional[int] = Field(None, ge=100, le=2000)
    overlap_tokens: Optional[int] = Field(None, ge=0, le=500)
    boundary_policy: Optional[str] = None
    preserve_bullets: Optional[bool] = None
    preserve_tables: Optional[bool] = None


class StructureSettingsRequest(BaseModel):
    """Request to update structure parsing settings."""
    detect_headings: Optional[bool] = None
    heading_regex_profile: Optional[str] = None
    keep_bullets_together: Optional[bool] = None
    keep_tables_as_blocks: Optional[bool] = None
    split_on_pagebreaks: Optional[bool] = None


class RetrievalSettingsRequest(BaseModel):
    """Request to update retrieval settings."""
    top_k_default: Optional[int] = Field(None, ge=1, le=100)
    rerank: Optional[str] = None
    rerank_top_n: Optional[int] = Field(None, ge=1, le=100)


class QualitySettingsRequest(BaseModel):
    """Request to update quality settings."""
    enable: Optional[bool] = None
    task_focus: Optional[str] = None
    quality_threshold: Optional[float] = Field(None, ge=0, le=1)
    base_score: Optional[float] = Field(None, ge=0, le=1)
    enable_llm_intent: Optional[bool] = None
    llm_intent_model: Optional[str] = None
    intent_labels: Optional[List[str]] = None
    mri_auto_threshold: Optional[float] = Field(None, ge=0, le=1)
    mri_assisted_threshold: Optional[float] = Field(None, ge=0, le=1)
    sri_auto_threshold: Optional[float] = Field(None, ge=0, le=1)
    sri_assisted_threshold: Optional[float] = Field(None, ge=0, le=1)


class ProfileInfo(BaseModel):
    """Information about a configuration profile."""
    id: str
    name: str
    description: str
    icon: Optional[str] = None
    use_cases: Optional[List[str]] = None
    settings_preview: Dict[str, Any]
    hints: Optional[Dict[str, str]] = None


# =============================================================================
# RAG Settings Endpoints
# =============================================================================

@router.get("/rag")
async def get_rag_settings(session_id: str = Query("default")) -> Dict[str, Any]:
    """
    Get current RAG settings.

    Returns:
        RAG configuration including embedding model, fusion strategy, weights, etc.
    """
    state = _get_settings_state(session_id)
    return {
        "session_id": session_id,
        "settings": state["rag"],
    }


@router.post("/rag")
async def update_rag_settings(
    request: RAGSettingsRequest,
    session_id: str = Query("default"),
) -> Dict[str, Any]:
    """
    Update RAG settings for the session.

    Only provided fields are updated; others remain unchanged.
    Changes are session-scoped and do not persist to ragix.yaml.
    """
    state = _get_settings_state(session_id)

    # Validate fusion strategy
    if request.fusion_strategy:
        valid_strategies = ["rrf", "weighted", "interleave", "bm25_only", "vector_only"]
        if request.fusion_strategy not in valid_strategies:
            raise HTTPException(400, f"Invalid fusion strategy. Valid: {valid_strategies}")

    # Validate embedding model
    if request.embedding_model:
        valid_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"]
        if request.embedding_model not in valid_models:
            raise HTTPException(400, f"Invalid embedding model. Valid: {valid_models}")

    # Update only provided fields
    for field in request.model_fields:
        value = getattr(request, field)
        if value is not None:
            state["rag"][field] = value

    return {
        "session_id": session_id,
        "settings": state["rag"],
        "status": "updated",
    }


# =============================================================================
# Quality Settings Endpoints
# =============================================================================

@router.get("/quality")
async def get_quality_settings(session_id: str = Query("default")) -> Dict[str, Any]:
    """
    Get current quality assessment settings.

    Returns:
        Quality configuration including thresholds, LLM settings, readiness indices.
    """
    state = _get_settings_state(session_id)
    return {
        "session_id": session_id,
        "settings": state["quality"],
    }


@router.post("/quality")
async def update_quality_settings(
    request: QualitySettingsRequest,
    session_id: str = Query("default"),
) -> Dict[str, Any]:
    """
    Update quality assessment settings for the session.

    Only provided fields are updated; others remain unchanged.
    """
    state = _get_settings_state(session_id)

    # Update only provided fields
    for field in request.model_fields:
        value = getattr(request, field)
        if value is not None:
            state["quality"][field] = value

    return {
        "session_id": session_id,
        "settings": state["quality"],
        "status": "updated",
    }


# =============================================================================
# Chunking Settings Endpoints
# =============================================================================

@router.get("/chunking")
async def get_chunking_settings(session_id: str = Query("default")) -> Dict[str, Any]:
    """
    Get current chunking settings.

    Returns:
        Chunking configuration including strategy, target tokens, overlap.
    """
    state = _get_settings_state(session_id)
    return {
        "session_id": session_id,
        "settings": state["chunking"],
    }


@router.post("/chunking")
async def update_chunking_settings(
    request: ChunkingSettingsRequest,
    session_id: str = Query("default"),
) -> Dict[str, Any]:
    """
    Update chunking settings for the session.

    Supports both char-based (chunk_size, overlap) and legacy token-based fields.
    Only provided fields are updated; others remain unchanged.
    """
    state = _get_settings_state(session_id)

    # Validate strategy
    if request.strategy:
        valid_strategies = ["semantic", "fixed", "hybrid", "token", "paragraph"]
        if request.strategy not in valid_strategies:
            raise HTTPException(400, f"Invalid chunking strategy. Valid: {valid_strategies}")

    # Update only provided fields
    for field in request.model_fields:
        value = getattr(request, field)
        if value is not None:
            state["chunking"][field] = value

    return {
        "session_id": session_id,
        "settings": state["chunking"],
        "status": "updated",
    }


# =============================================================================
# Indexing Mode Endpoints
# =============================================================================

@router.get("/indexing-modes")
async def list_indexing_modes() -> Dict[str, Any]:
    """List available indexing modes."""
    return {
        "modes": INDEXING_MODES,
        "total": len(INDEXING_MODES),
    }


@router.get("/indexing-mode")
async def get_indexing_mode(session_id: str = Query("default")) -> Dict[str, Any]:
    """Get current indexing mode for the session."""
    state = _get_settings_state(session_id)
    return {
        "session_id": session_id,
        "mode": state.get("indexing_mode", "pure_docs"),
        "mode_info": INDEXING_MODES.get(state.get("indexing_mode", "pure_docs")),
    }


@router.post("/indexing-mode")
async def set_indexing_mode(
    request: IndexingModeRequest,
    session_id: str = Query("default"),
) -> Dict[str, Any]:
    """Set the indexing mode for the session."""
    if request.mode not in INDEXING_MODES:
        raise HTTPException(400, f"Invalid mode. Valid: {list(INDEXING_MODES.keys())}")

    state = _get_settings_state(session_id)
    state["indexing_mode"] = request.mode

    return {
        "session_id": session_id,
        "mode": request.mode,
        "mode_info": INDEXING_MODES[request.mode],
        "status": "updated",
    }


# =============================================================================
# Structure Settings Endpoints
# =============================================================================

@router.get("/structure")
async def get_structure_settings(session_id: str = Query("default")) -> Dict[str, Any]:
    """Get current document structure parsing settings."""
    state = _get_settings_state(session_id)
    return {
        "session_id": session_id,
        "settings": state.get("structure", {}),
    }


@router.post("/structure")
async def update_structure_settings(
    request: StructureSettingsRequest,
    session_id: str = Query("default"),
) -> Dict[str, Any]:
    """Update document structure parsing settings."""
    state = _get_settings_state(session_id)

    # Validate heading_regex_profile
    if request.heading_regex_profile:
        valid_profiles = ["md", "docx", "pdf_heuristic", "custom"]
        if request.heading_regex_profile not in valid_profiles:
            raise HTTPException(400, f"Invalid heading profile. Valid: {valid_profiles}")

    # Update only provided fields
    for field in request.model_fields:
        value = getattr(request, field)
        if value is not None:
            state["structure"][field] = value

    return {
        "session_id": session_id,
        "settings": state["structure"],
        "status": "updated",
    }


# =============================================================================
# Retrieval Settings Endpoints
# =============================================================================

@router.get("/retrieval")
async def get_retrieval_settings(session_id: str = Query("default")) -> Dict[str, Any]:
    """Get current retrieval settings."""
    state = _get_settings_state(session_id)
    return {
        "session_id": session_id,
        "settings": state.get("retrieval", {}),
    }


@router.post("/retrieval")
async def update_retrieval_settings(
    request: RetrievalSettingsRequest,
    session_id: str = Query("default"),
) -> Dict[str, Any]:
    """Update retrieval settings."""
    state = _get_settings_state(session_id)

    # Validate rerank
    if request.rerank:
        valid_reranks = ["none", "cross_encoder", "llm"]
        if request.rerank not in valid_reranks:
            raise HTTPException(400, f"Invalid rerank method. Valid: {valid_reranks}")

    # Update only provided fields
    for field in request.model_fields:
        value = getattr(request, field)
        if value is not None:
            state["retrieval"][field] = value

    return {
        "session_id": session_id,
        "settings": state["retrieval"],
        "status": "updated",
    }


# =============================================================================
# Profile Management Endpoints (from YAML registry)
# =============================================================================

@router.get("/profiles")
async def list_profiles(mode: Optional[str] = Query(None)) -> Dict[str, Any]:
    """
    List available configuration profiles.

    Args:
        mode: Filter by indexing mode (pure_code, code_docs, pure_docs)

    Returns:
        List of profiles with their descriptions and preview of key settings.
    """
    # Load profiles from YAML registry
    yaml_profiles = _load_docs_profiles()

    profiles = []

    # Add docs profiles from YAML
    for profile_id, profile_data in yaml_profiles.items():
        # Skip metadata entries
        if profile_id in ("metadata", "guardrails"):
            continue

        if not isinstance(profile_data, dict) or "name" not in profile_data:
            continue

        # Build settings preview
        settings_preview = {}
        if "chunking" in profile_data:
            for key in ["strategy", "target_tokens", "min_tokens", "max_tokens"]:
                if key in profile_data["chunking"]:
                    settings_preview[f"chunking.{key}"] = profile_data["chunking"][key]
        if "quality" in profile_data:
            if "task_focus" in profile_data["quality"]:
                settings_preview["quality.task_focus"] = profile_data["quality"]["task_focus"]
            if "intent_classifier" in profile_data["quality"]:
                ic = profile_data["quality"]["intent_classifier"]
                if "enable" in ic:
                    settings_preview["quality.enable_llm_intent"] = ic["enable"]
        if "retrieval" in profile_data:
            for key in ["top_k_default", "rerank"]:
                if key in profile_data["retrieval"]:
                    settings_preview[f"retrieval.{key}"] = profile_data["retrieval"][key]

        profiles.append(ProfileInfo(
            id=profile_id,
            name=profile_data.get("name", profile_id),
            description=profile_data.get("description", ""),
            icon=profile_data.get("icon"),
            use_cases=profile_data.get("use_cases"),
            settings_preview=settings_preview,
            hints=profile_data.get("hints"),
        ))

    return {
        "profiles": [p.model_dump() for p in profiles],
        "total": len(profiles),
        "guardrails": yaml_profiles.get("guardrails", {}),
    }


@router.get("/profiles/{profile_id}")
async def get_profile_detail(profile_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific profile.

    Returns full profile configuration including all settings.
    """
    yaml_profiles = _load_docs_profiles()

    if profile_id not in yaml_profiles:
        raise HTTPException(404, f"Unknown profile: {profile_id}")

    profile_data = yaml_profiles[profile_id]

    return {
        "id": profile_id,
        "profile": profile_data,
    }


@router.post("/profiles/{profile_id}/apply")
async def apply_profile(
    profile_id: str,
    session_id: str = Query("default"),
) -> Dict[str, Any]:
    """
    Apply a configuration profile to the current session.

    This updates all settings to match the profile's recommended values.
    Shows any overrides that were previously applied.
    """
    # Load profiles from YAML
    yaml_profiles = _load_docs_profiles()

    if profile_id not in yaml_profiles:
        raise HTTPException(404, f"Unknown profile: {profile_id}")

    profile_data = yaml_profiles[profile_id]
    state = _get_settings_state(session_id)

    # Track what we're changing
    changes = []

    # Apply chunking settings
    if "chunking" in profile_data:
        for key, value in profile_data["chunking"].items():
            if key in state["chunking"]:
                old_value = state["chunking"][key]
                if old_value != value:
                    changes.append(f"chunking.{key}: {old_value} -> {value}")
                state["chunking"][key] = value
            else:
                state["chunking"][key] = value

    # Apply structure settings
    if "structure" in profile_data:
        for key, value in profile_data["structure"].items():
            if key in state["structure"]:
                state["structure"][key] = value
            else:
                state["structure"][key] = value

    # Apply quality settings
    if "quality" in profile_data:
        q = profile_data["quality"]
        if "enable" in q:
            state["quality"]["enable"] = q["enable"]
        if "task_focus" in q:
            state["quality"]["task_focus"] = q["task_focus"]
        if "intent_classifier" in q:
            ic = q["intent_classifier"]
            if "enable" in ic:
                state["quality"]["enable_llm_intent"] = ic["enable"]
            if "labels" in ic:
                state["quality"]["intent_labels"] = ic["labels"]

    # Apply retrieval settings
    if "retrieval" in profile_data:
        for key, value in profile_data["retrieval"].items():
            state["retrieval"][key] = value

    # Store the active profile
    state["active_profile"] = profile_id
    state["profile_overrides"] = {}  # Reset overrides when applying new profile

    return {
        "session_id": session_id,
        "profile_applied": profile_id,
        "settings": {
            "rag": state["rag"],
            "chunking": state["chunking"],
            "quality": state["quality"],
        },
        "status": "applied",
    }


# =============================================================================
# Unified Settings Endpoints
# =============================================================================

@router.get("/all")
async def get_all_settings(session_id: str = Query("default")) -> Dict[str, Any]:
    """
    Get all settings for the session.

    Returns:
        Complete settings including indexing mode, RAG, quality, chunking,
        structure, retrieval, and active profile.
    """
    state = _get_settings_state(session_id)
    return {
        "session_id": session_id,
        "indexing_mode": state.get("indexing_mode", "pure_docs"),
        "active_profile": state.get("active_profile"),
        "profile_overrides": state.get("profile_overrides", {}),
        "settings": {
            "rag": state["rag"],
            "quality": state["quality"],
            "chunking": state["chunking"],
            "structure": state.get("structure", {}),
            "retrieval": state.get("retrieval", {}),
        },
    }


@router.post("/reset")
async def reset_settings(session_id: str = Query("default")) -> Dict[str, Any]:
    """
    Reset all settings to defaults for the session.
    """
    if session_id in _settings_state:
        del _settings_state[session_id]

    state = _get_settings_state(session_id)
    return {
        "session_id": session_id,
        "indexing_mode": state.get("indexing_mode", "pure_docs"),
        "active_profile": state.get("active_profile"),
        "settings": {
            "rag": state["rag"],
            "quality": state["quality"],
            "chunking": state["chunking"],
            "structure": state.get("structure", {}),
            "retrieval": state.get("retrieval", {}),
        },
        "status": "reset",
    }


@router.get("/effective")
async def get_effective_settings(session_id: str = Query("default")) -> Dict[str, Any]:
    """
    Get the effective settings for indexing.

    Returns settings that would be used by the indexer, combining profile
    defaults with any user overrides.
    """
    state = _get_settings_state(session_id)
    active_profile = state.get("active_profile")
    overrides = state.get("profile_overrides", {})

    return {
        "session_id": session_id,
        "indexing_mode": state.get("indexing_mode", "pure_docs"),
        "active_profile": active_profile,
        "has_overrides": bool(overrides),
        "overrides": overrides,
        "effective": {
            "chunking": state["chunking"],
            "structure": state.get("structure", {}),
            "quality": state["quality"],
            "retrieval": state.get("retrieval", {}),
        },
    }


@router.post("/persist")
async def persist_settings(
    session_id: str = Query("default"),
    target: str = Query("ragix.yaml", description="Target: ragix.yaml or indexing_mode"),
) -> Dict[str, Any]:
    """
    Persist current session settings to ragix.yaml.

    WARNING: This modifies the configuration file on disk.

    Args:
        session_id: Session ID
        target: Where to save ("ragix.yaml" or current indexing mode name)
    """
    try:
        from ragix_core.config import get_config, save_config, find_config_file

        state = _get_settings_state(session_id)
        config = get_config()

        # Update SearchConfig
        config.search.embedding_model = state["rag"]["embedding_model"]
        config.search.fusion_strategy = state["rag"]["fusion_strategy"]
        config.search.bm25_weight = state["rag"]["bm25_weight"]
        config.search.vector_weight = state["rag"]["vector_weight"]
        config.search.top_k = state["rag"]["top_k"]
        config.search.rrf_k = state["rag"]["rrf_k"]
        config.search.chunk_size = state["rag"]["chunk_size"]
        config.search.chunk_overlap = state["rag"]["chunk_overlap"]
        config.search.max_file_size_mb = state["rag"]["max_file_size_mb"]

        # Find config path and save
        config_path = find_config_file()
        if config_path:
            save_config(config, config_path)
            return {
                "session_id": session_id,
                "status": "persisted",
                "config_path": str(config_path),
                "indexing_mode": state.get("indexing_mode"),
                "active_profile": state.get("active_profile"),
                "message": "Settings saved to ragix.yaml",
            }
        else:
            return {
                "session_id": session_id,
                "status": "error",
                "message": "No ragix.yaml found to save settings",
            }

    except Exception as e:
        raise HTTPException(500, f"Failed to persist settings: {e}")


# =============================================================================
# Settings Export/Import
# =============================================================================

@router.get("/export")
async def export_settings(session_id: str = Query("default")) -> Dict[str, Any]:
    """
    Export current settings as a JSON object for backup/sharing.
    """
    state = _get_settings_state(session_id)
    return {
        "session_id": session_id,
        "export_version": "1.0",
        "indexing_mode": state.get("indexing_mode", "pure_docs"),
        "active_profile": state.get("active_profile"),
        "settings": {
            "rag": state["rag"],
            "quality": state["quality"],
            "chunking": state["chunking"],
            "structure": state.get("structure", {}),
            "retrieval": state.get("retrieval", {}),
        },
    }


class ImportSettingsRequest(BaseModel):
    """Request to import settings."""
    settings: Dict[str, Any]
    indexing_mode: Optional[str] = None


@router.post("/import")
async def import_settings(
    request: ImportSettingsRequest,
    session_id: str = Query("default"),
) -> Dict[str, Any]:
    """
    Import settings from a previously exported JSON object.
    """
    state = _get_settings_state(session_id)

    # Apply imported settings
    settings = request.settings
    if "rag" in settings:
        state["rag"].update(settings["rag"])
    if "quality" in settings:
        state["quality"].update(settings["quality"])
    if "chunking" in settings:
        state["chunking"].update(settings["chunking"])
    if "structure" in settings:
        state["structure"].update(settings["structure"])
    if "retrieval" in settings:
        state["retrieval"].update(settings["retrieval"])

    if request.indexing_mode:
        state["indexing_mode"] = request.indexing_mode

    # Clear active profile since we imported custom settings
    state["active_profile"] = None

    return {
        "session_id": session_id,
        "status": "imported",
        "indexing_mode": state.get("indexing_mode"),
        "settings": {
            "rag": state["rag"],
            "quality": state["quality"],
            "chunking": state["chunking"],
            "structure": state.get("structure", {}),
            "retrieval": state.get("retrieval", {}),
        },
    }
