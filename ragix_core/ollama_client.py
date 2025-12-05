"""
Ollama API Client with Caching for RAGIX

This module provides a client for the Ollama API that:
- Fetches running model info (/api/ps)
- Fetches model details (/api/show)
- Caches results with configurable TTL
- Extracts VRAM usage, quantization, context size

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-12-05
"""

import logging
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import requests

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded Ollama model."""
    name: str
    # From /api/ps
    size_bytes: int = 0
    vram_bytes: int = 0
    expires_at: str = ""
    # From /api/show
    family: str = ""
    parameter_size: str = ""  # e.g., "7B", "14B"
    quantization: str = ""    # e.g., "Q4_K_M", "Q8_0", "F16"
    context_length: int = 0   # num_ctx from modelfile
    # Computed
    vram_gb: float = 0.0
    size_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "size_gb": self.size_gb,
            "vram_bytes": self.vram_bytes,
            "vram_gb": self.vram_gb,
            "family": self.family,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "expires_at": self.expires_at,
        }


@dataclass
class OllamaCache:
    """Cache for Ollama API responses."""
    # Cache entries: model_name -> (ModelInfo, timestamp)
    model_info: Dict[str, tuple] = field(default_factory=dict)
    # Running models cache
    running_models: List[str] = field(default_factory=list)
    running_models_timestamp: float = 0.0
    # TTL in seconds
    model_info_ttl: int = 300  # 5 minutes for model details
    running_models_ttl: int = 30  # 30 seconds for running status


class OllamaClient:
    """
    Client for Ollama API with caching.

    Example:
        client = OllamaClient()
        info = client.get_model_info("mistral:latest")
        print(f"VRAM: {info.vram_gb:.1f} GB, Quantization: {info.quantization}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_info_ttl: int = 300,
        running_models_ttl: int = 30,
        timeout: int = 10,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            model_info_ttl: Cache TTL for model info (seconds)
            running_models_ttl: Cache TTL for running models (seconds)
            timeout: Request timeout (seconds)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache = OllamaCache(
            model_info_ttl=model_info_ttl,
            running_models_ttl=running_models_ttl,
        )

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            return r.status_code == 200
        except Exception:
            return False

    def get_running_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of currently running/loaded models from /api/ps.

        Returns list of dicts with: name, size, vram, expires_at
        """
        now = time.time()

        # Check cache
        if not force_refresh:
            if (now - self._cache.running_models_timestamp) < self._cache.running_models_ttl:
                return self._cache.running_models

        try:
            r = requests.get(f"{self.base_url}/api/ps", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            models = []
            for m in data.get("models", []):
                model_data = {
                    "name": m.get("name", ""),
                    "size_bytes": m.get("size", 0),
                    "size_gb": m.get("size", 0) / (1024**3),
                    "vram_bytes": m.get("size_vram", 0),
                    "vram_gb": m.get("size_vram", 0) / (1024**3),
                    "expires_at": m.get("expires_at", ""),
                    "digest": m.get("digest", "")[:12],
                }
                models.append(model_data)

            # Update cache
            self._cache.running_models = models
            self._cache.running_models_timestamp = now

            logger.debug(f"Fetched {len(models)} running models from Ollama")
            return models

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get running models: {e}")
            return self._cache.running_models  # Return stale cache
        except Exception as e:
            logger.error(f"Error parsing running models: {e}")
            return []

    def get_model_details(self, model_name: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get detailed model info from /api/show.

        Returns dict with: family, parameter_size, quantization, context_length, etc.
        """
        now = time.time()
        cache_key = model_name.lower()

        # Check cache
        if not force_refresh and cache_key in self._cache.model_info:
            info, timestamp = self._cache.model_info[cache_key]
            if (now - timestamp) < self._cache.model_info_ttl:
                return info

        try:
            r = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()

            # Extract details from response
            details = data.get("details", {})
            model_info_raw = data.get("model_info", {})

            # Parse quantization from various places
            quantization = details.get("quantization_level", "")
            if not quantization:
                # Try to extract from model name or format
                quant_match = re.search(r'(Q[0-9]+[_\-]?[A-Z0-9]*|F16|F32|BF16)',
                                       model_name.upper())
                if quant_match:
                    quantization = quant_match.group(1)

            # Get context length from modelfile or model_info
            context_length = 0

            # Try model_info keys
            for key in model_info_raw:
                if "context" in key.lower():
                    try:
                        context_length = int(model_info_raw[key])
                        break
                    except (ValueError, TypeError):
                        pass

            # Try parsing from modelfile
            if context_length == 0:
                modelfile = data.get("modelfile", "")
                ctx_match = re.search(r'num_ctx\s+(\d+)', modelfile)
                if ctx_match:
                    context_length = int(ctx_match.group(1))

            # Get size from /api/tags (not available in /api/show)
            size_bytes = 0
            try:
                tags_resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
                if tags_resp.ok:
                    for m in tags_resp.json().get("models", []):
                        # Match by exact name or base name
                        m_name = m.get("name", "")
                        if m_name == model_name or m_name.split(":")[0] == model_name.split(":")[0]:
                            size_bytes = m.get("size", 0)
                            break
            except Exception:
                pass  # Size is optional, don't fail

            # Build result
            result = {
                "name": model_name,
                "family": details.get("family", ""),
                "families": details.get("families", []),
                "parameter_size": details.get("parameter_size", ""),
                "quantization": quantization,
                "format": details.get("format", ""),
                "context_length": context_length,
                "license": data.get("license", ""),
                "template": data.get("template", "")[:100] if data.get("template") else "",
                "size_bytes": size_bytes,
                "size_gb": size_bytes / (1024**3) if size_bytes else 0,
            }

            # Update cache
            self._cache.model_info[cache_key] = (result, now)

            logger.debug(f"Fetched details for model {model_name}: {result.get('parameter_size')}, {result.get('quantization')}")
            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get model details for {model_name}: {e}")
            # Return stale cache if available
            if cache_key in self._cache.model_info:
                return self._cache.model_info[cache_key][0]
            return None
        except Exception as e:
            logger.error(f"Error parsing model details for {model_name}: {e}")
            return None

    def get_model_info(self, model_name: str, force_refresh: bool = False) -> ModelInfo:
        """
        Get comprehensive model info combining /api/ps and /api/show.

        Args:
            model_name: Model name (e.g., "mistral:latest")
            force_refresh: Force refresh from API

        Returns:
            ModelInfo dataclass with all available info
        """
        info = ModelInfo(name=model_name)

        # Get running model info (VRAM, etc.)
        running = self.get_running_models(force_refresh)
        for m in running:
            if m["name"] == model_name or model_name in m["name"] or m["name"] in model_name:
                info.size_bytes = m["size_bytes"]
                info.size_gb = m["size_gb"]
                info.vram_bytes = m["vram_bytes"]
                info.vram_gb = m["vram_gb"]
                info.expires_at = m["expires_at"]
                break

        # Get detailed model info (quantization, context, etc.)
        details = self.get_model_details(model_name, force_refresh)
        if details:
            info.family = details.get("family", "")
            info.parameter_size = details.get("parameter_size", "")
            info.quantization = details.get("quantization", "")
            info.context_length = details.get("context_length", 0)
            # Use size from details if not from running
            if info.size_bytes == 0:
                info.size_bytes = details.get("size_bytes", 0)
                info.size_gb = details.get("size_gb", 0.0)

        return info

    def list_models(self) -> List[str]:
        """Get list of all available models from /api/tags."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.model_info.clear()
        self._cache.running_models = []
        self._cache.running_models_timestamp = 0.0
        logger.debug("Cleared Ollama client cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        return {
            "model_info_entries": len(self._cache.model_info),
            "running_models_count": len(self._cache.running_models),
            "running_models_age_seconds": now - self._cache.running_models_timestamp if self._cache.running_models_timestamp else None,
            "model_info_ttl": self._cache.model_info_ttl,
            "running_models_ttl": self._cache.running_models_ttl,
        }


# Global client instance (lazy initialization)
_global_client: Optional[OllamaClient] = None


def get_ollama_client(base_url: str = "http://localhost:11434") -> OllamaClient:
    """Get or create global Ollama client instance."""
    global _global_client
    if _global_client is None or _global_client.base_url != base_url:
        _global_client = OllamaClient(base_url=base_url)
    return _global_client


def get_dynamic_context_limit(model_name: str, base_url: str = "http://localhost:11434") -> int:
    """
    Get context limit from Ollama API, falling back to hardcoded limits.

    Args:
        model_name: Model name
        base_url: Ollama server URL

    Returns:
        Context limit in tokens
    """
    from .agent_config import get_model_context_limit, DEFAULT_CONTEXT_LIMIT

    client = get_ollama_client(base_url)
    info = client.get_model_info(model_name)

    if info.context_length > 0:
        logger.debug(f"Using dynamic context limit for {model_name}: {info.context_length}")
        return info.context_length

    # Fall back to hardcoded limits
    limit = get_model_context_limit(model_name)
    logger.debug(f"Using hardcoded context limit for {model_name}: {limit}")
    return limit
