"""
LLM Response Cache — Persistent caching for LLM responses with sovereignty tracking.

Author: Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
Date: 2026-01-18

Features:
- Hash-based cache key derivation (model + prompt + temperature)
- Model version/digest tracking
- Sovereignty metadata (hostname, user, endpoint, timestamps)
- Statistics tracking (hits, misses, savings)
- TTL support (optional)
- Cache modes: write_through, read_only, read_prefer, off
"""

import hashlib
import json
import os
import socket
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import logging


class CacheMode(Enum):
    """LLM cache operating modes."""
    WRITE_THROUGH = "write_through"  # Default: call LLM, cache result
    READ_ONLY = "read_only"          # Replay only, fail on cache miss
    READ_PREFER = "read_prefer"      # Use cache if exists, else call LLM
    OFF = "off"                      # No caching


class CacheMissError(Exception):
    """Raised when cache lookup fails in read_only mode."""
    pass

logger = logging.getLogger(__name__)


@dataclass
class SovereigntyInfo:
    """Sovereignty attestation for a cached response."""
    hostname: str
    user: str
    endpoint: str
    local: bool
    timestamp: str

    @classmethod
    def capture(cls, endpoint: str = "http://127.0.0.1:11434") -> "SovereigntyInfo":
        """Capture current sovereignty information."""
        is_local = "127.0.0.1" in endpoint or "localhost" in endpoint
        return cls(
            hostname=socket.gethostname(),
            user=os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            endpoint=endpoint,
            local=is_local,
            timestamp=datetime.now(timezone.utc).isoformat()
        )


@dataclass
class CacheEntry:
    """A cached LLM response with metadata."""
    cache_key: str
    model: str
    model_digest: str
    prompt_hash: str
    temperature: float
    created_at: str
    response: str
    response_hash: str
    sovereignty: Dict[str, Any]
    usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        return cls(**data)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    tokens_saved: int = 0
    time_saved_s: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "hit_rate": self.hit_rate
        }


class LLMCache:
    """
    Persistent cache for LLM responses.

    Features:
    - Hash-based key derivation
    - Model version awareness
    - Sovereignty metadata tracking
    - Statistics collection

    Usage:
        cache = LLMCache(workspace / ".KOAS/cache")

        # Check cache
        cached = cache.get("granite3.1-moe:3b", prompt)
        if cached:
            return cached

        # Generate and cache
        response = call_llm(model, prompt)
        cache.put("granite3.1-moe:3b", prompt, response, model_digest="b43d80d7fca7")
    """

    def __init__(self, cache_dir: Path, endpoint: str = "http://127.0.0.1:11434"):
        self.cache_dir = Path(cache_dir)
        self.endpoint = endpoint
        self.responses_dir = self.cache_dir / "llm_responses"
        self.responses_dir.mkdir(parents=True, exist_ok=True)

        self.stats = self._load_stats()
        self.index = self._load_index()

    def _compute_key(self, model: str, prompt: str, temperature: float = 0.3) -> str:
        """Compute deterministic cache key."""
        # Normalize model name (remove host prefix if any)
        model_name = model.split("/")[-1]
        key_data = f"{model_name}:{temperature:.2f}:{prompt}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_model_dir(self, model: str) -> Path:
        """Get directory for a specific model."""
        # Sanitize model name for filesystem
        safe_name = model.replace(":", "_").replace("/", "_")
        model_dir = self.responses_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _get_cache_path(self, model: str, key: str) -> Path:
        """Get path for a cache entry."""
        return self._get_model_dir(model) / f"{key}.json"

    def _load_stats(self) -> CacheStats:
        """Load statistics from disk."""
        stats_file = self.cache_dir / "cache_stats.json"
        if stats_file.exists():
            try:
                data = json.loads(stats_file.read_text())
                return CacheStats(**{k: v for k, v in data.items() if k != "hit_rate"})
            except Exception as e:
                logger.warning(f"Failed to load cache stats: {e}")
        return CacheStats()

    def _save_stats(self):
        """Save statistics to disk."""
        stats_file = self.cache_dir / "cache_stats.json"
        stats_file.write_text(json.dumps(self.stats.to_dict(), indent=2))

    def _load_index(self) -> Dict[str, str]:
        """Load cache index for fast lookup."""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                return json.loads(index_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}

    def _save_index(self):
        """Save cache index."""
        index_file = self.cache_dir / "cache_index.json"
        index_file.write_text(json.dumps(self.index, indent=2))

    def get(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.3,
        model_digest: Optional[str] = None
    ) -> Optional[str]:
        """
        Get cached response if available.

        Args:
            model: Model name (e.g., "granite3.1-moe:3b")
            prompt: The prompt text
            temperature: Temperature used (affects output)
            model_digest: Optional model digest for version matching

        Returns:
            Cached response string, or None if not cached
        """
        key = self._compute_key(model, prompt, temperature)
        cache_file = self._get_cache_path(model, key)

        self.stats.total_requests += 1

        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                entry = CacheEntry.from_dict(data)

                # Optionally check model digest
                if model_digest and entry.model_digest != model_digest:
                    logger.debug(f"Cache miss: model digest mismatch ({entry.model_digest} != {model_digest})")
                    self.stats.misses += 1
                    self._save_stats()
                    return None

                self.stats.hits += 1
                self.stats.tokens_saved += entry.usage.get("completion_tokens", 0)
                self._save_stats()

                logger.debug(f"Cache hit for key {key}")
                return entry.response

            except Exception as e:
                logger.warning(f"Failed to read cache entry {key}: {e}")

        self.stats.misses += 1
        self._save_stats()
        return None

    def put(
        self,
        model: str,
        prompt: str,
        response: str,
        temperature: float = 0.3,
        model_digest: str = "",
        usage: Optional[Dict[str, int]] = None
    ):
        """
        Store a response in cache.

        Args:
            model: Model name
            prompt: The prompt text
            response: The LLM response
            temperature: Temperature used
            model_digest: Model version/digest for tracking
            usage: Token usage statistics (optional)
        """
        key = self._compute_key(model, prompt, temperature)
        cache_file = self._get_cache_path(model, key)

        sovereignty = SovereigntyInfo.capture(self.endpoint)

        entry = CacheEntry(
            cache_key=key,
            model=model,
            model_digest=model_digest,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            temperature=temperature,
            created_at=datetime.now(timezone.utc).isoformat(),
            response=response,
            response_hash=hashlib.sha256(response.encode()).hexdigest(),
            sovereignty=asdict(sovereignty),
            usage=usage or {}
        )

        cache_file.write_text(json.dumps(entry.to_dict(), indent=2, ensure_ascii=False))

        # Update index
        self.index[key] = str(cache_file)
        self._save_index()

        logger.debug(f"Cached response for key {key}")

    def clear(self, model: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            model: If specified, only clear entries for this model
        """
        if model:
            model_dir = self._get_model_dir(model)
            if model_dir.exists():
                for f in model_dir.glob("*.json"):
                    f.unlink()
                logger.info(f"Cleared cache for model {model}")
        else:
            for model_dir in self.responses_dir.iterdir():
                if model_dir.is_dir():
                    for f in model_dir.glob("*.json"):
                        f.unlink()
            self.index = {}
            self._save_index()
            logger.info("Cleared all cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.stats.to_dict()

    def get_sovereignty_summary(self) -> Dict[str, Any]:
        """Get sovereignty summary from cached entries."""
        models_used = set()
        endpoints = set()
        hosts = set()

        for model_dir in self.responses_dir.iterdir():
            if model_dir.is_dir():
                for cache_file in model_dir.glob("*.json"):
                    try:
                        data = json.loads(cache_file.read_text())
                        models_used.add(data.get("model", "unknown"))
                        sov = data.get("sovereignty", {})
                        endpoints.add(sov.get("endpoint", "unknown"))
                        hosts.add(sov.get("hostname", "unknown"))
                    except:
                        pass

        return {
            "models_used": list(models_used),
            "endpoints": list(endpoints),
            "hosts": list(hosts),
            "all_local": all("127.0.0.1" in e or "localhost" in e for e in endpoints)
        }


class KernelCache:
    """
    Persistent cache for kernel outputs.

    Caches kernel computation results based on input_hash, allowing
    fast replay when document list is stable.

    Usage:
        cache = KernelCache(workspace / ".KOAS/cache")

        # Check cache
        cached = cache.get("doc_extract", input_hash)
        if cached:
            return cached

        # Compute and cache
        output = kernel.compute(input)
        cache.put("doc_extract", input_hash, output)
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.outputs_dir = self.cache_dir / "kernel_outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.stats = self._load_stats()

    def _load_stats(self) -> Dict[str, Any]:
        """Load kernel cache statistics."""
        stats_file = self.cache_dir / "kernel_cache_stats.json"
        if stats_file.exists():
            try:
                return json.loads(stats_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load kernel cache stats: {e}")
        return {"hits": 0, "misses": 0, "total_requests": 0}

    def _save_stats(self):
        """Save kernel cache statistics."""
        stats_file = self.cache_dir / "kernel_cache_stats.json"
        hit_rate = self.stats["hits"] / self.stats["total_requests"] if self.stats["total_requests"] > 0 else 0
        self.stats["hit_rate"] = hit_rate
        stats_file.write_text(json.dumps(self.stats, indent=2))

    def _get_cache_path(self, kernel_name: str, input_hash: str) -> Path:
        """Get cache file path for a kernel output."""
        return self.outputs_dir / f"{kernel_name}_{input_hash}.json"

    def get(self, kernel_name: str, input_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached kernel output if available.

        Args:
            kernel_name: Name of the kernel (e.g., "doc_extract")
            input_hash: Hash of kernel inputs

        Returns:
            Cached output dict, or None if not cached
        """
        cache_file = self._get_cache_path(kernel_name, input_hash)

        self.stats["total_requests"] += 1

        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                self.stats["hits"] += 1
                self._save_stats()
                logger.debug(f"[KernelCache] HIT for {kernel_name} (hash={input_hash})")
                return data.get("output")
            except Exception as e:
                logger.warning(f"Failed to read kernel cache {kernel_name}: {e}")

        self.stats["misses"] += 1
        self._save_stats()
        logger.debug(f"[KernelCache] MISS for {kernel_name} (hash={input_hash})")
        return None

    def put(self, kernel_name: str, input_hash: str, output: Dict[str, Any], metadata: Optional[Dict] = None):
        """
        Store kernel output in cache.

        Args:
            kernel_name: Name of the kernel
            input_hash: Hash of kernel inputs
            output: Kernel output dict
            metadata: Optional metadata (timing, etc.)
        """
        cache_file = self._get_cache_path(kernel_name, input_hash)

        entry = {
            "kernel": kernel_name,
            "input_hash": input_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "output": output,
            "metadata": metadata or {}
        }

        cache_file.write_text(json.dumps(entry, indent=2, ensure_ascii=False))
        logger.debug(f"[KernelCache] Stored {kernel_name} (hash={input_hash})")

    def invalidate(self, kernel_name: Optional[str] = None):
        """
        Invalidate cache entries.

        Args:
            kernel_name: If specified, only invalidate entries for this kernel
        """
        if kernel_name:
            for f in self.outputs_dir.glob(f"{kernel_name}_*.json"):
                f.unlink()
            logger.info(f"[KernelCache] Invalidated cache for {kernel_name}")
        else:
            for f in self.outputs_dir.glob("*.json"):
                f.unlink()
            logger.info("[KernelCache] Invalidated all kernel cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get kernel cache statistics."""
        return self.stats.copy()

    def list_cached_kernels(self) -> Dict[str, int]:
        """List cached kernels and their entry counts."""
        kernel_counts: Dict[str, int] = {}
        for f in self.outputs_dir.glob("*.json"):
            kernel_name = f.stem.rsplit("_", 1)[0]
            kernel_counts[kernel_name] = kernel_counts.get(kernel_name, 0) + 1
        return kernel_counts


def get_model_digest(model: str, endpoint: str = "http://127.0.0.1:11434") -> str:
    """
    Get model digest from Ollama.

    Args:
        model: Model name
        endpoint: Ollama endpoint

    Returns:
        Model digest (short form)
    """
    import httpx

    try:
        response = httpx.get(f"{endpoint}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            for m in data.get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    digest = m.get("digest", "")
                    # Return short form (first 12 chars)
                    return digest[:12] if digest else ""
    except Exception as e:
        logger.warning(f"Failed to get model digest: {e}")

    return ""
