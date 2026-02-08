"""
Dual LLM backend abstraction: Ollama (sovereign) + Claude API (non-sovereign).

OllamaBackend wraps the existing llm_call() from ragix_kernels/llm_wrapper.py.
ClaudeAPIBackend uses the anthropic SDK (optional dependency).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)


class SovereigntyError(Exception):
    """Raised when a non-sovereign backend is used with strict_sovereign=True."""
    pass


class LLMBackend(ABC):
    """Abstract base for LLM backends."""

    def __init__(self):
        self._last_envelope: Dict[str, Any] = {}

    @abstractmethod
    def call(
        self,
        prompt: str,
        temperature: float = 0.1,
        num_predict: int = 2048,
        timeout: int = 120,
        streaming: bool = False,
    ) -> str:
        """Send a prompt and return the response text.

        Args:
            streaming: If True, use streaming with marker-based
                short-circuit (END_EDIT_OPS) and instrumentation.
        """
        ...

    @property
    def last_envelope(self) -> Dict[str, Any]:
        """Last response envelope (Ollama fields, http_bytes, etc.)."""
        return self._last_envelope

    @abstractmethod
    def sovereignty_info(self) -> Dict[str, Any]:
        """Return sovereignty metadata for audit trail."""
        ...

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Whether this backend runs entirely locally."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier."""
        ...


class OllamaBackend(LLMBackend):
    """
    Local LLM backend via Ollama.

    Wraps ragix_kernels.llm_wrapper.llm_call() with cache and activity
    integration.
    """

    def __init__(
        self,
        model: str = "mistral:instruct",
        endpoint: str = "http://127.0.0.1:11434",
        cache=None,
        cache_mode=None,
    ):
        super().__init__()
        self._model = model
        self._endpoint = endpoint
        self._cache = cache
        self._cache_mode = cache_mode

    def call(
        self,
        prompt: str,
        temperature: float = 0.1,
        num_predict: int = 2048,
        timeout: int = 120,
        streaming: bool = False,
    ) -> str:
        try:
            from ragix_kernels.cache import CacheMode

            mode = self._cache_mode or CacheMode.WRITE_THROUGH
            if self._cache is None:
                mode = CacheMode.OFF

            if mode == CacheMode.OFF:
                if streaming:
                    # Streaming with marker-based short-circuit
                    from ragix_kernels.llm_wrapper import (
                        _call_ollama_streaming,
                    )
                    response_text, envelope = _call_ollama_streaming(
                        self._model, prompt, temperature,
                        self._endpoint, timeout, num_predict,
                    )
                    self._last_envelope = envelope
                    return response_text
                else:
                    # Non-streaming direct call with envelope capture
                    from ragix_kernels.llm_wrapper import (
                        _call_ollama_with_envelope,
                    )
                    _, envelope = _call_ollama_with_envelope(
                        self._model, prompt, temperature,
                        self._endpoint, timeout, num_predict,
                    )
                    self._last_envelope = envelope
                    return envelope.get("response", "")

            # With caching: use llm_call (cache boundary), then do a
            # lightweight envelope probe if response is empty
            from ragix_kernels.llm_wrapper import llm_call
            result = llm_call(
                model=self._model,
                prompt=prompt,
                temperature=temperature,
                cache=self._cache,
                mode=mode,
                endpoint=self._endpoint,
                timeout=timeout,
                num_predict=num_predict,
            )
            # Envelope not available through cache path; store minimal info
            self._last_envelope = {
                "response": result,
                "_http_bytes": len(result.encode("utf-8")) if result else 0,
                "_via_cache": True,
            }
            return result
        except ImportError:
            # Fallback: direct HTTP call
            return self._direct_call(prompt, temperature, num_predict, timeout)

    def _direct_call(
        self, prompt: str, temperature: float, num_predict: int, timeout: int
    ) -> str:
        """Direct HTTP call to Ollama API without wrapper."""
        import json
        import urllib.request

        url = f"{self._endpoint}/api/generate"
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
            result = json.loads(raw_bytes.decode("utf-8"))
            result["_http_bytes"] = len(raw_bytes)
            self._last_envelope = result
            return result.get("response", "")

    def sovereignty_info(self) -> Dict[str, Any]:
        import socket
        return {
            "local_only": True,
            "endpoint": self._endpoint,
            "model": self._model,
            "hostname": socket.gethostname(),
        }

    @property
    def is_local(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._model


class ClaudeAPIBackend(LLMBackend):
    """
    Claude API backend (non-sovereign).

    Requires: pip install anthropic
    Audit trail marks local_only: false.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", api_key: str = ""):
        super().__init__()
        self._model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install ragix[claude] or pip install anthropic"
                )
            import os
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError("No API key: set ANTHROPIC_API_KEY or pass api_key")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def call(
        self,
        prompt: str,
        temperature: float = 0.1,
        num_predict: int = 2048,
        timeout: int = 120,
        streaming: bool = False,
    ) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=num_predict,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return message.content[0].text

    def sovereignty_info(self) -> Dict[str, Any]:
        return {
            "local_only": False,
            "endpoint": "https://api.anthropic.com",
            "model": self._model,
        }

    @property
    def is_local(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return self._model


def get_backend(
    backend: str = "ollama",
    model: Optional[str] = None,
    endpoint: str = "http://127.0.0.1:11434",
    strict_sovereign: bool = True,
    api_key: str = "",
    cache=None,
    cache_mode=None,
) -> LLMBackend:
    """
    Factory for LLM backends.

    Args:
        backend: "ollama" or "claude"
        model: Model name (defaults per backend)
        endpoint: Ollama endpoint URL
        strict_sovereign: If True, refuse non-local backends
        api_key: Anthropic API key (for claude backend)
        cache: LLMCache instance (for ollama)
        cache_mode: CacheMode (for ollama)

    Returns:
        LLMBackend instance

    Raises:
        SovereigntyError: If claude is requested with strict_sovereign=True
    """
    if backend == "ollama":
        return OllamaBackend(
            model=model or "mistral:instruct",
            endpoint=endpoint,
            cache=cache,
            cache_mode=cache_mode,
        )
    elif backend == "claude":
        if strict_sovereign:
            raise SovereigntyError(
                "Claude API backend rejected: --strict-sovereign is active. "
                "Use --backend ollama or disable strict_sovereign."
            )
        return ClaudeAPIBackend(
            model=model or "claude-sonnet-4-5-20250929",
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama' or 'claude'.")
