"""
LLM Backend Abstraction for RAGIX
=================================

This module provides multiple LLM backends for RAGIX:

1. OllamaLLM     - 🟢 SOVEREIGN: 100% local, no data leaves your machine
2. ClaudeLLM    - 🔴 CLOUD: Uses Anthropic API, data sent to cloud
3. OpenAILLM    - 🔴 CLOUD: Uses OpenAI API, data sent to cloud

SOVEREIGNTY WARNING:
    When using cloud backends (Claude, OpenAI), your code and prompts
    are sent to external servers. Use OllamaLLM for fully local operation.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-25
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
import os
import logging
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Sovereignty Status
# =============================================================================

class SovereigntyStatus(str, Enum):
    """Indicates whether data leaves your machine."""
    SOVEREIGN = "sovereign"      # 🟢 100% local
    CLOUD = "cloud"              # 🔴 Data sent to external API
    HYBRID = "hybrid"            # 🟡 Some data may leave


# =============================================================================
# Base LLM Interface
# =============================================================================

class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.

    All backends must implement:
        - generate(): Generate response from prompt + history
        - sovereignty: Property indicating data privacy status
    """

    @property
    @abstractmethod
    def sovereignty(self) -> SovereigntyStatus:
        """Return the sovereignty status of this backend."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass

    @abstractmethod
    def generate(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Generate a response.

        Args:
            system_prompt: System instructions for the LLM
            history: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            Generated text response
        """
        pass

    def _log_sovereignty_warning(self):
        """Log a warning if using cloud backend."""
        if self.sovereignty == SovereigntyStatus.CLOUD:
            logger.warning(
                f"⚠️  SOVEREIGNTY WARNING: Using {self.__class__.__name__} "
                f"({self.model_name}) - data will be sent to external API"
            )


# =============================================================================
# 🟢 SOVEREIGN: Ollama Backend (100% Local)
# =============================================================================

class OllamaLLM(BaseLLM):
    """
    🟢 SOVEREIGN: Ollama backend - 100% local execution.

    All processing happens on your machine. No data leaves your network.

    Requires Ollama running locally:
        ollama serve

    Supported models:
        - mistral (recommended)
        - qwen2.5
        - deepseek-coder
        - granite
        - llama3.2
        - codellama

    Example:
        llm = OllamaLLM("mistral")
        response = llm.generate(
            system_prompt="You are a helpful assistant.",
            history=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama backend.

        Args:
            model: Ollama model name (e.g., "mistral", "qwen2.5")
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        logger.info(f"🟢 Initialized OllamaLLM (SOVEREIGN) with model: {model}")

    @property
    def sovereignty(self) -> SovereigntyStatus:
        return SovereigntyStatus.SOVEREIGN

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Generate response using local Ollama.

        🟢 SOVEREIGN: All processing happens locally.
        """
        response, _ = self.generate_with_stats(system_prompt, history)
        return response

    def generate_with_stats(self, system_prompt: str, history: List[Dict[str, str]]) -> Tuple[str, Dict]:
        """
        Generate response with token statistics.

        Returns:
            Tuple of (response_text, stats_dict)
            Stats include: prompt_eval_count, eval_count, total_duration, etc.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        r = requests.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

        # Extract token statistics from Ollama response
        stats = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_duration_ns": data.get("total_duration", 0),
            "load_duration_ns": data.get("load_duration", 0),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
            "eval_duration_ns": data.get("eval_duration", 0),
        }
        stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]

        # Calculate tokens per second if we have duration
        if stats["eval_duration_ns"] > 0:
            stats["tokens_per_second"] = stats["completion_tokens"] / (stats["eval_duration_ns"] / 1e9)
        else:
            stats["tokens_per_second"] = 0

        return data["message"]["content"].strip(), stats

    # Store last request stats for easy access
    _last_stats: Dict = {}

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            pass
        return False


# =============================================================================
# 🔴 CLOUD: Claude Backend (Anthropic API)
# =============================================================================

class ClaudeLLM(BaseLLM):
    """
    🔴 CLOUD: Claude backend - uses Anthropic API.

    ⚠️  SOVEREIGNTY WARNING ⚠️
    This backend sends your prompts and code to Anthropic's servers.
    Use OllamaLLM for fully local, sovereign operation.

    Requires:
        pip install anthropic
        export ANTHROPIC_API_KEY="sk-ant-..."

    Supported models:
        - claude-sonnet-4-20250514 (recommended)
        - claude-opus-4-20250514
        - claude-haiku-3-20240307

    Example:
        llm = ClaudeLLM("claude-sonnet-4-20250514")
        response = llm.generate(
            system_prompt="You are a helpful assistant.",
            history=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        """
        Initialize Claude backend.

        Args:
            model: Claude model name
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self._api_key:
            raise ValueError(
                "🔴 Claude API key required. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

        # Import anthropic here to make it optional
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "🔴 anthropic package required for Claude backend. "
                "Install with: pip install anthropic"
            )

        logger.warning(
            f"🔴 Initialized ClaudeLLM (CLOUD) with model: {model}\n"
            f"   ⚠️  Data will be sent to Anthropic API - sovereignty is broken"
        )

    @property
    def sovereignty(self) -> SovereigntyStatus:
        return SovereigntyStatus.CLOUD

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Generate response using Claude API.

        🔴 CLOUD: Data is sent to Anthropic servers.
        """
        self._log_sovereignty_warning()

        # Convert history to Claude format
        messages = []
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        )

        return response.content[0].text


# =============================================================================
# 🔴 CLOUD: OpenAI Backend (ChatGPT API)
# =============================================================================

class OpenAILLM(BaseLLM):
    """
    🔴 CLOUD: OpenAI backend - uses OpenAI API (ChatGPT).

    ⚠️  SOVEREIGNTY WARNING ⚠️
    This backend sends your prompts and code to OpenAI's servers.
    Use OllamaLLM for fully local, sovereign operation.

    Requires:
        pip install openai
        export OPENAI_API_KEY="sk-..."

    Supported models:
        - gpt-4o (recommended)
        - gpt-4o-mini
        - gpt-4-turbo
        - gpt-3.5-turbo

    Example:
        llm = OpenAILLM("gpt-4o")
        response = llm.generate(
            system_prompt="You are a helpful assistant.",
            history=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Initialize OpenAI backend.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError(
                "🔴 OpenAI API key required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        # Import openai here to make it optional
        try:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "🔴 openai package required for OpenAI backend. "
                "Install with: pip install openai"
            )

        logger.warning(
            f"🔴 Initialized OpenAILLM (CLOUD) with model: {model}\n"
            f"   ⚠️  Data will be sent to OpenAI API - sovereignty is broken"
        )

    @property
    def sovereignty(self) -> SovereigntyStatus:
        return SovereigntyStatus.CLOUD

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Generate response using OpenAI API.

        🔴 CLOUD: Data is sent to OpenAI servers.
        """
        self._log_sovereignty_warning()

        # Build messages with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content


# =============================================================================
# Factory Function
# =============================================================================

def create_llm_backend(
    backend: str = "ollama",
    model: Optional[str] = None,
    **kwargs
) -> BaseLLM:
    """
    Factory function to create LLM backend.

    Args:
        backend: Backend type - "ollama" (🟢), "claude" (🔴), or "openai" (🔴)
        model: Model name (optional, uses defaults)
        **kwargs: Additional backend-specific arguments

    Returns:
        Configured LLM backend instance

    Example:
        # 🟢 Sovereign (local)
        llm = create_llm_backend("ollama", model="mistral")

        # 🔴 Cloud (Anthropic)
        llm = create_llm_backend("claude", model="claude-sonnet-4-20250514")

        # 🔴 Cloud (OpenAI)
        llm = create_llm_backend("openai", model="gpt-4o")
    """
    backend = backend.lower()

    if backend == "ollama":
        model = model or os.environ.get("UNIX_RAG_MODEL", "mistral")
        return OllamaLLM(model=model, **kwargs)

    elif backend == "claude":
        model = model or os.environ.get("RAGIX_CLAUDE_MODEL", "claude-sonnet-4-20250514")
        return ClaudeLLM(model=model, **kwargs)

    elif backend in ("openai", "chatgpt", "gpt"):
        model = model or os.environ.get("RAGIX_OPENAI_MODEL", "gpt-4o")
        return OpenAILLM(model=model, **kwargs)

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Supported: ollama (🟢 sovereign), claude (🔴 cloud), openai (🔴 cloud)"
        )


def get_backend_from_env() -> BaseLLM:
    """
    Create LLM backend from environment variables.

    Environment variables:
        RAGIX_LLM_BACKEND: "ollama" | "claude" | "openai" (default: ollama)

        For Ollama:
            UNIX_RAG_MODEL: Model name (default: mistral)

        For Claude:
            RAGIX_CLAUDE_MODEL: Model name (default: claude-sonnet-4-20250514)
            ANTHROPIC_API_KEY: API key (required)

        For OpenAI:
            RAGIX_OPENAI_MODEL: Model name (default: gpt-4o)
            OPENAI_API_KEY: API key (required)

    Returns:
        Configured LLM backend
    """
    backend = os.environ.get("RAGIX_LLM_BACKEND", "ollama")
    return create_llm_backend(backend)
