"""
LLM Call Wrapper — Single boundary for all LLM calls with caching support.

Author: Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
Date: 2026-01-22

This module provides a unified interface for LLM calls with:
- Cache mode support (write_through, read_only, read_prefer, off)
- Sovereignty tracking via LLMCache
- Consistent error handling

Usage:
    from ragix_kernels.llm_wrapper import llm_call
    from ragix_kernels.cache import LLMCache, CacheMode

    cache = LLMCache(workspace / ".KOAS/cache")
    response = llm_call(
        model="granite3.1-moe:3b",
        prompt="Summarize this document...",
        temperature=0.3,
        cache=cache,
        mode=CacheMode.WRITE_THROUGH
    )
"""

import hashlib
import logging
import time
from typing import Optional

import httpx

from ragix_kernels.cache import LLMCache, CacheMode, CacheMissError
from ragix_kernels.activity import get_activity_writer

logger = logging.getLogger(__name__)


def _compute_hash(text: str) -> str:
    """Compute SHA256 hash prefix for traceability (first 16 chars)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _call_ollama(
    model: str,
    prompt: str,
    temperature: float,
    endpoint: str,
    timeout: int,
    num_predict: Optional[int] = None,
) -> str:
    """
    Direct call to Ollama API.

    Args:
        model: Model name (e.g., "granite3.1-moe:3b")
        prompt: The prompt text
        temperature: Sampling temperature
        endpoint: Ollama endpoint URL
        timeout: Request timeout in seconds
        num_predict: Max tokens to generate (optional)

    Returns:
        LLM response text

    Raises:
        httpx.HTTPError: On HTTP errors
        Exception: On other failures
    """
    options = {"temperature": temperature}
    if num_predict:
        options["num_predict"] = num_predict

    response = httpx.post(
        f"{endpoint}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def llm_call(
    model: str,
    prompt: str,
    temperature: float,
    cache: LLMCache,
    mode: CacheMode,
    endpoint: str = "http://127.0.0.1:11434",
    timeout: int = 120,
    num_predict: Optional[int] = None,
    model_digest: str = "",
) -> str:
    """
    Single LLM call boundary with caching.

    This function provides a unified interface for all LLM calls in the
    KOAS kernels, with full cache mode support.

    Args:
        model: Model name (e.g., "granite3.1-moe:3b")
        prompt: The prompt text
        temperature: Sampling temperature (affects cache key)
        cache: LLMCache instance
        mode: Cache mode (write_through, read_only, read_prefer, off)
        endpoint: Ollama endpoint URL
        timeout: Request timeout in seconds
        num_predict: Max tokens to generate (optional)
        model_digest: Model digest for version tracking (optional)

    Returns:
        LLM response text

    Raises:
        CacheMissError: In READ_ONLY mode when cache lookup fails
        httpx.HTTPError: On HTTP errors
        Exception: On other failures
    """
    activity_writer = get_activity_writer()
    prompt_hash = _compute_hash(prompt)
    start_time = time.time()

    # OFF mode: bypass cache entirely
    if mode == CacheMode.OFF:
        logger.debug(f"[llm_call] Cache OFF, calling LLM directly")
        response = _call_ollama(model, prompt, temperature, endpoint, timeout, num_predict)

        # Activity logging: LLM call without cache
        if activity_writer:
            duration_ms = int((time.time() - start_time) * 1000)
            activity_writer.emit_llm_call(
                model=model,
                cache_hit=False,
                prompt_hash=prompt_hash,
                response_hash=_compute_hash(response),
                duration_ms=duration_ms,
            )
        return response

    # Check cache first for all other modes
    cached = cache.get(model, prompt, temperature, model_digest or None)
    if cached:
        logger.debug(f"[llm_call] Cache HIT for model={model}")

        # Activity logging: cache hit
        if activity_writer:
            activity_writer.emit_llm_call(
                model=model,
                cache_hit=True,
                prompt_hash=prompt_hash,
                response_hash=_compute_hash(cached),
            )
        return cached

    # READ_ONLY mode: fail on cache miss
    if mode == CacheMode.READ_ONLY:
        prompt_preview = prompt[:80].replace("\n", " ") + "..." if len(prompt) > 80 else prompt
        raise CacheMissError(
            f"Cache miss in read_only mode: model={model}, "
            f"prompt_len={len(prompt)}, preview='{prompt_preview}'"
        )

    # WRITE_THROUGH or READ_PREFER: call LLM and cache result
    logger.debug(f"[llm_call] Cache MISS, calling LLM model={model}")
    response = _call_ollama(model, prompt, temperature, endpoint, timeout, num_predict)

    # Cache the response
    cache.put(model, prompt, response, temperature, model_digest)
    logger.debug(f"[llm_call] Cached response for model={model}")

    # Activity logging: LLM call with cache write
    if activity_writer:
        duration_ms = int((time.time() - start_time) * 1000)
        activity_writer.emit_llm_call(
            model=model,
            cache_hit=False,
            prompt_hash=prompt_hash,
            response_hash=_compute_hash(response),
            duration_ms=duration_ms,
        )

    return response


def llm_call_with_ollama_lib(
    model: str,
    prompt: str,
    temperature: float,
    cache: LLMCache,
    mode: CacheMode,
    num_predict: int = 300,
    model_digest: str = "",
) -> str:
    """
    LLM call using the ollama Python library (for kernels that use it).

    This variant uses the `ollama` package instead of direct HTTP calls.
    The caching logic remains identical.

    Args:
        model: Model name
        prompt: The prompt text
        temperature: Sampling temperature
        cache: LLMCache instance
        mode: Cache mode
        num_predict: Max tokens to generate
        model_digest: Model digest for version tracking

    Returns:
        LLM response text

    Raises:
        CacheMissError: In READ_ONLY mode when cache lookup fails
    """
    import ollama as ollama_lib

    activity_writer = get_activity_writer()
    prompt_hash = _compute_hash(prompt)
    start_time = time.time()

    # OFF mode: bypass cache entirely
    if mode == CacheMode.OFF:
        response = ollama_lib.generate(
            model=model,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": num_predict},
        )
        response_text = response.get("response", "")

        # Activity logging: LLM call without cache
        if activity_writer:
            duration_ms = int((time.time() - start_time) * 1000)
            activity_writer.emit_llm_call(
                model=model,
                cache_hit=False,
                prompt_hash=prompt_hash,
                response_hash=_compute_hash(response_text),
                duration_ms=duration_ms,
            )
        return response_text

    # Check cache first
    cached = cache.get(model, prompt, temperature, model_digest or None)
    if cached:
        logger.debug(f"[llm_call_ollama] Cache HIT for model={model}")

        # Activity logging: cache hit
        if activity_writer:
            activity_writer.emit_llm_call(
                model=model,
                cache_hit=True,
                prompt_hash=prompt_hash,
                response_hash=_compute_hash(cached),
            )
        return cached

    # READ_ONLY mode: fail on cache miss
    if mode == CacheMode.READ_ONLY:
        prompt_preview = prompt[:80].replace("\n", " ") + "..." if len(prompt) > 80 else prompt
        raise CacheMissError(
            f"Cache miss in read_only mode: model={model}, "
            f"prompt_len={len(prompt)}, preview='{prompt_preview}'"
        )

    # WRITE_THROUGH or READ_PREFER: call LLM and cache result
    logger.debug(f"[llm_call_ollama] Cache MISS, calling LLM model={model}")
    response = ollama_lib.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": num_predict},
    )
    response_text = response.get("response", "")

    # Cache the response
    cache.put(model, prompt, response_text, temperature, model_digest)

    # Activity logging: LLM call with cache write
    if activity_writer:
        duration_ms = int((time.time() - start_time) * 1000)
        activity_writer.emit_llm_call(
            model=model,
            cache_hit=False,
            prompt_hash=prompt_hash,
            response_hash=_compute_hash(response_text),
            duration_ms=duration_ms,
        )

    return response_text
