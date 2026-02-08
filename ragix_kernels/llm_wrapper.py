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
    _, envelope = _call_ollama_with_envelope(
        model, prompt, temperature, endpoint, timeout, num_predict,
    )
    return envelope.get("response", "")


def _call_ollama_with_envelope(
    model: str,
    prompt: str,
    temperature: float,
    endpoint: str,
    timeout: int,
    num_predict: Optional[int] = None,
) -> tuple:
    """
    Direct call to Ollama API, returning full response envelope.

    Returns:
        (http_bytes_len, envelope_dict) where envelope contains all Ollama
        fields: response, done, done_reason, eval_count, prompt_eval_count,
        total_duration, load_duration, etc.
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
    raw_bytes = response.content
    try:
        envelope = response.json()
    except Exception:
        envelope = {"response": "", "_json_decode_error": True}
    envelope["_http_bytes"] = len(raw_bytes)
    return len(raw_bytes), envelope


def _call_ollama_streaming(
    model: str,
    prompt: str,
    temperature: float,
    endpoint: str,
    timeout: int,
    num_predict: Optional[int] = None,
) -> tuple:
    """
    Streaming call to Ollama with marker-based short-circuit.

    No token-count early-abort (calibration proved t_first_visible ranges
    813–1882 for gpt-oss-safeguard:120b, making fixed thresholds unreliable).

    Instead:
      - Short-circuit on END_EDIT_OPS: stop streaming immediately once the
        end marker is detected (saves time on successful chunks).
      - Full instrumentation: t_first_visible, visible_chars, markers.
      - Degenerate runs (vis=0, no markers at full budget) are classified
        post-hoc by the kernel, not aborted mid-stream.

    Returns:
        (response_text, envelope_dict) where envelope includes:
        - Standard Ollama fields (eval_count, done_reason, etc.) from final chunk
        - _http_bytes: total bytes received
        - _tokens_seen: tokens observed before completion/short-circuit
        - _visible_chars: count of non-whitespace characters in response
        - _t_first_visible: token index of first visible character (-1 if none)
        - _begin_seen: whether BEGIN_EDIT_OPS marker was found
        - _end_seen: whether END_EDIT_OPS marker was found
        - _short_circuit: "end_marker" if stopped early on END_EDIT_OPS, "" otherwise
    """
    import json as _json

    options = {"temperature": temperature}
    if num_predict:
        options["num_predict"] = num_predict

    response_fragments = []
    accumulated = ""
    tokens_seen = 0
    visible_chars = 0
    t_first_visible = -1
    total_bytes = 0
    envelope = {}
    short_circuit = ""
    begin_seen = False
    end_seen = False

    try:
        with httpx.stream(
            "POST",
            f"{endpoint}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": options,
            },
            timeout=timeout,
        ) as stream:
            for line in stream.iter_lines():
                if not line.strip():
                    continue
                total_bytes += len(line.encode("utf-8")) + 1  # +1 for newline

                try:
                    chunk = _json.loads(line)
                except _json.JSONDecodeError:
                    continue

                if chunk.get("done"):
                    # Final chunk: contains full stats
                    envelope = chunk
                    break

                # Accumulate response fragment
                fragment = chunk.get("response", "")
                response_fragments.append(fragment)
                accumulated += fragment
                tokens_seen += 1

                # Count visible (non-whitespace) characters
                frag_visible = sum(1 for c in fragment if c.strip())
                if frag_visible > 0:
                    if t_first_visible < 0:
                        t_first_visible = tokens_seen
                    visible_chars += frag_visible

                # Marker detection (only when visible chars exist)
                if visible_chars > 0:
                    if not begin_seen and "BEGIN_EDIT_OPS" in accumulated:
                        begin_seen = True
                    if begin_seen and not end_seen and "END_EDIT_OPS" in accumulated:
                        end_seen = True
                        # Short-circuit: we have the complete payload
                        short_circuit = "end_marker"
                        logger.debug(
                            f"[llm_wrapper] END_EDIT_OPS at token {tokens_seen} "
                            f"— short-circuit"
                        )
                        break

    except httpx.ReadTimeout:
        envelope = {"response": "", "done": True, "done_reason": "timeout"}
    except Exception as e:
        envelope = {"response": "", "done": True, "done_reason": f"error: {e}"}

    response_text = "".join(response_fragments)

    # Merge response text and diagnostics into envelope
    envelope["response"] = response_text
    envelope["_http_bytes"] = total_bytes
    envelope["_tokens_seen"] = tokens_seen
    envelope["_visible_chars"] = visible_chars
    envelope["_t_first_visible"] = t_first_visible
    envelope["_begin_seen"] = begin_seen
    envelope["_end_seen"] = end_seen
    envelope["_short_circuit"] = short_circuit

    return response_text, envelope


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
