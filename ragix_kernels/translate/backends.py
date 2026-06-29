"""
LLM backend seam for the KOAS-Translate kernels.

A ``Backend`` is a plain ``prompt -> completion`` callable. Keeping the seam this
narrow lets every translate kernel (draft / qa / harmonize) be unit-tested with a
deterministic stub, while the default backend wraps
``ragix_core.llm_backends.OllamaLLM`` (replacing the pipeline's bespoke
ollama_client). Determinism comes from the pinned Ollama modelfile (greedy
decoding); ``parse_json_lenient`` recovers JSON from models that wrap or pad it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-06-27
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_LANG_PAIR = "en-fr"

#: A translation/QA backend: full prompt → completion text.
Backend = Callable[[str], str]

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def build_ollama_backend(model: str, base_url: str = DEFAULT_OLLAMA_URL) -> Backend:
    """Default backend: a single-prompt wrapper over ``llm_backends.OllamaLLM``."""
    from ragix_core.llm_backends import OllamaLLM
    llm = OllamaLLM(model=model, base_url=base_url)

    def _generate(prompt: str) -> str:
        return llm.generate(system_prompt=prompt, history=[])

    return _generate


def resolve_backend(injected: Optional[Backend], cfg: dict, default_model: str) -> Backend:
    """Return *injected* if provided (tests / programmatic), else build the default
    from ``cfg['model']`` (falling back to *default_model*) and ``cfg['ollama_url']``."""
    if injected is not None:
        return injected
    return build_ollama_backend(
        cfg.get("model", default_model),
        cfg.get("ollama_url", DEFAULT_OLLAMA_URL),
    )


def load_prompt(cfg: dict, default_path: Path) -> str:
    """Resolve a stage prompt template, honouring overrides and language pairs.

    Resolution order:
      1. ``cfg['prompt_template']`` — inline template string;
      2. ``cfg['prompt_path']`` — explicit file;
      3. ``<prompts_dir>/<stem>.<lang_pair>.txt`` — per-pair override
         (``lang_pair`` from ``cfg``, default ``en-fr``);
      4. *default_path* — the bundled ``en-fr`` method prompt.

    *default_path* is the en-fr default for the stage (e.g.
    ``prompts/translate.txt``); its stem names the stage. A ``lang_pair`` of
    ``en-fr`` therefore resolves straight to the default — fully backward
    compatible.
    """
    if cfg.get("prompt_template") is not None:
        return cfg["prompt_template"]
    if cfg.get("prompt_path"):
        return Path(cfg["prompt_path"]).read_text(encoding="utf-8")
    lang_pair = cfg.get("lang_pair", DEFAULT_LANG_PAIR)
    default_path = Path(default_path)
    if lang_pair != DEFAULT_LANG_PAIR:
        cand = default_path.parent / f"{default_path.stem}.{lang_pair}.txt"
        if cand.exists():
            return cand.read_text(encoding="utf-8")
    return default_path.read_text(encoding="utf-8")


def _extract_first_json_object(raw: str) -> Optional[str]:
    """Return the first balanced ``{ … }`` substring of *raw*, or None."""
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None


def parse_json_lenient(raw: str) -> Any:
    """Parse JSON from a model response, tolerating prose/fences.

    Order: (1) whole response, (2) ```json …``` fence, (3) first balanced
    ``{ … }`` block. Raises :class:`ValueError` only if all three fail.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, re.I)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    candidate = _extract_first_json_object(raw)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise ValueError(
        "Model did not return JSON parseable by any fallback. "
        f"Raw output (first 600 chars):\n{raw[:600]}"
    )
