"""
Jinja2 prompt templates for LLM-driven review operations.

Provides render_prompt() which loads .j2 templates from this directory.
Falls back to string.Template if Jinja2 is not installed.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

from pathlib import Path
from typing import Any, Dict

import logging

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    _env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,  # Plain text prompts, not HTML
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False
    _env = None
    logger.info("jinja2 not available, using string.Template fallback")


def render_prompt(template_name: str, **kwargs: Any) -> str:
    """
    Render a prompt template with the given variables.

    Args:
        template_name: Template filename (e.g. "review.j2")
        **kwargs: Variables to pass to the template

    Returns:
        Rendered prompt string
    """
    if _JINJA2_AVAILABLE and _env is not None:
        template = _env.get_template(template_name)
        return template.render(**kwargs)

    # Fallback: read file and do basic substitution
    template_path = _TEMPLATE_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    text = template_path.read_text(encoding="utf-8")
    # Basic {{ var }} substitution
    for key, value in kwargs.items():
        text = text.replace("{{ " + key + " }}", str(value))
        text = text.replace("{{" + key + "}}", str(value))
    # Strip Jinja2 control blocks in fallback mode
    import re
    text = re.sub(r"\{%.*?%\}", "", text)
    return text


def list_templates():
    """List available prompt templates."""
    return sorted(p.name for p in _TEMPLATE_DIR.glob("*.j2"))
