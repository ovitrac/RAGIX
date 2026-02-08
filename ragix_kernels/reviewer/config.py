"""
Configuration for the KOAS Reviewer kernel family.

Follows the DocsConfig pattern from ragix_kernels/docs/config.py.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for Markdown chunking."""
    max_chunk_tokens: int = 1800
    overlap_tokens: int = 100
    min_chunk_tokens: int = 50

    def __post_init__(self):
        self.max_chunk_tokens = max(200, self.max_chunk_tokens)
        self.overlap_tokens = max(0, min(self.overlap_tokens, self.max_chunk_tokens // 4))


@dataclass
class LLMConfig:
    """LLM backend configuration."""
    backend: str = "ollama"                     # ollama | claude | auto
    endpoint: str = "http://127.0.0.1:11434"
    pyramid_model: str = "granite3.1-moe:3b"
    edit_model: str = "mistral:instruct"
    tutor_model: Optional[str] = None           # None = single-model mode
    temperature: float = 0.1
    timeout: int = 120
    num_predict: int = 2048
    strict_sovereign: bool = True               # Reject non-local backends
    api_key: str = ""                           # For Claude API (from env)


@dataclass
class StyleRules:
    """House style rules for review."""
    silent_allowlist: List[str] = field(default_factory=lambda: [
        "typo", "punctuation", "capitalization", "spacing", "grammar",
    ])
    ai_leftover_patterns: List[str] = field(default_factory=lambda: [
        r"(?i)\bAs an AI\b",
        r"(?i)\bAs a language model\b",
        r"(?i)\bIn conclusion\b",
        r"(?i)\bSure,?\s+here['']?s\b",
        r"(?i)\bI hope this helps\b",
        r"(?i)\bLet me know if\b",
        r"(?i)\bHere is (?:a|an|the)\b",
        r"(?i)\bCertainly!",
        r"(?i)\bAbsolutely!",
        r"(?i)\bGreat question",
    ])
    protect_tables: bool = True
    protect_math: bool = True


@dataclass
class ReviewerConfig:
    """
    Top-level configuration for KOAS Reviewer.

    Combines chunk, LLM, and style sub-configurations.
    """
    # Sub-configs
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    style: StyleRules = field(default_factory=StyleRules)

    # Output mode
    in_place: bool = False                  # Default: *.REVIEWED.md
    output_suffix: str = ".REVIEWED.md"

    # Pipeline control
    skip_pyramid: bool = False
    no_llm: bool = False                    # Deterministic passes only
    strict: bool = False                    # Refuse edits touching protected areas

    # Pyramid
    pyramid_levels: int = 4                 # 1-4
    summary_max_tokens: int = 200

    # Cache
    cache_mode: str = "write_through"       # write_through|read_only|read_prefer|off

    # Alert mapping
    severity_alert_map: Dict[str, str] = field(default_factory=lambda: {
        "minor": "NOTE",
        "attention": "WARNING",
        "deletion": "CAUTION",
        "critical": "IMPORTANT",
    })

    def __post_init__(self):
        self.pyramid_levels = max(1, min(4, self.pyramid_levels))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": {
                "max_chunk_tokens": self.chunk.max_chunk_tokens,
                "overlap_tokens": self.chunk.overlap_tokens,
                "min_chunk_tokens": self.chunk.min_chunk_tokens,
            },
            "llm": {
                "backend": self.llm.backend,
                "endpoint": self.llm.endpoint,
                "pyramid_model": self.llm.pyramid_model,
                "edit_model": self.llm.edit_model,
                "tutor_model": self.llm.tutor_model,
                "temperature": self.llm.temperature,
                "timeout": self.llm.timeout,
                "strict_sovereign": self.llm.strict_sovereign,
            },
            "style": {
                "silent_allowlist": self.style.silent_allowlist,
                "ai_leftover_patterns": self.style.ai_leftover_patterns,
                "protect_tables": self.style.protect_tables,
                "protect_math": self.style.protect_math,
            },
            "in_place": self.in_place,
            "skip_pyramid": self.skip_pyramid,
            "no_llm": self.no_llm,
            "strict": self.strict,
            "pyramid_levels": self.pyramid_levels,
            "cache_mode": self.cache_mode,
            "severity_alert_map": self.severity_alert_map,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReviewerConfig":
        chunk_d = d.get("chunk", {})
        llm_d = d.get("llm", {})
        style_d = d.get("style", {})
        return cls(
            chunk=ChunkConfig(
                max_chunk_tokens=chunk_d.get("max_chunk_tokens", 1800),
                overlap_tokens=chunk_d.get("overlap_tokens", 100),
                min_chunk_tokens=chunk_d.get("min_chunk_tokens", 50),
            ),
            llm=LLMConfig(
                backend=llm_d.get("backend", "ollama"),
                endpoint=llm_d.get("endpoint", "http://127.0.0.1:11434"),
                pyramid_model=llm_d.get("pyramid_model", "granite3.1-moe:3b"),
                edit_model=llm_d.get("edit_model", "mistral:instruct"),
                tutor_model=llm_d.get("tutor_model"),
                temperature=llm_d.get("temperature", 0.1),
                timeout=llm_d.get("timeout", 120),
                strict_sovereign=llm_d.get("strict_sovereign", True),
            ),
            style=StyleRules(
                silent_allowlist=style_d.get("silent_allowlist",
                    ["typo", "punctuation", "capitalization", "spacing", "grammar"]),
                ai_leftover_patterns=style_d.get("ai_leftover_patterns", []),
                protect_tables=style_d.get("protect_tables", True),
                protect_math=style_d.get("protect_math", True),
            ),
            in_place=d.get("in_place", False),
            skip_pyramid=d.get("skip_pyramid", False),
            no_llm=d.get("no_llm", False),
            strict=d.get("strict", False),
            pyramid_levels=d.get("pyramid_levels", 4),
            cache_mode=d.get("cache_mode", "write_through"),
            severity_alert_map=d.get("severity_alert_map", {
                "minor": "NOTE",
                "attention": "WARNING",
                "deletion": "CAUTION",
                "critical": "IMPORTANT",
            }),
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_config: Optional[ReviewerConfig] = None


def get_reviewer_config() -> ReviewerConfig:
    """Get global reviewer configuration (lazy-loaded default)."""
    global _global_config
    if _global_config is None:
        _global_config = ReviewerConfig()
    return _global_config


def set_reviewer_config(config: ReviewerConfig) -> None:
    """Set the global reviewer configuration."""
    global _global_config
    _global_config = config
