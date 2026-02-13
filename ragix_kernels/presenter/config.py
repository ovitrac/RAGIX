"""
Configuration for the KOAS Presenter kernel family.

Follows the ReviewerConfig pattern from ragix_kernels/reviewer/config.py.
All resolved design decisions (R1-R10) from ROADMAP_KOAS_PRESENTER.md
are reflected in the default values.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

@dataclass
class FolderScanConfig:
    """Configuration for recursive folder scanning (S1)."""
    include_patterns: List[str] = field(default_factory=lambda: [
        "**/*.md", "**/*.txt", "**/*.rst",
    ])
    asset_patterns: List[str] = field(default_factory=lambda: [
        "**/*.svg", "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.pdf",
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/node_modules/**", "**/.git/**", "**/.*", "**/__pycache__/**",
    ])
    max_depth: int = 10
    follow_symlinks: bool = False


@dataclass
class ClusteringConfig:
    """Configuration for topic clustering (S2 normalizer)."""
    method: str = "embedding"               # heading_path | embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    algorithm: str = "hdbscan"              # hdbscan | agglomerative
    cross_file_similarity_threshold: float = 0.75


@dataclass
class LLMRefinementConfig:
    """Configuration for optional LLM cluster refinement (S2 normalizer)."""
    enabled: bool = True
    max_operations: int = 8                 # max merge/split/move per run
    method_tag: str = "llm_refine"          # logged in provenance


@dataclass
class DeduplicationConfig:
    """Configuration for near-duplicate detection."""
    enabled: bool = True
    threshold: float = 0.70                 # Jaccard (deterministic) or cosine (embedding)
    global_threshold: float = 0.80          # Cross-section dedupe (stricter to avoid cross-topic FP)


@dataclass
class ImportanceConfig:
    """Configuration for unit importance scoring."""
    boost_headings: bool = True             # H1/H2 get higher scores
    boost_findings: bool = True             # Units with metrics/numbers
    decay_depth: float = 0.1               # Score decay per heading depth level


@dataclass
class BudgetConfig:
    """Token budget for LLM operations (R1: tiered escalation)."""
    tier: str = "auto"                      # auto | T0 | T1 | T2 | T3
    max_llm_input_tokens_per_cluster: int = 1500
    max_llm_output_tokens_per_cluster: int = 300
    max_llm_total_input_tokens_per_run: int = 20000
    max_llm_calls_per_run: int = 16

    def __post_init__(self):
        valid_tiers = ("auto", "T0", "T1", "T2", "T3")
        if self.tier not in valid_tiers:
            raise ValueError(f"Invalid tier {self.tier!r}, must be one of {valid_tiers}")


@dataclass
class LanguageConfig:
    """Multi-language support configuration (R4)."""
    detection: str = "fasttext"             # fasttext | cld3 | none
    role_lexicon: str = "fr,en"             # bilingual keyword mapping


@dataclass
class NormalizerConfig:
    """Configuration for the semantic normalizer kernel (S2)."""
    enabled: bool = True                    # false → identity pass-through
    mode: str = "auto"                      # auto | deterministic | llm
    model: str = "mistral-small:24b"         # Ollama model for LLM ops
    max_clusters: int = 20                  # consolidate clusters above this count
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    llm_refinement: LLMRefinementConfig = field(default_factory=LLMRefinementConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    narrative_template: Optional[List[str]] = None  # null → auto-detect
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)

    def __post_init__(self):
        valid_modes = ("auto", "deterministic", "llm")
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode {self.mode!r}, must be one of {valid_modes}")


@dataclass
class TocConfig:
    """Table of Contents slide configuration."""
    enabled: bool = True                    # generate a TOC slide after title
    items_per_page: int = 12               # max entries per TOC page (0 = single page)
    min_level: int = 1                      # min heading level (1 = section dividers in MARP)
    max_level: int = 1                      # max heading level (1 = section dividers only)
    title: str = "Table of Contents"        # heading for the TOC slide
    continuation_title: str = "Table of Contents (continued)"  # heading for overflow pages


@dataclass
class SlidePlanConfig:
    """Configuration for slide planning (S2)."""
    mode: str = "auto"                      # auto | outline | normalized
    max_bullets_per_slide: int = 6
    max_words_per_slide: int = 80           # target, not hard limit
    split_long_lists: bool = True
    equation_standalone: bool = True        # block equations get own slide
    code_standalone_min_lines: int = 5
    merge_short_sections: bool = True
    min_slides: int = 8
    max_slides: int = 60
    # Compression mode (v1.2)
    compression: str = "full"               # full | compressed | executive
    max_slides_per_section: int = 0         # 0 = unlimited; >0 = hard cap per section
    annex_exclude_patterns: List[str] = field(default_factory=lambda: [
        "annexe", "annex", "appendix", "appendice",
    ])
    executive_min_importance: float = 0.5   # min importance for executive mode inclusion
    # TOC (v1.2.1)
    toc: TocConfig = field(default_factory=TocConfig)

    def __post_init__(self):
        self.min_slides = max(1, self.min_slides)
        self.max_slides = max(self.min_slides, self.max_slides)
        valid_compression = ("full", "compressed", "executive")
        if self.compression not in valid_compression:
            raise ValueError(
                f"Invalid compression {self.compression!r}, must be one of {valid_compression}"
            )


@dataclass
class TableOverflowConfig:
    """Table overflow handling policy (R7)."""
    max_rows: int = 12
    max_cols: int = 8
    max_cell_chars: int = 40                # monospace char count heuristic
    strategy: str = "split"                 # split | image | truncate

    def __post_init__(self):
        valid = ("split", "image", "truncate")
        if self.strategy not in valid:
            raise ValueError(f"Invalid strategy {self.strategy!r}, must be one of {valid}")


@dataclass
class ThemeConfig:
    """MARP theme configuration."""
    name: str = "default"                   # default | gaia | uncover | custom
    custom_css_path: Optional[str] = None
    size: str = "16:9"                      # 16:9 | 4:3
    math: str = "katex"                     # katex | mathjax
    header: str = "{organization}"
    footer: str = "{title} — {date}"
    paginate: bool = True
    logo: Optional[str] = None              # path to header logo
    colors: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#0066cc",
        "secondary": "#2d3436",
        "accent": "#e17055",
        "background": "#ffffff",
        "text": "#2d3436",
    })


@dataclass
class NotesConfig:
    """Speaker notes configuration (R5)."""
    depth: str = "file_line"                # none | section_only | file_line | full

    def __post_init__(self):
        valid = ("none", "section_only", "file_line", "full")
        if self.depth not in valid:
            raise ValueError(f"Invalid depth {self.depth!r}, must be one of {valid}")


@dataclass
class ExportConfig:
    """MARP export configuration."""
    format: str = "md"                      # md | pdf | html | pptx | png
    pdf_notes: bool = True
    pdf_outlines: bool = True
    symlink_assets: bool = False            # true → symlink instead of copy

    def __post_init__(self):
        valid = ("md", "pdf", "html", "pptx", "png")
        if self.format not in valid:
            raise ValueError(f"Invalid format {self.format!r}, must be one of {valid}")


@dataclass
class LLMConfig:
    """LLM backend configuration for presenter."""
    backend: str = "ollama"
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "mistral-small:24b"
    temperature: float = 0.1
    timeout: int = 120
    num_predict: int = 1024
    strict_sovereign: bool = True


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

@dataclass
class PresenterConfig:
    """
    Top-level configuration for the KOAS Presenter.

    Combines all sub-configurations. Default values reflect resolved
    design decisions R1-R10 from ROADMAP_KOAS_PRESENTER.md.
    """
    # S1: Collection
    folder_scan: FolderScanConfig = field(default_factory=FolderScanConfig)

    # S2: Structuring
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    slide_plan: SlidePlanConfig = field(default_factory=SlidePlanConfig)
    table_overflow: TableOverflowConfig = field(default_factory=TableOverflowConfig)

    # S3: Rendering
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    notes: NotesConfig = field(default_factory=NotesConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # LLM
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Metadata
    title: str = ""
    subtitle: str = ""
    author: str = ""
    organization: str = ""
    date: str = ""                          # ISO date or empty for auto
    lang: str = "auto"                      # auto | fr | en | ...

    # Pipeline control
    polished: bool = False                  # T3 tier: full LLM polish

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "folder_scan": {
                "include_patterns": self.folder_scan.include_patterns,
                "asset_patterns": self.folder_scan.asset_patterns,
                "exclude_patterns": self.folder_scan.exclude_patterns,
                "max_depth": self.folder_scan.max_depth,
                "follow_symlinks": self.folder_scan.follow_symlinks,
            },
            "normalizer": {
                "enabled": self.normalizer.enabled,
                "mode": self.normalizer.mode,
                "model": self.normalizer.model,
                "clustering": {
                    "method": self.normalizer.clustering.method,
                    "embedding_model": self.normalizer.clustering.embedding_model,
                    "algorithm": self.normalizer.clustering.algorithm,
                    "cross_file_similarity_threshold": self.normalizer.clustering.cross_file_similarity_threshold,
                },
                "llm_refinement": {
                    "enabled": self.normalizer.llm_refinement.enabled,
                    "max_operations": self.normalizer.llm_refinement.max_operations,
                },
                "deduplication": {
                    "enabled": self.normalizer.deduplication.enabled,
                    "threshold": self.normalizer.deduplication.threshold,
                    "global_threshold": self.normalizer.deduplication.global_threshold,
                },
                "importance": {
                    "boost_headings": self.normalizer.importance.boost_headings,
                    "boost_findings": self.normalizer.importance.boost_findings,
                    "decay_depth": self.normalizer.importance.decay_depth,
                },
                "narrative_template": self.normalizer.narrative_template,
                "budget": {
                    "tier": self.normalizer.budget.tier,
                    "max_llm_input_tokens_per_cluster": self.normalizer.budget.max_llm_input_tokens_per_cluster,
                    "max_llm_output_tokens_per_cluster": self.normalizer.budget.max_llm_output_tokens_per_cluster,
                    "max_llm_total_input_tokens_per_run": self.normalizer.budget.max_llm_total_input_tokens_per_run,
                    "max_llm_calls_per_run": self.normalizer.budget.max_llm_calls_per_run,
                },
                "language": {
                    "detection": self.normalizer.language.detection,
                    "role_lexicon": self.normalizer.language.role_lexicon,
                },
            },
            "slide_plan": {
                "mode": self.slide_plan.mode,
                "max_bullets_per_slide": self.slide_plan.max_bullets_per_slide,
                "max_words_per_slide": self.slide_plan.max_words_per_slide,
                "split_long_lists": self.slide_plan.split_long_lists,
                "equation_standalone": self.slide_plan.equation_standalone,
                "code_standalone_min_lines": self.slide_plan.code_standalone_min_lines,
                "merge_short_sections": self.slide_plan.merge_short_sections,
                "min_slides": self.slide_plan.min_slides,
                "max_slides": self.slide_plan.max_slides,
                "compression": self.slide_plan.compression,
                "max_slides_per_section": self.slide_plan.max_slides_per_section,
                "annex_exclude_patterns": self.slide_plan.annex_exclude_patterns,
                "executive_min_importance": self.slide_plan.executive_min_importance,
                "toc": {
                    "enabled": self.slide_plan.toc.enabled,
                    "items_per_page": self.slide_plan.toc.items_per_page,
                    "min_level": self.slide_plan.toc.min_level,
                    "max_level": self.slide_plan.toc.max_level,
                    "title": self.slide_plan.toc.title,
                    "continuation_title": self.slide_plan.toc.continuation_title,
                },
            },
            "table_overflow": {
                "max_rows": self.table_overflow.max_rows,
                "max_cols": self.table_overflow.max_cols,
                "max_cell_chars": self.table_overflow.max_cell_chars,
                "strategy": self.table_overflow.strategy,
            },
            "theme": {
                "name": self.theme.name,
                "custom_css_path": self.theme.custom_css_path,
                "size": self.theme.size,
                "math": self.theme.math,
                "header": self.theme.header,
                "footer": self.theme.footer,
                "paginate": self.theme.paginate,
                "logo": self.theme.logo,
                "colors": self.theme.colors,
            },
            "notes": {"depth": self.notes.depth},
            "export": {
                "format": self.export.format,
                "pdf_notes": self.export.pdf_notes,
                "pdf_outlines": self.export.pdf_outlines,
                "symlink_assets": self.export.symlink_assets,
            },
            "llm": {
                "backend": self.llm.backend,
                "endpoint": self.llm.endpoint,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "timeout": self.llm.timeout,
                "strict_sovereign": self.llm.strict_sovereign,
            },
            "title": self.title,
            "subtitle": self.subtitle,
            "author": self.author,
            "organization": self.organization,
            "date": self.date,
            "lang": self.lang,
            "polished": self.polished,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PresenterConfig":
        """Deserialize from dictionary."""
        fs = d.get("folder_scan", {})
        norm = d.get("normalizer", {})
        sp = d.get("slide_plan", {})
        to = d.get("table_overflow", {})
        th = d.get("theme", {})
        ex = d.get("export", {})
        llm = d.get("llm", {})
        clust = norm.get("clustering", {})
        refine = norm.get("llm_refinement", {})
        dedup = norm.get("deduplication", {})
        imp = norm.get("importance", {})
        budget = norm.get("budget", {})
        lang = norm.get("language", {})

        return cls(
            folder_scan=FolderScanConfig(
                include_patterns=fs.get("include_patterns", ["**/*.md", "**/*.txt", "**/*.rst"]),
                asset_patterns=fs.get("asset_patterns", ["**/*.svg", "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.pdf"]),
                exclude_patterns=fs.get("exclude_patterns", ["**/node_modules/**", "**/.git/**", "**/.*", "**/__pycache__/**"]),
                max_depth=fs.get("max_depth", 10),
                follow_symlinks=fs.get("follow_symlinks", False),
            ),
            normalizer=NormalizerConfig(
                enabled=norm.get("enabled", True),
                mode=norm.get("mode", "auto"),
                model=norm.get("model", "mistral-small:24b"),
                clustering=ClusteringConfig(
                    method=clust.get("method", "embedding"),
                    embedding_model=clust.get("embedding_model", "all-MiniLM-L6-v2"),
                    algorithm=clust.get("algorithm", "hdbscan"),
                    cross_file_similarity_threshold=clust.get("cross_file_similarity_threshold", 0.75),
                ),
                llm_refinement=LLMRefinementConfig(
                    enabled=refine.get("enabled", True),
                    max_operations=refine.get("max_operations", 8),
                    method_tag=refine.get("method_tag", "llm_refine"),
                ),
                deduplication=DeduplicationConfig(
                    enabled=dedup.get("enabled", True),
                    threshold=dedup.get("threshold", 0.70),
                    global_threshold=dedup.get("global_threshold", 0.80),
                ),
                importance=ImportanceConfig(
                    boost_headings=imp.get("boost_headings", True),
                    boost_findings=imp.get("boost_findings", True),
                    decay_depth=imp.get("decay_depth", 0.1),
                ),
                narrative_template=norm.get("narrative_template"),
                budget=BudgetConfig(
                    tier=budget.get("tier", "auto"),
                    max_llm_input_tokens_per_cluster=budget.get("max_llm_input_tokens_per_cluster", 1500),
                    max_llm_output_tokens_per_cluster=budget.get("max_llm_output_tokens_per_cluster", 300),
                    max_llm_total_input_tokens_per_run=budget.get("max_llm_total_input_tokens_per_run", 20000),
                    max_llm_calls_per_run=budget.get("max_llm_calls_per_run", 16),
                ),
                language=LanguageConfig(
                    detection=lang.get("detection", "fasttext"),
                    role_lexicon=lang.get("role_lexicon", "fr,en"),
                ),
            ),
            slide_plan=SlidePlanConfig(
                mode=sp.get("mode", "auto"),
                max_bullets_per_slide=sp.get("max_bullets_per_slide", 6),
                max_words_per_slide=sp.get("max_words_per_slide", 80),
                split_long_lists=sp.get("split_long_lists", True),
                equation_standalone=sp.get("equation_standalone", True),
                code_standalone_min_lines=sp.get("code_standalone_min_lines", 5),
                merge_short_sections=sp.get("merge_short_sections", True),
                min_slides=sp.get("min_slides", 8),
                max_slides=sp.get("max_slides", 60),
                compression=sp.get("compression", "full"),
                max_slides_per_section=sp.get("max_slides_per_section", 0),
                annex_exclude_patterns=sp.get("annex_exclude_patterns", [
                    "annexe", "annex", "appendix", "appendice",
                ]),
                executive_min_importance=sp.get("executive_min_importance", 0.5),
                toc=TocConfig(
                    enabled=sp.get("toc", {}).get("enabled", True),
                    items_per_page=sp.get("toc", {}).get("items_per_page", 12),
                    min_level=sp.get("toc", {}).get("min_level", 1),
                    max_level=sp.get("toc", {}).get("max_level", 1),
                    title=sp.get("toc", {}).get("title", "Table of Contents"),
                    continuation_title=sp.get("toc", {}).get("continuation_title", "Table of Contents (continued)"),
                ),
            ),
            table_overflow=TableOverflowConfig(
                max_rows=to.get("max_rows", 12),
                max_cols=to.get("max_cols", 8),
                max_cell_chars=to.get("max_cell_chars", 40),
                strategy=to.get("strategy", "split"),
            ),
            theme=ThemeConfig(
                name=th.get("name", "default"),
                custom_css_path=th.get("custom_css_path"),
                size=th.get("size", "16:9"),
                math=th.get("math", "katex"),
                header=th.get("header", "{organization}"),
                footer=th.get("footer", "{title} — {date}"),
                paginate=th.get("paginate", True),
                logo=th.get("logo"),
                colors=th.get("colors", {
                    "primary": "#0066cc",
                    "secondary": "#2d3436",
                    "accent": "#e17055",
                    "background": "#ffffff",
                    "text": "#2d3436",
                }),
            ),
            notes=NotesConfig(depth=d.get("notes", {}).get("depth", "file_line")),
            export=ExportConfig(
                format=ex.get("format", "md"),
                pdf_notes=ex.get("pdf_notes", True),
                pdf_outlines=ex.get("pdf_outlines", True),
                symlink_assets=ex.get("symlink_assets", False),
            ),
            llm=LLMConfig(
                backend=llm.get("backend", "ollama"),
                endpoint=llm.get("endpoint", "http://127.0.0.1:11434"),
                model=llm.get("model", "mistral-small:24b"),
                temperature=llm.get("temperature", 0.1),
                timeout=llm.get("timeout", 120),
                strict_sovereign=llm.get("strict_sovereign", True),
            ),
            title=d.get("title", ""),
            subtitle=d.get("subtitle", ""),
            author=d.get("author", ""),
            organization=d.get("organization", ""),
            date=d.get("date", ""),
            lang=d.get("lang", "auto"),
            polished=d.get("polished", False),
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_config: Optional[PresenterConfig] = None


def get_presenter_config() -> PresenterConfig:
    """Get global presenter configuration (lazy-loaded default)."""
    global _global_config
    if _global_config is None:
        _global_config = PresenterConfig()
    return _global_config


def set_presenter_config(config: PresenterConfig) -> None:
    """Set the global presenter configuration."""
    global _global_config
    _global_config = config
