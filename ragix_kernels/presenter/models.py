"""
Core data models for the KOAS Presenter kernel family.

All models are JSON-serializable dataclasses used across kernels for
content extraction, semantic normalization, slide planning, and rendering.

Follows the reviewer models.py pattern: enums, dataclasses with
to_dict()/from_dict(), and utility functions.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FileType(str, Enum):
    """Classification of scanned files."""
    DOCUMENT = "document"       # .md, .txt, .rst
    ASSET = "asset"             # .svg, .png, .jpg, .gif, .pdf
    DATA = "data"               # .json, .yaml, .csv, .tsv
    CONFIG = "config"           # .toml, .ini, .cfg
    UNKNOWN = "unknown"


class UnitType(str, Enum):
    """Types of semantic content units extracted from documents."""
    HEADING = "heading"                 # Section title (H1-H6)
    PARAGRAPH = "paragraph"             # Prose text block
    BULLET_LIST = "bullet_list"         # Unordered list
    NUMBERED_LIST = "numbered_list"     # Ordered list
    TABLE = "table"                     # Markdown table
    CODE_BLOCK = "code_block"           # Fenced code (```lang ... ```)
    EQUATION_BLOCK = "equation_block"   # Display math ($$...$$)
    EQUATION_INLINE = "equation_inline" # Inline math ($...$)
    BLOCKQUOTE = "blockquote"           # > quoted text
    IMAGE_REF = "image_ref"             # ![alt](path) reference
    MERMAID = "mermaid"                 # ```mermaid ... ``` diagram
    FRONT_MATTER = "front_matter"       # YAML front matter
    ADMONITION = "admonition"           # > [!NOTE] / > [!WARNING] etc.


class AssetType(str, Enum):
    """Types of presentable assets."""
    IMAGE = "image"         # SVG, PNG, JPG, GIF
    EQUATION = "equation"   # Block math expression
    TABLE = "table"         # Large table (cataloged for potential image rendering)
    CODE = "code"           # Code block
    DIAGRAM = "diagram"     # Mermaid or other diagram


class UnitRole(str, Enum):
    """Semantic role assigned by the normalizer."""
    CONTEXT = "context"                 # Background, setup
    PROBLEM = "problem"                 # Problem statement, gap, issue
    METHOD = "method"                   # Approach, methodology
    FINDING = "finding"                 # Result, observation, metric
    RECOMMENDATION = "recommendation"   # Action item, suggestion
    CONCLUSION = "conclusion"           # Summary, takeaway
    REFERENCE = "reference"             # Citation, link, appendix
    ILLUSTRATION = "illustration"       # Figure, diagram, chart
    METADATA = "metadata"               # Front-matter, boilerplate
    UNKNOWN = "unknown"                 # Unclassified


class SlideType(str, Enum):
    """Types of presentation slides."""
    TITLE = "title"
    SECTION = "section"
    CONTENT = "content"
    TWO_COLUMN = "two_column"
    IMAGE_TEXT = "image_text"
    IMAGE_FULL = "image_full"
    EQUATION = "equation"
    TABLE = "table"
    CODE = "code"
    QUOTE = "quote"
    SUMMARY = "summary"
    BLANK = "blank"


class ProvenanceMethod(str, Enum):
    """How a slide's content was obtained."""
    EXTRACTED = "extracted"             # Directly from source document
    SYNTHESIZED = "synthesized"         # Merged/summarized from multiple units
    USER_OUTLINE = "user_outline"       # Specified by user outline
    AUTO_SECTION = "auto_section"       # Auto-generated section divider


class NormalizationMode(str, Enum):
    """How the corpus was normalized."""
    IDENTITY = "identity"               # Pass-through (no normalization)
    DETERMINISTIC = "deterministic"     # Heuristics only (no LLM)
    LLM = "llm"                         # LLM-assisted normalization


# ---------------------------------------------------------------------------
# S1: Content Extraction Models
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    """A file discovered during folder scan."""
    path: str                   # Relative to folder root
    file_type: FileType
    size_bytes: int
    file_hash: str              # "sha256:..."
    line_count: int = 0         # 0 for binary files
    extension: str = ""
    front_matter: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "path": self.path,
            "file_type": self.file_type.value,
            "size_bytes": self.size_bytes,
            "file_hash": self.file_hash,
            "line_count": self.line_count,
            "extension": self.extension,
        }
        if self.front_matter is not None:
            d["front_matter"] = self.front_matter
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FileEntry:
        return cls(
            path=d["path"],
            file_type=FileType(d["file_type"]),
            size_bytes=d["size_bytes"],
            file_hash=d["file_hash"],
            line_count=d.get("line_count", 0),
            extension=d.get("extension", ""),
            front_matter=d.get("front_matter"),
        )


@dataclass
class OutlineNode:
    """A node in the cross-document heading tree."""
    id: str                     # "file-stem:H2.3"
    level: int                  # 1-6
    title: str
    source_file: str
    line: int                   # 1-based
    children: List[OutlineNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "source_file": self.source_file,
            "line": self.line,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> OutlineNode:
        return cls(
            id=d["id"],
            level=d["level"],
            title=d["title"],
            source_file=d["source_file"],
            line=d["line"],
            children=[cls.from_dict(c) for c in d.get("children", [])],
        )


@dataclass
class SemanticUnit:
    """
    Atomic content unit extracted from a document.

    The fundamental building block for slide generation. Each unit maps
    to a specific region in a source file and carries enough metadata
    for downstream planning.
    """
    id: str                             # "file-stem:L42-L58"
    type: UnitType
    content: str                        # Raw text
    source_file: str                    # Relative path from folder root
    source_lines: Tuple[int, int]       # (start, end) 1-based inclusive
    heading_path: List[str]             # ["H1 title", "H2 title", ...]
    depth: int                          # Heading nesting level (0=root)
    tokens: int                         # Estimated token count
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Type-specific metadata:
    #   CODE_BLOCK: {"language": "python"}
    #   IMAGE_REF:  {"alt": "...", "path": "...", "asset_id": "..."}
    #   TABLE:      {"rows": N, "cols": M, "has_header": true}
    #   EQUATION_BLOCK: {"latex": "..."}
    #   MERMAID:    {"diagram_type": "graph|flowchart|sequence|..."}
    #   HEADING:    {"level": N}
    #   FRONT_MATTER: {"keys": [...]}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "source_file": self.source_file,
            "source_lines": list(self.source_lines),
            "heading_path": self.heading_path,
            "depth": self.depth,
            "tokens": self.tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SemanticUnit:
        return cls(
            id=d["id"],
            type=UnitType(d["type"]),
            content=d["content"],
            source_file=d["source_file"],
            source_lines=tuple(d["source_lines"]),
            heading_path=d.get("heading_path", []),
            depth=d.get("depth", 0),
            tokens=d.get("tokens", 0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ContentCorpus:
    """All content extracted from a document folder (S1 output)."""
    root: str                           # Folder path
    files: List[FileEntry]              # Scanned files
    units: List[SemanticUnit]           # All extracted units (ordered)
    outline: List[OutlineNode]          # Heading tree across all docs
    total_tokens: int = 0
    total_files: int = 0
    total_documents: int = 0            # Document files only (not assets)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "files": [f.to_dict() for f in self.files],
            "units": [u.to_dict() for u in self.units],
            "outline": [o.to_dict() for o in self.outline],
            "total_tokens": self.total_tokens,
            "total_files": self.total_files,
            "total_documents": self.total_documents,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ContentCorpus:
        return cls(
            root=d["root"],
            files=[FileEntry.from_dict(f) for f in d.get("files", [])],
            units=[SemanticUnit.from_dict(u) for u in d.get("units", [])],
            outline=[OutlineNode.from_dict(o) for o in d.get("outline", [])],
            total_tokens=d.get("total_tokens", 0),
            total_files=d.get("total_files", 0),
            total_documents=d.get("total_documents", 0),
        )


@dataclass
class Asset:
    """A presentable asset (image, equation, table, code, diagram)."""
    id: str                             # "asset-NNN"
    type: AssetType
    source_file: str                    # Origin document
    source_lines: Tuple[int, int]       # (start, end) 1-based
    content: str                        # Raw content (equation text, table MD, code)
    path: Optional[str] = None          # File path for images/diagrams
    format: Optional[str] = None        # "svg", "png", "jpg", "mermaid"
    dimensions: Optional[Tuple[int, int]] = None  # (w, h) for images
    alt_text: Optional[str] = None
    caption: Optional[str] = None       # Extracted from surrounding text

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "source_file": self.source_file,
            "source_lines": list(self.source_lines),
            "content": self.content,
        }
        if self.path is not None:
            d["path"] = self.path
        if self.format is not None:
            d["format"] = self.format
        if self.dimensions is not None:
            d["dimensions"] = list(self.dimensions)
        if self.alt_text is not None:
            d["alt_text"] = self.alt_text
        if self.caption is not None:
            d["caption"] = self.caption
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Asset:
        dims = d.get("dimensions")
        return cls(
            id=d["id"],
            type=AssetType(d["type"]),
            source_file=d["source_file"],
            source_lines=tuple(d["source_lines"]),
            content=d["content"],
            path=d.get("path"),
            format=d.get("format"),
            dimensions=tuple(dims) if dims else None,
            alt_text=d.get("alt_text"),
            caption=d.get("caption"),
        )


@dataclass
class AssetCatalog:
    """Unified catalog of all presentable assets."""
    assets: List[Asset]
    by_type: Dict[str, List[str]] = field(default_factory=dict)     # type → [asset IDs]
    by_file: Dict[str, List[str]] = field(default_factory=dict)     # source file → [asset IDs]

    def get(self, asset_id: str) -> Optional[Asset]:
        """Look up asset by ID."""
        for a in self.assets:
            if a.id == asset_id:
                return a
        return None

    def get_by_path(self, path: str) -> Optional[Asset]:
        """Look up asset by source_file or path (normalized filename match)."""
        if not path:
            return None
        # Exact match first
        for a in self.assets:
            if a.source_file == path or a.path == path:
                return a
        # Basename match (handles assets/ prefix variations)
        from pathlib import PurePosixPath
        target = PurePosixPath(path).name
        for a in self.assets:
            if PurePosixPath(a.source_file).name == target:
                return a
            if a.path and PurePosixPath(a.path).name == target:
                return a
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": [a.to_dict() for a in self.assets],
            "by_type": self.by_type,
            "by_file": self.by_file,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AssetCatalog:
        return cls(
            assets=[Asset.from_dict(a) for a in d.get("assets", [])],
            by_type=d.get("by_type", {}),
            by_file=d.get("by_file", {}),
        )


# ---------------------------------------------------------------------------
# S2: Semantic Normalization Models
# ---------------------------------------------------------------------------

@dataclass
class NormalizedUnit:
    """SemanticUnit enriched with normalization metadata."""
    unit: SemanticUnit                          # Original unit (preserved)
    topic_cluster: Optional[str] = None         # "Architecture", "Security", ...
    importance: float = 0.5                     # 0.0–1.0, slide inclusion priority
    role: UnitRole = UnitRole.UNKNOWN
    summary: Optional[str] = None               # Extractive or abstractive
    duplicate_of: Optional[str] = None          # ID of canonical unit (if redundant)
    merge_group: Optional[str] = None           # Group ID for units to be merged
    lang: Optional[str] = None                  # Detected language code

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "unit": self.unit.to_dict(),
            "importance": round(self.importance, 4),
            "role": self.role.value,
        }
        if self.topic_cluster is not None:
            d["topic_cluster"] = self.topic_cluster
        if self.summary is not None:
            d["summary"] = self.summary
        if self.duplicate_of is not None:
            d["duplicate_of"] = self.duplicate_of
        if self.merge_group is not None:
            d["merge_group"] = self.merge_group
        if self.lang is not None:
            d["lang"] = self.lang
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NormalizedUnit:
        return cls(
            unit=SemanticUnit.from_dict(d["unit"]),
            topic_cluster=d.get("topic_cluster"),
            importance=d.get("importance", 0.5),
            role=UnitRole(d.get("role", "unknown")),
            summary=d.get("summary"),
            duplicate_of=d.get("duplicate_of"),
            merge_group=d.get("merge_group"),
            lang=d.get("lang"),
        )

    @classmethod
    def from_semantic_unit(cls, u: SemanticUnit) -> NormalizedUnit:
        """Create NormalizedUnit from raw SemanticUnit (identity normalization)."""
        return cls(unit=u)


@dataclass
class TopicCluster:
    """A group of semantically related content units."""
    id: str                                     # "cluster-NNN"
    label: str                                  # Human-readable topic name
    unit_ids: List[str]                         # Member unit IDs
    importance: float                           # Aggregate importance
    suggested_slides: int                       # Estimated slide count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "unit_ids": self.unit_ids,
            "importance": round(self.importance, 4),
            "suggested_slides": self.suggested_slides,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TopicCluster:
        return cls(
            id=d["id"],
            label=d["label"],
            unit_ids=d.get("unit_ids", []),
            importance=d.get("importance", 0.5),
            suggested_slides=d.get("suggested_slides", 1),
        )


@dataclass
class NarrativeArc:
    """Suggested presentation storyline."""
    sections: List[str]                         # Ordered section labels
    # e.g., ["Context", "Problem", "Analysis", "Findings", "Recommendations"]

    def to_dict(self) -> Dict[str, Any]:
        return {"sections": self.sections}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NarrativeArc:
        return cls(sections=d.get("sections", ["Content"]))


@dataclass
class NormalizedCorpus:
    """ContentCorpus augmented with semantic structure (S2 normalizer output)."""
    raw: ContentCorpus                          # Original corpus (preserved)
    units: List[NormalizedUnit]                 # Enriched units
    clusters: List[TopicCluster]                # Topic clusters
    narrative: NarrativeArc                     # Suggested ordering
    duplicates_removed: int                     # Count of deduplicated units
    merge_groups: Dict[str, List[str]]          # group_id → [unit_ids]
    normalization_mode: NormalizationMode        # identity | deterministic | llm

    def active_units(self) -> List[NormalizedUnit]:
        """Units not flagged as duplicates (included in slide planning)."""
        return [u for u in self.units if u.duplicate_of is None]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw.to_dict(),
            "units": [u.to_dict() for u in self.units],
            "clusters": [c.to_dict() for c in self.clusters],
            "narrative": self.narrative.to_dict(),
            "duplicates_removed": self.duplicates_removed,
            "merge_groups": self.merge_groups,
            "normalization_mode": self.normalization_mode.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> NormalizedCorpus:
        return cls(
            raw=ContentCorpus.from_dict(d["raw"]),
            units=[NormalizedUnit.from_dict(u) for u in d.get("units", [])],
            clusters=[TopicCluster.from_dict(c) for c in d.get("clusters", [])],
            narrative=NarrativeArc.from_dict(d.get("narrative", {})),
            duplicates_removed=d.get("duplicates_removed", 0),
            merge_groups=d.get("merge_groups", {}),
            normalization_mode=NormalizationMode(d.get("normalization_mode", "identity")),
        )

    @classmethod
    def identity(cls, corpus: ContentCorpus) -> NormalizedCorpus:
        """Create identity-normalized corpus (pass-through, no LLM)."""
        units = [NormalizedUnit.from_semantic_unit(u) for u in corpus.units]
        all_ids = [u.unit.id for u in units]
        return cls(
            raw=corpus,
            units=units,
            clusters=[TopicCluster(
                id="cluster-000",
                label="All",
                unit_ids=all_ids,
                importance=1.0,
                suggested_slides=len(all_ids),
            )],
            narrative=NarrativeArc(sections=["Content"]),
            duplicates_removed=0,
            merge_groups={},
            normalization_mode=NormalizationMode.IDENTITY,
        )


# ---------------------------------------------------------------------------
# S2: Slide Deck Models (THE CONTRACT)
# ---------------------------------------------------------------------------

@dataclass
class SlideProvenance:
    """Traceability link from a slide back to source documents."""
    source_file: str = ""
    source_lines: List[int] = field(default_factory=list)   # [start, end]
    heading_path: List[str] = field(default_factory=list)
    unit_ids: List[str] = field(default_factory=list)
    method: ProvenanceMethod = ProvenanceMethod.EXTRACTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_lines": self.source_lines,
            "heading_path": self.heading_path,
            "unit_ids": self.unit_ids,
            "method": self.method.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlideProvenance:
        return cls(
            source_file=d.get("source_file", ""),
            source_lines=d.get("source_lines", []),
            heading_path=d.get("heading_path", []),
            unit_ids=d.get("unit_ids", []),
            method=ProvenanceMethod(d.get("method", "extracted")),
        )


@dataclass
class SlideImage:
    """Image reference within a slide."""
    asset_id: str = ""
    path: str = ""
    alt: str = ""
    caption: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.asset_id:
            d["asset_id"] = self.asset_id
        if self.path:
            d["path"] = self.path
        if self.alt:
            d["alt"] = self.alt
        if self.caption:
            d["caption"] = self.caption
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlideImage:
        return cls(
            asset_id=d.get("asset_id", ""),
            path=d.get("path", ""),
            alt=d.get("alt", ""),
            caption=d.get("caption", ""),
        )


@dataclass
class SlideCode:
    """Code block within a slide."""
    language: str = ""
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"language": self.language, "text": self.text}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlideCode:
        return cls(language=d.get("language", ""), text=d.get("text", ""))


@dataclass
class SlideTable:
    """Table within a slide."""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    caption: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"headers": self.headers, "rows": self.rows}
        if self.caption:
            d["caption"] = self.caption
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlideTable:
        return cls(
            headers=d.get("headers", []),
            rows=d.get("rows", []),
            caption=d.get("caption", ""),
        )


@dataclass
class SlideColumn:
    """One column in a two-column slide."""
    width: str = "50%"
    body: List[str] = field(default_factory=list)
    bullets: List[str] = field(default_factory=list)
    image: Optional[SlideImage] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"width": self.width}
        if self.body:
            d["body"] = self.body
        if self.bullets:
            d["bullets"] = self.bullets
        if self.image:
            d["image"] = self.image.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlideColumn:
        img = d.get("image")
        return cls(
            width=d.get("width", "50%"),
            body=d.get("body", []),
            bullets=d.get("bullets", []),
            image=SlideImage.from_dict(img) if img else None,
        )


@dataclass
class SlideContent:
    """Content fields for a slide (presentation-agnostic)."""
    heading: str = ""
    subheading: str = ""
    body: List[str] = field(default_factory=list)
    bullets: List[str] = field(default_factory=list)
    numbered: List[str] = field(default_factory=list)
    code: Optional[SlideCode] = None
    equation: str = ""
    table: Optional[SlideTable] = None
    image: Optional[SlideImage] = None
    columns: List[SlideColumn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.heading:
            d["heading"] = self.heading
        if self.subheading:
            d["subheading"] = self.subheading
        if self.body:
            d["body"] = self.body
        if self.bullets:
            d["bullets"] = self.bullets
        if self.numbered:
            d["numbered"] = self.numbered
        if self.code:
            d["code"] = self.code.to_dict()
        if self.equation:
            d["equation"] = self.equation
        if self.table:
            d["table"] = self.table.to_dict()
        if self.image:
            d["image"] = self.image.to_dict()
        if self.columns:
            d["columns"] = [c.to_dict() for c in self.columns]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlideContent:
        code = d.get("code")
        table = d.get("table")
        image = d.get("image")
        return cls(
            heading=d.get("heading", ""),
            subheading=d.get("subheading", ""),
            body=d.get("body", []),
            bullets=d.get("bullets", []),
            numbered=d.get("numbered", []),
            code=SlideCode.from_dict(code) if code else None,
            equation=d.get("equation", ""),
            table=SlideTable.from_dict(table) if table else None,
            image=SlideImage.from_dict(image) if image else None,
            columns=[SlideColumn.from_dict(c) for c in d.get("columns", [])],
        )


@dataclass
class SlideLayout:
    """MARP layout directives for a slide (content-agnostic)."""
    template: str = ""
    css_class: str = ""                     # _class directive
    paginate: bool = True
    background_color: str = ""
    background_image: str = ""
    bg_position: str = ""                   # left | right | center | full
    bg_size: str = ""                       # contain | cover | 50% | etc.
    bg_filter: str = ""                     # blur | grayscale | etc.
    # v1.1 layout intelligence
    table_class: str = ""                   # "" | "table-small" | "table-tiny"
    inline_image: bool = False              # True → inline HTML, False → ![bg ...]
    image_class: str = ""                   # "" | "figure" | "figure-landscape" | "figure-full"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"paginate": self.paginate}
        if self.template:
            d["template"] = self.template
        if self.css_class:
            d["class"] = self.css_class
        if self.background_color:
            d["background_color"] = self.background_color
        if self.background_image:
            d["background_image"] = self.background_image
        if self.bg_position:
            d["bg_position"] = self.bg_position
        if self.bg_size:
            d["bg_size"] = self.bg_size
        if self.bg_filter:
            d["bg_filter"] = self.bg_filter
        if self.table_class:
            d["table_class"] = self.table_class
        if self.inline_image:
            d["inline_image"] = self.inline_image
        if self.image_class:
            d["image_class"] = self.image_class
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlideLayout:
        return cls(
            template=d.get("template", ""),
            css_class=d.get("class", ""),
            paginate=d.get("paginate", True),
            background_color=d.get("background_color", ""),
            background_image=d.get("background_image", ""),
            bg_position=d.get("bg_position", ""),
            bg_size=d.get("bg_size", ""),
            bg_filter=d.get("bg_filter", ""),
            table_class=d.get("table_class", ""),
            inline_image=d.get("inline_image", False),
            image_class=d.get("image_class", ""),
        )


@dataclass
class Slide:
    """A single slide in the presentation deck."""
    id: str                                 # "slide-NNN"
    type: SlideType
    content: SlideContent
    notes: str = ""                         # Speaker notes
    provenance: Optional[SlideProvenance] = None
    layout: Optional[SlideLayout] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "content": self.content.to_dict(),
        }
        if self.notes:
            d["notes"] = self.notes
        if self.provenance:
            d["provenance"] = self.provenance.to_dict()
        if self.layout:
            d["layout"] = self.layout.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Slide:
        prov = d.get("provenance")
        layout = d.get("layout")
        return cls(
            id=d["id"],
            type=SlideType(d["type"]),
            content=SlideContent.from_dict(d.get("content", {})),
            notes=d.get("notes", ""),
            provenance=SlideProvenance.from_dict(prov) if prov else None,
            layout=SlideLayout.from_dict(layout) if layout else None,
        )


@dataclass
class DeckMetadata:
    """Presentation metadata."""
    title: str
    author: str
    subtitle: str = ""
    organization: str = ""
    date: str = ""
    version: str = ""
    source_folder: str = ""
    generated_by: str = "koas-presenter"
    lang: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "title": self.title,
            "author": self.author,
            "generated_by": self.generated_by,
        }
        if self.subtitle:
            d["subtitle"] = self.subtitle
        if self.organization:
            d["organization"] = self.organization
        if self.date:
            d["date"] = self.date
        if self.version:
            d["version"] = self.version
        if self.source_folder:
            d["source_folder"] = self.source_folder
        if self.lang:
            d["lang"] = self.lang
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DeckMetadata:
        return cls(
            title=d.get("title", "Untitled"),
            author=d.get("author", ""),
            subtitle=d.get("subtitle", ""),
            organization=d.get("organization", ""),
            date=d.get("date", ""),
            version=d.get("version", ""),
            source_folder=d.get("source_folder", ""),
            generated_by=d.get("generated_by", "koas-presenter"),
            lang=d.get("lang", ""),
        )


@dataclass
class DeckTheme:
    """Theme specification for the deck."""
    name: str = "default"
    custom_css: str = ""
    size: str = "16:9"
    math: str = "katex"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "size": self.size, "math": self.math}
        if self.custom_css:
            d["custom_css"] = self.custom_css
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DeckTheme:
        return cls(
            name=d.get("name", "default"),
            custom_css=d.get("custom_css", ""),
            size=d.get("size", "16:9"),
            math=d.get("math", "katex"),
        )


@dataclass
class SlideDeck:
    """
    The central data structure — the clean boundary between content generation
    and layout rendering.

    Everything upstream (S1 + S2) produces this; everything downstream (S3) consumes it.
    Conforms to schema/slide_deck_v1.json.
    """
    metadata: DeckMetadata
    slides: List[Slide]
    theme: DeckTheme = field(default_factory=DeckTheme)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "theme": self.theme.to_dict(),
            "slides": [s.to_dict() for s in self.slides],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SlideDeck:
        return cls(
            metadata=DeckMetadata.from_dict(d.get("metadata", {})),
            theme=DeckTheme.from_dict(d.get("theme", {})),
            slides=[Slide.from_dict(s) for s in d.get("slides", [])],
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for European languages)."""
    return max(1, len(text) // 4)
