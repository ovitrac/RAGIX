"""
Core data models for the KOAS Reviewer kernel family.

All models are JSON-serializable dataclasses used across kernels for
change tracking, edit operations, protected regions, and pyramidal KB.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json
import re


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EditAction(str, Enum):
    """Allowed edit operation actions."""
    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"
    FLAG_ONLY = "flag_only"   # Mark without modifying


class Severity(str, Enum):
    """Issue and edit severity levels."""
    MINOR = "minor"
    ATTENTION = "attention"
    DELETION = "deletion"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """GitHub Alert block types, mapped to severity."""
    NOTE = "NOTE"
    TIP = "TIP"
    WARNING = "WARNING"
    CAUTION = "CAUTION"
    IMPORTANT = "IMPORTANT"


SEVERITY_TO_ALERT: Dict[Severity, AlertType] = {
    Severity.MINOR: AlertType.NOTE,
    Severity.ATTENTION: AlertType.WARNING,
    Severity.DELETION: AlertType.CAUTION,
    Severity.CRITICAL: AlertType.IMPORTANT,
}


class ProtectedKind(str, Enum):
    """Types of protected Markdown regions."""
    CODE_FENCE = "code_fence"
    INLINE_CODE = "inline_code"
    YAML_FRONT_MATTER = "yaml_front_matter"
    HTML_BLOCK = "html_block"
    TABLE = "table"
    MATH_BLOCK = "math_block"
    MATH_INLINE = "math_inline"
    LINK_REF_DEF = "link_ref_def"


# ---------------------------------------------------------------------------
# Change ID
# ---------------------------------------------------------------------------

class ChangeID:
    """
    Stable, zero-padded change identifier.

    Format: RVW-NNNN (canonical) or RVW-<NS>-NNNN (display).
    Revert always uses the canonical form.
    """

    _PREFIX = "RVW"
    _PATTERN = re.compile(r"^RVW-(\d{4,})$")

    def __init__(self, seq: int, namespace: Optional[str] = None):
        self.seq = seq
        self.namespace = namespace

    @property
    def canonical(self) -> str:
        return f"{self._PREFIX}-{self.seq:04d}"

    @property
    def display(self) -> str:
        if self.namespace:
            return f"{self._PREFIX}-{self.namespace}-{self.seq:04d}"
        return self.canonical

    @classmethod
    def parse(cls, s: str) -> "ChangeID":
        """Parse from canonical string 'RVW-NNNN'."""
        m = cls._PATTERN.match(s)
        if not m:
            raise ValueError(f"Invalid change ID: {s!r}")
        return cls(seq=int(m.group(1)))

    def __str__(self) -> str:
        return self.canonical

    def __repr__(self) -> str:
        return f"ChangeID({self.canonical!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ChangeID):
            return self.seq == other.seq
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.seq)

    def __lt__(self, other: "ChangeID") -> bool:
        return self.seq < other.seq


# ---------------------------------------------------------------------------
# Protected Span
# ---------------------------------------------------------------------------

@dataclass
class ProtectedSpan:
    """An immutable region in the Markdown document."""
    kind: ProtectedKind
    line_start: int       # 1-based inclusive
    line_end: int         # 1-based inclusive
    content_hash: str     # SHA-256 of the span text
    info: str = ""        # e.g. language tag for code fences

    def contains_line(self, line: int) -> bool:
        return self.line_start <= line <= self.line_end

    def overlaps(self, start: int, end: int) -> bool:
        return self.line_start <= end and start <= self.line_end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_hash": self.content_hash,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProtectedSpan":
        return cls(
            kind=ProtectedKind(d["kind"]),
            line_start=d["line_start"],
            line_end=d["line_end"],
            content_hash=d["content_hash"],
            info=d.get("info", ""),
        )


# ---------------------------------------------------------------------------
# Heading Node (for structure tree)
# ---------------------------------------------------------------------------

@dataclass
class HeadingNode:
    """A node in the Markdown heading tree."""
    id: str               # Stable ID, e.g. "S2.3"
    level: int            # 1-6
    title: str
    anchor: str           # URL-friendly anchor
    line_start: int       # 1-based
    line_end: int         # 1-based (end of section content)
    numbering: str = ""   # Explicit numbering if detected, e.g. "2.3"
    children: List["HeadingNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "anchor": self.anchor,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "numbering": self.numbering,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeadingNode":
        return cls(
            id=d["id"],
            level=d["level"],
            title=d["title"],
            anchor=d["anchor"],
            line_start=d["line_start"],
            line_end=d["line_end"],
            numbering=d.get("numbering", ""),
            children=[cls.from_dict(c) for c in d.get("children", [])],
        )


# ---------------------------------------------------------------------------
# Review Chunk
# ---------------------------------------------------------------------------

@dataclass
class ReviewChunk:
    """A chunk of Markdown text for review, aligned to document structure."""
    chunk_id: str           # Stable: f"{anchor}_{hash[:8]}"
    section_id: str         # Parent heading ID
    line_start: int         # 1-based inclusive
    line_end: int           # 1-based inclusive
    token_estimate: int
    content_hash: str       # SHA-256 of chunk text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_id": self.section_id,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "token_estimate": self.token_estimate,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReviewChunk":
        return cls(**d)


# ---------------------------------------------------------------------------
# Chunk Fingerprint (v7.2 Content Recipes)
# ---------------------------------------------------------------------------

@dataclass
class ChunkFingerprint:
    """
    Structural fingerprint of a review chunk for content-recipe triggering.

    Pure deterministic features extracted from chunk text, used to decide
    whether content-level masking recipes should be applied before the LLM
    call.  No LLM involved — only regex and Unicode inspection.
    """
    chunk_id: str
    table_rows: int           # total rows across all tables
    table_count: int          # number of distinct tables
    math_count: int           # display + inline math expressions
    emoji_count: int          # emoji code points
    digit_density: float      # digits / total non-whitespace chars
    bullet_count: int         # lines starting with -, *, N.
    blockquote_lines: int     # lines starting with >
    max_line_length: int
    safety_keyword_hits: int
    content_tokens: int
    code_fence_count: int = 0 # fenced code blocks (``` pairs)

    def triggers_recipe(self) -> bool:
        """Trigger if structural complexity exceeds safe thresholds.

        Standalone triggers (any single condition sufficient):
        - Tables with >= 3 data rows
        - Emoji count >= 5 (lowered from 8 based on full-document analysis)
        - Blockquote lines >= 5
        - Math expressions >= 3 (post-masking placeholder density)
        - Code fences >= 1 (Mermaid diagrams, code blocks)

        Combo triggers (lower thresholds when multiple features combine):
        - Math + table (any counts)
        """
        # --- Standalone triggers ---
        # Tables with >= 3 data rows (structural complexity for model decoding)
        if self.table_rows >= 3:
            return True
        # Emoji density (threshold lowered after full-document production run:
        # 5-emoji chunks degenerate at same rate as 8+ with zero false-positive
        # cost since Policy C only fires on already-degenerate chunks)
        if self.emoji_count >= 5:
            return True
        # Blockquote density (structural complexity from nested content)
        if self.blockquote_lines >= 5:
            return True
        # High math density (even after preflight masking, placeholder density
        # can cause degenerate decoding with many [MATH_N] markers)
        if self.math_count >= 3:
            return True
        # Code fences (Mermaid diagrams with data values, code blocks with
        # structured content — cause degenerate decoding even when not counted
        # as tables by the fingerprinter since they're in protected regions)
        if self.code_fence_count >= 1:
            return True
        # --- Combo triggers (lower individual thresholds) ---
        # Math + table combo (any counts)
        if self.math_count > 0 and self.table_count > 0:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "table_rows": self.table_rows,
            "table_count": self.table_count,
            "math_count": self.math_count,
            "emoji_count": self.emoji_count,
            "digit_density": round(self.digit_density, 4),
            "bullet_count": self.bullet_count,
            "blockquote_lines": self.blockquote_lines,
            "max_line_length": self.max_line_length,
            "safety_keyword_hits": self.safety_keyword_hits,
            "content_tokens": self.content_tokens,
            "code_fence_count": self.code_fence_count,
            "triggers_recipe": self.triggers_recipe(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkFingerprint":
        return cls(
            chunk_id=d["chunk_id"],
            table_rows=d["table_rows"],
            table_count=d["table_count"],
            math_count=d["math_count"],
            emoji_count=d["emoji_count"],
            digit_density=d["digit_density"],
            bullet_count=d["bullet_count"],
            blockquote_lines=d["blockquote_lines"],
            max_line_length=d["max_line_length"],
            safety_keyword_hits=d["safety_keyword_hits"],
            content_tokens=d["content_tokens"],
            code_fence_count=d.get("code_fence_count", 0),
        )


# ---------------------------------------------------------------------------
# Review Note
# ---------------------------------------------------------------------------

@dataclass
class ReviewNote:
    """GitHub Alert block injected after attention-requiring edits."""
    alert: AlertType
    text: str               # Must start with "REVIEWER: RVW-NNNN"

    def to_dict(self) -> Dict[str, Any]:
        return {"alert": self.alert.value, "text": self.text}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReviewNote":
        return cls(alert=AlertType(d["alert"]), text=d["text"])

    def validate(self, change_id: str) -> List[str]:
        """Check formatting invariants. Returns list of errors."""
        errors = []
        if not self.text.startswith("REVIEWER:"):
            errors.append("Review note must start with 'REVIEWER:'")
        if change_id not in self.text:
            errors.append(f"Review note must contain change ID {change_id}")
        return errors

    def render(self) -> str:
        """Render as GitHub Alert Markdown block."""
        return f"> [!{self.alert.value}]\n> {self.text}"


# ---------------------------------------------------------------------------
# Edit Operation Target
# ---------------------------------------------------------------------------

@dataclass
class EditTarget:
    """Locator for where an edit applies in the document."""
    chunk_id: str
    anchor: str = ""
    node_path: str = ""     # e.g. "2.3.p5"
    line_start: int = 0
    line_end: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "anchor": self.anchor,
            "node_path": self.node_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EditTarget":
        return cls(
            chunk_id=d["chunk_id"],
            anchor=d.get("anchor", ""),
            node_path=d.get("node_path", ""),
            line_start=d.get("line_start", 0),
            line_end=d.get("line_end", 0),
        )


# ---------------------------------------------------------------------------
# Edit Operation
# ---------------------------------------------------------------------------

@dataclass
class EditOp:
    """
    A single structured edit operation.

    Produced by md_edit_plan (LLM) or deterministic kernels.
    Validated and applied by md_apply_ops.
    """
    id: str                     # "RVW-0034"
    action: EditAction
    target: EditTarget
    before_hash: str            # SHA-256 of text before edit
    before_text: str = ""
    after_text: str = ""
    needs_attention: bool = False
    kind: str = ""              # e.g. "typo", "logic_flow", "structure"
    severity: Severity = Severity.MINOR
    silent: bool = False
    review_note: Optional[ReviewNote] = None
    rationale: str = ""

    def validate(self) -> List[str]:
        """Check structural invariants. Returns list of errors."""
        errors = []
        # Canonical ID format
        try:
            ChangeID.parse(self.id)
        except ValueError as e:
            errors.append(str(e))
        # Deletion requires review note
        if self.action == EditAction.DELETE and self.review_note is None:
            errors.append(f"{self.id}: deletion requires review_note")
        # Attention requires review note
        if self.needs_attention and self.review_note is None:
            errors.append(f"{self.id}: needs_attention=true requires review_note")
        # Review note format
        if self.review_note:
            errors.extend(self.review_note.validate(self.id))
        return errors

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "action": self.action.value,
            "target": self.target.to_dict(),
            "before_hash": self.before_hash,
            "before_text": self.before_text,
            "after_text": self.after_text,
            "needs_attention": self.needs_attention,
            "kind": self.kind,
            "severity": self.severity.value,
            "silent": self.silent,
            "rationale": self.rationale,
        }
        if self.review_note:
            d["review_note"] = self.review_note.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EditOp":
        note = None
        if "review_note" in d and d["review_note"]:
            note = ReviewNote.from_dict(d["review_note"])
        return cls(
            id=d["id"],
            action=EditAction(d["action"]),
            target=EditTarget.from_dict(d["target"]),
            before_hash=d["before_hash"],
            before_text=d.get("before_text", ""),
            after_text=d.get("after_text", ""),
            needs_attention=d.get("needs_attention", False),
            kind=d.get("kind", ""),
            severity=Severity(d.get("severity", "minor")),
            silent=d.get("silent", False),
            review_note=note,
            rationale=d.get("rationale", ""),
        )


# ---------------------------------------------------------------------------
# Ledger Entry
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    """
    One line in the append-only JSONL ledger.

    Records a single applied change (or revert event).
    """
    id: str                     # "RVW-0034" or "REVERT-RVW-0034"
    doc: str                    # Document path
    doc_hash_before: str
    doc_hash_after: str = ""
    timestamp: str = ""         # ISO 8601
    actor: Dict[str, str] = field(default_factory=dict)
    scope: Dict[str, str] = field(default_factory=dict)
    kind: str = ""
    severity: str = "minor"
    silent: bool = False
    summary: str = ""
    rationale: str = ""
    patch_forward: str = ""     # Path to .patch
    patch_inverse: str = ""     # Path to .inverse.patch
    review_note: Optional[Dict[str, str]] = None
    is_revert: bool = False
    reverted_id: str = ""       # If is_revert, the original change ID

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "doc": self.doc,
            "doc_hash_before": self.doc_hash_before,
            "doc_hash_after": self.doc_hash_after,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "scope": self.scope,
            "kind": self.kind,
            "severity": self.severity,
            "silent": self.silent,
            "summary": self.summary,
            "rationale": self.rationale,
            "patch_forward": self.patch_forward,
            "patch_inverse": self.patch_inverse,
            "is_revert": self.is_revert,
            "reverted_id": self.reverted_id,
        }
        if self.review_note:
            d["review_note"] = self.review_note
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LedgerEntry":
        return cls(
            id=d["id"],
            doc=d["doc"],
            doc_hash_before=d["doc_hash_before"],
            doc_hash_after=d.get("doc_hash_after", ""),
            timestamp=d.get("timestamp", ""),
            actor=d.get("actor", {}),
            scope=d.get("scope", {}),
            kind=d.get("kind", ""),
            severity=d.get("severity", "minor"),
            silent=d.get("silent", False),
            summary=d.get("summary", ""),
            rationale=d.get("rationale", ""),
            patch_forward=d.get("patch_forward", ""),
            patch_inverse=d.get("patch_inverse", ""),
            review_note=d.get("review_note"),
            is_revert=d.get("is_revert", False),
            reverted_id=d.get("reverted_id", ""),
        )

    def to_json_line(self) -> str:
        """Serialize as a single JSONL line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


# ---------------------------------------------------------------------------
# Pyramid Node
# ---------------------------------------------------------------------------

@dataclass
class PyramidNode:
    """A node in the single-document hierarchical summary pyramid."""
    node_id: str            # e.g. "S2.3" (matches heading numbering)
    level: int              # 0=abstract, 1=section, 2=subsection, 3=paragraph
    heading: str = ""
    anchor: str = ""
    content_hash: str = ""  # SHA-256 of source text
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "level": self.level,
            "heading": self.heading,
            "anchor": self.anchor,
            "content_hash": self.content_hash,
            "summary": self.summary,
            "keywords": self.keywords,
            "claims": self.claims,
            "dependencies": self.dependencies,
            "children": self.children,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PyramidNode":
        return cls(
            node_id=d["node_id"],
            level=d["level"],
            heading=d.get("heading", ""),
            anchor=d.get("anchor", ""),
            content_hash=d.get("content_hash", ""),
            summary=d.get("summary", ""),
            keywords=d.get("keywords", []),
            claims=d.get("claims", []),
            dependencies=d.get("dependencies", []),
            children=d.get("children", []),
            token_count=d.get("token_count", 0),
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)
