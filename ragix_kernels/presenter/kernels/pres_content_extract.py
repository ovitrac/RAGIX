"""
Kernel: pres_content_extract
Stage: 1 (Collection)

Line-by-line state machine parser that extracts all 13 UnitType values
from Markdown/text documents. Builds cross-document outline tree and
produces a ContentCorpus.

State machine with 9 states:
    NORMAL, CODE_FENCE, MATH_BLOCK, FRONT_MATTER, TABLE,
    BLOCKQUOTE, ADMONITION, BULLET_LIST, NUMBERED_LIST

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.presenter.models import (
    ContentCorpus,
    FileEntry,
    FileType,
    OutlineNode,
    SemanticUnit,
    UnitType,
)
from ragix_kernels.presenter.md_parse_utils import (
    ADMONITION_RE,
    BLOCKQUOTE_RE,
    BULLET_RE,
    CODE_FENCE_RE,
    HEADING_RE,
    IMAGE_REF_RE,
    INLINE_MATH_RE,
    MATH_BLOCK_RE,
    NUMBERED_RE,
    TABLE_SEP_RE,
    YAML_DELIM_RE,
    detect_front_matter,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parser states
# ---------------------------------------------------------------------------

class _State(Enum):
    NORMAL = auto()
    CODE_FENCE = auto()
    MATH_BLOCK = auto()
    FRONT_MATTER = auto()
    TABLE = auto()
    BLOCKQUOTE = auto()
    ADMONITION = auto()
    BULLET_LIST = auto()
    NUMBERED_LIST = auto()


# ---------------------------------------------------------------------------
# Document parser (module-level, testable independently)
# ---------------------------------------------------------------------------

def parse_document(
    text: str,
    source_file: str,
    file_stem: str,
) -> List[SemanticUnit]:
    """
    Parse a Markdown document into a list of SemanticUnit objects.

    Uses a line-by-line state machine to detect all 13 UnitType values.
    Each unit gets an ID of the form "{file_stem}:L{start}-L{end}".

    Args:
        text: Full document text.
        source_file: Relative path from folder root.
        file_stem: Short name for unit ID prefix.

    Returns:
        Ordered list of SemanticUnit objects.
    """
    lines = text.splitlines()
    units: List[SemanticUnit] = []
    heading_stack: List[Tuple[int, str]] = []  # (level, title)

    state = _State.NORMAL
    acc: List[str] = []          # accumulator for multi-line blocks
    acc_start: int = 0           # 1-based start line of current accumulator
    fence_marker: str = ""       # ` or ~ character
    fence_len: int = 0           # length of opening fence
    fence_lang: str = ""         # info string after opening fence

    def _heading_path() -> List[str]:
        return [t for _, t in heading_stack]

    def _depth() -> int:
        return len(heading_stack)

    def _make_unit(
        utype: UnitType,
        content: str,
        start: int,
        end: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticUnit:
        return SemanticUnit(
            id=f"{file_stem}:L{start}-L{end}",
            type=utype,
            content=content,
            source_file=source_file,
            source_lines=(start, end),
            heading_path=_heading_path(),
            depth=_depth(),
            tokens=estimate_tokens(content),
            metadata=metadata or {},
        )

    def _emit_paragraph():
        """Emit accumulated paragraph lines if non-empty."""
        nonlocal acc, acc_start
        if not acc:
            return
        text_block = "\n".join(acc).strip()
        if text_block:
            end_line = acc_start + len(acc) - 1
            # Check for standalone image references
            img_m = IMAGE_REF_RE.match(text_block)
            if img_m and "\n" not in text_block.strip():
                units.append(_make_unit(
                    UnitType.IMAGE_REF, text_block, acc_start, end_line,
                    {"alt": img_m.group(1), "path": img_m.group(2)},
                ))
            else:
                units.append(_make_unit(
                    UnitType.PARAGRAPH, text_block, acc_start, end_line,
                ))
        acc = []
        acc_start = 0

    def _emit_block(utype: UnitType, metadata: Optional[Dict[str, Any]] = None):
        """Emit the accumulated block as a unit of given type."""
        nonlocal acc, acc_start
        if not acc:
            return
        content = "\n".join(acc)
        end_line = acc_start + len(acc) - 1
        units.append(_make_unit(utype, content, acc_start, end_line, metadata))
        acc = []
        acc_start = 0

    # Handle front matter before main loop
    fm = detect_front_matter(lines)
    start_line = 0
    if fm is not None:
        fm_start, fm_end, fm_data = fm
        fm_content = "\n".join(lines[fm_start:fm_end + 1])
        units.append(_make_unit(
            UnitType.FRONT_MATTER, fm_content, 1, fm_end + 1,
            {"keys": list(fm_data.keys()) if isinstance(fm_data, dict) else []},
        ))
        start_line = fm_end + 1

    for i in range(start_line, len(lines)):
        line = lines[i]
        line_num = i + 1  # 1-based
        stripped = line.strip()

        # ---- STATE: CODE_FENCE ----
        if state == _State.CODE_FENCE:
            # Check for closing fence
            cm = CODE_FENCE_RE.match(stripped)
            if cm and cm.group(1)[0] == fence_marker and len(cm.group(1)) >= fence_len and cm.group(2) == "":
                acc.append(line)
                inner = "\n".join(acc[1:-1])  # content between fences
                if fence_lang.lower() in ("mermaid",):
                    # Detect mermaid diagram type from first line
                    diagram_type = ""
                    first_line = inner.strip().split("\n")[0] if inner.strip() else ""
                    if first_line:
                        diagram_type = first_line.split()[0] if first_line.split() else ""
                    _emit_block(UnitType.MERMAID, {"diagram_type": diagram_type})
                else:
                    _emit_block(UnitType.CODE_BLOCK, {"language": fence_lang})
                state = _State.NORMAL
            else:
                acc.append(line)
            continue

        # ---- STATE: MATH_BLOCK ----
        if state == _State.MATH_BLOCK:
            acc.append(line)
            if MATH_BLOCK_RE.match(stripped) and len(acc) > 1:
                # Extract LaTeX content between $$ markers
                latex = "\n".join(acc[1:-1]).strip()
                _emit_block(UnitType.EQUATION_BLOCK, {"latex": latex})
                state = _State.NORMAL
            continue

        # ---- STATE: TABLE ----
        if state == _State.TABLE:
            if "|" in line or TABLE_SEP_RE.match(stripped):
                acc.append(line)
                continue
            else:
                # End of table — compute metadata
                rows = [r for r in acc if not TABLE_SEP_RE.match(r.strip())]
                cols = 0
                if rows:
                    cols = max(r.count("|") - 1 for r in rows) if rows[0].strip().startswith("|") else max(r.count("|") for r in rows)
                    cols = max(1, cols)
                has_header = len(acc) >= 2 and TABLE_SEP_RE.match(acc[1].strip() if len(acc) > 1 else "")
                _emit_block(UnitType.TABLE, {
                    "rows": len(rows),
                    "cols": cols,
                    "has_header": bool(has_header),
                })
                state = _State.NORMAL
                # Reprocess current line
                i_reprocess = True  # handled below

        # ---- STATE: BLOCKQUOTE ----
        if state == _State.BLOCKQUOTE:
            bq_m = BLOCKQUOTE_RE.match(line)
            if bq_m:
                acc.append(bq_m.group(1))
                continue
            else:
                _emit_block(UnitType.BLOCKQUOTE)
                state = _State.NORMAL
                # Reprocess current line

        # ---- STATE: ADMONITION ----
        if state == _State.ADMONITION:
            bq_m = BLOCKQUOTE_RE.match(line)
            if bq_m:
                acc.append(bq_m.group(1))
                continue
            else:
                _emit_block(UnitType.ADMONITION, {"admonition_type": acc[0] if acc else ""})
                state = _State.NORMAL
                # Reprocess current line

        # ---- STATE: BULLET_LIST ----
        if state == _State.BULLET_LIST:
            bl_m = BULLET_RE.match(line)
            if bl_m or (stripped and not stripped.startswith("#") and line.startswith(" ")):
                acc.append(line)
                continue
            elif stripped == "":
                # Blank line might continue the list (check next line)
                # Conservative: end the list
                _emit_block(UnitType.BULLET_LIST)
                state = _State.NORMAL
                continue
            else:
                _emit_block(UnitType.BULLET_LIST)
                state = _State.NORMAL
                # Reprocess current line

        # ---- STATE: NUMBERED_LIST ----
        if state == _State.NUMBERED_LIST:
            nl_m = NUMBERED_RE.match(line)
            if nl_m or (stripped and not stripped.startswith("#") and line.startswith(" ")):
                acc.append(line)
                continue
            elif stripped == "":
                _emit_block(UnitType.NUMBERED_LIST)
                state = _State.NORMAL
                continue
            else:
                _emit_block(UnitType.NUMBERED_LIST)
                state = _State.NORMAL
                # Reprocess current line

        # ---- STATE: NORMAL ----
        if state != _State.NORMAL:
            continue

        # Blank line
        if stripped == "":
            _emit_paragraph()
            continue

        # Heading
        h_m = HEADING_RE.match(line)
        if h_m:
            _emit_paragraph()
            level = len(h_m.group(1))
            title = h_m.group(2).strip()
            # Update heading stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            units.append(_make_unit(
                UnitType.HEADING, title, line_num, line_num,
                {"level": level},
            ))
            continue

        # Code fence
        cf_m = CODE_FENCE_RE.match(stripped)
        if cf_m:
            _emit_paragraph()
            fence_marker = cf_m.group(1)[0]
            fence_len = len(cf_m.group(1))
            fence_lang = cf_m.group(2).strip()
            state = _State.CODE_FENCE
            acc = [line]
            acc_start = line_num
            continue

        # Math block ($$)
        if MATH_BLOCK_RE.match(stripped):
            _emit_paragraph()
            state = _State.MATH_BLOCK
            acc = [line]
            acc_start = line_num
            continue

        # Table: pipe + separator on next line
        if "|" in line and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1].strip()):
            _emit_paragraph()
            state = _State.TABLE
            acc = [line]
            acc_start = line_num
            continue

        # Admonition: > [!TYPE]
        adm_m = ADMONITION_RE.match(line)
        if adm_m:
            _emit_paragraph()
            state = _State.ADMONITION
            adm_type = adm_m.group(1)
            # Extract rest of first line after the admonition marker
            rest = line[adm_m.end():].strip()
            acc = [adm_type]  # First element is the type
            if rest:
                acc.append(rest)
            acc_start = line_num
            continue

        # Blockquote
        bq_m = BLOCKQUOTE_RE.match(line)
        if bq_m:
            _emit_paragraph()
            state = _State.BLOCKQUOTE
            acc = [bq_m.group(1)]
            acc_start = line_num
            continue

        # Bullet list
        bl_m = BULLET_RE.match(line)
        if bl_m:
            _emit_paragraph()
            state = _State.BULLET_LIST
            acc = [line]
            acc_start = line_num
            continue

        # Numbered list
        nl_m = NUMBERED_RE.match(line)
        if nl_m:
            _emit_paragraph()
            state = _State.NUMBERED_LIST
            acc = [line]
            acc_start = line_num
            continue

        # Standalone image reference (only content on the line)
        img_m = IMAGE_REF_RE.match(stripped)
        if img_m and stripped == img_m.group(0):
            _emit_paragraph()
            units.append(_make_unit(
                UnitType.IMAGE_REF, stripped, line_num, line_num,
                {"alt": img_m.group(1), "path": img_m.group(2)},
            ))
            continue

        # Regular text line — accumulate into paragraph
        if not acc:
            acc_start = line_num
        acc.append(line)

    # Flush remaining accumulator
    if state == _State.CODE_FENCE:
        _emit_block(UnitType.CODE_BLOCK, {"language": fence_lang})
    elif state == _State.MATH_BLOCK:
        latex = "\n".join(acc[1:]).strip() if len(acc) > 1 else ""
        _emit_block(UnitType.EQUATION_BLOCK, {"latex": latex})
    elif state == _State.TABLE:
        rows = [r for r in acc if not TABLE_SEP_RE.match(r.strip())]
        cols = 0
        if rows:
            cols = max(r.count("|") - 1 for r in rows) if rows[0].strip().startswith("|") else max(r.count("|") for r in rows)
            cols = max(1, cols)
        _emit_block(UnitType.TABLE, {"rows": len(rows), "cols": cols, "has_header": len(acc) >= 2})
    elif state == _State.BLOCKQUOTE:
        _emit_block(UnitType.BLOCKQUOTE)
    elif state == _State.ADMONITION:
        _emit_block(UnitType.ADMONITION, {"admonition_type": acc[0] if acc else ""})
    elif state == _State.BULLET_LIST:
        _emit_block(UnitType.BULLET_LIST)
    elif state == _State.NUMBERED_LIST:
        _emit_block(UnitType.NUMBERED_LIST)
    else:
        _emit_paragraph()

    return units


# ---------------------------------------------------------------------------
# Outline tree builder
# ---------------------------------------------------------------------------

def build_outline(
    all_units: Dict[str, List[SemanticUnit]],
) -> List[OutlineNode]:
    """
    Build a cross-document outline tree from heading units.

    Args:
        all_units: {source_file: [units]} mapping.

    Returns:
        List of root OutlineNode objects.
    """
    roots: List[OutlineNode] = []
    stack: List[OutlineNode] = []

    for source_file in sorted(all_units.keys()):
        units = all_units[source_file]
        file_stem = Path(source_file).stem
        counters: Dict[int, int] = {}

        for u in units:
            if u.type != UnitType.HEADING:
                continue
            level = u.metadata.get("level", 1)

            # Update counters
            counters[level] = counters.get(level, 0) + 1
            for deeper in list(counters.keys()):
                if deeper > level:
                    del counters[deeper]

            parts = [str(counters.get(lv, 0)) for lv in sorted(counters.keys()) if lv <= level]
            node_id = f"{file_stem}:H{'.'.join(parts)}"

            node = OutlineNode(
                id=node_id,
                level=level,
                title=u.content,
                source_file=source_file,
                line=u.source_lines[0],
            )

            # Nest under parent
            while stack and stack[-1].level >= level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)

    return roots


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class PresContentExtractKernel(Kernel):
    """Markdown/text → SemanticUnit[] with outline tree."""

    name = "pres_content_extract"
    version = "1.0.0"
    category = "presenter"
    stage = 1
    description = "Markdown/text parsing into SemanticUnit list with outline tree"

    requires: List[str] = ["pres_folder_scan"]
    provides: List[str] = ["content_corpus", "outline_tree"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load folder scan output
        scan_path = input.dependencies["pres_folder_scan"]
        scan_data = json.loads(scan_path.read_text(encoding="utf-8"))
        folder_root = Path(scan_data["data"]["root"])
        files = [FileEntry.from_dict(f) for f in scan_data["data"]["files"]]

        # Filter document files
        doc_files = [f for f in files if f.file_type == FileType.DOCUMENT]

        all_units: List[SemanticUnit] = []
        per_file_units: Dict[str, List[SemanticUnit]] = {}
        total_tokens = 0

        for fe in doc_files:
            doc_path = folder_root / fe.path
            if not doc_path.exists():
                logger.warning(f"[pres_content_extract] File not found: {doc_path}")
                continue

            try:
                text = doc_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"[pres_content_extract] Cannot read {doc_path}: {e}")
                continue

            file_stem = Path(fe.path).stem
            units = parse_document(text, fe.path, file_stem)
            per_file_units[fe.path] = units
            all_units.extend(units)
            total_tokens += sum(u.tokens for u in units)

        # Build outline tree
        outline = build_outline(per_file_units)

        corpus = ContentCorpus(
            root=str(folder_root),
            files=files,
            units=all_units,
            outline=outline,
            total_tokens=total_tokens,
            total_files=len(files),
            total_documents=len(doc_files),
        )

        logger.info(
            f"[pres_content_extract] {len(doc_files)} docs → "
            f"{len(all_units)} units, {total_tokens} tokens"
        )

        return corpus.to_dict()

    def summarize(self, data: Dict[str, Any]) -> str:
        n_units = len(data.get("units", []))
        n_docs = data.get("total_documents", 0)
        tokens = data.get("total_tokens", 0)
        return (
            f"Content extraction: {n_units} units from {n_docs} documents "
            f"({tokens} tokens)"
        )
