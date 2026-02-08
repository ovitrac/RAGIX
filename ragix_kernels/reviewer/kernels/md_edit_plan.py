"""
Kernel: md_edit_plan
Stage: 2 (Analysis)

Generates proposed edits per chunk; kernel normalizes them into EditOps using
a multi-pass extraction ladder; writes per-chunk artifacts and a streaming
status log.  Tolerant of any LLM output format (JSON, YAML, plain text).

Worker+Tutor orchestration: Worker proposes, Tutor validates (optional).
Graceful degradation: unparseable ops are salvaged or downgraded to flag_only.
Retry policy: empty responses are retried up to 2 times with jitter.
Timeout: configurable per-chunk timeout (default 300s for large models).

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-06
"""

import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.models import (
    ChunkFingerprint,
    EditOp,
    ProtectedSpan,
    ReviewChunk,
    PyramidNode,
    content_hash,
    estimate_tokens,
)
from ragix_kernels.reviewer.context import (
    assemble_edit_context,
    compute_context_tier,
    render_context_prompt,
)
from ragix_kernels.reviewer.llm_backend import LLMBackend, get_backend
from ragix_kernels.reviewer.prompts import render_prompt

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction ladder — each level returns Optional[List[Dict[str, Any]]]
# ---------------------------------------------------------------------------

def _extract_json_strict(text: str) -> Optional[List[Dict[str, Any]]]:
    """Level 1: Strict JSON array parse."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*\n?", "", text)
    text = text.strip()

    start = text.find("[")
    if start < 0:
        return None

    depth = 0
    end = -1
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end < 0:
        return None

    json_str = text[start:end]
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return None


def _extract_json_relaxed(text: str) -> Optional[List[Dict[str, Any]]]:
    """Level 2: JSON5/relaxed parse (trailing commas, comments, single quotes)."""
    text = re.sub(r"```(?:json5?|javascript)?\s*\n?", "", text)
    text = text.strip()

    start = text.find("[")
    if start < 0:
        # Try finding a single JSON object
        start = text.find("{")
        if start >= 0:
            text = "[" + text[start:] + "]"
            start = 0
        else:
            return None

    depth = 0
    end = -1
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' or ch == "'":
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end < 0:
        return None

    json_str = text[start:end]

    # Fix trailing commas
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    # Fix single quotes to double quotes (outside double-quoted strings)
    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
    # Remove // line comments
    json_str = re.sub(r"//[^\n]*", "", json_str)
    # Remove /* */ block comments
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass
    return None


def _extract_yaml(text: str) -> Optional[List[Dict[str, Any]]]:
    """Level 3: YAML parse."""
    try:
        import yaml
    except ImportError:
        return None

    # Extract from yaml code fences
    m = re.search(r"```yaml\s*\n(.*?)```", text, re.DOTALL)
    payload = m.group(1) if m else text

    try:
        parsed = yaml.safe_load(payload)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass
    return None


def _extract_tagged_payload(text: str) -> Optional[List[Dict[str, Any]]]:
    """Level 4: Extract from BEGIN_EDIT_OPS / END_EDIT_OPS markers or fenced blocks."""
    # Tagged markers
    m = re.search(
        r"BEGIN_EDIT_OPS\s*\n(.*?)\nEND_EDIT_OPS",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not m:
        # Also try <EDIT_OPS>...</EDIT_OPS>
        m = re.search(
            r"<EDIT_OPS>\s*\n?(.*?)\n?</EDIT_OPS>",
            text, re.DOTALL | re.IGNORECASE,
        )
    if not m:
        return None

    payload = m.group(1).strip()
    # Try JSON first, then YAML
    result = _extract_json_strict(payload)
    if result is not None:
        return result
    result = _extract_json_relaxed(payload)
    if result is not None:
        return result
    result = _extract_yaml(payload)
    if result is not None:
        return result
    return None


# Regex patterns for plain-text op parsing
_PLAINTEXT_OP_RE = re.compile(
    r"\[RVW\]\s*"
    r"(?:action\s*[=:]\s*(?P<action>\w+))?"
    r".*?"
    r"(?:lines?\s*[=:]\s*(?P<line_start>\d+)(?:\s*[-–]\s*(?P<line_end>\d+))?)?",
    re.IGNORECASE | re.DOTALL,
)

_REPLACE_PATTERN = re.compile(
    r"(?:replace|change|fix|correct|rewrite)\s*[:\-]?\s*"
    r"(?:[\"'`](.+?)[\"'`])\s*"
    r"(?:with|to|by|->|→|=>)\s*"
    r"(?:[\"'`](.+?)[\"'`])",
    re.IGNORECASE | re.DOTALL,
)

_DELETE_PATTERN = re.compile(
    r"(?:delete|remove|drop|eliminate)\s*[:\-]?\s*(?:[\"'`](.+?)[\"'`])",
    re.IGNORECASE,
)

_FLAG_PATTERN = re.compile(
    r"(?:flag|note|warning|attention|issue)\s*[:\-]?\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def _extract_plaintext_ops(
    text: str, chunk_line_start: int, chunk_line_end: int,
) -> Optional[List[Dict[str, Any]]]:
    """Level 5: Parse structured plain-text patterns into ops."""
    ops = []

    # Try [RVW] blocks
    for m in _PLAINTEXT_OP_RE.finditer(text):
        action = (m.group("action") or "flag_only").lower()
        line_s = int(m.group("line_start") or chunk_line_start)
        line_e = int(m.group("line_end") or line_s)
        op = {
            "action": action if action in ("replace", "insert", "delete", "flag_only") else "flag_only",
            "target": {"line_start": line_s, "line_end": line_e},
            "rationale": m.group(0).strip()[:200],
            "_salvage_method": "plaintext_rvw_block",
        }
        ops.append(op)

    if ops:
        return ops

    # Try replace patterns
    for m in _REPLACE_PATTERN.finditer(text):
        ops.append({
            "action": "replace",
            "target": {"line_start": chunk_line_start, "line_end": chunk_line_end},
            "before_text": m.group(1),
            "after_text": m.group(2),
            "rationale": f"Model suggested replacing \"{m.group(1)[:60]}\"",
            "_salvage_method": "regex_replace",
        })

    # Try delete patterns
    for m in _DELETE_PATTERN.finditer(text):
        ops.append({
            "action": "delete",
            "target": {"line_start": chunk_line_start, "line_end": chunk_line_end},
            "before_text": m.group(1),
            "rationale": f"Model suggested deleting \"{m.group(1)[:60]}\"",
            "_salvage_method": "regex_delete",
        })

    # Try flag patterns
    if not ops:
        for m in _FLAG_PATTERN.finditer(text):
            ops.append({
                "action": "flag_only",
                "target": {"line_start": chunk_line_start, "line_end": chunk_line_end},
                "rationale": m.group(1).strip()[:200],
                "_salvage_method": "regex_flag",
            })

    return ops if ops else None


def _salvage_freeform(
    text: str, chunk_id: str, chunk_line_start: int, chunk_line_end: int,
) -> List[Dict[str, Any]]:
    """Level 6: Last resort — emit a single flag_only from the raw model output."""
    # Check if the model said "no changes needed" or similar
    no_change_patterns = [
        r"no\s+(?:changes?|edits?|issues?|modifications?)\s+(?:needed|required|found|necessary)",
        r"the\s+(?:text|chunk|document)\s+(?:is|looks?|appears?)\s+(?:correct|fine|good|ok)",
        r"aucun(?:e)?\s+(?:modification|changement|erreur|problème)",
        r"\[\s*\]",  # empty JSON array
    ]
    for pat in no_change_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return []  # Legitimate: no changes needed

    # Model produced something but we couldn't parse it — flag for human review
    summary = text.strip()[:300].replace("\n", " ")
    return [{
        "action": "flag_only",
        "target": {
            "chunk_id": chunk_id,
            "line_start": chunk_line_start,
            "line_end": chunk_line_end,
        },
        "rationale": f"LLM output could not be structured; raw summary: {summary}",
        "kind": "unstructured_llm_output",
        "severity": "attention",
        "needs_attention": True,
        "_salvage_method": "freeform_salvage",
    }]


def _run_extraction_ladder(
    raw_text: str, chunk_id: str, chunk_line_start: int, chunk_line_end: int,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Run the multi-pass extraction ladder on raw LLM output.

    Returns:
        (ops_list, extraction_method) — method is the name of the level that succeeded.
        ops_list is None only when raw_text is empty.
    """
    if not raw_text or not raw_text.strip():
        return None, "empty_response"

    # Level 1: Strict JSON
    ops = _extract_json_strict(raw_text)
    if ops is not None:
        return ops, "json_strict"

    # Level 2: Relaxed JSON (JSON5-like)
    ops = _extract_json_relaxed(raw_text)
    if ops is not None:
        return ops, "json_relaxed"

    # Level 3: YAML
    ops = _extract_yaml(raw_text)
    if ops is not None:
        return ops, "yaml"

    # Level 4: Tagged payload (BEGIN_EDIT_OPS / END_EDIT_OPS)
    ops = _extract_tagged_payload(raw_text)
    if ops is not None:
        return ops, "tagged_payload"

    # Level 5: Structured plain-text patterns
    ops = _extract_plaintext_ops(raw_text, chunk_line_start, chunk_line_end)
    if ops is not None:
        return ops, "plaintext_regex"

    # Level 6: Freeform salvage (always produces something or [])
    ops = _salvage_freeform(raw_text, chunk_id, chunk_line_start, chunk_line_end)
    return ops, "freeform_salvage"


# ---------------------------------------------------------------------------
# Op normalization — kernel wraps raw extracted ops into compliant EditOps
# ---------------------------------------------------------------------------

def _normalize_op(
    raw_op: Dict[str, Any],
    chunk: ReviewChunk,
    lines: List[str],
    protected_line_set: set,
    silent_allowlist: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Normalize a raw extracted op into a compliant EditOp dict.

    Kernel-owned responsibilities:
    - Assign missing fields with safe defaults
    - Compute before_hash if missing
    - Enforce deletion policy (always needs attention + review_note)
    - Reject ops touching protected regions
    - Downgrade uncertain ops to flag_only

    Returns None if op should be discarded (e.g. touches protected region).
    """
    action = raw_op.get("action", "flag_only").lower()
    if action not in ("replace", "insert", "delete", "flag_only"):
        action = "flag_only"

    # Target resolution
    target = raw_op.get("target", {})
    line_start = target.get("line_start", chunk.line_start)
    line_end = target.get("line_end", line_start)

    # Clamp to chunk boundaries
    line_start = max(chunk.line_start, min(line_start, chunk.line_end))
    line_end = max(line_start, min(line_end, chunk.line_end))

    # ---- Blank-line targeting guard (v7.4) ----
    # If a replace targets a blank/whitespace-only span but the LLM provided
    # non-trivial before_text, the LLM has an off-by-one.  Search ±3 lines
    # for a line containing the first 40 chars of before_text.  If found,
    # relocate; otherwise discard (the edit would hallucinate content).
    if action == "replace":
        before_text_raw = raw_op.get("before_text", "")
        actual_span = "\n".join(
            lines[line_start - 1:line_end] if line_start >= 1 else []
        )
        if actual_span.strip() == "" and len(before_text_raw.strip()) > 10:
            relocated = False
            needle = before_text_raw.strip()[:40].lower()
            for delta in (1, -1, 2, -2, 3, -3):
                candidate = line_start + delta
                if chunk.line_start <= candidate <= chunk.line_end:
                    cand_text = lines[candidate - 1] if candidate >= 1 else ""
                    if needle in cand_text.lower():
                        line_start = candidate
                        line_end = candidate
                        relocated = True
                        break
            if not relocated:
                # Cannot locate real target — downgrade to flag_only
                action = "flag_only"
                raw_op["_blank_line_guard"] = "downgraded"
            else:
                raw_op["_blank_line_guard"] = f"relocated_delta={delta}"

    # Check protected regions
    if action != "flag_only":
        touches_protected = any(
            ln in protected_line_set for ln in range(line_start, line_end + 1)
        )
        if touches_protected:
            return None

    # Extract text fields
    before_text = raw_op.get("before_text", "")
    after_text = raw_op.get("after_text", "")

    # Compute before_hash if missing
    before_hash = raw_op.get("before_hash", "")
    if not before_hash and before_text:
        h = hashlib.sha256(before_text.encode("utf-8")).hexdigest()
        before_hash = f"sha256:{h}"

    # Kind and severity
    kind = raw_op.get("kind", "style").lower()
    severity = raw_op.get("severity", "minor").lower()

    # Silent flag
    silent = raw_op.get("silent", kind in silent_allowlist)
    needs_attention = raw_op.get("needs_attention", not silent)

    # Rationale
    rationale = raw_op.get("rationale", "")[:200]

    # ----- Deletion policy enforcement (Fix #5) -----
    if action == "delete":
        severity = "deletion"
        needs_attention = True
        silent = False
        # Ensure review_note exists
        review_note = raw_op.get("review_note", {})
        if not review_note or not review_note.get("text"):
            review_note = {
                "alert": "CAUTION",
                "text": f"REVIEWER: Deletion proposed — {rationale[:120]}",
            }
        raw_op["review_note"] = review_note

    # Build normalized op
    op = {
        "action": action,
        "target": {
            "chunk_id": chunk.chunk_id,
            "anchor": chunk.section_id,
            "line_start": line_start,
            "line_end": line_end,
        },
        "before_hash": before_hash,
        "before_text": before_text,
        "after_text": after_text,
        "needs_attention": needs_attention,
        "kind": kind,
        "severity": severity,
        "silent": silent,
        "rationale": rationale,
    }

    # Copy review_note if present
    if "review_note" in raw_op:
        op["review_note"] = raw_op["review_note"]

    # Copy salvage method for audit trail
    if "_salvage_method" in raw_op:
        op["_extraction_method"] = raw_op["_salvage_method"]

    # Copy blank-line guard outcome for audit trail
    if "_blank_line_guard" in raw_op:
        op["_blank_line_guard"] = raw_op["_blank_line_guard"]

    return op


# ---------------------------------------------------------------------------
# Verdict parsing (Tutor response)
# ---------------------------------------------------------------------------

_VERDICT_RE = re.compile(
    r"VERDICT:\s*(ACCEPTED|CORRECTED|REJECTED)",
    re.IGNORECASE,
)
_REASON_RE = re.compile(
    r"REASON:\s*(.+?)(?=\n(?:VERDICT|CORRECTION)|$)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_verdict(response: str) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
    """Parse tutor verdict response."""
    verdict_match = _VERDICT_RE.search(response)
    verdict = verdict_match.group(1).upper() if verdict_match else "REJECTED"

    reason_match = _REASON_RE.search(response)
    reason = reason_match.group(1).strip() if reason_match else ""

    corrected_ops = None
    if verdict in ("CORRECTED", "REJECTED"):
        corrected_ops, _ = _run_extraction_ladder(response, "", 0, 0)

    return verdict, reason, corrected_ops


# ---------------------------------------------------------------------------
# Circuit breaker — rolling failure tracker
# ---------------------------------------------------------------------------

class _CircuitBreaker:
    """Track rolling parse success/failure and trigger mode switches."""

    def __init__(self, window: int = 20, threshold: float = 0.60):
        self.window = window
        self.threshold = threshold
        self._results: List[bool] = []  # True = success, False = failure
        self.tripped = False
        self.trip_count = 0

    def record(self, success: bool) -> None:
        self._results.append(success)
        if len(self._results) > self.window:
            self._results.pop(0)

        if len(self._results) >= self.window:
            fail_rate = 1 - (sum(self._results) / len(self._results))
            if fail_rate >= self.threshold and not self.tripped:
                self.tripped = True
                self.trip_count += 1
                logger.warning(
                    f"[md_edit_plan] Circuit breaker tripped: "
                    f"{fail_rate:.0%} failure rate over last {self.window} chunks. "
                    f"Switching to plain-text prompt mode."
                )

    @property
    def should_use_plain_prompt(self) -> bool:
        return self.tripped


# ---------------------------------------------------------------------------
# Preflight: Math block masking
# ---------------------------------------------------------------------------

_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


def _mask_math_blocks(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace math expressions with inert placeholders to prevent LLM
    degenerate decoding loops triggered by LaTeX content.

    Processes display math ($$...$$) first, then inline ($...$).

    Returns:
        (masked_text, mapping) where mapping is {placeholder: original_math}
    """
    mapping: Dict[str, str] = {}
    counter = 0

    def _replace_display(m: re.Match) -> str:
        nonlocal counter
        key = f"[MATH_{counter}]"
        mapping[key] = m.group(0)
        counter += 1
        return key

    def _replace_inline(m: re.Match) -> str:
        nonlocal counter
        key = f"[MATH_{counter}]"
        mapping[key] = m.group(0)
        counter += 1
        return key

    # Display math first (greedy $$)
    text = _DISPLAY_MATH_RE.sub(_replace_display, text)
    # Then inline math
    text = _INLINE_MATH_RE.sub(_replace_inline, text)

    return text, mapping


def _unmask_ops(
    ops: List[Dict[str, Any]], mapping: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Restore original math content in op text fields after extraction.

    Replaces [MATH_N] placeholders back to original $...$ / $$...$$ content
    in before_text and after_text fields so md_apply_ops matches real content.
    """
    if not mapping:
        return ops
    for op in ops:
        for field in ("before_text", "after_text", "rationale"):
            val = op.get(field, "")
            if val:
                for placeholder, original in mapping.items():
                    val = val.replace(placeholder, original)
                op[field] = val
    return ops


def _has_math_content(text: str) -> bool:
    """Quick check whether text contains LaTeX math expressions."""
    return "$$" in text or _INLINE_MATH_RE.search(text) is not None


# ---------------------------------------------------------------------------
# Content Recipes — v7.2 masking transforms
# ---------------------------------------------------------------------------

# Reuse table patterns from md_fingerprint_chunk
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s\-:]+\|[\s\-:|]+\|?\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|")

# Emoji ranges (consistent with md_fingerprint_chunk)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended (colored circles)
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U00002600-\U000026FF"  # Misc symbols
    "\U000024C2-\U0001F251"  # Enclosed characters
    "]+"
)


def _mask_table_blocks(
    text: str, min_rows: int = 3,
) -> Tuple[str, Dict[str, str]]:
    """
    Replace Markdown pipe-table blocks with [[TABLE_BLOCK_N]] placeholders.

    Only masks tables with >= min_rows data rows (excluding separator).
    Uses double brackets to distinguish from [MATH_N] placeholders.

    Returns:
        (masked_text, mapping) where mapping is {placeholder: original_table}
    """
    mapping: Dict[str, str] = {}
    lines = text.split("\n")
    result_lines: List[str] = []
    counter = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        # Detect table start: line with | and next line is separator
        if _TABLE_ROW_RE.match(line) and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]):
            # Scan the full table block
            table_start = i
            data_rows = 1  # Header row
            i += 1  # Move past header
            # Separator line
            i += 1
            # Count data rows
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                if not _TABLE_SEP_RE.match(lines[i]):
                    data_rows += 1
                i += 1

            table_lines = lines[table_start:i]
            if data_rows >= min_rows:
                key = f"[[TABLE_BLOCK_{counter}]]"
                mapping[key] = "\n".join(table_lines)
                counter += 1
                result_lines.append(key)
            else:
                result_lines.extend(table_lines)
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines), mapping


_CODE_FENCE_RE_EDIT = re.compile(r"^\s*(`{3,}|~{3,})")


def _mask_code_fences(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace fenced code blocks (``` ... ```) with [[CODE_BLOCK_N]] placeholders.

    Targets Mermaid diagrams and other code blocks that cause degenerate
    decoding due to structured data inside fences (pie charts, gantt data).

    Returns:
        (masked_text, mapping) where mapping is {placeholder: original_block}
    """
    mapping: Dict[str, str] = {}
    lines = text.split("\n")
    result_lines: List[str] = []
    counter = 0
    i = 0

    while i < len(lines):
        m = _CODE_FENCE_RE_EDIT.match(lines[i])
        if m:
            fence_char = m.group(1)[0]  # '`' or '~'
            block_start = i
            i += 1
            # Scan to closing fence
            while i < len(lines):
                m2 = _CODE_FENCE_RE_EDIT.match(lines[i])
                if m2 and m2.group(1)[0] == fence_char:
                    i += 1
                    break
                i += 1
            block = lines[block_start:i]
            key = f"[[CODE_BLOCK_{counter}]]"
            mapping[key] = "\n".join(block)
            counter += 1
            result_lines.append(key)
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines), mapping


def _mask_emoji(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace emoji code points with [[EMOJI_N]] placeholders.

    Returns:
        (masked_text, mapping) where mapping is {placeholder: original_emoji}
    """
    mapping: Dict[str, str] = {}
    counter = 0

    def _replace(m: re.Match) -> str:
        nonlocal counter
        key = f"[[EMOJI_{counter}]]"
        mapping[key] = m.group(0)
        counter += 1
        return key

    masked = _EMOJI_RE.sub(_replace, text)
    return masked, mapping


def _truncate_digit_blocks(
    text: str,
    density_threshold: float = 0.25,
    min_lines: int = 5,
    keep_lines: int = 2,
) -> Tuple[str, Dict[str, str]]:
    """
    For paragraph blocks with high digit density, keep first/last lines
    and replace the middle with a [[NUMERIC_TRUNCATED_N]] placeholder.

    Reduces token pressure from numeric-heavy tables/lists that the LLM
    cannot meaningfully review.

    Returns:
        (modified_text, mapping) where mapping is {placeholder: truncated_middle}
    """
    mapping: Dict[str, str] = {}
    counter = 0

    # Split into paragraph blocks (separated by blank lines)
    paragraphs = re.split(r"\n\s*\n", text)
    result_parts: List[str] = []

    for para in paragraphs:
        para_lines = para.split("\n")
        if len(para_lines) < min_lines:
            result_parts.append(para)
            continue

        # Compute digit density for this block
        non_ws = re.sub(r"\s", "", para)
        if not non_ws:
            result_parts.append(para)
            continue

        density = sum(1 for c in non_ws if c.isdigit()) / len(non_ws)
        if density < density_threshold:
            result_parts.append(para)
            continue

        # Truncate: keep first/last keep_lines
        head = para_lines[:keep_lines]
        tail = para_lines[-keep_lines:]
        middle = para_lines[keep_lines:-keep_lines]

        if middle:
            key = f"[[NUMERIC_TRUNCATED_{counter}]]"
            mapping[key] = "\n".join(middle)
            counter += 1
            result_parts.append("\n".join(head + [key] + tail))
        else:
            result_parts.append(para)

    return "\n\n".join(result_parts), mapping


# Safety vocabulary synonym map (reduces model self-censoring)
_SYNONYM_MAP = {
    "vulnérabilité": "faiblesse",
    "vulnérabilités": "faiblesses",
    "attaque": "surface de risque",
    "attaques": "surfaces de risque",
    "exploiter": "détourner",
    "exploit": "détournement",
    "faille": "lacune",
    "failles": "lacunes",
    "injection": "insertion non contrôlée",
    "backdoor": "accès non documenté",
}


def _apply_synonym_map(
    text: str, synonyms: Dict[str, str],
) -> Tuple[str, Dict[str, str]]:
    """
    Word-boundary-aware substitution of safety keywords with neutral synonyms.

    Returns:
        (modified_text, reverse_mapping) where reverse_mapping is
        {synonym: original_word} for restoring in extracted ops.
    """
    reverse_mapping: Dict[str, str] = {}
    modified = text

    for original, replacement in synonyms.items():
        pattern = re.compile(
            r"\b" + re.escape(original) + r"\b",
            re.IGNORECASE,
        )
        if pattern.search(modified):
            modified = pattern.sub(replacement, modified)
            reverse_mapping[replacement] = original

    return modified, reverse_mapping


_BLOCKQUOTE_LINE_RE = re.compile(r"^\s*>")


def _flatten_blockquotes(
    text: str, min_lines: int = 5,
) -> Tuple[str, Dict[str, str]]:
    """
    Replace contiguous blockquote blocks (>= min_lines) with [[QUOTE_BLOCK_N]]
    placeholders.

    Blockquotes are identified by lines starting with ``>``.  Contiguous blocks
    of *min_lines* or more lines are replaced with a single placeholder.
    Shorter blocks are left intact.

    Returns:
        (masked_text, mapping) where mapping is {placeholder: original_blockquote}
    """
    mapping: Dict[str, str] = {}
    lines = text.split("\n")
    result_lines: List[str] = []
    counter = 0
    i = 0

    while i < len(lines):
        if _BLOCKQUOTE_LINE_RE.match(lines[i]):
            # Scan contiguous blockquote block
            block_start = i
            while i < len(lines) and _BLOCKQUOTE_LINE_RE.match(lines[i]):
                i += 1
            block_lines = lines[block_start:i]
            if len(block_lines) >= min_lines:
                key = f"[[QUOTE_BLOCK_{counter}]]"
                mapping[key] = "\n".join(block_lines)
                counter += 1
                result_lines.append(key)
            else:
                result_lines.extend(block_lines)
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines), mapping


_MATH_PLACEHOLDER_RE = re.compile(r"\[MATH_\d+\]")


def _consolidate_math_placeholders(
    text: str, min_placeholders: int = 3,
) -> Tuple[str, Dict[str, str]]:
    """
    Replace paragraph blocks dominated by [MATH_N] placeholders with a single
    [[FORMULA_BLOCK_N]] placeholder.

    After preflight math masking, chunks with many inline/display equations end up
    with high placeholder density (e.g., 10+ [MATH_N] markers in 186 tokens),
    which itself triggers degenerate model decoding.  This recipe consolidates
    consecutive lines containing >= min_placeholders total math markers into a
    single block placeholder, preserving the original text for unmasking.

    A "formula block" is a maximal run of consecutive non-empty lines where
    the combined text contains >= min_placeholders math placeholders.

    Returns:
        (masked_text, mapping) where mapping is {placeholder: original_lines}
    """
    mapping: Dict[str, str] = {}
    lines = text.split("\n")
    result_lines: List[str] = []
    counter = 0
    i = 0

    while i < len(lines):
        # Scan a run of non-empty lines
        if lines[i].strip() and _MATH_PLACEHOLDER_RE.search(lines[i]):
            block_start = i
            while i < len(lines) and lines[i].strip():
                i += 1
            block = lines[block_start:i]
            block_text = "\n".join(block)
            n_math = len(_MATH_PLACEHOLDER_RE.findall(block_text))
            if n_math >= min_placeholders:
                key = f"[[FORMULA_BLOCK_{counter}]]"
                mapping[key] = block_text
                counter += 1
                result_lines.append(key)
            else:
                result_lines.extend(block)
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines), mapping


def _apply_content_recipe(
    chunk_text: str,
    fingerprint: Optional[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, str], List[str]]:
    """
    Apply all triggered masking steps based on fingerprint and config.

    Order: blockquote_flatten → table → emoji → math_consolidate →
           digit truncation → synonym map.
    Merges all mappings into one dict for unified unmasking.

    Args:
        chunk_text: The chunk text to mask
        fingerprint: ChunkFingerprint.to_dict() or None
        cfg: content_recipes config section

    Returns:
        (masked_text, combined_mapping, recipe_names)
        recipe_names lists which recipes were actually applied.
    """
    if not cfg.get("enabled", True):
        return chunk_text, {}, []

    combined_mapping: Dict[str, str] = {}
    recipes: List[str] = []
    text = chunk_text

    # Blockquote flattening (first — exposes embedded content or removes it)
    bq_min = cfg.get("blockquote_min_lines", 5)
    if (cfg.get("blockquote_flatten", True)
            and fingerprint and fingerprint.get("blockquote_lines", 0) >= bq_min):
        text, bq_map = _flatten_blockquotes(text, min_lines=bq_min)
        if bq_map:
            combined_mapping.update(bq_map)
            recipes.append("blockquote_flatten")

    # Code fence masking (Mermaid diagrams, code blocks with structured data)
    if (cfg.get("code_fence_mask", True)
            and fingerprint and fingerprint.get("code_fence_count", 0) > 0):
        text, fence_map = _mask_code_fences(text)
        if fence_map:
            combined_mapping.update(fence_map)
            recipes.append("code_fence_mask")

    # Table masking (any table with >= min_rows data rows)
    min_rows = cfg.get("table_mask_min_rows", 3)
    if fingerprint and fingerprint.get("table_rows", 0) >= min_rows:
        text, tbl_map = _mask_table_blocks(text, min_rows=min_rows)
        if tbl_map:
            combined_mapping.update(tbl_map)
            recipes.append("table_mask")

    # Emoji masking
    if cfg.get("emoji_mask", True) and fingerprint and fingerprint.get("emoji_count", 0) > 0:
        text, emoji_map = _mask_emoji(text)
        if emoji_map:
            combined_mapping.update(emoji_map)
            recipes.append("emoji_mask")

    # Math placeholder consolidation (after preflight masking, high
    # [MATH_N] density still triggers degenerate decoding)
    math_consolidate_min = cfg.get("math_consolidate_min_placeholders", 3)
    if fingerprint and fingerprint.get("math_count", 0) >= math_consolidate_min:
        text, math_map = _consolidate_math_placeholders(
            text, min_placeholders=math_consolidate_min,
        )
        if math_map:
            combined_mapping.update(math_map)
            recipes.append("math_consolidate")

    # Digit truncation
    dt_threshold = cfg.get("digit_truncation_threshold", 0.25)
    dt_min_lines = cfg.get("digit_truncation_min_lines", 5)
    if fingerprint and fingerprint.get("digit_density", 0) >= dt_threshold:
        text, digit_map = _truncate_digit_blocks(
            text, density_threshold=dt_threshold, min_lines=dt_min_lines,
        )
        if digit_map:
            combined_mapping.update(digit_map)
            recipes.append("digit_truncation")

    # Synonym map
    if cfg.get("synonym_map_enabled", True) and fingerprint and fingerprint.get("safety_keyword_hits", 0) > 0:
        text, syn_map = _apply_synonym_map(text, _SYNONYM_MAP)
        if syn_map:
            combined_mapping.update(syn_map)
            recipes.append("synonym_map")

    return text, combined_mapping, recipes


# ---------------------------------------------------------------------------
# Policy D: Plain-text projection (markdown stripping)
# ---------------------------------------------------------------------------

# Regex for placeholder protection during markdown stripping
_PLACEHOLDER_RE = re.compile(r"\[\[[A-Z_]+\d*\]\]")

# Sentinel token that must never appear in real content
_SENTINEL_PREFIX = "\x00PHOLD_"


def _compute_post_mask_density(text: str) -> Dict[str, int]:
    """
    Compute residual markup density on (already masked) text.

    Returns a dict with counts of formatting markers that remain
    after content recipes have been applied. Used to gate Policy D.
    """
    lines = text.split("\n")

    # Count ** markers (pairs of double-asterisk, raw occurrences)
    bold_marker_count = text.count("**")

    # Count backtick characters (excluding those inside [[...]] placeholders)
    backtick_count = _PLACEHOLDER_RE.sub("", text).count("`")

    # Count code spans: `...` (simple greedy match)
    code_span_count = len(re.findall(r"`[^`]+`", _PLACEHOLDER_RE.sub("", text)))

    # Count empty headings: ### (with nothing or only whitespace after)
    empty_heading_count = sum(
        1 for line in lines
        if re.match(r"^\s{0,3}#{1,6}\s*$", line)
    )

    # Count horizontal rules: ---, ***, ___
    hr_count = sum(
        1 for line in lines
        if re.match(r"^\s{0,3}([-*_])\1{2,}\s*$", line)
    )

    # Count blockquote markers still present
    quote_marker_count = sum(
        1 for line in lines
        if re.match(r"^\s{0,3}>\s?", line)
    )

    return {
        "bold_marker_count": bold_marker_count,
        "backtick_count": backtick_count,
        "code_span_count": code_span_count,
        "empty_heading_count": empty_heading_count,
        "hr_count": hr_count,
        "quote_marker_count": quote_marker_count,
    }


def _exceeds_md_strip_threshold(
    density: Dict[str, int],
    cfg: Dict[str, Any],
) -> bool:
    """
    Check if post-mask markup density exceeds Policy D thresholds.

    Returns True if any standalone threshold or the combined threshold
    is exceeded. Thresholds are configurable for reproducibility.
    """
    if density.get("backtick_count", 0) >= cfg.get("backtick_threshold", 12):
        return True
    if density.get("bold_marker_count", 0) >= cfg.get("bold_threshold", 8):
        return True
    if density.get("empty_heading_count", 0) >= cfg.get("empty_heading_threshold", 1):
        return True
    if density.get("hr_count", 0) >= cfg.get("hr_threshold", 1):
        return True
    if density.get("code_span_count", 0) >= cfg.get("code_span_threshold", 8):
        return True

    # Combined threshold
    combined = (
        density.get("backtick_count", 0)
        + density.get("bold_marker_count", 0)
        + density.get("hr_count", 0)
        + density.get("empty_heading_count", 0)
    )
    if combined >= cfg.get("combined_threshold", 20):
        return True

    return False


def _markdown_strip(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Deterministic plain-text projection preserving placeholders and structure.

    Transforms masked Markdown into text that is easy for the model to read,
    while keeping [[PLACEHOLDER_BLOCK_i]] tokens exactly intact, line breaks
    (so anchors still work), and minimal semantic cues.

    Transform order:
        1. Protect [[...]] placeholders with sentinel tokens
        2. Remove horizontal rules (---+, ***+, ___+)
        3. Normalize headings (empty → remove, non-empty → strip # prefix)
        4. Flatten blockquotes (remove > prefix, keep content)
        5. Strip bold/italic markers (keep text)
        6. Strip inline code backticks (keep content)
        7. Normalize list markers to bullet dot
        8. Collapse multiple blank lines to max 1
        9. Restore placeholders from sentinels

    Returns:
        (stripped_text, transform_log) where transform_log records what changed.
    """
    transforms: List[str] = []

    # Step 1: Protect placeholders with sentinels
    placeholders: Dict[str, str] = {}
    counter = 0

    def _protect(m: re.Match) -> str:
        nonlocal counter
        sentinel = f"{_SENTINEL_PREFIX}{counter}\x00"
        placeholders[sentinel] = m.group(0)
        counter += 1
        return sentinel

    text = _PLACEHOLDER_RE.sub(_protect, text)

    # Also protect [MATH_N] single-bracket placeholders
    _math_ph_re = re.compile(r"\[MATH_\d+\]")
    text = _math_ph_re.sub(_protect, text)

    lines = text.split("\n")

    # Step 2: Remove horizontal rules
    hr_re = re.compile(r"^\s{0,3}([-*_])\1{2,}\s*$")
    new_lines = []
    hr_removed = 0
    for line in lines:
        if hr_re.match(line):
            hr_removed += 1
        else:
            new_lines.append(line)
    lines = new_lines
    if hr_removed:
        transforms.append(f"hr_removed={hr_removed}")

    # Step 3: Normalize headings
    heading_re = re.compile(r"^(\s{0,3})(#{1,6})\s*(.*?)\s*$")
    new_lines = []
    headings_stripped = 0
    headings_removed = 0
    for line in lines:
        m = heading_re.match(line)
        if m:
            content = m.group(3)
            if not content:
                headings_removed += 1
                # Remove the empty heading entirely
            else:
                headings_stripped += 1
                new_lines.append(content)
        else:
            new_lines.append(line)
    lines = new_lines
    if headings_stripped:
        transforms.append(f"headings_stripped={headings_stripped}")
    if headings_removed:
        transforms.append(f"empty_headings_removed={headings_removed}")

    # Step 4: Flatten blockquotes (remove > prefix)
    bq_re = re.compile(r"^(\s{0,3})>\s?(.*)$")
    bq_flattened = 0
    new_lines = []
    for line in lines:
        m = bq_re.match(line)
        if m:
            bq_flattened += 1
            new_lines.append(m.group(2))
        else:
            new_lines.append(line)
    lines = new_lines
    if bq_flattened:
        transforms.append(f"bq_flattened={bq_flattened}")

    # Rejoin for text-level transforms
    text = "\n".join(lines)

    # Step 5: Bold/italic stripping (up to 5 passes to handle nesting)
    bold_stripped = 0
    italic_stripped = 0
    for _ in range(5):
        new_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        if new_text != text:
            bold_stripped += text.count("**") - new_text.count("**")
            text = new_text
        else:
            break
    for _ in range(5):
        new_text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
        if new_text != text:
            italic_stripped += 1
            text = new_text
        else:
            break
    if bold_stripped:
        transforms.append(f"bold_stripped={bold_stripped // 2}")
    if italic_stripped:
        transforms.append(f"italic_stripped={italic_stripped}")

    # Step 6: Strip inline code backticks (keep content)
    code_stripped = 0
    new_text = re.sub(r"`([^`]+)`", r"\1", text)
    if new_text != text:
        code_stripped = text.count("`") - new_text.count("`")
        text = new_text
    if code_stripped:
        transforms.append(f"code_spans_stripped={code_stripped // 2}")

    # Step 7: Normalize list markers
    lines = text.split("\n")
    list_re = re.compile(r"^(\s*)([-*+])\s+")
    ordered_re = re.compile(r"^(\s*)\d+\.\s+")
    list_normalized = 0
    new_lines = []
    for line in lines:
        m = list_re.match(line)
        if m:
            list_normalized += 1
            new_lines.append(list_re.sub(r"\g<1>" + "\u2022 ", line))
        else:
            m2 = ordered_re.match(line)
            if m2:
                list_normalized += 1
                new_lines.append(ordered_re.sub(r"\g<1>" + "\u2022 ", line))
            else:
                new_lines.append(line)
    lines = new_lines
    if list_normalized:
        transforms.append(f"list_normalized={list_normalized}")

    # Step 8: Collapse multiple blank lines to max 1
    collapsed = 0
    new_lines = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                new_lines.append(line)
            else:
                collapsed += 1
        else:
            blank_run = 0
            new_lines.append(line)
    lines = new_lines
    if collapsed:
        transforms.append(f"blank_lines_collapsed={collapsed}")

    text = "\n".join(lines)

    # Step 9: Restore placeholders from sentinels
    for sentinel, original in placeholders.items():
        text = text.replace(sentinel, original)

    transform_log = {
        "transforms": transforms,
        "transform_count": len(transforms),
    }

    return text, transform_log


def _fixup_ops_for_stripped(
    ops: List[Dict[str, Any]],
    lines: List[str],
    chunk: "ReviewChunk",
) -> List[Dict[str, Any]]:
    """
    Fix up ops extracted from stripped text so they anchor to original.

    Since ops use line-number anchoring, the primary fix is replacing
    before_text (if present) with the actual original content from those
    lines, so before_hash verification passes in md_apply_ops.
    """
    for op in ops:
        target = op.get("target", {})
        ls = target.get("line_start", chunk.line_start)
        le = target.get("line_end", ls)
        # Clamp to valid range
        ls = max(0, min(ls - 1, len(lines) - 1))
        le = max(ls, min(le, len(lines)))
        original_text = "\n".join(lines[ls:le])
        if "before_text" in op and op["before_text"]:
            op["before_text"] = original_text
            # Recompute hash from original content
            h = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
            op["before_hash"] = f"sha256:{h}"
    return ops


# ---------------------------------------------------------------------------
# Preflight: Dynamic sub-chunk splitting
# ---------------------------------------------------------------------------

def _maybe_split_chunk(
    chunk: ReviewChunk,
    lines: List[str],
    protected_spans: List[ProtectedSpan],
    max_tokens: int = 900,
) -> List[ReviewChunk]:
    """
    Split an oversized chunk at a paragraph boundary.

    Returns [chunk] unchanged if within budget, or 2 sub-chunks with
    {chunk_id}_a / _b suffixes. Never splits inside protected spans.
    Single split level (no recursion).

    Args:
        chunk: The chunk to potentially split
        lines: All document lines (0-indexed list)
        protected_spans: Protected regions to avoid splitting inside
        max_tokens: Token threshold for splitting (default 900)

    Returns:
        List of 1 or 2 ReviewChunk instances
    """
    if chunk.token_estimate <= max_tokens:
        return [chunk]

    start_0 = chunk.line_start - 1  # Convert to 0-based
    end_0 = chunk.line_end           # Exclusive for slicing

    # Find paragraph breaks (0-based indices where paragraphs begin)
    breaks = []
    for i in range(start_0 + 1, end_0):
        if i > 0 and lines[i - 1].strip() == "" and lines[i].strip() != "":
            breaks.append(i)

    if not breaks:
        return [chunk]  # No paragraph breaks — cannot split

    # Build protected line set for this chunk
    protected_lines = set()
    for span in protected_spans:
        if span.overlaps(chunk.line_start, chunk.line_end):
            for ln in range(span.line_start, span.line_end + 1):
                protected_lines.add(ln - 1)  # Convert to 0-based

    # Find best split point near midpoint
    midpoint = (start_0 + end_0) // 2
    valid_breaks = []
    for b in breaks:
        # Check that the split point is not inside a protected span
        if b not in protected_lines and (b - 1) not in protected_lines:
            valid_breaks.append(b)

    if not valid_breaks:
        return [chunk]  # All breaks inside protected spans

    # Choose break closest to midpoint
    split_at = min(valid_breaks, key=lambda b: abs(b - midpoint))

    # Build sub-chunks (1-based line numbers)
    text_a = "\n".join(lines[start_0:split_at])
    text_b = "\n".join(lines[split_at:end_0])

    chunk_a = ReviewChunk(
        chunk_id=f"{chunk.chunk_id}_a",
        section_id=chunk.section_id,
        line_start=chunk.line_start,
        line_end=split_at,  # 1-based: split_at is 0-based start of b
        token_estimate=estimate_tokens(text_a),
        content_hash=content_hash(text_a),
    )
    chunk_b = ReviewChunk(
        chunk_id=f"{chunk.chunk_id}_b",
        section_id=chunk.section_id,
        line_start=split_at + 1,
        line_end=chunk.line_end,
        token_estimate=estimate_tokens(text_b),
        content_hash=content_hash(text_b),
    )
    logger.info(
        f"[preflight] Split {chunk.chunk_id} ({chunk.token_estimate}t) → "
        f"{chunk_a.chunk_id} ({chunk_a.token_estimate}t) + "
        f"{chunk_b.chunk_id} ({chunk_b.token_estimate}t) at line {split_at + 1}"
    )
    return [chunk_a, chunk_b]


# ---------------------------------------------------------------------------
# Empty response classification
# ---------------------------------------------------------------------------

def _classify_empty(envelope: Dict[str, Any]) -> str:
    """
    Classify an empty LLM response into a precise diagnostic bucket.

    Buckets (ordered by specificity):
      - http_body_empty:              HTTP body was 0 bytes (infra/protocol)
      - json_decode_fail:             Body non-empty but not valid JSON (stream mismatch)
      - eval_count_zero:              Valid JSON, 0 tokens generated (model stall/abort)
      - decoded_empty_nonzero_eval:   Tokens generated but response="" (decoder/tokenizer quirk)
      - json_response_empty:          Generic: valid JSON, response="" (unknown cause)
      - no_envelope:                  No envelope data available
    """
    if not envelope:
        return "no_envelope"

    http_bytes = envelope.get("_http_bytes", -1)

    if http_bytes == 0:
        return "http_body_empty"

    if envelope.get("_json_decode_error"):
        return "json_decode_fail"

    eval_count = envelope.get("eval_count", -1)
    response_text = envelope.get("response", "")
    is_empty = not response_text or not response_text.strip()

    if is_empty and eval_count == 0:
        return "eval_count_zero"

    if is_empty and eval_count > 0:
        # Model generated tokens but decoder output is empty —
        # prompt degeneracy or special-token emission
        return "decoded_empty_nonzero_eval"

    if is_empty:
        return "json_response_empty"

    # Response has content but caller still considers it "empty" — shouldn't happen
    return "json_response_empty"


def _is_degenerate(stats: Dict[str, Any]) -> bool:
    """Check if chunk result is decoded-empty saturation (model emitted tokens but nothing visible)."""
    return stats.get("extraction_method") == "decoded_empty_saturated"


# ---------------------------------------------------------------------------
# Single chunk processing (core LLM call + extraction)
# ---------------------------------------------------------------------------

def _process_single_chunk(
    chunk: ReviewChunk,
    chunk_text: str,
    lines: List[str],
    pyramid_nodes: Dict[str, PyramidNode],
    issues: List[Dict[str, Any]],
    style_rules_text: str,
    worker_backend: LLMBackend,
    tutor_backend: Optional[LLMBackend],
    silent_allowlist: List[str],
    protected_line_set: set,
    budget_tokens: int = 3500,
    use_plain_prompt: bool = False,
    chunk_timeout: int = 300,
    max_retries: int = 2,
    tier: Optional[int] = None,
    math_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Core LLM call + extraction for a single (possibly sub-)chunk.

    Called by _process_chunk after preflight guards have been applied.
    The chunk_text is pre-extracted and may have math blocks masked.

    Args:
        chunk: ReviewChunk metadata (may be a sub-chunk with _a/_b suffix)
        chunk_text: Pre-extracted text (possibly math-masked)
        lines: All document lines
        pyramid_nodes: Pyramid knowledge base
        issues: All issues from consistency scan
        style_rules_text: Rendered style rules
        worker_backend: LLM backend for Worker
        tutor_backend: Optional LLM backend for Tutor
        silent_allowlist: Kinds that don't need review notes
        protected_line_set: Line numbers in protected regions
        budget_tokens: Token budget for context assembly
        use_plain_prompt: Use plain-text prompt template
        chunk_timeout: Per-request timeout in seconds
        max_retries: Max retries on empty_response
        tier: Context tier (0-3) for context assembly
        math_mapping: Math placeholder mapping for unmasking ops

    Returns:
        (list of normalized ops, stats dict, raw_response_text)
    """
    t0 = time.monotonic()
    stats = {
        "chunk_id": chunk.chunk_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "worker_calls": 0,
        "tutor_calls": 0,
        "verdict": "none",
        "ops_proposed": 0,
        "ops_accepted": 0,
        "extraction_method": "none",
        "parse_success": False,
        "latency_ms": 0,
        "errors": [],
        "t_request_ms": 0,
        "retries": 0,
    }

    raw_response = ""

    # Assemble context using pyramid (with tier cap)
    context_parts = assemble_edit_context(
        chunk_text=chunk_text,
        chunk=chunk,
        pyramid_nodes=pyramid_nodes,
        issues=issues,
        style_rules=style_rules_text,
        budget_tokens=budget_tokens,
        tier=tier,
    )

    abstract = ""
    section_summary = ""
    sibling_summaries = []
    for label, content in context_parts:
        if label == "abstract":
            abstract = content
        elif label == "section_summary":
            section_summary = content
        elif label == "sibling":
            sibling_summaries.append(content)

    chunk_issues = [
        i for i in issues
        if chunk.line_start <= i.get("line", 0) <= chunk.line_end
    ]
    issues_for_template = [
        {"line": i.get("line", 0), "type": i.get("type", ""), "evidence": i.get("evidence", "")[:80]}
        for i in chunk_issues[:5]
    ]

    silent_list_str = ", ".join(f'"{s}"' for s in silent_allowlist)

    # ----- Choose prompt template -----
    template_name = "review_plain.j2" if use_plain_prompt else "review.j2"

    # Pass tier to template for conditional section rendering
    effective_tier = tier if tier is not None else 3

    worker_prompt = render_prompt(
        template_name,
        abstract=abstract or "(No document overview available)",
        section_summary=section_summary,
        sibling_summaries=sibling_summaries,
        issues=issues_for_template,
        style_rules=style_rules_text,
        section_heading=chunk.section_id,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        chunk_text=chunk_text,
        chunk_id=chunk.chunk_id,
        anchor=chunk.section_id,
        silent_allowlist=silent_list_str,
        tier=effective_tier,
    )

    # ----- WORKER: propose edit ops (with conditional retry) -----
    num_predict = 2048
    attempt = 0
    t_request_start = time.monotonic()
    base_temperature = 0.10
    envelope = {}
    current_prompt = worker_prompt

    while attempt <= max_retries:
        temp = base_temperature + (attempt * 0.05)

        try:
            raw_response = worker_backend.call(
                prompt=current_prompt,
                temperature=temp,
                num_predict=num_predict,
                timeout=chunk_timeout,
                streaming=True,
            )
            envelope = worker_backend.last_envelope
            stats["worker_calls"] += 1
        except Exception as e:
            logger.warning(f"[md_edit_plan] Worker LLM failed for {chunk.chunk_id}: {e}")
            stats["errors"].append(f"worker_llm_error: {str(e)[:100]}")
            stats["t_request_ms"] = int((time.monotonic() - t_request_start) * 1000)
            stats["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return [], stats, raw_response

        # Check if response is non-empty
        if raw_response and raw_response.strip():
            break

        # Classify the empty response
        empty_kind = _classify_empty(envelope)
        tokens_seen = envelope.get("_tokens_seen", 0)
        eval_count = envelope.get("eval_count", tokens_seen)
        done_reason = envelope.get("done_reason", "")
        stats.setdefault("empty_kinds", []).append(empty_kind)

        # Decoded-empty-saturated: retries never help
        if (empty_kind == "decoded_empty_nonzero_eval"
                and eval_count >= int(num_predict * 0.95)
                and done_reason == "length"):
            stats["retries"] = attempt
            logger.info(
                f"[md_edit_plan] Decoded-empty saturated for {chunk.chunk_id} "
                f"(eval={eval_count}, vis=0, no markers) — skipping retries"
            )
            break

        attempt += 1
        stats["retries"] = attempt
        if attempt <= max_retries:
            jitter = random.uniform(3.0, 8.0)
            logger.info(
                f"[md_edit_plan] Empty response for {chunk.chunk_id} "
                f"({empty_kind}), retry {attempt}/{max_retries} "
                f"(temp={temp+0.05:.2f}) after {jitter:.1f}s"
            )
            time.sleep(jitter)
        else:
            logger.warning(
                f"[md_edit_plan] Empty after {max_retries} retries for "
                f"{chunk.chunk_id} ({empty_kind})"
            )

    stats["t_request_ms"] = int((time.monotonic() - t_request_start) * 1000)

    # Capture envelope diagnostics for status.jsonl
    stats["ollama_eval_count"] = envelope.get("eval_count", envelope.get("_tokens_seen", -1))
    stats["ollama_prompt_eval_count"] = envelope.get("prompt_eval_count", -1)
    stats["ollama_done"] = envelope.get("done", None)
    stats["ollama_done_reason"] = envelope.get("done_reason", "")
    stats["http_bytes"] = envelope.get("_http_bytes", -1)
    # Streaming diagnostics
    if envelope.get("_tokens_seen") is not None:
        stats["tokens_seen"] = envelope["_tokens_seen"]
        stats["visible_chars"] = envelope.get("_visible_chars", 0)
        stats["t_first_visible"] = envelope.get("_t_first_visible", -1)
        stats["begin_marker_seen"] = envelope.get("_begin_seen", False)
        stats["end_marker_seen"] = envelope.get("_end_seen", False)
    if envelope.get("_short_circuit"):
        stats["short_circuit"] = envelope["_short_circuit"]

    # ----- Extraction ladder -----
    extracted_ops, method = _run_extraction_ladder(
        raw_response, chunk.chunk_id, chunk.line_start, chunk.line_end,
    )
    stats["extraction_method"] = method

    if extracted_ops is None:
        empty_kind = _classify_empty(envelope)
        stats["empty_kind"] = empty_kind

        eval_count = envelope.get("eval_count", envelope.get("_tokens_seen", 0))
        done_reason = envelope.get("done_reason", "")
        is_saturated = (
            empty_kind == "decoded_empty_nonzero_eval"
            and eval_count >= int(num_predict * 0.95)
            and done_reason == "length"
        )

        if not chunk_issues:
            stats["parse_success"] = True
            stats["verdict"] = "empty_noop_clean"
            stats["extraction_method"] = (
                "decoded_empty_saturated" if is_saturated else "empty_as_noop"
            )
            stats["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return [], stats, raw_response
        else:
            stats["parse_success"] = True
            stats["verdict"] = "empty_as_flag_with_issues"
            stats["extraction_method"] = (
                "decoded_empty_saturated" if is_saturated else "empty_as_flag"
            )
            flag_op = {
                "action": "flag_only",
                "target": {
                    "chunk_id": chunk.chunk_id,
                    "anchor": chunk.section_id,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                },
                "before_text": "",
                "after_text": "",
                "needs_attention": True,
                "kind": "structure",
                "severity": "attention",
                "silent": False,
                "rationale": (
                    f"LLM returned empty decoded output (eval_count={eval_count}). "
                    f"{len(chunk_issues)} known issue(s) in this span. Manual check recommended."
                ),
                "review_note": {
                    "alert": "WARNING",
                    "text": (
                        f"REVIEWER: LLM generated {eval_count} tokens but decoded to empty. "
                        f"Chunk has {len(chunk_issues)} flagged issue(s). Please review manually."
                    ),
                },
            }
            stats["ops_proposed"] = 1
            stats["ops_accepted"] = 1
            stats["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return [flag_op], stats, raw_response

    stats["parse_success"] = method not in ("empty_response", "freeform_salvage")
    stats["ops_proposed"] = len(extracted_ops)

    if not extracted_ops:
        stats["verdict"] = "empty"
        stats["parse_success"] = True
        stats["latency_ms"] = int((time.monotonic() - t0) * 1000)
        return [], stats, raw_response

    # ----- Unmask math in extracted ops -----
    if math_mapping:
        extracted_ops = _unmask_ops(extracted_ops, math_mapping)

    # ----- Normalize ops -----
    normalized_ops = []
    for raw_op in extracted_ops:
        norm = _normalize_op(raw_op, chunk, lines, protected_line_set, silent_allowlist)
        if norm is not None:
            normalized_ops.append(norm)

    if not normalized_ops:
        stats["verdict"] = "filtered"
        stats["latency_ms"] = int((time.monotonic() - t0) * 1000)
        return [], stats, raw_response

    # ----- TUTOR: validate (if configured) -----
    if tutor_backend is not None:
        tutor_prompt = render_prompt(
            "tutor_validate.j2",
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            chunk_text=chunk_text,
            proposed_ops_json=json.dumps(normalized_ops, indent=2),
        )

        try:
            tutor_response = tutor_backend.call(
                prompt=tutor_prompt,
                temperature=0.0,
                num_predict=3000,
            )
            stats["tutor_calls"] = 1
        except Exception as e:
            logger.warning(f"[md_edit_plan] Tutor LLM failed for {chunk.chunk_id}: {e}")
            stats["verdict"] = "tutor_error"
            stats["errors"].append(f"tutor_llm_error: {str(e)[:100]}")
            stats["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return [], stats, raw_response

        verdict, reason, corrected_ops = _parse_verdict(tutor_response)
        stats["verdict"] = verdict.lower()

        if verdict == "ACCEPTED":
            valid_ops = normalized_ops
        elif verdict == "CORRECTED" and corrected_ops:
            valid_ops = corrected_ops
        elif verdict == "REJECTED":
            valid_ops = []
            stats["errors"].append(f"rejected: {reason[:100]}")
        else:
            valid_ops = normalized_ops
            stats["verdict"] = "unknown_verdict_passthrough"
    else:
        valid_ops = normalized_ops
        stats["verdict"] = "single_model"

    stats["ops_accepted"] = len(valid_ops)
    stats["latency_ms"] = int((time.monotonic() - t0) * 1000)
    return valid_ops, stats, raw_response


# ---------------------------------------------------------------------------
# Preflight pipeline wrapper
# ---------------------------------------------------------------------------

def _write_degenerate_exemplar(
    exemplars_dir: Path,
    chunk_id: str,
    chunk_text: str,
    fingerprint: Optional[Dict[str, Any]],
    stats: Dict[str, Any],
    adaptive_log: List[Dict[str, Any]],
    recipes_applied: Optional[List[str]] = None,
) -> None:
    """Snapshot degenerate chunk for regression suite."""
    exemplars_dir.mkdir(parents=True, exist_ok=True)
    exemplar = {
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "fingerprint": fingerprint,
        "extraction_method": stats.get("extraction_method", "none"),
        "tier_history": adaptive_log,
        "recipes_applied": recipes_applied or [],
        "model": stats.get("model", ""),
        "ollama_eval_count": stats.get("ollama_eval_count", -1),
    }
    path = exemplars_dir / f"{chunk_id}.json"
    path.write_text(
        json.dumps(exemplar, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info(f"[exemplar] Wrote degenerate exemplar: {path.name}")


def _write_recipe_artifacts(
    masks_dir: Path,
    chunk_id: str,
    combined_mapping: Dict[str, str],
    recipes: List[str],
    tokens_before: int,
    tokens_after: int,
    recovered: Optional[bool] = None,
    ops_count: Optional[int] = None,
    extraction_method: Optional[str] = None,
) -> None:
    """Save content-recipe artifacts for traceability."""
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Save each table block as a separate .md file
    for key, val in combined_mapping.items():
        if key.startswith("[[TABLE_BLOCK_"):
            idx = key.strip("[]").split("_")[-1]
            (masks_dir / f"{chunk_id}_TABLE_{idx}.md").write_text(
                val, encoding="utf-8",
            )

    # Save each quote block as a separate .md file
    for key, val in combined_mapping.items():
        if key.startswith("[[QUOTE_BLOCK_"):
            idx = key.strip("[]").split("_")[-1]
            (masks_dir / f"{chunk_id}_QUOTE_{idx}.md").write_text(
                val, encoding="utf-8",
            )

    # Save each formula block as a separate .md file
    for key, val in combined_mapping.items():
        if key.startswith("[[FORMULA_BLOCK_"):
            idx = key.strip("[]").split("_")[-1]
            (masks_dir / f"{chunk_id}_FORMULA_{idx}.md").write_text(
                val, encoding="utf-8",
            )

    # Save each code block as a separate .md file
    for key, val in combined_mapping.items():
        if key.startswith("[[CODE_BLOCK_"):
            idx = key.strip("[]").split("_")[-1]
            (masks_dir / f"{chunk_id}_CODE_{idx}.md").write_text(
                val, encoding="utf-8",
            )

    # Save emoji mapping if present
    emoji_map = {k: v for k, v in combined_mapping.items() if k.startswith("[[EMOJI_")}
    if emoji_map:
        (masks_dir / f"{chunk_id}_EMOJI.json").write_text(
            json.dumps(emoji_map, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    # Save full recipe log with effectiveness metrics
    reduction_pct = round((1 - tokens_after / max(1, tokens_before)) * 100, 1)
    recipe_log: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "recipes": recipes,
        "mapping_keys": list(combined_mapping.keys()),
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "reduction_pct": reduction_pct,
    }
    if recovered is not None:
        recipe_log["recovered"] = recovered
    if ops_count is not None:
        recipe_log["ops_count"] = ops_count
    if extraction_method is not None:
        recipe_log["extraction_method"] = extraction_method
    (masks_dir / f"{chunk_id}_recipe.json").write_text(
        json.dumps(recipe_log, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _process_chunk(
    chunk: ReviewChunk,
    lines: List[str],
    pyramid_nodes: Dict[str, PyramidNode],
    issues: List[Dict[str, Any]],
    style_rules_text: str,
    worker_backend: LLMBackend,
    tutor_backend: Optional[LLMBackend],
    silent_allowlist: List[str],
    protected_line_set: set,
    budget_tokens: int = 3500,
    use_plain_prompt: bool = False,
    chunk_timeout: int = 300,
    max_retries: int = 2,
    context_tier_default: int = 1,
    context_tier_max: int = 2,
    max_edit_chunk_tokens: int = 900,
    math_masking: bool = True,
    protected_spans: Optional[List[ProtectedSpan]] = None,
    fingerprints: Optional[Dict[str, Dict[str, Any]]] = None,
    content_recipe_cfg: Optional[Dict[str, Any]] = None,
    md_strip_cfg: Optional[Dict[str, Any]] = None,
    masks_dir: Optional[Path] = None,
    exemplars_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Preflight-guarded chunk processing pipeline.

    Applies three deterministic guards before the LLM call:
    1. Math masking — replace $...$ / $$...$$ with [MATH_N] placeholders
    2. Sub-chunk splitting — split oversized chunks at paragraph boundaries
    3. Context tiering — downgrade context richness for large chunks

    Adaptive tier escalation policy (v7.1):
    - Policy A: degenerate at tier>0 → retry at tier=0
    - Policy B: ops=0 + known issues → escalate tier

    Content recipe policy (v7.2):
    - Policy C: degenerate at tier=0 + fingerprint triggers → apply content
      recipes (table/emoji masking, synonym map) and rerun once

    Markdown stripping policy (v7.3.1):
    - Policy D: degenerate after recipe + high markup density → strip
      residual formatting (bold, backticks, headings, rules) and retry

    Then delegates to _process_single_chunk for LLM call + extraction.

    Returns:
        (list of normalized ops, stats dict, raw_response_text)
    """
    # Extract chunk text
    start = chunk.line_start - 1
    end = chunk.line_end
    chunk_text = "\n".join(lines[start:end])

    # --- Preflight 1: Math masking ---
    math_mapping: Optional[Dict[str, str]] = None
    math_masked = False
    if math_masking and _has_math_content(chunk_text):
        chunk_text, math_mapping = _mask_math_blocks(chunk_text)
        math_masked = bool(math_mapping)
        if math_masked:
            logger.info(
                f"[preflight] Masked {len(math_mapping)} math block(s) "
                f"in {chunk.chunk_id}"
            )

    # --- Preflight 2: Sub-chunk splitting ---
    sub_chunks = _maybe_split_chunk(
        chunk, lines, protected_spans or [], max_tokens=max_edit_chunk_tokens,
    )
    was_split = len(sub_chunks) > 1

    if was_split:
        # Process each sub-chunk independently, merge results
        all_ops: List[Dict[str, Any]] = []
        all_raw: List[str] = []
        merged_stats: Dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "worker_calls": 0,
            "tutor_calls": 0,
            "verdict": "none",
            "ops_proposed": 0,
            "ops_accepted": 0,
            "extraction_method": "none",
            "parse_success": False,
            "latency_ms": 0,
            "errors": [],
            "t_request_ms": 0,
            "retries": 0,
            "preflight_tier": context_tier_default,
            "preflight_math_masked": math_masked,
            "preflight_split": True,
            "sub_chunks": [],
        }

        any_success = False
        t0_split = time.monotonic()

        for sub in sub_chunks:
            sub_start = sub.line_start - 1
            sub_end = sub.line_end
            sub_text = "\n".join(lines[sub_start:sub_end])

            # Apply math mask to sub-chunk text if needed
            sub_math_mapping: Optional[Dict[str, str]] = None
            if math_masking and _has_math_content(sub_text):
                sub_text, sub_math_mapping = _mask_math_blocks(sub_text)

            # Compute tier for sub-chunk
            sub_tier = compute_context_tier(
                sub.token_estimate, budget_tokens, context_tier_default,
            )

            # Sub-chunk issues for adaptive escalation
            sub_issues = [
                i for i in issues
                if sub.line_start <= i.get("line", 0) <= sub.line_end
            ]

            _sub_kwargs = dict(
                chunk=sub,
                chunk_text=sub_text,
                lines=lines,
                pyramid_nodes=pyramid_nodes,
                issues=issues,
                style_rules_text=style_rules_text,
                worker_backend=worker_backend,
                tutor_backend=tutor_backend,
                silent_allowlist=silent_allowlist,
                protected_line_set=protected_line_set,
                budget_tokens=budget_tokens,
                use_plain_prompt=use_plain_prompt,
                chunk_timeout=chunk_timeout,
                max_retries=max_retries,
                math_mapping=sub_math_mapping,
            )

            ops, stats, raw = _process_single_chunk(tier=sub_tier, **_sub_kwargs)

            # Adaptive tier for sub-chunks
            sub_adaptive: List[Dict[str, Any]] = []

            # Policy A: degenerate → retry at tier=0
            if _is_degenerate(stats) and sub_tier > 0:
                sub_adaptive.append({"from": sub_tier, "to": 0, "reason": "degenerate_fallback"})
                logger.info(
                    f"[adaptive] sub-chunk {sub.chunk_id} degenerate at tier={sub_tier}, "
                    f"retrying at tier=0"
                )
                ops, stats, raw = _process_single_chunk(tier=0, **_sub_kwargs)

            # Policy B: ops=0 + issues → escalate
            if (not _is_degenerate(stats) and not ops and sub_issues
                    and sub_tier < context_tier_max):
                esc = min(sub_tier + 1, context_tier_max)
                feasible = compute_context_tier(
                    sub.token_estimate, budget_tokens, esc,
                )
                if feasible >= esc:
                    sub_adaptive.append({
                        "from": sub_tier, "to": esc, "reason": "ops_escalation",
                        "issues": len(sub_issues),
                    })
                    logger.info(
                        f"[adaptive] sub-chunk {sub.chunk_id} ops=0 with {len(sub_issues)} issues, "
                        f"escalating tier={sub_tier} → {esc}"
                    )
                    ops, stats, raw = _process_single_chunk(tier=esc, **_sub_kwargs)

            # Policy C (v7.2): degenerate at tier=0 + fingerprint triggers
            sub_effective_tier = sub_adaptive[-1]["to"] if sub_adaptive else sub_tier
            sub_fingerprint = (fingerprints or {}).get(chunk.chunk_id)  # parent chunk fp
            if (_is_degenerate(stats) and sub_effective_tier == 0
                    and sub_fingerprint and sub_fingerprint.get("triggers_recipe")
                    and content_recipe_cfg and content_recipe_cfg.get("enabled", True)):
                masked_sub, sub_recipe_map, sub_recipes = _apply_content_recipe(
                    sub_text, sub_fingerprint, content_recipe_cfg,
                )
                if sub_recipes:
                    sub_combined = {**(sub_math_mapping or {}), **sub_recipe_map}
                    sub_adaptive.append({
                        "from": 0, "to": 0, "reason": "content_recipe",
                        "recipes": sub_recipes,
                    })
                    logger.info(
                        f"[adaptive] sub-chunk {sub.chunk_id} applying content recipe: {sub_recipes}"
                    )
                    _recipe_sub_kwargs = dict(_sub_kwargs)
                    _recipe_sub_kwargs["chunk_text"] = masked_sub
                    _recipe_sub_kwargs["math_mapping"] = sub_combined
                    ops, stats, raw = _process_single_chunk(tier=0, **_recipe_sub_kwargs)

                    # Policy D for sub-chunks: md strip after recipe failure
                    if (_is_degenerate(stats) and md_strip_cfg
                            and md_strip_cfg.get("enabled", True)):
                        sub_density = _compute_post_mask_density(masked_sub)
                        if _exceeds_md_strip_threshold(sub_density, md_strip_cfg):
                            stripped_sub, sub_strip_log = _markdown_strip(masked_sub)
                            sub_adaptive.append({
                                "from": 0, "to": 0, "reason": "md_strip",
                                "density": sub_density,
                                "transforms": sub_strip_log.get("transforms", []),
                            })
                            logger.info(
                                f"[adaptive] sub-chunk {sub.chunk_id} Policy D: md_strip"
                            )
                            _strip_sub_kwargs = dict(_sub_kwargs)
                            _strip_sub_kwargs["chunk_text"] = stripped_sub
                            _strip_sub_kwargs["math_mapping"] = sub_combined
                            ops, stats, raw = _process_single_chunk(
                                tier=0, **_strip_sub_kwargs,
                            )
                            if ops:
                                ops = _fixup_ops_for_stripped(ops, lines, sub)

            final_sub_tier = sub_adaptive[-1]["to"] if sub_adaptive else sub_tier

            all_ops.extend(ops)
            all_raw.append(raw)
            merged_stats["worker_calls"] += stats.get("worker_calls", 0)
            merged_stats["tutor_calls"] += stats.get("tutor_calls", 0)
            merged_stats["ops_proposed"] += stats.get("ops_proposed", 0)
            merged_stats["ops_accepted"] += stats.get("ops_accepted", 0)
            merged_stats["t_request_ms"] += stats.get("t_request_ms", 0)
            merged_stats["retries"] += stats.get("retries", 0)
            merged_stats["errors"].extend(stats.get("errors", []))
            merged_stats["sub_chunks"].append({
                "chunk_id": sub.chunk_id,
                "tier": final_sub_tier,
                "parse_success": stats.get("parse_success", False),
                "extraction_method": stats.get("extraction_method", "none"),
                "ops": len(ops),
                "adaptive": sub_adaptive if sub_adaptive else None,
            })
            if stats.get("parse_success"):
                any_success = True

        merged_stats["parse_success"] = any_success
        merged_stats["ops_accepted"] = len(all_ops)
        merged_stats["extraction_method"] = "split_merged"
        merged_stats["verdict"] = "split_merged"
        merged_stats["latency_ms"] = int((time.monotonic() - t0_split) * 1000)

        return all_ops, merged_stats, "\n---\n".join(all_raw)

    # --- Preflight 3: Context tiering (single chunk) ---
    tier = compute_context_tier(
        chunk.token_estimate, budget_tokens, context_tier_default,
    )
    tier_downgraded = tier < context_tier_default

    if tier_downgraded:
        logger.info(
            f"[preflight] Tier downgrade for {chunk.chunk_id}: "
            f"{context_tier_default} → {tier} "
            f"(chunk={chunk.token_estimate}t, budget={budget_tokens}t)"
        )

    # Chunk-scoped issues for adaptive escalation
    chunk_issues = [
        i for i in issues
        if chunk.line_start <= i.get("line", 0) <= chunk.line_end
    ]

    # Common kwargs for _process_single_chunk
    _single_kwargs = dict(
        chunk=chunk,
        chunk_text=chunk_text,
        lines=lines,
        pyramid_nodes=pyramid_nodes,
        issues=issues,
        style_rules_text=style_rules_text,
        worker_backend=worker_backend,
        tutor_backend=tutor_backend,
        silent_allowlist=silent_allowlist,
        protected_line_set=protected_line_set,
        budget_tokens=budget_tokens,
        use_plain_prompt=use_plain_prompt,
        chunk_timeout=chunk_timeout,
        max_retries=max_retries,
        math_mapping=math_mapping,
    )

    # --- First attempt at computed tier ---
    ops, stats, raw = _process_single_chunk(tier=tier, **_single_kwargs)

    # --- Adaptive tier escalation ---
    adaptive_log: List[Dict[str, Any]] = []

    effective_tier = tier  # tracks the tier actually used for the latest attempt

    # Policy A: degenerate at tier>0 → retry at tier=0 (skeleton only)
    if _is_degenerate(stats) and effective_tier > 0:
        adaptive_log.append({"from": effective_tier, "to": 0, "reason": "degenerate_fallback"})
        logger.info(
            f"[adaptive] {chunk.chunk_id} degenerate at tier={effective_tier}, "
            f"retrying at tier=0 (skeleton)"
        )
        ops, stats, raw = _process_single_chunk(tier=0, **_single_kwargs)
        effective_tier = 0

    # Policy B: parsed OK, ops=0, chunk has known issues → try richer context
    if (not _is_degenerate(stats) and not ops and chunk_issues
            and effective_tier < context_tier_max):
        esc_tier = min(effective_tier + 1, context_tier_max)
        # Only escalate if budget can sustain the higher tier
        feasible = compute_context_tier(
            chunk.token_estimate, budget_tokens, esc_tier,
        )
        if feasible >= esc_tier:
            adaptive_log.append({
                "from": effective_tier, "to": esc_tier, "reason": "ops_escalation",
                "issues": len(chunk_issues),
            })
            logger.info(
                f"[adaptive] {chunk.chunk_id} ops=0 with {len(chunk_issues)} issues, "
                f"escalating tier={effective_tier} → {esc_tier}"
            )
            ops, stats, raw = _process_single_chunk(tier=esc_tier, **_single_kwargs)
            effective_tier = esc_tier

    # Policy C (v7.2): degenerate at tier=0 + fingerprint triggers → content recipe
    recipes_applied: Optional[List[str]] = None
    masked_text: Optional[str] = None
    combined_mapping: Optional[Dict[str, str]] = None
    fingerprint = (fingerprints or {}).get(chunk.chunk_id)
    if (_is_degenerate(stats) and effective_tier == 0
            and fingerprint is not None and fingerprint.get("triggers_recipe")
            and content_recipe_cfg and content_recipe_cfg.get("enabled", True)):
        masked_text, recipe_mapping, recipes = _apply_content_recipe(
            chunk_text, fingerprint, content_recipe_cfg,
        )
        if recipes:
            recipes_applied = recipes
            # Merge math mapping + recipe mapping
            combined_mapping = {**(math_mapping or {}), **recipe_mapping}
            adaptive_log.append({
                "from": 0, "to": 0, "reason": "content_recipe",
                "recipes": recipes,
            })
            token_before = estimate_tokens(chunk_text)
            token_after = estimate_tokens(masked_text)
            logger.info(
                f"[adaptive] {chunk.chunk_id} applying content recipe: {recipes} "
                f"(tokens {token_before} → {token_after})"
            )

            # Rerun with masked text
            _recipe_kwargs = dict(_single_kwargs)
            _recipe_kwargs["chunk_text"] = masked_text
            _recipe_kwargs["math_mapping"] = combined_mapping
            ops, stats, raw = _process_single_chunk(tier=0, **_recipe_kwargs)
            effective_tier = 0

            # Save recipe artifacts with post-rerun effectiveness metrics
            if masks_dir:
                _write_recipe_artifacts(
                    masks_dir, chunk.chunk_id, recipe_mapping,
                    recipes, token_before, token_after,
                    recovered=not _is_degenerate(stats),
                    ops_count=len(ops),
                    extraction_method=stats.get("extraction_method"),
                )

    # Policy D (v7.3.1): degenerate after recipe + high markup density → md strip
    md_strip_applied = False
    if (_is_degenerate(stats) and recipes_applied
            and md_strip_cfg and md_strip_cfg.get("enabled", True)):
        # Compute residual markup density on the masked text (or chunk text as fallback)
        recipe_text = masked_text if masked_text is not None else chunk_text
        density = _compute_post_mask_density(recipe_text)
        if _exceeds_md_strip_threshold(density, md_strip_cfg):
            stripped_text, strip_log = _markdown_strip(recipe_text)
            md_strip_applied = True
            adaptive_log.append({
                "from": 0, "to": 0, "reason": "md_strip",
                "density": density,
                "transforms": strip_log.get("transforms", []),
            })
            token_stripped = estimate_tokens(stripped_text)
            logger.info(
                f"[adaptive] {chunk.chunk_id} Policy D: md_strip "
                f"(density={density}, transforms={strip_log['transforms']}, "
                f"tokens → {token_stripped})"
            )

            # Rerun with stripped text at tier=0
            _strip_kwargs = dict(_single_kwargs)
            _strip_kwargs["chunk_text"] = stripped_text
            # Keep combined mapping from recipe phase for unmasking
            if combined_mapping:
                _strip_kwargs["math_mapping"] = combined_mapping
            ops, stats, raw = _process_single_chunk(tier=0, **_strip_kwargs)
            effective_tier = 0

            # Fix up ops: replace before_text with original content
            if ops:
                ops = _fixup_ops_for_stripped(ops, lines, chunk)

            # Write MDSTRIP artifacts
            if masks_dir:
                masks_dir.mkdir(parents=True, exist_ok=True)
                # Save stripped text sent to LLM
                (masks_dir / f"{chunk.chunk_id}_MDSTRIP_0.txt").write_text(
                    stripped_text, encoding="utf-8",
                )
                # Update recipe log with Policy D info
                strip_recipe_log = {
                    "chunk_id": chunk.chunk_id,
                    "policy": "D",
                    "density": density,
                    "transforms": strip_log.get("transforms", []),
                    "tokens_masked": estimate_tokens(recipe_text),
                    "tokens_stripped": token_stripped,
                    "recovered": not _is_degenerate(stats),
                    "ops_count": len(ops),
                    "extraction_method": stats.get("extraction_method"),
                }
                (masks_dir / f"{chunk.chunk_id}_MDSTRIP_0.recipe.json").write_text(
                    json.dumps(strip_recipe_log, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    # Write degenerate exemplar if still degenerate after all policies
    if _is_degenerate(stats) and exemplars_dir:
        _write_degenerate_exemplar(
            exemplars_dir, chunk.chunk_id, chunk_text,
            fingerprint, stats, adaptive_log, recipes_applied,
        )

    # Inject preflight metadata into stats
    stats["preflight_tier"] = tier
    stats["preflight_tier_downgraded"] = tier_downgraded
    stats["preflight_math_masked"] = math_masked
    stats["preflight_split"] = False
    if recipes_applied:
        stats["content_recipe"] = recipes_applied
    if md_strip_applied:
        stats["md_strip"] = True
    if adaptive_log:
        stats["adaptive_escalation"] = adaptive_log
        # Record final tier actually used
        stats["preflight_tier"] = adaptive_log[-1]["to"]

    return ops, stats, raw


def _regenerate_chunk(
    chunk: ReviewChunk,
    chunk_text: str,
    issues: List[Dict[str, Any]],
    style_rules_text: str,
    backend: LLMBackend,
    chunk_id: str,
    anchor: str,
    rejection_reason: str,
) -> Optional[List[Dict[str, Any]]]:
    """Regenerate ops after rejection using tutor model with conservative prompt."""
    issues_for_template = [
        {"line": i.get("line", 0), "type": i.get("type", ""), "evidence": i.get("evidence", "")[:80]}
        for i in issues[:5]
    ]

    regen_prompt = render_prompt(
        "regenerate.j2",
        rejection_reason=rejection_reason,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        section_heading=chunk.section_id,
        chunk_text=chunk_text,
        issues=issues_for_template,
        style_rules=style_rules_text,
        chunk_id=chunk_id,
        anchor=anchor,
    )

    try:
        response = backend.call(
            prompt=regen_prompt,
            temperature=0.05,
            num_predict=2048,
        )
    except Exception as e:
        logger.warning(f"[md_edit_plan] Regeneration LLM failed for {chunk_id}: {e}")
        return None

    result, _ = _run_extraction_ladder(response, chunk_id, chunk.line_start, chunk.line_end)
    return result


# ---------------------------------------------------------------------------
# Per-chunk I/O — intermediate results (Fix #1)
# ---------------------------------------------------------------------------

def _write_chunk_artifacts(
    ops_dir: Path,
    status_path: Path,
    chunk_id: str,
    raw_response: str,
    ops: List[Dict[str, Any]],
    stats: Dict[str, Any],
    model_name: str,
) -> None:
    """Write per-chunk artifacts immediately after processing."""
    ops_dir.mkdir(parents=True, exist_ok=True)

    # Always write raw LLM output
    (ops_dir / f"ops_{chunk_id}.raw.txt").write_text(raw_response, encoding="utf-8")

    # Write canonical ops (if any)
    if ops:
        (ops_dir / f"ops_{chunk_id}.json").write_text(
            json.dumps({"chunk_id": chunk_id, "ops": ops}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Append to status.jsonl (with full Ollama diagnostics)
    status_line = {
        "chunk_id": chunk_id,
        "ts": stats.get("ts", datetime.now(timezone.utc).isoformat()),
        "model": model_name,
        "parse": "OK" if stats.get("parse_success") else "FAIL",
        "extraction_method": stats.get("extraction_method", "none"),
        "ops": len(ops),
        "errors": stats.get("errors", []),
        "latency_ms": stats.get("latency_ms", 0),
        "t_request_ms": stats.get("t_request_ms", 0),
        "retries": stats.get("retries", 0),
        # Ollama envelope diagnostics
        "http_bytes": stats.get("http_bytes", -1),
        "ollama_eval_count": stats.get("ollama_eval_count", -1),
        "ollama_prompt_eval_count": stats.get("ollama_prompt_eval_count", -1),
        "ollama_done": stats.get("ollama_done"),
        "ollama_done_reason": stats.get("ollama_done_reason", ""),
    }
    if stats.get("empty_kind"):
        status_line["empty_kind"] = stats["empty_kind"]
    if stats.get("empty_kinds"):
        status_line["empty_kinds"] = stats["empty_kinds"]
    # Streaming diagnostics
    if "tokens_seen" in stats:
        status_line["tokens_seen"] = stats["tokens_seen"]
        status_line["visible_chars"] = stats.get("visible_chars", 0)
        status_line["t_first_visible"] = stats.get("t_first_visible", -1)
        status_line["begin_marker_seen"] = stats.get("begin_marker_seen", False)
        status_line["end_marker_seen"] = stats.get("end_marker_seen", False)
    if stats.get("short_circuit"):
        status_line["short_circuit"] = stats["short_circuit"]
    # Preflight diagnostics (v7)
    if "preflight_tier" in stats:
        status_line["preflight_tier"] = stats["preflight_tier"]
    if stats.get("preflight_tier_downgraded"):
        status_line["preflight_tier_downgraded"] = True
    if stats.get("preflight_math_masked"):
        status_line["preflight_math_masked"] = True
    if stats.get("preflight_split"):
        status_line["preflight_split"] = True
        if "sub_chunks" in stats:
            status_line["sub_chunks"] = stats["sub_chunks"]
    # Content recipe diagnostics (v7.2)
    if stats.get("content_recipe"):
        status_line["content_recipe"] = stats["content_recipe"]
    # Policy D: markdown stripping diagnostics (v7.3.1)
    if stats.get("md_strip"):
        status_line["md_strip"] = True
    # Adaptive escalation log (v7.1+)
    if stats.get("adaptive_escalation"):
        status_line["adaptive_escalation"] = stats["adaptive_escalation"]
    with open(status_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(status_line, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class MdEditPlanKernel(Kernel):
    """
    LLM-driven edit plan generation with extraction ladder and streaming output.

    Worker (fast/sovereign model) proposes edits per chunk in any format.
    Kernel normalizes them into EditOps using a 6-level extraction ladder.
    Writes per-chunk artifacts (raw + canonical + status) immediately.
    Circuit breaker switches to plain-text prompt when failure rate is high.
    """

    name = "md_edit_plan"
    version = "2.4.0"
    category = "reviewer"
    stage = 2
    description = "LLM-driven edit ops per chunk with tolerant extraction and content recipes"

    requires: List[str] = ["md_chunk", "md_consistency_scan", "md_pyramid"]
    provides: List[str] = ["edit_plan"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load document
        snapshot_path = input.workspace / "stage1" / "doc.raw.md"
        text = snapshot_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Load chunks
        chunk_path = input.dependencies.get("md_chunk")
        chunks: List[ReviewChunk] = []
        if chunk_path and chunk_path.exists():
            chunk_data = json.loads(chunk_path.read_text())["data"]
            chunks = [ReviewChunk.from_dict(d) for d in chunk_data["chunks"]]

        # Load pyramid nodes
        pyramid_nodes: Dict[str, PyramidNode] = {}
        pyramid_path = input.workspace / "stage2" / "pyramid.json"
        if pyramid_path.exists():
            try:
                pyr_data = json.loads(pyramid_path.read_text())
                for nd in pyr_data.get("nodes", []):
                    node = PyramidNode.from_dict(nd)
                    pyramid_nodes[node.node_id] = node
            except Exception:
                pass

        # Load issues from consistency scan
        issues: List[Dict[str, Any]] = []
        issues_path = input.dependencies.get("md_consistency_scan")
        if issues_path and issues_path.exists():
            try:
                issues_data = json.loads(issues_path.read_text())["data"]
                issues = issues_data.get("issues", [])
            except Exception:
                pass

        # Load protected regions for filtering
        protected_line_set: set = set()
        protected_spans: List[ProtectedSpan] = []
        prot_path = input.workspace / "stage1" / "md_protected_regions.json"
        if prot_path.exists():
            try:
                prot_data = json.loads(prot_path.read_text())["data"]
                for span in prot_data.get("protected_spans", []):
                    for ln in range(span["line_start"], span["line_end"] + 1):
                        protected_line_set.add(ln)
                    protected_spans.append(ProtectedSpan.from_dict(span))
            except Exception:
                pass

        # Load fingerprints (v7.2 — optional dependency, graceful skip)
        fingerprints: Dict[str, Dict[str, Any]] = {}
        fp_path = input.workspace / "stage2" / "md_fingerprint_chunk.json"
        if fp_path.exists():
            try:
                fp_data = json.loads(fp_path.read_text())["data"]
                for fp in fp_data.get("fingerprints", []):
                    fingerprints[fp["chunk_id"]] = fp
                logger.info(
                    f"[md_edit_plan] Loaded {len(fingerprints)} chunk fingerprints "
                    f"({sum(1 for f in fingerprints.values() if f.get('triggers_recipe'))} trigger recipes)"
                )
            except Exception as e:
                logger.warning(f"[md_edit_plan] Could not load fingerprints: {e}")

        # Config
        reviewer_cfg = input.config.get("reviewer", {})
        llm_cfg = reviewer_cfg.get("llm", {})
        style_cfg = reviewer_cfg.get("style", {})

        no_llm = reviewer_cfg.get("no_llm", False)
        if no_llm:
            logger.info("[md_edit_plan] no_llm=true, skipping LLM edit plan")
            return self._empty_result()

        # Initialize LLM backends
        worker_backend = get_backend(
            backend=llm_cfg.get("backend", "ollama"),
            model=llm_cfg.get("edit_model", "mistral:instruct"),
            endpoint=llm_cfg.get("endpoint", "http://127.0.0.1:11434"),
            strict_sovereign=llm_cfg.get("strict_sovereign", True),
        )

        tutor_model = llm_cfg.get("tutor_model")
        tutor_backend = None
        if tutor_model:
            tutor_backend = get_backend(
                backend=llm_cfg.get("backend", "ollama"),
                model=tutor_model,
                endpoint=llm_cfg.get("endpoint", "http://127.0.0.1:11434"),
                strict_sovereign=llm_cfg.get("strict_sovereign", True),
            )

        # Style rules text
        silent_allowlist = style_cfg.get(
            "silent_allowlist",
            ["typo", "punctuation", "capitalization", "spacing", "grammar"],
        )
        style_rules_text = self._build_style_rules(style_cfg, silent_allowlist)

        # Budget
        budget_tokens = reviewer_cfg.get("context_budget_tokens", 3500)

        # Preflight config (v7)
        context_tier_default = reviewer_cfg.get("context_tier_default", 1)
        context_tier_max = reviewer_cfg.get("context_tier_max", 2)
        max_edit_chunk_tokens = reviewer_cfg.get("max_edit_chunk_tokens", 900)
        math_masking = reviewer_cfg.get("math_masking", True)

        # Content recipe config (v7.2)
        content_recipe_cfg = reviewer_cfg.get("content_recipes", {
            "enabled": True,
            "blockquote_flatten": True,
            "blockquote_min_lines": 5,
            "table_mask_min_rows": 2,
            "emoji_mask": True,
            "math_consolidate_min_placeholders": 3,
            "digit_truncation_threshold": 0.25,
            "digit_truncation_min_lines": 5,
            "synonym_map_enabled": True,
        })
        content_recipes_enabled = content_recipe_cfg.get("enabled", True) and bool(fingerprints)

        # Policy D: markdown stripping config (v7.3.1)
        md_strip_cfg = reviewer_cfg.get("md_strip", {
            "enabled": True,
            "backtick_threshold": 12,
            "bold_threshold": 8,
            "empty_heading_threshold": 1,
            "hr_threshold": 1,
            "code_span_threshold": 8,
            "combined_threshold": 20,
        })

        # Timeout and retry policy (large models need > 120s)
        chunk_timeout = llm_cfg.get("chunk_timeout", 300)
        max_retries = llm_cfg.get("empty_retries", 2)
        logger.info(
            f"[md_edit_plan] Chunk timeout: {chunk_timeout}s, "
            f"empty retries: {max_retries}, "
            f"tier_default: {context_tier_default}, tier_max: {context_tier_max}, "
            f"max_chunk_tokens: {max_edit_chunk_tokens}, "
            f"math_masking: {math_masking}, "
            f"content_recipes: {content_recipes_enabled}, "
            f"md_strip: {md_strip_cfg.get('enabled', True) if content_recipes_enabled else False}"
        )

        # Output directories
        stage2_dir = input.workspace / "stage2"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        ops_dir = stage2_dir / "ops"
        ops_dir.mkdir(parents=True, exist_ok=True)
        status_path = stage2_dir / "ops" / "status.jsonl"
        masks_dir = stage2_dir / "masks" if content_recipes_enabled else None
        exemplars_dir = stage2_dir / "exemplars"

        # Model name for audit trail
        model_name = llm_cfg.get("edit_model", "unknown")

        # Circuit breaker
        breaker = _CircuitBreaker(
            window=reviewer_cfg.get("circuit_breaker_window", 20),
            threshold=reviewer_cfg.get("circuit_breaker_threshold", 0.60),
        )

        # Process chunks
        all_ops: List[Dict[str, Any]] = []
        all_stats: List[Dict[str, Any]] = []
        id_counter = 1

        # Get starting ID from existing ledger
        ledger_path = input.workspace / "review" / "ledger.jsonl"
        if ledger_path.exists():
            from ragix_kernels.reviewer.ledger import Ledger
            ledger = Ledger(ledger_path)
            id_counter = ledger.next_seq

        for i, chunk in enumerate(chunks):
            ops, stats, raw_response = _process_chunk(
                chunk=chunk,
                lines=lines,
                pyramid_nodes=pyramid_nodes,
                issues=issues,
                style_rules_text=style_rules_text,
                worker_backend=worker_backend,
                tutor_backend=tutor_backend,
                silent_allowlist=silent_allowlist,
                protected_line_set=protected_line_set,
                budget_tokens=budget_tokens,
                use_plain_prompt=breaker.should_use_plain_prompt,
                chunk_timeout=chunk_timeout,
                max_retries=max_retries,
                context_tier_default=context_tier_default,
                context_tier_max=context_tier_max,
                max_edit_chunk_tokens=max_edit_chunk_tokens,
                math_masking=math_masking,
                protected_spans=protected_spans,
                fingerprints=fingerprints if content_recipes_enabled else None,
                content_recipe_cfg=content_recipe_cfg if content_recipes_enabled else None,
                md_strip_cfg=md_strip_cfg if content_recipes_enabled else None,
                masks_dir=masks_dir,
                exemplars_dir=exemplars_dir,
            )

            # Kernel-owned ID allocation (Fix #4)
            for op_dict in ops:
                op_dict["id"] = f"RVW-{id_counter:04d}"
                id_counter += 1

            # Update circuit breaker
            breaker.record(stats.get("parse_success", False))

            # Write per-chunk artifacts immediately (Fix #1)
            _write_chunk_artifacts(
                ops_dir, status_path, chunk.chunk_id,
                raw_response, ops, stats, model_name,
            )

            all_ops.extend(ops)
            all_stats.append(stats)

            # Progress logging every 25 chunks
            if (i + 1) % 25 == 0 or i == len(chunks) - 1:
                success = sum(1 for s in all_stats if s.get("parse_success"))
                total = len(all_stats)
                logger.info(
                    f"[md_edit_plan] Progress: {i+1}/{len(chunks)} chunks, "
                    f"{len(all_ops)} ops, {success}/{total} parsed "
                    f"({success/max(1,total):.0%})"
                    f"{' [PLAIN MODE]' if breaker.should_use_plain_prompt else ''}"
                )

        # Save merged edit plan
        plan_data = {
            "ops": all_ops,
            "total_ops": len(all_ops),
            "chunks_processed": len(chunks),
        }
        (stage2_dir / "edit_plan.json").write_text(
            json.dumps(plan_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Compute quality stats
        quality = self._compute_quality(all_stats, breaker)

        logger.info(
            f"[md_edit_plan] Complete: {len(all_ops)} ops from {len(chunks)} chunks. "
            f"Extraction: {quality['extraction_methods']}. "
            f"Circuit breaker trips: {breaker.trip_count}"
        )

        return {
            "ops": all_ops,
            "total_ops": len(all_ops),
            "chunks_processed": len(chunks),
            "quality": quality,
            "per_chunk_stats": all_stats,
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "ops": [],
            "total_ops": 0,
            "chunks_processed": 0,
            "quality": {"acceptance_rate": 0, "verdict_counts": {}, "extraction_methods": {}},
            "per_chunk_stats": [],
        }

    def _build_style_rules(
        self, style_cfg: Dict[str, Any], silent_allowlist: List[str]
    ) -> str:
        """Build human-readable style rules text for the prompt."""
        rules = [
            "Review this text for:",
            "- Typographical errors and grammar issues",
            "- Terminology consistency within the document",
            "- AI-generated leftover phrases (e.g. 'As an AI', 'Certainly!')",
            "- Logical flow and coherence between paragraphs",
            "- Duplicated content",
            "- Broken cross-references",
        ]

        # Language constraint — prevents translation ops on monolingual documents
        lang = style_cfg.get("language")
        if lang:
            lang_label = {"fr": "français", "en": "English", "de": "Deutsch"}.get(
                lang, lang
            )
            rules.append("")
            rules.append(f"IMPORTANT: This document is written in {lang_label}.")
            rules.append(
                f"Do NOT propose translations to another language. "
                f"All corrections must stay in {lang_label}. "
                f"Preserve the original language of every heading, paragraph, and term."
            )

        rules.extend([
            "",
            f"Silent (minor) change types: {', '.join(silent_allowlist)}",
            "These do NOT require review notes.",
            "",
            "All other changes require needs_attention=true and a review_note.",
            "Deletions ALWAYS require a review_note with alert=CAUTION.",
        ])
        return "\n".join(rules)

    def _compute_quality(
        self, stats: List[Dict[str, Any]], breaker: _CircuitBreaker,
    ) -> Dict[str, Any]:
        """Aggregate quality metrics from per-chunk stats."""
        total = len(stats)
        if total == 0:
            return {
                "acceptance_rate": 0, "verdict_counts": {},
                "extraction_methods": {}, "parse_success_rate": 0,
            }

        verdict_counts: Dict[str, int] = {}
        extraction_methods: Dict[str, int] = {}
        tier_distribution: Dict[int, int] = {}
        total_proposed = 0
        total_accepted = 0
        parse_successes = 0
        chunks_split = 0
        math_masked_chunks = 0
        adaptive_fallbacks = 0
        adaptive_escalations = 0
        recipe_applied = 0
        recipe_recovered = 0

        for s in stats:
            v = s.get("verdict", "none")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

            m = s.get("extraction_method", "none")
            extraction_methods[m] = extraction_methods.get(m, 0) + 1

            total_proposed += s.get("ops_proposed", 0)
            total_accepted += s.get("ops_accepted", 0)
            if s.get("parse_success"):
                parse_successes += 1

            # Preflight aggregates (v7)
            t = s.get("preflight_tier")
            if t is not None:
                tier_distribution[t] = tier_distribution.get(t, 0) + 1
            if s.get("preflight_split"):
                chunks_split += 1
            if s.get("preflight_math_masked"):
                math_masked_chunks += 1
            # Adaptive escalation tracking (v7.1)
            for esc in s.get("adaptive_escalation", []):
                if esc.get("reason") == "degenerate_fallback":
                    adaptive_fallbacks += 1
                elif esc.get("reason") == "ops_escalation":
                    adaptive_escalations += 1
                elif esc.get("reason") == "content_recipe":
                    recipe_applied += 1
            # Also count sub-chunk adaptives
            for sc in s.get("sub_chunks", []):
                for esc in (sc.get("adaptive") or []):
                    if esc.get("reason") == "degenerate_fallback":
                        adaptive_fallbacks += 1
                    elif esc.get("reason") == "ops_escalation":
                        adaptive_escalations += 1

            # Content recipe recovery: recipe was applied and chunk is no longer degenerate
            if s.get("content_recipe") and not _is_degenerate(s):
                recipe_recovered += 1

        acceptance_rate = total_accepted / max(1, total_proposed)

        return {
            "acceptance_rate": round(acceptance_rate, 3),
            "parse_success_rate": round(parse_successes / max(1, total), 3),
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "verdict_counts": verdict_counts,
            "extraction_methods": extraction_methods,
            "parse_errors": total - parse_successes,
            "circuit_breaker_trips": breaker.trip_count,
            "total_retries": sum(s.get("retries", 0) for s in stats),
            "chunks_retried": sum(1 for s in stats if s.get("retries", 0) > 0),
            # Preflight quality (v7)
            "tier_distribution": tier_distribution,
            "chunks_split": chunks_split,
            "math_masked_chunks": math_masked_chunks,
            "adaptive_fallbacks": adaptive_fallbacks,
            "adaptive_escalations": adaptive_escalations,
            # Content recipe quality (v7.2)
            "recipe_applied": recipe_applied,
            "recipe_recovered": recipe_recovered,
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        quality = data.get("quality", {})
        methods = quality.get("extraction_methods", {})
        return (
            f"Edit plan: {data['total_ops']} ops from {data['chunks_processed']} chunks. "
            f"Parse rate: {quality.get('parse_success_rate', 0):.0%}. "
            f"Extraction: {methods}"
        )
