"""
Kernel: md_consistency_scan
Stage: 2 (Analysis)

Detect multi-agent drift, AI leftovers, register shifts, duplicated paragraphs,
broken cross-references, numerical contradictions, table structure issues,
and terminology drift. v1 is fully deterministic (regex + heuristics).
v2 adds: French cross-refs, numerical consistency, table validation, term drift.
v2.1 fixes: footnote skip, scope-excluded rows, percentage independence guards,
term drift min-count threshold.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-08
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.reviewer.config import ReviewerConfig

import logging

logger = logging.getLogger(__name__)

# Default AI leftover patterns
DEFAULT_AI_PATTERNS = [
    r"(?i)\bAs an AI\b",
    r"(?i)\bAs a language model\b",
    r"(?i)\bIn conclusion\b",
    r"(?i)\bSure,?\s+here['']?s\b",
    r"(?i)\bI hope this helps\b",
    r"(?i)\bLet me know if\b",
    r"(?i)\bCertainly!",
    r"(?i)\bAbsolutely!",
    r"(?i)\bGreat question",
    r"(?i)\bHere is (?:a|an|the)\b",
]

# Broken referent patterns (English — v1)
BROKEN_REF_PATTERNS = [
    r"(?i)\bsee\s+(?:Section|Table|Figure|Equation)\s+(\d[\d.]*)",
    r"(?i)\b(?:Table|Figure|Equation)\s+(\d+)\b",
]

# French cross-reference patterns (v2)
FRENCH_XREF_PATTERNS = [
    re.compile(r"(?:§|section\s+)(\d[\d.]+)", re.IGNORECASE),
    re.compile(r"(?:voir|cf\.?)\s+(?:§|section\s+)?(\d[\d.]+)", re.IGNORECASE),
    re.compile(r"\((?:voir|cf\.?)\s+(\d[\d.]+)\)", re.IGNORECASE),
]

# Quantity-noun patterns for numerical consistency (v2)
# Matches "N approches/classes/méthodes/modules/étapes/workflows/..."
_QUANTITY_NOUN_RE = re.compile(
    r"\b(\d+)\s+"
    r"(approches?|classes?|méthodes?|modules?|étapes?|workflows?|"
    r"controllers?|contrôleurs?|services?|packages?|tables?|"
    r"catégories?|phases?|priorités?|recommandations?|actions?|"
    r"arêtes?|nœuds?|lignes?)\b",
    re.IGNORECASE,
)

# Table detection patterns
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:]*-+[\s:]*\|")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|")


def _paragraph_hash(text: str) -> str:
    """Normalize and hash a paragraph for duplicate detection."""
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()


def _detect_ai_leftovers(
    lines: List[str], patterns: List[str]
) -> List[Dict[str, Any]]:
    """Find AI-generated leftover phrases."""
    issues = []
    compiled = [re.compile(p) for p in patterns]
    for i, line in enumerate(lines):
        for j, pattern in enumerate(compiled):
            m = pattern.search(line)
            if m:
                issues.append({
                    "type": "ai_leftover",
                    "line": i + 1,
                    "match": m.group(),
                    "pattern": patterns[j],
                    "severity": "attention",
                    "evidence": line.strip()[:120],
                })
    return issues


def _detect_duplicated_paragraphs(lines: List[str]) -> List[Dict[str, Any]]:
    """Find duplicated paragraphs via hash comparison."""
    # Split into paragraphs
    paragraphs: List[Tuple[int, int, str]] = []
    current_start = 0
    current_lines: List[str] = []

    for i, line in enumerate(lines):
        if line.strip() == "":
            if current_lines:
                text = "\n".join(current_lines)
                if len(text.strip()) > 40:  # Skip trivial paragraphs
                    paragraphs.append((current_start + 1, i, text))
                current_lines = []
                current_start = i + 1
        else:
            if not current_lines:
                current_start = i
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines)
        if len(text.strip()) > 40:
            paragraphs.append((current_start + 1, len(lines), text))

    # Find duplicates by hash
    seen: Dict[str, Tuple[int, int]] = {}
    issues = []
    for start, end, text in paragraphs:
        h = _paragraph_hash(text)
        if h in seen:
            orig_start, orig_end = seen[h]
            issues.append({
                "type": "duplicated_paragraph",
                "line": start,
                "line_end": end,
                "original_line": orig_start,
                "severity": "attention",
                "evidence": text.strip()[:120],
            })
        else:
            seen[h] = (start, end)

    return issues


def _detect_broken_refs(
    lines: List[str], anchor_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Find references to non-existent sections/figures/tables."""
    issues = []
    for i, line in enumerate(lines):
        for pattern in BROKEN_REF_PATTERNS:
            for m in re.finditer(pattern, line):
                ref = m.group()
                # Check if the referenced item exists
                # This is a heuristic: we check if the number appears in anchors
                ref_num = m.group(1) if m.lastindex else ""
                # Simple check: if "Section X" mentioned, verify heading exists
                if "section" in ref.lower():
                    found = any(
                        ref_num in (s.get("numbering", "") or s.get("id", ""))
                        for s in anchor_map.values()
                    ) if isinstance(anchor_map, dict) else False
                    if not found:
                        issues.append({
                            "type": "broken_reference",
                            "line": i + 1,
                            "reference": ref,
                            "severity": "attention",
                            "evidence": line.strip()[:120],
                        })
    return issues


# ---------------------------------------------------------------------------
# v2 Detectors
# ---------------------------------------------------------------------------

def _detect_french_cross_refs(
    lines: List[str],
    section_numbers: Set[str],
    protected_lines: Set[int],
) -> List[Dict[str, Any]]:
    """
    Validate French cross-references (§X.Y, voir X.Y, cf. X.Y) against
    the section index extracted by md_structure.

    Suppresses false positives for explicitly deleted sections
    (lines containing "SUPPRIMÉ" near the reference).
    """
    issues = []
    # Track already-reported references to avoid duplicates from same line
    seen_refs: Set[Tuple[int, str]] = set()

    for i, line in enumerate(lines):
        if (i + 1) in protected_lines:
            continue
        for pattern in FRENCH_XREF_PATTERNS:
            for m in pattern.finditer(line):
                target = m.group(1).rstrip(".")
                key = (i + 1, target)
                if key in seen_refs:
                    continue
                seen_refs.add(key)

                # Check exact match or prefix match
                exact = target in section_numbers
                prefix = any(s.startswith(target + ".") for s in section_numbers)
                if exact or prefix:
                    continue

                # Suppress if the line mentions the section was deleted
                if re.search(r"(?i)supprim[ée]", line):
                    continue

                issues.append({
                    "type": "broken_cross_ref",
                    "line": i + 1,
                    "reference": target,
                    "severity": "attention",
                    "evidence": line.strip()[:120],
                })
    return issues


def _detect_numerical_contradictions(
    lines: List[str],
    protected_lines: Set[int],
) -> List[Dict[str, Any]]:
    """
    Cross-check quantity claims (e.g. "deux approches") against nearby table
    row counts.  Catches the "deux approches but 3 table rows" class of bugs.

    Strategy: for each "N <noun>" claim, if a markdown table starts within
    ±10 lines, count its data rows and compare to N.
    """
    issues = []
    # Index all table locations (start_line → data_row_count)
    table_info: Dict[int, int] = {}  # separator_line → data_rows
    for i, line in enumerate(lines):
        if _TABLE_SEP_RE.match(line):
            # Count data rows after separator
            j = i + 1
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                j += 1
            data_rows = j - i - 1
            table_info[i + 1] = data_rows  # 1-based

    # French number words
    _FR_NUMBERS = {
        "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
        "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    }
    _FR_NUM_RE = re.compile(
        r"\b(" + "|".join(_FR_NUMBERS.keys()) + r")\s+"
        r"(approches?|classes?|méthodes?|modules?|étapes?|workflows?|"
        r"catégories?|phases?|lignes?|contrôleurs?|controllers?)\b",
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):
        if (i + 1) in protected_lines:
            continue

        # Check digit-based claims
        for m in _QUANTITY_NOUN_RE.finditer(line):
            claimed_n = int(m.group(1))
            noun = m.group(2).lower()
            _check_nearby_table(
                issues, i + 1, claimed_n, noun, line, table_info, lines,
            )

        # Check French word-based claims
        for m in _FR_NUM_RE.finditer(line):
            claimed_n = _FR_NUMBERS[m.group(1).lower()]
            noun = m.group(2).lower()
            _check_nearby_table(
                issues, i + 1, claimed_n, noun, line, table_info, lines,
            )

    return issues


_SCOPE_EXCLUDE_RE = re.compile(
    r"(?i)hors\s+(?:scope|périmètre)|exclu[es]?|non\s+concern[ée]"
)


def _check_nearby_table(
    issues: List[Dict[str, Any]],
    claim_line: int,
    claimed_n: int,
    noun: str,
    line_text: str,
    table_info: Dict[int, int],
    lines: List[str],
) -> None:
    """
    Helper: if a table exists within the next 5 lines (claim → table direction),
    check row count vs claim.  Only flag when:
    - claim is 2-10 (small enough to count visually)
    - table is the very next table after the claim (not some unrelated table)
    - mismatch is clear (different count, not a subset/superset situation)

    v2.1: skip footnote lines; subtract scope-excluded rows from count.
    """
    if claimed_n > 10 or claimed_n < 2:
        return

    # Skip footnote lines: starts with * (not **bold**) followed by content
    stripped = line_text.lstrip()
    if stripped.startswith("*") and not stripped.startswith("**"):
        return

    # Find the nearest table AFTER the claim (forward-only, 5 lines)
    best_sep = None
    best_dist = 999
    for sep_line, data_rows in table_info.items():
        dist = sep_line - claim_line
        if 0 < dist <= 5 and dist < best_dist:
            best_sep = sep_line
            best_dist = dist
    if best_sep is None:
        return

    data_rows = table_info[best_sep]

    # Subtract scope-excluded rows ("hors scope", "exclu", etc.)
    excluded = 0
    for k in range(best_sep, best_sep + data_rows):
        if k < len(lines) and _SCOPE_EXCLUDE_RE.search(lines[k]):
            excluded += 1
    effective_rows = data_rows - excluded

    if effective_rows >= 2 and effective_rows != claimed_n:
        issues.append({
            "type": "numerical_contradiction",
            "line": claim_line,
            "claimed": claimed_n,
            "table_rows": effective_rows,
            "total_rows": data_rows,
            "excluded_rows": excluded,
            "noun": noun,
            "table_at_line": best_sep,
            "severity": "attention",
            "evidence": line_text.strip()[:120],
        })


def _detect_table_structure(
    lines: List[str],
) -> List[Dict[str, Any]]:
    """
    Validate markdown table structure:
    - Column count consistency (header vs separator vs data rows)
    - Percentage columns that should sum to ~100%
    """
    issues = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Detect table: row followed by separator
        if _TABLE_ROW_RE.match(line) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            header_cols = line.count("|") - 1
            sep_cols = lines[i + 1].count("|") - 1
            table_start = i + 1  # 1-based

            if header_cols != sep_cols and header_cols > 0 and sep_cols > 0:
                issues.append({
                    "type": "table_col_mismatch",
                    "line": table_start,
                    "header_cols": header_cols,
                    "separator_cols": sep_cols,
                    "severity": "attention",
                    "evidence": line.strip()[:80],
                })

            # Scan data rows and check column counts
            j = i + 2
            while j < n and _TABLE_ROW_RE.match(lines[j]):
                row_cols = lines[j].count("|") - 1
                if row_cols != header_cols and row_cols > 0 and header_cols > 0:
                    issues.append({
                        "type": "table_col_mismatch",
                        "line": j + 1,
                        "expected_cols": header_cols,
                        "actual_cols": row_cols,
                        "severity": "minor",
                        "evidence": lines[j].strip()[:80],
                    })
                j += 1

            # Check percentage sums in the table (v2.1: independence guards)
            pct_re = re.compile(r"(\d{1,3}(?:[.,]\d+)?)\s*%")
            pcts = []
            for k in range(i + 2, j):
                for pm in pct_re.finditer(lines[k]):
                    try:
                        pcts.append(float(pm.group(1).replace(",", ".")))
                    except ValueError:
                        pass
            if len(pcts) >= 3:
                total = sum(pcts)
                max_pct = max(pcts)

                # Guard 1: if any percentage >= 50%, these are likely
                # independent rates (prevalence, coverage) not a partition
                if max_pct >= 50:
                    pass
                # Guard 2: if sum > 120%, clearly not a partition
                elif total > 120:
                    pass
                # Only flag sums close to but not at 100%
                elif 80 < total < 95 or 105 < total <= 120:
                    issues.append({
                        "type": "table_pct_mismatch",
                        "line": table_start,
                        "total_pct": round(total, 1),
                        "row_count": len(pcts),
                        "severity": "minor",
                        "evidence": f"Percentages sum to {total:.1f}%",
                    })

            i = j
        else:
            i += 1

    return issues


def _detect_term_drift(
    lines: List[str],
    section_index: List[Dict[str, Any]],
    protected_lines: Set[int],
) -> List[Dict[str, Any]]:
    """
    Track key domain terms across sections and flag inconsistencies.

    Detects when the same concept is referred to by different names in
    different sections (e.g., "refactoring" vs "refonte", "code mort"
    vs "dead code", "habilitations" vs "autorisations").
    """
    issues = []

    # Term groups: variants that should be consistent within the document
    _TERM_GROUPS = [
        # (canonical, [variants]) — case-insensitive word-boundary match
        ("Option A", [r"\bOption\s+A\b"]),
        ("Option B", [r"\bOption\s+B\b"]),
        ("Greenfield", [r"\b[Gg]reen\s*field\b"]),
        ("Brownfield", [r"\b[Bb]rown\s*field\b"]),
    ]

    # Language mixing: detect sections mixing French and English for same terms
    _LANG_PAIRS = [
        (re.compile(r"\bcode\s+mort\b", re.I), re.compile(r"\bdead\s+code\b", re.I),
         "code mort", "dead code"),
        (re.compile(r"\bcontrôleurs?\b", re.I), re.compile(r"\bcontrollers?\b", re.I),
         "contrôleur", "controller"),
        (re.compile(r"\brefactoring\b", re.I), re.compile(r"\brefonte\b", re.I),
         "refactoring", "refonte"),
    ]

    # Per-section tracking for language pair mixing
    for sec in section_index:
        sec_start = sec.get("line_start", 0)
        sec_end = sec.get("line_end", 0)
        if sec_end <= sec_start:
            continue

        for fr_re, en_re, fr_term, en_term in _LANG_PAIRS:
            fr_count = 0
            en_count = 0
            for k in range(sec_start - 1, min(sec_end, len(lines))):
                if (k + 1) in protected_lines:
                    continue
                fr_count += len(fr_re.findall(lines[k]))
                en_count += len(en_re.findall(lines[k]))

            # Flag if both forms are used frequently in the same section
            # (v2.1: require min count >= 2 to suppress one-off comparisons)
            if fr_count >= 2 and en_count >= 2:
                issues.append({
                    "type": "term_drift",
                    "line": sec_start,
                    "section": sec.get("id", ""),
                    "term_fr": fr_term,
                    "count_fr": fr_count,
                    "term_en": en_term,
                    "count_en": en_count,
                    "severity": "minor",
                    "evidence": (
                        f"Section {sec.get('id', '')} uses both "
                        f"'{fr_term}' ({fr_count}×) and "
                        f"'{en_term}' ({en_count}×)"
                    ),
                })

    return issues


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------

class MdConsistencyScanKernel(Kernel):
    """Detect drift, AI leftovers, duplicates, refs, numbers, tables, terms."""

    name = "md_consistency_scan"
    version = "2.1.0"
    category = "reviewer"
    stage = 2
    description = "AI leftovers, duplicates, refs, numbers, tables, terms"

    requires: List[str] = ["md_chunk", "md_protected_regions"]
    provides: List[str] = ["issues"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load document
        snapshot_path = input.workspace / "stage1" / "doc.raw.md"
        text = snapshot_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Load protected spans to exclude
        prot_path = input.dependencies.get("md_protected_regions")
        protected_lines: Set[int] = set()
        if prot_path and prot_path.exists():
            prot_data = json.loads(prot_path.read_text())["data"]
            for span in prot_data.get("protected_spans", []):
                for ln in range(span["line_start"], span["line_end"] + 1):
                    protected_lines.add(ln)

        # Load section index for cross-ref and term drift checks
        section_numbers: Set[str] = set()
        section_index: List[Dict[str, Any]] = []
        try:
            struct_path = input.workspace / "stage1" / "md_structure.json"
            if struct_path.exists():
                struct_data = json.loads(struct_path.read_text())["data"]
                section_index = self._flatten_sections(
                    struct_data.get("heading_tree", [])
                )
                for sec in section_index:
                    num = sec.get("numbering", "")
                    if num:
                        section_numbers.add(num)
        except Exception:
            pass

        # Config
        reviewer_cfg = input.config.get("reviewer", {})
        style_cfg = reviewer_cfg.get("style", {})
        ai_patterns = style_cfg.get("ai_leftover_patterns", DEFAULT_AI_PATTERNS)

        all_issues: List[Dict[str, Any]] = []

        # 1. AI leftover detection (v1)
        ai_issues = _detect_ai_leftovers(lines, ai_patterns)
        ai_issues = [i for i in ai_issues if i["line"] not in protected_lines]
        all_issues.extend(ai_issues)

        # 2. Duplicated paragraphs (v1)
        dup_issues = _detect_duplicated_paragraphs(lines)
        all_issues.extend(dup_issues)

        # 3. Broken references — English patterns (v1)
        try:
            anchors_path = input.workspace / "stage1" / "anchors.json"
            anchor_map = json.loads(anchors_path.read_text()) if anchors_path.exists() else {}
        except Exception:
            anchor_map = {}
        ref_issues = _detect_broken_refs(lines, anchor_map)
        all_issues.extend(ref_issues)

        # 4. French cross-references (v2)
        if section_numbers:
            fr_ref_issues = _detect_french_cross_refs(
                lines, section_numbers, protected_lines
            )
            all_issues.extend(fr_ref_issues)

        # 5. Numerical contradictions (v2)
        num_issues = _detect_numerical_contradictions(
            lines, protected_lines
        )
        all_issues.extend(num_issues)

        # 6. Table structure validation (v2)
        table_issues = _detect_table_structure(lines)
        all_issues.extend(table_issues)

        # 7. Terminology drift (v2)
        if section_index:
            term_issues = _detect_term_drift(
                lines, section_index, protected_lines
            )
            all_issues.extend(term_issues)

        # Summary by type
        by_type: Dict[str, int] = Counter(i["type"] for i in all_issues)

        # Save
        stage_dir = input.workspace / "stage2"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "coherence_issues.json").write_text(
            json.dumps({
                "issues": all_issues,
                "total_issues": len(all_issues),
                "by_type": dict(by_type),
            }, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"[md_consistency_scan] {len(all_issues)} issues: "
            f"{dict(by_type)}"
        )

        return {
            "issues": all_issues,
            "total_issues": len(all_issues),
            "by_type": dict(by_type),
        }

    @staticmethod
    def _flatten_sections(
        tree: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Flatten heading tree into a flat list for iteration."""
        result = []
        for node in tree:
            result.append(node)
            children = node.get("children", [])
            if children:
                result.extend(
                    MdConsistencyScanKernel._flatten_sections(children)
                )
        return result

    def summarize(self, data: Dict[str, Any]) -> str:
        by_type = data.get("by_type", {})
        parts = [f"{v} {k}" for k, v in sorted(by_type.items())]
        return (
            f"Consistency scan: {data['total_issues']} issues. "
            f"{', '.join(parts) if parts else 'Clean.'}"
        )
