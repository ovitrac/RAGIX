"""
Kernel: pres_slide_plan
Stage: 2 (Structuring)

Converts a NormalizedCorpus into a SlideDeck by mapping semantic units
to slides with type-appropriate content structures. Handles splitting
of long content, merging of short sections, and bounds enforcement.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.presenter.models import (
    Asset,
    AssetCatalog,
    DeckMetadata,
    DeckTheme,
    NormalizedCorpus,
    NormalizedUnit,
    ProvenanceMethod,
    SemanticUnit,
    Slide,
    SlideCode,
    SlideContent,
    SlideDeck,
    SlideImage,
    SlideProvenance,
    SlideTable,
    SlideType,
    TopicCluster,
    UnitRole,
    UnitType,
)
from ragix_kernels.presenter.config import SlidePlanConfig, TableOverflowConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit → SlideType mapping
# ---------------------------------------------------------------------------

_UNIT_TYPE_TO_SLIDE: Dict[UnitType, SlideType] = {
    UnitType.PARAGRAPH: SlideType.CONTENT,
    UnitType.BULLET_LIST: SlideType.CONTENT,
    UnitType.NUMBERED_LIST: SlideType.CONTENT,
    UnitType.TABLE: SlideType.TABLE,
    UnitType.CODE_BLOCK: SlideType.CODE,
    UnitType.EQUATION_BLOCK: SlideType.EQUATION,
    UnitType.BLOCKQUOTE: SlideType.QUOTE,
    UnitType.ADMONITION: SlideType.CONTENT,
    UnitType.MERMAID: SlideType.IMAGE_FULL,
}


def _heading_slide_type(level: int) -> SlideType:
    """Map heading level to slide type."""
    if level <= 2:
        return SlideType.SECTION
    return SlideType.CONTENT


# ---------------------------------------------------------------------------
# Content builders
# ---------------------------------------------------------------------------

def _make_provenance(
    unit: SemanticUnit,
    method: ProvenanceMethod = ProvenanceMethod.EXTRACTED,
) -> SlideProvenance:
    """Build provenance from a SemanticUnit."""
    return SlideProvenance(
        source_file=unit.source_file,
        source_lines=list(unit.source_lines),
        heading_path=unit.heading_path,
        unit_ids=[unit.id],
        method=method,
    )


def _speaker_notes(unit: SemanticUnit) -> str:
    """Generate speaker notes with provenance info."""
    return f"Source: {unit.source_file}:L{unit.source_lines[0]}-L{unit.source_lines[1]}"


def _parse_table_content(content: str) -> Tuple[List[str], List[List[str]]]:
    """Parse Markdown table text into headers and rows."""
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    if not lines:
        return [], []

    def _split_row(line: str) -> List[str]:
        # Strip leading/trailing pipes and split
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [cell.strip() for cell in line.split("|")]

    headers: List[str] = []
    rows: List[List[str]] = []
    sep_re = re.compile(r'^[\s|:]*-[-:|\ ]*$')

    for i, line in enumerate(lines):
        if sep_re.match(line):
            continue  # skip separator row
        cells = _split_row(line)
        if not headers:
            headers = cells
        else:
            rows.append(cells)

    return headers, rows


def _split_bullets(items: List[str], max_per_slide: int) -> List[List[str]]:
    """Split a list of bullet items into chunks of max_per_slide."""
    if len(items) <= max_per_slide:
        return [items]
    return [items[i:i + max_per_slide] for i in range(0, len(items), max_per_slide)]


def _extract_bullets(content: str) -> List[str]:
    """Extract bullet items from a bullet/numbered list content string."""
    items = []
    for line in content.splitlines():
        line = line.strip()
        m = re.match(r'^[\s]*[*\-+]\s+(.+)', line) or re.match(r'^[\s]*\d+[.)]\s+(.+)', line)
        if m:
            items.append(m.group(1))
        elif line and items:
            # Continuation line
            items[-1] += " " + line
    return items


def _split_paragraph(text: str, max_words: int) -> List[str]:
    """Split a paragraph into chunks of approximately max_words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class PresSlidePlanKernel(Kernel):
    """NormalizedCorpus → SlideDeck (without layout directives)."""

    name = "pres_slide_plan"
    version = "1.0.0"
    category = "presenter"
    stage = 2
    description = "Map semantic units to typed slides with splitting/merging"

    requires: List[str] = ["pres_semantic_normalize", "pres_asset_catalog"]
    provides: List[str] = ["slide_deck"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load dependencies
        norm_path = input.dependencies["pres_semantic_normalize"]
        norm_data = json.loads(norm_path.read_text(encoding="utf-8"))
        normalized = NormalizedCorpus.from_dict(norm_data["data"])

        catalog_path = input.dependencies["pres_asset_catalog"]
        catalog_data = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog = AssetCatalog.from_dict({
            "assets": catalog_data["data"].get("assets", []),
            "by_type": catalog_data["data"].get("by_type", {}),
            "by_file": catalog_data["data"].get("by_file", {}),
        })

        # Parse config
        sp_cfg = input.config.get("slide_plan", {})
        compression = sp_cfg.get("compression", "full")
        plan_cfg = SlidePlanConfig(
            max_bullets_per_slide=sp_cfg.get("max_bullets_per_slide", 6),
            max_words_per_slide=sp_cfg.get("max_words_per_slide", 80),
            split_long_lists=sp_cfg.get("split_long_lists", True),
            equation_standalone=sp_cfg.get("equation_standalone", True),
            code_standalone_min_lines=sp_cfg.get("code_standalone_min_lines", 5),
            merge_short_sections=sp_cfg.get("merge_short_sections", True),
            min_slides=sp_cfg.get("min_slides", 8),
            max_slides=sp_cfg.get("max_slides", 60),
            compression=compression,
            max_slides_per_section=sp_cfg.get("max_slides_per_section", 0),
            annex_exclude_patterns=sp_cfg.get("annex_exclude_patterns", [
                "annexe", "annex", "appendix", "appendice",
            ]),
            executive_min_importance=sp_cfg.get("executive_min_importance", 0.5),
        )
        # Executive mode: auto-cap at 25 slides if no explicit max
        if compression == "executive" and sp_cfg.get("max_slides") is None:
            plan_cfg.max_slides = 25
        # Compressed mode: default per-section cap of 8
        if compression == "compressed" and plan_cfg.max_slides_per_section == 0:
            plan_cfg.max_slides_per_section = 8
        tbl_cfg_raw = input.config.get("table_overflow", {})
        tbl_cfg = TableOverflowConfig(
            max_rows=tbl_cfg_raw.get("max_rows", 12),
            max_cols=tbl_cfg_raw.get("max_cols", 8),
            strategy=tbl_cfg_raw.get("strategy", "split"),
        )

        # Metadata
        title = input.config.get("title", "")
        subtitle = input.config.get("subtitle", "")
        author = input.config.get("author", "")
        organization = input.config.get("organization", "")
        date_str = input.config.get("date", "") or date.today().isoformat()
        lang = input.config.get("lang", "auto")
        if lang == "auto":
            lang = "fr"

        # Auto-detect title from front-matter or first heading
        if not title:
            for nu in normalized.units:
                if nu.unit.type == UnitType.FRONT_MATTER:
                    fm_keys = nu.unit.metadata.get("keys", [])
                    if "title" in fm_keys:
                        # Parse YAML content for title
                        for line in nu.unit.content.splitlines():
                            if line.strip().startswith("title:"):
                                title = line.split(":", 1)[1].strip().strip("'\"")
                                break
                    break
            if not title:
                for nu in normalized.units:
                    if nu.unit.type == UnitType.HEADING and nu.unit.metadata.get("level") == 1:
                        title = nu.unit.content
                        break
            if not title:
                title = "Presentation"

        slides: List[Slide] = []
        slide_idx = 0

        def _next_id() -> str:
            nonlocal slide_idx
            slide_idx += 1
            return f"slide-{slide_idx:03d}"

        # --- Title slide ---
        slides.append(Slide(
            id=_next_id(),
            type=SlideType.TITLE,
            content=SlideContent(
                heading=title,
                subheading=subtitle,
                body=[f"{author}" + (f" | {organization}" if organization else ""), date_str] if author else [date_str],
            ),
            notes="Title slide",
            provenance=SlideProvenance(method=ProvenanceMethod.AUTO_SECTION),
        ))

        # --- TOC slide (expanded later by pres_marp_render) ---
        toc_cfg = sp_cfg.get("toc", {})
        toc_enabled = toc_cfg.get("enabled", True)
        if toc_enabled:
            toc_title = toc_cfg.get("title", "Table of Contents")
            slides.append(Slide(
                id=_next_id(),
                type=SlideType.CONTENT,
                content=SlideContent(
                    heading=toc_title,
                    body=["[TOC]"],
                ),
                notes="Table of contents — expanded by pres_marp_render",
                provenance=SlideProvenance(method=ProvenanceMethod.AUTO_SECTION),
            ))

        # --- Content slides with budget-aware allocation ---
        active = normalized.active_units()

        # Filter out boilerplate meta-clusters (TOC, figure lists, etc.)
        _META_PATTERNS = (
            "table des matières", "table des figures", "table des tableaux",
            "historique du document", "uncategorized",
            "table of contents", "list of figures", "list of tables",
        )

        # Map cluster label → units (excluding headings — they produce empty slides)
        cluster_units: Dict[str, List[NormalizedUnit]] = {}
        annex_excluded = 0
        exec_filtered = 0
        for nu in active:
            if nu.unit.type == UnitType.HEADING:
                continue  # headings waste budget; heading text is in content units via heading_path
            if nu.unit.type == UnitType.FRONT_MATTER:
                continue  # already handled in title slide
            cl = nu.topic_cluster or "Uncategorized"
            # Skip boilerplate meta-clusters (substring match for partial labels)
            cl_lower = cl.lower()
            if any(pat in cl_lower for pat in _META_PATTERNS):
                continue
            # --- Annex exclusion (compressed/executive modes) ---
            if compression in ("compressed", "executive"):
                if any(pat in cl_lower for pat in plan_cfg.annex_exclude_patterns):
                    annex_excluded += 1
                    continue
            # --- Executive mode: role filter ---
            if compression == "executive":
                _EXEC_ROLES = (UnitRole.FINDING, UnitRole.RECOMMENDATION, UnitRole.PROBLEM)
                if nu.role not in _EXEC_ROLES:
                    # Keep high-importance images and tables
                    if nu.unit.type in (UnitType.IMAGE_REF, UnitType.TABLE) and \
                       nu.importance >= plan_cfg.executive_min_importance:
                        pass  # include
                    else:
                        exec_filtered += 1
                        continue
            cluster_units.setdefault(cl, []).append(nu)

        if annex_excluded > 0:
            logger.info(f"[pres_slide_plan] Annex exclusion: {annex_excluded} units removed")
        if exec_filtered > 0:
            logger.info(f"[pres_slide_plan] Executive filter: {exec_filtered} units removed")

        # Section ordering: document order (first-appearance of cluster labels)
        # preserves the original structure instead of narrative arc reordering
        section_order_cfg = sp_cfg.get("section_order", "document")
        seen_labels: List[str] = []
        seen_set: set = set()
        for nu in active:
            cl = nu.topic_cluster or "Uncategorized"
            if cl not in seen_set and cl in cluster_units:
                seen_labels.append(cl)
                seen_set.add(cl)

        if section_order_cfg == "narrative":
            # Use narrative arc ordering (role-based)
            section_order = list(normalized.narrative.sections)
            covered = set(section_order)
            for cl_label in seen_labels:
                if cl_label not in covered:
                    section_order.append(cl_label)
            section_order = [s for s in section_order if s in cluster_units]
        else:
            # Document order (default) — preserves original heading structure
            section_order = seen_labels

        n_sections = len(section_order)

        # Budget allocation: overhead (title + optional TOC) + N sections + content_budget
        overhead = 2 if toc_enabled else 1  # title slide + TOC slide
        content_budget = plan_cfg.max_slides - overhead - n_sections
        if content_budget < n_sections:
            # Not enough room — drop smallest sections
            section_weights = [
                (s, len(cluster_units[s])) for s in section_order
            ]
            section_weights.sort(key=lambda x: -x[1])
            max_sections = (plan_cfg.max_slides - overhead) // 2
            kept = set(s for s, _ in section_weights[:max_sections])
            section_order = [s for s in section_order if s in kept]
            n_sections = len(section_order)
            content_budget = plan_cfg.max_slides - overhead - n_sections

        # Allocate content slots per section proportional to unit count
        total_units = sum(len(cluster_units.get(s, [])) for s in section_order)
        section_budgets: Dict[str, int] = {}
        section_cap = plan_cfg.max_slides_per_section  # 0 = unlimited
        for s in section_order:
            n_units = len(cluster_units.get(s, []))
            ratio = n_units / max(total_units, 1)
            budget_alloc = max(1, round(ratio * content_budget))
            # Apply per-section cap if set
            if section_cap > 0:
                budget_alloc = min(budget_alloc, section_cap)
            section_budgets[s] = budget_alloc

        # Adjust to fit exact budget (redistribute rounding residual)
        allocated = sum(section_budgets.values())
        if allocated > content_budget:
            sorted_secs = sorted(section_budgets, key=lambda s: section_budgets[s])
            for s in sorted_secs:
                if allocated <= content_budget:
                    break
                if section_budgets[s] > 1:
                    section_budgets[s] -= 1
                    allocated -= 1
        elif allocated < content_budget:
            sorted_secs = sorted(section_budgets, key=lambda s: -len(cluster_units.get(s, [])))
            for s in sorted_secs:
                if allocated >= content_budget:
                    break
                section_budgets[s] += 1
                allocated += 1

        logger.info(
            f"[pres_slide_plan] Budget: {n_sections} sections, "
            f"{content_budget} content slots across {total_units} units"
        )

        # Generate slides per section, respecting per-section budget
        for section_label in section_order:
            units_in_section = cluster_units.get(section_label, [])
            if not units_in_section:
                continue

            # Section divider slide
            slides.append(Slide(
                id=_next_id(),
                type=SlideType.SECTION,
                content=SlideContent(heading=section_label),
                notes=f"Section: {section_label}",
                provenance=SlideProvenance(method=ProvenanceMethod.AUTO_SECTION),
            ))

            # Generate content slides for this section, up to budget
            budget = section_budgets.get(section_label, 1)
            # Sort units by importance (highest first) for budget trimming
            sorted_units = sorted(
                units_in_section, key=lambda nu: -nu.importance,
            )
            section_slides: List[Slide] = []
            for nu in sorted_units:
                if len(section_slides) >= budget:
                    break
                u = nu.unit
                new_slides = self._unit_to_slides(
                    u, nu, catalog, plan_cfg, tbl_cfg, _next_id,
                )
                section_slides.extend(new_slides)

            # Trim to budget (in case unit_to_slides produced multiple slides)
            trimmed = section_slides[:budget]

            # Deduplicate headings within section: add " (suite)" to consecutive
            # slides sharing the same heading to avoid confusion
            seen_headings: Dict[str, int] = {}
            for sl in trimmed:
                h = sl.content.heading or ""
                if not h:
                    continue
                if h in seen_headings:
                    seen_headings[h] += 1
                    sl.content.heading = f"{h} ({seen_headings[h]})"
                else:
                    seen_headings[h] = 1

            slides.extend(trimmed)

        # --- Final bounds enforcement (safety net) ---
        if len(slides) > plan_cfg.max_slides:
            slides = self._trim_slides(slides, plan_cfg.max_slides)

        # Build deck
        metadata = DeckMetadata(
            title=title,
            author=author,
            subtitle=subtitle,
            organization=organization,
            date=date_str,
            source_folder=normalized.raw.root,
            lang=lang,
        )
        deck = SlideDeck(metadata=metadata, slides=slides)

        result = deck.to_dict()
        result["statistics"] = {
            "total_slides": len(slides),
            "slide_types": self._count_types(slides),
            "units_mapped": sum(
                len(s.provenance.unit_ids) if s.provenance else 0
                for s in slides
            ),
            "compression": compression,
            "annex_excluded": annex_excluded,
            "executive_filtered": exec_filtered,
        }
        return result

    def _unit_to_slides(
        self,
        u: SemanticUnit,
        nu: NormalizedUnit,
        catalog: AssetCatalog,
        plan_cfg: SlidePlanConfig,
        tbl_cfg: TableOverflowConfig,
        next_id,
    ) -> List[Slide]:
        """Convert a single SemanticUnit to one or more Slides."""
        slides: List[Slide] = []
        prov = _make_provenance(u)
        notes = _speaker_notes(u)

        # Skip headings (they become section dividers at cluster level)
        if u.type == UnitType.HEADING:
            level = u.metadata.get("level", 1)
            if level >= 3:
                # H3+ becomes a content slide with just a heading
                slides.append(Slide(
                    id=next_id(),
                    type=SlideType.CONTENT,
                    content=SlideContent(heading=u.content),
                    notes=notes,
                    provenance=prov,
                ))
            return slides

        if u.type == UnitType.FRONT_MATTER:
            return slides  # already handled in title

        # --- Paragraph ---
        if u.type == UnitType.PARAGRAPH:
            heading = u.heading_path[-1] if u.heading_path else ""
            chunks = _split_paragraph(u.content, plan_cfg.max_words_per_slide)
            for chunk in chunks:
                slides.append(Slide(
                    id=next_id(),
                    type=SlideType.CONTENT,
                    content=SlideContent(heading=heading, body=[chunk]),
                    notes=notes,
                    provenance=prov,
                ))
                heading = ""  # only first chunk gets heading
            return slides

        # --- Bullet list ---
        if u.type == UnitType.BULLET_LIST:
            heading = u.heading_path[-1] if u.heading_path else ""
            items = _extract_bullets(u.content)
            if plan_cfg.split_long_lists and len(items) > plan_cfg.max_bullets_per_slide:
                for chunk in _split_bullets(items, plan_cfg.max_bullets_per_slide):
                    slides.append(Slide(
                        id=next_id(),
                        type=SlideType.CONTENT,
                        content=SlideContent(heading=heading, bullets=chunk),
                        notes=notes,
                        provenance=prov,
                    ))
                    heading = ""
            else:
                slides.append(Slide(
                    id=next_id(),
                    type=SlideType.CONTENT,
                    content=SlideContent(heading=heading, bullets=items),
                    notes=notes,
                    provenance=prov,
                ))
            return slides

        # --- Numbered list ---
        if u.type == UnitType.NUMBERED_LIST:
            heading = u.heading_path[-1] if u.heading_path else ""
            items = _extract_bullets(u.content)
            if plan_cfg.split_long_lists and len(items) > plan_cfg.max_bullets_per_slide:
                for chunk in _split_bullets(items, plan_cfg.max_bullets_per_slide):
                    slides.append(Slide(
                        id=next_id(),
                        type=SlideType.CONTENT,
                        content=SlideContent(heading=heading, numbered=chunk),
                        notes=notes,
                        provenance=prov,
                    ))
                    heading = ""
            else:
                slides.append(Slide(
                    id=next_id(),
                    type=SlideType.CONTENT,
                    content=SlideContent(heading=heading, numbered=items),
                    notes=notes,
                    provenance=prov,
                ))
            return slides

        # --- Table ---
        if u.type == UnitType.TABLE:
            heading = u.heading_path[-1] if u.heading_path else ""
            headers, rows = _parse_table_content(u.content)
            n_rows = u.metadata.get("rows", len(rows))

            if n_rows > tbl_cfg.max_rows and tbl_cfg.strategy == "split":
                # Split into multiple slides
                for i in range(0, len(rows), tbl_cfg.max_rows):
                    chunk_rows = rows[i:i + tbl_cfg.max_rows]
                    slides.append(Slide(
                        id=next_id(),
                        type=SlideType.TABLE,
                        content=SlideContent(
                            heading=heading,
                            table=SlideTable(headers=headers, rows=chunk_rows),
                        ),
                        notes=notes,
                        provenance=prov,
                    ))
                    heading = ""
            else:
                slides.append(Slide(
                    id=next_id(),
                    type=SlideType.TABLE,
                    content=SlideContent(
                        heading=heading,
                        table=SlideTable(headers=headers, rows=rows),
                    ),
                    notes=notes,
                    provenance=prov,
                ))
            return slides

        # --- Code block ---
        if u.type == UnitType.CODE_BLOCK:
            heading = u.heading_path[-1] if u.heading_path else ""
            lang = u.metadata.get("language", "")
            # Extract code content (between fences)
            lines = u.content.splitlines()
            # Strip opening/closing fences if present
            if lines and lines[0].strip().startswith(("```", "~~~")):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith(("```", "~~~")):
                lines = lines[:-1]
            code_text = "\n".join(lines)
            slides.append(Slide(
                id=next_id(),
                type=SlideType.CODE,
                content=SlideContent(
                    heading=heading,
                    code=SlideCode(language=lang, text=code_text),
                ),
                notes=notes,
                provenance=prov,
            ))
            return slides

        # --- Equation block ---
        if u.type == UnitType.EQUATION_BLOCK:
            heading = u.heading_path[-1] if u.heading_path else ""
            latex = u.metadata.get("latex", u.content)
            slides.append(Slide(
                id=next_id(),
                type=SlideType.EQUATION,
                content=SlideContent(heading=heading, equation=latex),
                notes=notes,
                provenance=prov,
            ))
            return slides

        # --- Image reference ---
        if u.type == UnitType.IMAGE_REF:
            heading = u.heading_path[-1] if u.heading_path else ""
            img_path = u.metadata.get("path", "")
            alt = u.metadata.get("alt", "")
            asset_id = u.metadata.get("asset_id", "")
            slides.append(Slide(
                id=next_id(),
                type=SlideType.IMAGE_FULL if not alt else SlideType.IMAGE_TEXT,
                content=SlideContent(
                    heading=heading,
                    image=SlideImage(
                        asset_id=asset_id,
                        path=img_path,
                        alt=alt,
                    ),
                    body=[alt] if alt else [],
                ),
                notes=notes,
                provenance=prov,
            ))
            return slides

        # --- Mermaid diagram ---
        if u.type == UnitType.MERMAID:
            heading = u.heading_path[-1] if u.heading_path else ""
            slides.append(Slide(
                id=next_id(),
                type=SlideType.IMAGE_FULL,
                content=SlideContent(
                    heading=heading,
                    code=SlideCode(language="mermaid", text=u.content),
                ),
                notes=notes,
                provenance=prov,
            ))
            return slides

        # --- Blockquote ---
        if u.type == UnitType.BLOCKQUOTE:
            heading = u.heading_path[-1] if u.heading_path else ""
            slides.append(Slide(
                id=next_id(),
                type=SlideType.QUOTE,
                content=SlideContent(heading=heading, body=[u.content]),
                notes=notes,
                provenance=prov,
            ))
            return slides

        # --- Admonition ---
        if u.type == UnitType.ADMONITION:
            heading = u.heading_path[-1] if u.heading_path else ""
            adm_type = u.metadata.get("admonition_type", "NOTE")
            slides.append(Slide(
                id=next_id(),
                type=SlideType.CONTENT,
                content=SlideContent(
                    heading=heading or adm_type,
                    body=[u.content],
                ),
                notes=notes,
                provenance=prov,
            ))
            return slides

        # Fallback: generic content slide
        heading = u.heading_path[-1] if u.heading_path else ""
        slides.append(Slide(
            id=next_id(),
            type=SlideType.CONTENT,
            content=SlideContent(heading=heading, body=[u.content]),
            notes=notes,
            provenance=prov,
        ))
        return slides

    def _trim_slides(self, slides: List[Slide], max_slides: int) -> List[Slide]:
        """
        Trim slides to max_slides using a multi-phase strategy.

        Phase 1: Remove content slides with lowest importance (from the end).
        Phase 2: Remove empty section dividers (sections with no content
                 slides between them and the next section/end).
        Phase 3: Remove remaining section dividers (smallest first by
                 counting their content slides) if still over budget.
        """
        if len(slides) <= max_slides:
            return slides

        result = list(slides)

        # --- Phase 1: Remove content slides from the end ---
        while len(result) > max_slides:
            removed = False
            for i in range(len(result) - 1, 0, -1):
                if result[i].type not in (SlideType.TITLE, SlideType.SECTION):
                    result.pop(i)
                    removed = True
                    break
            if not removed:
                break  # only title/sections left

        if len(result) <= max_slides:
            return result

        # --- Phase 2: Remove empty section dividers ---
        # A section is "empty" if followed immediately by another section or end
        changed = True
        while changed and len(result) > max_slides:
            changed = False
            for i in range(len(result) - 1, 0, -1):
                if result[i].type != SlideType.SECTION:
                    continue
                # Check if next slide (if any) is also a section or end of list
                is_last = (i == len(result) - 1)
                next_is_section = (not is_last and result[i + 1].type == SlideType.SECTION)
                if is_last or next_is_section:
                    result.pop(i)
                    changed = True
                    break

        if len(result) <= max_slides:
            return result

        # --- Phase 3: Remove sections with fewest content slides ---
        # Count content slides belonging to each section
        while len(result) > max_slides:
            # Find section indices and their content counts
            section_content: List[tuple] = []  # (index, content_count)
            for i, s in enumerate(result):
                if s.type != SlideType.SECTION:
                    continue
                # Count content slides until next section or end
                count = 0
                for j in range(i + 1, len(result)):
                    if result[j].type == SlideType.SECTION:
                        break
                    if result[j].type != SlideType.TITLE:
                        count += 1
                section_content.append((i, count))

            if not section_content:
                break

            # Remove section with fewest content slides (plus its content)
            section_content.sort(key=lambda x: x[1])
            rm_idx, rm_count = section_content[0]
            # Remove the section + its content slides
            end_idx = rm_idx + 1
            while end_idx < len(result) and result[end_idx].type not in (SlideType.SECTION, SlideType.TITLE):
                end_idx += 1
            del result[rm_idx:end_idx]

            if len(result) <= max_slides:
                break

        return result

    def _count_types(self, slides: List[Slide]) -> Dict[str, int]:
        """Count slides by type."""
        counts: Dict[str, int] = {}
        for s in slides:
            counts[s.type.value] = counts.get(s.type.value, 0) + 1
        return counts

    def summarize(self, data: Dict[str, Any]) -> str:
        stats = data.get("statistics", {})
        total = stats.get("total_slides", 0)
        types = stats.get("slide_types", {})
        parts = [f"{v} {k}" for k, v in sorted(types.items())]
        return f"Slide plan: {total} slides ({', '.join(parts)})"
