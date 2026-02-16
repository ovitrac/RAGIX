"""
summary_generate — Stage 3: Structured Summary Generation

LLM-assisted generation of per-domain summary sections with
mandatory [MID: xxx] citations, citation retry loop, and
language-consistent headings.

v1.1.0 — P2 (citation contract) + P3 (language enforcement)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)

# ── Section heading translations (P3) ───────────────────────────────────
# Deterministic mapping for structured headings. Applied post-generation
# to guarantee headings match the corpus language.

_HEADING_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "French": {
        "constraints": "Contraintes",
        "architectural decisions": "Décisions architecturales",
        "operational rules": "Règles opérationnelles",
        "executive summary": "Résumé exécutif",
        "domain coverage": "Couverture des domaines",
        "observations": "Observations",
        "recommendations": "Recommandations",
        "findings": "Constats",
    },
}


def _enforce_heading_language(content: str, language: str) -> str:
    """
    Deterministic heading sanitizer (P3).

    Replaces English section headings with their translation
    in the target language. Only touches ### headings.
    """
    translations = _HEADING_TRANSLATIONS.get(language, {})
    if not translations:
        return content

    def _replace_heading(m: re.Match) -> str:
        prefix = m.group(1)  # "### " or "## "
        heading_text = m.group(2).strip()
        heading_lower = heading_text.lower()
        if heading_lower in translations:
            return f"{prefix}{translations[heading_lower]}"
        return m.group(0)

    return re.sub(r"^(#{2,3}\s+)(.+)$", _replace_heading, content, flags=re.MULTILINE)


def _count_citations(content: str) -> Tuple[int, int]:
    """
    Count bullets and citations in generated content.

    Returns (total_bullets, cited_bullets).
    """
    bullets = re.findall(r"^[-*]\s+.+$", content, re.MULTILINE)
    cited = [b for b in bullets if re.search(r"\[MID:\s*MEM-[a-f0-9]+\]", b)]
    return len(bullets), len(cited)


_SECTION_PROMPT = """\
You are a technical compliance analyst writing a structured summary.

Using ONLY the memory items provided below, write a structured summary
for the technology domain: {domain}

Format:
## {domain}
### {h_constraints}
- [constraint description] [MID: {mid_example}]
### {h_decisions}
- [decision description] [MID: {mid_example}]
### {h_rules}
- [rule description] [MID: {mid_example}]

MANDATORY RULES:
1. Every bullet MUST end with [MID: xxx] citing the exact source memory item ID
2. Do NOT write any bullet without a [MID: xxx] citation — this is a hard contract
3. Be factual and analytical — do NOT use consultative framing
4. Do NOT use phrases like "I can help you", "Here is what I found"
5. Write ALL text in {language} — headings AND content
6. Include version numbers (e.g., "PostgreSQL 13", "RHEL 9") in each rule
7. If a memory item has no relevant constraint/decision/rule, skip it
8. Each [MID: xxx] must appear in the MEMORY ITEMS section below

MEMORY ITEMS:
{items_text}

Structured summary:"""

_RETRY_PROMPT = """\
Your previous response had {uncited} uncited bullets out of {total}.
The citation contract requires EVERY bullet to end with [MID: xxx].

Rewrite the summary for {domain}, ensuring ALL bullets have citations.
If you cannot cite a bullet, remove it.

Previous response:
{previous}

MEMORY ITEMS (same as before):
{items_text}

Corrected structured summary:"""


class SummaryGenerateKernel(Kernel):
    name = "summary_generate"
    version = "1.1.0"
    category = "summary"
    stage = 3
    description = "Generate structured per-domain summary sections"
    requires = ["summary_budgeted_recall"]
    provides = ["summary_sections"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Generate structured per-domain summary sections via LLM with citation enforcement."""
        cfg = input.config
        model = cfg.get("model", "ibm/granite4:32b-a9b-h")
        ollama_url = cfg.get("ollama_url", "http://localhost:11434")
        language = cfg.get("language", "French")
        citation_min_rate = cfg.get("citation_min_rate", 0.80)

        # Localized headings (P3)
        headings = _HEADING_TRANSLATIONS.get(language, {})
        h_constraints = headings.get("constraints", "Constraints")
        h_decisions = headings.get("architectural decisions", "Architectural Decisions")
        h_rules = headings.get("operational rules", "Operational Rules")

        # Load budgeted recall output
        recall_file = input.dependencies.get("summary_budgeted_recall")
        if recall_file and recall_file.exists():
            recall_data = json.loads(recall_file.read_text())["data"]
        else:
            raise RuntimeError("Missing summary_budgeted_recall dependency")

        domain_items = recall_data.get("domain_items", {})

        # V3.0: Optional domain filter (--domains flag)
        target_domains = cfg.get("domains", None)
        if target_domains:
            target_set = {d.strip().lower() for d in target_domains}
            filtered = {d: v for d, v in domain_items.items() if d.lower() in target_set}
            logger.info(
                f"Domain filter active: {len(filtered)}/{len(domain_items)} domains selected"
            )
            domain_items = filtered

        # Load memory store for item formatting
        db_path = cfg.get("db_path", str(input.workspace / "memory.db"))
        from ragix_core.memory.store import MemoryStore
        fts_tokenizer = cfg.get("fts_tokenizer")
        store = MemoryStore(db_path, fts_tokenizer=fts_tokenizer)

        sections = []
        citation_stats = {"total_bullets": 0, "cited_bullets": 0, "retries": 0}

        for domain, items_info in domain_items.items():
            # Format items for injection
            items_text_parts = []
            mid_example = ""
            for info in items_info:
                item = store.read_item(info["id"])
                if item:
                    items_text_parts.append(item.format_inject())
                    if not mid_example:
                        mid_example = item.id

            if not items_text_parts:
                continue

            items_text = "\n\n".join(items_text_parts)

            prompt = _SECTION_PROMPT.format(
                domain=domain.upper(),
                mid_example=mid_example,
                items_text=items_text,
                language=language,
                h_constraints=h_constraints,
                h_decisions=h_decisions,
                h_rules=h_rules,
            )

            # First attempt
            content = self._call_llm(prompt, model, ollama_url)

            # P2: Citation enforcement — check and retry once if needed
            total_b, cited_b = _count_citations(content)
            rate = (cited_b / total_b) if total_b > 0 else 1.0

            if rate < citation_min_rate and total_b > 0:
                uncited = total_b - cited_b
                logger.info(
                    f"[{domain}] Citation rate {rate:.0%} < {citation_min_rate:.0%} "
                    f"({uncited}/{total_b} uncited) — retrying"
                )
                retry_prompt = _RETRY_PROMPT.format(
                    uncited=uncited,
                    total=total_b,
                    domain=domain.upper(),
                    previous=content,
                    items_text=items_text,
                )
                content2 = self._call_llm(retry_prompt, model, ollama_url)
                total_b2, cited_b2 = _count_citations(content2)
                rate2 = (cited_b2 / total_b2) if total_b2 > 0 else 1.0
                citation_stats["retries"] += 1

                if rate2 > rate:
                    logger.info(
                        f"[{domain}] Retry improved: {rate:.0%} → {rate2:.0%}"
                    )
                    content = content2
                    total_b, cited_b = total_b2, cited_b2
                else:
                    logger.info(
                        f"[{domain}] Retry did not improve ({rate2:.0%}), "
                        f"keeping original"
                    )

            # P3: Enforce heading language
            content = _enforce_heading_language(content, language)

            citation_stats["total_bullets"] += total_b
            citation_stats["cited_bullets"] += cited_b

            sections.append({
                "domain": domain,
                "title": domain.upper(),
                "content": content,
                "item_count": len(items_info),
                "citation_rate": round(
                    (cited_b / total_b * 100) if total_b > 0 else 100.0, 1
                ),
            })

        overall_rate = (
            citation_stats["cited_bullets"] / citation_stats["total_bullets"] * 100
            if citation_stats["total_bullets"] > 0 else 100.0
        )

        return {
            "sections": sections,
            "model": model,
            "domains_generated": len(sections),
            "citation_stats": {
                **citation_stats,
                "overall_rate_pct": round(overall_rate, 1),
            },
            "language": language,
        }

    def _call_llm(self, prompt: str, model: str, ollama_url: str) -> str:
        """Call Ollama LLM for section generation."""
        try:
            import requests
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 3000},
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"*Generation failed: {e}*"

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of generation results with citation stats."""
        cs = data.get("citation_stats", {})
        return (
            f"Generated {data.get('domains_generated', 0)} domain sections "
            f"using {data.get('model', 'unknown')}. "
            f"Citations: {cs.get('overall_rate_pct', 0)}% "
            f"({cs.get('cited_bullets', 0)}/{cs.get('total_bullets', 0)}), "
            f"{cs.get('retries', 0)} retries."
        )
