"""
Kernel: Document Summarize
Stage: 3 (Synthesis)

Generates LLM-based summaries for each individual document.
This addresses the core requirement of producing concise summaries
outlining scope and key content for each document.

Uses local LLM (Ollama/Granite) to generate summaries from:
- Document metadata (title, type, size)
- Extracted structure (sections, headings)
- Key sentences/extracts

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-01-18
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import logging
import json
import time
import re

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class DocSummarizeKernel(Kernel):
    """
    Generate per-document summaries using local LLM.

    This kernel produces a concise summary for each document including:
    - Document scope (what domain/process it covers)
    - Key content summary (2-3 sentences)
    - Main topics identified
    - Document type classification

    Configuration options:
        project.path: Path to the indexed project (required)
        llm_model: Ollama model name (default: "granite3.1-moe:3b")
        language: Output language "fr" or "en" (default: "fr")
        max_summary_length: Max chars for summary (default: 500)
        batch_size: Documents per batch for progress (default: 10)
        skip_types: File types to skip (default: [])

    Dependencies:
        doc_metadata: File inventory
        doc_structure: Document sections
        doc_extract: Key sentences
        doc_pyramid: Hierarchical context

    Output:
        summaries: Per-document summaries keyed by file_id
        by_domain: Summaries grouped by domain
        statistics: Generation statistics
    """

    name = "doc_summarize"
    version = "1.0.0"
    category = "docs"
    stage = 3
    description = "Generate LLM-based per-document summaries"

    requires = ["doc_metadata", "doc_structure", "doc_extract", "doc_pyramid"]
    provides = ["doc_summaries", "summaries_by_domain"]

    # Prompt templates
    PROMPT_FR = """Tu es un expert en analyse documentaire. Résume ce document de manière concise.

**Document:** {title}
**Type:** {doc_type}
**Sections:** {sections}

**Extraits clés:**
{extracts}

**Instructions:**
1. Identifie le PÉRIMÈTRE (quel domaine/processus ce document couvre)
2. Résume le CONTENU CLÉ en 2-3 phrases
3. Liste les THÈMES PRINCIPAUX (3-5 mots-clés)

**Format de réponse:**
PÉRIMÈTRE: [domaine couvert]
RÉSUMÉ: [contenu principal]
THÈMES: [mot1, mot2, mot3]"""

    PROMPT_EN = """You are a document analysis expert. Summarize this document concisely.

**Document:** {title}
**Type:** {doc_type}
**Sections:** {sections}

**Key extracts:**
{extracts}

**Instructions:**
1. Identify the SCOPE (what domain/process this document covers)
2. Summarize the KEY CONTENT in 2-3 sentences
3. List MAIN TOPICS (3-5 keywords)

**Response format:**
SCOPE: [domain covered]
SUMMARY: [main content]
TOPICS: [word1, word2, word3]"""

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Generate per-document summaries."""
        import ollama

        # Get configuration
        project_config = input.config.get("project", {})
        project_path_str = project_config.get("path")

        if not project_path_str:
            raise RuntimeError("Missing required config: project.path")

        llm_model = input.config.get("llm_model", "granite3.1-moe:3b")
        language = input.config.get("language", "fr")
        max_summary_length = input.config.get("max_summary_length", 500)
        batch_size = input.config.get("batch_size", 10)
        skip_types = set(input.config.get("skip_types", []))

        prompt_template = self.PROMPT_FR if language == "fr" else self.PROMPT_EN

        logger.info(f"[doc_summarize] Generating summaries with {llm_model}")

        # Load dependencies
        metadata_path = input.dependencies.get("doc_metadata")
        structure_path = input.dependencies.get("doc_structure")
        extract_path = input.dependencies.get("doc_extract")
        pyramid_path = input.dependencies.get("doc_pyramid")

        if not all(p and p.exists() for p in [metadata_path, structure_path, extract_path]):
            raise RuntimeError("Missing required dependencies")

        with open(metadata_path) as f:
            metadata_data = json.load(f).get("data", {})
        with open(structure_path) as f:
            structure_data = json.load(f).get("data", {})
        with open(extract_path) as f:
            extract_data = json.load(f).get("data", {})

        # Load pyramid for domain context (optional)
        domain_for_file = {}
        if pyramid_path and pyramid_path.exists():
            with open(pyramid_path) as f:
                pyramid_data = json.load(f).get("data", {}).get("pyramid", {})
            for domain in pyramid_data.get("level_3_domains", []):
                domain_id = domain.get("domain_id")
                domain_label = domain.get("label", domain_id)
                for fid in domain.get("file_ids", []):
                    domain_for_file[fid] = {"id": domain_id, "label": domain_label}

        # Build file info
        files = {f["file_id"]: f for f in metadata_data.get("files", [])}
        structures = structure_data.get("documents", {})
        extracts = extract_data.get("by_file", {})

        # Generate summaries
        summaries: Dict[str, Dict[str, Any]] = {}
        errors: List[Dict[str, str]] = []
        by_domain: Dict[str, List[str]] = defaultdict(list)

        start_time = time.time()
        total_files = len(files)
        processed = 0

        for file_id, file_info in files.items():
            file_path = file_info.get("path", "")
            file_kind = file_info.get("kind", "unknown")

            # Skip certain types if configured
            if file_kind in skip_types:
                continue

            # Get document title from path
            title = Path(file_path).name

            # Get sections
            doc_structure = structures.get(file_id, {})
            sections = doc_structure.get("sections", [])
            section_titles = [s.get("title", "")[:50] for s in sections[:10]]
            sections_text = ", ".join(section_titles) if section_titles else "Non structuré"

            # Get extracts
            doc_extracts = extracts.get(file_id, {})
            sentences = doc_extracts.get("sentences", [])
            extract_texts = [s.get("text", "")[:200] for s in sentences[:5]]
            extracts_text = "\n".join(f"- {t}" for t in extract_texts) if extract_texts else "Aucun extrait"

            # Build prompt
            prompt = prompt_template.format(
                title=title,
                doc_type=self._format_doc_type(file_kind, language),
                sections=sections_text,
                extracts=extracts_text
            )

            # Call LLM
            try:
                response = ollama.generate(
                    model=llm_model,
                    prompt=prompt,
                    options={
                        "temperature": 0.3,
                        "num_predict": 300,
                    }
                )
                raw_response = response.get("response", "")

                # Parse response
                parsed = self._parse_summary(raw_response, language)

                summary_entry = {
                    "file_id": file_id,
                    "path": file_path,
                    "title": title,
                    "kind": file_kind,
                    "scope": parsed.get("scope", ""),
                    "summary": parsed.get("summary", "")[:max_summary_length],
                    "topics": parsed.get("topics", []),
                    "domain": domain_for_file.get(file_id, {}),
                    "raw_response": raw_response,
                }
                summaries[file_id] = summary_entry

                # Index by domain
                domain_id = domain_for_file.get(file_id, {}).get("id", "unknown")
                by_domain[domain_id].append(file_id)

            except Exception as e:
                logger.warning(f"[doc_summarize] Error for {file_id}: {e}")
                errors.append({"file_id": file_id, "error": str(e)})

            processed += 1
            if processed % batch_size == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (total_files - processed) / rate if rate > 0 else 0
                logger.info(
                    f"[doc_summarize] Progress: {processed}/{total_files} "
                    f"({rate:.1f}/s, ETA: {eta:.0f}s)"
                )

        elapsed_total = time.time() - start_time

        # Statistics
        statistics = {
            "total_files": total_files,
            "summarized": len(summaries),
            "errors": len(errors),
            "domains": len(by_domain),
            "elapsed_seconds": round(elapsed_total, 1),
            "rate_per_second": round(len(summaries) / elapsed_total, 2) if elapsed_total > 0 else 0,
        }

        logger.info(
            f"[doc_summarize] Generated {len(summaries)} summaries "
            f"in {elapsed_total:.1f}s ({statistics['rate_per_second']}/s)"
        )

        return {
            "summaries": summaries,
            "by_domain": dict(by_domain),
            "errors": errors,
            "statistics": statistics,
        }

    def _format_doc_type(self, kind: str, language: str) -> str:
        """Format document type for display."""
        type_map_fr = {
            "doc_docx": "Document Word",
            "doc_pdf": "Document PDF",
            "doc_xlsx": "Classeur Excel",
            "doc_pptx": "Présentation PowerPoint",
            "doc_txt": "Fichier texte",
            "doc_md": "Document Markdown",
        }
        type_map_en = {
            "doc_docx": "Word document",
            "doc_pdf": "PDF document",
            "doc_xlsx": "Excel spreadsheet",
            "doc_pptx": "PowerPoint presentation",
            "doc_txt": "Text file",
            "doc_md": "Markdown document",
        }
        type_map = type_map_fr if language == "fr" else type_map_en
        return type_map.get(kind, kind)

    def _parse_summary(self, response: str, language: str) -> Dict[str, Any]:
        """Parse LLM response into structured summary."""
        result = {
            "scope": "",
            "summary": "",
            "topics": [],
        }

        # Define patterns based on language
        if language == "fr":
            scope_pattern = r"PÉRIMÈTRE\s*:\s*(.+?)(?=RÉSUMÉ|$)"
            summary_pattern = r"RÉSUMÉ\s*:\s*(.+?)(?=THÈMES|$)"
            topics_pattern = r"THÈMES\s*:\s*(.+?)$"
        else:
            scope_pattern = r"SCOPE\s*:\s*(.+?)(?=SUMMARY|$)"
            summary_pattern = r"SUMMARY\s*:\s*(.+?)(?=TOPICS|$)"
            topics_pattern = r"TOPICS\s*:\s*(.+?)$"

        # Extract fields
        scope_match = re.search(scope_pattern, response, re.IGNORECASE | re.DOTALL)
        if scope_match:
            result["scope"] = scope_match.group(1).strip()

        summary_match = re.search(summary_pattern, response, re.IGNORECASE | re.DOTALL)
        if summary_match:
            result["summary"] = summary_match.group(1).strip()

        topics_match = re.search(topics_pattern, response, re.IGNORECASE | re.DOTALL)
        if topics_match:
            topics_str = topics_match.group(1).strip()
            # Parse comma or bracket-separated topics
            topics_str = re.sub(r"[\[\]]", "", topics_str)
            topics = [t.strip() for t in re.split(r"[,;]", topics_str) if t.strip()]
            result["topics"] = topics[:10]  # Limit to 10

        # Fallback: if no structure found, use whole response
        if not result["summary"] and response:
            result["summary"] = response[:500]

        return result

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        total = stats.get("summarized", 0)
        errors = stats.get("errors", 0)
        elapsed = stats.get("elapsed_seconds", 0)
        rate = stats.get("rate_per_second", 0)

        return (
            f"Summaries: {total} documents summarized in {elapsed:.0f}s "
            f"({rate:.1f}/s). {errors} errors."
        )
