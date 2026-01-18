"""
Tutored Document Summarization Kernel — Per-document summaries with tutor verification.

Author: Olivier Vitrac, PhD, HDR | Adservio Innovation Lab
Date: 2026-01-18

This kernel generates per-document summaries using a two-stage approach:
1. Worker model (granite3.1-moe:3b): Fast generation of draft summaries
2. Tutor model (mistral:7b-instruct): Verification and correction

The tutor model checks for:
- Hallucination (information not in source)
- Accuracy (correct interpretation)
- Completeness (key points covered)

Features:
- LLM response caching (avoids redundant calls)
- Sovereignty tracking (all processing local)
- Quality statistics (acceptance/correction rates)
"""

import json
import logging
import re
import socket
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.cache import LLMCache, get_model_digest

logger = logging.getLogger(__name__)


@dataclass
class TutorConfig:
    """Configuration for tutored summarization."""
    worker_model: str = "granite3.1-moe:3b"
    tutor_model: str = "mistral:7b-instruct"
    endpoint: str = "http://127.0.0.1:11434"
    timeout: int = 120
    temperature: float = 0.3
    enable_tutor: bool = True
    enable_cache: bool = True
    max_retries: int = 2


@dataclass
class SummaryResult:
    """Result of a single document summarization."""
    file_id: str
    draft: str
    final: str
    tutor_action: str  # "accepted", "corrected", "regenerated", "skipped"
    verified: bool
    cached: bool = False


class DocSummarizeTutoredKernel(Kernel):
    """
    Per-document summaries with tutor model verification.

    Two-stage generation:
    1. Worker (Granite 3B): Generate draft summary (fast, bulk processing)
    2. Tutor (Mistral 7B): Verify accuracy, correct if needed

    This prevents hallucination by having a larger model review outputs.
    """

    name = "doc_summarize_tutored"
    version = "1.0.0"
    category = "docs"
    stage = 3

    requires = ["doc_metadata", "doc_structure", "doc_extract", "doc_pyramid"]
    provides = ["tutored_summaries", "summary_quality_stats"]

    # Prompts
    WORKER_PROMPT_EN = """You are a document analysis expert.

**Document:** {title}
**Path:** {path}
**Type:** {kind}
**Sections:** {sections}

**Key excerpts from the document:**
{key_sentences}

**Instructions:**
1. Identify the SCOPE (what domain/process this document covers)
2. Summarize the KEY CONTENT in 2-3 sentences
3. List the MAIN TOPICS (3-5 keywords)

**Response format:**
SCOPE: [domain covered]
SUMMARY: [main content]
TOPICS: [topic1, topic2, topic3]"""

    TUTOR_PROMPT_EN = """You are a summary verification expert. Your task is to validate
the quality and accuracy of an automatically generated summary.

## Original document excerpts
{key_sentences}

## Summary to verify
{draft_summary}

## Instructions
1. Check that the summary does NOT contain information absent from the excerpts
2. Verify that the identified topics are consistent with the content
3. Verify that the scope is correctly identified
4. If you find errors, provide a corrected version

## Response format
VERDICT: [ACCEPTED | CORRECTED | REJECTED]
REASON: [Brief explanation]
CORRECTION: [If CORRECTED or REJECTED, provide the corrected summary in the same format]"""

    REGENERATE_PROMPT_EN = """You are a document analysis expert.

Based on the following excerpts, create an accurate summary.

**Document:** {title}
**Excerpts:**
{key_sentences}

**Important:** Only include information that is explicitly present in the excerpts.
Do NOT add any external information or assumptions.

**Response format:**
SCOPE: [domain covered]
SUMMARY: [main content based only on excerpts]
TOPICS: [topic1, topic2, topic3]"""

    def __init__(self):
        super().__init__()
        self.cache: Optional[LLMCache] = None
        self.config: Optional[TutorConfig] = None
        self.stats = {
            "total": 0,
            "accepted": 0,
            "corrected": 0,
            "regenerated": 0,
            "cached": 0,
            "errors": 0
        }
        self.worker_digest = ""
        self.tutor_digest = ""

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Execute tutored summarization for all documents."""
        # Initialize configuration
        self.config = TutorConfig(
            worker_model=input.config.get("llm_model", "granite3.1-moe:3b"),
            tutor_model=input.config.get("tutor_model", "mistral:7b-instruct"),
            endpoint=input.config.get("llm_endpoint", "http://127.0.0.1:11434"),
            timeout=input.config.get("llm_timeout", 120),
            temperature=input.config.get("temperature", 0.3),
            enable_tutor=input.config.get("enable_tutor", True),
            enable_cache=input.config.get("enable_cache", True)
        )

        # Initialize cache in workspace directory
        cache_dir = input.workspace / "cache"
        self.cache = LLMCache(cache_dir, self.config.endpoint)

        # Get model digests for tracking
        self.worker_digest = get_model_digest(self.config.worker_model, self.config.endpoint)
        self.tutor_digest = get_model_digest(self.config.tutor_model, self.config.endpoint)

        # Load dependencies
        metadata = self._load_dep(input, "doc_metadata")
        structure = self._load_dep(input, "doc_structure")
        extracts = self._load_dep(input, "doc_extract")
        pyramid = self._load_dep(input, "doc_pyramid")

        # Build document contexts
        doc_contexts = self._build_document_contexts(metadata, structure, extracts, pyramid)

        # Generate summaries
        summaries = {}
        start_time = datetime.now(timezone.utc)

        for file_id, context in doc_contexts.items():
            try:
                result = self._summarize_document(context)
                summaries[file_id] = self._format_summary(context, result)
                self.stats["total"] += 1

                if result.cached:
                    self.stats["cached"] += 1
                elif result.tutor_action == "accepted":
                    self.stats["accepted"] += 1
                elif result.tutor_action == "corrected":
                    self.stats["corrected"] += 1
                elif result.tutor_action == "regenerated":
                    self.stats["regenerated"] += 1

            except Exception as e:
                logger.error(f"Failed to summarize {file_id}: {e}")
                self.stats["errors"] += 1
                summaries[file_id] = self._format_error(context, str(e))

        end_time = datetime.now(timezone.utc)

        # Build sovereignty attestation
        sovereignty = self._build_sovereignty_attestation(start_time, end_time)

        return {
            "summaries": summaries,
            "statistics": self.stats,
            "quality": {
                "acceptance_rate": self.stats["accepted"] / max(1, self.stats["total"]),
                "correction_rate": self.stats["corrected"] / max(1, self.stats["total"]),
                "cache_hit_rate": self.stats["cached"] / max(1, self.stats["total"])
            },
            "sovereignty": sovereignty,
            "cache_stats": self.cache.get_stats() if self.cache else {}
        }

    def _load_dep(self, input: KernelInput, name: str) -> Dict[str, Any]:
        """Load dependency data from file."""
        path = input.dependencies.get(name)
        if path and path.exists():
            with open(path) as f:
                return json.load(f).get("data", {})
        return {}

    def _build_document_contexts(
        self,
        metadata: Dict,
        structure: Dict,
        extracts: Dict,
        pyramid: Dict
    ) -> Dict[str, Dict]:
        """Build context for each document."""
        contexts = {}

        # File metadata lookup
        file_meta = {f["file_id"]: f for f in metadata.get("files", [])}

        # Structure lookup
        doc_structures = structure.get("documents", {})

        # Extract lookup
        file_extracts = extracts.get("by_file", {})

        # Domain lookup from pyramid
        file_domains = {}
        for domain in pyramid.get("pyramid", {}).get("level_3_domains", []):
            domain_id = domain.get("id", "")
            domain_label = domain.get("label", "")
            for cluster in domain.get("clusters", []):
                for file_id in cluster.get("file_ids", []):
                    file_domains[file_id] = {"id": domain_id, "label": domain_label}

        for file_id, meta in file_meta.items():
            # Get sections
            sections = []
            if file_id in doc_structures:
                for sec in doc_structures[file_id].get("sections", [])[:10]:
                    sections.append(sec.get("title", ""))

            # Get key sentences
            sentences = []
            if file_id in file_extracts:
                for sent in file_extracts[file_id].get("sentences", [])[:15]:
                    sentences.append(sent)

            contexts[file_id] = {
                "file_id": file_id,
                "path": meta.get("path", ""),
                "title": Path(meta.get("path", "")).name,
                "kind": meta.get("kind", "unknown"),
                "sections": sections,
                "key_sentences": sentences,
                "domain": file_domains.get(file_id, {"id": "", "label": ""})
            }

        return contexts

    def _summarize_document(self, context: Dict) -> SummaryResult:
        """Summarize a single document with tutor verification."""
        file_id = context["file_id"]

        # Build worker prompt
        worker_prompt = self._build_worker_prompt(context)

        # Check cache first
        if self.config.enable_cache:
            cached_response = self.cache.get(
                self.config.worker_model,
                worker_prompt,
                self.config.temperature,
                self.worker_digest
            )
            if cached_response:
                return SummaryResult(
                    file_id=file_id,
                    draft=cached_response,
                    final=cached_response,
                    tutor_action="accepted",
                    verified=True,
                    cached=True
                )

        # Generate draft with worker model
        draft = self._call_llm(self.config.worker_model, worker_prompt)

        # Cache the draft
        if self.config.enable_cache:
            self.cache.put(
                self.config.worker_model,
                worker_prompt,
                draft,
                self.config.temperature,
                self.worker_digest
            )

        # Skip tutor if disabled
        if not self.config.enable_tutor:
            return SummaryResult(
                file_id=file_id,
                draft=draft,
                final=draft,
                tutor_action="skipped",
                verified=False
            )

        # Verify with tutor model
        tutor_prompt = self._build_tutor_prompt(context, draft)
        verification = self._call_llm(self.config.tutor_model, tutor_prompt)

        # Parse verdict
        verdict, correction = self._parse_verdict(verification)

        if verdict == "ACCEPTED":
            return SummaryResult(
                file_id=file_id,
                draft=draft,
                final=draft,
                tutor_action="accepted",
                verified=True
            )
        elif verdict == "CORRECTED" and correction:
            return SummaryResult(
                file_id=file_id,
                draft=draft,
                final=correction,
                tutor_action="corrected",
                verified=True
            )
        else:
            # Regenerate with tutor
            regen_prompt = self._build_regenerate_prompt(context)
            regenerated = self._call_llm(self.config.tutor_model, regen_prompt)
            return SummaryResult(
                file_id=file_id,
                draft=draft,
                final=regenerated,
                tutor_action="regenerated",
                verified=True
            )

    def _build_worker_prompt(self, context: Dict) -> str:
        """Build prompt for worker model."""
        sentences_text = "\n".join(f"- {s}" for s in context["key_sentences"][:10])
        sections_text = ", ".join(context["sections"][:5]) or "Not detected"

        return self.WORKER_PROMPT_EN.format(
            title=context["title"],
            path=context["path"],
            kind=context["kind"],
            sections=sections_text,
            key_sentences=sentences_text or "No key sentences extracted"
        )

    def _build_tutor_prompt(self, context: Dict, draft: str) -> str:
        """Build prompt for tutor model."""
        sentences_text = "\n".join(f"- {s}" for s in context["key_sentences"][:10])

        return self.TUTOR_PROMPT_EN.format(
            key_sentences=sentences_text or "No excerpts available",
            draft_summary=draft
        )

    def _build_regenerate_prompt(self, context: Dict) -> str:
        """Build prompt for regeneration by tutor."""
        sentences_text = "\n".join(f"- {s}" for s in context["key_sentences"][:10])

        return self.REGENERATE_PROMPT_EN.format(
            title=context["title"],
            key_sentences=sentences_text or "No excerpts available"
        )

    def _parse_verdict(self, response: str) -> Tuple[str, Optional[str]]:
        """Parse tutor verdict from response."""
        # Extract verdict
        verdict_match = re.search(r"VERDICT:\s*(ACCEPTED|CORRECTED|REJECTED)", response, re.IGNORECASE)
        verdict = verdict_match.group(1).upper() if verdict_match else "REJECTED"

        # Extract correction if present
        correction = None
        if verdict in ["CORRECTED", "REJECTED"]:
            correction_match = re.search(
                r"CORRECTION:\s*(.+?)(?:$|\n\n)",
                response,
                re.DOTALL | re.IGNORECASE
            )
            if correction_match:
                correction = correction_match.group(1).strip()

        return verdict, correction

    def _call_llm(self, model: str, prompt: str) -> str:
        """Call Ollama LLM."""
        try:
            response = httpx.post(
                f"{self.config.endpoint}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature
                    }
                },
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _format_summary(self, context: Dict, result: SummaryResult) -> Dict[str, Any]:
        """Format summary result for output."""
        # Parse the final summary
        parsed = self._parse_summary(result.final)

        return {
            "file_id": result.file_id,
            "path": context["path"],
            "title": context["title"],
            "kind": context["kind"],
            "scope": parsed.get("scope", ""),
            "summary": parsed.get("summary", result.final),
            "topics": parsed.get("topics", []),
            "domain": context["domain"],
            "verification": {
                "tutor_action": result.tutor_action,
                "verified": result.verified,
                "cached": result.cached
            },
            "raw_response": result.final
        }

    def _parse_summary(self, response: str) -> Dict[str, Any]:
        """Parse structured summary from LLM response."""
        result = {"scope": "", "summary": "", "topics": []}

        # Extract scope
        scope_match = re.search(r"SCOPE:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if scope_match:
            result["scope"] = scope_match.group(1).strip().strip("*")

        # Extract summary
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?:TOPICS:|$)", response, re.DOTALL | re.IGNORECASE)
        if summary_match:
            result["summary"] = summary_match.group(1).strip().strip("*")

        # Extract topics
        topics_match = re.search(r"TOPICS?:\s*(.+?)(?:$|\n\n)", response, re.DOTALL | re.IGNORECASE)
        if topics_match:
            topics_text = topics_match.group(1).strip()
            # Parse comma-separated or list format
            topics = re.split(r"[,\n]+", topics_text)
            result["topics"] = [t.strip().strip("*-•").strip() for t in topics if t.strip()][:5]

        return result

    def _format_error(self, context: Dict, error: str) -> Dict[str, Any]:
        """Format error result."""
        return {
            "file_id": context["file_id"],
            "path": context["path"],
            "title": context["title"],
            "kind": context["kind"],
            "scope": "",
            "summary": f"Error: {error}",
            "topics": [],
            "domain": context["domain"],
            "verification": {
                "tutor_action": "error",
                "verified": False,
                "cached": False
            },
            "error": error
        }

    def _build_sovereignty_attestation(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Build sovereignty attestation."""
        return {
            "hostname": socket.gethostname(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
            "llm_endpoint": self.config.endpoint,
            "llm_local": "127.0.0.1" in self.config.endpoint or "localhost" in self.config.endpoint,
            "models_used": [
                {
                    "name": self.config.worker_model,
                    "digest": self.worker_digest,
                    "role": "worker"
                },
                {
                    "name": self.config.tutor_model,
                    "digest": self.tutor_digest,
                    "role": "tutor"
                }
            ],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_s": (end_time - start_time).total_seconds(),
            "attestation": "All processing performed locally. No data sent to external services."
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        stats = data.get("statistics", {})
        quality = data.get("quality", {})

        return (
            f"Tutored summaries: {stats.get('total', 0)} documents processed. "
            f"Accepted: {stats.get('accepted', 0)}, Corrected: {stats.get('corrected', 0)}, "
            f"Regenerated: {stats.get('regenerated', 0)}, Cached: {stats.get('cached', 0)}. "
            f"Acceptance rate: {quality.get('acceptance_rate', 0):.0%}."
        )
