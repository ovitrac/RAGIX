"""
summary_capabilities — Stage 3: Capabilities Manifest

Emits an auto-discoverable JSON manifest describing the pipeline's
kernels, features, supported tiers, and configuration schema.

Deterministic, no LLM, no external dependencies.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)


class SummaryCapabilitiesKernel(Kernel):
    name = "summary_capabilities"
    version = "1.0.0"
    category = "summary"
    stage = 3
    description = "Emit autodiscoverable capabilities manifest"
    requires = []  # independent
    provides = ["capabilities"]

    # Kernel registry mirrors summaryctl._KERNEL_MAP
    _PIPELINE_KERNELS = [
        {"name": "summary_collect",         "stage": 1, "description": "Corpus collection"},
        {"name": "summary_ingest",          "stage": 1, "description": "Rule extraction + memory storage"},
        {"name": "summary_build_graph",     "stage": 2, "description": "Graph-RAG edge construction"},
        {"name": "summary_consolidate",     "stage": 2, "description": "Cluster, merge, promote"},
        {"name": "summary_budgeted_recall", "stage": 2, "description": "Domain-balanced retrieval"},
        {"name": "summary_generate",        "stage": 3, "description": "LLM-assisted summary generation"},
        {"name": "summary_verify",          "stage": 3, "description": "Citation + consistency verification"},
        {"name": "summary_redact",          "stage": 3, "description": "Secrecy-tier redaction"},
        {"name": "summary_capabilities",    "stage": 3, "description": "Capabilities manifest"},
        {"name": "summary_report",          "stage": 3, "description": "Final report assembly"},
    ]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Emit autodiscoverable JSON manifest of pipeline kernels and features."""
        cfg = input.config

        # Check if graph is available
        graph_available = False
        graph_stats_path = input.workspace / "stage2" / "graph_stats.json"
        if graph_stats_path.exists():
            graph_available = True

        # Build manifest
        manifest = {
            "pipeline": "koas-summary",
            "version": "2.1.0",
            "kernels": self._PIPELINE_KERNELS,
            "capabilities": {
                "graph_rag": graph_available,
                "secrecy_tiers": ["S0", "S2", "S3"],
                "languages": ["French", "English"],
                "embedders": ["ollama", "sentence-transformers", "mock"],
                "chunking_strategies": ["pages", "headings", "windows"],
            },
            "config_keys": {
                "scope": "Memory scope label (e.g. grdf-rie)",
                "model": "LLM model for extraction/generation",
                "embedder_backend": "Embedding backend (mock/ollama/sentence-transformers)",
                "graph.enabled": "Enable graph-assisted consolidation",
                "graph.neighborhood_depth": "BFS depth for merge candidates",
                "graph.similarity_edge_threshold": "Cosine threshold for similar edges",
                "secrecy.tier": "Report redaction tier (S0/S2/S3)",
                "language": "Output language for headings and structure",
            },
        }

        # Write artifact
        stage3 = input.workspace / "stage3"
        stage3.mkdir(parents=True, exist_ok=True)
        artifact_path = stage3 / "capabilities.json"
        with open(artifact_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Capabilities manifest → {artifact_path}")
        return manifest

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of capabilities manifest."""
        n = len(data.get("kernels", []))
        caps = data.get("capabilities", {})
        return (
            f"Pipeline v{data.get('version', '?')}: {n} kernels, "
            f"graph_rag={caps.get('graph_rag', False)}, "
            f"tiers={caps.get('secrecy_tiers', [])}"
        )
