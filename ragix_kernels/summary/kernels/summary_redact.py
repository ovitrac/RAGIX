"""
summary_redact — Stage 3: Secrecy-Tier Redaction

Applies report-time redaction based on secrecy tiers:
  S0 = public:   redact paths, emails, hostnames, IPs
  S2 = internal: redact hostnames, IPs only
  S3 = audit:    no redaction (full detail preserved)

Deterministic, no LLM. Processes both summary.md and summary.json.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ragix_kernels.base import Kernel, KernelInput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redaction patterns
# ---------------------------------------------------------------------------

_REDACT_PATTERNS = {
    "paths": re.compile(r'(?:/[\w.-]+){2,}|[A-Z]:\\[\w\\.-]+'),
    "emails": re.compile(r'[\w.+-]+@[\w.-]+\.\w{2,}'),
    "hostnames": re.compile(
        r'\b[\w-]+\.(?:corp|internal|local|intra|lan|'
        r'company|enterprise|office|srv|server)\b',
        re.IGNORECASE,
    ),
    "ips": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    # Extended patterns (feedback-driven)
    "filenames": re.compile(r'\b[\w.-]+\.(?:pdf|docx?|xlsx?|pptx?|csv|txt|log|cfg|conf|yaml|yml|xml)\b', re.I),
    "pointer_ids": re.compile(r'\bMEM-[0-9a-f]{8,}\b', re.I),
    "hashes": re.compile(r'\bsha256:[0-9a-f]{16,}\b'),
    "entity_labels": re.compile(
        # Infrastructure-revealing entity labels (server names, internal services)
        r'\b[\w-]+\.(?:corp|internal|local|intra|lan)\b'
        r'|\b(?:srv|db|app|web|api|mgmt|admin)-\d{1,3}\b',
        re.IGNORECASE,
    ),
}

# Which patterns to apply per tier
# Full secrecy table:
#   Artifact        S0          S2          S3
#   paths           REDACT      keep        keep
#   emails          REDACT      keep        keep
#   hostnames       REDACT      REDACT      keep
#   ips             REDACT      REDACT      keep
#   filenames       REDACT      keep        keep
#   pointer_ids     REDACT      keep        keep
#   hashes          REDACT      REDACT      keep
#   entity_labels   REDACT      keep        keep
_TIER_RULES: Dict[str, List[str]] = {
    "S0": [
        "paths", "emails", "hostnames", "ips",
        "filenames", "pointer_ids", "hashes", "entity_labels",
    ],
    "S2": ["hostnames", "ips", "hashes"],
    "S3": [],
}

_REPLACEMENTS = {
    "paths": "[PATH]",
    "emails": "[EMAIL]",
    "hostnames": "[HOST]",
    "ips": "[IP]",
    "filenames": "[FILE]",
    "pointer_ids": "[MEM-ID]",
    "hashes": "[HASH]",
    "entity_labels": "[ENTITY]",
}


# ---------------------------------------------------------------------------
# Write-time redaction utility (called before storing canonical items)
# ---------------------------------------------------------------------------

def redact_for_storage(text: str, tier: str = "S3") -> str:
    """
    Redact sensitive content before writing to memory store.

    Called at merge/write time to ensure the canonical item
    does not contain sensitive strings that would leak on export.

    Provenance pointers (source_id, chunk_ids) are NOT redacted —
    only literal content strings.
    """
    if tier == "S3":
        return text  # audit tier — no redaction

    rules = _TIER_RULES.get(tier, [])
    for rule_name in rules:
        if rule_name in _REDACT_PATTERNS:
            text = _REDACT_PATTERNS[rule_name].sub(_REPLACEMENTS[rule_name], text)
    return text


class SummaryRedactKernel(Kernel):
    name = "summary_redact"
    version = "1.0.0"
    category = "summary"
    stage = 3
    description = "Apply secrecy-tier redaction to generated summary"
    requires = ["summary_generate"]
    provides = ["redacted_summary"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        """Apply secrecy-tier redaction to summary.md and summary.json artifacts."""
        cfg = input.config
        secrecy_cfg = cfg.get("secrecy", {})
        tier = secrecy_cfg.get("tier", cfg.get("secrecy_tier", "S3"))
        custom_patterns = secrecy_cfg.get("custom_patterns", [])

        stage3 = input.workspace / "stage3"
        summary_md = stage3 / "summary.md"
        summary_json = stage3 / "summary.json"

        result = {
            "tier": tier,
            "redactions": {},
            "total_redactions": 0,
            "artifacts": {},
        }

        # S3 = no-op
        if tier == "S3":
            logger.info("Secrecy tier S3: no redaction needed")
            result["artifacts"]["note"] = "S3 tier — no redaction applied"
            return result

        rules = _TIER_RULES.get(tier, [])
        if not rules and not custom_patterns:
            logger.info(f"Secrecy tier {tier}: no applicable rules")
            return result

        # Build active patterns
        active_patterns: List[Tuple[str, re.Pattern, str]] = []
        for rule_name in rules:
            if rule_name in _REDACT_PATTERNS:
                active_patterns.append(
                    (rule_name, _REDACT_PATTERNS[rule_name], _REPLACEMENTS[rule_name])
                )
        for i, pat_str in enumerate(custom_patterns):
            try:
                active_patterns.append(
                    (f"custom_{i}", re.compile(pat_str), "[REDACTED]")
                )
            except re.error as e:
                logger.warning(f"Invalid custom pattern '{pat_str}': {e}")

        # Redact summary.md
        if summary_md.exists():
            text = summary_md.read_text(encoding="utf-8")
            redacted_text, counts = self._redact_text(text, active_patterns)
            result["redactions"]["md"] = counts
            result["total_redactions"] += sum(counts.values())

            out_md = stage3 / f"summary_{tier}.md"
            out_md.write_text(redacted_text, encoding="utf-8")
            result["artifacts"]["md"] = str(out_md)
            logger.info(f"Redacted MD: {sum(counts.values())} replacements → {out_md}")
        else:
            logger.warning(f"summary.md not found at {summary_md}")

        # Redact summary.json
        if summary_json.exists():
            json_text = summary_json.read_text(encoding="utf-8")
            redacted_json, counts = self._redact_text(json_text, active_patterns)
            result["redactions"]["json"] = counts
            result["total_redactions"] += sum(counts.values())

            out_json = stage3 / f"summary_{tier}.json"
            out_json.write_text(redacted_json, encoding="utf-8")
            result["artifacts"]["json"] = str(out_json)

        # Write redaction report
        report_path = stage3 / "redaction_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        result["artifacts"]["report"] = str(report_path)

        return result

    @staticmethod
    def _redact_text(
        text: str,
        patterns: List[Tuple[str, re.Pattern, str]],
    ) -> Tuple[str, Dict[str, int]]:
        """Apply redaction patterns to text, return (redacted_text, counts)."""
        counts: Dict[str, int] = {}
        for name, pat, replacement in patterns:
            matches = pat.findall(text)
            counts[name] = len(matches)
            if matches:
                text = pat.sub(replacement, text)
        return text, counts

    def summarize(self, data: Dict[str, Any]) -> str:
        """Format one-line summary of redaction results."""
        tier = data.get("tier", "?")
        total = data.get("total_redactions", 0)
        if tier == "S3":
            return f"Secrecy tier {tier}: no redaction applied."
        return f"Secrecy tier {tier}: {total} redactions applied."
