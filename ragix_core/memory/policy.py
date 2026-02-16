"""
Write Governance — Memory Policy Engine

Evaluates memory proposals before storage. Two levels:
- Hard blocks: secrets, injection patterns, oversized content, missing provenance
- Soft blocks: low confidence → quarantine to STM with expiry

Never bypassed. Optional auditor LLM can refine type/tags but cannot override hard blocks.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from ragix_core.memory.config import PolicyConfig
from ragix_core.memory.types import MemoryItem, MemoryProposal, _now_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

PolicyAction = Literal["accept", "quarantine", "reject"]


@dataclass
class PolicyVerdict:
    """Result of policy evaluation on a proposal."""

    action: PolicyAction
    reasons: List[str] = field(default_factory=list)
    # If quarantined, these overrides are applied
    forced_tier: Optional[str] = None
    forced_validation: Optional[str] = None
    forced_expires_at: Optional[str] = None
    # V3.3: if True, item.injectable is set to False (excluded from recall/inject)
    forced_non_injectable: bool = False

    @property
    def accepted(self) -> bool:
        """Return True if the verdict is accept."""
        return self.action == "accept"

    @property
    def rejected(self) -> bool:
        """Return True if the verdict is reject."""
        return self.action == "reject"


# ---------------------------------------------------------------------------
# Pattern sets
# ---------------------------------------------------------------------------

# Secrets detection patterns (conservative)
_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", re.IGNORECASE),
    re.compile(r"-----BEGIN\s+PEM-----", re.IGNORECASE),
    re.compile(r"-----BEGIN\s+CERTIFICATE-----", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|apikey)\s*[:=]\s*\S{8,}", re.IGNORECASE),
    re.compile(r"(?:secret|token|password|passwd|pwd)\s*[:=]\s*\S{8,}", re.IGNORECASE),
    re.compile(r"(?:aws_access_key_id|aws_secret_access_key)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{36,}", re.IGNORECASE),  # GitHub PAT
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),    # OpenAI-style key
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}", re.IGNORECASE),  # JWT
    re.compile(r"[A-Za-z0-9+/]{60,}={1,2}", re.IGNORECASE),  # long base64 with padding (> 60 chars)
]

# Injection / prompt override patterns
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+", re.IGNORECASE),
    re.compile(r"store\s+this\s+(?:as\s+)?(?:a\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"override\s+(?:system|safety|security)", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[\s*SYSTEM\s*\]", re.IGNORECASE),
    re.compile(r"pretend\s+(?:to\s+be|you\s+are)", re.IGNORECASE),
]

# V3.3: Instructional-content patterns (§7.2)
# Items matching BLOCK patterns are rejected; QUARANTINE patterns are stored
# with injectable=False (excluded from recall/inject, visible in search).

# BLOCK: system/role fragments, tool invocation syntax, JSON payloads, MCP fragments
_INSTRUCTIONAL_BLOCK_PATTERNS = [
    re.compile(r"you\s+are\s+(?:Chat\s*GPT|Claude|GPT|Gemini|an?\s+AI)", re.IGNORECASE),
    re.compile(r"(?:^|\n)(?:System|Developer|Assistant|Human)\s*:", re.IGNORECASE),
    re.compile(r"(?:use|call|invoke|run)\s+memory_\w+", re.IGNORECASE),
    re.compile(r"(?:use|call|invoke|run)\s+(?:the\s+)?(?:tool|function)\s+", re.IGNORECASE),
    re.compile(r"\{\s*\"(?:tool_name|action|function_call|tool_use)\"\s*:", re.IGNORECASE),
    re.compile(r"\{\s*\"(?:parameters|arguments|params)\"\s*:\s*\{", re.IGNORECASE),
    re.compile(r"<\s*(?:tool_use|tool_result|result|function_call)\s*>", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(?:tool_use|tool_result|result|function_call)\s*>", re.IGNORECASE),
]

# QUARANTINE: imperative self-instructions (stored but injectable=False)
_INSTRUCTIONAL_QUARANTINE_PATTERNS = [
    re.compile(r"(?:always|never)\s+(?:remember|forget)\s+(?:to\s+)?", re.IGNORECASE),
    re.compile(r"in\s+(?:future|subsequent|later)\s+(?:sessions?|conversations?|turns?)", re.IGNORECASE),
    re.compile(r"(?:you\s+)?(?:must|should|shall)\s+(?:always|never)\s+", re.IGNORECASE),
    re.compile(r"(?:from\s+now\s+on|henceforth|going\s+forward)\s*[,.]?\s+", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class MemoryPolicy:
    """
    Evaluates memory proposals against hard and soft rules.

    Hard blocks (reject):
    - Secret patterns detected in content/title
    - Injection-like text
    - Content exceeds max length (must use pointer type)
    - Missing provenance for MTM/LTM tier

    Soft blocks (quarantine → STM with expiry):
    - Low confidence
    - No evidence / no provenance source_id
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        """Initialize policy engine with detection configuration."""
        self._config = config or PolicyConfig()

    def evaluate_proposal(self, proposal: MemoryProposal) -> PolicyVerdict:
        """Evaluate a memory proposal. Returns verdict with action and reasons."""
        reasons: List[str] = []
        action: PolicyAction = "accept"

        text = f"{proposal.title} {proposal.content}"

        # --- Hard blocks ---
        if self._config.secret_patterns_enabled:
            secret_hits = self._check_secrets(text)
            if secret_hits:
                reasons.extend(secret_hits)
                action = "reject"

        if self._config.injection_patterns_enabled:
            inject_hits = self._check_injection(text)
            if inject_hits:
                reasons.extend(inject_hits)
                action = "reject"

        # V3.3: Instructional-content block patterns
        if self._config.instructional_content_enabled:
            inst_block_hits = self._check_instructional_block(text)
            if inst_block_hits:
                reasons.extend(inst_block_hits)
                action = "reject"

        # Oversized content (must use pointer type)
        if (
            len(proposal.content) > self._config.max_content_length
            and proposal.type != "pointer"
        ):
            reasons.append(
                f"Content too long ({len(proposal.content)} chars > "
                f"{self._config.max_content_length}); use type='pointer'"
            )
            action = "reject"

        if action == "reject":
            return PolicyVerdict(action="reject", reasons=reasons)

        # --- Soft blocks ---
        quarantine_reasons: List[str] = []
        force_non_injectable = False

        # V3.3: Instructional-content quarantine patterns
        if self._config.instructional_content_enabled:
            inst_quarantine_hits = self._check_instructional_quarantine(text)
            if inst_quarantine_hits:
                quarantine_reasons.extend(inst_quarantine_hits)
                force_non_injectable = True

        # Low confidence
        if proposal.why_store == "":
            quarantine_reasons.append("Missing why_store justification")

        # No provenance source_id
        prov = proposal.provenance_hint or {}
        if not prov.get("source_id"):
            quarantine_reasons.append("Missing provenance source_id")

        if quarantine_reasons:
            expiry = datetime.now(timezone.utc) + timedelta(
                hours=self._config.quarantine_expiry_hours
            )
            return PolicyVerdict(
                action="quarantine",
                reasons=quarantine_reasons,
                forced_tier="stm",
                forced_validation="unverified",
                forced_expires_at=expiry.isoformat(),
                forced_non_injectable=force_non_injectable,
            )

        return PolicyVerdict(action="accept", reasons=[])

    def evaluate_item(self, item: MemoryItem) -> PolicyVerdict:
        """
        Evaluate an existing MemoryItem (for direct writes or tier promotion).
        """
        reasons: List[str] = []
        action: PolicyAction = "accept"

        text = f"{item.title} {item.content}"

        # Hard blocks
        if self._config.secret_patterns_enabled:
            secret_hits = self._check_secrets(text)
            if secret_hits:
                reasons.extend(secret_hits)
                action = "reject"

        if self._config.injection_patterns_enabled:
            inject_hits = self._check_injection(text)
            if inject_hits:
                reasons.extend(inject_hits)
                action = "reject"

        # V3.3: Instructional-content block patterns
        if self._config.instructional_content_enabled:
            inst_block_hits = self._check_instructional_block(text)
            if inst_block_hits:
                reasons.extend(inst_block_hits)
                action = "reject"

        if (
            len(item.content) > self._config.max_content_length
            and item.type != "pointer"
        ):
            reasons.append("Content too long for non-pointer type")
            action = "reject"

        # Provenance required for MTM/LTM
        if item.tier in self._config.require_provenance_for:
            if not item.provenance.source_id:
                reasons.append(
                    f"Provenance source_id required for tier={item.tier}"
                )
                action = "reject"

        if action == "reject":
            return PolicyVerdict(action="reject", reasons=reasons)

        return PolicyVerdict(action="accept", reasons=[])

    # -- Pattern checks ----------------------------------------------------

    def _check_secrets(self, text: str) -> List[str]:
        """Check for secret patterns. Returns list of match descriptions."""
        hits = []
        for i, pattern in enumerate(_SECRET_PATTERNS):
            if pattern.search(text):
                hits.append(f"HARD_BLOCK: secret pattern #{i} matched")
        return hits

    def _check_injection(self, text: str) -> List[str]:
        """Check for injection patterns. Returns list of match descriptions."""
        hits = []
        for i, pattern in enumerate(_INJECTION_PATTERNS):
            if pattern.search(text):
                hits.append(f"HARD_BLOCK: injection pattern #{i} matched")
        return hits

    def _check_instructional_block(self, text: str) -> List[str]:
        """Check for instructional-content BLOCK patterns (tool syntax, system fragments)."""
        hits = []
        for i, pattern in enumerate(_INSTRUCTIONAL_BLOCK_PATTERNS):
            if pattern.search(text):
                hits.append(f"HARD_BLOCK: instructional_content pattern #{i} matched")
        return hits

    def _check_instructional_quarantine(self, text: str) -> List[str]:
        """Check for instructional-content QUARANTINE patterns (imperative self-instructions)."""
        hits = []
        for i, pattern in enumerate(_INSTRUCTIONAL_QUARANTINE_PATTERNS):
            if pattern.search(text):
                hits.append(f"QUARANTINE: instructional_self_instruction pattern #{i} matched")
        return hits
