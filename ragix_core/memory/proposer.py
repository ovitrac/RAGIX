"""
LLM Output Parser for Memory Proposals

Extracts memory.propose tool calls from LLM responses.
Two strategies:
  A) Tool-only side channel (structured tool calls)
  B) Delimiter block parsing (<MEMORY_PROPOSALS_JSON>...</MEMORY_PROPOSALS_JSON>)

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ragix_core.memory.config import ProposerConfig
from ragix_core.memory.types import MemoryProposal

logger = logging.getLogger(__name__)


class MemoryProposer:
    """
    Parses LLM output for memory proposals.

    Supports both structured tool calls and delimiter-based parsing.
    """

    def __init__(self, config: Optional[ProposerConfig] = None):
        """Initialize proposer with parsing configuration and delimiter patterns."""
        self._config = config or ProposerConfig()
        self._delimiter_re = re.compile(
            re.escape(self._config.delimiter_open)
            + r"(.*?)"
            + re.escape(self._config.delimiter_close),
            re.DOTALL,
        )

    @property
    def system_instruction(self) -> str:
        """System instruction segment to append to LLM context."""
        return self._config.system_instruction

    def parse_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[MemoryProposal]:
        """
        Extract memory proposals from structured tool calls.

        Looks for calls with action="memory.propose" or name="memory.propose".
        """
        proposals = []
        for call in tool_calls:
            action = call.get("action", "") or call.get("name", "")
            if action in ("memory.propose", "memory_propose"):
                items = call.get("items", [])
                if not items and "arguments" in call:
                    # OpenAI-style tool call format
                    args = call["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            continue
                    items = args.get("items", [])
                for item_d in items:
                    try:
                        proposals.append(MemoryProposal.from_dict(item_d))
                    except Exception as e:
                        logger.warning(f"Failed to parse proposal: {e}")
        return proposals

    def parse_response_text(
        self, text: str
    ) -> Tuple[str, List[MemoryProposal]]:
        """
        Extract memory proposals from response text using delimiter blocks.

        Returns (cleaned_text, proposals) where cleaned_text has
        the delimiter blocks stripped.
        """
        proposals = []
        matches = self._delimiter_re.findall(text)

        for match in matches:
            match = match.strip()
            try:
                data = json.loads(match)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse delimiter block as JSON")
                continue

            # Support both {"items": [...]} and [...]
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("items", [])
            else:
                continue

            for item_d in items:
                try:
                    proposals.append(MemoryProposal.from_dict(item_d))
                except Exception as e:
                    logger.warning(f"Failed to parse proposal from delimiter: {e}")

        # Clean text
        cleaned = self._delimiter_re.sub("", text).strip()

        return cleaned, proposals

    def extract_proposals(
        self,
        response_text: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[MemoryProposal]]:
        """
        Extract proposals from both channels based on strategy config.

        Returns (cleaned_response_text, all_proposals).
        """
        proposals = []
        cleaned = response_text

        strategy = self._config.strategy

        # Strategy A: tool calls
        if strategy in ("tool", "both") and tool_calls:
            proposals.extend(self.parse_tool_calls(tool_calls))

        # Strategy B: delimiter blocks
        if strategy in ("delimiter", "both") and response_text:
            cleaned, delim_proposals = self.parse_response_text(response_text)
            proposals.extend(delim_proposals)

        if proposals:
            logger.info(f"Extracted {len(proposals)} memory proposal(s)")

        return cleaned, proposals
