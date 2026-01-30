"""
Output sanitizer for sovereign compliance.

Enforces MUST requirements M1, M2 and MUST NOT requirement N2
from docs/SOVEREIGN_LLM_OPERATIONS.md.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-01-30
"""

import re
import logging
from pathlib import Path
from typing import List, Set, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OutputLevel(Enum):
    """Output isolation levels for sovereign compliance."""
    INTERNAL = "internal"       # Full output with all metadata
    EXTERNAL = "external"       # Redacted for external delivery
    ORCHESTRATOR = "orchestrator"  # Metrics only, no text content
    COMPLIANCE = "compliance"   # Full output + attestation


# Denylist keys that MUST NOT appear in external outputs (N2)
DENYLIST_KEYS: List[str] = [
    "llm_trace",
    "call_hash",
    "inputs_merkle_root",
    "run_id",
    "endpoint",
    "model",
    "cache_status",
    "prompt_tokens",
    "completion_tokens",
    "digest",
    "prompt_hash",
    "response_hash",
    "sovereignty",
    "model_digest",
    "cache_key",
]

# Path patterns to redact in external outputs
PATH_PATTERNS: List[str] = [
    r"/home/[^/\s]+/",           # Unix home directories
    r"/Users/[^/\s]+/",          # macOS home directories
    r"C:\\Users\\[^\\]+\\",      # Windows home directories
    r"/tmp/[^/\s]+/",            # Temp directories
    r"/var/[^/\s]+/",            # Var directories
]

# ID patterns to anonymize
ID_PATTERNS: List[str] = [
    r"F\d{6}",                        # File IDs (F000123)
    r"run_\d{8}_\d{6}_[a-f0-9]+",     # Run IDs
    r"[a-f0-9]{32,}",                 # Long hashes (MD5+)
]


class SecurityViolation(Exception):
    """Raised when denylist keys appear in external output."""
    pass


@dataclass
class SanitizationResult:
    """Result of sanitization operation."""
    content: str
    redacted_paths: int = 0
    anonymized_ids: int = 0
    stripped_metadata: bool = False
    warnings: List[str] = field(default_factory=list)


def validate_external_report(
    report_path: Path,
    level: OutputLevel,
    strict: bool = True
) -> bool:
    """
    Validate report for external delivery.

    Args:
        report_path: Path to the report file
        level: Output isolation level
        strict: If True, raise exception on violation; if False, log warning

    Returns:
        True if valid, False if violations found (when strict=False)

    Raises:
        SecurityViolation: If denylist keys detected and strict=True
    """
    if level == OutputLevel.INTERNAL:
        return True  # No validation for internal

    content = report_path.read_text(encoding="utf-8")
    violations = []

    if level in (OutputLevel.EXTERNAL, OutputLevel.ORCHESTRATOR):
        for key in DENYLIST_KEYS:
            # Match as JSON key or standalone identifier
            patterns = [
                rf'"{key}"',           # JSON key
                rf"'{key}'",           # YAML key
                rf"\b{key}\b",         # Standalone word
            ]
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(key)
                    break

    if violations:
        msg = (
            f"Denylist keys found in {level.value} report: {violations}. "
            f"Build FAILED. Remove or redact before delivery."
        )
        if strict:
            raise SecurityViolation(msg)
        else:
            logger.warning(msg)
            return False

    return True


def redact_paths(content: str) -> tuple[str, int]:
    """
    Redact file system paths.

    Returns:
        Tuple of (redacted_content, count_of_redactions)
    """
    result = content
    count = 0

    for pattern in PATH_PATTERNS:
        matches = re.findall(pattern, result)
        count += len(matches)
        result = re.sub(pattern, "[PATH]/", result)

    return result, count


def anonymize_ids(content: str) -> tuple[str, int]:
    """
    Anonymize internal identifiers.

    Returns:
        Tuple of (anonymized_content, count_of_anonymizations)
    """
    result = content
    count = 0

    # Replace file IDs with sequential anonymous IDs
    file_ids = sorted(set(re.findall(r"F\d{6}", result)))
    for i, fid in enumerate(file_ids):
        result = result.replace(fid, f"[DOC-{chr(65 + (i % 26))}]")
        count += 1

    # Replace run IDs
    run_ids = re.findall(r"run_\d{8}_\d{6}_[a-f0-9]+", result)
    if run_ids:
        result = re.sub(r"run_\d{8}_\d{6}_[a-f0-9]+", "[RUN-ID]", result)
        count += len(set(run_ids))

    # Replace long hashes (but not short ones that might be legitimate)
    long_hashes = re.findall(r"\b[a-f0-9]{40,}\b", result)
    if long_hashes:
        result = re.sub(r"\b[a-f0-9]{40,}\b", "[HASH]", result)
        count += len(set(long_hashes))

    return result, count


def strip_metadata_blocks(content: str) -> tuple[str, bool]:
    """
    Strip YAML/TOML front-matter and provenance comments.

    Returns:
        Tuple of (stripped_content, was_stripped)
    """
    original_len = len(content)
    result = content

    # YAML front-matter (---\n...\n---)
    result = re.sub(r"^---\n.*?\n---\n", "", result, flags=re.DOTALL)

    # TOML front-matter (+++\n...\n+++)
    result = re.sub(r"^\+\+\+\n.*?\n\+\+\+\n", "", result, flags=re.DOTALL)

    # HTML provenance comments
    result = re.sub(r"<!--\s*PROVENANCE.*?-->", "", result, flags=re.DOTALL | re.IGNORECASE)

    # Fenced metadata blocks
    result = re.sub(r"```metadata\n.*?\n```", "", result, flags=re.DOTALL)
    result = re.sub(r"```json:metadata\n.*?\n```", "", result, flags=re.DOTALL)

    # KOAS-specific sovereignty blocks
    result = re.sub(r"```sovereignty\n.*?\n```", "", result, flags=re.DOTALL)

    was_stripped = len(result) < original_len
    return result, was_stripped


def extract_metrics_only(content: str) -> str:
    """
    Extract metrics-only view for orchestrator mode.

    Removes all text content, keeps only:
    - Section headers
    - Tables with numeric data
    - Summary statistics
    """
    lines = content.split("\n")
    metrics_lines = []

    in_table = False
    for line in lines:
        # Keep markdown headers
        if line.startswith("#"):
            metrics_lines.append(line)
            continue

        # Keep table headers and separators
        if "|" in line:
            # Check if it's a metrics table (contains numbers)
            if re.search(r"\|\s*\d", line) or re.match(r"\s*\|[-:| ]+\|", line):
                metrics_lines.append(line)
                in_table = True
                continue

        # Keep lines with explicit metrics
        if re.search(r"(count|total|score|time|duration|size):\s*\d", line, re.IGNORECASE):
            metrics_lines.append(line)
            continue

        in_table = False

    return "\n".join(metrics_lines)


def sanitize_for_level(
    content: str,
    level: OutputLevel,
    redact_paths_flag: bool = True,
    anonymize_ids_flag: bool = True,
    strip_metadata_flag: bool = True
) -> SanitizationResult:
    """
    Apply sanitization based on output level.

    Args:
        content: Raw content to sanitize
        level: Target output level
        redact_paths_flag: Whether to redact file paths
        anonymize_ids_flag: Whether to anonymize IDs
        strip_metadata_flag: Whether to strip metadata blocks

    Returns:
        SanitizationResult with sanitized content and statistics
    """
    if level == OutputLevel.INTERNAL:
        return SanitizationResult(content=content)

    result = SanitizationResult(content=content)
    warnings = []

    if level == OutputLevel.ORCHESTRATOR:
        # Metrics only - strip all text content
        result.content = extract_metrics_only(content)
        result.stripped_metadata = True
        return result

    # EXTERNAL and COMPLIANCE levels
    if strip_metadata_flag:
        result.content, result.stripped_metadata = strip_metadata_blocks(result.content)

    if redact_paths_flag:
        result.content, result.redacted_paths = redact_paths(result.content)

    if anonymize_ids_flag:
        result.content, result.anonymized_ids = anonymize_ids(result.content)

    # Log sanitization summary
    if result.redacted_paths or result.anonymized_ids or result.stripped_metadata:
        logger.info(
            f"[output_sanitizer] Sanitized for {level.value}: "
            f"paths={result.redacted_paths}, ids={result.anonymized_ids}, "
            f"metadata_stripped={result.stripped_metadata}"
        )

    result.warnings = warnings
    return result


def get_level_from_string(level_str: str) -> OutputLevel:
    """Parse output level from string."""
    try:
        return OutputLevel(level_str.lower())
    except ValueError:
        valid = [l.value for l in OutputLevel]
        raise ValueError(f"Invalid output level '{level_str}'. Valid: {valid}")
