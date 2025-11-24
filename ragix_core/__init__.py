"""
RAGIX Core - Shared orchestrator and tooling for RAGIX agents

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

__version__ = "0.5.0-dev"

from .llm_backends import OllamaLLM
from .tools_shell import ShellSandbox, CommandResult
from .logging_utils import AgentLogger, LogLevel, mask_secrets
from .profiles import (
    Profile,
    compute_dry_run_from_profile,
    get_profile_restrictions,
    merge_denylist_from_config,
    DANGEROUS_PATTERNS,
    GIT_DESTRUCTIVE_PATTERNS,
)
from .orchestrator import (
    extract_json_object,
    extract_json_with_diagnostics,
    validate_action_schema,
    create_retry_prompt,
)
from .retrieval import Retriever, GrepRetriever, RetrievalResult, format_retrieval_results

__all__ = [
    "OllamaLLM",
    "ShellSandbox",
    "CommandResult",
    "AgentLogger",
    "LogLevel",
    "mask_secrets",
    "Profile",
    "compute_dry_run_from_profile",
    "get_profile_restrictions",
    "merge_denylist_from_config",
    "extract_json_object",
    "extract_json_with_diagnostics",
    "validate_action_schema",
    "create_retry_prompt",
    "Retriever",
    "GrepRetriever",
    "RetrievalResult",
    "format_retrieval_results",
    "DANGEROUS_PATTERNS",
    "GIT_DESTRUCTIVE_PATTERNS",
]
