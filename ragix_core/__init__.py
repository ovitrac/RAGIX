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
from .agent_graph import (
    AgentNode,
    AgentEdge,
    AgentGraph,
    NodeStatus,
    TransitionCondition,
    create_linear_workflow,
)
from .graph_executor import (
    GraphExecutor,
    SyncGraphExecutor,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
)
from .agents import (
    BaseAgent,
    AgentCapability,
    CodeAgent,
    DocAgent,
    GitAgent,
    TestAgent,
)
from .chunking import (
    ChunkType,
    CodeChunk,
    PythonChunker,
    MarkdownChunker,
    GenericChunker,
    chunk_file,
)
from .embeddings import (
    EmbeddingBackend,
    EmbeddingConfig,
    SentenceTransformerBackend,
    DummyEmbeddingBackend,
    ChunkEmbedding,
    embed_code_chunks,
    save_embeddings,
    load_embeddings,
    create_embedding_backend,
)
from .vector_index import (
    SearchResult,
    VectorIndex,
    NumpyVectorIndex,
    FAISSVectorIndex,
    create_vector_index,
    build_index_from_embeddings,
)
from .batch_mode import (
    BatchExitCode,
    BatchConfig,
    WorkflowResult,
    BatchResult,
    BatchExecutor,
    run_batch_sync,
)
from .secrets_vault import (
    SecretMetadata,
    SecretProvider,
    InMemoryVault,
    EncryptedFileVault,
    AccessPolicy,
    VaultManager,
    create_vault,
)

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
    "AgentNode",
    "AgentEdge",
    "AgentGraph",
    "NodeStatus",
    "TransitionCondition",
    "create_linear_workflow",
    "GraphExecutor",
    "SyncGraphExecutor",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionStatus",
    "BaseAgent",
    "AgentCapability",
    "CodeAgent",
    "DocAgent",
    "GitAgent",
    "TestAgent",
    "ChunkType",
    "CodeChunk",
    "PythonChunker",
    "MarkdownChunker",
    "GenericChunker",
    "chunk_file",
    "EmbeddingBackend",
    "EmbeddingConfig",
    "SentenceTransformerBackend",
    "DummyEmbeddingBackend",
    "ChunkEmbedding",
    "embed_code_chunks",
    "save_embeddings",
    "load_embeddings",
    "create_embedding_backend",
    "SearchResult",
    "VectorIndex",
    "NumpyVectorIndex",
    "FAISSVectorIndex",
    "create_vector_index",
    "build_index_from_embeddings",
    "BatchExitCode",
    "BatchConfig",
    "WorkflowResult",
    "BatchResult",
    "BatchExecutor",
    "run_batch_sync",
    "SecretMetadata",
    "SecretProvider",
    "InMemoryVault",
    "EncryptedFileVault",
    "AccessPolicy",
    "VaultManager",
    "create_vault",
    "DANGEROUS_PATTERNS",
    "GIT_DESTRUCTIVE_PATTERNS",
]
