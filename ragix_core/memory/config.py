"""
Memory Subsystem Configuration

Nested dataclasses for all memory components: store, embedder, policy,
recall, Q*-search, palace, and consolidation.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-14
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class StoreConfig:
    """SQLite store configuration."""
    db_path: str = "memory.db"  # relative to workspace or absolute
    wal_mode: bool = True
    # FTS5 tokenizer string (passed as tokenize='...' in CREATE VIRTUAL TABLE).
    # Default: "unicode61 remove_diacritics 2" — accent-insensitive, good for French.
    # Alternatives: "porter" (English stemming), "unicode61" (accent-sensitive).
    fts_tokenizer: str = "unicode61 remove_diacritics 2"


@dataclass
class EmbedderConfig:
    """Embedding backend configuration."""
    backend: Literal["ollama", "sentence-transformers", "mock"] = "ollama"
    model: str = "nomic-embed-text"
    dimension: int = 768
    ollama_url: str = "http://localhost:11434"
    # Mock embedder settings (for tests)
    mock_seed: int = 42


@dataclass
class PolicyConfig:
    """Write governance configuration."""
    # Hard block thresholds
    max_content_length: int = 2000  # chars; longer must use pointer type
    secret_patterns_enabled: bool = True
    injection_patterns_enabled: bool = True
    instructional_content_enabled: bool = True  # V3.3: block/quarantine tool syntax & self-instructions
    require_provenance_for: List[str] = field(
        default_factory=lambda: ["mtm", "ltm"]
    )
    # Soft block / quarantine
    low_confidence_threshold: float = 0.3
    quarantine_expiry_hours: int = 72
    # Auditor LLM (optional)
    auditor_enabled: bool = False
    auditor_model: str = "granite3.2:3b"


@dataclass
class RecallConfig:
    """Hybrid retrieval configuration."""
    mode: Literal["inject", "catalog", "hybrid"] = "hybrid"
    inject_budget_tokens: int = 1500
    catalog_k: int = 10
    # Search weights
    tag_weight: float = 0.3
    embedding_weight: float = 0.5
    provenance_weight: float = 0.2
    # Filters
    default_tier_filter: Optional[str] = None  # None = all tiers
    exclude_archived: bool = True


@dataclass
class QSearchConfig:
    """Q*-style agenda search configuration."""
    enabled: bool = True
    max_expansions: int = 20
    max_retrieved_tokens: int = 3000
    max_time_seconds: float = 10.0
    score_threshold: float = 0.4
    # Scoring weights: S = w_r*R + w_p*P + w_c*C - w_d*D - w_x*X
    w_relevance: float = 0.35
    w_provenance: float = 0.20
    w_coverage: float = 0.25
    w_duplication: float = 0.10
    w_contradiction: float = 0.10
    # Operators
    enable_bridge: bool = False  # requires LLM for subgoal generation
    bridge_model: str = "granite3.2:3b"


@dataclass
class PalaceConfig:
    """Memory palace browse configuration."""
    enabled: bool = True
    auto_assign: bool = True  # auto-assign location on consolidation


@dataclass
class ConsolidateConfig:
    """Consolidation pipeline configuration."""
    enabled: bool = True
    model: str = "granite3.2:3b"
    ollama_url: str = "http://localhost:11434"
    # Triggering
    stm_threshold: int = 20  # auto-trigger when STM count exceeds this
    # Context-fraction trigger (consolidate when memory injection >= fraction of ctx)
    ctx_fraction_trigger: float = 0.15  # 15% of context budget
    ctx_limit_tokens: int = 32000  # model effective context size
    # Clustering
    cluster_distance_threshold: float = 0.3
    # Promotion rules
    usage_count_for_ltm: int = 5
    auto_promote_types: List[str] = field(
        default_factory=lambda: ["constraint", "decision", "definition"]
    )
    # Fallback if no LLM available
    fallback_to_deterministic: bool = True
    # Merge prompt style for RIE-type rules
    merge_style: str = "rules"  # "rules" | "prose"
    # V31-3: Merge confidence blending weights (must sum to 1.0)
    merge_w_max: float = 0.3        # weight for max(confidence) across cluster
    merge_w_overlap: float = 0.3    # weight for Jaccard tag overlap
    merge_w_recency: float = 0.2    # weight for exponential recency decay
    merge_w_validation: float = 0.2  # weight for validation-state bonus
    recency_lambda: float = 0.01    # decay constant for recency (exp(-lambda * days))


@dataclass
class GraphConfig:
    """Graph-RAG configuration (consolidation assist)."""
    enabled: bool = True
    similarity_edge_threshold: float = 0.85  # cosine for 'similar' edges
    neighborhood_depth: int = 2              # BFS depth for merge candidates
    max_neighborhood_size: int = 50          # cap to prevent runaway BFS
    # Locality constraints: union requires BFS + at least one of these
    require_same_doc: bool = True            # same source document
    require_same_tag: bool = True            # same primary tag
    locality_cosine_threshold: float = 0.90  # high cosine = locality signal
    # V2.4: Scalability controls
    similarity_top_k: int = 20              # keep only top-k similar edges per item
    max_edges_per_node: int = 100           # cap total edges per node (prevents hub explosion)


@dataclass
class SecrecyConfig:
    """
    Report-time + write-time redaction configuration.

    Secrecy table (what is redacted per tier):
        Artifact        S0          S2          S3
        paths           REDACT      keep        keep
        emails          REDACT      keep        keep
        hostnames       REDACT      REDACT      keep
        ips             REDACT      REDACT      keep
        filenames       REDACT      keep        keep
        pointer_ids     REDACT      keep        keep
        hashes          REDACT      REDACT      keep
        entity_labels   REDACT      keep        keep
    """
    tier: str = "S3"  # S0=public, S2=internal, S3=audit (full detail)
    redact_at_write_time: bool = True   # apply redaction when storing canonical items
    custom_patterns: List[str] = field(default_factory=list)


@dataclass
class ProposerConfig:
    """LLM proposal parsing configuration."""
    strategy: Literal["tool", "delimiter", "both"] = "both"
    delimiter_open: str = "<MEMORY_PROPOSALS_JSON>"
    delimiter_close: str = "</MEMORY_PROPOSALS_JSON>"
    system_instruction: str = (
        "If you see information that should persist across turns, emit "
        "a memory.propose tool call with concise items, tags, and provenance hints. "
        "Do not include secrets or raw data dumps."
    )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for the memory MCP server."""
    enabled: bool = True
    calls_per_minute: int = 60        # max tool calls per minute per session
    proposals_per_turn: int = 10      # max memory proposals per turn
    max_content_length: int = 5000    # max content length per proposal (chars)
    burst_multiplier: float = 1.5     # allows burst up to N * calls_per_minute


@dataclass
class MemoryConfig:
    """Top-level memory subsystem configuration."""
    store: StoreConfig = field(default_factory=StoreConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    qsearch: QSearchConfig = field(default_factory=QSearchConfig)
    palace: PalaceConfig = field(default_factory=PalaceConfig)
    consolidate: ConsolidateConfig = field(default_factory=ConsolidateConfig)
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    secrecy: SecrecyConfig = field(default_factory=SecrecyConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryConfig:
        """Build config from a nested dict (e.g. YAML/JSON)."""
        return cls(
            store=StoreConfig(**d.get("store", {})),
            embedder=EmbedderConfig(**d.get("embedder", {})),
            policy=PolicyConfig(**d.get("policy", {})),
            recall=RecallConfig(**d.get("recall", {})),
            qsearch=QSearchConfig(**d.get("qsearch", {})),
            palace=PalaceConfig(**d.get("palace", {})),
            consolidate=ConsolidateConfig(**d.get("consolidate", {})),
            proposer=ProposerConfig(**d.get("proposer", {})),
            graph=GraphConfig(**d.get("graph", {})),
            secrecy=SecrecyConfig(**d.get("secrecy", {})),
            rate_limit=RateLimitConfig(**d.get("rate_limit", {})),
        )
