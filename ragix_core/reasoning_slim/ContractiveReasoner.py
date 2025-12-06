#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContractiveReasoner: Branching Contractive Reasoning Engine for Slim LLMs
========================================================================

- Backend: Ollama (default: granite3.1-moe:3b)
- API: async, branch-wise exploration with entropy control & tree introspection
- Intended use: called from other Python code (e.g. RAGIX), or as a standalone tool.

Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio Innovation Lab
"""

from __future__ import annotations

import asyncio
import argparse
import dataclasses
import json
import math
import time
import uuid
from statistics import mean
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

import httpx  # pip install httpx

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

# ---------------------------------------------------------------------
# 0. Base system prompt (to be reused in external scripts if needed)
# ---------------------------------------------------------------------

CONTRACTIVE_SYSTEM_PROMPT = """
You are a contractive reasoning assistant.

Your job is NOT to jump to the final answer, but to:
1. Normalize the problem.
2. Decompose it into subproblems when it is too complex or uncertain.
3. Solve leaf subproblems simply and clearly.
4. Help aggregate child answers into a coherent parent answer.
5. Explicitly state uncertainties and missing information.
6. When you cannot reduce uncertainty further, ask precise clarification
   questions instead of hallucinating an answer.

You will receive instructions that tell you whether you should:
- DECOMPOSE: produce a tree/list of subquestions, with brief roles.
- SOLVE: answer a single, well-scoped question.
- COLLAPSE: integrate several subanswers into a higher-level answer.
- CHECK: analyze your own answer for possible failure modes.
- CLARIFY: propose questions to ask the human when uncertainty is high.

Follow those instructions carefully.
Keep answers concise and structured.
"""


# ---------------------------------------------------------------------
# 1. Reasoning tree structures
# ---------------------------------------------------------------------

@dataclass
class NodeMetrics:
    """Metrics attached to each reasoning node (per step)."""
    depth: int
    step_index: int
    timestamp: float

    # Entropy-related metrics
    entropy_model: Optional[float] = None
    entropy_struct: Optional[float] = None
    entropy_consistency: Optional[float] = None

    # BM25-like relevance to root / core question
    relevance_root: Optional[float] = None

    # Token usage (from Ollama)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Context / size
    approx_prompt_chars: Optional[int] = None

    # Raw diagnostic payloads if needed
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningNode:
    """A single node in the reasoning tree."""
    node_id: str
    parent_id: Optional[str]
    role: str  # "root", "subproblem", "alternative", "clarification", ...
    question: str

    # LLM outputs
    answer: Optional[str] = None
    state: str = "open"  # "open", "solved", "failed", "pruned", "clarification_needed"
    children: List[str] = field(default_factory=list)

    # Semantic relevance to root (for pruning decisions)
    relevance_to_root: Optional[float] = None

    # For DECOMPOSE, we may store the raw decomposition text
    raw_decomposition: Optional[str] = None

    # Pre-decomposition entropy (for auto-rebranch decision)
    pre_decomp_entropy: Optional[float] = None

    # Number of rebranch attempts (to avoid infinite loops)
    rebranch_count: int = 0

    # Cumulative token count for this branch path (root → this node)
    branch_tokens: int = 0

    # Summarized context (when branch exceeds token limit)
    summarized_context: Optional[str] = None

    # Peer-review tracking
    peer_reviewed: bool = False
    peer_verdict: Optional[str] = None  # "approved", "rejected", "needs_revision"
    peer_score: Optional[float] = None  # 0.0 to 1.0
    peer_feedback: Optional[str] = None

    # Node metrics
    metrics: List[NodeMetrics] = field(default_factory=list)


# ---------------------------------------------------------------------
# 2. Helper: simple BM25-style relevance to root
# ---------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words, filtering punctuation."""
    import re
    # Split on whitespace and punctuation, keep alphanumeric tokens
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 1]  # filter single chars


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between two texts based on token sets.
    Returns value in [0, 1] where 1 = identical token sets.
    """
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def semantic_relevance(node_text: str, root_text: str, method: str = "combined") -> float:
    """
    Compute semantic relevance of node_text to root_text.

    Parameters
    ----------
    node_text : str
        The node's question (and optionally answer).
    root_text : str
        The root question (the main topic).
    method : str
        "jaccard" - pure token overlap
        "bm25" - BM25-like scoring
        "combined" - weighted combination (default)

    Returns
    -------
    float
        Relevance score. For "combined", normalized to [0, 1].
    """
    if method == "jaccard":
        return jaccard_similarity(node_text, root_text)
    elif method == "bm25":
        return bm25_like_score(root_text, node_text)
    else:  # combined
        jacc = jaccard_similarity(node_text, root_text)
        bm25 = bm25_like_score(root_text, node_text)
        # Normalize BM25 (typical range 0-30) to [0,1]
        bm25_norm = min(1.0, bm25 / 20.0)
        # Weighted combination favoring Jaccard for scope detection
        return 0.6 * jacc + 0.4 * bm25_norm


def bm25_like_score(query: str, document: str, k1: float = 1.5, b: float = 0.75) -> float:
    """
    Extremely simplified BM25-like scoring for a single query/document pair.
    This is not a full corpus BM25, but enough to rank branches qualitatively.

    - query: root / core question
    - document: node question + answer text
    """
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(document)
    if not d_tokens:
        return 0.0

    # Term frequencies in document
    tf: Dict[str, int] = {}
    for t in d_tokens:
        tf[t] = tf.get(t, 0) + 1

    avgdl = len(d_tokens)
    score = 0.0
    # Because we don't have a corpus, approximate IDF ~ log(1 + 1/df) ≈ 1 for shared terms
    for t in set(q_tokens):
        f = tf.get(t, 0)
        if f == 0:
            continue
        # toy IDF: treat all query terms as equally informative
        idf = 1.0
        denom = f + k1 * (1.0 - b + b * len(d_tokens) / (avgdl + 1e-6))
        score += idf * f * (k1 + 1) / (denom + 1e-6)
    return score


# ---------------------------------------------------------------------
# 3. ContractiveReasoner class
# ---------------------------------------------------------------------

class ContractiveReasoner:
    """
    ContractiveReasoner
    ===================

    Async reasoning engine that:
    - Talks to an Ollama model (default granite3.1-moe:3b) via /api/chat and /api/show.
    - Builds a tree of ReasoningNode objects, with branching (DECOMPOSE) and collapsing (COLLAPSE).
    - Tracks entropy, relevance, tokens, and step metrics.
    - Allows browsing, restarting, rebranching, and exporting reasoning traces.

    Typical usage (sync wrapper):

        engine = ContractiveReasoner()
        result = engine.run("Design a safe-by-design migration study for X ...")
        print(result.final_answer)
        tree = result.tree   # or engine.tree

    Or async:

        engine = ContractiveReasoner()
        result = await engine.solve("...", max_depth=3)
    """

    # -----------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "granite3.1-moe:3b",
        system_prompt: str = CONTRACTIVE_SYSTEM_PROMPT,
        max_depth: int = 4,
        max_loops: int = 10,
        max_global_tokens: int = 64000,
        max_branch_tokens: int = 16000,
        max_concurrent_branches: int = 4,
        entropy_decompose_threshold: float = 0.9,
        entropy_collapse_threshold: float = 0.4,
        entropy_gamma_min_reduction: float = 0.05,
        k_entropy_samples: int = 4,
        min_relevance_threshold: float = 0.15,
        max_rebranch_attempts: int = 2,
        timeout_sec: int = 120,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Parameters
        ----------
        base_url : str
            Ollama base URL (default: http://localhost:11434).
        model : str
            Name of the Ollama model (default: granite3.1-moe:3b).
        system_prompt : str
            System prompt used for all reasoning calls.
        max_depth : int
            Maximum tree depth.
        max_loops : int
            Global max number of reasoning loops.
        max_global_tokens : int
            Global soft limit for tokens seen (prompt + completion).
        max_branch_tokens : int
            Soft limit for a single branch (before summarization/collapse).
        max_concurrent_branches : int
            Maximum number of nodes processed concurrently.
        entropy_decompose_threshold : float
            Above this entropy, we prefer DECOMPOSE over direct solving.
        entropy_collapse_threshold : float
            Below this entropy, we consider a node "stable enough" to collapse.
        entropy_gamma_min_reduction : float
            Minimal entropy reduction required to accept a decomposition.
        k_entropy_samples : int
            Number of LLM samples to estimate model entropy at a node.
        min_relevance_threshold : float
            Minimum semantic relevance to root question. Nodes below this
            threshold are pruned (marked out-of-scope) without LLM calls.
        max_rebranch_attempts : int
            Maximum number of automatic rebranch attempts per node when
            decomposition increases entropy instead of reducing it.
        timeout_sec : int
            Per-call timeout.
        client : Optional[httpx.AsyncClient]
            Optional custom HTTP client. If None, a new one will be created.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt

        self.max_depth = max_depth
        self.max_loops = max_loops
        self.max_global_tokens = max_global_tokens
        self.max_branch_tokens = max_branch_tokens
        self.max_concurrent_branches = max_concurrent_branches

        self.entropy_decompose_threshold = entropy_decompose_threshold
        self.entropy_collapse_threshold = entropy_collapse_threshold
        self.entropy_gamma_min_reduction = entropy_gamma_min_reduction
        self.k_entropy_samples = k_entropy_samples
        self.min_relevance_threshold = min_relevance_threshold
        self.max_rebranch_attempts = max_rebranch_attempts

        self.timeout_sec = timeout_sec

        self._external_client = client
        self._client: Optional[httpx.AsyncClient] = None
        self._ctx_window: Optional[int] = None  # from ollama /api/show

        # Tree state
        self.nodes: Dict[str, ReasoningNode] = {}
        self.root_id: Optional[str] = None
        self._step_counter = 0
        self._global_prompt_tokens = 0
        self._global_completion_tokens = 0

        # Concurrency
        self._sem = asyncio.Semaphore(max_concurrent_branches)

        # Optional event callback: called after each node update
        # signature: callback(engine: ContractiveReasoner, node: ReasoningNode, metrics: NodeMetrics) -> None
        self.event_callback: Optional[
            Callable[[ContractiveReasoner, ReasoningNode, NodeMetrics], None]
        ] = None

        # Peer reviewer (optional, for external validation)
        # Set this to enable peer-review of branches
        self.peer_reviewer: Optional[Any] = None  # PeerReviewer instance

    def reset(self) -> None:
        """
        Reset the engine state for a fresh reasoning run.

        Call this before reusing the same engine instance for a new question.
        Preserves configuration but clears the reasoning tree and counters.
        """
        self.nodes.clear()
        self.root_id = None
        self._step_counter = 0
        self._global_prompt_tokens = 0
        self._global_completion_tokens = 0

    # -----------------------------------------------------------------
    # Public API (sync wrapper)
    # -----------------------------------------------------------------

    def run(self, question: str, **kwargs) -> "ReasoningResult":
        """
        Synchronous wrapper around `solve`.

        Returns a ReasoningResult object (see bottom of file).
        """
        return asyncio.run(self.solve(question, **kwargs))

    # -----------------------------------------------------------------
    # Public API (async solve)
    # -----------------------------------------------------------------

    async def solve(
        self,
        question: str,
        *,
        max_depth: Optional[int] = None,
        max_loops: Optional[int] = None,
    ) -> "ReasoningResult":
        """
        Main entry point for contractive reasoning on a question.

        Parameters
        ----------
        question : str
            The root question/problem.
        max_depth : Optional[int]
            Optional override of global max_depth.
        max_loops : Optional[int]
            Optional override of global max_loops.

        Returns
        -------
        ReasoningResult
        """
        if max_depth is not None:
            self.max_depth = max_depth
        if max_loops is not None:
            self.max_loops = max_loops

        # Reset tree state for fresh reasoning (allows engine reuse)
        self.reset()

        if self._client is None:
            self._client = self._external_client or httpx.AsyncClient(timeout=self.timeout_sec)

        # Fetch model info, including context length (if available)
        await self._init_model_info()

        # Initialize root node
        root_id = self._new_node(
            parent_id=None,
            role="root",
            question=question,
        )
        self.root_id = root_id

        loops = 0
        while loops < self.max_loops:
            loops += 1

            # 1. Collect open frontier nodes (breadth-first is fine)
            frontier = [n for n in self.nodes.values() if n.state == "open"]
            if not frontier:
                break  # nothing left to do

            # Limit by depth
            frontier = [n for n in frontier if self._node_depth(n.node_id) <= self.max_depth]
            if not frontier:
                break

            # 2. Process in parallel with concurrency limit
            tasks = []
            for node in frontier:
                tasks.append(self._process_node(node))
            await asyncio.gather(*tasks)

            # 3. Try collapses where possible (separate pass)
            await self._try_collapses()

            # Global soft stop conditions
            if self._global_prompt_tokens + self._global_completion_tokens >= self.max_global_tokens:
                break

        # Final collapse at root if needed
        if self.root_id is not None:
            root = self.nodes[self.root_id]
            if root.state != "solved":
                await self._force_root_collapse()

        final_answer = self.nodes[self.root_id].answer if self.root_id else None

        # Close client if we created it
        if self._client is not None and self._external_client is None:
            await self._client.aclose()
            self._client = None

        return ReasoningResult(
            engine=self,
            final_answer=final_answer,
            root_id=self.root_id,
        )

    # -----------------------------------------------------------------
    # Node management / introspection API
    # -----------------------------------------------------------------

    def _new_node(self, parent_id: Optional[str], role: str, question: str) -> str:
        node_id = str(uuid.uuid4())
        node = ReasoningNode(
            node_id=node_id,
            parent_id=parent_id,
            role=role,
            question=question.strip(),
        )
        self.nodes[node_id] = node
        if parent_id is not None:
            self.nodes[parent_id].children.append(node_id)
        return node_id

    def get_node(self, node_id: str) -> ReasoningNode:
        return self.nodes[node_id]

    def get_path_to_root(self, node_id: str) -> List[ReasoningNode]:
        path = []
        current = self.nodes[node_id]
        while current is not None:
            path.append(current)
            if current.parent_id is None:
                break
            current = self.nodes[current.parent_id]
        return list(reversed(path))

    def get_subtree(self, node_id: str) -> List[ReasoningNode]:
        result: List[ReasoningNode] = []

        def _dfs(nid: str):
            n = self.nodes[nid]
            result.append(n)
            for cid in n.children:
                _dfs(cid)

        _dfs(node_id)
        return result

    def rebranch(self, node_id: str) -> None:
        """
        Delete all children of node_id and mark it open again.
        Does NOT change its question; can be used after a failed or unhelpful decomposition.
        """
        node = self.nodes[node_id]
        for cid in list(node.children):
            self._delete_subtree(cid)
        node.children = []
        node.state = "open"
        node.answer = None

    def _delete_subtree(self, node_id: str) -> None:
        node = self.nodes[node_id]
        for cid in node.children:
            self._delete_subtree(cid)
        del self.nodes[node_id]

    def _compute_branch_tokens(self, node_id: str) -> int:
        """
        Compute cumulative token count from root to this node.
        Sums prompt + completion tokens across all nodes in the path.
        """
        total = 0
        for n in self.get_path_to_root(node_id):
            for m in n.metrics:
                total += (m.prompt_tokens or 0) + (m.completion_tokens or 0)
        return total

    def _get_branch_context(self, node_id: str) -> str:
        """
        Build context string from root to this node for prompt construction.
        If any ancestor has a summarized_context, use that instead of full history.
        """
        path = self.get_path_to_root(node_id)
        parts = []

        # Find the most recent summarized context in the path
        summary_idx = -1
        for i, n in enumerate(path):
            if n.summarized_context:
                summary_idx = i

        if summary_idx >= 0:
            # Use summary + nodes after it
            parts.append(f"[SUMMARIZED CONTEXT]\n{path[summary_idx].summarized_context}\n")
            path = path[summary_idx + 1:]

        for n in path:
            parts.append(f"Q: {n.question}")
            if n.answer:
                parts.append(f"A: {n.answer[:500]}...")  # Truncate long answers

        return "\n".join(parts)

    async def _summarize_branch(self, node: ReasoningNode) -> None:
        """
        Summarize the branch history from root to this node to reduce context size.
        Stores the summary in node.summarized_context for use in child prompts.
        """
        path = self.get_path_to_root(node.node_id)
        context_parts = []
        for n in path:
            context_parts.append(f"Q: {n.question}")
            if n.answer:
                context_parts.append(f"A: {n.answer}")
        full_context = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "INSTRUCTION: SUMMARIZE_CONTEXT\n"
                    "Summarize the following reasoning chain into a concise context.\n"
                    "Preserve key facts, conclusions, and any unresolved questions.\n"
                    "Keep the summary under 500 words.\n\n"
                    f"REASONING CHAIN:\n{full_context}"
                ),
            },
        ]
        data = await self._chat(messages, temperature=0.2)
        self._record_token_usage(data)
        summary = data.get("content", "").strip()
        node.summarized_context = summary

    # -----------------------------------------------------------------
    # Ollama integration
    # -----------------------------------------------------------------

    async def _init_model_info(self) -> None:
        """
        Call /api/show once to retrieve model information, including context size if possible.
        """
        if self._ctx_window is not None:
            return
        url = f"{self.base_url}/api/show"
        payload = {"model": self.model}
        r = await self._client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

        # Strategy 1: parse "parameters" string for num_ctx
        num_ctx = None
        params = data.get("parameters")
        if isinstance(params, str):
            for line in params.splitlines():
                line = line.strip()
                if line.startswith("num_ctx"):
                    try:
                        num_ctx = int(line.split()[1])
                    except Exception:
                        pass

        # Strategy 2: use model_info context_length
        if num_ctx is None:
            mi = data.get("model_info", {})
            for k, v in mi.items():
                if isinstance(k, str) and k.endswith(".context_length"):
                    try:
                        num_ctx = int(v)
                        break
                    except Exception:
                        pass

        # Fallback: no info -> leave None (caller may configure)
        self._ctx_window = num_ctx

    async def _chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> Dict[str, Any]:
        """
        Wrapper around Ollama /api/chat.
        Returns full JSON, including prompt_eval_count and eval_count.

        Raises
        ------
        httpx.ConnectError
            If Ollama is not running or unreachable.
        httpx.HTTPStatusError
            If the API returns an error status.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            r = await self._client.post(url, json=payload)
            r.raise_for_status()
        except httpx.ConnectError as e:
            raise httpx.ConnectError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Try: ollama serve"
            ) from e
        except httpx.HTTPStatusError as e:
            # Model not found or other API error
            if e.response.status_code == 404:
                raise httpx.HTTPStatusError(
                    f"Model '{self.model}' not found. Try: ollama pull {self.model}",
                    request=e.request,
                    response=e.response,
                ) from e
            raise
        data = r.json()
        # Ollama returns a single "message"
        msg = data.get("message", {})
        content = msg.get("content", "")
        # Handle case where content is a list (some Ollama versions)
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        elif not isinstance(content, str):
            content = str(content) if content else ""
        # For compatibility with some variants, copy back
        data["content"] = content
        return data

    # -----------------------------------------------------------------
    # Token accounting
    # -----------------------------------------------------------------

    def _record_token_usage(self, data: Dict[str, Any]) -> Tuple[int, int]:
        """
        Extract prompt/completion token usage from an Ollama response and
        accumulate in the engine-level counters.
        """

        def _to_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        prompt_tokens = _to_int(
            data.get("prompt_eval_count")
            or data.get("prompt_tokens")
        )
        completion_tokens = _to_int(
            data.get("eval_count")
            or data.get("completion_eval_count")
            or data.get("completion_tokens")
        )

        self._global_prompt_tokens += prompt_tokens
        self._global_completion_tokens += completion_tokens
        return prompt_tokens, completion_tokens

    # -----------------------------------------------------------------
    # Entropy & metrics
    # -----------------------------------------------------------------

    async def _estimate_entropy_and_answer(self, node: ReasoningNode) -> Tuple[float, float, float, str, Dict[str, Any]]:
        """
        Estimate (entropy_model, entropy_struct, entropy_consistency) and return
        a representative answer for node.question (without decomposition).

        Uses k samples from the LLM and a simple clustering on exact strings.
        """
        # Structural entropy: cheap, local
        q = node.question
        len_q = len(q)
        n_and = q.count(" and ") + q.count(" AND ")
        n_or = q.count(" or ") + q.count(" OR ")
        n_constraints = sum(q.count(tok) for tok in [">=", "<=", "==", ">", "<"])
        entropy_struct = math.log(1 + len_q) / 10.0 + 0.1 * (n_and + n_or + n_constraints)

        # Model entropy: sample k answers
        answers = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        for _ in range(self.k_entropy_samples):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "INSTRUCTION: SOLVE\n"
                        "You are asked a single, well-scoped question.\n"
                        "Give a concise answer, 3–5 sentences.\n\n"
                        f"QUESTION:\n{node.question}"
                    ),
                },
            ]
            data = await self._chat(messages, temperature=0.5)
            p_tok, c_tok = self._record_token_usage(data)
            prompt_tokens_total += p_tok
            completion_tokens_total += c_tok
            ans = data.get("content", "").strip()
            answers.append(ans)

        # Cluster answers by semantic similarity (Jaccard) instead of exact match.
        # Two answers are in the same cluster if Jaccard similarity > threshold.
        SIMILARITY_THRESHOLD = 0.5
        clusters: List[List[str]] = []

        for ans in answers:
            ans_stripped = ans.strip()
            merged = False
            for cluster in clusters:
                # Check if this answer is similar to the cluster representative
                representative = cluster[0]
                if jaccard_similarity(ans_stripped, representative) >= SIMILARITY_THRESHOLD:
                    cluster.append(ans_stripped)
                    merged = True
                    break
            if not merged:
                clusters.append([ans_stripped])

        # Compute entropy from cluster sizes
        total = float(len(answers))
        entropy_model = 0.0
        for cluster in clusters:
            p = len(cluster) / total
            if p > 0:
                entropy_model -= p * math.log(p)
        # Normalize by max entropy (log(k_samples)) to get [0, 1] range
        max_entropy = math.log(self.k_entropy_samples) if self.k_entropy_samples > 1 else 1.0
        entropy_model = max(0.0, entropy_model / max_entropy) if max_entropy > 0 else 0.0

        # Consistency entropy: based on number of distinct semantic clusters
        distinct = len(clusters)
        entropy_consistency = (distinct - 1) / max(1, self.k_entropy_samples - 1)

        # Pick the answer from the largest cluster as representative
        largest_cluster = max(clusters, key=len)
        best_answer = largest_cluster[0]

        debug = {
            "all_samples": answers,
            "n_clusters": distinct,
            "cluster_sizes": [len(c) for c in clusters],
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "token_usage": {
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
            },
        }

        return entropy_model, entropy_struct, entropy_consistency, best_answer, debug

    def _node_depth(self, node_id: str) -> int:
        d = 0
        n = self.nodes[node_id]
        while n.parent_id is not None:
            d += 1
            n = self.nodes[n.parent_id]
        return d

    # -----------------------------------------------------------------
    # Node processing (DECOMPOSE vs SOLVE)
    # -----------------------------------------------------------------

    async def _process_node(self, node: ReasoningNode) -> None:
        async with self._sem:
            depth = self._node_depth(node.node_id)
            self._step_counter += 1
            step_index = self._step_counter

            t0 = time.time()

            # ============================================================
            # SEMANTIC PRUNING GATE (before any LLM calls)
            # ============================================================
            # For non-root nodes, check semantic relevance to root question.
            # If relevance is below threshold, prune the branch immediately.
            if node.parent_id is not None and self.root_id is not None:
                root_q = self.nodes[self.root_id].question
                pre_relevance = semantic_relevance(node.question, root_q)
                node.relevance_to_root = pre_relevance

                if pre_relevance < self.min_relevance_threshold:
                    # PRUNE: mark as out-of-scope without spending LLM tokens
                    node.state = "pruned"
                    node.answer = f"[Pruned: low relevance to root ({pre_relevance:.3f} < {self.min_relevance_threshold})]"

                    metrics = NodeMetrics(
                        depth=depth,
                        step_index=step_index,
                        timestamp=t0,
                        entropy_model=None,
                        entropy_struct=None,
                        entropy_consistency=None,
                        relevance_root=pre_relevance,
                        approx_prompt_chars=len(node.question),
                        debug={"pruned": True, "reason": "low_relevance"},
                    )
                    node.metrics.append(metrics)
                    if self.event_callback:
                        self.event_callback(self, node, metrics)
                    return  # Skip LLM processing entirely

            # ============================================================
            # NORMAL PROCESSING (if not pruned)
            # ============================================================

            # Check if branch is approaching token limit and needs summarization
            parent_tokens = 0
            if node.parent_id:
                parent = self.nodes[node.parent_id]
                parent_tokens = parent.branch_tokens

            # Estimate entropy & get a candidate answer (cheap baseline)
            entropy_model, entropy_struct, entropy_consistency, candidate_answer, debug = (
                await self._estimate_entropy_and_answer(node)
            )

            # Compute semantic relevance to root (post-answer, more accurate)
            if self.root_id is not None:
                root_q = self.nodes[self.root_id].question
                relevance_root = semantic_relevance(
                    node.question + "\n" + candidate_answer, root_q
                )
                node.relevance_to_root = relevance_root
            else:
                relevance_root = None

            # Combine entropies
            E = (
                0.5 * entropy_model
                + 0.3 * entropy_struct
                + 0.2 * entropy_consistency
            )

            metrics = NodeMetrics(
                depth=depth,
                step_index=step_index,
                timestamp=t0,
                entropy_model=entropy_model,
                entropy_struct=entropy_struct,
                entropy_consistency=entropy_consistency,
                relevance_root=relevance_root,
                approx_prompt_chars=len(node.question),
                debug=debug,
            )
            token_usage = debug.get("token_usage", {})
            metrics.prompt_tokens = token_usage.get("prompt_tokens")
            metrics.completion_tokens = token_usage.get("completion_tokens")

            # Update branch token count
            node_tokens = (metrics.prompt_tokens or 0) + (metrics.completion_tokens or 0)
            node.branch_tokens = parent_tokens + node_tokens

            # Check if branch needs summarization (approaching max_branch_tokens)
            if node.branch_tokens > 0.8 * self.max_branch_tokens:
                if not node.summarized_context:  # Don't re-summarize
                    await self._summarize_branch(node)
                    metrics.debug["branch_summarized"] = True

            # Decide DECOMPOSE vs SOLVE
            if E > self.entropy_decompose_threshold and depth < self.max_depth:
                # Store pre-decomposition entropy for later comparison
                node.pre_decomp_entropy = E
                # Attempt decomposition
                await self._decompose_node(node, metrics, candidate_answer)
            else:
                # Use candidate answer as node answer
                node.answer = candidate_answer
                node.state = "solved"
                node.metrics.append(metrics)
                if self.event_callback:
                    self.event_callback(self, node, metrics)

    async def _decompose_node(
        self, node: ReasoningNode, metrics: NodeMetrics, fallback_answer: Optional[str] = None
    ) -> None:
        """
        Ask the LLM to decompose node.question into subproblems.

        Parameters
        ----------
        node : ReasoningNode
            The node to decompose.
        metrics : NodeMetrics
            Metrics object to update with token usage.
        fallback_answer : Optional[str]
            If decomposition fails, use this as the node's answer.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "INSTRUCTION: DECOMPOSE\n"
                    "You will rewrite the problem as a list of subquestions.\n"
                    "For each subquestion, specify:\n"
                    " - type: AND (must be solved) or OR (alternative strategy)\n"
                    " - short role\n"
                    "Return your answer as JSON with key 'subquestions',\n"
                    "each entry: {\"type\": \"AND|OR\", \"question\": \"...\", \"role\": \"...\"}.\n\n"
                    f"PROBLEM:\n{node.question}"
                ),
            },
        ]
        data = await self._chat(messages, temperature=0.2)
        p_tok, c_tok = self._record_token_usage(data)
        metrics.prompt_tokens = (metrics.prompt_tokens or 0) + p_tok
        metrics.completion_tokens = (metrics.completion_tokens or 0) + c_tok
        content = data.get("content", "").strip()
        node.raw_decomposition = content

        # Try to extract JSON
        subqs: List[Dict[str, str]] = []
        try:
            # Find first '{' and last '}' defensively
            s = content.find("{")
            e = content.rfind("}")
            if s != -1 and e != -1:
                js = content[s : e + 1]
                parsed = json.loads(js)
                subqs = parsed.get("subquestions", [])
        except Exception:
            pass

        if not subqs:
            # Decomposition failed; use fallback answer if available
            if fallback_answer:
                node.answer = fallback_answer
                node.state = "solved"
            else:
                node.state = "failed"
            node.metrics.append(metrics)
            if self.event_callback:
                self.event_callback(self, node, metrics)
            return

        # Create children
        children_ids: List[str] = []
        for sq in subqs:
            q_child = sq.get("question", "").strip()
            if not q_child:
                continue
            role = sq.get("role", "subproblem")
            child_id = self._new_node(
                parent_id=node.node_id,
                role=role,
                question=q_child,
            )
            children_ids.append(child_id)

        if not children_ids:
            # No valid children created; use fallback answer if available
            if fallback_answer:
                node.answer = fallback_answer
                node.state = "solved"
            else:
                node.state = "failed"
            node.metrics.append(metrics)
            if self.event_callback:
                self.event_callback(self, node, metrics)
            return

        node.state = "partially_solved"
        node.metrics.append(metrics)
        if self.event_callback:
            self.event_callback(self, node, metrics)

    # -----------------------------------------------------------------
    # Collapsing nodes
    # -----------------------------------------------------------------

    async def _try_collapses(self) -> None:
        """
        Try collapsing nodes whose children are all resolved (solved, failed, or pruned).

        Process:
        1. PRE-REVIEW: Internal quality scoring of each branch
        2. RANKING: Rank branches by alignment with root question
        3. PEER REVIEW: External validation (if configured) - constructive, not gatekeeping
        4. COLLAPSE: Integrate best branches into parent answer
        """
        # Get root question for alignment scoring
        root_question = ""
        if self.root_id and self.root_id in self.nodes:
            root_question = self.nodes[self.root_id].question

        for node in list(self.nodes.values()):
            if not node.children:
                continue
            if node.state == "solved":
                continue
            # Check if all children are resolved (solved, failed, or pruned)
            children = [self.nodes[cid] for cid in node.children]
            if not all(c.state in ("solved", "failed", "pruned") for c in children):
                continue
            # Filter out pruned/failed children for collapse
            valid_children = [c for c in children if c.state == "solved"]

            if not valid_children:
                # All children failed or were pruned
                node.state = "failed"
                node.answer = "[All subproblems failed or were pruned]"
                continue

            # === PHASE 1: PRE-REVIEW SCORING ===
            # Internal quality check before sending to peer (like author self-review)
            try:
                from peer_review import pre_review_branch, rank_branches_for_collapse

                # Rank branches by alignment with root question
                ranking = rank_branches_for_collapse(valid_children, root_question)
                ranked_list = ranking.get_ranked()

                # Log ranking for analysis
                if self.event_callback and ranked_list:
                    for node_id, score in ranked_list[:3]:  # Top 3
                        child = self.nodes.get(node_id)
                        if child:
                            # Store pre-review score in node for later analysis
                            if not hasattr(child, 'pre_review_score'):
                                child.pre_review_score = score

            except ImportError:
                # peer_review module not available
                ranked_list = [(c.node_id, 0.5) for c in valid_children]

            # === PHASE 2: PEER REVIEW (Constructive) ===
            # If peer reviewer is configured, review branches before collapse
            # Goal: HELP improve, not gatekeep
            if self.peer_reviewer is not None and valid_children:
                reviewed_children = []
                for child in valid_children:
                    if child.state == "solved" and not child.peer_reviewed:
                        try:
                            from peer_review import PeerTiming, PeerVerdict

                            eligible, reason = self.peer_reviewer.is_eligible(
                                child, self, PeerTiming.BEFORE_COLLAPSE
                            )
                            if eligible:
                                result = await self.peer_reviewer.review(child, self)
                                action = self.peer_reviewer.apply_verdict(result, child)

                                # Only kill if truly rejected (fundamentally wrong)
                                # NEEDS_REVISION branches are kept but flagged
                                if action == "killed" and result.verdict == PeerVerdict.REJECTED:
                                    # Only reject if score is very low
                                    if result.score < self.peer_reviewer.config.rejection_threshold:
                                        continue
                                    # Otherwise, keep but note the issues
                                    child.peer_feedback = result.feedback

                        except Exception as e:
                            # Peer review failed - continue without it
                            pass
                    reviewed_children.append(child)
                valid_children = reviewed_children

            # === PHASE 3: COLLAPSE ===
            if valid_children:
                # Sort by pre-review score (best first) before collapse
                try:
                    valid_children = sorted(
                        valid_children,
                        key=lambda c: getattr(c, 'pre_review_score', 0.5),
                        reverse=True
                    )
                except Exception:
                    pass

                await self._collapse_node(node, valid_children)
            else:
                # All children were rejected
                node.state = "failed"
                node.answer = "[All subproblems were rejected by peer review]"

    async def _collapse_node(self, node: ReasoningNode, children: List[ReasoningNode]) -> None:
        """
        Ask LLM to integrate child answers into parent's answer.
        """
        depth = self._node_depth(node.node_id)
        self._step_counter += 1
        step_index = self._step_counter
        t0 = time.time()

        summary_parts = []
        for i, c in enumerate(children, 1):
            summary_parts.append(
                f"[Child {i}] role={c.role}, state={c.state}\n"
                f"QUESTION: {c.question}\n"
                f"ANSWER:\n{c.answer or 'None'}\n"
            )
        children_summary = "\n\n".join(summary_parts)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "INSTRUCTION: COLLAPSE\n"
                    "You are given a parent question and several child subanswers.\n"
                    "Integrate them into a single, coherent answer to the parent.\n"
                    "Explicitly state remaining uncertainties and any contradictions.\n\n"
                    f"PARENT QUESTION:\n{node.question}\n\n"
                    f"SUBANSWERS:\n{children_summary}"
                ),
            },
        ]
        data = await self._chat(messages, temperature=0.3)
        p_tok, c_tok = self._record_token_usage(data)
        ans = data.get("content", "").strip()

        # Post-collapse entropy estimation:
        # - Check for uncertainty markers in the answer
        # - Short answers are more uncertain
        # - Answers with hedging language are more uncertain
        len_ans = len(ans)
        ans_lower = ans.lower()

        # Base entropy from answer length (normalized, short = high entropy)
        # Answers < 50 chars are likely too brief
        length_entropy = max(0.0, 1.0 - math.log(1 + len_ans) / math.log(1 + 500))

        # Hedging/uncertainty markers increase entropy
        uncertainty_markers = [
            "uncertain", "unclear", "unknown", "not sure", "might", "maybe",
            "possibly", "perhaps", "could be", "contradiction", "conflicting",
            "insufficient", "cannot determine", "more information needed"
        ]
        hedging_count = sum(1 for marker in uncertainty_markers if marker in ans_lower)
        hedging_entropy = min(1.0, hedging_count * 0.15)

        # Combine: weighted average
        entropy_model = 0.4 * length_entropy + 0.6 * hedging_entropy
        entropy_struct = 0.0
        entropy_consistency = 0.0

        metrics = NodeMetrics(
            depth=depth,
            step_index=step_index,
            timestamp=t0,
            entropy_model=entropy_model,
            entropy_struct=entropy_struct,
            entropy_consistency=entropy_consistency,
            approx_prompt_chars=len(node.question) + len(children_summary),
        )
        metrics.prompt_tokens = p_tok
        metrics.completion_tokens = c_tok

        node.answer = ans

        # ============================================================
        # AUTO-REBRANCH CHECK: Did decomposition increase entropy?
        # ============================================================
        # Compare post-collapse entropy with pre-decomposition entropy.
        # If entropy increased (or didn't decrease enough), and we haven't
        # exceeded rebranch attempts, trigger a rebranch with different strategy.
        post_collapse_entropy = entropy_model
        pre_decomp = node.pre_decomp_entropy

        should_rebranch = False
        if pre_decomp is not None:
            entropy_reduction = pre_decomp - post_collapse_entropy
            # Rebranch if entropy increased OR reduction is below threshold
            if entropy_reduction < self.entropy_gamma_min_reduction:
                if node.rebranch_count < self.max_rebranch_attempts:
                    should_rebranch = True

        if should_rebranch:
            # Entropy didn't improve sufficiently → rebranch
            node.rebranch_count += 1
            # Clear children and reset state for re-processing
            for cid in list(node.children):
                self._delete_subtree(cid)
            node.children = []
            node.state = "open"  # Will be re-processed in next loop
            node.answer = None
            node.raw_decomposition = None
            # Add debug info to metrics
            metrics.debug["rebranch_triggered"] = True
            metrics.debug["entropy_reduction"] = entropy_reduction if pre_decomp else None
            metrics.debug["rebranch_attempt"] = node.rebranch_count
            node.metrics.append(metrics)
            if self.event_callback:
                self.event_callback(self, node, metrics)
            return  # Don't mark as solved yet

        # Normal completion
        if post_collapse_entropy < self.entropy_collapse_threshold:
            node.state = "solved"
        else:
            # Node still unstable; keep it partially solved, may re-decompose later
            node.state = "partially_solved"

        node.metrics.append(metrics)
        if self.event_callback:
            self.event_callback(self, node, metrics)

    async def _force_root_collapse(self) -> None:
        """
        As a last resort, synthesize a final answer from the whole tree.
        """
        if self.root_id is None:
            return
        root = self.nodes[self.root_id]
        if root.answer:
            return

        # Build a textual dump of the tree (question/answer pairs)
        def _collect(nid: str, level: int = 0) -> List[str]:
            n = self.nodes[nid]
            indent = "  " * level
            s = f"{indent}- Q: {n.question}\n"
            if n.answer:
                s += f"{indent}  A: {n.answer}\n"
            out = [s]
            for cid in n.children:
                out.extend(_collect(cid, level + 1))
            return out

        tree_text = "\n".join(_collect(self.root_id))

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "INSTRUCTION: FINAL_SUMMARY\n"
                    "You are given the reasoning tree (questions and partial answers).\n"
                    "Produce the best possible final answer to the original question.\n"
                    "Mention limitations where necessary.\n\n"
                    f"ORIGINAL QUESTION:\n{root.question}\n\n"
                    f"REASONING TREE:\n{tree_text}"
                ),
            },
        ]
        data = await self._chat(messages, temperature=0.3)
        self._record_token_usage(data)
        root.answer = data.get("content", "").strip()
        root.state = "solved"

    # -----------------------------------------------------------------
    # Visualization / export helpers
    # -----------------------------------------------------------------

    def to_mermaid(self) -> str:
        """
        Export reasoning tree as a Mermaid graph (flowchart).
        """
        if self.root_id is None:
            return "graph TD\n"

        lines = ["graph TD"]
        for node in self.nodes.values():
            label = node.role.upper()
            if node.state == "solved":
                label += " ✅"
            elif node.state == "failed":
                label += " ❌"
            elif node.state == "partially_solved":
                label += " ⚠️"
            q_short = (node.question[:60] + "...") if len(node.question) > 60 else node.question
            lines.append(f'  {node.node_id[:8]}["{label}\\n{q_short}"]')
        for node in self.nodes.values():
            for cid in node.children:
                lines.append(f"  {node.node_id[:8]} --> {cid[:8]}")
        return "\n".join(lines)

    def export_trace(self) -> Dict[str, Any]:
        """
        Export the full reasoning trace as a JSON-serializable dict.
        """
        return {
            "model": self.model,
            "root_id": self.root_id,
            "nodes": {
                nid: {
                    "parent_id": n.parent_id,
                    "role": n.role,
                    "question": n.question,
                    "answer": n.answer,
                    "state": n.state,
                    "children": n.children,
                    "metrics": [dataclasses.asdict(m) for m in n.metrics],
                    "raw_decomposition": n.raw_decomposition,
                }
                for nid, n in self.nodes.items()
            },
        }

    def summarize(self) -> Dict[str, Any]:
        """
        Aggregate high-level metrics for CLI reporting or monitoring.
        """
        nodes = list(self.nodes.values())
        depth_counts: Dict[int, int] = {}
        state_counts: Dict[str, int] = {}
        entropy_model_values: List[float] = []
        entropy_struct_values: List[float] = []
        entropy_consistency_values: List[float] = []
        relevance_values: List[float] = []
        branch_token_values: List[int] = []
        summarized_count = 0

        for n in nodes:
            depth = self._node_depth(n.node_id)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            state_counts[n.state] = state_counts.get(n.state, 0) + 1
            # Track branch tokens and summarization
            if n.branch_tokens > 0:
                branch_token_values.append(n.branch_tokens)
            if n.summarized_context:
                summarized_count += 1

            for m in n.metrics:
                if m.entropy_model is not None:
                    entropy_model_values.append(m.entropy_model)
                if m.entropy_struct is not None:
                    entropy_struct_values.append(m.entropy_struct)
                if m.entropy_consistency is not None:
                    entropy_consistency_values.append(m.entropy_consistency)
                if m.relevance_root is not None:
                    relevance_values.append(m.relevance_root)

        def _stats(values: List[float]) -> Optional[Dict[str, float]]:
            if not values:
                return None
            return {
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
            }

        max_depth = max(depth_counts.keys()) if depth_counts else 0

        return {
            "total_nodes": len(nodes),
            "states": state_counts,
            "depth": {
                "max": max_depth,
                "breadth": depth_counts,
            },
            "steps": self._step_counter,
            "tokens": {
                "prompt": self._global_prompt_tokens,
                "completion": self._global_completion_tokens,
            },
            "entropies": {
                "model": _stats(entropy_model_values),
                "struct": _stats(entropy_struct_values),
                "consistency": _stats(entropy_consistency_values),
            },
            "relevance_root": _stats(relevance_values),
            "branch_tokens": {
                "max": max(branch_token_values) if branch_token_values else 0,
                "mean": mean(branch_token_values) if branch_token_values else 0,
            },
            "summarized_branches": summarized_count,
            "ctx_window": self._ctx_window,
        }


# ---------------------------------------------------------------------
# 4. Result wrapper
# ---------------------------------------------------------------------

@dataclass
class ReasoningResult:
    engine: ContractiveReasoner
    final_answer: Optional[str]
    root_id: Optional[str]

    @property
    def tree(self) -> Dict[str, ReasoningNode]:
        return self.engine.nodes

    def to_mermaid(self) -> str:
        return self.engine.to_mermaid()

    def export_trace(self) -> Dict[str, Any]:
        return self.engine.export_trace()

    def summarize(self) -> Dict[str, Any]:
        return self.engine.summarize()


def format_summary(summary: Dict[str, Any]) -> str:
    """
    Human-readable rendering of the engine summary.
    """
    states = summary.get("states", {})
    depth = summary.get("depth", {})
    entropies = summary.get("entropies", {})
    relevance = summary.get("relevance_root")
    tokens = summary.get("tokens", {})

    def _fmt_stats(label: str, stats: Optional[Dict[str, float]]) -> str:
        if not stats:
            return f"{label}=n/a"
        return f"{label}=min {stats['min']:.2f} | mean {stats['mean']:.2f} | max {stats['max']:.2f}"

    lines = [
        f"Nodes: total={summary.get('total_nodes', 0)} "
        f"solved={states.get('solved', 0)} "
        f"open={states.get('open', 0)} "
        f"partial={states.get('partially_solved', 0)} "
        f"failed={states.get('failed', 0)} "
        f"pruned={states.get('pruned', 0)}",
        f"Depth: max={depth.get('max', 0)} breadth={depth.get('breadth', {})}",
        _fmt_stats("Entropy(model)", entropies.get("model")),
        _fmt_stats("Entropy(struct)", entropies.get("struct")),
        _fmt_stats("Entropy(consistency)", entropies.get("consistency")),
        _fmt_stats("Relevance(root)", relevance),
        f"Tokens: prompt={tokens.get('prompt', 0)} completion={tokens.get('completion', 0)}",
        f"Branch tokens: max={summary.get('branch_tokens', {}).get('max', 0)} "
        f"mean={summary.get('branch_tokens', {}).get('mean', 0):.0f} "
        f"summarized={summary.get('summarized_branches', 0)}",
        f"Steps: {summary.get('steps', 0)}  Context window: {summary.get('ctx_window') or 'n/a'}",
    ]
    return "\n".join(lines)


def parse_duration(value: str) -> int:
    """
    Parse a duration string into seconds.

    Supported formats:
    - "180" or "180s" -> 180 seconds
    - "3m" or "3m0s" -> 180 seconds
    - "2m30s" -> 150 seconds
    - "1h" or "1h0m" -> 3600 seconds
    - "1h30m" -> 5400 seconds

    Returns integer seconds.
    """
    import re

    if isinstance(value, (int, float)):
        return int(value)

    value = str(value).strip().lower()

    # Pure number (seconds)
    if value.isdigit():
        return int(value)

    total_seconds = 0

    # Match hours, minutes, seconds patterns
    hour_match = re.search(r'(\d+)\s*h', value)
    min_match = re.search(r'(\d+)\s*m(?!s)', value)  # m but not ms
    sec_match = re.search(r'(\d+)\s*s', value)

    if hour_match:
        total_seconds += int(hour_match.group(1)) * 3600
    if min_match:
        total_seconds += int(min_match.group(1)) * 60
    if sec_match:
        total_seconds += int(sec_match.group(1))

    # If no pattern matched but ends with 's', try stripping it
    if total_seconds == 0 and value.endswith('s'):
        try:
            total_seconds = int(value[:-1])
        except ValueError:
            pass

    if total_seconds == 0:
        raise ValueError(f"Cannot parse duration: {value}")

    return total_seconds


def _load_config_file(path: str) -> Dict[str, Any]:
    """
    Load a config file (YAML or JSON) into a dict.
    """
    lower = path.lower()
    if lower.endswith((".yaml", ".yml")):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs. Install with `pip install pyyaml`.")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    # default: JSON
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_config_overrides(args: argparse.Namespace, defaults: Dict[str, Any]) -> argparse.Namespace:
    """
    Override argparse defaults with values from a config file when provided.
    CLI flags take precedence over config values.

    Special handling:
    - Duration fields (timeout_sec, peer_timeout_sec) accept strings like "3m0s"
    - Nested 'peer' section is flattened with 'peer_' prefix
    """
    cfg_path = getattr(args, "config", None)
    if not cfg_path:
        return args

    cfg = _load_config_file(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must define a mapping (key/value pairs).")

    # Flatten nested 'peer' section if present
    # Maps config keys to CLI argument names
    peer_key_map = {
        "model": "peer_model",
        "base_url": "peer_base_url",
        "timeout_sec": "peer_timeout_sec",
        "timing": "peer_timing",
        "min_depth": "peer_min_depth",
        "min_nodes": "peer_min_nodes",
        "max_per_branch": "peer_max_per_branch",
        "max_calls": "peer_max_calls",
        "token_budget": "peer_token_budget",
        "approval_threshold": "peer_approval_threshold",
        "rejection_threshold": "peer_rejection_threshold",
        "kill_rejected": "peer_kill_rejected",
        "force_root": "peer_force_root",
        "as_tiebreaker": "peer_as_tiebreaker",
        "log_prompts": "peer_log_prompts",
        "experiment": "peer_experiment",
    }
    if "peer" in cfg and isinstance(cfg["peer"], dict):
        peer_cfg = cfg.pop("peer")
        for k, v in peer_cfg.items():
            # Map to CLI argument name
            flat_key = peer_key_map.get(k, f"peer_{k}")
            cfg[flat_key] = v

    # Fields that accept duration strings
    duration_fields = {"timeout_sec", "peer_timeout_sec"}

    for key, value in cfg.items():
        # Convert dashes to underscores for argparse compatibility
        attr_key = key.replace("-", "_")

        if not hasattr(args, attr_key):
            continue

        current = getattr(args, attr_key)
        default = defaults.get(attr_key)

        # Only override if current value equals the default (CLI didn't override)
        if current == default:
            # Handle duration fields
            if attr_key in duration_fields and isinstance(value, str):
                value = parse_duration(value)
            setattr(args, attr_key, value)

    return args


# ---------------------------------------------------------------------
# 5. CLI helper
# ---------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Contractive reasoning engine with telemetry.",
    )
    parser.add_argument("question", type=str, nargs="*", help="Question to ask.")

    parser.add_argument("--model", type=str, default="granite3.1-moe:3b")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434")

    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-loops", type=int, default=6)
    parser.add_argument("--max-global-tokens", type=int, default=64000)
    parser.add_argument("--max-branch-tokens", type=int, default=16000)
    parser.add_argument("--max-concurrent-branches", type=int, default=4)

    parser.add_argument("--entropy-decompose-threshold", type=float, default=0.9)
    parser.add_argument("--entropy-collapse-threshold", type=float, default=0.4)
    parser.add_argument("--entropy-gamma-min-reduction", type=float, default=0.05)
    parser.add_argument("--entropy-samples", type=int, default=4)
    parser.add_argument(
        "--min-relevance-threshold",
        type=float,
        default=0.15,
        help="Minimum semantic relevance to root; branches below this are pruned.",
    )
    parser.add_argument(
        "--max-rebranch-attempts",
        type=int,
        default=2,
        help="Max auto-rebranch attempts when decomposition increases entropy.",
    )
    parser.add_argument("--timeout-sec", type=int, default=120)

    parser.add_argument("--export-trace", type=str, help="Write full trace JSON to path.")
    parser.add_argument("--export-mermaid", type=str, help="Write Mermaid graph to path.")
    parser.add_argument("--log-events", type=str, help="Write per-step NDJSON metrics to path.")
    parser.add_argument(
        "--print-events",
        action="store_true",
        help="Print per-step entropy/relevance rows during execution.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip printing the full answer, keep telemetry only.",
    )
    parser.add_argument(
        "--no-mermaid",
        action="store_true",
        help="Do not print Mermaid to stdout (useful when piping).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=90,
        help="Wrap width for printed answers.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Load CLI parameters from a YAML or JSON config file (CLI flags override).",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive loop: repeatedly ask questions until exit.",
    )

    # Peer-review arguments
    parser.add_argument(
        "--peer-model",
        type=str,
        default=None,
        help="Peer model for external validation (e.g., mistral:7b-instruct). Enables peer review.",
    )
    parser.add_argument(
        "--peer-timing",
        type=str,
        default="before_collapse",
        choices=["before_collapse", "end_of_reasoning", "on_demand"],
        help="When to trigger peer review.",
    )
    parser.add_argument(
        "--peer-min-depth",
        type=int,
        default=2,
        help="Minimum branch depth before peer review is eligible.",
    )
    parser.add_argument(
        "--peer-min-nodes",
        type=int,
        default=3,
        help="Minimum nodes in branch before peer review is eligible.",
    )
    parser.add_argument(
        "--peer-max-calls",
        type=int,
        default=10,
        help="Maximum total peer review calls per solve.",
    )
    parser.add_argument(
        "--peer-token-budget",
        type=int,
        default=50000,
        help="Maximum token budget for peer review calls.",
    )
    parser.add_argument(
        "--peer-kill-rejected",
        action="store_true",
        default=True,
        help="Kill branches rejected by peer (default: True).",
    )
    parser.add_argument(
        "--no-peer-kill-rejected",
        action="store_true",
        help="Do not kill branches rejected by peer (only penalize).",
    )
    parser.add_argument(
        "--peer-force-root",
        action="store_true",
        help="Force peer review of root answer at end of reasoning.",
    )
    parser.add_argument(
        "--peer-log-prompts",
        action="store_true",
        help="Log full peer review prompts for analysis.",
    )
    parser.add_argument(
        "--peer-experiment",
        type=str,
        default=None,
        choices=["no_peer", "peer_before_collapse", "peer_final_only",
                 "peer_aggressive", "peer_conservative", "peer_budget_limited",
                 "peer_large_model"],
        help="Use a predefined peer review experiment configuration.",
    )
    parser.add_argument(
        "--peer-base-url",
        type=str,
        default=None,
        help="Base URL for peer model (defaults to --base-url if not set).",
    )
    parser.add_argument(
        "--peer-timeout-sec",
        type=int,
        default=180,
        help="Timeout for peer review calls in seconds (or duration string like '3m').",
    )
    parser.add_argument(
        "--peer-approval-threshold",
        type=float,
        default=0.6,
        help="Score >= this value is approved (default: 0.6).",
    )
    parser.add_argument(
        "--peer-rejection-threshold",
        type=float,
        default=0.3,
        help="Score <= this value is rejected (default: 0.3).",
    )
    parser.add_argument(
        "--peer-max-per-branch",
        type=int,
        default=1,
        help="Maximum peer review calls per branch path (default: 1).",
    )
    parser.add_argument(
        "--peer-as-tiebreaker",
        action="store_true",
        help="Only call peer on high-entropy (uncertain) nodes.",
    )
    return parser


async def run_cli(args: argparse.Namespace) -> None:
    question = " ".join(args.question).strip()

    # Setup peer reviewer if configured
    peer_reviewer = None
    if args.peer_model or args.peer_experiment:
        try:
            from peer_review import PeerConfig, PeerReviewer, PeerTiming, create_experiment_configs

            if args.peer_experiment:
                # Use predefined experiment config
                configs = create_experiment_configs()
                peer_config = configs.get(args.peer_experiment)
                if peer_config is None:
                    print(f"[WARN] Unknown peer experiment: {args.peer_experiment}")
                else:
                    # Override peer model if specified
                    if args.peer_model:
                        peer_config.peer_model = args.peer_model
                    peer_reviewer = PeerReviewer(peer_config)
            elif args.peer_model:
                # Build config from CLI args
                timing_map = {
                    "before_collapse": PeerTiming.BEFORE_COLLAPSE,
                    "end_of_reasoning": PeerTiming.END_OF_REASONING,
                    "on_demand": PeerTiming.ON_DEMAND,
                }
                # Use peer_base_url if provided, else fall back to main base_url
                peer_url = args.peer_base_url if args.peer_base_url else args.base_url
                peer_config = PeerConfig(
                    enabled=True,
                    peer_model=args.peer_model,
                    peer_base_url=peer_url,
                    peer_timeout_sec=args.peer_timeout_sec,
                    timing=timing_map.get(args.peer_timing, PeerTiming.BEFORE_COLLAPSE),
                    min_branch_depth=args.peer_min_depth,
                    min_branch_nodes=args.peer_min_nodes,
                    max_peer_calls_per_branch=args.peer_max_per_branch,
                    max_peer_calls_total=args.peer_max_calls,
                    peer_token_budget=args.peer_token_budget,
                    approval_threshold=args.peer_approval_threshold,
                    rejection_threshold=args.peer_rejection_threshold,
                    kill_rejected_branches=args.peer_kill_rejected and not args.no_peer_kill_rejected,
                    force_peer_on_root=args.peer_force_root,
                    log_peer_prompts=args.peer_log_prompts,
                    peer_as_tiebreaker=args.peer_as_tiebreaker,
                )
                peer_reviewer = PeerReviewer(peer_config)

            if peer_reviewer:
                print(f"[PEER] Enabled with model={peer_reviewer.config.peer_model}, "
                      f"timing={peer_reviewer.config.timing.value}")
        except ImportError as e:
            print(f"[WARN] Peer review module not available: {e}")

    async def _run_once(q: str) -> None:
        engine = ContractiveReasoner(
            base_url=args.base_url,
            model=args.model,
            max_depth=args.max_depth,
            max_loops=args.max_loops,
            max_global_tokens=args.max_global_tokens,
            max_branch_tokens=args.max_branch_tokens,
            max_concurrent_branches=args.max_concurrent_branches,
            entropy_decompose_threshold=args.entropy_decompose_threshold,
            entropy_collapse_threshold=args.entropy_collapse_threshold,
            entropy_gamma_min_reduction=args.entropy_gamma_min_reduction,
            k_entropy_samples=args.entropy_samples,
            min_relevance_threshold=args.min_relevance_threshold,
            max_rebranch_attempts=args.max_rebranch_attempts,
            timeout_sec=args.timeout_sec,
        )
        event_log: List[Dict[str, Any]] = []

        def _event_callback(engine: ContractiveReasoner, node: ReasoningNode, metrics: NodeMetrics) -> None:
            event = {
                "step": metrics.step_index,
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "role": node.role,
                "state": node.state,
                "depth": metrics.depth,
                "entropy_model": metrics.entropy_model,
                "entropy_struct": metrics.entropy_struct,
                "entropy_consistency": metrics.entropy_consistency,
                "relevance_root": metrics.relevance_root,
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "timestamp": metrics.timestamp,
            }
            event_log.append(event)
            if args.print_events:
                em = metrics.entropy_model if metrics.entropy_model is not None else 0.0
                es = metrics.entropy_struct if metrics.entropy_struct is not None else 0.0
                ec = metrics.entropy_consistency if metrics.entropy_consistency is not None else 0.0
                rel = metrics.relevance_root
                rel_str = f"{rel:.3f}" if rel is not None else "n/a"
                print(
                    f"[step {metrics.step_index:02d}] depth={metrics.depth} state={node.state:<16} "
                    f"E=({em:.3f},{es:.3f},{ec:.3f}) rel={rel_str} role={node.role}"
                )

        engine.event_callback = _event_callback

        # Attach peer reviewer if configured
        if peer_reviewer is not None:
            engine.peer_reviewer = peer_reviewer
            peer_reviewer.reset()  # Reset metrics for each question

        res = await engine.solve(q, max_depth=args.max_depth, max_loops=args.max_loops)

        # Handle final peer review if configured
        if peer_reviewer is not None and peer_reviewer.config.force_peer_on_root:
            try:
                from peer_review import peer_review_final
                final_result = await peer_review_final(engine, peer_reviewer)
                if args.print_events:
                    print(f"[PEER] Final review: {final_result.verdict.value} "
                          f"(score={final_result.score:.2f})")
            except Exception as e:
                if args.print_events:
                    print(f"[PEER] Final review failed: {e}")

        if not args.summary_only:
            print("\n=== FINAL ANSWER ===\n")
            print(textwrap.fill(res.final_answer or "<no answer>", width=args.width))

        if not args.no_mermaid:
            print("\n=== MERMAID ===\n")
            print(res.to_mermaid())

        summary = res.summarize()
        print("\n=== SUMMARY ===\n")
        print(format_summary(summary))

        # Print peer review metrics if available
        if peer_reviewer is not None and peer_reviewer.metrics.total_calls > 0:
            print("\n=== PEER REVIEW METRICS ===\n")
            pm = peer_reviewer.metrics
            print(f"  Total calls:      {pm.total_calls}")
            print(f"  Verdicts:         approved={pm.approved_count}, rejected={pm.rejected_count}, "
                  f"revision={pm.revision_count}, uncertain={pm.uncertain_count}")
            print(f"  Peer tokens:      prompt={pm.total_prompt_tokens}, completion={pm.total_completion_tokens}")
            print(f"  Peer time:        {pm.total_elapsed_sec:.2f}s")
            if pm.reviews:
                print(f"  Reviews:")
                for r in pm.reviews:
                    print(f"    - {r.node_id[:8]}... : {r.verdict.value} (score={r.score:.2f})")

        if args.export_trace:
            with open(args.export_trace, "w", encoding="utf-8") as f:
                json.dump(res.export_trace(), f, indent=2, ensure_ascii=False)

        if args.export_mermaid:
            with open(args.export_mermaid, "w", encoding="utf-8") as f:
                f.write(res.to_mermaid())

        if args.log_events:
            mode = "a" if args.chat else "w"
            with open(args.log_events, mode, encoding="utf-8") as f:
                for ev in event_log:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    try:
        if args.chat:
            print("Chat mode: type 'exit' or 'quit' to stop.")
            while True:
                q = question or input("Q> ").strip()
                question = ""  # only use CLI question once
                if not q:
                    continue
                if q.lower() in {"exit", "quit"}:
                    break
                await _run_once(q)
        else:
            if not question:
                question = "Design a stepwise protocol to evaluate the safety of a new food contact polymer."
            await _run_once(question)
    finally:
        # Cleanup peer reviewer
        if peer_reviewer is not None:
            await peer_reviewer.close()


if __name__ == "__main__":
    parser = build_arg_parser()
    defaults = vars(parser.parse_args([]))
    parsed_args = parser.parse_args()
    parsed_args = _apply_config_overrides(parsed_args, defaults)
    asyncio.run(run_cli(parsed_args))
