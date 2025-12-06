#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peer Review Module for ContractiveReasoner
==========================================

Implements external LLM peer-review for validation of reasoning branches.

Design Philosophy (Academic Peer Review Model):
-----------------------------------------------
The peer reviewer acts as a **constructive academic reviewer**, not a gatekeeper.
Its goal is to HELP the reasoning process, not to kill branches arbitrarily.

Key principles:
1. PRE-REVIEW: Internal quality check before sending to peer (like self-review)
2. ALIGNMENT: Measure how well each branch answers the original question
3. RANKING: Prefer branches by quality score, don't just kill/keep
4. CONSTRUCTIVE FEEDBACK: Suggest improvements, not just verdicts
5. CONFIDENCE METRICS: Provide pertinence scores to guide the peer

This mirrors the scientific peer review process:
- Author (main LLM) submits work
- Internal review (pre-review scoring) checks basic quality
- External peer (peer LLM) provides constructive feedback
- Revision cycle if needed (not immediate rejection)

Key Features:
- Configurable peer models (can use different/larger LLMs)
- Branch maturity checks (only review developed branches)
- Budget-aware (tracks peer tokens separately)
- Pre-review scoring (internal alignment check)
- Preference ranking (Borda count for multiple branches)
- Experimental modes for academic comparison

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio Innovation Lab
"""

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ContractiveReasoner import ContractiveReasoner, ReasoningNode


# =============================================================================
# PRE-REVIEW SCORING (Internal Quality Check)
# =============================================================================

@dataclass
class PreReviewScore:
    """
    Internal quality assessment before sending to external peer.

    This is like the author's self-review before submission:
    - Does the answer address the question?
    - Is there semantic alignment with the root question?
    - What is the confidence level?
    """
    node_id: str

    # Alignment metrics (0.0 to 1.0)
    semantic_alignment: float = 0.0     # How well answer aligns with question
    topic_coverage: float = 0.0         # Coverage of key topics from question
    answer_completeness: float = 0.0    # Does it look like a complete answer?

    # Confidence indicators
    hedging_ratio: float = 0.0          # Ratio of hedging words (uncertainty)
    assertion_strength: float = 0.0     # Strength of claims made

    # Composite score
    overall_score: float = 0.0

    # Recommendation
    ready_for_peer: bool = True
    issues: List[str] = field(default_factory=list)

    def compute_overall(self) -> float:
        """Compute weighted overall score."""
        weights = {
            "semantic_alignment": 0.35,
            "topic_coverage": 0.25,
            "answer_completeness": 0.20,
            "assertion_strength": 0.20,
        }
        score = (
            weights["semantic_alignment"] * self.semantic_alignment +
            weights["topic_coverage"] * self.topic_coverage +
            weights["answer_completeness"] * self.answer_completeness +
            weights["assertion_strength"] * self.assertion_strength
        )
        # Penalize high hedging
        score *= (1.0 - 0.3 * self.hedging_ratio)
        self.overall_score = max(0.0, min(1.0, score))
        return self.overall_score


@dataclass
class BranchRanking:
    """
    Preference-based ranking of multiple branches before collapse.

    Instead of binary kill/keep, we rank branches by quality.
    This allows selecting the best branch(es) for collapse.
    """
    rankings: List[Tuple[str, float]] = field(default_factory=list)  # (node_id, score)

    def add(self, node_id: str, score: float):
        """Add a branch with its score."""
        self.rankings.append((node_id, score))

    def get_ranked(self, descending: bool = True) -> List[Tuple[str, float]]:
        """Get branches sorted by score."""
        return sorted(self.rankings, key=lambda x: x[1], reverse=descending)

    def get_top_n(self, n: int) -> List[str]:
        """Get top N branch IDs."""
        ranked = self.get_ranked()
        return [node_id for node_id, _ in ranked[:n]]

    def borda_count(self) -> Dict[str, int]:
        """
        Compute Borda count scores for ranking.
        Each position gives points: last=0, ..., first=N-1
        """
        ranked = self.get_ranked()
        n = len(ranked)
        return {node_id: (n - 1 - i) for i, (node_id, _) in enumerate(ranked)}


def _tokenize_simple(text: str) -> List[str]:
    """Simple tokenization for alignment scoring."""
    return [t.lower() for t in re.findall(r'\b\w+\b', text) if len(t) > 2]


def compute_semantic_alignment(answer: str, question: str) -> float:
    """
    Compute semantic alignment between answer and question.

    Uses token overlap with IDF-like weighting.
    """
    q_tokens = set(_tokenize_simple(question))
    a_tokens = set(_tokenize_simple(answer))

    if not q_tokens or not a_tokens:
        return 0.0

    # Jaccard similarity
    intersection = len(q_tokens & a_tokens)
    union = len(q_tokens | a_tokens)
    jaccard = intersection / union if union > 0 else 0.0

    # Question coverage (what fraction of question words appear in answer)
    q_coverage = intersection / len(q_tokens) if q_tokens else 0.0

    # Weighted combination
    return 0.6 * q_coverage + 0.4 * jaccard


def compute_topic_coverage(answer: str, question: str) -> float:
    """
    Check if key topics from question are addressed in answer.

    Identifies noun phrases and checks coverage.
    """
    # Simple extraction of likely key terms (capitalized, longer words)
    q_tokens = _tokenize_simple(question)
    a_tokens_set = set(_tokenize_simple(answer))

    # Weight longer words as more important
    key_terms = [t for t in q_tokens if len(t) > 4]
    if not key_terms:
        key_terms = q_tokens

    if not key_terms:
        return 0.5  # No clear key terms

    covered = sum(1 for t in key_terms if t in a_tokens_set)
    return covered / len(key_terms)


def compute_answer_completeness(answer: str) -> float:
    """
    Estimate if answer looks complete (not truncated, has structure).
    """
    if not answer or len(answer.strip()) < 10:
        return 0.0

    score = 0.5  # Base score for having content

    # Length bonus (up to 500 chars)
    length_score = min(len(answer) / 500, 1.0) * 0.2
    score += length_score

    # Structure indicators
    if any(c in answer for c in '.!?'):  # Has sentence endings
        score += 0.1
    if '\n' in answer or any(answer.count(c) > 1 for c in ',-;:'):  # Structure
        score += 0.1
    if answer.strip()[-1] in '.!?"\'':  # Proper ending
        score += 0.1

    return min(score, 1.0)


def compute_hedging_ratio(text: str) -> float:
    """
    Compute ratio of hedging/uncertainty words in text.

    High hedging suggests low confidence.
    """
    hedging_words = {
        'maybe', 'perhaps', 'possibly', 'might', 'could', 'may', 'probably',
        'uncertain', 'unclear', 'unsure', 'approximately', 'roughly', 'about',
        'seems', 'appears', 'likely', 'unlikely', 'potentially', 'presumably',
        'supposedly', 'apparently', 'somewhat', 'fairly', 'rather', 'quite'
    }

    tokens = _tokenize_simple(text)
    if not tokens:
        return 0.0

    hedge_count = sum(1 for t in tokens if t in hedging_words)
    return hedge_count / len(tokens)


def compute_assertion_strength(text: str) -> float:
    """
    Estimate strength of assertions in text.

    Strong assertions suggest confidence.
    """
    strong_words = {
        'is', 'are', 'was', 'were', 'will', 'must', 'always', 'never',
        'definitely', 'certainly', 'clearly', 'obviously', 'absolutely',
        'exactly', 'precisely', 'correct', 'true', 'false', 'fact'
    }

    tokens = _tokenize_simple(text)
    if not tokens:
        return 0.5

    strong_count = sum(1 for t in tokens if t in strong_words)
    ratio = strong_count / len(tokens)

    # Normalize to 0-1 range (typical ratio is 0.05-0.15)
    return min(ratio * 5, 1.0)


def pre_review_branch(
    node: "ReasoningNode",
    root_question: str,
    min_score_for_peer: float = 0.3,
) -> PreReviewScore:
    """
    Perform internal pre-review of a branch before sending to peer.

    Parameters
    ----------
    node : ReasoningNode
        The node to pre-review.
    root_question : str
        The original root question.
    min_score_for_peer : float
        Minimum score to recommend for peer review.

    Returns
    -------
    PreReviewScore
        Pre-review assessment.
    """
    answer = node.answer or ""
    question = node.question or root_question

    score = PreReviewScore(node_id=node.node_id)

    # Compute metrics
    score.semantic_alignment = compute_semantic_alignment(answer, root_question)
    score.topic_coverage = compute_topic_coverage(answer, root_question)
    score.answer_completeness = compute_answer_completeness(answer)
    score.hedging_ratio = compute_hedging_ratio(answer)
    score.assertion_strength = compute_assertion_strength(answer)

    # Compute overall
    score.compute_overall()

    # Determine readiness
    issues = []
    if score.semantic_alignment < 0.2:
        issues.append("Low alignment with original question")
    if score.topic_coverage < 0.3:
        issues.append("Key topics not addressed")
    if score.answer_completeness < 0.3:
        issues.append("Answer appears incomplete")
    if score.hedging_ratio > 0.15:
        issues.append("High uncertainty in answer")

    score.issues = issues
    score.ready_for_peer = score.overall_score >= min_score_for_peer

    return score


def rank_branches_for_collapse(
    branches: List["ReasoningNode"],
    root_question: str,
) -> BranchRanking:
    """
    Rank multiple branches by quality before collapse.

    Instead of binary decisions, rank branches to select the best.

    Parameters
    ----------
    branches : List[ReasoningNode]
        Branches to rank.
    root_question : str
        The original root question.

    Returns
    -------
    BranchRanking
        Ranked branches with scores.
    """
    ranking = BranchRanking()

    for branch in branches:
        pre_score = pre_review_branch(branch, root_question)
        ranking.add(branch.node_id, pre_score.overall_score)

    return ranking


# =============================================================================
# PEER REVIEW CONFIGURATION
# =============================================================================

class PeerVerdict(Enum):
    """Possible peer review verdicts."""
    APPROVED = "approved"           # Answer is valid, branch can proceed
    REJECTED = "rejected"           # Answer is wrong, branch should be killed
    NEEDS_REVISION = "needs_revision"  # Answer has issues, needs improvement
    UNCERTAIN = "uncertain"         # Peer cannot determine validity
    SKIP = "skip"                   # Branch not eligible for review


class PeerTiming(Enum):
    """When peer review can be triggered."""
    BEFORE_COLLAPSE = "before_collapse"  # Review branches before collapsing
    AFTER_COLLAPSE = "after_collapse"    # Review collapsed answers
    END_OF_REASONING = "end_of_reasoning"  # Final review of root answer
    ON_DEMAND = "on_demand"              # Only when explicitly called


@dataclass
class PeerConfig:
    """Configuration for peer review behavior."""
    # Peer model settings
    peer_model: str = "mistral:7b-instruct"
    peer_base_url: str = "http://localhost:11434"
    peer_timeout_sec: int = 180

    # Eligibility criteria
    enabled: bool = True
    timing: PeerTiming = PeerTiming.BEFORE_COLLAPSE
    min_branch_depth: int = 2           # Minimum depth before peer can be called
    min_branch_nodes: int = 3           # Minimum nodes in branch before review
    max_peer_calls_per_branch: int = 1  # Limit peer calls per branch path
    max_peer_calls_total: int = 10      # Global limit on peer calls

    # Budget controls
    peer_token_budget: int = 50000      # Max tokens for all peer calls

    # Review behavior
    approval_threshold: float = 0.6     # Score >= this is approved
    rejection_threshold: float = 0.3    # Score <= this is rejected
    kill_rejected_branches: bool = True # Whether to mark rejected branches as failed

    # Experimental flags (for paper)
    log_peer_prompts: bool = True       # Log full prompts for analysis
    force_peer_on_root: bool = False    # Always peer-review root answer
    peer_as_tiebreaker: bool = False    # Only call peer on high-entropy nodes


@dataclass
class PeerReviewResult:
    """Result of a single peer review."""
    node_id: str
    verdict: PeerVerdict
    score: float                        # 0.0 to 1.0
    feedback: str
    reasoning: str                      # Peer's explanation

    # Metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_sec: float = 0.0

    # For analysis
    peer_model: str = ""
    timestamp: float = 0.0
    raw_response: str = ""


@dataclass
class PeerMetrics:
    """Aggregate metrics for peer review analysis."""
    total_calls: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    revision_count: int = 0
    uncertain_count: int = 0
    skipped_count: int = 0

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_elapsed_sec: float = 0.0

    # Agreement analysis
    agreement_with_main: int = 0        # Peer agreed with main model
    disagreement_with_main: int = 0     # Peer disagreed

    # Per-review details (for paper)
    reviews: List[PeerReviewResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics for analysis."""
        return {
            "total_calls": self.total_calls,
            "verdicts": {
                "approved": self.approved_count,
                "rejected": self.rejected_count,
                "needs_revision": self.revision_count,
                "uncertain": self.uncertain_count,
                "skipped": self.skipped_count,
            },
            "tokens": {
                "prompt": self.total_prompt_tokens,
                "completion": self.total_completion_tokens,
                "total": self.total_prompt_tokens + self.total_completion_tokens,
            },
            "time_sec": self.total_elapsed_sec,
            "agreement_rate": (
                self.agreement_with_main / max(1, self.total_calls - self.skipped_count)
            ),
        }


# =============================================================================
# PEER REVIEWER CLASS
# =============================================================================

class PeerReviewer:
    """
    External peer reviewer for ContractiveReasoner branches.

    The peer acts as an independent validator - given the original question
    and the proposed answer, it assesses validity and provides feedback.

    Design principles:
    - Peer has NO access to intermediate reasoning (only Q and A)
    - Peer uses a different model to avoid self-reinforcing errors
    - Peer calls are expensive and should be used judiciously
    """

    # ==========================================================================
    # CONSTRUCTIVE PEER REVIEW PROMPT
    # ==========================================================================
    # Designed to mirror academic peer review: help improve, not gatekeep.
    # The peer's goal is to assist the reasoning process, not to kill branches.
    #
    # Key principles:
    # 1. Constructive feedback over binary rejection
    # 2. Specific, actionable suggestions
    # 3. Recognition of partial correctness
    # 4. Clear indication of confidence level
    # ==========================================================================

    PEER_REVIEW_PROMPT = """INSTRUCTION: CONSTRUCTIVE PEER REVIEW

You are an academic peer reviewer helping to improve a colleague's work.
Your role is CONSTRUCTIVE: help strengthen the answer, not simply reject it.

IMPORTANT PRINCIPLES:
- An imperfect answer that addresses the question is better than no answer
- Partial correctness should be acknowledged and built upon
- Rejection is reserved for fundamentally wrong or harmful answers
- Always provide specific, actionable suggestions for improvement

ORIGINAL QUESTION:
{question}

PROPOSED ANSWER:
{answer}

PRE-REVIEW ASSESSMENT (internal quality check):
- Alignment score: {alignment:.2f}/1.0 (how well answer addresses question)
- Completeness: {completeness:.2f}/1.0 (structural completeness)
- Confidence: {confidence:.2f}/1.0 (assertion strength vs hedging)
{issues_text}

YOUR TASK:
1. Assess if the answer ADDRESSES the original question (relevance)
2. Check for factual accuracy (within your knowledge)
3. Identify what is CORRECT and should be preserved
4. Suggest specific improvements (not vague criticism)
5. Provide an overall recommendation

Respond with a JSON object:
{{
    "verdict": "approved" | "minor_revision" | "major_revision" | "reject",
    "score": <float 0.0 to 1.0>,
    "addresses_question": <bool>,
    "factually_correct": <bool or "partially" or "cannot_verify">,
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", ...],
    "suggestions": ["<specific actionable suggestion 1>", ...],
    "confidence": <float 0.0 to 1.0 - your confidence in this review>,
    "reasoning": "<brief explanation of your verdict>"
}}

VERDICT GUIDELINES:
- "approved": Answer is acceptable, addresses question, no major issues
- "minor_revision": Good foundation, needs small improvements
- "major_revision": Has merit but needs significant work
- "reject": Fundamentally wrong, off-topic, or harmful (use sparingly!)

Remember: Your goal is to HELP improve the reasoning, not to gatekeep.
A "minor_revision" with good suggestions is more valuable than a harsh "reject".
"""

    # Alternative prompt for final root review (more comprehensive)
    FINAL_REVIEW_PROMPT = """INSTRUCTION: FINAL ANSWER REVIEW

You are reviewing the FINAL synthesized answer to a complex question.
This answer was produced by decomposing the problem and integrating sub-answers.

ORIGINAL QUESTION:
{question}

FINAL SYNTHESIZED ANSWER:
{answer}

SYNTHESIS SUMMARY:
- Total reasoning steps: {total_nodes}
- Branches explored: {total_branches}
- Pre-review score: {pre_score:.2f}/1.0

Evaluate the final answer holistically:
1. Does it fully answer the original question?
2. Is the synthesis coherent (not contradictory)?
3. Are important aspects missing?
4. Is the confidence level appropriate?

Respond with a JSON object:
{{
    "verdict": "accept" | "accept_with_notes" | "needs_work" | "insufficient",
    "overall_score": <float 0.0 to 1.0>,
    "question_answered": <bool>,
    "coherent_synthesis": <bool>,
    "missing_aspects": ["<aspect 1>", ...],
    "quality_notes": "<assessment of answer quality>",
    "improvement_priority": ["<most important improvement>", ...],
    "confidence": <float 0.0 to 1.0>
}}

Focus on whether the answer SERVES THE USER'S NEED, not on perfection.
"""

    def __init__(self, config: Optional[PeerConfig] = None):
        """
        Initialize the peer reviewer.

        Parameters
        ----------
        config : PeerConfig, optional
            Configuration for peer review behavior.
        """
        self.config = config or PeerConfig()
        self.metrics = PeerMetrics()
        self._client: Optional[httpx.AsyncClient] = None
        self._branch_review_counts: Dict[str, int] = {}  # node_id -> review count

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for peer model."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.peer_timeout_sec)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def reset(self):
        """Reset metrics and state for a new reasoning run."""
        self.metrics = PeerMetrics()
        self._branch_review_counts.clear()

    def is_eligible(
        self,
        node: "ReasoningNode",
        engine: "ContractiveReasoner",
        timing: PeerTiming,
    ) -> Tuple[bool, str]:
        """
        Check if a node is eligible for peer review.

        Returns (eligible: bool, reason: str)
        """
        if not self.config.enabled:
            return False, "peer_review_disabled"

        if timing != self.config.timing and timing != PeerTiming.ON_DEMAND:
            return False, f"wrong_timing_{timing.value}"

        # Check global budget
        if self.metrics.total_calls >= self.config.max_peer_calls_total:
            return False, "max_total_calls_reached"

        total_tokens = self.metrics.total_prompt_tokens + self.metrics.total_completion_tokens
        if total_tokens >= self.config.peer_token_budget:
            return False, "token_budget_exhausted"

        # Check if already reviewed
        if node.peer_reviewed:
            return False, "already_reviewed"

        # Check branch depth
        depth = engine._node_depth(node.node_id)
        if depth < self.config.min_branch_depth:
            return False, f"insufficient_depth_{depth}"

        # Check branch node count
        path = engine.get_path_to_root(node.node_id)
        if len(path) < self.config.min_branch_nodes:
            return False, f"insufficient_nodes_{len(path)}"

        # Check per-branch call limit
        # Use root of the branch path as key
        branch_key = path[0].node_id if path else node.node_id
        current_calls = self._branch_review_counts.get(branch_key, 0)
        if current_calls >= self.config.max_peer_calls_per_branch:
            return False, "max_branch_calls_reached"

        # Node must have an answer to review
        if not node.answer:
            return False, "no_answer_to_review"

        return True, "eligible"

    async def review(
        self,
        node: "ReasoningNode",
        engine: "ContractiveReasoner",
        root_question: Optional[str] = None,
    ) -> PeerReviewResult:
        """
        Perform peer review on a node's answer.

        Parameters
        ----------
        node : ReasoningNode
            The node to review.
        engine : ContractiveReasoner
            The reasoning engine (for context).
        root_question : str, optional
            Override the root question (for testing).

        Returns
        -------
        PeerReviewResult
            The peer's verdict and feedback.
        """
        t0 = time.time()

        # Get root question
        if root_question is None:
            if engine.root_id and engine.root_id in engine.nodes:
                root_question = engine.nodes[engine.root_id].question
            else:
                root_question = node.question

        # Compute pre-review scores for context
        pre_score = pre_review_branch(node, root_question)

        # Format issues for prompt
        issues_text = ""
        if pre_score.issues:
            issues_text = "- Pre-review concerns: " + "; ".join(pre_score.issues)

        # Build prompt with pre-review context
        prompt = self.PEER_REVIEW_PROMPT.format(
            question=root_question,
            answer=node.answer or "(no answer)",
            alignment=pre_score.semantic_alignment,
            completeness=pre_score.answer_completeness,
            confidence=pre_score.assertion_strength,
            issues_text=issues_text,
        )

        # Call peer model
        client = await self._get_client()
        url = f"{self.config.peer_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.peer_model,
            "messages": [
                {"role": "system", "content": "You are a constructive academic peer reviewer. Your goal is to help improve the work, not to gatekeep."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }

        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            # Peer call failed - return uncertain
            result = PeerReviewResult(
                node_id=node.node_id,
                verdict=PeerVerdict.UNCERTAIN,
                score=0.5,
                feedback=f"Peer review failed: {e}",
                reasoning="",
                elapsed_sec=time.time() - t0,
                peer_model=self.config.peer_model,
                timestamp=t0,
            )
            self._record_result(result, node)
            return result

        # Parse response
        msg = data.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)

        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0

        # Extract JSON from response
        verdict, score, reasoning, feedback = self._parse_peer_response(content)

        result = PeerReviewResult(
            node_id=node.node_id,
            verdict=verdict,
            score=score,
            feedback=feedback,
            reasoning=reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_sec=time.time() - t0,
            peer_model=self.config.peer_model,
            timestamp=t0,
            raw_response=content if self.config.log_peer_prompts else "",
        )

        self._record_result(result, node)
        return result

    def _parse_peer_response(self, content: str) -> Tuple[PeerVerdict, float, str, str]:
        """
        Parse peer response and extract verdict, score, reasoning.

        Handles both old format (rejected/approved) and new constructive format
        (minor_revision/major_revision/reject).
        """
        try:
            # Find JSON in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)

                verdict_str = data.get("verdict", "uncertain").lower().replace("_", "")

                # Map both old and new verdict formats
                verdict_map = {
                    # New constructive format
                    "approved": PeerVerdict.APPROVED,
                    "minorrevision": PeerVerdict.NEEDS_REVISION,
                    "majorrevision": PeerVerdict.NEEDS_REVISION,
                    "reject": PeerVerdict.REJECTED,
                    # Old format (backwards compatible)
                    "rejected": PeerVerdict.REJECTED,
                    "needsrevision": PeerVerdict.NEEDS_REVISION,
                    "needs_revision": PeerVerdict.NEEDS_REVISION,
                    "uncertain": PeerVerdict.UNCERTAIN,
                    # Final review format
                    "accept": PeerVerdict.APPROVED,
                    "acceptwithnotes": PeerVerdict.APPROVED,
                    "needswork": PeerVerdict.NEEDS_REVISION,
                    "insufficient": PeerVerdict.REJECTED,
                }
                verdict = verdict_map.get(verdict_str, PeerVerdict.UNCERTAIN)

                # Get score (handle both 'score' and 'overall_score')
                score = float(data.get("score", data.get("overall_score", 0.5)))
                score = max(0.0, min(1.0, score))

                reasoning = data.get("reasoning", data.get("quality_notes", ""))

                # Build feedback from various fields
                feedback_parts = []

                # Strengths (new format)
                strengths = data.get("strengths", [])
                if strengths:
                    feedback_parts.append("Strengths: " + "; ".join(strengths[:3]))

                # Weaknesses (new format) or issues (old format)
                weaknesses = data.get("weaknesses", data.get("issues", []))
                if weaknesses:
                    feedback_parts.append("Weaknesses: " + "; ".join(weaknesses[:3]))

                # Suggestions
                suggestions = data.get("suggestions", data.get("improvement_priority", []))
                if suggestions:
                    feedback_parts.append("Suggestions: " + "; ".join(suggestions[:3]))

                feedback = " | ".join(feedback_parts) if feedback_parts else ""

                return verdict, score, reasoning, feedback

        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: try to infer from text (more nuanced)
        content_lower = content.lower()

        # Check for constructive language first
        if any(word in content_lower for word in ["minor revision", "small improvement", "mostly correct"]):
            return PeerVerdict.NEEDS_REVISION, 0.6, content[:200], ""
        elif any(word in content_lower for word in ["major revision", "significant issues", "needs work"]):
            return PeerVerdict.NEEDS_REVISION, 0.4, content[:200], ""
        elif any(word in content_lower for word in ["reject", "incorrect", "wrong", "fundamentally"]):
            return PeerVerdict.REJECTED, 0.3, content[:200], ""
        elif any(word in content_lower for word in ["approv", "correct", "valid", "accept"]):
            return PeerVerdict.APPROVED, 0.7, content[:200], ""
        else:
            return PeerVerdict.UNCERTAIN, 0.5, content[:200], ""

    def _record_result(self, result: PeerReviewResult, node: "ReasoningNode"):
        """Record result in metrics and update node."""
        self.metrics.total_calls += 1
        self.metrics.total_prompt_tokens += result.prompt_tokens
        self.metrics.total_completion_tokens += result.completion_tokens
        self.metrics.total_elapsed_sec += result.elapsed_sec
        self.metrics.reviews.append(result)

        # Count by verdict
        if result.verdict == PeerVerdict.APPROVED:
            self.metrics.approved_count += 1
        elif result.verdict == PeerVerdict.REJECTED:
            self.metrics.rejected_count += 1
        elif result.verdict == PeerVerdict.NEEDS_REVISION:
            self.metrics.revision_count += 1
        elif result.verdict == PeerVerdict.UNCERTAIN:
            self.metrics.uncertain_count += 1
        else:
            self.metrics.skipped_count += 1

        # Update node
        node.peer_reviewed = True
        node.peer_verdict = result.verdict.value
        node.peer_score = result.score
        node.peer_feedback = result.feedback

        # Track branch review count
        # (simplified: use node_id as key, in practice would use branch root)
        self._branch_review_counts[node.node_id] = (
            self._branch_review_counts.get(node.node_id, 0) + 1
        )

    def apply_verdict(
        self,
        result: PeerReviewResult,
        node: "ReasoningNode",
    ) -> str:
        """
        Apply peer verdict to node state.

        Returns action taken: "killed", "penalized", "approved", "no_action"
        """
        if result.verdict == PeerVerdict.REJECTED:
            if self.config.kill_rejected_branches:
                node.state = "failed"
                node.answer = f"[PEER REJECTED: {result.feedback or result.reasoning}]"
                return "killed"
            else:
                # Just penalize score, don't kill
                return "penalized"

        elif result.verdict == PeerVerdict.APPROVED:
            return "approved"

        elif result.verdict == PeerVerdict.NEEDS_REVISION:
            # Could trigger re-decomposition here
            return "needs_revision"

        return "no_action"


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

async def peer_review_before_collapse(
    node: "ReasoningNode",
    children: List["ReasoningNode"],
    engine: "ContractiveReasoner",
    reviewer: PeerReviewer,
) -> List["ReasoningNode"]:
    """
    Review children before collapse, optionally filtering out rejected ones.

    This is called from _try_collapses in the main engine.

    Returns filtered list of children to include in collapse.
    """
    valid_children = []

    for child in children:
        eligible, reason = reviewer.is_eligible(
            child, engine, PeerTiming.BEFORE_COLLAPSE
        )

        if eligible:
            result = await reviewer.review(child, engine)
            action = reviewer.apply_verdict(result, child)

            if action == "killed":
                continue  # Exclude from collapse

        valid_children.append(child)

    return valid_children


async def peer_review_final(
    engine: "ContractiveReasoner",
    reviewer: PeerReviewer,
) -> PeerReviewResult:
    """
    Final peer review of the root answer.

    Called after reasoning completes, reviews the final synthesized answer.
    """
    if not engine.root_id or engine.root_id not in engine.nodes:
        return PeerReviewResult(
            node_id="",
            verdict=PeerVerdict.SKIP,
            score=0.0,
            feedback="No root node",
            reasoning="",
        )

    root = engine.nodes[engine.root_id]

    # Force review even if not normally eligible
    result = await reviewer.review(root, engine, root_question=root.question)
    reviewer.apply_verdict(result, root)

    return result


# =============================================================================
# EXPERIMENTAL CONFIGURATIONS
# =============================================================================

def create_experiment_configs() -> Dict[str, PeerConfig]:
    """
    Create standard experiment configurations for academic comparison.

    Returns dict of named configs for different experimental conditions.
    """
    return {
        # Baseline: no peer review
        "no_peer": PeerConfig(enabled=False),

        # Standard peer review before collapse
        "peer_before_collapse": PeerConfig(
            enabled=True,
            timing=PeerTiming.BEFORE_COLLAPSE,
            min_branch_depth=2,
            min_branch_nodes=3,
            max_peer_calls_per_branch=1,
            kill_rejected_branches=True,
        ),

        # Final review only
        "peer_final_only": PeerConfig(
            enabled=True,
            timing=PeerTiming.END_OF_REASONING,
            force_peer_on_root=True,
            max_peer_calls_total=1,
        ),

        # Aggressive peer review
        "peer_aggressive": PeerConfig(
            enabled=True,
            timing=PeerTiming.BEFORE_COLLAPSE,
            min_branch_depth=1,
            min_branch_nodes=2,
            max_peer_calls_per_branch=2,
            max_peer_calls_total=20,
            approval_threshold=0.7,
            rejection_threshold=0.4,
        ),

        # Conservative (only high uncertainty)
        "peer_conservative": PeerConfig(
            enabled=True,
            timing=PeerTiming.BEFORE_COLLAPSE,
            min_branch_depth=3,
            min_branch_nodes=5,
            max_peer_calls_per_branch=1,
            max_peer_calls_total=5,
            peer_as_tiebreaker=True,
        ),

        # Cost-constrained
        "peer_budget_limited": PeerConfig(
            enabled=True,
            timing=PeerTiming.BEFORE_COLLAPSE,
            peer_token_budget=10000,
            max_peer_calls_total=3,
        ),

        # Large peer model
        "peer_large_model": PeerConfig(
            enabled=True,
            peer_model="deepseek-r1:14b",
            timing=PeerTiming.END_OF_REASONING,
            force_peer_on_root=True,
        ),
    }


def export_peer_metrics_for_paper(
    metrics: PeerMetrics,
    engine_summary: Dict[str, Any],
    config: PeerConfig,
) -> Dict[str, Any]:
    """
    Export combined metrics in format suitable for academic analysis.

    Returns a dict ready for CSV/JSON export and statistical analysis.
    """
    peer_dict = metrics.to_dict()

    return {
        # Experimental condition
        "peer_enabled": config.enabled,
        "peer_model": config.peer_model if config.enabled else None,
        "peer_timing": config.timing.value if config.enabled else None,

        # Main model metrics
        "main_total_nodes": engine_summary.get("total_nodes", 0),
        "main_solved_nodes": engine_summary.get("states", {}).get("solved", 0),
        "main_failed_nodes": engine_summary.get("states", {}).get("failed", 0),
        "main_pruned_nodes": engine_summary.get("states", {}).get("pruned", 0),
        "main_prompt_tokens": engine_summary.get("tokens", {}).get("prompt", 0),
        "main_completion_tokens": engine_summary.get("tokens", {}).get("completion", 0),

        # Peer metrics
        "peer_calls": peer_dict["total_calls"],
        "peer_approved": peer_dict["verdicts"]["approved"],
        "peer_rejected": peer_dict["verdicts"]["rejected"],
        "peer_needs_revision": peer_dict["verdicts"]["needs_revision"],
        "peer_prompt_tokens": peer_dict["tokens"]["prompt"],
        "peer_completion_tokens": peer_dict["tokens"]["completion"],
        "peer_time_sec": peer_dict["time_sec"],
        "peer_agreement_rate": peer_dict["agreement_rate"],

        # Combined
        "total_prompt_tokens": (
            engine_summary.get("tokens", {}).get("prompt", 0)
            + peer_dict["tokens"]["prompt"]
        ),
        "total_completion_tokens": (
            engine_summary.get("tokens", {}).get("completion", 0)
            + peer_dict["tokens"]["completion"]
        ),
    }
