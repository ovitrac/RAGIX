"""
Bounded recall-answer loop controller — RAGIX memory subsystem.

Implements an iterative loop where the LLM proposes and the controller
enforces bounds, fixed-point convergence, and query-cycle detection.

The loop reads an initial injection block from stdin (typically produced by
``ragix-memory pipe``), calls the LLM, parses its JSON protocol line, and
optionally triggers additional recall iterations until convergence or
max-calls is reached.

Protocol:
  The LLM's first output line MUST be a JSON object (see §6 of ROADMAP_LOOP):
    {"need_more": bool, "query": str|null, "rationale": str|null, "stop": bool}
  Remaining output is the answer text.

Trace:
  One JSONL line per iteration to stderr (or ``--trace-file``):
    {"iter": 1, "query": "...", "new_items": 5, "sim": null, "sim_method": "...", "action": "continue"}

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-19
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import IO, Any, Dict, List, Optional, Tuple

from ragix_core.memory.similarity import compute_similarity, detect_query_cycle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol system prompt (shared with memctl by convention)
# ---------------------------------------------------------------------------

LOOP_PROTOCOL_PROMPT = """\
You are answering a question using retrieved context. Follow this protocol exactly:

1. Your FIRST line of output MUST be a JSON object with these fields:
   {"need_more": <bool>, "query": "<string or null>", "rationale": "<string or null>", "stop": <bool>}

2. After the JSON line, leave ONE blank line, then write your answer.

3. If the provided context is SUFFICIENT to answer fully:
   {"need_more": false, "query": null, "rationale": null, "stop": true}

4. If the provided context is INSUFFICIENT and you need more information:
   {"need_more": true, "query": "specific refined search query", "rationale": "what is missing", "stop": false}

5. Do NOT emit anything before the JSON line. Do NOT wrap it in markdown."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LoopConfig:
    """Configuration for the recall-answer loop."""

    llm_command: str = ""
    llm_mode: str = "stdin"                # "stdin" or "file"
    protocol: str = "json"                 # "json", "regex", "passive"
    system_prompt: str = ""                # user system prompt (appended after protocol)
    max_calls: int = 3
    threshold: float = 0.92               # answer fixed-point threshold
    query_threshold: float = 0.90         # query cycle threshold
    stable_steps: int = 2                 # consecutive stable steps to converge
    stop_on_no_new: bool = True
    budget: int = 2200                    # token budget for context
    similarity_mode: str = "auto"         # "auto", "lexical", "embedding"
    strict: bool = False                  # exit 1 on max-calls without convergence
    trace: bool = False
    trace_file: Optional[str] = None      # redirect trace to file (else stderr)


# ---------------------------------------------------------------------------
# Protocol parsing
# ---------------------------------------------------------------------------


@dataclass
class ProtocolResponse:
    """Parsed LLM output: protocol header + answer body."""

    need_more: bool = False
    query: Optional[str] = None
    rationale: Optional[str] = None
    stop: bool = True
    answer: str = ""
    raw: str = ""
    parse_ok: bool = True


def parse_protocol_json(raw_output: str) -> ProtocolResponse:
    """Parse JSON-first-line protocol from LLM output.

    If the first line is not valid JSON, returns ``need_more=False``
    (conservative: treat unparseable output as final answer).
    """
    resp = ProtocolResponse(raw=raw_output)

    lines = raw_output.split("\n", 1)
    first_line = lines[0].strip()

    try:
        header = json.loads(first_line)
    except (json.JSONDecodeError, ValueError):
        # Fallback: entire output is the answer
        resp.parse_ok = False
        resp.need_more = False
        resp.stop = True
        resp.answer = raw_output.strip()
        return resp

    resp.need_more = bool(header.get("need_more", False))
    resp.query = header.get("query")
    resp.rationale = header.get("rationale")
    resp.stop = bool(header.get("stop", not resp.need_more))

    # Answer is everything after the first line (skip optional blank line)
    if len(lines) > 1:
        body = lines[1]
        if body.startswith("\n"):
            body = body[1:]
        resp.answer = body.strip()
    else:
        resp.answer = ""

    # Empty query with need_more → treat as stop
    if resp.need_more and (not resp.query or not resp.query.strip()):
        resp.need_more = False
        resp.stop = True

    return resp


def parse_protocol_regex(raw_output: str) -> ProtocolResponse:
    """Fallback regex parser: scan for NEED_MORE: / QUERY: patterns."""
    import re

    resp = ProtocolResponse(raw=raw_output)

    need_match = re.search(r"NEED_MORE:\s*(true|false)", raw_output, re.IGNORECASE)
    query_match = re.search(r"QUERY:\s*(.+)", raw_output, re.IGNORECASE)

    if need_match and need_match.group(1).lower() == "true":
        resp.need_more = True
        resp.stop = False
        if query_match:
            resp.query = query_match.group(1).strip()
    else:
        resp.need_more = False
        resp.stop = True

    resp.answer = raw_output.strip()
    return resp


def parse_protocol(raw_output: str, protocol: str = "json") -> ProtocolResponse:
    """Dispatch to the appropriate protocol parser."""
    if protocol == "json":
        return parse_protocol_json(raw_output)
    elif protocol == "regex":
        return parse_protocol_regex(raw_output)
    elif protocol == "passive":
        return ProtocolResponse(
            raw=raw_output,
            answer=raw_output.strip(),
            need_more=False,
            stop=True,
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------


def invoke_llm(
    command: str,
    input_text: str,
    system_prompt: str = "",
    mode: str = "stdin",
    include_protocol: bool = True,
) -> str:
    """Call the LLM subprocess and return its stdout.

    The full system prompt (protocol + user) is prepended to the input
    when mode is ``stdin`` (the LLM receives everything on stdin).

    When *include_protocol* is False (passive mode), the JSON-first-line
    protocol prompt is omitted — the LLM receives only the user system
    prompt and the context.

    Raises ``RuntimeError`` on non-zero exit or empty output.
    """
    if include_protocol:
        full_prompt = LOOP_PROTOCOL_PROMPT
        if system_prompt:
            full_prompt += "\n\n" + system_prompt
    else:
        full_prompt = system_prompt or ""

    if mode == "stdin":
        if full_prompt:
            stdin_text = full_prompt + "\n\n---\n\n" + input_text
        else:
            stdin_text = input_text
    else:
        stdin_text = input_text

    try:
        result = subprocess.run(
            command,
            shell=True,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"LLM command timed out (300s): {command}")

    if result.returncode != 0:
        stderr_preview = (result.stderr or "")[:200]
        raise RuntimeError(
            f"LLM command failed (exit {result.returncode}): {stderr_preview}"
        )

    output = result.stdout.strip()
    if not output:
        raise RuntimeError("LLM returned empty output")

    return output


# ---------------------------------------------------------------------------
# Trace emission
# ---------------------------------------------------------------------------


def emit_trace(
    trace_data: Dict[str, Any],
    trace_stream: IO[str],
) -> None:
    """Write a single JSONL trace line."""
    trace_stream.write(json.dumps(trace_data, ensure_ascii=False) + "\n")
    trace_stream.flush()


# ---------------------------------------------------------------------------
# Recall helper
# ---------------------------------------------------------------------------


def do_recall(
    dispatcher: Any,
    query: str,
    seen_ids: set,
    budget: int,
    tier: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Perform recall and return (all_items, new_items, new_count).

    Uses the existing dispatcher.dispatch("search", ...) + "read" pattern
    from cmd_pipe. Items already in *seen_ids* are filtered out.

    Returns:
        all_items: Full list after dedup and enrichment.
        new_items: Only the items not previously seen.
        new_count: len(new_items).
    """
    params: Dict[str, Any] = {"query": query, "k": 50}
    if tier:
        params["tier"] = tier

    result = dispatcher.dispatch("search", params)
    if result.get("status") != "ok":
        return [], [], 0

    items = result.get("items", [])
    if not items:
        return [], [], 0

    # Enrich with full content
    ids = [it["id"] for it in items]
    full = dispatcher.dispatch("read", {"ids": ids})
    if full.get("status") == "ok":
        items = full.get("items", items)

    # Filter non-injectable
    items = [it for it in items if it.get("injectable", True)]

    # Separate new vs seen
    new_items = [it for it in items if it["id"] not in seen_ids]
    new_count = len(new_items)

    # Update seen set
    seen_ids.update(it["id"] for it in new_items)

    return items, new_items, new_count


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


@dataclass
class LoopResult:
    """Outcome of the recall-answer loop."""

    answer: str = ""
    iterations: int = 0
    converged: bool = False
    exit_reason: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)


def run_loop(
    initial_context: str,
    config: LoopConfig,
    dispatcher: Any = None,
    embedder: Any = None,
    trace_stream: Optional[IO[str]] = None,
) -> LoopResult:
    """Execute the bounded recall-answer loop.

    Parameters
    ----------
    initial_context : str
        Initial injection block (from ``ragix-memory pipe`` or raw text).
    config : LoopConfig
        Loop configuration (thresholds, max calls, etc.).
    dispatcher : optional
        Memory dispatcher for recall. If None, no refinement recall is
        performed (LLM gets the same context every iteration).
    embedder : optional
        Embedder for Tier B similarity. If None, falls back to Tier A.
    trace_stream : optional
        Writable stream for JSONL trace. Defaults to stderr if
        ``config.trace`` is True.

    Returns
    -------
    LoopResult
        Final answer, iteration count, convergence status, and trace log.
    """
    from ragix_core.memory.mcp.formatting import format_injection_block

    result = LoopResult()

    # Trace stream setup
    if trace_stream is None and config.trace:
        if config.trace_file:
            trace_stream = open(config.trace_file, "w")
        else:
            trace_stream = sys.stderr
    close_trace = (
        trace_stream is not None
        and config.trace_file is not None
        and trace_stream is not sys.stderr
    )

    # State
    context = initial_context
    previous_answers: List[str] = []
    previous_queries: List[str] = []
    seen_ids: set = set()
    stable_count = 0

    try:
        for iteration in range(1, config.max_calls + 1):
            result.iterations = iteration

            # --- Call LLM ---
            try:
                raw_output = invoke_llm(
                    config.llm_command,
                    context,
                    system_prompt=config.system_prompt,
                    mode=config.llm_mode,
                    include_protocol=(config.protocol != "passive"),
                )
            except RuntimeError as e:
                result.exit_reason = f"llm_error: {e}"
                break

            # --- Parse protocol ---
            resp = parse_protocol(raw_output, config.protocol)
            result.answer = resp.answer

            # --- Fixed-point test ---
            sim_score: Optional[float] = None
            sim_method: Optional[str] = None
            if previous_answers:
                sim_score, sim_method = compute_similarity(
                    resp.answer,
                    previous_answers[-1],
                    mode=config.similarity_mode,
                    embedder=embedder,
                )
                if sim_score >= config.threshold:
                    stable_count += 1
                else:
                    stable_count = 0

            previous_answers.append(resp.answer)

            # --- Determine action ---
            action = "continue"

            # Check convergence: stable for N consecutive steps
            if stable_count >= config.stable_steps:
                action = "fixed_point"
                result.converged = True
                result.exit_reason = (
                    f"answer stable for {config.stable_steps} steps "
                    f"(sim={sim_score:.3f}, method={sim_method})"
                )

            # LLM says stop (ignored in passive mode — fixed-point only)
            elif config.protocol != "passive" and (resp.stop or not resp.need_more):
                action = "llm_stop"
                result.exit_reason = "llm signaled stop"

            # Query-cycle detection
            elif resp.query:
                is_cycle, cycle_reason = detect_query_cycle(
                    resp.query,
                    previous_queries,
                    threshold=config.query_threshold,
                    mode=config.similarity_mode,
                    embedder=embedder,
                )
                if is_cycle:
                    action = "query_cycle"
                    result.exit_reason = f"query cycle: {cycle_reason}"

            # Last iteration
            if iteration == config.max_calls and action == "continue":
                action = "max_calls"
                result.exit_reason = f"max_calls ({config.max_calls}) reached"

            # --- Recall for next iteration (if continuing) ---
            new_count = 0
            if action == "continue" and resp.query and dispatcher is not None:
                previous_queries.append(resp.query)

                all_items, new_items, new_count = do_recall(
                    dispatcher, resp.query, seen_ids, config.budget
                )

                if new_count == 0 and config.stop_on_no_new:
                    action = "no_new_items"
                    result.exit_reason = "no new items from recall"
                elif all_items:
                    # Rebuild context with merged items
                    context = format_injection_block(
                        all_items, config.budget, len(all_items)
                    )

            # --- Emit trace ---
            trace_entry = {
                "iter": iteration,
                "query": resp.query,
                "new_items": new_count,
                "sim": round(sim_score, 4) if sim_score is not None else None,
                "sim_method": sim_method,
                "action": action,
            }
            if resp.rationale:
                trace_entry["rationale"] = resp.rationale
            result.trace.append(trace_entry)

            if trace_stream is not None:
                emit_trace(trace_entry, trace_stream)

            # --- Stop? ---
            if action != "continue":
                break

    finally:
        if close_trace and trace_stream is not None:
            trace_stream.close()

    return result
