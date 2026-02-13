"""
Kernel: pres_semantic_normalize
Stage: 2 (Structuring)

Dual-mode normalizer that enriches a ContentCorpus with:
- Topic clustering (heading-path based)
- Semantic role assignment (keyword heuristics or LLM)
- Importance scoring
- Near-duplicate detection
- Narrative arc ordering

Modes (config.normalizer.mode):
    "deterministic" — pure heuristics, no LLM
    "llm"           — heuristics + LLM refinement via Ollama
    "auto"          — deterministic if no Ollama, else llm

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-02-11
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragix_kernels.base import Kernel, KernelInput
from ragix_kernels.presenter.models import (
    ContentCorpus,
    NarrativeArc,
    NormalizedCorpus,
    NormalizedUnit,
    NormalizationMode,
    SemanticUnit,
    TopicCluster,
    UnitRole,
    UnitType,
)
from ragix_kernels.presenter.config import (
    BudgetConfig,
    ImportanceConfig,
    NormalizerConfig,
)
from ragix_kernels.presenter.normalize_utils import (
    assign_role_by_keywords,
    cluster_by_heading_path,
    compute_importance,
    consolidate_clusters,
    detect_narrative_arc,
    find_duplicates,
    find_global_duplicates,
)

logger = logging.getLogger(__name__)

# Jinja2 (soft dependency — only needed for LLM mode)
try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_template() -> Optional[Any]:
    """Load the normalize_cluster.j2 Jinja2 template."""
    if not HAS_JINJA2:
        return None
    template_dir = Path(__file__).parent.parent / "prompts"
    template_path = template_dir / "normalize_cluster.j2"
    if not template_path.exists():
        logger.warning(f"[pres_semantic_normalize] Template not found: {template_path}")
        return None
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        undefined=jinja2.StrictUndefined,
    )
    return env.get_template("normalize_cluster.j2")


def _check_ollama(endpoint: str = "http://127.0.0.1:11434") -> bool:
    """Quick check if Ollama is reachable."""
    try:
        import httpx
        resp = httpx.get(f"{endpoint}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _llm_call_simple(
    model: str,
    prompt: str,
    endpoint: str = "http://127.0.0.1:11434",
    temperature: float = 0.1,
    timeout: int = 120,
    num_predict: int = 1024,
) -> str:
    """Direct Ollama call without caching (presenter-local)."""
    import httpx
    response = httpx.post(
        f"{endpoint}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract JSON from LLM response text."""
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown code fence
    import re
    m = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class PresSemanticNormalizeKernel(Kernel):
    """Dual-mode semantic normalizer (deterministic + optional LLM)."""

    name = "pres_semantic_normalize"
    version = "1.0.0"
    category = "presenter"
    stage = 2
    description = "Topic clustering, role assignment, dedup, narrative arc"

    requires: List[str] = ["pres_content_extract"]
    provides: List[str] = ["normalized_corpus", "narrative_arc"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        # Load S1 output
        extract_path = input.dependencies["pres_content_extract"]
        extract_data = json.loads(extract_path.read_text(encoding="utf-8"))
        corpus = ContentCorpus.from_dict(extract_data["data"])

        # Parse config
        norm_cfg = input.config.get("normalizer", {})
        enabled = norm_cfg.get("enabled", True)
        mode = norm_cfg.get("mode", "auto")
        model = norm_cfg.get("model", "mistral-small:24b")
        lang = input.config.get("lang", "fr")
        if lang == "auto":
            lang = "fr"

        imp_cfg = ImportanceConfig(
            boost_headings=norm_cfg.get("importance", {}).get("boost_headings", True),
            boost_findings=norm_cfg.get("importance", {}).get("boost_findings", True),
            decay_depth=norm_cfg.get("importance", {}).get("decay_depth", 0.1),
        )
        dedup_threshold = norm_cfg.get("deduplication", {}).get("threshold", 0.70)
        dedup_enabled = norm_cfg.get("deduplication", {}).get("enabled", True)

        max_clusters = norm_cfg.get("max_clusters", 20)

        budget = BudgetConfig(
            max_llm_calls_per_run=norm_cfg.get("budget", {}).get("max_llm_calls_per_run", 16),
            max_llm_total_input_tokens_per_run=norm_cfg.get("budget", {}).get("max_llm_total_input_tokens_per_run", 20000),
        )

        # Identity mode: pass-through
        if not enabled:
            result = NormalizedCorpus.identity(corpus)
            return _build_output(result, {"mode": "identity", "llm_calls": 0})

        # Resolve mode
        actual_mode = mode
        if mode == "auto":
            llm_cfg = input.config.get("llm", {})
            endpoint = llm_cfg.get("endpoint", "http://127.0.0.1:11434")
            if _check_ollama(endpoint):
                actual_mode = "llm"
            else:
                actual_mode = "deterministic"
                logger.info("[pres_semantic_normalize] Ollama not reachable, falling back to deterministic")

        # --- Step 1: Heading-path clustering + consolidation ---
        raw_clusters = cluster_by_heading_path(corpus.units)
        pre_consolidation = len(raw_clusters)
        raw_clusters = consolidate_clusters(raw_clusters, corpus.units, max_clusters=max_clusters)
        if len(raw_clusters) < pre_consolidation:
            logger.info(
                f"[pres_semantic_normalize] Consolidated {pre_consolidation} → "
                f"{len(raw_clusters)} clusters (max_clusters={max_clusters})"
            )

        # --- Step 2: Role assignment + importance scoring ---
        unit_map: Dict[str, SemanticUnit] = {u.id: u for u in corpus.units}
        normalized_units: List[NormalizedUnit] = []

        for u in corpus.units:
            role = assign_role_by_keywords(u, lang=lang)
            importance = compute_importance(u, role, imp_cfg)
            # Determine cluster for this unit
            topic = None
            for cluster_label, uids in raw_clusters.items():
                if u.id in uids:
                    topic = cluster_label
                    break
            normalized_units.append(NormalizedUnit(
                unit=u,
                topic_cluster=topic,
                importance=importance,
                role=role,
                lang=lang,
            ))

        # --- Step 3: LLM refinement (if llm mode) ---
        llm_calls = 0
        llm_tokens_used = 0
        if actual_mode == "llm":
            llm_calls, llm_tokens_used = self._llm_refine(
                raw_clusters, normalized_units, unit_map,
                model=model,
                lang=lang,
                budget=budget,
                config=input.config,
            )

        # --- Step 4: Deduplication ---
        duplicates_map: Dict[str, str] = {}
        if dedup_enabled:
            duplicates_map = find_duplicates(normalized_units, threshold=dedup_threshold)
            for nu in normalized_units:
                if nu.unit.id in duplicates_map:
                    nu.duplicate_of = duplicates_map[nu.unit.id]

        # --- Step 4b: Global (cross-cluster) deduplication ---
        compression = input.config.get("slide_plan", {}).get("compression", "full")
        global_dedup_count = 0
        if dedup_enabled and compression in ("compressed", "executive"):
            global_threshold = norm_cfg.get("deduplication", {}).get("global_threshold", 0.80)
            active_units = [nu for nu in normalized_units if nu.duplicate_of is None]
            global_dupes = find_global_duplicates(active_units, threshold=global_threshold)
            for nu in normalized_units:
                if nu.unit.id in global_dupes:
                    nu.duplicate_of = global_dupes[nu.unit.id]
            duplicates_map.update(global_dupes)
            global_dedup_count = len(global_dupes)
            if global_dedup_count > 0:
                logger.info(
                    f"[pres_semantic_normalize] Global dedup: {global_dedup_count} "
                    f"cross-cluster duplicates removed (threshold={global_threshold})"
                )

        # --- Step 5: Build TopicCluster objects ---
        clusters: List[TopicCluster] = []
        for idx, (label, uids) in enumerate(raw_clusters.items()):
            active_uids = [uid for uid in uids if uid not in duplicates_map]
            if not active_uids:
                continue
            # Aggregate importance
            avg_imp = sum(
                nu.importance for nu in normalized_units
                if nu.unit.id in active_uids
            ) / max(len(active_uids), 1)
            # Estimate slides
            suggested = max(1, len(active_uids) // 3)
            clusters.append(TopicCluster(
                id=f"cluster-{idx:03d}",
                label=label,
                unit_ids=active_uids,
                importance=avg_imp,
                suggested_slides=suggested,
            ))

        # --- Step 6: Narrative arc ---
        narrative = detect_narrative_arc(clusters, normalized_units)

        # --- Step 7: Build NormalizedCorpus ---
        norm_mode = NormalizationMode.LLM if actual_mode == "llm" else NormalizationMode.DETERMINISTIC
        result = NormalizedCorpus(
            raw=corpus,
            units=normalized_units,
            clusters=clusters,
            narrative=narrative,
            duplicates_removed=len(duplicates_map),
            merge_groups={},
            normalization_mode=norm_mode,
        )

        stats = {
            "mode": actual_mode,
            "units_total": len(normalized_units),
            "units_active": len(result.active_units()),
            "clusters": len(clusters),
            "duplicates_removed": len(duplicates_map),
            "llm_calls": llm_calls,
            "llm_tokens_used": llm_tokens_used,
            "narrative_sections": len(narrative.sections),
        }
        return _build_output(result, stats)

    def _llm_refine(
        self,
        raw_clusters: Dict[str, List[str]],
        normalized_units: List[NormalizedUnit],
        unit_map: Dict[str, SemanticUnit],
        model: str,
        lang: str,
        budget: BudgetConfig,
        config: Dict[str, Any],
    ) -> tuple:
        """
        LLM refinement pass: re-label clusters and override roles/importance.

        Returns (llm_calls, llm_tokens_used).
        """
        template = _load_template()
        if template is None:
            logger.warning("[pres_semantic_normalize] No Jinja2 template, skipping LLM refinement")
            return 0, 0

        llm_cfg = config.get("llm", {})
        endpoint = llm_cfg.get("endpoint", "http://127.0.0.1:11434")
        temperature = llm_cfg.get("temperature", 0.1)
        timeout = llm_cfg.get("timeout", 120)

        nu_map: Dict[str, NormalizedUnit] = {nu.unit.id: nu for nu in normalized_units}
        llm_calls = 0
        llm_tokens_used = 0

        for cluster_label, uids in raw_clusters.items():
            # Budget checks
            if llm_calls >= budget.max_llm_calls_per_run:
                logger.info("[pres_semantic_normalize] LLM call budget exhausted")
                break
            if llm_tokens_used >= budget.max_llm_total_input_tokens_per_run:
                logger.info("[pres_semantic_normalize] LLM token budget exhausted")
                break

            # Skip tiny clusters
            cluster_units = [unit_map[uid] for uid in uids if uid in unit_map]
            cluster_tokens = sum(u.tokens for u in cluster_units)
            if cluster_tokens < 50:
                continue

            # Render prompt
            prompt = template.render(lang=lang, units=cluster_units)
            prompt_tokens = len(prompt) // 4  # rough estimate
            llm_tokens_used += prompt_tokens

            try:
                response = _llm_call_simple(
                    model=model,
                    prompt=prompt,
                    endpoint=endpoint,
                    temperature=temperature,
                    timeout=timeout,
                )
                llm_calls += 1
            except Exception as e:
                logger.warning(f"[pres_semantic_normalize] LLM call failed: {e}")
                continue

            # Parse response
            parsed = _parse_llm_json(response)
            if parsed is None:
                logger.warning(f"[pres_semantic_normalize] Failed to parse LLM JSON for cluster '{cluster_label}'")
                continue

            # Apply refinements
            llm_label = parsed.get("label", cluster_label)
            llm_units = parsed.get("units", [])

            for llm_u in llm_units:
                uid = llm_u.get("id", "")
                nu = nu_map.get(uid)
                if nu is None:
                    continue
                # Override role
                llm_role = llm_u.get("role", "")
                try:
                    nu.role = UnitRole(llm_role)
                except ValueError:
                    pass  # keep heuristic role
                # Override importance
                llm_imp = llm_u.get("importance")
                if isinstance(llm_imp, (int, float)) and 0.0 <= llm_imp <= 1.0:
                    nu.importance = float(llm_imp)
                # Update cluster label
                nu.topic_cluster = llm_label

            # Update cluster label in raw_clusters (for downstream)
            # We don't mutate raw_clusters keys, but store in NormalizedUnits

        return llm_calls, llm_tokens_used

    def summarize(self, data: Dict[str, Any]) -> str:
        stats = data.get("statistics", {})
        mode = stats.get("mode", "?")
        n_units = stats.get("units_active", 0)
        n_clusters = stats.get("clusters", 0)
        n_dedup = stats.get("duplicates_removed", 0)
        llm = stats.get("llm_calls", 0)
        return (
            f"Normalization ({mode}): {n_units} active units, "
            f"{n_clusters} clusters, {n_dedup} duplicates removed"
            + (f", {llm} LLM calls" if llm > 0 else "")
        )


def _build_output(
    corpus: NormalizedCorpus,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine NormalizedCorpus data with statistics."""
    result = corpus.to_dict()
    result["statistics"] = stats
    return result
