"""
saqqara_index — stage 2: what stage 1 read becomes a store that can be queried.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

Gate K7.15, K7.16.

`requires=["document_tree"]` and `provides=["document_store"]`, so the registry
orders it after the reader rather than by anyone remembering to. It reads the
trees stage 1 produced — from the workspace, or handed over in memory — chunks
them, stores them, and embeds only what is missing.

It computes and stores. It does not read source files again: everything comes from
the trees, which are the only things carrying provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ragix_kernels.base import Kernel, KernelInput

from ..store.config import load_config
from ..store.embed import build_embedder, embed_missing
from ..store.refine import refine_store
from ..store.feed import feed_result, feed_tree
from ..store.ports import build_store

__all__ = ["SaqqaraIndexKernel"]


class SaqqaraIndexKernel(Kernel):
    """Turn read documents into a queryable store."""

    name = "saqqara_index"
    version = "0.1.0"
    category = "saqqara"
    stage = 2
    description = "Chunk read documents into a hybrid store and embed what is missing"
    requires: List[str] = ["document_tree"]
    provides = ["document_store"]

    def compute(self, input: KernelInput) -> Dict[str, Any]:
        config = load_config(input.config.get("config") or None,
                             **(input.config.get("overrides") or {}))

        store_section = config.section("store")
        path = Path(store_section.get("path", ".ragix/saqqara.db"))
        if not path.is_absolute():
            path = Path(input.workspace) / path
        store = build_store({**store_section, "path": str(path)})

        source = input.config.get("result")
        if source is None:
            # The envelope resolves `requires` into `dependencies`, so the declared
            # dependency is where the tree comes from. The workspace path is a
            # fallback for a direct caller, not the contract.
            stored = input.dependencies.get("document_tree")
            if stored is None:
                stored = Path(input.workspace) / "stage1" / "saqqara.json"
            if not Path(stored).is_file():
                raise FileNotFoundError(
                    f"no stage-1 result at {stored}: saqqara_index reads what the "
                    "reader produced and does not read source files itself"
                )
            source = Path(stored)

        chunker = config.section("chunker")
        fed = (feed_result(source, **chunker) if not isinstance(source, list)
               else [feed_tree(t, source_path=p, source_sha256=s, **chunker)
                     for t, p, s in source])

        embedder_section = config.section("embedder")
        provider = embedder_section.get("provider", "none")
        model_name = embedder_section.get("model", "") or provider
        embedder = build_embedder(
            provider, model=embedder_section.get("model", ""),
            **({"base_url": embedder_section["base_url"]}
               if embedder_section.get("base_url") else {}),
            **({"batch_size": embedder_section["batch_size"]}
               if embedder_section.get("batch_size") else {}),
        )

        documents, embedded, skipped = [], 0, 0
        refusals: list[Dict[str, Any]] = []
        for one in fed:
            store.upsert_document(one.document, objects=one.objects, edges=one.edges)
            written = store.replace_chunks(one.document.doc_id, one.chunks)
            plan = embed_missing(store, one.chunks, embedder,
                                 model=model_name if embedder else "")
            embedded += plan.embedded
            skipped += plan.skipped
            refusals.extend({**r, "path": one.document.source_path} for r in plan.refusals)
            documents.append({
                "doc_id": one.document.doc_id,
                "path": one.document.source_path,
                "format": one.document.doc_class,
                "chunks": written,
                "refused": plan.refused,
                **one.counts(),
            })

        # The refinement passes, if the run asked for them. After embedding, and on
        # the store as it stands: a pass reads the previous result by construction.
        refine_section = config.section("refine")
        budgets = [int(b) for b in (refine_section.get("budgets") or [])]
        refinement = None
        if budgets and embedder is not None:
            refinement = refine_store(
                store, embedder, model_name, budgets,
                overlap_children=int(refine_section.get("overlap_children", 0)),
            ).to_dict()
            embedded += sum(p["embedded"] for p in refinement["passes"])

        status = store.status()
        status["dense"] = ("disabled (no embedder)" if embedder is None
                           else f"{model_name} ({status['embeddings']} vectors, "
                                f"{status['embedding_refusals']} refused)")

        # The boundary between a lane with gaps and no lane at all. Refusals above
        # zero are a result and the stage completes with them listed; a lane that
        # ended with no vector is not a strict lane, it is an absent one, and the
        # condition is almost always the configuration rather than the corpus — a
        # model that is not there, or one that rejects everything it is sent. Zero
        # is the only line here nobody has to choose.
        if refusals and status["embeddings"] == 0:
            raise RuntimeError(
                f"the dense lane holds no vector and refused {len(refusals)} chunk(s) "
                f"under model {model_name!r}: a lane with nothing in it is not a lane "
                "with gaps. Check the model and the server before reading this as a "
                "fact about the corpus."
            )
        return {
            "documents": documents,
            "status": status,
            "embedded": embedded,
            "skipped": skipped,
            "refused": len(refusals),
            "embedder_refusals": refusals,
            "refinement": refinement,
            "config": config.to_dict(),
        }

    def summarize(self, data: Dict[str, Any]) -> str:
        """Three numbers, always. `refused` is not omitted when it is zero: a line
        that appears only on bad days teaches a reader that its absence means the
        question was not asked."""
        status = data.get("status", {})
        return (
            f"{len(data.get('documents', []))} document(s), {status.get('chunks', 0)} chunk(s). "
            f"Embedded {data.get('embedded', 0)}, already present {data.get('skipped', 0)}, "
            f"refused {data.get('refused', 0)}. "
            f"Dense: {status.get('dense', 'unknown')}."
        )
