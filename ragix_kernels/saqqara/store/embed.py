"""
saqqara.store.embed — vectors for the chunks that do not have one yet.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

Gate K7.6, K7.7.

Embedding is incremental by construction, not by optimisation. `chunk_id` is a
digest of the chunk's content (K7.5) and embeddings are keyed `(chunk_id, model)`,
so "what is missing" is a set difference the store can answer, and re-indexing an
unchanged corpus costs one query and no model call.

**`provider: none` writes nothing.** Not a zero vector, not a random one, not a
placeholder. A lexical-only store is a legitimate configuration, and a zero vector
is worse than an absent one: it is a point in the space, it has a cosine with
every query, and it makes an unembedded chunk look embedded to every layer above.
The status line says `dense: disabled (no embedder)` so nobody has to infer it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from .records import ChunkRecord, EmbeddingRecord, EmbeddingRefusalRecord

__all__ = ["PROVIDERS", "EmbedPlan", "build_embedder", "embed_missing"]

#: The providers a configuration may name. `none` is a provider, not the absence
#: of one: choosing to run without embeddings is a decision, and it is recorded.
PROVIDERS = ("none", "sentence-transformers", "ollama", "dummy")


class EmbedPlan:
    """What embedding did, in the terms the report needs.

    `skipped` and `refusals` are separate fields and must stay separate. `skipped`
    counts chunks this model had already embedded — a saving. A refusal is a chunk
    with no vector at all. Both reduce the number of records written, so a report
    that carries only `embedded` and `skipped` absorbs refusals into the count of
    work not needed, and reads as a complete lane. That is the failure this
    separation exists to prevent, and it is what the gate falsifies.
    """

    def __init__(self, model: str, embedded: int = 0, skipped: int = 0,
                 disabled: bool = False,
                 refusals: Optional[list[dict[str, Any]]] = None) -> None:
        self.model = model
        self.embedded = embedded
        self.skipped = skipped
        self.disabled = disabled
        #: One dict per refused chunk, the shape `ChunkPlan.refusals` already uses:
        #: a reason, a locator, and the signals the decision was made on.
        self.refusals: list[dict[str, Any]] = list(refusals or [])

    @property
    def refused(self) -> int:
        return len(self.refusals)

    def to_dict(self) -> dict[str, Any]:
        return {"disabled": self.disabled, "embedded": self.embedded,
                "model": self.model, "refusals": list(self.refusals),
                "refused": self.refused, "skipped": self.skipped}


def build_embedder(provider: str, model: str = "", **options: Any):
    """The backend a configuration asks for, or None for `provider: none`.

    Returning None rather than a null object is deliberate: a null embedder would
    be asked for vectors and would have to answer something, and every answer it
    could give is a lie. The caller checks for None once, at the top.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"embedder provider must be one of {PROVIDERS}, not {provider!r}")
    if provider == "none":
        return None

    from ragix_core.embeddings import EmbeddingConfig, create_embedding_backend

    if provider == "dummy":
        return create_embedding_backend("dummy")

    config = EmbeddingConfig(model_name=model) if model else EmbeddingConfig()
    for key, value in options.items():
        setattr(config, key, value)
    return create_embedding_backend(provider, config)


def embed_missing(
    store: Any,
    chunks: Iterable[ChunkRecord],
    embedder: Optional[Any],
    model: str,
    *,
    now: Optional[str] = None,
) -> EmbedPlan:
    """Embed only the chunks this model has not seen.

    `store.existing_embeddings` answers what is present; the difference is what is
    computed. Nothing is written when there is no embedder — see the module note.
    """
    chunks = list(chunks)
    if embedder is None:
        return EmbedPlan(model="", skipped=len(chunks), disabled=True)

    doc_ids = {c.doc_id for c in chunks}
    if len(doc_ids) > 1:
        raise ValueError(
            f"embed_missing works on one document at a time, got {len(doc_ids)}: "
            "the refusal register is replaced per document, and a mixed call would "
            "clear one document's refusals while writing another's"
        )
    doc_id = doc_ids.pop() if doc_ids else ""

    # A refusal is never consulted here. `existing_embeddings` answers what is
    # present, and a chunk refused by a previous run is missing, so it is asked
    # again: another model may accept it, and a permanent mark would turn one
    # server's answer into a property of the text.
    have = store.existing_embeddings([c.chunk_id for c in chunks], model)
    missing = [c for c in chunks if c.chunk_id not in have]
    if not missing:
        store.replace_embedding_refusals(doc_id, [])
        return EmbedPlan(model=model, embedded=0, skipped=len(chunks))

    stamp = now or datetime.now(timezone.utc).isoformat()

    # A backend that can attribute a refusal to one text is asked to; one that
    # cannot keeps the strict path, where any refusal raises. The capability is
    # asked for by name rather than by provider, so a second backend that grows it
    # is used without editing this line.
    recording = getattr(embedder, "embed_batch_recording_refusals", None)
    if recording is None:
        vectors = embedder.embed_batch([c.text for c in missing])
        refusals = []
    else:
        vectors, refusals = recording([c.text for c in missing])

    if len(vectors) != len(missing):
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for {len(missing)} chunks; "
            "a partial answer cannot be matched to its inputs"
        )

    refused_at = {r.index: r for r in refusals}
    records, refused_records, refused_report = [], [], []
    for position, (chunk, vector) in enumerate(zip(missing, vectors)):
        refusal = refused_at.get(position)
        if refusal is not None:
            if vector:
                raise RuntimeError(
                    f"chunk {chunk.chunk_id[:12]} is both refused and embedded; "
                    "a vector for a refused text cannot be trusted to be its own"
                )
            signals = {**refusal.to_dict(), "level": chunk.level, "seq": chunk.seq}
            signals.pop("reason", None)
            signals.pop("index", None)
            refused_records.append(EmbeddingRefusalRecord(
                chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, model=model,
                reason=refusal.reason, signals=signals, refused_at=stamp,
            ))
            refused_report.append({"reason": refusal.reason, "chunk_id": chunk.chunk_id,
                                   "doc_id": chunk.doc_id, "node_ids": list(chunk.node_ids),
                                   "signals": signals})
            continue
        if not vector:
            raise RuntimeError(
                f"empty vector for chunk {chunk.chunk_id[:12]}: an absent embedding "
                "is not a zero vector, and storing one would make it look present"
            )
        records.append(EmbeddingRecord(chunk_id=chunk.chunk_id, model=model,
                                       dimensions=len(vector), vector=tuple(vector),
                                       indexed_at=stamp))
    store.upsert_embeddings(records)
    # Written for every document, empty included: see `replace_embedding_refusals`.
    store.replace_embedding_refusals(doc_id, refused_records)
    return EmbedPlan(model=model, embedded=len(records),
                     skipped=len(chunks) - len(missing), refusals=refused_report)
