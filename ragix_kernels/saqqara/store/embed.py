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

from .records import ChunkRecord, EmbeddingRecord

__all__ = ["PROVIDERS", "EmbedPlan", "build_embedder", "embed_missing"]

#: The providers a configuration may name. `none` is a provider, not the absence
#: of one: choosing to run without embeddings is a decision, and it is recorded.
PROVIDERS = ("none", "sentence-transformers", "ollama", "dummy")


class EmbedPlan:
    """What embedding did, in the terms the report needs."""

    def __init__(self, model: str, embedded: int = 0, skipped: int = 0,
                 disabled: bool = False) -> None:
        self.model = model
        self.embedded = embedded
        self.skipped = skipped
        self.disabled = disabled

    def to_dict(self) -> dict[str, Any]:
        return {"disabled": self.disabled, "embedded": self.embedded,
                "model": self.model, "skipped": self.skipped}


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

    have = store.existing_embeddings([c.chunk_id for c in chunks], model)
    missing = [c for c in chunks if c.chunk_id not in have]
    if not missing:
        return EmbedPlan(model=model, embedded=0, skipped=len(chunks))

    stamp = now or datetime.now(timezone.utc).isoformat()
    vectors = embedder.embed_batch([c.text for c in missing])
    if len(vectors) != len(missing):
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors for {len(missing)} chunks; "
            "a partial answer cannot be matched to its inputs"
        )

    records = []
    for chunk, vector in zip(missing, vectors):
        if not vector:
            raise RuntimeError(
                f"empty vector for chunk {chunk.chunk_id[:12]}: an absent embedding "
                "is not a zero vector, and storing one would make it look present"
            )
        records.append(EmbeddingRecord(chunk_id=chunk.chunk_id, model=model,
                                       dimensions=len(vector), vector=tuple(vector),
                                       indexed_at=stamp))
    store.upsert_embeddings(records)
    return EmbedPlan(model=model, embedded=len(records), skipped=len(chunks) - len(records))
