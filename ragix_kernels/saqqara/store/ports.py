"""
saqqara.store.ports — the store seam, and the registry that fills it.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

One protocol, so the pipeline never names a provider. `build_store` turns a config
section into an adapter by registry lookup; adding a provider is one module and
one registry line. Only `sqlite` exists today, and the seam is here anyway — a
seam added after the second implementation is a refactor, and a refactor is what
this shape exists to avoid.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Protocol, runtime_checkable

from .records import ChunkRecord, DocumentRecord, EdgeRecord, EmbeddingRecord, Hit, ObjectRecord

__all__ = ["DocumentStore", "build_store", "register_store"]


@runtime_checkable
class DocumentStore(Protocol):
    """What every store must answer, whatever it is built on."""

    # ------------------------------------------------------------- documents
    def upsert_document(self, doc: DocumentRecord,
                        objects: Iterable[ObjectRecord] = (),
                        edges: Iterable[EdgeRecord] = ()) -> None: ...

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]: ...

    def list_documents(self, include_trashed: bool = False) -> list[DocumentRecord]: ...

    def find_doc_by_source(self, sha256: str) -> Optional[DocumentRecord]: ...

    def delete_document(self, doc_id: str, purge: bool = False) -> dict[str, int]: ...

    def restore_document(self, doc_id: str) -> bool: ...

    # ---------------------------------------------------------------- pieces
    def replace_chunks(self, doc_id: str, chunks: Iterable[ChunkRecord]) -> dict[str, int]: ...

    def get_chunks(self, doc_id: Optional[str] = None) -> list[ChunkRecord]: ...

    def get_objects(self, doc_id: str) -> list[ObjectRecord]: ...

    def get_edges(self, doc_id: str) -> list[EdgeRecord]: ...

    # ------------------------------------------------------------ embeddings
    def existing_embeddings(self, chunk_ids: Iterable[str], model: str) -> set[str]: ...

    def upsert_embeddings(self, records: Iterable[EmbeddingRecord]) -> int: ...

    # -------------------------------------------------------------- retrieval
    def search(self, vector: Iterable[float], top_k: int, model: str) -> list[Hit]: ...

    def lexical_search(self, query: str, top_k: int) -> list[Hit]: ...

    def status(self) -> dict[str, Any]: ...


_PROVIDERS: dict[str, Callable[..., DocumentStore]] = {}


def register_store(name: str, factory: Callable[..., DocumentStore]) -> None:
    """Add a provider. Re-registering a name is refused rather than silently won."""
    if name in _PROVIDERS:
        raise ValueError(f"store provider {name!r} is already registered")
    _PROVIDERS[name] = factory


#: Providers this package ships, and the module that registers each. Resolved
#: lazily so that registration does not depend on someone having imported the
#: right module first — it did, and the kernel (which imports only this file)
#: failed with "unknown store provider 'sqlite'; registered: none" in any process
#: that had not separately imported the sqlite module. The test suite could not
#: see it, because a test that imports SqliteDocumentStore registers it as a side
#: effect for everything after.
BUILTIN_PROVIDERS = {"sqlite": "ragix_kernels.saqqara.store.sqlite"}


def build_store(config: dict[str, Any]) -> DocumentStore:
    """Build the store a config asks for.

    The error names what is available: a provider typo is the likeliest cause of
    getting here, and a list is the answer to it.
    """
    provider = config.get("provider", "sqlite")
    if provider not in _PROVIDERS and provider in BUILTIN_PROVIDERS:
        import importlib

        importlib.import_module(BUILTIN_PROVIDERS[provider])
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(
            f"unknown store provider {provider!r}; registered: {sorted(set(_PROVIDERS) | set(BUILTIN_PROVIDERS))}"
        )
    return factory(**{k: v for k, v in config.items() if k != "provider"})
