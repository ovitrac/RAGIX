"""
The Ollama embedding backend of ragix_core.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2026-08-30

The live test is opt-in: nothing in this suite talks to a model server unless
OLLAMA_LIVE=1 is set. What can be checked without a server is checked without one —
the factory wiring, the refusal of an unknown backend, and the fact that the
dimension is learned rather than declared.
"""

from __future__ import annotations

import os

import pytest

from ragix_core.embeddings import (
    EmbeddingConfig,
    OllamaEmbeddingBackend,
    create_embedding_backend,
)


def test_the_factory_builds_the_ollama_backend():
    backend = create_embedding_backend("ollama", EmbeddingConfig(model_name="nomic-embed-text"))
    assert isinstance(backend, OllamaEmbeddingBackend)
    assert backend.model == "nomic-embed-text"


def test_an_unknown_backend_names_the_ones_that_exist():
    """The likeliest cause of arriving here is a typo; a list is the answer to it."""
    with pytest.raises(ValueError, match="ollama"):
        create_embedding_backend("word2vec")


def test_the_dimension_is_not_declared_up_front():
    """A hardcoded dimension is a claim about a model the caller chose.

    It is wrong the first time someone chooses another, so it is learned from the
    first answer instead — which means it is not known before one is asked for.
    """
    backend = OllamaEmbeddingBackend()
    assert backend._dimension is None


def test_the_base_url_loses_a_trailing_slash():
    assert OllamaEmbeddingBackend(base_url="http://localhost:11434/").base_url == \
        "http://localhost:11434"


@pytest.mark.skipif(os.environ.get("OLLAMA_LIVE") != "1",
                    reason="live embedding test; set OLLAMA_LIVE=1 to run")
def test_it_embeds_against_a_running_server():
    model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    backend = OllamaEmbeddingBackend(model=model)

    one = backend.embed_text("a sentence to embed")
    assert one and all(isinstance(v, float) for v in one)
    assert any(v != 0.0 for v in one), "a zero vector is not an embedding"
    assert backend.dimension == len(one)

    batch = backend.embed_batch(["first", "second"])
    assert len(batch) == 2 and all(len(v) == len(one) for v in batch)
