"""
Embedding Backend for Semantic Code Search

Provides interfaces and implementations for generating vector embeddings from code chunks.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from typing import List, Protocol, Optional, Dict, Any
from pathlib import Path
import json
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Protocol for embedding backends."""

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector (list of floats)
        """
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        ...

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Get model identifier."""
        ...


@dataclass
class EmbeddingConfig:
    """Configuration for embedding backend."""

    model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None  # 'cpu', 'cuda', or None for auto
    batch_size: int = 32
    normalize: bool = True
    cache_dir: Optional[str] = None


class SentenceTransformerBackend:
    """
    Embedding backend using sentence-transformers.

    This is the default backend for RAGIX. It provides high-quality
    embeddings while being fast and local-first (no API calls).
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize sentence-transformers backend.

        Args:
            config: Optional configuration. Defaults to all-MiniLM-L6-v2.
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension = None

    def _ensure_model(self):
        """Lazy-load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model: {self.config.model_name}")
        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
            cache_folder=self.config.cache_dir,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self._dimension}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        self._ensure_model()
        embedding = self._model.encode(
            text, normalize_embeddings=self.config.normalize, convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self._ensure_model()

        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )

        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._ensure_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return self.config.model_name


class DummyEmbeddingBackend:
    """
    Dummy backend for testing without sentence-transformers.

    Generates random but deterministic embeddings based on text hash.
    DO NOT USE IN PRODUCTION.
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize dummy backend.

        Args:
            dimension: Embedding dimension (default matches all-MiniLM-L6-v2)
        """
        self._dimension = dimension
        logger.warning("Using DummyEmbeddingBackend. Not suitable for production!")

    def embed_text(self, text: str) -> List[float]:
        """Generate deterministic random embedding based on text hash."""
        import hashlib
        import random

        # Use text hash as seed for reproducibility
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        # Generate random unit vector
        vec = [rng.gauss(0, 1) for _ in range(self._dimension)]

        # Normalize to unit length
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch."""
        return [self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return "dummy"


def _server_message(response: Any) -> str:
    """The server's own words for a rejection, kept short and never parsed."""
    try:
        body = response.json()
    except Exception:
        body = None
    text = body.get("error") if isinstance(body, dict) else None
    if not text:
        text = (getattr(response, "text", "") or "").strip()
    return str(text)[:500]


@dataclass(frozen=True)
class EmbeddingRefusal:
    """One text the embedder would not embed, and everything needed to judge it.

    A refusal is a fact about a **model and a text together**, not about the text:
    the same span may embed under another model, and the run that reports it must
    say which one refused. `batch_size` is here for the same reason — a reader
    asking "refused by what" is asking about the request that was actually sent.

    `message` is the server's own words, kept verbatim as a signal. Nothing in
    this module decides anything by reading it: what attributes a refusal to a
    text is that the text refuses **alone** (see `embed_batch_recording_refusals`),
    which is a measurement rather than a guess about someone else's wording.
    """

    index: int
    reason: str
    status: int
    message: str
    model: str
    batch_size: int
    chars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "reason": self.reason, "status": self.status,
            "message": self.message, "model": self.model,
            "batch_size": self.batch_size, "chars": self.chars,
        }


#: The reason a refusal carries. One word, and it stays one word until a second
#: condition is actually observed: a vocabulary listing conditions nobody has met
#: reads like knowledge and is a guess.
REFUSED_INPUT_REJECTED = "input-rejected"


class OllamaEmbeddingBackend:
    """Embeddings from a local Ollama server.

    A port implementation, deliberately minimal: no model management, and no
    dependency beyond `requests`. Ollama is a local service, so this keeps the
    sovereign posture — nothing leaves the machine.

    **A batch is one request, not a loop of them.** `/api/embed` takes a list and
    answers with one vector per input; sending the texts one at a time costs a
    round trip each, and the round trip is the cost. Measured on the executor
    (GB10, ollama 0.19, 300 chunks of a real corpus, 2026-09-05):

    ==============================  =========  ==========  ===========
    model                           one by one  32 a time  128 a time
    ==============================  =========  ==========  ===========
    snowflake-arctic-embed2 (1024)   4.3/s      75.5/s      93.7/s
    nomic-embed-text (768)          48.4/s     160.3/s     178.4/s
    ==============================  =========  ==========  ===========

    The vectors are the same either way — maximum absolute difference against the
    one-by-one vectors was **0.0** for every batched cell of that measurement, for
    both models — so this is a change of transport and not of result.

    Two endpoints exist across Ollama versions: /api/embed returns {"embeddings":
    [[...]]} and the older /api/embeddings returns {"embedding": [...]}. Both are
    accepted, because which one a given server speaks is not something a caller
    should have to know, and guessing wrong is a confusing failure rather than an
    obvious one.

    Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio
    """

    #: Texts per request when nothing says otherwise. 128 is the measured best of
    #: the three sizes probed; 32 already captures most of the gain, and 1 is the
    #: explicit way to ask for one request per text.
    DEFAULT_BATCH_SIZE = 128

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size is texts per request, at least 1: got {batch_size}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """The dimension this model produces, learned from its first answer.

        Not declared up front: a hardcoded dimension is a claim about a model the
        caller chose, and it is wrong the first time someone chooses another.
        """
        if self._dimension is None:
            self._dimension = len(self.embed_text("dimension probe"))
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        import requests

        for endpoint, payload, key in (
            ("/api/embed", {"model": self.model, "input": text}, "embeddings"),
            ("/api/embeddings", {"model": self.model, "prompt": text}, "embedding"),
        ):
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout
                )
            except Exception as exc:  # pragma: no cover - network shape
                raise RuntimeError(f"ollama embedding request failed: {exc}") from exc
            if response.status_code == 404:
                continue
            response.raise_for_status()
            data = response.json()
            vector = data.get(key)
            if key == "embeddings" and vector:
                vector = vector[0]
            if not vector:
                raise RuntimeError(
                    f"ollama returned no vector for model {self.model!r}; "
                    "an empty embedding is not a zero vector, and storing one "
                    "would make an unembedded chunk look embedded"
                )
            return [float(v) for v in vector]

        raise RuntimeError(
            f"no embedding endpoint on {self.base_url}: tried /api/embed and /api/embeddings"
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """One request per slice of `batch_size`, in order.

        `batch_size == 1` is the per-text path and is a configuration, not a
        fallback: nothing here decides on its own to send texts one at a time.

        A server that answers a batch without an `embeddings` array, or with the
        wrong number of vectors, **raises**. Retrying one by one would turn a
        server that cannot batch into a run that is twenty times slower and says
        nothing about why, and matching N inputs to fewer vectors cannot be done
        at all — the caller would store one chunk's embedding against another's.
        """
        if not texts:
            return []
        if self.batch_size == 1:
            return [self.embed_text(text) for text in texts]

        import requests

        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            slice_ = texts[start:start + self.batch_size]
            try:
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": slice_},
                    timeout=self.timeout,
                )
            except Exception as exc:  # pragma: no cover - network shape
                raise RuntimeError(f"ollama embedding request failed: {exc}") from exc

            if response.status_code == 404:
                raise RuntimeError(
                    f"no /api/embed on {self.base_url}: this server predates batched "
                    "embedding. Set batch_size=1 to send one text per request."
                )
            response.raise_for_status()
            answer = response.json().get("embeddings")
            if not isinstance(answer, list) or len(answer) != len(slice_):
                raise RuntimeError(
                    f"ollama answered {len(slice_)} input(s) with "
                    f"{len(answer) if isinstance(answer, list) else type(answer).__name__}: "
                    "a batch that cannot be matched to its inputs is not a partial "
                    "success, and pairing them by position would embed the wrong text"
                )
            for vector in answer:
                if not vector:
                    raise RuntimeError(
                        f"ollama returned an empty vector for model {self.model!r}; "
                        "an empty embedding is not a zero vector, and storing one "
                        "would make an unembedded chunk look embedded"
                    )
                vectors.append([float(v) for v in vector])
        return vectors

    def embed_batch_recording_refusals(
        self, texts: List[str]
    ) -> "tuple[List[Optional[List[float]]], List[EmbeddingRefusal]]":
        """Vectors aligned with `texts`, and the texts the server would not embed.

        `embed_batch` raises on a refusal and is unchanged: every caller that wants
        all-or-nothing keeps it. This is the other contract, for a lane that must
        report what it holds and what it refused rather than die on the first one —
        a stage that stops at the first refusal has no lane, not a strict one.

        **A refusal is attributed by isolation, not by reading the error.** One
        request carries up to `batch_size` texts and a rejection names the request,
        not the text. The slice is halved and retried until a single text refuses
        on its own; that text is the refusal, and the server's message rides along
        as a signal. Matching on the message instead would make this code depend on
        another project's wording, and would call a whole slice refused on the word
        of one sentence. The cost is bounded: k refusals in a slice of n cost about
        k·log2(n) extra requests — measured, 9 refusals in 51 598 chunks cost 63.

        Only a rejection of the request as sent (HTTP 4xx other than 404) is a
        refusal. A missing endpoint, a server error, a connection failure and an
        answer whose vector count does not match its inputs all raise: they say
        nothing about any particular text, and recording them per text would
        attribute a broken server to the corpus.

        The returned list has one entry per input, `None` where the text was
        refused, so a caller cannot lose track of which vector belongs to which
        text. Vectors are never silently dropped.
        """
        if not texts:
            return [], []

        vectors: List[Optional[List[float]]] = []
        refusals: List[EmbeddingRefusal] = []
        for start in range(0, len(texts), self.batch_size):
            got, refused = self._embed_slice(texts[start:start + self.batch_size], start)
            vectors.extend(got)
            refusals.extend(refused)
        return vectors, refusals

    def _embed_slice(
        self, slice_: List[str], offset: int
    ) -> "tuple[List[Optional[List[float]]], List[EmbeddingRefusal]]":
        """One request, halved and retried while the server rejects it."""
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": slice_},
                timeout=self.timeout,
            )
        except Exception as exc:  # pragma: no cover - network shape
            raise RuntimeError(f"ollama embedding request failed: {exc}") from exc

        if response.status_code == 404:
            raise RuntimeError(
                f"no /api/embed on {self.base_url}: this server predates batched "
                "embedding. Set batch_size=1 to send one text per request."
            )

        if 400 <= response.status_code < 500:
            message = _server_message(response)
            if len(slice_) == 1:
                return [None], [EmbeddingRefusal(
                    index=offset, reason=REFUSED_INPUT_REJECTED,
                    status=response.status_code, message=message,
                    model=self.model, batch_size=self.batch_size,
                    chars=len(slice_[0]),
                )]
            middle = len(slice_) // 2
            left_v, left_r = self._embed_slice(slice_[:middle], offset)
            right_v, right_r = self._embed_slice(slice_[middle:], offset + middle)
            return left_v + right_v, left_r + right_r

        response.raise_for_status()
        answer = response.json().get("embeddings")
        if not isinstance(answer, list) or len(answer) != len(slice_):
            raise RuntimeError(
                f"ollama answered {len(slice_)} input(s) with "
                f"{len(answer) if isinstance(answer, list) else type(answer).__name__}: "
                "a batch that cannot be matched to its inputs is not a partial "
                "success, and pairing them by position would embed the wrong text"
            )
        vectors: List[Optional[List[float]]] = []
        for vector in answer:
            if not vector:
                raise RuntimeError(
                    f"ollama returned an empty vector for model {self.model!r}; "
                    "an empty embedding is not a zero vector, and storing one "
                    "would make an unembedded chunk look embedded"
                )
            vectors.append([float(v) for v in vector])
        return vectors, []


def create_embedding_backend(
    backend_type: str = "sentence-transformers", config: Optional[EmbeddingConfig] = None
) -> EmbeddingBackend:
    """
    Factory function to create embedding backend.

    Args:
        backend_type: Type of backend ('sentence-transformers' or 'dummy')
        config: Optional configuration

    Returns:
        EmbeddingBackend instance

    Raises:
        ValueError: If backend_type is unknown
    """
    if backend_type == "sentence-transformers":
        return SentenceTransformerBackend(config)
    elif backend_type == "ollama":
        model = getattr(config, "model_name", None) or "nomic-embed-text"
        base_url = getattr(config, "base_url", None) or "http://localhost:11434"
        # `or` would swallow a 0 and quietly substitute the default; an invalid
        # size is the caller's mistake and is theirs to see.
        size = getattr(config, "batch_size", None)
        return OllamaEmbeddingBackend(
            model=model, base_url=base_url,
            batch_size=OllamaEmbeddingBackend.DEFAULT_BATCH_SIZE if size is None else size)
    elif backend_type == "dummy":
        dimension = 384
        if config and hasattr(config, "dimension"):
            dimension = config.dimension
        return DummyEmbeddingBackend(dimension)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Choose 'sentence-transformers', 'ollama' or 'dummy'."
        )


@dataclass
class ChunkEmbedding:
    """
    Embedding for a code chunk.

    Links a chunk identifier to its vector embedding and metadata.
    """

    chunk_id: str  # Unique identifier (e.g., "file.py:func_name")
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    embedding: List[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkEmbedding":
        """Deserialize from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            chunk_type=data["chunk_type"],
            name=data["name"],
            embedding=data["embedding"],
            metadata=data["metadata"],
        )


def embed_code_chunks(
    chunks: List[Any], backend: EmbeddingBackend, batch_size: int = 32
) -> List[ChunkEmbedding]:
    """
    Generate embeddings for a list of CodeChunk objects.

    Args:
        chunks: List of CodeChunk objects (from chunking.py)
        backend: Embedding backend to use
        batch_size: Batch size for embedding generation

    Returns:
        List of ChunkEmbedding objects with vectors
    """
    if not chunks:
        return []

    # Prepare texts for embedding
    texts = [chunk.content for chunk in chunks]

    # Generate embeddings in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = backend.embed_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)

    # Create ChunkEmbedding objects
    chunk_embeddings = []
    for chunk, embedding in zip(chunks, all_embeddings):
        # Generate unique ID: file_path:name
        chunk_id = f"{chunk.file_path}:{chunk.name}"

        chunk_emb = ChunkEmbedding(
            chunk_id=chunk_id,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type),
            name=chunk.name,
            embedding=embedding,
            metadata=chunk.metadata,
        )
        chunk_embeddings.append(chunk_emb)

    return chunk_embeddings


def save_embeddings(embeddings: List[ChunkEmbedding], output_path: Path):
    """
    Save embeddings to JSON file.

    Args:
        embeddings: List of ChunkEmbedding objects
        output_path: Path to output file
    """
    data = {
        "version": "0.6.0",
        "num_embeddings": len(embeddings),
        "dimension": len(embeddings[0].embedding) if embeddings else 0,
        "embeddings": [emb.to_dict() for emb in embeddings],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")


def load_embeddings(input_path: Path) -> List[ChunkEmbedding]:
    """
    Load embeddings from JSON file.

    Args:
        input_path: Path to input file

    Returns:
        List of ChunkEmbedding objects
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = [ChunkEmbedding.from_dict(emb_data) for emb_data in data["embeddings"]]

    logger.info(f"Loaded {len(embeddings)} embeddings from {input_path}")
    return embeddings
