"""Local embedding utilities for retrieval benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from researcher_rag_eval.retrieval.chunking import DocumentChunk

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel(Protocol):
    """Minimal interface needed from an embedding model."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float32]:
        """Return one embedding vector per input sentence."""


@dataclass(frozen=True)
class EmbeddedCorpus:
    """Document chunks together with their normalized embedding vectors."""

    model_name: str
    chunks: tuple[DocumentChunk, ...]
    vectors: NDArray[np.float32]


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> EmbeddingModel:
    """Load the local sentence-transformer model on first use."""
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: tuple[DocumentChunk, ...],
    model: EmbeddingModel,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> EmbeddedCorpus:
    """Embed document chunks with normalized vectors for cosine retrieval."""
    if not chunks:
        raise ValueError("At least one document chunk is required.")

    vectors = np.asarray(
        model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if vectors.shape[0] != len(chunks):
        raise ValueError("Each document chunk must have one embedding vector.")

    return EmbeddedCorpus(
        model_name=model_name,
        chunks=chunks,
        vectors=vectors,
    )


def embed_query(
    query: str,
    model: EmbeddingModel,
) -> NDArray[np.float32]:
    """Embed one non-empty retrieval query as a normalized vector."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    vectors = np.asarray(
        model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    if vectors.ndim != 2 or vectors.shape[0] != 1:
        raise ValueError("A query must produce exactly one embedding vector.")

    return vectors[0]