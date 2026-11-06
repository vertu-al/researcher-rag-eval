"""Deterministic in-memory vector retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from researcher_rag_eval.retrieval.chunking import DocumentChunk
from researcher_rag_eval.retrieval.embeddings import EmbeddedCorpus


@dataclass(frozen=True)
class RetrievalResult:
    """One chunk returned for a query, including rank and similarity score."""

    rank: int
    score: float
    chunk: DocumentChunk


def retrieve_top_k(
    query_vector: NDArray[np.float32],
    corpus: EmbeddedCorpus,
    *,
    top_k: int = 3,
) -> tuple[RetrievalResult, ...]:
    """Rank corpus chunks by dot product of normalized embedding vectors."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("Query embedding must be one-dimensional.")

    if corpus.vectors.ndim != 2:
        raise ValueError("Corpus embeddings must be two-dimensional.")

    if corpus.vectors.shape[0] != len(corpus.chunks):
        raise ValueError("Corpus chunks and vectors must have the same length.")

    if corpus.vectors.shape[1] != query.shape[0]:
        raise ValueError("Query and corpus embedding dimensions must match.")

    scores = corpus.vectors @ query
    ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

    return tuple(
        RetrievalResult(
            rank=rank,
            score=float(scores[index]),
            chunk=corpus.chunks[index],
        )
        for rank, index in enumerate(ranked_indices, start=1)
    )