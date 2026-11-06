from __future__ import annotations

import numpy as np
import pytest

from researcher_rag_eval.retrieval.chunking import DocumentChunk
from researcher_rag_eval.retrieval.embeddings import EmbeddedCorpus
from researcher_rag_eval.retrieval.search import retrieve_top_k


def _corpus() -> EmbeddedCorpus:
    chunks = (
        DocumentChunk(
            document_id="outcome",
            chunk_id="outcome::p001",
            source_path="outcome.md",
            position=1,
            text="Outcome evidence.",
        ),
        DocumentChunk(
            document_id="sample",
            chunk_id="sample::p001",
            source_path="sample.md",
            position=1,
            text="Sample evidence.",
        ),
    )

    return EmbeddedCorpus(
        model_name="demo-model",
        chunks=chunks,
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_retrieve_top_k_returns_chunks_in_score_order() -> None:
    results = retrieve_top_k(
        np.array([0.8, 0.6], dtype=np.float32),
        _corpus(),
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "outcome::p001",
        "sample::p001",
    ]
    assert [result.rank for result in results] == [1, 2]
    assert [result.score for result in results] == pytest.approx([0.8, 0.6])


def test_retrieve_top_k_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve_top_k(
            np.array([1.0, 0.0], dtype=np.float32),
            _corpus(),
            top_k=0,
        )


def test_retrieve_top_k_rejects_wrong_query_dimension() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        retrieve_top_k(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            _corpus(),
        )