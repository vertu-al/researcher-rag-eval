from __future__ import annotations

import numpy as np
import pytest

from researcher_rag_eval.retrieval.chunking import DocumentChunk
from researcher_rag_eval.retrieval.embeddings import embed_chunks


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.sentences: list[str] | None = None
        self.normalize_embeddings: bool | None = None
        self.convert_to_numpy: bool | None = None

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.sentences = sentences
        self.normalize_embeddings = normalize_embeddings
        self.convert_to_numpy = convert_to_numpy

        return np.tile(np.array([[0.6, 0.8]], dtype=np.float32),(len(sentences), 1),
        )


def _chunks() -> tuple[DocumentChunk, ...]:
    return (
        DocumentChunk(
            document_id="first",
            chunk_id="first::p001",
            source_path="first.md",
            position=1,
            text="First evidence passage.",
        ),
        DocumentChunk(
            document_id="second",
            chunk_id="second::p001",
            source_path="second.md",
            position=1,
            text="Second evidence passage.",
        ),
    )


def test_embed_chunks_returns_vectors_for_each_chunk() -> None:
    model = FakeEmbeddingModel()

    corpus = embed_chunks(_chunks(), model, model_name="demo-model")

    assert corpus.model_name == "demo-model"
    assert corpus.vectors.shape == (2, 2)
    assert model.sentences == [
        "First evidence passage.",
        "Second evidence passage.",
    ]
    assert model.normalize_embeddings is True
    assert model.convert_to_numpy is True


def test_embed_chunks_rejects_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="At least one document chunk"):
        embed_chunks((), FakeEmbeddingModel())


def test_embed_query_returns_one_vector() -> None:
    model = FakeEmbeddingModel()

    from researcher_rag_eval.retrieval.embeddings import embed_query

    vector = embed_query("What was the sample size?", model)

    assert vector.shape == (2,)
    assert model.sentences == ["What was the sample size?"]
    assert model.normalize_embeddings is True