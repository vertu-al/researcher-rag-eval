from __future__ import annotations

import pytest

from researcher_rag_eval.retrieval.chunking import DocumentChunk
from researcher_rag_eval.retrieval.metrics import evaluate_retrieval
from researcher_rag_eval.retrieval.search import RetrievalResult


def _result(rank: int, chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=1.0 / rank,
        chunk=DocumentChunk(
            document_id=chunk_id.split("::")[0],
            chunk_id=chunk_id,
            source_path="demo.md",
            position=rank,
            text="Demo evidence.",
        ),
    )


def test_retrieval_metrics_capture_rank_and_partial_recall() -> None:
    results = (
        _result(1, "irrelevant::p001"),
        _result(2, "relevant_a::p001"),
        _result(3, "relevant_b::p001"),
    )

    metrics = evaluate_retrieval(
        results,
        relevant_chunk_ids=("relevant_a::p001", "relevant_b::p001"),
    )

    assert metrics.top_k == 3
    assert metrics.hit_at_k is True
    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank == 0.5


def test_retrieval_metrics_return_zero_when_nothing_is_found() -> None:
    metrics = evaluate_retrieval(
        (_result(1, "irrelevant::p001"),),
        relevant_chunk_ids=("relevant::p001",),
    )

    assert metrics.hit_at_k is False
    assert metrics.recall_at_k == 0.0
    assert metrics.reciprocal_rank == 0.0


def test_retrieval_metrics_reject_unanswerable_cases() -> None:
    with pytest.raises(ValueError, match="at least one relevant chunk"):
        evaluate_retrieval((), relevant_chunk_ids=())