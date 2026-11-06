"""Metrics for deterministic retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from researcher_rag_eval.retrieval.search import RetrievalResult


@dataclass(frozen=True)
class RetrievalMetrics:
    """Retrieval quality metrics for one answerable evaluation case."""

    top_k: int
    hit_at_k: bool
    recall_at_k: float
    reciprocal_rank: float


def evaluate_retrieval(
    results: tuple[RetrievalResult, ...],
    relevant_chunk_ids: tuple[str, ...],
) -> RetrievalMetrics:
    """Evaluate ranked retrieval results against gold chunk identifiers."""
    if not relevant_chunk_ids:
        raise ValueError("Retrieval metrics require at least one relevant chunk.")

    relevant_ids = set(relevant_chunk_ids)
    retrieved_ids = [result.chunk.chunk_id for result in results]

    matched_ids = set(retrieved_ids).intersection(relevant_ids)
    first_relevant_rank = next(
        (
            result.rank
            for result in results
            if result.chunk.chunk_id in relevant_ids
        ),
        None,
    )

    return RetrievalMetrics(
        top_k=len(results),
        hit_at_k=bool(matched_ids),
        recall_at_k=len(matched_ids) / len(relevant_ids),
        reciprocal_rank=(
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
    )