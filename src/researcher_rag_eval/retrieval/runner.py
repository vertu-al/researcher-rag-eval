"""Execution and persistence of retrieval evaluation runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from researcher_rag_eval.retrieval.chunking import DocumentChunk
from researcher_rag_eval.retrieval.embeddings import (
    DEFAULT_MODEL_NAME,
    EmbeddingModel,
    embed_chunks,
    embed_query,
)
from researcher_rag_eval.retrieval.io import RetrievalBenchmark
from researcher_rag_eval.retrieval.metrics import evaluate_retrieval
from researcher_rag_eval.retrieval.search import retrieve_top_k

RUN_SCHEMA_VERSION = "0.1"
CHUNKING_METHOD = "paragraph-v1"


def run_retrieval_benchmark(
    benchmark: RetrievalBenchmark,
    chunks: tuple[DocumentChunk, ...],
    model: EmbeddingModel,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = 2,
    run_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Run a retrieval benchmark and return a serializable run artifact."""
    corpus = embed_chunks(chunks, model, model_name=model_name)

    case_results: list[dict[str, Any]] = []
    hit_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_rank_values: list[float] = []

    for case in benchmark.cases:
        query_vector = embed_query(case.question, model)
        results = retrieve_top_k(query_vector, corpus, top_k=top_k)

        metrics: dict[str, Any] | None = None
        if case.relevant_chunk_ids:
            case_metrics = evaluate_retrieval(
                results,
                case.relevant_chunk_ids,
            )
            hit_values.append(float(case_metrics.hit_at_k))
            recall_values.append(case_metrics.recall_at_k)
            reciprocal_rank_values.append(case_metrics.reciprocal_rank)
            metrics = {
                "hit_at_k": case_metrics.hit_at_k,
                "recall_at_k": case_metrics.recall_at_k,
                "reciprocal_rank": case_metrics.reciprocal_rank,
            }

        case_results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "expected_abstention": case.expected_abstention,
                "relevant_chunk_ids": list(case.relevant_chunk_ids),
                "retrieved_chunks": [
                    {
                        "rank": result.rank,
                        "chunk_id": result.chunk.chunk_id,
                        "score": result.score,
                    }
                    for result in results
                ],
                "metrics": metrics,
            }
        )

    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id or f"retrieval-{uuid4()}",
        "created_at": created_at or _utc_timestamp(),
        "run_type": "retrieval",
        "system": {
            "name": "researcher-rag-eval",
            "version": "local-dev",
            "embedding_model": model_name,
        },
        "dataset": {
            "id": benchmark.dataset_id,
            "version": benchmark.dataset_version,
        },
        "configuration": {
            "chunking_method": CHUNKING_METHOD,
            "top_k": top_k,
        },
        "summary": {
            "total_case_count": len(benchmark.cases),
            "scored_case_count": len(hit_values),
            "expected_abstention_case_count": sum(
                case.expected_abstention for case in benchmark.cases
            ),
            "mean_hit_at_k": _mean(hit_values),
            "mean_recall_at_k": _mean(recall_values),
            "mean_reciprocal_rank": _mean(reciprocal_rank_values),
        },
        "case_results": case_results,
    }


def write_run_artifact(run: dict[str, Any], output_path: Path) -> None:
    """Write a retrieval run artifact as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float | None:
    """Return a mean value, or None when no metric is applicable."""
    return sum(values) / len(values) if values else None


def _utc_timestamp() -> str:
    """Return an ISO 8601 timestamp in UTC."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")