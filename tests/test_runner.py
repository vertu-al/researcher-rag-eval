from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from researcher_rag_eval.retrieval.chunking import DocumentChunk
from researcher_rag_eval.retrieval.io import (
    RetrievalBenchmark,
    RetrievalCase,
)
from researcher_rag_eval.retrieval.runner import (
    run_retrieval_benchmark,
    write_run_artifact,
)


class FakeEmbeddingModel:
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        vectors_by_text = {
            "Outcome evidence.": [1.0, 0.0, 0.0],
            "Sample evidence.": [0.0, 1.0, 0.0],
            "Limitations evidence.": [0.0, 0.0, 1.0],
            "Outcome question?": [1.0, 0.0, 0.0],
            "Causal question?": [0.0, 0.0, 1.0],
            "Unknown question?": [0.5, 0.5, 0.0],
        }
        return np.array(
            [vectors_by_text[sentence] for sentence in sentences],
            dtype=np.float32,
        )


def _chunks() -> tuple[DocumentChunk, ...]:
    return (
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
        DocumentChunk(
            document_id="limitations",
            chunk_id="limitations::p001",
            source_path="limitations.md",
            position=1,
            text="Limitations evidence.",
        ),
    )


def _benchmark() -> RetrievalBenchmark:
    return RetrievalBenchmark(
        dataset_id="demo-retrieval",
        dataset_version="v1",
        cases=(
            RetrievalCase(
                case_id="outcome",
                question="Outcome question?",
                relevant_chunk_ids=("outcome::p001",),
                expected_abstention=False,
            ),
            RetrievalCase(
                case_id="causal",
                question="Causal question?",
                relevant_chunk_ids=("limitations::p001",),
                expected_abstention=True,
            ),
            RetrievalCase(
                case_id="unknown",
                question="Unknown question?",
                relevant_chunk_ids=(),
                expected_abstention=True,
            ),
        ),
    )


def test_runner_creates_a_serializable_retrieval_run() -> None:
    run = run_retrieval_benchmark(
        _benchmark(),
        _chunks(),
        FakeEmbeddingModel(),
        model_name="fake-model",
        top_k=2,
        run_id="test-run",
        created_at="2026-11-06T12:00:00Z",
    )

    assert run["run_id"] == "test-run"
    assert run["summary"] == {
        "total_case_count": 3,
        "scored_case_count": 2,
        "expected_abstention_case_count": 2,
        "mean_hit_at_k": 1.0,
        "mean_recall_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
    }
    assert run["case_results"][0]["metrics"]["reciprocal_rank"] == 1.0
    assert run["case_results"][2]["metrics"] is None


def test_write_run_artifact_creates_readable_json(tmp_path: Path) -> None:
    output_path = tmp_path / "runs" / "demo.json"
    run = {"run_id": "test-run", "summary": {"mean_hit_at_k": 1.0}}

    write_run_artifact(run, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == run