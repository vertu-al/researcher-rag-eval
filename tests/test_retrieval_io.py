from __future__ import annotations

import json
from pathlib import Path

import pytest

from researcher_rag_eval.retrieval.io import load_retrieval_benchmark


def _benchmark_payload() -> dict:
    return {
        "dataset_id": "demo-retrieval",
        "dataset_version": "v1",
        "cases": [
            {
                "case_id": "answerable",
                "question": "What does the evidence show?",
                "relevant_chunk_ids": ["evidence::p001"],
                "expected_abstention": False,
            },
            {
                "case_id": "unanswerable",
                "question": "What adverse events occurred?",
                "relevant_chunk_ids": [],
                "expected_abstention": True,
            },
        ],
    }


def test_load_retrieval_benchmark_reads_metadata_and_cases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(_benchmark_payload()), encoding="utf-8")

    benchmark = load_retrieval_benchmark(path)

    assert benchmark.dataset_id == "demo-retrieval"
    assert benchmark.dataset_version == "v1"
    assert [case.case_id for case in benchmark.cases] == [
        "answerable",
        "unanswerable",
    ]


def test_load_retrieval_benchmark_rejects_duplicate_case_ids(
    tmp_path: Path,
) -> None:
    payload = _benchmark_payload()
    payload["cases"].append(payload["cases"][0])

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be unique"):
        load_retrieval_benchmark(path)


def test_load_retrieval_benchmark_rejects_invalid_gold_chunk_ids(
    tmp_path: Path,
) -> None:
    payload = _benchmark_payload()
    payload["cases"][0]["relevant_chunk_ids"] = "evidence::p001"

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        load_retrieval_benchmark(path)