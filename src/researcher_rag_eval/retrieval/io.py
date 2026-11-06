"""Loading utilities for retrieval benchmark datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalCase:
    """One query with gold evidence chunks for retrieval evaluation."""

    case_id: str
    question: str
    relevant_chunk_ids: tuple[str, ...]
    expected_abstention: bool


@dataclass(frozen=True)
class RetrievalBenchmark:
    """Versioned retrieval dataset and its evaluation cases."""

    dataset_id: str
    dataset_version: str
    cases: tuple[RetrievalCase, ...]


def load_retrieval_benchmark(path: Path) -> RetrievalBenchmark:
    """Load and validate one retrieval benchmark JSON file."""
    with path.open(encoding="utf-8") as file:
        raw_benchmark: Any = json.load(file)

    if not isinstance(raw_benchmark, dict):
        raise ValueError("Retrieval benchmark must be a JSON object.")

    try:
        raw_cases = raw_benchmark["cases"]
        benchmark = RetrievalBenchmark(
            dataset_id=_require_string(raw_benchmark["dataset_id"], "dataset_id"),
            dataset_version=_require_string(
                raw_benchmark["dataset_version"],
                "dataset_version",
            ),
            cases=tuple(_parse_case(raw_case) for raw_case in raw_cases),
        )
    except KeyError as error:
        raise ValueError(
            f"Retrieval benchmark is missing required field: {error.args[0]!r}."
        ) from error

    if not isinstance(raw_cases, list):
        raise ValueError("Retrieval benchmark cases must be a JSON list.")

    case_ids = tuple(case.case_id for case in benchmark.cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Retrieval benchmark case IDs must be unique.")

    return benchmark


def _parse_case(raw_case: Any) -> RetrievalCase:
    """Translate one JSON object into an immutable retrieval case."""
    if not isinstance(raw_case, dict):
        raise ValueError("Every retrieval case must be a JSON object.")

    try:
        raw_chunk_ids = raw_case["relevant_chunk_ids"]
        if not isinstance(raw_chunk_ids, list) or not all(
            isinstance(chunk_id, str) and chunk_id
            for chunk_id in raw_chunk_ids
        ):
            raise ValueError(
                "relevant_chunk_ids must be a JSON list of non-empty strings."
            )

        expected_abstention = raw_case["expected_abstention"]
        if not isinstance(expected_abstention, bool):
            raise ValueError("expected_abstention must be a boolean.")

        return RetrievalCase(
            case_id=_require_string(raw_case["case_id"], "case_id"),
            question=_require_string(raw_case["question"], "question"),
            relevant_chunk_ids=tuple(raw_chunk_ids),
            expected_abstention=expected_abstention,
        )
    except KeyError as error:
        raise ValueError(
            f"Retrieval case is missing required field: {error.args[0]!r}."
        ) from error


def _require_string(value: Any, field_name: str) -> str:
    """Validate a required non-empty string field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")

    return value