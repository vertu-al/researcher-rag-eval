"""Tests for loading version-controlled JSON benchmark cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researcher_rag_eval.io import load_cases

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "controlled_cases.json"
)


def test_load_cases_loads_the_controlled_benchmark() -> None:
    cases = load_cases(BENCHMARK_PATH)

    assert len(cases) == 6
    assert tuple(case.case_id for case in cases) == (
        "supported-claim",
        "missing-citation",
        "irrelevant-citation",
        "unknown-citation",
        "failure-to-abstain",
        "appropriate-abstention",
    )
    assert cases[-1].candidate_abstained
    assert cases[-1].expected_abstention


def test_load_cases_rejects_a_non_list_json_document(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"case_id": "not-a-list"}', encoding="utf-8")

    with pytest.raises(ValueError, match="top-level list"):
        load_cases(invalid_path)


def test_load_cases_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    raw_cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    raw_cases.append(raw_cases[0])

    duplicate_path = tmp_path / "duplicate-cases.json"
    duplicate_path.write_text(json.dumps(raw_cases), encoding="utf-8")

    with pytest.raises(ValueError, match="case IDs must be unique"):
        load_cases(duplicate_path)