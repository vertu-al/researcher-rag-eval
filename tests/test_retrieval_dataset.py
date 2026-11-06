from __future__ import annotations

import json
from pathlib import Path

from researcher_rag_eval.retrieval.chunking import chunk_markdown_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "benchmark" / "retrieval_demo" / "documents"
CASES_PATH = PROJECT_ROOT / "benchmark" / "retrieval_demo" / "retrieval_cases.json"


def _load_dataset() -> dict:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def test_demo_corpus_produces_expected_chunk_ids() -> None:
    chunks = chunk_markdown_directory(DOCUMENTS_DIR)

    assert {chunk.chunk_id for chunk in chunks} == {
        "intervention_outcome::p001",
        "sample_characteristics::p001",
        "study_design_limitations::p001",
    }


def test_every_gold_retrieval_id_exists_in_the_demo_corpus() -> None:
    chunks = chunk_markdown_directory(DOCUMENTS_DIR)
    known_chunk_ids = {chunk.chunk_id for chunk in chunks}
    dataset = _load_dataset()

    referenced_chunk_ids = {
        chunk_id
        for case in dataset["cases"]
        for chunk_id in case["relevant_chunk_ids"]
    }

    assert referenced_chunk_ids <= known_chunk_ids


def test_causal_case_retrieves_limitations_but_requires_abstention() -> None:
    dataset = _load_dataset()
    cases_by_id = {case["case_id"]: case for case in dataset["cases"]}

    causal_case = cases_by_id["causal-inference"]

    assert causal_case["relevant_chunk_ids"] == [
        "study_design_limitations::p001"
    ]
    assert causal_case["expected_abstention"] is True