"""Tests for the benchmark command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from researcher_rag_eval.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "controlled_cases.json"
RETRIEVAL_CASES_PATH = (
    PROJECT_ROOT / "benchmark" / "retrieval_demo" / "retrieval_cases.json"
)
DOCUMENTS_DIR = PROJECT_ROOT / "benchmark" / "retrieval_demo" / "documents"


class FakeEmbeddingModel:
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        vectors_by_text = {
            (
                "In a controlled demonstration sample, intervention A "
                "improved the measured outcome relative to baseline."
            ): [1.0, 0.0, 0.0],
            "The demonstration sample included 40 participants.": [
                0.0,
                1.0,
                0.0,
            ],
            (
                "The observational data contained no control condition "
                "and therefore cannot establish a causal effect."
            ): [0.0, 0.0, 1.0],
            "Did intervention A improve the measured outcome?": [
                1.0,
                0.0,
                0.0,
            ],
            (
                "How many participants were included in the "
                "demonstration sample?"
            ): [0.0, 1.0, 0.0],
            (
                "Does the available evidence establish a causal effect "
                "of intervention A?"
            ): [0.0, 0.0, 1.0],
            "What was the measured outcome compared against?": [
                1.0,
                0.0,
                0.0,
            ],
            "Which adverse events were reported for intervention A?": [
                0.5,
                0.5,
                0.0,
            ],
        }
        return np.array(
            [vectors_by_text[sentence] for sentence in sentences],
            dtype=np.float32,
        )


def test_cli_runs_the_controlled_benchmark(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--cases", str(BENCHMARK_PATH)])
    output_lines = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert output_lines[0] == "case=supported-claim status=PASS"
    assert (
        "case=missing-citation status=FLAGGED issues=missing_citation"
        in output_lines
    )
    assert output_lines[-1] == "summary cases=6 passed=2 flagged=4"


def test_cli_runs_retrieval_benchmark(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "retrieval-run.json"
    monkeypatch.setattr(
        "researcher_rag_eval.cli.load_embedding_model",
        lambda model_name: FakeEmbeddingModel(),
    )

    exit_code = main(
        [
            "--retrieval-cases",
            str(RETRIEVAL_CASES_PATH),
            "--documents",
            str(DOCUMENTS_DIR),
            "--output",
            str(output_path),
            "--model-name",
            "fake-model",
            "--top-k",
            "2",
        ]
    )
    output_lines = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert output_lines[0] == f"saved={output_path}"
    assert output_lines[1] == (
        "summary cases=5 scored=4 expected_abstention=2 "
        "mean_hit@2=1.000 mean_recall@2=1.000 mean_rr=1.000"
    )

    run = json.loads(output_path.read_text(encoding="utf-8"))
    assert run["system"]["embedding_model"] == "fake-model"
    assert run["summary"]["mean_reciprocal_rank"] == 1.0