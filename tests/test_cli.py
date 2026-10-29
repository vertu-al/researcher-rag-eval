"""Tests for the benchmark command-line interface."""

from pathlib import Path

import pytest

from researcher_rag_eval.cli import main

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "controlled_cases.json"
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