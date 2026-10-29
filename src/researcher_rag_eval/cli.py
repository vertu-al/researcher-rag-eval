"""Command-line interface for running controlled RAG evaluation benchmarks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from researcher_rag_eval.evaluation import EvaluationReport, evaluate_case
from researcher_rag_eval.io import load_cases


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface definition."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate controlled evidence-grounded RAG cases from a JSON file."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="Path to a JSON list of controlled evaluation cases.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a benchmark and print an auditable summary."""
    args = build_parser().parse_args(argv)
    cases = load_cases(args.cases)
    reports = tuple(evaluate_case(case) for case in cases)

    for report in reports:
        print(_render_report(report))

    passed_count = sum(report.passed for report in reports)
    flagged_count = len(reports) - passed_count
    print(
        f"summary cases={len(reports)} "
        f"passed={passed_count} "
        f"flagged={flagged_count}"
    )
    return 0


def _render_report(report: EvaluationReport) -> str:
    """Render one evaluation report as a compact, readable terminal line."""
    issue_codes = [
        issue.value
        for claim_evaluation in report.claim_evaluations
        for issue in claim_evaluation.issues
    ]
    issue_codes.extend(issue.value for issue in report.case_issues)

    status = "PASS" if report.passed else "FLAGGED"
    line = f"case={report.case_id} status={status}"

    if issue_codes:
        line += f" issues={','.join(issue_codes)}"

    return line