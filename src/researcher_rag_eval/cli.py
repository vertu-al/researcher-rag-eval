"""Command-line interface for RAG evaluation benchmarks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from researcher_rag_eval.evaluation import EvaluationReport, evaluate_case
from researcher_rag_eval.io import load_cases
from researcher_rag_eval.retrieval.chunking import chunk_markdown_directory
from researcher_rag_eval.retrieval.embeddings import (
    DEFAULT_MODEL_NAME,
    load_embedding_model,
)
from researcher_rag_eval.retrieval.io import load_retrieval_benchmark
from researcher_rag_eval.retrieval.runner import (
    run_retrieval_benchmark,
    write_run_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface definition."""
    parser = argparse.ArgumentParser(
        description="Run controlled RAG evaluation benchmarks."
    )

    benchmark_group = parser.add_mutually_exclusive_group(required=True)
    benchmark_group.add_argument(
        "--cases",
        type=Path,
        help="Path to a JSON list of controlled citation evaluation cases.",
    )
    benchmark_group.add_argument(
        "--retrieval-cases",
        type=Path,
        help="Path to a JSON retrieval benchmark dataset.",
    )

    parser.add_argument(
        "--documents",
        type=Path,
        help="Directory containing Markdown documents for retrieval.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the generated retrieval run artifact.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of retrieved chunks per query. Default: 2.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformer model used for retrieval.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a benchmark and print an auditable summary."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cases is not None:
        return _run_controlled_benchmark(args.cases)

    if args.documents is None or args.output is None:
        parser.error(
            "--documents and --output are required with --retrieval-cases."
        )

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    return _run_retrieval_benchmark(
        cases_path=args.retrieval_cases,
        documents_dir=args.documents,
        output_path=args.output,
        model_name=args.model_name,
        top_k=args.top_k,
    )


def _run_controlled_benchmark(cases_path: Path) -> int:
    """Run the original citation and abstention benchmark."""
    cases = load_cases(cases_path)
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


def _run_retrieval_benchmark(
    *,
    cases_path: Path,
    documents_dir: Path,
    output_path: Path,
    model_name: str,
    top_k: int,
) -> int:
    """Run retrieval evaluation and save its immutable run artifact."""
    benchmark = load_retrieval_benchmark(cases_path)
    chunks = chunk_markdown_directory(documents_dir)
    model = load_embedding_model(model_name)

    run = run_retrieval_benchmark(
        benchmark,
        chunks,
        model,
        model_name=model_name,
        top_k=top_k,
    )
    write_run_artifact(run, output_path)

    print(f"saved={output_path}")
    print(_render_retrieval_summary(run))
    return 0


def _render_report(report: EvaluationReport) -> str:
    """Render one controlled-case result as a compact terminal line."""
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


def _render_retrieval_summary(run: dict[str, Any]) -> str:
    """Render aggregate retrieval metrics as a compact terminal line."""
    summary = run["summary"]
    top_k = run["configuration"]["top_k"]

    return (
        f"summary cases={summary['total_case_count']} "
        f"scored={summary['scored_case_count']} "
        f"expected_abstention={summary['expected_abstention_case_count']} "
        f"mean_hit@{top_k}={_format_metric(summary['mean_hit_at_k'])} "
        f"mean_recall@{top_k}={_format_metric(summary['mean_recall_at_k'])} "
        f"mean_rr={_format_metric(summary['mean_reciprocal_rank'])}"
    )


def _format_metric(value: float | None) -> str:
    """Format a metric, preserving the absence of applicable values."""
    return f"{value:.3f}" if value is not None else "n/a"