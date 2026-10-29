"""Loading utilities for version-controlled evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researcher_rag_eval.schema import (
    CandidateClaim,
    ClaimAnnotation,
    EvaluationCase,
    EvidencePassage,
    SupportStatus,
)


def load_cases(path: Path) -> tuple[EvaluationCase, ...]:
    """Load a JSON list of controlled evaluation cases."""
    with path.open(encoding="utf-8") as file:
        raw_cases: Any = json.load(file)

    if not isinstance(raw_cases, list):
        raise ValueError("Benchmark JSON must contain a top-level list of cases.")

    cases = tuple(_parse_case(raw_case) for raw_case in raw_cases)
    case_ids = tuple(case.case_id for case in cases)

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Benchmark case IDs must be unique.")

    return cases


def _parse_case(raw_case: Any) -> EvaluationCase:
    """Translate one JSON object into an immutable evaluation case."""
    if not isinstance(raw_case, dict):
        raise ValueError("Every benchmark case must be a JSON object.")

    try:
        return EvaluationCase(
            case_id=raw_case["case_id"],
            question=raw_case["question"],
            candidate_answer=raw_case["candidate_answer"],
            candidate_abstained=raw_case["candidate_abstained"],
            expected_abstention=raw_case["expected_abstention"],
            evidence=tuple(
                EvidencePassage(
                    evidence_id=raw_evidence["evidence_id"],
                    source_title=raw_evidence["source_title"],
                    text=raw_evidence["text"],
                )
                for raw_evidence in raw_case["evidence"]
            ),
            claims=tuple(
                CandidateClaim(
                    claim_id=raw_claim["claim_id"],
                    text=raw_claim["text"],
                    citation_ids=tuple(raw_claim["citation_ids"]),
                )
                for raw_claim in raw_case["claims"]
            ),
            annotations=tuple(
                ClaimAnnotation(
                    claim_id=raw_annotation["claim_id"],
                    support_status=SupportStatus(
                        raw_annotation["support_status"]
                    ),
                    supporting_evidence_ids=tuple(
                        raw_annotation["supporting_evidence_ids"]
                    ),
                )
                for raw_annotation in raw_case["annotations"]
            ),
        )
    except KeyError as error:
        raise ValueError(
            f"Benchmark case is missing required field: {error.args[0]!r}."
        ) from error