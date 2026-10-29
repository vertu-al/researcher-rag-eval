"""Tests for the controlled RAG evaluation data contract."""

import pytest

from researcher_rag_eval.schema import (
    CandidateClaim,
    ClaimAnnotation,
    EvaluationCase,
    EvidencePassage,
    SupportStatus,
)


def _evidence() -> tuple[EvidencePassage, ...]:
    return (
        EvidencePassage(
            evidence_id="evidence-01",
            source_title="Example study",
            text="The intervention improved the measured outcome in this sample.",
        ),
    )


def _supported_annotation() -> tuple[ClaimAnnotation, ...]:
    return (
        ClaimAnnotation(
            claim_id="claim-01",
            support_status=SupportStatus.SUPPORTED,
            supporting_evidence_ids=("evidence-01",),
        ),
    )


def test_evaluation_case_accepts_a_supported_claim() -> None:
    case = EvaluationCase(
        case_id="supported-claim",
        question="Did the intervention improve the outcome?",
        candidate_answer="Yes, the intervention improved the measured outcome.",
        candidate_abstained=False,
        expected_abstention=False,
        evidence=_evidence(),
        claims=(
            CandidateClaim(
                claim_id="claim-01",
                text="The intervention improved the measured outcome.",
                citation_ids=("evidence-01",),
            ),
        ),
        annotations=_supported_annotation(),
    )

    assert case.case_id == "supported-claim"


def test_schema_allows_a_claim_without_citations() -> None:
    case = EvaluationCase(
        case_id="missing-citation",
        question="Did the intervention improve the outcome?",
        candidate_answer="Yes, the intervention improved the measured outcome.",
        candidate_abstained=False,
        expected_abstention=False,
        evidence=_evidence(),
        claims=(
            CandidateClaim(
                claim_id="claim-01",
                text="The intervention improved the measured outcome.",
                citation_ids=(),
            ),
        ),
        annotations=_supported_annotation(),
    )

    assert case.claims[0].citation_ids == ()


def test_schema_allows_an_unknown_citation_for_the_evaluator_to_detect() -> None:
    case = EvaluationCase(
        case_id="unknown-citation",
        question="Did the intervention improve the outcome?",
        candidate_answer="Yes, the intervention improved the measured outcome.",
        candidate_abstained=False,
        expected_abstention=False,
        evidence=_evidence(),
        claims=(
            CandidateClaim(
                claim_id="claim-01",
                text="The intervention improved the measured outcome.",
                citation_ids=("evidence-does-not-exist",),
            ),
        ),
        annotations=_supported_annotation(),
    )

    assert case.claims[0].citation_ids == ("evidence-does-not-exist",)


def test_evaluation_case_requires_an_annotation_for_every_claim() -> None:
    with pytest.raises(
        ValueError,
        match="Every claim must have exactly one annotation",
    ):
        EvaluationCase(
            case_id="missing-annotation",
            question="Did the intervention improve the outcome?",
            candidate_answer="Yes, the intervention improved the measured outcome.",
            candidate_abstained=False,
            expected_abstention=False,
            evidence=_evidence(),
            claims=(
                CandidateClaim(
                    claim_id="claim-01",
                    text="The intervention improved the measured outcome.",
                    citation_ids=("evidence-01",),
                ),
            ),
            annotations=(),
        )


def test_supported_annotation_requires_supporting_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="Supported claims must name at least one supporting evidence passage",
    ):
        ClaimAnnotation(
            claim_id="claim-01",
            support_status=SupportStatus.SUPPORTED,
        )