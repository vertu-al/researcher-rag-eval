"""Tests for deterministic citation, support, and abstention checks."""

import pytest

from researcher_rag_eval.evaluation import (
    CaseIssue,
    ClaimIssue,
    evaluate_case,
)
from researcher_rag_eval.schema import (
    CandidateClaim,
    ClaimAnnotation,
    EvaluationCase,
    EvidencePassage,
    SupportStatus,
)


def _case(
    *,
    citation_ids: tuple[str, ...] = ("evidence-01",),
    support_status: SupportStatus = SupportStatus.SUPPORTED,
    supporting_evidence_ids: tuple[str, ...] = ("evidence-01",),
) -> EvaluationCase:
    return EvaluationCase(
        case_id="controlled-case",
        question="Did the intervention improve the outcome?",
        candidate_answer="The intervention improved the measured outcome.",
        candidate_abstained=False,
        expected_abstention=False,
        evidence=(
            EvidencePassage(
                evidence_id="evidence-01",
                source_title="Relevant study",
                text="The intervention improved the measured outcome.",
            ),
            EvidencePassage(
                evidence_id="evidence-02",
                source_title="Irrelevant study",
                text="The study reports a separate background observation.",
            ),
        ),
        claims=(
            CandidateClaim(
                claim_id="claim-01",
                text="The intervention improved the measured outcome.",
                citation_ids=citation_ids,
            ),
        ),
        annotations=(
            ClaimAnnotation(
                claim_id="claim-01",
                support_status=support_status,
                supporting_evidence_ids=supporting_evidence_ids,
            ),
        ),
    )


def test_evaluator_passes_a_fully_supported_claim() -> None:
    report = evaluate_case(_case())

    assert report.passed
    assert report.claim_evaluations[0].has_supporting_citation
    assert report.claim_evaluations[0].issues == ()


def test_evaluator_detects_a_missing_citation() -> None:
    report = evaluate_case(_case(citation_ids=()))

    assert not report.passed
    assert report.claim_evaluations[0].issues == (ClaimIssue.MISSING_CITATION,)


def test_evaluator_detects_an_unknown_citation() -> None:
    report = evaluate_case(
        _case(citation_ids=("evidence-does-not-exist",))
    )

    claim_evaluation = report.claim_evaluations[0]

    assert not report.passed
    assert claim_evaluation.unknown_citation_ids == (
        "evidence-does-not-exist",
    )
    assert set(claim_evaluation.issues) == {
        ClaimIssue.UNKNOWN_CITATION,
        ClaimIssue.CITATION_DOES_NOT_SUPPORT_CLAIM,
    }


def test_evaluator_detects_an_irrelevant_citation() -> None:
    report = evaluate_case(_case(citation_ids=("evidence-02",)))

    assert not report.passed
    assert report.claim_evaluations[0].issues == (
        ClaimIssue.CITATION_DOES_NOT_SUPPORT_CLAIM,
    )


def test_evaluator_detects_an_unsupported_claim() -> None:
    report = evaluate_case(
        _case(
            citation_ids=("evidence-01",),
            support_status=SupportStatus.UNSUPPORTED,
            supporting_evidence_ids=(),
        )
    )

    assert not report.passed
    assert report.claim_evaluations[0].issues == (
        ClaimIssue.UNSUPPORTED_CLAIM,
    )


@pytest.mark.parametrize(
    ("candidate_abstained", "expected_abstention", "expected_issue"),
    [
        (False, True, CaseIssue.FAILURE_TO_ABSTAIN),
        (True, False, CaseIssue.UNNECESSARY_ABSTENTION),
    ],
)
def test_evaluator_detects_abstention_mismatches(
    candidate_abstained: bool,
    expected_abstention: bool,
    expected_issue: CaseIssue,
) -> None:
    case = EvaluationCase(
        case_id="abstention-case",
        question="Can the evidence answer this question?",
        candidate_answer="The available evidence is insufficient.",
        candidate_abstained=candidate_abstained,
        expected_abstention=expected_abstention,
        evidence=(),
        claims=(),
        annotations=(),
    )

    report = evaluate_case(case)

    assert not report.passed
    assert report.case_issues == (expected_issue,)