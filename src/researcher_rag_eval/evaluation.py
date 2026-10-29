"""Deterministic checks for controlled evidence-grounded RAG evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from researcher_rag_eval.schema import (
    CandidateClaim,
    ClaimAnnotation,
    EvaluationCase,
    SupportStatus,
)


class ClaimIssue(StrEnum):
    """Failure modes evaluated independently for each substantive claim."""

    MISSING_CITATION = "missing_citation"
    UNKNOWN_CITATION = "unknown_citation"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CITATION_DOES_NOT_SUPPORT_CLAIM = "citation_does_not_support_claim"


class CaseIssue(StrEnum):
    """Failure modes evaluated at the whole-answer level."""

    FAILURE_TO_ABSTAIN = "failure_to_abstain"
    UNNECESSARY_ABSTENTION = "unnecessary_abstention"


@dataclass(frozen=True)
class ClaimEvaluation:
    """Audit record for one claim in a candidate answer."""

    claim_id: str
    citation_ids: tuple[str, ...]
    unknown_citation_ids: tuple[str, ...]
    has_supporting_citation: bool
    issues: tuple[ClaimIssue, ...]

    @property
    def passed(self) -> bool:
        """Return whether the claim has no detected evaluation issues."""
        return not self.issues


@dataclass(frozen=True)
class EvaluationReport:
    """Deterministic evaluation outcome for one controlled case."""

    case_id: str
    claim_evaluations: tuple[ClaimEvaluation, ...]
    case_issues: tuple[CaseIssue, ...]

    @property
    def passed(self) -> bool:
        """Return whether neither claims nor answer-level checks failed."""
        return not self.case_issues and all(
            evaluation.passed for evaluation in self.claim_evaluations
        )


def evaluate_case(case: EvaluationCase) -> EvaluationReport:
    """Evaluate citation, support, and abstention behaviour for one case."""
    annotations_by_claim_id = {
        annotation.claim_id: annotation for annotation in case.annotations
    }
    available_evidence_ids = {passage.evidence_id for passage in case.evidence}

    claim_evaluations = tuple(
        _evaluate_claim(
            claim=claim,
            annotation=annotations_by_claim_id[claim.claim_id],
            available_evidence_ids=available_evidence_ids,
        )
        for claim in case.claims
    )

    case_issues: list[CaseIssue] = []
    if case.expected_abstention and not case.candidate_abstained:
        case_issues.append(CaseIssue.FAILURE_TO_ABSTAIN)
    if not case.expected_abstention and case.candidate_abstained:
        case_issues.append(CaseIssue.UNNECESSARY_ABSTENTION)

    return EvaluationReport(
        case_id=case.case_id,
        claim_evaluations=claim_evaluations,
        case_issues=tuple(case_issues),
    )


def _evaluate_claim(
    claim: CandidateClaim,
    annotation: ClaimAnnotation,
    available_evidence_ids: set[str],
) -> ClaimEvaluation:
    """Evaluate one claim against its human reference annotation."""
    issues: list[ClaimIssue] = []
    unknown_citation_ids = tuple(
        citation_id
        for citation_id in claim.citation_ids
        if citation_id not in available_evidence_ids
    )

    if not claim.citation_ids:
        issues.append(ClaimIssue.MISSING_CITATION)

    if unknown_citation_ids:
        issues.append(ClaimIssue.UNKNOWN_CITATION)

    supporting_citation_ids = set(claim.citation_ids).intersection(
        annotation.supporting_evidence_ids
    )
    has_supporting_citation = bool(supporting_citation_ids)

    if annotation.support_status is SupportStatus.UNSUPPORTED:
        issues.append(ClaimIssue.UNSUPPORTED_CLAIM)
    elif claim.citation_ids and not has_supporting_citation:
        issues.append(ClaimIssue.CITATION_DOES_NOT_SUPPORT_CLAIM)

    return ClaimEvaluation(
        claim_id=claim.claim_id,
        citation_ids=claim.citation_ids,
        unknown_citation_ids=unknown_citation_ids,
        has_supporting_citation=has_supporting_citation,
        issues=tuple(issues),
    )