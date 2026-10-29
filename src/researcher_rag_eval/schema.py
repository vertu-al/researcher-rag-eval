"""Data contracts for controlled evidence-grounded RAG evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SupportStatus(StrEnum):
    """Human reference label for whether available evidence supports a claim."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class EvidencePassage:
    """A retrieved evidence passage available to the candidate answer."""

    evidence_id: str
    source_title: str
    text: str

    def __post_init__(self) -> None:
        _require_text(self.evidence_id, "evidence_id")
        _require_text(self.source_title, "source_title")
        _require_text(self.text, "text")


@dataclass(frozen=True)
class CandidateClaim:
    """One atomic substantive claim extracted from a candidate answer."""

    claim_id: str
    text: str
    citation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.claim_id, "claim_id")
        _require_text(self.text, "text")
        _require_unique_nonempty_ids(self.citation_ids, "citation_ids")


@dataclass(frozen=True)
class ClaimAnnotation:
    """Human reference annotation for one candidate claim."""

    claim_id: str
    support_status: SupportStatus
    supporting_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.claim_id, "claim_id")
        _require_unique_nonempty_ids(
            self.supporting_evidence_ids,
            "supporting_evidence_ids",
        )

        if (
            self.support_status is SupportStatus.SUPPORTED
            and not self.supporting_evidence_ids
        ):
            raise ValueError(
                "Supported claims must name at least one supporting evidence passage."
            )

        if (
            self.support_status is SupportStatus.UNSUPPORTED
            and self.supporting_evidence_ids
        ):
            raise ValueError(
                "Unsupported claims cannot name supporting evidence passages."
            )


@dataclass(frozen=True)
class EvaluationCase:
    """A controlled RAG response and its human-authored evaluation reference."""

    case_id: str
    question: str
    candidate_answer: str
    candidate_abstained: bool
    expected_abstention: bool
    evidence: tuple[EvidencePassage, ...]
    claims: tuple[CandidateClaim, ...]
    annotations: tuple[ClaimAnnotation, ...]

    def __post_init__(self) -> None:
        _require_text(self.case_id, "case_id")
        _require_text(self.question, "question")
        _require_text(self.candidate_answer, "candidate_answer")

        evidence_ids = tuple(passage.evidence_id for passage in self.evidence)
        claim_ids = tuple(claim.claim_id for claim in self.claims)
        annotated_claim_ids = tuple(
            annotation.claim_id for annotation in self.annotations
        )

        _require_unique_nonempty_ids(evidence_ids, "evidence IDs")
        _require_unique_nonempty_ids(claim_ids, "claim IDs")
        _require_unique_nonempty_ids(
            annotated_claim_ids,
            "annotated claim IDs",
        )

        if set(claim_ids) != set(annotated_claim_ids):
            raise ValueError(
                "Every claim must have exactly one annotation, and every annotation "
                "must refer to a claim."
            )


def _require_text(value: str, field_name: str) -> None:
    """Raise a clear error when a required text field is blank."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank.")


def _require_unique_nonempty_ids(
    values: tuple[str, ...],
    field_name: str,
) -> None:
    """Validate identifiers without requiring that a claim cite anything."""
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} must not contain blank identifiers.")

    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate identifiers.")