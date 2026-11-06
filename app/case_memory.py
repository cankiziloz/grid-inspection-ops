"""V2 helpers for creating an auditable, human-reviewed case-memory corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .domain import Decision, ReviewReasonCode, ReviewResolution, ReviewStatus
from .repository import SqliteInspectionRepository

DEMO_GLUE_PREFIX = "demo-glue-"
GLUE_RESIDUE_ANNOTATION = {
    "semantic_category": "glue",
    "feedback_origin": "seeded_demo",
    "eligible_for_similarity": True,
}


@dataclass(frozen=True)
class DemoGlueFeedbackResult:
    resolved: int
    annotated: int
    skipped: int


def seed_demo_glue_residue_feedback(
    repository: SqliteInspectionRepository,
    *,
    reviewer_id: str = "demo-v2-operator",
) -> DemoGlueFeedbackResult:
    """Resolve only pending demo glue INSPECTs as accepted process residue.

    This deliberately does not alter real records, automatic decisions, or an
    existing human resolution. It seeds a transparent starting corpus for V2.
    """
    resolved = 0
    annotated = 0
    skipped = 0
    for record in repository.list_by_part_prefix(DEMO_GLUE_PREFIX):
        if record.decision is not Decision.INSPECT:
            skipped += 1
            continue
        if record.review_status is ReviewStatus.PENDING:
            record = repository.resolve_review(
                record.inspection_id,
                ReviewResolution(
                    reviewer_id=reviewer_id,
                    final_decision=Decision.PASS,
                    reason_code=ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE,
                    reason=(
                        "Seeded V2 demo feedback: glue-like process residue was accepted "
                        "after human review."
                    ),
                    reviewed_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
            resolved += 1
        if (
            record.review_status is ReviewStatus.RESOLVED
            and record.review is not None
            and record.review.reason_code is ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE
        ):
            before = record.metadata.get("case_memory")
            repository.annotate_case_memory(record.inspection_id, GLUE_RESIDUE_ANNOTATION)
            if before != GLUE_RESIDUE_ANNOTATION:
                annotated += 1
        else:
            skipped += 1
    return DemoGlueFeedbackResult(resolved=resolved, annotated=annotated, skipped=skipped)
