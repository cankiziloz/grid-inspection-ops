from app.case_memory import GLUE_RESIDUE_ANNOTATION, seed_demo_glue_residue_feedback
from app.domain import Decision, InspectionRecord, ReviewReasonCode, ReviewResolution, ReviewStatus
from app.repository import SqliteInspectionRepository


def inspection(inspection_id: str, part_id: str) -> InspectionRecord:
    return InspectionRecord(
        inspection_id=inspection_id,
        part_id=part_id,
        image_uri="image.png",
        decision=Decision.INSPECT,
        review_status=ReviewStatus.PENDING,
    )


def test_demo_glue_feedback_resolves_and_annotates_only_pending_demo_glue(tmp_path):
    repository = SqliteInspectionRepository(str(tmp_path / "inspections.db"))
    repository.save(inspection("demo-glue-pending", "demo-glue-00001"))
    repository.save(inspection("real-glue-pending", "production-glue-00001"))
    repository.save(inspection("demo-glue-confirmed", "demo-glue-00002"))
    repository.resolve_review(
        "demo-glue-confirmed",
        ReviewResolution(
            reviewer_id="operator",
            final_decision=Decision.REJECT,
            reason_code=ReviewReasonCode.TRUE_DEFECT_CONFIRMED,
            reason="Confirmed quality defect.",
            reviewed_at="2026-11-06T12:00:00+00:00",
        ),
    )

    result = seed_demo_glue_residue_feedback(repository)

    marked = repository.get("demo-glue-pending")
    assert result == type(result)(resolved=1, annotated=1, skipped=1)
    assert marked is not None
    assert marked.final_decision is Decision.PASS
    assert marked.review is not None
    assert marked.review.reason_code is ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE
    assert marked.metadata["case_memory"] == GLUE_RESIDUE_ANNOTATION
    assert repository.get("real-glue-pending").review_status is ReviewStatus.PENDING
    assert repository.get("demo-glue-confirmed").final_decision is Decision.REJECT

    repeat = seed_demo_glue_residue_feedback(repository)
    assert repeat == type(repeat)(resolved=0, annotated=0, skipped=1)
