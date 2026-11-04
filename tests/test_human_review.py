from datetime import datetime, timezone

import pytest

from app.domain import Decision, InspectionRecord, ReviewResolution, ReviewStatus
from app.repository import SqliteInspectionRepository


def pending_record() -> InspectionRecord:
    return InspectionRecord(
        inspection_id="inspect-1", part_id="part-1", image_uri="image.png",
        decision=Decision.INSPECT, review_status=ReviewStatus.PENDING,
    )


def review(decision: Decision = Decision.REJECT) -> ReviewResolution:
    return ReviewResolution(reviewer_id="operator-7", final_decision=decision,
                            reason="Mask matches a confirmed defect.",
                            reviewed_at=datetime.now(timezone.utc).isoformat())


def test_human_review_resolves_pending_inspection_and_is_immutable(tmp_path):
    repository = SqliteInspectionRepository(str(tmp_path / "inspections.db"))
    repository.save(pending_record())
    resolved = repository.resolve_review("inspect-1", review())
    assert resolved.review_status is ReviewStatus.RESOLVED
    assert resolved.final_decision is Decision.REJECT
    assert resolved.review is not None
    with pytest.raises(ValueError, match="not awaiting"):
        repository.resolve_review("inspect-1", review(Decision.PASS))


def test_human_review_cannot_resolve_to_inspect(tmp_path):
    repository = SqliteInspectionRepository(str(tmp_path / "inspections.db"))
    repository.save(pending_record())
    with pytest.raises(ValueError, match="PASS or REJECT"):
        repository.resolve_review("inspect-1", review(Decision.INSPECT))


def test_dashboard_reports_review_outcomes_without_calling_them_accuracy(tmp_path):
    repository = SqliteInspectionRepository(str(tmp_path / "inspections.db"))
    for inspection_id in ("review-pass", "review-reject", "still-pending"):
        repository.save(
            InspectionRecord(
                inspection_id=inspection_id,
                part_id=inspection_id,
                image_uri="image.png",
                decision=Decision.INSPECT,
                review_status=ReviewStatus.PENDING,
            )
        )
    repository.resolve_review("review-pass", review(Decision.PASS))
    repository.resolve_review("review-reject", review(Decision.REJECT))

    summary = repository.dashboard_summary()
    outcomes = summary["windows"]["24h"]["review_outcomes"]

    assert summary["review_queue_depth"] == 1
    assert summary["oldest_pending_review_minutes"] is not None
    assert outcomes == {
        "resolved": 2,
        "operator_pass": 1,
        "operator_reject": 1,
        "confirmed_defect_rate": 0.5,
    }
