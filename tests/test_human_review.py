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
