from datetime import datetime, timedelta, timezone

import pytest

from app.domain import Decision, InspectionRecord, ReviewReasonCode, ReviewResolution, ReviewStatus
from app.repository import SqliteInspectionRepository


def pending_record() -> InspectionRecord:
    return InspectionRecord(
        inspection_id="inspect-1", part_id="part-1", image_uri="image.png",
        decision=Decision.INSPECT, review_status=ReviewStatus.PENDING,
    )


def review(
    decision: Decision = Decision.REJECT,
    reason_code: ReviewReasonCode | None = None,
    reviewed_at: datetime | None = None,
) -> ReviewResolution:
    return ReviewResolution(reviewer_id="operator-7", final_decision=decision,
                            reason="Review decision recorded with evidence.",
                            reviewed_at=(reviewed_at or datetime.now(timezone.utc)).isoformat(),
                            reason_code=reason_code)


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
    repository.resolve_review(
        "review-pass",
        review(Decision.PASS, ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE),
    )
    repository.resolve_review(
        "review-reject",
        review(Decision.REJECT, ReviewReasonCode.TRUE_DEFECT_CONFIRMED),
    )

    summary = repository.dashboard_summary()
    period_summary = summary["windows"]["24h"]
    outcomes = period_summary["review_outcomes"]

    assert summary["review_queue_depth"] == 1
    assert summary["oldest_pending_review_minutes"] is not None
    assert period_summary["final_disposition"] == {
        "completed": 2,
        "pass": 1,
        "reject": 1,
        "pending": 1,
        "pass_rate": 0.5,
        "reject_rate": 0.5,
    }
    assert outcomes == {
        "resolved": 2,
        "operator_pass": 1,
        "operator_reject": 1,
        "review_pass_rate": 0.5,
        "review_reject_rate": 0.5,
        "no_defect_found": 0,
        "no_defect_found_rate": 0.0,
        "accepted_minor_defect": 0,
        "accepted_minor_defect_rate": 0.0,
    }
    assert {
        item["reason_code"]: item["count"] for item in period_summary["review_reason_counts"]
    } == {
        "ACCEPTED_PROCESS_RESIDUE": 1,
        "TRUE_DEFECT_CONFIRMED": 1,
    }


def test_review_reason_code_requires_its_matching_disposition():
    with pytest.raises(ValueError, match="requires final decision REJECT"):
        review(Decision.PASS, ReviewReasonCode.TRUE_DEFECT_CONFIRMED)


def test_dashboard_review_outcomes_use_review_time_not_inspection_time(tmp_path):
    repository = SqliteInspectionRepository(str(tmp_path / "inspections.db"))
    for inspection_id in ("old-review", "recent-review"):
        repository.save(
            InspectionRecord(
                inspection_id=inspection_id,
                part_id=inspection_id,
                image_uri="image.png",
                decision=Decision.INSPECT,
                review_status=ReviewStatus.PENDING,
            )
        )
    repository.resolve_review(
        "old-review",
        review(
            Decision.PASS,
            ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE,
            datetime.now(timezone.utc) - timedelta(days=8),
        ),
    )
    repository.resolve_review(
        "recent-review",
        review(
            Decision.REJECT,
            ReviewReasonCode.TRUE_DEFECT_CONFIRMED,
        ),
    )

    summary = repository.dashboard_summary()

    assert summary["windows"]["24h"]["review_outcomes"]["resolved"] == 1
    assert summary["windows"]["7d"]["review_outcomes"]["resolved"] == 1
    assert summary["windows"]["30d"]["review_outcomes"]["resolved"] == 2
    assert summary["windows"]["all"]["review_outcomes"]["resolved"] == 2


def test_dashboard_separates_no_defect_from_minor_defect_accepted(tmp_path):
    repository = SqliteInspectionRepository(str(tmp_path / "inspections.db"))
    for inspection_id in ("no-defect", "minor-defect"):
        repository.save(
            InspectionRecord(
                inspection_id=inspection_id,
                part_id=inspection_id,
                image_uri="image.png",
                decision=Decision.INSPECT,
                review_status=ReviewStatus.PENDING,
            )
        )
    repository.resolve_review(
        "no-defect",
        review(Decision.PASS, ReviewReasonCode.NO_DEFECT_FOUND),
    )
    repository.resolve_review(
        "minor-defect",
        review(Decision.PASS, ReviewReasonCode.MINOR_DEFECT_ACCEPTED),
    )

    outcomes = repository.dashboard_summary()["windows"]["24h"]["review_outcomes"]

    assert outcomes["no_defect_found"] == 1
    assert outcomes["accepted_minor_defect"] == 1
    assert outcomes["no_defect_found_rate"] == 0.5
    assert outcomes["accepted_minor_defect_rate"] == 0.5


def test_dashboard_reports_p50_and_p95_latency_for_instrumented_inspections(tmp_path):
    repository = SqliteInspectionRepository(str(tmp_path / "inspections.db"))
    for index, end_to_end in enumerate((10.0, 100.0, 200.0), start=1):
        repository.save(
            InspectionRecord(
                inspection_id=f"latency-{index}",
                part_id=f"latency-{index}",
                image_uri="image.png",
                decision=Decision.PASS,
                final_decision=Decision.PASS,
                metadata={
                    "latency_ms": {
                        "end_to_end": end_to_end,
                        "vision": end_to_end / 2,
                        "retrieval": 5.0,
                        "explanation": 8.0,
                    }
                },
            )
        )

    latency = repository.dashboard_summary()["windows"]["24h"]["latency_ms"]

    assert latency["samples"] == 3
    assert latency["end_to_end"] == {"p50_ms": 100.0, "p95_ms": 200.0}
    assert latency["vision"] == {"p50_ms": 50.0, "p95_ms": 100.0}
