"""Seed clearly labelled, idempotent demo records for the operations dashboard.

The records intentionally span 24 hours, 7 days, 30 days, and older history so
each dashboard time window has meaningful data. Existing real inspections are
never changed. Re-running this script does not duplicate its demo records.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.domain import (
    Decision,
    InspectionRecord,
    ReviewReasonCode,
    ReviewResolution,
    ReviewStatus,
)
from app.repository import SqliteInspectionRepository


SEED_CASES = [
    ("001", 0.3, Decision.PASS, None, None, "ALL_CRITICAL_GATES_PASSED"),
    ("002", 1.1, Decision.REJECT, None, None, "SCORE_AT_OR_ABOVE_REJECT_THRESHOLD"),
    ("003", 2.4, Decision.INSPECT, Decision.PASS, ReviewReasonCode.ACCEPTED_PROCESS_RESIDUE, "SCORE_IN_REVIEW_BAND"),
    ("004", 4.8, Decision.INSPECT, Decision.REJECT, ReviewReasonCode.TRUE_DEFECT_CONFIRMED, "SCORE_IN_REVIEW_BAND"),
    ("005", 8.0, Decision.INSPECT, Decision.PASS, ReviewReasonCode.CAPTURE_ARTIFACT, "LOW_VALID_COVERAGE"),
    ("015", 12.0, Decision.INSPECT, Decision.PASS, ReviewReasonCode.MINOR_DEFECT_ACCEPTED, "SCORE_IN_REVIEW_BAND"),
    ("006", 18.0, Decision.PASS, None, None, "ALL_CRITICAL_GATES_PASSED"),
    ("007", 30.0, Decision.INSPECT, Decision.PASS, ReviewReasonCode.NO_DEFECT_FOUND, "SCORE_IN_REVIEW_BAND"),
    ("008", 55.0, Decision.REJECT, None, None, "SCORE_AT_OR_ABOVE_REJECT_THRESHOLD"),
    ("009", 90.0, Decision.INSPECT, Decision.PASS, ReviewReasonCode.ACCEPTABLE_VARIATION, "SCORE_IN_REVIEW_BAND"),
    ("010", 130.0, Decision.PASS, None, None, "ALL_CRITICAL_GATES_PASSED"),
    ("011", 190.0, Decision.INSPECT, Decision.REJECT, ReviewReasonCode.TRUE_DEFECT_CONFIRMED, "SCORE_IN_REVIEW_BAND"),
    ("012", 290.0, Decision.REJECT, None, None, "SCORE_AT_OR_ABOVE_REJECT_THRESHOLD"),
    ("013", 420.0, Decision.INSPECT, Decision.PASS, ReviewReasonCode.POLICY_EXCEPTION, "SCORE_IN_REVIEW_BAND"),
    ("014", 760.0, Decision.PASS, None, None, "ALL_CRITICAL_GATES_PASSED"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="runtime/inspections.db")
    args = parser.parse_args()

    repository = SqliteInspectionRepository(args.database)
    database = Path(args.database)
    now = datetime.now(timezone.utc)
    inserted = 0
    enriched = 0

    for suffix, age_hours, decision, final, reason_code, policy_reason in SEED_CASES:
        inspection_id = f"demo-dashboard-{suffix}"
        index = int(suffix)
        latency_ms = {
            "vision": float(80 + index * 3),
            "retrieval": float(15 + index),
            "explanation": float(35 + index * 2),
            "end_to_end": float(150 + index * 6),
        }
        existing = repository.get(inspection_id)
        if existing:
            if "latency_ms" not in existing.metadata:
                updated = existing.model_copy(
                    update={"metadata": {**existing.metadata, "latency_ms": latency_ms}}
                )
                with sqlite3.connect(database) as connection:
                    connection.execute(
                        "UPDATE inspections SET record_json = ? WHERE inspection_id = ?",
                        (updated.model_dump_json(), inspection_id),
                    )
                enriched += 1
            continue
        created_at = now - timedelta(hours=age_hours)
        review = None
        review_status = ReviewStatus.NOT_REQUIRED
        if final is not None:
            review_status = ReviewStatus.RESOLVED
            review = ReviewResolution(
                reviewer_id="demo-operator",
                final_decision=final,
                reason_code=reason_code,
                reason=f"Seeded dashboard example: {reason_code.value.lower()}.",
                reviewed_at=created_at.isoformat(),
            )
        record = InspectionRecord(
            inspection_id=inspection_id,
            part_id=f"demo-dashboard-part-{suffix}",
            image_uri="runtime/demo-placeholder.png",
            decision=decision,
            final_decision=final,
            review_status=review_status,
            review=review,
            metadata={
                "critical_policy": {"reason_codes": [policy_reason]},
                "latency_ms": latency_ms,
            },
        )
        repository.save(record)
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE inspections SET created_at = ? WHERE inspection_id = ?",
                (created_at.strftime("%Y-%m-%d %H:%M:%S"), inspection_id),
            )
        inserted += 1
    print(
        f"Inserted {inserted} and enriched {enriched} dashboard demo records into {database}."
    )


if __name__ == "__main__":
    main()
