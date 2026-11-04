from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .domain import Decision, InspectionRecord, ReviewResolution, ReviewStatus


class SqliteInspectionRepository:
    """Local operational store with idempotent edge-frame acceptance."""

    def __init__(self, database_path: str = "runtime/inspections.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    part_id TEXT NOT NULL,
                    frame_id TEXT,
                    station_id TEXT,
                    camera_id TEXT,
                    decision TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'NOT_REQUIRED',
                    final_decision TEXT,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._migrate_inspections_table(connection)
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_inspections_frame_id "
                "ON inspections(frame_id) WHERE frame_id IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspections_created_at ON inspections(created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspections_review_status ON inspections(review_status)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @staticmethod
    def _migrate_inspections_table(connection: sqlite3.Connection) -> None:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(inspections)")}
        migrations = {
            "frame_id": "TEXT",
            "station_id": "TEXT",
            "camera_id": "TEXT",
            "review_status": "TEXT NOT NULL DEFAULT 'NOT_REQUIRED'",
            "final_decision": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in columns:
                connection.execute(f"ALTER TABLE inspections ADD COLUMN {column} {definition}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _capture_fields(record: InspectionRecord) -> tuple[str | None, str | None, str | None]:
        capture = record.metadata.get("capture", {})
        return capture.get("frame_id"), capture.get("station_id"), capture.get("camera_id")

    def save(self, record: InspectionRecord) -> None:
        frame_id, station_id, camera_id = self._capture_fields(record)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO inspections (
                    inspection_id, part_id, frame_id, station_id, camera_id, decision,
                    review_status, final_decision, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.inspection_id,
                    record.part_id,
                    frame_id,
                    station_id,
                    camera_id,
                    record.decision.value,
                    record.review_status.value,
                    record.final_decision.value if record.final_decision else None,
                    record.model_dump_json(),
                ),
            )
            connection.execute(
                "INSERT INTO audit_events (inspection_id, event_type, payload_json) VALUES (?, ?, ?)",
                (record.inspection_id, "INSPECTION_CREATED", record.model_dump_json()),
            )

    def get_by_frame_id(self, frame_id: str) -> InspectionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM inspections WHERE frame_id = ?", (frame_id,)
            ).fetchone()
        return InspectionRecord.model_validate_json(row["record_json"]) if row else None

    def get(self, inspection_id: str) -> InspectionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM inspections WHERE inspection_id = ?", (inspection_id,)
            ).fetchone()
        return InspectionRecord.model_validate_json(row["record_json"]) if row else None

    def recent_summary(self, limit: int = 50) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT decision, COUNT(*) AS count FROM ("
                "SELECT decision FROM inspections ORDER BY created_at DESC LIMIT ?"
                ") GROUP BY decision",
                (limit,),
            ).fetchall()
        return {"window": limit, "decision_counts": {row["decision"]: row["count"] for row in rows}}

    @staticmethod
    def _reviewed_at(record: InspectionRecord) -> datetime | None:
        if not record.review:
            return None
        try:
            reviewed_at = datetime.fromisoformat(record.review.reviewed_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        if reviewed_at.tzinfo is None:
            reviewed_at = reviewed_at.replace(tzinfo=timezone.utc)
        return reviewed_at.astimezone(timezone.utc)

    def dashboard_summary(self) -> dict:
        """UTC-windowed counters for the live operations dashboard."""
        windows = {
            "24h": ("-24 hours", timedelta(hours=24)),
            "7d": ("-7 days", timedelta(days=7)),
            "30d": ("-30 days", timedelta(days=30)),
            "all": (None, None),
        }
        result: dict = {
            "windows": {},
            "review_queue_depth": 0,
            "oldest_pending_review_minutes": None,
        }
        with self._connect() as connection:
            resolved_rows = connection.execute(
                "SELECT record_json FROM inspections WHERE review_status = 'RESOLVED'"
            ).fetchall()
            resolved_records = [
                InspectionRecord.model_validate_json(row["record_json"]) for row in resolved_rows
            ]
            now = datetime.now(timezone.utc)
            for name, (interval, review_window) in windows.items():
                where = "" if interval is None else "WHERE created_at >= datetime('now', ?)"
                parameters: tuple[str, ...] = () if interval is None else (interval,)
                rows = connection.execute(
                    f"SELECT decision, COUNT(*) AS count FROM inspections {where} GROUP BY decision",
                    parameters,
                ).fetchall()
                counts = {decision.value: 0 for decision in (Decision.PASS, Decision.INSPECT, Decision.REJECT)}
                counts.update({row["decision"]: row["count"] for row in rows})
                total = sum(counts.values())
                policy_reasons: Counter[str] = Counter()
                final_counts = {Decision.PASS.value: 0, Decision.REJECT.value: 0}
                pending_reviews = 0
                record_rows = connection.execute(
                    f"SELECT record_json FROM inspections {where}",
                    parameters,
                ).fetchall()
                for row in record_rows:
                    record = InspectionRecord.model_validate_json(row["record_json"])
                    policy_reasons.update(
                        record.metadata.get("critical_policy", {}).get("reason_codes", [])
                    )
                    if record.final_decision in {Decision.PASS, Decision.REJECT}:
                        final_counts[record.final_decision.value] += 1
                    elif record.review_status is ReviewStatus.PENDING:
                        pending_reviews += 1
                completed = sum(final_counts.values())
                reviewed_records = [
                    record
                    for record in resolved_records
                    if review_window is None
                    or (
                        (reviewed_at := self._reviewed_at(record)) is not None
                        and reviewed_at >= now - review_window
                    )
                ]
                review_outcomes = {Decision.PASS.value: 0, Decision.REJECT.value: 0}
                review_reasons: Counter[str] = Counter()
                for record in reviewed_records:
                    if record.final_decision:
                        review_outcomes[record.final_decision.value] += 1
                    code = (
                        record.review.reason_code.value
                        if record.review and record.review.reason_code
                        else "LEGACY_UNCLASSIFIED"
                    )
                    review_reasons[code] += 1
                resolved = sum(review_outcomes.values())
                result["windows"][name] = {
                    "total": total,
                    "decision_counts": counts,
                    "attention_rate": (counts["INSPECT"] + counts["REJECT"]) / total if total else 0.0,
                    "reject_rate": counts["REJECT"] / total if total else 0.0,
                    "final_disposition": {
                        "completed": completed,
                        "pass": final_counts[Decision.PASS.value],
                        "reject": final_counts[Decision.REJECT.value],
                        "pending": pending_reviews,
                        "pass_rate": final_counts[Decision.PASS.value] / completed if completed else 0.0,
                        "reject_rate": final_counts[Decision.REJECT.value] / completed if completed else 0.0,
                    },
                    "review_outcomes": {
                        "resolved": resolved,
                        "operator_pass": review_outcomes[Decision.PASS.value],
                        "operator_reject": review_outcomes[Decision.REJECT.value],
                        "confirmed_defect_rate": review_outcomes[Decision.REJECT.value] / resolved if resolved else 0.0,
                    },
                    "review_reason_counts": [
                        {"reason_code": reason, "count": count}
                        for reason, count in review_reasons.most_common()
                    ],
                    "top_policy_reasons": [
                        {"reason_code": reason, "count": count}
                        for reason, count in policy_reasons.most_common(5)
                    ],
                }
            result["review_queue_depth"] = connection.execute(
                "SELECT COUNT(*) AS count FROM inspections WHERE review_status = 'PENDING'"
            ).fetchone()["count"]
            oldest_pending_minutes = connection.execute(
                "SELECT (julianday('now') - julianday(MIN(created_at))) * 24 * 60 AS minutes "
                "FROM inspections WHERE review_status = 'PENDING'"
            ).fetchone()["minutes"]
            result["oldest_pending_review_minutes"] = (
                round(oldest_pending_minutes, 1) if oldest_pending_minutes is not None else None
            )
        return result

    def list_recent(self, limit: int = 50) -> list[InspectionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM inspections ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [InspectionRecord.model_validate_json(row["record_json"]) for row in rows]

    def resolve_review(self, inspection_id: str, review: ReviewResolution) -> InspectionRecord:
        if review.final_decision not in {Decision.PASS, Decision.REJECT}:
            raise ValueError("A human review can resolve only to PASS or REJECT.")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM inspections WHERE inspection_id = ?", (inspection_id,)
            ).fetchone()
            if not row:
                raise KeyError(inspection_id)
            record = InspectionRecord.model_validate_json(row["record_json"])
            if record.review_status is not ReviewStatus.PENDING:
                raise ValueError("Inspection is not awaiting human review.")
            updated = record.model_copy(
                update={
                    "final_decision": review.final_decision,
                    "review_status": ReviewStatus.RESOLVED,
                    "review": review,
                }
            )
            connection.execute(
                "UPDATE inspections SET review_status = ?, final_decision = ?, record_json = ? WHERE inspection_id = ?",
                (ReviewStatus.RESOLVED.value, review.final_decision.value, updated.model_dump_json(), inspection_id),
            )
            connection.execute(
                "INSERT INTO audit_events (inspection_id, event_type, payload_json) VALUES (?, ?, ?)",
                (inspection_id, "HUMAN_REVIEW_RESOLVED", review.model_dump_json()),
            )
        return updated
