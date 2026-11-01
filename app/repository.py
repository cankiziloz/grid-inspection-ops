from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .domain import Decision, InspectionRecord, ReviewResolution, ReviewStatus


class SqliteInspectionRepository:
    """Local development repository; replace the connection with PostgreSQL in production."""

    def __init__(self, database_path: str = "runtime/inspections.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    part_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, record: InspectionRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO inspections (inspection_id, part_id, decision, record_json)
                VALUES (?, ?, ?, ?)
                """,
                (record.inspection_id, record.part_id, record.decision.value, record.model_dump_json()),
            )
            connection.execute(
                "INSERT INTO audit_events (inspection_id, event_type, payload_json) VALUES (?, ?, ?)",
                (record.inspection_id, "INSPECTION_CREATED", record.model_dump_json()),
            )

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
            updated = record.model_copy(update={
                "final_decision": review.final_decision,
                "review_status": ReviewStatus.RESOLVED,
                "review": review,
            })
            connection.execute(
                "UPDATE inspections SET record_json = ? WHERE inspection_id = ?",
                (updated.model_dump_json(), inspection_id),
            )
            connection.execute(
                "INSERT INTO audit_events (inspection_id, event_type, payload_json) VALUES (?, ?, ?)",
                (inspection_id, "HUMAN_REVIEW_RESOLVED", review.model_dump_json()),
            )
        return updated
