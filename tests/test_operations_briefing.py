from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import api
from app.operations_briefing import build_operations_briefing_snapshot


def dashboard_summary() -> dict:
    latency = {
        "samples": 3,
        "end_to_end": {"p50_ms": 200.0, "p95_ms": 300.0},
        "vision": {"p50_ms": 100.0, "p95_ms": 150.0},
        "retrieval": {"p50_ms": 20.0, "p95_ms": 30.0},
        "explanation": {"p50_ms": 40.0, "p95_ms": 60.0},
    }
    window = {
        "total": 12,
        "decision_counts": {"PASS": 6, "INSPECT": 4, "REJECT": 2},
        "attention_rate": 0.5,
        "reject_rate": 1 / 6,
        "review_outcomes": {
            "resolved": 3,
            "operator_pass": 2,
            "operator_reject": 1,
            "review_pass_rate": 2 / 3,
            "review_reject_rate": 1 / 3,
            "no_defect_found": 1,
            "no_defect_found_rate": 1 / 3,
            "accepted_minor_defect": 1,
            "accepted_minor_defect_rate": 1 / 3,
        },
        "top_policy_reasons": [{"reason_code": "LOW_VALID_COVERAGE", "count": 4}],
        "review_reason_counts": [{"reason_code": "NO_DEFECT_FOUND", "count": 1}],
        "latency_ms": latency,
    }
    return {
        "windows": {"24h": window},
        "review_queue_depth": 2,
        "oldest_pending_review_minutes": 18.5,
    }


def test_briefing_snapshot_is_typed_and_contains_only_dashboard_evidence():
    snapshot = build_operations_briefing_snapshot(
        dashboard_summary(),
        "24h",
        generated_at=datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert snapshot.schema_version == "operations-briefing-snapshot-v1"
    assert snapshot.generated_at == "2026-11-06T12:00:00+00:00"
    assert snapshot.incoming_activity.total == 12
    assert snapshot.review_outcomes.no_defect_found == 1
    assert snapshot.review_queue.oldest_pending_minutes == 18.5
    assert snapshot.top_policy_reasons[0].reason_code == "LOW_VALID_COVERAGE"
    assert snapshot.latency.end_to_end.p95_ms == 300.0


def test_briefing_snapshot_rejects_unknown_period():
    with pytest.raises(ValueError, match="Unsupported briefing period"):
        build_operations_briefing_snapshot(dashboard_summary(), "1h")  # type: ignore[arg-type]


def test_briefing_snapshot_endpoint_uses_deterministic_repository_data(monkeypatch):
    class Repository:
        def dashboard_summary(self) -> dict:
            return dashboard_summary()

    monkeypatch.setattr(api, "runtime", lambda: (None, Repository()))

    response = TestClient(api.app).get("/v1/operations/briefing-snapshot?period=24h")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "deterministic-dashboard"
    assert payload["incoming_activity"]["total"] == 12
    assert payload["latency"]["end_to_end"]["p95_ms"] == 300.0
