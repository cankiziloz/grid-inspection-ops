"""Deterministic input contract for the optional Operations Briefing LLM."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .procedure_routing import POLICY_SECTION_ROUTES
from .rag import load_procedure_chunks

BriefingPeriod = Literal["24h", "7d", "30d", "all"]

PERIOD_LABELS: dict[BriefingPeriod, str] = {
    "24h": "Last 24 hours",
    "7d": "Last 7 days",
    "30d": "Last 30 days",
    "all": "All available history",
}


class ReasonCount(BaseModel):
    reason_code: str
    count: int = Field(ge=0)


class LatencyPhaseSnapshot(BaseModel):
    p50_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)


class LatencySnapshot(BaseModel):
    samples: int = Field(ge=0)
    end_to_end: LatencyPhaseSnapshot
    vision: LatencyPhaseSnapshot
    retrieval: LatencyPhaseSnapshot
    explanation: LatencyPhaseSnapshot


class IncomingActivitySnapshot(BaseModel):
    total: int = Field(ge=0)
    automatic_pass: int = Field(ge=0)
    automatic_reject: int = Field(ge=0)
    sent_to_review: int = Field(ge=0)
    attention_rate: float = Field(ge=0, le=1)
    automatic_reject_rate: float = Field(ge=0, le=1)


class ReviewOutcomesSnapshot(BaseModel):
    resolved: int = Field(ge=0)
    pass_after_review: int = Field(ge=0)
    reject_after_review: int = Field(ge=0)
    no_defect_found: int = Field(ge=0)
    accepted_minor_defect: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    reject_rate: float = Field(ge=0, le=1)
    no_defect_found_rate: float = Field(ge=0, le=1)
    accepted_minor_defect_rate: float = Field(ge=0, le=1)


class ReviewQueueSnapshot(BaseModel):
    open_count: int = Field(ge=0)
    oldest_pending_minutes: float | None = Field(default=None, ge=0)


class SopEvidenceSnapshot(BaseModel):
    document_id: str
    title: str
    section: str
    excerpt: str
    source_reason_codes: list[str]
    related_event_count: int = Field(ge=0)


class OperationsBriefingSnapshot(BaseModel):
    schema_version: Literal["operations-briefing-snapshot-v1"]
    source: Literal["deterministic-dashboard"]
    generated_at: str
    period: BriefingPeriod
    period_label: str
    incoming_activity: IncomingActivitySnapshot
    review_outcomes: ReviewOutcomesSnapshot
    review_queue: ReviewQueueSnapshot
    top_policy_reasons: list[ReasonCount]
    review_disposition_mix: list[ReasonCount]
    latency: LatencySnapshot
    sop_evidence: list[SopEvidenceSnapshot] = Field(default_factory=list)


def _attach_sop_evidence(
    reasons: list[ReasonCount],
    docs_dir: Path | None,
) -> list[SopEvidenceSnapshot]:
    """Attach only SOP sections explicitly approved by policy routing."""
    if docs_dir is None:
        return []
    chunks = {
        (chunk.document_id, chunk.section): chunk
        for chunk in load_procedure_chunks(docs_dir)
    }
    evidence: dict[tuple[str, str], SopEvidenceSnapshot] = {}
    for reason in reasons:
        for route in POLICY_SECTION_ROUTES.get(reason.reason_code, ()):
            chunk = chunks.get((route.document_id, route.parent_section))
            if chunk is None:
                continue
            key = (route.document_id, route.parent_section)
            if key not in evidence:
                evidence[key] = SopEvidenceSnapshot(
                    document_id=chunk.document_id,
                    title=chunk.title,
                    section=chunk.section,
                    excerpt=" ".join(chunk.text.split())[:500],
                    source_reason_codes=[],
                    related_event_count=0,
                )
            item = evidence[key]
            item.source_reason_codes.append(reason.reason_code)
            item.related_event_count += reason.count
    return list(evidence.values())


def build_operations_briefing_snapshot(
    dashboard_summary: dict,
    period: BriefingPeriod,
    *,
    generated_at: datetime | None = None,
    docs_dir: Path | None = None,
) -> OperationsBriefingSnapshot:
    """Build a typed, LLM-free snapshot from repository dashboard aggregates."""
    window = dashboard_summary.get("windows", {}).get(period)
    if window is None:
        raise ValueError(f"Unsupported briefing period: {period}")
    counts = window.get("decision_counts", {})
    outcomes = window.get("review_outcomes", {})
    latency = window.get("latency_ms", {})
    timestamp = generated_at or datetime.now(timezone.utc)

    return OperationsBriefingSnapshot(
        schema_version="operations-briefing-snapshot-v1",
        source="deterministic-dashboard",
        generated_at=timestamp.astimezone(timezone.utc).isoformat(),
        period=period,
        period_label=PERIOD_LABELS[period],
        incoming_activity=IncomingActivitySnapshot(
            total=window.get("total", 0),
            automatic_pass=counts.get("PASS", 0),
            automatic_reject=counts.get("REJECT", 0),
            sent_to_review=counts.get("INSPECT", 0),
            attention_rate=window.get("attention_rate", 0.0),
            automatic_reject_rate=window.get("reject_rate", 0.0),
        ),
        review_outcomes=ReviewOutcomesSnapshot(
            resolved=outcomes.get("resolved", 0),
            pass_after_review=outcomes.get("operator_pass", 0),
            reject_after_review=outcomes.get("operator_reject", 0),
            no_defect_found=outcomes.get("no_defect_found", 0),
            accepted_minor_defect=outcomes.get("accepted_minor_defect", 0),
            pass_rate=outcomes.get("review_pass_rate", 0.0),
            reject_rate=outcomes.get("review_reject_rate", 0.0),
            no_defect_found_rate=outcomes.get("no_defect_found_rate", 0.0),
            accepted_minor_defect_rate=outcomes.get("accepted_minor_defect_rate", 0.0),
        ),
        review_queue=ReviewQueueSnapshot(
            open_count=dashboard_summary.get("review_queue_depth", 0),
            oldest_pending_minutes=dashboard_summary.get("oldest_pending_review_minutes"),
        ),
        top_policy_reasons=[
            ReasonCount.model_validate(item) for item in window.get("top_policy_reasons", [])
        ],
        review_disposition_mix=[
            ReasonCount.model_validate(item) for item in window.get("review_reason_counts", [])
        ],
        latency=LatencySnapshot.model_validate(
            {
                "samples": latency.get("samples", 0),
                "end_to_end": latency.get("end_to_end", {}),
                "vision": latency.get("vision", {}),
                "retrieval": latency.get("retrieval", {}),
                "explanation": latency.get("explanation", {}),
            }
        ),
        sop_evidence=_attach_sop_evidence(
            [ReasonCount.model_validate(item) for item in window.get("top_policy_reasons", [])],
            docs_dir,
        ),
    )
