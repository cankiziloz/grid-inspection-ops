"""Deterministic policy-to-procedure routing for safety-relevant workflows.

The routes name a parent SOP section, not an individual embedded child chunk.
Today each parent section fits in one chunk; when a procedure grows, multiple
child chunks can share its ``parent_section`` metadata and stay behind the same
governance boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

from .domain import InspectionRecord


@dataclass(frozen=True)
class ProcedureRoute:
    document_id: str
    parent_section: str


POLICY_SECTION_ROUTES: dict[str, tuple[ProcedureRoute, ...]] = {
    "SCORE_AT_OR_ABOVE_REJECT_THRESHOLD": (
        ProcedureRoute("SOP-QA-04", "Immediate actions"),
    ),
    "SCORE_IN_REVIEW_BAND": (
        ProcedureRoute("SOP-REV-02", "Required review steps"),
    ),
    "LOW_VALID_COVERAGE": (
        ProcedureRoute("SOP-REV-02", "Required review steps"),
        ProcedureRoute("SOP-ING-01", "Image acceptance rules"),
    ),
    "MISSING_IMAGE_QUALITY": (
        ProcedureRoute("SOP-REV-02", "Required review steps"),
        ProcedureRoute("SOP-ING-01", "Image acceptance rules"),
    ),
    "IMAGE_RESOLUTION_TOO_LOW": (
        ProcedureRoute("SOP-REV-02", "Required review steps"),
        ProcedureRoute("SOP-ING-01", "Image acceptance rules"),
    ),
    "IMAGE_EXPOSURE_OUT_OF_RANGE": (
        ProcedureRoute("SOP-REV-02", "Required review steps"),
        ProcedureRoute("SOP-ING-01", "Image acceptance rules"),
    ),
    "IMAGE_CONTRAST_TOO_LOW": (
        ProcedureRoute("SOP-REV-02", "Required review steps"),
        ProcedureRoute("SOP-ING-01", "Image acceptance rules"),
    ),
    "IMAGE_TOO_BLURRY": (
        ProcedureRoute("SOP-REV-02", "Required review steps"),
        ProcedureRoute("SOP-ING-01", "Image acceptance rules"),
    ),
    "ARTIFACT_NOT_VERIFIED": (ProcedureRoute("SOP-ESC-03", "Escalate when"),),
    "NON_FINITE_ANOMALY_SCORE": (ProcedureRoute("SOP-ESC-03", "Escalate when"),),
    "INVALID_THRESHOLD_ORDER": (ProcedureRoute("SOP-ESC-03", "Escalate when"),),
    "ALL_CRITICAL_GATES_PASSED": (ProcedureRoute("SOP-ING-01", "Response handling"),),
}


def routes_for_record(record: InspectionRecord) -> list[ProcedureRoute]:
    """Return ordered, deduplicated SOP parent sections for a policy outcome."""
    reasons = record.metadata.get("critical_policy", {}).get("reason_codes", [])
    return list(dict.fromkeys(route for reason in reasons for route in POLICY_SECTION_ROUTES.get(reason, ())))


def documents_for_reasons(reasons: list[str]) -> tuple[str, ...]:
    """Compatibility view used by the local lexical retriever."""
    return tuple(
        dict.fromkeys(
            route.document_id for reason in reasons for route in POLICY_SECTION_ROUTES.get(reason, ())
        )
    )
