from __future__ import annotations

import math
from dataclasses import dataclass

from .domain import Decision, VisionEvidence


@dataclass(frozen=True)
class PolicyOutcome:
    decision: Decision
    reason_codes: tuple[str, ...]
    policy_version: str = "grid-critical-policy-v1"


MIN_VALID_COVERAGE = 0.98


def evaluate(evidence: VisionEvidence) -> PolicyOutcome:
    """Fail closed: unknown, unverified, or low-quality inputs never PASS."""
    if not evidence.artifact_verified:
        return PolicyOutcome(Decision.INSPECT, ("ARTIFACT_NOT_VERIFIED",))
    if not math.isfinite(evidence.score):
        return PolicyOutcome(Decision.INSPECT, ("NON_FINITE_ANOMALY_SCORE",))
    if evidence.review_threshold > evidence.reject_threshold:
        return PolicyOutcome(Decision.INSPECT, ("INVALID_THRESHOLD_ORDER",))
    if evidence.quality is None:
        return PolicyOutcome(Decision.INSPECT, ("MISSING_IMAGE_QUALITY",))
    if not evidence.quality.acceptable:
        return PolicyOutcome(Decision.INSPECT, tuple(evidence.quality.failure_reasons) or ("IMAGE_QUALITY_GATE",))
    if evidence.valid_coverage < MIN_VALID_COVERAGE:
        return PolicyOutcome(Decision.INSPECT, ("LOW_VALID_COVERAGE",))
    if evidence.score >= evidence.reject_threshold:
        return PolicyOutcome(Decision.REJECT, ("SCORE_AT_OR_ABOVE_REJECT_THRESHOLD",))
    if evidence.score >= evidence.review_threshold:
        return PolicyOutcome(Decision.INSPECT, ("SCORE_IN_REVIEW_BAND",))
    return PolicyOutcome(Decision.PASS, ("ALL_CRITICAL_GATES_PASSED",))


def decide(evidence: VisionEvidence) -> Decision:
    """Safety-critical, deterministic decision policy.

    Never call an LLM from this function. A low-coverage image is routed to a
    human even if its anomaly score is low.
    """
    return evaluate(evidence).decision


def checklist_for(decision: Decision, evidence: VisionEvidence) -> list[str]:
    if decision is Decision.INSPECT:
        return [
            "Confirm the original image is sharp and the full grid area is visible.",
            "Review the anomaly-mask overlay and each highlighted region.",
            "Confirm whether the marked pattern is a real grid defect or a lighting/capture artifact.",
            "Record final disposition and a reason code for future calibration.",
        ]
    if decision is Decision.REJECT:
        return [
            "Quarantine the part and retain the image and anomaly mask.",
            "Verify the highlighted region before disposition if the production SOP requires it.",
            "Check recent grid reject rate for a possible process alert.",
        ]
    return ["Inspection passed under the current calibrated grid policy."]
