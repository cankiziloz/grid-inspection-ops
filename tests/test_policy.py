import math

import pytest

from app.domain import Decision, ImageQuality, VisionEvidence
from app.policy import decide, evaluate


def evidence(
    score: float,
    coverage: float = 1.0,
    *,
    verified: bool = True,
    quality: ImageQuality | None = None,
) -> VisionEvidence:
    return VisionEvidence(
        model_version="grid-v1",
        score=score,
        review_threshold=0.25,
        reject_threshold=0.40,
        valid_coverage=coverage,
        artifact_verified=verified,
        quality=quality or ImageQuality(
            width=512, height=512, mean_luminance=0.5, contrast=0.2,
            laplacian_variance=40.0, acceptable=True,
        ),
    )


def test_policy_returns_three_distinct_actions():
    assert decide(evidence(0.20)) is Decision.PASS
    assert decide(evidence(0.25)) is Decision.INSPECT
    assert decide(evidence(0.40)) is Decision.REJECT


@pytest.mark.parametrize(
    ("item", "reason"),
    [
        (evidence(0.10, verified=False), "ARTIFACT_NOT_VERIFIED"),
        (evidence(0.10, coverage=0.97), "LOW_VALID_COVERAGE"),
        (evidence(0.10, quality=ImageQuality(width=512, height=512, mean_luminance=0.5, contrast=0.0, laplacian_variance=40, acceptable=False, failure_reasons=["IMAGE_CONTRAST_TOO_LOW"])), "IMAGE_CONTRAST_TOO_LOW"),
        (evidence(math.nan), "NON_FINITE_ANOMALY_SCORE"),
    ],
)
def test_critical_gates_fail_closed(item, reason):
    outcome = evaluate(item)
    assert outcome.decision is Decision.INSPECT
    assert reason in outcome.reason_codes


def test_decision_is_monotonic_for_valid_evidence():
    rank = {Decision.PASS: 0, Decision.INSPECT: 1, Decision.REJECT: 2}
    decisions = [rank[decide(evidence(score))] for score in (0.0, 0.249, 0.25, 0.39, 0.4, 0.8)]
    assert decisions == sorted(decisions)
